// SPDX-License-Identifier: MIT
// qmm's f16 WMMA path (qmmh.h, PLAN.md "Speed" S2), for the op tests and tools/qmm_bench.py. Its own
// translation unit: one LDS-using kernel per TU (DESIGN.md "The megakernel").
#include "qmmh.h"
#include "moe.h"

#include "qmm_host.h"
#include "rt.h"

namespace mk {

// One dispatcher kernel in this TU (the LDS rule): which = 0 the matmul, 1 the split-K sum.
__global__ void __launch_bounds__(kMkBlock) k_qmmh(QmmArgs a, uint32_t which) {
  if (which == 0) op_qmmh(a, blockIdx.x, gridDim.x);
  else op_qmm_ksum(a, blockIdx.x, gridDim.x);
}

// The stages a fused staging replaces (tests: tests/ops/test_qmmh_xin.py). No LDS.
__global__ void __launch_bounds__(kMkBlock) k_ref_silu(SiluScaleArgs a) { op_silu_scale(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_ref_swiglu(SwigluArgs a) { op_swiglu(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_ref_hc_mix(HcMixArgs a) { op_hc_mix(a, blockIdx.x, gridDim.x); }
// x as the fused staging computes it, in f32 before the f16 rounding (which would hide a 1-ulp difference).
__global__ void __launch_bounds__(kMkBlock) k_xin_probe(QmmArgs a, float *x) {
  const uint64_t n4 = uint64_t(a.t) * a.k / 4;
  for (uint64_t i = uint64_t(blockIdx.x) * kMkBlock + threadIdx.x; i < n4; i += uint64_t(gridDim.x) * kMkBlock) {
    const uint32_t tok = uint32_t(i * 4 / a.k), col = uint32_t(i * 4 % a.k);
    *reinterpret_cast<float4 *>(x + uint64_t(tok) * a.k + col) = qmmh_x4(a, tok, col);
  }
}

}  // namespace mk

using namespace mk;

static void launch_qmmh(const QmmArgs &a, uint32_t grid) { k_qmmh<<<grid, kMkBlock>>>(a, 0); }

// Same C ABI as qwen38_qmm (qmm.hip); quantized types only.
extern "C" int qwen38_qmmh(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k, const float *x_host,
                       uint64_t x_bytes, uint32_t x_rows, const uint32_t *x_row_host, uint64_t x_row_bytes, uint32_t t,
                       float *y_host, uint64_t y_bytes, uint32_t y_rows, const uint32_t *y_row_host,
                       uint64_t y_row_bytes, uint32_t ldy, uint32_t grid) {
  if (!qmmh_type(type)) HIP_DIE("qwen38_qmmh: type %u has no f16 path", type);
  return run_qmm(launch_qmmh, kQmmhRows, kQmmhToks, "qwen38_qmmh", type, w_host, w_bytes, n, k, x_host, x_bytes, x_rows,
                 x_row_host, x_row_bytes, t, y_host, y_bytes, y_rows, y_row_host, y_row_bytes, ldy, grid);
}

extern "C" double qwen38_qmmh_bench(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k,
                                const float *x_host, uint64_t x_bytes, uint32_t t, uint32_t grid, uint32_t reps) {
  if (!qmmh_type(type)) HIP_DIE("qwen38_qmmh_bench: type %u has no f16 path", type);
  return bench_qmm(launch_qmmh, "qwen38_qmmh_bench", type, w_host, w_bytes, n, k, x_host, x_bytes, t, grid, reps);
}

// k_qmmh's registers, scratch (bytes per lane) and static LDS, for the resource test.
extern "C" int qwen38_qmmh_resources(int *regs, int *scratch, int *lds) {
  hipFuncAttributes fa{};
  HIP_CHECK(hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(&k_qmmh)));
  *regs = fa.numRegs;
  *scratch = int(fa.localSizeBytes);
  *lds = int(fa.sharedSizeBytes);
  return 0;
}

// Split-K (qmmh.h): y [t, n] = W x for x [t, k], the matmul in `ksplit` K slices and then their sum, both kernels
// on `grid` workgroups (0: 1024). For tests/ops/test_qmmh_split.py.
extern "C" int qwen38_qmmh_split(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k,
                             const float *x_host, uint64_t x_bytes, uint32_t t, float *y_host, uint64_t y_bytes,
                             uint32_t ksplit, uint32_t grid) {
  if (!qmmh_type(type)) HIP_DIE("qwen38_qmmh_split: type %u has no f16 path", type);
  if (!qmmh_ksplit_ok(k, ksplit)) HIP_DIE("qwen38_qmmh_split: ksplit %u leaves an empty slice of K=%u", ksplit, k);
  if (ksplit > kMaxKsplit) HIP_DIE("qwen38_qmmh_split: ksplit %u > %u", ksplit, kMaxKsplit);
  const uint64_t ldw = q::row_bytes(type, k);
  HIP_CHECK_BYTES("qwen38_qmmh_split", "w", w_bytes, ldw * n);
  HIP_CHECK_BYTES("qwen38_qmmh_split", "x", x_bytes, sizeof(float) * uint64_t(t) * k);
  HIP_CHECK_BYTES("qwen38_qmmh_split", "y", y_bytes, sizeof(float) * uint64_t(t) * n);
  uint8_t *w;
  float *x, *y, *part = nullptr;
  HIP_CHECK(hipMalloc(&w, ldw * n));
  HIP_CHECK(hipMemcpy(w, w_host, ldw * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&x, sizeof(float) * t * k));
  HIP_CHECK(hipMemcpy(x, x_host, sizeof(float) * t * k, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&y, sizeof(float) * t * n));
  HIP_CHECK(hipMemset(y, 0xff, sizeof(float) * t * n));
  if (ksplit > 1) {
    HIP_CHECK(hipMalloc(&part, sizeof(float) * ksplit * t * n));
    HIP_CHECK(hipMemset(part, 0xff, sizeof(float) * ksplit * t * n));
  }
  QmmArgs a{type, w, ldw, n, k, t, x, k, nullptr, y, n, nullptr, 0, ksplit, part};
  const uint32_t G = grid ? grid : 1024;
  k_qmmh<<<G, kMkBlock>>>(a, 0);
  if (ksplit > 1) k_qmmh<<<G, kMkBlock>>>(a, 1);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(y_host, y, sizeof(float) * t * n, hipMemcpyDeviceToHost));
  for (void *p : {static_cast<void *>(w), static_cast<void *>(x), static_cast<void *>(y), static_cast<void *>(part)})
    if (p) HIP_CHECK(hipFree(p));
  return 0;
}

// The engine's split policy (qmmh_ksplit): a function of the matrix shape alone.
extern "C" uint32_t qwen38_qmmh_ksplit(uint32_t n, uint32_t k) { return qmmh_ksplit(n, k); }

// Fused staging (QmmArgs::xin) against the stages it replaces. parts (and parts2 for swiglu): [xks][t][k] split-K
// partials of the matrices producing x. y_fused: the f16 matmul staging x = silu(scale * sum parts) (kind 1) or
// silu(sum parts) * sum parts2 (kind 2) itself; y_ref: op_qmm_ksum, then op_silu_scale or op_swiglu, then the plain
// matmul. W is `type` rows (Q8_0 or Q8_0 tiles). x_fused / x_ref: the f32 x each computes, before staging to f16.
extern "C" int qwen38_qmmh_xin(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k, uint32_t t,
                           const float *parts_host, uint64_t parts_bytes, const float *parts2_host, uint64_t parts2_bytes,
                           uint32_t xks, uint32_t kind, float scale, float *y_fused_host, uint64_t y_fused_bytes,
                           float *y_ref_host, uint64_t y_ref_bytes, float *x_fused_host, uint64_t x_fused_bytes,
                           float *x_ref_host, uint64_t x_ref_bytes, uint32_t grid) {
  if (!qmmh_type(type)) HIP_DIE("qwen38_qmmh_xin: type %u has no f16 path", type);
  if (kind != kXinSiluSum && kind != kXinSwigluSum) HIP_DIE("qwen38_qmmh_xin: kind %u", kind);
  if (xks < 2 || xks > kMaxKsplit) HIP_DIE("qwen38_qmmh_xin: xks %u", xks);
  if (t == 0 || k % 64 || n % 16) HIP_DIE("qwen38_qmmh_xin: shape %u x %u, t %u", n, k, t);
  const uint64_t ldw = q::row_bytes(type, k), pn = uint64_t(xks) * t * k;
  HIP_CHECK_BYTES("qwen38_qmmh_xin", "w", w_bytes, ldw * n);
  HIP_CHECK_BYTES("qwen38_qmmh_xin", "parts", parts_bytes, sizeof(float) * pn);
  HIP_CHECK_BYTES("qwen38_qmmh_xin", "parts2", parts2_bytes, kind == kXinSwigluSum ? sizeof(float) * pn : 0);
  HIP_CHECK_BYTES("qwen38_qmmh_xin", "y_fused", y_fused_bytes, sizeof(float) * uint64_t(t) * n);
  HIP_CHECK_BYTES("qwen38_qmmh_xin", "y_ref", y_ref_bytes, sizeof(float) * uint64_t(t) * n);
  HIP_CHECK_BYTES("qwen38_qmmh_xin", "x_fused", x_fused_bytes, sizeof(float) * uint64_t(t) * k);
  HIP_CHECK_BYTES("qwen38_qmmh_xin", "x_ref", x_ref_bytes, sizeof(float) * uint64_t(t) * k);
  uint8_t *w;
  float *p, *p2 = nullptr, *g, *u = nullptr, *x, *y;
  HIP_CHECK(hipMalloc(&w, ldw * n));
  HIP_CHECK(hipMemcpy(w, w_host, ldw * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&p, sizeof(float) * pn));
  HIP_CHECK(hipMemcpy(p, parts_host, sizeof(float) * pn, hipMemcpyHostToDevice));
  if (kind == kXinSwigluSum) {
    HIP_CHECK(hipMalloc(&p2, sizeof(float) * pn));
    HIP_CHECK(hipMemcpy(p2, parts2_host, sizeof(float) * pn, hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&u, sizeof(float) * t * k));
  }
  HIP_CHECK(hipMalloc(&g, sizeof(float) * t * k));
  HIP_CHECK(hipMalloc(&x, sizeof(float) * t * k));
  HIP_CHECK(hipMalloc(&y, sizeof(float) * t * n));
  const uint32_t G = grid ? grid : 1024;
  // fused
  QmmArgs a{};
  a.type = type; a.w = w; a.ldw = ldw; a.n = n; a.k = k; a.t = t; a.x = nullptr; a.ldx = k; a.y = y; a.ldy = n;
  a.xin = kind; a.xks = xks; a.xp = p; a.xp2 = p2; a.xscale = scale;
  HIP_CHECK(hipMemset(y, 0xff, sizeof(float) * t * n));
  k_qmmh<<<G, kMkBlock>>>(a, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(y_fused_host, y, sizeof(float) * t * n, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemset(x, 0xff, sizeof(float) * t * k));
  k_xin_probe<<<G, kMkBlock>>>(a, x);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(x_fused_host, x, sizeof(float) * t * k, hipMemcpyDeviceToHost));
  // reference: the stages the fusion replaces
  QmmArgs s{};
  s.n = k; s.t = t; s.ksplit = xks; s.part = p; s.y = g; s.ldy = k;
  k_qmmh<<<G, kMkBlock>>>(s, 1);
  if (kind == kXinSwigluSum) {
    s.part = p2; s.y = u;
    k_qmmh<<<G, kMkBlock>>>(s, 1);
    k_ref_swiglu<<<G, kMkBlock>>>(SwigluArgs{g, u, k, k, x, k, t, k});
  } else {
    k_ref_silu<<<G, kMkBlock>>>(SiluScaleArgs{g, x, t * k, scale});
  }
  QmmArgs r{};
  r.type = type; r.w = w; r.ldw = ldw; r.n = n; r.k = k; r.t = t; r.x = x; r.ldx = k; r.y = y; r.ldy = n;
  HIP_CHECK(hipMemset(y, 0xff, sizeof(float) * t * n));
  k_qmmh<<<G, kMkBlock>>>(r, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(y_ref_host, y, sizeof(float) * t * n, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(x_ref_host, x, sizeof(float) * t * k, hipMemcpyDeviceToHost));
  for (void *q : {static_cast<void *>(w), static_cast<void *>(p), static_cast<void *>(p2), static_cast<void *>(g),
                  static_cast<void *>(u), static_cast<void *>(x), static_cast<void *>(y)})
    if (q) HIP_CHECK(hipFree(q));
  return 0;
}

// hc up + hc_mix as one stage (QmmArgs::yout = kYoutHcMix; docs/narrow.md N1) against the pair it replaces. W is 4 * e rows
// of `k` (Q8_0 or Q8_0 tiles). x [t, k] (or, with parts non-null, x staged from the xks split-K partials [xks][t][k] by
// kXinSiluSum with `scale`, and x null); xn [t, 4e]; z [t, 4] or null. mixed_f / inject_f: the fused stage's outputs;
// mixed_r / inject_r: the plain matmul into `up`, then op_hc_mix. Outputs the caller pre-fills with NaN stay NaN where
// the stage doesn't write (inject without z).
extern "C" int qwen38_qmmh_hcmix(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t k, uint32_t e, uint32_t t,
                             const float *x_host, uint64_t x_bytes, const float *parts_host, uint64_t parts_bytes,
                             uint32_t xks, float scale, const float *xn_host, uint64_t xn_bytes, const float *z_host,
                             uint64_t z_bytes, float *mixed_f_host, uint64_t mixed_f_bytes, float *inject_f_host,
                             uint64_t inject_f_bytes, float *mixed_r_host, uint64_t mixed_r_bytes, float *inject_r_host,
                             uint64_t inject_r_bytes, uint32_t grid) {
  if (type != q::Q8_0 && type != q::Q8_0_TILES) HIP_DIE("qwen38_qmmh_hcmix: type %u (Q8_0 or Q8_0 tiles only)", type);
  if (t == 0 || e == 0 || e % 16 || k % 64) HIP_DIE("qwen38_qmmh_hcmix: shape e=%u k=%u t=%u", e, k, t);
  const bool xin = parts_host != nullptr;
  const uint64_t n = 4ull * e, ldw = q::row_bytes(type, k), tk = uint64_t(t) * k, pn = uint64_t(xks) * tk;
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "w", w_bytes, ldw * n);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "x", x_bytes, xin ? 0 : sizeof(float) * tk);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "parts", parts_bytes, xin ? sizeof(float) * pn : 0);
  if (xin && (xks < 2 || xks > kMaxKsplit)) HIP_DIE("qwen38_qmmh_hcmix: xks %u", xks);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "xn", xn_bytes, sizeof(float) * t * n);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "z", z_bytes, z_host ? sizeof(float) * 4 * t : 0);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "mixed_f", mixed_f_bytes, sizeof(float) * t * e);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "mixed_r", mixed_r_bytes, sizeof(float) * t * e);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "inject_f", inject_f_bytes, sizeof(float) * 4 * t);
  HIP_CHECK_BYTES("qwen38_qmmh_hcmix", "inject_r", inject_r_bytes, sizeof(float) * 4 * t);
  uint8_t *w;
  float *x = nullptr, *p = nullptr, *g = nullptr, *xn, *z = nullptr, *up, *mf, *jf, *mr, *jr;
  HIP_CHECK(hipMalloc(&w, ldw * n));
  HIP_CHECK(hipMemcpy(w, w_host, ldw * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&x, sizeof(float) * tk));
  if (xin) {
    HIP_CHECK(hipMalloc(&p, sizeof(float) * pn));
    HIP_CHECK(hipMemcpy(p, parts_host, sizeof(float) * pn, hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&g, sizeof(float) * tk));
  } else {
    HIP_CHECK(hipMemcpy(x, x_host, sizeof(float) * tk, hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMalloc(&xn, sizeof(float) * t * n));
  HIP_CHECK(hipMemcpy(xn, xn_host, sizeof(float) * t * n, hipMemcpyHostToDevice));
  if (z_host) {
    HIP_CHECK(hipMalloc(&z, sizeof(float) * 4 * t));
    HIP_CHECK(hipMemcpy(z, z_host, sizeof(float) * 4 * t, hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMalloc(&up, sizeof(float) * t * n));
  HIP_CHECK(hipMalloc(&mf, sizeof(float) * t * e));
  HIP_CHECK(hipMalloc(&mr, sizeof(float) * t * e));
  HIP_CHECK(hipMalloc(&jf, sizeof(float) * 4 * t));
  HIP_CHECK(hipMalloc(&jr, sizeof(float) * 4 * t));
  for (float *o : {mf, mr}) HIP_CHECK(hipMemset(o, 0xff, sizeof(float) * t * e));
  for (float *o : {jf, jr}) HIP_CHECK(hipMemset(o, 0xff, sizeof(float) * 4 * t));
  HIP_CHECK(hipMemset(up, 0xff, sizeof(float) * t * n));
  const uint32_t G = grid ? grid : 1024;
  QmmArgs a{};
  a.type = type; a.w = w; a.ldw = ldw; a.n = uint32_t(n); a.k = k; a.t = t; a.x = x; a.ldx = k; a.y = up; a.ldy = uint32_t(n);
  if (xin) { a.x = nullptr; a.xin = kXinSiluSum; a.xks = xks; a.xp = p; a.xscale = scale; }
  // the reference: stage x as the fused stage does (the ksum and silu stages), then the plain matmul into up, then hc_mix
  QmmArgs r = a;
  if (xin) {
    QmmArgs s{};
    s.n = k; s.t = t; s.ksplit = xks; s.part = p; s.y = g; s.ldy = k;
    k_qmmh<<<G, kMkBlock>>>(s, 1);
    k_ref_silu<<<G, kMkBlock>>>(SiluScaleArgs{g, x, uint32_t(tk), scale});
    r.x = x; r.xin = kXinPlain;
  }
  k_qmmh<<<G, kMkBlock>>>(r, 0);
  k_ref_hc_mix<<<G, kMkBlock>>>(HcMixArgs{xn, up, mr, z, jr, t, e});
  a.yout = kYoutHcMix; a.mix_xn = xn; a.mix_z = z; a.mix_out = mf; a.mix_inject = z ? jf : nullptr; a.mix_e = e;
  k_qmmh<<<G, kMkBlock>>>(a, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(mixed_f_host, mf, sizeof(float) * t * e, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(mixed_r_host, mr, sizeof(float) * t * e, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(inject_f_host, jf, sizeof(float) * 4 * t, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(inject_r_host, jr, sizeof(float) * 4 * t, hipMemcpyDeviceToHost));
  for (void *q : {static_cast<void *>(w), static_cast<void *>(x), static_cast<void *>(p), static_cast<void *>(g),
                  static_cast<void *>(xn), static_cast<void *>(z), static_cast<void *>(up), static_cast<void *>(mf),
                  static_cast<void *>(mr), static_cast<void *>(jf), static_cast<void *>(jr)})
    if (q) HIP_CHECK(hipFree(q));
  return 0;
}
