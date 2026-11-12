// SPDX-License-Identifier: MIT
#include "moe.h"

#include <vector>

#include "rt.h"

namespace qwen38 {
// Every matmul of the block goes through this one dispatcher kernel, the only kernel in this TU that reaches
// the LDS arena (qmmh): LDS-using ops stay inlined into a single kernel per TU (DESIGN.md "The megakernel").
enum MoeMm : uint32_t { kMmQmm, kMmQmmh, kMmGu, kMmDown };
struct MoeMmArgs {
  uint32_t which;
  QmmArgs q;
  MoeFfnArgs f;
};
__global__ void __launch_bounds__(kBlockThreads) k_moe_mm(MoeMmArgs a) {
  switch (a.which) {
    case kMmQmm: op_qmm(a.q, blockIdx.x, gridDim.x); break;
    case kMmQmmh: op_qmmh(a.q, blockIdx.x, gridDim.x); break;
    case kMmGu: op_moe_gu(a.f, blockIdx.x, gridDim.x); break;
    case kMmDown: op_moe_down(a.f, blockIdx.x, gridDim.x); break;
    default: HIP_TRAP(kTrapBadOp, kTrapFileMoeHip);
  }
}
__global__ void __launch_bounds__(kBlockThreads) k_moe_router(MoeRouterArgs a) { op_moe_router(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kBlockThreads) k_moe_group(MoeGroupArgs a) { op_moe_group(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kBlockThreads) k_swiglu(SwigluArgs a) { op_swiglu(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kBlockThreads) k_moe_combine(MoeCombineArgs a) { op_moe_combine(a, blockIdx.x, gridDim.x); }

template <class T>
static T *dev(const void *h, size_t bytes) {
  T *d;
  HIP_CHECK(hipMalloc(&d, bytes));
  if (h) HIP_CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
  else HIP_CHECK(hipMemset(d, 0xff, bytes));
  return d;
}
}  // namespace qwen38

using namespace qwen38;
using namespace qwen38::model;

// The whole MoE block for tests: x [T, 2560] -> out [T, 2560], plus the router's choices.
// Weights as stored in the GGUF (raw bytes of the full 512-expert tensors). fast = 1: the quantized matmuls
// (experts and the shared expert) on f16 WMMA, as the engine builds them with QWEN38_FAST_QMM=1.
extern "C" int qwen38_moe(const float *x, uint64_t x_bytes, uint32_t t, const float *router, uint64_t router_bytes,
                      const float *shexp_gate, uint64_t shexp_gate_bytes, uint32_t type_gate, const void *gate,
                      uint64_t gate_bytes, uint32_t type_up, const void *up, uint64_t up_bytes, uint32_t type_down,
                      const void *down, uint64_t down_bytes, const void *sh_gate, uint64_t sh_gate_bytes,
                      const void *sh_up, uint64_t sh_up_bytes, const void *sh_down, uint64_t sh_down_bytes, float *out,
                      uint64_t out_bytes, int32_t *topk_idx_out, uint64_t topk_idx_bytes, float *topk_w_out,
                      uint64_t topk_w_bytes, uint32_t grid, uint32_t fast) {
  if (fast > 1) HIP_DIE("qwen38_moe: fast=%u, want 0 or 1", fast);
  const uint32_t A = t * kTopK;
  for (uint32_t ty : {type_gate, type_up, type_down})
    if (q::block_bytes(ty) == 0) HIP_DIE("qwen38_moe: unknown quant type %u", ty);
  const uint64_t ldg = q::row_bytes(type_gate, kEmbd), ldu = q::row_bytes(type_up, kEmbd), ldd = q::row_bytes(type_down, kFf);
  const uint64_t ldsh = q::row_bytes(q::Q8_0, kEmbd), ldshd = q::row_bytes(q::Q8_0, kFfShared);
  HIP_CHECK_BYTES("qwen38_moe", "x", x_bytes, sizeof(float) * uint64_t(t) * kEmbd);
  HIP_CHECK_BYTES("qwen38_moe", "router", router_bytes, sizeof(float) * kExperts * kEmbd);
  HIP_CHECK_BYTES("qwen38_moe", "shexp_gate", shexp_gate_bytes, sizeof(float) * kEmbd);
  HIP_CHECK_BYTES("qwen38_moe", "gate", gate_bytes, ldg * kFf * kExperts);
  HIP_CHECK_BYTES("qwen38_moe", "up", up_bytes, ldu * kFf * kExperts);
  HIP_CHECK_BYTES("qwen38_moe", "down", down_bytes, ldd * kEmbd * kExperts);
  HIP_CHECK_BYTES("qwen38_moe", "sh_gate", sh_gate_bytes, ldsh * kFfShared);
  HIP_CHECK_BYTES("qwen38_moe", "sh_up", sh_up_bytes, ldsh * kFfShared);
  HIP_CHECK_BYTES("qwen38_moe", "sh_down", sh_down_bytes, ldshd * kEmbd);
  HIP_CHECK_BYTES("qwen38_moe", "out", out_bytes, sizeof(float) * uint64_t(t) * kEmbd);
  HIP_CHECK_BYTES("qwen38_moe", "topk_idx", topk_idx_bytes, sizeof(int32_t) * A);
  HIP_CHECK_BYTES("qwen38_moe", "topk_w", topk_w_bytes, sizeof(float) * A);
  float *dx = dev<float>(x, sizeof(float) * t * kEmbd);
  float *drouter = dev<float>(router, sizeof(float) * kExperts * kEmbd);
  float *dshg = dev<float>(shexp_gate, sizeof(float) * kEmbd);
  uint8_t *dgate = dev<uint8_t>(gate, ldg * kFf * kExperts), *dup = dev<uint8_t>(up, ldu * kFf * kExperts);
  uint8_t *ddown = dev<uint8_t>(down, ldd * kEmbd * kExperts);
  uint8_t *dsg = dev<uint8_t>(sh_gate, ldsh * kFfShared), *dsu = dev<uint8_t>(sh_up, ldsh * kFfShared);
  uint8_t *dsd = dev<uint8_t>(sh_down, ldshd * kEmbd);
  float *logits = dev<float>(nullptr, sizeof(float) * t * kExperts);
  int32_t *tidx = dev<int32_t>(nullptr, sizeof(int32_t) * A);
  float *tw = dev<float>(nullptr, sizeof(float) * A);
  uint32_t *cnt = dev<uint32_t>(nullptr, 4 * kExperts), *off = dev<uint32_t>(nullptr, 4 * (kExperts + 1));
  uint32_t *asg = dev<uint32_t>(nullptr, 4 * A), *tok = dev<uint32_t>(nullptr, 4 * A);
  uint32_t *igu = dev<uint32_t>(nullptr, 4 * (kExperts + 1)), *idn = dev<uint32_t>(nullptr, 4 * (kExperts + 1));
  float *gu = dev<float>(nullptr, sizeof(float) * A * 2 * kFf), *h = dev<float>(nullptr, sizeof(float) * A * kFf);
  float *y = dev<float>(nullptr, sizeof(float) * A * kEmbd);
  float *sg = dev<float>(nullptr, sizeof(float) * t * kFfShared), *su = dev<float>(nullptr, sizeof(float) * t * kFfShared);
  float *sh = dev<float>(nullptr, sizeof(float) * t * kFfShared), *sd = dev<float>(nullptr, sizeof(float) * t * kEmbd);
  float *sgt = dev<float>(nullptr, sizeof(float) * t), *dout = dev<float>(nullptr, sizeof(float) * t * kEmbd);
  const uint32_t G = grid ? grid : 1024;

  auto qmm = [&](const QmmArgs &q) {
    MoeMmArgs m{};
    m.which = fast && qmmh_type(q.type) ? kMmQmmh : kMmQmm;
    m.q = q;
    k_moe_mm<<<G, kBlockThreads>>>(m);
  };
  qmm(QmmArgs{q::F32, reinterpret_cast<const uint8_t *>(drouter), 4ull * kEmbd, kExperts, kEmbd, t, dx, kEmbd, nullptr,
              logits, kExperts, nullptr});
  k_moe_router<<<G, kBlockThreads>>>(MoeRouterArgs{logits, tidx, tw, t});
  k_moe_group<<<G, kBlockThreads>>>(MoeGroupArgs{tidx, t, cnt, off, asg, tok, igu, idn, fast});
  MoeFfnArgs f{type_gate, type_up, type_down, dgate, dup, ddown, ldg, ldu, ldd, dx, gu, h, y, cnt, off, asg, tok, igu, idn,
               fast};
  MoeMmArgs mf{};
  mf.f = f;
  mf.which = kMmGu;
  k_moe_mm<<<G, kBlockThreads>>>(mf);
  k_swiglu<<<G, kBlockThreads>>>(SwigluArgs{gu, gu + kFf, 2 * kFf, 2 * kFf, h, kFf, A, kFf});
  mf.which = kMmDown;
  k_moe_mm<<<G, kBlockThreads>>>(mf);
  qmm(QmmArgs{q::Q8_0, dsg, ldsh, kFfShared, kEmbd, t, dx, kEmbd, nullptr, sg, kFfShared, nullptr});
  qmm(QmmArgs{q::Q8_0, dsu, ldsh, kFfShared, kEmbd, t, dx, kEmbd, nullptr, su, kFfShared, nullptr});
  k_swiglu<<<G, kBlockThreads>>>(SwigluArgs{sg, su, kFfShared, kFfShared, sh, kFfShared, t, kFfShared});
  qmm(QmmArgs{q::Q8_0, dsd, ldshd, kEmbd, kFfShared, t, sh, kFfShared, nullptr, sd, kEmbd, nullptr});
  qmm(QmmArgs{q::F32, reinterpret_cast<const uint8_t *>(dshg), 4ull * kEmbd, 1, kEmbd, t, dx, kEmbd, nullptr, sgt, 1,
              nullptr});
  k_moe_combine<<<G, kBlockThreads>>>(MoeCombineArgs{y, tidx, tw, sd, sgt, dout, t});
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(out, dout, sizeof(float) * t * kEmbd, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(topk_idx_out, tidx, sizeof(int32_t) * A, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(topk_w_out, tw, sizeof(float) * A, hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{dx, drouter, dshg, dgate, dup, ddown, dsg, dsu, dsd, logits, tidx, tw, cnt, off, asg,
                                     tok, igu, idn, gu, h, y, sg, su, sh, sd, sgt, dout})
    HIP_CHECK(hipFree(p));
  return 0;
}

// The router's top-10 and the grouping stage alone, for tests/ops/test_moe_route.py: logits [T, 512] in, the
// router's choices and the group stage's bookkeeping out. fast picks the expert tile the item prefixes count.
extern "C" int qwen38_moe_route(const float *logits_host, uint32_t t, uint32_t fast, uint32_t grid, int32_t *topk_idx,
                            float *topk_w, uint32_t *count, uint32_t *offset, uint32_t *assign, uint32_t *token,
                            uint32_t *item_gu, uint32_t *item_dn) {
  if (fast > 1) HIP_DIE("qwen38_moe_route: fast=%u, want 0 or 1", fast);
  if (t == 0 || t > model::kMaxPassTokens) HIP_DIE("qwen38_moe_route: T=%u out of range (1..%u)", t, model::kMaxPassTokens);
  const uint32_t A = t * kTopK, G = grid ? grid : 1024;
  float *lg = dev<float>(logits_host, sizeof(float) * t * kExperts);
  int32_t *tidx = dev<int32_t>(nullptr, sizeof(int32_t) * A);
  float *tw = dev<float>(nullptr, sizeof(float) * A);
  uint32_t *cnt = dev<uint32_t>(nullptr, 4 * kExperts), *off = dev<uint32_t>(nullptr, 4 * (kExperts + 1));
  uint32_t *asg = dev<uint32_t>(nullptr, 4 * A), *tok = dev<uint32_t>(nullptr, 4 * A);
  uint32_t *igu = dev<uint32_t>(nullptr, 4 * (kExperts + 1)), *idn = dev<uint32_t>(nullptr, 4 * (kExperts + 1));
  k_moe_router<<<G, kBlockThreads>>>(MoeRouterArgs{lg, tidx, tw, t});
  k_moe_group<<<G, kBlockThreads>>>(MoeGroupArgs{tidx, t, cnt, off, asg, tok, igu, idn, fast});
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(topk_idx, tidx, sizeof(int32_t) * A, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(topk_w, tw, sizeof(float) * A, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(count, cnt, 4 * kExperts, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(offset, off, 4 * (kExperts + 1), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(assign, asg, 4 * A, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(token, tok, 4 * A, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(item_gu, igu, 4 * (kExperts + 1), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(item_dn, idn, 4 * (kExperts + 1), hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{lg, tidx, tw, cnt, off, asg, tok, igu, idn}) HIP_CHECK(hipFree(p));
  return 0;
}

// ---- the expert down stage alone: tests/ops/test_tiles.py (bit identity of the tiled layouts), tools/qmm_bench.py ----
namespace qwen38 {
__global__ void __launch_bounds__(kBlockThreads) k_moe_relayout_iq4(RelayoutArgs a) { op_relayout_iq4nl(a, blockIdx.x, gridDim.x); }
}  // namespace qwen38

// op_moe_down over raw = all 512 experts' down matrices as stored in the GGUF (type IQ4_NL or Q8_0, [512][2560][row
// bytes]; only IQ4_NL has tiles). counts[e] rows of h ([sum counts][640], grouped by expert in expert order) go to expert e, and expert
// rows' j-th result is scattered to y[assign[j]] ([sum counts][2560], NaN where unwritten): the group stage's
// bookkeeping, built here on the host by its formulas. mode bit 0: tiles (converted here, as the engine does at load);
// bit 1: the exact f32 kernel (else the f16 WMMA path). grid 0: 1024 workgroups. reps > 0: ms per launch instead, over
// rotating expert sets (a pool bigger than the caches; counts must name experts below 10 then), y untouched.
static double moe_down_run(const char *op, uint32_t type, const void *raw, uint64_t raw_bytes, const uint32_t *counts,
                           uint64_t counts_bytes, const uint32_t *assign, uint64_t assign_bytes, const float *h_host,
                           uint64_t h_bytes, float *y_host, uint64_t y_bytes, uint32_t mode, uint32_t grid, uint32_t reps) {
  if (type != q::IQ4_NL && type != q::Q8_0) HIP_DIE("qwen38_moe_down: type %u is neither IQ4_NL nor Q8_0", type);
  if (mode > 3) HIP_DIE("qwen38_moe_down: mode %u", mode);
  HIP_CHECK_BYTES(op, "counts", counts_bytes, sizeof(uint32_t) * kExperts);  // before counts is read
  const bool tiled = mode & 1, exact = mode & 2;
  if (tiled && type != q::IQ4_NL) HIP_DIE("qwen38_moe_down: only IQ4_NL down matrices have tiles");
  const uint32_t fast = exact ? 0 : 1;
  const MoeTile tile = moe_tile(fast);
  uint32_t off[kExperts + 1], items[kExperts + 1], maxe = 0;
  off[0] = items[0] = 0;
  for (uint32_t e = 0; e < kExperts; ++e) {
    off[e + 1] = off[e] + counts[e];
    items[e + 1] = items[e] + (kMoeDnRows / tile.rows) * ((counts[e] + tile.toks - 1) / tile.toks);
    if (counts[e]) maxe = e;
  }
  const uint32_t A = off[kExperts];
  if (A == 0) HIP_DIE("qwen38_moe_down: no assignments");
  HIP_CHECK_BYTES(op, "assign", assign_bytes, sizeof(uint32_t) * A);
  HIP_CHECK_BYTES(op, "h", h_bytes, sizeof(float) * uint64_t(A) * kFf);
  HIP_CHECK_BYTES(op, "y", y_bytes, sizeof(float) * uint64_t(A) * kEmbd);
  if (reps && maxe >= 10) HIP_DIE("qwen38_moe_down bench: expert %u used, want experts below 10", maxe);
  const uint64_t ldraw = q::row_bytes(type, kFf), bytes = ldraw * kMoeDnRows * kExperts;
  HIP_CHECK_BYTES(op, "raw", raw_bytes, bytes);
  uint8_t *w;
  HIP_CHECK(hipMalloc(&w, bytes));
  HIP_CHECK(hipMemcpy(w, raw, bytes, hipMemcpyHostToDevice));
  uint32_t wtype = type;
  if (tiled) {
    uint8_t *t;
    HIP_CHECK(hipMalloc(&t, bytes));
    HIP_CHECK(hipMemset(t, 0xAB, bytes));
    const RelayoutArgs ra{w, t, kMoeDnRows * kExperts, kFf};
    k_moe_relayout_iq4<<<1024, kBlockThreads>>>(ra);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(w));
    w = t;
    wtype = q::IQ4_NL_TILES;
  }
  const uint64_t ld = q::row_bytes(wtype, kFf);
  if (ld != ldraw) HIP_DIE("qwen38_moe_down: tiles take %lu bytes a row, the GGUF %lu", (unsigned long)ld, (unsigned long)ldraw);
  float *h, *y;
  uint32_t *cnt, *ofs, *asg, *idn;
  HIP_CHECK(hipMalloc(&h, sizeof(float) * uint64_t(A) * kFf));
  HIP_CHECK(hipMemcpy(h, h_host, sizeof(float) * uint64_t(A) * kFf, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&y, sizeof(float) * uint64_t(A) * kEmbd));
  HIP_CHECK(hipMemset(y, 0xff, sizeof(float) * uint64_t(A) * kEmbd));
  HIP_CHECK(hipMalloc(&cnt, 4 * kExperts));
  HIP_CHECK(hipMemcpy(cnt, counts, 4 * kExperts, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&ofs, 4 * (kExperts + 1)));
  HIP_CHECK(hipMemcpy(ofs, off, 4 * (kExperts + 1), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&asg, 4 * A));
  HIP_CHECK(hipMemcpy(asg, assign, 4 * A, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&idn, 4 * (kExperts + 1)));
  HIP_CHECK(hipMemcpy(idn, items, 4 * (kExperts + 1), hipMemcpyHostToDevice));
  MoeMmArgs m{};
  m.which = kMmDown;
  m.f.type_down = wtype;
  m.f.ld_down = ld;
  m.f.h = h;
  m.f.y = y;
  m.f.count = cnt;
  m.f.offset = ofs;
  m.f.assign = asg;
  m.f.item_dn = idn;
  m.f.fast = fast;
  const uint32_t G = grid ? grid : 1024;
  double ms = 0;
  if (reps == 0) {
    m.f.down = w;
    k_moe_mm<<<G, kBlockThreads>>>(m);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(y_host, y, sizeof(float) * uint64_t(A) * kEmbd, hipMemcpyDeviceToHost));
  } else {
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    const uint32_t sets = (kExperts - 10) / 10;  // shifts 0, 10, ..: experts shift .. shift + 9 stay in range
    for (uint32_t i = 0; i <= reps; ++i) {
      if (i == 1) HIP_CHECK(hipEventRecord(e0));
      m.f.down = w + uint64_t(i % sets) * 10 * kMoeDnRows * ld;
      k_moe_mm<<<G, kBlockThreads>>>(m);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventSynchronize(e1));
    float f = 0.f;
    HIP_CHECK(hipEventElapsedTime(&f, e0, e1));
    ms = double(f) / reps;
    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
  }
  for (void *p : std::vector<void *>{w, h, y, cnt, ofs, asg, idn}) HIP_CHECK(hipFree(p));
  return ms;
}

extern "C" int qwen38_moe_down(uint32_t type, const void *raw, uint64_t raw_bytes, const uint32_t *counts,
                           uint64_t counts_bytes, const uint32_t *assign, uint64_t assign_bytes, const float *h,
                           uint64_t h_bytes, float *y, uint64_t y_bytes, uint32_t mode, uint32_t grid) {
  moe_down_run("qwen38_moe_down", type, raw, raw_bytes, counts, counts_bytes, assign, assign_bytes, h, h_bytes, y, y_bytes, mode,
               grid, 0);
  return 0;
}

extern "C" double qwen38_moe_down_bench(uint32_t type, const void *raw, uint64_t raw_bytes, const uint32_t *counts,
                                    uint64_t counts_bytes, const uint32_t *assign, uint64_t assign_bytes, const float *h,
                                    uint64_t h_bytes, float *y, uint64_t y_bytes, uint32_t mode, uint32_t grid,
                                    uint32_t reps) {
  if (reps == 0) HIP_DIE("qwen38_moe_down_bench: reps 0");
  return moe_down_run("qwen38_moe_down_bench", type, raw, raw_bytes, counts, counts_bytes, assign, assign_bytes, h, h_bytes, y,
                      y_bytes, mode, grid, reps);
}

// ---- the expert gate/up stage alone, for the prefetch bench (tools/moe_gu_bench.py) and tests ----
namespace qwen38 {
__global__ void __launch_bounds__(kBlockThreads) k_moe_relayout_iq3s(RelayoutArgs a) { op_relayout_iq3s(a, blockIdx.x, gridDim.x); }
}  // namespace qwen38

// op_moe_gu over raw = 512 experts' IQ3_S matrices as stored in the GGUF ([512][640][row bytes]); experts 0..255 are the
// gate, 256..511 the up matrices (a bench reads both through one pool). counts[e] tokens go to expert e (e < 256; the
// group stage's bookkeeping is built here on the host by its formulas), token map = identity over x's rows, gu rows in
// grouped order out (NaN where unwritten). mode bit 0: tiles; bit 1: the exact f32 kernel. reps > 0: ms per launch over rotating expert sets (counts name experts below 10 then), gu untouched.
static double moe_gu_run(const char *op, const void *raw, uint64_t raw_bytes, const uint32_t *counts, uint64_t counts_bytes,
                         const float *x_host, uint64_t x_bytes, uint32_t nx, float *gu_host, uint64_t gu_bytes,
                         uint32_t mode, uint32_t grid, uint32_t reps) {
  if (mode > 3) HIP_DIE("qwen38_moe_gu: mode %u", mode);
  HIP_CHECK_BYTES(op, "counts", counts_bytes, sizeof(uint32_t) * kExperts);  // before counts is read
  const bool tiled = mode & 1, exact = mode & 2;
  const uint32_t fast = exact ? 0 : 1;
  const MoeTile tile = moe_tile(fast);
  uint32_t off[kExperts + 1], items[kExperts + 1], maxe = 0;
  off[0] = items[0] = 0;
  for (uint32_t e = 0; e < kExperts; ++e) {
    if (e >= 256 && counts[e]) HIP_DIE("qwen38_moe_gu: expert %u is an up matrix of the pool", e);
    off[e + 1] = off[e] + counts[e];
    items[e + 1] = items[e] + 2 * (kMoeGuRows / tile.rows) * ((counts[e] + tile.toks - 1) / tile.toks);
    if (counts[e]) maxe = e;
  }
  const uint32_t A = off[kExperts];
  if (A == 0 || A != nx) HIP_DIE("qwen38_moe_gu: %u assignments, %u activation rows", A, nx);
  HIP_CHECK_BYTES(op, "x", x_bytes, sizeof(float) * uint64_t(nx) * kEmbd);
  HIP_CHECK_BYTES(op, "gu", gu_bytes, sizeof(float) * uint64_t(A) * 2 * kMoeGuRows);
  if (reps && maxe >= 10) HIP_DIE("qwen38_moe_gu bench: expert %u used, want experts below 10", maxe);
  const uint64_t ldraw = q::row_bytes(q::IQ3_S, kEmbd), bytes = ldraw * kMoeGuRows * kExperts;
  HIP_CHECK_BYTES(op, "raw", raw_bytes, bytes);
  uint8_t *w;
  HIP_CHECK(hipMalloc(&w, bytes));
  HIP_CHECK(hipMemcpy(w, raw, bytes, hipMemcpyHostToDevice));
  uint32_t wtype = q::IQ3_S;
  if (tiled) {
    uint8_t *t;
    const uint64_t tbytes = q::row_bytes(q::IQ3_S_TILES, kEmbd) * kMoeGuRows * kExperts;  // 112 B a row-superblock, not 110
    HIP_CHECK(hipMalloc(&t, tbytes));
    HIP_CHECK(hipMemset(t, 0xAB, tbytes));
    k_moe_relayout_iq3s<<<1024, kBlockThreads>>>(RelayoutArgs{w, t, kMoeGuRows * kExperts, kEmbd});
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(w));
    w = t;
    wtype = q::IQ3_S_TILES;
  }
  const uint64_t ld = q::row_bytes(wtype, kEmbd);
  float *x, *gu;
  uint32_t *cnt, *ofs, *tok, *igu;
  HIP_CHECK(hipMalloc(&x, sizeof(float) * uint64_t(nx) * kEmbd));
  HIP_CHECK(hipMemcpy(x, x_host, sizeof(float) * uint64_t(nx) * kEmbd, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&gu, sizeof(float) * uint64_t(A) * 2 * kMoeGuRows));
  HIP_CHECK(hipMemset(gu, 0xff, sizeof(float) * uint64_t(A) * 2 * kMoeGuRows));
  HIP_CHECK(hipMalloc(&cnt, 4 * kExperts));
  HIP_CHECK(hipMemcpy(cnt, counts, 4 * kExperts, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&ofs, 4 * (kExperts + 1)));
  HIP_CHECK(hipMemcpy(ofs, off, 4 * (kExperts + 1), hipMemcpyHostToDevice));
  std::vector<uint32_t> tokh(A);
  for (uint32_t i = 0; i < A; ++i) tokh[i] = i;
  HIP_CHECK(hipMalloc(&tok, 4 * A));
  HIP_CHECK(hipMemcpy(tok, tokh.data(), 4 * A, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&igu, 4 * (kExperts + 1)));
  HIP_CHECK(hipMemcpy(igu, items, 4 * (kExperts + 1), hipMemcpyHostToDevice));
  MoeMmArgs m{};
  m.which = kMmGu;
  m.f.type_gate = m.f.type_up = wtype;
  m.f.ld_gate = m.f.ld_up = ld;
  m.f.x = x;
  m.f.gu = gu;
  m.f.count = cnt;
  m.f.offset = ofs;
  m.f.token = tok;
  m.f.item_gu = igu;
  m.f.fast = fast;
  const uint32_t G = grid ? grid : 1024;
  double ms = 0;
  const uint64_t half = 256ull * kMoeGuRows * ld;  // the up matrices: the pool's second half
  if (reps == 0) {
    m.f.gate = w;
    m.f.up = w + half;
    k_moe_mm<<<G, kBlockThreads>>>(m);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(gu_host, gu, sizeof(float) * uint64_t(A) * 2 * kMoeGuRows, hipMemcpyDeviceToHost));
  } else {
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    const uint32_t sets = (256 - 10) / 10;
    for (uint32_t i = 0; i <= reps; ++i) {
      if (i == 1) HIP_CHECK(hipEventRecord(e0));
      m.f.gate = w + uint64_t(i % sets) * 10 * kMoeGuRows * ld;
      m.f.up = m.f.gate + half;
      k_moe_mm<<<G, kBlockThreads>>>(m);
    }
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventSynchronize(e1));
    float f = 0.f;
    HIP_CHECK(hipEventElapsedTime(&f, e0, e1));
    ms = double(f) / reps;
    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
  }
  for (void *p : std::vector<void *>{w, x, gu, cnt, ofs, tok, igu}) HIP_CHECK(hipFree(p));
  return ms;
}

extern "C" int qwen38_moe_gu(const void *raw, uint64_t raw_bytes, const uint32_t *counts, uint64_t counts_bytes, const float *x,
                         uint64_t x_bytes, uint32_t nx, float *gu, uint64_t gu_bytes, uint32_t mode, uint32_t grid) {
  moe_gu_run("qwen38_moe_gu", raw, raw_bytes, counts, counts_bytes, x, x_bytes, nx, gu, gu_bytes, mode, grid, 0);
  return 0;
}

extern "C" double qwen38_moe_gu_bench(const void *raw, uint64_t raw_bytes, const uint32_t *counts, uint64_t counts_bytes,
                                  const float *x, uint64_t x_bytes, uint32_t nx, float *gu, uint64_t gu_bytes, uint32_t mode,
                                  uint32_t grid, uint32_t reps) {
  if (reps == 0) HIP_DIE("qwen38_moe_gu_bench: reps 0");
  return moe_gu_run("qwen38_moe_gu_bench", raw, raw_bytes, counts, counts_bytes, x, x_bytes, nx, gu, gu_bytes, mode, grid, reps);
}
