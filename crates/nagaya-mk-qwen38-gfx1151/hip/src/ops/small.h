// SPDX-License-Identifier: MIT
// Row-wise and elementwise ops: RMSNorm (grouped), the hyper-connection mix / inject / combine,
// and the activations between the hyper-connection's two projections.
//
// Reference: modeling_qwen4_exp.py Qwen4ExpTextRMSNorm, Qwen4ExpTextGatedResidual and
// Qwen4ExpTextDecoderLayer.forward. Gammas are passed as stored in the GGUF, i.e. already
// 1 + w (the converter's +1), so y = x * rsqrt(mean(x^2) + eps) * gamma.
//
// Reduction order: RMSNorm's sum of squares over a group of D values is thread i summing
// elements i, i + 256, ... in order, then block_sum (fixed). The mix's mean over the 4 streams
// sums streams 0..3 in order, then multiplies by 1/4 (exact).
#pragma once
#include "common.h"

namespace mk {

// ---- RMSNorm over groups: x [T, n_groups * D] -> y (may alias x) ----
struct RmsArgs {
  const float *x;
  uint32_t ldx;
  float *y;
  uint32_t ldy;
  const float *gamma;  // [n_groups * D] or [D] when gamma_per_group == 0
  uint32_t t, n_groups, d;
  uint32_t gamma_per_group;  // 1: gamma spans all groups (hc norms); 0: one D-vector for all groups (q/k norms)
  float eps;
};

// Each thread loads its values (i = tid, tid + 256, ...) and their gammas up front, then sums and scales them in
// that order: the loop with a runtime bound issued one load per round trip (at decode, one workgroup took ~15 us for
// what is ~1 us of work while 79 waited at the next barrier). Same order, same bits.
constexpr uint32_t kRmsPerThread = 10;  // d <= 2560
// The rest of one rms item, from the thread's values: sum of squares in element order, block_sum, scale, store.
// One function for op_rms and op_hc_combine_rms, so the two can't drift apart.
__device__ __attribute__((always_inline)) inline void rms_finish(const float (&xv)[kRmsPerThread], const float (&gv)[kRmsPerThread],
                                                                 uint32_t d, float eps, float *yr, uint32_t tid, float *scratch) {
  float s = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kRmsPerThread; ++j)
    if (tid + j * kBlockThreads < d) s = fmaf(xv[j], xv[j], s);
  s = block_sum(s, scratch);
  const float r = 1.0f / sqrtf(s / float(d) + eps);
#pragma unroll
  for (uint32_t j = 0; j < kRmsPerThread; ++j) {
    const uint32_t i = tid + j * kBlockThreads;
    if (i < d) yr[i] = xv[j] * r * gv[j];
  }
}
__device__ inline void op_rms(const RmsArgs &a, uint32_t wg, uint32_t nwg) {
  __shared__ float scratch[kBlockThreads / kWave];
  if (a.d == 0 || a.d > kRmsPerThread * kBlockThreads) HIP_TRAP(kTrapBadArgs, kTrapFileSmall);
  const uint32_t items = a.t * a.n_groups, tid = mk_tid();
  for (uint32_t it = wg; it < items; it += nwg) {
    const uint32_t t = it / a.n_groups, g = it % a.n_groups;
    const float *xr = a.x + uint64_t(t) * a.ldx + uint64_t(g) * a.d;
    const float *gm = a.gamma + (a.gamma_per_group ? uint64_t(g) * a.d : 0);
    float xv[kRmsPerThread], gv[kRmsPerThread];
#pragma unroll
    for (uint32_t j = 0; j < kRmsPerThread; ++j) {  // unconditional (clamped) loads: no per-load wait
      const uint32_t i = min(tid + j * kBlockThreads, a.d - 1);
      xv[j] = xr[i];
      gv[j] = gm[i];
    }
    rms_finish(xv, gv, a.d, a.eps, a.y + uint64_t(t) * a.ldy + uint64_t(g) * a.d, tid, scratch);
  }
}

// ---- elementwise ----
__device__ inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float siluf_(float x) { return x / (1.0f + expf(-x)); }

// y[i] = silu(x[i] * scale) over n values (the hyper-connection's silu(down / hc)).
struct SiluScaleArgs {
  const float *x;
  float *y;
  uint32_t n;
  float scale;
};
__device__ inline void op_silu_scale(const SiluScaleArgs &a, uint32_t wg, uint32_t nwg) {
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < a.n; i += nwg * kBlockThreads) a.y[i] = siluf_(a.x[i] * a.scale);
}

// mixed[t, c] = (sum_s sigmoid(up[t, s, c]) * xn[t, s, c]) / 4 over the 4 streams, in stream order;
// inject[t, s] = 2 * sigmoid(z[t, s] / 4) (z = xn . block_inject_weight, 4 per token).
struct HcMixArgs {
  const float *xn;  // [T, 4 * E]
  const float *up;  // [T, 4 * E]
  float *mixed;     // [T, E]
  const float *z;   // [T, 4] or null (the final mixer has no inject)
  float *inject;    // [T, 4]
  uint32_t t, e;
};
// One element of hc_mix, shared by op_hc_mix and the hc up matmul's fused epilogue (qmmh.h, QmmArgs::yout) so the two
// can't drift: mixed = (sum_s sigmoid(up[s]) * xn[s]) * 0.25 with the streams added in order 0..3 from 0, and the
// inject of one z. (The build has -ffp-contract=off, so neither site contracts.)
__device__ __attribute__((always_inline)) inline float hc_mix_elem(const float up[4], const float xn[4]) {
  float s = 0.f;
#pragma unroll
  for (uint32_t k = 0; k < 4; ++k) s += sigmoidf_(up[k]) * xn[k];
  return s * 0.25f;
}
__device__ __attribute__((always_inline)) inline float hc_inject_elem(float z) { return 2.0f * sigmoidf_(z * 0.25f); }

__device__ inline void op_hc_mix(const HcMixArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t n = a.t * a.e;
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < n; i += nwg * kBlockThreads) {
    const uint32_t t = i / a.e, c = i % a.e;
    const uint64_t base = uint64_t(t) * 4 * a.e + c;
    float up[4], xn[4];
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) up[k] = a.up[base + k * a.e], xn[k] = a.xn[base + k * a.e];
    a.mixed[i] = hc_mix_elem(up, xn);
    if (a.z != nullptr && c < 4) a.inject[t * 4 + c] = hc_inject_elem(a.z[t * 4 + c]);
  }
}

// h[t, s, c] += out[t, c] * inject[t, s]  (Qwen4ExpTextDecoderLayer: hyper_input + injection).
struct HcCombineArgs {
  float *h;            // [T, 4 * E], updated in place
  const float *out;    // [T, E]
  const float *inject; // [T, 4]
  uint32_t t, e;
};
__device__ inline void op_hc_combine(const HcCombineArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t n = a.t * 4 * a.e;
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < n; i += nwg * kBlockThreads) {
    const uint32_t t = i / (4 * a.e), r = i % (4 * a.e), s = r / a.e, c = r % a.e;
    a.h[i] = a.h[i] + a.out[uint64_t(t) * a.e + c] * a.inject[t * 4 + s];
  }
}

// hc_combine then the rms of the h it just wrote (a layer's hc_post and the next hc_pre's norm), as one stage whose item is
// one (token, stream group): the group's E = d elements of h are exactly what one op_rms item reads, so the item combines
// them (same expression as op_hc_combine, one thread per element i = tid + j * 256), stores h, and reduces the same values
// in registers through rms_finish, op_rms's own tail. Both outputs are bit-identical to the two ops in a row (h needs
// no other item's combine: the combine is elementwise and the item owns every element it reads).
struct HcCombineRmsArgs {
  HcCombineArgs c;  // h [T, 4 * e], out [T, e], inject [T, 4]
  RmsArgs r;        // x == c.h, y = xn, gamma [4 * e] per group, t == c.t, n_groups 4, d == c.e
};
__device__ inline void op_hc_combine_rms(const HcCombineRmsArgs &a, uint32_t wg, uint32_t nwg) {
  __shared__ float scratch[kBlockThreads / kWave];
  const uint32_t e = a.c.e;
  if (e == 0 || e > kRmsPerThread * kBlockThreads) HIP_TRAP(kTrapBadArgs, kTrapFileSmall);
  const uint32_t items = a.c.t * 4, tid = mk_tid();
  for (uint32_t it = wg; it < items; it += nwg) {
    const uint32_t t = it / 4, g = it % 4;
    float *hr = a.c.h + uint64_t(t) * 4 * e + uint64_t(g) * e;
    const float *gm = a.r.gamma + uint64_t(g) * e;
    float xv[kRmsPerThread], gv[kRmsPerThread];
#pragma unroll
    for (uint32_t j = 0; j < kRmsPerThread; ++j) {  // clamped loads as op_rms's
      const uint32_t c = min(tid + j * kBlockThreads, e - 1);
      xv[j] = hr[c] + a.c.out[uint64_t(t) * e + c] * a.c.inject[t * 4 + g];
      gv[j] = gm[c];
    }
#pragma unroll
    for (uint32_t j = 0; j < kRmsPerThread; ++j)
      if (tid + j * kBlockThreads < e) hr[tid + j * kBlockThreads] = xv[j];
    rms_finish(xv, gv, e, a.r.eps, a.r.y + uint64_t(t) * a.r.ldy + uint64_t(g) * e, tid, scratch);
  }
}

}  // namespace mk
