// SPDX-License-Identifier: MIT
// The gated DeltaNet (GDN) core of the 36 linear-attention layers: causal depthwise conv + silu
// over the qkv channels, the gated delta rule (a per-token recurrence over a [K, V] state per
// value head), then a sigmoid-gated RMSNorm. Everything here works in GGUF (tiled) V-head order
// throughout: GGUF value head p uses key/query head p % 16 (see model.h and tests/ops/test_gdn.py
// for the derivation). Reference: reference/transformers-5.17.0-qwen4_exp/modeling_qwen4_exp.py
// `Qwen4ExpTextGatedDeltaNet.forward`, `causal_conv1d_fn`, `torch_recurrent_gated_delta_rule`,
// `l2norm`, `Qwen4ExpTextRMSNormGated` (gate = sigmoid here, config `output_gate_type`).
//
// Two ops, two launches (op_gdn_conv must finish before op_gdn_core starts; the C ABI launches
// them on the same stream, so the runtime's own ordering is enough — no grid barrier needed).
//
// op_gdn_conv: item = channel (0..10239), one thread per item (no reduction, like op_silu_scale).
// For each channel, the causal window is exactly 4 taps of the length-(T+3) sequence
// [conv_state_in (3 old); x (T new)]; weight index j pairs with tap j of that window (no flip:
// causal_conv1d_fn's F.conv1d is cross-correlation). The new conv_state_out is the window's last
// 3 taps: a copy, never an arithmetic result, so it is exact (no rounding at all).
//
// op_gdn_core: item = value head p (0..47), one workgroup (kBlockThreads = 256 threads) per item. The head's
// [128 (k), 128 (v)] recurrent state lives in registers, 64 floats per thread: thread t owns columns
// c = t / 4 and c + 64, and within them the k-slice k = 4i + j (i = 0..31, j = t % 4), so the 4 lanes of a
// quad together hold two whole columns. (Run 2's kernel gave each thread a whole column, 128 floats; that
// op alone set the megakernel's registers to 256 with 1.3 KiB/lane of spills, PLAN.md Speed.)
// Per token, in the op's fixed order: l2norm q,k (over the whole 128-dim head vector: threads 0..127
// square q's components, 128..255 k's, each a wave xor-butterfly, then the half's 4 waves added in order,
// eps 1e-6, exactly like the reference's l2norm) and scale q by 1/sqrt(128); beta = sigmoid(b);
// g = ssm_a * softplus(a + dt_bias, threshold 20, like torch); S *= exp(g); kv = S^T k; delta = (v - kv) *
// beta; S += k (x) delta; out = S^T q; y = rms_normalize(out) * gamma * sigmoid(z). Each dot product over k
// is a 32-long fmaf chain per lane (i = 0..31) and then the fixed quad sum (x0 + x1) + (x2 + x3), which
// gives all 4 lanes the same bits (f32 addition commutes). The RMSNorm's sum of squares: lane j = 0 of each
// quad adds o_c^2 + o_{c+64}^2, a wave xor-butterfly, then the 8 waves in order. Three __syncthreads per
// token (after the l2norm partials, after the normalized q/k, after the o^2 partials); every thread reaches
// each one every iteration, and between any write and the next iteration's rewrite of the same shared
// array there are three of them (also across a workgroup's second head), so no extra fence is needed.
#pragma once
#include "common.h"
#include "model.h"
#include "small.h"  // sigmoidf_, siluf_

namespace mk {

static_assert(model::kGdnKDim == 128 && model::kGdnVDim == 128, "gdn.h assumes 128-wide K/V heads");
static_assert(model::kGdnKHeads == 16 && model::kGdnVHeads == 48, "gdn.h assumes 16 K-heads / 48 V-heads");

// ---- stage 1: causal depthwise conv + silu ----
struct GdnConvArgs {
  const float *x;          // [t, kGdnConvDim], pre-conv qkv (row-major, stride kGdnConvDim)
  const float *state_in;   // [kGdnConvDim, kGdnConv - 1], previous 3 raw inputs per channel
  const float *w;          // [kGdnConvDim, kGdnConv]
  float *y;                 // [t, kGdnConvDim], post conv + silu
  float *state_out;          // [kGdnConvDim, kGdnConv - 1], new (raw, pre-silu) trailing 3 inputs
  uint32_t t;
};

// Tap i (0 <= i <= t + kGdnConv - 2) of the extended per-channel sequence [state_in; x]: a copy,
// never a sum, so callers that only copy this value (state_out) get exact results.
__device__ inline float gdn_conv_ext(const GdnConvArgs &a, uint32_t c, uint32_t i) {
  constexpr uint32_t kState = model::kGdnConv - 1;  // 3
  return i < kState ? a.state_in[c * kState + i] : a.x[uint64_t(i - kState) * model::kGdnConvDim + c];
}

__device__ inline void gdn_conv_item(const GdnConvArgs &a, uint32_t c) {
  const float *wc = a.w + uint64_t(c) * model::kGdnConv;
  for (uint32_t t = 0; t < a.t; ++t) {
    float acc = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < model::kGdnConv; ++j) acc = fmaf(wc[j], gdn_conv_ext(a, c, t + j), acc);
    a.y[uint64_t(t) * model::kGdnConvDim + c] = siluf_(acc);
  }
#pragma unroll
  for (uint32_t j = 0; j < model::kGdnConv - 1; ++j)
    a.state_out[c * (model::kGdnConv - 1) + j] = gdn_conv_ext(a, c, a.t + j);
}

__device__ inline void op_gdn_conv(const GdnConvArgs &a, uint32_t wg, uint32_t nwg) {
  for (uint32_t c = wg * kBlockThreads + mk_tid(); c < model::kGdnConvDim; c += nwg * kBlockThreads) gdn_conv_item(a, c);
}

// ---- stage 2/3/4: l2norm + the gated delta rule + the gated RMSNorm, fused per value head ----
struct GdnCoreArgs {
  const float *qkv;      // [t, kGdnConvDim], post conv + silu (q | k | v, v tiled)
  const float *z;          // [t, kGdnValDim], tiled
  const float *a;           // [t, kGdnVHeads], tiled
  const float *b;            // [t, kGdnVHeads], tiled
  const float *ssm_a;          // [kGdnVHeads], tiled, already -exp(A_log)
  const float *dt_bias;          // [kGdnVHeads], tiled
  const float *gamma;              // [kGdnVDim] (128), shared by every head, not +1 shifted
  const float *state_in;             // [kGdnVHeads, kGdnKDim, kGdnVDim]
  float *y;                            // [t, kGdnValDim], tiled
  float *state_out;                     // [kGdnVHeads, kGdnKDim, kGdnVDim]
  uint32_t t;
};

// torch.nn.functional.softplus's default (beta=1, threshold=20): identity above the threshold so
// exp() never overflows, log1p(exp(x)) below it.
__device__ inline float gdn_softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

constexpr uint32_t kGdnQuad = 4;                                 // lanes per pair of S columns
constexpr uint32_t kGdnKPerLane = model::kGdnKDim / kGdnQuad;    // 32 state rows per lane and column
constexpr uint32_t kGdnColHalf = model::kGdnVDim / 2;            // a thread's columns: c and c + 64
static_assert(kBlockThreads == kGdnQuad * kGdnColHalf, "gdn core: one head per 256-thread workgroup");
static_assert(kBlockThreads == 2 * model::kGdnKDim, "gdn core: threads 0..127 take q's l2norm, 128..255 k's");

// (x0 + x1) + (x2 + x3) over a quad; every lane of the quad gets the same bits.
__device__ inline float gdn_quad_sum(float x) {
  x += __shfl_xor(x, 1, kWave);
  x += __shfl_xor(x, 2, kWave);
  return x;
}

__device__ inline void gdn_core_item(const GdnCoreArgs &a, uint32_t p) {
  __shared__ float qkL[2][model::kGdnKDim];   // [0] = normalized, scaled q; [1] = normalized k
  __shared__ float red_qk[kBlockThreads / kWave];  // per-wave sums of squares: waves 0..3 q, 4..7 k
  __shared__ float red_o[kBlockThreads / kWave];   // per-wave sums of o^2

  const uint32_t tid = mk_tid(), w = tid / kWave;
  const uint32_t j = tid % kGdnQuad, c = tid / kGdnQuad;
  const uint32_t half = tid / model::kGdnKDim, comp = tid % model::kGdnKDim;  // l2norm role
  const uint32_t col = c + (j & 1) * kGdnColHalf;  // the column this lane writes y for (lanes j = 0, 1)
  const uint32_t kh = p % model::kGdnKHeads;       // GGUF value head p uses key/query head p % 16
  const uint64_t qkoff = uint64_t(half) * model::kGdnKeyDim + uint64_t(kh) * model::kGdnKDim;
  const uint64_t voff = 2 * uint64_t(model::kGdnKeyDim) + uint64_t(p) * model::kGdnVDim;
  const uint64_t zoff = uint64_t(p) * model::kGdnVDim;
  const float g_scale = a.ssm_a[p];
  const float dtb = a.dt_bias[p];
  const float gamma_col = a.gamma[col];
  const float kQScale = rsqrtf(float(model::kGdnKDim));  // 1 / sqrt(128), reference's `query.shape[-1] ** -0.5`
  const float *st_in = a.state_in + uint64_t(p) * model::kGdnKDim * model::kGdnVDim;

  float s0[kGdnKPerLane], s1[kGdnKPerLane];  // S[4i + j, c], S[4i + j, c + 64]
#pragma unroll
  for (uint32_t i = 0; i < kGdnKPerLane; ++i) {
    s0[i] = st_in[(kGdnQuad * i + j) * model::kGdnVDim + c];
    s1[i] = st_in[(kGdnQuad * i + j) * model::kGdnVDim + c + kGdnColHalf];
  }

  for (uint32_t t = 0; t < a.t; ++t) {
    const float *row = a.qkv + uint64_t(t) * model::kGdnConvDim;
    const float x = row[qkoff + comp];
    const float sq = wave_sum(x * x);
    if (tid % kWave == 0) red_qk[w] = sq;
    __syncthreads();
    const uint32_t r = half * (model::kGdnKDim / kWave);
    const float ss = ((red_qk[r] + red_qk[r + 1]) + red_qk[r + 2]) + red_qk[r + 3];
    const float xn = x * rsqrtf(ss + 1e-6f);
    qkL[half][comp] = half ? xn : xn * kQScale;
    __syncthreads();

    const float v0 = row[voff + c], v1 = row[voff + c + kGdnColHalf];
    const float beta_t = sigmoidf_(a.b[uint64_t(t) * model::kGdnVHeads + p]);
    const float g_t = g_scale * gdn_softplus(a.a[uint64_t(t) * model::kGdnVHeads + p] + dtb);
    const float decay = expf(g_t);

    float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < kGdnKPerLane; ++i) {
      const float kk = qkL[1][kGdnQuad * i + j];
      s0[i] *= decay;
      s1[i] *= decay;
      kv0 = fmaf(s0[i], kk, kv0);
      kv1 = fmaf(s1[i], kk, kv1);
    }
    kv0 = gdn_quad_sum(kv0);
    kv1 = gdn_quad_sum(kv1);
    const float d0 = (v0 - kv0) * beta_t, d1 = (v1 - kv1) * beta_t;
    // Re-read k from LDS below instead of keeping the 32 values from the loop above live (register budget,
    // test_gdn_core_register_budget): the empty asm's memory clobber stops the compiler reusing them.
    asm volatile("" ::: "memory");

    float o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < kGdnKPerLane; ++i) {
      const float kk = qkL[1][kGdnQuad * i + j], qq = qkL[0][kGdnQuad * i + j];
      s0[i] = fmaf(kk, d0, s0[i]);
      s1[i] = fmaf(kk, d1, s1[i]);
      o0 = fmaf(s0[i], qq, o0);
      o1 = fmaf(s1[i], qq, o1);
    }
    o0 = gdn_quad_sum(o0);
    o1 = gdn_quad_sum(o1);

    const float osq = wave_sum(j == 0 ? o0 * o0 + o1 * o1 : 0.f);
    if (tid % kWave == 0) red_o[w] = osq;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < kBlockThreads / kWave; ++i) tot += red_o[i];
    if (j < 2) {
      const float o = j ? o1 : o0;
      const float rms = o * rsqrtf(tot / float(model::kGdnVDim) + model::kRmsEps);
      const float z_t = a.z[uint64_t(t) * model::kGdnValDim + zoff + col];
      a.y[uint64_t(t) * model::kGdnValDim + zoff + col] = rms * gamma_col * sigmoidf_(z_t);
    }
  }

  float *st_out = a.state_out + uint64_t(p) * model::kGdnKDim * model::kGdnVDim;
#pragma unroll
  for (uint32_t i = 0; i < kGdnKPerLane; ++i) {
    st_out[(kGdnQuad * i + j) * model::kGdnVDim + c] = s0[i];
    st_out[(kGdnQuad * i + j) * model::kGdnVDim + c + kGdnColHalf] = s1[i];
  }
}

// Item = one value head.
__device__ inline void op_gdn_core(const GdnCoreArgs &a, uint32_t wg, uint32_t nwg) {
  for (uint32_t p = wg; p < model::kGdnVHeads; p += nwg) {
    gdn_core_item(a, p);
  }
}

}  // namespace mk
