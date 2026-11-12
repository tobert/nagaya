// SPDX-License-Identifier: MIT
// Small ops the engine needs around the model's blocks: token embedding, the ×4 stream repeat,
// and the PLE block (modeling_qwen4_exp.py Qwen4ExpTextPLELayer.forward). Same conventions as
// src/ops/common.h.
#pragma once
#include "common.h"
#include "model.h"
#include "quant.h"
#include "small.h"

namespace mk {

// out[t, :] = dequant(W row ids[t]) for a Q8_0 table with K columns. Item = one 32-value unit.
struct EmbedArgs {
  const uint8_t *w;
  uint64_t ldw;
  uint32_t k;
  const int32_t *ids;
  uint32_t t;
  float *out;  // [T, K]
};
__device__ inline void op_embed(const EmbedArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t units = a.k / 32, n = a.t * units;
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < n; i += nwg * kBlockThreads) {
    const uint32_t t = i / units, u = i % units;
    float v[32];
    q::dequant32<q::Q8_0>(a.w + uint64_t(a.ids[t]) * a.ldw, u, v);
    float *o = a.out + uint64_t(t) * a.k + u * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) o[j] = v[j];
  }
}

// h[t, s * E + c] = e[t, c] for s in 0..3 (Qwen4ExpTextModel: inputs_embeds.repeat(1, 1, hc)).
struct Repeat4Args {
  const float *e;
  float *h;
  uint32_t t, d;
};
__device__ inline void op_repeat4(const Repeat4Args &a, uint32_t wg, uint32_t nwg) {
  const uint32_t n = a.t * 4 * a.d;
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < n; i += nwg * kBlockThreads) {
    const uint32_t t = i / (4 * a.d), c = i % a.d;
    a.h[i] = a.e[uint64_t(t) * a.d + c];
  }
}

// PLE gate. Item = (t, stream s): g = sum_c kn[t,s,c] * qn[t,s,c] / sqrt(E) (block_sum order),
// g = sqrt(max(|g|, 1e-6)) * sign(g), gv[t,s,c] = sigmoid(g) * pv[t,c].
struct PleGateArgs {
  const float *kn, *qn;  // [T, 4E]
  const float *pv;       // [T, E]
  float *gv;             // [T, 4E]
  uint32_t t, e;
};
__device__ inline void op_ple_gate(const PleGateArgs &a, uint32_t wg, uint32_t nwg) {
  __shared__ float scratch[kBlockThreads / kWave];
  for (uint32_t it = wg; it < a.t * 4; it += nwg) {
    const uint32_t t = it / 4, s = it % 4;
    const uint64_t base = uint64_t(t) * 4 * a.e + uint64_t(s) * a.e;
    float acc = 0.f;
    for (uint32_t c = mk_tid(); c < a.e; c += kBlockThreads) acc = fmaf(a.kn[base + c], a.qn[base + c], acc);
    float g = block_sum(acc, scratch) / sqrtf(float(a.e));
    const float sg = g > 0.f ? 1.f : (g < 0.f ? -1.f : 0.f);
    g = sqrtf(fmaxf(fabsf(g), 1e-6f)) * sg;
    const float gate = sigmoidf_(g);
    for (uint32_t c = mk_tid(); c < a.e; c += kBlockThreads) a.gv[base + c] = gate * a.pv[uint64_t(t) * a.e + c];
  }
}

// PLE short conv: depthwise, kernel 4, dilation 3, causal, then silu, over [state (9 rows); x].
// Item = channel c: y[t,c] = silu(sum_j w[c,j] * xp[t + 3j]) with j = 0..3 in order; then the
// channel's new state is the last 9 rows of xp. w is the GGUF ple_conv1d [C, 4].
struct PleConvArgs {
  const float *x;     // [T, C] normed gated values
  float *state;       // [9, C], read then overwritten
  const float *w;     // [C, 4]
  float *y;           // [T, C]
  uint32_t t, c;
};
__device__ inline void op_ple_conv(const PleConvArgs &a, uint32_t wg, uint32_t nwg) {
  constexpr uint32_t S = 9;
  for (uint32_t c = wg * kBlockThreads + mk_tid(); c < a.c; c += nwg * kBlockThreads) {
    auto xp = [&](uint32_t i) -> float {  // padded index: 0..8 state, 9.. new rows
      return i < S ? a.state[uint64_t(i) * a.c + c] : a.x[uint64_t(i - S) * a.c + c];
    };
    const float w0 = a.w[c * 4 + 0], w1 = a.w[c * 4 + 1], w2 = a.w[c * 4 + 2], w3 = a.w[c * 4 + 3];
    for (uint32_t t = 0; t < a.t; ++t) {
      float s = w0 * xp(t);
      s = fmaf(w1, xp(t + 3), s);
      s = fmaf(w2, xp(t + 6), s);
      s = fmaf(w3, xp(t + 9), s);
      a.y[uint64_t(t) * a.c + c] = siluf_(s);
    }
    float keep[S];
    for (uint32_t i = 0; i < S; ++i) keep[i] = xp(a.t + i);
    for (uint32_t i = 0; i < S; ++i) a.state[uint64_t(i) * a.c + c] = keep[i];
  }
}

// h[i] = h[i] + (a[i] + b[i])  (the PLE's hidden + (gated_value + conv)).
struct AddArgs {
  float *h;
  const float *a, *b;
  uint32_t n;
};
__device__ inline void op_add(const AddArgs &a, uint32_t wg, uint32_t nwg) {
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < a.n; i += nwg * kBlockThreads) a.h[i] = a.h[i] + (a.a[i] + a.b[i]);
}

}  // namespace mk
