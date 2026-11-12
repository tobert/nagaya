// SPDX-License-Identifier: MIT
// The oracle for qmm.h: run 2's quantized matmul, frozen (commit 639afd4), kept only for the
// bit-identity tests (tests/ops/test_qmm.py). Its reduction order is the op's definition:
// the K dimension is cut into 32-value units; lane l of the wave that owns row n takes units
// l, l + 32, l + 64, ... in that order, and within a unit values 0..31, accumulating with fmaf
// into one f32 per token; the 32 lane partials are summed by wave_sum's fixed butterfly.
// A work item is 8 rows x 8 tokens. It decodes with q::dequant32, independent of the engine
// kernel's decoder. Do not optimize this file: its job is to stay what it was.
#pragma once
#include "qmm.h"

namespace mk::oracle {

constexpr uint32_t kQmmRows = kMkBlock / kWave;  // 8, its own: the engine's tile may differ
constexpr uint32_t kQmmToks = 8;

template <uint32_t T>
__device__ inline void qmm_item(const QmmArgs &a, uint32_t item) {
  const uint32_t n_rg = (a.n + kQmmRows - 1) / kQmmRows;
  const uint32_t rg = item % n_rg, tt = item / n_rg;
  const uint32_t n = rg * kQmmRows + mk_tid() / kWave;
  const uint32_t lane = mk_tid() % kWave;
  const uint32_t t0 = tt * kQmmToks;
  const uint32_t nt = min(kQmmToks, a.t - t0);
  if (n >= a.n) return;
  const uint8_t *wrow = a.w + uint64_t(n) * a.ldw;
  const float *xr[kQmmToks];
#pragma unroll
  for (uint32_t t = 0; t < kQmmToks; ++t) {
    const uint32_t tok = t0 + (t < nt ? t : 0);
    xr[t] = a.x + uint64_t(a.x_row ? a.x_row[tok] : tok) * a.ldx;
  }
  float acc[kQmmToks] = {};
  float w[32];
  for (uint32_t u = lane; u < a.k / 32; u += kWave) {
    q::dequant32<T>(wrow, u, w);
#pragma unroll
    for (uint32_t t = 0; t < kQmmToks; ++t) {
      if (t < nt) {
        const float *xp = xr[t] + u * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) acc[t] = fmaf(w[j], xp[j], acc[t]);
      }
    }
  }
#pragma unroll
  for (uint32_t t = 0; t < kQmmToks; ++t) {
    const float s = wave_sum(acc[t]);
    if (t < nt && lane == 0) {
      const uint32_t tok = t0 + t;
      a.y[uint64_t(a.y_row ? a.y_row[tok] : tok) * a.ldy + n] = s;
    }
  }
}

__device__ inline uint32_t qmm_items(const QmmArgs &a) {
  return ((a.n + kQmmRows - 1) / kQmmRows) * ((a.t + kQmmToks - 1) / kQmmToks);
}

template <uint32_t T>
__device__ inline void op_qmm_t(const QmmArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t items = oracle::qmm_items(a);
  for (uint32_t i = wg; i < items; i += nwg) oracle::qmm_item<T>(a, i);
}

__device__ inline void op_qmm(const QmmArgs &a, uint32_t wg, uint32_t nwg) {
  switch (a.type) {
    case q::Q8_0: oracle::op_qmm_t<q::Q8_0>(a, wg, nwg); break;
    case q::Q6_K: oracle::op_qmm_t<q::Q6_K>(a, wg, nwg); break;
    case q::IQ4_NL: oracle::op_qmm_t<q::IQ4_NL>(a, wg, nwg); break;
    case q::IQ4_XS: oracle::op_qmm_t<q::IQ4_XS>(a, wg, nwg); break;
    case q::IQ3_S: oracle::op_qmm_t<q::IQ3_S>(a, wg, nwg); break;
    case q::F32: oracle::op_qmm_t<q::F32>(a, wg, nwg); break;
    case q::BF16: oracle::op_qmm_t<q::BF16>(a, wg, nwg); break;
    default: HIP_TRAP(kTrapBadOp, kTrapFileQmmOracle);
  }
}

}  // namespace mk::oracle
