// SPDX-License-Identifier: MIT
// y[t, n] = sum_k W[n, k] * x[t, k], W rows in a GGUF block format, x and y f32.
//
// Reduction order (the op's definition, independent of the grid): the K dimension is cut into
// 32-value units; lane l of the wave that owns row n takes units l, l + 32, l + 64, ... in that
// order, and within a unit values 0..31, accumulating with fmaf into one f32 per token. The 32
// lane partials are then summed by wave_sum's fixed butterfly.
//
// A work item is 8 rows x 8 tokens: wave w of the workgroup takes row 8 * rg + w. Optional row
// maps gather x rows (x_row[t]) and scatter y rows (y_row[t]) for the MoE.
//
// This is run 2's kernel, same as qmm_oracle.h (PLAN.md "Speed" S1). Attempt 2 (commit 6ee7d17: 32 rows x 8
// tokens, activations staged in LDS) made the unfused read 1.5x faster but the megakernel read 22% slower, so
// the engine keeps this one until attempt 3 (double-buffered staging) wins in the megakernel.
#pragma once
#include "common.h"
#include "quant.h"
#include "tiles.h"

namespace mk {

struct QmmArgs {
  uint32_t type;
  const uint8_t *w;   // N rows of row_bytes(type, K)
  uint64_t ldw;       // bytes between rows
  uint32_t n, k, t;   // rows, reduction length, tokens
  const float *x;     // x row t at x + (x_row ? x_row[t] : t) * ldx
  uint32_t ldx;
  const uint32_t *x_row;
  float *y;           // y row t at y + (y_row ? y_row[t] : t) * ldy, column n
  uint32_t ldy;
  const uint32_t *y_row;
  uint32_t wbulk;     // a bulky weight matrix (>= 8 MiB, or an expert's): the f16 path's weight-policy experiment
  uint32_t ksplit;    // f16 path only: K slices (0 or 1 = whole). > 1: items write partials to `part`, and a
  float *part;        // kStQmmKsum stage adds them into y ([ksplit][t][n], qmmh.h op_qmm_ksum)
  // f16 path only: how x is staged (qmmh.h qmmh_x4). kXinPlain: x itself. Otherwise x is computed from the split-K
  // partials ([xks][t][k]) of the matmul(s) that produce it, by the same expressions as the stages it replaces
  // (op_qmm_ksum's sum, then op_silu_scale or op_swiglu), so the staged values have the same bits.
  uint32_t xin;
  uint32_t xks;
  const float *xp, *xp2;  // partials; xp2: the swiglu's up
  float xscale;           // kXinSiluSum's scale
  // f16 path, Q8_0 / Q8_0_TILES, no split-K, no row maps: what happens to the finished accumulators. kYoutPlain: they are
  // stored to y. kYoutHcMix (docs/narrow.md N1): the matmul is the hyper-connection's up projection (n = 4 * mix_e) and
  // its epilogue is hc_mix: the item's row map becomes n = stream * mix_e + rg * 16 + r (qmmh.h qmmh_row), the four
  // streams' accumulators meet in LDS, and mix_out[t, c] = hc_mix_elem(up, mix_xn) and, from the item with rg == 0, mix_inject
  // [t, j] = hc_inject_elem(mix_z[t, j]) (mix_z null: no inject) are stored; y is not written.
  uint32_t yout;
  const float *mix_xn, *mix_z;  // [t, 4 * mix_e], [t, 4]
  float *mix_out, *mix_inject;  // [t, mix_e], [t, 4]
  uint32_t mix_e;
};
constexpr uint32_t kXinPlain = 0, kXinSiluSum = 1, kXinSwigluSum = 2;
constexpr uint32_t kYoutPlain = 0, kYoutHcMix = 1;

constexpr uint32_t kQmmRows = kMkBlock / kWave;  // 8
constexpr uint32_t kQmmToks = 8;


// The exact F32/BF16 stages at one token (router logits, hyper-connection inject, GDN alpha/beta, indexer) are latency
// bound: a lane owns at most NU units, each a dependent load -> 32 fmaf chain, so the row costs NU serialized memory round
// trips. This schedule keeps the arithmetic of qmm_item exactly (lane l takes units l, l + 32, ... in order, values 0..31
// in order, fmaf into one f32, then wave_sum's butterfly) and changes only when the loads happen: (1) for the long chain
// (NU = 10) every lane first touches the cache line of each of its later units' weights with loads it never uses,
// so all the round trips overlap and the later real loads hit L2; (2) the real loads are wide (float4 / uint4) and unconditional, clamped to the
// row's last unit, and the arithmetic of a unit a lane doesn't own is skipped by a branch, never fed clamped data.
template <uint32_t T>
struct NarrowLd;
template <>
struct NarrowLd<q::F32> {
  static constexpr uint32_t kUnitBytes = 128;
  static __device__ inline void w(const uint8_t *row, uint32_t u, float o[32]) {
    const float4 *p = reinterpret_cast<const float4 *>(row + uint64_t(u) * kUnitBytes);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 v = p[i];
      o[4 * i] = v.x, o[4 * i + 1] = v.y, o[4 * i + 2] = v.z, o[4 * i + 3] = v.w;
    }
  }
};
template <>
struct NarrowLd<q::BF16> {
  static constexpr uint32_t kUnitBytes = 64;
  static __device__ inline void w(const uint8_t *row, uint32_t u, float o[32]) {
    const uint4 *p = reinterpret_cast<const uint4 *>(row + uint64_t(u) * kUnitBytes);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = p[i];
      const uint32_t d[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        o[8 * i + 2 * e] = __uint_as_float(d[e] << 16);          // element 2e: the dword's low half
        o[8 * i + 2 * e + 1] = __uint_as_float(d[e] & 0xffff0000u);  // element 2e + 1: its high half
      }
    }
  }
};

template <uint32_t T, uint32_t NU>
__device__ inline float qmm_narrow_sum(const uint8_t *wrow, const float *x, uint32_t units, uint32_t lane) {
  using L = NarrowLd<T>;
  float wv[32], xv[32];
  // Touch loads: one per later unit's weight line, clamped to the last unit (a repeat is harmless). Only for the long
  // chains (the inject, NU = 10, one work item): 4 waves have nothing else to do and the chain is pure latency. On
  // the router (NU = 3, 64 items) the extra requests cost ~10% (bandwidth-bound), and touching x never paid.
  constexpr uint32_t kTouch = NU > 3 ? NU - 1 : 0;
  float touch[kTouch + 1];
#pragma unroll
  for (uint32_t j = 1; j <= kTouch; ++j) {
    const uint32_t u = min(lane + 32 * j, units - 1);
    touch[j - 1] = *reinterpret_cast<const float *>(wrow + uint64_t(u) * L::kUnitBytes);
  }
  float acc = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < NU; ++j) {
    const uint32_t u = min(lane + 32 * j, units - 1);
    L::w(wrow, u, wv);
    const float4 *xp = reinterpret_cast<const float4 *>(x + uint64_t(u) * 32);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 v = xp[i];
      xv[4 * i] = v.x, xv[4 * i + 1] = v.y, xv[4 * i + 2] = v.z, xv[4 * i + 3] = v.w;
    }
    if (lane + 32 * j < units) {
#pragma unroll
      for (int e = 0; e < 32; ++e) acc = fmaf(wv[e], xv[e], acc);
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < kTouch; ++i) asm volatile("" ::"v"(touch[i]));
  return acc;
}

template <uint32_t T>
__device__ inline void qmm_item(const QmmArgs &a, uint32_t item) {
  const uint32_t n_rg = (a.n + kQmmRows - 1) / kQmmRows;
  const uint32_t rg = item % n_rg, tt = item / n_rg;
  const uint32_t n = rg * kQmmRows + mk_tid() / kWave;
  const uint32_t lane = mk_tid() % kWave;
  const uint32_t t0 = tt * kQmmToks;
  const uint32_t nt = min(kQmmToks, a.t - t0);
  if (a.xin != kXinPlain) HIP_TRAP(kTrapBadArgs, kTrapFileQmm);  // fused staging is the f16 path's (the engine fuses only there)
  if (n >= a.n) return;
  const uint8_t *wrow = a.w + uint64_t(n) * a.ldw;
  const float *xr[kQmmToks];
#pragma unroll
  for (uint32_t t = 0; t < kQmmToks; ++t) {
    const uint32_t tok = t0 + (t < nt ? t : 0);
    xr[t] = a.x + uint64_t(a.x_row ? a.x_row[tok] : tok) * a.ldx;
  }
  if constexpr (T == q::F32 || T == q::BF16) {
    if (nt == 1) {
      const uint32_t units = a.k / 32, nu = (units + kWave - 1) / kWave;
      if (nu == 3 || nu == 10) {  // the engine's K = 2560 and 10240
        const float *x = xr[0];
        if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(wrow)) & 15) HIP_TRAP(kTrapBadArgs, kTrapFileQmm);  // wide loads
        const float s = wave_sum(nu == 3 ? qmm_narrow_sum<T, 3>(wrow, x, units, lane) : qmm_narrow_sum<T, 10>(wrow, x, units, lane));
        if (lane == 0) a.y[uint64_t(a.y_row ? a.y_row[t0] : t0) * a.ldy + n] = s;
        return;
      }
    }
  }
  float acc[kQmmToks] = {};
  float w[32];
  for (uint32_t u = lane; u < a.k / 32; u += kWave) {
    if constexpr (T == q::IQ3_S_TILES) {
      // Unit u of row n from its read-once tile (tiles.h): the same values, bit for bit, as dequant32<IQ3_S> on the
      // GGUF row (tests/ops/test_tiles.py), so this kernel's results don't depend on the layout.
      const uint32_t ib = u % 8;
      Iq3sLane L;
      L.load(a.w + (uint64_t(n / 16) * (a.k / 256) + u / 8) * kIq3sTile, n % 16 + 16 * (ib % 2));
      int c[32];
      L.codes(ib / 2, c);
      const float s = L.scale(ib / 2);
#pragma unroll
      for (int j = 0; j < 32; ++j) w[j] = s * float(c[j]);
    } else if constexpr (T == q::Q8_0_TILES) {
      // Unit u of row n from its Q8_0 tile (tiles.h): dequant32<Q8_0>'s expression on the same bytes.
      const uint8_t *tile = q8_tile(a.w, a.k, n / 16, u);
      const uint32_t slot = n % 16 + 16 * (u % 2);
      const float d = q::h2f(q::ld16(tile + 1024 + 2 * slot));
#pragma unroll
      for (int j = 0; j < 32; ++j) w[j] = float(int8_t(tile[32 * slot + j])) * d;
    } else if constexpr (T == q::IQ4_NL_TILES) {
      // Unit u of row n from its IQ4_NL tile (tiles.h): dequant32<IQ4_NL>'s expression on the same bytes.
      iq4_tile_dequant32(iq4_tile(a.w, a.k, n / 16, u / 4), n % 16, u % 4, w);
    } else if constexpr (T == q::Q6_K_TILES) {
      // Unit u of row n from its Q6_K tile (tiles.h): dequant32<Q6_K>'s expression on the same bytes.
      q6_tile_dequant32(q6_tile(a.w, a.k, n / 16, u / 8), n % 16, u % 8, w);
    } else {
      q::dequant32<T>(wrow, u, w);
    }
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
  const uint32_t items = qmm_items(a);
  for (uint32_t i = wg; i < items; i += nwg) qmm_item<T>(a, i);
}

__device__ inline void op_qmm(const QmmArgs &a, uint32_t wg, uint32_t nwg) {
  switch (a.type) {
    case q::Q8_0: op_qmm_t<q::Q8_0>(a, wg, nwg); break;
    case q::Q8_0_TILES: op_qmm_t<q::Q8_0_TILES>(a, wg, nwg); break;
    case q::Q6_K: op_qmm_t<q::Q6_K>(a, wg, nwg); break;
    case q::Q6_K_TILES: op_qmm_t<q::Q6_K_TILES>(a, wg, nwg); break;
    case q::IQ4_NL: op_qmm_t<q::IQ4_NL>(a, wg, nwg); break;
    case q::IQ4_XS: op_qmm_t<q::IQ4_XS>(a, wg, nwg); break;
    case q::IQ3_S: op_qmm_t<q::IQ3_S>(a, wg, nwg); break;
    case q::F32: op_qmm_t<q::F32>(a, wg, nwg); break;
    case q::BF16: op_qmm_t<q::BF16>(a, wg, nwg); break;
    default: HIP_TRAP(kTrapBadOp, kTrapFileQmm);
  }
}

}  // namespace mk
