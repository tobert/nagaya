// SPDX-License-Identifier: MIT
// The sparse MoE block (modeling_qwen4_exp.py Qwen4ExpTextSparseMoeBlock, TopKRouter, Experts).
//
// Stages (grid barrier between each):
//   router logits  qmm (F32 gate, 512 x 2560)                      [engine]
//   op_moe_router  per token: softmax over 512 (f32: max, then sum of exp(x - max) in block_sum
//                  order), top-10 by (probability desc, expert index asc) — a total order —
//                  weights renormalized by their sum in slot order. The top-10 is wave 0's: 10 rounds of
//                  a wave argmax under that order (a selection: no arithmetic, so the bits can't depend
//                  on how it's searched; it replaced thread 0's serial scan, ~190 us per token per layer).
//   op_moe_group   one workgroup: the expert ids staged in LDS, count per expert, offsets and work-item
//                  prefixes by an integer block scan (exact in any order), and the assignment list
//                  (a = t * 10 + slot) per expert in increasing a. No atomics.
//   op_moe_gu      expert gate and up projections over each expert's gathered tokens: the exact f32
//                  qmm, or with `fast` the f16 WMMA one (qmmh.h, QmmhMoe tiles; PLAN.md Speed S2).
//   op_swiglu      h = silu(gate) * up.
//   op_moe_down    expert down projections, scattered to their assignment rows.
//   op_moe_combine out[t] = (sum over the token's slots in ascending expert order of y * w)
//                  + sigmoid(shared_gate) * shared, like the reference's index_add_ loop over
//                  experts in ascending order, then + the shared expert.
// The shared expert is plain qmm + op_swiglu, run by the engine.
#pragma once
#include "common.h"
#include "model.h"
#include "qmm.h"
#include "qmmh.h"
#include "small.h"

namespace mk {

using model::kExperts;
using model::kTopK;

struct MoeRouterArgs {
  const float *logits;  // [T, 512]
  int32_t *topk_idx;    // [T, 10]
  float *topk_w;        // [T, 10]
  uint32_t t;
};

__device__ inline void op_moe_router(const MoeRouterArgs &a, uint32_t wg, uint32_t nwg) {
  __shared__ float scratch[kBlockThreads / kWave];
  float *p = reinterpret_cast<float *>(lds_arena());  // [512] probabilities, in the arena (LDS given back)
  static_assert(kExperts * sizeof(float) <= kLdsArenaBytes, "router probabilities fit the LDS arena");
  static_assert(kExperts % kWave == 0 && kExperts / kWave <= 32 && kTopK <= kWave, "wave 0's top-k layout");
  for (uint32_t t = wg; t < a.t; t += nwg) {
    const float *x = a.logits + uint64_t(t) * kExperts;
    // max: exact regardless of order
    float m = -INFINITY;
    for (uint32_t e = mk_tid(); e < kExperts; e += kBlockThreads) m = fmaxf(m, x[e]);
    __syncthreads();
    {
      const uint32_t w = mk_tid() / kWave, l = mk_tid() % kWave;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, kWave));
      if (l == 0) scratch[w] = m;
      __syncthreads();
      m = scratch[0];
      for (uint32_t i = 1; i < kBlockThreads / kWave; ++i) m = fmaxf(m, scratch[i]);
      __syncthreads();
    }
    float s = 0.f;
    for (uint32_t e = mk_tid(); e < kExperts; e += kBlockThreads) {
      const float v = expf(x[e] - m);
      p[e] = v;
      s += v;
    }
    s = block_sum(s, scratch);
    for (uint32_t e = mk_tid(); e < kExperts; e += kBlockThreads) p[e] = p[e] / s;
    __syncthreads();
    if (mk_tid() < kWave) {
      // Lane l owns experts l + 32 j (j = 0..15), scanned in increasing index, so a strict > keeps the lower index
      // of a tie; the xor butterfly under the same order leaves every lane holding the round's winner.
      const int32_t l = int32_t(mk_tid());
      uint32_t taken = 0;  // bit j: expert l + 32 j already chosen
      float my_v = 0.f;    // lane k < 10 keeps slot k
      int32_t my_i = -1;
      for (uint32_t k = 0; k < kTopK; ++k) {
        float bv = -1.f;  // below every probability; a NaN is never chosen
        int32_t bi = -1;
#pragma unroll 1  // reload from LDS each round: holding the lane's 16 probabilities spilled k_mega (192 VGPRs)
        for (uint32_t j = 0; j < kExperts / kWave; ++j) {
          const float v = p[uint32_t(l) + kWave * j];
          if (!((taken >> j) & 1u) && v > bv) {
            bv = v;
            bi = l + int32_t(kWave * j);
          }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          const float ov = __shfl_xor(bv, off, kWave);
          const int32_t oi = __shfl_xor(bi, off, kWave);
          if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
          }
        }
        if (bi < 0) HIP_TRAP(kTrapMoeRoute, kTrapFileMoe);  // nothing selectable: NaN probabilities
        if (bi % int32_t(kWave) == l) taken |= 1u << (bi / int32_t(kWave));
        if (uint32_t(l) == k) {
          my_v = bv;
          my_i = bi;
        }
      }
      float sum = 0.f;
      for (uint32_t k = 0; k < kTopK; ++k) sum += __shfl(my_v, int(k), kWave);  // slot order, as before
      if (uint32_t(l) < kTopK) {
        a.topk_idx[t * kTopK + uint32_t(l)] = my_i;
        a.topk_w[t * kTopK + uint32_t(l)] = my_v / sum;
      }
    }
    __syncthreads();
  }
}

// Group assignments by expert. Written by workgroup 0 only.
struct MoeGroupArgs {
  const int32_t *topk_idx;  // [T, 10]
  uint32_t t;
  uint32_t *count;          // [512]
  uint32_t *offset;         // [513] start of each expert's run in `assign`
  uint32_t *assign;         // [T * 10] assignment ids a = t * 10 + slot, grouped by expert
  uint32_t *token;          // [T * 10] token of assign[i]
  uint32_t *item_gu;        // [513] prefix of gate/up work items per expert (x2 for gate and up)
  uint32_t *item_dn;        // [513] prefix of down work items per expert
  uint32_t fast;            // item prefixes for the f16 WMMA tiles (MoeFfnArgs::fast must match)
};

constexpr uint32_t kMoeGuRows = model::kFf;     // 640 rows per expert (gate, and up)
constexpr uint32_t kMoeDnRows = model::kEmbd;   // 2560 rows per expert (down)
static_assert(kMoeGuRows % kQmmRows == 0 && kMoeDnRows % kQmmRows == 0, "expert rows must fill whole qmm items");
// The work-item tile of the expert matmuls: exact qmm (8 rows x 8 tokens) or f16 WMMA (QmmhMoe).
struct MoeTile {
  uint32_t rows, toks;
};
__host__ __device__ constexpr MoeTile moe_tile(uint32_t fast) {
  return fast ? MoeTile{QmmhMoe::kRows, QmmhMoe::kToks} : MoeTile{kQmmRows, kQmmToks};
}
static_assert(kMoeGuRows % QmmhMoe::kRows == 0 && kMoeDnRows % QmmhMoe::kRows == 0, "expert rows fill whole qmmh items");

__device__ inline void op_moe_group(const MoeGroupArgs &a, uint32_t wg, uint32_t nwg) {
  if (wg != 0) return;
  constexpr uint32_t kMaxA = model::kMaxPassTokens * kTopK;
  static_assert(kMaxA * 2 + kExperts * 4 + 3 * (kBlockThreads / kWave) * 4 <= kLdsArenaBytes, "group staging fits the arena");
  static_assert(kExperts == 2 * kBlockThreads, "each thread owns two experts");
  uint16_t *ids = reinterpret_cast<uint16_t *>(lds_arena());                     // [T*10] expert per assignment
  uint32_t *per = reinterpret_cast<uint32_t *>(lds_arena() + kMaxA * 2);         // [512] counts, then offsets
  uint32_t *wtot = per + kExperts;                                               // [3][8] wave totals
  const uint32_t na = a.t * kTopK, tid = mk_tid(), w = tid / kWave, l = tid % kWave;
  if (na > kMaxA) HIP_TRAP(kTrapMoeGroup, kTrapFileMoe);
  for (uint32_t i = tid; i < na; i += kBlockThreads) {
    const int32_t e = a.topk_idx[i];
    if (e < 0 || e >= int32_t(kExperts)) HIP_TRAP(kTrapMoeGroup, kTrapFileMoe);
    ids[i] = uint16_t(e);
  }
  __syncthreads();
  // counts: thread tid owns experts 2 tid and 2 tid + 1 (consecutive, so the scan below is in expert order)
  const uint32_t e0 = 2 * tid, e1 = 2 * tid + 1;
  uint32_t c0 = 0, c1 = 0;
  for (uint32_t i = 0; i < na; ++i) {
    const uint32_t v = ids[i];
    c0 += v == e0;
    c1 += v == e1;
  }
  a.count[e0] = c0;
  a.count[e1] = c1;
  // exclusive prefix over experts of (assignments, gate/up items, down items): integers, exact in any order
  const MoeTile tile = moe_tile(a.fast);
  const uint32_t t0 = (c0 + tile.toks - 1) / tile.toks, t1 = (c1 + tile.toks - 1) / tile.toks;
  const uint32_t gu_per = 2 * (kMoeGuRows / tile.rows), dn_per = kMoeDnRows / tile.rows;
  uint32_t v[3] = {c0 + c1, gu_per * (t0 + t1), dn_per * (t0 + t1)}, inc[3];
#pragma unroll  // unrolled, so the small arrays stay in registers (a dynamic index would put them in scratch)
  for (int q = 0; q < 3; ++q) {
    uint32_t x = v[q];
#pragma unroll
    for (int off = 1; off < int(kWave); off <<= 1) {
      const uint32_t y = __shfl_up(x, off, kWave);
      if (int(l) >= off) x += y;
    }
    inc[q] = x;
    if (l == kWave - 1) wtot[q * (kBlockThreads / kWave) + w] = x;
  }
  __syncthreads();
  uint32_t ex[3];
#pragma unroll
  for (int q = 0; q < 3; ++q) {
    uint32_t base = 0;
    for (uint32_t i = 0; i < w; ++i) base += wtot[q * (kBlockThreads / kWave) + i];
    ex[q] = base + inc[q] - v[q];
  }
  a.offset[e0] = ex[0];
  a.offset[e1] = ex[0] + c0;
  a.item_gu[e0] = ex[1];
  a.item_gu[e1] = ex[1] + gu_per * t0;
  a.item_dn[e0] = ex[2];
  a.item_dn[e1] = ex[2] + dn_per * t0;
  if (tid == kBlockThreads - 1) {
    a.offset[kExperts] = ex[0] + v[0];
    a.item_gu[kExperts] = ex[1] + v[1];
    a.item_dn[kExperts] = ex[2] + v[2];
  }
  per[e0] = ex[0];
  per[e1] = ex[0] + c0;
  __syncthreads();
  // assignments: each thread writes its two experts' runs in increasing a
  uint32_t j0 = per[e0], j1 = per[e1];
  for (uint32_t i = 0; i < na; ++i) {
    const uint32_t x = ids[i];
    if (x == e0) {
      a.assign[j0] = i;
      a.token[j0++] = i / kTopK;
    } else if (x == e1) {
      a.assign[j1] = i;
      a.token[j1++] = i / kTopK;
    }
  }
}

// Which expert owns global item i: the last e with prefix[e] <= i (binary search over 513).
__device__ inline uint32_t moe_expert_of(const uint32_t *prefix, uint32_t i) {
  uint32_t lo = 0, hi = kExperts;  // prefix[lo] <= i < prefix[hi]
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) / 2;
    if (prefix[mid] <= i) lo = mid;
    else hi = mid;
  }
  return lo;
}

struct MoeFfnArgs {
  uint32_t type_gate, type_up, type_down;
  const uint8_t *gate, *up, *down;  // [512][rows][row_bytes] each
  uint64_t ld_gate, ld_up, ld_down; // bytes per row
  const float *x;                   // [T, 2560] the block input
  float *gu;                        // [T*10, 1280]: gate | up, rows in grouped order
  float *h;                         // [T*10, 640]: swiglu, rows in grouped order
  float *y;                         // [T*10, 2560]: down, rows by assignment id
  // the group stage's outputs, read at run time (after a barrier)
  const uint32_t *count, *offset, *assign, *token, *item_gu, *item_dn;
  uint32_t fast;                    // f16 WMMA tiles (QmmhMoe); must match the group stage's
};

__device__ __attribute__((always_inline)) inline void qmm_item_any(uint32_t type, uint32_t fast, const QmmArgs &q,
                                                                   uint32_t local) {
  if (fast) {
    switch (type) {
      case q::Q8_0: qmmh_item_w<q::Q8_0, QmmhMoe>(q, local); break;
      case q::IQ4_NL: qmmh_item_w<q::IQ4_NL, QmmhMoe>(q, local); break;
      case q::IQ4_XS: qmmh_item_w<q::IQ4_XS, QmmhMoe>(q, local); break;
      case q::IQ3_S: qmmh_item_w<q::IQ3_S, QmmhMoe>(q, local); break;
      case q::IQ3_S_TILES: qmmh_item_iq3s_tiles(q, local); break;
      case q::IQ4_NL_TILES: qmmh_item_moe_tiles_w<q::IQ4_NL_TILES>(q, local); break;
      default: HIP_TRAP(kTrapBadOp, kTrapFileMoe);
    }
    return;
  }
  switch (type) {
    case q::Q8_0: qmm_item<q::Q8_0>(q, local); break;
    case q::IQ4_NL: qmm_item<q::IQ4_NL>(q, local); break;
    case q::IQ4_XS: qmm_item<q::IQ4_XS>(q, local); break;
    case q::IQ3_S: qmm_item<q::IQ3_S>(q, local); break;
    case q::IQ3_S_TILES: qmm_item<q::IQ3_S_TILES>(q, local); break;
    case q::IQ4_NL_TILES: qmm_item<q::IQ4_NL_TILES>(q, local); break;
    default: HIP_TRAP(kTrapBadOp, kTrapFileMoe);
  }
}

// An item count read from device memory can't exceed what T * 10 assignments allow; anything
// larger is corrupted group data, and the kernel traps rather than loop over it.
__device__ inline void moe_check_total(uint32_t total, uint32_t mats_x_rows, MoeTile tile) {
  if (total > mats_x_rows / tile.rows * (model::kMaxPassTokens * kTopK / tile.toks + kExperts)) HIP_TRAP(kTrapMoeTotal, kTrapFileMoe);
}

__device__ __attribute__((always_inline)) inline void op_moe_gu(const MoeFfnArgs &a, uint32_t wg, uint32_t nwg) {
  const MoeTile tile = moe_tile(a.fast);
  const uint32_t total = a.item_gu[kExperts];
  moe_check_total(total, 2 * kMoeGuRows, tile);
  for (uint32_t i = wg; i < total; i += nwg) {
    const uint32_t e = moe_expert_of(a.item_gu, i);
    const uint32_t local = i - a.item_gu[e];
    const uint32_t per = (kMoeGuRows / tile.rows) * ((a.count[e] + tile.toks - 1) / tile.toks);
    const bool is_up = local >= per;
    QmmArgs q{};
    q.type = is_up ? a.type_up : a.type_gate;
    q.ldw = is_up ? a.ld_up : a.ld_gate;
    q.w = (is_up ? a.up : a.gate) + uint64_t(e) * kMoeGuRows * q.ldw;
    q.n = kMoeGuRows;
    q.k = model::kEmbd;
    q.t = a.count[e];
    q.x = a.x;
    q.ldx = model::kEmbd;
    q.x_row = a.token + a.offset[e];
    q.y = a.gu + uint64_t(a.offset[e]) * 2 * kMoeGuRows + (is_up ? kMoeGuRows : 0);
    q.ldy = 2 * kMoeGuRows;
    q.wbulk = 1;  // expert matrices: bulky weights
    qmm_item_any(q.type, a.fast, q, is_up ? local - per : local);
  }
}

__device__ __attribute__((always_inline)) inline void op_moe_down(const MoeFfnArgs &a, uint32_t wg, uint32_t nwg) {
  const MoeTile tile = moe_tile(a.fast);
  const uint32_t total = a.item_dn[kExperts];
  moe_check_total(total, kMoeDnRows, tile);
  for (uint32_t i = wg; i < total; i += nwg) {
    const uint32_t e = moe_expert_of(a.item_dn, i);
    QmmArgs q{};
    q.type = a.type_down;
    q.ldw = a.ld_down;
    q.w = a.down + uint64_t(e) * kMoeDnRows * q.ldw;
    q.n = kMoeDnRows;
    q.k = kMoeGuRows;
    q.t = a.count[e];
    q.x = a.h + uint64_t(a.offset[e]) * kMoeGuRows;
    q.ldx = kMoeGuRows;
    q.y = a.y;
    q.ldy = kMoeDnRows;
    q.y_row = a.assign + a.offset[e];
    q.wbulk = 1;  // expert matrices: bulky weights
    qmm_item_any(q.type, a.fast, q, i - a.item_dn[e]);
  }
}

// out[r, j] = silu(g[r, j]) * u[r, j]  (Qwen4ExpTextMLP / Experts: act_fn(gate) * up), with
// torch's silu x * sigmoid(x).
struct SwigluArgs {
  const float *g, *u;
  uint32_t ldg, ldu;
  float *out;
  uint32_t ldo;
  uint32_t rows, cols;
};
__device__ inline void op_swiglu(const SwigluArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t n = a.rows * a.cols;
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < n; i += nwg * kBlockThreads) {
    const uint32_t r = i / a.cols, c = i % a.cols;
    const float g = a.g[uint64_t(r) * a.ldg + c];
    a.out[uint64_t(r) * a.ldo + c] = (g * sigmoidf_(g)) * a.u[uint64_t(r) * a.ldu + c];
  }
}

struct MoeCombineArgs {
  const float *y;           // [T*10, 2560] by assignment id
  const int32_t *topk_idx;  // [T, 10]
  const float *topk_w;      // [T, 10]
  const float *shared;      // [T, 2560] shared expert output
  const float *shared_gate; // [T] pre-sigmoid
  float *out;               // [T, 2560]
  uint32_t t;
};
__device__ inline void op_moe_combine(const MoeCombineArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t E = model::kEmbd, n = a.t * E;
  for (uint32_t i = wg * kBlockThreads + mk_tid(); i < n; i += nwg * kBlockThreads) {
    const uint32_t t = i / E, c = i % E;
    uint32_t order[kTopK];
    for (uint32_t k = 0; k < kTopK; ++k) order[k] = k;
    for (uint32_t k = 1; k < kTopK; ++k)  // insertion sort of the slots by expert id
      for (uint32_t j = k; j > 0 && a.topk_idx[t * kTopK + order[j]] < a.topk_idx[t * kTopK + order[j - 1]]; --j) {
        const uint32_t tmp = order[j];
        order[j] = order[j - 1];
        order[j - 1] = tmp;
      }
    float s = 0.f;
    for (uint32_t k = 0; k < kTopK; ++k) {
      const uint32_t slot = order[k];
      s = s + a.y[uint64_t(t * kTopK + slot) * E + c] * a.topk_w[t * kTopK + slot];
    }
    a.out[i] = s + sigmoidf_(a.shared_gate[t]) * a.shared[i];
  }
}

}  // namespace mk
