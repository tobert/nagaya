// SPDX-License-Identifier: MIT
// The attention cell mask (lab only, docs/lab/mask.md): cell ranges [lo, hi) of a context that no attention read may use.
// A masked cell keeps its position (and every later token keeps its own); it just stops being visible to the 12 QSA
// layers. Its K/V and raw index key are never read, so they may hold anything (tests fill them with NaN).
//
// Only the indexer's selection (stage 2) changes: stage 3 reads exactly the cells stage 2 lists, so a mask that keeps a
// cell out of the list keeps it out of every attention output. The unmasked selection (attn.h attn_select_item) is not
// touched: an unmasked context runs kStAttnSelect as before, and a masked one runs kStAttnSelectMasked, which only the
// unfused stage kernel (k_stage, engine.hip) dispatches. The megakernel's dispatcher (stage.h run_stage) doesn't know the
// op and traps on it; the host refuses a fused pass on a masked context before that (Engine::run_pass).
//
// The masked selection, for query token t (n_vis = p0 + t + 1 cells visible, nb = n_vis / 4 complete blocks):
// - A block is *clear* if none of its 4 cells is masked. Only clear blocks are scored (the pooled key would average a
//   masked cell's raw key in) and only clear blocks compete for the top-k. A clear block's score is computed by the
//   unmasked op's arithmetic, so it is the same bits.
// - A complete block that is partly masked (it straddles a mask edge) is not scored and does not compete: its unmasked
//   cells are always selected, like the tail cells (the incomplete last block, which the model also attends without
//   scoring). So a straddling block's unmasked cells are never lost and its masked ones never read, and the result
//   depends only on the cells' positions and the ranges: deterministic. Fully masked blocks contribute nothing.
// - Dense when the clear blocks number at most budget / ratio: every unmasked visible cell. Else sparse: the top
//   budget / ratio clear blocks by (score desc, index asc), exactly the unmasked op's order, plus every unmasked cell of
//   the straddling blocks, plus the unmasked tail cells.
// - Emitted in ascending cell order (the unmasked op's order: selected blocks in block order, then the tail, is also
//   ascending), so with no ranges the selection is the unmasked op's, bit for bit (tests/ops/test_attn_mask.py).
// - Count bound: dense or sparse, at most 4 (budget / ratio) clear-block cells, 3 cells for each of at most 2 straddling
//   blocks per range, and 3 tail cells: max_sel = budget + ratio - 1 + 2 kAttnMaskMax (ratio - 1) (kAttnMaxSelMasked).
// - Rules the host enforces (Engine::set_mask): at most kAttnMaskMax ranges, each lo < hi, ascending and disjoint, every
//   hi <= the context's n when set. Tokens only add cells >= n, so a pass never masks a token's own cell, and every
//   query sees at least itself (stage 3 traps on an empty selection).
#pragma once
#include <cmath>

#include "attn.h"

namespace mk {

constexpr uint32_t kAttnMaskMax = 4;
// The selection's row stride for a masked pass (see the count bound above); model::kIdxBudget + kIdxRatio - 1 unmasked.
constexpr uint32_t attn_max_sel_masked(uint32_t budget, uint32_t ratio) {
  return budget + ratio - 1 + 2 * kAttnMaskMax * (ratio - 1);
}
constexpr uint32_t kAttnMaxSelMasked = attn_max_sel_masked(model::kIdxBudget, model::kIdxRatio);

struct AttnCellMask {
  uint32_t n;  // ranges in use, 0..kAttnMaskMax
  uint32_t lo[kAttnMaskMax], hi[kAttnMaskMax];
};

struct AttnSelectMaskedArgs {
  AttnSelectArgs s;  // as kStAttnSelect's; s.max_sel is the masked stride (attn_max_sel_masked)
  AttnCellMask m;
};

__device__ __attribute__((always_inline)) inline bool attn_cell_masked(const AttnCellMask &m, uint32_t c) {
  bool in = false;
  for (uint32_t i = 0; i < m.n; ++i) in |= c >= m.lo[i] && c < m.hi[i];
  return in;
}

// No cell of block b (cells [b ratio, b ratio + ratio)) is masked.
__device__ __attribute__((always_inline)) inline bool attn_block_clear(const AttnCellMask &m, uint32_t b, uint32_t ratio) {
  const uint32_t c0 = b * ratio, c1 = c0 + ratio;
  bool hit = false;
  for (uint32_t i = 0; i < m.n; ++i) hit |= m.lo[i] < c1 && m.hi[i] > c0;
  return !hit;
}

// Token t's masked selection from its block scores (clear blocks only), by every thread of the workgroup. The radix
// select and the tie rule are attn_select_emit's, restricted to the clear blocks; the emission is one pass over the
// visible cells in ascending order. LDS: the radix select's 256 + 8 + 2 words, then one bit per block (kMaxContext / 4
// blocks: 256 words) marking the selected clear blocks.
__device__ __attribute__((always_inline)) inline void attn_select_masked_emit(const AttnSelectArgs &a, const AttnCellMask &m,
                                                                              uint32_t t) {
  static_assert(kBlockThreads == 256, "one histogram bin (and one 8-bit digit value) per thread");
  constexpr uint32_t kBitWords = model::kMaxContext / model::kIdxRatio / 32;
  static_assert(kBitWords <= kBlockThreads, "one selection word per thread to clear");
  static_assert((256 + kWavesPerBlock + 8 + kBitWords) * 4 <= kLdsArenaBytes, "the masked selection's LDS must fit the arena");
  const uint32_t tid = mk_tid();
  const uint32_t n_vis = a.p0 + t + 1;
  const uint32_t nb = n_vis / a.ratio;
  const float *scores = a.block_scores + uint64_t(t) * a.max_nb;
  const uint32_t tail_start = nb * a.ratio;
  const uint32_t block_topk = a.budget / a.ratio;
  int32_t *sel = a.sel + uint64_t(t) * a.max_sel;
  uint32_t *hist = reinterpret_cast<uint32_t *>(lds_arena());
  uint32_t *wtot = hist + 256;
  uint32_t *bc = wtot + kWavesPerBlock;
  uint32_t *picked = bc + 8;  // [kBitWords] bit b: clear block b is selected (sparse only)
  if (nb > kBitWords * 32) HIP_TRAP(kTrapAttnSelect, kTrapFileAttn);

  uint32_t n_clear = 0;
  for (uint32_t b = tid; b < nb; b += kBlockThreads) n_clear += attn_block_clear(m, b, a.ratio) ? 1u : 0u;
  uint32_t total;
  (void)wg_incl_scan(n_clear, wtot, total);
  n_clear = total;
  const bool dense = n_clear <= block_topk;

  if (!dense) {
    const uint32_t select_count = block_topk;
    picked[tid] = 0;
    uint32_t prefix = 0, mask = 0, k = select_count, above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
      hist[tid] = 0;
      lds_sync();
      for (uint32_t b = tid; b < nb; b += kBlockThreads) {
        if (!attn_block_clear(m, b, a.ratio)) continue;
        const uint32_t key = attn_score_key(scores[b]);
        if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
      }
      lds_sync();
      const uint32_t v = hist[255u - tid];
      const uint32_t ge = wg_incl_scan(v, wtot, total);
      if (ge >= k && ge - v < k) {
        bc[0] = 255u - tid;
        bc[1] = ge - v;
      }
      lds_sync();
      const uint32_t d = bc[0], hi = bc[1];
      above += hi;
      k -= hi;
      prefix |= d << shift;
      mask |= 255u << shift;
    }
    const uint32_t K = prefix, R = k;
    uint32_t carry_gt = 0, carry_eq = 0;
    for (uint32_t c0 = 0; c0 < nb; c0 += kBlockThreads) {
      const uint32_t b = c0 + tid;
      uint32_t gt = 0, eq = 0;
      if (b < nb && attn_block_clear(m, b, a.ratio)) {
        const uint32_t key = attn_score_key(scores[b]);
        gt = key > K;
        eq = key == K;
      }
      const uint32_t flags = gt | (eq << 16);
      const uint32_t excl = wg_incl_scan(flags, wtot, total) - flags;
      const uint32_t eq_before = carry_eq + (excl >> 16);
      if (gt || (eq && eq_before < R)) atomicOr(&picked[b >> 5], 1u << (b & 31u));
      carry_gt += total & 0xffffu;
      carry_eq += total >> 16;
    }
    if (carry_gt != above || carry_eq < R) HIP_TRAP(kTrapAttnSelect, kTrapFileAttn);
    lds_sync();
  }

  // Emission: every visible cell in ascending order that is unmasked and either in the tail, in a straddling block, in a
  // selected clear block, or anywhere (dense).
  uint32_t out_n = 0;
  for (uint32_t c0 = 0; c0 < n_vis; c0 += kBlockThreads) {
    const uint32_t c = c0 + tid;
    uint32_t in = 0;
    if (c < n_vis && !attn_cell_masked(m, c)) {
      const uint32_t b = c / a.ratio;
      in = dense || c >= tail_start || !attn_block_clear(m, b, a.ratio) || ((picked[b >> 5] >> (b & 31u)) & 1u) ? 1u : 0u;
    }
    const uint32_t slot = out_n + wg_incl_scan(in, wtot, total) - in;
    if (in) {
      if (slot >= a.max_sel) HIP_TRAP(kTrapAttnSelect, kTrapFileAttn);
      sel[slot] = int32_t(c);
    }
    out_n += total;
  }
  if (out_n == 0) HIP_TRAP(kTrapAttnNoSel, kTrapFileAttn);
  if (tid == 0) a.sel_count[t] = out_n;
  for (uint32_t k = out_n + tid; k < a.max_sel; k += kBlockThreads) sel[k] = -1;
  lds_sync();  // picked and the scan words are rewritten by the next token's selection
}

// attn_select_item with the mask: the same scoring arithmetic for clear blocks; a block with a masked cell is never
// scored (its raw keys are never read) and gets -inf in the inspection copy.
__device__ __attribute__((always_inline)) inline void attn_select_masked_item(const AttnSelectArgs &a, const AttnCellMask &m,
                                                                              uint32_t t) {
  const uint32_t w = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t n_vis = a.p0 + t + 1;
  const uint32_t nb = n_vis / a.ratio;
  float *scores = a.block_scores + uint64_t(t) * a.max_nb;
  constexpr uint32_t kId = model::kIdxDim;
  constexpr uint32_t kPl = kId / kWave;

  for (uint32_t b = w; b < nb; b += kWavesPerBlock) {
    if (!attn_block_clear(m, b, a.ratio)) {  // wave-uniform: b is the wave's
      if (lane == 0) scores[b] = -INFINITY;
      continue;
    }
    const uint32_t base = b * a.ratio;
    float pooled[kPl];
#pragma unroll
    for (uint32_t j = 0; j < kPl; ++j) {
      float s = 0.f;
      for (uint32_t c = 0; c < 4; ++c) s += a.ik_cache[uint64_t(base + c) * kId + lane + kWave * j];
      pooled[j] = s * 0.25f;
    }
    float ss = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < kPl; ++j) ss = fmaf(pooled[j], pooled[j], ss);
    ss = wave_sum(ss);
    const float r = 1.0f / sqrtf(ss / float(kId) + model::kRmsEps);
#pragma unroll
    for (uint32_t j = 0; j < kPl; ++j) pooled[j] *= r * a.idxk_gamma[lane + kWave * j];
    const float freq = float(base) * a.inv_freq[lane];
    rope_pair(pooled[0], pooled[1], cosf(freq), sinf(freq));

    float score = 0.f;
    for (uint32_t h = 0; h < model::kIdxHeads; ++h) {
      const float *q = a.iq_out + (uint64_t(t) * model::kIdxHeads + h) * kId;
      float dot = 0.f;
#pragma unroll
      for (uint32_t j = 0; j < kPl; ++j) dot = fmaf(q[lane + kWave * j], pooled[j], dot);
      dot = wave_sum(dot);
      score += fmaxf(dot, 0.f);  // relu
    }
    score *= 1.0f / sqrtf(float(kId));
    if (lane == 0) scores[b] = score;
  }
  __syncthreads();
  if (a.block_scores_out) {
    float *out = a.block_scores_out + uint64_t(t) * a.max_nb;
    for (uint32_t b = mk_tid(); b < nb; b += kBlockThreads) out[b] = scores[b];
  }
  __syncthreads();
  attn_select_masked_emit(a, m, t);
}

__device__ __attribute__((always_inline)) inline void op_attn_select_masked(const AttnSelectMaskedArgs &a, uint32_t wg,
                                                                            uint32_t nwg) {
  for (uint32_t t = wg; t < a.s.t; t += nwg) attn_select_masked_item(a.s, a.m, t);
}

}  // namespace mk
