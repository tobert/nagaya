// SPDX-License-Identifier: MIT
// The oracle for attn.h's indexer selection (attn_select_emit): the single-thread selection the op
// shipped with through e94328e, frozen, kept only for the bit-identity tests
// (tests/ops/test_attn_select.py). Given one token's block scores it emits, exactly as the op must:
// every block when there are no more than block_topk of them (dense), else the top
// select_count blocks by (score desc, index asc) -- select_count repeated arg-max scans with a
// strict `>` (a tie keeps the lower index), then an insertion sort to ascending order; each
// block's 4 cells in order, then the tail cells, then -1 padding. It consumes the scores it picks
// (writes -FLT_MAX over them) and uses `picked` as scratch. Do not optimize this file: its job is
// to stay what it was.
#pragma once
#include "attn.h"

namespace mk::oracle {

__device__ inline void attn_select_emit_serial(const AttnSelectArgs &a, uint32_t t) {
  if (mk_tid() != 0) return;
  const uint32_t n_vis = a.p0 + t + 1;
  const uint32_t nb = n_vis / a.ratio;
  float *scores = a.block_scores + uint64_t(t) * a.max_nb;
  const uint32_t tail_start = nb * a.ratio;
  const uint32_t block_topk = a.budget / a.ratio;
  const uint32_t select_count = nb < block_topk ? nb : block_topk;
  int32_t *sel = a.sel + uint64_t(t) * a.max_sel;
  int32_t *picked = a.picked + uint64_t(t) * a.block_topk_max;
  uint32_t out_n = 0;
  if (select_count == nb) {  // dense: every complete block, no need to look at scores
    for (uint32_t b = 0; b < nb; ++b) {
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) sel[out_n++] = int32_t(b * a.ratio + c);
    }
  } else {  // sparse: exact top-`select_count` by (score desc, index asc), then sort ascending
    for (uint32_t i = 0; i < select_count; ++i) {
      float best = -FLT_MAX;
      int32_t besti = -1;
      for (uint32_t b = 0; b < nb; ++b) {
        const float sc = scores[b];
        if (sc > best) {  // strict >: on a tie, the lower (already-seen) index wins
          best = sc;
          besti = int32_t(b);
        }
      }
      picked[i] = besti;
      scores[besti] = -FLT_MAX;  // consumed; real scores are >= 0 (relu-summed), never tie this
    }
    for (uint32_t i = 1; i < select_count; ++i) {  // insertion sort ascending; select_count is small
      const int32_t key = picked[i];
      uint32_t j = i;
      while (j > 0 && picked[j - 1] > key) {
        picked[j] = picked[j - 1];
        --j;
      }
      picked[j] = key;
    }
    for (uint32_t i = 0; i < select_count; ++i) {
      const int32_t b = picked[i];
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) sel[out_n++] = b * int32_t(a.ratio) + int32_t(c);
    }
  }
  for (uint32_t c = tail_start; c < n_vis; ++c) sel[out_n++] = int32_t(c);
  a.sel_count[t] = out_n;
  for (uint32_t k = out_n; k < a.max_sel; ++k) sel[k] = -1;
}

}  // namespace mk::oracle
