// SPDX-License-Identifier: MIT
// Compressed sparse attention (QSA), the 12 attention layers (il % 4 == 3).
//
// Reference: modeling_qwen4_exp.py Qwen4ExpTextAttention, Qwen4ExpTextQSAIndexer,
// eager_attention_forward, apply_rotary_pos_emb, rotate_half, Qwen4ExpTextRotaryEmbedding,
// Qwen4ExpTextRMSNorm. For text, all three mrope position components are equal (no vision
// grid), so Qwen4ExpTextRotaryEmbedding.recomposition_frequencies collapses to plain rope over
// the first `kRopeDims` (64) of `head_dim` dims: rotate_half pairs dim i with dim i + 32 (a
// split-half rotation, NOT the interleaved (2i, 2i+1) GPT-J style), each with its own
// frequency inv_freq[i], i in [0, 32). Dims [64, head_dim) pass through unrotated. Gammas are
// the GGUF's own (already 1 + w): y = x * rsqrt(mean(x^2) + eps) * gamma.
//
// The projections (q_proj/k_proj/v_proj, indexer.q_proj/indexer.k_proj) are the engine's job
// (qmm), not ours: this op receives their outputs directly, per token:
//   q_raw  [T, 24 * 512]   head h -> [h*512 .. h*512+256) = q, [h*512+256 .. h*512+512) = gate
//   k_raw  [T,  2 * 256], v_raw [T, 2 * 256]
//   iq_raw [T,  4 * 128]  (indexer q, 4 heads), ik_raw [T, 128] (raw index key, 1 head)
//
// The cache (per attention layer, owned by the caller): kv[cell][2 (k,v)][2 heads][256] f32,
// ik[cell][128] f32, both addressed by absolute position (cell = the token's position in the
// whole context, not relative to this pass). A pass of T tokens starting at p0 writes cells
// p0..p0+T-1; token t attends causally to cells 0..p0+t.
//
// Reduction order (the op's definition, independent of grid or workgroup count):
// - RMSNorm/rope: one wave (32 lanes) per (token, head); lane l owns elements l, l+32, ...,
//   l+D-32 of the D-dim vector (D in {128, 256}), so the rope pair (l, l+32) is always local to
//   one lane -- no cross-lane shuffle. Sum of squares: each lane's local sum, then wave_sum's
//   fixed butterfly.
// - Indexer block pooling: mean over the block's 4 raw keys, summed in ascending cell order
//   then * 0.25 (exact), same convention as small.h's hyper-connection mix.
// - Indexer top-k: scores computed in ascending block order; selection is exact top-`block_topk`
//   by (score desc, index asc) -- a repeated arg-max scan, ties broken by the ascending scan
//   order (strict `>`) which always keeps the lowest index. Below the budget, every block is
//   selected (dense) without needing the scan. The final per-query selection is emitted in
//   ascending cell order (selected blocks' cells, in block-index order, then the tail), which
//   is the fixed order stage 3 reads.
// - Attention: each (token, head) split into kAttnSlices = 8 contiguous slices of its n selected
//   cells (slice s = [s n / 8, (s + 1) n / 8) of the ascending order; a function of n alone, never
//   of T or the grid), one wave each (a wave may run the same slice of several heads of one KV head
//   at once, sharing the k/v loads: schedule only). A wave runs the online (running-max) softmax
//   over its slice in order: m_s = its max score, l_s = sum exp(x - m_s), acc_s = sum exp(x - m_s) v.
//   Then thread d merges element d in ascending slice order: M = max_s m_s, L = sum_s exp(m_s - M)
//   l_s, A_d = sum_s exp(m_s - M) acc_s,d, out_d = A_d / L * sigmoid(gate_d) (flash-decoding).
//   Mathematically the reference's softmax; no atomics; the same fixed order on every run. Each
//   cell's score x (8-term fmaf chain per lane, wave_sum's butterfly, * 1/16) is the same as before
//   the split, bit for bit: only the association of the softmax sums moved.
//
// Attention's numeric design (u = 2^-24, n cells, n_s = ceil(n / 8) the longest slice). Per term,
// the running recurrence costs at most ~4 u per step it is carried through (the rescale factor's
// exp, ~2 ulp, the rescale multiply, the product, the add; a step whose max doesn't change rescales
// by expf(0) = 1 exactly); the term's own exp ~2 u; the merge ~10 u (the slice's rescale exp and
// multiply, an 8-term sum); the division and the gate's sigmoid ~6 u; every exp's argument rounds
// by u |argument| <= u R, R = the scores' range, three times (the term's, the running rescales'
// telescoped sum, the merge's). So against the exact softmax of the same scores,
//   |out_d - exact_d| <= u (4 n_s + 24 + 3 R) (W_d + |o_d|) sig_d,  W_d = sum p_i |v_id| / sum p_i,
// a worst case (linear in the chain; rounding errors typically grow as its square root). The
// sequential order (one wave over all n cells, before 2026-09-29) has the same bound with 4 n in
// place of 4 n_s: the split's bound is ~8x tighter at n = 2051 (1032 u vs 8208 u + the rest), so
// the split moves each output by at most the two bounds' sum and typically by ~sqrt(n) u-sized
// steps. The per-cell score's own error (~13 u of sum_j |q_j k_j| / 16) is unchanged by the split.
// tests/ops/test_attn_split.py holds the op to this bound (x2 margin) and to the definition bit for
// bit; test_attn.py / test_attn_long.py keep the reference comparison's 5e-4.
#pragma once
#include <cfloat>

#include "common.h"
#include "model.h"

namespace mk {

constexpr uint32_t kWavesPerBlock = kMkBlock / kWave;  // 8

__device__ inline void rope_pair(float &a, float &b, float c, float s) {
  const float na = a * c - b * s;
  const float nb = b * c + a * s;
  a = na;
  b = nb;
}

// RMSNorm over D dims (D in {128,256}, D/32 per lane, lane l owns elements l, l+32, ...) then
// partial rope on the first kRopeDims dims (always exactly lane-local pairs (l, l+32), since
// kRopeDims == 2*kWave): `in`/`out` may alias only through the wave's own lane-local elements
// (safe here: caller passes disjoint buffers). `pos` is the absolute cache position.
template <uint32_t D>
__device__ inline void norm_rope_lane(const float *in, float *out, const float *gamma, uint32_t lane, int pos,
                                       const float *inv_freq) {
  static_assert(D % kWave == 0 && D >= model::kRopeDims, "D must be a multiple of kWave and cover the rope dims");
  constexpr uint32_t kPl = D / kWave;
  float local[kPl];
#pragma unroll
  for (uint32_t j = 0; j < kPl; ++j) local[j] = in[lane + kWave * j];
  float ss = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kPl; ++j) ss = fmaf(local[j], local[j], ss);
  ss = wave_sum(ss);
  const float r = 1.0f / sqrtf(ss / float(D) + model::kRmsEps);
#pragma unroll
  for (uint32_t j = 0; j < kPl; ++j) local[j] *= r * gamma[lane + kWave * j];
  const float freq = float(pos) * inv_freq[lane];
  rope_pair(local[0], local[1], cosf(freq), sinf(freq));
#pragma unroll
  for (uint32_t j = 0; j < kPl; ++j) out[lane + kWave * j] = local[j];
}

// ---- stage 1: prep (RMSNorm+rope for Q/K/index-Q, cache writes for K/V/raw-index-key) ----

struct AttnPrepArgs {
  const float *q_raw;   // [T, 24*512]
  const float *k_raw;   // [T, 2*256]
  const float *v_raw;   // [T, 2*256]
  const float *iq_raw;  // [T, 4*128]
  const float *ik_raw;  // [T, 128]
  const float *q_gamma;     // [256]
  const float *k_gamma;     // [256]
  const float *idxq_gamma;  // [128]
  const float *inv_freq;    // [32]
  uint32_t t, p0;
  float *q_out;     // [T, 24*256] out: normed+roped Q
  float *gate_out;  // [T, 24*256] out: raw gate (copied, no math)
  float *iq_out;    // [T, 4*128] out: normed+roped index Q
  float *kv_cache;  // [(p0+T)][2][2][256] in/out; this call writes rows p0..p0+T-1
  float *ik_cache;  // [(p0+T)][128] in/out; this call writes rows p0..p0+T-1
};

// One item = up to kWavesPerBlock (token, unit) assignments, one per wave. unit: 0..23 = Q head
// h (+ gate copy), 24..25 = K head (cache write), 26..27 = V head (cache write), 28..31 =
// index-Q head, 32 = raw index key (cache write).
constexpr uint32_t kAttnPrepUnits = model::kHeads + model::kKvHeads + model::kKvHeads + model::kIdxHeads + 1;  // 33

__device__ inline void attn_prep_item(const AttnPrepArgs &a, uint32_t linear) {
  const uint32_t tok = linear / kAttnPrepUnits, unit = linear % kAttnPrepUnits;
  if (tok >= a.t) return;
  const uint32_t lane = mk_tid() % kWave;
  const int pos = int(a.p0 + tok);
  constexpr uint32_t kHd = model::kHeadDim, kId = model::kIdxDim;
  if (unit < model::kHeads) {
    const uint32_t h = unit;
    const float *qin = a.q_raw + (uint64_t(tok) * model::kHeads + h) * 2 * kHd;
    float *qout = a.q_out + (uint64_t(tok) * model::kHeads + h) * kHd;
    norm_rope_lane<kHd>(qin, qout, a.q_gamma, lane, pos, a.inv_freq);
    float *gout = a.gate_out + (uint64_t(tok) * model::kHeads + h) * kHd;
#pragma unroll
    for (uint32_t j = 0; j < kHd / kWave; ++j) gout[lane + kWave * j] = qin[kHd + lane + kWave * j];
  } else if (unit < model::kHeads + model::kKvHeads) {
    const uint32_t kh = unit - model::kHeads;
    const float *kin = a.k_raw + (uint64_t(tok) * model::kKvHeads + kh) * kHd;
    float *kcache = a.kv_cache + ((uint64_t(pos) * 2 + 0) * model::kKvHeads + kh) * kHd;
    norm_rope_lane<kHd>(kin, kcache, a.k_gamma, lane, pos, a.inv_freq);
  } else if (unit < model::kHeads + 2 * model::kKvHeads) {
    const uint32_t vh = unit - (model::kHeads + model::kKvHeads);
    const float *vin = a.v_raw + (uint64_t(tok) * model::kKvHeads + vh) * kHd;
    float *vcache = a.kv_cache + ((uint64_t(pos) * 2 + 1) * model::kKvHeads + vh) * kHd;
#pragma unroll
    for (uint32_t j = 0; j < kHd / kWave; ++j) vcache[lane + kWave * j] = vin[lane + kWave * j];
  } else if (unit < model::kHeads + 2 * model::kKvHeads + model::kIdxHeads) {
    const uint32_t ih = unit - (model::kHeads + 2 * model::kKvHeads);
    const float *iqin = a.iq_raw + (uint64_t(tok) * model::kIdxHeads + ih) * kId;
    float *iqout = a.iq_out + (uint64_t(tok) * model::kIdxHeads + ih) * kId;
    norm_rope_lane<kId>(iqin, iqout, a.idxq_gamma, lane, pos, a.inv_freq);
  } else {
    const float *ikin = a.ik_raw + uint64_t(tok) * kId;
    float *ikcache = a.ik_cache + uint64_t(pos) * kId;
#pragma unroll
    for (uint32_t j = 0; j < kId / kWave; ++j) ikcache[lane + kWave * j] = ikin[lane + kWave * j];
  }
}

__device__ inline void op_attn_prep(const AttnPrepArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t w = mk_tid() / kWave;
  const uint32_t total = a.t * kAttnPrepUnits;
  const uint32_t items = (total + kWavesPerBlock - 1) / kWavesPerBlock;
  for (uint32_t it = wg; it < items; it += nwg) {
    const uint32_t linear = it * kWavesPerBlock + w;
    if (linear < total) attn_prep_item(a, linear);
  }
}

// ---- stage 2: indexer selection (Qwen4ExpTextQSAIndexer.forward) ----

struct AttnSelectArgs {
  const float *iq_out;     // [T, 4*128], from stage 1
  const float *ik_cache;   // [(p0+T)][128], raw keys, from stage 1 (this pass + prior)
  const float *idxk_gamma; // [128]
  const float *inv_freq;   // [32]
  uint32_t t, p0;
  uint32_t budget, ratio;  // runtime indexer params (engine uses model::kIdxBudget/kIdxRatio);
                           // pooling and expansion assume ratio == 4: callers check (mk_attn dies)
  float *block_scores;      // scratch [T][max_nb], max_nb = (p0+T)/ratio
  float *block_scores_out;  // out [T][max_nb], the untouched scores (for near-tie inspection)
  int32_t *picked;          // scratch [T][block_topk_max] for the serial oracle (attn_select_oracle.h); unused by the op
  int32_t *sel;             // out [T][max_sel], max_sel = budget+ratio-1; unused slots = -1
  uint32_t *sel_count;      // out [T]
  uint32_t max_nb, max_sel, block_topk_max;
  // The pooled block-key cache (goal 08 I2, docs/indexer.md): [max_nb][128] floats, block b's pooled, normed, roped key
  // (what attn_pool_block makes from the raw keys of cells 4b..4b+3) at b * 128, in the score's lane layout. Score:
  // non-null reads every block's key from it (the pool stage wrote the pass's new blocks before); null pools live from
  // ik_cache (the composed op, the masked lab path). Pool: where blocks [pool_b0, (p0 + t) / ratio) are written.
  float *pooled;
  uint32_t pool_b0;
};

// Order-preserving key of a block score: for floats in [+0, +inf] the IEEE bit patterns order like
// the values. -0.0 compares equal to +0.0 (the serial scan's float `>` saw a tie), so it maps to 0.
// Anything else negative, or NaN, can't come out of the relu sum: stop rather than rank it.
__device__ __attribute__((always_inline)) inline uint32_t attn_score_key(float s) {
  uint32_t u = __float_as_uint(s);
  if (u == 0x80000000u) u = 0;
  if (u > 0x7f800000u) MK_TRAP(kTrapAttnSelect, kTrapFileAttn);
  return u;
}

// Inclusive prefix sum of one uint32 per thread over the workgroup, in thread order (integer, so
// exact). `total` gets the workgroup's sum. `wtot` is kWavesPerBlock words of LDS; LDS-only
// barriers (the scores it's fed were ordered by the caller's __syncthreads or a stage barrier).
__device__ __attribute__((always_inline)) inline uint32_t wg_incl_scan(uint32_t v, uint32_t *wtot, uint32_t &total) {
  const uint32_t tid = mk_tid(), w = tid / kWave, l = tid % kWave;
#pragma unroll
  for (uint32_t off = 1; off < kWave; off <<= 1) {
    const uint32_t u = __shfl_up(v, off, kWave);
    if (l >= off) v += u;
  }
  if (l == kWave - 1) wtot[w] = v;
  lds_sync();
  uint32_t before = 0;
  total = 0;
#pragma unroll
  for (uint32_t i = 0; i < kWavesPerBlock; ++i) {
    const uint32_t x = wtot[i];
    before += i < w ? x : 0u;
    total += x;
  }
  lds_sync();  // wtot is rewritten by the next scan
  return v + before;
}

// Token t's selection from its block scores (a.block_scores[t], read only), by every thread of the
// workgroup. The result is the op's definition, identical to the serial scan it replaced
// (attn_select_oracle.h): the top select_count blocks by (score desc, index asc) in ascending
// block order, 4 cells each, then the tail, then -1 padding. The schedule:
// 1. Radix select, 4 passes of 8 bits from the top, of K = the select_count-th largest key. A
//    pass histograms the keys that match the digits chosen so far (LDS, integer atomics: the
//    counts are exact whatever the order), then a prefix sum over the bins from the top finds the
//    bin holding the remaining rank. G = keys > K (summed from the bins above each chosen digit),
//    R = select_count - G >= 1 of the keys == K are taken: the R lowest-indexed, as the strict
//    order says.
// 2. Emission in ascending block order, 256 blocks per round: one prefix sum over packed (gt, eq)
//    flags gives each block its count of earlier (key > K) and (key == K) blocks; block b is in
//    iff gt, or eq with fewer than R earlier ties, and its slot is gt_before + min(eq_before, R).
// LDS: 256 + 8 + 2 words of the arena.
__device__ __attribute__((always_inline)) inline void attn_select_emit(const AttnSelectArgs &a, uint32_t t) {
  const uint32_t tid = mk_tid();
  const uint32_t n_vis = a.p0 + t + 1;
  const uint32_t nb = n_vis / a.ratio;
  const float *scores = a.block_scores + uint64_t(t) * a.max_nb;
  const uint32_t tail_start = nb * a.ratio;
  const uint32_t block_topk = a.budget / a.ratio;
  const uint32_t select_count = nb < block_topk ? nb : block_topk;
  int32_t *sel = a.sel + uint64_t(t) * a.max_sel;
  uint32_t out_n;
  if (select_count == nb) {  // dense: every complete block then the tail = cells 0..n_vis-1
    for (uint32_t c = tid; c < n_vis; c += kMkBlock) sel[c] = int32_t(c);
    out_n = n_vis;
  } else {  // sparse
    static_assert(kMkBlock == 256, "one histogram bin (and one 8-bit digit value) per thread");
    uint32_t *hist = reinterpret_cast<uint32_t *>(lds_arena());
    uint32_t *wtot = hist + 256;
    uint32_t *bc = wtot + kWavesPerBlock;
    uint32_t prefix = 0, mask = 0, k = select_count, above = 0;
    if (select_count > 0) {
      for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        lds_sync();
        for (uint32_t b = tid; b < nb; b += kMkBlock) {
          const uint32_t key = attn_score_key(scores[b]);
          if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        lds_sync();
        // Thread i holds bin 255 - i, so the inclusive scan is the count of group keys with a digit >= that bin.
        const uint32_t v = hist[255u - tid];
        uint32_t total;
        const uint32_t ge = wg_incl_scan(v, wtot, total);
        if (ge >= k && ge - v < k) {  // exactly one bin: the k-th largest of the group is in it
          bc[0] = 255u - tid;
          bc[1] = ge - v;  // group keys in higher bins: all > K
        }
        lds_sync();
        const uint32_t d = bc[0], hi = bc[1];
        above += hi;
        k -= hi;
        prefix |= d << shift;
        mask |= 255u << shift;
      }
    }
    const uint32_t K = prefix, R = k;  // R >= 1 when select_count > 0 (the chosen bin held the k-th key)
    uint32_t carry_gt = 0, carry_eq = 0;
    for (uint32_t c0 = 0; c0 < nb; c0 += kMkBlock) {
      const uint32_t b = c0 + tid;
      uint32_t gt = 0, eq = 0;
      if (b < nb && select_count > 0) {
        const uint32_t key = attn_score_key(scores[b]);
        gt = key > K;
        eq = key == K;
      }
      const uint32_t flags = gt | (eq << 16);  // per-round counts <= 256: no carry between the halves
      uint32_t total;
      const uint32_t excl = wg_incl_scan(flags, wtot, total) - flags;
      const uint32_t gt_before = carry_gt + (excl & 0xffffu), eq_before = carry_eq + (excl >> 16);
      if (gt || (eq && eq_before < R)) {
        const uint32_t slot = gt_before + (eq_before < R ? eq_before : R);
#pragma unroll
        for (uint32_t c = 0; c < 4; ++c) sel[slot * 4 + c] = int32_t(b * a.ratio + c);
      }
      carry_gt += total & 0xffffu;
      carry_eq += total >> 16;
    }
    // The emission's own count must agree with the radix select's: else the two passes saw different keys.
    if (select_count > 0 && (carry_gt != above || carry_eq < R)) MK_TRAP(kTrapAttnSelect, kTrapFileAttn);
    out_n = select_count * 4;
    for (uint32_t c = tail_start + tid; c < n_vis; c += kMkBlock) sel[out_n + (c - tail_start)] = int32_t(c);
    out_n += n_vis - tail_start;
  }
  if (tid == 0) a.sel_count[t] = out_n;
  for (uint32_t k = out_n + tid; k < a.max_sel; k += kMkBlock) sel[k] = -1;
}

// One complete block's pooled key, by one wave (lane = the wave's lane): the block's 4 raw index keys summed in ascending
// cell order and scaled by 1/4, RMS-normalized (one wave_sum), idx_k_norm's gamma, roped at the block start. Element
// lane + 32 j of the 128 in pooled[j]. The one body the pool stage stores and the live score path uses, so a cached key
// is the register file a live pool would have made.
constexpr uint32_t kPoolPl = model::kIdxDim / kWave;
__device__ __attribute__((always_inline)) inline void attn_pool_block(const AttnSelectArgs &a, uint32_t b, uint32_t lane,
                                                                        float (&pooled)[kPoolPl]) {
  constexpr uint32_t kId = model::kIdxDim;
  const uint32_t base = b * a.ratio;
#pragma unroll
  for (uint32_t j = 0; j < kPoolPl; ++j) {
    float s = 0.f;
    for (uint32_t c = 0; c < 4; ++c) s += a.ik_cache[uint64_t(base + c) * kId + lane + kWave * j];
    pooled[j] = s * 0.25f;
  }
  float ss = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kPoolPl; ++j) ss = fmaf(pooled[j], pooled[j], ss);
  ss = wave_sum(ss);
  const float r = 1.0f / sqrtf(ss / float(kId) + model::kRmsEps);
#pragma unroll
  for (uint32_t j = 0; j < kPoolPl; ++j) pooled[j] *= r * a.idxk_gamma[lane + kWave * j];
  const float freq = float(base) * a.inv_freq[lane];
  rope_pair(pooled[0], pooled[1], cosf(freq), sinf(freq));
}

// The cached key of block b: the same 128 floats attn_pool_block made, loaded back. Its own function with its own loads
// (k_mega sits at the VGPR limit: the cached path must not widen the live path's register use).
__device__ __attribute__((always_inline)) inline void attn_pooled_load(const AttnSelectArgs &a, uint32_t b, uint32_t lane,
                                                                         float (&pooled)[kPoolPl]) {
  const float *src = a.pooled + uint64_t(b) * model::kIdxDim;
#pragma unroll
  for (uint32_t j = 0; j < kPoolPl; ++j) pooled[j] = src[lane + kWave * j];
}

// One complete block's score for token t, by one wave: its pooled key (from the cache if the args carry one, else pooled
// live), dotted against the token's index-query heads, relu-summed and scaled. The block's score is this one wave's
// instruction chain whoever runs it, so the composed select item and the score stage (op_attn_score) give the same
// bits. Writes a.block_scores[t][b] from lane 0.
__device__ __attribute__((always_inline)) inline void attn_score_block(const AttnSelectArgs &a, uint32_t t, uint32_t b,
                                                                         uint32_t lane) {
  float *scores = a.block_scores + uint64_t(t) * a.max_nb;
  constexpr uint32_t kId = model::kIdxDim;
  float pooled[kPoolPl];
  if (a.pooled) attn_pooled_load(a, b, lane, pooled);
  else attn_pool_block(a, b, lane, pooled);

  float score = 0.f;
  for (uint32_t h = 0; h < model::kIdxHeads; ++h) {
    const float *q = a.iq_out + (uint64_t(t) * model::kIdxHeads + h) * kId;
    float dot = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < kPoolPl; ++j) dot = fmaf(q[lane + kWave * j], pooled[j], dot);
    dot = wave_sum(dot);
    score += fmaxf(dot, 0.f);  // relu
  }
  score *= 1.0f / sqrtf(float(kId));
  if (lane == 0) scores[b] = score;
}

// The composed item: token t's scores by this workgroup's 8 waves (each takes every 8th block), then its selection.
__device__ __attribute__((always_inline)) inline void attn_select_item(const AttnSelectArgs &a, uint32_t t) {
  const uint32_t w = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t n_vis = a.p0 + t + 1;
  const uint32_t nb = n_vis / a.ratio;
  float *scores = a.block_scores + uint64_t(t) * a.max_nb;
  for (uint32_t b = w; b < nb; b += kWavesPerBlock) attn_score_block(a, t, b, lane);
  __syncthreads();
  if (a.block_scores_out) {
    float *out = a.block_scores_out + uint64_t(t) * a.max_nb;
    for (uint32_t b = mk_tid(); b < nb; b += kMkBlock) out[b] = scores[b];
  }
  __syncthreads();
  attn_select_emit(a, t);
}

__device__ __attribute__((always_inline)) inline void op_attn_select(const AttnSelectArgs &a, uint32_t wg, uint32_t nwg) {
  for (uint32_t t = wg; t < a.t; t += nwg) attn_select_item(a, t);
}

// The fused pass's split of the composed op (goal 08 I1, docs/indexer.md). Score: items (token t, block range
// [r0, r0 + kScoreRange)), t-major; the range's blocks go to the workgroup's 8 waves (r0 + w, r0 + w + 8, ...), each
// block through attn_score_block, so every score equals the composed op's; an item past its token's nb does nothing.
// Ranges per token cover the pass's largest nb, (p0 + T) / ratio. Needs no workgroup barrier: items write disjoint
// scores. A grid barrier separates it from op_attn_emit, which reads them (the selection alone, from attn_select_emit).
constexpr uint32_t kScoreRange = 32;
__host__ __device__ inline uint32_t attn_score_ranges(const AttnSelectArgs &a) {
  return ((a.p0 + a.t) / a.ratio + kScoreRange - 1) / kScoreRange;
}
__device__ __attribute__((always_inline)) inline void op_attn_score(const AttnSelectArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t w = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t nr = attn_score_ranges(a);
  const uint32_t items = a.t * nr;
  for (uint32_t i = wg; i < items; i += nwg) {
    const uint32_t t = i / nr, r0 = (i % nr) * kScoreRange;
    const uint32_t nb = (a.p0 + t + 1) / a.ratio;
    const uint32_t end = r0 + kScoreRange < nb ? r0 + kScoreRange : nb;
    for (uint32_t b = r0 + w; b < end; b += kWavesPerBlock) attn_score_block(a, t, b, lane);
  }
}
__device__ __attribute__((always_inline)) inline void op_attn_emit(const AttnSelectArgs &a, uint32_t wg, uint32_t nwg) {
  for (uint32_t t = wg; t < a.t; t += nwg) {
    lds_sync();  // the previous token's emission is done with the arena
    attn_select_emit(a, t);
  }
}

// The pool stage (goal 08 I2): blocks [pool_b0, (p0 + t) / ratio) of the context's pooled-key cache, one wave per block (a
// workgroup's 8 waves take 8 consecutive blocks; item i of the stage is workgroup-sized). Each block is attn_pool_block's
// registers stored to a.pooled[b][128] by its one wave: one writer per address, no workgroup barrier. Runs after prep (the
// raw keys of this pass's cells are written) and before the score stage, a grid barrier each side.
__host__ __device__ inline uint32_t attn_pool_blocks(const AttnSelectArgs &a) {
  return (a.p0 + a.t) / a.ratio - a.pool_b0;  // the host checks pool_b0 <= (p0 + t) / ratio at build
}
__device__ __attribute__((always_inline)) inline void op_attn_pool(const AttnSelectArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t w = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t n = attn_pool_blocks(a), items = (n + kWavesPerBlock - 1) / kWavesPerBlock;
  for (uint32_t i = wg; i < items; i += nwg) {
    const uint32_t k = i * kWavesPerBlock + w;
    if (k >= n) continue;  // wave-uniform
    const uint32_t b = a.pool_b0 + k;
    float pooled[kPoolPl];
    attn_pool_block(a, b, lane, pooled);
#pragma unroll
    for (uint32_t j = 0; j < kPoolPl; ++j) a.pooled[uint64_t(b) * model::kIdxDim + lane + kWave * j] = pooled[j];
  }
}

// ---- stage 3: attention (eager_attention_forward, GQA repeat_kv, gated output) ----

struct AttnAttendArgs {
  const float *q_out;     // [T, 24*256], from stage 1
  const float *gate_out;  // [T, 24*256], from stage 1
  const float *kv_cache;  // [(p0+T)][2][2][256]
  const int32_t *sel;       // [T][max_sel], ascending, from stage 2
  const uint32_t *sel_count;// [T]
  uint32_t t, max_sel;
  float *attn_out;  // [T, 24*256] out
};

// The split (flash-decoding; the op's definition, see the header): slice s of a (token, head)'s n selected cells
// is [s n / kAttnSlices, (s + 1) n / kAttnSlices) of the ascending selection. A function of n alone: never of T,
// the grid, or the workgroup that runs it. Slices past n's cells are empty (n < kAttnSlices) and skipped.
constexpr uint32_t kAttnSlices = kWavesPerBlock;  // 8: one wave per slice
__device__ __attribute__((always_inline)) inline uint32_t attn_slice_lo(uint32_t s, uint32_t n) {
  return s * n / kAttnSlices;
}

// One wave's slice for kH heads of one KV head at once: for each head, the running-max online softmax over
// sel[0 .. len) in order, len >= 1. Each head's arithmetic is exactly the kH = 1 chain (the heads share only the
// k/v loads), so a head's bits don't depend on kH: kH is schedule. Out per head: m (the slice's max score, exact),
// l = sum exp(x - m), acc[j] = sum exp(x - m) v[lane + 32 j].
template <uint32_t kH>
__device__ __attribute__((always_inline)) inline void attn_slice(const AttnAttendArgs &a, const int32_t *sel, uint32_t len,
                                                                 const float (&qlocal)[kH][model::kHeadDim / kWave],
                                                                 uint32_t kvh, uint32_t lane, float (&m)[kH], float (&l)[kH],
                                                                 float (&acc)[kH][model::kHeadDim / kWave]) {
  constexpr uint32_t kHd = model::kHeadDim;
  constexpr uint32_t kPl = kHd / kWave;
  const float scale = 1.0f / sqrtf(float(kHd));
#pragma unroll
  for (uint32_t hh = 0; hh < kH; ++hh) {
    m[hh] = -FLT_MAX;
    l[hh] = 0.f;
  }
  // Each step is a dependent chain (cell index -> k -> dot -> wave_sum -> exp -> v): latency-bound, so the prefetch
  // is explicit: 32 cell indices per load (lane i holds sel[i0 + i], broadcast with a shuffle) and the next cell's
  // k and v loaded while this cell's are used.
  const auto kv_row = [&](int32_t cell, uint32_t kv) {
    return a.kv_cache + ((uint64_t(cell) * 2 + kv) * model::kKvHeads + kvh) * kHd;
  };
  int32_t cells = sel[lane < len ? lane : 0];
  float kn[kPl], vn[kPl];
  {
    const int32_t c0 = __shfl(cells, 0, kWave);
#pragma unroll
    for (uint32_t j = 0; j < kPl; ++j) {
      kn[j] = kv_row(c0, 0)[lane + kWave * j];
      vn[j] = kv_row(c0, 1)[lane + kWave * j];
    }
  }
  for (uint32_t i = 0; i < len; ++i) {
    float kc[kPl], vc[kPl];
#pragma unroll
    for (uint32_t j = 0; j < kPl; ++j) {
      kc[j] = kn[j];
      vc[j] = vn[j];
    }
    if (i + 1 < len) {
      if ((i + 1) % kWave == 0) {
        const uint32_t e = i + 1 + lane;
        cells = sel[e < len ? e : i + 1];
      }
      const int32_t cn = __shfl(cells, (i + 1) % kWave, kWave);
#pragma unroll
      for (uint32_t j = 0; j < kPl; ++j) {
        kn[j] = kv_row(cn, 0)[lane + kWave * j];
        vn[j] = kv_row(cn, 1)[lane + kWave * j];
      }
    }
#pragma unroll
    for (uint32_t hh = 0; hh < kH; ++hh) {
      float dot = 0.f;
#pragma unroll
      for (uint32_t j = 0; j < kPl; ++j) dot = fmaf(qlocal[hh][j], kc[j], dot);
      dot = wave_sum(dot) * scale;
      const float new_m = fmaxf(m[hh], dot);
      const float corr = expf(m[hh] - new_m);  // 0 on the first step (m starts at -FLT_MAX)
      const float p = expf(dot - new_m);
      l[hh] = l[hh] * corr + p;
#pragma unroll
      for (uint32_t j = 0; j < kPl; ++j) acc[hh][j] = acc[hh][j] * corr + p * vc[j];
      m[hh] = new_m;
    }
  }
}

// One item = kH heads h0 .. h0 + kH - 1 (one KV head's) of one token, the whole workgroup: wave w runs slice w of
// every one of them, then, head by head, thread d merges element d. LDS: kAttnSlices x (kHeadDim + 2) floats of
// the arena, reused per head.
template <uint32_t kH>
__device__ __attribute__((always_inline)) inline void attn_attend_item(const AttnAttendArgs &a, uint32_t tok, uint32_t h0) {
  const uint32_t tid = mk_tid(), w = tid / kWave, lane = tid % kWave;
  constexpr uint32_t kHd = model::kHeadDim;
  constexpr uint32_t kPl = kHd / kWave;
  constexpr uint32_t kGroup = model::kHeads / model::kKvHeads;
  static_assert(kGroup % kH == 0, "an item's heads share one KV head");
  static_assert(kMkBlock == kHd, "the merge maps one thread to one output element");
  static_assert(kAttnSlices * (kHd + 2) * sizeof(float) <= kLdsArenaBytes, "the slices' partials must fit the arena");
  const uint32_t kvh = h0 / kGroup;
  float *part = reinterpret_cast<float *>(lds_arena());  // [slice][kHd] acc
  float *pm = part + kAttnSlices * kHd;                  // [slice] m
  float *pl = pm + kAttnSlices;                          // [slice] l

  const uint32_t n_sel = a.sel_count[tok];
  if (n_sel == 0) MK_TRAP(kTrapAttnNoSel, kTrapFileAttn);  // causal self-visibility means this can't happen
  const int32_t *sel = a.sel + uint64_t(tok) * a.max_sel;

  const uint32_t lo = attn_slice_lo(w, n_sel), hi = attn_slice_lo(w + 1, n_sel);
  float m[kH] = {}, l[kH] = {}, acc[kH][kPl] = {};
  if (lo < hi) {
    float qlocal[kH][kPl];
#pragma unroll
    for (uint32_t hh = 0; hh < kH; ++hh) {
      const float *q = a.q_out + (uint64_t(tok) * model::kHeads + h0 + hh) * kHd;
#pragma unroll
      for (uint32_t j = 0; j < kPl; ++j) qlocal[hh][j] = q[lane + kWave * j];
    }
    attn_slice<kH>(a, sel + lo, hi - lo, qlocal, kvh, lane, m, l, acc);
  }
#pragma unroll
  for (uint32_t hh = 0; hh < kH; ++hh) {
    if (lo < hi) {
#pragma unroll
      for (uint32_t j = 0; j < kPl; ++j) part[w * kHd + lane + kWave * j] = acc[hh][j];
      if (lane == 0) {
        pm[w] = m[hh];
        pl[w] = l[hh];
      }
    }
    lds_sync();
    // The merge, in ascending slice order: M = max_s m_s (exact), L = sum_s exp(m_s - M) l_s, A = sum_s exp(m_s - M) acc_s.
    float mx = -FLT_MAX;
    for (uint32_t s = 0; s < kAttnSlices; ++s)
      if (attn_slice_lo(s, n_sel) < attn_slice_lo(s + 1, n_sel)) mx = fmaxf(mx, pm[s]);
    float L = 0.f, A = 0.f;
    for (uint32_t s = 0; s < kAttnSlices; ++s) {
      if (attn_slice_lo(s, n_sel) == attn_slice_lo(s + 1, n_sel)) continue;
      const float c = expf(pm[s] - mx);
      L = L + c * pl[s];
      A = A + c * part[s * kHd + tid];
    }
    const uint64_t o = (uint64_t(tok) * model::kHeads + h0 + hh) * kHd + tid;
    a.attn_out[o] = (A / L) * (1.0f / (1.0f + expf(-a.gate_out[o])));
    lds_sync();  // the arena is rewritten by the next head or item
  }
}

// The schedule (bits don't depend on it): at T = 1 one head per item, 24 workgroups of 8 waves (the step is
// latency-bound); at T > 1 kAttnHeadsPerWave heads of one KV head per item share each cell's k/v loads (prefill is
// bound by the K/V traffic: one head per item made a 256-token pass's attention 1.6x slower at 4096 tokens).
constexpr uint32_t kAttnHeadsPerWave = 4;
__device__ __attribute__((always_inline)) inline void op_attn_attend(const AttnAttendArgs &a, uint32_t wg, uint32_t nwg) {
  if (a.t == 1) {
    for (uint32_t it = wg; it < model::kHeads; it += nwg) attn_attend_item<1>(a, 0, it);
  } else {
    constexpr uint32_t kPer = model::kHeads / kAttnHeadsPerWave;
    const uint32_t total = a.t * kPer;
    for (uint32_t it = wg; it < total; it += nwg) attn_attend_item<kAttnHeadsPerWave>(a, it / kPer, (it % kPer) * kAttnHeadsPerWave);
  }
}

}  // namespace mk
