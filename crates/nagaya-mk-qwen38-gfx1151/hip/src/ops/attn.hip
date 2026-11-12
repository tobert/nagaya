// SPDX-License-Identifier: MIT
#include "attn.h"
#include "attn_mask.h"

#include <vector>

#include "rt.h"

namespace mk {

__global__ void __launch_bounds__(kMkBlock) k_attn_prep(AttnPrepArgs a) { op_attn_prep(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_attn_select(AttnSelectArgs a) {
  op_attn_select(a, blockIdx.x, gridDim.x);
}
__global__ void __launch_bounds__(kMkBlock) k_attn_attend(AttnAttendArgs a) {
  op_attn_attend(a, blockIdx.x, gridDim.x);
}
__global__ void __launch_bounds__(kMkBlock) k_attn_select_masked(AttnSelectMaskedArgs a) {
  op_attn_select_masked(a, blockIdx.x, gridDim.x);
}

template <class T>
static T *up(const T *h, size_t n) {
  T *d;
  HIP_CHECK(hipMalloc(&d, sizeof(T) * n));
  if (h) HIP_CHECK(hipMemcpy(d, h, sizeof(T) * n, hipMemcpyHostToDevice));
  else HIP_CHECK(hipMemset(d, 0, sizeof(T) * n));
  return d;
}

}  // namespace mk

using namespace mk;

// Test C ABI: host pointers in (including the prior cache contents and p0) and out. `grid`
// overrides every stage's launch grid (0 = one workgroup per item), for grid-independence tests.
extern "C" int qwen38_attn(uint32_t t, uint32_t p0, const float *q_raw, const float *k_raw, const float *v_raw,
                       const float *iq_raw, const float *ik_raw, const float *q_gamma, const float *k_gamma,
                       const float *idxq_gamma, const float *idxk_gamma, const float *inv_freq, float *kv_cache,
                       float *ik_cache, uint32_t budget, uint32_t ratio, int32_t *sel_out, uint32_t *sel_count_out,
                       float *block_scores_out, float *attn_out, uint32_t grid) {
  if (t == 0) HIP_DIE("qwen38_attn: T must be >= 1");
  if (ratio == 0 || budget < ratio) HIP_DIE("qwen38_attn: budget (%u) must be >= ratio (%u) >= 1", budget, ratio);
  // attn.h pools and expands blocks of exactly 4 cells (kaibo M2 review): any other ratio dies.
  if (ratio != model::kIdxRatio) HIP_DIE("qwen38_attn: ratio %u, but the op is specialized to %u", ratio, model::kIdxRatio);
  const uint32_t cap = p0 + t;
  const uint32_t max_nb = cap / ratio;
  const uint32_t max_sel = budget + ratio - 1;
  const uint32_t block_topk_max = budget / ratio;

  // ---- device buffers ----
  float *dq_raw = up(q_raw, size_t(t) * model::kHeads * 2 * model::kHeadDim);
  float *dk_raw = up(k_raw, size_t(t) * model::kKvHeads * model::kHeadDim);
  float *dv_raw = up(v_raw, size_t(t) * model::kKvHeads * model::kHeadDim);
  float *diq_raw = up(iq_raw, size_t(t) * model::kIdxHeads * model::kIdxDim);
  float *dik_raw = up(ik_raw, size_t(t) * model::kIdxDim);
  float *dqg = up(q_gamma, model::kHeadDim), *dkg = up(k_gamma, model::kHeadDim);
  float *diqg = up(idxq_gamma, model::kIdxDim), *dikg = up(idxk_gamma, model::kIdxDim);
  float *dinv = up(inv_freq, model::kRopeDims / 2);
  float *dkv = up(kv_cache, size_t(cap) * 2 * model::kKvHeads * model::kHeadDim);
  float *dik = up(ik_cache, size_t(cap) * model::kIdxDim);

  float *dq_out = up<float>(nullptr, size_t(t) * model::kHeads * model::kHeadDim);
  float *dgate_out = up<float>(nullptr, size_t(t) * model::kHeads * model::kHeadDim);
  float *diq_out = up<float>(nullptr, size_t(t) * model::kIdxHeads * model::kIdxDim);

  float *dscores = up<float>(nullptr, size_t(t) * (max_nb ? max_nb : 1));
  float *dscores_out = up<float>(nullptr, size_t(t) * (max_nb ? max_nb : 1));
  int32_t *dpicked = up<int32_t>(nullptr, size_t(t) * (block_topk_max ? block_topk_max : 1));
  int32_t *dsel = up<int32_t>(nullptr, size_t(t) * max_sel);
  uint32_t *dsel_count = up<uint32_t>(nullptr, t);

  float *dattn_out = up<float>(nullptr, size_t(t) * model::kHeads * model::kHeadDim);
  HIP_CHECK(hipMemset(dattn_out, 0xff, sizeof(float) * size_t(t) * model::kHeads * model::kHeadDim));

  // ---- stage 1: prep ----
  AttnPrepArgs pa{dq_raw, dk_raw, dv_raw, diq_raw, dik_raw, dqg, dkg, diqg, dinv,
                  t,      p0,     dq_out, dgate_out, diq_out, dkv, dik};
  const uint32_t prep_items = (t * kAttnPrepUnits + kWavesPerBlock - 1) / kWavesPerBlock;
  k_attn_prep<<<grid ? grid : prep_items, kMkBlock>>>(pa);

  // ---- stage 2: indexer selection ----
  AttnSelectArgs sa{diq_out, dik, dikg, dinv, t, p0, budget, ratio, dscores, dscores_out,
                     dpicked, dsel, dsel_count, max_nb ? max_nb : 1, max_sel, block_topk_max ? block_topk_max : 1};
  k_attn_select<<<grid ? grid : t, kMkBlock>>>(sa);

  // ---- stage 3: attention ----
  AttnAttendArgs aa{dq_out, dgate_out, dkv, dsel, dsel_count, t, max_sel, dattn_out};
  const uint32_t attend_items = t == 1 ? model::kHeads : t * (model::kHeads / kAttnHeadsPerWave);
  k_attn_attend<<<grid ? grid : attend_items, kMkBlock>>>(aa);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(kv_cache, dkv, sizeof(float) * size_t(cap) * 2 * model::kKvHeads * model::kHeadDim,
                     hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ik_cache, dik, sizeof(float) * size_t(cap) * model::kIdxDim, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sel_out, dsel, sizeof(int32_t) * size_t(t) * max_sel, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sel_count_out, dsel_count, sizeof(uint32_t) * t, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(block_scores_out, dscores_out, sizeof(float) * size_t(t) * (max_nb ? max_nb : 1),
                     hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(attn_out, dattn_out, sizeof(float) * size_t(t) * model::kHeads * model::kHeadDim,
                     hipMemcpyDeviceToHost));

  for (void *p : std::vector<void *>{dq_raw, dk_raw, dv_raw, diq_raw, dik_raw, dqg, dkg, diqg, dikg, dinv, dkv, dik,
                                     dq_out, dgate_out, diq_out, dscores, dscores_out, dpicked, dsel, dsel_count,
                                     dattn_out})
    HIP_CHECK(hipFree(p));
  return 0;
}

// Test C ABI for the selection alone (tests/ops/test_attn_select.py): token t's block scores in, its sel/sel_count
// out, the same shapes as qwen38_attn (max_nb = (p0 + t) / ratio, max_sel = budget + ratio - 1). Same signature as the
// serial oracle's qwen38_attn_select_serial. Unwritten sel cells come back as 0x5a5a5a5a.
namespace mk {
__global__ void __launch_bounds__(kMkBlock) k_attn_select_emit(AttnSelectArgs a) {
  for (uint32_t t = blockIdx.x; t < a.t; t += gridDim.x) attn_select_emit(a, t);
}
}  // namespace mk

extern "C" int qwen38_attn_select_emit(uint32_t t, uint32_t p0, uint32_t budget, uint32_t ratio, const float *scores,
                                   int32_t *sel_out, uint32_t *sel_count_out, uint32_t grid) {
  if (t == 0) HIP_DIE("qwen38_attn_select_emit: T must be >= 1");
  if (ratio != model::kIdxRatio || budget < ratio) HIP_DIE("qwen38_attn_select_emit: budget %u ratio %u", budget, ratio);
  const uint32_t max_nb = (p0 + t) / ratio, max_sel = budget + ratio - 1;
  const size_t ns = size_t(t) * (max_nb ? max_nb : 1);
  float *dscores = up<float>(nullptr, ns);
  if (max_nb) HIP_CHECK(hipMemcpy(dscores, scores, sizeof(float) * size_t(t) * max_nb, hipMemcpyHostToDevice));
  int32_t *dsel = up<int32_t>(nullptr, size_t(t) * max_sel);
  HIP_CHECK(hipMemset(dsel, 0x5a, sizeof(int32_t) * size_t(t) * max_sel));
  uint32_t *dcount = up<uint32_t>(nullptr, t);
  HIP_CHECK(hipMemset(dcount, 0x5a, sizeof(uint32_t) * t));
  AttnSelectArgs a{};
  a.t = t;
  a.p0 = p0;
  a.budget = budget;
  a.ratio = ratio;
  a.block_scores = dscores;
  a.sel = dsel;
  a.sel_count = dcount;
  a.max_nb = max_nb ? max_nb : 1;
  a.max_sel = max_sel;
  a.block_topk_max = budget / ratio;
  k_attn_select_emit<<<grid ? grid : t, kMkBlock>>>(a);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(sel_out, dsel, sizeof(int32_t) * size_t(t) * max_sel, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sel_count_out, dcount, sizeof(uint32_t) * t, hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{dscores, dsel, dcount}) HIP_CHECK(hipFree(p));
  return 0;
}

// Test C ABI for stage 3 alone (tests/ops/test_attn_split.py): q_out, gate [T][24][256], kv [n_cells][2][2][256],
// sel [T][max_sel] (cells < n_cells), sel_count [T] in; attn_out [T][24][256] out. `grid` 0 = one workgroup per item.
extern "C" int qwen38_attn_attend(uint32_t t, uint32_t n_cells, uint32_t max_sel, const float *q_out, const float *gate,
                              const float *kv, const int32_t *sel, const uint32_t *sel_count, float *attn_out,
                              uint32_t grid) {
  if (t == 0 || n_cells == 0) HIP_DIE("qwen38_attn_attend: T %u, %u cells", t, n_cells);
  for (uint32_t i = 0; i < t; ++i) {
    if (sel_count[i] == 0 || sel_count[i] > max_sel) HIP_DIE("qwen38_attn_attend: token %u selects %u of %u", i, sel_count[i], max_sel);
    for (uint32_t k = 0; k < sel_count[i]; ++k)
      if (sel[size_t(i) * max_sel + k] < 0 || uint32_t(sel[size_t(i) * max_sel + k]) >= n_cells)
        HIP_DIE("qwen38_attn_attend: token %u slot %u: cell %d of %u", i, k, sel[size_t(i) * max_sel + k], n_cells);
  }
  const size_t nq = size_t(t) * model::kHeads * model::kHeadDim;
  float *dq = up(q_out, nq), *dg = up(gate, nq);
  float *dkv = up(kv, size_t(n_cells) * 2 * model::kKvHeads * model::kHeadDim);
  int32_t *dsel = up(sel, size_t(t) * max_sel);
  uint32_t *dcnt = up(sel_count, t);
  float *dout = up<float>(nullptr, nq);
  HIP_CHECK(hipMemset(dout, 0xff, sizeof(float) * nq));
  AttnAttendArgs aa{dq, dg, dkv, dsel, dcnt, t, max_sel, dout};
  k_attn_attend<<<grid ? grid : (t == 1 ? model::kHeads : t * (model::kHeads / kAttnHeadsPerWave)), kMkBlock>>>(aa);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(attn_out, dout, sizeof(float) * nq, hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{dq, dg, dkv, dsel, dcnt, dout}) HIP_CHECK(hipFree(p));
  return 0;
}

// Test C ABI for the masked selection (attn_mask.h; tests/ops/test_attn_mask.py): qwen38_attn with the attention cell mask,
// n_mask ranges [mask_lo[i], mask_hi[i]) (the engine's rules: <= kAttnMaskMax, lo < hi, ascending, disjoint, hi <= p0), and
// the masked selection stage in place of the unmasked one, n_mask = 0 included. sel_out is [T][attn_max_sel_masked(budget,
// ratio)]; kv_bytes / ik_bytes are the caches' sizes, checked against (p0 + T) cells.
extern "C" int qwen38_attn_masked(uint32_t t, uint32_t p0, const float *q_raw, const float *k_raw, const float *v_raw,
                              const float *iq_raw, const float *ik_raw, const float *q_gamma, const float *k_gamma,
                              const float *idxq_gamma, const float *idxk_gamma, const float *inv_freq, float *kv_cache,
                              uint64_t kv_bytes, float *ik_cache, uint64_t ik_bytes, uint32_t budget, uint32_t ratio,
                              uint32_t n_mask, const uint32_t *mask_lo, const uint32_t *mask_hi, int32_t *sel_out,
                              uint32_t *sel_count_out, float *block_scores_out, float *attn_out, uint32_t grid) {
  if (t == 0) HIP_DIE("qwen38_attn_masked: T must be >= 1");
  if (ratio != model::kIdxRatio || budget < ratio) HIP_DIE("qwen38_attn_masked: budget %u ratio %u", budget, ratio);
  const uint32_t cap = p0 + t;
  HIP_CHECK_BYTES("qwen38_attn_masked", "kv_cache", kv_bytes, sizeof(float) * size_t(cap) * 2 * model::kKvHeads * model::kHeadDim);
  HIP_CHECK_BYTES("qwen38_attn_masked", "ik_cache", ik_bytes, sizeof(float) * size_t(cap) * model::kIdxDim);
  if (n_mask > kAttnMaskMax) HIP_DIE("qwen38_attn_masked: %u ranges, at most %u", n_mask, kAttnMaskMax);
  AttnCellMask m{};
  m.n = n_mask;
  for (uint32_t i = 0; i < n_mask; ++i) {
    if (mask_lo[i] >= mask_hi[i] || mask_hi[i] > p0 || (i && mask_lo[i] < mask_hi[i - 1]))
      HIP_DIE("qwen38_attn_masked: range %u [%u, %u) breaks the mask's rules (p0 %u)", i, mask_lo[i], mask_hi[i], p0);
    m.lo[i] = mask_lo[i];
    m.hi[i] = mask_hi[i];
  }
  const uint32_t max_nb = cap / ratio;
  const uint32_t max_sel = attn_max_sel_masked(budget, ratio);
  const uint32_t block_topk_max = budget / ratio;

  float *dq_raw = up(q_raw, size_t(t) * model::kHeads * 2 * model::kHeadDim);
  float *dk_raw = up(k_raw, size_t(t) * model::kKvHeads * model::kHeadDim);
  float *dv_raw = up(v_raw, size_t(t) * model::kKvHeads * model::kHeadDim);
  float *diq_raw = up(iq_raw, size_t(t) * model::kIdxHeads * model::kIdxDim);
  float *dik_raw = up(ik_raw, size_t(t) * model::kIdxDim);
  float *dqg = up(q_gamma, model::kHeadDim), *dkg = up(k_gamma, model::kHeadDim);
  float *diqg = up(idxq_gamma, model::kIdxDim), *dikg = up(idxk_gamma, model::kIdxDim);
  float *dinv = up(inv_freq, model::kRopeDims / 2);
  float *dkv = up(kv_cache, size_t(cap) * 2 * model::kKvHeads * model::kHeadDim);
  float *dik = up(ik_cache, size_t(cap) * model::kIdxDim);
  float *dq_out = up<float>(nullptr, size_t(t) * model::kHeads * model::kHeadDim);
  float *dgate_out = up<float>(nullptr, size_t(t) * model::kHeads * model::kHeadDim);
  float *diq_out = up<float>(nullptr, size_t(t) * model::kIdxHeads * model::kIdxDim);
  float *dscores = up<float>(nullptr, size_t(t) * (max_nb ? max_nb : 1));
  float *dscores_out = up<float>(nullptr, size_t(t) * (max_nb ? max_nb : 1));
  int32_t *dpicked = up<int32_t>(nullptr, size_t(t) * (block_topk_max ? block_topk_max : 1));
  int32_t *dsel = up<int32_t>(nullptr, size_t(t) * max_sel);
  HIP_CHECK(hipMemset(dsel, 0x5a, sizeof(int32_t) * size_t(t) * max_sel));
  uint32_t *dsel_count = up<uint32_t>(nullptr, t);
  float *dattn_out = up<float>(nullptr, size_t(t) * model::kHeads * model::kHeadDim);
  HIP_CHECK(hipMemset(dattn_out, 0xff, sizeof(float) * size_t(t) * model::kHeads * model::kHeadDim));

  AttnPrepArgs pa{dq_raw, dk_raw, dv_raw, diq_raw, dik_raw, dqg, dkg, diqg, dinv,
                  t,      p0,     dq_out, dgate_out, diq_out, dkv, dik};
  const uint32_t prep_items = (t * kAttnPrepUnits + kWavesPerBlock - 1) / kWavesPerBlock;
  k_attn_prep<<<grid ? grid : prep_items, kMkBlock>>>(pa);
  AttnSelectMaskedArgs sa{AttnSelectArgs{diq_out, dik, dikg, dinv, t, p0, budget, ratio, dscores, dscores_out, dpicked, dsel,
                                         dsel_count, max_nb ? max_nb : 1, max_sel, block_topk_max ? block_topk_max : 1},
                          m};
  k_attn_select_masked<<<grid ? grid : t, kMkBlock>>>(sa);
  AttnAttendArgs aa{dq_out, dgate_out, dkv, dsel, dsel_count, t, max_sel, dattn_out};
  const uint32_t attend_items = t == 1 ? model::kHeads : t * (model::kHeads / kAttnHeadsPerWave);
  k_attn_attend<<<grid ? grid : attend_items, kMkBlock>>>(aa);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(kv_cache, dkv, kv_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ik_cache, dik, ik_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sel_out, dsel, sizeof(int32_t) * size_t(t) * max_sel, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sel_count_out, dsel_count, sizeof(uint32_t) * t, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(block_scores_out, dscores_out, sizeof(float) * size_t(t) * (max_nb ? max_nb : 1), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(attn_out, dattn_out, sizeof(float) * size_t(t) * model::kHeads * model::kHeadDim, hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{dq_raw, dk_raw, dv_raw, diq_raw, dik_raw, dqg, dkg, diqg, dikg, dinv, dkv, dik, dq_out,
                                     dgate_out, diq_out, dscores, dscores_out, dpicked, dsel, dsel_count, dattn_out})
    HIP_CHECK(hipFree(p));
  return 0;
}
