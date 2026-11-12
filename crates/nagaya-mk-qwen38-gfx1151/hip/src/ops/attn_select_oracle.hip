// SPDX-License-Identifier: MIT
// The serial selection (attn_select_oracle.h) in its own library (libqwen38_attn_select_oracle.so), for
// test_attn_select's bit-identity tests. Same C ABI as attn.hip's qwen38_attn_select_emit.
#include "attn_select_oracle.h"

#include "rt.h"

namespace mk {
__global__ void __launch_bounds__(kMkBlock) k_attn_select_serial(AttnSelectArgs a) {
  for (uint32_t t = blockIdx.x; t < a.t; t += gridDim.x) oracle::attn_select_emit_serial(a, t);
}
}  // namespace mk

using namespace mk;

// scores [t][max_nb] with max_nb = (p0 + t) / ratio; sel_out [t][budget + ratio - 1]; grid 0 = t.
extern "C" int qwen38_attn_select_serial(uint32_t t, uint32_t p0, uint32_t budget, uint32_t ratio, const float *scores,
                                     int32_t *sel_out, uint32_t *sel_count_out, uint32_t grid) {
  if (t == 0) MK_DIE("qwen38_attn_select_serial: T must be >= 1");
  if (ratio != model::kIdxRatio || budget < ratio) MK_DIE("qwen38_attn_select_serial: budget %u ratio %u", budget, ratio);
  const uint32_t max_nb = (p0 + t) / ratio, max_sel = budget + ratio - 1, topk = budget / ratio;
  const size_t ns = size_t(t) * (max_nb ? max_nb : 1);
  float *dscores;
  int32_t *dpicked, *dsel;
  uint32_t *dcount;
  MK_CHECK(hipMalloc(&dscores, sizeof(float) * ns));
  MK_CHECK(hipMemset(dscores, 0, sizeof(float) * ns));
  if (max_nb) MK_CHECK(hipMemcpy(dscores, scores, sizeof(float) * size_t(t) * max_nb, hipMemcpyHostToDevice));
  MK_CHECK(hipMalloc(&dpicked, sizeof(int32_t) * size_t(t) * topk));
  MK_CHECK(hipMalloc(&dsel, sizeof(int32_t) * size_t(t) * max_sel));
  MK_CHECK(hipMemset(dsel, 0x5a, sizeof(int32_t) * size_t(t) * max_sel));  // unwritten cells show
  MK_CHECK(hipMalloc(&dcount, sizeof(uint32_t) * t));
  MK_CHECK(hipMemset(dcount, 0x5a, sizeof(uint32_t) * t));
  AttnSelectArgs a{};
  a.t = t;
  a.p0 = p0;
  a.budget = budget;
  a.ratio = ratio;
  a.block_scores = dscores;
  a.picked = dpicked;
  a.sel = dsel;
  a.sel_count = dcount;
  a.max_nb = max_nb ? max_nb : 1;
  a.max_sel = max_sel;
  a.block_topk_max = topk;
  k_attn_select_serial<<<grid ? grid : t, kMkBlock>>>(a);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(sel_out, dsel, sizeof(int32_t) * size_t(t) * max_sel, hipMemcpyDeviceToHost));
  MK_CHECK(hipMemcpy(sel_count_out, dcount, sizeof(uint32_t) * t, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(dscores));
  MK_CHECK(hipFree(dpicked));
  MK_CHECK(hipFree(dsel));
  MK_CHECK(hipFree(dcount));
  return 0;
}
