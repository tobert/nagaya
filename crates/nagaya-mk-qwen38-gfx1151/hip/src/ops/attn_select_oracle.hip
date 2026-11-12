// SPDX-License-Identifier: MIT
// The serial selection (attn_select_oracle.h) in its own library (libqwen38_attn_select_oracle.so), for
// test_attn_select's bit-identity tests. Same C ABI as attn.hip's qwen38_attn_select_emit.
#include "attn_select_oracle.h"

#include "rt.h"

namespace mk {
__global__ void __launch_bounds__(kBlockThreads) k_attn_select_serial(AttnSelectArgs a) {
  for (uint32_t t = blockIdx.x; t < a.t; t += gridDim.x) oracle::attn_select_emit_serial(a, t);
}
}  // namespace mk

using namespace mk;

// scores [t][max_nb] with max_nb = (p0 + t) / ratio; sel_out [t][budget + ratio - 1]; grid 0 = t.
extern "C" int qwen38_attn_select_serial(uint32_t t, uint32_t p0, uint32_t budget, uint32_t ratio, const float *scores,
                                     int32_t *sel_out, uint32_t *sel_count_out, uint32_t grid) {
  if (t == 0) HIP_DIE("qwen38_attn_select_serial: T must be >= 1");
  if (ratio != model::kIdxRatio || budget < ratio) HIP_DIE("qwen38_attn_select_serial: budget %u ratio %u", budget, ratio);
  const uint32_t max_nb = (p0 + t) / ratio, max_sel = budget + ratio - 1, topk = budget / ratio;
  const size_t ns = size_t(t) * (max_nb ? max_nb : 1);
  float *dscores;
  int32_t *dpicked, *dsel;
  uint32_t *dcount;
  HIP_CHECK(hipMalloc(&dscores, sizeof(float) * ns));
  HIP_CHECK(hipMemset(dscores, 0, sizeof(float) * ns));
  if (max_nb) HIP_CHECK(hipMemcpy(dscores, scores, sizeof(float) * size_t(t) * max_nb, hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&dpicked, sizeof(int32_t) * size_t(t) * topk));
  HIP_CHECK(hipMalloc(&dsel, sizeof(int32_t) * size_t(t) * max_sel));
  HIP_CHECK(hipMemset(dsel, 0x5a, sizeof(int32_t) * size_t(t) * max_sel));  // unwritten cells show
  HIP_CHECK(hipMalloc(&dcount, sizeof(uint32_t) * t));
  HIP_CHECK(hipMemset(dcount, 0x5a, sizeof(uint32_t) * t));
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
  k_attn_select_serial<<<grid ? grid : t, kBlockThreads>>>(a);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(sel_out, dsel, sizeof(int32_t) * size_t(t) * max_sel, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sel_count_out, dcount, sizeof(uint32_t) * t, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(dscores));
  HIP_CHECK(hipFree(dpicked));
  HIP_CHECK(hipFree(dsel));
  HIP_CHECK(hipFree(dcount));
  return 0;
}
