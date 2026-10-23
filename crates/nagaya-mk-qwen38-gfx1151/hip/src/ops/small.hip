// SPDX-License-Identifier: MIT
#include "small.h"

#include <vector>

#include "qmm.h"
#include "rt.h"

namespace mk {
__global__ void __launch_bounds__(kMkBlock) k_rms(RmsArgs a) { op_rms(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_silu_scale(SiluScaleArgs a) { op_silu_scale(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_hc_mix(HcMixArgs a) { op_hc_mix(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_hc_combine(HcCombineArgs a) { op_hc_combine(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_hc_combine_rms(HcCombineRmsArgs a) { op_hc_combine_rms(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_qmm2(QmmArgs a) { op_qmm(a, blockIdx.x, gridDim.x); }

template <class T>
static T *up(const T *h, size_t n) {
  T *d;
  MK_CHECK(hipMalloc(&d, sizeof(T) * n));
  if (h) MK_CHECK(hipMemcpy(d, h, sizeof(T) * n, hipMemcpyHostToDevice));
  else MK_CHECK(hipMemset(d, 0xff, sizeof(T) * n));
  return d;
}
}  // namespace mk

using namespace mk;

// The hyper-connection "pre" half (Qwen4ExpTextGatedResidual.forward), for tests:
// h [T, 4E] -> mixed [T, E], inject [T, 4] (if w_inject), xn [T, 4E].
// Weights as stored in the GGUF: gamma F32 [4E] (1 + w), down Q8_0 [R, 4E], up Q8_0 [4E, R],
// inject F32 [4, 4E] or null.
extern "C" int mk_hc_pre(const float *h, uint32_t t, uint32_t e, uint32_t r, const float *gamma, const void *w_down,
                         const void *w_up, const float *w_inject, float *mixed_out, float *inject_out, float *xn_out) {
  const uint32_t d4 = 4 * e;
  float *dh = up(h, size_t(t) * d4), *dg = up(gamma, d4), *dxn = up<float>(nullptr, size_t(t) * d4);
  const uint64_t ld_down = q::row_bytes(q::Q8_0, d4), ld_up = q::row_bytes(q::Q8_0, r);
  uint8_t *dwd = up(static_cast<const uint8_t *>(w_down), ld_down * r);
  uint8_t *dwu = up(static_cast<const uint8_t *>(w_up), ld_up * d4);
  float *dwi = w_inject ? up(w_inject, size_t(4) * d4) : nullptr;
  float *ddown = up<float>(nullptr, size_t(t) * r), *dact = up<float>(nullptr, size_t(t) * r);
  float *dup = up<float>(nullptr, size_t(t) * d4), *dmix = up<float>(nullptr, size_t(t) * e);
  float *dz = up<float>(nullptr, size_t(t) * 4), *dinj = up<float>(nullptr, size_t(t) * 4);

  RmsArgs ra{dh, d4, dxn, d4, dg, t, 4, e, 1, 1e-6f};
  k_rms<<<t * 4, kMkBlock>>>(ra);
  QmmArgs qa{};
  qa = {q::Q8_0, dwd, ld_down, r, d4, t, dxn, d4, nullptr, ddown, r, nullptr};
  k_qmm2<<<512, kMkBlock>>>(qa);
  SiluScaleArgs sa{ddown, dact, t * r, 0.25f};
  k_silu_scale<<<64, kMkBlock>>>(sa);
  qa = {q::Q8_0, dwu, ld_up, d4, r, t, dact, r, nullptr, dup, d4, nullptr};
  k_qmm2<<<512, kMkBlock>>>(qa);
  if (dwi) {
    qa = {q::F32, reinterpret_cast<const uint8_t *>(dwi), uint64_t(d4) * 4, 4, d4, t, dxn, d4, nullptr, dz, 4, nullptr};
    k_qmm2<<<64, kMkBlock>>>(qa);
  }
  HcMixArgs ma{dxn, dup, dmix, dwi ? dz : nullptr, dinj, t, e};
  k_hc_mix<<<64, kMkBlock>>>(ma);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(mixed_out, dmix, sizeof(float) * t * e, hipMemcpyDeviceToHost));
  if (dwi) MK_CHECK(hipMemcpy(inject_out, dinj, sizeof(float) * t * 4, hipMemcpyDeviceToHost));
  MK_CHECK(hipMemcpy(xn_out, dxn, sizeof(float) * t * d4, hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{dh, dg, dxn, dwd, dwu, dwi, ddown, dact, dup, dmix, dz, dinj})
    if (p) MK_CHECK(hipFree(p));
  return 0;
}

// h += out (x) inject, for tests.
extern "C" int mk_hc_combine(float *h, const float *out, const float *inject, uint32_t t, uint32_t e) {
  float *dh = up(h, size_t(t) * 4 * e), *dout = up(out, size_t(t) * e), *dinj = up(inject, size_t(t) * 4);
  HcCombineArgs a{dh, dout, dinj, t, e};
  k_hc_combine<<<64, kMkBlock>>>(a);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(h, dh, sizeof(float) * t * 4 * e, hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{dh, dout, dinj}) MK_CHECK(hipFree(p));
  return 0;
}

// hc_combine then rms over h (4 groups of e, gamma [4e] per group, eps 1e-6), for tests: fused = 0 runs the two ops,
// fused = 1 the one fused stage (op_hc_combine_rms) on `grid` workgroups. Outputs the updated h and xn.
extern "C" int mk_hc_combine_rms(const float *h, const float *out, const float *inject, const float *gamma, uint32_t t,
                                 uint32_t e, uint32_t grid, uint32_t fused, float *h_out, float *xn_out) {
  float *dh = up(h, size_t(t) * 4 * e), *dout = up(out, size_t(t) * e), *dinj = up(inject, size_t(t) * 4);
  float *dg = up(gamma, size_t(4) * e), *dxn = up<float>(nullptr, size_t(t) * 4 * e);
  HcCombineArgs ca{dh, dout, dinj, t, e};
  RmsArgs ra{dh, 4 * e, dxn, 4 * e, dg, t, 4, e, 1, 1e-6f};
  if (fused) {
    k_hc_combine_rms<<<grid, kMkBlock>>>(HcCombineRmsArgs{ca, ra});
  } else {
    k_hc_combine<<<grid, kMkBlock>>>(ca);
    k_rms<<<grid, kMkBlock>>>(ra);
  }
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(h_out, dh, sizeof(float) * t * 4 * e, hipMemcpyDeviceToHost));
  MK_CHECK(hipMemcpy(xn_out, dxn, sizeof(float) * t * 4 * e, hipMemcpyDeviceToHost));
  for (void *p : std::vector<void *>{dh, dout, dinj, dg, dxn}) MK_CHECK(hipFree(p));
  return 0;
}
