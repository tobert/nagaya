// SPDX-License-Identifier: MIT
#include "gdn.h"

#include <vector>

#include "rt.h"

namespace qwen38 {

__global__ void __launch_bounds__(kBlockThreads) k_gdn_conv(GdnConvArgs a) { op_gdn_conv(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kBlockThreads) k_gdn_core(GdnCoreArgs a) { op_gdn_core(a, blockIdx.x, gridDim.x); }

template <class T>
static T *up(const T *h, size_t n) {
  T *d;
  HIP_CHECK(hipMalloc(&d, sizeof(T) * n));
  HIP_CHECK(hipMemcpy(d, h, sizeof(T) * n, hipMemcpyHostToDevice));
  return d;
}

template <class T>
static T *upv(size_t n) {  // uninitialized output buffer, NaN-filled so unwritten values show
  T *d;
  HIP_CHECK(hipMalloc(&d, sizeof(T) * n));
  HIP_CHECK(hipMemset(d, 0xff, sizeof(T) * n));
  return d;
}

}  // namespace qwen38

using namespace qwen38;

// The gated DeltaNet core for one layer's T tokens: causal conv + silu (op_gdn_conv), then the
// gated delta rule and gated RMSNorm per value head (op_gdn_core). Host pointers in and out,
// tiled (GGUF) order throughout (see gdn.h). `grid` 0 means one workgroup per item for each
// kernel (conv: ceil(10240 / kBlockThreads); core: 48); tests pass other sizes to show the result
// doesn't depend on the grid.
extern "C" int qwen38_gdn(const float *qkv_host, const float *z_host, const float *a_host, const float *b_host,
                      uint32_t t, const float *conv_state_in_host, const float *conv_w_host,
                      const float *ssm_a_host, const float *dt_bias_host, const float *gamma_host,
                      const float *state_in_host, float *y_out_host, float *conv_state_out_host,
                      float *state_out_host, uint32_t grid) {
  if (t == 0 || t > model::kMaxPassTokens) HIP_DIE("qwen38_gdn: T=%u out of range (1..%u)", t, model::kMaxPassTokens);

  const uint32_t C = model::kGdnConvDim, K = model::kGdnConv, VH = model::kGdnVHeads;
  const uint32_t VDIM = model::kGdnValDim, KD = model::kGdnKDim, VD = model::kGdnVDim;

  float *qkv = up(qkv_host, size_t(t) * C);
  float *z = up(z_host, size_t(t) * VDIM);
  float *a_in = up(a_host, size_t(t) * VH);
  float *b_in = up(b_host, size_t(t) * VH);
  float *conv_state_in = up(conv_state_in_host, size_t(C) * (K - 1));
  float *conv_w = up(conv_w_host, size_t(C) * K);
  float *ssm_a = up(ssm_a_host, size_t(VH));
  float *dt_bias = up(dt_bias_host, size_t(VH));
  float *gamma = up(gamma_host, size_t(VD));
  float *state_in = up(state_in_host, size_t(VH) * KD * VD);

  float *conv_y = upv<float>(size_t(t) * C);
  float *conv_state_out = upv<float>(size_t(C) * (K - 1));
  float *y = upv<float>(size_t(t) * VDIM);
  float *state_out = upv<float>(size_t(VH) * KD * VD);

  GdnConvArgs ca{qkv, conv_state_in, conv_w, conv_y, conv_state_out, t};
  const uint32_t conv_items = (C + kBlockThreads - 1) / kBlockThreads;
  k_gdn_conv<<<grid ? grid : conv_items, kBlockThreads>>>(ca);

  GdnCoreArgs ga{conv_y, z, a_in, b_in, ssm_a, dt_bias, gamma, state_in, y, state_out, t};
  k_gdn_core<<<grid ? grid : VH, kBlockThreads>>>(ga);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(y_out_host, y, sizeof(float) * size_t(t) * VDIM, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(conv_state_out_host, conv_state_out, sizeof(float) * size_t(C) * (K - 1), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(state_out_host, state_out, sizeof(float) * size_t(VH) * KD * VD, hipMemcpyDeviceToHost));

  for (void *p : std::vector<void *>{qkv, z, a_in, b_in, conv_state_in, conv_w, ssm_a, dt_bias, gamma, state_in,
                                     conv_y, conv_state_out, y, state_out})
    HIP_CHECK(hipFree(p));
  return 0;
}

// k_gdn_core's register count and scratch (bytes per lane), for the register-budget test.
extern "C" int qwen38_gdn_core_resources(int *regs, int *scratch) {
  hipFuncAttributes fa{};
  HIP_CHECK(hipFuncGetAttributes(&fa, reinterpret_cast<const void *>(&k_gdn_core)));
  *regs = fa.numRegs;
  *scratch = int(fa.localSizeBytes);
  return 0;
}
