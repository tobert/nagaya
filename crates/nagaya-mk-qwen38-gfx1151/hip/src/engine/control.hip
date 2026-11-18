// SPDX-License-Identifier: MIT
// The control op on the device (engine-abi "The residual of the Qwen3.8 engine"): add, ablate or clamp one vector in
// the rows of h, between two launches of a cut pass. The host op of src/control.rs is its oracle, and the gate is equal
// bits. So each sum keeps the host's order: 8 lanes by index modulo 8, each in index order, joined in pairs. Each
// product and each sum is its own rounded operation (__fmul_rn, __fadd_rn): no fused multiply-add.
//
// Its own translation unit with one kernel, as round.hip. Two kernels in one TU, and LDS reached through a non-inlined
// call, have hung this GPU silently. The kernel uses no LDS and no atomics. One block holds one row. Each thread
// computes the whole dot product of its row, so no thread waits for another before the coefficient. One barrier
// separates the reads of the dot product from the writes of the row.
#include "engine.h"

namespace qwen38 {

namespace {

constexpr uint32_t kControlBlock = 64;
constexpr uint32_t kControlLanes = 8;
constexpr uint32_t kControlWidth = model::kHcDim;
static_assert(kControlWidth == 10240, "the residual of one token: 4 streams of 2,560 values (src/control.rs WIDTH)");
static_assert(kControlWidth % kControlLanes == 0, "the lanes split the row evenly");

// The op codes of src/binding.rs ControlApply.
constexpr uint32_t kOpAdd = 0, kOpAblate = 1, kOpClamp = 2;

__global__ void __launch_bounds__(kControlBlock)
    k_control(float *h, const float *v, float norm, uint32_t op, float strength) {
  float *row = h + size_t(blockIdx.x) * kControlWidth;
  float c = strength;
  if (op != kOpAdd) {
    float s[kControlLanes] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (uint32_t j = 0; j < kControlWidth; j += kControlLanes) {
#pragma unroll
      for (uint32_t k = 0; k < kControlLanes; ++k) s[k] = __fadd_rn(s[k], __fmul_rn(row[j + k], v[j + k]));
    }
    const float dot = __fadd_rn(__fadd_rn(__fadd_rn(s[0], s[1]), __fadd_rn(s[2], s[3])),
                                __fadd_rn(__fadd_rn(s[4], s[5]), __fadd_rn(s[6], s[7])));
    const float along = __fdiv_rn(dot, norm);
    c = op == kOpAblate ? __fdiv_rn(__fmul_rn(-strength, along), norm) : __fdiv_rn(__fsub_rn(strength, along), norm);
  }
  __syncthreads();
  for (uint32_t j = threadIdx.x; j < kControlWidth; j += kControlBlock) row[j] = __fadd_rn(row[j], __fmul_rn(c, v[j]));
}

}  // namespace

void control_apply(float *h, uint32_t rows, const float *v, float sum_squares, uint32_t op, float strength) {
  if (rows == 0 || rows > model::kMaxPassTokens) HIP_DIE("control_apply: %u rows: want 1 to %u", rows, model::kMaxPassTokens);
  if (op > kOpClamp) HIP_DIE("control_apply: op %u: want 0 to 2", op);
  if (!(sum_squares > 0.f)) HIP_DIE("control_apply: the sum of squares %g is not above 0: the host skips a zero vector", sum_squares);
  k_control<<<rows, kControlBlock>>>(h, v, __builtin_sqrtf(sum_squares), op, strength);
  HIP_CHECK(hipGetLastError());
  // The next launch of the pass resets its control block with the stream idle (rt.h), so wait here.
  HIP_CHECK(hipDeviceSynchronize());
}

}  // namespace qwen38
