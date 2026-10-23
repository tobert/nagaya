// SPDX-License-Identifier: MIT
// Load-time weight conversion into read-once tiles (src/ops/tiles.h). Its own translation unit: the re-layout
// kernel reaches no LDS, and k_stage / k_mega each keep a TU to themselves (engine.hip).
#include "engine.h"
#include "tiles.h"

namespace mk {

__global__ void __launch_bounds__(kMkBlock) k_convert_iq3s(RelayoutArgs a) { op_relayout_iq3s(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_convert_q8(RelayoutArgs a) { op_relayout_q8(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_convert_q6(RelayoutArgs a) { op_relayout_q6(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_convert_iq4(RelayoutArgs a) { op_relayout_iq4nl(a, blockIdx.x, gridDim.x); }

void relayout_q8_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k) {
  if (rows % 16 || k % 64) MK_DIE("Q8_0 tiles: %u x %u is not 16-row groups of 64-value unit pairs", rows, k);
  k_convert_q8<<<1024, kMkBlock>>>(RelayoutArgs{src, dst, rows, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
}

void relayout_q6_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k) {
  if (rows % 16 || k % 256) MK_DIE("Q6_K tiles: %u x %u is not 16-row groups of 256-value superblocks", rows, k);
  k_convert_q6<<<1024, kMkBlock>>>(RelayoutArgs{src, dst, rows, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
}

void relayout_iq4_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k) {
  if (rows % 16 || k % 128) MK_DIE("IQ4_NL tiles: %u x %u is not 16-row groups of 128-value tiles", rows, k);
  k_convert_iq4<<<1024, kMkBlock>>>(RelayoutArgs{src, dst, rows, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
}

void relayout_iq3s_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k) {
  if (rows % 16 || k % 256) MK_DIE("IQ3_S tiles: %u x %u is not 16-row groups of 256-value superblocks", rows, k);
  k_convert_iq3s<<<1024, kMkBlock>>>(RelayoutArgs{src, dst, rows, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
}

}  // namespace mk
