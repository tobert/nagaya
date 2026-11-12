// SPDX-License-Identifier: MIT
// Run 2's qmm (qmm_oracle.h) in its own translation unit and library (libqwen38_qmm_oracle.so), for
// test_qmm's bit-identity tests and tools/qmm_bench.py.
#include "qmm_oracle.h"

#include "qmm_host.h"
#include "rt.h"

namespace mk {
__global__ void __launch_bounds__(kMkBlock) k_qmm_oracle(QmmArgs a) { oracle::op_qmm(a, blockIdx.x, gridDim.x); }
}  // namespace mk

using namespace mk;

static void launch_oracle(const QmmArgs &a, uint32_t grid) { k_qmm_oracle<<<grid, kMkBlock>>>(a); }

extern "C" int qwen38_qmm_oracle(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k,
                             const float *x_host, uint64_t x_bytes, uint32_t x_rows, const uint32_t *x_row_host,
                             uint64_t x_row_bytes, uint32_t t, float *y_host, uint64_t y_bytes, uint32_t y_rows,
                             const uint32_t *y_row_host, uint64_t y_row_bytes, uint32_t ldy, uint32_t grid) {
  return run_qmm(launch_oracle, oracle::kQmmRows, oracle::kQmmToks, "qwen38_qmm_oracle", type, w_host, w_bytes, n, k, x_host,
                 x_bytes, x_rows, x_row_host, x_row_bytes, t, y_host, y_bytes, y_rows, y_row_host, y_row_bytes, ldy, grid);
}

extern "C" double qwen38_qmm_oracle_bench(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k,
                                      const float *x_host, uint64_t x_bytes, uint32_t t, uint32_t grid, uint32_t reps) {
  return bench_qmm(launch_oracle, "qwen38_qmm_oracle_bench", type, w_host, w_bytes, n, k, x_host, x_bytes, t, grid, reps);
}
