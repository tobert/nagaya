// SPDX-License-Identifier: MIT
// The engine's qmm (qmm.h) and the GPU dequant, for the op tests and tools/qmm_bench.py.
#include "qmm.h"

#include "qmm_host.h"
#include "rt.h"

namespace mk {

__global__ void __launch_bounds__(kMkBlock) k_qmm(QmmArgs a) { op_qmm(a, blockIdx.x, gridDim.x); }

template <uint32_t T>
__global__ void __launch_bounds__(kMkBlock) k_dequant(const uint8_t *w, uint64_t ldw, uint32_t rows, uint32_t k, float *out) {
  const uint32_t units = k / 32;
  const uint64_t total = uint64_t(rows) * units;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t r = uint32_t(i / units), u = uint32_t(i % units);
    // The engine's decoder (q::Unit); the oracle's dequant32 is checked through test_qmm's oracle tests.
    const q::Unit<T> dec(w + r * ldw, u);
    for (uint32_t c = 0; c < 4; ++c) {
      float v[8];
      dec.get8(c, v);
      for (int j = 0; j < 8; ++j) out[uint64_t(r) * k + u * 32 + c * 8 + j] = v[j];
    }
  }
}

}  // namespace mk

using namespace mk;

extern "C" int qwen38_dequant(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t rows, uint32_t k, float *out_host,
                          uint64_t out_bytes) {
  check_shape(type, k);
  const uint64_t ldw = q::row_bytes(type, k);
  MK_CHECK_BYTES("qwen38_dequant", "w", w_bytes, ldw * rows);
  MK_CHECK_BYTES("qwen38_dequant", "out", out_bytes, sizeof(float) * uint64_t(rows) * k);
  uint8_t *w;
  float *out;
  MK_CHECK(hipMalloc(&w, ldw * rows));
  MK_CHECK(hipMalloc(&out, sizeof(float) * uint64_t(rows) * k));
  MK_CHECK(hipMemcpy(w, w_host, ldw * rows, hipMemcpyHostToDevice));
  const int grid = 256;
  switch (type) {
    case q::Q8_0: k_dequant<q::Q8_0><<<grid, kMkBlock>>>(w, ldw, rows, k, out); break;
    case q::Q6_K: k_dequant<q::Q6_K><<<grid, kMkBlock>>>(w, ldw, rows, k, out); break;
    case q::IQ4_NL: k_dequant<q::IQ4_NL><<<grid, kMkBlock>>>(w, ldw, rows, k, out); break;
    case q::IQ4_XS: k_dequant<q::IQ4_XS><<<grid, kMkBlock>>>(w, ldw, rows, k, out); break;
    case q::IQ3_S: k_dequant<q::IQ3_S><<<grid, kMkBlock>>>(w, ldw, rows, k, out); break;
    case q::F32: k_dequant<q::F32><<<grid, kMkBlock>>>(w, ldw, rows, k, out); break;
    case q::BF16: k_dequant<q::BF16><<<grid, kMkBlock>>>(w, ldw, rows, k, out); break;
    default: MK_DIE("unknown quant type %u", type);
  }
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(out_host, out, sizeof(float) * uint64_t(rows) * k, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(w));
  MK_CHECK(hipFree(out));
  return 0;
}

static void launch_qmm(const QmmArgs &a, uint32_t grid) { k_qmm<<<grid, kMkBlock>>>(a); }

extern "C" int qwen38_qmm(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k, const float *x_host,
                      uint64_t x_bytes, uint32_t x_rows, const uint32_t *x_row_host, uint64_t x_row_bytes, uint32_t t,
                      float *y_host, uint64_t y_bytes, uint32_t y_rows, const uint32_t *y_row_host, uint64_t y_row_bytes,
                      uint32_t ldy, uint32_t grid) {
  return run_qmm(launch_qmm, kQmmRows, kQmmToks, "qwen38_qmm", type, w_host, w_bytes, n, k, x_host, x_bytes, x_rows, x_row_host,
                 x_row_bytes, t, y_host, y_bytes, y_rows, y_row_host, y_row_bytes, ldy, grid);
}

extern "C" double qwen38_qmm_bench(uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k,
                               const float *x_host, uint64_t x_bytes, uint32_t t, uint32_t grid, uint32_t reps) {
  return bench_qmm(launch_qmm, "qwen38_qmm_bench", type, w_host, w_bytes, n, k, x_host, x_bytes, t, grid, reps);
}
