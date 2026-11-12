// SPDX-License-Identifier: MIT
// Host side of the qmm test and bench entry points, shared by qmm.hip (the engine's kernel) and
// qmm_oracle.hip (run 2's): each launches its own kernel through `launch(args, grid)`. The two
// kernels live in separate translation units, one LDS-using kernel per TU (DESIGN.md "The megakernel").
#pragma once
#include "qmm.h"
#include "rt.h"

namespace mk {

inline void check_shape(uint32_t type, uint32_t k) {
  const uint32_t bv = q::block_values(type);
  if (q::block_bytes(type) == 0) HIP_DIE("unknown quant type %u", type);
  if (k % 32 != 0 || k % bv != 0) HIP_DIE("K=%u is not a multiple of 32 and of type %u's block (%u)", k, type, bv);
}

// y[y_row[t], n] = W[n, :] . x[x_row[t], :] (a null map is the identity), with x holding x_rows rows of
// K and y holding y_rows rows of ldy >= n floats; y starts as NaN, so unwritten outputs show. `grid` 0 means
// one workgroup per item of `rows_per_item` x `toks_per_item` (tests pass other sizes to show the
// result doesn't depend on the grid). Every buffer comes with its byte size, which must be what the layout needs
// (A6): W is n rows of row_bytes(type, k), x x_rows rows of K floats, y y_rows rows of ldy floats, each row map t
// uint32s (0 bytes with a null map). `op` names the entry point in the refusal.
template <class Launch>
int run_qmm(Launch launch, uint32_t rows_per_item, uint32_t toks_per_item, const char *op, uint32_t type,
            const void *w_host, uint64_t w_bytes, uint32_t n, uint32_t k, const float *x_host, uint64_t x_bytes,
            uint32_t x_rows, const uint32_t *x_row_host, uint64_t x_row_bytes, uint32_t t, float *y_host,
            uint64_t y_bytes, uint32_t y_rows, const uint32_t *y_row_host, uint64_t y_row_bytes, uint32_t ldy,
            uint32_t grid) {
  check_shape(type, k);
  HIP_CHECK_BYTES(op, "w", w_bytes, q::row_bytes(type, k) * n);
  HIP_CHECK_BYTES(op, "x", x_bytes, sizeof(float) * uint64_t(x_rows) * k);
  HIP_CHECK_BYTES(op, "y", y_bytes, sizeof(float) * uint64_t(y_rows) * ldy);
  HIP_CHECK_BYTES(op, "x_row", x_row_bytes, x_row_host ? sizeof(uint32_t) * uint64_t(t) : 0);
  HIP_CHECK_BYTES(op, "y_row", y_row_bytes, y_row_host ? sizeof(uint32_t) * uint64_t(t) : 0);
  if (!x_row_host && x_rows != t) HIP_DIE("qmm: %u x rows for %u tokens without a row map", x_rows, t);
  if (!y_row_host && y_rows != t) HIP_DIE("qmm: %u y rows for %u tokens without a row map", y_rows, t);
  if (ldy < n) HIP_DIE("qmm: ldy %u < n %u", ldy, n);
  QmmArgs a{};
  a.type = type;
  a.ldw = q::row_bytes(type, k);
  a.n = n;
  a.k = k;
  a.t = t;
  a.ldx = k;
  a.ldy = ldy;
  uint8_t *w;
  float *x, *y;
  uint32_t *xr = nullptr, *yr = nullptr;
  HIP_CHECK(hipMalloc(&w, a.ldw * n));
  HIP_CHECK(hipMalloc(&x, sizeof(float) * uint64_t(x_rows) * k));
  HIP_CHECK(hipMalloc(&y, sizeof(float) * uint64_t(y_rows) * ldy));
  HIP_CHECK(hipMemcpy(w, w_host, a.ldw * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(x, x_host, sizeof(float) * uint64_t(x_rows) * k, hipMemcpyHostToDevice));
  if (y_row_host) {
    for (uint32_t i = 0; i < t; ++i)
      if (y_row_host[i] >= y_rows) HIP_DIE("qmm: y_row[%u] = %u >= %u rows", i, y_row_host[i], y_rows);
    HIP_CHECK(hipMalloc(&yr, sizeof(uint32_t) * t));
    HIP_CHECK(hipMemcpy(yr, y_row_host, sizeof(uint32_t) * t, hipMemcpyHostToDevice));
  }
  if (x_row_host) {
    for (uint32_t i = 0; i < t; ++i)
      if (x_row_host[i] >= x_rows) HIP_DIE("qmm: x_row[%u] = %u >= %u rows", i, x_row_host[i], x_rows);
    HIP_CHECK(hipMalloc(&xr, sizeof(uint32_t) * t));
    HIP_CHECK(hipMemcpy(xr, x_row_host, sizeof(uint32_t) * t, hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemset(y, 0xff, sizeof(float) * uint64_t(y_rows) * ldy));  // NaN: unwritten outputs show
  a.w = w;
  a.x = x;
  a.x_row = xr;
  a.y = y;
  a.y_row = yr;
  const uint32_t items = ((n + rows_per_item - 1) / rows_per_item) * ((t + toks_per_item - 1) / toks_per_item);
  launch(a, grid ? grid : items);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(y_host, y, sizeof(float) * uint64_t(y_rows) * ldy, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(w));
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(y));
  if (xr) HIP_CHECK(hipFree(xr));
  if (yr) HIP_CHECK(hipFree(yr));
  return 0;
}

// Timing for tools/qmm_bench.py: `reps` launches of a qmm kernel, after one warm-up launch. Returns milliseconds per launch. The weights are copied
// into enough device copies to exceed 256 MiB and each launch takes the next copy, so a launch
// can't find its weights in the 32 MiB Infinity Cache from the one before (a single copy re-read
// in a loop "read" 344 GB/s, above DRAM's ~240: the first S1 bench, 2026-09-27).
template <class Launch>
double bench_qmm(Launch launch, const char *op, uint32_t type, const void *w_host, uint64_t w_bytes, uint32_t n,
                 uint32_t k, const float *x_host, uint64_t x_bytes, uint32_t t, uint32_t grid, uint32_t reps) {
  check_shape(type, k);
  HIP_CHECK_BYTES(op, "w", w_bytes, q::row_bytes(type, k) * n);
  HIP_CHECK_BYTES(op, "x", x_bytes, sizeof(float) * uint64_t(t) * k);
  if (reps == 0 || grid == 0) HIP_DIE("qmm bench: reps %u, grid %u", reps, grid);
  QmmArgs a{};
  a.type = type;
  a.ldw = q::row_bytes(type, k);
  a.n = n;
  a.k = k;
  a.t = t;
  a.ldx = k;
  a.ldy = n;
  const uint64_t wbytes = a.ldw * n, pool_min = 256ull << 20;
  const uint64_t copies = (pool_min + wbytes - 1) / wbytes;
  uint8_t *w;
  float *x, *y;
  HIP_CHECK(hipMalloc(&w, wbytes * copies));
  HIP_CHECK(hipMalloc(&x, sizeof(float) * uint64_t(t) * k));
  HIP_CHECK(hipMalloc(&y, sizeof(float) * uint64_t(t) * n));
  for (uint64_t c = 0; c < copies; ++c) HIP_CHECK(hipMemcpy(w + c * wbytes, w_host, wbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(x, x_host, sizeof(float) * uint64_t(t) * k, hipMemcpyHostToDevice));
  a.x = x;
  a.y = y;
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  for (uint32_t i = 0; i <= reps; ++i) {
    if (i == 1) HIP_CHECK(hipEventRecord(e0));
    a.w = w + (i % copies) * wbytes;
    launch(a, grid);
  }
  HIP_CHECK(hipEventRecord(e1));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventSynchronize(e1));
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipFree(w));
  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(y));
  return double(ms) / reps;
}

}  // namespace mk
