// SPDX-License-Identifier: MIT
// Read-once weight tiles (tiles.h): the re-layout kernel and a tile dequantizer, for tests/ops/test_tiles.py.
#include "tiles.h"
#include "qmmh.h"
#include "qmm_host.h"

#include "rt.h"

namespace mk {

__global__ void __launch_bounds__(kMkBlock) k_relayout_iq3s(RelayoutArgs a) { op_relayout_iq3s(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_relayout_q8(RelayoutArgs a) { op_relayout_q8(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_relayout_q6(RelayoutArgs a) { op_relayout_q6(a, blockIdx.x, gridDim.x); }
__global__ void __launch_bounds__(kMkBlock) k_relayout_iq4(RelayoutArgs a) { op_relayout_iq4nl(a, blockIdx.x, gridDim.x); }

// Every value of an IQ4_NL matrix from its tiles (thread = one row's unit).
__global__ void __launch_bounds__(kMkBlock) k_tile_dequant_iq4(const uint8_t *tiled, uint32_t n, uint32_t k, float *out) {
  const uint32_t units = k / 32;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < uint64_t(n) * units; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t row = uint32_t(i / units), u = uint32_t(i % units);
    float w[32];
    iq4_tile_dequant32(iq4_tile(tiled, k, row / 16, u / 4), row % 16, u % 4, w);
    for (int j = 0; j < 32; ++j) out[uint64_t(row) * k + u * 32 + j] = w[j];
  }
}

// Every value of a Q6_K matrix from its tiles (thread = one row's unit).
__global__ void __launch_bounds__(kMkBlock) k_tile_dequant_q6(const uint8_t *tiled, uint32_t n, uint32_t k, float *out) {
  const uint32_t units = k / 32;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < uint64_t(n) * units; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t row = uint32_t(i / units), u = uint32_t(i % units);
    float w[32];
    q6_tile_dequant32(q6_tile(tiled, k, row / 16, u / 8), row % 16, u % 8, w);
    for (int j = 0; j < 32; ++j) out[uint64_t(row) * k + u * 32 + j] = w[j];
  }
}

// Every value of a Q8_0 matrix from its tiles (thread = one row's unit).
__global__ void __launch_bounds__(kMkBlock) k_tile_dequant_q8(const uint8_t *tiled, uint32_t n, uint32_t k, float *out) {
  const uint32_t units = k / 32;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < uint64_t(n) * units; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t row = uint32_t(i / units), u = uint32_t(i % units);
    const uint8_t *tile = q8_tile(tiled, k, row / 16, u);
    const uint32_t slot = row % 16 + 16 * (u % 2);
    const float d = q::h2f(q::ld16(tile + 1024 + 2 * slot));
    for (int j = 0; j < 32; ++j) out[uint64_t(row) * k + u * 32 + j] = float(int8_t(tile[32 * slot + j])) * d;
  }
}

// Every value of the matrix from its tiles: thread (row, unit) finds the tile and the lane that decodes it.
__global__ void __launch_bounds__(kMkBlock) k_tile_dequant_iq3s(const uint8_t *tiled, uint32_t n, uint32_t k, float *out) {
  const uint32_t nsb = k / 256, units = k / 32;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < uint64_t(n) * units; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint32_t row = uint32_t(i / units), u = uint32_t(i % units);
    const uint32_t g = row / 16, r = row % 16, sb = u / 8, ib = u % 8, h = ib % 2, p = ib / 2;
    Iq3sLane L;
    L.load(tiled + (uint64_t(g) * nsb + sb) * kIq3sTile, r + 16 * h);
    int c[32];
    L.codes(p, c);
    const float s = L.scale(p);
    for (int j = 0; j < 32; ++j) out[uint64_t(row) * k + u * 32 + j] = s * float(c[j]);
  }
}

// The only kernel in this TU that reaches the LDS arena. mode bit 0: tiles (else GGUF rows); bit 1: the exact kernel
// (else the f16 path); bit 2: a dense-shape matrix, Q8_0 or Q6_K by a.type (QmmhDense, with a.ksplit slices; else IQ3_S on the MoE shape);
// bit 3: the split-K sum instead of the matmul.
__global__ void __launch_bounds__(kMkBlock) k_tile_qmm(QmmArgs a, uint32_t mode) {
  const bool tiled = mode & 1, exact = mode & 2, q8 = mode & 4;
  if (mode & 8) {
    op_qmm_ksum(a, blockIdx.x, gridDim.x);
    return;
  }
  if (q8) {
    if (exact) op_qmm(a, blockIdx.x, gridDim.x);
    else op_qmmh(a, blockIdx.x, gridDim.x);
    return;
  }
  const uint32_t items = exact ? qmm_items(a) : qmmh_items<QmmhMoe>(a);
  for (uint32_t i = blockIdx.x; i < items; i += gridDim.x) {
    if (exact) {
      if (tiled) qmm_item<q::IQ3_S_TILES>(a, i);
      else qmm_item<q::IQ3_S>(a, i);
    } else {
      if (tiled) qmmh_item_iq3s_tiles(a, i);
      else qmmh_item<q::IQ3_S, QmmhMoe>(a, i);
    }
  }
}

}  // namespace mk

using namespace mk;

static void check_iq3s_shape(uint32_t n, uint32_t k) {
  if (n == 0 || n % 16 || k == 0 || k % 256) MK_DIE("IQ3_S tiles: %u x %u is not 16-row groups of 256-value superblocks", n, k);
}

// raw: N rows of GGUF IQ3_S; tiled_out: iq3s_tiled_bytes(N, K). tiled_out's prior contents are uploaded first, so a
// byte the kernel doesn't write keeps its value (the test fills it to catch that).
extern "C" int qwen38_relayout_iq3s(const void *raw, uint64_t raw_bytes, uint32_t n, uint32_t k, void *tiled_out,
                                uint64_t tiled_bytes) {
  check_iq3s_shape(n, k);
  const uint64_t src_bytes = uint64_t(n) * (k / 256) * 110, dst_bytes = iq3s_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_relayout_iq3s", "raw", raw_bytes, src_bytes);
  MK_CHECK_BYTES("qwen38_relayout_iq3s", "tiled_out", tiled_bytes, dst_bytes);
  uint8_t *src, *dst;
  MK_CHECK(hipMalloc(&src, src_bytes));
  MK_CHECK(hipMalloc(&dst, dst_bytes));
  MK_CHECK(hipMemcpy(src, raw, src_bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemcpy(dst, tiled_out, dst_bytes, hipMemcpyHostToDevice));
  k_relayout_iq3s<<<256, kMkBlock>>>(RelayoutArgs{src, dst, n, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(tiled_out, dst, dst_bytes, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(src));
  MK_CHECK(hipFree(dst));
  return 0;
}

extern "C" int qwen38_tile_dequant_iq3s(const void *tiled, uint64_t tiled_bytes, uint32_t n, uint32_t k, float *out_host,
                                    uint64_t out_bytes) {
  check_iq3s_shape(n, k);
  const uint64_t bytes = iq3s_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_tile_dequant_iq3s", "tiled", tiled_bytes, bytes);
  MK_CHECK_BYTES("qwen38_tile_dequant_iq3s", "out", out_bytes, sizeof(float) * uint64_t(n) * k);
  uint8_t *t;
  float *out;
  MK_CHECK(hipMalloc(&t, bytes));
  MK_CHECK(hipMalloc(&out, sizeof(float) * uint64_t(n) * k));
  MK_CHECK(hipMemcpy(t, tiled, bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemset(out, 0xff, sizeof(float) * uint64_t(n) * k));
  k_tile_dequant_iq3s<<<256, kMkBlock>>>(t, n, k, out);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(out_host, out, sizeof(float) * uint64_t(n) * k, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(t));
  MK_CHECK(hipFree(out));
  return 0;
}

// y = W x on the MoE tile shape, W given as GGUF IQ3_S rows (re-laid out here when tiled = 1).
// mode: bit 0 = tiles (else GGUF rows), bit 1 = the exact kernel (else the f16 MoE-shape path).
extern "C" int qwen38_tile_qmm_iq3s(const void *raw, uint64_t raw_size, uint32_t n, uint32_t k, const float *x_host,
                                uint64_t x_size, uint32_t t, float *y_host, uint64_t y_size, uint32_t mode, uint32_t grid) {
  check_iq3s_shape(n, k);
  if (mode > 3) MK_DIE("qwen38_tile_qmm_iq3s: mode %u", mode);
  const bool tiled = mode & 1, exact = mode & 2;
  if (t == 0) MK_DIE("qwen38_tile_qmm_iq3s: no tokens");
  const uint64_t raw_bytes = uint64_t(n) * (k / 256) * 110, tiled_bytes = iq3s_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_tile_qmm_iq3s", "raw", raw_size, raw_bytes);
  MK_CHECK_BYTES("qwen38_tile_qmm_iq3s", "x", x_size, sizeof(float) * uint64_t(t) * k);
  MK_CHECK_BYTES("qwen38_tile_qmm_iq3s", "y", y_size, sizeof(float) * uint64_t(t) * n);
  uint8_t *src, *w;
  float *x, *y;
  MK_CHECK(hipMalloc(&src, raw_bytes));
  MK_CHECK(hipMemcpy(src, raw, raw_bytes, hipMemcpyHostToDevice));
  if (tiled) {
    MK_CHECK(hipMalloc(&w, tiled_bytes));
    k_relayout_iq3s<<<256, kMkBlock>>>(RelayoutArgs{src, w, n, k});
  } else {
    w = src;
  }
  MK_CHECK(hipMalloc(&x, sizeof(float) * uint64_t(t) * k));
  MK_CHECK(hipMalloc(&y, sizeof(float) * uint64_t(t) * n));
  MK_CHECK(hipMemcpy(x, x_host, sizeof(float) * uint64_t(t) * k, hipMemcpyHostToDevice));
  MK_CHECK(hipMemset(y, 0xff, sizeof(float) * uint64_t(t) * n));
  QmmArgs a{};
  a.type = tiled ? q::IQ3_S_TILES : q::IQ3_S;
  a.w = w;
  a.ldw = q::row_bytes(a.type, k);
  a.n = n;
  a.k = k;
  a.t = t;
  a.x = x;
  a.ldx = k;
  a.y = y;
  a.ldy = n;
  const uint32_t rows_i = exact ? kQmmRows : QmmhMoe::kRows, toks_i = exact ? kQmmToks : QmmhMoe::kToks;
  const uint32_t items = ((n + rows_i - 1) / rows_i) * ((t + toks_i - 1) / toks_i);
  k_tile_qmm<<<grid ? grid : items, kMkBlock>>>(a, mode);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(y_host, y, sizeof(float) * uint64_t(t) * n, hipMemcpyDeviceToHost));
  if (tiled) MK_CHECK(hipFree(w));
  MK_CHECK(hipFree(src));
  MK_CHECK(hipFree(x));
  MK_CHECK(hipFree(y));
  return 0;
}

static void check_q8_shape(uint32_t n, uint32_t k) {
  if (n == 0 || n % 16 || k == 0 || k % 64) MK_DIE("Q8_0 tiles: %u x %u is not 16-row groups of 64-value unit pairs", n, k);
}

// raw: N rows of GGUF Q8_0; tiled_out: q8_tiled_bytes(N, K) (the same size), uploaded first so unwritten bytes show.
extern "C" int qwen38_relayout_q8(const void *raw, uint64_t raw_bytes, uint32_t n, uint32_t k, void *tiled_out,
                              uint64_t tiled_bytes) {
  check_q8_shape(n, k);
  const uint64_t bytes = q8_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_relayout_q8", "raw", raw_bytes, bytes);
  MK_CHECK_BYTES("qwen38_relayout_q8", "tiled_out", tiled_bytes, bytes);
  if (bytes != q::row_bytes(q::Q8_0, k) * n) MK_DIE("Q8_0 tiles: %lu bytes for %u x %u", (unsigned long)bytes, n, k);
  uint8_t *src, *dst;
  MK_CHECK(hipMalloc(&src, bytes));
  MK_CHECK(hipMalloc(&dst, bytes));
  MK_CHECK(hipMemcpy(src, raw, bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemcpy(dst, tiled_out, bytes, hipMemcpyHostToDevice));
  k_relayout_q8<<<256, kMkBlock>>>(RelayoutArgs{src, dst, n, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(tiled_out, dst, bytes, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(src));
  MK_CHECK(hipFree(dst));
  return 0;
}

extern "C" int qwen38_tile_dequant_q8(const void *tiled, uint64_t tiled_bytes, uint32_t n, uint32_t k, float *out_host,
                                  uint64_t out_bytes) {
  check_q8_shape(n, k);
  const uint64_t bytes = q8_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_tile_dequant_q8", "tiled", tiled_bytes, bytes);
  MK_CHECK_BYTES("qwen38_tile_dequant_q8", "out", out_bytes, sizeof(float) * uint64_t(n) * k);
  uint8_t *t;
  float *out;
  MK_CHECK(hipMalloc(&t, bytes));
  MK_CHECK(hipMalloc(&out, sizeof(float) * uint64_t(n) * k));
  MK_CHECK(hipMemcpy(t, tiled, bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemset(out, 0xff, sizeof(float) * uint64_t(n) * k));
  k_tile_dequant_q8<<<256, kMkBlock>>>(t, n, k, out);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(out_host, out, sizeof(float) * uint64_t(n) * k, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(t));
  MK_CHECK(hipFree(out));
  return 0;
}

// y = W x for Q8_0 rows (re-laid out here when tiled): mode bit 0 = tiles, bit 1 = the exact kernel (else the f16
// dense path with `ksplit` K slices, then their sum). grid 0: 1024 workgroups.
extern "C" int qwen38_tile_qmm_q8(const void *raw, uint64_t raw_size, uint32_t n, uint32_t k, const float *x_host,
                              uint64_t x_size, uint32_t t, float *y_host, uint64_t y_size, uint32_t mode, uint32_t ksplit,
                              uint32_t grid) {
  check_q8_shape(n, k);
  if (mode > 3) MK_DIE("qwen38_tile_qmm_q8: mode %u", mode);
  const bool tiled = mode & 1, exact = mode & 2;
  if (t == 0) MK_DIE("qwen38_tile_qmm_q8: no tokens");
  if (exact && ksplit > 1) MK_DIE("qwen38_tile_qmm_q8: the exact kernel has no split");
  if (!exact && !qmmh_ksplit_ok(k, ksplit)) MK_DIE("qwen38_tile_qmm_q8: ksplit %u leaves an empty slice of K=%u", ksplit, k);
  if (ksplit > kMaxKsplit) MK_DIE("qwen38_tile_qmm_q8: ksplit %u > %u", ksplit, kMaxKsplit);
  const uint64_t bytes = q8_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_tile_qmm_q8", "raw", raw_size, bytes);
  MK_CHECK_BYTES("qwen38_tile_qmm_q8", "x", x_size, sizeof(float) * uint64_t(t) * k);
  MK_CHECK_BYTES("qwen38_tile_qmm_q8", "y", y_size, sizeof(float) * uint64_t(t) * n);
  uint8_t *src, *w;
  float *x, *y, *part = nullptr;
  MK_CHECK(hipMalloc(&src, bytes));
  MK_CHECK(hipMemcpy(src, raw, bytes, hipMemcpyHostToDevice));
  if (tiled) {
    MK_CHECK(hipMalloc(&w, bytes));
    k_relayout_q8<<<256, kMkBlock>>>(RelayoutArgs{src, w, n, k});
  } else {
    w = src;
  }
  MK_CHECK(hipMalloc(&x, sizeof(float) * uint64_t(t) * k));
  MK_CHECK(hipMalloc(&y, sizeof(float) * uint64_t(t) * n));
  MK_CHECK(hipMemcpy(x, x_host, sizeof(float) * uint64_t(t) * k, hipMemcpyHostToDevice));
  MK_CHECK(hipMemset(y, 0xff, sizeof(float) * uint64_t(t) * n));
  if (ksplit > 1) {
    MK_CHECK(hipMalloc(&part, sizeof(float) * uint64_t(ksplit) * t * n));
    MK_CHECK(hipMemset(part, 0xff, sizeof(float) * uint64_t(ksplit) * t * n));
  }
  QmmArgs a{};
  a.type = tiled ? q::Q8_0_TILES : q::Q8_0;
  a.w = w;
  a.ldw = q::row_bytes(a.type, k);
  a.n = n;
  a.k = k;
  a.t = t;
  a.x = x;
  a.ldx = k;
  a.y = y;
  a.ldy = n;
  a.ksplit = ksplit;
  a.part = part;
  const uint32_t G = grid ? grid : 1024;
  k_tile_qmm<<<G, kMkBlock>>>(a, mode | 4);
  if (ksplit > 1) k_tile_qmm<<<G, kMkBlock>>>(a, 8);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(y_host, y, sizeof(float) * uint64_t(t) * n, hipMemcpyDeviceToHost));
  for (void *p : {static_cast<void *>(src), static_cast<void *>(x), static_cast<void *>(y), static_cast<void *>(part)})
    if (p) MK_CHECK(hipFree(p));
  if (tiled) MK_CHECK(hipFree(w));
  return 0;
}

static void check_q6_shape(uint32_t n, uint32_t k) {
  if (n == 0 || n % 16 || k == 0 || k % 256) MK_DIE("Q6_K tiles: %u x %u is not 16-row groups of 256-value superblocks", n, k);
}

// raw: N rows of GGUF Q6_K; tiled_out: q6_tiled_bytes(N, K) (the same size), uploaded first so unwritten bytes show.
extern "C" int qwen38_relayout_q6(const void *raw, uint64_t raw_bytes, uint32_t n, uint32_t k, void *tiled_out,
                              uint64_t tiled_bytes) {
  check_q6_shape(n, k);
  const uint64_t bytes = q6_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_relayout_q6", "raw", raw_bytes, bytes);
  MK_CHECK_BYTES("qwen38_relayout_q6", "tiled_out", tiled_bytes, bytes);
  if (bytes != q::row_bytes(q::Q6_K, k) * n) MK_DIE("Q6_K tiles: %lu bytes for %u x %u", (unsigned long)bytes, n, k);
  uint8_t *src, *dst;
  MK_CHECK(hipMalloc(&src, bytes));
  MK_CHECK(hipMalloc(&dst, bytes));
  MK_CHECK(hipMemcpy(src, raw, bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemcpy(dst, tiled_out, bytes, hipMemcpyHostToDevice));
  k_relayout_q6<<<256, kMkBlock>>>(RelayoutArgs{src, dst, n, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(tiled_out, dst, bytes, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(src));
  MK_CHECK(hipFree(dst));
  return 0;
}

extern "C" int qwen38_tile_dequant_q6(const void *tiled, uint64_t tiled_bytes, uint32_t n, uint32_t k, float *out_host,
                                  uint64_t out_bytes) {
  check_q6_shape(n, k);
  const uint64_t bytes = q6_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_tile_dequant_q6", "tiled", tiled_bytes, bytes);
  MK_CHECK_BYTES("qwen38_tile_dequant_q6", "out", out_bytes, sizeof(float) * uint64_t(n) * k);
  uint8_t *t;
  float *out;
  MK_CHECK(hipMalloc(&t, bytes));
  MK_CHECK(hipMalloc(&out, sizeof(float) * uint64_t(n) * k));
  MK_CHECK(hipMemcpy(t, tiled, bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemset(out, 0xff, sizeof(float) * uint64_t(n) * k));
  k_tile_dequant_q6<<<256, kMkBlock>>>(t, n, k, out);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(out_host, out, sizeof(float) * uint64_t(n) * k, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(t));
  MK_CHECK(hipFree(out));
  return 0;
}

// y = W x for Q6_K rows (re-laid out here when tiled): mode bit 0 = tiles, bit 1 = the exact kernel (else the f16
// dense path with `ksplit` K slices, then their sum). grid 0: 1024 workgroups.
extern "C" int qwen38_tile_qmm_q6(const void *raw, uint64_t raw_size, uint32_t n, uint32_t k, const float *x_host,
                              uint64_t x_size, uint32_t t, float *y_host, uint64_t y_size, uint32_t mode, uint32_t ksplit,
                              uint32_t grid) {
  check_q6_shape(n, k);
  if (mode > 3) MK_DIE("qwen38_tile_qmm_q6: mode %u", mode);
  const bool tiled = mode & 1, exact = mode & 2;
  if (t == 0) MK_DIE("qwen38_tile_qmm_q6: no tokens");
  if (exact && ksplit > 1) MK_DIE("qwen38_tile_qmm_q6: the exact kernel has no split");
  if (!exact && !qmmh_ksplit_ok(k, ksplit)) MK_DIE("qwen38_tile_qmm_q6: ksplit %u leaves an empty slice of K=%u", ksplit, k);
  if (ksplit > kMaxKsplit) MK_DIE("qwen38_tile_qmm_q6: ksplit %u > %u", ksplit, kMaxKsplit);
  const uint64_t bytes = q6_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_tile_qmm_q6", "raw", raw_size, bytes);
  MK_CHECK_BYTES("qwen38_tile_qmm_q6", "x", x_size, sizeof(float) * uint64_t(t) * k);
  MK_CHECK_BYTES("qwen38_tile_qmm_q6", "y", y_size, sizeof(float) * uint64_t(t) * n);
  uint8_t *src, *w;
  float *x, *y, *part = nullptr;
  MK_CHECK(hipMalloc(&src, bytes));
  MK_CHECK(hipMemcpy(src, raw, bytes, hipMemcpyHostToDevice));
  if (tiled) {
    MK_CHECK(hipMalloc(&w, bytes));
    k_relayout_q6<<<256, kMkBlock>>>(RelayoutArgs{src, w, n, k});
  } else {
    w = src;
  }
  MK_CHECK(hipMalloc(&x, sizeof(float) * uint64_t(t) * k));
  MK_CHECK(hipMalloc(&y, sizeof(float) * uint64_t(t) * n));
  MK_CHECK(hipMemcpy(x, x_host, sizeof(float) * uint64_t(t) * k, hipMemcpyHostToDevice));
  MK_CHECK(hipMemset(y, 0xff, sizeof(float) * uint64_t(t) * n));
  if (ksplit > 1) {
    MK_CHECK(hipMalloc(&part, sizeof(float) * uint64_t(ksplit) * t * n));
    MK_CHECK(hipMemset(part, 0xff, sizeof(float) * uint64_t(ksplit) * t * n));
  }
  QmmArgs a{};
  a.type = tiled ? q::Q6_K_TILES : q::Q6_K;
  a.w = w;
  a.ldw = q::row_bytes(a.type, k);
  a.n = n;
  a.k = k;
  a.t = t;
  a.x = x;
  a.ldx = k;
  a.y = y;
  a.ldy = n;
  a.ksplit = ksplit;
  a.part = part;
  const uint32_t G = grid ? grid : 1024;
  k_tile_qmm<<<G, kMkBlock>>>(a, mode | 4);  // bit 2: the dense shape (the type is in a.type)
  if (ksplit > 1) k_tile_qmm<<<G, kMkBlock>>>(a, 8);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(y_host, y, sizeof(float) * uint64_t(t) * n, hipMemcpyDeviceToHost));
  for (void *p : {static_cast<void *>(src), static_cast<void *>(x), static_cast<void *>(y), static_cast<void *>(part)})
    if (p) MK_CHECK(hipFree(p));
  if (tiled) MK_CHECK(hipFree(w));
  return 0;
}

static void check_iq4_shape(uint32_t n, uint32_t k) {
  if (n == 0 || n % 16 || k == 0 || k % 128) MK_DIE("IQ4_NL tiles: %u x %u is not 16-row groups of 128-value tiles", n, k);
}

// raw: N rows of GGUF IQ4_NL; tiled_out: iq4_tiled_bytes(N, K) (the same size), uploaded first so unwritten bytes show.
extern "C" int qwen38_relayout_iq4nl(const void *raw, uint64_t raw_bytes, uint32_t n, uint32_t k, void *tiled_out,
                              uint64_t tiled_bytes) {
  check_iq4_shape(n, k);
  const uint64_t bytes = iq4_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_relayout_iq4nl", "raw", raw_bytes, bytes);
  MK_CHECK_BYTES("qwen38_relayout_iq4nl", "tiled_out", tiled_bytes, bytes);
  if (bytes != q::row_bytes(q::IQ4_NL, k) * n) MK_DIE("IQ4_NL tiles: %lu bytes for %u x %u", (unsigned long)bytes, n, k);
  uint8_t *src, *dst;
  MK_CHECK(hipMalloc(&src, bytes));
  MK_CHECK(hipMalloc(&dst, bytes));
  MK_CHECK(hipMemcpy(src, raw, bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemcpy(dst, tiled_out, bytes, hipMemcpyHostToDevice));
  k_relayout_iq4<<<256, kMkBlock>>>(RelayoutArgs{src, dst, n, k});
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(tiled_out, dst, bytes, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(src));
  MK_CHECK(hipFree(dst));
  return 0;
}

extern "C" int qwen38_tile_dequant_iq4nl(const void *tiled, uint64_t tiled_bytes, uint32_t n, uint32_t k, float *out_host,
                                  uint64_t out_bytes) {
  check_iq4_shape(n, k);
  const uint64_t bytes = iq4_tiled_bytes(n, k);
  MK_CHECK_BYTES("qwen38_tile_dequant_iq4nl", "tiled", tiled_bytes, bytes);
  MK_CHECK_BYTES("qwen38_tile_dequant_iq4nl", "out", out_bytes, sizeof(float) * uint64_t(n) * k);
  uint8_t *t;
  float *out;
  MK_CHECK(hipMalloc(&t, bytes));
  MK_CHECK(hipMalloc(&out, sizeof(float) * uint64_t(n) * k));
  MK_CHECK(hipMemcpy(t, tiled, bytes, hipMemcpyHostToDevice));
  MK_CHECK(hipMemset(out, 0xff, sizeof(float) * uint64_t(n) * k));
  k_tile_dequant_iq4<<<256, kMkBlock>>>(t, n, k, out);
  MK_CHECK(hipGetLastError());
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(out_host, out, sizeof(float) * uint64_t(n) * k, hipMemcpyDeviceToHost));
  MK_CHECK(hipFree(t));
  MK_CHECK(hipFree(out));
  return 0;
}
