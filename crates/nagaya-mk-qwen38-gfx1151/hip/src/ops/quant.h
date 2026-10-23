// SPDX-License-Identifier: MIT
// Device dequantization of the GGUF block formats this model ships, in units of 32 consecutive
// values: a whole Q8_0 / IQ4_NL block, or one 32-value sub-block of an IQ4_XS / IQ3_S / Q6_K
// superblock. Each value is computed with the same float expression, in the same order, as
// ggml's dequantize_row_* (ggml-quants.c at llama.cpp 5cf3a3528), so it is bit-identical to
// gguf-py and to our CPU dequant (test_quant.py checks that for every type).
// Block layouts and code tables come from src/gguf (types.hpp, tables.hpp: ggml, MIT).
#pragma once
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <utility>

#include "../gguf/tables.hpp"
#include "../gguf/types.hpp"

namespace mk::q {

enum Type : uint32_t {  // ggml type ids
  F32 = 0,
  Q8_0 = 8,
  Q6_K = 14,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ4_XS = 23,
  BF16 = 30,
  // Not a ggml type: IQ3_S re-laid out into read-once tiles (tiles.h). 256 values per superblock, 112 bytes per row
  // per superblock (1792-byte tiles of 16 rows); decoded only by the f16 path (qmmh_item_iq3s_tiles).
  IQ3_S_TILES = 121,
  // Not a ggml type: Q8_0 re-laid out into read-once tiles of 16 rows x one unit pair (tiles.h). The same bytes,
  // reordered: 64 values per 68 bytes of a row.
  Q8_0_TILES = 108,
  // Not a ggml type: Q6_K re-laid out into read-once tiles of 16 rows x one superblock (tiles.h). The same bytes,
  // reordered: 256 values per 210 bytes of a row, as Q6_K.
  Q6_K_TILES = 114,
  // Not a ggml type: IQ4_NL re-laid out into read-once tiles of 16 rows x 4 units (tiles.h), for the expert down matrices.
  // The same bytes, reordered: 128 values per 72 bytes of a row (4 x 18).
  IQ4_NL_TILES = 120,
};

// Values per block and bytes per block.
__host__ __device__ constexpr uint32_t block_values(uint32_t t) {
  return (t == Q8_0 || t == IQ4_NL) ? 32 : (t == F32 || t == BF16) ? 1 : t == Q8_0_TILES ? 64 : t == IQ4_NL_TILES ? 128 : 256;
}
__host__ __device__ constexpr uint32_t block_bytes(uint32_t t) {
  switch (t) {
    case F32: return 4;
    case BF16: return 2;
    case Q8_0: return 34;
    case IQ4_NL: return 18;
    case Q6_K: return 210;
    case IQ3_S: return 110;
    case IQ4_XS: return 136;
    case IQ3_S_TILES: return 112;
    case Q8_0_TILES: return 68;
    case Q6_K_TILES: return 210;
    case IQ4_NL_TILES: return 72;
  }
  return 0;
}
__host__ __device__ constexpr uint64_t row_bytes(uint32_t t, uint32_t k) {
  return uint64_t(k) / block_values(t) * block_bytes(t);
}

namespace detail {
template <class T, size_t N, size_t... I>
constexpr std::array<T, N> copy_table(const T (&src)[N], std::index_sequence<I...>) {
  return {src[I]...};
}
}  // namespace detail

__constant__ constexpr std::array<uint32_t, 512> kIq3sGridDev =
    detail::copy_table(mkgguf::tables::kIq3sGrid, std::make_index_sequence<512>{});
__constant__ constexpr std::array<int8_t, 16> kIq4nlDev =
    detail::copy_table(mkgguf::tables::kValuesIq4nl, std::make_index_sequence<16>{});

__device__ inline float h2f(uint16_t h) { return float(__builtin_bit_cast(_Float16, h)); }  // exact

__device__ inline uint16_t ld16(const uint8_t *p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }

// Unit `u` (32 values) of a row stored as blocks of type T, starting at `row`. Writes out[0..32).
template <uint32_t T>
__device__ inline void dequant32(const uint8_t *row, uint32_t u, float out[32]) {
  if constexpr (T == Q8_0) {
    const uint8_t *b = row + size_t(u) * 34;
    const float d = h2f(ld16(b));
#pragma unroll
    for (int j = 0; j < 32; ++j) out[j] = float(int8_t(b[2 + j])) * d;
  } else if constexpr (T == IQ4_NL) {
    const uint8_t *b = row + size_t(u) * 18;
    const float d = h2f(ld16(b));
    const uint8_t *qs = b + 2;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      out[j] = d * float(kIq4nlDev[qs[j] & 0xf]);
      out[j + 16] = d * float(kIq4nlDev[qs[j] >> 4]);
    }
  } else if constexpr (T == IQ4_XS) {
    const uint8_t *b = row + size_t(u / 8) * 136;
    const uint32_t ib = u % 8;
    const float d = h2f(ld16(b));
    const uint16_t scales_h = ld16(b + 2);
    const uint8_t *scales_l = b + 4;
    const uint8_t *qs = b + 8 + ib * 16;
    const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
    const float dl = d * float(ls - 32);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      out[j] = dl * float(kIq4nlDev[qs[j] & 0xf]);
      out[j + 16] = dl * float(kIq4nlDev[qs[j] >> 4]);
    }
  } else if constexpr (T == IQ3_S) {
    const uint8_t *b = row + size_t(u / 8) * 110;
    const uint32_t ib = u % 8;
    const float d = h2f(ld16(b));
    const uint8_t *qs = b + 2 + ib * 8;
    const uint8_t qh = b[2 + 64 + ib];
    const uint8_t *signs = b + 2 + 64 + 8 + ib * 4;
    const uint8_t sc = b[2 + 64 + 8 + 32 + ib / 2];
    const float db = d * float(1 + 2 * ((ib % 2) ? (sc >> 4) : (sc & 0xf)));
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = kIq3sGridDev[qs[2 * l + 0] | ((uint32_t(qh) << (8 - 2 * l)) & 256)];
      const uint32_t g2 = kIq3sGridDev[qs[2 * l + 1] | ((uint32_t(qh) << (7 - 2 * l)) & 256)];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        out[8 * l + j] = db * float((g1 >> (8 * j)) & 0xff) * ((signs[l] & (1u << j)) ? -1.f : 1.f);
        out[8 * l + 4 + j] = db * float((g2 >> (8 * j)) & 0xff) * ((signs[l] & (1u << (j + 4))) ? -1.f : 1.f);
      }
    }
  } else if constexpr (T == Q6_K) {
    const uint8_t *b = row + size_t(u / 8) * 210;
    const uint32_t s = u % 8, n = s / 4, q = s % 4;
    const uint8_t *ql = b + 64 * n;
    const uint8_t *qh = b + 128 + 32 * n;
    const int8_t *sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * n;
    const float d = h2f(ld16(b + 208));
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      int v;
      if (q == 0) v = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      else if (q == 1) v = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      else if (q == 2) v = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      else v = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      out[l] = d * float(sc[is + 2 * q]) * float(v);
    }
  } else if constexpr (T == F32) {
    const float *p = reinterpret_cast<const float *>(row) + size_t(u) * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) out[j] = p[j];
  } else if constexpr (T == BF16) {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(row) + size_t(u) * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) out[j] = __uint_as_float(uint32_t(p[j]) << 16);
  } else {
    static_assert(T == F32, "unsupported quant type");
  }
}


// The engine's decoder: unit `u` of a row, emitted 8 values at a time (chunk c = values 8c..8c+7),
// so a kernel never holds a unit's 32 floats at once. Each value is the same float expression as
// dequant32's (and so ggml's): the bits are identical, which test_quant (gguf-py) and the qmm
// oracle tests check. The constructor reads the unit's header (scales) once.
template <uint32_t T>
struct Unit;

template <>
struct Unit<Q8_0> {
  const uint8_t *b;
  float d;
  __device__ Unit(const uint8_t *row, uint32_t u) : b(row + size_t(u) * 34), d(h2f(ld16(b))) {}
  __device__ void get8(uint32_t c, float out[8]) const {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = float(int8_t(b[2 + 8 * c + j])) * d;
  }
};

// IQ4_NL and IQ4_XS: values 0..15 are the low nibbles of qs[0..15], values 16..31 the high ones.
template <>
struct Unit<IQ4_NL> {
  const uint8_t *qs;
  float d;
  __device__ Unit(const uint8_t *row, uint32_t u) : qs(row + size_t(u) * 18 + 2), d(h2f(ld16(row + size_t(u) * 18))) {}
  __device__ void get8(uint32_t c, float out[8]) const {
    const uint8_t *p = qs + 8 * (c % 2);
    const uint32_t sh = c < 2 ? 0 : 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = d * float(kIq4nlDev[(p[j] >> sh) & 0xf]);
  }
};

template <>
struct Unit<IQ4_XS> {
  const uint8_t *qs;
  float dl;
  __device__ Unit(const uint8_t *row, uint32_t u) {
    const uint8_t *b = row + size_t(u / 8) * 136;
    const uint32_t ib = u % 8;
    const float d = h2f(ld16(b));
    const uint16_t scales_h = ld16(b + 2);
    const uint8_t *scales_l = b + 4;
    qs = b + 8 + ib * 16;
    const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
    dl = d * float(ls - 32);
  }
  __device__ void get8(uint32_t c, float out[8]) const {
    const uint8_t *p = qs + 8 * (c % 2);
    const uint32_t sh = c < 2 ? 0 : 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = dl * float(kIq4nlDev[(p[j] >> sh) & 0xf]);
  }
};

// IQ3_S: chunk c is dequant32's l = c (two grid entries of 4 values each).
template <>
struct Unit<IQ3_S> {
  const uint8_t *qs, *signs;
  uint8_t qh;
  float db;
  __device__ Unit(const uint8_t *row, uint32_t u) {
    const uint8_t *b = row + size_t(u / 8) * 110;
    const uint32_t ib = u % 8;
    const float d = h2f(ld16(b));
    qs = b + 2 + ib * 8;
    qh = b[2 + 64 + ib];
    signs = b + 2 + 64 + 8 + ib * 4;
    const uint8_t sc = b[2 + 64 + 8 + 32 + ib / 2];
    db = d * float(1 + 2 * ((ib % 2) ? (sc >> 4) : (sc & 0xf)));
  }
  __device__ void get8(uint32_t l, float out[8]) const {
    const uint32_t g1 = kIq3sGridDev[qs[2 * l + 0] | ((uint32_t(qh) << (8 - 2 * l)) & 256)];
    const uint32_t g2 = kIq3sGridDev[qs[2 * l + 1] | ((uint32_t(qh) << (7 - 2 * l)) & 256)];
    const uint8_t sg = signs[l];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = db * float((g1 >> (8 * j)) & 0xff) * ((sg & (1u << j)) ? -1.f : 1.f);
      out[4 + j] = db * float((g2 >> (8 * j)) & 0xff) * ((sg & (1u << (j + 4))) ? -1.f : 1.f);
    }
  }
};

// Q6_K: unit u is sub-block s = u % 8 of its superblock (n = s / 4, q = s % 4); value l of the unit
// uses scale sc[l / 16 + 2 q].
template <>
struct Unit<Q6_K> {
  const uint8_t *ql, *qh;
  const int8_t *sc;
  uint32_t q;
  float d;
  __device__ Unit(const uint8_t *row, uint32_t u) {
    const uint8_t *b = row + size_t(u / 8) * 210;
    const uint32_t s = u % 8, n = s / 4;
    q = s % 4;
    ql = b + 64 * n + (q % 2) * 32;
    qh = b + 128 + 32 * n;
    sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * n + 2 * q;
    d = h2f(ld16(b + 208));
  }
  __device__ void get8(uint32_t c, float out[8]) const {
    const uint32_t lo_sh = q < 2 ? 0 : 4, hi_sh = 2 * q;
    const float ds = d * float(sc[c / 2]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t l = 8 * c + j;
      const int v = int(((ql[l] >> lo_sh) & 0xF) | (((qh[l] >> hi_sh) & 3) << 4)) - 32;
      out[j] = ds * float(v);
    }
  }
};

template <>
struct Unit<F32> {
  const float *p;
  __device__ Unit(const uint8_t *row, uint32_t u) : p(reinterpret_cast<const float *>(row) + size_t(u) * 32) {}
  __device__ void get8(uint32_t c, float out[8]) const {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = p[8 * c + j];
  }
};

template <>
struct Unit<BF16> {
  const uint16_t *p;
  __device__ Unit(const uint8_t *row, uint32_t u) : p(reinterpret_cast<const uint16_t *>(row) + size_t(u) * 32) {}
  __device__ void get8(uint32_t c, float out[8]) const {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = __uint_as_float(uint32_t(p[8 * c + j]) << 16);
  }
};

}  // namespace mk::q
