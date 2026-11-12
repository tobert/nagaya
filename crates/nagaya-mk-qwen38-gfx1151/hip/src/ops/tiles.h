// SPDX-License-Identifier: MIT
// Read-once weight tiles (read-once weight fetch). The GGUF stores a matrix row-major, and a
// decoder that follows it makes each lane read scattered 1-4 byte pieces of 16 different rows, so every cache
// line is fetched many times (probes/fetch.hip: 110 GB/s for IQ3_S experts). Here each matrix is re-laid out
// once, at load, into tiles stored in the order the lanes of a wave consume them: each lane's bytes are
// contiguous and aligned, every byte is read once by one lane, and a wave streams its tiles in order (the
// probe: 223 GB/s, a plain stream 231). The values are unchanged: decoding a tile gives the same bits as
// decoding the GGUF row (tests/ops/test_tiles.py).
//
// IQ3_S (the expert gate and up matrices): one tile per 16 rows x one 256-value superblock, 1792 B. Lane l
// (0..31) handles row l % 16 and decodes units ib = 2p + h, p = 0..3, h = l / 16:
//   bytes    0 .. 1023   grid indices: lane l at 32 l, unit p's 8 bytes at + 8 p   (raw qs[8 ib .. 8 ib + 7])
//   bytes 1024 .. 1535   signs:        lane l at 1024 + 16 l, unit p's 4 at + 4 p   (raw signs[4 ib .. + 3])
//   bytes 1536 .. 1791   meta:         lane l at 1536 + 8 l: [0..3] qh of units p = 0..3 (raw qh[ib]),
//                        [4] scale nibbles of p = 0 (low) and 1 (high), [5] of p = 2 and 3, [6..7] d (f16).
// 1792 = 16 x 110 + 32: the only extra bytes are d, stored once per half-wave. Tiles of a matrix are ordered by
// 16-row group, then superblock, so a row group's tiles are contiguous: 112 B per row per superblock, and an
// N-row matrix of K values takes N * K / 256 * 112 bytes (an expert's stride).
#pragma once
#include "common.h"
#include "quant.h"

namespace qwen38 {

constexpr uint32_t kIq3sTile = 1792;
constexpr uint32_t kIq3sTileRowBytes = kIq3sTile / 16;  // 112 per row per superblock

// Bytes of an N x K IQ3_S matrix in tiles.
__host__ __device__ constexpr uint64_t iq3s_tiled_bytes(uint32_t n, uint32_t k) {
  return uint64_t(n / 16) * (k / 256) * kIq3sTile;
}

// ---- re-layout: GGUF row-major IQ3_S -> tiles. Item = one tile; thread t < 32 writes lane t's 56 bytes. ----
struct RelayoutArgs {
  const uint8_t *src;  // N rows of K / 256 * 110 bytes
  uint8_t *dst;        // iq3s_tiled_bytes(N, K)
  uint32_t n, k;
};

__device__ inline void op_relayout_iq3s(const RelayoutArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t nsb = a.k / 256, tiles = (a.n / 16) * nsb;
  for (uint32_t i = wg; i < tiles; i += nwg) {
    const uint32_t g = i / nsb, sb = i % nsb, l = thread_id();
    if (l >= 32) continue;
    const uint32_t r = l % 16, h = l / 16;
    const uint8_t *b = a.src + uint64_t(g * 16 + r) * nsb * 110 + sb * 110;
    uint8_t *t = a.dst + uint64_t(i) * kIq3sTile;
    auto nib = [&](uint32_t ib) { return uint32_t(b[2 + 64 + 8 + 32 + ib / 2] >> (4 * (ib % 2))) & 0xf; };
    for (uint32_t p = 0; p < 4; ++p) {
      const uint32_t ib = 2 * p + h;
      for (uint32_t j = 0; j < 8; ++j) t[32 * l + 8 * p + j] = b[2 + 8 * ib + j];
      for (uint32_t j = 0; j < 4; ++j) t[1024 + 16 * l + 4 * p + j] = b[2 + 64 + 8 + 4 * ib + j];
      t[1536 + 8 * l + p] = b[2 + 64 + ib];
    }
    t[1536 + 8 * l + 4] = uint8_t(nib(h) | nib(2 + h) << 4);
    t[1536 + 8 * l + 5] = uint8_t(nib(4 + h) | nib(6 + h) << 4);
    t[1536 + 8 * l + 6] = b[0];
    t[1536 + 8 * l + 7] = b[1];
  }
}

// ---- decode: lane l's 56 bytes of one tile, then any of its 4 units ----
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

struct Iq3sLane {
  u32x4 q0, q1, sg;  // grid indices of units p = 0, 1 (q0) and 2, 3 (q1); signs of p = 0..3
  u32x2 m;           // qh x4 | scale nibbles x4, d

  // Three wide loads from the tile at `tile` (16-byte aligned), lane `l`.
  __device__ __attribute__((always_inline)) void load(const uint8_t *tile, uint32_t l) {
    q0 = *reinterpret_cast<const u32x4 *>(tile + 32 * l);
    q1 = *reinterpret_cast<const u32x4 *>(tile + 32 * l + 16);
    sg = *reinterpret_cast<const u32x4 *>(tile + 1024 + 16 * l);
    m = *reinterpret_cast<const u32x2 *>(tile + 1536 + 8 * l);
  }
  // The unit's scale: d * (1 + 2 * nibble), as q::dequant32<IQ3_S> computes db.
  __device__ __attribute__((always_inline)) float scale(uint32_t p) const {
    const float d = q::h2f(uint16_t(m.y >> 16));
    const uint32_t nb = (m.y >> (4 * p)) & 0xf;
    return d * float(1 + 2 * nb);
  }
  // The unit's 32 signed codes (grid byte times sign), value j of the unit at out[j]: dequant32's value j is
  // exactly scale(p) * out[j]. OutT = int (tests) or _Float16 (the matmul: every code is exact in f16).
  template <class OutT>
  __device__ __attribute__((always_inline)) void codes(uint32_t p, OutT out[32]) const {
    const uint32_t lo = p < 2 ? (p == 0 ? q0.x : q0.z) : (p == 2 ? q1.x : q1.z);
    const uint32_t hi = p < 2 ? (p == 0 ? q0.y : q0.w) : (p == 2 ? q1.y : q1.w);
    const uint32_t signs = p == 0 ? sg.x : p == 1 ? sg.y : p == 2 ? sg.z : sg.w;
    const uint32_t qh = (m.x >> (8 * p)) & 0xff;
#pragma unroll
    for (uint32_t l = 0; l < 4; ++l) {
      const uint32_t w2 = l < 2 ? lo : hi, sh = 16 * (l % 2);
      const uint32_t i1 = (w2 >> sh) & 0xff, i2 = (w2 >> (sh + 8)) & 0xff;
      const uint32_t g1 = q::kIq3sGridDev[i1 | ((qh << (8 - 2 * l)) & 256)];
      const uint32_t g2 = q::kIq3sGridDev[i2 | ((qh << (7 - 2 * l)) & 256)];
      const uint32_t sgn = (signs >> (8 * l)) & 0xff;
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const int v1 = int((g1 >> (8 * j)) & 0xff), v2 = int((g2 >> (8 * j)) & 0xff);
        out[8 * l + j] = OutT((sgn & (1u << j)) ? -v1 : v1);
        out[8 * l + 4 + j] = OutT((sgn & (1u << (j + 4))) ? -v2 : v2);
      }
    }
  }
};

// The matmul's decode of a unit pair p: the same 32 f16 codes as codes<_Float16>(p), packed two per word (word i =
// values 2i, 2i + 1), built without the int -> f32 -> f16 converts. An IQ3_S grid byte v is an odd integer, 1..15.
// Taking a grid byte as the low byte of an f16's bits under the high byte 0x64 gives 0x6400 | v, which is 1024 + v:
// the exponent field is fixed at 2^10, the ulp there is exactly 1, and v < 1024 fills the mantissa, so the value is
// exact and subtracting 1024 (exact: a small integer) leaves v. Zero never occurs (v >= 1), so the sign flip below is
// never a -0 where the old negate gave +0. The sign flips bit 15 (an XOR: what the old negate did to a nonzero value).
// One v_perm_b32 spreads two grid bytes to two halves; the grid is read from `grid` (the 512 words, in LDS).
__device__ __attribute__((always_inline)) inline void iq3s_codes_f16w(const Iq3sLane &L, uint32_t p, int32_t w[16],
                                                                     const uint32_t *grid) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  const uint32_t lo = p < 2 ? (p == 0 ? L.q0.x : L.q0.z) : (p == 2 ? L.q1.x : L.q1.z);
  const uint32_t hi = p < 2 ? (p == 0 ? L.q0.y : L.q0.w) : (p == 2 ? L.q1.y : L.q1.w);
  const uint32_t signs = p == 0 ? L.sg.x : p == 1 ? L.sg.y : p == 2 ? L.sg.z : L.sg.w;
  const uint32_t qh = (L.m.x >> (8 * p)) & 0xff;
  const h2 bias = {_Float16(-1024.f), _Float16(-1024.f)};
#pragma unroll
  for (uint32_t l = 0; l < 4; ++l) {
    const uint32_t w2 = l < 2 ? lo : hi, sh = 16 * (l % 2);
    const uint32_t i1 = (w2 >> sh) & 0xff, i2 = (w2 >> (sh + 8)) & 0xff;
    const uint32_t g[2] = {grid[i1 | ((qh << (8 - 2 * l)) & 256)], grid[i2 | ((qh << (7 - 2 * l)) & 256)]};
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {  // word k of this l: grid g[k / 2], bytes 2 (k % 2), + 1; sign bits 2k, 2k + 1
      const uint32_t f = __builtin_amdgcn_perm(0x64646464u, g[k / 2], k % 2 ? 0x04030402u : 0x04010400u);
      const uint32_t t = (signs >> (8 * l + 2 * k)) & 3u;
      const uint32_t m = (t * 0x40008000u) & 0x80008000u;  // bit 0 -> 15, bit 1 -> 31
      const h2 v = __builtin_bit_cast(h2, f) + bias;
      w[4 * l + k] = int32_t(__builtin_bit_cast(uint32_t, v) ^ m);
    }
  }
}

// ---- Q8_0 (the dense projections: the dense f16 path) ----
// One tile per 16 rows x one unit pair (64 values), 1088 B. Lane slot s = r + 16 u (row r of the group, unit u of the
// pair; in the f16 path, the slot lane s decodes a half of): bytes 32 s .. 32 s + 31 are the unit's codes (raw
// qs[0..31]), bytes 1024 + 2 s .. + 1 its scale d (f16). 1088 = 16 x 2 x 34: no bytes added. Tiles are ordered by
// 16-row group, then pair, so a row group's tiles are contiguous; the f16 path reads a unit's 16-code half with one
// 16-byte load, and a wave's 16 rows x one unit as 512 contiguous bytes (the GGUF layout: 16 two-byte loads per lane
// from 16 rows 34 B apart per unit).
constexpr uint32_t kQ8Tile = 1088;

__host__ __device__ constexpr uint64_t q8_tiled_bytes(uint32_t n, uint32_t k) { return uint64_t(n / 16) * (k / 64) * kQ8Tile; }

// The tile holding unit u of row group g, in an N x K matrix's tiles at w.
__host__ __device__ inline const uint8_t *q8_tile(const uint8_t *w, uint32_t k, uint32_t g, uint32_t u) {
  return w + (uint64_t(g) * (k / 64) + u / 2) * kQ8Tile;
}

// Item = one tile; thread s < 32 writes slot s's 34 bytes.
__device__ inline void op_relayout_q8(const RelayoutArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t pairs = a.k / 64, tiles = (a.n / 16) * pairs;
  for (uint32_t i = wg; i < tiles; i += nwg) {
    const uint32_t g = i / pairs, pr = i % pairs, s = thread_id();
    if (s >= 32) continue;
    const uint32_t r = s % 16, u = 2 * pr + s / 16;
    const uint8_t *b = a.src + uint64_t(g * 16 + r) * (a.k / 32) * 34 + uint64_t(u) * 34;
    uint8_t *t = a.dst + uint64_t(i) * kQ8Tile;
    for (uint32_t j = 0; j < 32; ++j) t[32 * s + j] = b[2 + j];
    t[1024 + 2 * s] = b[0];
    t[1024 + 2 * s + 1] = b[1];
  }
}

// ---- Q6_K (the output head: the Q6_K output head) ----
// One tile per 16 rows x one 256-value superblock, 3360 B = 16 x 210: the GGUF's bytes, reordered, none added. The
// shape follows D1's Q8_0 reasoning, worked for this type:
//  - 16 rows because a wave's WMMA takes 16 weight rows (B = W^T, N = 16), and lane = r + 16 h (row r, half h of a
//    unit: the 16 codes a lane decodes) is what the f16 path already does, so every lane reads its own bytes.
//  - One superblock because that is Q6_K's whole self-contained unit (ql 128 + qh 64 + scales 16 + d 2) and 8 units
//    is the decode path's K chunk (kQ8tChunkUnits: 16 tokens x 8 units of f16 fill the LDS arena). A wave's chunk is
//    then one 3360-byte tile, and a row group's superblocks are consecutive tiles, so a wave streams K in order.
//  - Every load instruction of a wave reads 512 contiguous bytes, one 16-byte load per lane (lane order r + 16 h in
//    the address: 32 r + 16 h). Lane (r, h) needs, for the superblock's 8 units (n = the 128-value half, q = the
//    quarter within it), 16-byte runs of ql (4 of them: raw ql[64 n + 32 m + 16 h + j], m = q % 2, index c = 2 n + m),
//    16-byte runs of qh (2: raw qh[32 n + 16 h + j]), and the row's scales and d.
//  bytes    0 .. 2047   ql: 512 c + 32 r + 16 h, c = 0..3
//  bytes 2048 .. 3071   qh: 2048 + 512 n + 32 r + 16 h, n = 0..1
//  bytes 3072 .. 3327   scales: 3072 + 16 r, the row's 16 int8 scales verbatim (a unit's two 16-value scales sit in
//                       different h, and each lane needs both: both halves of a row read the same 16 bytes, so no
//                       shuffle and no duplicated storage)
//  bytes 3328 .. 3359   d: 3328 + 2 r, the row's f16 d
// 3360 is a multiple of 16, so every tile is 16-byte aligned. A lane's superblock is 6 wide loads plus the 16 scale
// bytes and d: 8 loads per 256 values (Q8_0 tiles: 16 per 256).
constexpr uint32_t kQ6Tile = 3360;

__host__ __device__ constexpr uint64_t q6_tiled_bytes(uint32_t n, uint32_t k) { return uint64_t(n / 16) * (k / 256) * kQ6Tile; }

// The tile holding superblock sb of row group g, in an N x K matrix's tiles at w.
__host__ __device__ inline const uint8_t *q6_tile(const uint8_t *w, uint32_t k, uint32_t g, uint32_t sb) {
  return w + (uint64_t(g) * (k / 256) + sb) * kQ6Tile;
}

// Item = one tile; thread s < 32 (lane slot: row r = s % 16, half h = s / 16) writes its ql and qh runs, and the
// h = 0 threads the row's scales and d.
__device__ inline void op_relayout_q6(const RelayoutArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t nsb = a.k / 256, tiles = (a.n / 16) * nsb;
  for (uint32_t i = wg; i < tiles; i += nwg) {
    const uint32_t g = i / nsb, sb = i % nsb, s = thread_id();
    if (s >= 32) continue;
    const uint32_t r = s % 16, h = s / 16;
    const uint8_t *b = a.src + uint64_t(g * 16 + r) * nsb * 210 + uint64_t(sb) * 210;
    uint8_t *t = a.dst + uint64_t(i) * kQ6Tile;
    for (uint32_t c = 0; c < 4; ++c)
      for (uint32_t j = 0; j < 16; ++j) t[512 * c + 32 * r + 16 * h + j] = b[64 * (c / 2) + 32 * (c % 2) + 16 * h + j];
    for (uint32_t n = 0; n < 2; ++n)
      for (uint32_t j = 0; j < 16; ++j) t[2048 + 512 * n + 32 * r + 16 * h + j] = b[128 + 32 * n + 16 * h + j];
    if (h == 0) {
      for (uint32_t j = 0; j < 16; ++j) t[3072 + 16 * r + j] = b[192 + j];
      t[3328 + 2 * r] = b[208];
      t[3328 + 2 * r + 1] = b[209];
    }
  }
}

// Unit s (0..7) of row r of a tile, as q::dequant32<Q6_K> computes it: out[l] = d * float(scale) * float(code), the
// same expression on the same bytes (the exact kernel and the tests).
__device__ inline void q6_tile_dequant32(const uint8_t *tile, uint32_t r, uint32_t s, float out[32]) {
  const uint32_t n = s / 4, qq = s % 4;
  const float d = q::h2f(q::ld16(tile + 3328 + 2 * r));
  const int8_t *sc = reinterpret_cast<const int8_t *>(tile + 3072 + 16 * r) + 8 * n;
#pragma unroll
  for (int l = 0; l < 32; ++l) {
    const uint32_t h = l / 16, j = l % 16;
    const uint32_t cl = tile[512 * (2 * n + qq % 2) + 32 * r + 16 * h + j];
    const uint32_t ch = tile[2048 + 512 * n + 32 * r + 16 * h + j];
    const int v = int((qq < 2 ? (cl & 0xF) : (cl >> 4)) | (((ch >> (2 * qq)) & 3) << 4)) - 32;
    out[l] = d * float(sc[h + 2 * qq]) * float(v);
  }
}

// ---- IQ4_NL (the expert down matrices: the expert down matrices) ----
// 43 of the 48 layers' ffn_down_exps ([512 experts][2560 rows][K = 640], per expert contiguous) are IQ4_NL: 32-value
// blocks of 18 bytes, an f16 d and 16 bytes of qs, where value j < 16 is table[qs[j] & 15] and value 16 + j is
// table[qs[j] >> 4]. A T = 1 step reads ~10 experts x 0.92 MB per layer; the GGUF layout has each lane read 16
// scattered 1-2 byte pieces of a row (119 GB/s at grid 80 against ~215 for a stream). The tile is worked from the
// consumer (qmmh.h, qmmh_item_moe_tiles: the MoE shape, 8 waves x 16 rows, one block of <= 16 tokens):
//  - 16 rows because a wave's WMMA takes 16 weight rows (B = W^T, N = 16); lane slot s = r + 16 h is row r of the
//    group and half h of the two lanes of a row that split a unit between them.
//  - The unit's 16 qs bytes carry both its 16-value halves (low and high nibbles), and a lane needs 16 codes. Splitting
//    by nibble would make both half-wave lanes of a row read the same 16 bytes (half the load's bytes wasted). Split
//    by byte instead: lane (r, h) takes qs[8 h .. 8 h + 7], decodes the low nibbles to values 8 h .. 8 h + 7 and the
//    high nibbles to values 16 + 8 h .. 16 + 8 h + 7, and one exchange with the partner lane (shfl_xor 16) gives both
//    the 16-value B fragments (b_lo = values 0..15 = the h = 0 lane's low run then the h = 1 lane's, b_hi likewise).
//  - A lane then needs 8 bytes of a unit; two units in one 16-byte load (units 2 p and 2 p + 1) makes every load
//    instruction of the wave read 512 contiguous bytes, one 16-byte load per lane, and a tile of 4 units (2 such
//    loads) is 1024 B of qs.
//  - d: a lane needs the row's d for each unit (both lanes of a row the same). 4 units x 2 B = 8 B a row, 128 B a
//    tile, one 8-byte load per lane (the two lanes of a row read the same 8 bytes: one cache line, as the Q6_K
//    tile's scales).
//  - 4 units (128 values, 1152 B = 16 x 4 x 18, no byte added) because K = 640 = 5 tiles: a K chunk of the decode
//    path is 8 units (16 tokens x 8 units of f16 fill the LDS arena) = 2 tiles, and 640 = 8 + 8 + 4 units. A whole
//    5-tile K in flight would take 50 VGPRs a lane; 2 tiles is 20 (k_mega's budget).
//  bytes    0 .. 1023   qs: 512 p + 16 s + 8 v + i = raw qs[8 h + i] of unit 2 p + v   (p, v = 0..1, i = 0..7, s = r + 16 h)
//  bytes 1024 .. 1151   d:  1024 + 8 r + 2 u = the f16 d of row r's unit u (u = 0..3)
// Tiles are ordered by 16-row group, then tile (K order), so a row group's tiles are contiguous: 72 B a row per tile,
// 360 B a row for K = 640, and an expert's stride (2560 rows) is 921600 B, a multiple of the 16-byte alignment.
constexpr uint32_t kIq4Tile = 1152;

__host__ __device__ constexpr uint64_t iq4_tiled_bytes(uint32_t n, uint32_t k) { return uint64_t(n / 16) * (k / 128) * kIq4Tile; }

// Item = one tile; thread s < 32 (row r = s % 16, half h = s / 16) writes its 4 qs runs, and the h = 0 threads the
// row's 4 d.
__device__ inline void op_relayout_iq4nl(const RelayoutArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t ntl = a.k / 128, tiles = (a.n / 16) * ntl;
  for (uint32_t i = wg; i < tiles; i += nwg) {
    const uint32_t g = i / ntl, ti = i % ntl, s = thread_id();
    if (s >= 32) continue;
    const uint32_t r = s % 16, h = s / 16;
    const uint8_t *b = a.src + uint64_t(g * 16 + r) * (a.k / 32) * 18 + uint64_t(ti) * 4 * 18;
    uint8_t *t = a.dst + uint64_t(i) * kIq4Tile;
    for (uint32_t p = 0; p < 2; ++p)
      for (uint32_t v = 0; v < 2; ++v)
        for (uint32_t j = 0; j < 8; ++j) t[512 * p + 16 * s + 8 * v + j] = b[18 * (2 * p + v) + 2 + 8 * h + j];
    if (h == 0) {
      for (uint32_t u = 0; u < 4; ++u) {
        t[1024 + 8 * r + 2 * u] = b[18 * u];
        t[1024 + 8 * r + 2 * u + 1] = b[18 * u + 1];
      }
    }
  }
}

// The tile holding tile ti of row group g, in an N x K matrix's tiles at w.
__host__ __device__ inline const uint8_t *iq4_tile(const uint8_t *w, uint32_t k, uint32_t g, uint32_t ti) {
  return w + (uint64_t(g) * (k / 128) + ti) * kIq4Tile;
}

// Unit ub (0..3) of row r of a tile, as q::dequant32<IQ4_NL> computes it: out[j] = d * float(table[..]), the same
// expression on the same bytes (the exact kernel and the tests).
__device__ inline void iq4_tile_dequant32(const uint8_t *tile, uint32_t r, uint32_t ub, float out[32]) {
  const float d = q::h2f(q::ld16(tile + 1024 + 8 * r + 2 * ub));
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t byte = tile[512 * (ub / 2) + 16 * (r + 16 * (j / 8)) + 8 * (ub % 2) + j % 8];
    out[j] = d * float(q::kIq4nlDev[byte & 0xf]);
    out[j + 16] = d * float(q::kIq4nlDev[byte >> 4]);
  }
}

}  // namespace qwen38
