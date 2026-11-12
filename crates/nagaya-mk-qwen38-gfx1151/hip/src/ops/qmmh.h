// SPDX-License-Identifier: MIT
// qmm's fast path (PLAN.md "Speed" S2): y[t, n] = sum_k W[n, k] * x[t, k] for the quantized types, on
// f16 WMMA (v_wmma_f32_16x16x16_f16) with f32 accumulation.
//
// Numerics (the op's definition; test_qmm's tolerance comes from it):
// - Weights stay exact. Each 32-value unit's integer codes (-128 <= code <= 127, exact in f16) are the
//   WMMA's B operand; the unit's scale (Q6_K: one per 16 values) multiplies the unit's f32 partial
//   afterwards, acc = fmaf(scale, partial, acc), units in K order.
// - Activations are rounded once to f16 (round to nearest even) when staged in LDS: relative error <= 2^-11,
//   except below 2^-14 where f16 is subnormal and the error is <= 2^-25 absolute. A value beyond f16's range
//   (or NaN) traps.
// - Code times f16 is exact in f32; the WMMA sums in f32 in the hardware's fixed order. Every output
//   is computed by one item, so the bits don't depend on the grid.
//
// Schedule (QmmhDense; the MoE experts use QmmhMoe, 8 waves along 128 rows x one block of 16 tokens, with the
// same code): a work item is 64 rows x 64 tokens. The workgroup's 8 waves are 4 along the rows
// (wave w % 4 owns rows 16 (w % 4) .. + 15 of the item, the WMMA's N dimension, B = W^T) by 2 along the tokens
// (wave w / 4 owns 2 of the item's 4 blocks of 16 tokens, the M dimension, A = x). The K dimension goes in
// chunks of 2 units (64 values): the workgroup stages the chunk's activations for the item's tokens in the
// LDS arena as f16 (64 x 72 halves, 9 KiB: the arena is the megakernel's LDS budget), then each wave walks
// the chunk's units. Per unit, lane l
// decodes half h = l / 16 (16 codes) of row l % 16; the two half-waves swap halves (shfl_xor 16), so each
// lane holds both 16-value B fragments of its row. WMMA fragment layout on gfx11 wave32: A lane l holds row
// l % 16 (both half-waves the same), B lane l holds column l % 16, and D lane l holds column l % 16 at rows
// 2i + l / 16, i = 0..7. So D's column is the lane's own weight row, and the scale is lane-local.
//
// Every thread of the workgroup must call qmmh_item with the same item (it synchronizes on LDS).
#pragma once
#include "qmm.h"
#include "small.h"
#include "tiles.h"

// MK_EXPERT_LOOKAHEAD: the IQ3_S expert item (qmmh_item_iq3s_tiles) runs one superblock ahead, issuing the next
// step's activation and tile loads before this step's staging and decode. The lookahead holds the next step's
// activations in registers (16 VGPRs), so the TU that compiles k_stage sets it to 0 (engine.hip) to keep its register
// budget; the other TUs (k_mega, the op tests) take the default, 1. Both give the same bits.
#ifndef MK_EXPERT_LOOKAHEAD
#define MK_EXPERT_LOOKAHEAD 1
#endif

namespace mk {

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

// A tile shape: WN waves along the rows (16 rows each) by 8 / WN along the tokens, TB blocks of 16 tokens.
template <uint32_t WN, uint32_t TB>
struct QmmhShape {
  static constexpr uint32_t kWavesN = WN;
  static constexpr uint32_t kWavesM = kMkBlock / kWave / WN;
  static constexpr uint32_t kRows = 16 * WN;
  static constexpr uint32_t kTokBlocks = TB;
  static constexpr uint32_t kToks = 16 * TB;
  static constexpr uint32_t kWaveBlocks = TB / kWavesM;  // token blocks per wave
  static_assert(kWavesN * kWavesM == kMkBlock / kWave && TB % kWavesM == 0, "waves tile the item evenly");
};
// Dense projections: 64 rows x 64 tokens (4 x 2 waves). A 51-token read is one token tile.
using QmmhDense = QmmhShape<4, 4>;
// MoE experts: 128 rows x 16 tokens (8 x 1 waves). An expert sees ~1-5 tokens per read, so wider token tiles
// would leave waves on empty blocks.
using QmmhMoe = QmmhShape<8, 1>;
constexpr uint32_t kQmmhRows = QmmhDense::kRows, kQmmhToks = QmmhDense::kToks;
constexpr uint32_t kQmmhChunkUnits = 2;                                // 64 values per K chunk
constexpr uint32_t kQmmhRowHalves = kQmmhChunkUnits * 32 + 8;          // f16 per staged token row (+8 pad)
static_assert(QmmhDense::kToks * kQmmhRowHalves * 2 <= kLdsArenaBytes, "the f16 tile lives in the LDS arena");
static_assert(QmmhMoe::kToks * kQmmhRowHalves * 2 <= kLdsArenaBytes, "the f16 tile lives in the LDS arena");

__host__ __device__ constexpr bool qmmh_type(uint32_t t) {
  return t == q::Q8_0 || t == q::Q8_0_TILES || t == q::Q6_K || t == q::IQ4_NL || t == q::IQ4_XS || t == q::IQ3_S ||
         t == q::IQ3_S_TILES || t == q::Q6_K_TILES || t == q::IQ4_NL_TILES;
}

// Cache policy of bulky weight loads, as a template parameter P. The engine uses P = -1: plain loads.
// P = 0 is buffer loads with no policy; 2 = slc (L2 streaming), 4 = dlc (GL1 streaming), 6 = both.
// A policy is a hint: values, and so every output bit, are unchanged.

// Weight bytes of one matrix (base = the matrix's first byte, wave-uniform). P < 0: plain loads. P >= 0: buffer
// loads with cache policy P, in 16-bit pairs (every multi-byte run the decoders read is 2-byte aligned).
template <int P>
struct WLd {
  const uint8_t *base;
  __amdgpu_buffer_rsrc_t r;
  __device__ explicit WLd(const uint8_t *b) : base(b) {
    if constexpr (P >= 0) r = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(b), 0, 0x7fffffff, 0x31004000);
  }
  __device__ __attribute__((always_inline)) uint8_t u8(const uint8_t *p) const {
    if constexpr (P < 0) return *p;
    else return __builtin_amdgcn_raw_buffer_load_b8(r, int(p - base), 0, P);
  }
  __device__ __attribute__((always_inline)) uint16_t u16(const uint8_t *p) const {
    if constexpr (P < 0) return q::ld16(p);
    else return __builtin_amdgcn_raw_buffer_load_b16(r, int(p - base), 0, P);
  }
  // 16 bytes from a 16-byte aligned p.
  __device__ __attribute__((always_inline)) u32x4 v16(const uint8_t *p) const {
    if constexpr (P < 0) return *reinterpret_cast<const u32x4 *>(p);
    else return __builtin_amdgcn_raw_buffer_load_b128(r, int(p - base), 0, P);
  }
  // 8 bytes from an 8-byte aligned p.
  __device__ __attribute__((always_inline)) u32x2 v8(const uint8_t *p) const {
    if constexpr (P < 0) return *reinterpret_cast<const u32x2 *>(p);
    else return __builtin_amdgcn_raw_buffer_load_b64(r, int(p - base), 0, P);
  }
  // n bytes (n even) from a 2-byte aligned p.
  template <int N>
  __device__ __attribute__((always_inline)) void run(const uint8_t *p, uint8_t out[N]) const {
    static_assert(N % 2 == 0, "runs are 16-bit pairs");
    if constexpr (P < 0) {
#pragma unroll
      for (int j = 0; j < N; ++j) out[j] = p[j];
    } else {
#pragma unroll
      for (int j = 0; j < N / 2; ++j) {
        const uint16_t v = u16(p + 2 * j);
        out[2 * j] = uint8_t(v);
        out[2 * j + 1] = uint8_t(v >> 8);
      }
    }
  }
};

namespace q {

// Integer codes and scales of unit u of a row: value j = scale(j / 16) * code[j], the same value
// (bits) as Unit<T>'s get8. half(h, out) writes codes 16h .. 16h + 15 as f16. Weight bytes come through L.
template <uint32_t T, int P>
struct Codes;

template <int P>
struct Codes<Q8_0, P> {
  WLd<P> L;
  const uint8_t *b;
  float d;
  __device__ Codes(const WLd<P> &l, const uint8_t *row, uint32_t u) : L(l), b(row + size_t(u) * 34), d(h2f(L.u16(b))) {}
  __device__ float scale(uint32_t) const { return d; }
  __device__ __attribute__((always_inline)) void half(uint32_t h, _Float16 out[16]) const {
    uint8_t c[16];
    L.template run<16>(b + 2 + 16 * h, c);
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = _Float16(int8_t(c[j]));
  }
};

// Q8_0 tiles (tiles.h): `grp` is the row group's first tile and r the row within it. Half h of unit u is one aligned
// 16-byte load; the same codes and scale as Codes<Q8_0> on the GGUF row, so the same bits.
template <int P>
struct Codes<Q8_0_TILES, P> {
  WLd<P> L;
  const uint8_t *c;
  float d;
  __device__ Codes(const WLd<P> &l, const uint8_t *grp, uint32_t u, uint32_t r) : L(l) {
    const uint8_t *tile = grp + uint64_t(u / 2) * kQ8Tile;
    const uint32_t slot = r + 16 * (u % 2);
    c = tile + 32 * slot;
    d = h2f(L.u16(tile + 1024 + 2 * slot));
  }
  __device__ float scale(uint32_t) const { return d; }
  __device__ __attribute__((always_inline)) void half(uint32_t h, _Float16 out[16]) const {
    const u32x4 v = L.v16(c + 16 * h);
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = _Float16(int8_t(v[j / 4] >> (8 * (j % 4))));
  }
};

template <int P>
struct Codes<IQ4_NL, P> {
  WLd<P> L;
  const uint8_t *qs;
  float d;
  __device__ Codes(const WLd<P> &l, const uint8_t *row, uint32_t u)
      : L(l), qs(row + size_t(u) * 18 + 2), d(h2f(L.u16(row + size_t(u) * 18))) {}
  __device__ float scale(uint32_t) const { return d; }
  __device__ __attribute__((always_inline)) void half(uint32_t h, _Float16 out[16]) const {
    const uint32_t sh = 4 * h;  // values 0..15: low nibbles of qs[0..15]; 16..31: high nibbles
    uint8_t c[16];
    L.template run<16>(qs, c);
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = _Float16(kIq4nlDev[(c[j] >> sh) & 0xf]);
  }
};

template <int P>
struct Codes<IQ4_XS, P> {
  WLd<P> L;
  const uint8_t *qs;
  float dl;
  __device__ Codes(const WLd<P> &l, const uint8_t *row, uint32_t u) : L(l) {
    const uint8_t *b = row + size_t(u / 8) * 136;
    const uint32_t ib = u % 8;
    const float d = h2f(L.u16(b));
    const uint16_t scales_h = L.u16(b + 2);
    const uint8_t *scales_l = b + 4;
    qs = b + 8 + ib * 16;
    const int ls = ((L.u8(scales_l + ib / 2) >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
    dl = d * float(ls - 32);
  }
  __device__ float scale(uint32_t) const { return dl; }
  __device__ __attribute__((always_inline)) void half(uint32_t h, _Float16 out[16]) const {
    const uint32_t sh = 4 * h;
    uint8_t c[16];
    L.template run<16>(qs, c);
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = _Float16(kIq4nlDev[(c[j] >> sh) & 0xf]);
  }
};

template <int P>
struct Codes<IQ3_S, P> {
  WLd<P> L;
  const uint8_t *qs, *signs;
  uint8_t qh;
  float db;
  __device__ Codes(const WLd<P> &l, const uint8_t *row, uint32_t u) : L(l) {
    const uint8_t *b = row + size_t(u / 8) * 110;
    const uint32_t ib = u % 8;
    const float d = h2f(L.u16(b));
    qs = b + 2 + ib * 8;
    qh = L.u8(b + 2 + 64 + ib);
    signs = b + 2 + 64 + 8 + ib * 4;
    const uint8_t sc = L.u8(b + 2 + 64 + 8 + 32 + ib / 2);
    db = d * float(1 + 2 * ((ib % 2) ? (sc >> 4) : (sc & 0xf)));
  }
  __device__ float scale(uint32_t) const { return db; }
  // Half h is dequant32's l = 2h and 2h + 1 (8 values each): grid indices qs[4h .. 4h + 3], signs[2h .. 2h + 1].
  __device__ __attribute__((always_inline)) void half(uint32_t h, _Float16 out[16]) const {
    uint8_t g[4], sg2[2];
    L.template run<4>(qs + 4 * h, g);
    L.template run<2>(signs + 2 * h, sg2);
#pragma unroll
    for (uint32_t ll = 0; ll < 2; ++ll) {
      const uint32_t l = 2 * h + ll;
      const uint32_t g1 = kIq3sGridDev[g[2 * ll + 0] | ((uint32_t(qh) << (8 - 2 * l)) & 256)];
      const uint32_t g2 = kIq3sGridDev[g[2 * ll + 1] | ((uint32_t(qh) << (7 - 2 * l)) & 256)];
      const uint8_t sg = sg2[ll];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int v1 = int((g1 >> (8 * j)) & 0xff), v2 = int((g2 >> (8 * j)) & 0xff);
        out[8 * ll + j] = _Float16((sg & (1u << j)) ? -v1 : v1);
        out[8 * ll + 4 + j] = _Float16((sg & (1u << (j + 4))) ? -v2 : v2);
      }
    }
  }
};

template <int P>
struct Codes<Q6_K, P> {
  WLd<P> L;
  const uint8_t *ql, *qh, *sc;
  uint32_t q;
  float d;
  __device__ Codes(const WLd<P> &l, const uint8_t *row, uint32_t u) : L(l) {
    const uint8_t *b = row + size_t(u / 8) * 210;
    const uint32_t s = u % 8, n = s / 4;
    q = s % 4;
    ql = b + 64 * n + (q % 2) * 32;
    qh = b + 128 + 32 * n;
    sc = b + 192 + 8 * n + 2 * q;
    d = h2f(L.u16(b + 208));
  }
  __device__ float scale(uint32_t h) const { return d * float(int8_t(L.u8(sc + h))); }  // values 16h..16h+15
  __device__ __attribute__((always_inline)) void half(uint32_t h, _Float16 out[16]) const {
    const uint32_t lo_sh = q < 2 ? 0 : 4, hi_sh = 2 * q;
    uint8_t cl[16], ch[16];
    L.template run<16>(ql + 16 * h, cl);
    L.template run<16>(qh + 16 * h, ch);
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = _Float16(int(((cl[j] >> lo_sh) & 0xF) | (((ch[j] >> hi_sh) & 3) << 4)) - 32);
  }
};

// Q6_K tiles (tiles.h): `grp` is the row group's first tile and r the row within it. Half h of unit u is one aligned
// 16-byte load of ql and one of qh (plus the row's scale bytes and d); the same codes and scales, and the same
// expressions, as Codes<Q6_K> on the GGUF row, so the same bits.
template <int P>
struct Codes<Q6_K_TILES, P> {
  WLd<P> L;
  const uint8_t *ql, *qh, *sc;
  uint32_t q;
  float d;
  __device__ Codes(const WLd<P> &l, const uint8_t *grp, uint32_t u, uint32_t r) : L(l) {
    const uint8_t *tile = grp + uint64_t(u / 8) * kQ6Tile;
    const uint32_t s = u % 8, n = s / 4;
    q = s % 4;
    ql = tile + 512 * (2 * n + q % 2) + 32 * r;
    qh = tile + 2048 + 512 * n + 32 * r;
    sc = tile + 3072 + 16 * r + 8 * n + 2 * q;
    d = h2f(L.u16(tile + 3328 + 2 * r));
  }
  __device__ float scale(uint32_t h) const { return d * float(int8_t(L.u8(sc + h))); }  // values 16h..16h+15
  __device__ __attribute__((always_inline)) void half(uint32_t h, _Float16 out[16]) const {
    const uint32_t lo_sh = q < 2 ? 0 : 4, hi_sh = 2 * q;
    const u32x4 vl = L.v16(ql + 16 * h), vh = L.v16(qh + 16 * h);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const uint32_t cl = (vl[j / 4] >> (8 * (j % 4))) & 0xff, ch = (vh[j / 4] >> (8 * (j % 4))) & 0xff;
      out[j] = _Float16(int(((cl >> lo_sh) & 0xF) | (((ch >> hi_sh) & 3) << 4)) - 32);
    }
  }
};

}  // namespace q

constexpr uint32_t kMaxKsplit = 16;  // qmmh_ksplit's cap; op_qmm_ksum and qmmh_x4 hold one value per slice

// Slices s < ks of partials p ([ks][t][k]) at (tok, col .. col + 3), added in slice order: op_qmm_ksum's fold. All
// slices are loaded before the first add (clamped, unconditional: one round trip).
__device__ __attribute__((always_inline)) inline float4 qmmh_psum4(const float *p, uint32_t ks, uint64_t tk, uint64_t at) {
  float4 v[kMaxKsplit];
#pragma unroll
  for (uint32_t s = 0; s < kMaxKsplit; ++s) v[s] = *reinterpret_cast<const float4 *>(p + uint64_t(min(s, ks - 1)) * tk + at);
  float4 r = v[0];
#pragma unroll
  for (uint32_t s = 1; s < kMaxKsplit; ++s)
    if (s < ks) {
      r.x += v[s].x;
      r.y += v[s].y;
      r.z += v[s].z;
      r.w += v[s].w;
    }
  return r;
}

// Values col .. col + 3 of x's row for token tok, as the f16 path stages them (QmmArgs::xin). The fused forms replace
// kStQmmKsum (+ kStSiluScale | kStSwiglu) stages at decode (engine.hip fuse_prologues): same expressions, same bits.
__device__ __attribute__((always_inline)) inline float4 qmmh_x4(const QmmArgs &a, uint32_t tok, uint32_t col) {
  if (a.xin == kXinPlain) return *reinterpret_cast<const float4 *>(a.x + uint64_t(a.x_row ? a.x_row[tok] : tok) * a.ldx + col);
  if (a.xks < 2 || a.xks > kMaxKsplit || a.x_row) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  const uint64_t tk = uint64_t(a.t) * a.k, at = uint64_t(tok) * a.k + col;
  const float4 g = qmmh_psum4(a.xp, a.xks, tk, at);
  if (a.xin == kXinSiluSum)
    return make_float4(siluf_(g.x * a.xscale), siluf_(g.y * a.xscale), siluf_(g.z * a.xscale), siluf_(g.w * a.xscale));
  if (a.xin != kXinSwigluSum) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  const float4 u = qmmh_psum4(a.xp2, a.xks, tk, at);
  return make_float4((g.x * sigmoidf_(g.x)) * u.x, (g.y * sigmoidf_(g.y)) * u.y, (g.z * sigmoidf_(g.z)) * u.z,
                     (g.w * sigmoidf_(g.w)) * u.w);
}

// Split-K (PLAN.md Speed S5). A matrix with few 64-row groups (hc down: 5; the shared expert: 10; attention k/v: 8)
// left most of the megakernel's 80 workgroups idle while a few walked all of K. With ksplit > 1, item i covers K
// slice i % ksplit: units [s * us, min((s + 1) * us, units)), us = the units over ksplit rounded up to whole chunks,
// accumulated in K order as always, and the item writes that partial to part[s][t][n]; op_qmm_ksum then adds the
// slices in slice order. The engine's ksplit is a function of the matrix shape alone (never T, never the grid),
// so a token's bits don't depend on the pass it's in.
__host__ __device__ constexpr uint32_t qmmh_slice_units(uint32_t units, uint32_t ksplit) {
  return ((units + ksplit - 1) / ksplit + kQmmhChunkUnits - 1) / kQmmhChunkUnits * kQmmhChunkUnits;
}
// A usable split: every slice non-empty.
__host__ __device__ constexpr bool qmmh_ksplit_ok(uint32_t k, uint32_t ksplit) {
  return ksplit >= 1 && k % 32 == 0 && (ksplit - 1) * qmmh_slice_units(k / 32, ksplit) < k / 32;
}
// The engine's policy: aim for ~80 items at one token tile when a matrix has fewer than 20 row groups, with at least
// 8 units (256 values) per slice and at most 16 slices.
__host__ __device__ constexpr uint32_t qmmh_ksplit(uint32_t n, uint32_t k) {
  const uint32_t groups = (n + kQmmhRows - 1) / kQmmhRows, units = k / 32;
  if (groups >= 20 || units < 16) return 1;
  uint32_t ks = (80 + groups - 1) / groups;
  if (ks > 16) ks = 16;
  if (ks > units / 8) ks = units / 8;
  while (ks > 1 && !qmmh_ksplit_ok(k, ks)) --ks;
  return ks;
}

// hc up + hc_mix (docs/narrow.md N1, QmmArgs::yout = kYoutHcMix). The item's row map: wave wn is hyper-connection stream
// wn (kHc streams, one per wave along the rows), row group rg covers channels rg * 16 .. + 15 of every stream, so the
// item holds up[t, s, c] for all four streams of those channels and can run hc_mix itself. Each wave still owns one whole
// 16-row tile (n / 16 = wn * (mix_e / 16) + rg), so it decodes exactly the rows the plain map would give some other item.
template <class Sh>
__device__ __attribute__((always_inline)) inline uint32_t qmmh_row(const QmmArgs &a, uint32_t rg, uint32_t wn, uint32_t r) {
  return a.yout == kYoutHcMix ? wn * a.mix_e + rg * 16 + r : rg * Sh::kRows + wn * 16 + r;
}

// A fused-epilogue request the item can't honour crashes (never a plain store of something else).
template <uint32_t T, class Sh>
__device__ __attribute__((always_inline)) inline void qmmh_check_yout(const QmmArgs &a, uint32_t ks) {
  if (a.yout == kYoutPlain) return;
  constexpr bool ok_type = T == q::Q8_0 || T == q::Q8_0_TILES;
  if constexpr (!ok_type || Sh::kWavesN != 4) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  if (a.yout != kYoutHcMix || ks > 1 || a.y_row || a.mix_e == 0 || a.mix_e % 16 || a.n != 4 * a.mix_e || a.mix_xn == nullptr ||
      a.mix_out == nullptr || (a.mix_z != nullptr) != (a.mix_inject != nullptr))
    MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
}

// The epilogue's LDS: slot g (one token block's four streams) = up[wn][tok][r], 4 * 16 * 16 floats = 4 KiB. The dense
// general loop has two groups of waves along the tokens, so two slots (8 KiB <= kLdsArenaBytes).
constexpr uint32_t kHcMixSlotFloats = 4 * 16 * 16;
static_assert(2 * kHcMixSlotFloats * 4 <= kLdsArenaBytes, "the hc_mix epilogue's two slots live in the LDS arena");

// This lane's accumulators (column r, token rows 2i + h of a block) into slot memory `slot`, stream wn.
__device__ __attribute__((always_inline)) inline void qmmh_hcmix_put(float *slot, const v8f &acc, uint32_t wn, uint32_t h,
                                                                     uint32_t r) {
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) slot[(wn * 16 + 2 * i + h) * 16 + r] = acc[i];
}
// The thread's xn operands for (token tok of the item, channel rg * 16 + r), loaded ahead of the barrier.
__device__ __attribute__((always_inline)) inline void qmmh_hcmix_load(const QmmArgs &a, uint32_t rg, uint32_t tok,
                                                                      uint32_t r, float xn[4]) {
  const uint64_t base = uint64_t(tok) * 4 * a.mix_e + rg * 16 + r;
#pragma unroll
  for (uint32_t s = 0; s < 4; ++s) xn[s] = a.mix_xn[base + s * a.mix_e];
}
// mixed[tok, rg * 16 + r] from slot memory (the four streams in stream order, small.h's own expression), and the
// inject of z[tok, r] from the item with rg == 0.
__device__ __attribute__((always_inline)) inline void qmmh_hcmix_store(const QmmArgs &a, const float *slot, const float xn[4],
                                                                       uint32_t rg, uint32_t tok, uint32_t tl, uint32_t r) {
  float up[4];
#pragma unroll
  for (uint32_t s = 0; s < 4; ++s) up[s] = slot[(s * 16 + tl) * 16 + r];
  a.mix_out[uint64_t(tok) * a.mix_e + rg * 16 + r] = hc_mix_elem(up, xn);
  if (a.mix_z != nullptr && rg == 0 && r < 4) a.mix_inject[tok * 4 + r] = hc_inject_elem(a.mix_z[tok * 4 + r]);
}

// Decode (PLAN.md Speed, "Decode: the dense f16 path"): Q8_0 tiles on the dense shape when an item has one block of
// <= 16 tokens. The general loop's 2-unit chunk is one serialized trip per 4 KB of weights (loads issued after the
// barrier, waited on before the WMMAs): latency-bound at T = 1. Here a chunk is 8 units (the 16 staged tokens x 256
// values fit the arena), and each lane issues its 8 half-unit loads (one 16-byte load each) before the barriers that
// stage the chunk's activations, so the fetch overlaps the staging. Only the schedule changes: the same codes, the same
// WMMAs, units folded in K order with acc = fmaf(scale, partial, acc), so the same bits as qmmh_item.
constexpr uint32_t kQ8tChunkUnits = 8;
constexpr uint32_t kQ8tRowHalves = kQ8tChunkUnits * 32 + 8;
static_assert(16 * kQ8tRowHalves * 2 <= kLdsArenaBytes, "16 tokens x 8 units of f16 live in the LDS arena");

template <int P>
__device__ __attribute__((always_inline)) inline void qmmh_item_q8t_decode(const QmmArgs &a, uint32_t rg, uint32_t t0,
                                                                           uint32_t nt, uint32_t u0, uint32_t u1,
                                                                           uint32_t slice, uint32_t ks) {
  using Sh = QmmhDense;
  const uint32_t wave = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t wn = wave % Sh::kWavesN, wm = wave / Sh::kWavesN;
  const uint32_t r = lane % 16, h = lane / 16;
  const uint32_t n = qmmh_row<Sh>(a, rg, wn, r);
  _Float16 *xs = reinterpret_cast<_Float16 *>(lds_arena());
  const uint8_t *grp = q8_tile(a.w, a.k, (n < a.n ? n : 0) / 16, 0);
  const WLd<P> wl(a.w);
  v8f acc = {};
  for (uint32_t cu0 = u0; cu0 < u1; cu0 += kQ8tChunkUnits) {
    const uint32_t cn = min(kQ8tChunkUnits, u1 - cu0);  // even: slices and K are whole pairs
    // This lane's half of each unit of the chunk (slot r + 16 (u % 2) of pair u / 2), and the unit's scale. Units past
    // the chunk's end load the last unit again, unused: guarded loads became register copies and a wait per load.
    u32x4 wv[kQ8tChunkUnits];
    uint16_t dv[kQ8tChunkUnits];
    if (wm == 0) {
#pragma unroll
      for (uint32_t uu = 0; uu < kQ8tChunkUnits; ++uu) {
        const uint32_t u = min(cu0 + uu, u1 - 1);
        const uint8_t *tile = grp + uint64_t(u / 2) * kQ8Tile;
        const uint32_t slot = r + 16 * (u % 2);
        wv[uu] = wl.v16(tile + 32 * slot + 16 * h);
        dv[uu] = wl.u16(tile + 1024 + 2 * slot);
      }
    }
    lds_sync();  // every wave is done reading the previous chunk's (or item's) activations
    for (uint32_t i = mk_tid(); i < 16 * kQ8tChunkUnits * 8; i += kMkBlock) {
      const uint32_t t = i / (kQ8tChunkUnits * 8), q4 = i % (kQ8tChunkUnits * 8);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < nt && q4 / 8 < cn) {
        const uint32_t tok = t0 + t;
        v = qmmh_x4(a, tok, cu0 * 32 + q4 * 4);
        if (!(fabsf(v.x) <= 65504.f && fabsf(v.y) <= 65504.f && fabsf(v.z) <= 65504.f && fabsf(v.w) <= 65504.f))
          MK_TRAP(kTrapActRange, kTrapFileQmmh);  // beyond f16 (or NaN): crash, never inf
      }
      _Float16 *dst = xs + t * kQ8tRowHalves + q4 * 4;
      dst[0] = _Float16(v.x);
      dst[1] = _Float16(v.y);
      dst[2] = _Float16(v.z);
      dst[3] = _Float16(v.w);
    }
    lds_sync();
    if (wm != 0) continue;  // one token block: the second row of waves only stages
#pragma unroll
    for (uint32_t uu = 0; uu < kQ8tChunkUnits; ++uu) {
      if (uu < cn) {
        union {
          v16h v;
          _Float16 e[16];
          int32_t w[8];
        } mine, other;
#pragma unroll
        for (int j = 0; j < 16; ++j) mine.e[j] = _Float16(int8_t(wv[uu][j / 4] >> (8 * (j % 4))));
#pragma unroll
        for (int j = 0; j < 8; ++j) other.w[j] = __shfl_xor(mine.w[j], 16, kWave);
        const v16h b_lo = h == 0 ? mine.v : other.v, b_hi = h == 0 ? other.v : mine.v;
        const float s = q::h2f(dv[uu]);
        const _Float16 *ap = xs + r * kQ8tRowHalves + uu * 32;
        const v16h a_lo = *reinterpret_cast<const v16h *>(ap);
        const v16h a_hi = *reinterpret_cast<const v16h *>(ap + 16);
        const v8f zero = {};
        const v8f p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b_hi,
                                                                 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_lo, zero));
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(s, p[i], acc[i]);
      }
      // One unit's activations and codes live at a time: unscheduled, the compiler hoisted every unit's LDS loads
      // (240 VGPRs, past k_mega's 192).
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if (a.yout == kYoutHcMix) {  // one token block: the epilogue (qmmh_check_yout vetted the request)
    float *slot = reinterpret_cast<float *>(lds_arena());
    const uint32_t tl = mk_tid() / 16, tr = mk_tid() % 16;
    float xnv[4] = {};
    if (tl < nt) qmmh_hcmix_load(a, rg, t0 + tl, tr, xnv);
    lds_sync();  // every wave is done with the x tile, which the slot overlays
    if (wm == 0) qmmh_hcmix_put(slot, acc, wn, h, r);
    lds_sync();
    if (tl < nt) qmmh_hcmix_store(a, slot, xnv, rg, t0 + tl, tl, tr);
    lds_sync();
    return;
  }
  if (wm == 0 && n < a.n) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t t = 2 * i + h;
      if (t < nt) {
        const uint32_t tok = t0 + t;
        if (ks > 1) a.part[(uint64_t(slice) * a.t + tok) * a.n + n] = acc[i];
        else a.y[uint64_t(a.y_row ? a.y_row[tok] : tok) * a.ldy + n] = acc[i];
      }
    }
  }
}

// Decode on Q6_K tiles (the output head): the same schedule as qmmh_item_q8t_decode, one superblock (= one tile = 8
// units) per chunk. Each lane issues its 8 loads of the tile (4 ql runs, 2 qh runs, the row's 16 scale bytes, d) before
// the barriers that stage the chunk's activations. The slice [u0, u1) must be whole superblocks (the caller checks).
// The same codes and WMMAs as qmmh_item's general loop, each unit's two 16-value partials folded as
// acc = fmaf(s_hi, p_hi, fmaf(s_lo, p_lo, acc)) in K order, so the same bits.
template <int P>
__device__ __attribute__((always_inline)) inline void qmmh_item_q6t_decode(const QmmArgs &a, uint32_t rg, uint32_t t0,
                                                                           uint32_t nt, uint32_t u0, uint32_t u1,
                                                                           uint32_t slice, uint32_t ks) {
  using Sh = QmmhDense;
  const uint32_t wave = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t wn = wave % Sh::kWavesN, wm = wave / Sh::kWavesN;
  const uint32_t r = lane % 16, h = lane / 16;
  const uint32_t n = rg * Sh::kRows + wn * 16 + r;
  if (a.yout != kYoutPlain) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  _Float16 *xs = reinterpret_cast<_Float16 *>(lds_arena());
  const uint8_t *grp = q6_tile(a.w, a.k, (n < a.n ? n : 0) / 16, 0);
  const WLd<P> wl(a.w);
  v8f acc = {};
  for (uint32_t cu0 = u0; cu0 < u1; cu0 += 8) {
    u32x4 vql[4], vqh[2], vsc;
    uint16_t dv;
    if (wm == 0) {
      const uint8_t *tile = grp + uint64_t(cu0 / 8) * kQ6Tile;
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) vql[c] = wl.v16(tile + 512 * c + 32 * r + 16 * h);
#pragma unroll
      for (uint32_t c = 0; c < 2; ++c) vqh[c] = wl.v16(tile + 2048 + 512 * c + 32 * r + 16 * h);
      vsc = wl.v16(tile + 3072 + 16 * r);
      dv = wl.u16(tile + 3328 + 2 * r);
    }
    lds_sync();  // every wave is done reading the previous chunk's (or item's) activations
    for (uint32_t i = mk_tid(); i < 16 * kQ8tChunkUnits * 8; i += kMkBlock) {
      const uint32_t t = i / (kQ8tChunkUnits * 8), q4 = i % (kQ8tChunkUnits * 8);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < nt) {
        const uint32_t tok = t0 + t;
        v = qmmh_x4(a, tok, cu0 * 32 + q4 * 4);
        if (!(fabsf(v.x) <= 65504.f && fabsf(v.y) <= 65504.f && fabsf(v.z) <= 65504.f && fabsf(v.w) <= 65504.f))
          MK_TRAP(kTrapActRange, kTrapFileQmmh);  // beyond f16 (or NaN): crash, never inf
      }
      _Float16 *dst = xs + t * kQ8tRowHalves + q4 * 4;
      dst[0] = _Float16(v.x);
      dst[1] = _Float16(v.y);
      dst[2] = _Float16(v.z);
      dst[3] = _Float16(v.w);
    }
    lds_sync();
    if (wm != 0) continue;  // one token block: the second row of waves only stages
    const float d = q::h2f(dv);
#pragma unroll
    for (uint32_t uu = 0; uu < 8; ++uu) {
      const uint32_t nn = uu / 4, qq = uu % 4, lo_sh = qq < 2 ? 0 : 4, hi_sh = 2 * qq;
      const u32x4 vl = vql[2 * nn + qq % 2], vh = vqh[nn];
      union {
        v16h v;
        _Float16 e[16];
        int32_t w[8];
      } mine, other;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const uint32_t cl = (vl[j / 4] >> (8 * (j % 4))) & 0xff, ch = (vh[j / 4] >> (8 * (j % 4))) & 0xff;
        mine.e[j] = _Float16(int(((cl >> lo_sh) & 0xF) | (((ch >> hi_sh) & 3) << 4)) - 32);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) other.w[j] = __shfl_xor(mine.w[j], 16, kWave);
      const v16h b_lo = h == 0 ? mine.v : other.v, b_hi = h == 0 ? other.v : mine.v;
      const uint32_t si = 8 * nn + 2 * qq;  // the unit's two scales: raw scales[si], scales[si + 1]
      const float s_lo = d * float(int8_t(vsc[si / 4] >> (8 * (si % 4)))),
                  s_hi = d * float(int8_t(vsc[(si + 1) / 4] >> (8 * ((si + 1) % 4))));
      const _Float16 *ap = xs + r * kQ8tRowHalves + uu * 32;
      const v16h a_lo = *reinterpret_cast<const v16h *>(ap);
      const v16h a_hi = *reinterpret_cast<const v16h *>(ap + 16);
      const v8f zero = {};
      const v8f p_lo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_lo, zero);
      const v8f p_hi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b_hi, zero);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = fmaf(s_hi, p_hi[i], fmaf(s_lo, p_lo[i], acc[i]));
      // One unit's activations and codes live at a time (as in the Q8_0 decode path).
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if (wm == 0 && n < a.n) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t t = 2 * i + h;
      if (t < nt) {
        const uint32_t tok = t0 + t;
        if (ks > 1) a.part[(uint64_t(slice) * a.t + tok) * a.n + n] = acc[i];
        else a.y[uint64_t(a.y_row ? a.y_row[tok] : tok) * a.ldy + n] = acc[i];
      }
    }
  }
}

template <uint32_t T, class Sh = QmmhDense, int P = -1>
__device__ __attribute__((always_inline)) inline void qmmh_item(const QmmArgs &a, uint32_t item) {
  constexpr uint32_t kQmmhWavesN = Sh::kWavesN, kQmmhToks = Sh::kToks;
  constexpr uint32_t kQmmhWaveBlocks = Sh::kWaveBlocks, kQmmhWavesM = Sh::kWavesM;
  const uint32_t ks = a.ksplit > 1 ? a.ksplit : 1;
  const uint32_t slice = item % ks;
  item /= ks;
  const uint32_t n_tt = (a.t + kQmmhToks - 1) / kQmmhToks;
  const uint32_t rg = item / n_tt, tt = item % n_tt;
  const uint32_t wave = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t wn = wave % kQmmhWavesN, wm = wave / kQmmhWavesN;
  const uint32_t r = lane % 16, h = lane / 16;
  const uint32_t n = qmmh_row<Sh>(a, rg, wn, r);
  const uint32_t t0 = tt * kQmmhToks;
  const uint32_t nt = min(kQmmhToks, a.t - t0);
  const uint32_t ntb = (nt + 15) / 16;
  const uint32_t units = a.k / 32;
  const uint32_t us = ks > 1 ? qmmh_slice_units(units, ks) : units;
  const uint32_t u0 = slice * us, u1 = min(u0 + us, units);
  if ((reinterpret_cast<uintptr_t>(a.x) & 15) || (a.ldx & 3)) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  if (ks > 1 && (u0 >= u1 || a.y_row)) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);  // an empty slice; partials are token-indexed only
  qmmh_check_yout<T, Sh>(a, ks);
  _Float16 *xs = reinterpret_cast<_Float16 *>(lds_arena());
  // Rows past N (the last item of a matrix) decode row 0 and are never written. Tiles: the row group's first tile.
  constexpr bool kQ8Tiles = T == q::Q8_0_TILES, kQ6Tiles = T == q::Q6_K_TILES, kTiles = kQ8Tiles || kQ6Tiles;
  if (kQ8Tiles && (a.n % 16 || a.k % 64 || (reinterpret_cast<uintptr_t>(a.w) & 15))) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  if (kQ6Tiles && (a.n % 16 || a.k % 256 || (reinterpret_cast<uintptr_t>(a.w) & 15))) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  if constexpr (kQ8Tiles && __is_same(Sh, QmmhDense)) {
    if (ntb == 1) {
      qmmh_item_q8t_decode<P>(a, rg, t0, nt, u0, u1, slice, ks);
      return;
    }
  }
  if constexpr (kQ6Tiles && __is_same(Sh, QmmhDense)) {
    // The decode path's chunk is a superblock: a K slice that cuts one (split-K slices are whole 2-unit chunks) takes
    // the general loop, which decodes any unit.
    if (ntb == 1 && u0 % 8 == 0 && u1 % 8 == 0) {
      qmmh_item_q6t_decode<P>(a, rg, t0, nt, u0, u1, slice, ks);
      return;
    }
  }
  const uint8_t *wrow = kQ8Tiles   ? q8_tile(a.w, a.k, (n < a.n ? n : 0) / 16, 0)
                        : kQ6Tiles ? q6_tile(a.w, a.k, (n < a.n ? n : 0) / 16, 0)
                                   : a.w + uint64_t(n < a.n ? n : 0) * a.ldw;
  const WLd<P> wl(a.w);
  v8f acc[kQmmhWaveBlocks] = {};
  for (uint32_t cu0 = u0; cu0 < u1; cu0 += kQmmhChunkUnits) {
    const uint32_t cn = min(kQmmhChunkUnits, u1 - cu0);
    // Every wave is done reading the tile: the previous chunk's, or the previous item's last.
    lds_sync();
    // Stage: float4 i of the tile is token i / (8 cu), values 4 (i % (8 cu)) .. + 3 of the chunk; tokens
    // past nt (up to the last used block's end) and units past cn are zero. Blocks past ntb are never read.
    for (uint32_t i = mk_tid(); i < ntb * 16 * kQmmhChunkUnits * 8; i += kMkBlock) {
      const uint32_t t = i / (kQmmhChunkUnits * 8), q4 = i % (kQmmhChunkUnits * 8);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < nt && q4 / 8 < cn) {
        const uint32_t tok = t0 + t;
        v = qmmh_x4(a, tok, cu0 * 32 + q4 * 4);
        if (!(fabsf(v.x) <= 65504.f && fabsf(v.y) <= 65504.f && fabsf(v.z) <= 65504.f && fabsf(v.w) <= 65504.f))
          MK_TRAP(kTrapActRange, kTrapFileQmmh);  // beyond f16 (or NaN): crash, never inf
      }
      _Float16 *dst = xs + t * kQmmhRowHalves + q4 * 4;
      dst[0] = _Float16(v.x);
      dst[1] = _Float16(v.y);
      dst[2] = _Float16(v.z);
      dst[3] = _Float16(v.w);
    }
    lds_sync();
    // A wave with no token block in this item (T <= 32: the second row of waves) has nothing to multiply, so it
    // decodes nothing. Its outputs are untouched either way: this only removes a discarded fetch and decode. (A loop
    // bound, not a `continue`: the branch cost k_mega 8 B/lane of scratch at its 192-VGPR ceiling.)
    const uint32_t wcn = wm * kQmmhWaveBlocks < ntb ? cn : 0;
    for (uint32_t uu = 0; uu < wcn; ++uu) {
      const q::Codes<T, P> dec = [&] {
        if constexpr (kTiles) return q::Codes<T, P>(wl, wrow, cu0 + uu, r);
        else return q::Codes<T, P>(wl, wrow, cu0 + uu);
      }();
      union {
        v16h v;
        _Float16 e[16];
        int32_t w[8];
      } mine, other;
      dec.half(h, mine.e);
#pragma unroll
      for (int j = 0; j < 8; ++j) other.w[j] = __shfl_xor(mine.w[j], 16, kWave);
      const v16h b_lo = h == 0 ? mine.v : other.v;  // values 0..15 of row n's unit
      const v16h b_hi = h == 0 ? other.v : mine.v;  // values 16..31
      const float s_lo = dec.scale(0), s_hi = dec.scale(1);
#pragma unroll
      for (uint32_t b = 0; b < kQmmhWaveBlocks; ++b) {
        const uint32_t tb = wm * kQmmhWaveBlocks + b;
        if (tb < ntb) {
          const _Float16 *ap = xs + (tb * 16 + r) * kQmmhRowHalves + uu * 32;
          const v16h a_lo = *reinterpret_cast<const v16h *>(ap);
          const v16h a_hi = *reinterpret_cast<const v16h *>(ap + 16);
          const v8f zero = {};
          if constexpr (T == q::Q6_K || T == q::Q6_K_TILES) {  // a scale per 16 values: two partials
            const v8f p_lo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_lo, zero);
            const v8f p_hi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b_hi, zero);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[b][i] = fmaf(s_hi, p_hi[i], fmaf(s_lo, p_lo[i], acc[b][i]));
          } else {
            const v8f p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_hi, b_hi, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_lo, zero));
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[b][i] = fmaf(s_lo, p[i], acc[b][i]);
          }
        }
      }
    }
  }
  if (a.yout == kYoutHcMix) {
    // Per round b the two groups of waves (wm) hold token blocks wm * kQmmhWaveBlocks + b: each puts its streams' accumulators in
    // its slot, then all threads take the blocks in turn (thread = token tl, channel tr of the block).
    float *lds = reinterpret_cast<float *>(lds_arena());
    const uint32_t tl = mk_tid() / 16, tr = mk_tid() % 16;
#pragma unroll
    for (uint32_t b = 0; b < kQmmhWaveBlocks; ++b) {
      float xnv[kQmmhWavesM][4] = {};
#pragma unroll
      for (uint32_t g = 0; g < kQmmhWavesM; ++g) {
        const uint32_t tok = (g * kQmmhWaveBlocks + b) * 16 + tl;
        if (tok < nt) qmmh_hcmix_load(a, rg, t0 + tok, tr, xnv[g]);
      }
      lds_sync();  // every wave is done with the x tile (round 0) or the previous round's slots
      if (wm * kQmmhWaveBlocks + b < ntb) qmmh_hcmix_put(lds + wm * kHcMixSlotFloats, acc[b], wn, h, r);
      lds_sync();
#pragma unroll
      for (uint32_t g = 0; g < kQmmhWavesM; ++g) {
        const uint32_t tok = (g * kQmmhWaveBlocks + b) * 16 + tl;
        if (tok < nt) qmmh_hcmix_store(a, lds + g * kHcMixSlotFloats, xnv[g], rg, t0 + tok, tl, tr);
      }
    }
    lds_sync();
    return;
  }
  if (n < a.n) {
#pragma unroll
    for (uint32_t b = 0; b < kQmmhWaveBlocks; ++b) {
      const uint32_t tb = wm * kQmmhWaveBlocks + b;
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t t = tb * 16 + 2 * i + h;
        if (t < nt) {
          const uint32_t tok = t0 + t;
          if (ks > 1) a.part[(uint64_t(slice) * a.t + tok) * a.n + n] = acc[b][i];
          else a.y[uint64_t(a.y_row ? a.y_row[tok] : tok) * a.ldy + n] = acc[b][i];
        }
      }
    }
  }
}

// The f16 WMMA qmm on IQ3_S read-once tiles (tiles.h), on the MoE tile shape (QmmhMoe: 8 waves x 16 rows, one
// block of 16 tokens). Same numerics as qmmh_item<IQ3_S, QmmhMoe> and the same bits: the same f16 activations, the
// same integer codes, and each unit's partial accumulated in K order with acc = fmaf(scale, partial, acc). What
// changes is the fetch. A K chunk is one superblock (8 units): the workgroup stages its 256 activations per token
// in the LDS arena (16 x 264 halves), each lane pulls its 56 tile bytes with 3 wide loads and decodes its 4 whole
// units (2p + h), and the half-waves swap units so both hold each unit's 32 codes, as the B fragment needs.
constexpr uint32_t kTileChunkUnits = 8;
constexpr uint32_t kTileRowHalves = kTileChunkUnits * 32 + 8;  // f16 per staged token row (+8 pad)
static_assert(QmmhMoe::kToks * kTileRowHalves * 2 <= kLdsArenaBytes, "the superblock tile lives in the LDS arena");
static_assert(QmmhMoe::kTokBlocks == 1 && QmmhMoe::kWaveBlocks == 1, "the tiled path assumes one token block per wave");

__device__ __attribute__((always_inline)) inline void qmmh_item_iq3s_tiles(const QmmArgs &a, uint32_t item) {
  using Sh = QmmhMoe;
  const uint32_t n_tt = (a.t + Sh::kToks - 1) / Sh::kToks;
  const uint32_t rg = item / n_tt, tt = item % n_tt;
  const uint32_t wave = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t r = lane % 16, h = lane / 16;
  const uint32_t n0 = rg * Sh::kRows + wave * 16, n = n0 + r;
  const uint32_t t0 = tt * Sh::kToks, nt = min(Sh::kToks, a.t - t0);
  const uint32_t nsb = a.k / 256;
  if (a.n % 16 || a.k % 256 || (reinterpret_cast<uintptr_t>(a.w) & 15) || a.xin != kXinPlain || a.yout != kYoutPlain) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  if ((reinterpret_cast<uintptr_t>(a.x) & 15) || (a.ldx & 3)) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  _Float16 *xs = reinterpret_cast<_Float16 *>(lds_arena());
  // A row group past N (the last item of a matrix) decodes group 0 and is never written.
  const uint8_t *tiles = a.w + uint64_t(n0 < a.n ? n0 / 16 : 0) * nsb * kIq3sTile;
  v8f acc = {};
  // The IQ3_S grid (512 words, 2 KB) copied to LDS: the decode's eight grid gathers a unit pair are LDS reads, not
  // global loads. Every item writes the same values; the loop's first lds_sync publishes them.
  __shared__ uint32_t grid_lds[512];
  for (uint32_t i = mk_tid(); i < 512; i += kMkBlock) grid_lds[i] = q::kIq3sGridDev[i];
#if MK_EXPERT_LOOKAHEAD
  // One step of lookahead for both operands. vmcnt counts loads in issue order, so a wait for a load waits for every
  // load issued before it: with a step's tile issued first and its activations only at the staging, waiting for the
  // activations also waited for the tile (and the "tile before the barriers" overlap of D1 was void). Here the next
  // superblock's activations (unconditional loads, the token clamped into range and zeroed at the staging) are issued
  // before its tile, and both before this step's staging and decode, so the wait for this step's data, issued a whole
  // step earlier, never waits for the next step's. Only the schedule changes.
  constexpr uint32_t kXj = Sh::kToks * kTileChunkUnits * 8 / kMkBlock;  // 4 float4 of activations per thread
  const float *xrow[kXj];
#pragma unroll
  for (uint32_t j = 0; j < kXj; ++j) {
    const uint32_t i = mk_tid() + j * kMkBlock, t = min(i / (kTileChunkUnits * 8), nt - 1), q4 = i % (kTileChunkUnits * 8);
    const uint32_t tok = t0 + t;
    xrow[j] = a.x + uint64_t(a.x_row ? a.x_row[tok] : tok) * a.ldx + q4 * 4;
  }
  float4 xn[kXj];
#pragma unroll
  for (uint32_t j = 0; j < kXj; ++j) xn[j] = *reinterpret_cast<const float4 *>(xrow[j]);
  Iq3sLane Lnext;
  Lnext.load(tiles, lane);
#endif
  for (uint32_t sb = 0; sb < nsb; ++sb) {
#if MK_EXPERT_LOOKAHEAD
    const Iq3sLane L = Lnext;
    float4 xc[kXj];
#pragma unroll
    for (uint32_t j = 0; j < kXj; ++j) xc[j] = xn[j];
    {
      const uint32_t sn = min(sb + 1, nsb - 1);  // the last step re-reads its own superblock (unused)
#pragma unroll
      for (uint32_t j = 0; j < kXj; ++j) xn[j] = *reinterpret_cast<const float4 *>(xrow[j] + sn * 256);
      Lnext.load(tiles + uint64_t(sn) * kIq3sTile, lane);
    }
    lds_sync();  // every wave is done reading the previous superblock's activations
#pragma unroll
    for (uint32_t j = 0; j < kXj; ++j) {
      const uint32_t i = mk_tid() + j * kMkBlock, t = i / (kTileChunkUnits * 8), q4 = i % (kTileChunkUnits * 8);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < nt) {
        v = xc[j];
        if (!(fabsf(v.x) <= 65504.f && fabsf(v.y) <= 65504.f && fabsf(v.z) <= 65504.f && fabsf(v.w) <= 65504.f))
          MK_TRAP(kTrapActRange, kTrapFileQmmh);  // beyond f16 (or NaN): crash, never inf
      }
      _Float16 *dst = xs + t * kTileRowHalves + q4 * 4;
      dst[0] = _Float16(v.x);
      dst[1] = _Float16(v.y);
      dst[2] = _Float16(v.z);
      dst[3] = _Float16(v.w);
    }
    lds_sync();
#else
  // Without the lookahead (k_stage's TU: its register budget has no room for the prefetched activations): this
  // superblock's tile bytes are loaded before the barriers that stage its activations, so the fetch overlaps the
  // staging (PLAN.md Speed D1). Same bits as the lookahead.
    Iq3sLane L;
    L.load(tiles + uint64_t(sb) * kIq3sTile, lane);
    lds_sync();  // every wave is done reading the previous superblock's activations
    for (uint32_t i = mk_tid(); i < Sh::kToks * kTileChunkUnits * 8; i += kMkBlock) {
      const uint32_t t = i / (kTileChunkUnits * 8), q4 = i % (kTileChunkUnits * 8);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < nt) {
        const uint32_t tok = t0 + t;
        v = *reinterpret_cast<const float4 *>(a.x + uint64_t(a.x_row ? a.x_row[tok] : tok) * a.ldx + sb * 256 + q4 * 4);
        if (!(fabsf(v.x) <= 65504.f && fabsf(v.y) <= 65504.f && fabsf(v.z) <= 65504.f && fabsf(v.w) <= 65504.f))
          MK_TRAP(kTrapActRange, kTrapFileQmmh);  // beyond f16 (or NaN): crash, never inf
      }
      _Float16 *dst = xs + t * kTileRowHalves + q4 * 4;
      dst[0] = _Float16(v.x);
      dst[1] = _Float16(v.y);
      dst[2] = _Float16(v.z);
      dst[3] = _Float16(v.w);
    }
    lds_sync();
#endif
#pragma unroll 1
    for (uint32_t p = 0; p < 4; ++p) {  // rolled: one unit pair's codes live at a time (k_mega's register budget)
      union {
        v16h v[2];
        _Float16 e[32];
        int32_t w[16];
      } mine, other;
      iq3s_codes_f16w(L, p, mine.w, grid_lds);
#pragma unroll
      for (int j = 0; j < 16; ++j) other.w[j] = __shfl_xor(mine.w[j], 16, kWave);
      const float s_mine = L.scale(p), s_other = __shfl_xor(s_mine, 16, kWave);
#pragma unroll
      for (uint32_t uu = 0; uu < 2; ++uu) {  // K order: unit 2p (held by half 0), then 2p + 1 (half 1)
        const bool own = h == uu;
        const v16h b_lo = own ? mine.v[0] : other.v[0], b_hi = own ? mine.v[1] : other.v[1];
        const float s = own ? s_mine : s_other;
        const _Float16 *ap = xs + r * kTileRowHalves + (2 * p + uu) * 32;
        const v16h a_lo = *reinterpret_cast<const v16h *>(ap);
        const v16h a_hi = *reinterpret_cast<const v16h *>(ap + 16);
        const v8f zero = {};
        const v8f pr = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b_hi,
                                                                  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_lo, zero));
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(s, pr[i], acc[i]);
      }
    }
  }
  if (n < a.n) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t t = 2 * i + h;
      if (t < nt) {
        const uint32_t tok = t0 + t;
        a.y[uint64_t(a.y_row ? a.y_row[tok] : tok) * a.ldy + n] = acc[i];
      }
    }
  }
}

// The IQ4_NL expert down matrices on read-once tiles (tiles.h), on the MoE shape (QmmhMoe: 8 waves x 16 rows, one block
// of <= 16 tokens, so every item is a decode-path item). Same numerics as qmmh_item<IQ4_NL, QmmhMoe> and the same bits: the same f16 activations, the same integer codes, each unit's
// partial accumulated in K order with acc = fmaf(d, partial, acc). What changes is the fetch, on the D1/D4 pattern:
// a K chunk is 8 units (16 tokens x 8 units of f16 fill the LDS arena), each lane issues the chunk's weight loads
// (wide, contiguous, unconditional: units past K's end reload the last tile, unused) before the barriers that stage
// the chunk's activations, so the fetch overlaps the staging.
template <uint32_t T>
struct MoeTileLoads;

// IQ4_NL: 2 tiles of 4 units a chunk; per lane 2 x (two 16-byte qs loads + one 8-byte d load). (The Q8_0 expert down
// matrices, 5 layers, stay in the GGUF layout: its 32-byte contiguous runs already read at ~180 GB/s in the microbench and
// the dense Q8_0 tiles here measured 195 at one token and 157 at sixteen, PLAN.md.)
// The IQ4_NL table in registers: its bytes (code + 128, so 1..241 and unsigned), four per word, at compile time from the
// table itself.
__host__ __device__ constexpr uint32_t iq4_biased_word(uint32_t w) {
  uint32_t r = 0;
  for (uint32_t b = 0; b < 4; ++b) r |= uint32_t(uint8_t(int(mkgguf::tables::kValuesIq4nl[4 * w + b]) + 128)) << (8 * b);
  return r;
}
// Four nibble indices (one per byte of idx, each 0..15) to four f16 codes, packed two per word (lo: values 0, 1; hi:
// 2, 3), the same f16 bits as _Float16(kIq4nlDev[i]). The lookup is two v_perm_b32 (selector bits 0..2 pick a byte of
// each half of the table) and a select on bit 3. The f16 is built like the IQ3_S codes' (tiles.h iq3s_codes_f16w): the
// byte u = code + 128 as the low byte under 0x64 is the f16 1024 + u, exactly (the ulp at 2^10 is 1, u < 1024), and
// subtracting 1152 = 1024 + 128 (an exact integer subtraction) leaves the code, -127..113. A table entry of 0 would come
// out as +0 (1152 - 1152 rounds to +0), as the converted int gave; the table has none.
__device__ __attribute__((always_inline)) inline void iq4_f16x4(uint32_t idx, int32_t &lo, int32_t &hi) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  constexpr uint32_t T0 = iq4_biased_word(0), T1 = iq4_biased_word(1), T2 = iq4_biased_word(2), T3 = iq4_biased_word(3);
  const uint32_t sel = idx & 0x07070707u;
  const uint32_t a = __builtin_amdgcn_perm(T1, T0, sel), b = __builtin_amdgcn_perm(T3, T2, sel);
  const uint32_t m = ((idx >> 3) & 0x01010101u) * 0xffu;
  const uint32_t u = (b & m) | (a & ~m);
  const h2 bias = {_Float16(1152.f), _Float16(1152.f)};
  const h2 f0 = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)) - bias;
  const h2 f1 = __builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)) - bias;
  lo = __builtin_bit_cast(int32_t, f0);
  hi = __builtin_bit_cast(int32_t, f1);
}

template <>
struct MoeTileLoads<q::IQ4_NL_TILES> {
  static constexpr uint32_t kTileValues = 128;
  u32x4 qs[2][2];
  u32x2 d[2];
  template <int P>
  __device__ __attribute__((always_inline)) void load(const WLd<P> &wl, const uint8_t *grp, uint32_t k, uint32_t cu0,
                                                      uint32_t lane) {
    const uint32_t ntl = k / 128;
#pragma unroll
    for (uint32_t ti = 0; ti < 2; ++ti) {
      const uint8_t *tile = grp + uint64_t(min(cu0 / 4 + ti, ntl - 1)) * kIq4Tile;
      qs[ti][0] = wl.v16(tile + 16 * lane);
      qs[ti][1] = wl.v16(tile + 512 + 16 * lane);
      d[ti] = wl.v8(tile + 1024 + 8 * (lane % 16));
    }
  }
  // Unit uu of the chunk: this lane's 16 codes (half h of the unit, see tiles.h: bytes 8 h .. 8 h + 7's low nibbles are
  // values 8 h ..., their high nibbles values 16 + 8 h ...), exchanged with the partner lane into the two B fragments.
  __device__ __attribute__((always_inline)) float unit(uint32_t uu, uint32_t h, v16h &b_lo, v16h &b_hi) const {
    const uint32_t ti = uu / 4, ub = uu % 4, p = ub / 2, v = ub % 2;
    union F {
      v16h v;
      _Float16 e[16];
      int32_t w[8];
    } mine, other, lo, hi;
    // The 16-entry table lookup in registers (v_perm_b32, four nibbles at a time) instead of sixteen gathers from the
    // table in memory; the f16 codes are built exactly, see iq4_f16x4.
    {
      const uint32_t w0 = qs[ti][p][2 * v], w1 = qs[ti][p][2 * v + 1];
      iq4_f16x4(w0 & 0x0f0f0f0fu, mine.w[0], mine.w[1]);
      iq4_f16x4(w1 & 0x0f0f0f0fu, mine.w[2], mine.w[3]);
      iq4_f16x4((w0 >> 4) & 0x0f0f0f0fu, mine.w[4], mine.w[5]);
      iq4_f16x4((w1 >> 4) & 0x0f0f0f0fu, mine.w[6], mine.w[7]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) other.w[j] = __shfl_xor(mine.w[j], 16, kWave);
    // b_lo = values 0..15: the h = 0 lane's low nibbles (values 0..7), then the h = 1 lane's (8..15); b_hi likewise.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      lo.w[j] = h == 0 ? mine.w[j] : other.w[j];
      lo.w[4 + j] = h == 0 ? other.w[j] : mine.w[j];
      hi.w[j] = h == 0 ? mine.w[4 + j] : other.w[4 + j];
      hi.w[4 + j] = h == 0 ? other.w[4 + j] : mine.w[4 + j];
    }
    b_lo = lo.v;
    b_hi = hi.v;
    return q::h2f(uint16_t(d[ti][ub / 2] >> (16 * (ub % 2))));
  }
};

template <uint32_t T, int P>
__device__ __attribute__((always_inline)) inline void qmmh_item_moe_tiles(const QmmArgs &a, uint32_t item) {
  using Sh = QmmhMoe;
  static_assert(T == q::IQ4_NL_TILES, "IQ4_NL tiles only");
  static_assert(Sh::kTokBlocks == 1 && Sh::kWaveBlocks == 1, "the tiled MoE path assumes one token block per wave");
  static_assert(16 * kTileRowHalves * 2 <= kLdsArenaBytes && kTileChunkUnits == 8, "8-unit chunks");
  const uint32_t n_tt = (a.t + Sh::kToks - 1) / Sh::kToks;
  const uint32_t rg = item / n_tt, tt = item % n_tt;
  const uint32_t wave = mk_tid() / kWave, lane = mk_tid() % kWave;
  const uint32_t r = lane % 16, h = lane / 16;
  const uint32_t n0 = rg * Sh::kRows + wave * 16, n = n0 + r;
  const uint32_t t0 = tt * Sh::kToks, nt = min(Sh::kToks, a.t - t0);
  const uint32_t units = a.k / 32;
  if (a.n % 16 || a.k % MoeTileLoads<T>::kTileValues || (reinterpret_cast<uintptr_t>(a.w) & 15)) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  if ((reinterpret_cast<uintptr_t>(a.x) & 15) || (a.ldx & 3) || a.ksplit > 1 || a.yout != kYoutPlain) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  _Float16 *xs = reinterpret_cast<_Float16 *>(lds_arena());
  // A row group past N (the last item of a matrix) decodes group 0 and is never written.
  const uint32_t g = n0 < a.n ? n0 / 16 : 0;
  const uint8_t *grp = iq4_tile(a.w, a.k, g, 0);
  const WLd<P> wl(a.w);
  v8f acc = {};
  for (uint32_t cu0 = 0; cu0 < units; cu0 += kTileChunkUnits) {
    const uint32_t cn = min(kTileChunkUnits, units - cu0);
    MoeTileLoads<T> W;
    W.load(wl, grp, a.k, cu0, lane);
    lds_sync();  // every wave is done reading the previous chunk's activations
    for (uint32_t i = mk_tid(); i < Sh::kToks * kTileChunkUnits * 8; i += kMkBlock) {
      const uint32_t t = i / (kTileChunkUnits * 8), q4 = i % (kTileChunkUnits * 8);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (t < nt && q4 / 8 < cn) {
        const uint32_t tok = t0 + t;
        v = qmmh_x4(a, tok, cu0 * 32 + q4 * 4);
        if (!(fabsf(v.x) <= 65504.f && fabsf(v.y) <= 65504.f && fabsf(v.z) <= 65504.f && fabsf(v.w) <= 65504.f))
          MK_TRAP(kTrapActRange, kTrapFileQmmh);  // beyond f16 (or NaN): crash, never inf
      }
      _Float16 *dst = xs + t * kTileRowHalves + q4 * 4;
      dst[0] = _Float16(v.x);
      dst[1] = _Float16(v.y);
      dst[2] = _Float16(v.z);
      dst[3] = _Float16(v.w);
    }
    lds_sync();
#pragma unroll
    for (uint32_t uu = 0; uu < kTileChunkUnits; ++uu) {
      if (uu < cn) {
        v16h b_lo, b_hi;
        const float s = W.unit(uu, h, b_lo, b_hi);
        const _Float16 *ap = xs + r * kTileRowHalves + uu * 32;
        const v16h a_lo = *reinterpret_cast<const v16h *>(ap);
        const v16h a_hi = *reinterpret_cast<const v16h *>(ap + 16);
        const v8f zero = {};
        const v8f p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi, b_hi,
                                                                 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b_lo, zero));
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(s, p[i], acc[i]);
      }
      // One unit's activations and codes live at a time (as in the dense decode paths).
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if (n < a.n) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t t = 2 * i + h;
      if (t < nt) {
        const uint32_t tok = t0 + t;
        a.y[uint64_t(a.y_row ? a.y_row[tok] : tok) * a.ldy + n] = acc[i];
      }
    }
  }
}

// The item with plain weight loads.
template <uint32_t T>
__device__ __attribute__((always_inline)) inline void qmmh_item_moe_tiles_w(const QmmArgs &a, uint32_t item) {
  qmmh_item_moe_tiles<T, -1>(a, item);
}

template <uint32_t T, class Sh>
__device__ __attribute__((always_inline)) inline void qmmh_item_w(const QmmArgs &a, uint32_t item) {
  qmmh_item<T, Sh, -1>(a, item);
}

template <class Sh = QmmhDense>
__device__ inline uint32_t qmmh_items(const QmmArgs &a) {
  return ((a.n + Sh::kRows - 1) / Sh::kRows) * ((a.t + Sh::kToks - 1) / Sh::kToks) * (a.ksplit > 1 ? a.ksplit : 1);
}

// y[t, n] = part[0][t][n] + part[1][t][n] + ... in slice order (a left fold): one thread per output.
// The slices are loaded together (unconditional, clamped), then added in order: a runtime-bound loop made one round
// trip per slice (~9 us for 16 slices at decode). Same order, same bits.
__device__ inline void op_qmm_ksum(const QmmArgs &a, uint32_t wg, uint32_t nwg) {
  if (a.ksplit < 2 || a.ksplit > kMaxKsplit) MK_TRAP(kTrapBadArgs, kTrapFileQmmh);
  const uint64_t tn = uint64_t(a.t) * a.n;
  for (uint64_t i = uint64_t(wg) * kMkBlock + mk_tid(); i < tn; i += uint64_t(nwg) * kMkBlock) {
    float v[kMaxKsplit];
#pragma unroll
    for (uint32_t k = 0; k < kMaxKsplit; ++k) v[k] = a.part[uint64_t(min(k, a.ksplit - 1)) * tn + i];
    float s = v[0];
#pragma unroll
    for (uint32_t k = 1; k < kMaxKsplit; ++k)
      if (k < a.ksplit) s += v[k];
    const uint32_t tok = uint32_t(i / a.n), n = uint32_t(i % a.n);
    a.y[uint64_t(tok) * a.ldy + n] = s;
  }
}

template <uint32_t T>
__device__ __attribute__((always_inline)) inline void op_qmmh_t(const QmmArgs &a, uint32_t wg, uint32_t nwg) {
  const uint32_t items = qmmh_items<QmmhDense>(a);
  for (uint32_t i = wg; i < items; i += nwg) qmmh_item_w<T, QmmhDense>(a, i);
}

// The f16 WMMA qmm for the quantized types (qmmh_type); anything else traps: F32 and BF16 weights keep the
// exact kernel (the engine picks per stage).
__device__ __attribute__((always_inline)) inline void op_qmmh(const QmmArgs &a, uint32_t wg, uint32_t nwg) {
  switch (a.type) {
    case q::Q8_0: op_qmmh_t<q::Q8_0>(a, wg, nwg); break;
    case q::Q8_0_TILES: op_qmmh_t<q::Q8_0_TILES>(a, wg, nwg); break;
    case q::Q6_K: op_qmmh_t<q::Q6_K>(a, wg, nwg); break;
    case q::Q6_K_TILES: op_qmmh_t<q::Q6_K_TILES>(a, wg, nwg); break;
    case q::IQ4_NL: op_qmmh_t<q::IQ4_NL>(a, wg, nwg); break;
    case q::IQ4_XS: op_qmmh_t<q::IQ4_XS>(a, wg, nwg); break;
    case q::IQ3_S: op_qmmh_t<q::IQ3_S>(a, wg, nwg); break;
    default: MK_TRAP(kTrapBadOp, kTrapFileQmmh);
  }
}

}  // namespace mk
