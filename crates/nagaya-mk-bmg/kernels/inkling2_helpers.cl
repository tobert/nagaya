// The helpers of the Inkling 2 ops (MK_INKLING2), at file scope. megakernel.cl includes this file before the kernel.
//
// The expert planes are in the order of src/repack.rs of the Inkling 2 crate: tile, step, word, lane. A step is 16
// columns of one row, and each lane holds one row of a tile of 16. The expansion gives the bits of the host ops in
// crates/nagaya-mk-inkling2-bmg/src/formats.rs: the product in f32, in the order (e2m1 * scale) * factor, then a
// round to bf16, to the nearest, ties to even.

// 2 to the power e, for e from -126 to 127, built from its bits so it is exact.
inline float inkling2_pow2(int e) {
    return as_float((uint)(e + 127) << 23);
}

// The value of one E2M1 code: 0, 0.5, 1, 1.5, 2, 3, 4, 6, and the negative copies for codes 8 to 15. The value comes
// from the bits, with no table: a program-scope table read zeros on the card (2026-11-21).
inline float inkling2_e2m1(uint code) {
    uint m = code & 7u;
    float magnitude = m < 2u ? 0.5f * (float)m : (1.0f + 0.5f * (float)(m & 1u)) * inkling2_pow2((int)(m >> 1) - 1);
    return (code & 8u) ? -magnitude : magnitude;
}

// One E4M3 code (4 exponent bits, bias 7, 3 mantissa bits). The load refuses the NaN codes 0x7F and 0xFF, so they
// never reach the card. The result is exact, and it keeps the sign of zero.
inline float inkling2_e4m3(uint code) {
    int exponent = (int)((code >> 3) & 0xFu);
    float mantissa = (float)(code & 7u);
    float magnitude = exponent == 0 ? mantissa * inkling2_pow2(-9) : (8.0f + mantissa) * inkling2_pow2(exponent - 10);
    return (code & 0x80u) ? -magnitude : magnitude;
}

// One UE8M0 scale: 2 to the power code - 127. Code 0 is 2^-127, a subnormal. The load refuses the code 255.
inline float inkling2_ue8m0(uint code) {
    return code == 0u ? as_float(1u << 22) : as_float(code << 23);
}

// An f32 to bf16 bits, to the nearest, ties to even. The values are never NaN: the load checks the factors.
inline ushort inkling2_bf16(float value) {
    uint bits = as_uint(value);
    return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// The 16 bf16 values of step `step` of this lane's row of NVFP4 tile `tile`, as the B operand of a dpas: value 2i in
// the low half of word i, value 2i + 1 in the high half. `steps` is the count of steps of a row.
inline int8 inkling2_nvfp4_step(__global const uint* codes, __global const uchar* scales, uint tile, uint steps,
                                uint step, uint lane, float factor) {
    ulong at = (ulong)tile * steps + step;
    uint w0 = codes[(at * 2u + 0u) * 16u + lane];
    uint w1 = codes[(at * 2u + 1u) * 16u + lane];
    float scale = inkling2_e4m3(scales[at * 16u + lane]);
    uint out[8];
    for (uint i = 0; i < 8; i++) {
        uint byte = ((i < 4 ? w0 : w1) >> (8u * (i % 4u))) & 0xFFu;
        ushort lo = inkling2_bf16((inkling2_e2m1(byte & 0xFu) * scale) * factor);
        ushort hi = inkling2_bf16((inkling2_e2m1(byte >> 4) * scale) * factor);
        out[i] = (uint)lo | ((uint)hi << 16);
    }
    return (int8)((int)out[0], (int)out[1], (int)out[2], (int)out[3], (int)out[4], (int)out[5], (int)out[6],
                  (int)out[7]);
}

// The same for MXFP8: 4 words of E4M3 values for each step, and one UE8M0 scale for each 2 steps.
inline int8 inkling2_mxfp8_step(__global const uint* codes, __global const uchar* scales, uint tile, uint steps,
                                uint step, uint lane) {
    ulong at = (ulong)tile * steps + step;
    float scale = inkling2_ue8m0(scales[((ulong)tile * (steps / 2u) + step / 2u) * 16u + lane]);
    uint out[8];
    for (uint i = 0; i < 8; i++) {
        uint word = codes[(at * 4u + i / 2u) * 16u + lane];
        uint pair = word >> (16u * (i % 2u));
        ushort lo = inkling2_bf16(inkling2_e4m3(pair & 0xFFu) * scale);
        ushort hi = inkling2_bf16(inkling2_e4m3((pair >> 8) & 0xFFu) * scale);
        out[i] = (uint)lo | ((uint)hi << 16);
    }
    return (int8)((int)out[0], (int)out[1], (int)out[2], (int)out[3], (int)out[4], (int)out[5], (int)out[6],
                  (int)out[7]);
}

// The 16 bf16 values of one step of one row of MXFP8 from its 4 code words and its scale code, as
// inkling2_mxfp8_step gives them. The row path takes it for a scale code outside 10 to 245.
inline int8 inkling2_mxfp8_values(uint4 words, uint scale_code) {
    float scale = inkling2_ue8m0(scale_code);
    uint out[8];
    for (uint i = 0; i < 8; i++) {
        uint word = words[i / 2u];
        uint pair = word >> (16u * (i % 2u));
        ushort lo = inkling2_bf16(inkling2_e4m3(pair & 0xFFu) * scale);
        ushort hi = inkling2_bf16(inkling2_e4m3((pair >> 8) & 0xFFu) * scale);
        out[i] = (uint)lo | ((uint)hi << 16);
    }
    return (int8)((int)out[0], (int)out[1], (int)out[2], (int)out[3], (int)out[4], (int)out[5], (int)out[6],
                  (int)out[7]);
}

// The bytes of one expert in a slab of 128 experts: the codes, then the scales. The slab holds the codes of the 128
// experts, then their scales, then their factors as f32. `rows` and `cols` are the rows and the columns of one
// expert matrix. NVFP4 holds 8 code bytes and 1 scale for each step; MXFP8 holds 16 code bytes and 1 scale for each
// 2 steps.
inline ulong inkling2_code_words(uint rows, uint cols, bool mx) {
    return (ulong)rows * (mx ? cols : cols / 2u) / 4u;
}

inline ulong inkling2_scale_bytes(uint rows, uint cols, bool mx) {
    return (ulong)rows * (mx ? cols / 32u : cols / 16u);
}

// B for step `step` of this lane's row of tile `tile` of expert `e` in slab `slab`.
inline int8 inkling2_expert_step(__global const uchar* slab, uint e, uint rows, uint cols, uint tile, uint step,
                                 uint lane, bool mx) {
    ulong words = inkling2_code_words(rows, cols, mx), scales = inkling2_scale_bytes(rows, cols, mx);
    __global const uint* codes = (__global const uint*)slab + (ulong)e * words;
    __global const uchar* sc = slab + NUM_EXPERTS * words * 4u + (ulong)e * scales;
    if (mx) return inkling2_mxfp8_step(codes, sc, tile, cols / 16u, step, lane);
    __global const float* factors = (__global const float*)(slab + NUM_EXPERTS * (words * 4u + scales));
    return inkling2_nvfp4_step(codes, sc, tile, cols / 16u, step, lane, factors[e]);
}

// The XMX sums of NB tiles of one expert (lane rows of tiles tile0 and tile1) against MTS tiles of 8 rows of X from
// row t0 (n rows of k floats), over every step in order. Each x splits into three bf16 parts, as in XMX_ROWS_TILES of
// megakernel.cl, so the sums keep f32 precision. Rows past n read as zero.
#define INKLING2_XMX(NB, MTS) \
    for (uint step = 0; step < K / 16u; step++) { \
        int8 b0 = inkling2_expert_step(slab, e, rows, K, tile0, step, lane, mx); \
        int8 b1 = NB == 2 ? inkling2_expert_step(slab, e, rows, K, tile1, step, lane, mx) : b0; \
        _Pragma("unroll") for (uint mt = 0; mt < MTS; mt++) { \
            uint xv[8]; \
            intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)X, K * 4, n, K * 4, (int2)(step * 16u, t0 + 8 * mt), xv); \
            short8 hi, mid, lo; \
            _Pragma("unroll") for (uint r = 0; r < 8; r++) { \
                ulong p = bf16_split3(as_float(xv[r])); \
                hi[r] = (short)p; \
                mid[r] = (short)(p >> 16); \
                lo[r] = (short)(p >> 32); \
            } \
            acc0[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(lo, b0, acc0[mt]); \
            acc0[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(mid, b0, acc0[mt]); \
            acc0[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(hi, b0, acc0[mt]); \
            if (NB == 2) { \
                acc1[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(lo, b1, acc1[mt]); \
                acc1[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(mid, b1, acc1[mt]); \
                acc1[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(hi, b1, acc1[mt]); \
            } \
        } \
    }

// Gate and up rows of one lane: tiles tile0 (gate) and tile1 (up), 2 tiles of 8 tokens.
inline void inkling2_xmx_gate_up(__global const uchar* slab, uint e, uint rows, uint K, uint tile0, uint tile1,
                                 __global const float* X, uint t0, uint n, uint lane, bool mx, float8* acc0,
                                 float8* acc1) {
    for (uint mt = 0; mt < 2; mt++) { acc0[mt] = 0.0f; acc1[mt] = 0.0f; }
    INKLING2_XMX(2, 2)
}

// The down rows of one lane: tile tile0, 4 tiles of 8 tokens.
inline void inkling2_xmx_down(__global const uchar* slab, uint e, uint rows, uint K, uint tile0,
                              __global const float* X, uint t0, uint n, uint lane, bool mx, float8* acc0) {
    uint tile1 = tile0;
    float8* acc1 = acc0;
    for (uint mt = 0; mt < 4; mt++) acc0[mt] = 0.0f;
    INKLING2_XMX(1, 4)
}

// The row path of the expert ops (milestone 36 item 1) takes a block of at most INKLING2_ROW_TOKENS tokens. Each
// token picks an expert once, so an expert then has at most INKLING2_ROW_TOKENS entries.
#define INKLING2_ROW_TOKENS 8
// The parts of the columns of a row on the row path: subgroups of one work-group that share one tile. Each part is
// a fixed range of steps, so the order of a sum does not depend on the count of work-groups. With 9 steps in each
// part, the 1,152 tiles of 8 experts fill the 2,048 subgroups of the card in 9 rounds (up) or 4.5 rounds (down).
#define INKLING2_UP_PARTS   16
#define INKLING2_DOWN_PARTS 8

// The magnitudes of the E2M1 codes 0 to 7 of one step of this lane's row of NVFP4, into the table `tab` of the
// subgroup (8 x 16 floats, code m of lane l at 16 m + l): bf16((e2m1(m) * scale) * factor), the bits of
// inkling2_nvfp4_step. A code with the sign bit is the negative of its magnitude, also for a zero. Thus 8 products
// of each step serve its 16 codes, for any scale and factor, and the lane reads each magnitude back from local memory.
inline void inkling2_nvfp4_table(uint scale_code, float factor, __local float* tab, uint lane) {
    float scale = inkling2_e4m3(scale_code);
    tab[0u * 16u + lane] = as_float((uint)inkling2_bf16((0.0f * scale) * factor) << 16);
    tab[1u * 16u + lane] = as_float((uint)inkling2_bf16((0.5f * scale) * factor) << 16);
    tab[2u * 16u + lane] = as_float((uint)inkling2_bf16((1.0f * scale) * factor) << 16);
    tab[3u * 16u + lane] = as_float((uint)inkling2_bf16((1.5f * scale) * factor) << 16);
    tab[4u * 16u + lane] = as_float((uint)inkling2_bf16((2.0f * scale) * factor) << 16);
    tab[5u * 16u + lane] = as_float((uint)inkling2_bf16((3.0f * scale) * factor) << 16);
    tab[6u * 16u + lane] = as_float((uint)inkling2_bf16((4.0f * scale) * factor) << 16);
    tab[7u * 16u + lane] = as_float((uint)inkling2_bf16((6.0f * scale) * factor) << 16);
}

// The 16 weights of one step of one row of NVFP4, as f32: the values of inkling2_nvfp4_step, from the table of
// inkling2_nvfp4_table. Each work item reads only its own entries, so no barrier is needed.
inline void inkling2_nvfp4_weights(uint w0, uint w1, __local const float* tab, uint lane, float* w) {
    __local const float* own = tab + lane;
    for (uint i = 0; i < 16; i++) {
        uint c = ((i < 8 ? w0 : w1) >> (4u * (i % 8u))) & 0xFu;
        w[i] = as_float(as_uint(own[(c & 7u) * 16u]) ^ ((c & 8u) << 28));
    }
}

// The 16 weights of one step of one row of MXFP8, as f32: the values of inkling2_mxfp8_step. An E4M3 value has 4
// significant bits, so its product with the power of 2 of the scale is exact, and bf16 holds it, when the product is
// normal. That holds for the scale codes 10 to 245. Then the lane adds the exponent of the scale to the bits of the
// code, with no branch. For another scale code, the lane expands each value, as the XMX path.
inline void inkling2_mxfp8_weights(uint4 words, uint scale_code, float* w) {
    if (scale_code >= 10u && scale_code <= 245u) {
        uint shift = (scale_code - 7u) << 23;
        float sub = as_float((scale_code - 9u) << 23);
        for (uint i = 0; i < 16; i++) {
            uint c = (words[i / 4u] >> (8u * (i % 4u))) & 0xFFu;
            // A normal code: its exponent and mantissa bits in place, plus the exponent of the scale. A subnormal
            // code (exponent 0): its mantissa times 2^-9 times the scale.
            uint normal = ((c & 0x7Fu) << 20) + shift;
            uint small = as_uint((float)(c & 7u) * sub);
            w[i] = as_float(select(normal, small, (uint)((c & 0x78u) == 0u)) ^ ((c & 0x80u) << 24));
        }
    } else {
        int8 v = inkling2_mxfp8_values(words, scale_code);
        for (uint i = 0; i < 8; i++) {
            uint u = (uint)v[i];
            w[2 * i] = as_float(u << 16);
            w[2 * i + 1] = as_float(u & 0xFFFF0000u);
        }
    }
}

// The entries of the row path that one pass over the steps of a part serves. A pass expands each weight once for
// them; more entries need more registers.
#define INKLING2_ROW_CHUNK  2

// The dot products of this lane's row of tile `tile` of expert `e` with the rows x + j k for j < qn (at most
// INKLING2_ROW_TOKENS), over the steps s_lo to s_hi of a row of k columns, into tab[128 + 16 j + lane]. The order is
// fixed: 4 sums, sum i takes the columns i, i + 4, ... of the steps in order, with a fused multiply-add, and the
// result is (sum 0 + sum 1) + (sum 2 + sum 3). The weights have the bits of inkling2_expert_step. `tab` is the local
// memory of the subgroup, 256 floats: the first 128 hold the table of a step of NVFP4.
inline void inkling2_part_dots(__global const uchar* slab, uint e, uint rows, uint k, uint tile, uint s_lo,
                               uint s_hi, __global const float* x, uint qn, uint lane, bool mx, __local float* tab) {
#pragma OPENCL FP_CONTRACT OFF
    ulong words = inkling2_code_words(rows, k, mx), scales = inkling2_scale_bytes(rows, k, mx);
    uint steps = k / 16u, per_step = mx ? 4u : 2u;
    // The codes and the scales of the tile: `steps` steps of `per_step` words of 16 lanes, and the scale bytes.
    __global const uint* codes = (__global const uint*)slab + (ulong)e * words + (ulong)tile * steps * per_step * 16u;
    __global const uchar* sc = slab + NUM_EXPERTS * words * 4u + (ulong)e * scales
                               + (ulong)tile * (mx ? steps / 2u : steps) * 16u + lane;
    float factor = mx ? 1.0f : ((__global const float*)(slab + NUM_EXPERTS * (words * 4u + scales)))[e];
    __attribute__((opencl_unroll_hint(1)))
    for (uint j0 = 0; j0 < qn; j0 += INKLING2_ROW_CHUNK) {
        uint jn = min((uint)INKLING2_ROW_CHUNK, qn - j0);
        float a[INKLING2_ROW_CHUNK][4];
        for (uint j = 0; j < INKLING2_ROW_CHUNK; j++) {
            for (uint i = 0; i < 4; i++) a[j][i] = 0.0f;
        }
        for (uint step = s_lo; step < s_hi; step++) {
            // A block read brings the words of one step to the 16 lanes in one message: a tile holds them lane
            // after lane.
            uint4 raw;
            if (mx) raw = intel_sub_group_block_read4(codes + step * 64u);
            else raw = (uint4)(intel_sub_group_block_read2(codes + step * 32u), 0u, 0u);
            uint code = sc[(mx ? step / 2u : step) * 16u];
            float w[16];
            if (mx) inkling2_mxfp8_weights(raw, code, w);
            else {
                inkling2_nvfp4_table(code, factor, tab, lane);
                inkling2_nvfp4_weights(raw.x, raw.y, tab, lane, w);
            }
            for (uint j = 0; j < INKLING2_ROW_CHUNK; j++) {
                if (j < jn) {
                    float16 xv = vload16(step, x + (ulong)(j0 + j) * k);
                    for (uint i = 0; i < 16; i++) a[j][i % 4u] = fma(w[i], xv[i], a[j][i % 4u]);
                }
            }
        }
        for (uint j = 0; j < INKLING2_ROW_CHUNK; j++) {
            if (j < jn) tab[128u + (j0 + j) * 16u + lane] = (a[j][0] + a[j][1]) + (a[j][2] + a[j][3]);
        }
    }
}

// The softcapped SwiGLU of Inkling 2: 4 tanh(g / 4) * sigmoid(g) * 25 tanh(u / 25), in that order
// (Inkling2LoopSoftcappedSwiGLU). The sigmoid sees the raw gate.
inline float inkling2_swiglu(float g, float u) {
    return ((4.0f * tanh(g / 4.0f)) * (1.0f / (1.0f + exp(-g)))) * (25.0f * tanh(u / 25.0f));
}
