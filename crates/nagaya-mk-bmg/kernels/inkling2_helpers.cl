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

// The softcapped SwiGLU of Inkling 2: 4 tanh(g / 4) * sigmoid(g) * 25 tanh(u / 25), in that order
// (Inkling2LoopSoftcappedSwiGLU). The sigmoid sees the raw gate.
inline float inkling2_swiglu(float g, float u) {
    return ((4.0f * tanh(g / 4.0f)) * (1.0f / (1.0f + exp(-g)))) * (25.0f * tanh(u / 25.0f));
}
