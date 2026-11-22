// The helpers of the layer-part ops of Inkling 2 (MK_INKLING2), at file scope. megakernel.cl includes this file
// before the kernel, after inkling2_helpers.cl.
//
// The host forms are in crates/nagaya-mk-inkling2-bmg/src/host/ops.rs. Each sum here has the order of the host
// `dot`: 16 lanes, lane i takes the values i, i + 16, ..., and the lanes join in pairs 8 apart, then 4, 2 and 1.
// The multiplies and the adds stay apart (FP_CONTRACT OFF), as in the host, so each such sum has the bits of the
// host. The divisions, the square roots and the transcendental functions of the card are not those of the host.

// The positions of one work item of OP_RESIDUAL_CONV. It is at least 3, so only the first chunk reads the state.
#define INKLING2_CONV_CHUNK 64
// The f32 of sqrt(5) and sqrt(9): the divisors of the fixed and the dynamic convolution (host `short_conv` and
// `dynamic_conv`). They are literals, because a program-scope table read zeros on the card (2026-11-21).
#define INKLING2_SQRT5      0x1.1e377ap+1f
#define INKLING2_SQRT9      3.0f
// The epsilon of each RMS norm (host `EPS`).
#define INKLING2_EPS        1e-6f

// The sum of one value of each lane of a subgroup, in the order of the host `dot`. Each lane gets the sum.
inline float inkling2_lane_sum(float v) {
#pragma OPENCL FP_CONTRACT OFF
    v = v + intel_sub_group_shuffle_xor(v, 8u);
    v = v + intel_sub_group_shuffle_xor(v, 4u);
    v = v + intel_sub_group_shuffle_xor(v, 2u);
    v = v + intel_sub_group_shuffle_xor(v, 1u);
    return v;
}

// The dot product of two f32 rows of `len` values over one subgroup, in the order of the host `dot`. `len` is a
// multiple of 16. Each lane gets the sum.
inline float inkling2_sg_dot(__global const float* a, __global const float* b, uint len, uint lane) {
#pragma OPENCL FP_CONTRACT OFF
    float s = 0.0f;
    for (uint i = lane; i < len; i += 16u) s = s + a[i] * b[i];
    return inkling2_lane_sum(s);
}

// The factor of an RMS norm with no weight over one subgroup: 1 / sqrt(mean(x^2) + eps) (host `rms_factor`).
inline float inkling2_sg_rms_factor(__global const float* x, uint len, uint lane) {
#pragma OPENCL FP_CONTRACT OFF
    float mean = inkling2_sg_dot(x, x, len, lane) / (float)len;
    return 1.0f / sqrt(mean + INKLING2_EPS);
}

// The steps of a lane in one block read of a subgroup: a row of 128 values gives lane l the values l, l + 16, ...,
// l + 112, the order of the host `dot`. One read instruction thus loads 8 steps of a sum. A loop that loads one
// value for each step waits once for each step: a row of 2,304 values cost about 25 us (profile, 2026-11-22).
#define INKLING2_BLOCK      128
// The block reads that a sum issues before their adds. The adds keep their order, so each sum keeps its bits.
#define INKLING2_AHEAD      4
// The rank of the message gate (host `GATE_RANK`): g has 64 values.
#define INKLING2_GATE_RANK  64
// The positions of a tile of OP_MESSAGE_GATE_DOWN and of OP_MESSAGE_GATE. A tile shares each read of a weight. The
// second is at most 16, one position for each lane.
#define INKLING2_GATE_TILE  4
#define INKLING2_GATE_TILE2 16

// The sums of a subgroup over rows in global memory (no suffix) and in local memory (suffix _slm). Each row is
// 16-byte aligned, and `len` is a multiple of 16. Each sum has the order of the host `dot`, so both give the same bits.
//
// - inkling2_steps8: the 8 steps of each lane from x, a row of floats.
// - inkling2_bf16_steps8: the 8 steps of each lane from w, a row of bf16 values, as f32.
// - inkling2_rms_factor_block: the factor of an RMS norm with no weight (inkling2_sg_rms_factor).
// - inkling2_scaled_dot_block: s plus the sum of a[i] * (x[i] * f) over the values i of this lane. a is bf16 when
//   `wide` is false, else f32. The lanes do not join.
#define INKLING2_SUMS(SPACE, SUFFIX)                                                                                   \
inline float8 inkling2_steps8##SUFFIX(SPACE const float* x) {                                                         \
    return as_float8(intel_sub_group_block_read8((SPACE const uint*)x));                                              \
}                                                                                                                      \
inline float8 inkling2_bf16_steps8##SUFFIX(SPACE const ushort* w) {                                                   \
    return as_float8(convert_uint8(intel_sub_group_block_read_us8(w)) << 16);                                         \
}                                                                                                                      \
inline float inkling2_rms_factor_block##SUFFIX(SPACE const float* x, uint len, uint lane) {                          \
    _Pragma("OPENCL FP_CONTRACT OFF")                                                                                  \
    float s = 0.0f;                                                                                                    \
    uint full = len / INKLING2_BLOCK * INKLING2_BLOCK;                                                                \
    for (uint c = 0; c < full; c += INKLING2_AHEAD * INKLING2_BLOCK) {                                               \
        float8 v[INKLING2_AHEAD];                                                                                      \
        _Pragma("unroll")                                                                                              \
        for (uint u = 0; u < INKLING2_AHEAD; u++)                                                                      \
            v[u] = inkling2_steps8##SUFFIX(x + min(c + u * INKLING2_BLOCK, full - INKLING2_BLOCK));                    \
        _Pragma("unroll")                                                                                              \
        for (uint u = 0; u < INKLING2_AHEAD; u++) {                                                                    \
            if (c + u * INKLING2_BLOCK < full) {                                                                       \
                _Pragma("unroll")                                                                                      \
                for (uint k = 0; k < 8u; k++) s = s + v[u][k] * v[u][k];                                               \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    for (uint i = full + lane; i < len; i += 16u) s = s + x[i] * x[i];                                                \
    float mean = inkling2_lane_sum(s) / (float)len;                                                                    \
    return 1.0f / sqrt(mean + INKLING2_EPS);                                                                           \
}                                                                                                                      \
inline float inkling2_scaled_dot_block##SUFFIX(float s, SPACE const void* a, bool wide, SPACE const float* x, float f, \
                                               uint len, uint lane) {                                                  \
    _Pragma("OPENCL FP_CONTRACT OFF")                                                                                  \
    SPACE const float* af = (SPACE const float*)a;                                                                     \
    SPACE const ushort* ab = (SPACE const ushort*)a;                                                                   \
    uint full = len / INKLING2_BLOCK * INKLING2_BLOCK;                                                                \
    for (uint c = 0; c < full; c += INKLING2_AHEAD * INKLING2_BLOCK) {                                               \
        float8 av[INKLING2_AHEAD], xv[INKLING2_AHEAD];                                                                 \
        _Pragma("unroll")                                                                                              \
        for (uint u = 0; u < INKLING2_AHEAD; u++) {                                                                    \
            uint at = min(c + u * INKLING2_BLOCK, full - INKLING2_BLOCK);                                              \
            av[u] = wide ? inkling2_steps8##SUFFIX(af + at) : inkling2_bf16_steps8##SUFFIX(ab + at);                   \
            xv[u] = inkling2_steps8##SUFFIX(x + at);                                                                   \
        }                                                                                                              \
        _Pragma("unroll")                                                                                              \
        for (uint u = 0; u < INKLING2_AHEAD; u++) {                                                                    \
            if (c + u * INKLING2_BLOCK < full) {                                                                       \
                _Pragma("unroll")                                                                                      \
                for (uint k = 0; k < 8u; k++) s = s + av[u][k] * (xv[u][k] * f);                                       \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    for (uint i = full + lane; i < len; i += 16u) s = s + (wide ? af[i] : bf16_to_f32(ab[i])) * (x[i] * f);           \
    return s;                                                                                                          \
}
INKLING2_SUMS(__global, )
INKLING2_SUMS(__local, _slm)

// The floats of local memory that an op may stage: slm_part of megakernel.cl, 64 KiB.
#define INKLING2_SLM_FLOATS (2u * 64u * 128u)
// The block reads that a subgroup issues before their writes in a copy to local memory.
#define INKLING2_STAGE      4

// The dot product of a bf16 row and an f32 row of local memory, both of `len` values, in one work item. `len` is a
// multiple of 16. The order is that of the host `dot`, with 16 lanes held in private memory.
inline float inkling2_serial_dot(__global const ushort* w, __local const float* g, uint len) {
#pragma OPENCL FP_CONTRACT OFF
    float l[16];
    for (uint i = 0; i < 16u; i++) l[i] = 0.0f;
    for (uint c = 0; c < len; c += 16u) {
        for (uint i = 0; i < 16u; i++) l[i] = l[i] + bf16_to_f32(w[c + i]) * g[c + i];
    }
    for (uint width = 8u; width >= 1u; width /= 2u) {
        for (uint i = 0; i < width; i++) l[i] = l[i] + l[i + width];
    }
    return l[0];
}

// The sigmoid 1 / (1 + exp(-x)) (host `sigmoid`).
inline float inkling2_sigmoid(float x) {
#pragma OPENCL FP_CONTRACT OFF
    return 1.0f / (1.0f + exp(-x));
}

// The log scale of a full layer at position p: 1 + 0.1 * ln(max((p + 1) / 8192, 1)) (host `log_scale`). Below
// position 8,191 the ratio is 1, its log is 0, and the scale is 1 exactly.
inline float inkling2_log_scale(uint position) {
#pragma OPENCL FP_CONTRACT OFF
    // The product by 2^-13 is exact, so it gives the bits of the division of the host.
    float ratio = fmax((float)(position + 1u) * 0x1p-13f, 1.0f);
    return 1.0f + 0.1f * log(ratio);
}

// The key of an f32 in the total order of the host `f32::total_cmp`: a larger key is a larger value, and NaN with
// its sign bit clear is above every number.
inline int inkling2_total_key(float v) {
    int bits = as_int(v);
    return bits ^ (int)(((uint)(bits >> 31)) >> 1);
}
