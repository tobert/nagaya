// The quantizers of the activations of the experts: OP_QUANT_NVFP4 and OP_QUANT_MXFP8.
//
// Each op gives the bytes of the host quantizers of `nagaya-mk-inkling2-sm121/src/quant.rs` bit for bit, and those
// give the bytes of the shipped vLLM kernels. Every f32 operation here is the same operation in the same order as in
// `quant.rs`. This needs an IEEE division, no flush of denormals to zero, and no fused multiply-add. `build.rs` sets
// `-prec-div=true -ftz=false --fmad=false`.
//
// The ops use the native conversions `cvt.rn.satfinite.e2m1x2.f32` and `cvt.rn.satfinite.e4m3x2.f32`. They round to
// nearest even and saturate, as `encode_e2m1` and `encode_e4m3` do. `tests/cuda_quant.rs` of nagaya-mk-inkling2-sm121
// checks the bytes on the rows of the fixture and on random rows.
//
// The layout of a buffer of quantized rows (`rows` is the capacity, and `k` is the length of a row, a multiple of 128):
//   NVFP4: the codes (`rows * k / 2` bytes, an even element in the low nibble), then the scales (`rows * k / 16`
//          bytes, one `e4m3` for 16 values), then the row amaxes (`rows` floats).
//   MXFP8: the codes (`rows * k` bytes, `e4m3`), then the scales (`rows * k / 32` bytes, one `ue8m0` for 32 values).
// The expert ops read the same layout.
//
// This header needs `mk_protocol.h` and the worker functions of megakernel.cu.

#pragma once

#include <float.h>

// The reciprocals as the host has them: the f64 quotient, rounded once to f32 (`reciprocal_constant` of quant.rs).
constexpr uint32_t QUANT_RECIP_1536_BITS = 0x3a2aaaabu;
constexpr uint32_t QUANT_RECIP_448_BITS = 0x3b124925u;
constexpr uint32_t QUANT_RECIP_6_BITS = 0x3e2aaaabu;

__device__ __forceinline__ float quant_const(uint32_t bits) { return __uint_as_float(bits); }

// The `e4m3` byte of a float: nearest even, saturated at 448 (code 0x7E).
__device__ __forceinline__ uint32_t cvt_e4m3(float value) {
    uint16_t pair;
    // The first source goes in the high byte, and the second in the low byte.
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(value));
    return pair & 0xFFu;
}

// The `e2m1` code of a float (4 bits, the sign is bit 3): nearest even, saturated at 6.0.
__device__ __forceinline__ uint32_t cvt_e2m1(float value) {
    uint32_t pair;
    // The first source goes in the high nibble, and the second in the low nibble.
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %1, %2; cvt.u32.u8 %0, b; }" : "=r"(pair) : "f"(0.0f), "f"(value));
    return pair & 0xFu;
}

// The value of an `e4m3` byte. Every step is exact in f32. The byte is not a NaN.
__device__ __forceinline__ float decode_e4m3(uint32_t code) {
    const uint32_t exponent = (code >> 3) & 0xFu;
    const float mantissa = static_cast<float>(code & 7u);
    const float magnitude = exponent == 0 ? mantissa * __uint_as_float(118u << 23)
                                          : (8.0f + mantissa) * __uint_as_float((exponent - 10u + 127u) << 23);
    return (code & 0x80u) != 0 ? -magnitude : magnitude;
}

// The value of an `e2m1` code. A code of 8 to 15 is the negative of the code minus 8, and the code 8 is -0.0.
__device__ __forceinline__ float decode_e2m1(uint32_t code) {
    const uint32_t c = code & 7u;
    const float magnitude =
        c < 4 ? static_cast<float>(c) * 0.5f : static_cast<float>(2u + (c & 1u)) * static_cast<float>(1u << ((c >> 1) - 2u));
    return (code & 8u) != 0 ? -magnitude : magnitude;
}

// The `ue8m0` byte of a block with the largest magnitude `amax`: the exponent that rounds `amax * (1 / 448)` up to a
// power of two. A raw scale of 2^-127 or less gives 0, and the byte stops at 254 (`encode_ue8m0_ceil`).
__device__ __forceinline__ uint32_t ue8m0_ceil(float amax) {
    const float raw = amax * quant_const(QUANT_RECIP_448_BITS);
    if (raw <= __uint_as_float(1u << 22)) {
        return 0;
    }
    const uint32_t exponent = ((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23;
    return exponent < 254u ? exponent : 254u;
}

// The exact reciprocal of a `ue8m0` scale: 2^127 for the byte 0 (`quantize_mxfp8_ue8m0`).
__device__ __forceinline__ float ue8m0_inverse(uint32_t byte) {
    if (byte == 0) {
        return __uint_as_float(254u << 23);
    }
    return 1.0f / __uint_as_float(byte << 23);
}

// The largest magnitude of the row, in every thread. A value that is not finite is an error: the op traps (spec 8.5).
// The max does not depend on the order, so the tree does not matter.
__device__ __forceinline__ float quant_row_max_abs(const float* x, uint32_t k) {
    __shared__ float part[MK_CONSUMER_THREADS / 32];
    float m = 0.0f;
    for (uint32_t i = worker_id(); i < k; i += worker_count()) {
        const float v = fabsf(x[i]);
        if (!(v <= FLT_MAX)) {
            __trap();
        }
        m = fmaxf(m, v);
    }
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    }
    if ((worker_id() & 31) == 0) {
        part[worker_id() >> 5] = m;
    }
    worker_sync();
    float row = part[0];
    for (uint32_t w = 1; w < MK_CONSUMER_THREADS / 32; ++w) {
        row = fmaxf(row, part[w]);
    }
    worker_sync();
    return row;
}

// The lanes of a half-warp hold the 16 values of a block. The sum of their errors is the pairwise tree of
// `tree_sum16`: adjacent pairs first, then pairs of pairs. Addition of two floats does not depend on their order, so
// every lane of the half-warp gets the same bits.
__device__ __forceinline__ float tree_sum16_lanes(float error) {
    error += __shfl_xor_sync(0xffffffffu, error, 1);
    error += __shfl_xor_sync(0xffffffffu, error, 2);
    error += __shfl_xor_sync(0xffffffffu, error, 4);
    error += __shfl_xor_sync(0xffffffffu, error, 8);
    return error;
}

// One lane of a candidate of an NVFP4 block (`nvfp4_candidate`): the `e2m1` code of its value and the error of the
// block. The reconstruction is `((e2m1 * scale) * outer)`: keep this order.
__device__ __forceinline__ float nvfp4_candidate(float value, float scale, float decoder, float outer, uint32_t& code) {
    const float inverse = fminf(1.0f / (scale * decoder), FLT_MAX);
    code = cvt_e2m1(value * inverse);
    const float error = fabsf((decode_e2m1(code) * scale) * outer - value);
    return tree_sum16_lanes(error);
}

// Quantize one row of `k` floats to NVFP4 with `fouroversix_mae` (`quantize_nvfp4_fouroversix_mae`). A half-warp
// takes a block of 16 values: one lane for each value.
__device__ __forceinline__ void quant_nvfp4_row(const float* x, uint32_t k, uint8_t* codes, uint8_t* scales, float* amax_out) {
    const float amax = quant_row_max_abs(x, k);
    const float encoder = fminf(1536.0f / amax, FLT_MAX);
    const float decoder = 1.0f / encoder;
    const float outer = amax * quant_const(QUANT_RECIP_1536_BITS);
    const float encoder6 = encoder * quant_const(QUANT_RECIP_6_BITS);
    const float encoder4 = encoder * 0.25f;
    const uint32_t lane = worker_id() & 31;
    const uint32_t half = lane >> 4;
    const uint32_t element = lane & 15;
    const uint32_t first = (worker_id() >> 5) * 2 + half;
    const uint32_t stride = (worker_count() >> 5) * 2;
    for (uint32_t block = first; block < k / 16; block += stride) {
        const float value = x[block * 16 + element];
        float block_max = fabsf(value);
        for (uint32_t offset = 8; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffffu, block_max, offset));
        }
        const uint32_t byte6 = cvt_e4m3(fminf(block_max * encoder6, 448.0f));
        const uint32_t byte4 = cvt_e4m3(fminf(block_max * encoder4, 448.0f));
        uint32_t code6, code4;
        const float error6 = nvfp4_candidate(value, decode_e4m3(byte6), decoder, outer, code6);
        const float error4 = nvfp4_candidate(value, decode_e4m3(byte4), decoder, outer, code4);
        // The block takes the scale of 4 only when its error is less. A tie keeps the scale of 6.
        const bool use4 = error4 < error6;
        const uint32_t code = use4 ? code4 : code6;
        const uint32_t neighbor = __shfl_xor_sync(0xffffffffu, code, 1);
        if (element % 2 == 0) {
            codes[block * 8 + element / 2] = static_cast<uint8_t>(code | (neighbor << 4));
        }
        if (element == 0) {
            scales[block] = static_cast<uint8_t>(use4 ? byte4 : byte6);
        }
    }
    if (worker_id() == 0) {
        *amax_out = amax;
    }
}

// Quantize one row of `k` floats to MXFP8 with `ue8m0` scales (`quantize_mxfp8_ue8m0`). A warp takes a block of 32
// values: one lane for each value.
__device__ __forceinline__ void quant_mxfp8_row(const float* x, uint32_t k, uint8_t* codes, uint8_t* scales) {
    const uint32_t lane = worker_id() & 31;
    for (uint32_t block = worker_id() >> 5; block < k / 32; block += worker_count() >> 5) {
        const float value = x[block * 32 + lane];
        float block_max = fabsf(value);
        if (!(block_max <= FLT_MAX)) {
            __trap();
        }
        for (uint32_t offset = 16; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffffu, block_max, offset));
        }
        const uint32_t byte = ue8m0_ceil(block_max);
        const float inverse = ue8m0_inverse(byte);
        codes[block * 32 + lane] = static_cast<uint8_t>(cvt_e4m3(fminf(fmaxf(value * inverse, -448.0f), 448.0f)));
        if (lane == 0) {
            scales[block] = static_cast<uint8_t>(byte);
        }
    }
}

// The input rows are `a` (`const float*`, `k` floats in each). The output is the buffer of quantized rows at `c`, with
// room for `rows` rows. Item `i` quantizes the row `i`. `items` is the count of rows to quantize, and it is at most
// `rows`. A row that this op does not write keeps its bytes.
__device__ __forceinline__ void op_quant_nvfp4(const MkTaskDescriptor& task, uint32_t item) {
    const uint32_t k = task.k;
    if (worker_count() != MK_CONSUMER_THREADS || (k % MK_EXPERT_K_GROUP) != 0 || item >= task.rows) {
        __trap();  // The layout needs whole groups of 128 values, and the tree of the max needs 256 workers.
    }
    const float* x = reinterpret_cast<const float*>(task.a) + static_cast<uint64_t>(item) * k;
    uint8_t* base = reinterpret_cast<uint8_t*>(task.c);
    uint8_t* codes = base + static_cast<uint64_t>(item) * (k / 2);
    uint8_t* scales = base + static_cast<uint64_t>(task.rows) * (k / 2) + static_cast<uint64_t>(item) * (k / 16);
    float* amax = reinterpret_cast<float*>(base + static_cast<uint64_t>(task.rows) * (k / 2 + k / 16)) + item;
    quant_nvfp4_row(x, k, codes, scales, amax);
}

__device__ __forceinline__ void op_quant_mxfp8(const MkTaskDescriptor& task, uint32_t item) {
    const uint32_t k = task.k;
    if (worker_count() != MK_CONSUMER_THREADS || (k % MK_EXPERT_K_GROUP) != 0 || item >= task.rows) {
        __trap();
    }
    const float* x = reinterpret_cast<const float*>(task.a) + static_cast<uint64_t>(item) * k;
    uint8_t* base = reinterpret_cast<uint8_t*>(task.c);
    uint8_t* codes = base + static_cast<uint64_t>(item) * k;
    uint8_t* scales = base + static_cast<uint64_t>(task.rows) * k + static_cast<uint64_t>(item) * (k / 32);
    quant_mxfp8_row(x, k, codes, scales);
}
