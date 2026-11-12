// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// PORTED (this file): the block struct field ORDER and WIDTH (block_q8_0, block_q6_K,
// block_iq3_s, block_iq4_nl, block_iq4_xs), the fp16_to_fp32 / bf16_to_fp32 conversions
// (ggml_compute_fp16_to_fp32, ggml_compute_bf16_to_fp32), and the GgmlType numeric ids are
// copied verbatim from ggml/llama.cpp (see THIRD_PARTY.md for the source paths and commit),
// so this reader lays out and reads bytes exactly the way ggml and gguf-py do. Verified
// 2026-09-26 line-by-line against ggml/src/ggml-common.h and ggml/src/ggml-impl.h at
// llama.cpp commit 5cf3a3528: struct field order/widths, GGMLQuantizationType ids
// (gguf-py/gguf/constants.py) and the two conversion functions all match; nothing here was
// changed from the reference. Everything else in this file (the enum framing, comments, our
// own GgufValueType helper) is original MIT-licensed work of this repo (LICENSE-MIT).
#pragma once

#include <cstdint>
#include <cstring>

namespace gguf {

// GGML tensor element type ids, exactly as they appear in a GGUF tensor-info record and in
// gguf-py's GGMLQuantizationType. Only the ids present in the Qwen3.8-Flash-Next UD-IQ4_XS
// GGUF are given names. The Rust reader (src/gguf.rs) refuses any other id.
enum class GgmlType : uint32_t {
    F32 = 0,
    BF16 = 30,
    Q8_0 = 8,
    Q6_K = 14,
    IQ3_S = 21,
    IQ4_NL = 20,
    IQ4_XS = 23,
};

// True for every id this reader knows how to size and dequantize.
inline bool ggml_type_is_known(uint32_t id) {
    switch (id) {
        case 0: case 30: case 8: case 14: case 21: case 20: case 23:
            return true;
        default:
            return false;
    }
}

// GGUF metadata value type ids (GGUF spec / gguf-py GGUFValueType), used by the KV reader.
// Original to this repo (not a ggml port): these ids are the GGUF file format's own spec,
// cross-checked against gguf-py/gguf/constants.py GGUFValueType.
enum class GgufValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// ---- fp16 <-> fp32, ported verbatim from ggml_compute_fp16_to_fp32 (ggml-impl.h) ----
// ref: https://github.com/Maratyszcza/FP16
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    float normalized_value;
    {
        uint32_t bits = (two_w >> 4) + exp_offset;
        std::memcpy(&normalized_value, &bits, sizeof(bits));
        normalized_value *= exp_scale;
    }

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    float denormalized_value;
    {
        uint32_t bits = (two_w >> 17) | magic_mask;
        std::memcpy(&denormalized_value, &bits, sizeof(bits));
        denormalized_value -= magic_bias;
    }

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    uint32_t result_bits;
    float result_f;
    if (two_w < denormalized_cutoff) {
        std::memcpy(&result_bits, &denormalized_value, sizeof(float));
    } else {
        std::memcpy(&result_bits, &normalized_value, sizeof(float));
    }
    result_bits = sign | result_bits;
    std::memcpy(&result_f, &result_bits, sizeof(float));
    return result_f;
}

// ---- bf16 -> fp32, ported verbatim from ggml_compute_bf16_to_fp32 (ggml-impl.h): a shift. ----
inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---- Block layouts (packed, no implicit padding; each is verified with a static_assert). ----
// ggml_half is a raw 16-bit float bit pattern; converted with fp16_to_fp32 above.

#pragma pack(push, 1)

struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "block_q8_0 layout drifted from ggml");

// QK_K = 256 everywhere below, matching ggml-common.h's QK_K.
struct BlockQ6_K {
    uint8_t ql[128];      // QK_K/2
    uint8_t qh[64];       // QK_K/4
    int8_t scales[16];    // QK_K/16
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "block_q6_K layout drifted from ggml");

struct BlockIq3S {
    uint16_t d;
    uint8_t qs[64];       // QK_K/4
    uint8_t qh[8];        // QK_K/32
    uint8_t signs[32];    // QK_K/8
    uint8_t scales[4];    // IQ3S_N_SCALE = QK_K/64
};
static_assert(sizeof(BlockIq3S) == 110, "block_iq3_s layout drifted from ggml");

struct BlockIq4Nl {
    uint16_t d;
    uint8_t qs[16];       // QK4_NL/2
};
static_assert(sizeof(BlockIq4Nl) == 18, "block_iq4_nl layout drifted from ggml");

struct BlockIq4Xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];  // QK_K/64
    uint8_t qs[128];       // QK_K/2
};
static_assert(sizeof(BlockIq4Xs) == 136, "block_iq4_xs layout drifted from ggml");

#pragma pack(pop)

}  // namespace gguf
