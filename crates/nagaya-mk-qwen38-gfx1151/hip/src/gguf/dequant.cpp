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
// Dequant bodies below are ports of ggml's dequantize_row_q8_0 / _q6_K / _iq3_s / _iq4_nl /
// _iq4_xs (ggml/src/ggml-quants.c; see THIRD_PARTY.md). The port keeps ggml's variable names
// and loop structure so it stays checkable line-by-line against the original; only the C
// array indexing (`x[i].field`) was translated to this file's block struct names in
// types.hpp. See dequant.hpp for the 2026-09-26 verification note.
#include "dequant.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#include "tables.hpp"

namespace mkgguf {

const char* ggml_type_name(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return "F32";
        case GgmlType::BF16: return "BF16";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::IQ3_S: return "IQ3_S";
        case GgmlType::IQ4_NL: return "IQ4_NL";
        case GgmlType::IQ4_XS: return "IQ4_XS";
    }
    throw std::runtime_error("mkgguf: ggml_type_name: unhandled GgmlType " +
                              std::to_string(static_cast<uint32_t>(t)));
}

GgmlTypeTraits ggml_type_traits(GgmlType t) {
    switch (t) {
        case GgmlType::F32: return {1, 4};
        case GgmlType::BF16: return {1, 2};
        case GgmlType::Q8_0: return {32, sizeof(BlockQ8_0)};
        case GgmlType::Q6_K: return {256, sizeof(BlockQ6_K)};
        case GgmlType::IQ3_S: return {256, sizeof(BlockIq3S)};
        case GgmlType::IQ4_NL: return {32, sizeof(BlockIq4Nl)};
        case GgmlType::IQ4_XS: return {256, sizeof(BlockIq4Xs)};
    }
    throw std::runtime_error("mkgguf: ggml_type_traits: unhandled GgmlType " +
                              std::to_string(static_cast<uint32_t>(t)));
}

namespace {

void dequantize_row_f32(const float* x, float* y, int64_t k) {
    std::memcpy(y, x, static_cast<size_t>(k) * sizeof(float));
}

void dequantize_row_bf16(const uint16_t* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k; ++i) {
        y[i] = bf16_to_fp32(x[i]);
    }
}

// Ported from ggml's dequantize_row_q8_0.
void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int64_t k) {
    const int qk = 32;
    const int64_t nb = k / qk;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i * qk + j] = x[i].qs[j] * d;
        }
    }
}

// Ported from ggml's dequantize_row_q6_K.
void dequantize_row_q6_K(const BlockQ6_K* x, float* y, int64_t k) {
    const int64_t QK_K = 256;
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);

        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// Ported from ggml's dequantize_row_iq3_s. iq3s_grid entries are read byte-by-byte in native
// (little-endian) order, exactly as ggml's `(const uint8_t *)(iq3s_grid + idx)` cast does on
// this machine's x86_64 target.
void dequantize_row_iq3_s(const BlockIq3S* x, float* y, int64_t k) {
    const int64_t QK_K = 256;
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
                    &tables::kIq3sGrid[qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(
                    &tables::kIq3sGrid[qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * grid1[j] * (signs[l] & tables::kMaskIq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db1 * grid2[j] * (signs[l] & tables::kMaskIq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
                    &tables::kIq3sGrid[qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(
                    &tables::kIq3sGrid[qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * grid1[j] * (signs[l] & tables::kMaskIq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db2 * grid2[j] * (signs[l] & tables::kMaskIq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

// Ported from ggml's dequantize_row_iq4_nl.
void dequantize_row_iq4_nl(const BlockIq4Nl* x, float* y, int64_t k) {
    const int64_t QK4_NL = 32;
    const int64_t nb = k / QK4_NL;

    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j + 0] = d * tables::kValuesIq4nl[qs[j] & 0xf];
            y[j + QK4_NL / 2] = d * tables::kValuesIq4nl[qs[j] >> 4];
        }
        y += QK4_NL;
    }
}

// Ported from ggml's dequantize_row_iq4_xs.
void dequantize_row_iq4_xs(const BlockIq4Xs* x, float* y, int64_t k) {
    const int64_t QK_K = 256;
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                           (((x[i].scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * tables::kValuesIq4nl[qs[j] & 0xf];
                y[j + 16] = dl * tables::kValuesIq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

}  // namespace

void dequant_f32(GgmlType type, const void* src, int64_t n_elements, float* dst) {
    if (!ggml_type_is_known(static_cast<uint32_t>(type))) {
        throw std::runtime_error("mkgguf: dequant_f32: unknown ggml type id " +
                                  std::to_string(static_cast<uint32_t>(type)));
    }
    const GgmlTypeTraits tr = ggml_type_traits(type);
    if (n_elements % tr.block_size != 0) {
        throw std::runtime_error(
            "mkgguf: dequant_f32: n_elements=" + std::to_string(n_elements) +
            " is not a multiple of " + ggml_type_name(type) +
            "'s block size " + std::to_string(tr.block_size));
    }

    switch (type) {
        case GgmlType::F32:
            dequantize_row_f32(reinterpret_cast<const float*>(src), dst, n_elements);
            return;
        case GgmlType::BF16:
            dequantize_row_bf16(reinterpret_cast<const uint16_t*>(src), dst, n_elements);
            return;
        case GgmlType::Q8_0:
            dequantize_row_q8_0(reinterpret_cast<const BlockQ8_0*>(src), dst, n_elements);
            return;
        case GgmlType::Q6_K:
            dequantize_row_q6_K(reinterpret_cast<const BlockQ6_K*>(src), dst, n_elements);
            return;
        case GgmlType::IQ3_S:
            dequantize_row_iq3_s(reinterpret_cast<const BlockIq3S*>(src), dst, n_elements);
            return;
        case GgmlType::IQ4_NL:
            dequantize_row_iq4_nl(reinterpret_cast<const BlockIq4Nl*>(src), dst, n_elements);
            return;
        case GgmlType::IQ4_XS:
            dequantize_row_iq4_xs(reinterpret_cast<const BlockIq4Xs*>(src), dst, n_elements);
            return;
    }
    throw std::runtime_error("mkgguf: dequant_f32: unhandled GgmlType " +
                              std::to_string(static_cast<uint32_t>(type)));
}

}  // namespace mkgguf
