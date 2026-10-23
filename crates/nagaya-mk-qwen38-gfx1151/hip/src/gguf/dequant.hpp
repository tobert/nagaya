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
// Our C++ GGUF dequantizer: turns a run of GGML-quantized bytes into float32.
//
// PORTED (dequant.cpp): every per-type body is a straight-line port of ggml's
// dequantize_row_* (ggml/src/ggml-quants.c; see THIRD_PARTY.md for the source path and
// commit). Same float ops, same order, no fast-math (see src/gguf/CMakeLists.txt), so results
// are bit-identical to gguf-py's numpy dequant. Verified 2026-09-26 line-by-line against
// ggml/src/ggml-quants.c at llama.cpp commit 5cf3a3528: dequantize_row_q8_0, _q6_K, _iq3_s,
// _iq4_nl and _iq4_xs all match (the only difference is renaming ggml's `x[i].field` to this
// file's block struct field names). This is the ONLY dequant code the engine runs; gguf-py is
// a test-time cross-check only (tests/gguf), never a runtime dependency.
#pragma once

#include <cstddef>
#include <cstdint>

#include "types.hpp"

namespace mkgguf {

// Dequantizes `n_elements` elements of type `type`, starting at `src`, into `dst` (already
// allocated, n_elements floats). n_elements must be an exact multiple of the type's block
// size, and `type` must be one of the ids ggml_type_is_known() accepts — anything else is a
// programmer error and this function crashes (throws std::runtime_error) rather than silently
// producing garbage or a partial result.
void dequant_f32(GgmlType type, const void* src, int64_t n_elements, float* dst);

}  // namespace mkgguf
