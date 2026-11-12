// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next (qwen4exp) dimensions, from the GGUF and the original config.json.
#pragma once
#include <cstdint>

namespace qwen38::model {
constexpr uint32_t kLayers = 48;
constexpr uint32_t kEmbd = 2560;
constexpr uint32_t kHc = 4;                    // hyper-connection streams
constexpr uint32_t kHcDim = kHc * kEmbd;       // 10240
constexpr uint32_t kHcRank = 320;
constexpr uint32_t kVocab = 248320;
constexpr float kRmsEps = 1e-6f;
// attention (QSA) layers: il % 4 == 3
constexpr uint32_t kHeads = 24, kKvHeads = 2, kHeadDim = 256, kRopeDims = 64;
constexpr float kRopeTheta = 1e7f;
constexpr uint32_t kIdxHeads = 4, kIdxDim = 128, kIdxRatio = 4, kIdxBudget = 2048;
// gated DeltaNet layers
constexpr uint32_t kGdnKHeads = 16, kGdnVHeads = 48, kGdnKDim = 128, kGdnVDim = 128, kGdnConv = 4;
constexpr uint32_t kGdnKeyDim = kGdnKHeads * kGdnKDim;              // 2048
constexpr uint32_t kGdnValDim = kGdnVHeads * kGdnVDim;              // 6144
constexpr uint32_t kGdnConvDim = 2 * kGdnKeyDim + kGdnValDim;       // 10240
// MoE
constexpr uint32_t kExperts = 512, kTopK = 10, kFf = 640, kFfShared = 640;
// PLE (layer 1 only)
constexpr uint32_t kPleLayer = 1, kPleHeads = 16, kPleHeadDim = 160, kPleConv = 4, kPleDilation = 3;
constexpr uint32_t kPleEos = 248044;
constexpr uint32_t kMaxPassTokens = 256;
// A context holds at most this many tokens (the kaiseki app's n_ctx; DESIGN.md "Service"), in multiples of a pass.
constexpr uint32_t kMaxContext = 32768;
inline constexpr bool is_attn_layer(uint32_t il) { return il % 4 == 3; }
}  // namespace qwen38::model
