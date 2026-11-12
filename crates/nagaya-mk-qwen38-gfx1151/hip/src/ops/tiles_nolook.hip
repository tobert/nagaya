// SPDX-License-Identifier: MIT
// tiles.hip built without the expert item's lookahead (qmmh.h QWEN38_EXPERT_LOOKAHEAD = 0): the path k_stage runs. See
// moe_nolook.hip.
#define QWEN38_EXPERT_LOOKAHEAD 0
#include "tiles.hip"
