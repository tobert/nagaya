// SPDX-License-Identifier: MIT
// moe.hip built without the expert item's lookahead (qmmh.h QWEN38_EXPERT_LOOKAHEAD = 0): the path k_stage runs. The op tests
// (tests/ops/test_tiles.py, test_moe.py) run the same cases on libqwen38_moe (lookahead on, k_mega's) and libqwen38_moe_nolook.
#define QWEN38_EXPERT_LOOKAHEAD 0
#include "moe.hip"
