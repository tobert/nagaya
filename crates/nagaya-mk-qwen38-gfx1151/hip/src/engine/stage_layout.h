// SPDX-License-Identifier: MIT
// The C++ half of the layout check of stage_layout.def: each line is a static_assert on the C++ types. The Rust
// schedule (src/schedule.rs) reads the stages of a pass through its mirror of these types (src/stage.rs).
#pragma once
#include <cstddef>

#include "stage.h"

namespace qwen38 {
namespace stage_layout {

#define STAGE_SIZE(T, bytes) static_assert(sizeof(T) == (bytes), "stage_layout.def: the size of " #T " changed");
#define STAGE_FIELD(T, f, offset) static_assert(offsetof(T, f) == (offset), "stage_layout.def: " #T "::" #f " moved");
#define STAGE_UNION(T, f, rust, offset) static_assert(offsetof(T, f) == (offset), "stage_layout.def: " #T "::" #f " moved");
#define STAGE_CONST(name, rust, value) static_assert(uint64_t(name) == uint64_t(value), "stage_layout.def: " #name " changed");
#include "stage_layout.def"
#undef STAGE_SIZE
#undef STAGE_FIELD
#undef STAGE_UNION
#undef STAGE_CONST

}  // namespace stage_layout
}  // namespace qwen38
