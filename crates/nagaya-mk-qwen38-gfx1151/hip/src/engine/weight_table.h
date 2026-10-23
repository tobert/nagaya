// SPDX-License-Identifier: MIT
// The weight table: the tensors of the arena, as the Rust host planned and loaded them (src/arena.rs, src/table.rs).
// The engine reads no GGUF and fills no arena (milestone 15 steps 3 and 6). The arena stays mapped for the engine's life.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "rt.h"

extern "C" {
// One access of a stage's footprint (src/schedule.rs): the bytes [lo, hi), written when write is 1.
struct mk_access {
  uint64_t lo, hi;
  uint32_t write;
};

// One tensor of the arena in its final layout: the type (a ggml type or a tile type, quant.h), the shape, the bytes of
// one row of ne[0] values, and its first byte in the arena.
struct mk_weight {
  const char *name;
  uint32_t type;
  uint32_t pad;
  uint64_t ne[3];
  uint64_t row_bytes;
  const uint8_t *d;
};

// The table that the Rust host gives to mk_engine_open. Every pointer stays valid for the call; the arena stays valid
// for the engine's life.
struct mk_weight_table {
  uint64_t n_weights;
  const mk_weight *weights;
  // The schedule of the n stages (mk::Stage) of a pass in build order (src/schedule.rs). It writes the build index of
  // the k-th stage to run to order[k], and its Stage::sched to sched[k]. With elide 0, each stage keeps its place with
  // a grid barrier after it. Returns the grid barriers. Stops the program on an error.
  uint32_t (*schedule)(const void *stages, uint32_t n, const uint32_t *window_starts, uint32_t n_windows, uint32_t elide,
                       uint32_t tail, uint32_t *order, uint32_t *sched);
  // The footprint of one stage (src/schedule.rs): at most cap accesses to out, and the fence flag. Returns the count.
  // Stops the program on an error, or when the accesses are more than cap.
  uint32_t (*footprint)(const void *stage, mk_access *out, uint32_t cap, uint32_t *fence);
};
}

// The layouts of the two structs, which src/table.rs mirrors. The Rust test src/layout.rs reads these lines and
// compares each number with the Rust struct, so a change on one side fails a test. Keep each assert on one line.
static_assert(sizeof(mk_weight) == 56, "mk_weight");
static_assert(offsetof(mk_weight, name) == 0, "mk_weight.name");
static_assert(offsetof(mk_weight, type) == 8, "mk_weight.type, MkWeight.kind in Rust");
static_assert(offsetof(mk_weight, ne) == 16, "mk_weight.ne");
static_assert(offsetof(mk_weight, row_bytes) == 40, "mk_weight.row_bytes");
static_assert(offsetof(mk_weight, d) == 48, "mk_weight.d");
static_assert(sizeof(mk_weight_table) == 32, "mk_weight_table");
static_assert(offsetof(mk_weight_table, n_weights) == 0, "mk_weight_table.n_weights");
static_assert(offsetof(mk_weight_table, weights) == 8, "mk_weight_table.weights");
static_assert(offsetof(mk_weight_table, schedule) == 16, "mk_weight_table.schedule");
static_assert(offsetof(mk_weight_table, footprint) == 24, "mk_weight_table.footprint");
static_assert(sizeof(mk_access) == 24, "mk_access");
static_assert(offsetof(mk_access, lo) == 0, "mk_access.lo");
static_assert(offsetof(mk_access, hi) == 8, "mk_access.hi");
static_assert(offsetof(mk_access, write) == 16, "mk_access.write");

namespace mk {

// The engine's copy of the schedule calls of a mk_weight_table. A null field stops the program.
class ScheduleCalls {
 public:
  explicit ScheduleCalls(const mk_weight_table *t) {
    if (!t) MK_DIE("the weight table is null");
    if (!t->schedule || !t->footprint) MK_DIE("the weight table has no schedule calls");
    schedule_ = t->schedule;
    footprint_ = t->footprint;
  }
  uint32_t schedule(const void *stages, uint32_t n, const uint32_t *window_starts, uint32_t n_windows, bool elide, bool tail,
                    uint32_t *order, uint32_t *sched) const {
    return schedule_(stages, n, window_starts, n_windows, elide ? 1 : 0, tail ? 1 : 0, order, sched);
  }
  uint32_t footprint(const void *stage, mk_access *out, uint32_t cap, uint32_t *fence) const {
    return footprint_(stage, out, cap, fence);
  }

 private:
  uint32_t (*schedule_)(const void *, uint32_t, const uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t *, uint32_t *) = nullptr;
  uint32_t (*footprint_)(const void *, mk_access *, uint32_t, uint32_t *) = nullptr;
};

}  // namespace mk
