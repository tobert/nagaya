// SPDX-License-Identifier: MIT
// The extern layer: the C calls of the engine that the Rust binding makes (src/binding.rs, milestone 15 step 4).
// Each call takes C types and calls one Engine method or reads one field. It checks nothing: the checks, the
// refusals and the result rules are in Rust. Add a call here only with its declaration in src/binding.rs.
#include <cstddef>
#include <cstdint>

#include "engine.h"

using namespace qwen38;

// The layouts that src/context.rs and src/binding.rs mirror, and the model numbers that size the host's buffers
// (src/spec.rs). The Rust test src/layout.rs reads these lines. Keep each assert on one line.
static_assert(sizeof(bool) == 1, "bool, which Rust's bool is");
static_assert(sizeof(Context) == 72, "Context, Qwen38Context in Rust");
static_assert(offsetof(Context, n) == 0, "Context.n");
static_assert(offsetof(Context, max) == 4, "Context.max");
static_assert(offsetof(Context, gdn_state) == 8, "Context.gdn_state");
static_assert(offsetof(Context, gdn_conv) == 16, "Context.gdn_conv");
static_assert(offsetof(Context, ple_conv) == 24, "Context.ple_conv");
static_assert(offsetof(Context, kv) == 32, "Context.kv");
static_assert(offsetof(Context, idx_keys) == 40, "Context.idx_keys");
static_assert(offsetof(Context, idx_pooled) == 48, "Context.idx_pooled");
static_assert(offsetof(Context, pooled_n) == 56, "Context.pooled_n");
static_assert(offsetof(Context, ple_hist) == 60, "Context.ple_hist");
static_assert(offsetof(Context, poisoned) == 68, "Context.poisoned");
static_assert(model::kVocab == 248320, "spec.rs VOCAB");
static_assert(model::kMaxPassTokens == 256, "spec.rs MAX_PASS_TOKENS");
static_assert(model::kMaxContext == 32768, "spec.rs MAX_CONTEXT");

static_assert(sizeof(LaunchResult) == 16, "LaunchResult, Qwen38Launch in Rust");
static_assert(offsetof(LaunchResult, status) == 0, "LaunchResult.status");
static_assert(offsetof(LaunchResult, site) == 4, "LaunchResult.site");
static_assert(offsetof(LaunchResult, wg) == 8, "LaunchResult.wg");
static_assert(offsetof(LaunchResult, watchdog_stopped) == 12, "LaunchResult.watchdog_stopped");

extern "C" {
// The engine's runtime, with the watchdog's deadline in seconds.
void *qwen38_engine_open(double watchdog_deadline_s) { return new Engine(watchdog_deadline_s); }
// Re-lays out a rows x k matrix at src into read-once tiles at dst (tile: 1 IQ3_S, 2 Q8_0, 3 Q6_K, 4 IQ4_NL; tiles.h),
// and waits for the device. src/arena.rs names the tile ids.
void qwen38_relayout(uint32_t tile, const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k) {
  switch (tile) {
    case 1: relayout_iq3s_tiles(src, dst, rows, k); break;
    case 2: relayout_q8_tiles(src, dst, rows, k); break;
    case 3: relayout_q6_tiles(src, dst, rows, k); break;
    case 4: relayout_iq4_tiles(src, dst, rows, k); break;
    default: HIP_DIE("qwen38_relayout: tile %u: want 1 to 4", tile);
  }
  HIP_CHECK(hipDeviceSynchronize());
}
void qwen38_ctx_round_gdn_bf16(void *e, void *c) { static_cast<Engine *>(e)->round_gdn_bf16(static_cast<Context *>(c)); }
// The launches of the stages that the Rust host built (engine.h Engine::launch_fused, Engine::launch_each).
void qwen38_launch_fused(void *e, const Stage *ds, uint32_t n, int grid, uint32_t skip, LaunchResult *out) {
  *out = static_cast<Engine *>(e)->launch_fused(ds, n, grid, skip);
}
void qwen38_launch_each(void *e, const Stage *ds, uint32_t n, LaunchResult *out) {
  *out = static_cast<Engine *>(e)->launch_each(ds, n);
}
// The most co-resident workgroups of the megakernel (rt.h max_coop_grid).
int qwen38_max_coop_grid() { return max_coop_grid(mega_kernel(), kBlockThreads, 0); }
// The name of a control block status (rt.h status_name).
const char *qwen38_status_name(uint32_t status) { return status_name(status); }
}
