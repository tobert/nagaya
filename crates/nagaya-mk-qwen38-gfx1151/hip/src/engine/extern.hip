// SPDX-License-Identifier: MIT
// The extern layer: the C calls of the engine that the Rust binding makes (src/binding.rs, milestone 15 step 4).
// Each call takes C types and calls one Engine method or reads one field. It checks nothing: the checks, the
// refusals and the result rules are in Rust. Add a call here only with its declaration in src/binding.rs.
#include <cstddef>
#include <cstdint>

#include "engine.h"

using namespace mk;

// The layout of a segment, which src/binding.rs mirrors as MkSegment, and the numbers that size the binding's
// buffers (src/spec.rs). The Rust test src/layout.rs reads these lines. Keep each assert on one line.
static_assert(sizeof(Segment) == 24, "Segment");
static_assert(offsetof(Segment, c) == 0, "Segment.c, MkSegment.context in Rust");
static_assert(offsetof(Segment, tokens) == 8, "Segment.tokens");
static_assert(offsetof(Segment, t) == 16, "Segment.t");
static_assert(sizeof(bool) == 1, "bool, which Rust's bool is");
static_assert(sizeof(Context) == 72, "Context, MkContext in Rust");
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
static_assert(kMaxSegments == 16, "spec.rs MAX_SEGMENTS");
static_assert(model::kVocab == 248320, "spec.rs VOCAB");
static_assert(model::kMaxPassTokens == 256, "spec.rs MAX_PASS_TOKENS");
static_assert(model::kMaxContext == 32768, "spec.rs MAX_CONTEXT");

extern "C" {
// Gives len bytes of text, with no NUL, to the binding's buffer out.
typedef void (*mk_text_sink)(void *out, const char *text, uint64_t len);

// The weight table comes from the Rust host (weight_table.h). The arena must stay mapped for the engine's life.
void *mk_engine_open(const mk_weight_table *t) { return new Engine(t); }
// Re-lays out a rows x k matrix at src into read-once tiles at dst (tile: 1 IQ3_S, 2 Q8_0, 3 Q6_K, 4 IQ4_NL; tiles.h),
// and waits for the device. src/arena.rs names the tile ids.
void mk_relayout(uint32_t tile, const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k) {
  switch (tile) {
    case 1: relayout_iq3s_tiles(src, dst, rows, k); break;
    case 2: relayout_q8_tiles(src, dst, rows, k); break;
    case 3: relayout_q6_tiles(src, dst, rows, k); break;
    case 4: relayout_iq4_tiles(src, dst, rows, k); break;
    default: MK_DIE("mk_relayout: tile %u: want 1 to 4", tile);
  }
  MK_CHECK(hipDeviceSynchronize());
}
void mk_engine_recover_spin_timeouts(void *e, bool on) { static_cast<Engine *>(e)->set_recover_spin_timeouts(on); }

// A context is the Rust host's (src/context.rs). This call pools its complete blocks (engine.h Engine::pool_context).
void mk_ctx_pool(void *e, const void *c) { static_cast<Engine *>(e)->pool_context(static_cast<const Context *>(c)); }
void mk_ctx_round_gdn_bf16(void *e, void *c) { static_cast<Engine *>(e)->round_gdn_bf16(static_cast<Context *>(c)); }

// The last pass's error, "" after a clean pass.
void mk_last_error(void *e, mk_text_sink put, void *out) {
  const std::string &w = static_cast<Engine *>(e)->last_error();
  put(out, w.data(), w.size());
}
// The first problem of n segments as one pass, or "" (engine.h Engine::segments_problem).
void mk_segments_problem(void *e, const Segment *s, uint32_t n, mk_text_sink put, void *out) {
  const std::string w = static_cast<Engine *>(e)->segments_problem(s, n);
  put(out, w.data(), w.size());
}
// A fused pass of n segments (engine.h Engine::pass_segments), and its wall time in ms. The logits go straight into
// logits. ple holds the PLE rows of the pass's tokens.
double mk_pass_segments(void *e, const Segment *s, uint32_t n, bool want_logits, float *logits, const float *ple) {
  return static_cast<Engine *>(e)->pass_segments(s, n, want_logits, logits, ple).ms;
}
}
