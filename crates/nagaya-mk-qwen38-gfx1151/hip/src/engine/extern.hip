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
static_assert(kMaxSegments == 16, "spec.rs MAX_SEGMENTS");
static_assert(model::kVocab == 248320, "spec.rs VOCAB");
static_assert(model::kMaxPassTokens == 256, "spec.rs MAX_PASS_TOKENS");
static_assert(model::kMaxContext == 32768, "spec.rs MAX_CONTEXT");

extern "C" {
// Gives len bytes of text, with no NUL, to the binding's buffer out.
typedef void (*mk_text_sink)(void *out, const char *text, uint64_t len);

// The tensor table comes from the Rust reader (tensor_table.h). Its tensor data must stay mapped for the engine's life.
void *mk_engine_open(const mk_tensor_table *t) { return new Engine(t); }
uint64_t mk_engine_arena_bytes(void *e) { return static_cast<Engine *>(e)->arena_bytes(); }
void mk_engine_recover_spin_timeouts(void *e, bool on) { static_cast<Engine *>(e)->set_recover_spin_timeouts(on); }

void *mk_ctx_new(void *e, uint32_t max_tokens) { return static_cast<Engine *>(e)->new_context(max_tokens); }
void mk_ctx_free(void *e, void *c) { static_cast<Engine *>(e)->free_context(static_cast<Context *>(c)); }
uint32_t mk_ctx_n(void *c) { return static_cast<Context *>(c)->n; }
uint32_t mk_ctx_max(void *c) { return static_cast<Context *>(c)->max; }
bool mk_ctx_owns_kv(void *c) { return static_cast<Context *>(c)->owns_kv; }
bool mk_ctx_poisoned(void *c) { return static_cast<Context *>(c)->poisoned; }

// The parts of a context (engine.h Engine::state_read). Each Engine method stops the program on input it refuses.
uint64_t mk_ctx_state_bytes(uint32_t part) { return Engine::state_bytes(part); }
uint64_t mk_ctx_part_bytes(uint32_t part, uint32_t n) { return Engine::state_bytes(part, n); }
void mk_ctx_state_read(void *e, void *c, uint32_t part, void *dst) {
  static_cast<Engine *>(e)->state_read(static_cast<Context *>(c), part, dst);
}
void mk_ctx_state_write(void *e, void *c, uint32_t part, const void *src) {
  static_cast<Engine *>(e)->state_write(static_cast<Context *>(c), part, src);
}
void mk_ctx_cells_write(void *e, void *c, uint32_t part, uint32_t n, const void *src) {
  static_cast<Engine *>(e)->state_write_cells(static_cast<Context *>(c), part, n, src);
}
void mk_ctx_set_n(void *e, void *c, uint32_t n) { static_cast<Engine *>(e)->restore_n(static_cast<Context *>(c), n); }
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
// A pass of n segments (engine.h Engine::pass_segments), and its wall time in ms. The logits go straight into logits.
double mk_pass_segments(void *e, const Segment *s, uint32_t n, bool want_logits, bool fused, float *logits,
                        float *hidden) {
  return static_cast<Engine *>(e)->pass_segments(s, n, want_logits, fused, logits, hidden).ms;
}
}
