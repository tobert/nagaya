// SPDX-License-Identifier: MIT
// The engine: weights on the GPU, contexts, and passes built as stage lists (stage.h).
#pragma once
#include <hip/hip_runtime.h>

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arena.h"
#include "tensor_table.h"
#include "model.h"
#include "rt.h"
#include "stage.h"

namespace mk {
// A pass's quantized qmm stages run on the f16 WMMA path (S2) by default (Amy, 2026-09-27: "flip fast to
// default"); MK_FAST_QMM=0 builds the exact f32 engine, which check-m3 tests against the reference. Read at
// every pass build; any value but 0 or 1 stops the program.
// A bulky weight matrix (>= 8 MiB): the weight-policy experiment applies to these and to every expert matrix;
// routers, hyper-connection and other small matrices keep plain loads (Amy: "only on the bulky weights, leave
// routers and other small stuff alone").
inline uint32_t bulky_weight(uint64_t row_bytes, uint32_t rows) { return row_bytes * rows >= (8ull << 20) ? 1u : 0u; }

inline bool fast_qmm_enabled() {
  const char *e = getenv("MK_FAST_QMM");
  if (!e || !strcmp(e, "1")) return true;
  if (!strcmp(e, "0")) return false;
  MK_DIE("MK_FAST_QMM=%s: want 0 or 1", e);
}
}  // namespace mk

namespace mk {

struct Tensor {
  uint32_t type = 0;
  uint64_t ne[3] = {1, 1, 1};
  uint64_t row_bytes = 0;  // bytes per ne0 row
  const uint8_t *d = nullptr;
  const float *f32() const;
};

struct Layer {
  // hyper-connections: [0] attention side, [1] MoE side
  Tensor hc_norm[2], hc_down[2], hc_up[2], hc_inject[2];
  // gated DeltaNet
  Tensor qkv, zgate, alpha, beta, ssm_a, dt_bias, conv1d, ssm_norm, ssm_out;
  // attention
  Tensor q, k, v, o, q_norm, k_norm, idx_q, idx_k, idx_q_norm, idx_k_norm;
  // MoE
  Tensor gate_inp, gate_inp_shexp, gate_shexp, up_shexp, down_shexp, gate_exps, up_exps, down_exps;
  // PLE (layer 1)
  Tensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
};

// A context: everything a pass continues from.
struct Context {
  uint32_t n = 0;       // tokens so far
  uint32_t max = 0;     // K/V capacity
  float *gdn_state = nullptr;   // [36][48][128][128]
  float *gdn_conv = nullptr;    // [36][3][10240]  (per layer: the last 3 raw qkv inputs, oldest first, each a row of channels)
  float *ple_conv = nullptr;    // [9][10240]
  float *kv = nullptr;          // [12][max][2 (k,v)][2 heads][256]
  float *idx_keys = nullptr;    // [12][max][128]  raw index keys
  // The pooled block-key cache (goal 08 I2, docs/indexer.md): [12][max/4][128], block b's pooled, normed, roped key from the
  // raw keys of cells 4b..4b+3, in the score's lane layout. Derived state, never stored: blocks [0, pooled_n) are valid,
  // and pooled_n == n / 4 after any pass or restore_n (a pass dies at build if it finds pooled_n past n / 4).
  float *idx_pooled = nullptr;
  uint32_t pooled_n = 0;
  int32_t ple_hist[2] = {int32_t(model::kPleEos), int32_t(model::kPleEos)};  // last 2 token ids
  bool owns_kv = true;          // every context owns its K/V: the engine has no fork
  // A pass extending it stopped on a spin-timeout (Engine::set_recover_spin_timeouts): its recurrent state and its
  // cells >= n are half-written (n itself never advanced). Any later pass of it stops the program; free it.
  bool poisoned = false;
  // What has been written into an empty context to wake a parked one (Engine::state_write_cells, state_write of ple_hist),
  // for Engine::restore_n's check: the K/V and index-key cells written, and whether the PLE history was.
  uint32_t restored_kv = 0, restored_idx = 0;
  bool restored_ple = false;
};

// One segment of a pass (goal 04 S): t tokens continuing context c. A pass of several segments runs every weight stage
// once over all their rows (in the order given) and each state stage (PLE conv, GDN conv and core, attention prep,
// select and attend) once per segment: the stage a pass of that segment alone would run, on its rows and its context.
// So each segment gets the bits a pass of it alone gives (tests/engine/test_segments.py).
struct Segment {
  Context *c;
  const int32_t *tokens;
  uint32_t t;
};
// The most segments in one pass: the head's logits scratch is kMaxSegments x kVocab floats (15.9 MB).
constexpr uint32_t kMaxSegments = 16;

// The pass's named device buffers: one per name, allocated (zeroed) at its first ask and kept for the engine's life.
// A later ask of a name for more floats than its buffer holds dies: stages built on it would run past its end.
struct ScratchPool {
  struct Buf {
    float *p;
    size_t n;
  };
  std::unordered_map<std::string, Buf> bufs;
  float *get(const char *name, size_t n);
  void free_all();
};

struct PassOut {
  double ms = 0;
  std::string error;  // non-empty: the pass stopped on a spin-timeout (set_recover_spin_timeouts); no outputs, contexts poisoned
};

// The megakernel's entry point (mega.hip; its own translation unit, see engine.hip).
const void *mega_kernel();
// Re-lay out an IQ3_S matrix (rows x k, GGUF row-major at src) into read-once tiles at dst (tiles.h); synchronizes.
void relayout_iq3s_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
// The same for a Q8_0 matrix (the dense projections); the tiles take the same bytes.
void relayout_q8_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
// The same for a Q6_K matrix (the output head).
void relayout_q6_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
// The same for an IQ4_NL matrix (the expert down matrices).
void relayout_iq4_tiles(const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k);
// A tiled matrix is addressed by 16-row group from its first byte, so a matmul must cover all of its rows.
inline void check_whole_tiled(const Tensor &w, uint32_t n) {
  if ((w.type == q::Q8_0_TILES || w.type == q::IQ3_S_TILES || w.type == q::Q6_K_TILES || w.type == q::IQ4_NL_TILES) && w.ne[1] * w.ne[2] != n)
    MK_DIE("qmm over %u rows of a tiled matrix with %lu", n, (unsigned long)(w.ne[1] * w.ne[2]));
}
// MK_TILES=0 keeps the GGUF layout; unset or 1 converts the IQ3_S expert gate/up, the IQ4_NL and Q8_0 expert down
// matrices, the Q8_0 dense matrices and the Q6_K output head to read-once tiles at load.
inline bool tiles_enabled() {
  const char *e = getenv("MK_TILES");
  if (!e || !strcmp(e, "1")) return true;
  if (!strcmp(e, "0")) return false;
  MK_DIE("MK_TILES=%s: want 0 or 1", e);
}
const void *stage_kernel();
WatchdogConfig engine_watchdog_config();  // 10 s, or MK_WATCHDOG_DEADLINE_S (a test knob)
// A tensor's read-once tile layout (tiles.h), or none.
enum class Tile { kNone, kIq3s, kQ8, kQ6, kIq4 };

class Engine {
 public:
  // Loads the weights into a new sealed arena (arena.h): the pages the kernels read. The tensor table comes from the
  // Rust reader (tensor_table.h).
  explicit Engine(const mk_tensor_table *table);
  ~Engine();
  Context *new_context(uint32_t max_tokens);
  void free_context(Context *c);
  // A context, one part at a time, to and from host memory (exact bytes; parking, docs/park.md, builds on it). The fixed
  // state: part 0 gdn_state f32 [36][48][128][128], 1 gdn_conv f32 [36][3][10240], 2 ple_conv f32 [9][10240]. The parts
  // that depend on the token count n: 3 kv f32 [12][n][2][2][256] and 4 idx_keys f32 [12][n][128] (cells 0..n-1,
  // compacted from the capacity-strided layout), 5 ple_hist int32 [2]. state_bytes(part, n) is a part's size for n tokens
  // (state_bytes(part) for parts 0-2 and 5; it dies for 3 and 4). All die on a poisoned context or an unknown part.
  // state_write: parts 0-2 into any context (the host side is not inspected); part 5 (ids in 0..kVocab-1) only into an
  // empty context (n == 0); parts 3-4 die (state_write_cells).
  // state_write_cells: part 3 or 4, n cells (1..capacity) of src into cells 0..n-1, only into an empty context.
  // restore_n: sets the token count of such a context to n (1..capacity), once the K/V and index keys were written with
  // exactly n cells and the PLE history was written; it is then a context of n tokens like any other.
  static uint64_t state_bytes(uint32_t part, uint32_t n);
  static uint64_t state_bytes(uint32_t part);
  void state_read(const Context *c, uint32_t part, void *dst);
  void state_write(Context *c, uint32_t part, const void *src);
  void state_write_cells(Context *c, uint32_t part, uint32_t n, const void *src);
  void restore_n(Context *c, uint32_t n);
  // The rounding rule gdn-bf16-v1 (docs/park.md; round.hip): c's GDN recurrent state rounded in place to bf16 (round to
  // nearest, ties to even, on the f32 bit pattern; kept as f32 with the low 16 bits zero). Only the recurrent state moves.
  // A non-finite element, or a finite one that rounds to infinity, stops the program naming the first one (layer, head,
  // index, value); so does a poisoned context.
  void round_gdn_bf16(Context *c);
  // "" if the segments can run as one pass, else the first problem: 1..kMaxSegments segments, each of t >= 1 tokens,
  // kMaxPassTokens in all, each fitting its context's capacity, and no two sharing a K/V or index-key buffer (one
  // context twice: they would write the same cells).
  std::string segments_problem(const Segment *s, uint32_t n) const;
  // One fused pass of n segments (dies on segments_problem), advancing each context. logits_out: [n][kVocab], each
  // segment's logits at its last token (want_logits), or null when want_logits is false.
  PassOut pass_segments(const Segment *s, uint32_t n, bool want_logits, float *logits_out);
  // Off (the default): a pass that stops early for any reason stops the program. On: a fused pass that stops on a
  // spin-timeout (a grid barrier's bounded wait ran out, as under another tenant's GPU contention; PLAN.md Problems
  // 2026-10-01 11:47) returns instead, with last_error() set, and poisons every context it was extending
  // (Context::poisoned); the next pass starts from ControlBlock's full reset. A device trap, the watchdog's abort, or a
  // spin-timeout the watchdog also asked to stop still stop the program.
  void set_recover_spin_timeouts(bool on) { recover_spin_timeouts_ = on; }
  // The last pass's error ("" after a clean pass).
  const std::string &last_error() const { return last_error_; }
  uint64_t arena_bytes() const { return arena_bytes_; }

 private:
  struct Planned {  // one uploaded tensor: its GGUF index, final layout and place in the arena
    size_t gguf = 0;
    std::string name;
    Tensor t;  // final type, ne, row_bytes (d is set once the arena is mapped)
    Tile tile = Tile::kNone;
    uint32_t rows = 0, k = 0;
    uint64_t src_bytes = 0, bytes = 0, offset = 0;
  };
  static std::vector<Planned> plan(const TensorTable &table, uint64_t *arena_bytes);
  static arena::Header arena_header(const std::vector<Planned> &P, uint64_t arena_bytes, const std::vector<std::string> &shards);
  static arena::Entry arena_entry(const Planned &p);
  void create_arena(const std::vector<Planned> &P);
  void wire();
  Tensor get(const std::string &name) const;
  PassOut run_pass(const Segment *s, uint32_t n, bool want_logits, float *logits_out);
  // t: the pass's tokens, the sum of the segments'.
  std::vector<Stage> build(const Segment *seg, uint32_t ns, uint32_t t, bool want_logits, std::vector<uint32_t> *layer_ends);
  std::vector<Stage> build_stages(const Segment *seg, uint32_t ns, uint32_t t, bool want_logits, std::vector<uint32_t> *layer_ends);
  void ple_rows(const Context *c, const int32_t *tokens, uint32_t t, float *out) const;
  // The model's three blocks, appended as stages: mixed [T, E] in, out [T, E] out. The state stages, one per segment.
  void add_gdn(std::vector<Stage> &S, const Segment *seg, uint32_t ns, uint32_t il, uint32_t t, const float *mixed, float *out);
  void add_attn(std::vector<Stage> &S, const Segment *seg, uint32_t ns, uint32_t il, uint32_t t, const float *mixed, float *out);
  void add_moe(std::vector<Stage> &S, uint32_t il, uint32_t t, const float *mixed, float *out);
  // One pool stage per QSA layer for blocks [b0, c->n / 4) of c's cache (restore_n's one-off launch; blocks.hip).
  std::vector<Stage> pool_stages(const Context *c, uint32_t b0);
  // Runs stages one launch each under the control block and the watchdog, outside any pass.
  void run_stages_now(const std::vector<Stage> &S);

  TensorTable table_;
  std::vector<std::string> shards_;
  std::unordered_map<std::string, Tensor> w_;
  int arena_fd_ = -1;
  uint8_t *arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
  Layer layers_[model::kLayers];
  Tensor tok_embd_, out_hc_norm_, out_hc_down_, out_hc_up_, output_;
  const GgufTensor *ple_ = nullptr;  // the PLE table, left mapped (ple_rows gathers from it)
  // pass scratch, sized for kMaxPassTokens
  ScratchPool scratch_;
  int32_t *d_tokens_ = nullptr;
  uint32_t *d_head_rows_ = nullptr;  // [kMaxSegments]: a pass of several segments' last rows (the head's x_row)
  Stage *d_stages_ = nullptr;
  size_t stage_cap_ = 0;
  ControlBlock cb_;
  // One watchdog thread for the engine's life, begun and finished around each launch sequence (goal 03 C). One 10 s
  // deadline serves both: measured unfused passes (256 tokens, empty or 3584-token context, 2026-09-30, quiet GPU)
  // take 0.62-0.74 s, fused ones 0.48-0.58 s, so 10 s is >10x the longest and still far under the gfx ring timeout and the
  // 29 minutes an unwatched unfused pass once held the GPU.
  Watchdog wd_{cb_, engine_watchdog_config(), /*armed=*/false};
  hipEvent_t launched_ev_ = nullptr;  // recorded after each launch (the watchdog's "kernel still runs" query)
  bool recover_spin_timeouts_ = false;
  std::string last_error_;  // the last pass's error, "" if it was clean
  PassOut failed_pass(const Segment *seg, uint32_t ns, const char *why, double ms);
  float *scratch(const char *name, size_t n);
};

}  // namespace mk
