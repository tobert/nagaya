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

// A context: everything a pass continues from. A snapshot is a context nobody passes into; a
// read runs on a scratch copy of its fixed state, and writes K/V only in cells >= n.
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
  // and pooled_n == n / 4 after any pass, copy or restore_n (a pass dies at build if it finds pooled_n past n / 4). A fork
  // shares the buffer like idx_keys and holds its parent's pooled_n; its passes write blocks >= that, the region the
  // parent treats as scratch, as its prep writes cells >= n.
  float *idx_pooled = nullptr;
  uint32_t pooled_n = 0;
  int32_t ple_hist[2] = {int32_t(model::kPleEos), int32_t(model::kPleEos)};  // last 2 token ids
  bool owns_kv = true;          // a fork shares its snapshot's K/V and index keys
  // A pass extending it stopped on a spin-timeout (Engine::set_recover_spin_timeouts): its recurrent state and its
  // cells >= n are half-written (n itself never advanced). Any later pass, fork or copy of it stops the program; free it.
  bool poisoned = false;
  // The attention cell mask (lab only; src/ops/attn_mask.h, docs/lab/mask.md): cell ranges no attention read uses. Empty
  // (n = 0) for every context the service makes. A fork and a copy inherit it. Non-empty: unfused passes only (a fused pass
  // on it stops the program). Set through Engine::set_mask, which checks its rules.
  AttnCellMask mask{};
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
  std::vector<float> logits;                 // [vocab] at the last token, if asked
  std::vector<float> layer_h;                // [layers][T][10240], if asked (debug / M3 check)
  std::vector<int32_t> routes;               // [layers][T][10] expert ids, if asked (unfused; M3 attribution)
  std::vector<float> gdn_ab;                 // [36 GDN layers][T][2][48] the gate projections a (0) and b (1), if asked (unfused; lab)
  std::vector<float> gdn_params;             // [36][2][48] the GDN layers' ssm_a (0) and dt_bias (1), if asked, same order
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
  // Loads the weights into a new sealed arena (arena.h), or, given attach (a path like /proc/<pid>/fd/<n>), maps
  // another engine's arena read-only after checking it against this build (attach_arena); either way the same pages
  // the kernels read. The tensor table comes from the Rust reader (tensor_table.h).
  explicit Engine(const mk_tensor_table *table, const char *attach = nullptr);
  ~Engine();
  Context *new_context(uint32_t max_tokens);
  void free_context(Context *c);
  // Copies c's fixed state into a new context sharing c's K/V buffers (cells < c->n are
  // never written by passes on the copy, since they only write cells >= their own n). The fixed state's buffers come
  // from the engine's free list when it has a set (free_context of a fork returns them there), else from hipMalloc: a
  // read forks and frees per request, and steady reads make no allocation (PLAN.md Problems 2026-10-05: the live
  // service faulted on a fork's freshly mapped gdn_conv after 13 h of one hipMalloc/hipFree set per read). Every byte
  // of a reused set is overwritten by the copy, so reuse can't change bits.
  Context *fork(const Context *c);
  // A deep copy of c into a new context of capacity max_tokens (>= c->n) that owns all of its buffers: its K/V
  // cells and index keys are copied (re-strided: a layer's stride is the capacity), so nothing is shared.
  Context *copy(const Context *c, uint32_t max_tokens);
  // A context, one part at a time, to and from host memory (exact bytes; the lab's checkpoint experiments and parking,
  // docs/park.md, build on it). The fixed state: part 0 gdn_state f32 [36][48][128][128], 1 gdn_conv f32 [36][3][10240],
  // 2 ple_conv f32 [9][10240]. The parts that depend on the token count n: 3 kv f32 [12][n][2][2][256] and 4 idx_keys f32
  // [12][n][128] (cells 0..n-1, compacted from the capacity-strided layout), 5 ple_hist int32 [2]. state_bytes(part, n)
  // is a part's size for n tokens (state_bytes(part) for parts 0-2 and 5; it dies for 3 and 4). All die on a poisoned
  // context or an unknown part.
  // state_read: any part of any context; a fork's kv / idx_keys are its parent's cells below the fork's n.
  // state_write: parts 0-2 into any context (the host side is not inspected: harness/engine.py refuses non-finite values);
  // part 5 (ids in 0..kVocab-1) only into an empty context (n == 0) that owns its K/V; parts 3-4 die (state_write_cells).
  // state_write_cells: part 3 or 4, n cells (1..capacity) of src into cells 0..n-1, only into an empty context that owns
  // its K/V. restore_n: sets the token count of such a context to n (1..capacity), once the K/V and index keys were
  // written with exactly n cells and the PLE history was written; it is then a context of n tokens like any other.
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
  // The attention cell mask of c (Context::mask; lab only): n ranges [lo[i], hi[i]), n <= kAttnMaskMax, each lo < hi,
  // ascending and disjoint (lo[i] >= hi[i - 1]), every hi <= c->n. n = 0 clears it. Dies on anything else, or on a
  // poisoned context.
  void set_mask(Context *c, uint32_t n, const uint32_t *lo, const uint32_t *hi);
  // Lab and tests only: overwrites cells [lo, hi) (hi <= c->n) of c's K/V (what bit 0) and raw index keys (bit 1) in all 12
  // attention layers with NaN, so a test can show that nothing reads them. c must own its K/V (a fork shares its parent's).
  void poison_cells(Context *c, uint32_t lo, uint32_t hi, uint32_t what);
  // Runs one pass of T tokens on c (advancing it). fused: one cooperative launch (M4) instead
  // of one launch per stage (M3).
  // logits_out: where the logits (kVocab floats, want_logits) go; the copy from the device lands there directly and
  // PassOut::logits stays empty. Null: the logits come back in PassOut::logits.
  PassOut pass(Context *c, const int32_t *tokens, uint32_t t, bool want_logits, bool want_layers, bool fused,
               bool want_routes = false, float *logits_out = nullptr, bool want_gdn_ab = false);
  // "" if the segments can run as one pass, else the first problem: 1..kMaxSegments segments, each of t >= 1 tokens,
  // kMaxPassTokens in all, each fitting its context's capacity, and no two sharing a K/V or index-key buffer (forks of
  // one context, a fork and its parent or a fork of it, one context twice: they would write the same cells).
  std::string segments_problem(const Segment *s, uint32_t n) const;
  // A pass of n segments (dies on segments_problem), advancing each context. logits_out: [n][kVocab], each segment's
  // logits at its last token (want_logits); hidden_out: [n][kEmbd], each segment's final hidden state at its last token
  // (last_hidden's value for a pass of it alone), or null.
  PassOut pass_segments(const Segment *s, uint32_t n, bool want_logits, bool fused, float *logits_out, float *hidden_out);
  // Off (the default): a pass that stops early for any reason stops the program. On: a fused pass that stops on a
  // spin-timeout (a grid barrier's bounded wait ran out, as under another tenant's GPU contention; PLAN.md Problems
  // 2026-10-01 11:47) returns instead, with PassOut::error set (and last_error()), and poisons every context it was
  // extending (Context::poisoned); the next pass starts from ControlBlock's full reset. A device trap, the watchdog's
  // abort, or a spin-timeout the watchdog also asked to stop still stop the program.
  void set_recover_spin_timeouts(bool on) { recover_spin_timeouts_ = on; }
  // The last pass's error ("" after a clean pass).
  const std::string &last_error() const { return last_error_; }
  uint64_t dispatches_last_pass() const { return dispatches_; }
  // With MK_STAGE_TIMING set, an unfused pass records each stage's (op, layer, ms) here.
  std::vector<float> stage_ms_;
  std::vector<uint32_t> stage_op_;
  // With MK_MEGA_EVENTS set, a fused pass's per-stage timing from the event ring (DESIGN.md "The event ring"):
  // span (first workgroup begin to last end), gap (that end to the next stage's first begin: the barrier)
  // and the mean workgroup's own begin-to-end, in ms; kernel_ms is the first begin to the last end.
  std::vector<uint32_t> det_op_;
  std::vector<float> det_span_, det_gap_, det_work_mean_;
  std::vector<uint32_t> det_hwid_;  // per workgroup of the last events pass: HW_ID1 at stage 0
  double det_kernel_ms_ = 0;
  std::vector<uint64_t> det_b_, det_e_;  // [stage][workgroup] begin and end ticks (kWallHz) of the last events pass
  uint32_t mega_grid_ = 0;  // workgroups in the last fused pass's launch
  // The last pass's final hidden state at its last token (the final mixer's output: the output head's input), kEmbd
  // floats into out; dies if no pass has run, or if the last pass had several segments (pass_segments returns theirs).
  void last_hidden(float *out);
  std::vector<uint8_t> det_bar_;  // per stage of the last fused pass: 1 = a grid barrier after it (stage_sched.h)
  const Layer &layer(uint32_t il) const { return layers_[il]; }
  int arena_fd() const { return arena_fd_; }
  uint64_t arena_bytes() const { return arena_bytes_; }
  // Whether the arena at path is exactly what this build makes from these GGUF files: "" if so, else the first
  // difference (Engine::arena_problem). Needs the GPU for the spot check's conversions; builds no engine.
  static std::string check_arena(const mk_tensor_table *table, const char *path);

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
  static void expected_bytes(const TensorTable &table, const Planned &p, std::vector<uint8_t> &out);
  static std::string arena_problem(const TensorTable &table, const std::vector<std::string> &shards, const std::vector<Planned> &P,
                                   uint64_t arena_bytes, int fd, const char *path, size_t *types);
  void create_arena(const std::vector<Planned> &P);
  void attach_arena(const char *path, const std::vector<Planned> &P);
  void wire();
  Tensor get(const std::string &name) const;
  PassOut run_pass(const Segment *s, uint32_t n, bool want_logits, bool want_layers, bool fused, bool want_routes,
                   float *logits_out, float *hidden_out, bool want_gdn_ab = false);
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
  // Runs stages one launch each under the unfused pass's rails (control block, watchdog), outside any pass.
  void run_stages_now(const std::vector<Stage> &S);

  TensorTable table_;
  std::vector<std::string> shards_;
  std::unordered_map<std::string, Tensor> w_;
  int arena_fd_ = -1;
  uint8_t *arena_ = nullptr;  // read-only when attached
  uint64_t arena_bytes_ = 0;
  Layer layers_[model::kLayers];
  Tensor tok_embd_, out_hc_norm_, out_hc_down_, out_hc_up_, output_;
  const GgufTensor *ple_ = nullptr;  // the PLE table, left mapped (ple_rows gathers from it)
  // pass scratch, sized for kMaxPassTokens
  ScratchPool scratch_;
  // A fork's fixed state (gdn_state, gdn_conv, ple_conv), kept for the next fork (Engine::fork).
  struct ForkState {
    float *gdn_state, *gdn_conv, *ple_conv;
  };
  std::vector<ForkState> fork_free_;  // freed only with the engine; HostStats counts the sets allocated and reused
  int32_t *d_tokens_ = nullptr;
  uint32_t *d_head_rows_ = nullptr;  // [kMaxSegments]: a pass of several segments' last rows (the head's x_row)
  Stage *d_stages_ = nullptr;
  size_t stage_cap_ = 0;
  ControlBlock cb_;
  // One watchdog thread for the engine's life, begun and finished around each launch sequence, fused or unfused (goal 03 C).
  // One 10 s deadline serves both: measured unfused passes (256 tokens, empty or 3584-token context, 2026-09-30, quiet GPU)
  // take 0.62-0.74 s, fused ones 0.48-0.58 s, so 10 s is >10x the longest and still far under the gfx ring timeout and the
  // 29 minutes an unwatched unfused pass once held the GPU. MK_STAGE_TIMING's per-stage syncs add little.
  Watchdog wd_{cb_, engine_watchdog_config(), /*armed=*/false};
  hipEvent_t launched_ev_ = nullptr;  // recorded after each fused launch (the watchdog's "kernel still runs" query)
  std::unique_ptr<EventRing> ring_;  // the fused pass's event ring, made on first use
  uint64_t dispatches_ = 0;
  uint32_t last_t_ = 0;  // tokens in the last pass (last_hidden)
  uint32_t last_segments_ = 0;  // segments in the last pass
  bool recover_spin_timeouts_ = false;
  std::string last_error_;  // the last pass's error, "" if it was clean
  PassOut failed_pass(const Segment *seg, uint32_t ns, const char *why, double ms);
  float *scratch(const char *name, size_t n);
};

}  // namespace mk
