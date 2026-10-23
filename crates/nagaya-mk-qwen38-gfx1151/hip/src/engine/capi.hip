// SPDX-License-Identifier: MIT
// C ABI over the engine, for the Python harness (harness/engine.py) and the scoreboard.
#include <cstdio>
#include <cstring>
#include <vector>

#include "engine.h"

using namespace mk;

extern "C" {
// The tensor table comes from the Rust reader (tensor_table.h). Its tensor data must stay mapped for the engine's life.
void *mk_engine_open(const mk_tensor_table *t) { return new Engine(t); }
// An engine on another engine's weight arena (arena.h): path is /proc/<pid>/fd/<n>. Dies on an arena it can't trust.
void *mk_engine_attach(const mk_tensor_table *t, const char *path) { return new Engine(t, path); }
// Whether the arena at path is exactly what this build makes from these shards (engine.h Engine::check_arena): 0,
// or 1 with the first difference in why (cap bytes, NUL-terminated).
int mk_arena_check(const mk_tensor_table *t, const char *path, char *why, uint32_t cap) {
  const std::string w = Engine::check_arena(t, path);
  snprintf(why, cap, "%s", w.c_str());
  return w.empty() ? 0 : 1;
}
// The engine's weight arena: its fd in this process and its size.
void mk_engine_arena(void *e, int32_t *fd, uint64_t *bytes) {
  *fd = static_cast<Engine *>(e)->arena_fd();
  *bytes = static_cast<Engine *>(e)->arena_bytes();
}
void mk_engine_close(void *e) { delete static_cast<Engine *>(e); }
void *mk_ctx_new(void *e, uint32_t max_tokens) { return static_cast<Engine *>(e)->new_context(max_tokens); }
uint32_t mk_ctx_pooled_n(void *c) { return static_cast<Context *>(c)->pooled_n; }
void *mk_ctx_fork(void *e, void *c) { return static_cast<Engine *>(e)->fork(static_cast<Context *>(c)); }
void mk_ctx_free(void *e, void *c) { static_cast<Engine *>(e)->free_context(static_cast<Context *>(c)); }
void *mk_ctx_copy(void *e, void *c, uint32_t max_tokens) {
  return static_cast<Engine *>(e)->copy(static_cast<Context *>(c), max_tokens);
}
// A context, one part at a time (engine.h Engine::state_read): part 0 gdn_state, 1 gdn_conv, 2 ple_conv, 3 kv, 4 idx_keys,
// 5 ple_hist. mk_ctx_state_bytes is a fixed part's size in bytes (it dies for 3 and 4), mk_ctx_part_bytes any part's for
// n tokens; dst / src is that many bytes of host memory (a read of kv / idx_keys: the context's own n).
uint64_t mk_ctx_state_bytes(uint32_t part) { return Engine::state_bytes(part); }
uint64_t mk_ctx_part_bytes(uint32_t part, uint32_t n) { return Engine::state_bytes(part, n); }
void mk_ctx_state_read(void *e, void *c, uint32_t part, void *dst) {
  static_cast<Engine *>(e)->state_read(static_cast<Context *>(c), part, dst);
}
// Parts 0-2 into any context; part 5 (ple_hist) only into an empty context that owns its K/V (engine.h).
void mk_ctx_state_write(void *e, void *c, uint32_t part, const void *src) {
  static_cast<Engine *>(e)->state_write(static_cast<Context *>(c), part, src);
}
// Waking a parked context (docs/park.md; engine.h Engine::state_write_cells, restore_n): into an empty context that owns
// its K/V, n cells of part 3 (kv) or 4 (idx_keys), then its token count. Each dies on anything else.
void mk_ctx_cells_write(void *e, void *c, uint32_t part, uint32_t n, const void *src) {
  static_cast<Engine *>(e)->state_write_cells(static_cast<Context *>(c), part, n, src);
}
void mk_ctx_set_n(void *e, void *c, uint32_t n) { static_cast<Engine *>(e)->restore_n(static_cast<Context *>(c), n); }
// 1 if c owns its K/V and index keys (0: a fork, sharing its parent's).
int mk_ctx_owns_kv(void *c) { return static_cast<Context *>(c)->owns_kv ? 1 : 0; }
// The rounding rule gdn-bf16-v1 on c's recurrent state, in place (engine.h Engine::round_gdn_bf16; round.hip).
void mk_ctx_round_gdn_bf16(void *e, void *c) { static_cast<Engine *>(e)->round_gdn_bf16(static_cast<Context *>(c)); }
uint32_t mk_ctx_n(void *c) { return static_cast<Context *>(c)->n; }
// The attention cell mask (lab only; engine.h Engine::set_mask, src/ops/attn_mask.h): n ranges [lo[i], hi[i]); n = 0 clears.
void mk_ctx_set_mask(void *e, void *c, uint32_t n, const uint32_t *lo, const uint32_t *hi) {
  static_cast<Engine *>(e)->set_mask(static_cast<Context *>(c), n, lo, hi);
}
// c's mask into lo, hi (cap entries each); returns its range count (dies if cap is too small).
uint32_t mk_ctx_mask(void *c, uint32_t *lo, uint32_t *hi, uint32_t cap) {
  const AttnCellMask &m = static_cast<Context *>(c)->mask;
  if (m.n > cap) MK_DIE("mk_ctx_mask: %u ranges, room for %u", m.n, cap);
  for (uint32_t i = 0; i < m.n; ++i) {
    lo[i] = m.lo[i];
    hi[i] = m.hi[i];
  }
  return m.n;
}
// Lab and tests only: NaN into cells [lo, hi) of c's K/V (what bit 0) and index keys (bit 1) (engine.h Engine::poison_cells).
void mk_ctx_poison_cells(void *e, void *c, uint32_t lo, uint32_t hi, uint32_t what) {
  static_cast<Engine *>(e)->poison_cells(static_cast<Context *>(c), lo, hi, what);
}
// 1 if a pass extending c stopped on a spin-timeout (engine.h Context::poisoned): free it, nothing else.
int mk_ctx_poisoned(void *c) { return static_cast<Context *>(c)->poisoned ? 1 : 0; }
// Opt in (on = 1) to recoverable spin-timeouts (engine.h Engine::set_recover_spin_timeouts): mk_pass and mk_pass_segments
// then return -1 for a pass that stopped on one, with the reason in mk_last_error. Off by default, so a caller that never
// looks at the return value still dies on every failed pass, as before.
void mk_engine_recover_spin_timeouts(void *e, int on) { static_cast<Engine *>(e)->set_recover_spin_timeouts(on != 0); }
// The last pass's error into why (cap bytes, NUL-terminated); returns its length (0: the last pass was clean).
uint32_t mk_last_error(void *e, char *why, uint32_t cap) {
  const std::string &w = static_cast<Engine *>(e)->last_error();
  snprintf(why, cap, "%s", w.c_str());
  return uint32_t(w.size());
}
uint32_t mk_ctx_max(void *c) { return static_cast<Context *>(c)->max; }
// Runs a pass. logits: [vocab] or null; layers: [48][t][10240] or null; routes: [48][t][10] expert ids or null
// (layers and routes: unfused only).
// Returns the pass wall time in ms, or -1 if it stopped on a spin-timeout (only with mk_engine_recover_spin_timeouts on;
// nothing was written to logits, layers or routes, and c is poisoned).
double mk_pass(void *e, void *c, const int32_t *tokens, uint32_t t, int fused, float *logits, float *layers,
               int32_t *routes) {
  auto *E = static_cast<Engine *>(e);
  // The logits go from the device straight into the caller's buffer (one copy); po.logits stays empty.
  PassOut po = E->pass(static_cast<Context *>(c), tokens, t, logits != nullptr, layers != nullptr, fused != 0,
                       routes != nullptr, logits);
  if (!po.error.empty()) return -1.0;
  if (layers) memcpy(layers, po.layer_h.data(), sizeof(float) * po.layer_h.size());
  if (routes) memcpy(routes, po.routes.data(), sizeof(int32_t) * po.routes.size());
  return po.ms;
}
// An unfused pass that also hands back the GDN layers' gate projections (lab item 1, the memory horizon): ab
// [36 GDN layers][t][2][48] floats, the raw a then b the gated delta rule reads at each token (tiled value-head order),
// and params [36][2][48], the layers' ssm_a then dt_bias. The pass is mk_pass's, bit for bit (copies only). Returns the
// pass wall time in ms, or -1 on a recovered spin-timeout (unfused passes have none; kept for the one error convention).
double mk_pass_gdn_ab(void *e, void *c, const int32_t *tokens, uint32_t t, float *logits, float *ab, float *params) {
  auto *E = static_cast<Engine *>(e);
  PassOut po = E->pass(static_cast<Context *>(c), tokens, t, logits != nullptr, false, false, false, logits, true);
  if (!po.error.empty()) return -1.0;
  memcpy(ab, po.gdn_ab.data(), sizeof(float) * po.gdn_ab.size());
  memcpy(params, po.gdn_params.data(), sizeof(float) * po.gdn_params.size());
  return po.ms;
}
// Segments per pass (engine.h Segment): n contexts, lens[n] tokens each, tokens their concatenation (sum of lens).
static std::vector<Segment> segments_of(uint32_t n, void *const *ctxs, const uint32_t *lens, const int32_t *tokens) {
  std::vector<Segment> s(n);
  uint64_t r0 = 0;
  for (uint32_t i = 0; i < n; ++i) {
    s[i] = Segment{static_cast<Context *>(ctxs[i]), tokens + r0, lens[i]};
    r0 += lens[i];
  }
  return s;
}
// Whether the n segments can run as one pass (Engine::segments_problem): 0, or 1 with the first problem in why (cap
// bytes, NUL-terminated). Never dies; n past kMaxSegments reads no arrays.
int mk_segments_problem(void *e, uint32_t n, void *const *ctxs, const uint32_t *lens, char *why, uint32_t cap) {
  std::string w;
  if (n > kMaxSegments) {
    const Segment none{};
    w = static_cast<Engine *>(e)->segments_problem(&none, n);
  } else {
    static const int32_t no_tokens = 0;  // the check reads no tokens, only that a pass would have them
    std::vector<Segment> s(n);
    for (uint32_t i = 0; i < n; ++i) s[i] = Segment{static_cast<Context *>(ctxs[i]), &no_tokens, lens[i]};
    w = static_cast<Engine *>(e)->segments_problem(s.data(), n);
  }
  snprintf(why, cap, "%s", w.c_str());
  return w.empty() ? 0 : 1;
}
// A pass of n segments (dies on a segments problem). logits: [n][vocab] or null; hidden: [n][2560] or null, each
// segment's at its last token. Returns the pass wall time in ms, or -1 as mk_pass does (every context poisoned).
double mk_pass_segments(void *e, uint32_t n, void *const *ctxs, const uint32_t *lens, const int32_t *tokens, int fused,
                        float *logits, float *hidden) {
  if (n > kMaxSegments) MK_DIE("mk_pass_segments: %u segments past %u", n, kMaxSegments);
  const std::vector<Segment> s = segments_of(n, ctxs, lens, tokens);
  const PassOut po = static_cast<Engine *>(e)->pass_segments(s.data(), n, logits != nullptr, fused != 0, logits, hidden);
  return po.error.empty() ? po.ms : -1.0;
}
// The scratch pool's size check, no model (tests/engine/test_segments.py): asks one name for first, then second
// floats. Dies if the second ask is larger than the first; 0 otherwise.
int mk_scratch_selftest(uint64_t first, uint64_t second) {
  ScratchPool p;
  p.get("x", first);
  p.get("x", second);
  p.free_all();
  return 0;
}
// The host cost counters (rt.h HostStats), in this order: watchdog threads spawned, occupancy queries, events
// created, control-block memsets, control-block device syncs, logits device->host copies, logits that went through a PassOut vector (a second copy
// by the caller), fork states allocated and reused, (layer, block) pairs the pool stages write (goal 08 I2).
// Writes up to cap values; returns how many there are.
uint32_t mk_host_stats(uint64_t *out, uint32_t cap) {
  HostStats &s = host_stats();
  const uint64_t v[] = {s.watchdog_threads, s.occupancy_queries, s.events_created, s.cb_memsets,
                        s.cb_syncs,         s.logits_dtoh,       s.logits_staged,  s.fork_state_allocs,
                        s.fork_state_reuses, s.pool_blocks};
  const uint32_t n = sizeof v / sizeof v[0];
  for (uint32_t i = 0; i < n && i < cap; ++i) out[i] = v[i];
  return n;
}
// The last pass's final hidden state at its last token: kEmbd (2560) floats.
void mk_last_hidden(void *e, float *out) { static_cast<Engine *>(e)->last_hidden(out); }
uint64_t mk_dispatches(void *e) { return static_cast<Engine *>(e)->dispatches_last_pass(); }
// The last unfused pass's per-stage timings (MK_STAGE_TIMING set). Returns how many stages.
uint32_t mk_stage_times(void *e, uint32_t *op, float *ms, uint32_t cap) {
  auto *E = static_cast<Engine *>(e);
  const uint32_t n = uint32_t(E->stage_ms_.size());
  for (uint32_t i = 0; i < n && i < cap; ++i) {
    op[i] = E->stage_op_[i];
    ms[i] = E->stage_ms_[i];
  }
  return n;
}
// The last fused pass's per-stage timing from the event ring (MK_MEGA_EVENTS set; engine.h). Returns how many
// stages (0: none recorded).
// HW_ID1 of each workgroup in the last events pass (MK_MEGA_EVENTS). Returns the grid.
uint32_t mk_mega_hwid(void *e, uint32_t *out, uint32_t cap) {
  auto *E = static_cast<Engine *>(e);
  const uint32_t n = uint32_t(E->det_hwid_.size());
  if (n > cap) MK_DIE("mk_mega_hwid: %u workgroups, room for %u", n, cap);
  for (uint32_t i = 0; i < n; ++i) out[i] = E->det_hwid_[i];
  return n;
}
uint32_t mk_stage_detail(void *e, uint32_t *op, float *span, float *gap, float *work_mean, double *kernel_ms, uint32_t cap) {
  auto *E = static_cast<Engine *>(e);
  const uint32_t n = uint32_t(E->det_op_.size());
  if (n > cap) MK_DIE("mk_stage_detail: %u stages, room for %u", n, cap);
  for (uint32_t i = 0; i < n; ++i) {
    op[i] = E->det_op_[i];
    span[i] = E->det_span_[i];
    gap[i] = E->det_gap_[i];
    work_mean[i] = E->det_work_mean_[i];
  }
  *kernel_ms = E->det_kernel_ms_;
  return n;
}
// The last events pass's raw timestamps: begin and end ticks (100 MHz wall clock) per [stage][workgroup], for
// barrier studies. Returns stages x workgroups written.
uint64_t mk_stage_raw(void *e, uint64_t *begin, uint64_t *end, uint64_t cap) {
  auto *E = static_cast<Engine *>(e);
  const uint64_t n = E->det_b_.size();
  if (n > cap) MK_DIE("mk_stage_raw: %lu records, room for %lu", (unsigned long)n, (unsigned long)cap);
  std::copy(E->det_b_.begin(), E->det_b_.end(), begin);
  std::copy(E->det_e_.begin(), E->det_e_.end(), end);
  return n;
}
// The last fused pass's barriers: out[i] = 1 if a grid barrier follows stage i (stage_sched.h). Returns the stages.
uint32_t mk_stage_barriers(void *e, uint8_t *out, uint32_t cap) {
  auto *E = static_cast<Engine *>(e);
  const uint32_t n = uint32_t(E->det_bar_.size());
  if (n > cap) MK_DIE("mk_stage_barriers: %u stages, room for %u", n, cap);
  std::copy(E->det_bar_.begin(), E->det_bar_.end(), out);
  return n;
}
// Workgroups in the last fused pass's cooperative launch (0: none yet).
uint32_t mk_mega_grid(void *e) { return static_cast<Engine *>(e)->mega_grid_; }
// A kernel's resources, no engine needed: which 0 = k_mega, 1 = k_stage. VGPRs, scratch bytes per lane, static
// LDS bytes, and the occupancy API's co-resident workgroups of kMkBlock threads.
int mk_kernel_resources(int which, int *regs, int *scratch, int *lds, int *max_grid) {
  const void *k = which == 0 ? mega_kernel() : which == 1 ? stage_kernel() : nullptr;
  if (!k) return 1;
  hipFuncAttributes fa{};
  MK_CHECK(hipFuncGetAttributes(&fa, k));
  *regs = fa.numRegs;
  *scratch = int(fa.localSizeBytes);
  *lds = int(fa.sharedSizeBytes);
  *max_grid = max_coop_grid(k, kMkBlock, 0);
  return 0;
}
// Whether this library's trap sink is installed (the Engine constructor does it): 1 once the record exists.
int mk_trap_sink_installed() { return trap_sink_installed() ? 1 : 0; }
// A pass of one stage whose op no dispatcher knows, through k_mega (fused) or k_stage: the stage dispatcher's default
// traps (kTrapBadOp, stage.h) in the engine's own kernels, so tests/engine/test_trap_report.py can see the trap report
// with no model loaded. The caller installs the sink first (as the Engine does). Never returns: the trap kills the queue
// and the runtime aborts the process.
// k_stage's entry check, no model (tests/engine/test_stage_abort.py): one add stage over n floats (h = 1, a = 2, b = 3, so
// 6 where it ran) launched on 80 workgroups with the abort word preset or not. Writes h to h_out (n * 4 bytes, checked) and
// the control block's status and site. 0 on success.
int mk_stage_abort_selftest(int preset_abort, uint32_t site, float *h_out, uint64_t h_bytes, uint32_t *status, uint32_t *status_site) {
  constexpr uint32_t n = 1024;
  MK_CHECK_BYTES("mk_stage_abort_selftest", "h_out", h_bytes, n * sizeof(float));
  float *d;
  MK_CHECK(hipMalloc(&d, 3 * n * sizeof(float)));
  std::vector<float> v(3 * n);
  for (uint32_t i = 0; i < n; ++i) v[i] = 1.f, v[n + i] = 2.f, v[2 * n + i] = 3.f;
  MK_CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice));
  Stage s{};
  s.op = kStAdd;
  s.add = AddArgs{d, d + n, d + 2 * n, n};
  Stage *ds;
  MK_CHECK(hipMalloc(&ds, sizeof s));
  MK_CHECK(hipMemcpy(ds, &s, sizeof s, hipMemcpyHostToDevice));
  ControlBlock cb;
  if (preset_abort) cb.request_abort();
  Control *ctl = cb.device();
  const Stage *dss = ds;
  void *args[] = {&dss, &ctl, &site};
  MK_CHECK(hipLaunchKernel(stage_kernel(), dim3(80), dim3(kMkBlock), args, 0, 0));
  MK_CHECK(hipDeviceSynchronize());
  MK_CHECK(hipMemcpy(h_out, d, n * sizeof(float), hipMemcpyDeviceToHost));
  *status = cb.status();
  *status_site = cb.host()->status_site;
  MK_CHECK(hipFree(ds));
  MK_CHECK(hipFree(d));
  return 0;
}

int mk_trap_selftest(int fused) {
  Stage s{};
  s.op = 0xFFFFu;
  Stage *ds;
  MK_CHECK(hipMalloc(&ds, sizeof s));
  MK_CHECK(hipMemcpy(ds, &s, sizeof s, hipMemcpyHostToDevice));
  if (fused) {
    ControlBlock cb;
    Control *ctl = cb.device();
    Barrier *bar = cb.barrier();
    uint32_t n = 1, skip = UINT32_MAX;
    EventRingView ev{};
    const Stage *dss = ds;
    void *args[] = {&dss, &n, &ctl, &bar, &skip, &ev};
    launch_coop(mega_kernel(), 1, kMkBlock, args, 0, 0);
    MK_CHECK(hipDeviceSynchronize());
  } else {
    const Stage *dss = ds;
    ControlBlock cb;
    Control *ctl = cb.device();
    uint32_t site = 0;
    void *args[] = {&dss, &ctl, &site};
    MK_CHECK(hipLaunchKernel(stage_kernel(), dim3(1), dim3(kMkBlock), args, 0, 0));
    MK_CHECK(hipDeviceSynchronize());
  }
  return 0;
}
}
