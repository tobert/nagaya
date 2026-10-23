// SPDX-License-Identifier: MIT
// k_stage's register budget (tests/engine/test_mega_resources.py: grid >= 80 and no scratch, for k_mega and k_stage
// alike) has no room for the expert item's one-step lookahead (qmmh.h: 16 VGPRs of prefetched activations), so this TU,
// the one that compiles k_stage, runs the ops without it. Both paths give the same bits. Before any ops include.
#define MK_EXPERT_LOOKAHEAD 0
#include "engine.h"
#include "stage_layout.h"
#include "device.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <climits>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>



namespace mk {

using namespace model;

const float *Tensor::f32() const {
  if (type != q::F32) MK_DIE("tensor is type %u, not F32", type);
  return reinterpret_cast<const float *>(d);
}

// ---- kernels: one stage per launch (M3), or the whole list in one cooperative launch (M4) ----
// s: __restrict__ as in k_mega (mega.hip): nothing writes the stage list during the pass.
// ctl, site: the unfused pass's watchdog (Engine::pass). Every workgroup looks at the control words once on entry and, if the
// host asked to stop or something already failed, returns without running its stage: after an abort the rest of the pass's
// launches drain in microseconds. The first to see the abort word records kAborted at `site` (the stage's index). One
// thread reads and the workgroup shares the answer, so every thread agrees (run_stage has __syncthreads in it). This stops a
// long or runaway sequence of stages BETWEEN stages; a stage that hangs inside an op can't be stopped (ops have no polls).
__global__ void __launch_bounds__(kMkBlock) k_stage(const Stage *__restrict__ s, Control *ctl, uint32_t site) {
  __shared__ uint32_t stop;
  if (mk_tid() == 0) {
    uint32_t v = 0;
    if (ld_acquire_sys(&ctl->abort) != 0) {
      set_status(ctl, kAborted, site);
      v = 1;
    } else if (ld_acquire_sys(&ctl->status) != 0) {
      v = 1;
    }
    stop = v;
  }
  __syncthreads();
  if (stop) return;
  // The lab's masked selection (attn_mask.h) runs here only, never in k_mega. The engine builds no such stage.
  if (s->op == kStAttnSelectMasked) {
    op_attn_select_masked(s->aselm, blockIdx.x, gridDim.x);
    return;
  }
  run_stage(*s, blockIdx.x, gridDim.x);
}

// The engine's watchdog: 10 s a pass (the longest pass is a bit over a second: PLAN.md), and MK_WATCHDOG_DEADLINE_S for a test
// that needs the deadline to fire inside a pass. The deadline must be positive and finite, or the engine dies.
WatchdogConfig engine_watchdog_config() {
  WatchdogConfig c{.deadline_s = 10.0};
  if (const char *e = getenv("MK_WATCHDOG_DEADLINE_S")) {
    char *end = nullptr;
    const double v = strtod(e, &end);
    if (end == e || *end != 0 || !(v > 0.0) || v > 3600.0) MK_DIE("MK_WATCHDOG_DEADLINE_S=%s: want seconds in (0, 3600]", e);
    c.deadline_s = v;
  }
  return c;
}

const void *stage_kernel() { return reinterpret_cast<const void *>(&k_stage); }

// k_mega lives alone in mega.hip. With two kernels in one translation unit that both reach the
// ops' LDS-using inline functions, the compiler's module-LDS lowering produced a k_stage that
// never finished, even for an empty stage (run 2, 2026-09-26; bisected with tests/engine/probe_stage.hip).
// Rule: k_stage and k_mega each get their own TU, and no other kernel may share one with them.

constexpr uint32_t kUnfusedGrid = 1024;


// ---- weights: the arena that the Rust host loaded (src/arena.rs) ----
Engine::Engine(const mk_weight_table *table) : table_(table) {
  trap_sink_install();  // a device trap names itself on stderr before the runtime aborts (trap.h)
  if (!table->weights || table->n_weights == 0) MK_DIE("the weight table holds no weights");
  for (uint64_t i = 0; i < table->n_weights; ++i) {
    const mk_weight &w = table->weights[i];
    if (!w.name || !w.d) MK_DIE("weight %lu of the table has no name or no data", (unsigned long)i);
    Tensor t;
    t.type = w.type;
    for (int j = 0; j < 3; ++j) t.ne[j] = w.ne[j];
    t.row_bytes = w.row_bytes;
    t.d = w.d;
    if (!w_.emplace(w.name, t).second) MK_DIE("the weight table holds %s twice", w.name);
  }
  wire();
}

Engine::~Engine() {
  scratch_.free_all();
  (void)hipFree(d_tokens_);
  (void)hipFree(d_head_rows_);
  (void)hipHostFree(d_stages_);
  if (launched_ev_) (void)hipEventDestroy(launched_ev_);
}

void Engine::wire() {
  tok_embd_ = get("token_embd.weight");
  out_hc_norm_ = get("output_hc_norm.weight");
  out_hc_down_ = get("output_hc_down.weight");
  out_hc_up_ = get("output_hc_up.weight");
  output_ = get("output.weight");
  for (uint32_t il = 0; il < kLayers; ++il) {
    Layer &L = layers_[il];
    const std::string b = "blk." + std::to_string(il) + ".";
    const char *side[2] = {"attn", "ffn"};
    for (int s = 0; s < 2; ++s) {
      L.hc_norm[s] = get(b + "hc_" + side[s] + "_norm.weight");
      L.hc_down[s] = get(b + "hc_" + side[s] + "_down.weight");
      L.hc_up[s] = get(b + "hc_" + side[s] + "_up.weight");
      L.hc_inject[s] = get(b + "hc_" + side[s] + "_inject.weight");
    }
    if (is_attn_layer(il)) {
      L.q = get(b + "attn_q.weight");
      L.k = get(b + "attn_k.weight");
      L.v = get(b + "attn_v.weight");
      L.o = get(b + "attn_output.weight");
      L.q_norm = get(b + "attn_q_norm.weight");
      L.k_norm = get(b + "attn_k_norm.weight");
      L.idx_q = get(b + "indexer.q_proj.weight");
      L.idx_k = get(b + "indexer.k_proj.weight");
      L.idx_q_norm = get(b + "indexer.q_norm.weight");
      L.idx_k_norm = get(b + "indexer.k_norm.weight");
    } else {
      L.qkv = get(b + "attn_qkv.weight");
      L.zgate = get(b + "attn_gate.weight");
      L.alpha = get(b + "ssm_alpha.weight");
      L.beta = get(b + "ssm_beta.weight");
      L.ssm_a = get(b + "ssm_a");
      L.dt_bias = get(b + "ssm_dt.bias");
      L.conv1d = get(b + "ssm_conv1d.weight");
      L.ssm_norm = get(b + "ssm_norm.weight");
      L.ssm_out = get(b + "ssm_out.weight");
    }
    L.gate_inp = get(b + "ffn_gate_inp.weight");
    L.gate_inp_shexp = get(b + "ffn_gate_inp_shexp.weight");
    L.gate_shexp = get(b + "ffn_gate_shexp.weight");
    L.up_shexp = get(b + "ffn_up_shexp.weight");
    L.down_shexp = get(b + "ffn_down_shexp.weight");
    L.gate_exps = get(b + "ffn_gate_exps.weight");
    L.up_exps = get(b + "ffn_up_exps.weight");
    L.down_exps = get(b + "ffn_down_exps.weight");
    if (il == kPleLayer) {
      L.ple_key = get(b + "ple_key.weight");
      L.ple_value = get(b + "ple_value.weight");
      L.ple_norm_key = get(b + "ple_norm_key.weight");
      L.ple_norm_query = get(b + "ple_norm_query.weight");
      L.ple_norm_conv = get(b + "ple_norm_conv.weight");
      L.ple_conv1d = get(b + "ple_conv1d.weight");
    }
  }
  MK_CHECK(hipMalloc(&d_tokens_, sizeof(int32_t) * kMaxPassTokens));
  MK_CHECK(hipMalloc(&d_head_rows_, sizeof(uint32_t) * kMaxSegments));
}

Tensor Engine::get(const std::string &name) const {
  auto it = w_.find(name);
  if (it == w_.end()) MK_DIE("no tensor %s", name.c_str());
  return it->second;
}

float *ScratchPool::get(const char *name, size_t n) {
  auto it = bufs.find(name);
  if (it != bufs.end()) {
    if (n > it->second.n) MK_DIE("scratch %s: asked for %zu floats, its buffer holds %zu", name, n, it->second.n);
    return it->second.p;
  }
  float *p;
  MK_CHECK(hipMalloc(&p, sizeof(float) * n));
  MK_CHECK(hipMemset(p, 0, sizeof(float) * n));
  bufs[name] = Buf{p, n};
  return p;
}

void ScratchPool::free_all() {
  for (auto &kv : bufs) (void)hipFree(kv.second.p);
  bufs.clear();
}

float *Engine::scratch(const char *name, size_t n) { return scratch_.get(name, n); }

// ---- contexts: the Rust host makes them (src/context.rs) ----
void Engine::pool_context(const Context *c) {
  if (c->n / kIdxRatio) run_stages_now(pool_stages(c, 0));
}

void Engine::run_stages_now(const std::vector<Stage> &S) {
  if (S.size() > stage_cap_) {
    if (d_stages_) MK_CHECK(hipHostFree(d_stages_));
    stage_cap_ = S.size();
    MK_CHECK(hipHostMalloc(&d_stages_, sizeof(Stage) * stage_cap_, hipHostMallocDefault));
  }
  MK_CHECK(hipMemcpy(d_stages_, S.data(), sizeof(Stage) * S.size(), hipMemcpyHostToDevice));
  cb_.begin_pass();
  Control *ctl = cb_.device();
  if (!launched_ev_) {
    MK_CHECK(hipEventCreate(&launched_ev_));
    ++host_stats().events_created;
  }
  wd_.begin();
  for (uint32_t i = 0; i < S.size(); ++i) {
    k_stage<<<kUnfusedGrid, kMkBlock>>>(d_stages_ + i, ctl, i);
    MK_CHECK(hipGetLastError());
    MK_CHECK(hipEventRecord(launched_ev_, 0));
    if (i == 0) wd_.arm(launched_ev_);
  }
  MK_CHECK(hipDeviceSynchronize());
  wd_.finish();
  cb_.end_pass();
  if (cb_.status() != kOk)
    MK_DIE("stage launch outside a pass stopped: %s at stage %u of %zu (wg %u)", status_name(cb_.status()), cb_.host()->status_site,
           S.size(), cb_.host()->status_wg);
}

// ---- pass building ----
namespace {
Stage st(uint32_t op, uint32_t layer) {
  Stage s;
  memset(&s, 0, sizeof s);
  s.op = op;
  s.layer = layer;
  return s;
}
Stage qmm_stage(uint32_t layer, const Tensor &w, uint32_t n, uint32_t k, const float *x, uint32_t ldx, uint32_t t, float *y,
                uint32_t ldy) {
  if (w.ne[0] != k) MK_DIE("qmm: weight has K=%lu, want %u", (unsigned long)w.ne[0], k);
  check_whole_tiled(w, n);
  Stage s = st(qmm_op(w.type, fast_qmm_enabled()), layer);
  s.qmm = QmmArgs{w.type, w.d, w.row_bytes, n, k, t, x, ldx, nullptr, y, ldy, nullptr, bulky_weight(w.row_bytes, n),
                  /*ksplit: whole K until the split-K pass below*/ 0, /*part*/ nullptr, kXinPlain, /*xks*/ 0,
                  /*xp*/ nullptr, /*xp2*/ nullptr, /*xscale*/ 0.f};
  return s;
}
Stage rms_stage(uint32_t layer, const float *x, float *y, const Tensor &g, uint32_t t, uint32_t groups, uint32_t d,
                bool per_group) {
  Stage s = st(kStRms, layer);
  s.rms = RmsArgs{x, groups * d, y, groups * d, g.f32(), t, groups, d, per_group ? 1u : 0u, kRmsEps};
  return s;
}
}  // namespace

// Split-K (qmmh.h, PLAN.md Speed S5): every f16 qmm stage on a narrow matrix gets qmmh_ksplit(n, k) K slices,
// writing partials to `part`, and a kStQmmKsum stage right after it. One partial buffer serves them all: a split
// stage and its sum are adjacent, with a barrier (fused) or a launch (unfused) between any two stages. The split
// depends on the matrix shape alone, never on T, so a token's bits don't depend on the pass it's in.
// own_ranges (the barrier schedule, MK_ELIDE): each split gets its own partial range, dealt round the buffer, so two
// splits and their sums can share a barrier-free segment (src/schedule.rs); ranges still in use when the buffer wraps
// are hazards the schedule keeps in order. The values, and so the bits, don't depend on where the partials live.
static void split_k(std::vector<Stage> &S, std::vector<uint32_t> *layer_ends, float *part, size_t part_floats, bool own_ranges) {
  std::vector<Stage> out;
  out.reserve(S.size() + S.size() / 4);
  std::vector<uint32_t> at(S.size() + 1);
  size_t next = 0;  // own_ranges: the next free float in part
  for (size_t i = 0; i < S.size(); ++i) {
    at[i] = uint32_t(out.size());
    Stage s = S[i];
    const uint32_t ks = s.op == kStQmmH && !s.qmm.x_row && !s.qmm.y_row ? qmmh_ksplit(s.qmm.n, s.qmm.k) : 1;
    if (ks > 1) {
      if (size_t(ks) * s.qmm.t * s.qmm.n > part_floats)
        MK_DIE("split-K partials for %ux%u x %u tokens (ksplit %u) exceed the buffer", s.qmm.n, s.qmm.k, s.qmm.t, ks);
      if (ks > kMaxKsplit) MK_DIE("split-K: ksplit %u for %ux%u > the sum's %u", ks, s.qmm.n, s.qmm.k, kMaxKsplit);
      s.qmm.ksplit = ks;
      s.qmm.part = part;
      if (own_ranges) {
        const size_t need = (size_t(ks) * s.qmm.t * s.qmm.n + 63) / 64 * 64;  // whole 256-byte lines: float4 loads
        if (next + need > part_floats) next = 0;
        s.qmm.part = part + next;
        next += need;
      }
      out.push_back(s);
      s.op = kStQmmKsum;
    }
    out.push_back(s);
  }
  at[S.size()] = uint32_t(out.size());
  if (layer_ends)
    for (uint32_t &e : *layer_ends) e = at[e];
  S.swap(out);
}

// Fused staging (PLAN.md Speed D3; qmmh.h qmmh_x4). At decode most stages are a few microseconds of work plus a grid
// barrier, so the split-K sums and the activations between two f16 matmuls move into the consumer's staging, which
// computes the same values from the producer's partials:
//   A (the hyper-connections): QmmH(split, y=D) Ksum SiluScale(D -> X) QmmH(x=X)  ->  QmmH(split) QmmH(xin=silu-sum)
//   B (the shared expert):     QmmH gate(split, y=G) Ksum QmmH up(split, y=U) Ksum Swiglu(G, U -> X) QmmH(x=X)
//                              ->  QmmH gate QmmH up(partials in part2) QmmH(xin=swiglu-sum)
// Only for t <= kFuseMaxT: every consumer item re-sums the partials it stages, trivial for a few tokens and ~1 GB of
// L2 reads at 256. The skipped buffers (D, X, G, U) are read by nothing else (hc_pre, add_moe). The values, and so the
// bits, don't depend on the choice (tests/ops/test_qmmh_xin.py; tools/bits_snapshot.py with MK_FUSE=0 and 1).
constexpr uint32_t kFuseMaxT = 16;
static void fuse_prologues(std::vector<Stage> &S, std::vector<uint32_t> *layer_ends, float *part2, size_t part_floats) {
  std::vector<Stage> out;
  out.reserve(S.size());
  std::vector<uint32_t> at(S.size() + 1);
  auto qh = [&](size_t i) { return i < S.size() && S[i].op == kStQmmH; };
  auto split = [&](size_t i) { return qh(i) && S[i].qmm.ksplit > 1 && i + 1 < S.size() && S[i + 1].op == kStQmmKsum; };
  auto plain_consumer = [&](size_t i, const float *x, uint32_t k) {
    return qh(i) && S[i].qmm.x == x && S[i].qmm.k == k && S[i].qmm.ldx == k && !S[i].qmm.x_row && S[i].qmm.ksplit <= 1 &&
           S[i].qmm.xin == kXinPlain;
  };
  size_t i = 0;
  // A layer end (the old index of a layer's first stage) inside a fused group would be remapped to the consumer and
  // capture a layer's output one stage off (kaibo review): none does today; stop if one ever does.
  auto no_layer_end_inside = [&](size_t b, size_t last) {
    if (layer_ends)
      for (uint32_t e : *layer_ends)
        if (e > b && e <= last) MK_DIE("fused staging: layer end %u inside the fused stages %zu..%zu", e, b, last);
  };
  auto keep = [&](size_t j, const Stage &s) {
    for (; i <= j; ++i) at[i] = uint32_t(out.size());
    out.push_back(s);
  };
  while (i < S.size()) {
    // A
    if (split(i) && i + 3 < S.size() && S[i + 2].op == kStSiluScale && S[i + 2].silu.x == S[i].qmm.y &&
        S[i + 2].silu.n == S[i].qmm.t * S[i].qmm.n && plain_consumer(i + 3, S[i + 2].silu.y, S[i].qmm.n)) {
      Stage c = S[i + 3];
      c.qmm.xin = kXinSiluSum;
      c.qmm.xks = S[i].qmm.ksplit;
      c.qmm.xp = S[i].qmm.part;
      c.qmm.xscale = S[i + 2].silu.scale;
      c.qmm.x = nullptr;
      const size_t b = i;  // keep() advances i
      no_layer_end_inside(b, b + 3);
      keep(b, S[b]);
      keep(b + 3, c);
      continue;
    }
    // B
    if (split(i) && split(i + 2) && i + 5 < S.size() && S[i + 4].op == kStSwiglu) {
      const QmmArgs &g = S[i].qmm, &u = S[i + 2].qmm;
      const SwigluArgs &w = S[i + 4].swiglu;
      if (g.n == u.n && g.t == u.t && g.ksplit == u.ksplit && w.g == g.y && w.u == u.y && w.ldg == g.n && w.ldu == u.n &&
          w.rows == g.t && w.cols == g.n && w.ldo == g.n && plain_consumer(i + 5, w.out, g.n)) {
        if (size_t(u.ksplit) * u.t * u.n > part_floats) MK_DIE("fused staging: up partials exceed part2");
        Stage gs = S[i], us = S[i + 2], c = S[i + 5];
        us.qmm.part = part2;
        c.qmm.xin = kXinSwigluSum;
        c.qmm.xks = g.ksplit;
        c.qmm.xp = g.part;
        c.qmm.xp2 = part2;
        c.qmm.x = nullptr;
        const size_t b = i;  // keep() advances i
        no_layer_end_inside(b, b + 5);
        keep(b, gs);
        keep(b + 2, us);
        keep(b + 5, c);
        continue;
      }
    }
    keep(i, S[i]);
  }
  at[S.size()] = uint32_t(out.size());
  if (layer_ends)
    for (uint32_t &e : *layer_ends) e = at[e];
  S.swap(out);
}

// Fused hc_post + next hc_pre norm (PLAN.md Speed D11 (fuse-hc lane); small.h op_hc_combine_rms): a kStHcCombine followed directly by the
// kStRms over the same h becomes one stage, one item per (token, stream group). Only for t <= kFuseMaxT: the fused stage
// has t * 4 items where the combine alone has t * 40. Both outputs (h, xn) are bit-identical to the two stages
// (tests/ops/test_hc_fused.py). A layer end sits between the combine and its rms (the layer's output is h after the
// combine): it moves to after the fused stage, so the capture still sees the combined h.
static void fuse_hc(std::vector<Stage> &S, std::vector<uint32_t> *layer_ends) {
  std::vector<Stage> out;
  out.reserve(S.size());
  std::vector<uint32_t> at(S.size() + 1);
  for (size_t i = 0; i < S.size(); ++i) {
    at[i] = uint32_t(out.size());
    const bool pair = S[i].op == kStHcCombine && i + 1 < S.size() && S[i + 1].op == kStRms;
    if (pair) {
      const HcCombineArgs &c = S[i].comb;
      const RmsArgs &r = S[i + 1].rms;
      if (r.x == c.h && r.t == c.t && r.n_groups == 4 && r.d == c.e && r.ldx == 4 * c.e && r.gamma_per_group == 1) {
        Stage f = S[i];
        f.op = kStHcCombineRms;
        f.sub = 0;
        f.combrms.c = c;
        f.combrms.r = r;
        // An old layer end at i + 1 (between the pair) maps to the stage after the fused one, like an end at i + 2.
        out.push_back(f);
        ++i;
        at[i] = uint32_t(out.size());
        continue;
      }
    }
    out.push_back(S[i]);
  }
  at[S.size()] = uint32_t(out.size());
  if (layer_ends)
    for (uint32_t &e : *layer_ends) e = at[e];
  S.swap(out);
}

// The footprint of a stage from the Rust schedule (src/schedule.rs), through the table's call, without a K/V reach:
// its accesses, or a fence.
struct StageFootprint {
  bool fence = false;
  std::vector<mk_access> acc;
};
static StageFootprint rust_footprint(const ScheduleCalls &table, const Stage &s) {
  constexpr uint32_t kCap = 16;
  mk_access acc[kCap];
  uint32_t fence = 0;
  const uint32_t n = table.footprint(&s, acc, kCap, &fence);
  StageFootprint f;
  f.fence = fence != 0;
  f.acc.assign(acc, acc + n);
  return f;
}

// hc up + hc_mix as one stage (docs/narrow.md N1; qmmh.h QmmArgs::yout). hc_mix's only input from its layer's work is `up`,
// which the hc up matmul writes a stage or two earlier (the inject matmul sits between them). The matmul moves to hc_mix's
// slot and its epilogue does hc_mix: mixed and inject come out of the one stage, `up` is never written, and one level
// (a grid barrier, a stage-argument fetch, a load round trip) goes. The inject matmul stays before: it writes z, which the
// fused stage reads. Every float op is the same ones in the same order (small.h hc_mix_elem; the matmul's accumulators are
// bit-for-bit the values it used to store), so the bits don't change. Any t: the epilogue handles every token tile.
static void fuse_hc_mix(std::vector<Stage> &S, std::vector<uint32_t> *layer_ends, const ScheduleCalls &table) {
  std::vector<Stage> out;
  out.reserve(S.size());
  std::vector<uint32_t> at(S.size() + 1);
  std::vector<int64_t> producer_of(S.size(), -1);  // hc_mix stage -> its up matmul's index
  std::vector<char> dropped(S.size(), 0);
  auto overlaps = [](uintptr_t alo, uintptr_t ahi, uintptr_t blo, uintptr_t bhi) { return alo < bhi && blo < ahi; };
  for (size_t m = 0; m < S.size(); ++m) {
    if (S[m].op != kStHcMix) continue;
    const HcMixArgs &x = S[m].mix;
    int64_t j = int64_t(m) - 1;
    for (; j >= 0; --j)
      if ((S[j].op == kStQmmH || S[j].op == kStQmm) && S[j].qmm.y == x.up) break;
    if (j < 0) MK_DIE("fuse_hc_mix: no matmul writes the up of hc_mix stage %zu", m);
    const QmmArgs &u = S[j].qmm;
    if (S[j].op != kStQmmH) MK_DIE("fuse_hc_mix: hc_mix stage %zu: its up matmul is an exact qmm (op %u); MK_FUSE_HCMIX=0 for that", m, S[j].op);
    if (S[j].layer != S[m].layer || u.n != 4 * x.e || u.ldy != u.n || u.t != x.t || u.ksplit > 1 || u.x_row || u.y_row ||
        u.yout != kYoutPlain || (u.type != q::Q8_0 && u.type != q::Q8_0_TILES) || x.e % 16)
      MK_DIE("fuse_hc_mix: hc_mix stage %zu: the up matmul at %lld is not a plain Q8_0 4E-row projection of its layer", m, (long long)j);
    if (layer_ends)
      for (uint32_t e : *layer_ends)
        if (e >= j && e <= m) MK_DIE("fuse_hc_mix: layer end %u inside the stages %lld..%zu", e, (long long)j, m);
    // The matmul runs later now: nothing between may write what it reads, and nothing between may touch `up` (which
    // is no longer written) or write xn (which the epilogue reads at the new slot).
    const StageFootprint fu = rust_footprint(table, S[j]);
    const uintptr_t xlo = uintptr_t(x.xn), xhi = xlo + sizeof(float) * uint64_t(x.t) * 4 * x.e, ulo = uintptr_t(x.up),
                    uhi = ulo + sizeof(float) * uint64_t(x.t) * 4 * x.e;
    for (size_t k = size_t(j) + 1; k < m; ++k) {
      const StageFootprint fk = rust_footprint(table, S[k]);
      if (fk.fence) MK_DIE("fuse_hc_mix: a fence stage (op %u) between the up matmul and its hc_mix", S[k].op);
      for (const mk_access &a : fk.acc) {
        if (overlaps(a.lo, a.hi, ulo, uhi)) MK_DIE("fuse_hc_mix: stage %zu (op %u) touches up", k, S[k].op);
        if (a.write && overlaps(a.lo, a.hi, xlo, xhi)) MK_DIE("fuse_hc_mix: stage %zu (op %u) writes xn", k, S[k].op);
        if (a.write)
          for (const mk_access &r : fu.acc)
            if (!r.write && overlaps(a.lo, a.hi, r.lo, r.hi)) MK_DIE("fuse_hc_mix: stage %zu (op %u) writes the up matmul's input", k, S[k].op);
      }
    }
    // Nothing after the hc_mix reads `up` either (it is no longer written).
    for (size_t k = m + 1; k < S.size(); ++k) {
      const StageFootprint fk = rust_footprint(table, S[k]);
      if (fk.fence) continue;  // fences have no derived ranges; none reads this layer's up by construction
      bool rewritten = false, reads = false;
      for (const mk_access &a : fk.acc)
        if (overlaps(a.lo, a.hi, ulo, uhi)) (a.write ? rewritten : reads) = true;
      if (reads && !rewritten) MK_DIE("fuse_hc_mix: stage %zu (op %u) reads up after its hc_mix %zu", k, S[k].op, m);
      if (rewritten) break;  // the next layer's up matmul: later reads see its data, not this one's
    }
    producer_of[m] = j;
    dropped[size_t(j)] = 1;
  }
  for (size_t i = 0; i < S.size(); ++i) {
    at[i] = uint32_t(out.size());
    if (dropped[i]) continue;
    if (S[i].op == kStHcMix && producer_of[i] >= 0) {
      const HcMixArgs &x = S[i].mix;
      Stage f = S[size_t(producer_of[i])];
      f.qmm.yout = kYoutHcMix;
      f.qmm.mix_xn = x.xn;
      f.qmm.mix_z = x.z;
      f.qmm.mix_out = x.mixed;
      f.qmm.mix_inject = x.z ? x.inject : nullptr;
      f.qmm.mix_e = x.e;
      f.qmm.y = nullptr;
      f.qmm.ldy = 0;
      out.push_back(f);
      continue;
    }
    out.push_back(S[i]);
  }
  at[S.size()] = uint32_t(out.size());
  if (layer_ends)
    for (uint32_t &e : *layer_ends) e = at[e];
  S.swap(out);
}

// MK_FUSE=0 turns the fused staging (and the fused hc_post + norm) off (A/B and bit checks); unset or 1 fuses at decode.
static bool fuse_enabled() {
  const char *e = getenv("MK_FUSE");
  if (!e || !strcmp(e, "1")) return true;
  if (!strcmp(e, "0")) return false;
  MK_DIE("MK_FUSE=%s: want 0 or 1", e);
}

// MK_FUSE_HC=0 turns off only the combine + rms fusion (fuse_hc), for A/B and its own bit check; needs MK_FUSE on.
static bool fuse_hc_enabled() {
  const char *e = getenv("MK_FUSE_HC");
  if (!e || !strcmp(e, "1")) return true;
  if (!strcmp(e, "0")) return false;
  MK_DIE("MK_FUSE_HC=%s: want 0 or 1", e);
}

// MK_FUSE_HCMIX=0 turns off only the hc up + hc_mix fusion (fuse_hc_mix), for A/B and its own bit check; needs MK_FUSE on.
static bool fuse_hcmix_enabled() {
  const char *e = getenv("MK_FUSE_HCMIX");
  if (!e || !strcmp(e, "1")) return true;
  if (!strcmp(e, "0")) return false;
  MK_DIE("MK_FUSE_HCMIX=%s: want 0 or 1", e);
}

// MK_ELIDE=0 turns the barrier schedule off (src/schedule.rs; A/B and bit checks): every stage in build order with a
// grid barrier after it, and one partial range for every split. Unset or 1 schedules.
static bool elide_enabled() {
  const char *e = getenv("MK_ELIDE");
  if (!e || !strcmp(e, "1")) return true;
  if (!strcmp(e, "0")) return false;
  MK_DIE("MK_ELIDE=%s: want 0 or 1", e);
}

// MK_TAIL=0 turns off the barrier tail (docs/narrow.md N3, stage.h kSchedTail): a T = 1 fused combine + rms then runs as a
// normal stage with its own barrier. Unset or 1 marks it; it needs the schedule (MK_ELIDE) and the fused path, and the
// unfused path ignores the mark.
static bool tail_enabled() {
  const char *e = getenv("MK_TAIL");
  if (!e || !strcmp(e, "1")) return true;
  if (!strcmp(e, "0")) return false;
  MK_DIE("MK_TAIL=%s: want 0 or 1", e);
}

// The Rust schedule (src/schedule.rs), through the table's call, on S: the order and the sched word of each stage.
// Returns the grid barriers. A plan that is not a permutation of the stages stops the program.
static uint32_t apply_schedule(const ScheduleCalls &table, std::vector<Stage> &S, const std::vector<uint32_t> &ends, bool elide,
                               bool tail) {
  const uint32_t n = uint32_t(S.size());
  std::vector<uint32_t> order(n), sched(n);
  const uint32_t barriers =
      table.schedule(S.data(), n, ends.data(), uint32_t(ends.size()), elide, tail, order.data(), sched.data());
  std::vector<Stage> out;
  out.reserve(n);
  std::vector<char> seen(n, 0);
  for (uint32_t k = 0; k < n; ++k) {
    if (order[k] >= n || seen[order[k]]) MK_DIE("schedule: the plan names stage %u twice or past %u stages", order[k], n);
    seen[order[k]] = 1;
    Stage s = S[order[k]];
    s.sched = sched[k];
    out.push_back(s);
  }
  S.swap(out);
  return barriers;
}

std::vector<Stage> Engine::build(const Segment *seg, uint32_t ns, uint32_t t, bool want_logits, std::vector<uint32_t> *layer_ends) {
  std::vector<Stage> S = build_stages(seg, ns, t, want_logits, layer_ends);
  constexpr size_t kPartFloats = size_t(16) * kMaxPassTokens * 640;  // the largest split: 16 x 320 = 8 x 640 = 5120
  const bool elide = elide_enabled();
  split_k(S, layer_ends, scratch("qmm_part", kPartFloats), kPartFloats, elide);
  if (t <= kFuseMaxT && fuse_enabled()) {
    fuse_prologues(S, layer_ends, scratch("qmm_part2", kPartFloats), kPartFloats);
    if (fuse_hc_enabled()) fuse_hc(S, layer_ends);
  }
  // The fold lives in the f16 matmul's epilogue (qmmh.h); the exact engine (MK_FAST_QMM=0) has only kStQmm stages, so its
  // passes keep their hc_mix stages. With the f16 path on, a kStQmm up producer is a surprise and fuse_hc_mix stops.
  if (fuse_enabled() && fuse_hcmix_enabled() && fast_qmm_enabled()) fuse_hc_mix(S, layer_ends, table_);
  // The schedule reorders within layers only (windows start at the layer ends), so the unfused path's per-layer
  // captures stay put; the unfused path runs the same order, one launch per stage.
  const std::vector<uint32_t> no_ends;
  const std::vector<uint32_t> &ends = layer_ends ? *layer_ends : no_ends;
  const bool tail = tail_enabled();
  apply_schedule(table_, S, ends, elide, tail);
  return S;
}

std::vector<Stage> Engine::build_stages(const Segment *seg, uint32_t ns, uint32_t t, bool want_logits,
                                        std::vector<uint32_t> *layer_ends) {
  std::vector<Stage> S;
  float *e = scratch("e", size_t(kMaxPassTokens) * kEmbd);
  float *h = scratch("h", size_t(kMaxPassTokens) * kHcDim);
  float *xn = scratch("xn", size_t(kMaxPassTokens) * kHcDim);
  float *down = scratch("down", size_t(kMaxPassTokens) * kHcRank);
  float *act = scratch("act", size_t(kMaxPassTokens) * kHcRank);
  float *up = scratch("up", size_t(kMaxPassTokens) * kHcDim);
  float *z4 = scratch("z4", size_t(kMaxPassTokens) * 4);
  float *inj = scratch("inj", size_t(kMaxPassTokens) * 4);
  float *mixed = scratch("mixed", size_t(kMaxPassTokens) * kEmbd);
  float *out = scratch("out", size_t(kMaxPassTokens) * kEmbd);

  Stage s = st(kStEmbed, 0);
  s.embed = EmbedArgs{tok_embd_.d, tok_embd_.row_bytes, kEmbd, d_tokens_, t, e};
  S.push_back(s);
  s = st(kStRepeat4, 0);
  s.rep = Repeat4Args{e, h, t, kEmbd};
  S.push_back(s);

  auto hc_pre = [&](uint32_t il, int side, bool inject) {
    const Layer *L = il < kLayers ? &layers_[il] : nullptr;  // il == kLayers: the final mixer
    const Tensor &gn = L ? L->hc_norm[side] : out_hc_norm_;
    const Tensor &wd = L ? L->hc_down[side] : out_hc_down_;
    const Tensor &wu = L ? L->hc_up[side] : out_hc_up_;
    S.push_back(rms_stage(il, h, xn, gn, t, 4, kEmbd, true));
    S.push_back(qmm_stage(il, wd, kHcRank, kHcDim, xn, kHcDim, t, down, kHcRank));
    Stage a = st(kStSiluScale, il);
    a.silu = SiluScaleArgs{down, act, t * kHcRank, 1.0f / float(kHc)};
    S.push_back(a);
    S.push_back(qmm_stage(il, wu, kHcDim, kHcRank, act, kHcRank, t, up, kHcDim));
    if (inject) S.push_back(qmm_stage(il, L->hc_inject[side], 4, kHcDim, xn, kHcDim, t, z4, 4));
    Stage m = st(kStHcMix, il);
    m.mix = HcMixArgs{xn, up, mixed, inject ? z4 : nullptr, inj, t, kEmbd};
    S.push_back(m);
  };
  auto hc_post = [&](uint32_t il) {
    Stage cmb = st(kStHcCombine, il);
    cmb.comb = HcCombineArgs{h, out, inj, t, kEmbd};
    S.push_back(cmb);
  };

  // Bring-up only: MK_DEBUG_LAYERS=n builds the first n layers and no head (per-layer checks).
  uint32_t n_layers = kLayers;
  if (const char *dl = getenv("MK_DEBUG_LAYERS")) {
    n_layers = uint32_t(atoi(dl));
    if (n_layers == 0 || n_layers > kLayers) MK_DIE("MK_DEBUG_LAYERS=%s out of 1..%u", dl, kLayers);
    if (want_logits) MK_DIE("MK_DEBUG_LAYERS builds no head; ask for layers, not logits");
    fprintf(stderr, "engine: MK_DEBUG_LAYERS=%u (bring-up)\n", n_layers);
  }
  for (uint32_t il = 0; il < n_layers; ++il) {
    const Layer &L = layers_[il];
    if (il == kPleLayer) {
      float *pemb = scratch("ple_emb", size_t(kMaxPassTokens) * kEmbd);
      float *pk = scratch("ple_k", size_t(kMaxPassTokens) * kHcDim);
      float *kn = scratch("ple_kn", size_t(kMaxPassTokens) * kHcDim);
      float *pv = scratch("ple_v", size_t(kMaxPassTokens) * kEmbd);
      float *gv = scratch("ple_gv", size_t(kMaxPassTokens) * kHcDim);
      float *gvn = scratch("ple_gvn", size_t(kMaxPassTokens) * kHcDim);
      float *py = scratch("ple_y", size_t(kMaxPassTokens) * kHcDim);
      S.push_back(qmm_stage(il, L.ple_key, kHcDim, kEmbd, pemb, kEmbd, t, pk, kHcDim));
      S.push_back(rms_stage(il, pk, kn, L.ple_norm_key, t, 4, kEmbd, true));
      S.push_back(qmm_stage(il, L.ple_value, kEmbd, kEmbd, pemb, kEmbd, t, pv, kEmbd));
      S.push_back(rms_stage(il, h, xn, L.ple_norm_query, t, 4, kEmbd, true));
      Stage g = st(kStPleGate, il);
      g.pgate = PleGateArgs{kn, xn, pv, gv, t, kEmbd};
      S.push_back(g);
      S.push_back(rms_stage(il, gv, gvn, L.ple_norm_conv, t, 4, kEmbd, true));
      // The conv's history is the context's: one stage per segment, on its rows (Stage::sub = segment + 1 when several).
      for (uint32_t i = 0, r0 = 0; i < ns; r0 += seg[i].t, ++i) {
        Stage cv = st(kStPleConv, il);
        cv.sub = ns > 1 ? i + 1 : 0;
        cv.pconv = PleConvArgs{gvn + size_t(r0) * kHcDim, seg[i].c->ple_conv, L.ple_conv1d.f32(), py + size_t(r0) * kHcDim,
                               seg[i].t, kHcDim};
        S.push_back(cv);
      }
      Stage ad = st(kStAdd, il);
      ad.add = AddArgs{h, gv, py, t * kHcDim};
      S.push_back(ad);
    }
    hc_pre(il, 0, true);
    if (is_attn_layer(il)) add_attn(S, seg, ns, il, t, mixed, out);
    else add_gdn(S, seg, ns, il, t, mixed, out);
    hc_post(il);
    hc_pre(il, 1, true);
    add_moe(S, il, t, mixed, out);
    hc_post(il);
    if (layer_ends) layer_ends->push_back(uint32_t(S.size()));
  }
  if (n_layers < kLayers) return S;
  hc_pre(kLayers, 0, false);  // the final mixer: output_hc_*, no inject
  if (want_logits) {
    float *logits = scratch("logits", size_t(kMaxSegments) * kVocab);
    if (ns == 1) {
      S.push_back(qmm_stage(kLayers, output_, kVocab, kEmbd, mixed + size_t(t - 1) * kEmbd, kEmbd, 1, logits, kVocab));
    } else {
      // Each segment's last row, through the x_row map (run_pass uploads d_head_rows_): logits row i is segment i's.
      Stage hd = qmm_stage(kLayers, output_, kVocab, kEmbd, mixed, kEmbd, ns, logits, kVocab);
      hd.qmm.x_row = d_head_rows_;
      S.push_back(hd);
    }
  }
  return S;
}

std::string Engine::segments_problem(const Segment *s, uint32_t n) const {
  char why[256];
  if (n == 0 || n > kMaxSegments) {
    snprintf(why, sizeof why, "%u segments: a pass takes 1..%u segments", n, kMaxSegments);
    return why;
  }
  uint64_t total = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (!s[i].c || !s[i].tokens) {
      snprintf(why, sizeof why, "segment %u: no context or no tokens pointer", i);
      return why;
    }
    if (s[i].t == 0) {
      snprintf(why, sizeof why, "segment %u has no tokens", i);
      return why;
    }
    total += s[i].t;
  }
  if (total > kMaxPassTokens) {
    snprintf(why, sizeof why, "segments of %lu tokens in all: a pass takes 1..%u", (unsigned long)total, kMaxPassTokens);
    return why;
  }
  for (uint32_t i = 0; i < n; ++i)
    if (uint64_t(s[i].c->n) + s[i].t > s[i].c->max) {
      snprintf(why, sizeof why, "segment %u: context full: %u + %u tokens past its capacity %u", i, s[i].c->n, s[i].t, s[i].c->max);
      return why;
    }
  // Every pass writes cells [n, n + t): two segments on one buffer would write the same cells.
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t j = i + 1; j < n; ++j)
      if (s[i].c->kv == s[j].c->kv || s[i].c->idx_keys == s[j].c->idx_keys) {
        snprintf(why, sizeof why, "segments %u and %u share a K/V buffer (one context twice)", i, j);
        return why;
      }
  return "";
}

PassOut Engine::pass_segments(const Segment *s, uint32_t n, bool want_logits, float *logits_out, const float *ple) {
  return run_pass(s, n, want_logits, logits_out, ple);
}

// A fused pass that stopped on a spin-timeout (set_recover_spin_timeouts). Nothing it wrote is used: its contexts are
// poisoned and not advanced (run_pass advances n and the PLE history only after a clean pass). The control block's
// barrier count may be left mid-count, so end_pass marked it dirty and the next begin_pass does the full reset().
PassOut Engine::failed_pass(const Segment *seg, uint32_t ns, const char *why, double ms) {
  for (uint32_t i = 0; i < ns; ++i) seg[i].c->poisoned = true;
  last_error_ = std::string(why) + "; the pass's " + std::to_string(ns) + " context(s) are poisoned";
  fprintf(stderr, "engine: %s\n", last_error_.c_str());
  PassOut po;
  po.error = last_error_;
  po.ms = ms;
  return po;
}

uint32_t Engine::build_into(const Segment *seg, uint32_t ns, bool want_logits, Stage *out, uint32_t cap) {
  const std::string problem = segments_problem(seg, ns);
  if (!problem.empty()) MK_DIE("build: %s", problem.c_str());
  uint32_t t = 0;
  for (uint32_t i = 0; i < ns; ++i) t += seg[i].t;
  std::vector<uint32_t> ends;
  const std::vector<Stage> S = build(seg, ns, t, want_logits, &ends);
  if (S.size() > cap) MK_DIE("build: %zu stages, the buffer holds %u", S.size(), cap);
  memcpy(out, S.data(), sizeof(Stage) * S.size());
  return uint32_t(S.size());
}

PassOut Engine::pass_stages(const Segment *s, uint32_t n, bool want_logits, float *logits_out, const float *ple,
                            const Stage *stages, uint32_t n_stages) {
  if (!stages || n_stages == 0) MK_DIE("pass_stages: no stages");
  return run_pass(s, n, want_logits, logits_out, ple, stages, n_stages);
}

PassOut Engine::run_pass(const Segment *seg, uint32_t ns, bool want_logits, float *logits_out, const float *ple,
                         const Stage *stages, uint32_t n_stages) {
  for (uint32_t i = 0; i < ns; ++i)
    if (seg[i].c && seg[i].c->poisoned)
      MK_DIE("pass on a context poisoned by a spin-timeout (segment %u, %u tokens): its state is half-written", i, seg[i].c->n);
  const std::string problem = segments_problem(seg, ns);
  if (!problem.empty()) MK_DIE("pass: %s", problem.c_str());
  if (want_logits && !logits_out) MK_DIE("pass of %u segments: want_logits needs logits_out", ns);
  if (!ple) MK_DIE("pass of %u segments: no PLE rows", ns);
  uint32_t t = 0;
  std::vector<int32_t> tokens;
  std::vector<uint32_t> head_rows(ns);
  for (uint32_t i = 0; i < ns; ++i) {
    tokens.insert(tokens.end(), seg[i].tokens, seg[i].tokens + seg[i].t);
    t += seg[i].t;
    head_rows[i] = t - 1;
  }
  PassOut po;
  const auto t0 = std::chrono::steady_clock::now();
  MK_CHECK(hipMemcpy(d_tokens_, tokens.data(), sizeof(int32_t) * t, hipMemcpyHostToDevice));
  if (ns > 1) MK_CHECK(hipMemcpy(d_head_rows_, head_rows.data(), sizeof(uint32_t) * ns, hipMemcpyHostToDevice));
  // Each segment's n-gram rows, which the host gathered from its own context's history and tokens.
  MK_CHECK(hipMemcpy(scratch("ple_emb", size_t(kMaxPassTokens) * kEmbd), ple, sizeof(float) * size_t(t) * kEmbd,
                     hipMemcpyHostToDevice));
  std::vector<uint32_t> ends;
  std::vector<Stage> S = stages ? std::vector<Stage>(stages, stages + n_stages) : build(seg, ns, t, want_logits, &ends);
  if (S.size() > stage_cap_) {
    if (d_stages_) MK_CHECK(hipHostFree(d_stages_));
    stage_cap_ = S.size();
    MK_CHECK(hipHostMalloc(&d_stages_, sizeof(Stage) * stage_cap_, hipHostMallocDefault));
  }
  MK_CHECK(hipMemcpy(d_stages_, S.data(), sizeof(Stage) * S.size(), hipMemcpyHostToDevice));
  cb_.begin_pass();
  Control *ctl = cb_.device();
  Barrier *bar = cb_.barrier();
  uint32_t n = uint32_t(S.size());
  const Stage *ds = d_stages_;
  uint32_t skip = getenv("MK_MUTATE_SKIP_BARRIER") ? uint32_t(atoi(getenv("MK_MUTATE_SKIP_BARRIER"))) : UINT32_MAX;
  // MK_MUTATE_STALL_BARRIER=i (a test knob, tests/engine/test_spin_timeout.py): workgroup 0 leaves the kernel at the
  // first grid barrier at or after stage i without arriving, so every other workgroup waits out its bounded spin (1 s,
  // rt.h kDefaultSpinTicks) and the pass stops on a real kSpinTimeout, as a stalled or descheduled workgroup makes it.
  // Passed to k_mega in the skip word with the top bit set (mega.hip).
  if (const char *st = getenv("MK_MUTATE_STALL_BARRIER")) {
    char *end = nullptr;
    const unsigned long v = strtoul(st, &end, 10);
    if (end == st || *end || v >= n) MK_DIE("MK_MUTATE_STALL_BARRIER=%s: want a stage in 0..%u", st, n - 1);
    if (skip != UINT32_MAX) MK_DIE("MK_MUTATE_STALL_BARRIER: not with MK_MUTATE_SKIP_BARRIER");
    skip = 0x80000000u | uint32_t(v);
  }
  // 80 = 4 workgroups per WGP (8 waves/SIMD). The grid sweep (2026-09-27) ran 80 cleanly with no scratch;
  // tests/engine/test_mega_resources.py keeps k_mega inside that budget.
  int grid = std::min(max_coop_grid(mega_kernel(), kMkBlock, 0), 80);
  // MK_MEGA_GRID sets the grid for an experiment, above 40 too; launch_coop dies past the occupancy API's
  // limit, and the barrier's bounded spin stops a grid that isn't really co-resident (run 2's 60).
  if (const char *g = getenv("MK_MEGA_GRID")) grid = atoi(g);
  // The kernel's event ring stays off: no record is made (mega.hip).
  EventRingView ev{};
  void *args[] = {&ds, &n, &ctl, &bar, &skip, &ev};
  // Run 2: the occupancy API said 60 co-resident workgroups, but a 60-workgroup launch deadlocked at its
  // first barrier (spin timeout) while the kernel had 1.4 KiB/lane of scratch, which likely caps waves in
  // flight below what the API reports. Hence the no-scratch test. Every op is grid-independent, so any
  // grid gives the same bits. PLAN.md Problems.
  wd_.begin();
  if (!launched_ev_) {
    MK_CHECK(hipEventCreate(&launched_ev_));
    ++host_stats().events_created;
  }
  launch_coop(mega_kernel(), grid, kMkBlock, args, 0, 0);
  MK_CHECK(hipEventRecord(launched_ev_, 0));
  wd_.arm(launched_ev_);
  MK_CHECK(hipDeviceSynchronize());
  wd_.finish();
  cb_.end_pass();
  if (cb_.status() != kOk) {
    char why[256];
    snprintf(why, sizeof why, "megakernel stopped: %s at stage %u (wg %u)", status_name(cb_.status()), cb_.host()->status_site,
             cb_.host()->status_wg);
    // Recoverable only when asked, only a spin-timeout, and only when the watchdog didn't ask the kernel to stop too
    // (its deadline or the memory floor: those stay fatal).
    if (cb_.status() == kSpinTimeout && recover_spin_timeouts_ && wd_.reason().empty())
      return failed_pass(seg, ns, why, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    MK_DIE("%s", why);
  }
  if (want_logits) {
    MK_CHECK(hipMemcpy(logits_out, scratch("logits", size_t(kMaxSegments) * kVocab), sizeof(float) * kVocab * ns,
                       hipMemcpyDeviceToHost));
    ++host_stats().logits_dtoh;
  }
  last_error_.clear();
  // advance each context
  for (uint32_t i = 0; i < ns; ++i) {
    Context *c = seg[i].c;
    const int32_t *tk = seg[i].tokens;
    const uint32_t ti = seg[i].t;
    c->n += ti;
    c->pooled_n = c->n / kIdxRatio;  // the pass's pool stages wrote the blocks it completed (add_attn)
    if (ti >= 2) {
      c->ple_hist[0] = tk[ti - 2];
      c->ple_hist[1] = tk[ti - 1];
    } else {
      c->ple_hist[0] = c->ple_hist[1];
      c->ple_hist[1] = tk[0];
    }
  }
  po.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return po;
}

}  // namespace mk
