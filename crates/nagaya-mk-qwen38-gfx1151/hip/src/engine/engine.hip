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


// ---- weights: the arena (arena.h) ----
namespace {

double since_s(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

uint64_t round_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

// Coarse-grained, always (weights never change while a kernel runs): HIP registers host memory fine-grained by
// default (hip_runtime_api.h). Measured 2026-09-30, 51-token fused read p50: fine-grained arena 452-456 ms,
// coarse-grained 448 ms, hipMalloc baseline 454 ms (unfused ~209 all three): no difference that decides it. The
// relayout kernels' writes reach memory at each kernel's end (the fill synchronizes after each one). Note
// hipHostRegisterReadOnly is the same bit (0x8) in this ROCm and the device reports no read-only support: the
// attacher's protection is its PROT_READ mapping and the seals, not a flag.
constexpr unsigned kArenaRegister = hipExtHostRegisterCoarseGrained;

// 16 CPU threads: one thread's memcpy out of the GGUF mapping is far below what the page cache and the arena take.
void par_memcpy(uint8_t *dst, const uint8_t *src, uint64_t n) {
  const int T = 16;
  std::vector<std::thread> th;
  const uint64_t per = (n + T - 1) / T;
  for (int k = 0; k < T; ++k) {
    const uint64_t o = k * per;
    if (o >= n) break;
    th.emplace_back([=] { memcpy(dst + o, src + o, std::min(per, n - o)); });
  }
  for (auto &x : th) x.join();
}

void relayout(Tile kind, const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k) {
  switch (kind) {
    case Tile::kQ6: relayout_q6_tiles(src, dst, rows, k); break;
    case Tile::kQ8: relayout_q8_tiles(src, dst, rows, k); break;
    case Tile::kIq4: relayout_iq4_tiles(src, dst, rows, k); break;
    case Tile::kIq3s: relayout_iq3s_tiles(src, dst, rows, k); break;
    case Tile::kNone: MK_DIE("relayout of an untiled tensor");
  }
}

}  // namespace

arena::ShardId arena::shard_id(const std::string &path) {
  ShardId s;
  memset(&s, 0, sizeof s);
  char real[PATH_MAX];
  if (!realpath(path.c_str(), real)) MK_DIE("%s: %s", path.c_str(), strerror(errno));
  if (strlen(real) >= sizeof s.path) MK_DIE("%s: path longer than %zu bytes", real, sizeof s.path - 1);
  struct stat st;
  if (stat(real, &st)) MK_DIE("stat %s: %s", real, strerror(errno));
  s.size = uint64_t(st.st_size);
  s.mtime_ns = int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
  memcpy(s.path, real, strlen(real));
  return s;
}

Engine::Engine(const mk_tensor_table *table) : table_(table), shards_(table_.shards()) {
  trap_sink_install();  // a device trap names itself on stderr before the runtime aborts (trap.h)
  // The PLE table stays mapped; the host gathers 16 rows per token (ple_rows). The rows a pass needs are scattered
  // over ~29 GB: a fault should read its own page, not the ~128 KiB of readahead around it (ple_rows asks for each
  // row's pages itself, all at once).
  ple_ = &table_.tensor("per_layer_token_embd.weight");
  const uintptr_t pg = uintptr_t(sysconf(_SC_PAGESIZE)), a = uintptr_t(ple_->data) & ~(pg - 1);
  if (madvise(reinterpret_cast<void *>(a), uintptr_t(ple_->data) + ple_->n_bytes - a, MADV_RANDOM))
    MK_DIE("madvise(MADV_RANDOM) on the PLE table: %s", strerror(errno));
  const std::vector<Planned> P = plan(table_, &arena_bytes_);
  create_arena(P);
  for (const Planned &p : P) {
    Tensor t = p.t;
    t.d = arena_ + p.offset;
    w_[p.name] = t;
  }
  wire();
}

Engine::~Engine() {
  scratch_.free_all();
  (void)hipFree(d_tokens_);
  (void)hipFree(d_head_rows_);
  (void)hipHostFree(d_stages_);
  if (launched_ev_) (void)hipEventDestroy(launched_ev_);
  if (arena_) {
    (void)hipHostUnregister(arena_);
    munmap(arena_, arena_bytes_);
  }
  if (arena_fd_ >= 0) close(arena_fd_);
}

// Every uploaded tensor's place in the arena and its final layout, from the GGUF and this build alone. Read-once tiles (PLAN.md Speed): the IQ3_S expert gate and
// up matrices, the IQ4_NL expert down matrices (each expert's tiles contiguous, so the MoE addresses expert e at e x
// rows x row bytes as before), every Q8_0 matrix the matmuls read (all but the token embedding table, which is
// gathered by row, and the 5 layers' Q8_0 expert down matrices, which measured no gain in tiles: PLAN.md), and the
// Q6_K output head. Values are unchanged; both the f16 and the exact kernels decode tiles.
std::vector<Engine::Planned> Engine::plan(const TensorTable &table, uint64_t *arena_bytes) {
  const bool tiles = tiles_enabled();
  std::vector<Planned> P;
  for (size_t i = 0; i < table.n_tensors(); ++i) {
    const auto &ti = table.tensor(i);
    if (ti.name == "per_layer_token_embd.weight") continue;  // stays in the GGUF mapping (ple_rows)
    const std::string &nm = ti.name;
    if (nm.size() >= sizeof(arena::Entry::name)) MK_DIE("%s: name longer than the arena's %zu bytes", nm.c_str(), sizeof(arena::Entry::name) - 1);
    Planned p;
    p.gguf = i;
    p.name = nm;
    p.src_bytes = ti.n_bytes;
    Tensor &t = p.t;
    t.type = uint32_t(ti.type);
    for (int j = 0; j < 3; ++j) t.ne[j] = ti.ne[j];
    if (q::block_bytes(t.type) == 0) MK_DIE("%s: unsupported type %u", nm.c_str(), t.type);
    if (ti.ne[0] % q::block_values(t.type) != 0 || (q::block_values(t.type) > 1 && ti.ne[0] % 32 != 0))
      MK_DIE("%s: ne0 %lu is not a multiple of its block (%u)", nm.c_str(), (unsigned long)ti.ne[0], q::block_values(t.type));
    t.row_bytes = q::row_bytes(t.type, uint32_t(ti.ne[0]));
    p.bytes = ti.n_bytes;
    p.k = uint32_t(t.ne[0]);
    p.rows = uint32_t(t.ne[1] * t.ne[2]);
    if (tiles) {
      const bool expert_gu = nm.size() > 20 && (nm.find(".ffn_gate_exps.weight") != std::string::npos ||
                                                nm.find(".ffn_up_exps.weight") != std::string::npos);
      const bool expert_dn = nm.size() > 20 && nm.find(".ffn_down_exps.weight") != std::string::npos;
      if (expert_dn && t.type != q::IQ4_NL && t.type != q::Q8_0)
        MK_DIE("%s: expert down matrices are IQ4_NL or Q8_0, this one is type %u", nm.c_str(), t.type);
      p.tile = (expert_gu && t.type == q::IQ3_S) ? Tile::kIq3s
               : (t.type == q::Q8_0 && nm != "token_embd.weight" && nm.find("_exps.") == std::string::npos) ? Tile::kQ8
               : (t.type == q::Q6_K && nm == "output.weight") ? Tile::kQ6
               : (expert_dn && t.type == q::IQ4_NL) ? Tile::kIq4 : Tile::kNone;
      if (p.tile != Tile::kNone) {
        const uint64_t bytes = p.tile == Tile::kQ6 ? q6_tiled_bytes(p.rows, p.k) : p.tile == Tile::kQ8 ? q8_tiled_bytes(p.rows, p.k)
                               : p.tile == Tile::kIq4 ? iq4_tiled_bytes(p.rows, p.k) : iq3s_tiled_bytes(p.rows, p.k);
        if (p.tile != Tile::kIq3s && bytes != t.row_bytes * p.rows)
          MK_DIE("%s: type %u tiles take %lu bytes, the rows %lu", nm.c_str(), t.type, (unsigned long)bytes, (unsigned long)(t.row_bytes * p.rows));
        t.type = p.tile == Tile::kQ6 ? q::Q6_K_TILES : p.tile == Tile::kQ8 ? q::Q8_0_TILES
                 : p.tile == Tile::kIq4 ? q::IQ4_NL_TILES : q::IQ3_S_TILES;
        t.row_bytes = q::row_bytes(t.type, p.k);
        p.bytes = bytes;
      }
    }
    P.push_back(p);
  }
  uint64_t off = round_up(arena::kHeaderBytes + P.size() * sizeof(arena::Entry), arena::kDataAlign);
  for (Planned &p : P) {
    p.offset = off;
    off = round_up(off + p.bytes, arena::kAlign);
  }
  *arena_bytes = off;
  return P;
}

arena::Header Engine::arena_header(const std::vector<Planned> &P, uint64_t arena_bytes, const std::vector<std::string> &shards_) {
  arena::Header h;
  memset(&h, 0, sizeof h);
  memcpy(h.magic, arena::kMagic, sizeof h.magic);
  h.format = arena::kFormat;
  h.tiles = tiles_enabled() ? 1 : 0;
  h.n_tensors = uint32_t(P.size());
  if (shards_.size() > arena::kMaxShards) MK_DIE("%zu shards, the arena records %u", shards_.size(), arena::kMaxShards);
  h.n_shards = uint32_t(shards_.size());
  h.bytes = arena_bytes;
  h.manifest_off = arena::kHeaderBytes;
  h.data_off = P.empty() ? arena_bytes : P[0].offset;
  for (size_t i = 0; i < shards_.size(); ++i) h.shards[i] = arena::shard_id(shards_[i]);
  return h;
}

arena::Entry Engine::arena_entry(const Planned &p) {
  arena::Entry e;
  memset(&e, 0, sizeof e);
  memcpy(e.name, p.name.data(), p.name.size());
  e.type = p.t.type;
  for (int j = 0; j < 3; ++j) e.ne[j] = p.t.ne[j];
  e.row_bytes = p.t.row_bytes;
  e.offset = p.offset;
  e.bytes = p.bytes;
  return e;
}

// Fill a new arena: every tensor copied out of the GGUF mapping by the CPU straight into place, the tiled ones through
// a pinned scratch buffer and the relayout kernel; each chunk's file pages dropped from the page cache once copied
// (best effort, the reader's drop_cached: tensor_table.h): the weights are read once, and the cache is better spent on the PLE
// table the host keeps gathering from, and on the other tenants. Then the header and manifest, then the seals.
void Engine::create_arena(const std::vector<Planned> &P) {
  const auto t0 = std::chrono::steady_clock::now();
  arena_fd_ = memfd_create("mk-weights", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (arena_fd_ < 0) MK_DIE("memfd_create: %s", strerror(errno));
  if (ftruncate(arena_fd_, off_t(arena_bytes_))) MK_DIE("ftruncate the arena to %lu: %s", (unsigned long)arena_bytes_, strerror(errno));
  void *m = mmap(nullptr, arena_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, arena_fd_, 0);
  if (m == MAP_FAILED) MK_DIE("mmap the arena: %s", strerror(errno));
  arena_ = static_cast<uint8_t *>(m);
  if (madvise(arena_, arena_bytes_, MADV_HUGEPAGE)) MK_DIE("madvise(MADV_HUGEPAGE) on the arena: %s", strerror(errno));
  MK_CHECK(hipHostRegister(arena_, arena_bytes_, kArenaRegister));
  uint64_t scratch_bytes = 0;
  for (const Planned &p : P)
    if (p.tile != Tile::kNone) scratch_bytes = std::max(scratch_bytes, p.src_bytes);
  uint8_t *scratch = nullptr;
  if (scratch_bytes) MK_CHECK(hipHostMalloc(&scratch, scratch_bytes, hipHostMallocDefault));
  constexpr uint64_t kChunk = 512ull << 20;
  uint64_t tiled = 0;
  for (const Planned &p : P) {
    const auto &ti = table_.tensor(p.gguf);
    uint8_t *to = p.tile == Tile::kNone ? arena_ + p.offset : scratch;
    for (uint64_t o = 0; o < p.src_bytes; o += kChunk) {
      const uint64_t n = std::min(kChunk, p.src_bytes - o);
      par_memcpy(to + o, ti.data + o, n);
      table_.drop_cached(ti.data + o, n);
    }
    if (p.tile != Tile::kNone) {
      relayout(p.tile, scratch, arena_ + p.offset, p.rows, p.k);
      MK_CHECK(hipDeviceSynchronize());  // the scratch is reused by the next tiled tensor
      tiled += p.bytes;
    }
  }
  if (scratch) MK_CHECK(hipHostFree(scratch));
  const arena::Header h = arena_header(P, arena_bytes_, shards_);
  memcpy(arena_, &h, sizeof h);
  for (size_t i = 0; i < P.size(); ++i) {
    const arena::Entry e = arena_entry(P[i]);
    memcpy(arena_ + arena::kHeaderBytes + i * sizeof e, &e, sizeof e);
  }
  if (fcntl(arena_fd_, F_ADD_SEALS, arena::kSeals)) MK_DIE("sealing the arena: %s", strerror(errno));
  if (unsigned(fcntl(arena_fd_, F_GET_SEALS)) != arena::kSeals) MK_DIE("the arena's seals are %#x", fcntl(arena_fd_, F_GET_SEALS));
  fprintf(stderr, "engine: loaded %.2f GiB of weights (%.2f GiB in read-once tiles) into the arena in %.1f s\n",
          arena_bytes_ / double(1ull << 30), tiled / double(1ull << 30), since_s(t0));
}

void Engine::wire() {
  // The PLE hash's reset token is compiled in (model.h); the file must agree (kaibo M3/M4 review).
  if (table_.ple_eos_token_id() != int64_t(kPleEos))
    MK_DIE("GGUF qwen4exp.ple.eos_token_id %ld != compiled kPleEos %u", (long)table_.ple_eos_token_id(), kPleEos);
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

// ---- contexts ----
static void check_capacity(uint32_t max_tokens) {
  if (max_tokens == 0 || max_tokens > kMaxContext || max_tokens % kMaxPassTokens)
    MK_DIE("context capacity %u: want a multiple of %u in 1..%u", max_tokens, kMaxPassTokens, kMaxContext);
}

Context *Engine::new_context(uint32_t max_tokens) {
  check_capacity(max_tokens);
  auto *c = new Context;
  c->max = max_tokens;
  const size_t n_gdn = 36, n_att = 12;
  MK_CHECK(hipMalloc(&c->gdn_state, sizeof(float) * n_gdn * kGdnVHeads * kGdnKDim * kGdnVDim));
  MK_CHECK(hipMalloc(&c->gdn_conv, sizeof(float) * n_gdn * (kGdnConv - 1) * kGdnConvDim));
  MK_CHECK(hipMalloc(&c->ple_conv, sizeof(float) * 9 * kHcDim));
  MK_CHECK(hipMalloc(&c->kv, sizeof(float) * n_att * size_t(max_tokens) * 2 * kKvHeads * kHeadDim));
  MK_CHECK(hipMalloc(&c->idx_keys, sizeof(float) * n_att * size_t(max_tokens) * kIdxDim));
  // Filled with NaN (0xff bytes), so a block read before it was pooled scores as a poisoned one instead of as whatever the
  // allocator left there; nothing reads a block past pooled_n.
  MK_CHECK(hipMalloc(&c->idx_pooled, sizeof(float) * n_att * size_t(max_tokens / kIdxRatio) * kIdxDim));
  MK_CHECK(hipMemset(c->idx_pooled, 0xff, sizeof(float) * n_att * size_t(max_tokens / kIdxRatio) * kIdxDim));
  MK_CHECK(hipMemset(c->gdn_state, 0, sizeof(float) * n_gdn * kGdnVHeads * kGdnKDim * kGdnVDim));
  MK_CHECK(hipMemset(c->gdn_conv, 0, sizeof(float) * n_gdn * (kGdnConv - 1) * kGdnConvDim));
  MK_CHECK(hipMemset(c->ple_conv, 0, sizeof(float) * 9 * kHcDim));
  return c;
}

void Engine::free_context(Context *c) {
  if (!c) return;
  (void)hipFree(c->gdn_state);
  (void)hipFree(c->gdn_conv);
  (void)hipFree(c->ple_conv);
  (void)hipFree(c->kv);
  (void)hipFree(c->idx_keys);
  (void)hipFree(c->idx_pooled);
  delete c;
}

static float *state_part(const Context *c, uint32_t part) {
  if (part == 0) return c->gdn_state;
  if (part == 1) return c->gdn_conv;
  if (part == 2) return c->ple_conv;
  MK_DIE("context state part %u: want 0 (gdn_state), 1 (gdn_conv) or 2 (ple_conv)", part);
}

// The cell parts (3 kv, 4 idx_keys): floats per cell. A layer's cells are c->max apart on the device.
static size_t cell_floats(uint32_t part) {
  if (part == 3) return size_t(2) * kKvHeads * kHeadDim;
  if (part == 4) return kIdxDim;
  MK_DIE("context cell part %u: want 3 (kv) or 4 (idx_keys)", part);
}

uint64_t Engine::state_bytes(uint32_t part, uint32_t n) {
  const size_t n_gdn = 36, n_att = 12;
  if (part == 0) return sizeof(float) * n_gdn * kGdnVHeads * kGdnKDim * kGdnVDim;
  if (part == 1) return sizeof(float) * n_gdn * (kGdnConv - 1) * kGdnConvDim;
  if (part == 2) return sizeof(float) * 9 * kHcDim;
  if (part == 3 || part == 4) return sizeof(float) * n_att * size_t(n) * cell_floats(part);
  if (part == 5) return sizeof(int32_t) * 2;
  MK_DIE("context state part %u: want 0 (gdn_state), 1 (gdn_conv), 2 (ple_conv), 3 (kv), 4 (idx_keys) or 5 (ple_hist)", part);
}

uint64_t Engine::state_bytes(uint32_t part) {
  if (part == 3 || part == 4) MK_DIE("context state part %u: its size depends on the token count (state_bytes(part, n))", part);
  return state_bytes(part, 0);
}

void Engine::state_read(const Context *c, uint32_t part, void *dst) {
  if (c->poisoned) MK_DIE("state read of a context poisoned by a spin-timeout (%u tokens): its state is half-written", c->n);
  if (part == 3 || part == 4) {
    // One row per attention layer: cells 0..n-1 of a layer (stride: the capacity) into a compact [n] block.
    if (c->n == 0) return;
    const size_t cell = sizeof(float) * cell_floats(part), n_att = 12;
    MK_CHECK(hipMemcpy2D(dst, cell * c->n, part == 3 ? c->kv : c->idx_keys, cell * c->max, cell * c->n, n_att,
                         hipMemcpyDeviceToHost));
    return;
  }
  if (part == 5) {
    memcpy(dst, c->ple_hist, sizeof(c->ple_hist));
    return;
  }
  MK_CHECK(hipMemcpy(dst, state_part(c, part), state_bytes(part), hipMemcpyDeviceToHost));
}

// The rules for writing what a woken context restores (the cell parts, the PLE history, its token count).
static void check_restorable(const Context *c, const char *what) {
  if (c->poisoned) MK_DIE("%s into a context poisoned by a spin-timeout (%u tokens)", what, c->n);
  if (!c->owns_kv) MK_DIE("%s into a context that does not own its K/V", what);
  if (c->n != 0) MK_DIE("%s into a context of %u tokens: only an empty context (n = 0) is restored", what, c->n);
}

void Engine::state_write(Context *c, uint32_t part, const void *src) {
  if (c->poisoned) MK_DIE("state write into a context poisoned by a spin-timeout (%u tokens)", c->n);
  if (part == 3 || part == 4) MK_DIE("state write of part %u (cells): use state_write_cells with the cell count", part);
  if (part == 5) {
    check_restorable(c, "ple_hist write");
    int32_t h[2];
    memcpy(h, src, sizeof(h));
    for (int i = 0; i < 2; ++i)
      if (h[i] < 0 || uint32_t(h[i]) >= kVocab) MK_DIE("ple_hist write: ple_hist[%d] = %d, not a token id (0..%u)", i, h[i], kVocab - 1);
    c->ple_hist[0] = h[0];
    c->ple_hist[1] = h[1];
    c->restored_ple = true;
    return;
  }
  MK_CHECK(hipMemcpy(state_part(c, part), src, state_bytes(part), hipMemcpyHostToDevice));
}

void Engine::state_write_cells(Context *c, uint32_t part, uint32_t n, const void *src) {
  const char *what = part == 3 ? "K/V write" : part == 4 ? "index-key write" : nullptr;
  if (!what) MK_DIE("state_write_cells of part %u: want 3 (kv) or 4 (idx_keys)", part);
  check_restorable(c, what);
  if (n == 0 || n > c->max) MK_DIE("%s of %u cells into a context of capacity %u", what, n, c->max);
  const size_t cell = sizeof(float) * cell_floats(part), n_att = 12;
  MK_CHECK(hipMemcpy2D(part == 3 ? c->kv : c->idx_keys, cell * c->max, src, cell * n, cell * n, n_att, hipMemcpyHostToDevice));
  (part == 3 ? c->restored_kv : c->restored_idx) = n;
}

void Engine::restore_n(Context *c, uint32_t n) {
  if (c->poisoned) MK_DIE("set_n(%u) on a context poisoned by a spin-timeout (%u tokens)", n, c->n);
  if (!c->owns_kv) MK_DIE("set_n(%u) on a context that does not own its K/V", n);
  if (c->n != 0) MK_DIE("set_n(%u) on a context of %u tokens: only an empty context (n = 0) is restored", n, c->n);
  if (n == 0 || n > c->max) MK_DIE("set_n(%u) past the capacity %u (want 1..capacity)", n, c->max);
  if (c->restored_kv != n || c->restored_idx != n || !c->restored_ple)
    MK_DIE("set_n(%u): K/V written for %u cells, index keys for %u, PLE history %s: want both %u cells and the history", n,
           c->restored_kv, c->restored_idx, c->restored_ple ? "written" : "not written", n);
  c->n = n;
  // The pooled-key cache is derived state, not parked: pool every complete block of the written index keys, through the
  // unfused path (one launch per QSA layer), so the invariant pooled_n == n / 4 holds and a fork of this context never has
  // to write below its parent's n.
  if (n / kIdxRatio) run_stages_now(pool_stages(c, 0));
  c->pooled_n = n / kIdxRatio;
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

// ---- PLE rows (host): Qwen4ExpTextNGramEmbedding.forward for one pass ----
void Engine::ple_rows(const Context *c, const int32_t *tokens, uint32_t t, float *out) const {
  static std::vector<int64_t> mult, offs, sizes;
  if (mult.empty()) {
    mult = table_.ple_layer_multipliers();
    offs = table_.ple_head_offsets();
    sizes = table_.ple_head_vocab_sizes();
    if (mult.size() != 3 || offs.size() != kPleHeads || sizes.size() != kPleHeads) MK_DIE("PLE constants have the wrong shape");
  }
  const int64_t eos = kPleEos;
  std::vector<int64_t> hist(2 + t);
  hist[0] = c->ple_hist[0];
  hist[1] = c->ple_hist[1];
  for (uint32_t i = 0; i < t; ++i) hist[2 + i] = tokens[i];
  if (!ple_) MK_DIE("PLE table not found at load");
  const uint64_t row_bytes = ple_->n_bytes / (ple_->n_elements / ple_->ne[0]), n_rows = ple_->n_elements / ple_->ne[0];
  if (ple_->ne[0] != kPleHeadDim) MK_DIE("PLE rows of %lu values, want %u", (unsigned long)ple_->ne[0], kPleHeadDim);
  // Two sweeps over the same positions: first every row id (asking the kernel for each row's pages, so the reads
  // of a pass's 16 x T rows overlap instead of faulting one after another), then the rows, in the same order.
  std::vector<int64_t> ids;
  ids.reserve(size_t(t) * kPleHeads);
  const uintptr_t pg = uintptr_t(sysconf(_SC_PAGESIZE));
  int64_t last_eos = -1;  // last eos strictly before p
  for (uint32_t p = 0; p < 2 + t; ++p) {
    const int64_t seg_start = last_eos + 1, pos_in_seg = int64_t(p) - seg_start;
    if (p >= 2) {
      int64_t tok[3];
      for (int s = 0; s < 3; ++s) tok[s] = (pos_in_seg >= s && int64_t(p) - s >= 0) ? hist[p - s] : eos;
      for (int n = 2; n <= 3; ++n) {
        uint64_t mixed = uint64_t(tok[0]) * uint64_t(mult[0]);
        for (int s = 1; s < n; ++s) mixed ^= uint64_t(tok[s]) * uint64_t(mult[s]);
        const int64_t m = int64_t(mixed);
        for (uint32_t h = 0; h < 8; ++h) {
          const uint32_t head = (n - 2) * 8 + h;
          const int64_t v = sizes[head];
          const int64_t id = ((m % v) + v) % v + offs[head];
          if (id < 0 || uint64_t(id) >= n_rows) MK_DIE("PLE row %ld out of %lu", (long)id, (unsigned long)n_rows);
          ids.push_back(id);
          const uintptr_t r0 = uintptr_t(ple_->data + uint64_t(id) * row_bytes), a = r0 & ~(pg - 1);
          (void)madvise(reinterpret_cast<void *>(a), r0 + row_bytes - a, MADV_WILLNEED);  // a hint: faults still work
        }
      }
    }
    if (hist[p] == eos) last_eos = p;
  }
  for (size_t i = 0; i < ids.size(); ++i)
    table_.dequant_f32(ple_->type, ple_->data + uint64_t(ids[i]) * row_bytes, kPleHeadDim,
                       out + (i / kPleHeads) * kEmbd + (i % kPleHeads) * kPleHeadDim);
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
static StageFootprint rust_footprint(const TensorTable &table, const Stage &s) {
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
static void fuse_hc_mix(std::vector<Stage> &S, std::vector<uint32_t> *layer_ends, const TensorTable &table) {
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
static uint32_t apply_schedule(const TensorTable &table, std::vector<Stage> &S, const std::vector<uint32_t> &ends, bool elide,
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

PassOut Engine::pass_segments(const Segment *s, uint32_t n, bool want_logits, float *logits_out) {
  return run_pass(s, n, want_logits, logits_out);
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

PassOut Engine::run_pass(const Segment *seg, uint32_t ns, bool want_logits, float *logits_out) {
  for (uint32_t i = 0; i < ns; ++i)
    if (seg[i].c && seg[i].c->poisoned)
      MK_DIE("pass on a context poisoned by a spin-timeout (segment %u, %u tokens): its state is half-written", i, seg[i].c->n);
  const std::string problem = segments_problem(seg, ns);
  if (!problem.empty()) MK_DIE("pass: %s", problem.c_str());
  if (want_logits && !logits_out) MK_DIE("pass of %u segments: want_logits needs logits_out", ns);
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
  {
    // Each segment's n-gram rows hash its own context's history and tokens.
    std::vector<float> pe(size_t(t) * kEmbd);
    for (uint32_t i = 0, r0 = 0; i < ns; r0 += seg[i].t, ++i) ple_rows(seg[i].c, seg[i].tokens, seg[i].t, pe.data() + size_t(r0) * kEmbd);
    float *pemb = scratch("ple_emb", size_t(kMaxPassTokens) * kEmbd);
    MK_CHECK(hipMemcpy(pemb, pe.data(), sizeof(float) * pe.size(), hipMemcpyHostToDevice));
  }
  std::vector<uint32_t> ends;
  std::vector<Stage> S = build(seg, ns, t, want_logits, &ends);
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
