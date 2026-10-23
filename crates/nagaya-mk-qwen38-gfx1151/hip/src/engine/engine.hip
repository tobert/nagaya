// SPDX-License-Identifier: MIT
// k_stage's register budget (tests/engine/test_mega_resources.py: grid >= 80 and no scratch, for k_mega and k_stage
// alike) has no room for the expert item's one-step lookahead (qmmh.h: 16 VGPRs of prefetched activations), so this TU,
// the one that compiles k_stage, runs the ops without it. Both paths give the same bits. Before any ops include.
#define MK_EXPERT_LOOKAHEAD 0
#include "engine.h"
#include "stage_sched.h"
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
  // The lab's masked selection (attn_mask.h) runs here only, never in k_mega: run_stage, the megakernel's dispatcher too,
  // stays as it was.
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

Engine::Engine(const mk_tensor_table *table, const char *attach) : table_(table), shards_(table_.shards()) {
  trap_sink_install();  // a device trap names itself on stderr before the runtime aborts (trap.h)
  if (getenv("MK_SCRATCH_WARMUP")) {
    // Bring-up experiment: run one k_stage (an empty add) before the weights exist, so the
    // queue's scratch is allocated in a fresh address space.
    Stage *ds;
    Stage s;
    memset(&s, 0, sizeof s);
    s.op = kStAdd;
    MK_CHECK(hipMalloc(&ds, sizeof s));
    MK_CHECK(hipMemcpy(ds, &s, sizeof s, hipMemcpyHostToDevice));
    k_stage<<<1, kMkBlock>>>(ds, cb_.device(), 0);
    MK_CHECK(hipDeviceSynchronize());
    MK_CHECK(hipFree(ds));
    fprintf(stderr, "engine: scratch warm-up done\n");
  }
  // The PLE table stays mapped; the host gathers 16 rows per token (ple_rows). The rows a pass needs are scattered
  // over ~29 GB: a fault should read its own page, not the ~128 KiB of readahead around it (ple_rows asks for each
  // row's pages itself, all at once).
  ple_ = &table_.tensor("per_layer_token_embd.weight");
  const uintptr_t pg = uintptr_t(sysconf(_SC_PAGESIZE)), a = uintptr_t(ple_->data) & ~(pg - 1);
  if (madvise(reinterpret_cast<void *>(a), uintptr_t(ple_->data) + ple_->n_bytes - a, MADV_RANDOM))
    MK_DIE("madvise(MADV_RANDOM) on the PLE table: %s", strerror(errno));
  const std::vector<Planned> P = plan(table_, &arena_bytes_);
  if (attach) attach_arena(attach, P);
  else create_arena(P);
  for (const Planned &p : P) {
    Tensor t = p.t;
    t.d = arena_ + p.offset;
    w_[p.name] = t;
  }
  wire();
}

Engine::~Engine() {
  scratch_.free_all();
  for (const ForkState &f : fork_free_) {
    (void)hipFree(f.gdn_state);
    (void)hipFree(f.gdn_conv);
    (void)hipFree(f.ple_conv);
  }
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

// Every uploaded tensor's place in the arena and its final layout, from the GGUF and this build alone (so an attach
// can compare an arena's manifest with it field by field). Read-once tiles (PLAN.md Speed): the IQ3_S expert gate and
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

// This build's bytes for p, from the GGUF (the tiled ones through the relayout kernel), into out.
void Engine::expected_bytes(const TensorTable &table, const Planned &p, std::vector<uint8_t> &out) {
  const auto &ti = table.tensor(p.gguf);
  out.resize(p.bytes);
  if (p.tile == Tile::kNone) {
    par_memcpy(out.data(), ti.data, p.bytes);
  } else {
    uint8_t *src, *dst;
    MK_CHECK(hipHostMalloc(&src, p.src_bytes, hipHostMallocDefault));
    MK_CHECK(hipMalloc(&dst, p.bytes));
    par_memcpy(src, ti.data, p.src_bytes);
    relayout(p.tile, src, dst, p.rows, p.k);
    MK_CHECK(hipDeviceSynchronize());
    MK_CHECK(hipMemcpy(out.data(), dst, p.bytes, hipMemcpyDeviceToHost));
    MK_CHECK(hipFree(dst));
    MK_CHECK(hipHostFree(src));
  }
  table.drop_cached(ti.data, p.src_bytes);
}

// Whether the arena open at fd (named path in messages) is exactly what this build would make from these GGUF files:
// "" if so, else the first difference. Everything is read through pread (a hole in a shmem file reads as zeros without
// allocating), before a page is mapped or registered: the seals, the header against the GGUF files and tile setting,
// the manifest against this build's plan field by field, and the smallest tensor of every type against this build's
// own conversion of the GGUF (a tile format that changed between builds). *types: how many types were spot-checked.
std::string Engine::arena_problem(const TensorTable &table, const std::vector<std::string> &shards, const std::vector<Planned> &P,
                                  uint64_t arena_bytes, int fd, const char *path, size_t *types) {
  char msg[768];
  auto say = [&](const char *fmt, auto... a) {
    snprintf(msg, sizeof msg, fmt, a...);
    return std::string(path) + ": " + msg;
  };
  struct stat st;
  if (fstat(fd, &st)) return say("fstat: %s", strerror(errno));
  const int seals = fcntl(fd, F_GET_SEALS);
  if (seals < 0 || unsigned(seals) != arena::kSeals) return say("not sealed as a weight arena (seals %#x, want %#x)", seals, arena::kSeals);
  auto pread_all = [&](void *buf, uint64_t n, uint64_t off) -> bool {
    uint8_t *b = static_cast<uint8_t *>(buf);
    for (uint64_t got = 0; got < n;) {
      const ssize_t r = pread(fd, b + got, n - got, off_t(off + got));
      if (r <= 0) return false;
      got += uint64_t(r);
    }
    return true;
  };
  arena::Header h;
  if (uint64_t(st.st_size) < sizeof h || !pread_all(&h, sizeof h, 0)) return say("too short for an arena header");
  if (memcmp(h.magic, arena::kMagic, sizeof h.magic)) return say("not a weight arena (magic)");
  if (h.format != arena::kFormat) return say("arena format %u, this build reads %u", h.format, arena::kFormat);
  const arena::Header want = arena_header(P, arena_bytes, shards);
  if (h.tiles != want.tiles)
    return say("made with tiles %s, this engine runs with tiles %s (MK_TILES)", h.tiles ? "on" : "off", want.tiles ? "on" : "off");
  if (h.n_shards != want.n_shards) return say("%u shards, this engine opened %u", h.n_shards, want.n_shards);
  for (uint32_t i = 0; i < h.n_shards; ++i) {
    const arena::ShardId &a = h.shards[i], &b = want.shards[i];
    if (a.size != b.size || a.mtime_ns != b.mtime_ns || strncmp(a.path, b.path, sizeof a.path))
      return say("shard %u is %.240s (%lu bytes, mtime %ld ns), this engine opened %.240s (%lu bytes, mtime %ld ns)", i,
                 a.path, (unsigned long)a.size, (long)a.mtime_ns, b.path, (unsigned long)b.size, (long)b.mtime_ns);
  }
  if (h.bytes != want.bytes || uint64_t(st.st_size) != want.bytes || h.n_tensors != want.n_tensors ||
      h.manifest_off != want.manifest_off || h.data_off != want.data_off)
    return say("%lu bytes (file %lu), %u tensors, manifest at %lu, data at %lu; this build plans %lu, %u, %lu, %lu",
               (unsigned long)h.bytes, (unsigned long)st.st_size, h.n_tensors, (unsigned long)h.manifest_off, (unsigned long)h.data_off,
               (unsigned long)want.bytes, want.n_tensors, (unsigned long)want.manifest_off, (unsigned long)want.data_off);
  std::vector<arena::Entry> man(P.size());
  if (!pread_all(man.data(), man.size() * sizeof(arena::Entry), h.manifest_off)) return say("manifest past the end");
  for (size_t i = 0; i < P.size(); ++i) {
    const arena::Entry e = arena_entry(P[i]);
    if (memcmp(&e, &man[i], sizeof e)) return say("manifest entry %zu (%.64s) differs from this build's plan (%s)", i, man[i].name, P[i].name.c_str());
  }
  std::unordered_map<uint32_t, size_t> smallest;  // final type -> the plan index of its smallest tensor
  for (size_t i = 0; i < P.size(); ++i) {
    auto it = smallest.find(P[i].t.type);
    if (it == smallest.end() || P[i].bytes < P[it->second].bytes) smallest[P[i].t.type] = i;
  }
  std::vector<uint8_t> want_b, got_b;
  for (const auto &kv : smallest) {
    const Planned &p = P[kv.second];
    expected_bytes(table, p, want_b);
    got_b.resize(p.bytes);
    if (!pread_all(got_b.data(), p.bytes, p.offset)) return say("%s past the end", p.name.c_str());
    if (memcmp(want_b.data(), got_b.data(), p.bytes))
      return say("%s (type %u): the arena's bytes differ from this build's conversion of the GGUF", p.name.c_str(), p.t.type);
  }
  if (types) *types = smallest.size();
  return "";
}

std::string Engine::check_arena(const mk_tensor_table *t, const char *path) {
  const int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return std::string(path) + ": " + strerror(errno);
  const TensorTable table(t);
  uint64_t bytes = 0;
  const std::vector<Planned> P = plan(table, &bytes);
  std::string why = arena_problem(table, table.shards(), P, bytes, fd, path, nullptr);
  close(fd);
  return why;
}

// Map an arena another engine made (a restarted service's, from systemd's fd store; a test's) after arena_problem
// finds nothing, read-only (the seals forbid a writable mapping), and register it. KFD charges every registration
// against one ~117 GiB system-memory limit, and a process that just exited gives its registration back only with
// KFD's deferred teardown, a moment later: a refused registration is retried for MK_ATTACH_WAIT_S (default 30) s.
void Engine::attach_arena(const char *path, const std::vector<Planned> &P) {
  const auto t0 = std::chrono::steady_clock::now();
  const int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) MK_DIE("%s: %s", path, strerror(errno));
  size_t types = 0;
  const std::string why = arena_problem(table_, shards_, P, arena_bytes_, fd, path, &types);
  if (!why.empty()) MK_DIE("%s", why.c_str());
  void *m = mmap(nullptr, arena_bytes_, PROT_READ, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) MK_DIE("mmap %s: %s", path, strerror(errno));
  arena_ = static_cast<uint8_t *>(m);
  arena_fd_ = fd;
  if (madvise(arena_, arena_bytes_, MADV_HUGEPAGE)) MK_DIE("madvise(MADV_HUGEPAGE) on %s: %s", path, strerror(errno));
  const char *w = getenv("MK_ATTACH_WAIT_S");
  const double wait_s = w ? atof(w) : 30.0;
  for (int tries = 1;; ++tries) {
    const hipError_t e = hipHostRegister(arena_, arena_bytes_, kArenaRegister);
    if (e == hipSuccess) break;
    (void)hipGetLastError();
    if (since_s(t0) > wait_s)
      MK_DIE("%s: hipHostRegister: %s after %d tries over %.1f s (KFD charges every registration against one "
             "system-memory limit: does another process still hold one of this arena?)", path, hipGetErrorString(e), tries, since_s(t0));
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  fprintf(stderr, "engine: attached %s: %.2f GiB of weights in %.1f s (%zu types spot-checked)\n", path,
          arena_bytes_ / double(1ull << 30), since_s(t0), types);
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
  if (!c->owns_kv) {  // a fork: its fixed state goes back to the free list (fork), its K/V and index keys are its parent's
    fork_free_.push_back(ForkState{c->gdn_state, c->gdn_conv, c->ple_conv});
    delete c;
    return;
  }
  (void)hipFree(c->gdn_state);
  (void)hipFree(c->gdn_conv);
  (void)hipFree(c->ple_conv);
  (void)hipFree(c->kv);
  (void)hipFree(c->idx_keys);
  (void)hipFree(c->idx_pooled);
  delete c;
}

Context *Engine::fork(const Context *s) {
  if (s->poisoned) MK_DIE("fork of a context poisoned by a spin-timeout (%u tokens): its state is half-written", s->n);
  if (s->pooled_n != s->n / kIdxRatio)
    MK_DIE("fork of a %u-token context whose pooled-key watermark is %u, want %u", s->n, s->pooled_n, s->n / kIdxRatio);
  auto *c = new Context(*s);
  const size_t n_gdn = 36;
  if (!fork_free_.empty()) {
    const ForkState f = fork_free_.back();
    fork_free_.pop_back();
    c->gdn_state = f.gdn_state;
    c->gdn_conv = f.gdn_conv;
    c->ple_conv = f.ple_conv;
    ++host_stats().fork_state_reuses;
  } else {
    MK_CHECK(hipMalloc(&c->gdn_state, sizeof(float) * n_gdn * kGdnVHeads * kGdnKDim * kGdnVDim));
    MK_CHECK(hipMalloc(&c->gdn_conv, sizeof(float) * n_gdn * (kGdnConv - 1) * kGdnConvDim));
    MK_CHECK(hipMalloc(&c->ple_conv, sizeof(float) * 9 * kHcDim));
    ++host_stats().fork_state_allocs;
  }
  MK_CHECK(hipMemcpy(c->gdn_state, s->gdn_state, sizeof(float) * n_gdn * kGdnVHeads * kGdnKDim * kGdnVDim, hipMemcpyDeviceToDevice));
  MK_CHECK(hipMemcpy(c->gdn_conv, s->gdn_conv, sizeof(float) * n_gdn * (kGdnConv - 1) * kGdnConvDim, hipMemcpyDeviceToDevice));
  MK_CHECK(hipMemcpy(c->ple_conv, s->ple_conv, sizeof(float) * 9 * kHcDim, hipMemcpyDeviceToDevice));
  c->owns_kv = false;  // K/V and index keys are the snapshot's; passes on c write only cells >= n
  return c;
}

Context *Engine::copy(const Context *s, uint32_t max_tokens) {
  if (s->poisoned) MK_DIE("copy of a context poisoned by a spin-timeout (%u tokens): its state is half-written", s->n);
  if (max_tokens < s->n) MK_DIE("copy of a %u-token context into capacity %u", s->n, max_tokens);
  if (s->pooled_n != s->n / kIdxRatio)
    MK_DIE("copy of a %u-token context whose pooled-key watermark is %u, want %u", s->n, s->pooled_n, s->n / kIdxRatio);
  Context *c = new_context(max_tokens);
  const size_t n_gdn = 36, n_att = 12;
  MK_CHECK(hipMemcpy(c->gdn_state, s->gdn_state, sizeof(float) * n_gdn * kGdnVHeads * kGdnKDim * kGdnVDim, hipMemcpyDeviceToDevice));
  MK_CHECK(hipMemcpy(c->gdn_conv, s->gdn_conv, sizeof(float) * n_gdn * (kGdnConv - 1) * kGdnConvDim, hipMemcpyDeviceToDevice));
  MK_CHECK(hipMemcpy(c->ple_conv, s->ple_conv, sizeof(float) * 9 * kHcDim, hipMemcpyDeviceToDevice));
  if (s->n) {
    // One row per attention layer: the first n cells, from a layer stride of s->max cells to one of max_tokens.
    const size_t kv_cell = sizeof(float) * 2 * kKvHeads * kHeadDim, ik_cell = sizeof(float) * kIdxDim;
    MK_CHECK(hipMemcpy2D(c->kv, kv_cell * max_tokens, s->kv, kv_cell * s->max, kv_cell * s->n, n_att, hipMemcpyDeviceToDevice));
    MK_CHECK(hipMemcpy2D(c->idx_keys, ik_cell * max_tokens, s->idx_keys, ik_cell * s->max, ik_cell * s->n, n_att,
                         hipMemcpyDeviceToDevice));
    // The pooled keys of the complete blocks, the same way (a layer's stride is max / 4 blocks): the copy owns them.
    const size_t pk_block = sizeof(float) * kIdxDim;
    if (s->pooled_n)
      MK_CHECK(hipMemcpy2D(c->idx_pooled, pk_block * (max_tokens / kIdxRatio), s->idx_pooled, pk_block * (s->max / kIdxRatio),
                           pk_block * s->pooled_n, n_att, hipMemcpyDeviceToDevice));
  }
  c->pooled_n = s->pooled_n;
  c->n = s->n;
  c->ple_hist[0] = s->ple_hist[0];
  c->ple_hist[1] = s->ple_hist[1];
  c->mask = s->mask;
  return c;
}

void Engine::set_mask(Context *c, uint32_t n, const uint32_t *lo, const uint32_t *hi) {
  if (c->poisoned) MK_DIE("set_mask on a context poisoned by a spin-timeout (%u tokens)", c->n);
  if (n > kAttnMaskMax) MK_DIE("set_mask: %u ranges, at most %u", n, kAttnMaskMax);
  for (uint32_t i = 0; i < n; ++i) {
    if (lo[i] >= hi[i]) MK_DIE("set_mask: range %u is [%u, %u): want lo < hi", i, lo[i], hi[i]);
    if (hi[i] > c->n) MK_DIE("set_mask: range %u is [%u, %u), past the context's %u tokens", i, lo[i], hi[i], c->n);
    if (i && lo[i] < hi[i - 1]) MK_DIE("set_mask: range %u [%u, %u) overlaps or precedes range %u [%u, %u)", i, lo[i], hi[i], i - 1,
                                      lo[i - 1], hi[i - 1]);
  }
  AttnCellMask m{};
  m.n = n;
  for (uint32_t i = 0; i < n; ++i) {
    m.lo[i] = lo[i];
    m.hi[i] = hi[i];
  }
  c->mask = m;
}

void Engine::poison_cells(Context *c, uint32_t lo, uint32_t hi, uint32_t what) {
  if (c->poisoned) MK_DIE("poison_cells on a context poisoned by a spin-timeout (%u tokens)", c->n);
  if (!c->owns_kv) MK_DIE("poison_cells on a fork: its K/V are its parent's");
  if (lo >= hi || hi > c->n) MK_DIE("poison_cells [%u, %u) of a %u-token context", lo, hi, c->n);
  if (what == 0 || what > 3) MK_DIE("poison_cells: what = %u, want 1 (K/V), 2 (index keys) or 3 (both)", what);
  const size_t n_att = 12, kv_cell = 2 * kKvHeads * kHeadDim;
  for (size_t l = 0; l < n_att; ++l) {
    if (what & 1) {
      std::vector<float> nan(size_t(hi - lo) * kv_cell, NAN);
      MK_CHECK(hipMemcpy(c->kv + (l * c->max + lo) * kv_cell, nan.data(), sizeof(float) * nan.size(), hipMemcpyHostToDevice));
    }
    if (what & 2) {
      std::vector<float> nan(size_t(hi - lo) * kIdxDim, NAN);
      MK_CHECK(hipMemcpy(c->idx_keys + (l * c->max + lo) * kIdxDim, nan.data(), sizeof(float) * nan.size(), hipMemcpyHostToDevice));
      // The cache holds pool(raw) for the complete blocks below pooled_n: a block with a poisoned cell pools to NaN in
      // every element (a NaN cell makes each element's sum NaN), so write exactly that. Not a watermark cut: a fork made
      // before this call holds its own pooled_n over the shared buffer and would keep trusting a clean cache.
      // Blocks past pooled_n are written too: a fork that extended the context may have cached its partial last block.
      const uint32_t b0 = lo / kIdxRatio, b1 = (hi + kIdxRatio - 1) / kIdxRatio;
      if (b0 < b1) {
        std::vector<float> nanb(size_t(b1 - b0) * kIdxDim, NAN);
        MK_CHECK(hipMemcpy(c->idx_pooled + (l * (c->max / kIdxRatio) + b0) * kIdxDim, nanb.data(), sizeof(float) * nanb.size(),
                           hipMemcpyHostToDevice));
      }
    }
  }
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
    // One row per attention layer: cells 0..n-1 of a layer (stride: the capacity) into a compact [n] block. A fork reads
    // its parent's buffers: cells below the fork point are the parent's; cells from there to the fork's n are its own passes'.
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
  if (!c->owns_kv) MK_DIE("%s into a fork: its K/V and index keys are its parent's", what);
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
  if (!c->owns_kv) MK_DIE("set_n(%u) on a fork: its K/V and index keys are its parent's", n);
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

void Engine::last_hidden(float *out) {
  if (last_segments_ > 1) MK_DIE("last_hidden after a pass of %u segments: pass_segments returned their hidden states", last_segments_);
  if (!last_error_.empty()) MK_DIE("last_hidden after a pass that failed: %s", last_error_.c_str());
  if (last_t_ == 0) MK_DIE("last_hidden before any pass");
  MK_CHECK(hipMemcpy(out, scratch("mixed", size_t(kMaxPassTokens) * kEmbd) + size_t(last_t_ - 1) * kEmbd,
                     sizeof(float) * kEmbd, hipMemcpyDeviceToHost));
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
// splits and their sums can share a barrier-free segment (stage_sched.h); ranges still in use when the buffer wraps
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

// hc up + hc_mix as one stage (docs/narrow.md N1; qmmh.h QmmArgs::yout). hc_mix's only input from its layer's work is `up`,
// which the hc up matmul writes a stage or two earlier (the inject matmul sits between them). The matmul moves to hc_mix's
// slot and its epilogue does hc_mix: mixed and inject come out of the one stage, `up` is never written, and one level
// (a grid barrier, a stage-argument fetch, a load round trip) goes. The inject matmul stays before: it writes z, which the
// fused stage reads. Every float op is the same ones in the same order (small.h hc_mix_elem; the matmul's accumulators are
// bit-for-bit the values it used to store), so the bits don't change. Any t: the epilogue handles every token tile.
static void fuse_hc_mix(std::vector<Stage> &S, std::vector<uint32_t> *layer_ends) {
  using namespace sched_detail;
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
    const Footprint fu = stage_footprint(S[j]);
    const uintptr_t xlo = uintptr_t(x.xn), xhi = xlo + fl(uint64_t(x.t) * 4 * x.e), ulo = uintptr_t(x.up), uhi = ulo + fl(uint64_t(x.t) * 4 * x.e);
    for (size_t k = size_t(j) + 1; k < m; ++k) {
      const Footprint fk = stage_footprint(S[k]);
      if (fk.fence) MK_DIE("fuse_hc_mix: a fence stage (op %u) between the up matmul and its hc_mix", S[k].op);
      for (const Access &a : fk.acc) {
        if (overlaps(a.lo, a.hi, ulo, uhi)) MK_DIE("fuse_hc_mix: stage %zu (op %u) touches up", k, S[k].op);
        if (a.write && overlaps(a.lo, a.hi, xlo, xhi)) MK_DIE("fuse_hc_mix: stage %zu (op %u) writes xn", k, S[k].op);
        if (a.write)
          for (const Access &r : fu.acc)
            if (!r.write && overlaps(a.lo, a.hi, r.lo, r.hi)) MK_DIE("fuse_hc_mix: stage %zu (op %u) writes the up matmul's input", k, S[k].op);
      }
    }
    // Nothing after the hc_mix reads `up` either (it is no longer written).
    for (size_t k = m + 1; k < S.size(); ++k) {
      const Footprint fk = stage_footprint(S[k]);
      if (fk.fence) continue;  // fences have no derived ranges; none reads this layer's up by construction
      bool rewritten = false, reads = false;
      for (const Access &a : fk.acc)
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

// MK_ELIDE=0 turns the barrier schedule off (stage_sched.h; A/B and bit checks): every stage in build order with a
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
  if (fuse_enabled() && fuse_hcmix_enabled() && fast_qmm_enabled()) fuse_hc_mix(S, layer_ends);
  // The schedule reorders within layers only (windows start at the layer ends), so the unfused path's per-layer
  // captures stay put; the unfused path runs the same order, one launch per stage.
  if (elide) schedule_stages(S, layer_ends ? *layer_ends : std::vector<uint32_t>{}, tail_enabled());
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
  // Forks of one context share its K/V and index keys, and every pass writes cells [n, n + t): two forks of one parent
  // would write the same cells, and a fork beside its parent would read cells the parent writes in the same pass.
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t j = i + 1; j < n; ++j)
      if (s[i].c->kv == s[j].c->kv || s[i].c->idx_keys == s[j].c->idx_keys) {
        snprintf(why, sizeof why, "segments %u and %u share a K/V buffer (forks of one context, a fork and its parent, or one context twice)", i, j);
        return why;
      }
  return "";
}

PassOut Engine::pass(Context *c, const int32_t *tokens, uint32_t t, bool want_logits, bool want_layers, bool fused,
                     bool want_routes, float *logits_out, bool want_gdn_ab) {
  const Segment s{c, tokens, t};
  return run_pass(&s, 1, want_logits, want_layers, fused, want_routes, logits_out, nullptr, want_gdn_ab);
}

PassOut Engine::pass_segments(const Segment *s, uint32_t n, bool want_logits, bool fused, float *logits_out, float *hidden_out) {
  return run_pass(s, n, want_logits, false, fused, false, logits_out, hidden_out);
}

// A fused pass that stopped on a spin-timeout (set_recover_spin_timeouts). Nothing it wrote is used: its contexts are
// poisoned and not advanced (run_pass advances n and the PLE history only after a clean pass). A held parent of a fork
// it extended is intact: the fork has its own recurrent state and conv history (Engine::fork), and its pass writes K/V
// and index keys only in cells >= the fork's n, which the parent never reads. The control block's barrier count may be
// left mid-count, so end_pass marked it dirty and the next begin_pass does the full reset(). The event ring may hold a
// partial pass: it is dropped and made again on its next use.
PassOut Engine::failed_pass(const Segment *seg, uint32_t ns, const char *why, double ms) {
  for (uint32_t i = 0; i < ns; ++i) seg[i].c->poisoned = true;
  last_error_ = std::string(why) + "; the pass's " + std::to_string(ns) + " context(s) are poisoned";
  last_t_ = 0;
  last_segments_ = 0;
  ring_.reset();
  fprintf(stderr, "engine: %s\n", last_error_.c_str());
  PassOut po;
  po.error = last_error_;
  po.ms = ms;
  return po;
}

PassOut Engine::run_pass(const Segment *seg, uint32_t ns, bool want_logits, bool want_layers, bool fused, bool want_routes,
                         float *logits_out, float *hidden_out, bool want_gdn_ab) {
  for (uint32_t i = 0; i < ns; ++i)
    if (seg[i].c && seg[i].c->poisoned)
      MK_DIE("pass on a context poisoned by a spin-timeout (segment %u, %u tokens): its state is half-written", i, seg[i].c->n);
  const std::string problem = segments_problem(seg, ns);
  if (!problem.empty()) MK_DIE("pass: %s", problem.c_str());
  if (fused)  // the megakernel doesn't run the masked selection (attn_mask.h): lab passes on masked contexts are unfused
    for (uint32_t i = 0; i < ns; ++i)
      if (seg[i].c->mask.n)
        MK_DIE("fused pass on a context with an attention cell mask (segment %u, %u ranges): masked contexts run unfused only",
               i, seg[i].c->mask.n);
  if (ns > 1 && want_logits && !logits_out) MK_DIE("pass of %u segments: want_logits needs logits_out", ns);
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
  const bool timing = getenv("MK_STAGE_TIMING") != nullptr;
  auto since = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
  {
    // Each segment's n-gram rows hash its own context's history and tokens.
    std::vector<float> pe(size_t(t) * kEmbd);
    for (uint32_t i = 0, r0 = 0; i < ns; r0 += seg[i].t, ++i) ple_rows(seg[i].c, seg[i].tokens, seg[i].t, pe.data() + size_t(r0) * kEmbd);
    if (timing) fprintf(stderr, "pass: ple rows at %.1f ms\n", since());
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
  if (timing) {
    fprintf(stderr, "pass: %zu stages built and uploaded at %.1f ms (Stage is %zu bytes)\n", S.size(), since(), sizeof(Stage));
    const EmbedArgs &ea = S[0].embed;
    fprintf(stderr, "stage 0: op %u w %p ldw %lu k %u ids %p t %u out %p; d_stages %p\n", S[0].op, (const void *)ea.w,
            (unsigned long)ea.ldw, ea.k, (const void *)ea.ids, ea.t, (void *)ea.out, (void *)d_stages_);
    Stage back;
    MK_CHECK(hipMemcpy(&back, d_stages_, sizeof back, hipMemcpyDeviceToHost));
    fprintf(stderr, "stage 0 read back from the device: op %u k %u t %u (same bytes: %d)\n", back.op, back.embed.k,
            back.embed.t, memcmp(&back, &S[0], sizeof back) == 0);
  }
  if (want_layers) po.layer_h.assign(size_t(kLayers) * t * kHcDim, NAN);
  if (want_routes) po.routes.assign(size_t(kLayers) * t * kTopK, -1);
  // The router's top-10 ids: one buffer every layer's router overwrites (blocks.hip add_moe), so a layer's routes
  // are read at its end, like its h; nothing between a router and its layer's end writes them again.
  const int32_t *tidx = reinterpret_cast<const int32_t *>(scratch("moe_tidx", size_t(kMaxPassTokens) * kTopK));
  float *h = scratch("h", size_t(kMaxPassTokens) * kHcDim);
  // The GDN gate projections (lab item 1, the memory horizon): the raw a and b every GDN layer's core reads
  // (blocks.hip add_gdn; one scratch pair every GDN layer overwrites, so read at the layer's end like h and the routes),
  // and the layers' ssm_a and dt_bias, so the host forms g and beta with the core's own formula. Copies only: no stage
  // changes, so the pass's bits are the same with or without them.
  constexpr uint32_t kGdnLayers = 36;
  const float *gdn_a = scratch("gdn_a", size_t(kMaxPassTokens) * kGdnVHeads);
  const float *gdn_b = scratch("gdn_b", size_t(kMaxPassTokens) * kGdnVHeads);
  if (want_gdn_ab) {
    po.gdn_ab.assign(size_t(kGdnLayers) * t * 2 * kGdnVHeads, NAN);
    po.gdn_params.assign(size_t(kGdnLayers) * 2 * kGdnVHeads, NAN);
    for (uint32_t il = 0, g = 0; il < kLayers; ++il) {
      if (il % 4 == 3) continue;  // every 4th layer is full attention (blocks.hip gdn_index)
      MK_CHECK(hipMemcpy(po.gdn_params.data() + size_t(g) * 2 * kGdnVHeads, layers_[il].ssm_a.f32(), sizeof(float) * kGdnVHeads, hipMemcpyDeviceToHost));
      MK_CHECK(hipMemcpy(po.gdn_params.data() + (size_t(g) * 2 + 1) * kGdnVHeads, layers_[il].dt_bias.f32(), sizeof(float) * kGdnVHeads, hipMemcpyDeviceToHost));
      ++g;
    }
    if (po.gdn_params.size() != size_t(kGdnLayers) * 2 * kGdnVHeads) MK_DIE("gdn_ab: counted GDN layers wrong");
  }
  dispatches_ = 0;
  stage_ms_.clear();
  stage_op_.clear();
  if (!fused) {
    uint32_t li = 0;
    const uint32_t grid = getenv("MK_UNFUSED_GRID") ? uint32_t(atoi(getenv("MK_UNFUSED_GRID"))) : kUnfusedGrid;
    // The fused pass's rails, for the unfused one (goal 03 C): the control block and the watchdog around the launches.
    // k_stage checks abort/status on entry, so once the deadline (or the memory floor) sets abort every remaining launch
    // drains in microseconds. It stops a long or runaway SEQUENCE of stages between stages; one stage that hangs inside an op
    // can't be stopped (ops have no polls), and then only the host watchdog's kill remains, which waits for the kernel.
    cb_.begin_pass();
    Control *ctl = cb_.device();
    if (!launched_ev_) {
      MK_CHECK(hipEventCreate(&launched_ev_));
      ++host_stats().events_created;
    }
    wd_.begin();
    const auto tw = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < S.size(); ++i) {
      const auto ts = std::chrono::steady_clock::now();
      k_stage<<<grid, kMkBlock>>>(d_stages_ + i, ctl, i);
      MK_CHECK(hipGetLastError());
      // After every launch (the same event, re-recorded): the watchdog's "a kernel is still running" query then
      // looks at the newest launch; armed once, after the first record.
      MK_CHECK(hipEventRecord(launched_ev_, 0));
      if (i == 0) wd_.arm(launched_ev_);
      ++dispatches_;
      if (timing) {
        MK_CHECK(hipDeviceSynchronize());
        stage_op_.push_back(S[i].op);
        stage_ms_.push_back(float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count()));
        if (getenv("MK_STAGE_TIMING_PRINT")) fprintf(stderr, "stage %4u op %2u layer %2u: %8.3f ms\n", i, S[i].op, S[i].layer,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count());
      }
      if (li < ends.size() && ends[li] == i + 1) {
        if (want_layers) MK_CHECK(hipMemcpy(po.layer_h.data() + size_t(li) * t * kHcDim, h, sizeof(float) * t * kHcDim, hipMemcpyDeviceToHost));
        if (want_routes) MK_CHECK(hipMemcpy(po.routes.data() + size_t(li) * t * kTopK, tidx, sizeof(int32_t) * t * kTopK, hipMemcpyDeviceToHost));
        if (want_gdn_ab && li % 4 != 3) {
          const uint32_t g = li - (li + 1) / 4;  // gdn_index(li)
          float *dst = po.gdn_ab.data() + size_t(g) * t * 2 * kGdnVHeads;
          // [t][2][48]: a's row then b's row, per token (a strided copy each)
          const size_t pitch = sizeof(float) * 2 * kGdnVHeads, row = sizeof(float) * kGdnVHeads;
          MK_CHECK(hipMemcpy2D(dst, pitch, gdn_a, row, row, t, hipMemcpyDeviceToHost));
          MK_CHECK(hipMemcpy2D(dst + kGdnVHeads, pitch, gdn_b, row, row, t, hipMemcpyDeviceToHost));
        }
        ++li;
      }
    }
    MK_CHECK(hipDeviceSynchronize());
    wd_.finish();
    cb_.end_pass();
    if (cb_.status() != kOk)
      MK_DIE("unfused pass stopped: %s at stage %u of %zu (wg %u) after %.1f ms", status_name(cb_.status()), cb_.host()->status_site,
             S.size(), cb_.host()->status_wg,
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count());
  } else {
    if (want_layers || want_routes || want_gdn_ab) MK_DIE("per-layer capture (layers, routes, gdn_ab) needs the unfused path");
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
    // The grid is set by occupancy, and the megakernel's static LDS is the sum over every op: past ~32 KiB
    // only one workgroup fits per WGP and the grid halves to 20 (S1 attempt 2: 41 KiB, grid 20, and the
    // same work took 1110 ms instead of 810). Recorded per pass so tools can print it.
    mega_grid_ = uint32_t(grid);
    // Event ring: one lane per workgroup, room for every stage of the pass, drained after it.
    const bool events = getenv("MK_MEGA_EVENTS") != nullptr;
    EventRingView ev{};
    constexpr uint32_t kRingCap = 4096;
    if (events) {
      if (n > kRingCap) MK_DIE("MK_MEGA_EVENTS: %u stages exceed the ring's %u records per lane", n, kRingCap);
      if (!ring_ || ring_->view().lanes != uint32_t(grid)) ring_ = std::make_unique<EventRing>(uint32_t(grid), kRingCap);
      ev = ring_->view();
    }
    det_op_.clear();
    det_span_.clear();
    det_gap_.clear();
    det_work_mean_.clear();
    det_hwid_.assign(size_t(grid), 0);
    det_bar_.resize(n);
    // A tail stage's release is the second half of the barrier its predecessor arrived at (mega.hip): one barrier for the pair,
    // counted on the predecessor.
    for (uint32_t i = 0; i < n; ++i) det_bar_[i] = (S[i].sched & (kSchedNoBarrier | kSchedTail)) ? 0 : 1;
    void *args[] = {&ds, &n, &ctl, &bar, &skip, &ev};
    // Run 2: the occupancy API said 60 co-resident workgroups, but a 60-workgroup launch deadlocked at its
    // first barrier (spin timeout) while the kernel had 1.4 KiB/lane of scratch, which likely caps waves in
    // flight below what the API reports. Hence the no-scratch test. Every op is grid-independent, so any
    // grid gives the same bits. PLAN.md Problems.
    if (getenv("MK_STAGE_TIMING")) fprintf(stderr, "megakernel: grid %d (occupancy max %d)\n", grid, max_coop_grid(mega_kernel(), kMkBlock, 0));
    wd_.begin();
    // MK_MEGA_PER_STAGE=1 (diagnostic, tools/fuse_gap.py): the megakernel's own code launched once per stage
    // (n = 1, same grid), timed on the host like the unfused path. Against k_stage per stage it isolates code
    // generation; against the fused pass, what one launch changes (barriers, caches, clocks).
    const bool per_stage = getenv("MK_MEGA_PER_STAGE") != nullptr;
    if (per_stage && (events || skip != UINT32_MAX)) MK_DIE("MK_MEGA_PER_STAGE: not with events or a skipped barrier");
    const uint32_t launches = per_stage ? n : 1;
    uint32_t one = 1;
    const Stage *dsi = ds;
    void *args1[] = {&dsi, &one, &ctl, &bar, &skip, &ev};
    if (!launched_ev_) {
      MK_CHECK(hipEventCreate(&launched_ev_));
      ++host_stats().events_created;
    }
    for (uint32_t i = 0; i < launches; ++i) {
      const auto ts = std::chrono::steady_clock::now();
      dsi = ds + i;
      launch_coop(mega_kernel(), grid, kMkBlock, per_stage ? args1 : args, 0, 0);
      MK_CHECK(hipEventRecord(launched_ev_, 0));  // every launch (per_stage has n of them), the same event
      if (i == 0) wd_.arm(launched_ev_);
      ++dispatches_;
      if (per_stage) {
        MK_CHECK(hipDeviceSynchronize());
        if (cb_.status() != kOk) break;
        stage_op_.push_back(S[i].op);
        stage_ms_.push_back(float(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count()));
      }
    }
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
    if (events) {
      if (uint32_t(grid) != ev.lanes) MK_DIE("MK_MEGA_EVENTS: grid %d, ring has %u lanes", grid, ev.lanes);
      std::vector<uint64_t> b(size_t(n) * grid, 0), e(size_t(n) * grid, 0);
      std::vector<uint32_t> next(grid, 0), op(n, 0);
      uint64_t got = ring_->drain([&](const EventRecord &r) {
        if (r.wg >= uint32_t(grid) || r.stage >= n) MK_DIE("event ring: record wg %u stage %u out of range", r.wg, r.stage);
        if (r.stage != next[r.wg]++) MK_DIE("event ring: wg %u stage %u out of order", r.wg, r.stage);
        if (r.t_end < r.t_begin) MK_DIE("event ring: wg %u stage %u ends before it begins", r.wg, r.stage);
        b[size_t(r.stage) * grid + r.wg] = r.t_begin;
        if (r.stage == 0) det_hwid_[r.wg] = r.payload[1];
        e[size_t(r.stage) * grid + r.wg] = r.t_end;
        op[r.stage] = r.payload[0];
      });
      if (got != uint64_t(n) * grid) MK_DIE("event ring: %lu records, want %u stages x %d workgroups", (unsigned long)got, n, grid);
      constexpr double kMsPerTick = 1e3 / double(kWallHz);
      auto bmin = [&](uint32_t i) { return *std::min_element(b.begin() + size_t(i) * grid, b.begin() + size_t(i + 1) * grid); };
      auto emax = [&](uint32_t i) { return *std::max_element(e.begin() + size_t(i) * grid, e.begin() + size_t(i + 1) * grid); };
      for (uint32_t i = 0; i < n; ++i) {
        double work = 0;
        for (int g = 0; g < grid; ++g) work += double(e[size_t(i) * grid + g] - b[size_t(i) * grid + g]);
        det_op_.push_back(op[i]);
        det_span_.push_back(float(double(emax(i) - bmin(i)) * kMsPerTick));
        det_gap_.push_back(i + 1 < n ? float(double(int64_t(bmin(i + 1) - emax(i))) * kMsPerTick) : 0.f);
        det_work_mean_.push_back(float(work / grid * kMsPerTick));
      }
      det_kernel_ms_ = double(emax(n - 1) - bmin(0)) * kMsPerTick;
      det_b_.swap(b);
      det_e_.swap(e);
    }
  }
  if (want_logits) {
    float *dst = logits_out;
    if (!dst) {
      po.logits.resize(kVocab);
      dst = po.logits.data();
      ++host_stats().logits_staged;
    }
    MK_CHECK(hipMemcpy(dst, scratch("logits", size_t(kMaxSegments) * kVocab), sizeof(float) * kVocab * ns, hipMemcpyDeviceToHost));
    ++host_stats().logits_dtoh;
  }
  if (hidden_out) {
    const float *mixed = scratch("mixed", size_t(kMaxPassTokens) * kEmbd);
    for (uint32_t i = 0; i < ns; ++i)
      MK_CHECK(hipMemcpy(hidden_out + size_t(i) * kEmbd, mixed + size_t(head_rows[i]) * kEmbd, sizeof(float) * kEmbd,
                         hipMemcpyDeviceToHost));
  }
  last_t_ = t;
  last_segments_ = ns;
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
