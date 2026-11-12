// SPDX-License-Identifier: MIT
// Runtime rails for the megakernel: control block, bounded spins, grid barrier, event ring,
// watchdog. Host and device declarations; device code lives in device.h.
#pragma once
#include <hip/hip_runtime.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#define HIP_CHECK(expr)                                                                      \
  do {                                                                                      \
    hipError_t hip_e_ = (expr);                                                              \
    if (hip_e_ != hipSuccess) {                                                              \
      fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, hipGetErrorString(hip_e_)); \
      abort();                                                                              \
    }                                                                                       \
  } while (0)

#define HIP_DIE(...)                                        \
  do {                                                     \
    fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);        \
    fprintf(stderr, __VA_ARGS__);                          \
    fputc('\n', stderr);                                   \
    abort();                                               \
  } while (0)

// A C ABI that takes a buffer takes its byte size too, and dies unless it is exactly what the layout needs (A6;
// PLAN.md Problems 2026-09-29: a tile pool sized at 110 B a row-superblock instead of 112 was written past). Checked on
// the host before the first HIP call of the entry point. `op` is the entry point's symbol, `buf` the parameter's name.
#define HIP_CHECK_BYTES(op, buf, given, need)                                                         \
  do {                                                                                              \
    const uint64_t hip_given_ = uint64_t(given), hip_need_ = uint64_t(need);                                  \
    if (hip_given_ != hip_need_)                                                                             \
      HIP_DIE("%s: buffer '%s' is %llu B, the layout needs %llu B", op, buf, (unsigned long long)hip_given_, \
             (unsigned long long)hip_need_);                                                            \
  } while (0)

namespace qwen38 {

// Why a kernel stopped early. The first nonzero status wins (atomicCAS from 0).
enum Status : uint32_t {
  kOk = 0,
  kAborted = 1,       // the host set Control::abort
  kSpinTimeout = 2,   // a bounded spin ran out of time
  // Device traps (trap.h: HIP_TRAP records one of these, with the site, then traps; the host reports it). First value
  // kTrapFirst; status_name() knows each.
  kTrapActRange = 16,  // qmmh: an activation beyond f16's range, or NaN/inf (crash, never inf)
  kTrapBadArgs,        // an op's precondition on its arguments: shape, alignment, tile, split, fused-staging setup
  kTrapMoeRoute,       // moe router: nothing selectable (NaN probabilities)
  kTrapMoeGroup,       // moe group: too many assignments, or an expert index out of range
  kTrapMoeTotal,       // moe_check_total: an item count beyond what T * 10 assignments allow (corrupt group data)
  kTrapAttnSelect,     // attention selection: a NaN/negative block score, or the emission disagreeing with the radix select
  kTrapAttnNoSel,      // attention attend: a token with no selected cells (causal self-visibility says it can't happen)
  kTrapBadOp,          // a dispatcher's default: an op or weight type it doesn't know
  kTrapLast = kTrapBadOp,
};
constexpr uint32_t kTrapFirst = kTrapActRange;
const char *status_name(uint32_t s);  // async-signal-safe (the trap report uses it)

// Host-written, device-read. Lives in coherent host memory so the host can set abort while the
// kernel runs, and read status after (or during) it.
struct Control {
  uint32_t abort;       // nonzero: every spin returns and the kernel exits
  uint32_t status;      // first failure (Status)
  uint32_t status_wg;   // workgroup that recorded it
  uint32_t status_site; // caller-chosen site id
};

// ---- device trap report (trap.h) ----
// Idempotent, host. Allocates the process's trap record (a Control in coherent host memory), binds it into every
// translation unit's g_trap_ctl that has registered so far, and installs a SIGABRT handler (once) that reports a
// recorded trap to stderr before the abort proceeds (and chains to any handler installed before it). Call it before
// the first launch that can trap; call it again after loading another library that has device code (it binds only
// the translation units it hasn't yet). Returns the record.
Control *trap_sink_install();
bool trap_sink_installed();
// The report line for a record, into buf (async-signal-safe, no allocation); returns its length, or 0 if the record
// holds no trap. Exposed so tests can format a record without killing a process.
size_t trap_report_format(const Control &c, char *buf, size_t cap);

// Device-memory barrier state (agent-scope atomics, cached in L2, not host memory).
struct Barrier {
  uint32_t count;
  uint32_t generation;
};

// Default bound for a spin: a pass never waits this long at one barrier. 100 MHz ticks.
// Kept at 1 s, well under the gfx ring timeout: on this driver MES can't preempt a spinning
// queue, and run 1's 20 s spin starved the desktop's ring into a MODE2 reset (PLAN.md Problems).
constexpr uint64_t kWallHz = 100'000'000;
constexpr uint64_t kDefaultSpinTicks = kWallHz;  // 1 s
constexpr uint64_t kMaxSpinTicks = kWallHz;      // no spin anywhere may be bounded above this

// ---- host cost counters ----
// What the host does per pass, counted so tests can assert it (tests/engine/test_host_stats.py, exposed by the C ABI's
// mk_host_stats): a cost that is paid once must stay once, and a counter that moves per pass is a regression.
struct HostStats {
  std::atomic<uint64_t> watchdog_threads{0};   // std::threads spawned by Watchdog
  std::atomic<uint64_t> occupancy_queries{0};  // hipOccupancyMaxActiveBlocksPerMultiprocessor calls
  std::atomic<uint64_t> events_created{0};     // hipEventCreate calls by the engine's fused pass
  std::atomic<uint64_t> cb_memsets{0};         // ControlBlock barrier hipMemsets
  std::atomic<uint64_t> cb_syncs{0};           // ControlBlock hipDeviceSynchronize calls
  std::atomic<uint64_t> logits_dtoh{0};        // device -> host copies of the logits
  std::atomic<uint64_t> logits_staged{0};      // logits returned through a PassOut vector (the caller then copies them again)
  std::atomic<uint64_t> fork_state_allocs{0};  // fork fixed states hipMalloc'd (Engine::fork: the free list was empty)
  std::atomic<uint64_t> fork_state_reuses{0};  // fork fixed states taken from the free list
  std::atomic<uint64_t> pool_blocks{0};        // (QSA layer, block) pairs the pool stages built (passes and restore_n) will write
};
HostStats &host_stats();

// ---- event ring ----
// 64-byte records, one lane per workgroup, in coherent host memory. The device writes a record,
// then publishes head with a system-scope release; the host drains up to head and publishes tail.
struct EventRecord {
  uint32_t seq;      // per-lane sequence number, from 0
  uint32_t pass;
  uint16_t stage;
  uint16_t layer;
  uint32_t wg;
  uint64_t t_begin;  // wall_clock64(), 100 MHz
  uint64_t t_end;
  uint32_t payload[8];
};
static_assert(sizeof(EventRecord) == 64);

struct alignas(128) RingLaneHeader {
  uint64_t head;  // records published by the device
  uint64_t tail;  // records consumed by the host
};

struct EventRingView {  // passed to kernels by value
  EventRecord *records;     // [lanes][capacity]
  RingLaneHeader *headers;  // [lanes]
  uint32_t lanes;
  uint32_t capacity;        // power of two
};

class EventRing {
 public:
  EventRing(uint32_t lanes, uint32_t capacity);
  ~EventRing();
  EventRing(const EventRing &) = delete;
  EventRingView view() const { return v_; }
  // Copies every published, unconsumed record of every lane to sink(record) and advances tail.
  // Returns the number drained.
  template <class F>
  uint64_t drain(F &&sink);

 private:
  EventRingView v_{};
};

template <class F>
uint64_t EventRing::drain(F &&sink) {
  uint64_t n = 0;
  for (uint32_t l = 0; l < v_.lanes; ++l) {
    RingLaneHeader *h = &v_.headers[l];
    uint64_t head = __atomic_load_n(&h->head, __ATOMIC_ACQUIRE);
    uint64_t tail = __atomic_load_n(&h->tail, __ATOMIC_RELAXED);
    for (; tail < head; ++tail, ++n) sink(v_.records[size_t(l) * v_.capacity + (tail & (v_.capacity - 1))]);
    __atomic_store_n(&h->tail, tail, __ATOMIC_RELEASE);
  }
  return n;
}

// ---- control block ----
class ControlBlock {
 public:
  ControlBlock();
  ~ControlBlock();
  ControlBlock(const ControlBlock &) = delete;
  Control *host() { return c_; }  // same pointer on this APU; coherent host memory
  Control *device() { return c_; }
  Barrier *barrier() { return b_; }
  void reset();  // zero control and barrier state before a launch (host, stream idle): always a memset and a sync
  // The per-pass form of reset() for a host that launches pass after pass (Engine::pass): zero the host-side control
  // words with plain stores, and touch the device-side barrier only if the last pass left it dirty. The barrier's
  // generation never needs a reset: grid_barrier reads it on entry and waits for it to change, so any value works.
  // Its count is back at 0 whenever every workgroup of a launch got through every barrier (the last to arrive
  // stores 0 before it bumps the generation), and a workgroup leaves a barrier early only through spin_until, or as
  // the last arriver that finds a status or an abort and doesn't release, which record a nonzero status (device.h). So "status was kOk at the end of the last launch" (end_pass)
  // is exactly "count is 0", and a launch that failed gets the full reset(). The abort word is cleared on every
  // call: the watchdog may set it after a clean launch ended. begin_pass() before the launch, end_pass() after it
  // has completed (the stream is synchronized), so a stale word cannot reach the next kernel and a stale count
  // cannot hang it.
  void begin_pass();
  void end_pass();
  void request_abort() { __atomic_store_n(&c_->abort, 1u, __ATOMIC_RELEASE); }
  uint32_t status() const { return __atomic_load_n(&c_->status, __ATOMIC_ACQUIRE); }

 private:
  Control *c_ = nullptr;
  Barrier *b_ = nullptr;
  bool clean_ = false;  // the device barrier's count is known to be 0 (see begin_pass)
};

// ---- cooperative launch ----
// The largest co-resident grid for this kernel, block size and dynamic LDS. It is a function of the compiled kernel
// and the device alone, so the occupancy query runs once per (device, kernel, block, lds) and is remembered.
int max_coop_grid(const void *kernel, int block, size_t lds);
// Launches cooperatively; dies if grid exceeds max_coop_grid (we never rely on the runtime alone).
void launch_coop(const void *kernel, int grid, int block, void **args, size_t lds, hipStream_t s);

// ---- watchdog ----
// A host thread that watches one launch. Past the deadline, or when MemAvailable drops below
// the floor, it sets Control::abort. Every device spin is bounded (<= kMaxSpinTicks), so a
// kernel ends on its own; the process is killed (SIGKILL) only if the host still hasn't called
// finish() after the grace period, which must outlast the longest spin bound, AND the launch's
// stream is idle. Killing a process with a live kernel is NOT safe on this driver (run 1), so the
// kill is for a hung host, never the way a kernel is stopped: while a kernel still runs, the
// watchdog waits for it (run 2 added the stream check after a kill landed mid-kernel).
struct WatchdogConfig {
  double deadline_s = 30.0;
  double grace_s = 3.0;  // > kMaxSpinTicks plus a pass's own run time
  uint64_t mem_floor_bytes = 12ull << 30;
  double poll_s = 0.05;
};

// One thread for the life of the object (a fused decode pass is ~7 ms, and spawning a thread per pass cost ~0.1 ms of
// it): begin() starts a watch and finish() ends it, each pass. finish() returns only once the thread is out of its
// watch, so after it nothing of the watch can run: no abort set, no kill. `armed` (the default) begins the first
// watch at construction, for a caller that watches one launch (the m0 tests).
class Watchdog {
 public:
  Watchdog(ControlBlock &cb, WatchdogConfig cfg, bool armed = true);
  ~Watchdog();  // calls finish(), then joins the thread
  Watchdog(const Watchdog &) = delete;
  void begin();   // start watching a new launch: clears the reason and the launch event, restarts the deadline
  void finish();  // the launch completed; stop watching (a no-op while not watching)
  // An event recorded right after the launch: while it hasn't completed, a kernel is running and
  // the watchdog never kills (hipStreamQuery on the null stream blocked from this thread, run 2).
  void arm(hipEvent_t launched) { launched_.store(launched, std::memory_order_release); }
  // Why abort was requested in the last watch ("" if not): "deadline" or "memory". Kept until the next begin().
  std::string reason() const;

 private:
  enum class State { kIdle, kWatching, kStopping };
  void run();
  void watch(std::unique_lock<std::mutex> &lk);
  ControlBlock &cb_;
  WatchdogConfig cfg_;
  std::atomic<int> reason_{0};
  std::atomic<hipEvent_t> launched_{nullptr};
  // State, quit_ and both condition variables are guarded by m_. run() waits on cv_ between polls, so begin(),
  // finish() and the destructor wake it at once (a plain sleep_for made every fused pass wait up to one poll interval;
  // tests/m0/test_watchdog.hip case "prompt"); finish() waits on idle_cv_ for the thread to leave its watch.
  std::mutex m_;
  std::condition_variable cv_, idle_cv_;
  State state_ = State::kIdle;
  bool quit_ = false;
  std::thread t_;
};

// MemAvailable from /proc/meminfo, in bytes. Dies if it can't be read.
uint64_t mem_available_bytes();

}  // namespace qwen38
