// SPDX-License-Identifier: MIT
#include "rt.h"

#define HIP_TRAP_NO_BINDER  // rt.cpp has no device code: nothing of its own to bind (trap.h)
#include "trap.h"

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <tuple>

namespace mk {

const char *status_name(uint32_t s) {
  switch (s) {
    case kOk: return "ok";
    case kAborted: return "aborted";
    case kSpinTimeout: return "spin-timeout";
    case kTrapActRange: return "trap-activation-range";
    case kTrapBadArgs: return "trap-bad-args";
    case kTrapMoeRoute: return "trap-moe-route";
    case kTrapMoeGroup: return "trap-moe-group";
    case kTrapMoeTotal: return "trap-moe-total";
    case kTrapAttnSelect: return "trap-attn-select";
    case kTrapAttnNoSel: return "trap-attn-no-selection";
    case kTrapBadOp: return "trap-bad-op";
  }
  return "unknown";
}

const char *trap_file_name(uint32_t file) {
  switch (file) {
    case kTrapFileQmmh: return "qmmh.h";
    case kTrapFileQmm: return "qmm.h";
    case kTrapFileQmmOracle: return "qmm_oracle.h";
    case kTrapFileSmall: return "small.h";
    case kTrapFileMoe: return "moe.h";
    case kTrapFileMoeHip: return "moe.hip";
    case kTrapFileAttn: return "attn.h";
    case kTrapFileStage: return "stage.h";
  }
  return "unknown-file";
}

// ---- device trap report (trap.h) ----
namespace {
constexpr int kMaxBinders = 64;
std::atomic<void (*)(Control *)> g_binders[kMaxBinders];  // filled at static init, one per translation unit
std::atomic<int> g_nbinders{0};
std::atomic<Control *> g_trap_record{nullptr};  // read by the signal handler
struct sigaction g_old_abrt;
std::mutex g_sink_m;
int g_bound = 0;  // binders already run

// Append text / a decimal number to buf (no allocation, no stdio: the signal handler uses these).
struct Out {
  char *b;
  size_t cap, n = 0;
  void str(const char *s) {
    while (*s && n + 1 < cap) b[n++] = *s++;
  }
  void num(uint64_t v) {
    char t[24];
    int k = 0;
    do t[k++] = char('0' + v % 10), v /= 10;
    while (v);
    while (k && n + 1 < cap) b[n++] = t[--k];
  }
};

void on_sigabrt(int sig, siginfo_t *info, void *ctx) {
  Control *c = g_trap_record.load(std::memory_order_acquire);
  if (c != nullptr) {
    char buf[192];
    const size_t n = trap_report_format(*c, buf, sizeof buf);
    if (n) {
      ssize_t w = write(2, buf, n);
      (void)w;
    }
  }
  // Chain: a handler installed before ours runs next; with none (or the default), returning from a SIGABRT handler
  // lets abort() carry on to its default action.
  if (g_old_abrt.sa_flags & SA_SIGINFO) {
    if (g_old_abrt.sa_sigaction) g_old_abrt.sa_sigaction(sig, info, ctx);
  } else if (g_old_abrt.sa_handler != SIG_DFL && g_old_abrt.sa_handler != SIG_IGN && g_old_abrt.sa_handler) {
    g_old_abrt.sa_handler(sig);
  }
}
}  // namespace

void trap_register_binder(void (*bind)(Control *)) {
  const int i = g_nbinders.fetch_add(1);
  if (i >= kMaxBinders) HIP_DIE("trap: more than %d translation units with device traps", kMaxBinders);
  g_binders[i].store(bind);
}

size_t trap_report_format(const Control &c, char *buf, size_t cap) {
  const uint32_t status = __atomic_load_n(&c.status, __ATOMIC_ACQUIRE);
  if (status < kTrapFirst || status > kTrapLast) return 0;
  const uint32_t site = __atomic_load_n(&c.status_site, __ATOMIC_ACQUIRE), wg = __atomic_load_n(&c.status_wg, __ATOMIC_RELAXED);
  Out o{buf, cap};
  o.str("device trap: ");
  o.str(status_name(status));
  o.str(" at site ");
  o.str(trap_file_name(site >> 16));
  o.str(":");
  o.num(site & 0xffff);
  o.str(" (wg ");
  o.num(wg);
  o.str(")\n");
  return o.n;
}

Control *trap_sink_install() {
  std::lock_guard<std::mutex> lk(g_sink_m);
  Control *c = g_trap_record.load();
  if (c == nullptr) {
    HIP_CHECK(hipHostMalloc((void **)&c, sizeof(Control), hipHostMallocCoherent));
    c->abort = c->status = c->status_wg = c->status_site = 0;
    struct sigaction sa {};
    sa.sa_sigaction = on_sigabrt;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGABRT, &sa, &g_old_abrt) != 0) HIP_DIE("trap: sigaction(SIGABRT) failed");
    g_trap_record.store(c, std::memory_order_release);
  }
  const int n = g_nbinders.load();
  for (; g_bound < n; ++g_bound) g_binders[g_bound].load()(c);
  return c;
}

HostStats &host_stats() {
  static HostStats s;
  return s;
}

// ---- event ring ----
EventRing::EventRing(uint32_t lanes, uint32_t capacity) {
  if (lanes == 0 || capacity == 0 || (capacity & (capacity - 1)) != 0)
    HIP_DIE("event ring needs lanes > 0 and a power-of-two capacity, got %u x %u", lanes, capacity);
  HIP_CHECK(hipHostMalloc((void **)&v_.records, sizeof(EventRecord) * lanes * capacity, hipHostMallocCoherent));
  HIP_CHECK(hipHostMalloc((void **)&v_.headers, sizeof(RingLaneHeader) * lanes, hipHostMallocCoherent));
  memset(v_.records, 0, sizeof(EventRecord) * lanes * capacity);
  memset(v_.headers, 0, sizeof(RingLaneHeader) * lanes);
  v_.lanes = lanes;
  v_.capacity = capacity;
}

EventRing::~EventRing() {
  (void)hipHostFree(v_.records);
  (void)hipHostFree(v_.headers);
}

// ---- control block ----
ControlBlock::ControlBlock() {
  HIP_CHECK(hipHostMalloc((void **)&c_, sizeof(Control), hipHostMallocCoherent));
  HIP_CHECK(hipMalloc((void **)&b_, sizeof(Barrier)));
  reset();
}

ControlBlock::~ControlBlock() {
  (void)hipHostFree(c_);
  (void)hipFree(b_);
}

void ControlBlock::reset() {
  __atomic_store_n(&c_->abort, 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&c_->status_wg, 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&c_->status_site, 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&c_->status, 0u, __ATOMIC_RELEASE);
  HIP_CHECK(hipMemset(b_, 0, sizeof(Barrier)));
  ++host_stats().cb_memsets;
  HIP_CHECK(hipDeviceSynchronize());
  ++host_stats().cb_syncs;
  clean_ = true;
}

void ControlBlock::begin_pass() {
  if (!clean_) {
    reset();
    return;
  }
  __atomic_store_n(&c_->abort, 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&c_->status_wg, 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&c_->status_site, 0u, __ATOMIC_RELAXED);
  __atomic_store_n(&c_->status, 0u, __ATOMIC_RELEASE);
}

void ControlBlock::end_pass() { clean_ = status() == kOk; }

bool trap_sink_installed() { return g_trap_record.load(std::memory_order_acquire) != nullptr; }

// For ctypes (tests/ops/mkops.py lib()): every op-test library installs the sink once it is loaded.
extern "C" int mk_trap_sink_install() {
  trap_sink_install();
  return 0;
}

// For tests/ops/test_traps.py: the report line for a record (status, site, wg); returns its length, 0 if not a trap.
extern "C" size_t mk_trap_format(uint32_t status, uint32_t site, uint32_t wg, char *buf, size_t cap) {
  Control c{0, status, wg, site};
  return trap_report_format(c, buf, cap);
}

// ---- cooperative launch ----
int max_coop_grid(const void *kernel, int block, size_t lds) {
  static std::mutex m;
  static std::map<std::tuple<int, const void *, int, size_t>, int> cache;
  int dev = 0, sms = 0, per_sm = 0;
  HIP_CHECK(hipGetDevice(&dev));
  const auto key = std::make_tuple(dev, kernel, block, lds);
  std::lock_guard<std::mutex> lk(m);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  HIP_CHECK(hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, dev));
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, block, lds));
  ++host_stats().occupancy_queries;
  return cache[key] = sms * per_sm;
}

void launch_coop(const void *kernel, int grid, int block, void **args, size_t lds, hipStream_t s) {
  const int max = max_coop_grid(kernel, block, lds);
  if (grid <= 0 || grid > max) HIP_DIE("cooperative grid %d outside 1..%d (block %d, lds %zu)", grid, max, block, lds);
  HIP_CHECK(hipLaunchCooperativeKernel(kernel, dim3(grid), dim3(block), args, lds, s));
}

// ---- watchdog ----
uint64_t mem_available_bytes() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  uint64_t kb = 0;
  std::string unit;
  while (f >> key >> kb >> unit) {
    if (key == "MemAvailable:") return kb * 1024;
  }
  HIP_DIE("MemAvailable not found in /proc/meminfo");
}

Watchdog::Watchdog(ControlBlock &cb, WatchdogConfig cfg, bool armed) : cb_(cb), cfg_(cfg) {
  if (cfg_.grace_s * kWallHz <= double(kMaxSpinTicks))
    HIP_DIE("watchdog grace %.2f s must outlast the longest device spin bound (%.2f s)", cfg_.grace_s,
           double(kMaxSpinTicks) / kWallHz);
  t_ = std::thread([this] { run(); });
  ++host_stats().watchdog_threads;
  if (armed) begin();
}

Watchdog::~Watchdog() {
  finish();
  {
    std::lock_guard<std::mutex> lk(m_);
    quit_ = true;
  }
  cv_.notify_all();
  if (t_.joinable()) t_.join();
}

void Watchdog::begin() {
  {
    std::lock_guard<std::mutex> lk(m_);
    if (state_ != State::kIdle) HIP_DIE("watchdog: begin() while a watch is still open (finish() first)");
    reason_.store(0);
    launched_.store(nullptr);
    state_ = State::kWatching;
  }
  cv_.notify_all();
}

void Watchdog::finish() {
  std::unique_lock<std::mutex> lk(m_);
  if (state_ == State::kIdle) return;
  state_ = State::kStopping;
  cv_.notify_all();
  idle_cv_.wait(lk, [&] { return state_ == State::kIdle; });
}

std::string Watchdog::reason() const {
  switch (reason_.load()) {
    case 1: return "deadline";
    case 2: return "memory";
  }
  return "";
}

void Watchdog::run() {
  std::unique_lock<std::mutex> lk(m_);
  for (;;) {
    cv_.wait(lk, [&] { return quit_ || state_ != State::kIdle; });
    if (quit_) return;
    if (state_ == State::kWatching) watch(lk);  // with lk held; it drops the lock while it works
    // kStopping before the watch even started lands here too: nothing to do but acknowledge.
    state_ = State::kIdle;
    idle_cv_.notify_all();
  }
}

// One watch, from begin() to finish(). Called with m_ held; every pass through the loop body runs unlocked so
// finish() and begin() are never blocked behind a meminfo read, and the lock is back before each wait.
void Watchdog::watch(std::unique_lock<std::mutex> &lk) {
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  auto secs = [&] { return std::chrono::duration<double>(clk::now() - t0).count(); };
  double aborted_at = -1;
  bool warned_live = false;
  while (state_ == State::kWatching && !quit_) {
    lk.unlock();
    const double t = secs();
    if (aborted_at < 0) {
      int why = 0;
      if (t > cfg_.deadline_s) why = 1;
      else if (mem_available_bytes() < cfg_.mem_floor_bytes) why = 2;
      if (why) {
        reason_ = why;
        aborted_at = t;
        fprintf(stderr, "watchdog: %s at %.2f s; setting abort\n", why == 1 ? "deadline" : "MemAvailable below floor", t);
        cb_.request_abort();
      }
    } else if (t - aborted_at > cfg_.grace_s) {
      hipEvent_t ev = launched_.load(std::memory_order_acquire);
      if (ev != nullptr && hipEventQuery(ev) == hipErrorNotReady) {
        if (!warned_live) {
          fprintf(stderr, "watchdog: a kernel is still running %.2f s after abort; not killing, waiting for it\n", t - aborted_at);
          warned_live = true;
        }
      } else if (warned_live) {
        // the kernel just ended: give the host a fresh grace period to return and finish()
        warned_live = false;
        aborted_at = t;
      } else {
        fprintf(stderr, "watchdog: host still not finished %.2f s after abort; killing pid %d\n", t - aborted_at, getpid());
        fflush(stderr);
        kill(getpid(), SIGKILL);
      }
    }
    lk.lock();
    cv_.wait_for(lk, std::chrono::duration<double>(cfg_.poll_s), [&] { return state_ != State::kWatching || quit_; });
  }
}

}  // namespace mk

// The thin runtime calls of the Rust host (src/binding.rs, milestone 15 step 6). Each call makes one HIP call and stops
// the program when it fails, as HIP_CHECK does. The host makes every call on its one worker thread.
extern "C" {
// Registers n bytes of host memory at p with the device, coarse-grained: weights never change while a kernel runs.
// HIP registers host memory fine-grained by default (hip_runtime_api.h). Measured 2026-09-30, 51-token fused read p50:
// fine-grained arena 452-456 ms, coarse-grained 448 ms, hipMalloc baseline 454 ms: no difference that decides it.
void mk_rt_host_register(void *p, uint64_t n) { HIP_CHECK(hipHostRegister(p, n, hipExtHostRegisterCoarseGrained)); }
// Removes the registration of mk_rt_host_register, so the host memory can be unmapped (the close of the weights).
void mk_rt_host_unregister(void *p) { HIP_CHECK(hipHostUnregister(p)); }
// n bytes of pinned host memory.
void *mk_rt_host_malloc(uint64_t n) {
  void *p = nullptr;
  HIP_CHECK(hipHostMalloc(&p, n, hipHostMallocDefault));
  return p;
}
void mk_rt_host_free(void *p) { HIP_CHECK(hipHostFree(p)); }
// n bytes of device memory.
void *mk_rt_malloc(uint64_t n) {
  void *p = nullptr;
  HIP_CHECK(hipMalloc(&p, n));
  return p;
}
void mk_rt_free(void *p) { HIP_CHECK(hipFree(p)); }
// Sets n bytes of device memory at p to the byte value.
void mk_rt_memset(void *p, int value, uint64_t n) { HIP_CHECK(hipMemset(p, value, n)); }
// kind: 1 host to device, 2 device to host (hipMemcpyKind). The copies wait for the device.
void mk_rt_memcpy(void *dst, const void *src, uint64_t n, uint32_t kind) {
  HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyKind(kind)));
}
void mk_rt_memcpy_2d(void *dst, uint64_t dpitch, const void *src, uint64_t spitch, uint64_t width, uint64_t height,
                     uint32_t kind) {
  HIP_CHECK(hipMemcpy2D(dst, dpitch, src, spitch, width, height, hipMemcpyKind(kind)));
}
}
