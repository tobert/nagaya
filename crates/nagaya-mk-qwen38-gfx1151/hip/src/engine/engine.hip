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


// ---- kernels: one stage per launch (M3), or the whole list in one cooperative launch (M4) ----
// s: __restrict__ as in k_mega (mega.hip): nothing writes the stage list during the pass.
// ctl, site: the watchdog of the launches one stage each (Engine::launch_each). Every workgroup looks at the control words once on entry and, if the
// host asked to stop or something already failed, returns without running its stage: after an abort the rest of the pass's
// launches drain in microseconds. The first to see the abort word records kAborted at `site` (the stage's index). One
// thread reads and the workgroup shares the answer, so every thread agrees (run_stage has __syncthreads in it). This stops a
// long or runaway sequence of stages BETWEEN stages; a stage that hangs inside an op can't be stopped (ops have no polls).
__global__ void __launch_bounds__(kBlockThreads) k_stage(const Stage *__restrict__ s, Control *ctl, uint32_t site) {
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

const void *stage_kernel() { return reinterpret_cast<const void *>(&k_stage); }

// k_mega lives alone in mega.hip. With two kernels in one translation unit that both reach the
// ops' LDS-using inline functions, the compiler's module-LDS lowering produced a k_stage that
// never finished, even for an empty stage (run 2, 2026-09-26; bisected with tests/engine/probe_stage.hip).
// Rule: k_stage and k_mega each get their own TU, and no other kernel may share one with them.

constexpr uint32_t kUnfusedGrid = 1024;

Engine::Engine(double watchdog_deadline_s) : wd_(cb_, WatchdogConfig{.deadline_s = watchdog_deadline_s}, /*armed=*/false) {
  trap_sink_install();  // a device trap names itself on stderr before the runtime aborts (trap.h)
}

Engine::~Engine() {
  if (launched_ev_) (void)hipEventDestroy(launched_ev_);
}

void Engine::make_event() {
  if (!launched_ev_) {
    HIP_CHECK(hipEventCreate(&launched_ev_));
    ++host_stats().events_created;
  }
}

LaunchResult Engine::result() {
  return LaunchResult{cb_.status(), cb_.host()->status_site, cb_.host()->status_wg, wd_.reason().empty() ? 0u : 1u};
}

// One cooperative launch of the megakernel over n stages, under the control block and the watchdog.
LaunchResult Engine::launch_fused(const Stage *ds, uint32_t n, int grid, uint32_t skip) {
  cb_.begin_pass();
  Control *ctl = cb_.device();
  Barrier *bar = cb_.barrier();
  // The kernel's event ring stays off: no record is made (mega.hip).
  EventRingView ev{};
  void *args[] = {&ds, &n, &ctl, &bar, &skip, &ev};
  wd_.begin();
  make_event();
  launch_coop(mega_kernel(), grid, kBlockThreads, args, 0, 0);
  HIP_CHECK(hipEventRecord(launched_ev_, 0));
  wd_.arm(launched_ev_);
  HIP_CHECK(hipDeviceSynchronize());
  wd_.finish();
  cb_.end_pass();
  return result();
}

// One launch of k_stage for each of n stages, under the control block and the watchdog. k_stage checks abort and status
// on entry, so once the deadline sets abort every remaining launch drains in microseconds.
LaunchResult Engine::launch_each(const Stage *ds, uint32_t n) {
  cb_.begin_pass();
  Control *ctl = cb_.device();
  make_event();
  wd_.begin();
  for (uint32_t i = 0; i < n; ++i) {
    k_stage<<<kUnfusedGrid, kBlockThreads>>>(ds + i, ctl, i);
    HIP_CHECK(hipGetLastError());
    // After every launch (the same event, re-recorded): the watchdog's "a kernel is still running" query then looks at
    // the newest launch; armed once, after the first record.
    HIP_CHECK(hipEventRecord(launched_ev_, 0));
    if (i == 0) wd_.arm(launched_ev_);
  }
  HIP_CHECK(hipDeviceSynchronize());
  wd_.finish();
  cb_.end_pass();
  return result();
}

}  // namespace mk
