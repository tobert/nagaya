// SPDX-License-Identifier: MIT
// The megakernel: one cooperative launch walks the pass's stage list, with a grid barrier
// between stages (except inside the schedule's barrier-free segments, src/schedule.rs). It must be the only kernel in this translation unit (see engine.hip).
#include "device.h"
#include "stage.h"

namespace mk {

// skip_barrier: the mutation test's knob (M4 check) — the barrier after that stage is left out,
// which must show up as a bit difference. UINT32_MAX in every real pass. With its top bit set it is the stall knob
// instead (MK_MUTATE_STALL_BARRIER, below).
// stages: __restrict__ is a promise that nothing writes the stage list during the pass (the host uploads it
// before launch; no op's output points into it). Without it the compiler can't rule out an op's store
// aliasing the list, so it loads every op's arguments with vector loads into VGPRs: with it, scalar loads
// into SGPRs, and k_mega went 219 -> 136 VGPRs (2026-09-27; tests/engine/test_mega_resources.py).
// ev: the event ring (DESIGN.md "The event ring"), or records == nullptr for none. With it, thread 0 of
// each workgroup records every stage's begin and end (after the workgroup's own threads are done, before
// the grid barrier) in its lane. The outputs don't depend on it.
// amdgpu_waves_per_eu(8): the grid-80 design target (4 workgroups of 8 waves per WGP) given to the compiler, so
// it schedules the whole kernel for <= 192 VGPRs; left to itself it took 201 with the f16 qmm inlined.
__global__ void __launch_bounds__(kBlockThreads) __attribute__((amdgpu_waves_per_eu(8))) k_mega(const Stage *__restrict__ stages, uint32_t n, Control *ctl, Barrier *bar,
                                                   uint32_t skip_barrier, EventRingView ev) {
  const bool events = ev.records != nullptr;
  // The barrier tail (docs/narrow.md N3, stage.h kSchedTail): when stage i + 1 carries the bit, the grid barrier after stage i
  // is only its first half (grid_arrive_hold): the last arriver is the runner (1), the rest waiters (2). Stage i + 1 then runs
  // on the runner alone, as (wg 0, nwg 1), and the barrier after it is the second half (grid_tail_release). 0 = no tail in
  // flight (a skipped barrier, the single-stage launch of MK_MEGA_PER_STAGE: the stage runs as any other).
  uint32_t tail_role = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const uint64_t t_begin = events ? uint64_t(wall_clock64()) : 0;
    // The schedule (src/schedule.rs): a rotation deals this stage's items after the ones before it in its segment.
    const uint32_t sched = stages[i].sched;
    const uint32_t rot = sched & kSchedRotMask;
    if (tail_role != 2)  // a waiter has nothing to run in the tail stage
      run_stage(stages[i], tail_role ? 0u : (rot ? (blockIdx.x + rot) % gridDim.x : blockIdx.x), tail_role ? 1u : gridDim.x);
    if (events) {
      __syncthreads();
      if (threadIdx.x == 0) {
        EventRecord r{};
        r.stage = uint16_t(i);
        r.layer = uint16_t(stages[i].layer);
        r.wg = blockIdx.x;
        r.t_begin = t_begin;
        r.t_end = uint64_t(wall_clock64());
        r.payload[0] = stages[i].op;
        // Where this workgroup runs (HW_ID1: wave, SIMD, WGP, shader array, shader engine), for placement
        // studies (tools/fuse_gap.py). s_getreg of the whole 32-bit register 23.
        r.payload[1] = __builtin_amdgcn_s_getreg((31 << 11) | 23);
        // A full lane waits (bounded); on a timeout or abort ring_push sets the status and the next grid
        // barrier stops every workgroup. The site marks the stage with the top bit.
        ring_push(ctl, ev, blockIdx.x, r, 0x80000000u | i);
      }
    }
    if (tail_role != 0) {  // the tail stage's own barrier is the release; the mutation knobs below act on arrivals only
      const bool ok = grid_tail_release(ctl, bar, tail_role, i);
      tail_role = 0;
      if (!ok) return;
      continue;
    }
    if (i == skip_barrier) continue;
    // No grid barrier inside a segment: the next stage touches nothing this one writes, nor writes what it reads.
    // The ops share the workgroup's LDS (the arena, their tiles), so this workgroup's waves still meet here.
    if (sched & kSchedNoBarrier) {
      lds_sync();
      continue;
    }
    // MK_MUTATE_STALL_BARRIER (engine.hip, a test knob): skip_barrier = 0x80000000 | s makes workgroup 0 leave at the first
    // grid barrier at or after stage s without arriving, so the rest wait out their bounded spin: a real spin-timeout.
    if (skip_barrier != UINT32_MAX && (skip_barrier & 0x80000000u) && i >= (skip_barrier & 0x7fffffffu) && blockIdx.x == 0) return;
    if (i + 1 < n && (stages[i + 1].sched & kSchedTail)) {  // the barrier is the next stage's tail: arrive, don't release
      tail_role = grid_arrive_hold(ctl, bar, gridDim.x, i);
      if (tail_role == 0) return;
      continue;
    }
    if (!grid_barrier(ctl, bar, gridDim.x, i)) return;
  }
}

const void *mega_kernel() { return reinterpret_cast<const void *>(&k_mega); }

}  // namespace mk
