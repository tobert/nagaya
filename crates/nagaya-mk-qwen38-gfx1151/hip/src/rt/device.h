// SPDX-License-Identifier: MIT
// Device side of the runtime rails. Every wait is bounded: it returns false on abort or timeout
// and records why in Control. Callers exit the kernel when a wait fails; nothing spins forever.
#pragma once
#include "rt.h"

namespace qwen38 {

__device__ inline uint32_t ld_acquire_sys(const uint32_t *p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline uint32_t ld_acquire_agent(const uint32_t *p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

// Relaxed loads for polling. An acquire load comes with cache invalidates on gfx11 (buffer_gl0_inv and
// buffer_gl1_inv; GL1 is shared by the CUs of a shader array), so a workgroup spinning on acquire loads kept
// invalidating the caches the working workgroups read through (2026-09-27, tools/fuse_gap.py). Spin on these,
// and order the data afterwards with one acquire fence (grid_barrier does, for every thread).
__device__ inline uint32_t ld_relaxed_sys(const uint32_t *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline uint32_t ld_relaxed_agent(const uint32_t *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Record the first failure. Any thread may call it.
__device__ inline void set_status(Control *c, uint32_t status, uint32_t site) {
  uint32_t expected = 0;
  if (__hip_atomic_compare_exchange_strong(&c->status, &expected, status, __ATOMIC_ACQ_REL,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)) {
    __hip_atomic_store(&c->status_wg, blockIdx.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&c->status_site, site, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// True if the host asked to stop or another workgroup already failed.
__device__ inline bool should_stop(Control *c) {
  return ld_acquire_sys(&c->abort) != 0 || ld_acquire_sys(&c->status) != 0;
}

// Spin until pred() holds. Returns false (and records status) on abort, on another workgroup's
// failure, or after `ticks` of wall clock. One thread calls this.
template <class Pred>
__device__ inline bool spin_until(Control *c, Pred pred, uint32_t site,
                                  uint64_t ticks = kDefaultSpinTicks) {
  if (ticks > kMaxSpinTicks) ticks = kMaxSpinTicks;  // no caller may ask for a longer bound
  const uint64_t t0 = wall_clock64();
  for (uint32_t i = 0;; ++i) {
    if (pred()) return true;
    // Back off: 79 workgroups polling every ~25 ns (s_sleep 1, flags every 16 polls) slowed the ones still
    // working by up to 6x (moe_group, a one-workgroup stage: 6.4 ms unfused, 16-40 ms fused) and made fused
    // passes vary 420-550 ms. s_sleep 32 (~0.8 us) with the flags every 64 polls: 396 ms, steady, and the
    // barrier gaps didn't grow (2026-09-27, tools/fuse_gap.py). The bound still holds: time is checked every
    // 64 polls (~50 us).
    if ((i & 63) == 0) {
      if (ld_relaxed_sys(&c->abort) != 0) {  // flags: nothing is read on their account
        set_status(c, kAborted, site);
        return false;
      }
      if (ld_relaxed_sys(&c->status) != 0) return false;
      if (uint64_t(wall_clock64()) - t0 > ticks) {
        set_status(c, kSpinTimeout, site);
        return false;
      }
    }
    __builtin_amdgcn_s_sleep(32);
  }
}

// Grid barrier over `nblocks` co-resident workgroups (a cooperative launch guarantees that).
// Generation-counting, agent-scope atomics in device memory. Every thread of the block calls it;
// all return the same bool. false = stop: the caller must return from the kernel.
__device__ inline bool grid_barrier(Control *c, Barrier *b, uint32_t nblocks, uint32_t site) {
  __shared__ uint32_t ok_shared;
  // Has anything failed or been aborted? Asked first, so the (host-coherent, ~1-2 us) load is in flight while the release
  // fence and __syncthreads below run and its latency is hidden: asked only after the arrival it cost the last arriver
  // about 1.5-2 us a barrier, 3-4 ms a pass (A/B 2026-10-05, tools/slope.py). The answer is "as of the start of this
  // barrier": a failure recorded before it began stops it (tests/m0/test_barrier_status.hip); one that lands in the next
  // microsecond is caught at the next barrier, as before.
  uint32_t stop_seen = 0;
  if (threadIdx.x == 0) stop_seen = ld_relaxed_sys(&c->abort) | (ld_relaxed_sys(&c->status) << 1);
  // Publish this block's writes before arriving.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t gen = ld_acquire_agent(&b->generation);
    const uint32_t arrived =
        __hip_atomic_fetch_add(&b->count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1;
    bool ok = true;
    if (arrived == nblocks) {
      // The last arriver releases the grid only if nothing had failed when the barrier began: otherwise it leaves the
      // generation alone, so no workgroup comes out of this barrier into the next stage (it used to run one more, and
      // the failure site named an earlier stage than the last one written). The waiters' own bounded polls see the
      // status or the abort within ~50 us and leave. The count stays at nblocks: end_pass marks the control block dirty
      // on a nonzero status and the next begin_pass resets it.
      if (stop_seen != 0) {
        if ((stop_seen & 1) != 0) set_status(c, kAborted, site);  // a no-op when a status is already recorded
        ok = false;
      } else {
        __hip_atomic_store(&b->count, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&b->generation, gen + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
    } else {
      ok = spin_until(c, [&] { return ld_relaxed_agent(&b->generation) != gen; }, site);  // acquire: the fence below
    }
    ok_shared = ok;
  }
  __syncthreads();
  // See every other block's writes (all threads: in WGP mode a block spans two L0 caches).
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return ok_shared != 0;
}

// The barrier as two halves, for a stage that one workgroup runs alone while the rest wait (docs/narrow.md N3; the
// "tail" of a barrier). grid_arrive_hold is the barrier's first half: every thread calls it, all get the same role.
//   0  stop: a failure or abort was recorded before the barrier began (the last arriver recorded/kept it, as
//      grid_barrier does); the caller returns from the kernel. Only the last arriver can return it.
//   1  runner: the LAST arriver. It holds the acquire on every workgroup's writes and the count at nblocks; the grid is
//      not released. It runs the tail alone, then calls grid_tail_release.
//   2  waiter: arrived, nothing waited for yet. It skips the tail and calls grid_tail_release, which waits (bounded,
//      abort-aware: spin_until) for the runner's release.
// grid_tail_release is the second half: the runner publishes the tail's writes and releases (count = 0, then
// generation + 1, as grid_barrier does); a waiter waits for the generation to move off the one it arrived at, then
// acquires. false = stop (waiters only: abort, another failure, or the bound). The pair is one barrier: arrive_hold ...
// tail ... tail_release replaces grid_barrier; every workgroup must call both, in order, once.
// The generation seen at arrival is kept in LDS between the halves (thread 0 uses it).
__device__ __attribute__((always_inline)) inline uint32_t &tail_gen_slot() {
  __shared__ uint32_t gen;
  return gen;
}
__device__ __attribute__((always_inline)) inline uint32_t grid_arrive_hold(Control *c, Barrier *b, uint32_t nblocks, uint32_t site) {
  __shared__ uint32_t role_shared;
  uint32_t stop_seen = 0;  // as grid_barrier: asked first, its latency hidden behind the fence and the block sync
  if (threadIdx.x == 0) stop_seen = ld_relaxed_sys(&c->abort) | (ld_relaxed_sys(&c->status) << 1);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t gen = ld_acquire_agent(&b->generation);
    const uint32_t arrived = __hip_atomic_fetch_add(&b->count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1;
    uint32_t role = 2;
    if (arrived == nblocks) {
      if (stop_seen != 0) {  // the count stays at nblocks, the generation unmoved: the waiters' bounded polls see the status
        if ((stop_seen & 1) != 0) set_status(c, kAborted, site);
        role = 0;
      } else {
        role = 1;
      }
    }
    tail_gen_slot() = gen;
    role_shared = role;
  }
  __syncthreads();
  const uint32_t role = __builtin_amdgcn_readfirstlane(role_shared);
  // The runner sees every other block's writes (all threads: in WGP mode a block spans two L0 caches).
  if (role == 1) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return role;
}
__device__ __attribute__((always_inline)) inline bool grid_tail_release(Control *c, Barrier *b, uint32_t role, uint32_t site) {
  __shared__ uint32_t ok_shared;
  if (role == 1) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");  // publish the tail's writes
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint32_t gen = tail_gen_slot();
      __hip_atomic_store(&b->count, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&b->generation, gen + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    return true;
  }
  if (role != 2) return false;  // a stopped workgroup has nothing to release or wait for
  if (threadIdx.x == 0) {
    const uint32_t gen = tail_gen_slot();
    ok_shared = spin_until(c, [&] { return ld_relaxed_agent(&b->generation) != gen; }, site);  // acquire: the fence below
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return ok_shared != 0;
}

// Append one record to this workgroup's lane. One thread calls it. Waits (bounded) while the
// lane is full. false = stop.
__device__ inline bool ring_push(Control *c, const EventRingView &r, uint32_t lane,
                                 EventRecord rec, uint32_t site) {
  RingLaneHeader *h = &r.headers[lane];
  const uint64_t head = __hip_atomic_load(&h->head, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  if (head - __hip_atomic_load(&h->tail, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) >=
      r.capacity) {
    const bool ok = spin_until(
        c,
        [&] {
          return head - __hip_atomic_load(&h->tail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) <
                 r.capacity;  // relaxed: nothing is read after the wait, only the record written
        },
        site);
    if (!ok) return false;
  }
  rec.seq = uint32_t(head);
  EventRecord *dst = &r.records[size_t(lane) * r.capacity + (head & (r.capacity - 1))];
  // Field by field: the struct copy `*dst = rec` went through a stack temporary and gave the whole megakernel
  // 32 B/lane of scratch, events on or off (2026-09-27; tests/engine/test_mega_resources.py).
  dst->seq = rec.seq;
  dst->pass = rec.pass;
  dst->stage = rec.stage;
  dst->layer = rec.layer;
  dst->wg = rec.wg;
  dst->t_begin = rec.t_begin;
  dst->t_end = rec.t_end;
#pragma unroll
  for (int i = 0; i < 8; ++i) dst->payload[i] = rec.payload[i];
  __hip_atomic_store(&h->head, head + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  return true;
}

}  // namespace qwen38
