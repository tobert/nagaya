// SPDX-License-Identifier: MIT
// Conventions every op follows, so the unfused engine (M3) and the megakernel (M4) run the same
// device code and produce the same bits.
//
// - An op is a __device__ function `void op_<name>(const <Name>Args &a, uint32_t wg, uint32_t nwg)`
//   called by every thread of a kMkBlock-thread workgroup. It splits its work into items and
//   takes items wg, wg + nwg, wg + 2*nwg, ... The value of every output element depends only on
//   its item, never on nwg or on which workgroup ran it: no float atomics, no cross-item
//   reductions, reduction order fixed by the op's definition. So any grid size gives the same
//   bits, and a single launch (grid = work items) equals the megakernel (grid = co-resident WGs).
// - Every op has a wrapper kernel `k_<name>` (<<<grid, kMkBlock>>> calling op_<name>(args,
//   blockIdx.x, gridDim.x)) and a C ABI entry `mk_<name>(...)` for tests (host pointers in and
//   out; allocates, copies, launches, synchronizes, frees; returns 0 or dies with a message).
// - Shapes are compile-time constants of this model (model.h); a runtime shape that disagrees
//   stops the program (HIP_DIE), it never falls back.
// - Activations and accumulators are f32. Weights stay in their GGUF block format in memory and
//   are dequantized exactly into f32 registers (quant.h).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#include "trap.h"  // HIP_TRAP: the ops' traps name themselves (PLAN.md Problems 2026-09-29)

constexpr uint32_t kMkBlock = 256;  // threads per workgroup: 8 wave32s
constexpr uint32_t kWave = 32;

// threadIdx.x for ops: the same value, passed through an empty asm so the compiler can't prove it loop-invariant.
// The megakernel runs every op inside one loop over ~1800 stages, and LLVM's IR LICM (blind to register pressure)
// hoisted each op's thread-derived address math out of that loop and held it live for the whole kernel: k_mega
// sat at its 192-VGPR ceiling with a block of precomputed LDS offsets and SGPR spills in VGPR lanes, and one more
// register anywhere spilled (2026-09-28). Recomputing them per stage costs a few ALU ops.
// A workgroup barrier that orders LDS only. __syncthreads() is a workgroup-scope fence, which in WGP mode waits for
// every outstanding global load (s_waitcnt vmcnt(0)) and invalidates L0 (buffer_gl0_inv), so a weight load issued
// before it can't overlap it. Use this only where the threads communicate through LDS alone (global data they read
// was written by an earlier stage, ordered by the grid barrier).
__device__ __attribute__((always_inline)) inline void lds_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __attribute__((always_inline)) inline uint32_t mk_tid() {
  uint32_t t = threadIdx.x;
  asm volatile("" : "+v"(t));
  return t;
}

// The kernel-wide LDS arena. Each __shared__ array an op declares gets its own space, so a kernel's static
// LDS is the *sum* over every op it can run, and in the megakernel that sum sets the co-resident grid
// (PLAN.md Speed: 41 KiB halved the grid to 20; a WGP has 64 KiB, so grid 80 = 4 workgroups per WGP allows
// 16 KiB each, arena included; tests/engine/test_mega_resources.py). Ops whose tiles are
// large take them from this one arena instead, so those cost the *max*. The arena is only valid inside one
// op: stages are separated by a grid barrier (every thread of the workgroup passes a __syncthreads) or by a
// kernel launch. Always inlined: big LDS reached through a non-inlined call hung k_stage (stage.h).
constexpr uint32_t kLdsArenaBytes = 9216;
__device__ __attribute__((always_inline)) inline uint8_t *lds_arena() {
  __shared__ __attribute__((aligned(16))) uint8_t arena[kLdsArenaBytes];
  return arena;
}

// Deterministic wave-wide sum: a fixed xor-butterfly, lane i ends with the same bits on every run.
__device__ inline float wave_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
  return v;
}

// Deterministic block-wide sum over kMkBlock threads: wave sums, then wave 0 adds the 8 partials
// in wave order. Every thread gets the result. `scratch` needs kMkBlock / kWave floats of LDS.
__device__ inline float block_sum(float v, float *scratch) {
  const uint32_t w = threadIdx.x / kWave, l = threadIdx.x % kWave;
  v = wave_sum(v);
  __syncthreads();
  if (l == 0) scratch[w] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (uint32_t i = 0; i < kMkBlock / kWave; ++i) t += scratch[i];
  __syncthreads();
  return t;
}
