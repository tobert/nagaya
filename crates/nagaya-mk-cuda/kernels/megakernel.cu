// Persistent megakernel task-ring interpreter for CUDA GPUs (GB10 / sm_121a).
//
// Launched with one thread block per SM (or a configured grid width). Every block walks the same
// task ring in order. Before each task, thread 0 waits until scoreboard[wait_event] reaches
// gridDim.x using GPU-scope acquire loads; after each task, thread 0 signals scoreboard[signal_event]
// using GPU-scope release atomics.

#include <stddef.h>
#include <stdint.h>

// The opcodes, the flags, the sizes and the offsets come from src/protocol.rs. `build.rs` writes this header.
#include "mk_protocol.h"

// A block that polls a scoreboard event this many times traps. A normal run never reaches the limit, so the bits do
// not change. The limit turns a lost signal into an error of the launch, not a hang. At about 100 ns for each poll,
// the default is about 7 minutes.
#ifndef MK_SPIN_LIMIT
#define MK_SPIN_LIMIT (1ull << 32)
#endif

struct alignas(64) MkTaskDescriptor {
    uint16_t opcode;
    uint16_t flags;
    uint32_t layer;
    uint32_t wait_event;
    uint32_t signal_event;
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
    uint32_t p0;
    uint32_t p1;
    uint32_t p2;
    uint32_t p3;
};

#define MK_CHECK_OFFSET(type, field, offset) \
    static_assert(offsetof(type, field) == (offset), #type "::" #field " is not where src/protocol.rs puts it")

static_assert(sizeof(MkTaskDescriptor) == MK_DESC_BYTES, "MkTaskDescriptor differs in size from src/protocol.rs");
static_assert(alignof(MkTaskDescriptor) == MK_DESC_ALIGN, "MkTaskDescriptor differs in alignment from src/protocol.rs");
MK_CHECK_OFFSET(MkTaskDescriptor, opcode, MK_DESC_OFF_opcode);
MK_CHECK_OFFSET(MkTaskDescriptor, flags, MK_DESC_OFF_flags);
MK_CHECK_OFFSET(MkTaskDescriptor, layer, MK_DESC_OFF_layer);
MK_CHECK_OFFSET(MkTaskDescriptor, wait_event, MK_DESC_OFF_wait_event);
MK_CHECK_OFFSET(MkTaskDescriptor, signal_event, MK_DESC_OFF_signal_event);
MK_CHECK_OFFSET(MkTaskDescriptor, a, MK_DESC_OFF_a);
MK_CHECK_OFFSET(MkTaskDescriptor, b, MK_DESC_OFF_b);
MK_CHECK_OFFSET(MkTaskDescriptor, c, MK_DESC_OFF_c);
MK_CHECK_OFFSET(MkTaskDescriptor, d, MK_DESC_OFF_d);
MK_CHECK_OFFSET(MkTaskDescriptor, p0, MK_DESC_OFF_p0);
MK_CHECK_OFFSET(MkTaskDescriptor, p1, MK_DESC_OFF_p1);
MK_CHECK_OFFSET(MkTaskDescriptor, p2, MK_DESC_OFF_p2);
MK_CHECK_OFFSET(MkTaskDescriptor, p3, MK_DESC_OFF_p3);

struct BlockStats {
    uint32_t tasks_run;
    uint32_t pad;
    uint64_t wait_ns;
    uint64_t exec_ns;
};

static_assert(sizeof(BlockStats) == MK_STATS_BYTES, "BlockStats differs in size from src/protocol.rs");
MK_CHECK_OFFSET(BlockStats, tasks_run, MK_STATS_OFF_tasks_run);
MK_CHECK_OFFSET(BlockStats, pad, MK_STATS_OFF_pad);
MK_CHECK_OFFSET(BlockStats, wait_ns, MK_STATS_OFF_wait_ns);
MK_CHECK_OFFSET(BlockStats, exec_ns, MK_STATS_OFF_exec_ns);

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* ptr) {
    uint32_t val;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ uint32_t atom_release_gpu_add_u32(uint32_t* ptr, uint32_t val) {
    uint32_t old;
    asm volatile("atom.release.gpu.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val));
    return old;
}

// The dynamic shared memory of the block. The launch gives its size. Only the probe ops use it for now.
extern __shared__ __align__(128) uint8_t mk_smem[];

__device__ __forceinline__ void op_copy(const MkTaskDescriptor& task) {
    // a = dst (float*), b = src (const float*), p0 = count of floats.
    float* dst = reinterpret_cast<float*>(task.a);
    const float* src = reinterpret_cast<const float*>(task.b);
    // The index is 64 bits: with a 32-bit index, `i += stride` wraps for a count near 2^32 and the loop never ends.
    const uint64_t n = task.p0;
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// The value that OP_SELFTEST_FILL writes. It is never zero, so a word that no block wrote is wrong. The same function
// is `selftest_value` in src/protocol.rs.
__device__ __forceinline__ uint32_t selftest_value(uint32_t seed, uint32_t owner, uint32_t index) {
    return ((seed * 0x9E3779B1u) ^ (owner * 0x85EBCA6Bu) ^ (index * 0xC2B2AE35u)) | 1u;
}

// Block n writes p0 words at a + n * p0 words. Block p2 waits p1 ns first, so that it is the last to write.
__device__ __forceinline__ void op_selftest_fill(const MkTaskDescriptor& task) {
    uint32_t* data = reinterpret_cast<uint32_t*>(task.a);
    const uint32_t words = task.p0;
    if (blockIdx.x == task.p2 && threadIdx.x == 0) {
        const uint64_t until = globaltimer_ns() + task.p1;
        while (globaltimer_ns() < until) {
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
        data[static_cast<uint64_t>(blockIdx.x) * words + i] = selftest_value(task.p3, blockIdx.x, i);
    }
}

// Each block reads the words of all blocks and counts the words that differ from selftest_value. Thread 0 writes the
// count to d[blockIdx.x].
__device__ __forceinline__ void op_selftest_check(const MkTaskDescriptor& task) {
    const uint32_t* data = reinterpret_cast<const uint32_t*>(task.a);
    uint32_t* out = reinterpret_cast<uint32_t*>(task.d);
    const uint32_t words = task.p0;
    const uint64_t total = static_cast<uint64_t>(gridDim.x) * words;
    uint32_t wrong = 0;
    for (uint64_t base = 0; base < total; base += blockDim.x) {
        const uint64_t k = base + threadIdx.x;
        bool differs = false;
        if (k < total) {
            differs = data[k] != selftest_value(task.p3, static_cast<uint32_t>(k / words), static_cast<uint32_t>(k % words));
        }
        wrong += __syncthreads_count(differs);
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = wrong;
    }
}

#include "probe_ops.cuh"

extern "C" __global__ void mk_persistent_interpreter(
    const MkTaskDescriptor* __restrict__ tasks,
    uint32_t num_tasks,
    uint32_t* __restrict__ scoreboard,
    uint32_t base_event,
    BlockStats* __restrict__ stats,
    float* __restrict__ scratch,
    uint64_t* __restrict__ profile
) {
    (void)scratch;
    uint32_t bid = blockIdx.x;
    uint32_t grid = gridDim.x;
    uint32_t tasks_run = 0;
    uint64_t wait_ns = 0;
    uint64_t exec_ns = 0;

    for (uint32_t idx = 0; idx < num_tasks; ++idx) {
        MkTaskDescriptor task = tasks[idx];
        uint64_t t_start = 0;
        uint64_t t_after_wait = 0;

        if (threadIdx.x == 0) {
            t_start = globaltimer_ns();
            if (task.wait_event != NO_EVENT) {
                const uint32_t* ev = scoreboard + (base_event + task.wait_event);
                uint64_t polls = 0;
                while (ld_acquire_gpu_u32(ev) < grid) {
                    // Spin on GPU-scope acquire load, up to the limit.
                    if (++polls >= MK_SPIN_LIMIT) {
                        __trap();
                    }
                }
            }
            t_after_wait = globaltimer_ns();
            wait_ns += (t_after_wait - t_start);
        }
        __syncthreads();

        if (task.opcode == OP_HALT) {
            if (threadIdx.x == 0) {
                uint64_t t_end = globaltimer_ns();
                exec_ns += (t_end - t_after_wait);
#ifdef MK_PROFILE
                if (profile != nullptr) {
                    uint64_t* slot = profile + ((static_cast<uint64_t>(idx) * grid + bid) * MK_PROFILE_WORDS);
                    slot[0] = t_start;
                    slot[1] = t_after_wait;
                    slot[2] = t_end;
                }
#endif
            }
            break;
        }

        switch (task.opcode) {
            case OP_NOP:
                break;
            case OP_COPY:
                op_copy(task);
                break;
            case OP_SELFTEST_FILL:
                op_selftest_fill(task);
                break;
            case OP_SELFTEST_CHECK:
                op_selftest_check(task);
                break;
            case OP_PROBE_READ:
                op_probe_read(task);
                break;
            case OP_PROBE_L2:
                op_probe_l2(task);
                break;
            case OP_PROBE_TIMER:
                op_probe_timer(task);
                break;
            case OP_PROBE_MMA_TILE:
                op_probe_mma_tile(task);
                break;
            case OP_PROBE_MMA_RATE:
                op_probe_mma_rate(task);
                break;
            default:
                // Unknown opcode: trap rather than silently corrupt state (spec 8.5).
                __trap();
                break;
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t t_end = globaltimer_ns();
            exec_ns += (t_end - t_after_wait);
            tasks_run += 1;
#ifdef MK_PROFILE
            if (profile != nullptr) {
                uint64_t* slot = profile + ((static_cast<uint64_t>(idx) * grid + bid) * MK_PROFILE_WORDS);
                slot[0] = t_start;
                slot[1] = t_after_wait;
                slot[2] = t_end;
            }
#endif
            if (task.signal_event != NO_EVENT) {
                atom_release_gpu_add_u32(scoreboard + (base_event + task.signal_event), 1u);
            }
        }
    }

    if (threadIdx.x == 0 && stats != nullptr) {
        stats[bid].tasks_run = tasks_run;
        stats[bid].pad = 0;
        stats[bid].wait_ns = wait_ns;
        stats[bid].exec_ns = exec_ns;
    }
}
