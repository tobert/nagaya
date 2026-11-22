// Persistent megakernel task-ring interpreter for CUDA GPUs (GB10 / sm_121a).
//
// Launched with one thread block per SM (or a configured grid width). Every block walks the same
// task ring in order. Before each task, thread 0 waits until scoreboard[wait_event] reaches
// gridDim.x using GPU-scope acquire loads; after each task, thread 0 signals scoreboard[signal_event]
// using GPU-scope release atomics.

#include <stdint.h>

// Opcodes (must match src/protocol.rs).
#define OP_NOP   0u
#define OP_COPY  21u
#define OP_HALT  255u

#define NO_EVENT 0xFFFFFFFFu

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

static_assert(sizeof(MkTaskDescriptor) == 64, "MkTaskDescriptor must be 64 bytes");
static_assert(alignof(MkTaskDescriptor) == 64, "MkTaskDescriptor must be 64-byte aligned");

struct BlockStats {
    uint32_t tasks_run;
    uint32_t pad;
    uint64_t wait_ns;
    uint64_t exec_ns;
};

static_assert(sizeof(BlockStats) == 24, "BlockStats must be 24 bytes");

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

__device__ __forceinline__ void op_copy(const MkTaskDescriptor& task) {
    // a = dst (float*), b = src (const float*), p0 = count of floats.
    float* dst = reinterpret_cast<float*>(task.a);
    const float* src = reinterpret_cast<const float*>(task.b);
    uint32_t n = task.p0;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

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
                while (ld_acquire_gpu_u32(ev) < grid) {
                    // Spin on GPU-scope acquire load.
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
                    uint64_t* slot = profile + ((static_cast<uint64_t>(idx) * grid + bid) * 3u);
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
                uint64_t* slot = profile + ((static_cast<uint64_t>(idx) * grid + bid) * 3u);
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
