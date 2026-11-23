// Persistent megakernel task-ring interpreter for CUDA GPUs (GB10 / sm_121a).
//
// Launched with one thread block per SM (or a configured grid width). Every block walks the same task ring in order.
// A task waits on up to two events before the first item of the block, with GPU-scope acquire loads. Block b runs the
// items b, b + gridDim.x and so on. After the last of its items, the block adds the count of its items to the event
// `signal`, one time, with a GPU-scope release.
//
// A block has one of two shapes (see run_producer_shape and run_flat_shape). The producer shape has 288 threads: a
// producer warp that fetches the descriptors, waits at the scoreboard and prefetches, and eight consumer warps that
// run the items. The flat shape is any other width: every thread is a consumer, and thread 0 waits. Only the probes
// use the flat shape.

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

struct alignas(128) MkTaskDescriptor {
    uint16_t opcode;
    uint16_t flags;
    uint32_t items;
    uint32_t wait0_event;
    uint32_t wait0_target;
    uint32_t wait1_event;
    uint32_t wait1_target;
    uint32_t signal;
    uint32_t rows;
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
    uint64_t e;
    uint64_t f;
    uint32_t n;
    uint32_t k;
    uint32_t kt;
    uint32_t group_items;
    uint32_t item_bytes;
    uint32_t scale_bytes;
    uint32_t pos0;
    uint32_t layer;
    float p0;
    float p1;
    float p2;
    float p3;
};

// The task table of a task: `MK_TABLE_POINTERS` device pointers that stay the same for a slot and a run. `d` of the
// descriptor points to it, or `d` is 0 when the task has none. The producer copies the table to shared memory with the
// descriptor. An op names the slots that it uses (for example MK_EXPERT_TABLE_AMAX), in its own file.
struct alignas(128) MkTaskTable {
    uint64_t ptr[MK_TABLE_POINTERS];
};

#define MK_CHECK_OFFSET(type, field, offset) \
    static_assert(offsetof(type, field) == (offset), #type "::" #field " is not where src/protocol.rs puts it")

static_assert(sizeof(MkTaskDescriptor) == MK_DESC_BYTES, "MkTaskDescriptor differs in size from src/protocol.rs");
static_assert(alignof(MkTaskDescriptor) == MK_DESC_ALIGN, "MkTaskDescriptor differs in alignment from src/protocol.rs");
MK_CHECK_OFFSET(MkTaskDescriptor, opcode, MK_DESC_OFF_opcode);
MK_CHECK_OFFSET(MkTaskDescriptor, flags, MK_DESC_OFF_flags);
MK_CHECK_OFFSET(MkTaskDescriptor, items, MK_DESC_OFF_items);
MK_CHECK_OFFSET(MkTaskDescriptor, wait0_event, MK_DESC_OFF_wait0_event);
MK_CHECK_OFFSET(MkTaskDescriptor, wait0_target, MK_DESC_OFF_wait0_target);
MK_CHECK_OFFSET(MkTaskDescriptor, wait1_event, MK_DESC_OFF_wait1_event);
MK_CHECK_OFFSET(MkTaskDescriptor, wait1_target, MK_DESC_OFF_wait1_target);
MK_CHECK_OFFSET(MkTaskDescriptor, signal, MK_DESC_OFF_signal);
MK_CHECK_OFFSET(MkTaskDescriptor, rows, MK_DESC_OFF_rows);
MK_CHECK_OFFSET(MkTaskDescriptor, a, MK_DESC_OFF_a);
MK_CHECK_OFFSET(MkTaskDescriptor, b, MK_DESC_OFF_b);
MK_CHECK_OFFSET(MkTaskDescriptor, c, MK_DESC_OFF_c);
MK_CHECK_OFFSET(MkTaskDescriptor, d, MK_DESC_OFF_d);
MK_CHECK_OFFSET(MkTaskDescriptor, e, MK_DESC_OFF_e);
MK_CHECK_OFFSET(MkTaskDescriptor, f, MK_DESC_OFF_f);
MK_CHECK_OFFSET(MkTaskDescriptor, n, MK_DESC_OFF_n);
MK_CHECK_OFFSET(MkTaskDescriptor, k, MK_DESC_OFF_k);
MK_CHECK_OFFSET(MkTaskDescriptor, kt, MK_DESC_OFF_kt);
MK_CHECK_OFFSET(MkTaskDescriptor, group_items, MK_DESC_OFF_group_items);
MK_CHECK_OFFSET(MkTaskDescriptor, item_bytes, MK_DESC_OFF_item_bytes);
MK_CHECK_OFFSET(MkTaskDescriptor, scale_bytes, MK_DESC_OFF_scale_bytes);
MK_CHECK_OFFSET(MkTaskDescriptor, pos0, MK_DESC_OFF_pos0);
MK_CHECK_OFFSET(MkTaskDescriptor, layer, MK_DESC_OFF_layer);
MK_CHECK_OFFSET(MkTaskDescriptor, p0, MK_DESC_OFF_p0);
MK_CHECK_OFFSET(MkTaskDescriptor, p1, MK_DESC_OFF_p1);
MK_CHECK_OFFSET(MkTaskDescriptor, p2, MK_DESC_OFF_p2);
MK_CHECK_OFFSET(MkTaskDescriptor, p3, MK_DESC_OFF_p3);

static_assert(sizeof(MkTaskTable) == MK_TABLE_BYTES, "MkTaskTable differs in size from src/protocol.rs");
static_assert(alignof(MkTaskTable) == MK_TABLE_ALIGN, "MkTaskTable differs in alignment from src/protocol.rs");
MK_CHECK_OFFSET(MkTaskTable, ptr, MK_TABLE_OFF_ptr);

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

// The timer, read after `dependency` is known. A barrier with DEFER_BLOCKING lets a later instruction that does not
// need the barrier run before the barrier ends. A timestamp then reads too early. This read needs a value that the
// code loads after the barrier, and it adds the high bits of that value (always 0) to the time.
__device__ __forceinline__ uint64_t globaltimer_after(uint32_t dependency) {
    uint64_t t;
    asm volatile(
        "{ .reg .u64 g; .reg .u32 z; mov.u64 g, %%globaltimer; shr.u32 z, %1, 16; cvt.u64.u32 %0, z; add.u64 %0, %0, g; }"
        : "=l"(t)
        : "r"(dependency));
    return t;
}

__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* ptr) {
    uint32_t val;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    return val;
}

// A release add at GPU scope. A release is cumulative: it publishes each write that the thread has seen, and the
// __syncthreads before it makes the writes of the other threads of the block seen. Thus no separate fence is needed.
__device__ __forceinline__ void red_release_gpu_add_u32(uint32_t* ptr, uint32_t val) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

// An add at GPU scope that releases the writes before it and acquires the writes of the adds before it. The thread
// that gets the last value of a merge counter sees the partials that every other item wrote.
__device__ __forceinline__ uint32_t atom_acq_rel_gpu_add_u32(uint32_t* ptr, uint32_t val) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(ptr), "r"(val) : "memory");
    return old;
}

// The merge rule. The items of one merge group (the K slices of a tile, or the tiles of a row) each write a partial and
// then call this function with the counter of the group. The item that gets the last count returns true, and it merges
// the partials. That item also sets the counter back to 0 before it merges, because every other item of the group has
// added its count already. So a counter needs a host zero only when the ring opens, and after a launch that failed. A
// new merge group or finisher must call this function and must not reset its counter in another way. The merging
// thread signals the task after the reset, and a later task that reuses the counter waits on that signal, so the reset
// is seen. Only one thread of the item calls this function.
__device__ __forceinline__ bool merge_arrive(uint32_t* counter, uint32_t group_items) {
    if (atom_acq_rel_gpu_add_u32(counter, 1u) != group_items - 1) {
        return false;
    }
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(counter), "r"(0u) : "memory");
    return true;
}

// The threads that do the work of an item. A sum that an op cuts over the workers has this many lanes, whatever the
// grid is, so the bits of a sum do not depend on the launch.
constexpr uint32_t MK_WORKERS = 256;

// The dynamic shared memory of the block. The launch gives its size. Only the probe ops use it for now.
extern __shared__ __align__(128) uint8_t mk_smem[];

// The two shapes of a block. A block of MK_PRODUCER_THREADS threads has a producer warp (warp 0) and
// MK_CONSUMER_THREADS consumer threads. A block of another width is flat: every thread is a consumer. An op runs on
// the consumers only. It uses the functions below, so that it works in both shapes.
__device__ __forceinline__ bool has_producer() { return blockDim.x == MK_PRODUCER_THREADS; }

// The index of the calling consumer among the consumers of the block.
__device__ __forceinline__ uint32_t worker_id() { return has_producer() ? threadIdx.x - 32 : threadIdx.x; }

__device__ __forceinline__ uint32_t worker_count() { return has_producer() ? MK_CONSUMER_THREADS : blockDim.x; }

// The named barriers. Barrier 0 is __syncthreads. Barrier 1 is for the consumers. The producer shape uses a pair of
// barriers for each of the two slots of descriptors (see run_producer_shape).
constexpr uint32_t MK_BAR_CONSUMERS = 1;
constexpr uint32_t MK_BAR_FULL = 2;   // slots 0 and 1: barriers 2 and 3
constexpr uint32_t MK_BAR_EMPTY = 4;  // slots 0 and 1: barriers 4 and 5

__device__ __forceinline__ void bar_sync(uint32_t id, uint32_t threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// The thread counts as arrived and does not wait.
__device__ __forceinline__ void bar_arrive(uint32_t id, uint32_t threads) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// A barrier of the consumers.
__device__ __forceinline__ void worker_sync() {
    if (has_producer()) {
        bar_sync(MK_BAR_CONSUMERS, MK_CONSUMER_THREADS);
    } else {
        __syncthreads();
    }
}

// Item `item` copies `n` floats: the floats from `item * n` up to `(item + 1) * n`. The task covers `items * n` floats.
// The cut comes from the shape of the task, not from the grid, so the bits do not depend on which block runs an item.
// a = dst (float*), b = src (const float*), n = floats in each item.
__device__ __forceinline__ void op_copy(const MkTaskDescriptor& task, uint32_t item) {
    float* dst = reinterpret_cast<float*>(task.a);
    const float* src = reinterpret_cast<const float*>(task.b);
    // The offsets are 64 bits: a task can cover more than 2^32 floats.
    const uint64_t first = static_cast<uint64_t>(item) * task.n;
    for (uint32_t i = worker_id(); i < task.n; i += worker_count()) {
        dst[first + i] = src[first + i];
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
    const uint32_t words = task.n;
    if (blockIdx.x == task.kt && worker_id() == 0) {
        const uint64_t until = globaltimer_ns() + task.k;
        while (globaltimer_ns() < until) {
        }
    }
    worker_sync();
    for (uint32_t i = worker_id(); i < words; i += worker_count()) {
        data[static_cast<uint64_t>(blockIdx.x) * words + i] = selftest_value(task.group_items, blockIdx.x, i);
    }
}

// Each block reads the words of all blocks and counts the words that differ from selftest_value. Thread 0 writes the
// count to e[blockIdx.x].
__device__ __forceinline__ void op_selftest_check(const MkTaskDescriptor& task) {
    const uint32_t* data = reinterpret_cast<const uint32_t*>(task.a);
    uint32_t* out = reinterpret_cast<uint32_t*>(task.e);
    const uint32_t words = task.n;
    const uint64_t total = static_cast<uint64_t>(gridDim.x) * words;
    uint32_t wrong = 0;
    __shared__ uint32_t warp_wrong[32];
    for (uint64_t base = 0; base < total; base += worker_count()) {
        const uint64_t k = base + worker_id();
        bool differs = false;
        if (k < total) {
            differs = data[k] != selftest_value(task.group_items, static_cast<uint32_t>(k / words), static_cast<uint32_t>(k % words));
        }
        // The count of the words that differ, over the consumers of the block.
        const uint32_t in_warp = __popc(__ballot_sync(0xffffffffu, differs));
        if ((worker_id() & 31) == 0) {
            warp_wrong[worker_id() >> 5] = in_warp;
        }
        worker_sync();
        for (uint32_t w = 0; w < worker_count() / 32; ++w) {
            wrong += warp_wrong[w];
        }
        worker_sync();
    }
    if (worker_id() == 0) {
        out[blockIdx.x] = wrong;
    }
}

// Item `item` writes the words of its task table to `c + item * MK_TABLE_POINTERS` words. A task with no table traps.
__device__ __forceinline__ void op_selftest_table(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    if (table == nullptr) {
        __trap();
    }
    uint64_t* out = reinterpret_cast<uint64_t*>(task.c);
    if (worker_id() < MK_TABLE_POINTERS) {
        out[static_cast<uint64_t>(item) * MK_TABLE_POINTERS + worker_id()] = table->ptr[worker_id()];
    }
}

#include "probe_ops.cuh"
#include "matmul_bf16.cuh"
#include "quant.cuh"
#include "cache_layout.cuh"
#include "experts.cuh"
#include "ops_dense.cuh"
#include "ops_rows.cuh"
#include "ops_moe.cuh"
#include "ops_attn.cuh"

// Thread 0 polls the event until it reaches the target. A lost signal traps at the spin limit.
__device__ __forceinline__ void wait_for_event(const uint32_t* event, uint32_t target) {
    uint64_t polls = 0;
    while (ld_acquire_gpu_u32(event) < target) {
        if (++polls >= MK_SPIN_LIMIT) {
            __trap();
        }
    }
}

// Run work item `item` of the task. `table` is the task table, or null when `d` is 0. A per-item op uses `item`. The probe and test ops run once in each block, so their
// item is blockIdx.x, and they read that.
__device__ __forceinline__ void run_op(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    switch (task.opcode) {
        case OP_NOP:
            break;
        case OP_COPY:
            op_copy(task, item);
            break;
        case OP_MATMUL_BF16:
            op_matmul_bf16(task, table, item);
            break;
        case OP_SELFTEST_FILL:
            op_selftest_fill(task);
            break;
        case OP_SELFTEST_CHECK:
            op_selftest_check(task);
            break;
        case OP_SELFTEST_TABLE:
            op_selftest_table(task, table, item);
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
        case OP_REDUCE_PROBE:
            op_reduce_probe(task, item);
            break;
        case OP_QUANT_NVFP4:
            op_quant_nvfp4(task, item);
            break;
        case OP_QUANT_MXFP8:
            op_quant_mxfp8(task, item);
            break;
        case OP_EXPERT_NVFP4:
            op_expert_nvfp4(task, table, item);
            break;
        case OP_EXPERT_MXFP8:
            op_expert_mxfp8(task, table, item);
            break;

        // The cases below are reserved for the lanes of wave B. Each calls a stub that traps, in the file of its lane.
        // A lane changes the stub of its own ops, and not this switch.
        case OP_PROJ:
            op_proj(task, table, item);
            break;

        case OP_OUT_MIX:
            op_out_mix(task, table, item);
            break;

        case OP_ROUTE:
            op_route(task, table, item);
            break;

        case OP_EMBED:
            op_embed(task, table, item);
            break;

        case OP_MIX:
            op_mix(task, table, item);
            break;

        case OP_GATE_DOWN:
            op_gate_down(task, table, item);
            break;

        case OP_GATE_UP:
            op_gate_up(task, table, item);
            break;

        case OP_MIX_OUT:
            op_mix_out(task, table, item);
            break;

        case OP_ROWS_PROBE:
            op_rows_probe(task, table, item);
            break;

        case OP_ATTN:
            op_attn(task, table, item);
            break;

        default:
            // Unknown opcode: trap rather than silently corrupt state (spec 8.5).
            __trap();
            break;
    }
}

// The parameters of the kernel, read again from the parameter bank at each use. The loops of the interpreter use these
// functions, and not the parameters as values, so that no parameter takes a register across an op. The parameter bank is
// read-only and a read is as cheap as a register move. The names are those of the PTX of the kernel at the end of this
// file, so these functions must be inline in the kernel.
__device__ __forceinline__ uint32_t mk_num_tasks() {
    uint32_t v;
    asm volatile("ld.param.u32 %0, [mk_persistent_interpreter_param_1];" : "=r"(v));
    return v;
}
__device__ __forceinline__ uint32_t* mk_scoreboard() {
    uint64_t v;
    asm volatile("ld.param.u64 %0, [mk_persistent_interpreter_param_2];" : "=l"(v));
    return reinterpret_cast<uint32_t*>(v);
}
__device__ __forceinline__ uint32_t mk_base_event() {
    uint32_t v;
    asm volatile("ld.param.u32 %0, [mk_persistent_interpreter_param_3];" : "=r"(v));
    return v;
}
__device__ __forceinline__ BlockStats* mk_stats() {
    uint64_t v;
    asm volatile("ld.param.u64 %0, [mk_persistent_interpreter_param_4];" : "=l"(v));
    return reinterpret_cast<BlockStats*>(v);
}
__device__ __forceinline__ uint64_t* mk_profile() {
    uint64_t v;
    asm volatile("ld.param.u64 %0, [mk_persistent_interpreter_param_6];" : "=l"(v));
    return reinterpret_cast<uint64_t*>(v);
}

// Write the three times of a task in this block to the profile.
__device__ __forceinline__ void record_profile(uint64_t* profile, uint32_t idx, uint64_t t_start, uint64_t t_after_wait, uint64_t t_end) {
#ifdef MK_PROFILE
    if (profile != nullptr) {
        uint64_t* slot = profile + ((static_cast<uint64_t>(idx) * gridDim.x + blockIdx.x) * MK_PROFILE_WORDS);
        slot[0] = t_start;
        slot[1] = t_after_wait;
        slot[2] = t_end;
    }
#else
    (void)profile; (void)idx; (void)t_start; (void)t_after_wait; (void)t_end;
#endif
}

// The count of items of a task. A task with no items is a JOIN: block 0 runs one item that does no work, and it
// signals once. A task whose live count comes from a plan on the device has a static bound here, and the op reads the
// live count: an item beyond it does nothing, and it counts as done. So the count is never read from the device, and
// 0xFFFFFFFF is a defect of the ring.
__device__ __forceinline__ uint32_t item_count(const MkTaskDescriptor& task) {
    if (task.items == 0xFFFFFFFFu) {
        __trap();
    }
    return task.items == 0 ? 1u : task.items;
}

// The consumers run the items b, b + G, b + 2G and so on of the task, with b the block and G the grid width. Then the
// block adds the count of its items to the event of the task, one time. A block with no item adds nothing. The
// barrier before the add orders the writes of every consumer before the release of the add.
__device__ __forceinline__ void run_items(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t count) {
    // The loop keeps the item and no other counter across an op: the count of the items that this block ran follows from
    // the item at the end, because the item steps from `blockIdx.x` by `gridDim.x`.
    uint32_t item = blockIdx.x;
    for (; item < count; item += gridDim.x) {
        if (task.items != 0) {
            run_op(task, table, item);
        }
        worker_sync();
    }
    if (worker_id() == 0 && item != blockIdx.x && task.signal != NO_EVENT) {
        red_release_gpu_add_u32(mk_scoreboard() + (mk_base_event() + task.signal), (item - blockIdx.x) / gridDim.x);
    }
}

// The wait of a task: each event that the task names reaches its target.
__device__ __forceinline__ void wait_for_task(const MkTaskDescriptor& task, const uint32_t* scoreboard, uint32_t base_event) {
    if (task.wait0_event != NO_EVENT) {
        wait_for_event(scoreboard + (base_event + task.wait0_event), task.wait0_target);
    }
    if (task.wait1_event != NO_EVENT) {
        wait_for_event(scoreboard + (base_event + task.wait1_event), task.wait1_target);
    }
}

__device__ __forceinline__ void write_stats(BlockStats* stats, uint32_t tasks_run, uint64_t wait_ns, uint64_t exec_ns) {
    if (stats != nullptr) {
        stats[blockIdx.x].tasks_run = tasks_run;
        stats[blockIdx.x].pad = 0;
        stats[blockIdx.x].wait_ns = wait_ns;
        stats[blockIdx.x].exec_ns = exec_ns;
    }
}

// The flat shape: every thread is a consumer. Thread 0 waits at the scoreboard before the first item of a task. Only
// the probes use this shape.
__device__ __forceinline__ void run_flat_shape(
    const MkTaskDescriptor* tasks, uint32_t num_tasks, uint32_t* scoreboard, uint32_t base_event, BlockStats* stats, uint64_t* profile) {
    // The totals and the timestamps of thread 0 are in shared memory, so that they take no register across an op.
    __shared__ uint64_t clocks[5];  // tasks run, wait ns, exec ns, t_start, t_after_wait
    if (threadIdx.x == 0) {
        clocks[0] = clocks[1] = clocks[2] = 0;
    }

    for (uint32_t idx = 0; idx < mk_num_tasks(); ++idx) {
        const MkTaskDescriptor task = tasks[idx];
        const bool halt = task.opcode == OP_HALT;
        const uint32_t count = halt ? 0 : item_count(task);

        if (threadIdx.x == 0) {
            clocks[3] = globaltimer_ns();
            // A block that has no item of the task does not wait. A wait on the task still sees every item done.
            if (!halt && blockIdx.x < count) {
                wait_for_task(task, mk_scoreboard(), mk_base_event());
            }
            clocks[4] = globaltimer_ns();
            clocks[1] += clocks[4] - clocks[3];
        }
        __syncthreads();

        if (!halt) {
            // The flat shape reads the table from global memory. It is the same for the whole run.
            run_items(task, task.d != 0 ? reinterpret_cast<const MkTaskTable*>(task.d) : nullptr, count);
        }
        if (threadIdx.x == 0) {
            const uint64_t t_end = globaltimer_ns();
            clocks[2] += t_end - clocks[4];
            record_profile(mk_profile(), idx, clocks[3], clocks[4], t_end);
            clocks[0] += halt ? 0 : 1;
        }
        if (halt) {
            break;
        }
    }
    if (threadIdx.x == 0) {
        write_stats(mk_stats(), static_cast<uint32_t>(clocks[0]), clocks[1], clocks[2]);
    }
}

// The barriers of the producer shape use all the threads of the block: the 32 of the producer and the consumers.
constexpr uint32_t MK_BAR_THREADS = MK_PRODUCER_THREADS;

// The producer shape. Warp 0 is the producer and warps 1 to 8 are the consumers. The two sides pass tasks through two
// slots of descriptors in shared memory. For the slot s of a task t (s = t mod 2):
// - The producer waits for the barrier EMPTY[s] (the consumers are done with the task t - 2). It copies the
//   descriptor and the task table to the slot, asks L2 to prefetch the first item of the block, and waits at the scoreboard. Then it
//   arrives at FULL[s]. It runs ahead of the consumers by up to two tasks, so the wait of a task that does not depend
//   on the task before it ends while the consumers work.
// - The consumers wait at FULL[s], run the items and arrive at EMPTY[s].
// The arrive of the producer orders the descriptor and the data that its acquire saw before the reads of the
// consumers. A named barrier orders the memory of the threads that use it.
__device__ __forceinline__ void run_producer_shape(
    const MkTaskDescriptor* tasks, uint32_t num_tasks, uint32_t* scoreboard, uint32_t base_event, BlockStats* stats, uint64_t* profile) {
    __shared__ MkTaskDescriptor slots[2];
    __shared__ MkTaskTable tables[2];

    if (threadIdx.x < 32) {
        const bool lead = threadIdx.x == 0;
        uint32_t halt_at = num_tasks;
        for (uint32_t t = 0; t < num_tasks; ++t) {
            const uint32_t slot = t & 1;
            if (t >= 2) {
                bar_sync(MK_BAR_EMPTY + slot, MK_BAR_THREADS);
            }
            if (lead) {
                slots[slot] = tasks[t];
                // The table is the same for the whole run, so the producer copies it now, with the descriptor, before
                // its wait. The consumers read it after the arrive at FULL, as they read the descriptor.
                const uint64_t table_at = slots[slot].d;
                if (table_at != 0) {
                    if ((table_at & (MK_TABLE_ALIGN - 1)) != 0) {
                        __trap();  // A table is 128-byte aligned. A nonzero `d` that is not one is a defect of the task.
                    }
                    tables[slot] = *reinterpret_cast<const MkTaskTable*>(table_at);
                }
            }
            __syncwarp();
            const MkTaskDescriptor& task = slots[slot];
            const bool halt = task.opcode == OP_HALT;
            if (!halt) {
                const uint32_t count = item_count(task);
                // A block that has no item of the task does not wait.
                if (blockIdx.x < count) {
                    // A weight does not depend on the task before it, so the prefetch starts before the wait.
                    if (lead && task.item_bytes != 0) {
                        if ((task.item_bytes & 15u) != 0 || (task.b & 15u) != 0) {
                            __trap();  // A bulk prefetch needs a size and an address of 16 bytes.
                        }
                        bulk_prefetch_l2(reinterpret_cast<const uint8_t*>(task.b) + static_cast<uint64_t>(blockIdx.x) * task.item_bytes, task.item_bytes);
                    }
                    // Every lane polls the same word, which is one load. Thus the lanes stay converged.
                    wait_for_task(task, scoreboard, base_event);
                }
            }
            __syncwarp();
            bar_arrive(MK_BAR_FULL + slot, MK_BAR_THREADS);
            if (halt) {
                halt_at = t;
                break;
            }
        }
        if (halt_at == num_tasks) {
            __trap();  // The ring has no OP_HALT: the host appends one, so this is a defect.
        }
        // The consumers arrived at EMPTY for each task. The loop above waited for all of them except the last.
        if (halt_at >= 1) {
            bar_sync(MK_BAR_EMPTY + ((halt_at - 1) & 1), MK_BAR_THREADS);
        }
        return;
    }

    // The totals and the timestamps of the lead consumer are in shared memory, so that they take no register across an op.
    __shared__ uint64_t clocks[5];  // tasks run, wait ns, exec ns, t_start, t_after_wait
    const bool lead = worker_id() == 0;
    if (lead) {
        clocks[0] = clocks[1] = clocks[2] = 0;
    }
    for (uint32_t idx = 0; idx < mk_num_tasks(); ++idx) {
        const uint32_t slot = idx & 1;
        if (lead) {
            clocks[3] = globaltimer_ns();
        }
        bar_sync(MK_BAR_FULL + slot, MK_BAR_THREADS);
        const MkTaskDescriptor& task = slots[slot];
        const uint32_t opcode = task.opcode;
        // The opcode is loaded after the barrier, so the timer below reads after the barrier too.
        if (lead) {
            clocks[4] = globaltimer_after(opcode);
        }
        const bool halt = opcode == OP_HALT;
        if (!halt) {
            run_items(task, task.d != 0 ? &tables[slot] : nullptr, item_count(task));
        }
        if (lead) {
            const uint64_t t_end = globaltimer_ns();
            clocks[1] += clocks[4] - clocks[3];
            clocks[2] += t_end - clocks[4];
            record_profile(mk_profile(), idx, clocks[3], clocks[4], t_end);
            clocks[0] += halt ? 0 : 1;
        }
        if (halt) {
            break;
        }
        bar_arrive(MK_BAR_EMPTY + slot, MK_BAR_THREADS);
    }
    if (lead) {
        write_stats(mk_stats(), static_cast<uint32_t>(clocks[0]), clocks[1], clocks[2]);
    }
}

// The production shape sets the register budget. A block of 288 threads has 9 warps. A multiprocessor has 4 partitions
// of 16,384 registers, and the warps go to them in turn, so one partition holds 3 warps: a thread has at most 168
// registers (not 65,536 / 288). A flat shape has at most 288 threads. The build fails on a spill (build.rs).
extern "C" __global__ void __launch_bounds__(288, 1) mk_persistent_interpreter(
    const MkTaskDescriptor* __restrict__ tasks,
    uint32_t num_tasks,
    uint32_t* __restrict__ scoreboard,
    uint32_t base_event,
    BlockStats* __restrict__ stats,
    float* __restrict__ scratch,
    uint64_t* __restrict__ profile
) {
    (void)scratch;
    if (has_producer()) {
        run_producer_shape(tasks, num_tasks, scoreboard, base_event, stats, profile);
    } else {
        run_flat_shape(tasks, num_tasks, scoreboard, base_event, stats, profile);
    }
}
