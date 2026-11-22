// The row ops of one Inkling 2 run: embedding, mixes and message gates (OP_EMBED, OP_MIX, OP_GATE_DOWN, OP_GATE_UP,
// OP_MIX_OUT), and the probe of the row finisher (OP_ROWS_PROBE).
//
// This file belongs to wave B lane R (rows). `run_op` in megakernel.cu calls each op.
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor`, `MkTaskTable` and the worker functions of megakernel.cu.
//
// The row factor. A row of the residual stream has a factor `rf = 1 / sqrt(mean(v^2) + 1e-6)`. The task that writes
// a row also writes its `rf`, in an array of one float for each row. A task that reads the row as a normed row
// computes `v * rf` and does not read the whole row again (design rule 1).

#pragma once

constexpr uint32_t RW_WARPS = 8;
constexpr uint32_t RW_HIDDEN = 2304;
// The epsilon of the norms (`ops::EPS` of the host form).
constexpr float RW_EPS = 1e-6f;
// The most sums that `rw_block_sum` joins in one call.
constexpr uint32_t RW_MAX_SUMS = 16;
// Shared memory of the row ops: the scratch of `rw_block_sum`, then 16 bytes of flags and values.
constexpr uint32_t RW_SUM_FLOATS = RW_MAX_SUMS * RW_WARPS;
constexpr uint32_t RW_SMEM_BYTES = RW_SUM_FLOATS * 4 + 64;

// The sums of the 256 workers, each one joined in a fixed order: a butterfly of shuffles in each warp, then the 8
// warps as ((w0 + w1) + (w2 + w3)) + ((w4 + w5) + (w6 + w7)). Every worker gets the same bits, and they do not depend
// on the grid. `scratch` has `N * 8` floats. The function ends with a barrier, so a caller can call it again.
template <uint32_t N>
__device__ __forceinline__ void rw_block_sum(float (&v)[N], float* scratch) {
    const uint32_t warp = worker_id() >> 5;
    const uint32_t lane = worker_id() & 31;
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
#pragma unroll
        for (uint32_t offset = 16; offset > 0; offset >>= 1) {
            v[i] += __shfl_xor_sync(0xFFFFFFFFu, v[i], offset);
        }
        if (lane == 0) {
            scratch[i * RW_WARPS + warp] = v[i];
        }
    }
    worker_sync();
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
        const float* s = scratch + i * RW_WARPS;
        v[i] = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    }
    worker_sync();
}

// The row finisher: a merge group of a second level. Each item of a row tile writes its partial for each column tile
// that it covers, and then every worker of the item calls this function with the counter of the row tile. The item
// that arrives last returns true in all its workers, and it does the work of the whole row: it reads the partials
// with `__ldcg`, adds them in the order of the tiles with `rw_ordered_sum`, and writes the results. The counter is
// reset by `merge_arrive`, so a counter needs a zero only at the open of the ring. `flag` is 4 bytes of shared memory.
__device__ __forceinline__ bool rw_finisher_arrive(uint32_t* counter, uint32_t group_items, uint32_t* flag) {
    worker_sync();  // The partials of all workers are written, so the release of the add publishes them.
    if (worker_id() == 0) {
        *flag = merge_arrive(counter, group_items) ? 1u : 0u;
    }
    worker_sync();
    const bool last = *flag != 0;
    worker_sync();  // Every worker has read the flag, so the next call can write it.
    return last;
}

// The sum of `count` partials `p[0]`, `p[stride]`, ... in the order of the index.
__device__ __forceinline__ float rw_ordered_sum(const float* p, uint32_t count, uint32_t stride) {
    float sum = __ldcg(p);
    for (uint32_t i = 1; i < count; ++i) {
        sum += __ldcg(p + static_cast<uint64_t>(i) * stride);
    }
    return sum;
}

// The row factor of a sum of squares over `width` values.
__device__ __forceinline__ float rw_factor(float sum_squares, uint32_t width) {
    return 1.0f / sqrtf(sum_squares / static_cast<float>(width) + RW_EPS);
}

// OP_ROWS_PROBE: the probe of the row finisher. `rows` rows (1 to 16) of `group_items * n` floats are at `a`. Item `i`
// is the column tile `i`: the `n` floats of each row from `i * n`. It writes the sum of squares of the tile of each
// row to a partial, and the last item of the group sums the partials in the order of the tiles. Then it writes
// the sums to `c[0 .. rows)`, the row factors to `c[rows .. 2 rows)`, and the normed rows `x * rf` to `e`
// (`rows` rows of `group_items * n` floats). `f` holds the counter, the count of finishers that ran (it must be 1 for
// each run), two words that the op does not use, then `group_items * rows` partials.
__device__ __forceinline__ void op_rows_probe(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    (void)table;
    const uint32_t rows = task.rows;
    const uint32_t tiles = task.group_items;
    const uint32_t width = task.n;
    float* scratch = reinterpret_cast<float*>(mk_smem);
    uint32_t* flag = reinterpret_cast<uint32_t*>(mk_smem + RW_SUM_FLOATS * 4);
    if (worker_count() != MK_WORKERS || rows == 0 || rows > RW_MAX_SUMS || tiles == 0 || width == 0 ||
        task.items != tiles || dynamic_smem_bytes() < RW_SMEM_BYTES || (task.a & 15) != 0 || (task.f & 15) != 0 ||
        (task.e & 15) != 0 || (task.c & 3) != 0) {
        __trap();
    }
    const float* x = reinterpret_cast<const float*>(task.a);
    const uint64_t stride = static_cast<uint64_t>(tiles) * width;
    uint32_t* counter = reinterpret_cast<uint32_t*>(task.f);
    float* partials = reinterpret_cast<float*>(task.f) + 4;

    float sums[RW_MAX_SUMS];
#pragma unroll
    for (uint32_t r = 0; r < RW_MAX_SUMS; ++r) {
        sums[r] = 0.0f;
        if (r < rows) {
            const float* row = x + r * stride + static_cast<uint64_t>(item) * width;
            for (uint32_t j = worker_id(); j < width; j += MK_WORKERS) {
                const float v = __ldcg(row + j);
                sums[r] += v * v;
            }
        }
    }
    rw_block_sum(sums, scratch);
    if (worker_id() < rows) {
        float mine = 0.0f;
#pragma unroll
        for (uint32_t r = 0; r < RW_MAX_SUMS; ++r) {
            mine = (r == worker_id()) ? sums[r] : mine;
        }
        __stcg(partials + static_cast<uint64_t>(item) * rows + worker_id(), mine);
    }
    if (!rw_finisher_arrive(counter, tiles, flag)) {
        return;
    }
    if (worker_id() == 0) {
        atomicAdd(counter + 1, 1u);
    }
    float* factors = scratch;  // `rows` floats: the scratch of the sums is free now.
    if (worker_id() < rows) {
        const float total = rw_ordered_sum(partials + worker_id(), tiles, rows);
        const float rf = rw_factor(total, tiles * width);
        reinterpret_cast<float*>(task.c)[worker_id()] = total;
        reinterpret_cast<float*>(task.c)[rows + worker_id()] = rf;
        factors[worker_id()] = rf;
    }
    worker_sync();
    float* out = reinterpret_cast<float*>(task.e);
    for (uint64_t i = worker_id(); i < rows * stride; i += MK_WORKERS) {
        out[i] = __ldcg(x + i) * factors[i / stride];
    }
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_embed(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    (void)task;
    (void)table;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_mix(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    (void)task;
    (void)table;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_gate_down(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    (void)task;
    (void)table;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_gate_up(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    (void)task;
    (void)table;
    (void)item;
    __trap();
}

// Reserved for wave B lane R (rows).
__device__ __forceinline__ void op_mix_out(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    (void)task;
    (void)table;
    (void)item;
    __trap();
}
