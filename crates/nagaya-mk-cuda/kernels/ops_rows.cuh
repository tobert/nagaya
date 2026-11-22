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

// ---------------------------------------------------------------------------------------------------------------
// OP_EMBED, OP_MIX and OP_MIX_OUT.
// ---------------------------------------------------------------------------------------------------------------

constexpr uint32_t RW_PER_THREAD = RW_HIDDEN / MK_WORKERS;  // 9 values of a row for each worker
static_assert(RW_HIDDEN % MK_WORKERS == 0, "a row must divide over the workers");

// The slots of the task table of OP_EMBED.
constexpr uint32_t RW_EMBED_NORM = 0;     // the weight of the embedding norm (f32, 2304)
constexpr uint32_t RW_EMBED_PLANE0 = 1;   // the first plane of the attention residual, B[0] (f32 rows)
constexpr uint32_t RW_EMBED_PLANE0_RF = 2;  // the row factors of B[0]

// OP_EMBED: item `r` is the row `r`. a = ids (u32, `rows`), b = the embedding (bf16, `n` rows of 2304), c = x (f32,
// `rows` rows), e = the row factors of x, n = the count of rows of the embedding. It writes the normed row
// `w * (v * f)` to x and to B[0], with `f` the factor of the raw row. Then it writes the row factor of x to `e` and
// to the factors of B[0].
__device__ __forceinline__ void op_embed(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    float* scratch = reinterpret_cast<float*>(mk_smem);
    if (worker_count() != MK_WORKERS || table == nullptr || task.items != task.rows || item >= task.rows ||
        dynamic_smem_bytes() < RW_SMEM_BYTES || task.n == 0 || (task.a & 3) != 0 || (task.b & 1) != 0 ||
        (task.c & 15) != 0 || (task.e & 3) != 0 || table->ptr[RW_EMBED_NORM] == 0 ||
        table->ptr[RW_EMBED_PLANE0] == 0 || table->ptr[RW_EMBED_PLANE0_RF] == 0) {
        __trap();
    }
    const uint32_t id = __ldg(reinterpret_cast<const uint32_t*>(task.a) + item);
    if (id >= task.n) {
        __trap();  // The token is not in the vocabulary. The host checks it, so this is a defect.
    }
    const uint16_t* row = reinterpret_cast<const uint16_t*>(task.b) + static_cast<uint64_t>(id) * RW_HIDDEN;
    const float* w = reinterpret_cast<const float*>(table->ptr[RW_EMBED_NORM]);
    float v[RW_PER_THREAD];
    float sum[1] = {0.0f};
#pragma unroll
    for (uint32_t j = 0; j < RW_PER_THREAD; ++j) {
        v[j] = __uint_as_float(static_cast<uint32_t>(__ldg(row + worker_id() + j * MK_WORKERS)) << 16);
        sum[0] += v[j] * v[j];
    }
    rw_block_sum(sum, scratch);
    const float rf_raw = rw_factor(sum[0], RW_HIDDEN);
    float* x = reinterpret_cast<float*>(task.c) + static_cast<uint64_t>(item) * RW_HIDDEN;
    float* plane0 = reinterpret_cast<float*>(table->ptr[RW_EMBED_PLANE0]) + static_cast<uint64_t>(item) * RW_HIDDEN;
    float squares[1] = {0.0f};
#pragma unroll
    for (uint32_t j = 0; j < RW_PER_THREAD; ++j) {
        const uint32_t e = worker_id() + j * MK_WORKERS;
        const float out = __ldg(w + e) * (v[j] * rf_raw);
        x[e] = out;
        plane0[e] = out;
        squares[0] += out * out;
    }
    // The row factor that the next task reads is the factor of the row that it wrote: the normed row.
    rw_block_sum(squares, scratch);
    if (worker_id() == 0) {
        const float rf = rw_factor(squares[0], RW_HIDDEN);
        reinterpret_cast<float*>(task.e)[item] = rf;
        reinterpret_cast<float*>(table->ptr[RW_EMBED_PLANE0_RF])[item] = rf;
    }
}

// The slots of the task table of OP_MIX and OP_MIX_OUT.
constexpr uint32_t RW_MIX_STACK = 0;     // slots 0 to 4: the rows of the stack (f32, `rows` rows each)
constexpr uint32_t RW_MIX_STACK_RF = 5;  // slots 5 to 9: their row factors
constexpr uint32_t RW_MIX_MSG = 10;      // slots 10, 11: the stack of the message mix
constexpr uint32_t RW_MIX_MSG_RF = 12;   // slots 12, 13: its row factors
constexpr uint32_t RW_MIX_QUERIES = 14;  // the queries of the attention mix (f32, `n` rows of 2304)
constexpr uint32_t RW_MIX_MSG_QUERIES = 15;  // the queries of the message mix (f32, `kt` rows of 2304)
constexpr uint32_t RW_MAX_STACK = 5;

// One mix item: `out = sum_s softmax_s(q . normed(stack_s) * scaling) * normed(stack_s)`, then the row factor of `out`.
// The stack is `depth` rows, each `ptr[s] + row * 2304` with the factor `rf[s][row]`.
__device__ __forceinline__ void rw_mix_item(const MkTaskTable* table, uint32_t stack0, uint32_t rf0, uint32_t depth,
                                            const float* query, uint32_t row, float scaling, float* out,
                                            float* out_rf) {
    float* scratch = reinterpret_cast<float*>(mk_smem);
    const float* planes[RW_MAX_STACK];
    float factor[RW_MAX_STACK];
#pragma unroll
    for (uint32_t s = 0; s < RW_MAX_STACK; ++s) {
        planes[s] = nullptr;
        factor[s] = 0.0f;
        if (s < depth) {
            planes[s] = reinterpret_cast<const float*>(table->ptr[stack0 + s]) + static_cast<uint64_t>(row) * RW_HIDDEN;
            factor[s] = __ldcg(reinterpret_cast<const float*>(table->ptr[rf0 + s]) + row);
        }
    }
    // Pass 1: the score of each row of the stack. The rows are read again in pass 2, from L1 or L2, so that the
    // block holds few registers (the kernel has a bound).
    float score[RW_MAX_STACK];
#pragma unroll
    for (uint32_t s = 0; s < RW_MAX_STACK; ++s) {
        score[s] = 0.0f;
    }
#pragma unroll
    for (uint32_t j = 0; j < RW_PER_THREAD; ++j) {
        const float q = __ldg(query + worker_id() + j * MK_WORKERS);
#pragma unroll
        for (uint32_t s = 0; s < RW_MAX_STACK; ++s) {
            if (s < depth) {
                score[s] += q * (__ldcg(planes[s] + worker_id() + j * MK_WORKERS) * factor[s]);
            }
        }
    }
    rw_block_sum(score, scratch);
    // The softmax of the rows of the stack, in the order of the host form.
    float top = -INFINITY;
#pragma unroll
    for (uint32_t s = 0; s < RW_MAX_STACK; ++s) {
        if (s < depth) {
            score[s] *= scaling;
            top = fmaxf(top, score[s]);
        }
    }
    float weight[RW_MAX_STACK];
    float total = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < RW_MAX_STACK; ++s) {
        weight[s] = 0.0f;
        if (s < depth) {
            weight[s] = expf(score[s] - top);
            total += weight[s];
        }
    }
#pragma unroll
    for (uint32_t s = 0; s < RW_MAX_STACK; ++s) {
        weight[s] = weight[s] / total;
    }
    // Pass 2: the mixed row and the sum of its squares.
    float squares[1] = {0.0f};
#pragma unroll
    for (uint32_t j = 0; j < RW_PER_THREAD; ++j) {
        float acc = 0.0f;
#pragma unroll
        for (uint32_t s = 0; s < RW_MAX_STACK; ++s) {
            if (s < depth) {
                acc += weight[s] * (__ldcg(planes[s] + worker_id() + j * MK_WORKERS) * factor[s]);
            }
        }
        out[worker_id() + j * MK_WORKERS] = acc;
        squares[0] += acc * acc;
    }
    rw_block_sum(squares, scratch);
    if (worker_id() == 0) {
        *out_rf = rw_factor(squares[0], RW_HIDDEN);
    }
}

// OP_MIX: the attention residual of `n` queries over a stack of `k` rows, and the message mix of `kt` queries over a
// stack of `group_items` rows. Item `i` is one (query, row): the first `n * rows` items are the attention mix, in
// order of query then row, and the rest are the message mix. `a` = the outputs of the attention mix (f32, `n` x `rows`
// rows), `b` = their row factors, `c` = the outputs of the message mix (`kt` x `rows` rows), `e` = their row factors,
// `p0` = the scaling `2304^-0.5`. The table names the rows (RW_MIX_*).
__device__ __forceinline__ void rw_mix_dispatch(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item,
                                                uint32_t queries, uint32_t depth, uint32_t message_queries,
                                                uint32_t message_depth) {
    const uint32_t rows = task.rows;
    if (worker_count() != MK_WORKERS || table == nullptr || dynamic_smem_bytes() < RW_SMEM_BYTES || rows == 0 ||
        queries + message_queries == 0 || task.items != rows * (queries + message_queries) ||
        (queries != 0 && (depth < 1 || depth > RW_MAX_STACK || table->ptr[RW_MIX_QUERIES] == 0 || (task.a & 15) != 0)) ||
        (message_queries != 0 &&
         (message_depth < 1 || message_depth > 2 || table->ptr[RW_MIX_MSG_QUERIES] == 0 || (task.c & 15) != 0)) ||
        !(task.p0 > 0.0f)) {
        __trap();
    }
    if (item < rows * queries) {
        const uint32_t q = item / rows;
        const uint32_t row = item % rows;
        rw_mix_item(table, RW_MIX_STACK, RW_MIX_STACK_RF, depth,
                    reinterpret_cast<const float*>(table->ptr[RW_MIX_QUERIES]) + static_cast<uint64_t>(q) * RW_HIDDEN, row,
                    task.p0, reinterpret_cast<float*>(task.a) + (static_cast<uint64_t>(q) * rows + row) * RW_HIDDEN,
                    reinterpret_cast<float*>(task.b) + q * rows + row);
    } else {
        const uint32_t j = item - rows * queries;
        const uint32_t q = j / rows;
        const uint32_t row = j % rows;
        rw_mix_item(table, RW_MIX_MSG, RW_MIX_MSG_RF, message_depth,
                    reinterpret_cast<const float*>(table->ptr[RW_MIX_MSG_QUERIES]) + static_cast<uint64_t>(q) * RW_HIDDEN,
                    row, task.p0, reinterpret_cast<float*>(task.c) + (static_cast<uint64_t>(q) * rows + row) * RW_HIDDEN,
                    reinterpret_cast<float*>(task.e) + q * rows + row);
    }
}

__device__ __forceinline__ void op_mix(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    rw_mix_dispatch(task, table, item, task.n, task.k, task.kt, task.group_items);
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
