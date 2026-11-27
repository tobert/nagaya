// OP_OUT_MIX: the output projection of the attention, its short convolution, the residual, and the mix of the MLP input.
//
// This file belongs to wave B lane D (dense). It needs `matmul_bf16.cuh` (the partials and the merge), and
// `ops_rows.cuh` (`rw_mix_item`, `rw_finisher_arrive`, `rw_block_sum`).
//
// The host form is the part of `Model::layer` (`host/forward.rs`) between the attention and the experts:
//
//   o      = wo . attn                                      the matmul, K = 3,072, 2,304 columns
//   c      = (conv(state, o) + o) / sqrt(5)                 `ops::short_conv`, `attn_conv`: [2304][4], tap 3 on the current row
//   r      = x + c                                          FLAG_ACCUMULATE; the first layer of a block has r = c
//   e      = mix(B[0 .. b), r)                              `ops::attention_residual` with the query `mlp_query`
//   h_e    = mlp_norm * (e * rf(e))                         FLAG_NORM: the input of the router, when no message gate follows
//
// The matmul is 144 tiles of 16 columns with 4 K slices of 768 (576 items), the rule of `OP_MATMUL_BF16`: the item that
// arrives last in the group of a tile merges the 4 partials in slice order. Then, in the same item and for the 16
// channels of the tile, over all the rows of the task:
//   1. The merge writes the raw sums `o` to the scratch, as `[tile][row][16]`.
//   2. The convolution: row `t` takes the 3 rows before it from the raw sums, or from the state before the block. The
//      sum is `w0 p3 + w1 p2 + w2 p1 + w3 o`, as in `ops::short_conv`. The result goes to `c` (`r` with FLAG_ACCUMULATE:
//      `x + conv`). The state after the block is the last 3 raw rows. The state rows of the tile are not read again.
//   3. The item arrives at the counter of the 144 tiles. With no finisher items (`items` = 576), the item that arrives last
//      runs the finisher for every row, in order. With finisher items (below), it only arrives.
// The finisher items. A task can have `ceil(rows / MK_OUT_MIX_FINISH_ROWS)` items after the 576 of the matmul. Item
// `576 + j` runs the finisher for the rows `j * F` to `min((j + 1) * F, rows) - 1` (`F` = MK_OUT_MIX_FINISH_ROWS), so
// all the blocks share the rows. A finisher item polls the counter of the tiles until it reads 144 (an acquire load, the
// trap at MK_SPIN_LIMIT as for an event). It cannot deadlock: the interpreter gives out the items in order (item b to
// block b, then the claims in rising order), so every item of the matmul is held by a block that runs it without a wait
// before any block holds a finisher item. The finisher items arrive at a second counter (word 145), and the one that
// arrives last sets both counters back to 0: every other finisher item has read the counter of the tiles by then.
// The rows are independent, so each row has its sums in its order, whichever block runs it.
// The finisher runs for each row, in the 256 workers of one block:
//   a. The row factor of `r` (`1 / sqrt(mean(r^2) + eps)`), written to the table slot of the factors of `r`.
//   b. `rw_mix_item` of lane R (read only): the softmax mix of the stack, `e`, and the row factor of `e`.
//   c. With FLAG_NORM, `h_e = w * (e * rf)`, the order of `ops::rms_norm`.
// The quantization of `h_e` for the experts is not here. It belongs to the prologue of ROUTE, or to a quantizer task
// after this op (`OP_QUANT_NVFP4`): a row of `h_e` is complete only after the finisher, so no tile can quantize it.
//
// The descriptor: a = the attention output (f32, `[rows][3072]`), b = the repacked `wo`, c = `r` (f32, `[rows][2304]`),
// d = the task table, e = `x` (f32, `[rows][2304]`, with FLAG_ACCUMULATE, else 0), f = the scratch, n = 2304, k = 3072,
// kt = 768, group_items = 4, items = 576, item_bytes = 16 * kt * 2, rows = 1 to 512, layer = the depth of the stack
// (the rows of the blocks, then `r`: 2 to 5), p0 = the scale of a score, 2304^-0.5, scale_bytes = the floats of the
// scratch that the host gives (checked), flags = FLAG_ACCUMULATE and FLAG_NORM, in any mix.
// The split form (FLAG_BLOCK_FORM, a block of 28 rows or more): an earlier block-form OP_MATMUL_BF16 task wrote the raw
// sums `[row][2304]` to the scratch, at `f + 4 * (160 + 576 * rows * 16)` bytes. This task then has no matmul: a = that
// address (checked), b = 0, item_bytes = 0, items = 144 + ceil(rows / MK_OUT_MIX_FINISH_ROWS). Item `j` below 144
// does the convolution, the residual and the state of the tile `j`, and the finisher items follow as above. The
// task has no weights to prefetch, and the other fields are those of the fused form.
// The scratch, in floats: 144 counters of the tiles, the counter of the tiles that are done (word 144, used with
// finisher items) and the counter of the finisher items that are done (word 145), padding to 160, the partials
// (`576 * rows * 16`), the raw sums (`2304 * rows`). The counters start at 0, and the merger sets each back to 0 (with
// finisher items, the last finisher item sets word 144 back). A launch that failed leaves the counters dirty: the
// ring zeroes only the scoreboard, so the host must zero the scratch again (`Ring::zero_device`). A stale word 144
// would let a finisher item read rows before the tiles wrote them, with no error. In the profile build the words 146
// to 148 hold the spin of the finisher items (sum of ns, count, longest) and are never set back.
//
// The table (`OUT_MIX_TABLE_*` in `src/protocol.rs`): slots 0 to 4 are the stack as `OP_MIX` has it: the planes B[0..],
// then `r` at the slot `layer - 1`, and slots 5 to 9 are their row factors. The slot of the factors of `r` points to
// `rows` floats that this op writes. Slot 10 is the convolution weight (bf16, `[2304][4]`), 11 the conv state (f32,
// `[3][2304]`, oldest row first), 12 the norm weight of `h_e` (f32, 2304, with FLAG_NORM), 13 the outputs (f32): `e`
// `[rows][2304]`, then `h_e` `[rows][2304]`, then the row factors of `e` `[rows]`, 14 the query (f32, 2304), and slot 15
// must be 0.

#pragma once

constexpr uint32_t OM_TILES = 144;
constexpr uint32_t OM_WIDTH = 2304;
constexpr uint32_t OM_K = 3072;
constexpr uint32_t OM_COUNTERS = 160;
// The items of the matmul: 144 tiles of 4 slices.
constexpr uint32_t OM_TILE_ITEMS = OM_TILES * 4;
// The counter of the finisher items that are done: the word after the counter of the tiles.
constexpr uint32_t OM_FINISHED = OM_TILES + 1;
// The slots of the task table after the stack.
constexpr uint32_t OM_RF_SLOT = 5;
// The slot that must be 0: the matmul has no norm.
constexpr uint32_t OM_NO_NORM = 15;

// The scratch floats that a task of `rows` rows needs (the number that `OutMixLayout::scratch_floats` gives).
__device__ __forceinline__ uint64_t om_scratch_floats(uint32_t rows) {
    return OM_COUNTERS + static_cast<uint64_t>(OM_TILE_ITEMS) * rows * MM_TILE + static_cast<uint64_t>(OM_WIDTH) * rows;
}

__device__ __forceinline__ float om_bf16(const uint16_t* p, uint32_t i) {
    return __uint_as_float(static_cast<uint32_t>(__ldg(p + i)) << 16);
}

// The finisher of one row: the factor of `r`, the mix, the factor of `e`, and `h_e` with the flag.
__device__ __forceinline__ void om_finish_row(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t row) {
    const uint32_t depth = task.layer;
    const float* r = reinterpret_cast<const float*>(task.c) + static_cast<uint64_t>(row) * OM_WIDTH;
    float* scratch = reinterpret_cast<float*>(mk_smem);
    float sums[1] = {0.0f};
#pragma unroll
    for (uint32_t j = 0; j < RW_PER_THREAD; ++j) {
        const float v = __ldcg(r + worker_id() + j * MK_WORKERS);
        sums[0] += v * v;
    }
    rw_block_sum(sums, scratch);
    if (worker_id() == 0) {
        __stcg(reinterpret_cast<float*>(table->ptr[OM_RF_SLOT + depth - 1]) + row, rw_factor(sums[0], OM_WIDTH));
    }
    worker_sync();
    const uint64_t rows = task.rows;
    float* out = reinterpret_cast<float*>(table->ptr[MK_OUT_MIX_TABLE_OUT]);
    float* e = out + static_cast<uint64_t>(row) * OM_WIDTH;
    float* e_rf = out + 2 * rows * OM_WIDTH + row;
    rw_mix_item(table, 0, OM_RF_SLOT, depth, reinterpret_cast<const float*>(table->ptr[MK_OUT_MIX_TABLE_QUERY]), row,
                task.p0, e, e_rf);
    if ((task.flags & FLAG_NORM) != 0) {
        worker_sync();
        const float factor = __ldcg(e_rf);
        const float* w = reinterpret_cast<const float*>(table->ptr[MK_OUT_MIX_TABLE_NORM]);
        float* h = out + (rows + row) * OM_WIDTH;
#pragma unroll
        for (uint32_t j = 0; j < RW_PER_THREAD; ++j) {
            const uint32_t i = worker_id() + j * MK_WORKERS;
            h[i] = w[i] * (e[i] * factor);
        }
    }
    worker_sync();
}

// The raw sums of the task: after the counters and the 576 partials of its rows (the place that the fused op uses). In
// the split form an earlier block-form matmul writes them here, `[row][2304]`.
__device__ __forceinline__ uint64_t om_raw_address(const MkTaskDescriptor& task) {
    return task.f + (OM_COUNTERS + static_cast<uint64_t>(OM_TILE_ITEMS) * task.rows * MM_TILE) * 4;
}

__device__ __forceinline__ void om_check(const MkTaskDescriptor& task, const MkTaskTable* table) {
    const bool accumulate = (task.flags & FLAG_ACCUMULATE) != 0;
    const bool norm = (task.flags & FLAG_NORM) != 0;
    const bool split = (task.flags & FLAG_BLOCK_FORM) != 0;
    const uint32_t rows = task.rows;
    const uint32_t finisher_items = (rows + MK_OUT_MIX_FINISH_ROWS - 1) / MK_OUT_MIX_FINISH_ROWS;
    if (worker_count() != MK_WORKERS || table == nullptr || rows == 0 || rows > MM_MAX_ROWS || task.n != OM_WIDTH ||
        task.k != OM_K || task.kt != 768 || task.group_items != 4 ||
        !(split ? task.items == OM_TILES + finisher_items && finisher_items != 0
                : (task.items == OM_TILE_ITEMS || task.items == OM_TILE_ITEMS + finisher_items)) ||
        task.item_bytes != (split ? 0 : MM_TILE * 768 * 2) || dynamic_smem_bytes() < MM_SMEM_BYTES || task.layer < 2 ||
        task.layer > RW_MAX_STACK || (task.flags & ~(FLAG_ACCUMULATE | FLAG_NORM | FLAG_BLOCK_FORM)) != 0 ||
        (task.a & 7) != 0 || (split ? task.b != 0 || task.a != om_raw_address(task) : (task.b & 15) != 0) || (task.c & 15) != 0 || (task.f & 15) != 0 || task.a == 0 || task.c == 0 || task.f == 0 ||
        accumulate != (task.e != 0) || (task.e & 15) != 0 || !(task.p0 > 0.0f) ||
        om_scratch_floats(rows) > task.scale_bytes) {
        __trap();
    }
    // The stack: every plane and its factors, and `r` in the last slot, which this task writes.
    if (table->ptr[task.layer - 1] != task.c || table->ptr[OM_RF_SLOT + task.layer - 1] == 0 ||
        (table->ptr[OM_RF_SLOT + task.layer - 1] & 3) != 0) {
        __trap();
    }
    for (uint32_t s = 0; s + 1 < task.layer; ++s) {
        if (table->ptr[s] == 0 || (table->ptr[s] & 15) != 0 || table->ptr[OM_RF_SLOT + s] == 0 ||
            (table->ptr[OM_RF_SLOT + s] & 3) != 0) {
            __trap();
        }
    }
    const uint64_t conv_w = table->ptr[MK_OUT_MIX_TABLE_CONV];
    const uint64_t state = table->ptr[MK_OUT_MIX_TABLE_STATE];
    const uint64_t out = table->ptr[MK_OUT_MIX_TABLE_OUT];
    const uint64_t query = table->ptr[MK_OUT_MIX_TABLE_QUERY];
    const uint64_t weight = table->ptr[MK_OUT_MIX_TABLE_NORM];
    if (conv_w == 0 || (conv_w & 1) != 0 || state == 0 || (state & 15) != 0 || out == 0 || (out & 15) != 0 ||
        query == 0 || (query & 15) != 0 || norm != (weight != 0) || (weight & 15) != 0) {
        __trap();
    }
}

// One item of the matmul and of the steps 1 to 3 for its tile. It returns true in every worker of the item that must run
// the finisher for all rows: the item that arrives last over the tiles, when the task has no finisher items.
// The profile build times the phases of a tile item (the test `the_tile_phase_split` reads them). The stamps are in shared
// memory so that they take no register. Worker 0 adds the ns of each phase to scratch words that nothing sets back:
// 150 the matmul partials of an item, 151 the count of items, 152 the arrival and the merge of the 4 slices, 153 the
// convolution and the residual, 154 the state (the words 152 to 154 are for the 144 items that merge), and the words
// 156/157 and 158/159 hold, as u64, the complement of the first start and the last end of a tile item.
#ifdef MK_PROFILE
__shared__ uint64_t om_stamp[5];
#define OM_STAMP(i)                         \
    if (worker_id() == 0) {                 \
        om_stamp[i] = globaltimer_ns();     \
    }
__device__ __forceinline__ void om_prof_add(uint32_t* counters, uint32_t word, uint32_t from, uint32_t to) {
    if (worker_id() == 0) {
        atomicAdd(counters + word, static_cast<uint32_t>(om_stamp[to] - om_stamp[from]));
    }
}
__device__ __forceinline__ void om_prof_edges(uint32_t* counters, uint32_t last) {
    if (worker_id() == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(counters + 156), ~static_cast<unsigned long long>(om_stamp[0]));
        atomicMax(reinterpret_cast<unsigned long long*>(counters + 158), static_cast<unsigned long long>(om_stamp[last]));
    }
}
#else
#define OM_STAMP(i)
#endif

__device__ __forceinline__ bool om_tile_item(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {

    // The matmul has no norm: the slot 15 is 0. The code of `mm_item_partials` for a null norm that the compiler can see holds
    // more registers than the kernel has (28 bytes of spill). A null that it reads from the table gives the code that the
    // other matmul ops use, which fits. The check reads the value through `asm`, so that it does not teach the compiler that
    // the value is 0.
    const uint64_t no_norm = table->ptr[OM_NO_NORM];
    uint64_t seen = no_norm;
    asm volatile("" : "+l"(seen));
    if (seen != 0) {
        __trap();
    }
    const bool split = (task.flags & FLAG_BLOCK_FORM) != 0;
    OM_STAMP(0);
    if (!split) {
        mm_item_partials(task, item, reinterpret_cast<const float*>(task.a), reinterpret_cast<const float*>(no_norm),
                         reinterpret_cast<const uint16_t*>(no_norm), reinterpret_cast<float*>(task.f) + OM_COUNTERS,
                         OM_TILE_ITEMS);
    }
    OM_STAMP(1);
    const uint32_t rows = task.rows;
    const uint32_t tile = split ? item : item / 4;
    const uint64_t partial_floats = static_cast<uint64_t>(rows) * MM_TILE;
    uint32_t* counters = reinterpret_cast<uint32_t*>(task.f);
    float* partials = reinterpret_cast<float*>(task.f) + OM_COUNTERS;
    float* raw = partials + static_cast<uint64_t>(OM_TILE_ITEMS) * partial_floats;
    uint32_t* flag = reinterpret_cast<uint32_t*>(mk_smem + MM_TREE_FLOATS * 4);
    if (!split && worker_id() == 0) {
        *flag = merge_arrive(counters + tile, 4) ? 1u : 0u;
    }
    if (!split) {
        worker_sync();
    }
    if (!split && *flag == 0) {
#ifdef MK_PROFILE
        om_prof_add(counters, 150, 0, 1);
        if (worker_id() == 0) {
            atomicAdd(counters + 151, 1u);
        }
        OM_STAMP(2);
        om_prof_edges(counters, 2);
#endif
        return false;
    }

    // 1. The merge of the 4 slices: the raw sums of the tile, `[row][16]`. The split form has the raw sums of all tiles
    // from the matmul task, `[row][2304]`, and has no merge. `rb` is the first value of the tile and `stride` the floats
    // from a row to the next; the order of the values is the same in both forms.
    float* tile_raw = raw + static_cast<uint64_t>(tile) * partial_floats;
    if (!split) {
        for (uint64_t i = worker_id(); i < partial_floats; i += MK_WORKERS) {
            __stcg(tile_raw + i, mm_merged_sum(partials, tile, 4, partial_floats, i));
        }
        worker_sync();
    }
    const float* rb = split ? raw + static_cast<uint64_t>(tile) * MM_TILE : tile_raw;
    const uint64_t stride = split ? OM_WIDTH : MM_TILE;
    OM_STAMP(2);

    // 2. The convolution, the residual and the conv state.
    const uint16_t* conv_w = reinterpret_cast<const uint16_t*>(table->ptr[MK_OUT_MIX_TABLE_CONV]);
    float* state = reinterpret_cast<float*>(table->ptr[MK_OUT_MIX_TABLE_STATE]);
    const float* x_in = reinterpret_cast<const float*>(task.e);
    float* r_out = reinterpret_cast<float*>(task.c);
    const bool accumulate = (task.flags & FLAG_ACCUMULATE) != 0;
    for (uint64_t i = worker_id(); i < partial_floats; i += MK_WORKERS) {
        const uint32_t t = static_cast<uint32_t>(i / MM_TILE);
        const uint32_t c = static_cast<uint32_t>(i % MM_TILE);
        const uint32_t channel = tile * MM_TILE + c;
        const float x = __ldcg(rb + t * stride + c);
        // The rows at the lags 1, 2 and 3: a row of the block, or a row of the state.
        float p[3];
#pragma unroll
        for (uint32_t lag = 1; lag <= 3; ++lag) {
            p[lag - 1] = t >= lag ? __ldcg(rb + (t - lag) * stride + c)
                                  : __ldcg(state + static_cast<uint64_t>(3 + t - lag) * OM_WIDTH + channel);
        }
        float w[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            w[j] = om_bf16(conv_w, channel * 4 + j);
        }
        const float sum = w[0] * p[2] + w[1] * p[1] + w[2] * p[0] + w[3] * x;
        float value = (sum + x) / __uint_as_float(0x400f1bbdu);
        if (accumulate) {
            value = value + __ldcg(x_in + static_cast<uint64_t>(t) * OM_WIDTH + channel);
        }
        __stcg(r_out + static_cast<uint64_t>(t) * OM_WIDTH + channel, value);
    }
    worker_sync();
    OM_STAMP(3);
    // The state after the block: the last 3 of the three old rows and the raw rows. Only this item reads these channels.
    // A worker holds its new value in a register until every worker has read, because the new rows replace the old rows.
    {
        const uint32_t i = worker_id();
        const bool mine = i < 3 * MM_TILE;
        const uint32_t j = i / MM_TILE;
        const uint32_t c = i % MM_TILE;
        const uint32_t index = rows + j;
        float keep = 0.0f;
        if (mine) {
            keep = index < 3 ? __ldcg(state + index * OM_WIDTH + tile * MM_TILE + c)
                             : __ldcg(rb + static_cast<uint64_t>(index - 3) * stride + c);
        }
        worker_sync();
        if (mine) {
            __stcg(state + j * OM_WIDTH + tile * MM_TILE + c, keep);
        }
    }

    OM_STAMP(4);
#ifdef MK_PROFILE
    om_prof_add(counters, 150, 0, 1);
    if (worker_id() == 0) {
        atomicAdd(counters + 151, 1u);
    }
    om_prof_add(counters, 152, 1, 2);
    om_prof_add(counters, 153, 2, 3);
    om_prof_add(counters, 154, 3, 4);
    om_prof_edges(counters, 4);
#endif
    // 3. The arrival at the counter of the tiles.
    if (task.items != OM_TILE_ITEMS) {
        // The finisher items follow (the split form has no other form). The barrier orders the stores of all workers (`r` and the state) before the release add
        // of worker 0, which publishes them to the finisher items.
        worker_sync();
        if (worker_id() == 0) {
            atom_acq_rel_gpu_add_u32(counters + OM_TILES, 1u);
        }
        return false;
    }
    return rw_finisher_arrive(counters + OM_TILES, OM_TILES, flag);
}

// The items that come before the finisher items.
__device__ __forceinline__ uint32_t om_tile_items(const MkTaskDescriptor& task) {
    return (task.flags & FLAG_BLOCK_FORM) != 0 ? OM_TILES : OM_TILE_ITEMS;
}

// A finisher item: it waits for the 144 tiles, runs its rows and counts itself done.
__device__ __forceinline__ void om_finisher_item(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t j) {
    uint32_t* counters = reinterpret_cast<uint32_t*>(task.f);
    if (worker_id() == 0) {
        // The wait may depend only on items that never wait: the 576 items of the matmul. The proof that it cannot
        // deadlock rests on this (see the header). A new wait in a matmul item, or a wait of one finisher item on
        // another, breaks it.
#ifdef MK_PROFILE
        const uint64_t t0 = globaltimer_ns();
#endif
        uint64_t polls = 0;
        while (ld_acquire_gpu_u32(counters + OM_TILES) < OM_TILES) {
            if (++polls >= MK_SPIN_LIMIT) {
                __trap();
            }
        }
#ifdef MK_PROFILE
        // The profile build adds the time of the spin: the sum of ns (word 146), the count of items (147) and the longest
        // spin (148). They are never set back, so a test reads them after the run and the zero-counter test does not use
        // the profile build.
        const uint32_t spun = static_cast<uint32_t>(globaltimer_ns() - t0);
        atomicAdd(counters + 146, spun);
        atomicAdd(counters + 147, 1u);
        atomicMax(counters + 148, spun);
#endif
    }
    worker_sync();
    const uint32_t first = j * MK_OUT_MIX_FINISH_ROWS;
    const uint32_t end = first + MK_OUT_MIX_FINISH_ROWS < task.rows ? first + MK_OUT_MIX_FINISH_ROWS : task.rows;
    for (uint32_t row = first; row < end; ++row) {
        om_finish_row(task, table, row);
    }
    // The last finisher item sets the counter of the tiles back to 0. `merge_arrive` sets its own counter back. The
    // store is a plain relaxed store, not a `merge_arrive`, and it is safe by the rule of `merge_arrive`: every other
    // finisher item has read the counter already (it arrived after its wait), and this thread signals the task after
    // the store (`run_items`), so the next task that reuses the counter waits on a signal that follows the reset.
    if (worker_id() == 0 && merge_arrive(counters + OM_FINISHED, task.items - om_tile_items(task))) {
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(counters + OM_TILES), "r"(0u) : "memory");
    }
}

__device__ __forceinline__ void op_out_mix(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    om_check(task, table);
    if (item >= om_tile_items(task)) {
        om_finisher_item(task, table, item - om_tile_items(task));
        return;
    }
    if (!om_tile_item(task, table, item)) {
        return;
    }
    for (uint32_t row = 0; row < task.rows; ++row) {
        om_finish_row(task, table, row);
    }
}
