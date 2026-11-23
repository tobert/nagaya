// The MoE ops of one Inkling 2 run that experts.cuh and quant.cuh do not hold: OP_ROUTE.
//
// This file belongs to wave B lane M (MoE).
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor`, `MkTaskTable`, experts.cuh (for `ExpertPair`) and the worker
// functions of megakernel.cu.
//
// OP_ROUTE: the router matmul, the sigmoid, the top 8 and the list of pairs of the expert ops. The router is
// `gate.weight` (128 experts by 2,304 values, `bf16`) and `gate.bias` (128 `f32`). The op holds its own copy of the
// `bf16` MMA step, so a change of the dense lane does not move its bits.
//
// The descriptor:
//   a      the rows `x`: `rows` rows of 2,304 `f32`. With FLAG_NORM they are the rows before the norm: the op reads
//          `w * (v * rf)`, where `rf` is the row factor that an earlier task wrote (the order of `ops::rms_norm`).
//   b      the router, repacked as a dense matmul of N = 128 and K = 2,304 (`MatmulShape::new(128, 2304)`: 8 tiles of 16
//          columns, 3 K slices of 768, an item of 24,576 bytes).
//   c      the picks of each row: `rows * 8` `u32` expert ids in ascending order, then `rows * 8` `f32` route weights.
//   d      the task table: `ROUTE_TABLE_BIAS` the bias (128 `f32`). With FLAG_NORM also `ROUTE_TABLE_NORM_WEIGHT` the
//          norm weight (2,304 `f32`) and `ROUTE_TABLE_ROW_FACTOR` the row factors (`rows` `f32`).
//   e      the list of pairs: the header with the live count, then room for `expert_pair_bound(rows)` pairs. The op
//          writes the whole live list. The pairs go by expert, ascending. An expert with more than 16 rows has several
//          pairs, and the rows of a pair are in ascending order. `x_row` is the row, `y_row` is the slot `row * 8 + j`,
//          where `j` is the place of the expert among the 8 picks of the row, and `weight` is its route weight.
//   f      the scratch (`route_scratch_words` in src/experts.rs): the counters, which are 0 at the start and stay 0,
//          then the logits, then the partials.
//   rows   the count of rows, 1 to 1,024.   n = 128, k = 2,304, kt = 768, group_items = 3.   p0 is the route scale (8).
//   items  `ceil(rows / 16) * 24`: a row tile, a column tile and a K slice.
//   item_bytes  0, or 24,576 (the bytes of an item of the router) for `rows` up to 16. Then the producer warp prefetches
//          the item `blockIdx.x` of the router before it waits, as it does for a dense matmul. For more rows the blocks
//          from 24 on have no weight at that place, so the item bytes must be 0. The op prefetches the next item itself.
//
// The three levels. 1: the 3 K slices of a (row tile, column tile) merge in slice order and write the logits. 2: the
// last of the 8 column tiles of a row tile takes the picks of its rows: the sigmoid, the key `score + bias`, the 8
// largest keys (a tie takes the lower id), the ids in ascending order, and the weights (`ops::route`). 3: the last row
// tile builds the list of pairs. Each level resets its counter (`merge_arrive`). No bit depends on the grid or on the
// count of rows: a row is a column of the MMA, and its sums have one order.
//
// The sigmoid is `1 / (1 + expf(-x))` of the device. The host uses the `expf` of its libm, which can differ by an
// ulp. A pick can differ only where two keys are closer than that error.

#pragma once

constexpr uint32_t ROUTE_TABLE_BIAS = 0;
constexpr uint32_t ROUTE_TABLE_NORM_WEIGHT = 1;
constexpr uint32_t ROUTE_TABLE_ROW_FACTOR = 2;

constexpr uint32_t RT_EXPERTS = 128;
constexpr uint32_t RT_K = 2304;
constexpr uint32_t RT_KT = 768;
constexpr uint32_t RT_SLICES = 3;
constexpr uint32_t RT_COL_TILES = 8;
constexpr uint32_t RT_ITEMS_PER_ROW_TILE = RT_COL_TILES * RT_SLICES;
constexpr uint32_t RT_ROWS = 16;
constexpr uint32_t RT_TOP = 8;
constexpr uint32_t RT_MAX_TOKENS = 1024;
constexpr uint32_t RT_WARPS = 8;
constexpr uint32_t RT_STEPS_PER_WARP = RT_KT / (RT_WARPS * 16);
constexpr uint32_t RT_ITEM_BYTES = 16 * RT_KT * 2;
constexpr uint32_t RT_PARTIAL_FLOATS = RT_ROWS * 16;

// The shared memory: 4 flag words, then one region that the tree of the MMAs, the picks and the plan use one after the
// other. The plan has the bits of the picks (4 words for each row), the counts, the first pair and the warp sums.
constexpr uint32_t RT_OFF_FLAGS = 0;
constexpr uint32_t RT_OFF_REGION = 256;
constexpr uint32_t RT_OFF_TREE = RT_OFF_REGION;
constexpr uint32_t RT_OFF_BITS = RT_OFF_REGION;
constexpr uint32_t RT_OFF_COUNTS = RT_OFF_BITS + RT_MAX_TOKENS * 16;
constexpr uint32_t RT_OFF_WSUM = RT_OFF_COUNTS + RT_EXPERTS * 4;
constexpr uint32_t RT_SMEM_BYTES = RT_OFF_WSUM + RT_WARPS * 4;
static_assert(RT_OFF_TREE + RT_WARPS * 2 * 32 * 4 * 4 <= RT_SMEM_BYTES, "the tree of the MMAs must fit the region");

// The words of the scratch, as `route_scratch_words` in src/experts.rs has them. The counters come first: a counter for
// each (row tile, column tile), a counter for each row tile, and one for the plan, rounded to 4 words. Their room is for
// the most rows, so the logits start at a place that does not depend on the rows (`ROUTE_COUNTER_WORDS`).
constexpr uint32_t RT_COUNTER_WORDS = ((RT_MAX_TOKENS / 16) * RT_COL_TILES + RT_MAX_TOKENS / 16 + 1 + 3) & ~3u;

// The same bound as `expert_pair_bound` in src/protocol.rs.
__device__ __forceinline__ uint32_t rt_pair_bound(uint32_t rows) {
    const uint32_t all = 8 * rows;
    const uint32_t packed = 8 * rows / MK_EXPERT_MAX_ROWS + RT_EXPERTS;
    return all < packed ? all : packed;
}

// Two `f32` values to two `bf16` values in one register, rounded to nearest even. `e0` goes to the low half.
__device__ __forceinline__ uint32_t rt_pack_bf16(float e0, float e1) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(e1), "f"(e0));
    return packed;
}

__device__ __forceinline__ float rt_lo_half(uint32_t packed) { return __uint_as_float(packed << 16); }
__device__ __forceinline__ float rt_hi_half(uint32_t packed) { return __uint_as_float(packed & 0xFFFF0000u); }

// Design rule 9: x = hi + mid + lo in three `bf16` terms, and each subtraction is exact in `f32`.
struct RtTerms {
    uint32_t hi, mid, lo;
};

__device__ __forceinline__ RtTerms rt_split(float e0, float e1) {
    RtTerms t;
    t.hi = rt_pack_bf16(e0, e1);
    const float r0 = e0 - rt_lo_half(t.hi);
    const float r1 = e1 - rt_hi_half(t.hi);
    t.mid = rt_pack_bf16(r0, r1);
    t.lo = rt_pack_bf16(r0 - rt_lo_half(t.mid), r1 - rt_hi_half(t.mid));
    return t;
}

__device__ __forceinline__ void rt_mma(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// The k16 step of one n8 tile: lo, then mid, then hi, so that the small terms enter first.
__device__ __forceinline__ void rt_step(float (&acc)[4], const uint4& a, float2 first, float2 second) {
    const RtTerms f = rt_split(first.x, first.y);
    const RtTerms s = rt_split(second.x, second.y);
    rt_mma(acc, a, f.lo, s.lo);
    rt_mma(acc, a, f.mid, s.mid);
    rt_mma(acc, a, f.hi, s.hi);
}

// Two values of a row of `x`, with the norm of the row when `norm` is set. The row came from an earlier task.
__device__ __forceinline__ float2 rt_load_x(const float* row, const float* weight, float factor, bool valid, bool norm) {
    if (!valid) {
        return make_float2(0.0f, 0.0f);
    }
    float2 v = __ldcg(reinterpret_cast<const float2*>(row));
    if (norm) {
        const float2 w = *reinterpret_cast<const float2*>(weight);
        v.x = w.x * (v.x * factor);
        v.y = w.y * (v.y * factor);
    }
    return v;
}

// The sigmoid and the division of this file use the fast division (error of 2 ulps, no call to the slow path). A division
// by `/` can call that path, a call makes the compiler save the live registers of the interpreter, and the build fails on
// the spill that this makes. The device `expf` differs from the host `expf` by an ulp too, so the picks and the weights are
// not a bit gate against the host.
__device__ __forceinline__ float rt_sigmoid(float x) { return __fdividef(1.0f, 1.0f + expf(-x)); }

// `a` beats `b` when its key is larger, or when the keys are equal and its id is lower.
__device__ __forceinline__ bool rt_better(float key_a, uint32_t id_a, float key_b, uint32_t id_b) {
    return key_a > key_b || (key_a == key_b && id_a < id_b);
}

// The picks of one row, in one warp. The warp holds the 128 logits of the row, 4 for each lane. It takes the best key 8
// times (`ops::route` sorts by the key and then by the id), sorts the 8 ids, and gives the lane `i` the weight of the
// pick `i`. The lanes 0 to 7 write the ids and the weights. `sorted` has 8 words of shared memory for this warp.
__device__ __forceinline__ void route_row(
    const float* logits, const float* bias, float scale, uint32_t* sel_expert, float* sel_weight, uint32_t* sorted) {
    const uint32_t lane = worker_id() & 31;
    float key[4];
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
        const uint32_t e = lane + 32 * j;
        key[j] = rt_sigmoid(__ldcg(logits + e)) + bias[e];
        if (!(fabsf(key[j]) <= FLT_MAX)) {
            __trap();  // A logit or a bias that is not finite gives no pick.
        }
    }
    uint32_t taken = 0;  // bit j: this lane has taken the key j
    uint32_t mine = 0;
#pragma unroll
    for (uint32_t pick = 0; pick < RT_TOP; ++pick) {
        float best_key = -FLT_MAX;
        uint32_t best_id = 0xFFFFFFFFu;
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            if (((taken >> j) & 1u) == 0 && rt_better(key[j], lane + 32 * j, best_key, best_id)) {
                best_key = key[j];
                best_id = lane + 32 * j;
            }
        }
#pragma unroll
        for (uint32_t offset = 16; offset > 0; offset >>= 1) {
            const float other_key = __shfl_xor_sync(0xffffffffu, best_key, offset);
            const uint32_t other_id = __shfl_xor_sync(0xffffffffu, best_id, offset);
            if (rt_better(other_key, other_id, best_key, best_id)) {
                best_key = other_key;
                best_id = other_id;
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            if ((best_id & 31) == lane && (best_id >> 5) == j) {
                taken |= 1u << j;
            }
        }
        if (lane == pick) {
            mine = best_id;
        }
    }
    // The rank of each pick among the 8: the lanes 0 to 7 hold them.
    uint32_t rank = 0;
#pragma unroll
    for (uint32_t i = 0; i < RT_TOP; ++i) {
        rank += __shfl_sync(0xffffffffu, mine, i) < mine ? 1u : 0u;
    }
    if (lane < RT_TOP) {
        sorted[rank] = mine;
    }
    __syncwarp();
    const uint32_t id = lane < RT_TOP ? sorted[lane] : 0;
    const float score = rt_sigmoid(__ldcg(logits + id));
    // The sum is in the order of the ids, then `+ 1e-20`, as the host has it.
    float total = __shfl_sync(0xffffffffu, score, 0);
#pragma unroll
    for (uint32_t i = 1; i < RT_TOP; ++i) {
        total += __shfl_sync(0xffffffffu, score, i);
    }
    total += 1e-20f;
    if (lane < RT_TOP) {
        sel_expert[lane] = id;
        sel_weight[lane] = __fdividef(score, total) * scale;
    }
    __syncwarp();
}

// The list of pairs from the picks of all rows. It runs in the last item of the task: all 256 consumers. Thread `e` of
// the first 128 takes the expert `e`: its rows in ascending order, 16 rows for each pair.
__device__ __forceinline__ void route_plan(const MkTaskDescriptor& task, const uint32_t* sel_expert, const float* sel_weight) {
    const uint32_t rows = task.rows;
    uint32_t* bits = reinterpret_cast<uint32_t*>(mk_smem + RT_OFF_BITS);
    uint32_t* warp_sum = reinterpret_cast<uint32_t*>(mk_smem + RT_OFF_WSUM);
    const uint32_t tid = worker_id();
    for (uint32_t i = tid; i < rows * 4; i += MK_CONSUMER_THREADS) {
        bits[i] = 0;
    }
    worker_sync();
    for (uint32_t i = tid; i < rows * RT_TOP; i += MK_CONSUMER_THREADS) {
        const uint32_t e = __ldcg(sel_expert + i);
        if (e >= RT_EXPERTS) {
            __trap();
        }
        atomicOr(&bits[(i / RT_TOP) * 4 + (e >> 5)], 1u << (e & 31));
    }
    worker_sync();
    const bool owner = tid < RT_EXPERTS;
    uint32_t count = 0;
    if (owner) {
        for (uint32_t r = 0; r < rows; ++r) {
            count += (bits[r * 4 + (tid >> 5)] >> (tid & 31)) & 1u;
        }
    }
    const uint32_t pairs = (count + MK_EXPERT_MAX_ROWS - 1) / MK_EXPERT_MAX_ROWS;
    // The scan of the counts of pairs: inside a warp, then over the warps.
    uint32_t inclusive = pairs;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t up = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if ((tid & 31) >= offset) {
            inclusive += up;
        }
    }
    if ((tid & 31) == 31) {
        warp_sum[tid >> 5] = inclusive;
    }
    worker_sync();
    uint32_t first = inclusive - pairs;
    for (uint32_t w = 0; w < (tid >> 5); ++w) {
        first += warp_sum[w];
    }
    uint32_t live = 0;
    for (uint32_t w = 0; w < RT_WARPS; ++w) {
        live += warp_sum[w];
    }
    if (live > rt_pair_bound(rows)) {
        __trap();  // The bound of the list follows from the counts, so this is a defect.
    }
    ExpertPair* list = reinterpret_cast<ExpertPair*>(task.e + MK_EXPERT_PAIR_HEADER_BYTES);
    if (tid == 0) {
        reinterpret_cast<ExpertPairHeader*>(task.e)->pairs = live;
    }
    if (owner) {
        uint32_t fill = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t* mask = bits + r * 4;
            if (((mask[tid >> 5] >> (tid & 31)) & 1u) == 0) {
                continue;
            }
            uint32_t j = __popc(mask[tid >> 5] & ((1u << (tid & 31)) - 1u));
            for (uint32_t w = 0; w < (tid >> 5); ++w) {
                j += __popc(mask[w]);
            }
            ExpertPair* pair = list + (first + fill / MK_EXPERT_MAX_ROWS);
            const uint32_t at = fill % MK_EXPERT_MAX_ROWS;
            pair->x_row[at] = static_cast<uint16_t>(r);
            pair->y_row[at] = static_cast<uint16_t>(r * RT_TOP + j);
            pair->weight[at] = __ldcg(sel_weight + r * RT_TOP + j);
            ++fill;
        }
        for (uint32_t p = 0; p < pairs; ++p) {
            ExpertPair* pair = list + (first + p);
            pair->expert = tid;
            const uint32_t left = count - p * MK_EXPERT_MAX_ROWS;
            pair->count = left < MK_EXPERT_MAX_ROWS ? left : MK_EXPERT_MAX_ROWS;
        }
    }
}

__device__ __forceinline__ void op_route(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    const uint32_t rows = task.rows;
    const uint32_t row_tiles = (rows + RT_ROWS - 1) / RT_ROWS;
    const bool norm = (task.flags & FLAG_NORM) != 0;
    if (worker_count() != MK_CONSUMER_THREADS || (task.flags & ~(FLAG_NORM | FLAG_PAIRS)) != 0 || rows == 0 ||
        rows > RT_MAX_TOKENS || task.n != RT_EXPERTS || task.k != RT_K || task.kt != RT_KT ||
        task.group_items != RT_SLICES || task.items != row_tiles * RT_ITEMS_PER_ROW_TILE ||
        (task.item_bytes != 0 && (task.item_bytes != RT_ITEM_BYTES || rows > RT_ROWS)) ||
        table == nullptr || dynamic_smem_bytes() < RT_SMEM_BYTES || (task.a & 7) != 0 || (task.b & 15) != 0 ||
        (task.c & 15) != 0 || (task.e & 15) != 0 || (task.f & 15) != 0 || table->ptr[ROUTE_TABLE_BIAS] == 0 ||
        (norm && (table->ptr[ROUTE_TABLE_NORM_WEIGHT] == 0 || table->ptr[ROUTE_TABLE_ROW_FACTOR] == 0))) {
        __trap();  // A shape or a flag that the op does not cover.
    }
    float* tree = reinterpret_cast<float*>(mk_smem + RT_OFF_TREE);
    uint32_t* flags = reinterpret_cast<uint32_t*>(mk_smem + RT_OFF_FLAGS);

    const uint32_t row_tile = item / RT_ITEMS_PER_ROW_TILE;
    const uint32_t in_tile = item % RT_ITEMS_PER_ROW_TILE;
    const uint32_t col_tile = in_tile / RT_SLICES;
    const uint32_t slice = in_tile % RT_SLICES;
    const uint32_t first_row = row_tile * RT_ROWS;
    const uint32_t live_rows = rows - first_row < RT_ROWS ? rows - first_row : RT_ROWS;
    const uint32_t warp = worker_id() >> 5;
    const uint32_t lane = worker_id() & 31;
    const uint32_t g = lane >> 2;
    const uint32_t t = lane & 3;
    const bool two_tiles = live_rows > 8;
    const bool valid0 = g < live_rows;
    const bool valid1 = g + 8 < live_rows;

    const uint8_t* weights = reinterpret_cast<const uint8_t*>(task.b);
    const uint64_t next = mk_next_item();
    if (worker_id() == 0 && next < task.items) {
        bulk_prefetch_l2(weights + (next % RT_ITEMS_PER_ROW_TILE) * RT_ITEM_BYTES, RT_ITEM_BYTES);
    }

    const uint32_t first_step = warp * RT_STEPS_PER_WARP;
    const uint8_t* mine = weights + static_cast<uint64_t>(in_tile) * RT_ITEM_BYTES + (first_step * 32 + lane) * 16;
    const uint32_t k_offset = slice * RT_KT + first_step * 16 + 2 * t;
    const float* x = reinterpret_cast<const float*>(task.a);
    const float* row0 = x + static_cast<uint64_t>(first_row + g) * RT_K + k_offset;
    const float* row1 = row0 + 8 * static_cast<uint64_t>(RT_K);
    const float* norm_weight =
        norm ? reinterpret_cast<const float*>(table->ptr[ROUTE_TABLE_NORM_WEIGHT]) + k_offset : nullptr;
    const float* factors = norm ? reinterpret_cast<const float*>(table->ptr[ROUTE_TABLE_ROW_FACTOR]) + first_row : nullptr;
    const float factor0 = norm && valid0 ? __ldcg(factors + g) : 0.0f;
    const float factor1 = norm && valid1 ? __ldcg(factors + g + 8) : 0.0f;

    uint4 a[RT_STEPS_PER_WARP];
#pragma unroll
    for (uint32_t j = 0; j < RT_STEPS_PER_WARP; ++j) {
        a[j] = ld_global_nc_v4(mine + static_cast<uint64_t>(j) * 512);
    }
    float acc[2][4] = {};
#pragma unroll
    for (uint32_t j = 0; j < RT_STEPS_PER_WARP; ++j) {
        const uint32_t k0 = j * 16;
        rt_step(acc[0], a[j], rt_load_x(row0 + k0, norm_weight + k0, factor0, valid0, norm),
                rt_load_x(row0 + k0 + 8, norm_weight + k0 + 8, factor0, valid0, norm));
        if (two_tiles) {
            rt_step(acc[1], a[j], rt_load_x(row1 + k0, norm_weight + k0, factor1, valid1, norm),
                    rt_load_x(row1 + k0 + 8, norm_weight + k0 + 8, factor1, valid1, norm));
        }
    }

    // The tree of the 8 warps, as the dense matmul has it.
    reinterpret_cast<float4*>(tree)[(warp * 2) * 32 + lane] = make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);
    if (two_tiles) {
        reinterpret_cast<float4*>(tree)[(warp * 2 + 1) * 32 + lane] =
            make_float4(acc[1][0], acc[1][1], acc[1][2], acc[1][3]);
    }
    worker_sync();
    const uint32_t i = worker_id();
    const uint32_t token = (i >> 7) * 8 + 2 * ((i >> 2) & 3) + (i & 1);
    const uint32_t column = ((i >> 2) & 31) / 4 + 8 * ((i >> 1) & 1);
    uint32_t* counters = reinterpret_cast<uint32_t*>(task.f);
    float* logits = reinterpret_cast<float*>(task.f) + RT_COUNTER_WORDS;
    float* partials = logits + static_cast<uint64_t>(rows) * RT_EXPERTS;
    if (token < live_rows) {
        const float p01 = tree[i] + tree[256 + i];
        const float p23 = tree[512 + i] + tree[768 + i];
        const float p45 = tree[1024 + i] + tree[1280 + i];
        const float p67 = tree[1536 + i] + tree[1792 + i];
        __stcg(partials + static_cast<uint64_t>(item) * RT_PARTIAL_FLOATS + token * 16 + column,
               (p01 + p23) + (p45 + p67));
    }
    worker_sync();

    // Level 1: the last K slice of a (row tile, column tile) sums the slices in order and writes the logits.
    if (worker_id() == 0) {
        flags[0] = merge_arrive(counters + row_tile * RT_COL_TILES + col_tile, RT_SLICES) ? 1u : 0u;
    }
    worker_sync();
    if (flags[0] == 0) {
        return;
    }
    if (i < live_rows * 16) {
        const float* p = partials +
                         static_cast<uint64_t>(row_tile * RT_ITEMS_PER_ROW_TILE + col_tile * RT_SLICES) * RT_PARTIAL_FLOATS +
                         i;
        float sum = __ldcg(p);
        for (uint32_t s = 1; s < RT_SLICES; ++s) {
            sum += __ldcg(p + s * RT_PARTIAL_FLOATS);
        }
        __stcg(logits + static_cast<uint64_t>(first_row + i / 16) * RT_EXPERTS + col_tile * 16 + (i % 16), sum);
    }
    worker_sync();

    // Level 2: the last column tile of the row tile takes the picks of its rows, a row for each warp in turn.
    if (worker_id() == 0) {
        flags[1] = merge_arrive(counters + row_tiles * RT_COL_TILES + row_tile, RT_COL_TILES) ? 1u : 0u;
    }
    worker_sync();
    if (flags[1] == 0) {
        return;
    }
    uint32_t* sel_expert = reinterpret_cast<uint32_t*>(task.c);
    float* sel_weight = reinterpret_cast<float*>(task.c) + static_cast<uint64_t>(rows) * RT_TOP;
    const float* bias = reinterpret_cast<const float*>(table->ptr[ROUTE_TABLE_BIAS]);
    for (uint32_t r = warp; r < live_rows; r += RT_WARPS) {
        const uint32_t row = first_row + r;
        route_row(logits + static_cast<uint64_t>(row) * RT_EXPERTS, bias, task.p0, sel_expert + row * RT_TOP,
                  sel_weight + row * RT_TOP, reinterpret_cast<uint32_t*>(mk_smem + RT_OFF_REGION) + warp * RT_TOP);
    }
    worker_sync();

    // Level 3: the last row tile builds the list of pairs.
    if (worker_id() == 0) {
        flags[2] = merge_arrive(counters + row_tiles * RT_COL_TILES + row_tiles, row_tiles) ? 1u : 0u;
    }
    worker_sync();
    if (flags[2] == 0) {
        return;
    }
    route_plan(task, sel_expert, sel_weight);
}
