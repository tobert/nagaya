// OP_MATMUL_BF16: y[r, n] = sum_k x[r, k] * W[n, k] for 1 to 512 rows. x and y are f32, and W is bf16.
//
// This header needs `megakernel.cu` and `probe_ops.cuh` (the loads, the barriers and the merge counter).
//
// A work item is a tile of 16 output columns and a K slice of `kt` values. `src/repack.rs` describes the layout of W:
// the weight of one item is one span, and a lane of a warp loads one 16-byte vector for each MMA, which is its A
// fragment of `mma.sync.m16n8k16`. The tokens are operand B, in columns of 8. A pass of 9 to 16 rows runs two MMAs
// that share one A fragment. A pass of 8 rows or fewer runs one. The columns of an MMA are independent, so the bits
// of a row do not depend on the count of rows. A row that the pass does not have is zero.
//
// A pass of more than 16 rows is a loop over tiles of 16 rows inside the item. A tile runs the same steps as a pass of
// 16 rows, so a row has the same bits at each count of rows and at each place in the pass. When the K slice of a warp
// has `MM_BATCH` steps or fewer, the warp keeps its A fragments in registers and reads them once for all the tiles.
//
// With `FLAG_NORM` the op applies a norm while it loads x: the value is `w[k] * (x[r, k] * rf[r])`, the order of
// `ops::rms_norm`. `rf` is the row factor `1 / sqrt(mean(x^2) + eps)` that the task before wrote, in `e`. The weight
// `w` is bf16, and the table of the task holds its address in the slot `MK_MATMUL_TABLE_NORM_WEIGHT`.
// `FLAG_NORM_UNWEIGHTED` leaves out `w`. With `FLAG_SOFTCAP_30` the merge writes `30 * tanh(y / 30)`.
//
// Design rule 9: each f32 of x is the sum of three bf16 terms, hi + mid + lo, which hold its 24 bits of mantissa.
// Each product of a term and a weight is exact in f32. For each k16 step the MMA adds the terms in the order lo, mid,
// hi into one f32 accumulator, so the small terms enter first.
//
// The order of a sum is a function of the shape only:
// 1. A consumer warp runs its `kt / 128` consecutive k16 steps of the item, in ascending order, in one accumulator.
// 2. A fixed tree joins the 8 warps in shared memory: ((w0 + w1) + (w2 + w3)) + ((w4 + w5) + (w6 + w7)).
// 3. The item writes its partial. The item that arrives last in its group of K slices adds the partials in slice
//    order, writes y, and sets the counter of the group back to 0.
// Thus the bits depend on neither the grid, nor the block that runs an item or a merge, nor the count of rows.

#pragma once

constexpr uint32_t MM_TILE = 16;        // output columns of an item
constexpr uint32_t MM_K_STEP = 16;      // K values of one MMA
constexpr uint32_t MM_WARPS = 8;        // consumer warps: the K split of an item
constexpr uint32_t MM_BATCH = 6;        // k16 steps whose weights a warp loads before it computes
constexpr uint32_t MM_ROW_TILE = 16;
constexpr uint32_t MM_MAX_ROWS = MK_MATMUL_MAX_ROWS;
// Dynamic shared memory of the op: the partial of each warp for each of the 256 results, then 16 words of flags.
constexpr uint32_t MM_TREE_FLOATS = MM_WARPS * 2 * 32 * 4;
constexpr uint32_t MM_SMEM_BYTES = MM_TREE_FLOATS * 4 + 64;

// Two f32 values to two bf16 values in one register, rounded to nearest even. `e0` goes to the low half.
__device__ __forceinline__ uint32_t mm_pack_bf16(float e0, float e1) {
    uint32_t packed;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(e1), "f"(e0));
    return packed;
}

__device__ __forceinline__ float mm_lo_half(uint32_t packed) { return __uint_as_float(packed << 16); }
__device__ __forceinline__ float mm_hi_half(uint32_t packed) { return __uint_as_float(packed & 0xFFFF0000u); }

// The three bf16 terms of two consecutive K values of one row.
struct MmTerms {
    uint32_t hi;
    uint32_t mid;
    uint32_t lo;
};

// hi = bf16(x), mid = bf16(x - hi) and lo = bf16(x - hi - mid). Each subtraction is exact in f32.
__device__ __forceinline__ MmTerms mm_split(float e0, float e1) {
    MmTerms t;
    t.hi = mm_pack_bf16(e0, e1);
    const float r0 = e0 - mm_lo_half(t.hi);
    const float r1 = e1 - mm_hi_half(t.hi);
    t.mid = mm_pack_bf16(r0, r1);
    t.lo = mm_pack_bf16(r0 - mm_lo_half(t.mid), r1 - mm_hi_half(t.mid));
    return t;
}

__device__ __forceinline__ void mm_mma(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// The k16 step of one tile of 8 tokens: lo, then mid, then hi. `row` holds the f32 x of the token of this lane at
// the K values 2t and 2t + 1 (`first`) and 8 + 2t and 9 + 2t (`second`).
__device__ __forceinline__ void mm_step(float (&acc)[4], const uint4& a, float2 first, float2 second) {
    const MmTerms f = mm_split(first.x, first.y);
    const MmTerms s = mm_split(second.x, second.y);
    mm_mma(acc, a, f.lo, s.lo);
    mm_mma(acc, a, f.mid, s.mid);
    mm_mma(acc, a, f.hi, s.hi);
}

__device__ __forceinline__ float2 mm_load_x(const float* row, bool valid) {
    return valid ? *reinterpret_cast<const float2*>(row) : make_float2(0.0f, 0.0f);
}

__device__ __forceinline__ float mm_bf16_at(const uint16_t* w, uint32_t index) {
    return __uint_as_float(static_cast<uint32_t>(__ldg(w + index)) << 16);
}

// The two values of a row at the K values `k0` and `k0 + 1`, with the norm of the flags. `w` is null when the norm has
// no weight, and `rf` is null when there is no norm.
__device__ __forceinline__ float2 mm_load_normed(const float* row, bool valid, float rf, const uint16_t* w) {
    if (!valid) {
        return make_float2(0.0f, 0.0f);
    }
    float2 v = *reinterpret_cast<const float2*>(row);
    v.x = v.x * rf;
    v.y = v.y * rf;
    if (w != nullptr) {
        v.x = mm_bf16_at(w, 0) * v.x;
        v.y = mm_bf16_at(w, 1) * v.y;
    }
    return v;
}

// The partials of one item, for each tile of 16 rows. `x` holds `rows` rows of `task.k` floats, `factors` the row
// factors (null with no norm), `norm_w` the norm weight (null with none), and `partials` the place of the partials of the
// task: item `i` writes `rows * 16` floats at `partials + i * rows * 16`, in the order row, column of the tile. Every
// consumer calls this function. It ends with a barrier of the consumers.
__device__ __forceinline__ void mm_item_partials(const MkTaskDescriptor& task, uint32_t item, const float* x,
                                                 const float* factors, const uint16_t* norm_w, float* partials) {
    const uint32_t rows = task.rows;
    const uint32_t kt = task.kt;
    const uint32_t slices = task.group_items;
    const uint32_t steps_per_warp = kt / (MM_WARPS * MM_K_STEP);
    float* tree = reinterpret_cast<float*>(mk_smem);

    const uint32_t slice = item % slices;
    const uint32_t warp = worker_id() >> 5;
    const uint32_t lane = worker_id() & 31;
    const uint32_t g = lane >> 2;
    const uint32_t t = lane & 3;

    const uint8_t* weights = reinterpret_cast<const uint8_t*>(task.b);
    // The next item of this block: its first bytes are on the way to L2 while this item runs.
    const uint64_t next = mk_next_item();
    if (worker_id() == 0 && next < task.items) {
        bulk_prefetch_l2(weights + next * task.item_bytes, task.item_bytes);
    }

    const uint32_t first_step = warp * steps_per_warp;
    const uint8_t* mine = weights + static_cast<uint64_t>(item) * task.item_bytes + (first_step * 32 + lane) * 16;
    const uint32_t k_first = slice * kt + first_step * MM_K_STEP + 2 * t;
    const uint64_t partial_floats = static_cast<uint64_t>(rows) * MM_TILE;
    const bool norm = factors != nullptr;

    // When the steps of a warp fit in one batch, the A fragments stay in registers for all the row tiles.
    const bool resident = steps_per_warp <= MM_BATCH;
    // Zero first: a fragment that only some paths load must not look live from the item before, or the registers of all
    // the fragments stay taken through every op of the kernel.
    uint4 a[MM_BATCH] = {};
    const uint32_t row_tiles = (rows + MM_ROW_TILE - 1) / MM_ROW_TILE;
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        const uint32_t base = rt * MM_ROW_TILE;
        const uint32_t in_tile = rows - base < MM_ROW_TILE ? rows - base : MM_ROW_TILE;
        const bool two_tiles = in_tile > 8;
        const bool valid0 = g < in_tile;
        const bool valid1 = g + 8 < in_tile;
        const float* row0 = x + static_cast<uint64_t>(base + g) * task.k + k_first;
        const float* row1 = row0 + 8 * static_cast<uint64_t>(task.k);
        float rf0 = 1.0f;
        float rf1 = 1.0f;
        if (norm) {
            rf0 = valid0 ? __ldcg(factors + base + g) : 0.0f;
            rf1 = valid1 ? __ldcg(factors + base + g + 8) : 0.0f;
        }
        const uint16_t* wn = norm_w != nullptr ? norm_w + k_first : nullptr;

        float acc[2][4] = {};
        for (uint32_t s0 = 0; s0 < steps_per_warp; s0 += MM_BATCH) {
            if (!resident || rt == 0) {
#pragma unroll
                for (uint32_t j = 0; j < MM_BATCH; ++j) {
                    if (s0 + j < steps_per_warp) {
                        a[j] = ld_global_nc_v4(mine + static_cast<uint64_t>(s0 + j) * 512);
                    }
                }
            }
#pragma unroll
            for (uint32_t j = 0; j < MM_BATCH; ++j) {
                if (s0 + j < steps_per_warp) {
                    const uint32_t k0 = (s0 + j) * MM_K_STEP;
                    const uint16_t* w0 = wn != nullptr ? wn + k0 : nullptr;
                    const uint16_t* w1 = wn != nullptr ? wn + k0 + 8 : nullptr;
                    mm_step(acc[0], a[j], mm_load_normed(row0 + k0, valid0, rf0, w0),
                            mm_load_normed(row0 + k0 + 8, valid0, rf0, w1));
                    if (two_tiles) {
                        mm_step(acc[1], a[j], mm_load_normed(row1 + k0, valid1, rf1, w0),
                                mm_load_normed(row1 + k0 + 8, valid1, rf1, w1));
                    }
                }
            }
        }

        // The tree of the 8 warps. Thread i takes the result i of the tile, and it adds the 8 partials in a fixed order.
        reinterpret_cast<float4*>(tree)[(warp * 2) * 32 + lane] =
            make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);
        if (two_tiles) {
            reinterpret_cast<float4*>(tree)[(warp * 2 + 1) * 32 + lane] =
                make_float4(acc[1][0], acc[1][1], acc[1][2], acc[1][3]);
        }
        worker_sync();

        // Result i is: tile m = i / 128, lane l = (i / 4) % 32, word r = i % 4. Then the token is 8 m + 2 (l % 4) +
        // r % 2 and the column of the tile is l / 4 + 8 (r / 2). This is the layout of the result of the MMA.
        const uint32_t i = worker_id();
        const uint32_t token = (i >> 7) * 8 + 2 * ((i >> 2) & 3) + (i & 1);
        const uint32_t column = ((i >> 2) & 31) / 4 + 8 * ((i >> 1) & 1);
        if (token < in_tile) {
            const float p01 = tree[i] + tree[256 + i];
            const float p23 = tree[512 + i] + tree[768 + i];
            const float p45 = tree[1024 + i] + tree[1280 + i];
            const float p67 = tree[1536 + i] + tree[1792 + i];
            __stcg(partials + item * partial_floats + (base + token) * MM_TILE + column,
                   (p01 + p23) + (p45 + p67));
        }
        // The next tile writes the tree again.
        worker_sync();
    }
}

// The sum of the partials of the K slices of one tile, for the result `i` (row `i / 16`, column `i % 16` of the tile),
// in slice order. The partials of the tile start at `partials + tile * slices * rows * 16`.
__device__ __forceinline__ float mm_merged_sum(const float* partials, uint32_t tile, uint32_t slices,
                                               uint64_t partial_floats, uint64_t i) {
    const float* mine = partials + static_cast<uint64_t>(tile) * slices * partial_floats + i;
    float sum = __ldcg(mine);
    for (uint32_t s = 1; s < slices; ++s) {
        sum += __ldcg(mine + s * partial_floats);
    }
    return sum;
}

// a = x (const float*, `rows` rows of `k` floats), b = repacked W, c = y (float*, `rows` rows of `n` floats),
// e = the row factors (`rows` floats, with FLAG_NORM or FLAG_NORM_UNWEIGHTED),
// f = counters then partials (see MatmulShape::scratch_floats), n, k, kt, items, group_items = slices, rows,
// item_bytes = 16 * kt * 2. Every field is checked: a shape that the op does not cover traps.
__device__ __forceinline__ void op_matmul_bf16(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    const uint32_t rows = task.rows;
    const uint32_t kt = task.kt;
    const uint32_t slices = task.group_items;
    const bool norm = (task.flags & (FLAG_NORM | FLAG_NORM_UNWEIGHTED)) != 0;
    const bool weighted = (task.flags & FLAG_NORM) != 0;
    const bool softcap = (task.flags & FLAG_SOFTCAP_30) != 0;
    if (worker_count() != MK_WORKERS || rows == 0 || rows > MM_MAX_ROWS || kt == 0 ||
        kt % (MM_WARPS * MM_K_STEP) != 0 || slices == 0 || task.k != slices * kt || task.n % MM_TILE != 0 ||
        task.items != (task.n / MM_TILE) * slices || task.item_bytes != MM_TILE * kt * 2 ||
        dynamic_smem_bytes() < MM_SMEM_BYTES || (task.a & 7) != 0 || (task.b & 15) != 0 || (task.f & 3) != 0 ||
        (task.flags & ~(FLAG_NORM | FLAG_NORM_UNWEIGHTED | FLAG_SOFTCAP_30)) != 0 ||
        (task.flags & FLAG_NORM) != 0 && (task.flags & FLAG_NORM_UNWEIGHTED) != 0 ||
        norm && (task.e == 0 || (task.e & 3) != 0) || !norm && task.e != 0 ||
        weighted && (table == nullptr || table->ptr[MK_MATMUL_TABLE_NORM_WEIGHT] == 0 ||
                     (table->ptr[MK_MATMUL_TABLE_NORM_WEIGHT] & 3) != 0)) {
        __trap();
    }
    uint32_t* merge_flag = reinterpret_cast<uint32_t*>(mk_smem + MM_TREE_FLOATS * 4);
    const uint32_t tile = item / slices;
    const uint64_t partial_floats = static_cast<uint64_t>(rows) * MM_TILE;
    // The scratch holds one counter for each group, then the partials. The counters come first, so that their place
    // does not depend on the count of rows: a launch with other rows finds them at 0.
    uint32_t* counters = reinterpret_cast<uint32_t*>(task.f);
    float* partials = reinterpret_cast<float*>(task.f) + (task.n / MM_TILE);

    mm_item_partials(task, item, reinterpret_cast<const float*>(task.a), norm ? reinterpret_cast<const float*>(task.e) : nullptr,
                     weighted ? reinterpret_cast<const uint16_t*>(table->ptr[MK_MATMUL_TABLE_NORM_WEIGHT]) : nullptr,
                     partials);

    // The item that arrives last in its group merges. The release of the add publishes the partials of this block,
    // and the acquire shows the partials of the others.
    if (worker_id() == 0) {
        *merge_flag = merge_arrive(counters + tile, slices) ? 1u : 0u;
    }
    worker_sync();
    if (*merge_flag != 0) {
        for (uint64_t i = worker_id(); i < partial_floats; i += MK_WORKERS) {
            float sum = mm_merged_sum(partials, tile, slices, partial_floats, i);
            if (softcap) {
                sum = tanhf(sum / 30.0f) * 30.0f;
            }
            reinterpret_cast<float*>(task.c)[(i / MM_TILE) * task.n + tile * MM_TILE + (i % MM_TILE)] = sum;
        }
    }
}
