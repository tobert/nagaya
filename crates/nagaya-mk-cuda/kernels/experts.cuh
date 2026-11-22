// The expert ops: OP_EXPERT_NVFP4 and OP_EXPERT_MXFP8. For a list of pairs of an expert and its rows, they compute
// `y = W_e * x` with the block-scaled MMA of the tensor cores. The weight is operand A, and the quantized rows are
// operand B (the columns n8 of the MMA). The sum is in f32.
//
// The descriptor:
//   a      the buffer of quantized rows (the layout is in quant.cuh). `rows` is its capacity.
//   b      the bank of repacked weights: the item `(expert, tile)` is at `b + (expert * (n / 64) + tile) * item_bytes`,
//          and an item holds the 64 weight rows of a column tile (`src/experts.rs` writes the layout).
//   c      the output, f32: `y_row` of a pair is a row of `n` floats.
//   d      the task table. NVFP4 needs it: the slot MK_EXPERT_TABLE_AMAX holds the pointer to the amax of each expert
//          (`f32`, 128 of them).
//   e      the list of pairs: a header with the live count of pairs, then the pairs (`ExpertPairHeader` and
//          `ExpertPair`, in protocol.rs). Item `i` takes the pair `i / (n / 64)`. `f` is free: it is for the scratch of a
//          merge.
//   n      the rows of the weight: a multiple of 64.
//   k      the values of a row: a multiple of 128.
//   items  a static bound: the most pairs that the list can hold, times `n / 64`. The live count is in the header. An
//          item of a pair at or after the live count does nothing, and an empty pair (count 0) does nothing. Both
//          count as done for the signal.
//   p0     the expert scale (1.0 on sm_120 and sm_121).
//   p1     NVFP4 only: the outer scale denominator of the weight.
//   item_bytes is 0: the items are not in the order of the item index, so the producer warp prefetches nothing. The op
//   prefetches the next item of its block itself (`experts_prefetch_next`).
//
// The cut. An item is a tile of 64 weight rows, which are 4 tiles of 16 rows of the MMA, and all of K. The 8 consumer
// warps take a tile of 16 rows each and a half of the groups of K. The first half is the groups 0 to `(K / 128 + 1) /
// 2 - 1`. A warp walks its groups in order with one chain of MMAs. The two halves join in shared memory: the first
// half plus the second half, in this order. The cut comes from the shapes only, so the bits do not depend on the grid,
// on the block that runs an item, or on the count of rows. Each column of the MMA is independent of the others, so a
// row has the same bits alone and in a pair of 16.
//
// The epilogue of NVFP4 (engine-abi "The expert epilogue of NVFP4"): `(total * expert_scale) * row_scale`, with
// `row_scale = (x_amax * w_amax) * (1 / (1536 * w_denominator))`. The epilogue of MXFP8 is `total * expert_scale`.
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor`, quant.cuh and the worker functions of megakernel.cu.

#pragma once

struct ExpertPair {
    uint32_t expert;
    uint32_t count;
    uint16_t x_row[MK_EXPERT_MAX_ROWS];
    uint16_t y_row[MK_EXPERT_MAX_ROWS];
    float weight[MK_EXPERT_MAX_ROWS];
};

struct ExpertPairHeader {
    uint32_t pairs;
    uint32_t reserved[3];
};

static_assert(sizeof(ExpertPair) == MK_EXPERT_PAIR_BYTES, "ExpertPair differs in size from src/protocol.rs");
MK_CHECK_OFFSET(ExpertPair, expert, MK_EXPERT_PAIR_OFF_expert);
MK_CHECK_OFFSET(ExpertPair, count, MK_EXPERT_PAIR_OFF_count);
MK_CHECK_OFFSET(ExpertPair, x_row, MK_EXPERT_PAIR_OFF_x_row);
MK_CHECK_OFFSET(ExpertPair, y_row, MK_EXPERT_PAIR_OFF_y_row);
MK_CHECK_OFFSET(ExpertPair, weight, MK_EXPERT_PAIR_OFF_weight);
static_assert(sizeof(ExpertPairHeader) == MK_EXPERT_PAIR_HEADER_BYTES, "ExpertPairHeader differs in size from src/protocol.rs");
MK_CHECK_OFFSET(ExpertPairHeader, pairs, MK_EXPERT_PAIR_HEADER_OFF_pairs);

// The live count of pairs in the list at `e`. An earlier task writes the list, so the read skips L1.
__device__ __forceinline__ uint32_t live_pairs(const MkTaskDescriptor& task) {
    return __ldcg(&reinterpret_cast<const ExpertPairHeader*>(task.e)->pairs);
}

// Pair `index` of the list at `e`. The caller checked `index` against the live count.
__device__ __forceinline__ const ExpertPair* pair_at(const MkTaskDescriptor& task, uint32_t index) {
    return reinterpret_cast<const ExpertPair*>(task.e + MK_EXPERT_PAIR_HEADER_BYTES) + index;
}

// A pair in global memory. An earlier task writes the list, so every field is read with `__ldcg`, which skips L1.
struct PairRef {
    const ExpertPair* p;
    __device__ __forceinline__ uint32_t expert() const { return __ldcg(&p->expert); }
    __device__ __forceinline__ uint32_t count() const { return __ldcg(&p->count); }
    __device__ __forceinline__ uint32_t x_row(uint32_t i) const { return __ldcg(&p->x_row[i]); }
    __device__ __forceinline__ uint32_t y_row(uint32_t i) const { return __ldcg(&p->y_row[i]); }
    __device__ __forceinline__ float weight(uint32_t i) const { return __ldcg(&p->weight[i]); }
};

// Loads of the weights and of the rows. The weights never change while the kernel runs, so they use the loads of
// read-only data, and they do not allocate a line in L1. An earlier task of the ring writes the quantized rows, the
// pairs and the amaxes, and the next layer writes them again. These loads skip L1 (`.cg`): a read-only load of data
// that the kernel writes is not defined, and a line of L1 could hold the bytes of the layer before.
__device__ __forceinline__ uint32_t ld_cg_u32(const uint8_t* p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_cg_v2(const uint8_t* p) {
    uint2 v;
    asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_nc_v2_weight(const uint8_t* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ld_nc_u32_weight(const uint8_t* p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

// The MMA of NVFP4: 16 weight rows, 8 rows of x, 64 values of K. The scale registers hold 4 `ue4m3` bytes.
__device__ __forceinline__ void mma_nvfp4(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1, uint32_t scale_a, uint32_t scale_b) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%12}, %11, {%12,%12};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b), "n"(0));
}

// The MMA of MXFP8: 16 weight rows, 8 rows of x, 32 values of K. The scale register holds one `ue8m0` byte.
__device__ __forceinline__ void mma_mxfp8(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1, uint32_t scale_a, uint32_t scale_b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%12}, %11, {%12,%12};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b), "n"(0));
}

// Everything that differs between the formats.
struct ExpertNvfp4 {
    static constexpr uint32_t STEPS = 2;                         // MMAs in a group
    static constexpr uint32_t RECORD_BYTES = MK_EXPERT_NVFP4_RECORD_BYTES;
    static __device__ __forceinline__ uint64_t code_bytes(uint32_t k) { return k / 2; }
    static __device__ __forceinline__ uint64_t scale_bytes(uint32_t k) { return k / 16; }
};

struct ExpertMxfp8 {
    static constexpr uint32_t STEPS = 4;
    static constexpr uint32_t RECORD_BYTES = MK_EXPERT_MXFP8_RECORD_BYTES;
    static __device__ __forceinline__ uint64_t code_bytes(uint32_t k) { return k; }
    static __device__ __forceinline__ uint64_t scale_bytes(uint32_t k) { return k / 32; }
};

// What one lane holds for one group of K: the fragments of A and their scales. The weights are the bytes that come from
// DRAM. The rows of x are small, and every block reads them again, so they come from L2 just before the MMAs.
template <typename F>
struct WeightRegs {
    uint4 a[F::STEPS];
    uint32_t scale_a[2];  // NVFP4: the words of the 2 steps. MXFP8: scale_a[0] holds the 4 bytes of the 4 steps.
};

// The columns of B of one lane. `NB` is the count of n8 tiles: 1 for up to 8 rows, and 2 for up to 16.
template <typename F, uint32_t NB>
struct ExpertLane {
    const uint8_t* x_codes[NB];  // the codes of the row of this lane's column of B
    const uint8_t* x_scales[NB];
    bool valid[NB];
    uint32_t lane;
};

// Load the fragments of A of one group. `rec` is the record of the group.
template <typename F>
__device__ __forceinline__ void load_weights(WeightRegs<F>& r, uint32_t lane, const uint8_t* rec) {
    const uint32_t t = lane & 3;
    const uint32_t row = lane >> 2;
#pragma unroll
    for (uint32_t j = 0; j < F::STEPS; ++j) {
        const uint8_t* p = rec + j * 512 + lane * 16;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r.a[j].x), "=r"(r.a[j].y), "=r"(r.a[j].z), "=r"(r.a[j].w)
                     : "l"(p));
    }
    r.scale_a[0] = 0;
    r.scale_a[1] = 0;
    if (t < 2) {
        // The lane holds the scales of the row `row + 8 * t`.
        const uint8_t* p = rec + F::STEPS * 512 + (row * 2 + t) * (F::STEPS == 2 ? 8 : 4);
        if constexpr (F::STEPS == 2) {
            const uint2 s = ld_nc_v2_weight(p);
            r.scale_a[0] = s.x;
            r.scale_a[1] = s.y;
        } else {
            r.scale_a[0] = ld_nc_u32_weight(p);
        }
    }
}

// The MMAs of one group, with the columns of B of the group `g`.
template <typename F, uint32_t NB>
__device__ __forceinline__ void mma_group(float (&acc)[NB][4], const WeightRegs<F>& w, const ExpertLane<F, NB>& l, uint32_t g) {
    const uint32_t t = l.lane & 3;
    uint32_t scale_b[NB][2] = {};
#pragma unroll
    for (uint32_t nb = 0; nb < NB; ++nb) {
        if (t == 0 && l.valid[nb]) {
            if constexpr (F::STEPS == 2) {
                // 2 steps of 4 scale bytes each.
                const uint2 s = ld_cg_v2(l.x_scales[nb] + static_cast<uint64_t>(g) * 8);
                scale_b[nb][0] = s.x;
                scale_b[nb][1] = s.y;
            } else {
                scale_b[nb][0] = ld_cg_u32(l.x_scales[nb] + static_cast<uint64_t>(g) * 4);
            }
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < F::STEPS; ++j) {
#pragma unroll
        for (uint32_t nb = 0; nb < NB; ++nb) {
            uint32_t b0 = 0, b1 = 0;
            if (l.valid[nb]) {
                // The step of K of this MMA. A step has 32 bytes of codes for each row of x, in both formats.
                const uint8_t* p = l.x_codes[nb] + (static_cast<uint64_t>(g) * F::STEPS + j) * 32 + 4 * t;
                b0 = ld_cg_u32(p);
                b1 = ld_cg_u32(p + 16);
            }
            if constexpr (F::STEPS == 2) {
                mma_nvfp4(acc[nb], w.a[j], b0, b1, w.scale_a[j], scale_b[nb][j]);
            } else {
                // The byte j of the words that hold the scales of the 4 steps.
                mma_mxfp8(acc[nb], w.a[j], b0, b1, (w.scale_a[0] >> (8 * j)) & 0xFFu, (scale_b[nb][0] >> (8 * j)) & 0xFFu);
            }
        }
    }
}

// The sum of the groups `first` to `last - 1` of one warp, in order. The warp loads the weights of a group and then runs
// its MMAs. The 8 warps of the block and the prefetch of the next item keep the memory busy. A register double buffer
// would take 160 registers, and the probes of 512 threads need 128.
template <typename F, uint32_t NB>
__device__ __forceinline__ void sum_groups(
    float (&acc)[NB][4], const ExpertLane<F, NB>& l, const uint8_t* tile_stream, uint32_t first, uint32_t last) {
    WeightRegs<F> now;
#pragma unroll 1
    for (uint32_t g = first; g < last; ++g) {
        load_weights<F>(now, l.lane, tile_stream + static_cast<uint64_t>(g) * F::RECORD_BYTES);
        mma_group<F, NB>(acc, now, l, g);
    }
}

template <typename F, uint32_t NB>
__device__ __forceinline__ void expert_item_nb(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item, const PairRef pair) {
    const uint32_t n = task.n;
    const uint32_t k = task.k;
    const uint32_t tiles = n / MK_EXPERT_TILE_ROWS;
    const uint32_t tile = item % tiles;
    const uint32_t groups = k / MK_EXPERT_K_GROUP;
    const uint64_t item_bytes = static_cast<uint64_t>(4) * groups * F::RECORD_BYTES;
    const uint8_t* weights =
        reinterpret_cast<const uint8_t*>(task.b) + (static_cast<uint64_t>(pair.expert()) * tiles + tile) * item_bytes;

    const uint32_t warp = worker_id() >> 5;
    const uint32_t lane = worker_id() & 31;
    const uint32_t mt = warp & 3;  // the tile of 16 weight rows
    const uint32_t half = warp >> 2;
    const uint32_t split = (groups + 1) / 2;

    // The rows of x of the columns of B: the column of this lane is `g` of the n8 tile.
    const uint8_t* x_base = reinterpret_cast<const uint8_t*>(task.a);
    const uint8_t* x_scale_base = x_base + static_cast<uint64_t>(task.rows) * F::code_bytes(k);
    ExpertLane<F, NB> l;
    l.lane = lane;
#pragma unroll
    for (uint32_t nb = 0; nb < NB; ++nb) {
        const uint32_t token = nb * 8 + (lane >> 2);
        l.valid[nb] = token < pair.count();
        const uint32_t x_row = l.valid[nb] ? pair.x_row(token) : 0;
        if (x_row >= task.rows) {
            __trap();  // The pair names a row that the buffer of quantized rows does not hold.
        }
        l.x_codes[nb] = x_base + static_cast<uint64_t>(x_row) * F::code_bytes(k);
        l.x_scales[nb] = x_scale_base + static_cast<uint64_t>(x_row) * F::scale_bytes(k);
    }

    float acc[NB][4];
#pragma unroll
    for (uint32_t nb = 0; nb < NB; ++nb) {
        acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.0f;
    }
    const uint8_t* stream = weights + static_cast<uint64_t>(mt) * groups * F::RECORD_BYTES;
    if (half == 0) {
        sum_groups<F, NB>(acc, l, stream, 0, split);
    } else {
        sum_groups<F, NB>(acc, l, stream, split, groups);
    }

    // The second half goes to shared memory. The first half adds it to its own sum, and it writes the result.
    float* part = reinterpret_cast<float*>(mk_smem);
    if (half == 1) {
#pragma unroll
        for (uint32_t nb = 0; nb < NB; ++nb) {
            *reinterpret_cast<float4*>(part + ((mt * 2 + nb) * 32 + lane) * 4) =
                make_float4(acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3]);
        }
    }
    worker_sync();
    if (half == 0) {
        const uint32_t t = lane & 3;
        const uint32_t col = tile * MK_EXPERT_TILE_ROWS + mt * 16 + (lane >> 2);
        const float expert_scale = task.p0;
        float* y = reinterpret_cast<float*>(task.c);
        float w_amax = 0.0f;
        float inverse_row_scale = 0.0f;
        if constexpr (F::STEPS == 2) {
            w_amax = reinterpret_cast<const float*>(table->ptr[MK_EXPERT_TABLE_AMAX])[pair.expert()];
            inverse_row_scale = 1.0f / (1536.0f * task.p1);
        }
#pragma unroll
        for (uint32_t nb = 0; nb < NB; ++nb) {
            const float4 second = *reinterpret_cast<const float4*>(part + ((mt * 2 + nb) * 32 + lane) * 4);
            const float total[4] = {acc[nb][0] + second.x, acc[nb][1] + second.y, acc[nb][2] + second.z, acc[nb][3] + second.w};
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                // d0, d1: the row g, the columns 2t and 2t + 1. d2, d3: the row g + 8.
                const uint32_t token = nb * 8 + 2 * t + (i & 1);
                if (token < pair.count()) {
                    float value = total[i] * expert_scale;
                    if constexpr (F::STEPS == 2) {
                        const float x_amax = __ldcg(reinterpret_cast<const float*>(x_base + static_cast<uint64_t>(task.rows) * (F::code_bytes(k) + F::scale_bytes(k))) + pair.x_row(token));
                        value = value * ((x_amax * w_amax) * inverse_row_scale);
                    }
                    y[static_cast<uint64_t>(pair.y_row(token)) * n + col + (i >> 1) * 8] = value;
                }
            }
        }
    }
}

// The warps ask L2 for the bytes that they read in the next item of this block, so the first loads of that item do not
// wait for DRAM. One bulk prefetch for each stream of a warp. A bulk prefetch needs a multiple of 16 bytes.
template <typename F>
__device__ __forceinline__ void experts_prefetch_next(const MkTaskDescriptor& task, uint32_t item) {
    const uint32_t next = item + gridDim.x;
    const uint32_t tiles = task.n / MK_EXPERT_TILE_ROWS;
    if (next >= task.items || (worker_id() & 31) != 0) {
        return;
    }
    // The next item may be past the live pairs, or it may hold an empty pair. Then there is nothing to prefetch.
    if (next / tiles >= live_pairs(task)) {
        return;
    }
    const PairRef pair{pair_at(task, next / tiles)};
    if (pair.count() == 0 || pair.count() > MK_EXPERT_MAX_ROWS) {
        return;
    }
    const uint32_t groups = task.k / MK_EXPERT_K_GROUP;
    const uint64_t item_bytes = static_cast<uint64_t>(4) * groups * F::RECORD_BYTES;
    const uint32_t warp = worker_id() >> 5;
    const uint32_t mt = warp & 3;
    const uint32_t half = warp >> 2;
    const uint32_t split = (groups + 1) / 2;
    const uint32_t first = half == 0 ? 0 : split;
    const uint32_t last = half == 0 ? split : groups;
    const uint8_t* at = reinterpret_cast<const uint8_t*>(task.b) +
                        (static_cast<uint64_t>(pair.expert()) * tiles + next % tiles) * item_bytes +
                        (static_cast<uint64_t>(mt) * groups + first) * F::RECORD_BYTES;
    bulk_prefetch_l2(at, (last - first) * F::RECORD_BYTES);
}

template <typename F>
__device__ __forceinline__ void op_expert(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    if (worker_count() != MK_CONSUMER_THREADS || (task.n % MK_EXPERT_TILE_ROWS) != 0 || (task.k % MK_EXPERT_K_GROUP) != 0 ||
        dynamic_smem_bytes() < MK_EXPERT_SMEM_BYTES || task.item_bytes != 0 ||
        (F::STEPS == 2 && table == nullptr)) {
        __trap();  // The cut needs 8 warps, whole tiles and groups, and the shared memory of the partial sums.
    }
    const uint32_t tiles = task.n / MK_EXPERT_TILE_ROWS;
    // The count of items is the static bound of pairs times the tiles. The list has room for the bound.
    if ((task.e & 15) != 0 || task.items % tiles != 0 || live_pairs(task) > task.items / tiles) {
        __trap();
    }
    // An item of a pair that is not live, and an item of an empty pair, do nothing. All the consumers take the same
    // way, and the item counts as done.
    if (item / tiles >= live_pairs(task)) {
        return;
    }
    const PairRef pair{pair_at(task, item / tiles)};
    if (pair.count() > MK_EXPERT_MAX_ROWS) {
        __trap();
    }
    if (pair.count() == 0) {
        return;
    }
    experts_prefetch_next<F>(task, item);
    if (pair.count() <= 8) {
        expert_item_nb<F, 1>(task, table, item, pair);
    } else {
        expert_item_nb<F, 2>(task, table, item, pair);
    }
}

__device__ __forceinline__ void op_expert_nvfp4(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    op_expert<ExpertNvfp4>(task, table, item);
}

__device__ __forceinline__ void op_expert_mxfp8(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    op_expert<ExpertMxfp8>(task, table, item);
}
