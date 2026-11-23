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
//
// The modes of the task (wave B). The flags `FLAG_UP` and `FLAG_DOWN` pick the mode. A task with neither flag is the plain
// matmul above. The other fields are as above, and the additions are these.
//
// The cut of the items (all modes). `pos0` is the count of tiles of a pair that one item covers: 0 means 1. An item runs
// its tiles one after the other. A decode step of 8 pairs with 36 tiles has 288 tiles, and with `pos0` 6 it has 48 items,
// one for each block, so a block counts one arrival at a merge counter and not six. `n / 64` must be a multiple of
// `pos0`, and `items` is the bound of pairs times `n / 64 / pos0`. The sums of a tile do not depend on `pos0`.
//
// FLAG_UP, the up matmul of a layer. `n` is 2,304: row `g` of an MMA tile is the gate `i` and row `g + 8` is the up
// `i` of the activation column `i` (`repack_expert_up` in src/experts.rs makes this order from the interleaved rows
// of the checkpoint). So a thread holds the gate and the up of one value. The epilogue gives
// `swiglu(gate, up) * route_weight` (`ops::swiglu`; the route weight comes before the quantizer, as the binding says).
//   kt           the count of tokens: the buffers hold `8 * kt` slots.
//   group_items  the items of a pair, `n / 64 / pos0`: the last item of a pair quantizes its rows.
//   c            the activation in `f32`: slot `y_row` is a row of `n / 2` floats.
//   d            the table: the slot EXPERT_TABLE_QACT holds the buffer of quantized rows, `8 * kt` rows of `n / 2`
//                values (the layout of quant.cuh), and the slot MK_EXPERT_TABLE_AMAX the amax of the weights.
//   f            one `u32` counter for each pair of the bound. They start at 0 and the op leaves them at 0.
// The finisher quantizes each row of the pair with the rules of `quant.rs`: NVFP4 for OP_EXPERT_NVFP4, MXFP8 for
// OP_EXPERT_MXFP8. The row `y_row` of the pair goes to the row `y_row` of the buffer.
//
// FLAG_DOWN, the down matmul. It reads the quantized activation row of the slot `y_row` (not `x_row`), so one list of
// pairs serves both matmuls. `rows` is `8 * kt`, the rows of the buffer `a`.
//   kt           the count of tokens.
//   c            the slot rows in `f32`: `8 * kt` rows of `n` floats. Slot `8 t + j` is the output of the expert `j` of
//                the token `t`, with the experts in ascending order.
//   d            the table: MK_EXPERT_TABLE_AMAX; EXPERT_TABLE_RESIDUAL (`kt` rows of `n` floats, the residual `r`);
//                EXPERT_TABLE_CONV_WEIGHT (`n` rows of 4 floats, tap 3 weights the current row);
//                EXPERT_TABLE_CONV_STATE (3 rows of `n` floats, the 3 last sums, oldest first); EXPERT_TABLE_OUT
//                (`kt` rows of `n` floats, the output `x`; it may be the residual).
//                EXPERT_TABLE_OUT_RF (`kt` floats: the row factor of each row of `x`).
//   f            the scratch, `ex_down_scratch_words(n, kt)` words, at 0 and left at 0: one `u32` counter for each column
//                tile (`n / 64` of them), the counter of the row finisher, rounded up to a multiple of 4 words, and then
//                2 floats for each token of each tile (the sums of squares of the two halves of the tile).
// The last item of a column tile does the finisher of the tile. It waits for every live pair, so a pair with no rows
// still counts. For each token in order it sums the 8 slots in slot order from 0, runs `ops::short_conv` with the state,
// and writes `x = r + conv`. Then it pushes the last 3 sums into the state. A token has exactly 8 slots.
// The row factor of `x`. The finisher of a tile also writes the sum of squares of its 64 values of each token, in two
// halves of 32 columns, each half in the butterfly of the shuffles (offsets 16, 8, 4, 2, 1). Then it arrives at the
// counter of the row finisher. The last tile to arrive is the row finisher: for each token it adds the 36 tiles in
// tile order, each tile as `half 0 + half 1`, starting from the tile 0, and it writes `rf = 1 / sqrt(sum / n + 1e-6)`
// to EXPERT_TABLE_OUT_RF. The sum order is fixed, and no float is added with an atomic. The merger resets the counter.
//
// The up mode needs a dynamic shared memory of MK_EXPERT_SMEM_BYTES + 16 bytes (4,112), and the down mode needs
// MK_EXPERT_SMEM_BYTES + 16 + EX_DOWN_SMEM_BYTES bytes (21,264).
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

// The three modes of an expert op. EX_PLAIN is the matmul of wave A: `y = W x` for the rows of a pair. EX_UP is the up
// matmul of the layer: the epilogue gives the activation of the experts, and a row finisher quantizes it. EX_DOWN is the
// down matmul: the epilogue writes a slot row, and a column finisher sums the slots, runs the convolution and adds the
// residual. The mode is a template argument, so a mode has only its own code.
constexpr uint32_t EX_PLAIN = 0;
constexpr uint32_t EX_UP = 1;
constexpr uint32_t EX_DOWN = 2;

// The slots of the task table of the modes UP and DOWN. The slot 0 is MK_EXPERT_TABLE_AMAX. The same numbers are in
// src/experts.rs.
constexpr uint32_t EXPERT_TABLE_QACT = 1;       // UP: the buffer of quantized activation rows, for `8 * kt` rows.
constexpr uint32_t EXPERT_TABLE_RESIDUAL = 2;   // DOWN: the residual `r`, `kt` rows of `n` floats.
constexpr uint32_t EXPERT_TABLE_CONV_WEIGHT = 3;  // DOWN: the weights of `mlp_conv`, `n` rows of 4 floats.
constexpr uint32_t EXPERT_TABLE_CONV_STATE = 4;   // DOWN: the state of the convolution, 3 rows of `n` floats.
constexpr uint32_t EXPERT_TABLE_OUT = 5;          // DOWN: the output `x`, `kt` rows of `n` floats.
constexpr uint32_t EXPERT_TABLE_OUT_RF = 6;       // DOWN: the row factor of each row of `x`, `kt` floats.

// The epsilon of the row factor (`ops::EPS` of the host form; it is RW_EPS of ops_rows.cuh, which comes after this file).
constexpr float EX_RF_EPS = 1e-6f;

// The word of the scratch `f` of the down op where the partial sums start: after one counter for each column tile and
// the counter of the row finisher, rounded up to 4 words. `src/experts.rs` has the same rule (`down_scratch_words`).
__device__ __forceinline__ uint32_t ex_down_partials_offset(uint32_t tiles) { return (tiles + 1 + 3) & ~3u; }

// The columns of the activation of one expert, and the values of K of the down matmul: the half of `n` of the up matmul.
constexpr uint32_t EXPERT_UP_TILE_COLUMNS = MK_EXPERT_TILE_ROWS / 2;

// The shared memory after the partial sums (MK_EXPERT_SMEM_BYTES) holds the flag of an arrival (16 bytes), and then the
// sums of the down finisher.
constexpr uint32_t EX_FINISH_OFFSET = MK_EXPERT_SMEM_BYTES + 16;

// The division of this file and of ops_moe.cuh. A `/` or `__frcp_rn` can call the slow path of the division, and a call
// makes the compiler save every live register, which the build reports as a spill (the kernel has 168 registers and no
// room). So the sigmoid, the softcaps and the route weight use the fast division, which has no call and an error of 2
// ulps. The device `expf` and `tanhf` already differ from the host by an ulp, so no gate of these values is a bit gate.
__device__ __forceinline__ float ex_sigmoid(float x) { return __fdividef(1.0f, 1.0f + expf(-x)); }

// `softcap(x, 4)` and `softcap(x, 25)` of `ops::softcap`. The factor 0.25 is exact, and 0.04 is the `f32` of 1 / 25.
__device__ __forceinline__ float ex_softcap4(float x) { return tanhf(x * 0.25f) * 4.0f; }
__device__ __forceinline__ float ex_softcap25(float x) { return tanhf(x * 0.04f) * 25.0f; }

// The correctly rounded `1 / y` for a normal `y`, with no call to the slow path (see above). The approximation of the
// hardware is within 1 ulp, and one fused step makes it correct (Markstein). `tests/cuda_experts.rs` checks the bits
// of the NVFP4 epilogue with several denominators.
__device__ __forceinline__ float ex_reciprocal(float y) {
    float approx;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(approx) : "f"(y));
    return __fmaf_rn(__fmaf_rn(-approx, y, 1.0f), approx, approx);
}

// The quotient of `x` by the constant `d`, correctly rounded, with `r` the correctly rounded `1 / d` as a constant:
// one fused step corrects `x * r` (Markstein). It has no call to the slow path. The fused multiply-add is explicit,
// because the build turns the contraction of `a * b + c` off.
__device__ __forceinline__ float ex_div_const(float x, float d, float r) {
    const float q = x * r;
    return __fmaf_rn(__fmaf_rn(-q, d, x), r, q);
}

// The softcapped SwiGLU of an expert (`ops::swiglu`): the sigmoid sees the raw gate.
__device__ __forceinline__ float ex_swiglu(float gate, float up) {
    return ex_softcap4(gate) * ex_sigmoid(gate) * ex_softcap25(up);
}

// The row of x of the column `token` of the pair. The down matmul reads the activation row of the slot that the up
// matmul wrote, and that slot is `y_row`.
template <uint32_t MODE>
__device__ __forceinline__ uint32_t pair_x_row(const PairRef& pair, uint32_t token) {
    return MODE == EX_DOWN ? pair.y_row(token) : pair.x_row(token);
}

// The finisher of the pair `pair_index` in the up op, in the last of its 36 tile items: every consumer quantizes the
// rows of the pair, one after the other. The slot `y_row` of a row is its row in the activation and in the buffer.
template <typename F>
__device__ __forceinline__ void up_finish_rows(const MkTaskDescriptor& task, const MkTaskTable* table, const PairRef& pair) {
    const uint32_t k = task.n / 2;
    const uint32_t capacity = 8 * task.kt;
    const float* act = reinterpret_cast<const float*>(task.c);
    uint8_t* base = reinterpret_cast<uint8_t*>(table->ptr[EXPERT_TABLE_QACT]);
    for (uint32_t i = 0; i < pair.count(); ++i) {
        const uint64_t slot = pair.y_row(i);
        const float* x = act + slot * k;
        if constexpr (F::STEPS == 2) {
            quant_nvfp4_row_short(x, k, base + slot * (k / 2), base + static_cast<uint64_t>(capacity) * (k / 2) + slot * (k / 16),
                            reinterpret_cast<float*>(base + static_cast<uint64_t>(capacity) * (k / 2 + k / 16)) + slot);
        } else {
            quant_mxfp8_row(x, k, base + slot * k, base + static_cast<uint64_t>(capacity) * k + slot * (k / 32));
        }
    }
}

// Count the arrival of an item of a merge group, and tell the whole block if it is the last one. The caller has run a
// barrier of the consumers after its last write.
__device__ __forceinline__ bool expert_arrive(uint32_t* counter, uint32_t group_items) {
    uint32_t* flag = reinterpret_cast<uint32_t*>(mk_smem + MK_EXPERT_SMEM_BYTES);
    if (worker_id() == 0) {
        *flag = merge_arrive(counter, group_items) ? 1u : 0u;
    }
    worker_sync();
    // No barrier after the read: the next write of the flag is in a later item, after more barriers.
    return *flag != 0;
}

// The rows of the chunk of the finisher of the down op: tokens at a time. A chunk holds 3 rows of the tokens before it,
// and the sums of its own tokens.
constexpr uint32_t EX_DOWN_CHUNK = 64;
constexpr uint32_t EX_DOWN_SMEM_BYTES = (EX_DOWN_CHUNK + 3) * MK_EXPERT_TILE_ROWS * 4;

// The sum of the 8 slots of a token, in the order of the slots, for the column `column` (`ops::experts` sums the
// experts in ascending order from 0).
__device__ __forceinline__ float down_slot_sum(const float* slots, uint32_t token, uint32_t n, uint32_t column) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < 8; ++s) {
        sum += __ldcg(slots + (static_cast<uint64_t>(token) * 8 + s) * n + column);
    }
    return sum;
}

// The finisher of the column tile `tile` of the down op, in the last of the live pairs that arrive at the tile. For the
// tokens of the task in order it sums the slots, runs the short convolution `(w0 f[t-3] + w1 f[t-2] + w2 f[t-1] +
// w3 f[t] + f[t]) / sqrt(5)` (`ops::short_conv`), and writes `x = r + conv`. Then it pushes the last 3 sums into the
// state. The output of a token does not need the output of the token before it, only the sums, so a chunk of tokens
// runs in parallel. A chunk reads the sums of the 3 tokens before it again, and the first chunk reads the state.
__device__ __forceinline__ void down_finish_tile(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t tile) {
    const uint32_t n = task.n;
    const uint32_t tokens = task.kt;
    const uint32_t tid = worker_id();
    const uint32_t column = tid % MK_EXPERT_TILE_ROWS;
    const uint32_t lane_row = tid / MK_EXPERT_TILE_ROWS;  // 0 to 3
    const uint32_t col = tile * MK_EXPERT_TILE_ROWS + column;
    const float* slots = reinterpret_cast<const float*>(task.c);
    const float* residual = reinterpret_cast<const float*>(table->ptr[EXPERT_TABLE_RESIDUAL]);
    const float* conv_weight = reinterpret_cast<const float*>(table->ptr[EXPERT_TABLE_CONV_WEIGHT]);
    float* state = reinterpret_cast<float*>(table->ptr[EXPERT_TABLE_CONV_STATE]);
    float* out = reinterpret_cast<float*>(table->ptr[EXPERT_TABLE_OUT]);
    float* sums = reinterpret_cast<float*>(mk_smem + EX_FINISH_OFFSET);  // (3 + chunk) rows of 64 columns
    const float w0 = conv_weight[static_cast<uint64_t>(col) * 4 + 0];
    const float w1 = conv_weight[static_cast<uint64_t>(col) * 4 + 1];
    const float w2 = conv_weight[static_cast<uint64_t>(col) * 4 + 2];
    const float w3 = conv_weight[static_cast<uint64_t>(col) * 4 + 3];
    const float divisor = 2.23606801f;  // (5 as f64).sqrt() as f32
    const float inverse = __uint_as_float(0x3ee4f92eu);  // 1 / divisor, rounded to f32
    for (uint32_t first = 0; first < tokens; first += EX_DOWN_CHUNK) {
        const uint32_t count = tokens - first < EX_DOWN_CHUNK ? tokens - first : EX_DOWN_CHUNK;
        // The row `j` of the buffer holds the token `first + j - 3`: a token before the first one is in the state.
        for (uint32_t j = lane_row; j < count + 3; j += MK_CONSUMER_THREADS / MK_EXPERT_TILE_ROWS) {
            const int64_t token = static_cast<int64_t>(first) + j - 3;
            sums[j * MK_EXPERT_TILE_ROWS + column] =
                token < 0 ? __ldcg(state + (3 + token) * n + col) : down_slot_sum(slots, static_cast<uint32_t>(token), n, col);
        }
        worker_sync();
        for (uint32_t j = lane_row; j < count; j += MK_CONSUMER_THREADS / MK_EXPERT_TILE_ROWS) {
            const float* f = sums + j * MK_EXPERT_TILE_ROWS + column;  // f[t-3] is f[0], and f[t] is f[3]
            const float sum = w0 * f[0] + w1 * f[MK_EXPERT_TILE_ROWS] + w2 * f[2 * MK_EXPERT_TILE_ROWS] +
                              w3 * f[3 * MK_EXPERT_TILE_ROWS];
            const float conv = ex_div_const(sum + f[3 * MK_EXPERT_TILE_ROWS], divisor, inverse);
            const uint64_t at = static_cast<uint64_t>(first + j) * n + col;
            const float x = __ldcg(residual + at) + conv;
            out[at] = x;
            // The sum of the squares of the 32 columns of this warp. All the lanes of a warp have the same `j`.
            float square = x * x;
#pragma unroll
            for (uint32_t offset = 16; offset >= 1; offset >>= 1) {
                square += __shfl_xor_sync(0xFFFFFFFFu, square, offset);
            }
            if ((tid & 31) == 0) {
                __stcg(reinterpret_cast<float*>(task.f) + ex_down_partials_offset(n / MK_EXPERT_TILE_ROWS) +
                           (static_cast<uint64_t>(tile) * tokens + first + j) * 2 + ((tid >> 5) & 1),
                       square);
            }
        }
        worker_sync();
    }
    // The new state: the last 3 of the sums `state[0..3]`, `f[0..tokens]`. The values go to registers first, because
    // the new rows overlap the old ones.
    float next[3] = {0.0f, 0.0f, 0.0f};
    if (tid < MK_EXPERT_TILE_ROWS) {
#pragma unroll
        for (uint32_t j = 0; j < 3; ++j) {
            const int64_t token = static_cast<int64_t>(tokens) - 3 + j;
            next[j] = token < 0 ? __ldcg(state + (3 + token) * n + col) : down_slot_sum(slots, static_cast<uint32_t>(token), n, col);
        }
    }
    worker_sync();
    if (tid < MK_EXPERT_TILE_ROWS) {
#pragma unroll
        for (uint32_t j = 0; j < 3; ++j) {
            __stcg(state + static_cast<uint64_t>(j) * n + col, next[j]);
        }
    }
    // The row finisher: the last tile to arrive adds the sums of squares of all the tiles. The sums of this tile are
    // written, so the barrier comes first and the release of the add publishes them.
    const uint32_t tiles = n / MK_EXPERT_TILE_ROWS;
    worker_sync();
    if (!expert_arrive(reinterpret_cast<uint32_t*>(task.f) + tiles, tiles)) {
        return;
    }
    const float* partials = reinterpret_cast<const float*>(task.f) + ex_down_partials_offset(tiles);
    float* factors = reinterpret_cast<float*>(table->ptr[EXPERT_TABLE_OUT_RF]);
    for (uint32_t token = tid; token < tokens; token += MK_CONSUMER_THREADS) {
        float total = 0.0f;
        for (uint32_t t = 0; t < tiles; ++t) {
            const float2 half = __ldcg(reinterpret_cast<const float2*>(partials) + static_cast<uint64_t>(t) * tokens + token);
            const float sum = half.x + half.y;
            total = t == 0 ? sum : total + sum;
        }
        __stcg(factors + token, 1.0f / sqrtf(total / static_cast<float>(n) + EX_RF_EPS));
    }
}

// One pass of an item: the 8 columns `token_base` to `token_base + 7` of the pair, which are one n8 tile of the MMA. A pair
// of 9 to 16 rows takes two passes that read the weights twice (the second time from L2). The two-tile form of wave A
// held 4 more registers for the second tile, and the kernel has no room for them (168 registers, and the build fails on a
// spill). A column of the MMA is independent of the others, so the bits do not change.
template <typename F, uint32_t MODE>
__device__ __forceinline__ void expert_pass(
    const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t tile, const PairRef pair, uint32_t token_base) {
    constexpr uint32_t NB = 1;
    const uint32_t n = task.n;
    const uint32_t k = task.k;
    const uint32_t tiles = n / MK_EXPERT_TILE_ROWS;
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
        const uint32_t token = token_base + nb * 8 + (lane >> 2);
        l.valid[nb] = token < pair.count();
        const uint32_t x_row = l.valid[nb] ? pair_x_row<MODE>(pair, token) : 0;
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
            inverse_row_scale = ex_reciprocal(1536.0f * task.p1);
        }
        // The value of a result for the column `token`: the total, the expert scale, and for NVFP4 the row scale.
        auto scaled = [&](float total, uint32_t token) {
            float value = total * expert_scale;
            if constexpr (F::STEPS == 2) {
                const float x_amax = __ldcg(
                    reinterpret_cast<const float*>(x_base + static_cast<uint64_t>(task.rows) * (F::code_bytes(k) + F::scale_bytes(k))) +
                    pair_x_row<MODE>(pair, token));
                value = value * ((x_amax * w_amax) * inverse_row_scale);
            }
            return value;
        };
#pragma unroll
        for (uint32_t nb = 0; nb < NB; ++nb) {
            const float4 second = *reinterpret_cast<const float4*>(part + ((mt * 2 + nb) * 32 + lane) * 4);
            const float total[4] = {acc[nb][0] + second.x, acc[nb][1] + second.y, acc[nb][2] + second.z, acc[nb][3] + second.w};
            if constexpr (MODE == EX_UP) {
                // The row g holds the gate `i` and the row g + 8 holds the up `i` of the activation column `i`. The
                // activation is the SwiGLU times the route weight (the weight comes before the quantizer, as the binding
                // says).
#pragma unroll
                for (uint32_t e = 0; e < 2; ++e) {
                    const uint32_t token = token_base + nb * 8 + 2 * t + e;
                    if (token < pair.count()) {
                        const uint32_t slot = pair.y_row(token);
                        if (slot >= 8 * task.kt) {
                            __trap();
                        }
                        const float gate = scaled(total[e], token);
                        const float up = scaled(total[2 + e], token);
                        __stcg(y + static_cast<uint64_t>(slot) * (n / 2) + tile * EXPERT_UP_TILE_COLUMNS + mt * 8 + (lane >> 2),
                               ex_swiglu(gate, up) * pair.weight(token));
                    }
                }
            } else {
#pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    // d0, d1: the row g, the columns 2t and 2t + 1. d2, d3: the row g + 8.
                    const uint32_t token = token_base + nb * 8 + 2 * t + (i & 1);
                    if (token < pair.count()) {
                        const uint32_t slot = pair.y_row(token);
                        if (MODE == EX_DOWN && slot >= 8 * task.kt) {
                            __trap();
                        }
                        y[static_cast<uint64_t>(slot) * n + col + (i >> 1) * 8] = scaled(total[i], token);
                    }
                }
            }
        }
    }
}

// The warps ask L2 for the bytes that they read in the next tile of this block, so the first loads of that tile do not
// wait for DRAM. The next tile is the tile after this one in the same item, or the first tile of the next item of this
// block. One bulk prefetch for each stream of a warp. A bulk prefetch needs a multiple of 16 bytes.
template <typename F>
__device__ __forceinline__ void experts_prefetch_next(
    const MkTaskDescriptor& task, uint32_t item, uint32_t item_tiles, uint32_t tpi, uint32_t in_item) {
    const uint32_t tiles = task.n / MK_EXPERT_TILE_ROWS;
    uint32_t next_pair = item / item_tiles;
    uint32_t next_tile = (item % item_tiles) * tpi + in_item + 1;
    if (in_item + 1 == tpi) {
        const uint32_t next = item + gridDim.x;
        if (next >= task.items) {
            return;
        }
        next_pair = next / item_tiles;
        next_tile = (next % item_tiles) * tpi;
    }
    if ((worker_id() & 31) != 0) {
        return;
    }
    // The next item may be past the live pairs, or it may hold an empty pair. Then there is nothing to prefetch.
    if (next_pair >= live_pairs(task)) {
        return;
    }
    const PairRef pair{pair_at(task, next_pair)};
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
                        (static_cast<uint64_t>(pair.expert()) * tiles + next_tile) * item_bytes +
                        (static_cast<uint64_t>(mt) * groups + first) * F::RECORD_BYTES;
    bulk_prefetch_l2(at, (last - first) * F::RECORD_BYTES);
}

template <typename F, uint32_t MODE>
__device__ __forceinline__ void op_expert_mode(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    const uint32_t tiles = task.n / MK_EXPERT_TILE_ROWS;
    // An item covers `tpi` tiles of a pair, one after the other (`pos0`, 0 means 1). Fewer items mean fewer arrivals at a
    // merge counter, and each arrival costs about a microsecond. A tile has the same sums in any item.
    const uint32_t tpi = task.pos0 == 0 ? 1 : task.pos0;
    const uint32_t item_tiles = tiles / tpi;
    const uint32_t live = live_pairs(task);
    // The count of items is the static bound of pairs times the items of a pair. The list has room for the bound.
    if ((task.e & 15) != 0 || tiles % tpi != 0 || task.items % item_tiles != 0 || live > task.items / item_tiles) {
        __trap();
    }
    // An item of a pair that is not live does nothing, and the item counts as done. All the consumers take the same way.
    if (item / item_tiles >= live) {
        return;
    }
    const PairRef pair{pair_at(task, item / item_tiles)};
    if (pair.count() > MK_EXPERT_MAX_ROWS) {
        __trap();
    }
    const uint32_t first_tile = (item % item_tiles) * tpi;
    if (pair.count() == 0) {
        // An empty pair does nothing. In the down op it still counts at each of its tiles, because the finisher of a tile
        // waits for each live pair.
        if constexpr (MODE == EX_DOWN) {
            for (uint32_t j = 0; j < tpi; ++j) {
                if (expert_arrive(reinterpret_cast<uint32_t*>(task.f) + first_tile + j, live)) {
                    down_finish_tile(task, table, first_tile + j);
                }
            }
        }
        return;
    }
#pragma unroll 1
    for (uint32_t j = 0; j < tpi; ++j) {
        if (j > 0) {
            worker_sync();  // The tile before has read the partial sums of shared memory.
        }
        experts_prefetch_next<F>(task, item, item_tiles, tpi, j);
        expert_pass<F, MODE>(task, table, first_tile + j, pair, 0);
        if (pair.count() > 8) {
            worker_sync();  // The first pass has read the partial sums of shared memory.
            expert_pass<F, MODE>(task, table, first_tile + j, pair, 8);
        }
        if constexpr (MODE == EX_DOWN) {
            // The finisher reads the task again, and it does not keep `pair` or `live` across the MMAs: the registers of
            // the main loop are all in use (168), and a build with a spill fails.
            worker_sync();
            if (expert_arrive(reinterpret_cast<uint32_t*>(task.f) + first_tile + j, live_pairs(task))) {
                down_finish_tile(task, table, first_tile + j);
            }
        }
    }
    if constexpr (MODE == EX_UP) {
        worker_sync();
        const uint32_t pair_index = item / item_tiles;
        if (expert_arrive(reinterpret_cast<uint32_t*>(task.f) + pair_index, item_tiles)) {
            up_finish_rows<F>(task, table, PairRef{pair_at(task, pair_index)});
        }
    }
}

template <typename F>
__device__ __forceinline__ void op_expert(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    const uint32_t mode_flags = task.flags & (FLAG_UP | FLAG_DOWN);
    if (worker_count() != MK_CONSUMER_THREADS || (task.n % MK_EXPERT_TILE_ROWS) != 0 || (task.k % MK_EXPERT_K_GROUP) != 0 ||
        dynamic_smem_bytes() < MK_EXPERT_SMEM_BYTES || task.item_bytes != 0 || (F::STEPS == 2 && table == nullptr) ||
        (task.flags & ~(FLAG_PAIRS | FLAG_UP | FLAG_DOWN)) != 0 || mode_flags == (FLAG_UP | FLAG_DOWN)) {
        __trap();  // The cut needs 8 warps, whole tiles and groups, and the shared memory of the partial sums.
    }
    if (mode_flags == FLAG_UP) {
        // The up matmul: `n` rows are the gate and up of `n / 2` activation columns. The quantized rows have `k = n / 2`.
        if (table == nullptr || table->ptr[EXPERT_TABLE_QACT] == 0 || ((task.n / 2) % MK_EXPERT_K_GROUP) != 0 || task.kt == 0 ||
            task.group_items != (task.n / MK_EXPERT_TILE_ROWS) / (task.pos0 == 0 ? 1 : task.pos0) || (task.f & 3) != 0 || (table->ptr[EXPERT_TABLE_QACT] & 15) != 0 ||
            (task.c & 15) != 0 || dynamic_smem_bytes() < EX_FINISH_OFFSET) {
            __trap();
        }
        op_expert_mode<F, EX_UP>(task, table, item);
    } else if (mode_flags == FLAG_DOWN) {
        if (table == nullptr || table->ptr[EXPERT_TABLE_RESIDUAL] == 0 || table->ptr[EXPERT_TABLE_CONV_WEIGHT] == 0 ||
            table->ptr[EXPERT_TABLE_CONV_STATE] == 0 || table->ptr[EXPERT_TABLE_OUT] == 0 ||
            table->ptr[EXPERT_TABLE_OUT_RF] == 0 || (table->ptr[EXPERT_TABLE_OUT_RF] & 3) != 0 || task.kt == 0 ||
            task.rows != 8 * task.kt || (task.f & 7) != 0 ||
            dynamic_smem_bytes() < EX_FINISH_OFFSET + EX_DOWN_SMEM_BYTES) {
            __trap();
        }
        op_expert_mode<F, EX_DOWN>(task, table, item);
    } else {
        op_expert_mode<F, EX_PLAIN>(task, table, item);
    }
}

__device__ __forceinline__ void op_expert_nvfp4(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    op_expert<ExpertNvfp4>(task, table, item);
}

__device__ __forceinline__ void op_expert_mxfp8(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    op_expert<ExpertMxfp8>(task, table, item);
}
