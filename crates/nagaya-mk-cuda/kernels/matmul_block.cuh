// The block form of OP_MATMUL_BF16 (FLAG_BLOCK_FORM): a prefill of 17 to 512 rows. It gives the same bits as the decode
// form of `matmul_bf16.cuh` for each row, at each count of rows and at each grid.
//
// This header needs `matmul_bf16.cuh` (the split, the MMA and the sizes), `probe_ops.cuh` (the bulk copy and the
// mbarrier) and the worker functions of `megakernel.cu`.
//
// A work item is a group of up to 8 column tiles (128 columns), a block of 32 rows, and all of K. The item number is
// `group * row_blocks + row_block`, so the row blocks of one group run together, and they read the same weights.
//
// The 8 consumer warps split the outputs, and not K. A warp is `wc` (0 or 1: 4 column tiles, so 64 columns) and `wr`
// (0 to 3: 8 rows). It keeps 4 independent chains, one for each of its column tiles (operand A) against its 8 rows
// (operand B). Each chain keeps the order of the decode form:
// 1. A chunk is the `kt / 128` consecutive k16 steps that one warp of the decode form runs. The chain starts at zero
//    at the start of each chunk, never at the start of a stage.
// 2. In a step the chain adds the terms lo, mid and hi of x, in this order. The weight is exact, so a product is exact.
// 3. The 8 chunks of a slice join as ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)): the tree of the decode form.
// 4. The slices add in order: total = S0, total += S1, and so on. The item has all the slices, so no partial goes to
//    memory, and no counter is needed.
// The tensor core gives each result from its own accumulator and its own products. Thus the bits do not depend on the
// count of rows, on the place of a row in the block, or on the grid.
//
// The ring has `MB_STAGES` stages. A stage is 3 k16 steps (48 K values), the size that divides the chunks of 6 and of
// 9 steps. It holds the weights of the 8 tiles and the f32 x of the 32 rows. All the consumers copy the stage with
// `cp.async` pieces of 16 bytes, and each thread arrives at the mbarrier of the stage when its pieces are in
// (`cp.async.mbarrier.arrive.noinc`). The consumers of the stage wait for the barrier. Copies of one tile (1.5 KB) or
// one row (192 B) with `cp.async.bulk` took about 30 ns each, and 40 of them were slower than the MMAs of the stage.
// A warp splits the x that it needs into the three bf16 terms, and it applies the norm, while it runs the MMAs.
//
// The tree of a lane has 5 sets of 16 floats. Two live in registers (the chunk in progress and one pair), and the
// first quad and the total live in shared memory, because the kernel has only 168 registers.
//
// The dynamic shared memory: MB_META (the barriers), the stages, then the two sets. The task needs `MB_SMEM_BYTES`.

#pragma once

constexpr uint32_t MB_GROUP_TILES = 8;     // column tiles of an item
constexpr uint32_t MB_ROW_BLOCK = 32;      // rows of an item
constexpr uint32_t MB_STAGE_STEPS = 3;     // k16 steps of a stage: it divides the chunks of 6 and of 9 steps
constexpr uint32_t MB_STAGE_K = MB_STAGE_STEPS * MM_K_STEP;
constexpr uint32_t MB_STAGES = 3;          // stages in the ring
constexpr uint32_t MB_TILE_STAGE_BYTES = MB_STAGE_STEPS * 512;  // one tile in one stage: one bulk copy
constexpr uint32_t MB_W_STAGE_BYTES = MB_GROUP_TILES * MB_TILE_STAGE_BYTES;
// One row of x in a stage: 48 floats, in a pitch of 56 floats. A half warp reads 4 rows and 4 lanes of 8 bytes, and the
// pitch puts the 4 rows 8 banks apart.
constexpr uint32_t MB_X_ROW_BYTES = MB_STAGE_K * 4;
constexpr uint32_t MB_X_PITCH = 224;
constexpr uint32_t MB_X_STAGE_BYTES = MB_ROW_BLOCK * MB_X_PITCH;
constexpr uint32_t MB_STAGE_BYTES = MB_W_STAGE_BYTES + MB_X_STAGE_BYTES;
constexpr uint32_t MB_META = 2048;
constexpr uint32_t MB_BARRIERS = 1024;     // inside MB_META: one 8-byte barrier for each stage
constexpr uint32_t MB_RING_AT = MB_META;
// The two sets in shared memory, 16 floats for each consumer, in the order [word][consumer], so that a warp reads
// consecutive words. The sums of the slices, and the first quad of the slice in progress.
constexpr uint32_t MB_TOTAL_AT = MB_RING_AT + MB_STAGES * MB_STAGE_BYTES;
constexpr uint32_t MB_Q_AT = MB_TOTAL_AT + 16 * MK_CONSUMER_THREADS * 4;
constexpr uint32_t MB_SMEM_BYTES = MB_Q_AT + 16 * MK_CONSUMER_THREADS * 4;

__device__ __forceinline__ uint2 mb_ld_v2(uint32_t addr) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ float mb_ld_f32(uint32_t addr) {
    float v;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
    return v;
}

__device__ __forceinline__ void mb_st_f32(uint32_t addr, float v) {
    asm volatile("st.shared.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory");
}

__device__ __forceinline__ uint4 mb_ld_v4(uint32_t addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
    return v;
}

// The 16 accumulators of a lane: [tile of 16 columns][word of the MMA result].
struct MbAcc {
    float v[4][4];
};

__device__ __forceinline__ void mb_zero(MbAcc& a) {
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
            a.v[i][r] = 0.0f;
        }
    }
}

// dst = p.
__device__ __forceinline__ void mb_copy(MbAcc& dst, const MbAcc& p) {
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) {
            dst.v[i][r] = p.v[i][r];
        }
    }
}

// The leaf `chunk` (0 to 7) of a slice joins the tree of the decode form: `a` and `h` hold the pair sums, `q` (in
// shared memory) holds p0 + p1 + p2 + p3. After the leaf 7, `total` (also in shared memory) takes the slice sum: it is
// the sum itself for the slice 0, and it adds the sum for each later slice. `q` and `total` are the words of this
// consumer. Every lane runs the same branch, so the warp does not diverge.
__device__ __forceinline__ void mb_leaf(const MbAcc& p, MbAcc& a, MbAcc& h, uint32_t q, uint32_t total, uint32_t chunk,
                                        uint32_t slice) {
    if ((chunk & 1) == 0) {
        mb_copy(a, p);
    } else if (chunk == 1 || chunk == 5) {
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
                h.v[i][r] = a.v[i][r] + p.v[i][r];
            }
        }
    } else if (chunk == 3) {
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
                mb_st_f32(q + (i * 4 + r) * MK_CONSUMER_THREADS * 4, h.v[i][r] + (a.v[i][r] + p.v[i][r]));
            }
        }
    } else {
        const bool first = slice == 0;
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
                const float s = mb_ld_f32(q + (i * 4 + r) * MK_CONSUMER_THREADS * 4) + (h.v[i][r] + (a.v[i][r] + p.v[i][r]));
                const uint32_t word = total + (i * 4 + r) * MK_CONSUMER_THREADS * 4;
                mb_st_f32(word, first ? s : mb_ld_f32(word) + s);
            }
        }
    }
}

// The geometry of one item, as the consumers see it.
struct MbItem {
    uint32_t first_tile;  // the first column tile of the item
    uint32_t tiles;       // column tiles of the item: 1 to 8
    uint32_t row0;        // the first row of the block
    uint32_t rows;        // rows of the block: 1 to 32
};

__device__ __forceinline__ MbItem mb_item_of(const MkTaskDescriptor& task, uint32_t item) {
    const uint32_t row_blocks = (task.rows + MB_ROW_BLOCK - 1) / MB_ROW_BLOCK;
    const uint32_t group = item / row_blocks;
    MbItem it;
    it.first_tile = group * MB_GROUP_TILES;
    const uint32_t left = task.n / MM_TILE - it.first_tile;
    it.tiles = left < MB_GROUP_TILES ? left : MB_GROUP_TILES;
    it.row0 = (item % row_blocks) * MB_ROW_BLOCK;
    const uint32_t rows_left = task.rows - it.row0;
    it.rows = rows_left < MB_ROW_BLOCK ? rows_left : MB_ROW_BLOCK;
    return it;
}

// A copy of 16 bytes from global to shared memory, through L2 and not L1, that the thread does not wait for.
__device__ __forceinline__ void mb_copy16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}

// The barrier gets one arrival from this thread when all its earlier copies are done. `noinc` leaves the count of the
// barrier as it was, so the barrier expects one arrival from each consumer in each phase.
__device__ __forceinline__ void mb_arrive_when_copied(uint32_t barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(barrier) : "memory");
}

// Every consumer copies its share of the stage `index`: the weights of the tiles (96 pieces of 16 bytes for each tile)
// and the x of the rows (12 pieces for each row), 768 + 384 pieces for 256 threads. A stage lies inside one slice.
// Small bulk copies (192 bytes for a row) were slower than these: about 30 ns for each copy, so 40 copies took more
// than the compute of the stage. Then each thread arrives at the barrier of the stage.
__device__ __forceinline__ void mb_issue(uint32_t smem, const MkTaskDescriptor& task, const MbItem& it, const float* x,
                                         uint32_t index) {
    const uint32_t buffer = index % MB_STAGES;
    const uint32_t stage_at = smem + MB_RING_AT + buffer * MB_STAGE_BYTES;
    const uint32_t stages_per_slice = task.kt / MB_STAGE_K;
    const uint32_t slice = index / stages_per_slice;
    const uint32_t in_slice = index % stages_per_slice;
    const uint8_t* weights = reinterpret_cast<const uint8_t*>(task.b) + static_cast<uint64_t>(in_slice) * MB_TILE_STAGE_BYTES;
    const uint32_t pieces_per_tile = MB_TILE_STAGE_BYTES / 16;
#pragma unroll 1
    for (uint32_t h = 0; h < 3; ++h) {
        const uint32_t p = worker_id() + h * MK_WORKERS;
        const uint32_t tile = p / pieces_per_tile;
        if (tile < it.tiles) {
            const uint64_t item_at = static_cast<uint64_t>(it.first_tile + tile) * task.group_items + slice;
            mb_copy16(stage_at + p * 16, weights + item_at * task.item_bytes + (p % pieces_per_tile) * 16);
        }
    }
    const uint32_t pieces_per_row = MB_X_ROW_BYTES / 16;
#pragma unroll 1
    for (uint32_t h = 0; h < 2; ++h) {
        const uint32_t p = worker_id() + h * MK_WORKERS;
        const uint32_t row = p / pieces_per_row;
        if (row < it.rows) {
            mb_copy16(stage_at + MB_W_STAGE_BYTES + row * MB_X_PITCH + (p % pieces_per_row) * 16,
                      x + static_cast<uint64_t>(it.row0 + row) * task.k + static_cast<uint64_t>(index) * MB_STAGE_K +
                          (p % pieces_per_row) * 4);
        }
    }
    mb_arrive_when_copied(smem + MB_BARRIERS + buffer * 8);
}

// The sums of the item go to `MB_TOTAL_AT` in shared memory, 16 words for each consumer. Word `i * 4 + r` is the
// result `r` of the MMA for the tile `wc * 4 + i` of the item and the 8 rows `wr * 8`. Every consumer calls this
// function, and it ends with the barrier of the consumers. `factors` is null with no norm, and `norm_w` is null with
// no weight.
__device__ __forceinline__ void mb_core(const MkTaskDescriptor& task, const MbItem& it, const float* x,
                                        const float* factors, const uint16_t* norm_w) {
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(mk_smem));
    const uint32_t id = worker_id();
    const uint32_t warp = id >> 5;
    const uint32_t lane = id & 31;
    const uint32_t g = lane >> 2;
    const uint32_t t = lane & 3;
    const uint32_t wc = warp & 1;
    const uint32_t wr = warp >> 1;
    const uint32_t nstages = task.k / MB_STAGE_K;
    const uint32_t stages_per_chunk = task.kt / (MM_WARPS * MB_STAGE_K);

    // The row of this lane in the block, and its row factor.
    const uint32_t lrow = wr * 8 + g;
    const float rf = factors != nullptr && lrow < it.rows ? __ldcg(factors + it.row0 + lrow) : 1.0f;
    const uint16_t* ws = norm_w != nullptr ? norm_w + 2 * t : nullptr;

    if (id == 0) {
#pragma unroll 1
        for (uint32_t s = 0; s < MB_STAGES; ++s) {
            mbarrier_init(smem + MB_BARRIERS + s * 8, MK_WORKERS);
        }
        fence_mbarrier_init();
    }
    // A row that the block does not have reads as zero: the copies never write its place in the stages.
    if (it.rows < MB_ROW_BLOCK) {
        const uint32_t missing = (MB_ROW_BLOCK - it.rows) * (MB_X_ROW_BYTES / 16);
#pragma unroll 1
        for (uint32_t e = id; e < MB_STAGES * missing; e += MK_WORKERS) {
            const uint32_t buffer = e / missing;
            const uint32_t in_buffer = e % missing;
            const uint32_t row = it.rows + in_buffer / (MB_X_ROW_BYTES / 16);
            const uint32_t at = smem + MB_RING_AT + buffer * MB_STAGE_BYTES + MB_W_STAGE_BYTES + row * MB_X_PITCH +
                                (in_buffer % (MB_X_ROW_BYTES / 16)) * 16;
            asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(at), "r"(0u) : "memory");
        }
    }
    worker_sync();
    MbAcc acc, a, h;
    const uint32_t q = smem + MB_Q_AT + id * 4;
    const uint32_t total = smem + MB_TOTAL_AT + id * 4;
    mb_zero(acc);
    mb_zero(a);
    mb_zero(h);
    uint32_t in_chunk = 0;
    uint32_t leaf = 0;
    // One loop issues and computes: the iteration `i` computes the stage `i - MB_STAGES` (when there is one) and then
    // asks for the stage `i` into the buffer that the computed stage freed. One call of `mb_issue` serves both uses.
#pragma unroll 1
    for (uint32_t i = 0; i < nstages + MB_STAGES; ++i) {
        const uint32_t stage = i - MB_STAGES;
        if (i >= MB_STAGES) {
            const uint32_t buffer = stage % MB_STAGES;
            mbarrier_wait_parity(smem + MB_BARRIERS + buffer * 8, (stage / MB_STAGES) & 1);
            const uint32_t stage_at = smem + MB_RING_AT + buffer * MB_STAGE_BYTES;
            const uint32_t wbase = stage_at + wc * 4 * MB_TILE_STAGE_BYTES + lane * 16;
            const uint32_t xbase = stage_at + MB_W_STAGE_BYTES + (wr * 8 + g) * MB_X_PITCH + t * 8;
    #pragma unroll
            for (uint32_t step = 0; step < MB_STAGE_STEPS; ++step) {
                // The values of this lane at the K values 2t and 2t + 1 (b0), and 8 + 2t and 9 + 2t (b1).
                const uint2 raw0 = mb_ld_v2(xbase + step * 64);
                const uint2 raw1 = mb_ld_v2(xbase + step * 64 + 32);
                float2 first = make_float2(__uint_as_float(raw0.x) * rf, __uint_as_float(raw0.y) * rf);
                float2 second = make_float2(__uint_as_float(raw1.x) * rf, __uint_as_float(raw1.y) * rf);
                if (ws != nullptr) {
                    const uint32_t k0 = stage * MB_STAGE_K + step * MM_K_STEP;
                    const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(ws + k0));
                    const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(ws + k0 + 8));
                    first.x = mm_lo_half(w0) * first.x;
                    first.y = mm_hi_half(w0) * first.y;
                    second.x = mm_lo_half(w1) * second.x;
                    second.y = mm_hi_half(w1) * second.y;
                }
                const MmTerms f = mm_split(first.x, first.y);
                const MmTerms s = mm_split(second.x, second.y);
                // Two tiles at a time keep 8 registers of A. A chain still runs lo, mid, hi in order.
    #pragma unroll
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint4 a0 = mb_ld_v4(wbase + (2 * pair) * MB_TILE_STAGE_BYTES + step * 512);
                    const uint4 a1 = mb_ld_v4(wbase + (2 * pair + 1) * MB_TILE_STAGE_BYTES + step * 512);
                    mm_mma(acc.v[2 * pair], a0, f.lo, s.lo);
                    mm_mma(acc.v[2 * pair + 1], a1, f.lo, s.lo);
                    mm_mma(acc.v[2 * pair], a0, f.mid, s.mid);
                    mm_mma(acc.v[2 * pair + 1], a1, f.mid, s.mid);
                    mm_mma(acc.v[2 * pair], a0, f.hi, s.hi);
                    mm_mma(acc.v[2 * pair + 1], a1, f.hi, s.hi);
                }
            }
            // A chunk is 2 or 3 whole stages, so a chunk never ends inside a stage.
            if (++in_chunk == stages_per_chunk) {
                in_chunk = 0;
                mb_leaf(acc, a, h, q, total, leaf & (MM_WARPS - 1), leaf >> 3);
                mb_zero(acc);
                ++leaf;
            }
            // Every consumer has finished the stage: its buffer takes the copy of a later stage.
            worker_sync();
        }
        if (i < nstages) {
            mb_issue(smem, task, it, x, i);
        }
    }
    // All the stages were waited for. The barriers go before another op uses this memory.
    if (id == 0) {
#pragma unroll 1
        for (uint32_t s = 0; s < MB_STAGES; ++s) {
            mbarrier_inval(smem + MB_BARRIERS + s * 8);
        }
    }
    worker_sync();
}

// OP_MATMUL_BF16 with FLAG_BLOCK_FORM. The geometry and the checks are here, and the epilogue is here. `op_block_form`
// in `ops_dense.cuh` calls them and `mb_core`, so that the kernel has one copy of the heavy loop for both ops.
//
// a = x (const float*, `rows` rows of `k` floats, 16-byte aligned), b = repacked W, c = y (float*, `rows` rows of `n`
// floats), e = the row factors (with FLAG_NORM or FLAG_NORM_UNWEIGHTED), n, k, kt, group_items = slices, rows,
// items = column groups * row blocks, item_bytes = 16 * kt * 2. `f` is not used. Every field is checked: a shape that
// the op does not cover traps.
__device__ __forceinline__ void mb_matmul_check(const MkTaskDescriptor& task, const MkTaskTable* table) {
    const uint32_t rows = task.rows;
    const uint32_t kt = task.kt;
    const bool norm = (task.flags & (FLAG_NORM | FLAG_NORM_UNWEIGHTED)) != 0;
    const bool weighted = (task.flags & FLAG_NORM) != 0;
    const uint32_t tiles = task.n / MM_TILE;
    const uint32_t row_blocks = (rows + MB_ROW_BLOCK - 1) / MB_ROW_BLOCK;
    if (worker_count() != MK_WORKERS || rows == 0 || rows > MM_MAX_ROWS || kt == 0 ||
        kt % (MM_WARPS * MB_STAGE_K) != 0 || task.group_items == 0 || task.k != task.group_items * kt ||
        task.n % MM_TILE != 0 || task.items != ((tiles + MB_GROUP_TILES - 1) / MB_GROUP_TILES) * row_blocks ||
        task.item_bytes != MM_TILE * kt * 2 || dynamic_smem_bytes() < MB_SMEM_BYTES || (task.a & 15) != 0 ||
        (task.b & 15) != 0 ||
        (task.flags & ~(FLAG_NORM | FLAG_NORM_UNWEIGHTED | FLAG_SOFTCAP_30 | FLAG_BLOCK_FORM)) != 0 ||
        (task.flags & FLAG_NORM) != 0 && (task.flags & FLAG_NORM_UNWEIGHTED) != 0 ||
        norm && (task.e == 0 || (task.e & 3) != 0) || !norm && task.e != 0 ||
        weighted && (table == nullptr || table->ptr[MK_MATMUL_TABLE_NORM_WEIGHT] == 0 ||
                     (table->ptr[MK_MATMUL_TABLE_NORM_WEIGHT] & 3) != 0)) {
        __trap();
    }
}

// The outputs of the item: the sums in shared memory, with the softcap of the flags, to y.
__device__ __forceinline__ void mb_matmul_epilogue(const MkTaskDescriptor& task, const MbItem& it) {
    const bool softcap = (task.flags & FLAG_SOFTCAP_30) != 0;
    const float* totals = reinterpret_cast<const float*>(mk_smem + MB_TOTAL_AT) + worker_id();
    const uint32_t warp = worker_id() >> 5;
    const uint32_t lane = worker_id() & 31;
    const uint32_t wc = warp & 1;
    const uint32_t wr = warp >> 1;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t tile = wc * 4 + i;
        if (tile < it.tiles) {
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
                const uint32_t row = it.row0 + wr * 8 + 2 * (lane & 3) + (r & 1);
                const uint32_t column = (it.first_tile + tile) * MM_TILE + (lane >> 2) + 8 * (r >> 1);
                float sum = totals[(i * 4 + r) * MK_CONSUMER_THREADS];
                if (softcap) {
                    sum = tanhf(sum / 30.0f) * 30.0f;
                }
                if (row < task.rows) {
                    reinterpret_cast<float*>(task.c)[static_cast<uint64_t>(row) * task.n + column] = sum;
                }
            }
        }
    }
}
