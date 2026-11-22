// Probe ops of the CUDA interpreter. They measure the GB10, and an engine does not use them (`OP_PROBE_*` in
// src/protocol.rs). `tests/cuda_probe.rs` builds their rings and checks each result against the host. Each op
// returns a checksum or a result tile, so that a compiler cannot remove its work and a wrong value fails a test.
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor`, `globaltimer_ns` and `mk_smem` from megakernel.cu.

#pragma once

// ---------------------------------------------------------------------------------------------------------------
// Loads, prefetches and bulk copies.
// ---------------------------------------------------------------------------------------------------------------

constexpr uint32_t PROBE_V4_PER_ITEM = MK_PROBE_ITEM_BYTES / 16;
constexpr uint32_t PROBE_LINES_PER_ITEM = MK_PROBE_ITEM_BYTES / 128;
constexpr uint32_t PROBE_UNROLL = 8;

__device__ __forceinline__ uint32_t sum_words(uint4 v) { return v.x + v.y + v.z + v.w; }

// A plain 16-byte load: it can use the L1 cache.
__device__ __forceinline__ uint4 ld_global_v4(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// A 16-byte load that does not allocate a line in the L1 cache. The design of the interpreter reads weights so.
__device__ __forceinline__ uint4 ld_global_nc_v4(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_shared_v4(uint32_t addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ void prefetch_l2_line(const uint8_t* p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// One thread asks the copy engine to bring `bytes` into L2. `bytes` is a multiple of 16.
__device__ __forceinline__ void bulk_prefetch_l2(const uint8_t* p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint32_t bar, uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(arrivals) : "memory");
}

__device__ __forceinline__ void mbarrier_inval(uint32_t bar) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

// Make the initialization of the barriers visible to the copy engine.
__device__ __forceinline__ void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

// The thread arrives at the barrier, and the barrier expects `bytes` from a bulk copy.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

// Copy `bytes` from global memory to shared memory, and complete the barrier when the bytes arrive.
__device__ __forceinline__ void bulk_copy_to_shared(uint32_t dst, const uint8_t* src, uint32_t bytes, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
        "l"(src), "r"(bytes), "r"(bar)
        : "memory");
}

// Wait until the phase with `parity` of the barrier completes. A wait that never ends traps, as the scoreboard does.
__device__ __forceinline__ void mbarrier_wait_parity(uint32_t bar, uint32_t parity) {
    uint64_t polls = 0;
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(done)
            : "r"(bar), "r"(parity)
            : "memory");
        if (!done && ++polls >= MK_SPIN_LIMIT) {
            __trap();
        }
    }
}

__device__ __forceinline__ uint32_t dynamic_smem_bytes() {
    uint32_t bytes;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
    return bytes;
}

// The sum of `value` over the block, in thread 0. The other threads get a value that is not the sum. The sum of
// `u32` words wraps and does not depend on the order, so the host can compute the same word.
__device__ uint32_t probe_block_sum(uint32_t value) {
    __shared__ uint32_t partial[32];
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x >> 5] = value;
    }
    __syncthreads();
    uint32_t total = 0;
    if (threadIdx.x == 0) {
        for (uint32_t w = 0; w < (blockDim.x + 31) / 32; ++w) {
            total += partial[w];
        }
    }
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------------------------------------------
// OP_PROBE_READ
// ---------------------------------------------------------------------------------------------------------------

// Block `b` of `G` reads the items `b`, `b + G`, ... of `items` items at `base`, with the variants of direct loads. The
// prefetch modes ask for the item `distance` items ahead (in the steps of `G`) when the block starts an item.
// `NC` selects the load. `PREFETCH` is 0 for none, 1 for `prefetch.global.L2` of each line, and 2 for one bulk
// prefetch of the item.
template <bool NC, int PREFETCH>
__device__ uint32_t probe_read_direct(const uint8_t* base, uint32_t items, uint32_t distance) {
    uint32_t sum = 0;
    auto prefetch_item = [&](uint64_t item) {
        if (PREFETCH == 0 || item >= items) {
            return;
        }
        const uint8_t* at = base + item * MK_PROBE_ITEM_BYTES;
        if (PREFETCH == 1) {
            for (uint32_t line = threadIdx.x; line < PROBE_LINES_PER_ITEM; line += blockDim.x) {
                prefetch_l2_line(at + static_cast<uint64_t>(line) * 128);
            }
        } else if (threadIdx.x == 0) {
            bulk_prefetch_l2(at, MK_PROBE_ITEM_BYTES);
        }
    };
    for (uint32_t k = 0; k < distance; ++k) {
        prefetch_item(blockIdx.x + static_cast<uint64_t>(k) * gridDim.x);
    }
    for (uint32_t item = blockIdx.x; item < items; item += gridDim.x) {
        prefetch_item(item + static_cast<uint64_t>(distance) * gridDim.x);
        const uint8_t* at = base + static_cast<uint64_t>(item) * MK_PROBE_ITEM_BYTES;
        for (uint32_t first = 0; first < PROBE_V4_PER_ITEM; first += blockDim.x * PROBE_UNROLL) {
            uint4 v[PROBE_UNROLL];
#pragma unroll
            for (uint32_t j = 0; j < PROBE_UNROLL; ++j) {
                const uint32_t index = first + j * blockDim.x + threadIdx.x;
                v[j] = make_uint4(0, 0, 0, 0);
                if (index < PROBE_V4_PER_ITEM) {
                    v[j] = NC ? ld_global_nc_v4(at + static_cast<uint64_t>(index) * 16)
                              : ld_global_v4(at + static_cast<uint64_t>(index) * 16);
                }
            }
#pragma unroll
            for (uint32_t j = 0; j < PROBE_UNROLL; ++j) {
                sum += sum_words(v[j]);
            }
        }
    }
    return sum;
}

// The same items, copied by 1D bulk copies into a ring of `stages` stages of shared memory. Thread 0 issues each
// copy, `stages` chunks ahead of the chunks that the block reads. The block reads a chunk from shared memory when
// its barrier completes. The shared memory holds the stages, then one 8-byte barrier for each stage.
__device__ uint32_t probe_read_smem_bulk(const uint8_t* base, uint32_t items, uint32_t stages) {
    constexpr uint32_t chunks_per_item = MK_PROBE_ITEM_BYTES / MK_PROBE_STAGE_BYTES;
    constexpr uint32_t v4_per_chunk = MK_PROBE_STAGE_BYTES / 16;
    if (stages == 0 || stages > 32 || dynamic_smem_bytes() < stages * (MK_PROBE_STAGE_BYTES + 8)) {
        __trap();  // The launch gave too little shared memory.
    }
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(mk_smem));
    const uint32_t barriers = smem + stages * MK_PROBE_STAGE_BYTES;
    const uint32_t my_items = items > blockIdx.x ? (items - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
    const uint32_t chunks = my_items * chunks_per_item;
    auto issue = [&](uint32_t j) {
        const uint32_t stage = j % stages;
        const uint64_t item = blockIdx.x + static_cast<uint64_t>(j / chunks_per_item) * gridDim.x;
        const uint8_t* src = base + item * MK_PROBE_ITEM_BYTES + (j % chunks_per_item) * MK_PROBE_STAGE_BYTES;
        mbarrier_arrive_expect_tx(barriers + stage * 8, MK_PROBE_STAGE_BYTES);
        bulk_copy_to_shared(smem + stage * MK_PROBE_STAGE_BYTES, src, MK_PROBE_STAGE_BYTES, barriers + stage * 8);
    };
    if (threadIdx.x == 0) {
        for (uint32_t s = 0; s < stages; ++s) {
            mbarrier_init(barriers + s * 8, 1);
        }
        fence_mbarrier_init();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t j = 0; j < chunks && j < stages; ++j) {
            issue(j);
        }
    }
    uint32_t sum = 0;
    for (uint32_t j = 0; j < chunks; ++j) {
        const uint32_t stage = j % stages;
        mbarrier_wait_parity(barriers + stage * 8, (j / stages) & 1);
        const uint32_t at = smem + stage * MK_PROBE_STAGE_BYTES;
        for (uint32_t index = threadIdx.x; index < v4_per_chunk; index += blockDim.x) {
            sum += sum_words(ld_shared_v4(at + index * 16));
        }
        __syncthreads();  // Every thread has read the stage, so the copy of a later chunk can overwrite it.
        if (threadIdx.x == 0 && j + stages < chunks) {
            issue(j + stages);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t s = 0; s < stages; ++s) {
            mbarrier_inval(barriers + s * 8);
        }
    }
    return sum;
}

// a = the buffer, c = `u32` checksum of each block (the wrapping sum of the words that the block read),
// p0 = items, p1 = the variant (`PROBE_READ_*`), p2 = the distance of a prefetch, or the stages of the bulk copy.
__device__ __forceinline__ void op_probe_read(const MkTaskDescriptor& task) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(task.a);
    uint32_t sum = 0;
    switch (task.p1) {
        case PROBE_READ_PLAIN:
            sum = probe_read_direct<false, 0>(base, task.p0, 0);
            break;
        case PROBE_READ_NC:
            sum = probe_read_direct<true, 0>(base, task.p0, 0);
            break;
        case PROBE_READ_PREFETCH_LINE:
            sum = probe_read_direct<true, 1>(base, task.p0, task.p2);
            break;
        case PROBE_READ_PREFETCH_BULK:
            sum = probe_read_direct<true, 2>(base, task.p0, task.p2);
            break;
        case PROBE_READ_SMEM_BULK:
            sum = probe_read_smem_bulk(base, task.p0, task.p2);
            break;
        default:
            __trap();
    }
    const uint32_t total = probe_block_sum(sum);
    if (threadIdx.x == 0) {
        reinterpret_cast<uint32_t*>(task.c)[blockIdx.x] = total;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// OP_PROBE_L2
// ---------------------------------------------------------------------------------------------------------------

// Block `b` reads the slice of `p0` bytes at `a + b * p0`, `p1` times, with loads that skip the L1 cache. The slices
// do not overlap, so the footprint is the grid width times `p0`. A footprint that fits in L2 reads from L2 after the
// first pass. c = `u32` checksum of each block: the wrapping sum of the words of its slice, times `p1`.
__device__ __forceinline__ void op_probe_l2(const MkTaskDescriptor& task) {
    const uint8_t* slice = reinterpret_cast<const uint8_t*>(task.a) + static_cast<uint64_t>(blockIdx.x) * task.p0;
    const uint32_t v4_count = task.p0 / 16;
    uint32_t sum = 0;
    for (uint32_t pass = 0; pass < task.p1; ++pass) {
        for (uint32_t first = 0; first < v4_count; first += blockDim.x * PROBE_UNROLL) {
            uint4 v[PROBE_UNROLL];
#pragma unroll
            for (uint32_t j = 0; j < PROBE_UNROLL; ++j) {
                const uint32_t index = first + j * blockDim.x + threadIdx.x;
                v[j] = make_uint4(0, 0, 0, 0);
                if (index < v4_count) {
                    v[j] = ld_global_nc_v4(slice + static_cast<uint64_t>(index) * 16);
                }
            }
#pragma unroll
            for (uint32_t j = 0; j < PROBE_UNROLL; ++j) {
                sum += sum_words(v[j]);
            }
        }
    }
    const uint32_t total = probe_block_sum(sum);
    if (threadIdx.x == 0) {
        reinterpret_cast<uint32_t*>(task.c)[blockIdx.x] = total;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// OP_PROBE_TIMER
// ---------------------------------------------------------------------------------------------------------------

// Thread 0 of block 0 reads `%globaltimer` `p0` times in a row. c = four `u64` words: the smallest nonzero step
// in ns (0 if the value never changed), the count of steps, the ns from the first read to the last change, and
// the count of reads.
__device__ __forceinline__ void op_probe_timer(const MkTaskDescriptor& task) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint64_t* out = reinterpret_cast<uint64_t*>(task.c);
    const uint64_t first = globaltimer_ns();
    uint64_t last = first;
    uint64_t smallest = 0;
    uint64_t steps = 0;
    for (uint32_t i = 0; i < task.p0; ++i) {
        const uint64_t now = globaltimer_ns();
        if (now != last) {
            const uint64_t step = now - last;
            smallest = (smallest == 0 || step < smallest) ? step : smallest;
            ++steps;
            last = now;
        }
    }
    out[0] = smallest;
    out[1] = steps;
    out[2] = last - first;
    out[3] = task.p0;
}

// ---------------------------------------------------------------------------------------------------------------
// OP_PROBE_MMA_TILE and OP_PROBE_MMA_RATE
// ---------------------------------------------------------------------------------------------------------------
//
// One MMA tile has 16 rows of A, 8 columns of B and 16 x 8 results. The operands sit in global memory in logical
// order, and each lane gathers its own fragment. For every kind, row `m` of A and column `n` of B have 32 bytes of
// K values: `A[m]` and `B[n]`, 16 `bf16`, 64 `e2m1` (low nibble first) or 32 `e4m3`. The scales are bytes: `s`
// holds the scales of the 16 rows of A, then those of the 8 columns of B, each with `scale_bytes` bytes (one for
// each 16 values of K for NVFP4, one for each 32 for MXFP8).
//
// Let `g` be lane / 4 and `t` be lane % 4. A lane holds these 32-bit words, and each word holds consecutive K values:
//   A: a0 = A[g] word t, a1 = A[g + 8] word t, a2 = A[g] word t + 4, a3 = A[g + 8] word t + 4
//   B: b0 = B[g] word t, b1 = B[g] word t + 4
// The scales go to the lanes with t = 0 (scale of B column g, and of A row g) and t = 1 (scale of A row g + 8).
// The result of a lane: d0, d1 = row g, columns 2t, 2t + 1; d2, d3 = row g + 8, the same columns.
// The tests confirm this layout: a wrong layout gives a wrong result.

struct MmaFragment {
    uint32_t a[4];
    uint32_t b[2];
    uint32_t scale_a;
    uint32_t scale_b;
};

__device__ __forceinline__ uint32_t load_word(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ MmaFragment mma_load_fragment(
    uint32_t kind, const uint8_t* a, const uint8_t* b, const uint8_t* scales) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t g = lane / 4;
    const uint32_t t = lane % 4;
    MmaFragment f;
    f.a[0] = load_word(a + g * 32 + t * 4);
    f.a[1] = load_word(a + (g + 8) * 32 + t * 4);
    f.a[2] = load_word(a + g * 32 + 16 + t * 4);
    f.a[3] = load_word(a + (g + 8) * 32 + 16 + t * 4);
    f.b[0] = load_word(b + g * 32 + t * 4);
    f.b[1] = load_word(b + g * 32 + 16 + t * 4);
    f.scale_a = 0;
    f.scale_b = 0;
    if (kind == PROBE_MMA_NVFP4) {
        // Four scales for each row: a word.
        if (t == 0) {
            f.scale_a = load_word(scales + g * 4);
            f.scale_b = load_word(scales + 16 * 4 + g * 4);
        } else if (t == 1) {
            f.scale_a = load_word(scales + (g + 8) * 4);
        }
    } else if (kind == PROBE_MMA_MXFP8) {
        // One scale for each row: a byte.
        if (t == 0) {
            f.scale_a = scales[g];
            f.scale_b = scales[16 + g];
        } else if (t == 1) {
            f.scale_a = scales[g + 8];
        }
    }
    return f;
}

template <uint32_t KIND>
__device__ __forceinline__ void mma_tile(float (&d)[4], const MmaFragment& f) {
    if constexpr (KIND == PROBE_MMA_BF16) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]));
    } else if constexpr (KIND == PROBE_MMA_NVFP4) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%12}, %11, {%12,%12};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]), "r"(f.scale_a),
              "r"(f.scale_b), "n"(0));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12,%12}, %11, {%12,%12};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]), "r"(f.scale_a),
              "r"(f.scale_b), "n"(0));
    }
}

// Warp 0 of block 0 runs one tile from zero. a, b = the operands, d = the scales, c = 128 `f32` words: the tile
// of results in row-major order (16 rows of 8). p1 = the kind (`PROBE_MMA_*`).
__device__ __forceinline__ void op_probe_mma_tile(const MkTaskDescriptor& task) {
    if (blockIdx.x != 0 || threadIdx.x >= 32) {
        return;
    }
    const MmaFragment f = mma_load_fragment(
        task.p1, reinterpret_cast<const uint8_t*>(task.a), reinterpret_cast<const uint8_t*>(task.b),
        reinterpret_cast<const uint8_t*>(task.d));
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    switch (task.p1) {
        case PROBE_MMA_BF16:
            mma_tile<PROBE_MMA_BF16>(d, f);
            break;
        case PROBE_MMA_NVFP4:
            mma_tile<PROBE_MMA_NVFP4>(d, f);
            break;
        case PROBE_MMA_MXFP8:
            mma_tile<PROBE_MMA_MXFP8>(d, f);
            break;
        default:
            __trap();
    }
    const uint32_t g = (threadIdx.x & 31) / 4;
    const uint32_t t = (threadIdx.x & 31) % 4;
    float* out = reinterpret_cast<float*>(task.c);
    out[g * 8 + 2 * t] = d[0];
    out[g * 8 + 2 * t + 1] = d[1];
    out[(g + 8) * 8 + 2 * t] = d[2];
    out[(g + 8) * 8 + 2 * t + 1] = d[3];
}

// Every warp of every block loads the tile (a, b, scales in d) and runs `p0` MMAs in each of `MK_PROBE_MMA_CHAINS`
// chains. A chain adds each result to its own accumulator, so the chains are independent and keep the tensor
// cores busy. c = the four accumulators of each chain for each thread, in the order of block, thread, chain, word.
template <uint32_t KIND>
__device__ __forceinline__ void probe_mma_rate(const MkTaskDescriptor& task) {
    const MmaFragment f = mma_load_fragment(
        KIND, reinterpret_cast<const uint8_t*>(task.a), reinterpret_cast<const uint8_t*>(task.b),
        reinterpret_cast<const uint8_t*>(task.d));
    float acc[MK_PROBE_MMA_CHAINS][4] = {};
    for (uint32_t i = 0; i < task.p0; ++i) {
#pragma unroll
        for (uint32_t chain = 0; chain < MK_PROBE_MMA_CHAINS; ++chain) {
            mma_tile<KIND>(acc[chain], f);
        }
    }
    float4* out = reinterpret_cast<float4*>(task.c) +
                  (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * MK_PROBE_MMA_CHAINS;
#pragma unroll
    for (uint32_t chain = 0; chain < MK_PROBE_MMA_CHAINS; ++chain) {
        out[chain] = make_float4(acc[chain][0], acc[chain][1], acc[chain][2], acc[chain][3]);
    }
}

__device__ __forceinline__ void op_probe_mma_rate(const MkTaskDescriptor& task) {
    if ((blockDim.x & 31) != 0) {
        __trap();  // An MMA needs whole warps.
    }
    switch (task.p1) {
        case PROBE_MMA_BF16:
            probe_mma_rate<PROBE_MMA_BF16>(task);
            break;
        case PROBE_MMA_NVFP4:
            probe_mma_rate<PROBE_MMA_NVFP4>(task);
            break;
        case PROBE_MMA_MXFP8:
            probe_mma_rate<PROBE_MMA_MXFP8>(task);
            break;
        default:
            __trap();
    }
}
