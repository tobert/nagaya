// OP_PROBE_SMEM: the rate of shared memory, and the rate of MMAs that read their fragments from it. This is the
// probe of the prefill form of the dense matmul (`docs/engine-abi.md`): a stage of that form moves tens of kilobytes
// through shared memory for each ring of MMAs. The op needs `probe_ops.cuh`, `mk_smem` and the worker functions.
//
// The op fills the whole dynamic shared memory with the word `PROBE_SMEM_WORD`, which is 1.0 in both halves as bf16.
// Then it runs `n` iterations of the variant `k` and returns a checksum in `c[block]`. Every load of a lane in order
// is conflict free: lane `l` reads the bytes `l * size` to `l * size + size - 1` of a row of the area.

#pragma once

constexpr uint32_t PROBE_SMEM_WORD = 0x3F803F80u;
constexpr uint32_t PROBE_SMEM_ROWS = 8;  // rows of one iteration of a load or store variant

__device__ __forceinline__ void st_shared_v4(uint32_t addr, uint32_t value) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(addr), "r"(value) : "memory");
}

__device__ __forceinline__ void st_shared_v2(uint32_t addr, uint32_t value) {
    asm volatile("st.shared.v2.u32 [%0], {%1,%1};" ::"r"(addr), "r"(value) : "memory");
}

__device__ __forceinline__ uint2 ld_shared_v2(uint32_t addr) {
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr));
    return v;
}

__device__ __forceinline__ void probe_mma_bf16(float (&d)[4], const uint4& a, const uint2& b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

// One warp tile of the prefill form: 2 tiles of 16 columns (A) and 2 tiles of 8 rows (B), so 4 chains. A step runs
// the terms lo, mid and hi on each chain: 12 MMAs. With `FED` the fragments come from shared memory: 2 A vectors of
// 16 bytes and 6 B vectors of 8 bytes for each lane, 2.5 KB for each warp. Without it they stay in registers.
template <bool FED>
__device__ __forceinline__ uint32_t probe_smem_mma(uint32_t smem, uint32_t steps, uint32_t iterations) {
    const uint32_t lane = worker_id() & 31;
    float acc[2][2][4] = {};
    uint4 a[2] = {make_uint4(PROBE_SMEM_WORD, PROBE_SMEM_WORD, PROBE_SMEM_WORD, PROBE_SMEM_WORD),
                  make_uint4(PROBE_SMEM_WORD, PROBE_SMEM_WORD, PROBE_SMEM_WORD, PROBE_SMEM_WORD)};
    uint2 b[2][3];
#pragma unroll
    for (uint32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
        for (uint32_t term = 0; term < 3; ++term) {
            b[ni][term] = make_uint2(PROBE_SMEM_WORD, PROBE_SMEM_WORD);
        }
    }
    for (uint32_t it = 0; it < iterations; ++it) {
        for (uint32_t s = 0; s < steps; ++s) {
            if (FED) {
                // Eight rotating positions: 1 KB of A and 1.5 KB of B each.
                const uint32_t position = s & 7;
                const uint32_t a_at = smem + position * 1024 + lane * 16;
                const uint32_t b_at = smem + 8192 + position * 1536 + lane * 8;
                a[0] = ld_shared_v4(a_at);
                a[1] = ld_shared_v4(a_at + 512);
#pragma unroll
                for (uint32_t ni = 0; ni < 2; ++ni) {
#pragma unroll
                    for (uint32_t term = 0; term < 3; ++term) {
                        b[ni][term] = ld_shared_v2(b_at + (ni * 3 + term) * 256);
                    }
                }
            }
#pragma unroll
            for (uint32_t term = 0; term < 3; ++term) {
#pragma unroll
                for (uint32_t mi = 0; mi < 2; ++mi) {
#pragma unroll
                    for (uint32_t ni = 0; ni < 2; ++ni) {
                        probe_mma_bf16(acc[mi][ni], a[mi], b[ni][term]);
                    }
                }
            }
        }
    }
    float total = 0.0f;
#pragma unroll
    for (uint32_t mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (uint32_t ni = 0; ni < 2; ++ni) {
            total += (acc[mi][ni][0] + acc[mi][ni][1]) + (acc[mi][ni][2] + acc[mi][ni][3]);
        }
    }
    return __float_as_uint(total);
}

// a = unused, c = `u32` checksum of each block, n = iterations, k = the variant (`PROBE_SMEM_*`), kt = k16 steps of
// each iteration of the MMA variants. The checksums: a load variant sums every word that it reads. A store variant
// stores the pattern, and its checksum is the sum of the words of the area after the stores. An MMA variant gives the
// sum of its 16 accumulators of one thread, which is exact (each product is 1, and a term adds 16 to a chain),
// summed over the threads of the block.
__device__ __forceinline__ void op_probe_smem(const MkTaskDescriptor& task) {
    const uint32_t bytes = dynamic_smem_bytes() & ~15u;
    if (bytes < 32768 || (worker_count() & 31) != 0 || task.n == 0) {
        __trap();  // The launch gave too little shared memory.
    }
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(mk_smem));
    // Fill the whole area, also the place where an earlier op left a barrier.
    for (uint32_t at = worker_id() * 16; at < bytes; at += worker_count() * 16) {
        st_shared_v4(smem + at, PROBE_SMEM_WORD);
    }
    worker_sync();
    uint32_t sum = 0;
    const uint32_t id = worker_id();
    switch (task.k) {
        case PROBE_SMEM_LD_V4:
            for (uint32_t i = 0; i < task.n; ++i) {
#pragma unroll
                for (uint32_t row = 0; row < PROBE_SMEM_ROWS; ++row) {
                    sum += sum_words(ld_shared_v4(smem + row * 4096 + id * 16));
                }
            }
            break;
        case PROBE_SMEM_LD_V2:
            for (uint32_t i = 0; i < task.n; ++i) {
#pragma unroll
                for (uint32_t row = 0; row < PROBE_SMEM_ROWS; ++row) {
                    const uint2 v = ld_shared_v2(smem + row * 2048 + id * 8);
                    sum += v.x + v.y;
                }
            }
            break;
        case PROBE_SMEM_ST_V4:
            for (uint32_t i = 0; i < task.n; ++i) {
#pragma unroll
                for (uint32_t row = 0; row < PROBE_SMEM_ROWS; ++row) {
                    st_shared_v4(smem + row * 4096 + id * 16, PROBE_SMEM_WORD);
                }
            }
            break;
        case PROBE_SMEM_ST_V2:
            for (uint32_t i = 0; i < task.n; ++i) {
#pragma unroll
                for (uint32_t row = 0; row < PROBE_SMEM_ROWS; ++row) {
                    st_shared_v2(smem + row * 2048 + id * 8, PROBE_SMEM_WORD);
                }
            }
            break;
        case PROBE_SMEM_MMA_FED:
            sum = probe_smem_mma<true>(smem, task.kt, task.n);
            break;
        case PROBE_SMEM_MMA_REGS:
            sum = probe_smem_mma<false>(smem, task.kt, task.n);
            break;
        default:
            __trap();
    }
    if (task.k == PROBE_SMEM_ST_V4 || task.k == PROBE_SMEM_ST_V2) {
        worker_sync();
        for (uint32_t at = worker_id() * 16; at < 32768; at += worker_count() * 16) {
            sum += sum_words(ld_shared_v4(smem + at));
        }
    }
    const uint32_t total = probe_block_sum(sum);
    if (worker_id() == 0) {
        reinterpret_cast<uint32_t*>(task.c)[blockIdx.x] = total;
    }
}
