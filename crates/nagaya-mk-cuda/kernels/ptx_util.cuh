// PTX helpers that the ops of the engine and the probe ops share: loads, prefetches, bulk copies, the barriers of
// memory (`mbarrier`) and the size of the dynamic shared memory. The engine ops use some of them, so this file is in every
// cubin, and `probe_ops.cuh` is in the probe cubins only.
//
// This header needs `mk_protocol.h` and `MkTaskDescriptor` from megakernel.cu.

#pragma once

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
