// The layout of the cache of one run, on the device: the address of a value of a key row or a value row.
//
// The host form is `row_of` and `row_offset` in `src/attn.rs`, and `tests/cuda_cache.rs` of nagaya-mk-inkling2-sm121
// checks that the device writes where the host reads. A cache has two planes (the keys and the values). A plane has
// one block of `plane_rows` rows for each KV head, and a row holds the 128 values of one position in one format: f32
// (512 bytes), bf16 (256 bytes) or e4m3 with a unit scale (128 bytes). A full run keeps position j in row j, and a sliding run keeps it
// in row j % plane_rows.
//
// This header needs only `stdint.h`.

#pragma once

constexpr uint32_t CL_HEAD_DIM = 128;

// The row of a position in a plane of `plane_rows` rows. `ring` is true for a sliding run.
__device__ __forceinline__ uint32_t cache_row_of(uint32_t position, bool ring, uint32_t plane_rows) {
    return ring ? position % plane_rows : position;
}

// The formats of the cache, and the cubin that holds each one.
//
// The default cubin holds the f32 and the e4m3 formats, and it picks one with the flags of the task. A cubin that is built
// with one of `MK_CACHE_F32`, `MK_CACHE_BF16` or `MK_CACHE_E4M3` holds that format only, and a task whose flags name
// another format traps. A cubin of one format has the code of that format only, so it needs no more registers than the
// default cubin.
constexpr uint32_t CF_F32 = 0;
constexpr uint32_t CF_E4M3 = 1;
constexpr uint32_t CF_BF16 = 2;

#if defined(MK_CACHE_F32)
#define MK_CACHE_FIXED CF_F32
#elif defined(MK_CACHE_BF16)
#define MK_CACHE_FIXED CF_BF16
#elif defined(MK_CACHE_E4M3)
#define MK_CACHE_FIXED CF_E4M3
#endif

// The flags of the format `fmt`.
__host__ __device__ constexpr uint32_t cache_flags_of(uint32_t fmt) { return fmt == CF_F32 ? FLAG_F32 : (fmt == CF_BF16 ? FLAG_BF16 : 0u); }

// The bytes of one value, and of one row of 128 values.
__host__ __device__ constexpr uint32_t cache_value_bytes(uint32_t fmt) { return fmt == CF_F32 ? 4u : (fmt == CF_BF16 ? 2u : 1u); }

// True when the cubin can run a task with these flags: the flags name a format that the cubin holds.
__device__ __forceinline__ bool cache_flags_ok(uint32_t flags) {
    const uint32_t cache = flags & (FLAG_F32 | FLAG_BF16);
#ifdef MK_CACHE_FIXED
    return cache == cache_flags_of(MK_CACHE_FIXED);
#else
    return (cache & FLAG_BF16) == 0;
#endif
}

// The byte offset in a plane of channel `channel` of row `row` of KV head `kvh`.
template <uint32_t FMT>
__device__ __forceinline__ uint64_t cache_value_offset(uint32_t kvh, uint32_t row, uint32_t plane_rows, uint32_t channel) {
    return ((static_cast<uint64_t>(kvh) * plane_rows + row) * CL_HEAD_DIM + channel) * cache_value_bytes(FMT);
}

// Write one value of a cache row, in the format `FMT`. An e4m3 value is nearest even and saturated at 448.
template <uint32_t FMT>
__device__ __forceinline__ void cache_store_value(uint8_t* plane, uint32_t kvh, uint32_t row, uint32_t plane_rows,
                                                  uint32_t channel, float value) {
    uint8_t* at = plane + cache_value_offset<FMT>(kvh, row, plane_rows, channel);
    if constexpr (FMT == CF_F32) {
        *reinterpret_cast<float*>(at) = value;
    } else if constexpr (FMT == CF_BF16) {
        uint16_t half;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(half) : "f"(value));
        *reinterpret_cast<uint16_t*>(at) = half;
    } else {
        *at = static_cast<uint8_t>(cvt_e4m3(value));
    }
}

// The same, with the format that the task names. A cubin of one format has no branch.
__device__ __forceinline__ void cache_store(uint32_t flags, uint8_t* plane, uint32_t kvh, uint32_t row, uint32_t plane_rows,
                                            uint32_t channel, float value) {
#ifdef MK_CACHE_FIXED
    cache_store_value<MK_CACHE_FIXED>(plane, kvh, row, plane_rows, channel, value);
#else
    if ((flags & FLAG_F32) != 0) {
        cache_store_value<CF_F32>(plane, kvh, row, plane_rows, channel, value);
    } else {
        cache_store_value<CF_E4M3>(plane, kvh, row, plane_rows, channel, value);
    }
#endif
}
