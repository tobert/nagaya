// The layout of the cache of one run, on the device: the address of a value of a key row or a value row.
//
// The host form is `row_of` and `row_offset` in `src/attn.rs`, and `tests/cuda_cache.rs` of nagaya-mk-inkling2-sm121
// checks that the device writes where the host reads. A cache has two planes (the keys and the values). A plane has
// one block of `plane_rows` rows for each KV head, and a row holds the 128 values of one position in one format: f32
// (512 bytes) or e4m3 with a unit scale (128 bytes). A full run keeps position j in row j, and a sliding run keeps it
// in row j % plane_rows.
//
// This header needs only `stdint.h`.

#pragma once

constexpr uint32_t CL_HEAD_DIM = 128;

// The row of a position in a plane of `plane_rows` rows. `ring` is true for a sliding run.
__device__ __forceinline__ uint32_t cache_row_of(uint32_t position, bool ring, uint32_t plane_rows) {
    return ring ? position % plane_rows : position;
}

// The byte offset in a plane of channel `channel` of row `row` of KV head `kvh`.
__device__ __forceinline__ uint64_t cache_value_offset(uint32_t kvh, uint32_t row, uint32_t plane_rows, uint32_t channel,
                                                       bool f32) {
    return ((static_cast<uint64_t>(kvh) * plane_rows + row) * CL_HEAD_DIM + channel) * (f32 ? 4u : 1u);
}
