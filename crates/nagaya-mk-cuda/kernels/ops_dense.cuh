// The dense op of one Inkling 2 run: OP_PROJ. OP_MATMUL_BF16 is in matmul_bf16.cuh, and OP_OUT_MIX is in ops_out_mix.cuh.
//
// This file belongs to wave B lane D (dense). `run_op` in megakernel.cu calls each op.
//
// This header needs `mk_protocol.h`, `MkTaskDescriptor`, `MkTaskTable`, the worker functions of megakernel.cu and
// `matmul_bf16.cuh`.

#pragma once

// ---------------------------------------------------------------------------------------------------------------
// OP_PROJ: the joined projection of the attention input.
// ---------------------------------------------------------------------------------------------------------------
//
// One matmul of the joined matrix J (K = 2304, repacked as one `MatmulShape`) with a norm prologue, and an epilogue for
// each segment of the rows of J. The rows of J, in order, and the tiles of 16 rows that they make (see `src/proj.rs`):
//
//   taps_k    32 rows,   2 tiles   `weight_proj` of the key convolution. A full layer only. Head h has the taps 4h to 4h + 3.
//   k      1,024 rows,  64 tiles   `wk`. Head h (8 of them) is the tiles 8h to 8h + 7.
//   taps_v    32 rows,   2 tiles   `weight_proj` of the value convolution. A full layer only.
//   v      1,024 rows,  64 tiles   `wv`.
//   q      3,072 rows, 192 tiles   `wq`. Head h (24 of them) is the tiles 8h to 8h + 7.
//   rel      384 rows,  24 tiles   `wr`. Head h is one tile.
//   sink      32 rows,   2 tiles   `sink`. The first 24 rows are used.
//   gate   3,072 rows, 192 tiles   `gate_attn`.
//
// A sliding layer has 538 tiles (8,608 rows), and a full layer has 542 (8,672 rows). The items run in the order of the tiles,
// so the last tiles are the gate, whose epilogue is the cheapest: no head finisher is left at the end of the task.
//
// The norm. The q, rel, sink and gate rows read the input row `a` with the factor `e` and the weight in the slot
// MK_PROJ_TABLE_NORM_Q. The taps_k and k rows read `x_k`, `rf_k` and `w_k`, and the taps_v and v rows read `x_v`, `rf_v` and `w_v`. On a
// sliding layer the three inputs are one row. Each value is `w * (x * rf)`, the order of `ops::rms_norm`.
//
// The epilogue. The three K slices of a tile merge in slice order (the rule of OP_MATMUL_BF16). Then, by segment:
//   q      The merge writes the raw sums to the output. The item that merges the last tile of a head runs the finisher of
//          the head: for each row, the scale `q_scale`, then the L2 clip to 22.627417, then, on a full layer, the log
//          scale of the position `pos0 + row`. The finisher writes the row in place.
//   rel    One tile is one head, so the merge does it all: the scale `rel_scale`, the L2 clip to 8, the log scale.
//   sink   The merge writes the first 24 values of each row, `[rows][24]`.
//   gate   The merge writes the raw sums. OP_ATTN applies the sigmoid, once, where it multiplies the output.
//   taps   The merge writes the raw taps to the scratch, `[rows][32]` for the key and again for the value.
//   k, v   The merge writes the raw rows to the scratch, `[rows][1024]`. The item that merges the last tile of a head (the
//          taps tile too, on a full layer) runs the finisher: for each row, the short convolution with the
//          3 rows before it (the raw rows of the block, then the conv state), then the clip of the key to 22.627417 or the
//          rms of the value, written to the cache of the attention (see `cache_layout.cuh`): row `pos0 + row` of the head, modulo `layer` rows on a
//          sliding layer, in f32 (FLAG_F32), bf16 (FLAG_BF16) or e4m3 with a unit scale (no flag).
//          After the last row it pushes the last 3 raw rows into the conv state. A sliding layer has a fixed
//          convolution (`[1024][4]`, tap 3 on the current row, divisor sqrt(5)). A full layer has a dynamic one
//          (`[4][1024]` plus the 4 taps of the head, lag 0 on the current row, divisor 3).
//
// The descriptor: a = x_q, b = the repacked J, c = the output (below), d = the task table, e = rf_q, f = the scratch (below),
// n = rows of J, k = 2304, kt = 768, group_items = 3, item_bytes = 16 * kt * 2, items = (n / 16) * 3, rows = the count of
// rows (1 to 512), pos0 = the position of the first row, layer = the rows of a plane of the cache (a multiple of 64), scale_bytes = the
// floats of the scratch that the host gives (checked: the op traps when it needs more), flags = FLAG_NORM, on a full
// layer FLAG_FULL_LAYER, and FLAG_F32 or FLAG_BF16 for the cache. A cubin of one cache format traps on the flags of another.
//
// The output at `c`, in floats: q `[rows][3072]`, then rel `[rows][384]`, then sink `[rows][24]`, then gate `[rows][3072]`.
// The scratch at `f`, in floats: one counter for each tile, then 64 counters (24 heads of q, 8 of k, 8 of v, 24 spare), then
// the partials (`items * rows * 16`), then the raw keys `[rows][1024]`, the raw values `[rows][1024]`, the raw key taps
// `[rows][32]` and the raw value taps `[rows][32]`. The counters start at 0, and the merger sets each back to 0.
//
// The table slots are named in `src/protocol.rs` (`PROJ_TABLE_*`).

constexpr uint32_t PR_KV_TILES = 64;
constexpr uint32_t PR_Q_TILES = 192;
constexpr uint32_t PR_REL_TILES = 24;
constexpr uint32_t PR_SINK_TILES = 2;
constexpr uint32_t PR_GATE_TILES = 192;
constexpr uint32_t PR_HEAD_COUNTERS = 64;
constexpr uint32_t PR_Q_WIDTH = 3072;
constexpr uint32_t PR_REL_WIDTH = 384;
constexpr uint32_t PR_SINKS = 24;
constexpr uint32_t PR_KV_WIDTH = 1024;
constexpr uint32_t PR_TAPS = 32;
constexpr uint32_t PR_HEAD_DIM = 128;
constexpr float PR_QK_CLIP = 22.627417f;
constexpr float PR_REL_CLIP = 8.0f;
constexpr float PR_EPS = 1e-6f;

// The shared words after the merge flag: the flags of the heads that this item finishes, and the sums of the warps.
constexpr uint32_t PR_SMEM_FLAGS = MM_TREE_FLOATS * 4 + 16;

// The first tile of each segment of J. A tile below `k` holds the taps of the key, a tile from `tv` below `v` the taps of
// the value (a full layer only).
struct PrTiles {
    uint32_t tk, k, tv, v, q, rel, sink, gate, end;
};

__device__ __forceinline__ PrTiles pr_tiles(bool full) {
    PrTiles t;
    const uint32_t taps = full ? 2 : 0;
    t.tk = 0;
    t.k = t.tk + taps;
    t.tv = t.k + PR_KV_TILES;
    t.v = t.tv + taps;
    t.q = t.v + PR_KV_TILES;
    t.rel = t.q + PR_Q_TILES;
    t.sink = t.rel + PR_REL_TILES;
    t.gate = t.sink + PR_SINK_TILES;
    t.end = t.gate + PR_GATE_TILES;
    return t;
}

__device__ __forceinline__ float pr_bf16(const uint16_t* p, uint32_t i) {
    return __uint_as_float(static_cast<uint32_t>(__ldg(p + i)) << 16);
}

// The sum of a value over the 16 lanes of a half warp, as a tree. Every lane gets the same bits.
__device__ __forceinline__ float pr_sum16(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

// The log scale of the position `p` (`ops::log_scale`): 1 + 0.1 * ln(max((p + 1) / 8192, 1)).
__device__ __forceinline__ float pr_log_scale(uint32_t p) {
    const float ratio = fmaxf(static_cast<float>(p + 1) / 8192.0f, 1.0f);
    return 1.0f + 0.1f * logf(ratio);
}

// The sum of the squares over the 128 channels of a head, for the thread layout of the finishers: channel `tid % 128`,
// row `tid / 128` of a pair of rows. The 32 lanes of a warp join as a tree, and then the four warps of a row, in order.
// `slot` is 0 or 1 and alternates in the loop of the caller, so that one barrier for each pair of rows is enough. Every
// consumer calls this function.
__device__ __forceinline__ float pr_head_sum(float value, uint32_t slot) {
    float* red = reinterpret_cast<float*>(mk_smem);
    value += __shfl_xor_sync(0xffffffffu, value, 16);
    value += __shfl_xor_sync(0xffffffffu, value, 8);
    value += __shfl_xor_sync(0xffffffffu, value, 4);
    value += __shfl_xor_sync(0xffffffffu, value, 2);
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    const uint32_t warp = worker_id() >> 5;
    if ((worker_id() & 31) == 0) {
        red[slot * 8 + warp] = value;
    }
    worker_sync();
    const float* mine = red + slot * 8 + (warp >> 2) * 4;
    return (mine[0] + mine[1]) + (mine[2] + mine[3]);
}

// The places of the scratch (see the header comment): the raw rows of stream 0 (key) or 1 (value), and their taps.
__device__ __forceinline__ float* pr_raw(const MkTaskDescriptor& task, uint32_t stream) {
    const uint32_t tiles = task.n / MM_TILE;
    float* partials = reinterpret_cast<float*>(task.f) + tiles + PR_HEAD_COUNTERS;
    // The block form keeps every sum in the item, so it has no partials.
    const uint64_t partial_floats =
        (task.flags & FLAG_BLOCK_FORM) != 0 ? 0 : task.items * static_cast<uint64_t>(task.rows) * MM_TILE;
    return partials + partial_floats + stream * static_cast<uint64_t>(task.rows) * PR_KV_WIDTH;
}

__device__ __forceinline__ float* pr_raw_taps(const MkTaskDescriptor& task, uint32_t stream) {
    return pr_raw(task, 2) + stream * static_cast<uint64_t>(task.rows) * PR_TAPS;
}

// The finisher of one head of q: the scale, the L2 clip, and on a full layer the log scale, in place.
__device__ __forceinline__ void pr_finish_q(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t head,
                                            bool full) {
    float* q_out = reinterpret_cast<float*>(task.c);
    const uint16_t* scale = reinterpret_cast<const uint16_t*>(table->ptr[MK_PROJ_TABLE_Q_SCALE]);
    const uint32_t c = worker_id() & (PR_HEAD_DIM - 1);
    const uint32_t channel = head * PR_HEAD_DIM + c;
    const float s = pr_bf16(scale, channel);
    uint32_t slot = 0;
    for (uint32_t rb = 0; rb < task.rows; rb += 2, slot ^= 1) {
        const uint32_t r = rb + (worker_id() >> 7);
        const bool valid = r < task.rows;
        float v = 0.0f;
        if (valid) {
            v = __ldcg(q_out + static_cast<uint64_t>(r) * PR_Q_WIDTH + channel) * s;
        }
        const float total = pr_head_sum(v * v, slot);
        if (valid) {
            const float f = fminf(PR_QK_CLIP / (sqrtf(total) + 1e-6f), 1.0f);
            v = v * f;
            if (full) {
                v = v * pr_log_scale(task.pos0 + r);
            }
            q_out[static_cast<uint64_t>(r) * PR_Q_WIDTH + channel] = v;
        }
    }
    worker_sync();
}

// The finisher of one head of k or of v: the short convolution over the raw rows of the block and the conv state, then
// the clip of a key or the rms of a value, written to the slot rows. Then the last 3 raw rows go into the conv state.
__device__ __forceinline__ void pr_finish_kv(const MkTaskDescriptor& task, const MkTaskTable* table, bool key,
                                             uint32_t head, bool full) {
    const float* raw = pr_raw(task, key ? 0 : 1);
    const float* taps = pr_raw_taps(task, key ? 0 : 1);
    const uint16_t* conv_w = reinterpret_cast<const uint16_t*>(table->ptr[key ? MK_PROJ_TABLE_K_CONV : MK_PROJ_TABLE_V_CONV]);
    float* state = reinterpret_cast<float*>(table->ptr[key ? MK_PROJ_TABLE_K_STATE : MK_PROJ_TABLE_V_STATE]);
    uint8_t* out = reinterpret_cast<uint8_t*>(table->ptr[key ? MK_PROJ_TABLE_K_OUT : MK_PROJ_TABLE_V_OUT]);
    const uint32_t rows = task.rows;
    const uint32_t rsel = worker_id() >> 7;
    const uint32_t channel = head * PR_HEAD_DIM + (worker_id() & (PR_HEAD_DIM - 1));
    // The three rows before the block, oldest first.
    const float s0 = __ldcg(state + channel);
    const float s1 = __ldcg(state + PR_KV_WIDTH + channel);
    const float s2 = __ldcg(state + 2 * PR_KV_WIDTH + channel);
    float w[4];
    for (uint32_t i = 0; i < 4; ++i) {
        // A full layer stores `[lag][1024]`, and a sliding layer stores `[1024][tap]`.
        w[i] = pr_bf16(conv_w, full ? i * PR_KV_WIDTH + channel : channel * 4 + i);
    }
    uint32_t slot = 0;
    for (uint32_t rb = 0; rb < rows; rb += 2, slot ^= 1) {
        const uint32_t r = rb + rsel;
        const bool valid = r < rows;
        float result = 0.0f;
        if (valid) {
            const float x = __ldcg(raw + static_cast<uint64_t>(r) * PR_KV_WIDTH + channel);
            // The rows at the lags 1, 2 and 3: a row of the block, or a row of the state.
            float p[3];
            for (uint32_t lag = 1; lag <= 3; ++lag) {
                p[lag - 1] = r >= lag ? __ldcg(raw + static_cast<uint64_t>(r - lag) * PR_KV_WIDTH + channel)
                                      : (lag - r == 1 ? s2 : (lag - r == 2 ? s1 : s0));
            }
            float sum;
            if (full) {
                // Taps: lag 0 is the current row, and the tap of a lag is the static weight plus the tap of the head.
                const float* t = taps + static_cast<uint64_t>(r) * PR_TAPS + head * 4;
                sum = 0.0f;
                sum += x * (w[0] + __ldcg(t));
                sum += p[0] * (w[1] + __ldcg(t + 1));
                sum += p[1] * (w[2] + __ldcg(t + 2));
                sum += p[2] * (w[3] + __ldcg(t + 3));
                result = (sum + x) / __uint_as_float(0x40400000u);
            } else {
                // The tap 3 is the current row, and the tap 0 is the oldest.
                sum = w[0] * p[2] + w[1] * p[1] + w[2] * p[0] + w[3] * x;
                result = (sum + x) / __uint_as_float(0x400f1bbdu);
            }
        }
        const float total = pr_head_sum(result * result, slot);
        if (valid) {
            const float f = key ? fminf(PR_QK_CLIP / (sqrtf(total) + 1e-6f), 1.0f)
                                : 1.0f / sqrtf(total / static_cast<float>(PR_HEAD_DIM) + PR_EPS);
            // The row of the head in the plane of the cache: the layout of `cache_layout.cuh`, in the format of the task.
            const uint32_t row = cache_row_of(task.pos0 + r, !full, task.layer);
            cache_store(task.flags, out, head, row, task.layer, worker_id() & (PR_HEAD_DIM - 1), result * f);
        }
    }
    worker_sync();
    // The conv state after the block: the last 3 of the three old rows and the raw rows.
    if (rsel == 0) {
        for (uint32_t j = 0; j < 3; ++j) {
            const uint32_t index = rows + j;
            const float v = index < 3 ? (index == 0 ? s0 : (index == 1 ? s1 : s2))
                                      : __ldcg(raw + static_cast<uint64_t>(index - 3) * PR_KV_WIDTH + channel);
            state[j * PR_KV_WIDTH + channel] = v;
        }
    }
    worker_sync();
}

// The checks of the task. Every field that the op reads is checked, and the op traps when one is wrong.
__device__ __forceinline__ void pr_check(const MkTaskDescriptor& task, const MkTaskTable* table) {
    const bool full = (task.flags & FLAG_FULL_LAYER) != 0;
    const uint32_t rows = task.rows;
    const uint32_t tiles = task.n / MM_TILE;
    const uint64_t scratch_floats = static_cast<uint64_t>(tiles) + PR_HEAD_COUNTERS +
                                    task.items * static_cast<uint64_t>(rows) * MM_TILE +
                                    2 * static_cast<uint64_t>(rows) * PR_KV_WIDTH +
                                    2 * static_cast<uint64_t>(rows) * PR_TAPS;
    if (worker_count() != MK_WORKERS || table == nullptr || rows == 0 || rows > MM_MAX_ROWS || task.kt != 768 ||
        task.k != 2304 || task.group_items != 3 || tiles != pr_tiles(full).end || task.items != tiles * 3 ||
        task.item_bytes != MM_TILE * 768 * 2 || dynamic_smem_bytes() < MM_SMEM_BYTES ||
        (task.flags & ~(FLAG_NORM | FLAG_FULL_LAYER | FLAG_F32 | FLAG_BF16)) != 0 || !cache_flags_ok(task.flags) ||
        task.layer == 0 || task.layer % 64 != 0 || (full ? task.pos0 + rows > task.layer : rows > task.layer) ||
        (table->ptr[MK_PROJ_TABLE_K_OUT] & 15) != 0 || (table->ptr[MK_PROJ_TABLE_V_OUT] & 15) != 0 || (task.flags & FLAG_NORM) == 0 || (task.a & 7) != 0 ||
        (task.b & 15) != 0 || (task.c & 3) != 0 || (task.e & 3) != 0 || (task.f & 3) != 0 || task.a == 0 ||
        task.c == 0 || task.e == 0 || task.f == 0 || scratch_floats > task.scale_bytes) {
        __trap();
    }
    // The loop stays a loop: unrolled, it would keep the 15 pointers in registers for the whole op.
#pragma unroll 1
    for (uint32_t slot = 0; slot < 15; ++slot) {
        // The slots 0 to 14 are all pointers of the op, and the sliding layer shares them between q, k and v.
        const uint64_t p = table->ptr[slot];
        const bool is_row = slot == MK_PROJ_TABLE_X_K || slot == MK_PROJ_TABLE_X_V;
        if (p == 0 || (is_row && (p & 7) != 0) || (p & 3) != 0) {
            __trap();
        }
    }
}

// The merge of one tile and its epilogue. The partials of the three slices are written, and this item arrived last.
__device__ __forceinline__ void pr_merge(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t tile) {
    const bool full = (task.flags & FLAG_FULL_LAYER) != 0;
    const PrTiles t = pr_tiles(full);
    const uint32_t rows = task.rows;
    const uint32_t tiles = task.n / MM_TILE;
    const uint64_t partial_floats = static_cast<uint64_t>(rows) * MM_TILE;
    uint32_t* counters = reinterpret_cast<uint32_t*>(task.f);
    uint32_t* head_counters = counters + tiles;
    float* partials = reinterpret_cast<float*>(task.f) + tiles + PR_HEAD_COUNTERS;
    float* q_out = reinterpret_cast<float*>(task.c);
    float* rel_out = q_out + static_cast<uint64_t>(rows) * PR_Q_WIDTH;
    float* sink_out = rel_out + static_cast<uint64_t>(rows) * PR_REL_WIDTH;
    float* gate_out = sink_out + static_cast<uint64_t>(rows) * PR_SINKS;
    uint32_t* flags = reinterpret_cast<uint32_t*>(mk_smem + MM_TREE_FLOATS * 4);

    // The stream of a tile below `q`: 0 for the key, 1 for the value. `main` is its first tile of rows, and a tile below
    // it holds taps.
    const bool in_v = tile >= t.tv;
    const uint32_t stream = in_v ? 1 : 0;
    const uint32_t main_tile = in_v ? t.v : t.k;
    const uint32_t tap_tile = in_v ? t.tv : t.tk;
    float* raw = pr_raw(task, stream);
    float* raw_taps = pr_raw_taps(task, stream);

    // This item merges the tile. It writes the sums by segment.
    const uint64_t rounded = (partial_floats + MK_WORKERS - 1) / MK_WORKERS * MK_WORKERS;
    for (uint64_t i = worker_id(); i < rounded; i += MK_WORKERS) {
        const bool valid = i < partial_floats;
        const float sum = valid ? mm_merged_sum(partials, tile, 3, partial_floats, i) : 0.0f;
        const uint64_t row = i / MM_TILE;
        const uint32_t col = i % MM_TILE;
        if (tile < main_tile) {
            if (valid) {
                __stcg(raw_taps + row * PR_TAPS + (tile - tap_tile) * MM_TILE + col, sum);
            }
        } else if (tile < t.q) {
            if (valid) {
                __stcg(raw + row * PR_KV_WIDTH + (tile - main_tile) * MM_TILE + col, sum);
            }
        } else if (tile < t.rel) {
            if (valid) {
                __stcg(q_out + row * PR_Q_WIDTH + (tile - t.q) * MM_TILE + col, sum);
            }
        } else if (tile < t.sink) {
            const uint32_t head = tile - t.rel;
            float v = sum * pr_bf16(reinterpret_cast<const uint16_t*>(table->ptr[MK_PROJ_TABLE_REL_SCALE]), head * 16 + col);
            const float total = pr_sum16(v * v);
            const float f = fminf(PR_REL_CLIP / (sqrtf(total) + 1e-6f), 1.0f);
            v = v * f;
            if (full) {
                v = v * pr_log_scale(task.pos0 + static_cast<uint32_t>(row));
            }
            if (valid) {
                rel_out[row * PR_REL_WIDTH + head * 16 + col] = v;
            }
        } else if (tile < t.gate) {
            const uint32_t at = (tile - t.sink) * MM_TILE + col;
            if (valid && at < PR_SINKS) {
                sink_out[row * PR_SINKS + at] = sum;
            }
        } else if (valid) {
            gate_out[row * PR_Q_WIDTH + (tile - t.gate) * MM_TILE + col] = sum;
        }
    }
    if (tile >= t.rel) {
        return;
    }

    // The head finishers. The counter of a head counts its 8 tiles, and on a full layer the taps tile of its stream.
    // The counters of the heads: q 0 to 23, k 24 to 31, v 32 to 39.
    worker_sync();
    const uint32_t group = full ? 9 : 8;
    if (worker_id() == 0) {
        if (tile >= t.q) {
            flags[0] = merge_arrive(head_counters + (tile - t.q) / 8, 8) ? 1u : 0u;
        } else if (tile < main_tile) {
            // The taps tile `n` serves the 4 heads `4 n` to `4 n + 3` of its stream.
            for (uint32_t j = 0; j < 4; ++j) {
                flags[j] = merge_arrive(head_counters + 24 + 8 * stream + 4 * (tile - tap_tile) + j, group) ? 1u : 0u;
            }
        } else {
            flags[0] = merge_arrive(head_counters + 24 + 8 * stream + (tile - main_tile) / 8, group) ? 1u : 0u;
        }
    }
    worker_sync();
    if (tile >= t.q) {
        if (flags[0] != 0) {
            pr_finish_q(task, table, (tile - t.q) / 8, full);
        }
        return;
    }
    // A taps tile finishes up to 4 heads, and a tile of rows finishes one.
    const bool taps = tile < main_tile;
    for (uint32_t j = 0; j < (taps ? 4u : 1u); ++j) {
        if (flags[j] != 0) {
            pr_finish_kv(task, table, stream == 0, taps ? 4 * (tile - tap_tile) + j : (tile - main_tile) / 8, full);
        }
    }
}

__device__ __forceinline__ void op_proj(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    pr_check(task, table);
    const bool full = (task.flags & FLAG_FULL_LAYER) != 0;
    const PrTiles t = pr_tiles(full);
    const uint32_t tiles = task.n / MM_TILE;
    const uint32_t tile = item / 3;
    // The input of the tile: the taps_k and k rows read the second input, the taps_v and v rows the third, and the q, rel,
    // sink and gate rows the first.
    const bool in_v = tile >= t.tv && tile < t.q;
    const bool in_k = tile < t.tv;
    const float* x = reinterpret_cast<const float*>(in_v ? table->ptr[MK_PROJ_TABLE_X_V]
                                                         : (in_k ? table->ptr[MK_PROJ_TABLE_X_K] : task.a));
    const float* factors = reinterpret_cast<const float*>(in_v ? table->ptr[MK_PROJ_TABLE_RF_V]
                                                              : (in_k ? table->ptr[MK_PROJ_TABLE_RF_K] : task.e));
    const uint16_t* norm_w = reinterpret_cast<const uint16_t*>(
        table->ptr[in_v ? MK_PROJ_TABLE_NORM_V : (in_k ? MK_PROJ_TABLE_NORM_K : MK_PROJ_TABLE_NORM_Q)]);
    mm_item_partials(task, item, x, factors, norm_w,
                     reinterpret_cast<float*>(task.f) + tiles + PR_HEAD_COUNTERS);

    uint32_t* flags = reinterpret_cast<uint32_t*>(mk_smem + MM_TREE_FLOATS * 4);
    if (worker_id() == 0) {
        flags[0] = merge_arrive(reinterpret_cast<uint32_t*>(task.f) + tile, 3) ? 1u : 0u;
    }
    worker_sync();
    if (flags[0] != 0) {
        pr_merge(task, table, tile);
    }
}


// ---------------------------------------------------------------------------------------------------------------
// OP_PROJ in the block form (FLAG_BLOCK_FORM): a prefill of 17 to 512 rows.
// ---------------------------------------------------------------------------------------------------------------
//
// The items are groups of up to 8 tiles of one segment, times blocks of 32 rows (`matmul_block.cuh`), the group number
// first: item = group * row_blocks + row_block. A group never crosses a segment: taps_k 1 group of 2 tiles (full layer),
// k 8 groups, taps_v 1 (full), v 8, q 24, rel 3, sink 1 group of 2 tiles, gate 24. A group of q is one head, and a
// group of k or of v is one head. Each item has all of K, so it has the whole sum of its outputs: no partial, no tile
// counter. The epilogue gets the same sums as the decode form, so the bits are the same.
//
// The sums go to a staging buffer in shared memory, `[row][column]` of the block, and then:
//   q      the item runs the finisher of the head on its 32 rows. The finisher is the one of the decode form, and
//          no counter is needed because a head is one group.
//   rel    the epilogue of a tile of the decode form, with the thread layout of the decode form (16 lanes for a row).
//   sink   the first 24 columns of each row.   gate   the raw sums.
//   k, v   the raw rows and the raw taps go to the scratch, as in the decode form. The head finisher needs all the rows,
//          which are in several blocks of rows. A counter of the head counts the arrivals of the blocks of rows: the
//          count of row blocks for the group of the head, and on a full layer the same count again for the group of
//          the taps, which arrives on the 8 heads of its stream. The item that makes the last arrival runs
//          `pr_finish_kv` over all the rows.
//
// The descriptor is that of `OP_PROJ`, with FLAG_BLOCK_FORM and `items` = groups * row blocks. The scratch holds one
// word for each tile (not used), then the 64 counters of the heads, then the raw rows. It has no partials.

constexpr uint32_t PRB_PITCH = 129;  // floats in a row of the staging buffer; odd, so that a column is spread over the banks
constexpr uint32_t PRB_FLAGS_AT = 512;  // byte of the flags of the heads, in the shared memory before the ring

// The groups of tile in the order of the items. `tiles` is the count of tiles of the group, `first_tile` its first tile.
__device__ __forceinline__ uint32_t prb_groups(bool full) { return (full ? 2u : 0u) + 8 + 8 + 24 + 3 + 1 + 24; }

__device__ __forceinline__ void prb_group(bool full, uint32_t group, uint32_t& first_tile, uint32_t& tiles) {
    const PrTiles t = pr_tiles(full);
    const uint32_t taps = full ? 1 : 0;
    const uint32_t g_k = taps;
    const uint32_t g_tv = g_k + 8;
    const uint32_t g_v = g_tv + taps;
    const uint32_t g_q = g_v + 8;
    const uint32_t g_rel = g_q + 24;
    const uint32_t g_sink = g_rel + 3;
    const uint32_t g_gate = g_sink + 1;
    uint32_t group0;
    uint32_t tile0;
    uint32_t end;
    if (group < g_k) {
        group0 = 0, tile0 = t.tk, end = t.k;
    } else if (group < g_tv) {
        group0 = g_k, tile0 = t.k, end = t.tv;
    } else if (group < g_v) {
        group0 = g_tv, tile0 = t.tv, end = t.v;
    } else if (group < g_q) {
        group0 = g_v, tile0 = t.v, end = t.q;
    } else if (group < g_rel) {
        group0 = g_q, tile0 = t.q, end = t.rel;
    } else if (group < g_sink) {
        group0 = g_rel, tile0 = t.rel, end = t.sink;
    } else if (group < g_gate) {
        group0 = g_sink, tile0 = t.sink, end = t.gate;
    } else {
        group0 = g_gate, tile0 = t.gate, end = t.end;
    }
    first_tile = tile0 + (group - group0) * MB_GROUP_TILES;
    const uint32_t left = end - first_tile;
    tiles = left < MB_GROUP_TILES ? left : MB_GROUP_TILES;
}

// The checks of the task in the block form. Every field that the op reads is checked, and the op traps when one is wrong.
__device__ __forceinline__ void prb_check(const MkTaskDescriptor& task, const MkTaskTable* table) {
    const bool full = (task.flags & FLAG_FULL_LAYER) != 0;
    const uint32_t rows = task.rows;
    const uint32_t tiles = task.n / MM_TILE;
    const uint32_t row_blocks = (rows + MB_ROW_BLOCK - 1) / MB_ROW_BLOCK;
    const uint64_t scratch_floats = static_cast<uint64_t>(tiles) + PR_HEAD_COUNTERS +
                                    2 * static_cast<uint64_t>(rows) * PR_KV_WIDTH +
                                    2 * static_cast<uint64_t>(rows) * PR_TAPS;
    if (worker_count() != MK_WORKERS || table == nullptr || rows == 0 || rows > MM_MAX_ROWS || task.kt != 768 ||
        task.k != 2304 || task.group_items != 3 || tiles != pr_tiles(full).end ||
        task.items != prb_groups(full) * row_blocks || task.item_bytes != MM_TILE * 768 * 2 ||
        dynamic_smem_bytes() < MB_SMEM_BYTES ||
        (task.flags & ~(FLAG_NORM | FLAG_FULL_LAYER | FLAG_F32 | FLAG_BF16 | FLAG_BLOCK_FORM)) != 0 ||
        !cache_flags_ok(task.flags) || task.layer == 0 ||
        task.layer % 64 != 0 || (full ? task.pos0 + rows > task.layer : rows > task.layer) ||
        (table->ptr[MK_PROJ_TABLE_K_OUT] & 15) != 0 || (table->ptr[MK_PROJ_TABLE_V_OUT] & 15) != 0 ||
        (task.flags & FLAG_NORM) == 0 || (task.a & 15) != 0 || (task.b & 15) != 0 || (task.c & 3) != 0 ||
        (task.e & 3) != 0 || (task.f & 3) != 0 || task.a == 0 || task.c == 0 || task.e == 0 || task.f == 0 ||
        scratch_floats > task.scale_bytes) {
        __trap();
    }
    // The loop stays a loop: unrolled, it would keep the 15 pointers in registers for the whole op.
#pragma unroll 1
    for (uint32_t slot = 0; slot < 15; ++slot) {
        // The slots 0 to 14 are all pointers of the op, and the sliding layer shares them between q, k and v. The rows
        // of x are the source of bulk copies, so they need 16 bytes.
        const uint64_t p = table->ptr[slot];
        const bool is_row = slot == MK_PROJ_TABLE_X_K || slot == MK_PROJ_TABLE_X_V;
        if (p == 0 || (is_row && (p & 15) != 0) || (p & 3) != 0) {
            __trap();
        }
    }
}

// The finisher of one head of q on the rows of one block, from the staging buffer: the same expressions as `pr_finish_q`.
__device__ __forceinline__ void prb_finish_q(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t head,
                                             bool full, const float* staging, uint32_t row0, uint32_t rows_here) {
    float* q_out = reinterpret_cast<float*>(task.c);
    const uint16_t* scale = reinterpret_cast<const uint16_t*>(table->ptr[MK_PROJ_TABLE_Q_SCALE]);
    const uint32_t c = worker_id() & (PR_HEAD_DIM - 1);
    const uint32_t channel = head * PR_HEAD_DIM + c;
    const float s = pr_bf16(scale, channel);
    uint32_t slot = 0;
    for (uint32_t rb = 0; rb < rows_here; rb += 2, slot ^= 1) {
        const uint32_t r = rb + (worker_id() >> 7);
        const bool valid = r < rows_here;
        float v = 0.0f;
        if (valid) {
            v = staging[r * PRB_PITCH + c] * s;
        }
        const float total = pr_head_sum(v * v, slot);
        if (valid) {
            const float f = fminf(PR_QK_CLIP / (sqrtf(total) + 1e-6f), 1.0f);
            v = v * f;
            if (full) {
                v = v * pr_log_scale(task.pos0 + row0 + r);
            }
            q_out[static_cast<uint64_t>(row0 + r) * PR_Q_WIDTH + channel] = v;
        }
    }
    worker_sync();
}

// The geometry and the inputs of an item of `OP_PROJ` in the block form.
struct PrbInput {
    const float* x;
    const float* factors;
    const uint16_t* norm_w;
};

__device__ __forceinline__ PrbInput prb_input(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item,
                                              MbItem& it) {
    prb_check(task, table);
    const bool full = (task.flags & FLAG_FULL_LAYER) != 0;
    const PrTiles t = pr_tiles(full);
    const uint32_t rows = task.rows;
    const uint32_t row_blocks = (rows + MB_ROW_BLOCK - 1) / MB_ROW_BLOCK;
    const uint32_t group = item / row_blocks;
    prb_group(full, group, it.first_tile, it.tiles);
    it.row0 = (item % row_blocks) * MB_ROW_BLOCK;
    it.rows = rows - it.row0 < MB_ROW_BLOCK ? rows - it.row0 : MB_ROW_BLOCK;
    const uint32_t tile = it.first_tile;

    // The input of the group, as in the decode form: the k rows read the second input, the v rows the third, and q, rel,
    // sink and gate the first.
    const bool in_v = tile >= t.tv && tile < t.q;
    const bool in_k = tile < t.tv;
    PrbInput in;
    in.x = reinterpret_cast<const float*>(in_v ? table->ptr[MK_PROJ_TABLE_X_V]
                                               : (in_k ? table->ptr[MK_PROJ_TABLE_X_K] : task.a));
    in.factors = reinterpret_cast<const float*>(in_v ? table->ptr[MK_PROJ_TABLE_RF_V]
                                                     : (in_k ? table->ptr[MK_PROJ_TABLE_RF_K] : task.e));
    in.norm_w = reinterpret_cast<const uint16_t*>(
        table->ptr[in_v ? MK_PROJ_TABLE_NORM_V : (in_k ? MK_PROJ_TABLE_NORM_K : MK_PROJ_TABLE_NORM_Q)]);
    return in;
}

// The epilogue of an item of `OP_PROJ` in the block form: the sums are in shared memory (`MB_TOTAL_AT`).
__device__ __forceinline__ void prb_epilogue(const MkTaskDescriptor& task, const MkTaskTable* table, const MbItem& it) {
    const bool full = (task.flags & FLAG_FULL_LAYER) != 0;
    const PrTiles t = pr_tiles(full);
    const uint32_t rows = task.rows;
    const uint32_t row_blocks = (rows + MB_ROW_BLOCK - 1) / MB_ROW_BLOCK;
    const uint32_t tile = it.first_tile;

    // The staging buffer takes the stages of the ring, which are free. Each thread writes its 16 sums.
    float* staging = reinterpret_cast<float*>(mk_smem + MB_RING_AT);
    {
        const uint32_t id = worker_id();
        const uint32_t warp = id >> 5;
        const uint32_t lane = id & 31;
        const float* totals = reinterpret_cast<const float*>(mk_smem + MB_TOTAL_AT) + id;
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
            for (uint32_t r = 0; r < 4; ++r) {
                const uint32_t row = (warp >> 1) * 8 + 2 * (lane & 3) + (r & 1);
                const uint32_t col = ((warp & 1) * 4 + i) * MM_TILE + (lane >> 2) + 8 * (r >> 1);
                staging[row * PRB_PITCH + col] = totals[(i * 4 + r) * MK_CONSUMER_THREADS];
            }
        }
    }
    worker_sync();

    const uint32_t cols = it.tiles * MM_TILE;
    const uint32_t elems = it.rows * cols;
    float* q_out = reinterpret_cast<float*>(task.c);
    float* rel_out = q_out + static_cast<uint64_t>(rows) * PR_Q_WIDTH;
    float* sink_out = rel_out + static_cast<uint64_t>(rows) * PR_REL_WIDTH;
    float* gate_out = sink_out + static_cast<uint64_t>(rows) * PR_SINKS;
    if (tile >= t.q && tile < t.rel) {
        prb_finish_q(task, table, (tile - t.q) / MB_GROUP_TILES, full, staging, it.row0, it.rows);
        return;
    }
    if (tile >= t.gate) {
        for (uint32_t e = worker_id(); e < elems; e += MK_WORKERS) {
            const uint32_t row = e / cols;
            const uint32_t c = e % cols;
            gate_out[static_cast<uint64_t>(it.row0 + row) * PR_Q_WIDTH + (tile - t.gate) * MM_TILE + c] =
                staging[row * PRB_PITCH + c];
        }
        return;
    }
    if (tile >= t.sink) {
        for (uint32_t e = worker_id(); e < elems; e += MK_WORKERS) {
            const uint32_t row = e / cols;
            const uint32_t c = e % cols;
            if (c < PR_SINKS) {
                sink_out[static_cast<uint64_t>(it.row0 + row) * PR_SINKS + c] = staging[row * PRB_PITCH + c];
            }
        }
        return;
    }
    if (tile >= t.rel) {
        // One tile is one head. A row of a tile is 16 lanes of a half warp, so the sum over the 16 columns is a tree of
        // shuffles, as in the decode form. The loop covers a multiple of 256 elements so that every lane of a warp runs
        // the shuffles.
        const uint16_t* rel_scale = reinterpret_cast<const uint16_t*>(table->ptr[MK_PROJ_TABLE_REL_SCALE]);
        const uint32_t rounded = (elems + MK_WORKERS - 1) / MK_WORKERS * MK_WORKERS;
        for (uint32_t e = worker_id(); e < rounded; e += MK_WORKERS) {
            const bool valid = e < elems;
            const uint32_t col = e % MM_TILE;
            const uint32_t tile_in = (e / MM_TILE) % it.tiles;
            const uint32_t row = (e / MM_TILE) / it.tiles;
            const uint32_t head = it.first_tile + tile_in - t.rel;
            const float sum = valid ? staging[row * PRB_PITCH + tile_in * MM_TILE + col] : 0.0f;
            float v = sum * pr_bf16(rel_scale, head * 16 + col);
            const float total = pr_sum16(v * v);
            const float f = fminf(PR_REL_CLIP / (sqrtf(total) + 1e-6f), 1.0f);
            v = v * f;
            if (full) {
                v = v * pr_log_scale(task.pos0 + it.row0 + row);
            }
            if (valid) {
                rel_out[static_cast<uint64_t>(it.row0 + row) * PR_REL_WIDTH + head * 16 + col] = v;
            }
        }
        return;
    }

    // The taps and the rows of k and of v: raw to the scratch.
    const bool stream_v = tile >= t.tv;
    const uint32_t stream = stream_v ? 1 : 0;
    const uint32_t main_tile = stream_v ? t.v : t.k;
    const uint32_t tap_tile = stream_v ? t.tv : t.tk;
    const bool taps = tile < main_tile;
    float* dst = taps ? pr_raw_taps(task, stream) : pr_raw(task, stream);
    const uint32_t stride = taps ? PR_TAPS : PR_KV_WIDTH;
    const uint32_t offset = (tile - (taps ? tap_tile : main_tile)) * MM_TILE;
    for (uint32_t e = worker_id(); e < elems; e += MK_WORKERS) {
        const uint32_t row = e / cols;
        const uint32_t c = e % cols;
        __stcg(dst + static_cast<uint64_t>(it.row0 + row) * stride + offset + c, staging[row * PRB_PITCH + c]);
    }
    // The arrivals at the heads. The counters of the heads: q 0 to 23 (not used here), k 24 to 31, v 32 to 39.
    worker_sync();
    uint32_t* head_counters = reinterpret_cast<uint32_t*>(task.f) + task.n / MM_TILE;
    uint32_t* flags = reinterpret_cast<uint32_t*>(mk_smem + PRB_FLAGS_AT);
    const uint32_t arrivals = row_blocks * (full ? 2 : 1);
    if (worker_id() == 0) {
        if (taps) {
            // The taps group serves the 8 heads of its stream.
            for (uint32_t j = 0; j < 8; ++j) {
                flags[j] = merge_arrive(head_counters + 24 + 8 * stream + j, arrivals) ? 1u : 0u;
            }
        } else {
            flags[0] = merge_arrive(head_counters + 24 + 8 * stream + (tile - main_tile) / MB_GROUP_TILES, arrivals)
                           ? 1u
                           : 0u;
        }
    }
    worker_sync();
    for (uint32_t j = 0; j < (taps ? 8u : 1u); ++j) {
        if (flags[j] != 0) {
            pr_finish_kv(task, table, stream == 0, taps ? j : (tile - main_tile) / MB_GROUP_TILES, full);
        }
    }
}


// One item of the block form, for `OP_MATMUL_BF16` and for `OP_PROJ`. The kernel has one call of `mb_core`, so one copy of
// its code. The geometry of the item is computed again after the loop, so that no register holds it through the loop.
__device__ __forceinline__ void op_block_form(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    const bool proj = task.opcode == OP_PROJ;
    {
        MbItem it;
        PrbInput in;
        if (proj) {
            in = prb_input(task, table, item, it);
        } else {
            mb_matmul_check(task, table);
            it = mb_item_of(task, item);
            const bool norm = (task.flags & (FLAG_NORM | FLAG_NORM_UNWEIGHTED)) != 0;
            const bool weighted = (task.flags & FLAG_NORM) != 0;
            in.x = reinterpret_cast<const float*>(task.a);
            in.factors = norm ? reinterpret_cast<const float*>(task.e) : nullptr;
            in.norm_w = weighted ? reinterpret_cast<const uint16_t*>(table->ptr[MK_MATMUL_TABLE_NORM_WEIGHT]) : nullptr;
        }
        mb_core(task, it, in.x, in.factors, in.norm_w);
    }
    if (task.opcode == OP_PROJ) {
        MbItem it;
        prb_input(task, table, item, it);
        prb_epilogue(task, table, it);
    } else {
        mb_matmul_epilogue(task, mb_item_of(task, item));
    }
}
