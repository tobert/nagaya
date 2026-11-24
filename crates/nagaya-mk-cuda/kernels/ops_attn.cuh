// The attention op of one Inkling 2 run: OP_ATTN. This file belongs to wave B lane A (attention).
//
// The host form is `attend`, `relative_logits` and `relative_bias` in nagaya-mk-inkling2-bmg (`host/ops.rs`). The
// layout of the cache and the fields of the task are in `src/attn.rs`. The fields:
// - a: the queries `q` (f32, [rows][24][128]), after the clip and the log scale.
// - b, e: the key plane and the value plane of the run. Each is 8 heads of `k` rows of one position (see `src/attn.rs`).
// - c: the output (f32, [rows][24][128]): the attention times the sigmoid of the gate.
// - f: the scratch. The split mode has one counter for each merge group, then the chunk partials.
// - d: the task table: the relative states, the sinks, the gate and the relative projection (MK_ATTN_TABLE_*).
// - rows, pos0: the rows of the task are the positions pos0 .. pos0 + rows - 1 of one context.
// - n: the window of a sliding run (1,024), or 0 for a full run. k: the rows of a plane for each head. A sliding run
//   keeps position j in row j % k. A full run keeps it in row j.
// - kt: the keys of one chunk (MK_ATTN_CHUNK_KEYS). p0: the scale of a score, 1 / sqrt(128). group_items: the chunks.
// - flags: FLAG_F32 for an f32 cache, FLAG_BF16 for a bf16 cache, or no flag for e4m3 with a unit scale. scale_bytes: the bytes of one cache row.
//
// The cut. A row of the MMA is a pair (position, query head): row i of a tile of 16 is position i / 3 and the head
// 3 * kvh + i % 3 of its KV head. So the 3 query heads of a KV head share one read of the cache, and 16 rows are
// 16 / 3 positions. A work item is a chunk of keys, a KV head and a tile of rows. Eight warps run an item together:
//   1. Warp w takes the 16 channels 16 * w to 16 * w + 15. It computes its part of the scores of a step of keys on the
//      MMA, as S^T = Q K^T, and writes it to shared memory.
//   2. Each of the 256 threads takes one row and one key (or more). It adds the 8 parts in warp order, applies the
//      scale, the relative bias and the mask, and does the online softmax.
//   3. Warp w multiplies the probabilities by its 16 channels of the values on the MMA, and adds to its output.
// The score of a key reads each of the 128 channels in one place, in one fixed order. The softmax state and the output
// of a chunk start new in each chunk. A chunk gives a partial (max, sum, output). The partials of a (KV head, tile)
// merge in chunk order, from the state (sink, 1, 0), and a chunk with no key of a row (sum 0) is skipped.
//
// Two modes of the same arithmetic. The split mode (rows <= MK_ATTN_SPLIT_ROWS) runs one item for each chunk, and the
// item that arrives last in its group merges the partials of the scratch. The whole mode (a prefill block) runs one
// item for each (KV head, tile), and it merges each chunk in shared memory at once. Both start from the sink and take
// the chunks in the same order, so the bits are the same, and they do not depend on the rows or on the grid.
//
// The MMA. A score is a sum over 128 channels of products of f32 values. Each f32 is three bf16 terms (design rule 9).
// An f32 cache gives 6 MMAs for each product tile (the terms that are not below 2^-24), and an e4m3 or bf16 cache gives
// 3, because such a value is exact in bf16. The expansion of a cache value happens in registers.
//
// This header needs `matmul_bf16.cuh` (the MMA helpers) and the worker functions of `megakernel.cu`.

#pragma once

// The keys of one step of the loop: TPI tiles of 16.
template <uint32_t FMT>
constexpr uint32_t AtTilesKeys() { return 16; }

constexpr uint32_t AT_HEADS = 24;
constexpr uint32_t AT_KV_HEADS = 8;
constexpr uint32_t AT_GROUP = 3;
constexpr uint32_t AT_DIM = 128;
constexpr uint32_t AT_REL_STATES = 16;
constexpr uint32_t AT_EXTENT = 1024;
constexpr uint32_t AT_CHUNK = MK_ATTN_CHUNK_KEYS;
constexpr uint32_t AT_PARTIAL_ROW = MK_ATTN_PARTIAL_ROW_FLOATS;
// The words of the counters at the start of the scratch: a counter for each group of the most tiles of the split mode.
// The partials follow at this offset for every row count (`ATTN_COUNTER_WORDS` in src/attn.rs).
constexpr uint32_t AT_COUNTER_WORDS = (AT_KV_HEADS * ((AT_GROUP * MK_ATTN_SPLIT_ROWS + 15) / 16) + 3) & ~3u;
constexpr uint32_t AT_MAX_ROWS = 512;
// The offsets in floats of the shared memory. The part of the scores is sized for a step of up to 16 keys.
constexpr uint32_t AT_MAX_STEP = 16;
constexpr uint32_t AT_SM_PART = 0;                                  // [8 warps][16 rows][KT]
constexpr uint32_t AT_SM_P = AT_SM_PART + 8 * 16 * AT_MAX_STEP;     // [16 rows][KT + 4]
constexpr uint32_t AT_SM_ALPHA = AT_SM_P + 16 * (AT_MAX_STEP + 4);  // [16]
constexpr uint32_t AT_SM_REL = AT_SM_ALPHA + 16;        // [16 rows][16]
constexpr uint32_t AT_SM_OC = AT_SM_REL + 16 * 16;      // [16 rows][128]
constexpr uint32_t AT_SM_MC = AT_SM_OC + 16 * 128;      // [16]
constexpr uint32_t AT_SM_LC = AT_SM_MC + 16;            // [16]
constexpr uint32_t AT_SM_FLAG = AT_SM_LC + 16;          // one word
constexpr uint32_t AT_SM_RUN = AT_SM_FLAG + 4;          // [m, l, o][8][256]: the merged state of the whole mode
constexpr uint32_t AT_SM_QA = AT_SM_RUN + 3 * 8 * 256;  // [12][256]: the terms of the queries of each thread
constexpr uint32_t AT_SM_RLO = AT_SM_QA + 12 * 256;     // [l, o][8][256]: the low words of the merged state of the whole mode
static_assert((AT_SM_RLO + 2 * 8 * 256) * 4 <= MK_ATTN_SMEM_BYTES, "the shared memory of OP_ATTN is too small");
static_assert(AtTilesKeys<CF_F32>() <= AT_MAX_STEP && AtTilesKeys<CF_BF16>() <= AT_MAX_STEP, "a step has at most AT_MAX_STEP keys");
static_assert(AT_CHUNK % 64 == 0, "a chunk is a whole count of steps");

// Two e4m3 values (the low 16 bits of `two`, the first value in the low byte) to two f32 values. The expansion is exact.
__device__ __forceinline__ float2 at_e4m3x2(uint32_t two) {
    uint32_t halves;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(static_cast<unsigned short>(two)));
    float2 r;
    asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
        : "=f"(r.x), "=f"(r.y)
        : "r"(halves));
    return r;
}

// The terms of an operand of the MMA: A has 4 registers and B has 2, for each of the terms hi, mid and lo. An operand
// that is exact in bf16 has the term hi only.
struct AtA {
    uint4 h, m, l;
};
struct AtB {
    uint32_t h0, h1, m0, m1, l0, l1;
};

// The A operand of the MMA for rows g and g + 8, and the K slots 2t, 2t + 1 (`lo`) and 2t + 8, 2t + 9 (`hi`).
__device__ __forceinline__ AtA at_a_terms(float2 g_lo, float2 g8_lo, float2 g_hi, float2 g8_hi) {
    AtA a;
    const MmTerms t0 = mm_split(g_lo.x, g_lo.y);
    const MmTerms t1 = mm_split(g8_lo.x, g8_lo.y);
    const MmTerms t2 = mm_split(g_hi.x, g_hi.y);
    const MmTerms t3 = mm_split(g8_hi.x, g8_hi.y);
    a.h = make_uint4(t0.hi, t1.hi, t2.hi, t3.hi);
    a.m = make_uint4(t0.mid, t1.mid, t2.mid, t3.mid);
    a.l = make_uint4(t0.lo, t1.lo, t2.lo, t3.lo);
    return a;
}

// The B operand for the slots 2t, 2t + 1 (x0, x1) and 2t + 8, 2t + 9 (x2, x3) of one column.
template <uint32_t FMT>
__device__ __forceinline__ AtB at_b_terms(float x0, float x1, float x2, float x3) {
    AtB b;
    if (FMT == CF_F32) {
        const MmTerms t0 = mm_split(x0, x1);
        const MmTerms t1 = mm_split(x2, x3);
        b = {t0.hi, t1.hi, t0.mid, t1.mid, t0.lo, t1.lo};
    } else {
        b = {mm_pack_bf16(x0, x1), mm_pack_bf16(x2, x3), 0u, 0u, 0u, 0u};
    }
    return b;
}

// d += A B over the terms, the small terms first. The terms below 2^-24 of the product are left out.
template <uint32_t FMT>
__device__ __forceinline__ void at_mma(float (&d)[4], const AtA& a, const AtB& b) {
    if (FMT == CF_F32) {
        mm_mma(d, a.l, b.h0, b.h1);
        mm_mma(d, a.h, b.l0, b.l1);
        mm_mma(d, a.m, b.m0, b.m1);
        mm_mma(d, a.m, b.h0, b.h1);
        mm_mma(d, a.h, b.m0, b.m1);
        mm_mma(d, a.h, b.h0, b.h1);
    } else {
        mm_mma(d, a.l, b.h0, b.h1);
        mm_mma(d, a.m, b.h0, b.h1);
        mm_mma(d, a.h, b.h0, b.h1);
    }
}

// The registers of the cache values that one step of the loop loads. A step is TPI tiles of 16 keys. Lane (g, t) holds,
// for each tile, the 4 channels 16 w + 4 t .. + 3 of the keys g and 8 + g (`k`), and the channels 16 w + 2 g and + 1 of
// the keys 2t, 2t + 1, 2t + 8 and 2t + 9 (`v`). An f32 value fills a register, a bf16 value half of one, and an e4m3 value a byte.
template <uint32_t FMT>
struct AtTiles {
    static constexpr uint32_t TPI = 1;
    static constexpr uint32_t KEYS = AtTilesKeys<FMT>();
    uint4 k[TPI][2];
    uint2 v[TPI][4];
};

// Load the keys of the step whose first key is at position `key_base` and in row `row_base` of the plane. A key outside
// of [klo, khi] reads as zero: its row can hold no finite value, and a zero keeps the masked products exact.
template <uint32_t FMT>
__device__ __forceinline__ void at_load_k(
    AtTiles<FMT>& buf, const uint8_t* kp, uint32_t row_base, uint32_t key_base, uint32_t klo, uint32_t khi, uint32_t d0,
    uint32_t g, uint32_t t4) {
    constexpr uint32_t ROW = 128 * cache_value_bytes(FMT);
#pragma unroll
    for (uint32_t u = 0; u < AtTiles<FMT>::TPI; ++u) {
#pragma unroll
        for (uint32_t nt = 0; nt < 2; ++nt) {
            const uint32_t off = 16 * u + 8 * nt + g;
            const uint32_t j = key_base + off;
            uint4 raw = make_uint4(0, 0, 0, 0);
            if (j >= klo && j <= khi) {
                const uint8_t* p = kp + static_cast<uint64_t>(row_base + off) * ROW + (d0 + 4 * t4) * cache_value_bytes(FMT);
                if constexpr (FMT == CF_F32) {
                    const float4 x = __ldcg(reinterpret_cast<const float4*>(p));
                    raw = make_uint4(__float_as_uint(x.x), __float_as_uint(x.y), __float_as_uint(x.z), __float_as_uint(x.w));
                } else if constexpr (FMT == CF_BF16) {
                    const uint2 x = __ldcg(reinterpret_cast<const uint2*>(p));
                    raw.x = x.x;
                    raw.y = x.y;
                } else {
                    raw.x = __ldcg(reinterpret_cast<const unsigned int*>(p));
                }
            }
            buf.k[u][nt] = raw;
        }
    }
}

// The same for the values.
template <uint32_t FMT>
__device__ __forceinline__ void at_load_v(
    AtTiles<FMT>& buf, const uint8_t* vp, uint32_t row_base, uint32_t key_base, uint32_t klo, uint32_t khi, uint32_t d0,
    uint32_t g, uint32_t t4) {
    constexpr uint32_t ROW = 128 * cache_value_bytes(FMT);
#pragma unroll
    for (uint32_t u = 0; u < AtTiles<FMT>::TPI; ++u) {
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
            const uint32_t off = 16 * u + 2 * t4 + (q & 1) + 8 * (q >> 1);
            const uint32_t j = key_base + off;
            uint2 raw = make_uint2(0, 0);
            if (j >= klo && j <= khi) {
                const uint8_t* p = vp + static_cast<uint64_t>(row_base + off) * ROW + (d0 + 2 * g) * cache_value_bytes(FMT);
                if constexpr (FMT == CF_F32) {
                    const float2 x = __ldcg(reinterpret_cast<const float2*>(p));
                    raw = make_uint2(__float_as_uint(x.x), __float_as_uint(x.y));
                } else if constexpr (FMT == CF_BF16) {
                    raw.x = __ldcg(reinterpret_cast<const unsigned int*>(p));
                } else {
                    raw.x = __ldcg(reinterpret_cast<const unsigned short*>(p));
                }
            }
            buf.v[u][q] = raw;
        }
    }
}

// A bulk prefetch of `bytes` (a multiple of 16) into L2. It does not write memory, so it has no memory clobber, and it does
// not order the loads around it.
__device__ __forceinline__ void at_prefetch(const uint8_t* p, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes));
}

// d = d * a + t, where d is the double-float value h + l and t is the exact value of a product (its rounded value t and the
// error te of the rounding). The result is correct to about 2^-46 of the larger term. The intrinsics keep the compiler
// from fusing or reordering the steps: both modes of the op must give the same bits.
__device__ __forceinline__ void at_df_scale_add(float& h, float& l, float a, float t, float te) {
    const float ph = __fmul_rn(h, a);
    const float pe = __fmaf_rn(h, a, -ph);
    const float pl = __fadd_rn(__fmul_rn(l, a), pe);
    const float s = __fadd_rn(ph, t);
    const float bb = __fsub_rn(s, ph);
    const float err = __fadd_rn(__fsub_rn(ph, __fsub_rn(s, bb)), __fsub_rn(t, bb));
    const float lo = __fadd_rn(__fadd_rn(pl, err), te);
    const float hh = __fadd_rn(s, lo);
    l = __fsub_rn(lo, __fsub_rn(hh, s));
    h = hh;
}

// The merge of a chunk partial (mc, lc, oc) into the state (m, l, o) of one output value. The sum l and the output o are
// double-float values (high word, low word): a serial f32 sum of 2,048 chunks drifts with the depth (2.1e-5 of the row
// maximum at 524,288 positions). A chunk with no key of the row has the sum 0, and it is skipped. The order of the chunks is
// the order of the calls: the bits depend only on the chunk indices.
__device__ __forceinline__ void at_merge(
    float& m, float& lh, float& ll, float& oh, float& ol, float mc, float lc, float oc) {
    if (lc > 0.0f) {
        const float nm = fmaxf(m, mc);
        const float a = expf(m - nm);
        const float b = expf(mc - nm);
        const float tl = __fmul_rn(lc, b);
        at_df_scale_add(lh, ll, a, tl, __fmaf_rn(lc, b, -tl));
        const float to = __fmul_rn(oc, b);
        at_df_scale_add(oh, ol, a, to, __fmaf_rn(oc, b, -to));
        m = nm;
    }
}

// The state of the 8 output values of a thread of the merge: the values (row 2u + w / 128, channel w % 128), with
// u from 0 to 7 and w the worker. The sum l and the output o have a low word each.
struct AtRun {
    float m[8], l[8], ll[8], o[8], ol[8];
};

// The position, the query head and the liveness of row `row_in_tile` of the tile `mt`.
__device__ __forceinline__ void at_row_of(
    uint32_t mt, uint32_t row_in_tile, uint32_t rows, uint32_t kvh, bool& live, uint32_t& t, uint32_t& h) {
    const uint32_t row = 16 * mt + row_in_tile;
    live = row < AT_GROUP * rows;
    t = row / AT_GROUP;
    h = AT_GROUP * kvh + (row - t * AT_GROUP);
}

// The state of the merge at its start: the sink of each row, as one more score with the value 0.
__device__ __forceinline__ void at_run_init(AtRun& run, const float* sinks, uint32_t mt, uint32_t rows, uint32_t kvh) {
#pragma unroll
    for (uint32_t u = 0; u < 8; ++u) {
        bool live;
        uint32_t t, h;
        at_row_of(mt, 2 * u + (worker_id() >> 7), rows, kvh, live, t, h);
        run.m[u] = live ? __ldcg(sinks + static_cast<uint64_t>(t) * AT_HEADS + h) : 0.0f;
        run.l[u] = 1.0f;
        run.ll[u] = 0.0f;
        run.o[u] = 0.0f;
        run.ol[u] = 0.0f;
    }
}

template <uint32_t FMT>
__device__ __forceinline__ void op_attn_item(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
    using Tiles = AtTiles<FMT>;
    constexpr uint32_t TPI = Tiles::TPI;
    constexpr uint32_t KT = Tiles::KEYS;
    constexpr uint32_t KTP = KT + 4;
    constexpr uint32_t ROW = 128 * cache_value_bytes(FMT);

    const uint32_t rows = task.rows;
    const uint32_t pos0 = task.pos0;
    const uint32_t window = task.n;
    const uint32_t plane_rows = task.k;
    const uint32_t mtiles = (AT_GROUP * rows + 15) / 16;
    const bool split_mode = rows <= MK_ATTN_SPLIT_ROWS;
    const uint32_t lowest_block = window != 0 && pos0 + 1 > window ? pos0 + 1 - window : 0;
    const uint32_t c_first = lowest_block / AT_CHUNK;
    const uint32_t chunks = (pos0 + rows - 1) / AT_CHUNK - c_first + 1;

    float* sm = reinterpret_cast<float*>(mk_smem);
    const uint32_t wid = worker_id();
    const uint32_t warp = wid >> 5;
    const uint32_t lane = wid & 31;
    const uint32_t g = lane >> 2;
    const uint32_t t4 = lane & 3;
    const uint32_t d0 = 16 * warp;

    // The item: (chunk cc, KV head, tile mt). The tile is the fastest, so the blocks of one wave share the cache reads.
    const uint32_t mt = item % mtiles;
    const uint32_t kvh = (item / mtiles) % AT_KV_HEADS;
    const uint32_t cc_item = item / (mtiles * AT_KV_HEADS);
    const uint32_t group = kvh * mtiles + mt;
    const uint32_t live_rows = min(16u, AT_GROUP * rows - 16 * mt);

    // Each pointer below is read where it is used: a pointer that the loop does not use holds no register in the loop.
    const auto rel_ptr = [&]() { return reinterpret_cast<const float*>(table->ptr[MK_ATTN_TABLE_REL]); };
    const auto sinks_ptr = [&]() { return reinterpret_cast<const float*>(table->ptr[MK_ATTN_TABLE_SINKS]); };
    const auto gate_ptr = [&]() { return reinterpret_cast<const float*>(table->ptr[MK_ATTN_TABLE_GATE]); };
    const auto proj_ptr = [&]() { return reinterpret_cast<const unsigned short*>(table->ptr[MK_ATTN_TABLE_PROJ]); };
    const auto q_ptr = [&]() { return reinterpret_cast<const float*>(task.a); };
    const auto out_ptr = [&]() { return reinterpret_cast<float*>(task.c); };
    const uint8_t* kp = reinterpret_cast<const uint8_t*>(task.b) + static_cast<uint64_t>(kvh) * plane_rows * ROW;
    const uint8_t* vp = reinterpret_cast<const uint8_t*>(task.e) + static_cast<uint64_t>(kvh) * plane_rows * ROW;
    const auto counter_ptr = [&]() { return reinterpret_cast<uint32_t*>(task.f) + group; };
    const auto partials_ptr = [&]() { return reinterpret_cast<float*>(task.f) + AT_COUNTER_WORDS; };

    // The keys that the tile of rows reads: from the lowest key of its first row to its last position.
    const uint32_t r_first = 16 * mt;
    const uint32_t r_last = r_first + live_rows - 1;
    const uint32_t p_first = pos0 + r_first / AT_GROUP;
    const uint32_t p_last = pos0 + r_last / AT_GROUP;
    const uint32_t item_lo = window != 0 && p_first + 1 > window ? p_first + 1 - window : 0;
    const uint32_t item_hi = p_last;

    // The role of a thread in the softmax: row r, key kk of each tile.
    const uint32_t r = wid >> 4;
    const uint32_t kk = wid & 15;
    bool r_live;
    uint32_t r_t, r_h;
    at_row_of(mt, r, rows, kvh, r_live, r_t, r_h);
    const uint32_t r_pos = pos0 + r_t;
    const uint32_t r_lo = window != 0 && r_pos + 1 > window ? r_pos + 1 - window : 0;

    // The relative states of the 16 rows.
    sm[AT_SM_REL + wid] = r_live ? __ldcg(rel_ptr() + (static_cast<uint64_t>(r_t) * AT_HEADS + r_h) * AT_REL_STATES + kk) : 0.0f;

    // The A operand: the 16 channels of this warp of the queries of rows g and g + 8, as 3 bf16 terms.
    {
        float4 qv[2];
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
            bool live;
            uint32_t t, h;
            at_row_of(mt, g + 8 * half, rows, kvh, live, t, h);
            qv[half] = live ? __ldcg(reinterpret_cast<const float4*>(q_ptr() + (static_cast<uint64_t>(t) * AT_HEADS + h) * AT_DIM + d0 + 4 * t4))
                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        // The terms go to shared memory, and each step reads them again: a thread holds no register for them.
        const AtA qa = at_a_terms(make_float2(qv[0].x, qv[0].y), make_float2(qv[1].x, qv[1].y), make_float2(qv[0].z, qv[0].w),
                                  make_float2(qv[1].z, qv[1].w));
        uint4* slot = reinterpret_cast<uint4*>(sm + AT_SM_QA) + wid;
        slot[0] = qa.h;
        slot[256] = qa.m;
        slot[512] = qa.l;
    }
    worker_sync();

    // The whole mode keeps its merged state in shared memory, so that it holds no register across the loop.
    float* run_sm = sm + AT_SM_RUN;
    if (!split_mode) {
        AtRun init;
        at_run_init(init, sinks_ptr(), mt, rows, kvh);
#pragma unroll
        for (uint32_t u = 0; u < 8; ++u) {
            run_sm[(0 * 8 + u) * 256 + wid] = init.m[u];
            run_sm[(1 * 8 + u) * 256 + wid] = init.l[u];
            run_sm[(2 * 8 + u) * 256 + wid] = init.o[u];
            sm[AT_SM_RLO + (0 * 8 + u) * 256 + wid] = 0.0f;
            sm[AT_SM_RLO + (1 * 8 + u) * 256 + wid] = 0.0f;
        }
    }

    const uint32_t first_cc = split_mode ? cc_item : 0;
    const uint32_t last_cc = split_mode ? cc_item : chunks - 1;
    for (uint32_t cc = first_cc; cc <= last_cc; ++cc) {
        const uint32_t cs = (c_first + cc) * AT_CHUNK;
        const uint32_t lo = max(cs, item_lo);
        const uint32_t hi = min(cs + AT_CHUNK - 1, item_hi);
        float m_run = -INFINITY;
        float l_run = 0.0f;
        float o[2][4] = {};
        if (lo <= hi) {
            const uint32_t b0 = lo / KT;
            const uint32_t b1 = hi / KT;
            Tiles cur;
            const uint32_t base0 = b0 * KT;
            // The row of a block follows the row of the block before it. A sliding run wraps at the end of its ring.
            const auto next_row = [&](uint32_t row) {
                row += KT;
                return window != 0 && row >= plane_rows ? row - plane_rows : row;
            };
            uint32_t row_cur = window != 0 ? base0 % plane_rows : base0;
            uint32_t row_1 = next_row(row_cur);
            uint32_t row_2 = next_row(row_1);
            at_load_k<FMT>(cur, kp, row_cur, base0, item_lo, item_hi, d0, g, t4);
            at_load_v<FMT>(cur, vp, row_cur, base0, item_lo, item_hi, d0, g, t4);
            // The blocks after the next one go to L2 ahead of the loads, so the loads hit L2.
            if (wid == 0 && b0 < b1) {
                at_prefetch(kp + static_cast<uint64_t>(row_1) * ROW, KT * ROW);
                at_prefetch(vp + static_cast<uint64_t>(row_1) * ROW, KT * ROW);
            }
            for (uint32_t blk = b0; blk <= b1; ++blk) {
                const uint32_t key_base = blk * KT;
                if (wid == 0 && blk + 2 <= b1) {
                    at_prefetch(kp + static_cast<uint64_t>(row_2) * ROW, KT * ROW);
                    at_prefetch(vp + static_cast<uint64_t>(row_2) * ROW, KT * ROW);
                }
                // 1. The part of the scores of this warp: S^T[row][key] over its 16 channels.
#pragma unroll
                for (uint32_t u = 0; u < TPI; ++u) {
#pragma unroll
                    for (uint32_t nt = 0; nt < 2; ++nt) {
                        AtB kb;
                        {
                            float x0, x1, x2, x3;
                            if constexpr (FMT == CF_BF16) {
                                x0 = __uint_as_float(cur.k[u][nt].x << 16);
                                x1 = __uint_as_float(cur.k[u][nt].x & 0xFFFF0000u);
                                x2 = __uint_as_float(cur.k[u][nt].y << 16);
                                x3 = __uint_as_float(cur.k[u][nt].y & 0xFFFF0000u);
                            } else if constexpr (FMT == CF_F32) {
                                x0 = __uint_as_float(cur.k[u][nt].x);
                                x1 = __uint_as_float(cur.k[u][nt].y);
                                x2 = __uint_as_float(cur.k[u][nt].z);
                                x3 = __uint_as_float(cur.k[u][nt].w);
                            } else {
                                const float2 lo2 = at_e4m3x2(cur.k[u][nt].x);
                                const float2 hi2 = at_e4m3x2(cur.k[u][nt].x >> 16);
                                x0 = lo2.x;
                                x1 = lo2.y;
                                x2 = hi2.x;
                                x3 = hi2.y;
                            }
                            kb = at_b_terms<FMT>(x0, x1, x2, x3);
                        }
                        const uint4* qslot = reinterpret_cast<const uint4*>(sm + AT_SM_QA) + wid;
                        const AtA qa = {qslot[0], qslot[256], qslot[512]};
                        float s[4] = {};
                        at_mma<FMT>(s, qa, kb);
                        float* part = sm + AT_SM_PART + (warp * 16 + g) * KT + 16 * u + 8 * nt + 2 * t4;
                        *reinterpret_cast<float2*>(part) = make_float2(s[0], s[1]);
                        *reinterpret_cast<float2*>(part + 8 * KT) = make_float2(s[2], s[3]);
                    }
                }
                // The keys of the next step are on the way while this step does its softmax and its values.
                if (blk < b1) {
                    const uint32_t nb = key_base + KT;
                    at_load_k<FMT>(cur, kp, row_1, nb, item_lo, item_hi, d0, g, t4);
                }
                worker_sync();

                // 2. The scores of row r, keys kk + 16 u: the sum of the 8 parts, the scale, the bias, the mask, and the
                // online softmax over this step.
                float sc[TPI];
                bool ok[TPI];
                float tile_max = -INFINITY;
#pragma unroll
                for (uint32_t u = 0; u < TPI; ++u) {
                    const float* part = sm + AT_SM_PART + r * KT + 16 * u + kk;
                    const float p01 = part[0] + part[16 * KT];
                    const float p23 = part[32 * KT] + part[48 * KT];
                    const float p45 = part[64 * KT] + part[80 * KT];
                    const float p67 = part[96 * KT] + part[112 * KT];
                    const float sum = (p01 + p23) + (p45 + p67);
                    const uint32_t j = key_base + 16 * u + kk;
                    ok[u] = r_live && j >= r_lo && j <= r_pos;
                    float bias = 0.0f;
                    if (ok[u] && r_pos - j < AT_EXTENT) {
                        const uint32_t dist = r_pos - j;
                        float acc = 0.0f;
#pragma unroll
                        for (uint32_t dd = 0; dd < AT_REL_STATES; ++dd) {
                            const float pw = __uint_as_float(static_cast<uint32_t>(__ldg(proj_ptr() + dd * AT_EXTENT + dist)) << 16);
                            acc += sm[AT_SM_REL + r * AT_REL_STATES + dd] * pw;
                        }
                        bias = acc;
                    }
                    sc[u] = sum * task.p0 + bias;
                    tile_max = fmaxf(tile_max, ok[u] ? sc[u] : -INFINITY);
                }
#pragma unroll
                for (uint32_t w = 1; w < 16; w <<= 1) {
                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, w));
                }
                const float m_new = fmaxf(m_run, tile_max);
                const float alpha = m_new == -INFINITY ? 1.0f : expf(m_run - m_new);
                float lsum = 0.0f;
#pragma unroll
                for (uint32_t u = 0; u < TPI; ++u) {
                    const float p = ok[u] ? expf(sc[u] - m_new) : 0.0f;
                    sm[AT_SM_P + r * KTP + 16 * u + kk] = p;
                    lsum += p;
                }
#pragma unroll
                for (uint32_t w = 1; w < 16; w <<= 1) {
                    lsum += __shfl_xor_sync(0xffffffffu, lsum, w);
                }
                l_run = l_run * alpha + lsum;
                m_run = m_new;
                if (kk == 0) {
                    sm[AT_SM_ALPHA + r] = alpha;
                }
                worker_sync();

                // 3. The output of this warp: O[row][channel] = alpha O + P V over the 16 channels of the warp.
                const float alpha_g = sm[AT_SM_ALPHA + g];
                const float alpha_g8 = sm[AT_SM_ALPHA + g + 8];
#pragma unroll
                for (uint32_t j = 0; j < 2; ++j) {
                    o[j][0] *= alpha_g;
                    o[j][1] *= alpha_g;
                    o[j][2] *= alpha_g8;
                    o[j][3] *= alpha_g8;
                }
#pragma unroll
                for (uint32_t u = 0; u < TPI; ++u) {
                    const float* pr = sm + AT_SM_P + 16 * u + 2 * t4;
                    const AtA pa = at_a_terms(
                        *reinterpret_cast<const float2*>(pr + g * KTP), *reinterpret_cast<const float2*>(pr + (g + 8) * KTP),
                        *reinterpret_cast<const float2*>(pr + g * KTP + 8), *reinterpret_cast<const float2*>(pr + (g + 8) * KTP + 8));
                    {
                        float2 v[4];
#pragma unroll
                        for (uint32_t q = 0; q < 4; ++q) {
                            v[q] = FMT == CF_F32 ? make_float2(__uint_as_float(cur.v[u][q].x), __uint_as_float(cur.v[u][q].y))
                                 : (FMT == CF_BF16 ? make_float2(__uint_as_float(cur.v[u][q].x << 16), __uint_as_float(cur.v[u][q].x & 0xFFFF0000u))
                                                   : at_e4m3x2(cur.v[u][q].x));
                        }
#pragma unroll
                        for (uint32_t j = 0; j < 2; ++j) {
                            const AtB vb = at_b_terms<FMT>(j == 0 ? v[0].x : v[0].y, j == 0 ? v[1].x : v[1].y,
                                                           j == 0 ? v[2].x : v[2].y, j == 0 ? v[3].x : v[3].y);
                            at_mma<FMT>(o[j], pa, vb);
                        }
                    }
                }
                if (blk < b1) {
                    const uint32_t nb = key_base + KT;
                    at_load_v<FMT>(cur, vp, row_1, nb, item_lo, item_hi, d0, g, t4);
                }
                row_1 = row_2;
                row_2 = next_row(row_2);
            }
        }

        // The partial of the chunk: the maximum and the sum of each row, and the output of each channel. Channel
        // d0 + 4 t + 2 c + j of rows g and g + 8 is in o[j][c], o[j][2 + c].
        if (split_mode) {
            float* pr = partials_ptr() + (static_cast<uint64_t>(group) * chunks + cc) * 16 * AT_PARTIAL_ROW;
            if (kk == 0 && r < live_rows) {
                __stcg(pr + r * AT_PARTIAL_ROW, m_run);
                __stcg(pr + r * AT_PARTIAL_ROW + 1, l_run);
            }
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
                for (uint32_t c = 0; c < 2; ++c) {
                    const uint32_t d = d0 + 4 * t4 + 2 * c + j;
                    if (g < live_rows) {
                        __stcg(pr + g * AT_PARTIAL_ROW + 4 + d, o[j][c]);
                    }
                    if (g + 8 < live_rows) {
                        __stcg(pr + (g + 8) * AT_PARTIAL_ROW + 4 + d, o[j][2 + c]);
                    }
                }
            }
        } else if (lo <= hi) {
            // The whole mode merges the chunk in shared memory, in chunk order.
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
                for (uint32_t c = 0; c < 2; ++c) {
                    const uint32_t d = d0 + 4 * t4 + 2 * c + j;
                    sm[AT_SM_OC + g * 128 + d] = o[j][c];
                    sm[AT_SM_OC + (g + 8) * 128 + d] = o[j][2 + c];
                }
            }
            if (kk == 0) {
                sm[AT_SM_MC + r] = m_run;
                sm[AT_SM_LC + r] = l_run;
            }
            worker_sync();
#pragma unroll
            for (uint32_t u = 0; u < 8; ++u) {
                const uint32_t row = 2 * u + (wid >> 7);
                float m = run_sm[(0 * 8 + u) * 256 + wid];
                float l = run_sm[(1 * 8 + u) * 256 + wid];
                float ov = run_sm[(2 * 8 + u) * 256 + wid];
                float ll = sm[AT_SM_RLO + (0 * 8 + u) * 256 + wid];
                float ol = sm[AT_SM_RLO + (1 * 8 + u) * 256 + wid];
                at_merge(m, l, ll, ov, ol, sm[AT_SM_MC + row], sm[AT_SM_LC + row], sm[AT_SM_OC + row * 128 + (wid & 127)]);
                run_sm[(0 * 8 + u) * 256 + wid] = m;
                run_sm[(1 * 8 + u) * 256 + wid] = l;
                run_sm[(2 * 8 + u) * 256 + wid] = ov;
                sm[AT_SM_RLO + (0 * 8 + u) * 256 + wid] = ll;
                sm[AT_SM_RLO + (1 * 8 + u) * 256 + wid] = ol;
            }
            worker_sync();
        }
    }

    bool merge_here = !split_mode;
    AtRun run;
    if (!split_mode) {
#pragma unroll
        for (uint32_t u = 0; u < 8; ++u) {
            run.m[u] = run_sm[(0 * 8 + u) * 256 + wid];
            run.l[u] = run_sm[(1 * 8 + u) * 256 + wid];
            run.o[u] = run_sm[(2 * 8 + u) * 256 + wid];
            run.ll[u] = sm[AT_SM_RLO + (0 * 8 + u) * 256 + wid];
            run.ol[u] = sm[AT_SM_RLO + (1 * 8 + u) * 256 + wid];
        }
    }
    if (split_mode) {
        // The item that arrives last in its group merges the partials of all the chunks, in chunk order.
        worker_sync();
        uint32_t* flag = reinterpret_cast<uint32_t*>(sm + AT_SM_FLAG);
        if (wid == 0) {
            *flag = merge_arrive(counter_ptr(), chunks) ? 1u : 0u;
        }
        worker_sync();
        merge_here = *flag != 0;
        if (merge_here) {
            at_run_init(run, sinks_ptr(), mt, rows, kvh);
            for (uint32_t cc = 0; cc < chunks; ++cc) {
                const float* pr = partials_ptr() + (static_cast<uint64_t>(group) * chunks + cc) * 16 * AT_PARTIAL_ROW;
#pragma unroll
                for (uint32_t u = 0; u < 8; ++u) {
                    const uint32_t row = 2 * u + (wid >> 7);
                    if (row < live_rows) {
                        const float mc = __ldcg(pr + row * AT_PARTIAL_ROW);
                        const float lc = __ldcg(pr + row * AT_PARTIAL_ROW + 1);
                        if (lc > 0.0f) {
                            at_merge(run.m[u], run.l[u], run.ll[u], run.o[u], run.ol[u], mc, lc, __ldcg(pr + row * AT_PARTIAL_ROW + 4 + (wid & 127)));
                        }
                    }
                }
            }
        }
    }
    if (merge_here) {
        // The output: the normalized sum times the sigmoid of the gate.
#pragma unroll
        for (uint32_t u = 0; u < 8; ++u) {
            bool live;
            uint32_t t, h;
            at_row_of(mt, 2 * u + (wid >> 7), rows, kvh, live, t, h);
            if (live) {
                const uint64_t at = (static_cast<uint64_t>(t) * AT_HEADS + h) * AT_DIM + (wid & 127);
                const float gate = __ldcg(gate_ptr() + at);
                const float sig = 1.0f / (1.0f + expf(-gate));
                out_ptr()[at] = ((run.o[u] + run.ol[u]) / (run.l[u] + run.ll[u])) * sig;
            }
        }
    }
}

// Run work item `item` of OP_ATTN. Every field is checked: a shape that the op does not cover traps.
__device__ __forceinline__ void op_attn(const MkTaskDescriptor& task, const MkTaskTable* table, uint32_t item) {
#ifdef MK_CACHE_BF16
    constexpr uint32_t fmt = CF_BF16;
#else
    const uint32_t fmt = (task.flags & FLAG_F32) != 0 ? CF_F32 : CF_E4M3;
#endif
    const uint32_t rows = task.rows;
    if (worker_count() != MK_WORKERS || table == nullptr || rows == 0 || rows > AT_MAX_ROWS || task.kt != AT_CHUNK ||
        !cache_flags_ok(task.flags) || (task.flags & ~static_cast<uint32_t>(FLAG_F32 | FLAG_BF16)) != 0 ||
        task.scale_bytes != 128 * cache_value_bytes(fmt) || task.k == 0 ||
        task.k % 64 != 0 || dynamic_smem_bytes() < MK_ATTN_SMEM_BYTES || (task.a & 15) != 0 || (task.b & 15) != 0 ||
        (task.e & 15) != 0 || (task.c & 3) != 0 || (task.f & 15) != 0) {
        __trap();
    }
    const uint32_t window = task.n;
    // A full run keeps every position in its own row. A sliding run keeps a position in the row j % k, and the window
    // and the rows of the task must not share a row.
    if ((window == 0 && task.pos0 + rows > task.k) || (window != 0 && task.k < window + rows - 1 && task.pos0 + rows > task.k)) {
        __trap();
    }
    const uint32_t mtiles = (AT_GROUP * rows + 15) / 16;
    const uint32_t lowest_block = window != 0 && task.pos0 + 1 > window ? task.pos0 + 1 - window : 0;
    const uint32_t chunks = (task.pos0 + rows - 1) / AT_CHUNK - lowest_block / AT_CHUNK + 1;
    const uint32_t items = (rows <= MK_ATTN_SPLIT_ROWS ? chunks : 1) * AT_KV_HEADS * mtiles;
    if (task.items != items || task.group_items != chunks) {
        __trap();
    }
    // The shape of the dispatch is a tuned fact. ptxas spills a value of the interpreter when a cubin has one instance of
    // the op (or three), and it does not spill with this pair of instances (milestone 37 step 11). The check above traps
    // on the flags that the cubin does not hold, so the second instance of a fixed cubin never runs.
#ifdef MK_CACHE_BF16
    if ((task.flags & FLAG_BF16) != 0) {
        op_attn_item<CF_BF16>(task, table, item);
    } else {
        op_attn_item<CF_E4M3>(task, table, item);
    }
#else
    if (fmt == CF_F32) {
        op_attn_item<CF_F32>(task, table, item);
    } else {
        op_attn_item<CF_E4M3>(task, table, item);
    }
#endif
}
