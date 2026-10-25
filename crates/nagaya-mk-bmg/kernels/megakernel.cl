// mkviolet: per-card task interpreter for Gemma 4 26B-A4B and EmbeddingGemma 2 on Arc Pro B70.
// Compiled offline by build.rs: ocloc compile -file kernels/megakernel.cl -device bmg-g31 -options "-cl-std=CL3.0"
//
// Launched as 32 work-groups (one per Xe2 core) of 1024 work-items (64 subgroups of 16). Every
// work-group walks the whole task ring. Before a task runs, each work-group waits until
// scoreboard[wait_event] reaches the work-group count; after it, each work-group adds 1 to
// scoreboard[signal_event]. Most ops give each subgroup whole rows, so they need no
// work-group barrier inside.
//
// Activations are f32, weights BF16. Every op handles a block of `n` tokens, whose K and V go
// to cache rows p0..p0+n. Without a token table token t sits at position p0 + t. With one
// (kernel argument 3, one uint4 per token: position, then a row range [y, z) the token must
// not attend to), the block can hold several continuations of one prefix: each continuation's
// tokens keep the positions they would have alone and skip the rows of the ones before it.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Opcodes (must match src/protocol.rs)
#define OP_NOP              0
#define OP_RMSNORM          1
#define OP_MATMUL           2
#define OP_GEGLU            3
#define OP_ATTN             5
#define OP_ROUTER           6
#define OP_MOE_PLAN         7
#define OP_MOE_GATE_UP      8
#define OP_MOE_DOWN         9
#define OP_MEAN_POOL_L2     11
#define OP_ROUTER_LOGITS    12
#define OP_ATTN_OUT         13
#define OP_FFN_OUT          14
#define OP_ATTN_COMBINE     15
#define OP_ATTN_SCORES      16
#define OP_ATTN_SOFTMAX     17
#define OP_ATTN_PV          18
#define OP_ATTN_PV_SUM      19
#define OP_MOE_GATHER       20
#define OP_COPY             21
#define OP_HALT             255

// Flags (must match src/protocol.rs)
#define FLAG_ACCUMULATE     1u   // dst += result
#define FLAG_SOFTCAP_30     4u   // result = 30 * tanh(result / 30)
#define FLAG_ROPE_GLOBAL    8u   // theta 1e6, proportional RoPE with partial_rotary_factor 0.25
#define FLAG_ROPE           32u  // OP_RMSNORM on head rows: rotate each normalized row; see there

#define NO_EVENT            0xFFFFFFFFu
#define NUM_EXPERTS         128
#define TOP_K               8
#define XMX_TOKENS          32  // tokens per XMX work item: four dpas tiles of 8 rows
#define XMX_TOKENS2         16  // with two weight rows per lane, two tiles
#define MAX_PLAN_TOKENS     1024
#define SPLIT_K             4   // subgroups sharing one row's K in one-token matmuls
#define SPLIT_K_MAX_ROWS    65536   // taller one-token matmuls (the vocabulary head) keep whole rows
#define KSPLIT              8   // subgroups sharing one XMX tile's K in matmuls of at most 8 tokens
#define PLAN_SEGMENTS       8   // OP_MOE_PLAN splits the picks into this many runs per expert
#define XKC                 64  // K columns of x a work-group stages in local memory per step
#define ATTN_CHUNK          32  // keys per work item in one-token attention
#define ATTN_TILE           16  // query tokens per work item in multi-token attention
#define KV_RING_SPILL       1024 // rows past a KV ring a block may write before OP_COPY moves them

typedef struct __attribute__((aligned(64))) {
    ushort opcode;
    ushort flags;
    uint   layer;
    uint   wait_event;
    uint   signal_event;
    ulong  a;
    ulong  b;
    ulong  c;
    uint   m;
    uint   n;
    uint   k;
    uint   p0;
    ulong  x;
} MkTaskDescriptor;

typedef struct __attribute__((aligned(64))) {
    uint  top_k_indices[8];
    float top_k_weights[8];   // renormalized top-8 router probabilities * per_expert_scale
    float router_entropy;     // entropy of the renormalized top-8 distribution, nats
    float router_lse;         // log-sum-exp of all 128 router logits
    uint  layer_idx;
    uint  token;
    uint  reserved[4];
} MoERouteDecision;

// Written by OP_MOE_PLAN: the block's (token, pick) slots grouped by expert.
typedef struct {
    uint  count[NUM_EXPERTS];
    uint  offset[NUM_EXPERTS + 1];
    uint  n_active;                 // experts with at least one entry, listed in active[]
    uint  active[NUM_EXPERTS];
    uint  entry_slot[MAX_PLAN_TOKENS * TOP_K];   // t * TOP_K + pick
    float entry_weight[MAX_PLAN_TOKENS * TOP_K];
} MoEPlan;

typedef struct __attribute__((aligned(64))) {
    uint core_id;
    uint tasks_executed;
    uint spin_polls;
    uint last_opcode;
    uint xmx_tiles_executed;
    uint reserved[11];
} XeCoreStats;

inline uint usm_event_load(__global atomic_uint* counter) {
    return atomic_load_explicit(counter, memory_order_acquire, memory_scope_device);
}

inline void usm_event_signal(__global atomic_uint* counter, uint delta) {
    atomic_fetch_add_explicit(counter, delta, memory_order_acq_rel, memory_scope_device);
}

inline float bf16_to_f32(ushort b) {
    return as_float(((uint)b) << 16);
}

inline float4 bf16x4_to_f32(ushort4 w) {
    return (float4)(bf16_to_f32(w.s0), bf16_to_f32(w.s1), bf16_to_f32(w.s2), bf16_to_f32(w.s3));
}

// gelu_pytorch_tanh
inline float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f; // sqrt(2 / pi)
    const float k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanh(k0 * (x + k1 * x * x * x)));
}

// Part [lo, hi) of 8-wide steps of one or two weight rows (w1 may be null) against one x row,
// reduced over the subgroup.
inline float2 part_dots(__global const ushort* w0, __global const ushort* w1, __global const float* x,
                        uint lo, uint hi, uint lane) {
    float a0 = 0.0f, a1 = 0.0f;
    for (uint k8 = lo + lane; k8 < hi; k8 += 16) {
        ushort8 u0 = vload8(k8, w0);
        float8 xv = vload8(k8, x);
        a0 += dot(bf16x4_to_f32(u0.lo), xv.lo) + dot(bf16x4_to_f32(u0.hi), xv.hi);
        if (w1) {
            ushort8 u1 = vload8(k8, w1);
            a1 += dot(bf16x4_to_f32(u1.lo), xv.lo) + dot(bf16x4_to_f32(u1.hi), xv.hi);
        }
    }
    return (float2)(sub_group_reduce_add(a0), sub_group_reduce_add(a1));
}

// x as three BF16 parts, hi + mid + lo, which hold its 24 significant bits: (hi, mid, lo, 0)
// packed low to high.
inline ulong bf16_split3(float x) {
    uint u = as_uint(x) & 0xFFFF0000u;
    float x1 = x - as_float(u);
    uint v = as_uint(x1) & 0xFFFF0000u;
    uint w = as_uint(x1 - as_float(v));
    w += 0x7FFFu + ((w >> 16) & 1u);
    return (ulong)(u >> 16) | ((ulong)(v >> 16) << 16) | ((ulong)(w >> 16) << 32);
}

// XMX dot products for multi-token matmuls over columns [k_lo, k_hi): 16 weight rows (one per
// lane: w0, and w1 when NB == 2, point at this lane's row) against consecutive rows t0.. of X (n
// rows of k floats), in mts tiles of 8. acc0[mt] element r is token t0 + 8 mt + r against this
// lane's row. One 2D block read brings 8 rows by 16 columns of x; rows past n read as zero. Each
// x splits into three BF16 parts (bf16_split3), each with its own dpas, so the sum keeps f32
// precision; the weights are BF16 already, and row-major rows are the VNNI layout dpas takes
// for W^T.
#define XMX_ROWS_TILES(NB, MTS) \
    for (uint k0 = k_lo; k0 < k_hi; k0 += 16) { \
        int8 b0 = as_int8(vload8(0, (__global const uint*)(w0 + k0))); \
        int8 b1 = NB == 2 ? as_int8(vload8(0, (__global const uint*)(w1 + k0))) : b0; \
        _Pragma("unroll") for (uint mt = 0; mt < MTS; mt++) { \
            uint xv[8]; \
            intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)X, K * 4, n, K * 4, (int2)(k0, t0 + 8 * mt), xv); \
            short8 hi, mid, lo; \
            _Pragma("unroll") for (uint r = 0; r < 8; r++) { \
                ulong p = bf16_split3(as_float(xv[r])); \
                hi[r] = (short)p; \
                mid[r] = (short)(p >> 16); \
                lo[r] = (short)(p >> 32); \
            } \
            acc0[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(lo, b0, acc0[mt]); \
            acc0[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(mid, b0, acc0[mt]); \
            acc0[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(hi, b0, acc0[mt]); \
            if (NB == 2) { \
                acc1[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(lo, b1, acc1[mt]); \
                acc1[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(mid, b1, acc1[mt]); \
                acc1[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(hi, b1, acc1[mt]); \
            } \
        } \
    }

inline void xmx_rows_1(__global const ushort* w0, __global const float* X, uint t0, uint n, uint K, uint mts,
                       uint k_lo, uint k_hi, float8* acc0) {
    __global const ushort* w1 = w0;
    float8* acc1 = acc0;
    for (uint mt = 0; mt < 4; mt++) acc0[mt] = 0.0f;
    if (mts == 1) { XMX_ROWS_TILES(1, 1) }
    else          { XMX_ROWS_TILES(1, 4) }
}

inline void xmx_rows_2(__global const ushort* w0, __global const ushort* w1, __global const float* X, uint t0, uint n, uint K,
                       uint mts, uint k_lo, uint k_hi, float8* acc0, float8* acc1) {
    for (uint mt = 0; mt < 2; mt++) { acc0[mt] = 0.0f; acc1[mt] = 0.0f; }
    if (mts == 1) { XMX_ROWS_TILES(2, 1) }
    else          { XMX_ROWS_TILES(2, 2) }
}

// A work-group's tile of the dense matmuls: x rows t0.. (n rows of k floats, up to XMX_TOKENS of
// them) against 16 weight rows per subgroup (w0: this lane's row; none when w0 is 0, but the
// subgroup still joins the barriers). Each XKC-column step stages x in local memory once,
// already split into three BF16 parts in dpas A order (xs[part][k16][mt][r][lane]), so the
// 64 subgroups share one load and one split.
inline void xmx_tile_1(__global const ushort* w0, __global const float* X, uint t0, uint n, uint K, uint mts,
                       __local ushort* xs, uint lid, uint lsize, uint lane, float8* acc) {
    for (uint mt = 0; mt < 4; mt++) acc[mt] = 0.0f;
    for (uint kc = 0; kc < K; kc += XKC) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = lid; i < XMX_TOKENS * XKC; i += lsize) {
            uint t = i / XKC, k = i % XKC;
            float v = (t0 + t < n && kc + k < K) ? X[(ulong)(t0 + t) * K + kc + k] : 0.0f;
            ulong p = bf16_split3(v);
            uint at = ((k / 16) * 4 + t / 8) * 128 + (t % 8) * 16 + k % 16;
            xs[at] = (ushort)p;
            xs[XMX_TOKENS * XKC + at] = (ushort)(p >> 16);
            xs[2 * XMX_TOKENS * XKC + at] = (ushort)(p >> 32);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (w0 == 0) continue;
        for (uint ks = 0; ks < XKC / 16 && kc + 16 * ks < K; ks++) {
            int8 b0 = as_int8(vload8(0, (__global const uint*)(w0 + kc + 16 * ks)));
            for (uint mt = 0; mt < mts; mt++) {
                __local const ushort* a = xs + (ks * 4 + mt) * 128;
                short8 hi = as_short8(intel_sub_group_block_read_us8(a));
                short8 mid = as_short8(intel_sub_group_block_read_us8(a + XMX_TOKENS * XKC));
                short8 lo = as_short8(intel_sub_group_block_read_us8(a + 2 * XMX_TOKENS * XKC));
                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(lo, b0, acc[mt]);
                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(mid, b0, acc[mt]);
                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(hi, b0, acc[mt]);
            }
        }
    }
}

// A KV cache row is a key position, or with a ring of `ring` rows (sliding layers of a long
// slot) the position modulo the ring. Attention ops carry `ring` in the task's `layer` field.
inline ulong kv_row(uint j, uint ring) {
    return ring ? (ulong)(j % ring) : (ulong)j;
}

// Rows of one KV plane: the value plane starts this many rows after the key plane.
inline ulong kv_plane_rows(ulong max_seq, uint ring) {
    return ring ? (ulong)ring + KV_RING_SPILL : max_seq;
}

// The surface height for a 2D block read of key rows: every row of a ring once the keys pass
// it, else the keys so far, past which a read returns zeros.
inline uint kv_surface_rows(uint key_rows, uint ring) {
    return (ring && key_rows > ring) ? ring : key_rows;
}

// One-token attention over keys [k_lo, k_hi] for G query heads h0.. sharing kv head kvh, with
// DPL = head_dim / 16 dims per lane. Each key's score is a subgroup reduction, so lanes read K
// and V rows together. Writes each head's running max, sum, and unnormalized output to
// part + h * part_stride: [max, sum, o[D]].
#define DEFINE_ATTN_CHUNK(DPL, G) \
inline void attn_chunk_##DPL##_##G(__global const float* Q, __global const float* Kc, __global const float* Vc, \
                                   __global float* part, ulong part_stride, uint h0, uint kvh, uint D, uint KV, \
                                   uint k_lo, uint k_hi, uint ring, uint lane) { \
    float q[G][DPL], o[G][DPL], run_max[G], run_sum[G]; \
    for (uint g = 0; g < G; g++) { \
        for (uint dd = 0; dd < DPL; dd++) { q[g][dd] = Q[(ulong)(h0 + g) * D + lane + 16 * dd]; o[g][dd] = 0.0f; } \
        run_max[g] = -INFINITY; run_sum[g] = 0.0f; \
    } \
    for (uint j0 = k_lo; j0 <= k_hi; j0 += 16) { \
        uint jn = min(16u, k_hi + 1 - j0); \
        float score[G]; \
        for (uint g = 0; g < G; g++) score[g] = -INFINITY; \
        for (uint jj = 0; jj < 16; jj++) { \
            if (jj >= jn) break; \
            __global const float* kr = Kc + (kv_row(j0 + jj, ring) * KV + kvh) * D; \
            float kv[DPL]; \
            for (uint dd = 0; dd < DPL; dd++) kv[dd] = kr[lane + 16 * dd]; \
            for (uint g = 0; g < G; g++) { \
                float sg = 0.0f; \
                for (uint dd = 0; dd < DPL; dd++) sg += q[g][dd] * kv[dd]; \
                sg = sub_group_reduce_add(sg); \
                if (lane == jj) score[g] = sg; \
            } \
        } \
        for (uint g = 0; g < G; g++) { \
            float new_max = fmax(run_max[g], sub_group_reduce_max(score[g])); \
            float pr = exp(score[g] - new_max); \
            float corr = exp(run_max[g] - new_max); \
            run_sum[g] = run_sum[g] * corr + sub_group_reduce_add(pr); \
            for (uint dd = 0; dd < DPL; dd++) o[g][dd] *= corr; \
            run_max[g] = new_max; \
            score[g] = pr; \
        } \
        for (uint jj = 0; jj < 16; jj++) { \
            if (jj >= jn) break; \
            __global const float* vr = Vc + (kv_row(j0 + jj, ring) * KV + kvh) * D; \
            float vv[DPL]; \
            for (uint dd = 0; dd < DPL; dd++) vv[dd] = vr[lane + 16 * dd]; \
            for (uint g = 0; g < G; g++) { \
                float pj = sub_group_broadcast(score[g], jj); \
                for (uint dd = 0; dd < DPL; dd++) o[g][dd] += pj * vv[dd]; \
            } \
        } \
    } \
    for (uint g = 0; g < G; g++) { \
        __global float* pp = part + (ulong)(h0 + g) * part_stride; \
        if (lane == 0) { pp[0] = run_max[g]; pp[1] = run_sum[g]; } \
        for (uint dd = 0; dd < DPL; dd++) pp[2 + lane + 16 * dd] = o[g][dd]; \
    } \
}

DEFINE_ATTN_CHUNK(16, 2)
DEFINE_ATTN_CHUNK(32, 1)

// The causal key range of the one-token attention at position p0 and its chunk count.
inline uint attn_keys(uint p0, uint window, uint* lo) {
    *lo = (window > 0 && p0 + 1 > window) ? p0 + 1 - window : 0;
    return (p0 - *lo) / ATTN_CHUNK + 1;
}

// Multi-token attention runs as three ops over a score buffer S[head][token][key row] with rows
// of attn_row_floats floats: OP_ATTN_SCORES writes q . k, OP_ATTN_SOFTMAX turns each row into
// probabilities, OP_ATTN_PV sums them against V. All three walk the same key tiles: for the
// ATTN_TILE tokens from t0, the 16-aligned tiles from the lowest key any of them reads to the
// highest. Keys a token must not read hold probability 0.

// Floats per score row for a block of n tokens at cache rows p0..: room for the last key tile.
inline uint attn_row_floats(uint p0, uint n) {
    return (p0 + n + 15) / 16 * 16 + 16;
}

// Key rows [lo, hi] token t reads, less its gap [gap_lo, gap_hi). Rows before a token's gap
// sit at their own positions, so its sliding window starts at a row.
inline uint4 attn_token_keys(uint t, uint p0, uint window, __global const uint4* token_meta) {
    uint row = p0 + t;
    uint pos = token_meta ? token_meta[t].x : row;
    uint lo = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
    return (uint4)(lo, row, token_meta ? token_meta[t].y : 0, token_meta ? token_meta[t].z : 0);
}

// The 16-aligned key tiles [first, end) the tokens t0.. of a tile read; lane l handles t0 + l.
inline uint2 attn_tile_keys(uint t0, uint n, uint p0, uint window, __global const uint4* token_meta, uint lane) {
    uint t = t0 + lane;
    uint4 r = attn_token_keys(min(t, n - 1), p0, window, token_meta);
    uint lo = sub_group_reduce_min(t < n ? r.x : 0xFFFFFFFFu);
    uint hi = sub_group_reduce_max(t < n ? r.y : 0u);
    return (uint2)(lo / 16 * 16, hi / 16 * 16 + 16);
}

// Splits 16 floats into three BF16 parts (bf16_split3) packed in pairs, the dpas B layout for
// 16 values along K: element 2j in the low half of int j.
inline void split3_pairs(const uint* f, int8* hi, int8* mid, int8* lo) {
    _Pragma("unroll") for (uint j = 0; j < 8; j++) {
        ulong a = bf16_split3(as_float(f[2 * j])), b = bf16_split3(as_float(f[2 * j + 1]));
        (*hi)[j]  = (int)((uint)(a & 0xFFFFu) | (uint)(b & 0xFFFFu) << 16);
        (*mid)[j] = (int)((uint)((a >> 16) & 0xFFFFu) | (uint)((b >> 16) & 0xFFFFu) << 16);
        (*lo)[j]  = (int)((uint)((a >> 32) & 0xFFFFu) | (uint)((b >> 32) & 0xFFFFu) << 16);
    }
}

// Splits 8 floats, one per A row, into three BF16 parts.
inline void split3_rows(const uint* f, short8* hi, short8* mid, short8* lo) {
    _Pragma("unroll") for (uint r = 0; r < 8; r++) {
        ulong p = bf16_split3(as_float(f[r]));
        (*hi)[r] = (short)p;
        (*mid)[r] = (short)(p >> 16);
        (*lo)[r] = (short)(p >> 32);
    }
}

// acc += A . B with both operands in three parts, dropping the products below 2^-24 of the
// whole (mid * lo, lo * mid, lo * lo); smallest terms first.
inline float8 dpas_split6(short8 ah, short8 am, short8 al, int8 bh, int8 bm, int8 bl, float8 acc) {
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(al, bh, acc);
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(ah, bl, acc);
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(am, bm, acc);
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(am, bh, acc);
    acc = intel_sub_group_bf16_bf16_matrix_mad_k16(ah, bm, acc);
    return intel_sub_group_bf16_bf16_matrix_mad_k16(ah, bh, acc);
}

// Sum of v over the work-group, every work-item's partial added in subgroup order.
inline float wg_sum(float v, __local float* slm, uint lane, uint sg_id, uint num_sgs) {
    v = sub_group_reduce_add(v);
    if (lane == 0) slm[sg_id] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    float total = 0.0f;
    for (uint sg = 0; sg < num_sgs; sg++) total += slm[sg];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

// Max of v over the work-group.
inline float wg_max(float v, __local float* slm, uint lane, uint sg_id, uint num_sgs) {
    v = sub_group_reduce_max(v);
    if (lane == 0) slm[sg_id] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    float m = -INFINITY;
    for (uint sg = 0; sg < num_sgs; sg++) m = fmax(m, slm[sg]);
    barrier(CLK_LOCAL_MEM_FENCE);
    return m;
}

// Key segments OP_ATTN_PV splits each tile's keys into, so a small block still has enough
// work items: about 4096 items in all, at least 2 key tiles per segment, and partial outputs
// that fit the one-token attention scratch (16 heads x (max_seq / ATTN_CHUNK + 1) x 514
// floats). With 1 segment OP_ATTN_PV writes its output directly.
inline uint attn_pv_segments(uint H, uint N, uint D, uint key_rows, ulong max_seq) {
    uint items = H * ((N + ATTN_TILE - 1) / ATTN_TILE) * (D / 32);
    uint segs = (4096 + items - 1) / items;
    segs = min(segs, max(1u, (key_rows + 15) / 16 / 2));
    ulong room = 16ul * (max_seq / ATTN_CHUNK + 1) * 514;
    return (uint)max(1ul, min((ulong)segs, room / ((ulong)N * H * D)));
}

// Copies the plan's active experts and offsets into local memory for the work-group.
inline void load_plan(__global const MoEPlan* P, __local uint* active, __local uint* offset, uint lid, uint lsize) {
    for (uint i = lid; i < NUM_EXPERTS + 1; i += lsize) {
        if (i < NUM_EXPERTS) active[i] = P->active[i];
        offset[i] = P->offset[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// chunk[i] = how many chunks of `size` entries the active experts before i hold, for i up to
// n_active; work items then take (16 rows, one chunk), so an expert with many entries spreads
// over many subgroups.
inline void plan_chunks(__local const uint* active, __local const uint* offset, uint n_active, uint size,
                        __local uint* chunk, uint lid) {
    if (lid == 0) {
        uint c = 0;
        for (uint i = 0; i < n_active; i++) {
            chunk[i] = c;
            uint e = active[i];
            c += (offset[e + 1] - offset[e] + size - 1) / size;
        }
        chunk[n_active] = c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// The active expert index i whose chunks hold chunk c.
inline uint chunk_expert(__local const uint* chunk, uint n_active, uint c) {
    uint lo = 0, hi = n_active;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (chunk[mid] <= c) lo = mid; else hi = mid;
    }
    return lo;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void mk_persistent_interpreter(
    __global const MkTaskDescriptor* task_ring,
    uint                             num_tasks,
    __global atomic_uint*            scoreboard,
    __global const uint4*            token_meta,
    __global XeCoreStats*            core_stats,
    uint                             gpu_id,
    __global float*                  scratch
) {
    const uint wg_id   = get_group_id(0);
    const uint num_wgs = get_num_groups(0);
    const uint lid     = get_local_id(0);
    const uint lsize   = get_local_size(0);
    const uint sg_id   = get_sub_group_id();
    const uint lane    = get_sub_group_local_id();
    const uint num_sgs = get_num_sub_groups();
    // Global subgroup id, counted across work-groups first so that a task with fewer rows than
    // subgroups still spreads over every Xe2 core.
    const uint gsg     = sg_id * num_wgs + wg_id;
    const uint n_gsg   = num_wgs * num_sgs;
    // One-token matmuls: SPLIT_K subgroups of a work-group share each row, subgroup sg_id taking
    // part sg_id % SPLIT_K of K; a round covers num_sgs / SPLIT_K rows per work-group.
    const uint k_part  = sg_id % SPLIT_K;
    const uint k_slot  = sg_id / SPLIT_K;
    const uint split_stride = (num_sgs / SPLIT_K) * num_wgs;

    __local float slm_logits[NUM_EXPERTS];
    __local float slm_reduce[64];  // one per subgroup; a work-group has at most 64
    __local float slm_reduce2[64];
    __local float slm_part[2][64 * 128];   // K-split XMX partial sums: [g or u][subgroup][token][row]
    __local uint  slm_active[NUM_EXPERTS];       // the MoE plan's active experts and offsets
    __local uint  slm_offset[NUM_EXPERTS + 1];
    __local uint  slm_chunk[NUM_EXPERTS + 1];    // MoE work chunks before each active expert
    __local ushort slm_x[3 * XMX_TOKENS * XKC];  // dense matmuls' x tile, split (xmx_tile_1)
    __local uint  slm_spin_count;
    __local uchar slm_picks[MAX_PLAN_TOKENS * TOP_K];     // OP_MOE_PLAN: the block's expert picks
    __local uint  slm_seg[PLAN_SEGMENTS][NUM_EXPERTS];    // and their counts per segment

    if (lid == 0) slm_spin_count = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint tasks_done = 0;
    uint last_op = OP_NOP;

    for (uint pc = 0; pc < num_tasks; pc++) {
        MkTaskDescriptor task = task_ring[pc];
        ushort op = task.opcode;
        if (op == OP_HALT) {
            last_op = OP_HALT;
            break;
        }

        if (task.wait_event != NO_EVENT) {
            // The acquire fence invalidates the Xe2 core's whole L1 (lsc_fence.ugm.invalidate),
            // so one subgroup issues it while the rest wait at the barrier: their loads after
            // the barrier miss L1. One per subgroup made a task boundary 4.5 us instead of ~1.
            if (sg_id == 0) {
                if (lane == 0) {
                    uint spins = 0;
                    while (usm_event_load(&scoreboard[task.wait_event]) < num_wgs) spins++;
                    slm_spin_count += spins;
                }
                sub_group_barrier(CLK_LOCAL_MEM_FENCE);
                atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }

        const uint flags = task.flags;
        const bool accumulate = (flags & FLAG_ACCUMULATE) != 0;

        if (op == OP_RMSNORM) {
            // c[r] = a[r] * (mean(a[r]^2) + 1e-6)^-0.5 * w for n rows of k; w = 1 when b == 0.
            // FLAG_ACCUMULATE adds to c[r]. With FLAG_ROPE (head rows, in place), row r belongs
            // to token r / m at position p0 + r / m, and pairs (i, i + k/2) of the normalized row
            // rotate by pos * theta^(-2i/k); global layers (FLAG_ROPE_GLOBAL) rotate only i < k/8
            // (partial_rotary_factor 0.25). Both halves of a pair fall to one lane.
            __global const float*  A = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        C = (__global float*)task.c;
            uint K = task.k;
            // Rows of the hidden size get a work-group each; head-sized rows get a subgroup.
            if (K >= lsize) {
                for (uint r = wg_id; r < task.n; r += num_wgs) {
                    __global const float* x = A + (ulong)r * K;
                    __global float*       y = C + (ulong)r * K;
                    float ss = 0.0f;
                    for (uint i = lid; i < K; i += lsize) ss += x[i] * x[i];
                    float inv = rsqrt(wg_sum(ss, slm_reduce, lane, sg_id, num_sgs) / (float)K + 1e-6f);
                    for (uint i = lid; i < K; i += lsize) {
                        float v = x[i] * inv * (W ? bf16_to_f32(W[i]) : 1.0f);
                        y[i] = accumulate ? y[i] + v : v;
                    }
                }
            }
            else if (flags & FLAG_ROPE) {
                uint half_d = K / 2, heads = task.m;
                bool global_rope = (flags & FLAG_ROPE_GLOBAL) != 0;
                float theta = global_rope ? 1000000.0f : 10000.0f;
                uint n_rot = global_rope ? K / 8 : half_d;
                for (uint r = gsg; r < task.n; r += n_gsg) {
                    __global const float* x = A + (ulong)r * K;
                    __global float*       y = C + (ulong)r * K;
                    float ss = 0.0f;
                    for (uint i = lane; i < K; i += 16) ss += x[i] * x[i];
                    ss = sub_group_reduce_add(ss);
                    float inv = rsqrt(ss / (float)K + 1e-6f);
                    uint t = r / heads;
                    float pos = (float)(token_meta ? token_meta[t].x : task.p0 + t);
                    for (uint i = lane; i < half_d; i += 16) {
                        float x0 = x[i] * inv * (W ? bf16_to_f32(W[i]) : 1.0f);
                        float x1 = x[i + half_d] * inv * (W ? bf16_to_f32(W[i + half_d]) : 1.0f);
                        if (i < n_rot) {
                            float inv_freq = 1.0f / pow(theta, (float)(2 * i) / (float)K);
                            float ang = pos * inv_freq;
                            float cs = cos(ang), sn = sin(ang);
                            y[i]          = x0 * cs - x1 * sn;
                            y[i + half_d] = x1 * cs + x0 * sn;
                        } else {
                            y[i] = x0;
                            y[i + half_d] = x1;
                        }
                    }
                }
            }
            else for (uint r = gsg; r < task.n; r += n_gsg) {
                __global const float* x = A + (ulong)r * K;
                __global float*       y = C + (ulong)r * K;
                float ss = 0.0f;
                for (uint i = lane; i < K; i += 16) ss += x[i] * x[i];
                ss = sub_group_reduce_add(ss);
                float inv = rsqrt(ss / (float)K + 1e-6f);
                for (uint i = lane; i < K; i += 16) {
                    float v = x[i] * inv * (W ? bf16_to_f32(W[i]) : 1.0f);
                    y[i] = accumulate ? y[i] + v : v;
                }
            }
        }
        else if (op == OP_ATTN_OUT) {
            // After attention, per row r of n rows of k (one work-group each), with b packing the
            // weights post_attention_layernorm, pre_feedforward_layernorm, pre_feedforward_layernorm_2:
            // c[r] += rmsnorm(a[r]) * w0, then x[r] = rmsnorm(c[r]) * w1 and x[r] + m = rmsnorm(c[r]) * w2.
            __global const float*  A = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        C = (__global float*)task.c;
            __global float*        O = (__global float*)task.x;
            uint K = task.k;
            for (uint r = wg_id; r < task.n; r += num_wgs) {
                __global const float* a = A + (ulong)r * K;
                __global float* c = C + (ulong)r * K;
                __global float* o1 = O + (ulong)r * K;
                __global float* o2 = o1 + task.m;
                float ss = 0.0f;
                for (uint i = lid; i < K; i += lsize) ss += a[i] * a[i];
                float inv = rsqrt(wg_sum(ss, slm_reduce, lane, sg_id, num_sgs) / (float)K + 1e-6f);
                float ss2 = 0.0f;
                for (uint i = lid; i < K; i += lsize) {
                    float v = c[i] + a[i] * inv * bf16_to_f32(W[i]);
                    c[i] = v;
                    ss2 += v * v;
                }
                float inv2 = rsqrt(wg_sum(ss2, slm_reduce, lane, sg_id, num_sgs) / (float)K + 1e-6f);
                for (uint i = lid; i < K; i += lsize) {
                    o1[i] = c[i] * inv2 * bf16_to_f32(W[K + i]);
                    o2[i] = c[i] * inv2 * bf16_to_f32(W[2 * K + i]);
                }
            }
        }
        else if (op == OP_FFN_OUT) {
            // End of a layer, per row r of n rows of k (one work-group each), with b packing the
            // weights post_feedforward_layernorm_1, _2, and post_feedforward_layernorm:
            // e = the sum of x's 8 expert slots of row r (OP_MOE_DOWN's output),
            // h = rmsnorm(a[r]) * w0 + rmsnorm(e) * w1 (shared MLP, routed experts), then
            // c[r] = (c[r] + rmsnorm(h) * w2) * as_float(p0), the layer scalar.
            __global const float*  A = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        C = (__global float*)task.c;
            __global const float*  E = (__global const float*)task.x;
            uint K = task.k;
            float scalar = as_float(task.p0);
            for (uint r = wg_id; r < task.n; r += num_wgs) {
                __global const float* a = A + (ulong)r * K;
                __global float* c = C + (ulong)r * K;
                // k <= 3 * lsize, so a work-item holds its e and h values in registers.
                float e[3], h[3];
                float ssa = 0.0f, sse = 0.0f;
                for (uint j = 0; j < 3; j++) {
                    uint i = lid + j * lsize;
                    e[j] = 0.0f;
                    if (i < K) {
                        for (uint p = 0; p < TOP_K; p++) e[j] += E[((ulong)r * TOP_K + p) * K + i];
                        ssa += a[i] * a[i];
                        sse += e[j] * e[j];
                    }
                }
                float inva = rsqrt(wg_sum(ssa, slm_reduce, lane, sg_id, num_sgs) / (float)K + 1e-6f);
                float inve = rsqrt(wg_sum(sse, slm_reduce, lane, sg_id, num_sgs) / (float)K + 1e-6f);
                float ssh = 0.0f;
                for (uint j = 0; j < 3; j++) {
                    uint i = lid + j * lsize;
                    h[j] = 0.0f;
                    if (i < K) {
                        h[j] = a[i] * inva * bf16_to_f32(W[i]);
                        h[j] = h[j] + e[j] * inve * bf16_to_f32(W[K + i]);
                        ssh += h[j] * h[j];
                    }
                }
                float invh = rsqrt(wg_sum(ssh, slm_reduce, lane, sg_id, num_sgs) / (float)K + 1e-6f);
                for (uint j = 0; j < 3; j++) {
                    uint i = lid + j * lsize;
                    if (i < K) c[i] = (c[i] + h[j] * invh * bf16_to_f32(W[2 * K + i])) * scalar;
                }
            }
        }
        else if (op == OP_MATMUL) {
            // c[t, r] = sum_k a[t, k] * b[r, k] for n tokens, m rows, k columns; when n > 1, m and
            // k are multiples of 16.
            __global const float*  X = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        Y = (__global float*)task.c;
            uint M = task.m, N = task.n, K = task.k;
            bool softcap = (flags & FLAG_SOFTCAP_30) != 0;
            if (N == 1 && M <= SPLIT_K_MAX_ROWS) {
                uint per = (K / 8 + SPLIT_K - 1) / SPLIT_K, lo = k_part * per, hi = min(K / 8, lo + per);
                for (uint base = 0; base < M; base += split_stride) {
                    uint r = base + k_slot * num_wgs + wg_id;
                    float a = r < M ? part_dots(W + (ulong)r * K, 0, X, lo, hi, lane).x : 0.0f;
                    if (lane == 0) slm_reduce[sg_id] = a;
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (k_part == 0 && lane == 0 && r < M) {
                        float sum = 0.0f;
                        for (uint p = 0; p < SPLIT_K; p++) sum += slm_reduce[sg_id + p];
                        float v = softcap ? 30.0f * tanh(sum / 30.0f) : sum;
                        Y[r] = accumulate ? Y[r] + v : v;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            // At most 8 tokens: XMX tiles of 16 rows with K split over KSPLIT subgroups of a
            // work-group; the tile's first subgroup adds their partial sums in order.
            else if (N <= 8) {
                uint row_tiles = M / 16, steps = K / 16, kp = sg_id % KSPLIT, slot = sg_id / KSPLIT;
                uint k_lo = 16 * (steps * kp / KSPLIT), k_hi = 16 * (steps * (kp + 1) / KSPLIT);
                for (uint base = 0; base < row_tiles; base += (num_sgs / KSPLIT) * num_wgs) {
                    uint tile = base + slot * num_wgs + wg_id;
                    float8 acc[4];
                    acc[0] = 0.0f;
                    if (tile < row_tiles) xmx_rows_1(W + (ulong)(tile * 16 + lane) * K, X, 0, N, K, 1, k_lo, k_hi, acc);
                    for (uint r = 0; r < 8; r++) slm_part[0][sg_id * 128 + r * 16 + lane] = acc[0][r];
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (kp == 0 && tile < row_tiles) {
                        for (uint t = 0; t < N; t++) {
                            float sum = 0.0f;
                            for (uint p = 0; p < KSPLIT; p++) sum += slm_part[0][(sg_id + p) * 128 + t * 16 + lane];
                            float v = softcap ? 30.0f * tanh(sum / 30.0f) : sum;
                            __global float* y = Y + (ulong)t * M + tile * 16 + lane;
                            *y = accumulate ? *y + v : v;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            // Several tokens: a work-group takes 64 tiles of 16 rows by XMX_TOKENS tokens, sharing
            // the tokens' x through local memory.
            else if (N >= 16) {
                uint row_tiles = M / 16, tok_tiles = (N + XMX_TOKENS - 1) / XMX_TOKENS;
                uint blocks = (row_tiles + num_sgs - 1) / num_sgs;
                for (uint wi = wg_id; wi < blocks * tok_tiles; wi += num_wgs) {
                    uint tile = (wi % blocks) * num_sgs + sg_id, t0 = (wi / blocks) * XMX_TOKENS;
                    uint r0 = tile * 16, tn = min((uint)XMX_TOKENS, N - t0);
                    float8 acc[4];
                    xmx_tile_1(tile < row_tiles ? W + (ulong)(r0 + lane) * K : 0, X, t0, N, K, (tn + 7) / 8, slm_x, lid, lsize, lane, acc);
                    if (tile >= row_tiles) continue;
                    for (uint mt = 0; mt < 4; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            uint t = 8 * mt + r;
                            if (t < tn) {
                                float v = softcap ? 30.0f * tanh(acc[mt][r] / 30.0f) : acc[mt][r];
                                __global float* y = Y + (ulong)(t0 + t) * M + r0 + lane;
                                *y = accumulate ? *y + v : v;
                            }
                        }
                    }
                }
            }
            // 9 to 15 tokens: XMX work items of 16 rows by XMX_TOKENS tokens.
            else {
                uint row_tiles = M / 16, tok_tiles = (N + XMX_TOKENS - 1) / XMX_TOKENS;
                for (uint item = gsg; item < row_tiles * tok_tiles; item += n_gsg) {
                    uint r0 = (item % row_tiles) * 16, t0 = (item / row_tiles) * XMX_TOKENS;
                    uint tn = min((uint)XMX_TOKENS, N - t0);
                    float8 acc[4];
                    xmx_rows_1(W + (ulong)(r0 + lane) * K, X, t0, N, K, (tn + 7) / 8, 0, K, acc);
                    for (uint mt = 0; mt < 4; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            uint t = 8 * mt + r;
                            if (t < tn) {
                                float v = softcap ? 30.0f * tanh(acc[mt][r] / 30.0f) : acc[mt][r];
                                __global float* y = Y + (ulong)(t0 + t) * M + r0 + lane;
                                *y = accumulate ? *y + v : v;
                            }
                        }
                    }
                }
            }
        }
        else if (op == OP_GEGLU) {
            // c[t, r] = gelu(a[t] . b[r]) * (a[t] . b[m + r]) for m intermediate rows; m and k as OP_MATMUL.
            __global const float*  X = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        Y = (__global float*)task.c;
            uint M = task.m, N = task.n, K = task.k;
            if (N == 1) {
                uint per = (K / 8 + SPLIT_K - 1) / SPLIT_K, lo = k_part * per, hi = min(K / 8, lo + per);
                for (uint base = 0; base < M; base += split_stride) {
                    uint r = base + k_slot * num_wgs + wg_id;
                    float2 a = r < M ? part_dots(W + (ulong)r * K, W + (ulong)(M + r) * K, X, lo, hi, lane) : (float2)(0.0f);
                    if (lane == 0) { slm_reduce[sg_id] = a.x; slm_reduce2[sg_id] = a.y; }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (k_part == 0 && lane == 0 && r < M) {
                        float gs = 0.0f, us = 0.0f;
                        for (uint p = 0; p < SPLIT_K; p++) { gs += slm_reduce[sg_id + p]; us += slm_reduce2[sg_id + p]; }
                        Y[r] = gelu_tanh(gs) * us;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            // At most 8 tokens: as OP_MATMUL's, K split over KSPLIT subgroups.
            else if (N <= 8) {
                uint row_tiles = M / 16, steps = K / 16, kp = sg_id % KSPLIT, slot = sg_id / KSPLIT;
                uint k_lo = 16 * (steps * kp / KSPLIT), k_hi = 16 * (steps * (kp + 1) / KSPLIT);
                for (uint base = 0; base < row_tiles; base += (num_sgs / KSPLIT) * num_wgs) {
                    uint tile = base + slot * num_wgs + wg_id;
                    float8 g[2], u[2];
                    g[0] = 0.0f;
                    u[0] = 0.0f;
                    if (tile < row_tiles) xmx_rows_2(W + (ulong)(tile * 16 + lane) * K, W + (ulong)(M + tile * 16 + lane) * K, X, 0, N, K, 1, k_lo, k_hi, g, u);
                    for (uint r = 0; r < 8; r++) {
                        slm_part[0][sg_id * 128 + r * 16 + lane] = g[0][r];
                        slm_part[1][sg_id * 128 + r * 16 + lane] = u[0][r];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (kp == 0 && tile < row_tiles) {
                        for (uint t = 0; t < N; t++) {
                            float gs = 0.0f, us = 0.0f;
                            for (uint p = 0; p < KSPLIT; p++) {
                                gs += slm_part[0][(sg_id + p) * 128 + t * 16 + lane];
                                us += slm_part[1][(sg_id + p) * 128 + t * 16 + lane];
                            }
                            Y[(ulong)t * M + tile * 16 + lane] = gelu_tanh(gs) * us;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            else {
                uint row_tiles = M / 16, tok_tiles = (N + XMX_TOKENS2 - 1) / XMX_TOKENS2;
                for (uint item = gsg; item < row_tiles * tok_tiles; item += n_gsg) {
                    uint r0 = (item % row_tiles) * 16, t0 = (item / row_tiles) * XMX_TOKENS2;
                    uint tn = min((uint)XMX_TOKENS2, N - t0);
                    float8 g[2], u[2];
                    xmx_rows_2(W + (ulong)(r0 + lane) * K, W + (ulong)(M + r0 + lane) * K, X, t0, N, K, (tn + 7) / 8, 0, K, g, u);
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            uint t = 8 * mt + r;
                            if (t < tn) Y[(ulong)(t0 + t) * M + r0 + lane] = gelu_tanh(g[mt][r]) * u[mt][r];
                        }
                    }
                }
            }
        }
        else if (op == OP_ATTN) {
            // One token at position p0 without a token table, causal: query rows a[h, :]; the key
            // cache is b[pos, kv, :] and the value cache follows it at b + max_seq * kv * k.
            // m = heads | kv_heads << 16; x = max_seq << 32 | window (0: no window). Scale 1.0.
            // layer = the cache's ring rows (0: rows are positions); see kv_row.
            // Work items take (chunk of ATTN_CHUNK keys, kv head, query group); OP_ATTN_COMBINE
            // merges the chunks from scratch, [head][chunk][max, sum, o[D]]. Blocks of several
            // tokens use OP_ATTN_SCORES, OP_ATTN_SOFTMAX, and OP_ATTN_PV.
            __global const float* Q  = (__global const float*)task.a;
            __global const float* Kc = (__global const float*)task.b;
            uint D = task.k, H = task.m & 0xFFFFu, KV = task.m >> 16;
            uint window = (uint)(task.x & 0xFFFFFFFFul);
            ulong max_seq = task.x >> 32;
            uint ring = task.layer;
            __global const float* Vc = Kc + kv_plane_rows(max_seq, ring) * KV * D;
            uint lo, chunks = attn_keys(task.p0, window, &lo);
            uint G = D == 256 ? 2 : 1, QG = H / KV / G;
            for (uint item = gsg; item < chunks * KV * QG; item += n_gsg) {
                uint qg = item % QG, kvh = (item / QG) % KV, c = item / (QG * KV);
                uint h0 = kvh * (H / KV) + qg * G;
                uint k_lo = lo + c * ATTN_CHUNK, k_hi = min(task.p0, k_lo + ATTN_CHUNK - 1);
                __global float* part = scratch + (ulong)c * (D + 2);
                ulong stride = (ulong)chunks * (D + 2);
                if (D == 256) attn_chunk_16_2(Q, Kc, Vc, part, stride, h0, kvh, D, KV, k_lo, k_hi, ring, lane);
                else          attn_chunk_32_1(Q, Kc, Vc, part, stride, h0, kvh, D, KV, k_lo, k_hi, ring, lane);
            }
        }
        else if (op == OP_ATTN_SCORES) {
            // c[h, t, j] = a[t, h, :] . k[j, h / (heads / kv), :] for n tokens and the key rows j
            // of their tiles (see attn_row_floats); fields as OP_ATTN's. Work items: (head, tile
            // of ATTN_TILE tokens, 32 keys).
            __global const float* Q  = (__global const float*)task.a;
            __global const float* Kc = (__global const float*)task.b;
            __global float*       S  = (__global float*)task.c;
            uint D = task.k, H = task.m & 0xFFFFu, KV = task.m >> 16, N = task.n, p0 = task.p0;
            uint window = (uint)(task.x & 0xFFFFFFFFul);
            uint row_f = attn_row_floats(p0, N), key_rows = p0 + N;
            uint ring = task.layer, k_height = kv_surface_rows(key_rows, ring);
            uint t_tiles = (N + ATTN_TILE - 1) / ATTN_TILE, k_pairs = (key_rows + 31) / 32;
            for (uint item = gsg; item < H * t_tiles * k_pairs; item += n_gsg) {
                uint kp = item % k_pairs, tt = (item / k_pairs) % t_tiles, h = item / (k_pairs * t_tiles);
                uint t0 = tt * ATTN_TILE, kvh = h / (H / KV);
                uint2 span = attn_tile_keys(t0, N, p0, window, token_meta, lane);
                uint j0 = kp * 32;
                if (j0 + 32 <= span.x || j0 >= span.y) continue;
                float8 acc[2][2];
                for (uint a = 0; a < 2; a++) for (uint b = 0; b < 2; b++) acc[a][b] = 0.0f;
                // The rows of the pair's two key tiles, as in OP_ATTN_PV: no modulo in the loop.
                int krow[2] = {(int)sub_group_broadcast((uint)kv_row(j0, ring), 0), (int)sub_group_broadcast((uint)kv_row(j0 + 16, ring), 0)};
                for (uint k0 = 0; k0 < D; k0 += 16) {
                    short8 qh[2], qm[2], ql[2];
                    _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) {
                        uint qv[8];
                        intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)Q, H * D * 4, N, H * D * 4, (int2)(h * D + k0, t0 + 8 * mt), qv);
                        split3_rows(qv, &qh[mt], &qm[mt], &ql[mt]);
                    }
                    _Pragma("unroll") for (uint nt = 0; nt < 2; nt++) {
                        uint kv[16];
                        int ky = krow[nt];
                        intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)Kc, KV * D * 4, k_height, KV * D * 4, (int2)(kvh * D + k0, ky), kv);
                        intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)Kc, KV * D * 4, k_height, KV * D * 4, (int2)(kvh * D + k0 + 8, ky), kv + 8);
                        int8 kh, km, kl;
                        split3_pairs(kv, &kh, &km, &kl);
                        _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) acc[mt][nt] = dpas_split6(qh[mt], qm[mt], ql[mt], kh, km, kl, acc[mt][nt]);
                    }
                }
                for (uint mt = 0; mt < 2; mt++) {
                    for (uint r = 0; r < 8; r++) {
                        uint t = t0 + 8 * mt + r;
                        if (t < N) {
                            __global float* srow = S + ((ulong)h * N + t) * row_f;
                            srow[j0 + lane] = acc[mt][0][r];
                            srow[j0 + 16 + lane] = acc[mt][1][r];
                        }
                    }
                }
            }
        }
        else if (op == OP_ATTN_SOFTMAX) {
            // Turns each score row of c (OP_ATTN_SCORES') into probabilities over the keys its
            // token reads, with 0 for the rest of its tile's keys. One subgroup per (head, token).
            __global float* S = (__global float*)task.c;
            uint H = task.m & 0xFFFFu, N = task.n, p0 = task.p0;
            uint window = (uint)(task.x & 0xFFFFFFFFul);
            uint row_f = attn_row_floats(p0, N);
            // Few rows (a small block): a work-group per row.
            if (H * N <= 4 * num_wgs) for (uint item = wg_id; item < H * N; item += num_wgs) {
                uint t = item % N, h = item / N;
                uint2 span = attn_tile_keys(t / ATTN_TILE * ATTN_TILE, N, p0, window, token_meta, lane);
                uint4 r = attn_token_keys(t, p0, window, token_meta);
                __global float* srow = S + ((ulong)h * N + t) * row_f;
                float m = -INFINITY;
                for (uint j = r.x + lid; j <= r.y; j += lsize) {
                    if (!(j >= r.z && j < r.w)) m = fmax(m, srow[j]);
                }
                m = wg_max(m, slm_reduce, lane, sg_id, num_sgs);
                float sum = 0.0f;
                for (uint j = r.x + lid; j <= r.y; j += lsize) {
                    if (!(j >= r.z && j < r.w)) sum += exp(srow[j] - m);
                }
                float inv = 1.0f / wg_sum(sum, slm_reduce, lane, sg_id, num_sgs);
                for (uint j = span.x + lid; j < span.y; j += lsize) {
                    bool valid = j >= r.x && j <= r.y && !(j >= r.z && j < r.w);
                    srow[j] = valid ? exp(srow[j] - m) * inv : 0.0f;
                }
            }
            else for (uint item = gsg; item < H * N; item += n_gsg) {
                uint t = item % N, h = item / N;
                uint2 span = attn_tile_keys(t / ATTN_TILE * ATTN_TILE, N, p0, window, token_meta, lane);
                uint4 r = attn_token_keys(t, p0, window, token_meta);
                __global float* srow = S + ((ulong)h * N + t) * row_f;
                float m = -INFINITY;
                for (uint j = r.x + lane; j <= r.y; j += 16) {
                    if (!(j >= r.z && j < r.w)) m = fmax(m, srow[j]);
                }
                m = sub_group_reduce_max(m);
                float sum = 0.0f;
                for (uint j = r.x + lane; j <= r.y; j += 16) {
                    if (!(j >= r.z && j < r.w)) sum += exp(srow[j] - m);
                }
                float inv = 1.0f / sub_group_reduce_add(sum);
                for (uint j = span.x + lane; j < span.y; j += 16) {
                    bool valid = j >= r.x && j <= r.y && !(j >= r.z && j < r.w);
                    srow[j] = valid ? exp(srow[j] - m) * inv : 0.0f;
                }
            }
        }
        else if (op == OP_ATTN_PV) {
            // c[t, h, :] = sum_j a[h, t, j] v[j, h / (heads / kv), :] over OP_ATTN_SOFTMAX's
            // probabilities; b is the cache as OP_ATTN's. Work items: (head, tile of ATTN_TILE
            // tokens, 32 output dims).
            __global const float* S  = (__global const float*)task.a;
            __global const float* Kc = (__global const float*)task.b;
            __global float*       O  = (__global float*)task.c;
            uint D = task.k, H = task.m & 0xFFFFu, KV = task.m >> 16, N = task.n, p0 = task.p0;
            uint window = (uint)(task.x & 0xFFFFFFFFul);
            ulong max_seq = task.x >> 32;
            uint ring = task.layer;
            __global const float* Vc = Kc + kv_plane_rows(max_seq, ring) * KV * D;
            uint row_f = attn_row_floats(p0, N), key_rows = p0 + N, k_height = kv_surface_rows(key_rows, ring);
            uint t_tiles = (N + ATTN_TILE - 1) / ATTN_TILE, d_pairs = D / 32;
            uint segs = attn_pv_segments(H, N, D, key_rows, max_seq);
            for (uint item = gsg; item < H * t_tiles * d_pairs * segs; item += n_gsg) {
                uint seg = item % segs, rest = item / segs;
                uint dp = rest % d_pairs, tt = (rest / d_pairs) % t_tiles, h = rest / (d_pairs * t_tiles);
                uint t0 = tt * ATTN_TILE, kvh = h / (H / KV), d0 = dp * 32;
                uint2 span = attn_tile_keys(t0, N, p0, window, token_meta, lane);
                uint tiles = (span.y - span.x) / 16;
                uint j_lo = span.x + 16 * (tiles * seg / segs), j_hi = span.x + 16 * (tiles * (seg + 1) / segs);
                __global const float* Sh = S + (ulong)h * N * row_f;
                // With several segments each writes its partial sum to scratch [seg][t][h][d].
                __global float* dst = segs > 1 ? scratch + (ulong)seg * N * H * D : O;
                float8 acc[2][2];
                for (uint a = 0; a < 2; a++) for (uint b = 0; b < 2; b++) acc[a][b] = 0.0f;
                // The row of key j0, stepped with j0. One modulo per item, made uniform: a block
                // read whose row comes from a modulo in this loop hangs the card.
                uint vy = sub_group_broadcast((uint)kv_row(j_lo, ring), 0);
                for (uint j0 = j_lo; j0 < j_hi; j0 += 16) {
                    short8 ph[2], pm[2], pl[2];
                    _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) {
                        uint pv[8];
                        intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)Sh, row_f * 4, N, row_f * 4, (int2)(j0, t0 + 8 * mt), pv);
                        split3_rows(pv, &ph[mt], &pm[mt], &pl[mt]);
                    }
                    _Pragma("unroll") for (uint nt = 0; nt < 2; nt++) {
                        uint vv[16];
                        intel_sub_group_2d_block_read_32b_16r16x1c((__global void*)Vc, KV * D * 4, k_height, KV * D * 4, (int2)(kvh * D + d0 + 16 * nt, (int)vy), vv);
                        int8 vh, vm, vl;
                        split3_pairs(vv, &vh, &vm, &vl);
                        _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) acc[mt][nt] = dpas_split6(ph[mt], pm[mt], pl[mt], vh, vm, vl, acc[mt][nt]);
                    }
                    vy += 16;
                    if (ring && vy >= ring) vy -= ring;
                }
                for (uint mt = 0; mt < 2; mt++) {
                    for (uint r = 0; r < 8; r++) {
                        uint t = t0 + 8 * mt + r;
                        if (t < N) {
                            __global float* orow = dst + ((ulong)t * H + h) * D + d0;
                            orow[lane] = acc[mt][0][r];
                            orow[16 + lane] = acc[mt][1][r];
                        }
                    }
                }
            }
        }
        else if (op == OP_ATTN_PV_SUM) {
            // Adds OP_ATTN_PV's partial sums from scratch into c[t, h, :] in segment order;
            // fields as OP_ATTN_PV's. Nothing to do with one segment.
            __global float* O = (__global float*)task.c;
            uint D = task.k, H = task.m & 0xFFFFu, N = task.n;
            uint segs = attn_pv_segments(H, N, D, task.p0 + N, task.x >> 32);
            if (segs > 1) {
                ulong total = (ulong)N * H * D;
                for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                    float v = 0.0f;
                    for (uint seg = 0; seg < segs; seg++) v += scratch[seg * total + i];
                    O[i] = v;
                }
            }
        }
        else if (op == OP_ATTN_COMBINE) {
            // Merges one-token attention's chunks from scratch into c[h, :] for every head, in
            // chunk order. m = heads, k = head_dim; p0 and x (window) as OP_ATTN's. Work items:
            // (head, 16 output dims), one dim per lane.
            __global float* O = (__global float*)task.c;
            uint D = task.k, H = task.m;
            uint lo, chunks = attn_keys(task.p0, (uint)(task.x & 0xFFFFFFFFul), &lo);
            for (uint item = gsg; item < H * (D / 16); item += n_gsg) {
                uint h = item / (D / 16), d = (item % (D / 16)) * 16 + lane;
                __global const float* part = scratch + (ulong)h * chunks * (D + 2);
                float m = -INFINITY, l = 0.0f, o = 0.0f;
                for (uint c = 0; c < chunks; c++) {
                    __global const float* pp = part + (ulong)c * (D + 2);
                    float mc = pp[0], lc = pp[1];
                    float nm = fmax(m, mc);
                    float a = exp(m - nm), b = exp(mc - nm);
                    l = l * a + lc * b;
                    o = o * a + pp[2 + d] * b;
                    m = nm;
                }
                O[(ulong)h * D + d] = o * (1.0f / l);
            }
        }
        else if (op == OP_ROUTER_LOGITS) {
            // c[t, e] = sum_i b[e, i] * scale[i] * a[t, i]: Gemma4TextRouter's projection of
            // n tokens of k before the RMS factor, which OP_ROUTER applies. b packs proj
            // [128, k] then scale [k]. One subgroup per (token, expert).
            __global const float*  X = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        L = (__global float*)task.c;
            uint K = task.k;
            __global const ushort* r_scale = W + (ulong)NUM_EXPERTS * K;
            for (uint item = gsg; item < task.n * NUM_EXPERTS; item += n_gsg) {
                uint t = item / NUM_EXPERTS, e = item % NUM_EXPERTS;
                __global const ushort* w_row = W + (ulong)e * K;
                __global const float* x = X + (ulong)t * K;
                float acc = 0.0f;
                for (uint k8 = lane; k8 < K / 8; k8 += 16) {
                    ushort8 u = vload8(k8, w_row), sc = vload8(k8, r_scale);
                    float8 xv = vload8(k8, x);
                    acc += dot(bf16x4_to_f32(u.lo) * bf16x4_to_f32(sc.lo), xv.lo)
                         + dot(bf16x4_to_f32(u.hi) * bf16x4_to_f32(sc.hi), xv.hi);
                }
                acc = sub_group_reduce_add(acc);
                if (lane == 0) L[(ulong)t * NUM_EXPERTS + e] = acc;
            }
        }
        else if (op == OP_ROUTER) {
            // Gemma4TextRouter for n tokens of k <= 2816 from OP_ROUTER_LOGITS' sums in x:
            // logits = sums * rms(a[t])^-1 * k^-0.5, then the top 8 renormalized and scaled by
            // per_expert_scale (b + 129 k). One work-group per token.
            __global const float*  X = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global MoERouteDecision* R = (__global MoERouteDecision*)task.c;
            __global const float*  S = (__global const float*)task.x;
            uint K = task.k;
            __global const ushort* pe_scale = W + (ulong)(NUM_EXPERTS + 1) * K;
            for (uint t = wg_id; t < task.n; t += num_wgs) {
                __global const float* x = X + (ulong)t * K;
                float ss = 0.0f;
                for (uint i = lid; i < K; i += lsize) ss += x[i] * x[i];
                ss = sub_group_reduce_add(ss);
                if (lane == 0) slm_reduce[sg_id] = ss;
                barrier(CLK_LOCAL_MEM_FENCE);
                float total = 0.0f;
                for (uint sg = 0; sg < num_sgs; sg++) total += slm_reduce[sg];
                float scale = rsqrt(total / (float)K + 1e-6f) * rsqrt((float)K);
                for (uint e = lid; e < NUM_EXPERTS; e += lsize) slm_logits[e] = S[(ulong)t * NUM_EXPERTS + e] * scale;
                barrier(CLK_LOCAL_MEM_FENCE);
                // Subgroup 0 picks the top 8; lane l holds experts l, l + 16, ... Ties go to the
                // lowest expert index.
                if (sg_id == 0) {
                    float v[NUM_EXPERTS / 16];
                    for (uint j = 0; j < NUM_EXPERTS / 16; j++) v[j] = slm_logits[lane + 16 * j];
                    float lmax = -INFINITY;
                    for (uint j = 0; j < NUM_EXPERTS / 16; j++) lmax = fmax(lmax, v[j]);
                    float gmax = sub_group_reduce_max(lmax);
                    float lsum = 0.0f;
                    for (uint j = 0; j < NUM_EXPERTS / 16; j++) lsum += exp(v[j] - gmax);
                    float gsum = sub_group_reduce_add(lsum);
                    uint  idx[TOP_K];
                    float val[TOP_K];
                    for (uint pick = 0; pick < TOP_K; pick++) {
                        float best = -INFINITY;
                        uint  bi = NUM_EXPERTS;
                        for (uint j = 0; j < NUM_EXPERTS / 16; j++) {
                            if (v[j] > best) { best = v[j]; bi = lane + 16 * j; }
                        }
                        float m = sub_group_reduce_max(best);
                        uint e = sub_group_reduce_min(best == m ? bi : NUM_EXPERTS);
                        for (uint j = 0; j < NUM_EXPERTS / 16; j++) {
                            if (lane + 16 * j == e) v[j] = -INFINITY;
                        }
                        idx[pick] = e;
                        val[pick] = m;
                    }
                    float top = val[0];
                    float psum = 0.0f;
                    for (uint pick = 0; pick < TOP_K; pick++) {
                        val[pick] = exp(val[pick] - top);
                        psum += val[pick];
                    }
                    float entropy = 0.0f;
                    for (uint pick = 0; pick < TOP_K; pick++) {
                        float pr = val[pick] / psum;
                        if (lane == 0) {
                            R[t].top_k_indices[pick] = idx[pick];
                            R[t].top_k_weights[pick] = pr * bf16_to_f32(pe_scale[idx[pick]]);
                        }
                        if (pr > 0.0f) entropy -= pr * log(pr);
                    }
                    if (lane == 0) {
                        R[t].router_entropy = entropy;
                        R[t].router_lse = gmax + log(gsum);
                        R[t].layer_idx = task.layer;
                        R[t].token = t;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        else if (op == OP_MOE_PLAN) {
            // Groups the n tokens' top-8 picks in a (MoERouteDecision[n]) by expert into c (MoEPlan).
            __global const MoERouteDecision* R = (__global const MoERouteDecision*)task.a;
            __global MoEPlan* P = (__global MoEPlan*)task.c;
            // The last work-group, so work-group 0 is free for the one-row norms beside it. Work
            // item (expert e, segment g) counts, then places, e's picks in segment g of the
            // picks in token order, so the plan lists each expert's entries in token order.
            if (wg_id == num_wgs - 1) {
                uint picks = task.n * TOP_K;
                for (uint i = lid; i < picks; i += lsize) slm_picks[i] = (uchar)R[i / TOP_K].top_k_indices[i % TOP_K];
                barrier(CLK_LOCAL_MEM_FENCE);
                uint seg_len = (picks + PLAN_SEGMENTS - 1) / PLAN_SEGMENTS;
                for (uint w = lid; w < NUM_EXPERTS * PLAN_SEGMENTS; w += lsize) {
                    uint e = w % NUM_EXPERTS, g = w / NUM_EXPERTS, c = 0;
                    for (uint i = g * seg_len; i < min(picks, (g + 1) * seg_len); i++) c += slm_picks[i] == e;
                    slm_seg[g][e] = c;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (lid == 0) {
                    uint off = 0, n_active = 0;
                    for (uint e = 0; e < NUM_EXPERTS; e++) {
                        P->offset[e] = off;
                        uint total = 0;
                        for (uint g = 0; g < PLAN_SEGMENTS; g++) {
                            uint c = slm_seg[g][e];
                            slm_seg[g][e] = off + total;
                            total += c;
                        }
                        P->count[e] = total;
                        if (total > 0) P->active[n_active++] = e;
                        off += total;
                    }
                    P->offset[NUM_EXPERTS] = off;
                    P->n_active = n_active;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint w = lid; w < NUM_EXPERTS * PLAN_SEGMENTS; w += lsize) {
                    uint e = w % NUM_EXPERTS, g = w / NUM_EXPERTS, at = slm_seg[g][e];
                    for (uint i = g * seg_len; i < min(picks, (g + 1) * seg_len); i++) {
                        if (slm_picks[i] == e) {
                            P->entry_slot[at] = i;
                            P->entry_weight[at] = R[i / TOP_K].top_k_weights[i % TOP_K];
                            at++;
                        }
                    }
                }
            }
        }
        else if (op == OP_MOE_GATHER) {
            // c[q] = a[t] for each entry q of plan x, t its token: the rows OP_MOE_GATE_UP reads,
            // in plan order, k floats each. One subgroup per entry.
            __global const float*  A = (__global const float*)task.a;
            __global float*        C = (__global float*)task.c;
            __global const MoEPlan* P = (__global const MoEPlan*)task.x;
            uint K = task.k, entries = task.n * TOP_K;
            for (uint q = gsg; q < entries; q += n_gsg) {
                __global const float4* src = (__global const float4*)(A + (ulong)(P->entry_slot[q] / TOP_K) * K);
                __global float4* dst = (__global float4*)(C + (ulong)q * K);
                for (uint i = lane; i < K / 4; i += 16) dst[i] = src[i];
            }
        }
        else if (op == OP_COPY) {
            // c[i] = a[i] for n 32-bit words, bit for bit: moves KV rows a block wrote past its
            // ring back to the ring's start.
            __global const uint* A = (__global const uint*)task.a;
            __global uint* C = (__global uint*)task.c;
            for (ulong i = (ulong)gsg * 16 + lane; i < task.n; i += (ulong)n_gsg * 16) C[i] = A[i];
        }
        else if (op == OP_MOE_GATE_UP) {
            // For expert e and row r < m: c[q, r] = w * gelu(a[t] . g[e, r]) * (a[t] . u[e, r])
            // over the plan's entries q (token t, weight w) for e, in plan order; b =
            // gate_up_proj [128, 2m, k], x = plan. With one token a is the token's row; with
            // several, OP_MOE_GATHER's rows (row q for entry q). m and k are multiples of 16.
            __global const float*  X = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        Y = (__global float*)task.c;
            __global const MoEPlan* P = (__global const MoEPlan*)task.x;
            uint n_active = P->n_active;
            load_plan(P, slm_active, slm_offset, lid, lsize);
            uint M = task.m, K = task.k;
            if (task.n == 1) {
                // One token: each active expert has exactly one entry.
                uint per = (K / 8 + SPLIT_K - 1) / SPLIT_K, lo = k_part * per, hi = min(K / 8, lo + per);
                for (uint base = 0; base < n_active * M; base += split_stride) {
                    uint item = base + k_slot * num_wgs + wg_id;
                    bool live = item < n_active * M;
                    uint e = live ? slm_active[item / M] : 0, r = item % M;
                    float2 a = live ? part_dots(W + ((ulong)e * 2 * M + r) * K, W + ((ulong)e * 2 * M + M + r) * K, X, lo, hi, lane)
                                    : (float2)(0.0f);
                    if (lane == 0) { slm_reduce[sg_id] = a.x; slm_reduce2[sg_id] = a.y; }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (k_part == 0 && lane == 0 && live) {
                        float gs = 0.0f, us = 0.0f;
                        for (uint p = 0; p < SPLIT_K; p++) { gs += slm_reduce[sg_id + p]; us += slm_reduce2[sg_id + p]; }
                        uint q = slm_offset[e];
                        Y[(ulong)q * M + r] = P->entry_weight[q] * gelu_tanh(gs) * us;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
            // Several tokens: a is OP_MOE_GATHER's rows in plan order. XMX work items of 16 rows
            // of one active expert, over its entries XMX_TOKENS2 at a time; rows past the
            // expert's entries belong to the next and are not stored.
            else {
                plan_chunks(slm_active, slm_offset, n_active, XMX_TOKENS2, slm_chunk, lid);
                uint entries = slm_offset[NUM_EXPERTS];
                for (uint item = gsg; item < slm_chunk[n_active] * (M / 16); item += n_gsg) {
                    uint c = item / (M / 16), r0 = (item % (M / 16)) * 16;
                    uint i = chunk_expert(slm_chunk, n_active, c), e = slm_active[i];
                    uint q0 = slm_offset[e] + (c - slm_chunk[i]) * XMX_TOKENS2;
                    uint tn = min((uint)XMX_TOKENS2, slm_offset[e + 1] - q0);
                    __global const ushort* g_row = W + ((ulong)e * 2 * M + r0 + lane) * K;
                    float8 g[2], u[2];
                    xmx_rows_2(g_row, g_row + (ulong)M * K, X, q0, entries, K, (tn + 7) / 8, 0, K, g, u);
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            uint j = 8 * mt + r;
                            if (j < tn) {
                                uint q = q0 + j;
                                Y[(ulong)q * M + r0 + lane] = P->entry_weight[q] * gelu_tanh(g[mt][r]) * u[mt][r];
                            }
                        }
                    }
                }
            }
        }
        else if (op == OP_MOE_DOWN) {
            // For expert e and row r < m: c[slot, r] = a[q] . b[e, r] over the plan's entries q
            // for e; b = down_proj [128, m, k], x = plan. m and k are multiples of 16.
            __global const float*  X = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        Y = (__global float*)task.c;
            __global const MoEPlan* P = (__global const MoEPlan*)task.x;
            uint n_active = P->n_active;
            load_plan(P, slm_active, slm_offset, lid, lsize);
            uint M = task.m, K = task.k;
            // XMX work items of 16 rows of one active expert by XMX_TOKENS of its entries; a holds
            // OP_MOE_GATE_UP's rows in plan order.
            plan_chunks(slm_active, slm_offset, n_active, XMX_TOKENS, slm_chunk, lid);
            uint entries = slm_offset[NUM_EXPERTS];
            for (uint item = gsg; item < slm_chunk[n_active] * (M / 16); item += n_gsg) {
                uint c = item / (M / 16), r0 = (item % (M / 16)) * 16;
                uint i = chunk_expert(slm_chunk, n_active, c), e = slm_active[i];
                uint q0 = slm_offset[e] + (c - slm_chunk[i]) * XMX_TOKENS;
                uint tn = min((uint)XMX_TOKENS, slm_offset[e + 1] - q0);
                float8 acc[4];
                xmx_rows_1(W + ((ulong)e * M + r0 + lane) * K, X, q0, entries, K, (tn + 7) / 8, 0, K, acc);
                for (uint mt = 0; mt < 4; mt++) {
                    for (uint r = 0; r < 8; r++) {
                        uint j = 8 * mt + r;
                        if (j < tn) Y[(ulong)P->entry_slot[q0 + j] * M + r0 + lane] = acc[mt][r];
                    }
                }
            }
        }
        else if (op == OP_MEAN_POOL_L2) {
            // c[0..k) = L2-normalized mean of the first k columns of n rows with stride m.
            __global const float* A = (__global const float*)task.a;
            __global float*       C = (__global float*)task.c;
            uint K = task.k, stride = task.m;
            if (wg_id == 0 && sg_id == 0) {
                float ss = 0.0f;
                for (uint i = lane; i < K; i += 16) {
                    float s = 0.0f;
                    for (uint t = 0; t < task.n; t++) s += A[(ulong)t * stride + i];
                    s /= (float)task.n;
                    C[i] = s;
                    ss += s * s;
                }
                float inv = rsqrt(sub_group_reduce_add(ss) + 1e-12f);
                for (uint i = lane; i < K; i += 16) C[i] *= inv;
            }
        }

        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_release, memory_scope_device);
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if (lid == 0 && task.signal_event != NO_EVENT) {
            usm_event_signal(&scoreboard[task.signal_event], 1u);
        }
        tasks_done++;
        last_op = op;
    }

    if (lid == 0) {
        core_stats[wg_id].core_id            = (gpu_id << 16) | wg_id;
        core_stats[wg_id].tasks_executed     = tasks_done;
        core_stats[wg_id].spin_polls         = slm_spin_count;
        core_stats[wg_id].last_opcode        = last_op;
        core_stats[wg_id].xmx_tiles_executed = 0;
    }
}
