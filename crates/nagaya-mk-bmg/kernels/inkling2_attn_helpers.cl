// The helpers of the attention of Inkling 2 (MK_INKLING2), at file scope. megakernel.cl includes this file before
// the kernel, and kernels/inkling2_attn_ops.cl holds the ops.
//
// OP_INKLING2_ATTN is the attention of a block of n positions with a relative bias, and OP_INKLING2_ATTN_SUM merges
// its chunks with the sink of each head. A decode step is a block of one position. The ops follow OP_ATTN_STREAM
// of the LFM3 binary: a team of 4 subgroups takes (chunk, head, tile of 16 positions) and walks its chunk in rounds
// of 128 keys, with the XMX sums of three bf16 parts.
//
// The chunks and the rounds sit on a grid of absolute key positions, not on the first key of the block. Chunk c
// holds the keys c * chunk_keys to (c + 1) * chunk_keys - 1. Thus the sums of one query position do not depend on
// the block that holds it. See the op comment for what is bit-equal across splits.

#define INKLING2_HEADS       24
#define INKLING2_KV_HEADS    8
#define INKLING2_HEAD_DIM    128
#define INKLING2_REL_STATES  16    // the relative state of one head (D_REL)
#define INKLING2_REL_EXTENT  1024  // the distances that the bias covers; the proj row of one state is this long
#define INKLING2_WINDOW      1024  // the window of a sliding layer: the key positions q - 1023 to q
#define INKLING2_ATTN_KEYS   128   // the keys of one round: a pair of key tiles for each of the 4 subgroups
#define INKLING2_ATTN_TEAM_FLOATS 2176 // the scratch of one team: probabilities [16][128], maxes [4][16], sums [4][16]

// The lowest key position that query position `pos` reads. A ring (ring > 0) holds a sliding layer.
inline uint inkling2_lowest_key(uint pos, uint ring) {
    return (ring && pos + 1 > INKLING2_WINDOW) ? pos + 1 - INKLING2_WINDOW : 0;
}

// The relative bias of one score: the dot product of the 16 relative states of the query and the column of the
// distance d in proj ([16][1024] bf16, the distance contiguous). Lane i of relv holds state i, and every lane calls
// this function. The order is the order of `relative_logits` in the host form: from 0, state 0 first, one product
// and one sum at each step. FP_CONTRACT OFF keeps the two roundings, so the bias has the bits of the host profile.
inline float inkling2_rel_bias(float relv, __global const ushort* proj, uint d) {
#pragma OPENCL FP_CONTRACT OFF
    float bias = 0.0f;
    for (uint i = 0; i < INKLING2_REL_STATES; i++) {
        float p = bf16_to_f32(proj[i * INKLING2_REL_EXTENT + d]);
        bias = bias + sub_group_broadcast(relv, i) * p;
    }
    return bias;
}

// One score: q . k times 128^-0.5, plus the bias, with two roundings as in the host `attend`. The scale is the f32
// of 128^-0.5, 0x3DB504F3 (`tasks::INKLING2_ATTN_SCALE_BITS`, tested on the host).
inline float inkling2_score(float qk, float bias) {
#pragma OPENCL FP_CONTRACT OFF
    return qk * as_float(0x3DB504F3u) + bias;
}

// The first chunk and the count of chunks of a block of n positions from p0. `tasks::Inkling2Attention` has the
// host copy, which sizes the scratch.
inline uint2 inkling2_attn_chunks(uint n, uint p0, uint chunk_keys, uint ring) {
    uint first = inkling2_lowest_key(p0, ring) / chunk_keys;
    return (uint2)(first, (p0 + n - 1) / chunk_keys - first + 1);
}
