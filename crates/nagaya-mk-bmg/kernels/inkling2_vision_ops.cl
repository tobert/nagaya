// The ops of the vision tower of Inkling 2 (MK_INKLING2): opcodes 180 to 189. inkling2_ops.cl includes this file
// inside the task loop of megakernel.cl, so each op sees the locals of the loop. The opcodes must match
// src/protocol.rs. The host form is crates/nagaya-mk-inkling2-bmg/src/host/vision.rs, and the helpers are in
// kernels/inkling2_vision_helpers.cl.
//
// The tower also uses OP_MATMUL, OP_BIAS, OP_LAYERNORM, OP_GELU and OP_SCALE_ADD. The rows of one image are its class
// row, then its 4T patches in the order of the merge (engine-abi "The images of Inkling 2"). Activations are f32
// and row-major, 1,024 values in a row.

        else if (op == OP_INKLING2_VISION_EMBED) {
            // The position embedding of one image (host `Vision::tower`, the rows before ln_pre): c[1 + p] = a[p] +
            // the bilinear sum of 4 rows of the table b for patch p. Row 0 of c, the class row, is an OP_COPY before
            // this op. The fields: a = the patch embedding [n][k] f32; b = the table [32 * 32][k] bf16; x = the taps
            // of the image, from the host (`vision::axes`): for each patch line, then for each patch column, the two
            // taps as uint and their two weights as f32 bits; c = [n + 1][k]; n = 4T patches; k = the width;
            // p0 = the rows | the columns << 16 of the image, in image tokens. A lane takes one value. The weights
            // are the host's, and the 4 taps sum in the order of the host from 0, with no fused multiply-add. Thus
            // the op gives the bits of the host form.
#pragma OPENCL FP_CONTRACT OFF
            __global const float*  A = (__global const float*)task.a;
            __global const ushort* T = (__global const ushort*)task.b;
            __global const uint4*  AX = (__global const uint4*)task.x;
            __global float*        C = (__global float*)task.c;
            uint K = task.k, rows = task.p0 & 0xFFFFu, columns = task.p0 >> 16;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16u + lane; i < total; i += (ulong)n_gsg * 16u) {
                uint p = (uint)(i / K), j = (uint)(i % K);
                uint2 at = inkling2_patch_position(p, columns);
                uint4 h = AX[at.x], w = AX[2u * rows + at.y];
                float pos = 0.0f;
                for (uint hy = 0; hy < 2u; hy++) {
                    uint ht = hy ? h.y : h.x;
                    float hw = as_float(hy ? h.w : h.z);
                    for (uint wx = 0; wx < 2u; wx++) {
                        uint wt = wx ? w.y : w.x;
                        float ww = as_float(wx ? w.w : w.z);
                        float weight = hw * ww;
                        float v = bf16_to_f32(T[(ulong)(ht * INKLING2_VISION_GRID + wt) * K + j]);
                        pos = pos + v * weight;
                    }
                }
                C[i + K] = A[i] + pos;
            }
        }
        else if (op == OP_INKLING2_VISION_ROPE) {
            // The rotary embedding in two axes of the queries a and the keys b of one image, in place (host `rope`
            // and `rope_angle`). Row 0 is the class row at (0, 0), and row 1 + p is patch p at (line + 1,
            // column + 1). Pair j of a head of 64 turns by the column for j < 16 and by the line for j >= 16, with
            // frequency j mod 16. The fields: n = the rows, 4T + 1; k = the width, 1,024 (16 heads of 64);
            // p0 = the columns of the image in image tokens; x = the table [207][16] of cosines, then [207][16] of
            // sines, which the host computes as its host form does. A lane takes one pair of one row of a or b, and
            // writes only that pair. With the table, the op gives the bits of the host form.
#pragma OPENCL FP_CONTRACT OFF
            __global float*       Q = (__global float*)task.a;
            __global float*       Kr = (__global float*)task.b;
            __global const float* COS = (__global const float*)task.x;
            __global const float* SIN = COS + INKLING2_VISION_POSITIONS * INKLING2_VISION_FREQS;
            uint pairs = task.k / 2u, columns = task.p0;
            ulong per = (ulong)task.n * pairs;
            for (ulong i = (ulong)gsg * 16u + lane; i < 2u * per; i += (ulong)n_gsg * 16u) {
                __global float* rows = i < per ? Q : Kr;
                ulong e = i < per ? i : i - per;
                uint r = (uint)(e / pairs), pair = (uint)(e % pairs), j = pair % 32u;
                uint line = 0, column = 0;
                if (r > 0) {
                    uint2 at = inkling2_patch_position(r - 1u, columns);
                    line = at.x + 1u;
                    column = at.y + 1u;
                }
                uint p = j < INKLING2_VISION_FREQS ? column : line;
                uint t = p * INKLING2_VISION_FREQS + j % INKLING2_VISION_FREQS;
                float c = COS[t], s = SIN[t];
                __global float* v = rows + (ulong)r * task.k + 2u * pair;
                float x1 = v[0], x2 = v[1];
                v[0] = x1 * c + -x2 * s;
                v[1] = x2 * c + x1 * s;
            }
        }
        else if (op == OP_INKLING2_VISION_ATTN) {
            // The attention of the tower in both directions over the n rows of one image: OP_VISION_ATTN of the LFM3
            // binary, copied under this opcode, because this binary does not hold the LFM3 ops. The fields:
            // a = the queries [n][H][D], b = the keys and x = the values [n][KV][D], c = [n][H][D]; m = H | KV << 16;
            // p0 = D, 16 to 64 in 16s; layer = the bits of the scale. One subgroup takes a tile of
            // INKLING2_VISION_ATTN_TILE query rows of one head, and it reads each key row one time for the tile. The
            // arithmetic of a query row: the dot in 16 lanes, the online softmax and the sums of the values, in key
            // order. Thus a row does not read the other rows of its tile.
            __global const float* Q = (__global const float*)task.a;
            __global const float* Kr = (__global const float*)task.b;
            __global const float* V = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint N = task.n, H = task.m & 0xFFFFu, KV = task.m >> 16, D = task.p0;
            uint per = D / 16, group = H / KV;
            float scale = task.layer ? as_float(task.layer) : 1.0f;
            uint tiles = (N + INKLING2_VISION_ATTN_TILE - 1) / INKLING2_VISION_ATTN_TILE;
            for (uint item = gsg; item < H * tiles; item += n_gsg) {
                uint h = item / tiles, t0 = (item % tiles) * INKLING2_VISION_ATTN_TILE, g = h / group;
                float qv[INKLING2_VISION_ATTN_TILE][4], acc[INKLING2_VISION_ATTN_TILE][4];
                _Pragma("unroll") for (uint i = 0; i < INKLING2_VISION_ATTN_TILE; i++) {
                    // A tile past the last row reads the last row again, and it writes nothing for it.
                    uint t = min(t0 + i, N - 1);
                    __global const float* q = Q + ((ulong)t * H + h) * D;
                    _Pragma("unroll") for (uint r = 0; r < 4; r++) {
                        qv[i][r] = r < per ? q[lane + 16 * r] : 0.0f;
                        acc[i][r] = 0.0f;
                    }
                }
                // Lane i holds the max and the sum of query row i.
                float mx = -INFINITY, l = 0.0f;
                for (uint s0 = 0; s0 < N; s0 += 8) {
                    // The 8 keys from y0 hold the keys from s0. The last 8 keys start before s0 when N is no multiple
                    // of 8, so no read goes past the plane. Fewer than 8 keys are read one row at a time.
                    uint y0 = N >= 8 ? min(s0, N - 8) : 0;
                    float sc[8];
                    {
                        float kb[8][4];
                        _Pragma("unroll") for (uint r = 0; r < 4; r++) {
                            uint rows[8];
                            _Pragma("unroll") for (uint j = 0; j < 8; j++) rows[j] = 0;
                            if (r < per && N >= 8) {
                                intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)Kr, KV * D * 4, N, KV * D * 4, (int2)(g * D + 16 * r, y0), rows);
                            } else if (r < per) {
                                _Pragma("unroll") for (uint j = 0; j < 8; j++)
                                    rows[j] = as_uint(Kr[((ulong)min(j, N - 1) * KV + g) * D + lane + 16 * r]);
                            }
                            _Pragma("unroll") for (uint j = 0; j < 8; j++) kb[j][r] = as_float(rows[j]);
                        }
                        _Pragma("unroll") for (uint j = 0; j < 8; j++) {
                            float score = 0.0f;
                            _Pragma("unroll") for (uint i = 0; i < INKLING2_VISION_ATTN_TILE; i++) {
                                float dot = 0.0f;
                                _Pragma("unroll") for (uint r = 0; r < 4; r++) if (r < per) dot += qv[i][r] * kb[j][r];
                                dot = sub_group_reduce_add(dot) * scale;
                                score = lane == i ? dot : score;
                            }
                            sc[j] = score;
                        }
                    }
                    float vb[8][4];
                    _Pragma("unroll") for (uint r = 0; r < 4; r++) {
                        uint rows[8];
                        _Pragma("unroll") for (uint j = 0; j < 8; j++) rows[j] = 0;
                        if (r < per && N >= 8) {
                            intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)V, KV * D * 4, N, KV * D * 4, (int2)(g * D + 16 * r, y0), rows);
                        } else if (r < per) {
                            _Pragma("unroll") for (uint j = 0; j < 8; j++)
                                rows[j] = as_uint(V[((ulong)min(j, N - 1) * KV + g) * D + lane + 16 * r]);
                        }
                        _Pragma("unroll") for (uint j = 0; j < 8; j++) vb[j][r] = as_float(rows[j]);
                    }
                    _Pragma("unroll") for (uint j = 0; j < 8; j++) {
                        uint s = y0 + j;
                        if (s >= s0 && s < N) {
                            float mnew = fmax(mx, sc[j]);
                            float corr = exp(mx - mnew), p = exp(sc[j] - mnew);
                            l = l * corr + p;
                            mx = mnew;
                            _Pragma("unroll") for (uint i = 0; i < INKLING2_VISION_ATTN_TILE; i++) {
                                float ci = sub_group_broadcast(corr, i), pi = sub_group_broadcast(p, i);
                                _Pragma("unroll") for (uint r = 0; r < 4; r++) if (r < per) acc[i][r] = acc[i][r] * ci + pi * vb[j][r];
                            }
                        }
                    }
                }
                _Pragma("unroll") for (uint i = 0; i < INKLING2_VISION_ATTN_TILE; i++) {
                    if (t0 + i < N) {
                        float li = sub_group_broadcast(l, i);
                        __global float* out = C + ((ulong)(t0 + i) * H + h) * D;
                        _Pragma("unroll") for (uint r = 0; r < 4; r++) if (r < per) out[lane + 16 * r] = acc[i][r] / li;
                    }
                }
            }
        }
