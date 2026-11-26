// The ops of the LFM3 vision tower. megakernel.cl includes this file inside the loop of
// mk_persistent_interpreter when MK_LFM3 is defined, beside lfm3_ops.cl, so these branches see its locals. The tower
// also uses OP_MATMUL, OP_RMSNORM, OP_GEGLU (SwiGLU in this binary), OP_LAYERNORM, OP_BIAS and OP_GELU. Its attention
// is OP_VISION_ATTN. OP_BIATTN stays in the binary: the gate of OP_VISION_ATTN compares the two.
//
// The rows of an image are its patches in the order of the merge blocks of 2 by 2 patches, and the blocks go across
// first (image_processing_lfm3_vl.py:186). Row r of a frame of grid_w patches across is patch r % 4 of block r / 4.
// Activations are f32 and token-major. The weights are BF16. The opcodes 100 to 109 are for these ops.

        else if (op == OP_VISION_ROPE) {
            // The norm and the axial rotation of the head rows of the vision attention, in place
            // (modeling_lfm3_vl.py:85-131, 226-235). a = c = the rows [n][k], heads rows for each patch; b = the BF16
            // norm weights [k]; n = rows, m = heads, k = the head size (a multiple of 4); p0 = grid_w |
            // grid_h << 16, the patches of one frame; x = the bits of eps | the bits of theta << 32. Each row is
            // normalized, then the pair (i, i + k/2) turns by the down coordinate times inv_freq(i) for i < k/4, and by
            // the across coordinate times inv_freq(i - k/4) for the rest, with inv_freq(j) = theta^(-2j / (k/2)).
            // One subgroup for each row, as the rotation of OP_RMSNORM.
            __global const float*  A = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float*        C = (__global float*)task.c;
            uint K = task.k, heads = task.m, half_d = K / 2, quarter = K / 4;
            uint grid_w = task.p0 & 0xFFFFu, grid_h = task.p0 >> 16, frame = grid_w * grid_h;
            uint blocks_across = grid_w / 2;
            float eps = as_float((uint)(task.x & 0xFFFFFFFFul));
            float theta = as_float((uint)(task.x >> 32));
            for (uint r = gsg; r < task.n; r += n_gsg) {
                __global const float* x = A + (ulong)r * K;
                __global float*       y = C + (ulong)r * K;
                float ss = 0.0f;
                for (uint i = lane; i < K; i += 16) ss += x[i] * x[i];
                ss = sub_group_reduce_add(ss);
                float inv = rsqrt(ss / (float)K + eps);
                uint patch = (r / heads) % frame, block = patch / 4, within = patch % 4;
                float down = (float)((block / blocks_across) * 2 + within / 2);
                float across = (float)((block % blocks_across) * 2 + within % 2);
                // A lane writes only the pairs that it reads, so the rows may be rotated in place.
                for (uint i = lane; i < half_d; i += 16) {
                    float x0 = x[i] * inv * bf16_to_f32(W[i]);
                    float x1 = x[i + half_d] * inv * bf16_to_f32(W[i + half_d]);
                    uint j = i < quarter ? i : i - quarter;
                    float inv_freq = 1.0f / pow(theta, (float)(2 * j) / (float)half_d);
                    float ang = (i < quarter ? down : across) * inv_freq;
                    float cs = cos(ang), sn = sin(ang);
                    y[i]          = x0 * cs - x1 * sn;
                    y[i + half_d] = x1 * cs + x0 * sn;
                }
            }
        }
        else if (op == OP_VISION_CONV) {
            // The gated depthwise 3 x 3 convolution of a vision conv layer (modeling_lfm3_vl.py:300-338): with
            // u = B * x, c[r, d] = C[r, d] * the sum of w[d][kh][kw] * u[the patch (down + kh - 1, across + kw - 1)],
            // with zeros past the edge of the frame of row r. a = the in_proj rows [n][3 * k] in the chunks B, C and
            // x; b = the BF16 taps [k][9], down then across; c = [n][k]; n = rows, whole frames; k = the channels (a
            // multiple of 16); p0 = grid_w | grid_h << 16 (each even). A lane takes one channel of one row, and sums
            // the taps down then across with fma.
            __global const float*  BCX = (__global const float*)task.a;
            __global const ushort* Wt = (__global const ushort*)task.b;
            __global float*        Y = (__global float*)task.c;
            uint D = task.k, groups = D / 16;
            int grid_w = (int)(task.p0 & 0xFFFFu), grid_h = (int)(task.p0 >> 16);
            uint frame = (uint)(grid_w * grid_h), blocks_across = (uint)grid_w / 2;
            for (uint item = gsg; item < task.n * groups; item += n_gsg) {
                uint r = item / groups, d = (item % groups) * 16 + lane;
                uint base = r - r % frame, patch = r % frame, block = patch / 4, within = patch % 4;
                int down = (int)((block / blocks_across) * 2 + within / 2);
                int across = (int)((block % blocks_across) * 2 + within % 2);
                float acc = 0.0f;
                for (int kh = 0; kh < 3; kh++) {
                    int h = down + kh - 1;
                    if (h < 0 || h >= grid_h) continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int w = across + kw - 1;
                        if (w < 0 || w >= grid_w) continue;
                        uint nb = ((uint)(h / 2) * blocks_across + (uint)(w / 2)) * 4 + (uint)(h % 2) * 2 + (uint)(w % 2);
                        __global const float* row = BCX + (ulong)(base + nb) * 3 * D;
                        float u = row[d] * row[2 * D + d];
                        acc = fma(bf16_to_f32(Wt[(ulong)d * 9 + kh * 3 + kw]), u, acc);
                    }
                }
                Y[(ulong)r * D + d] = BCX[(ulong)r * 3 * D + D + d] * acc;
            }
        }
        else if (op == OP_VISION_ATTN) {
            // The attention of the tower: OP_BIATTN with every key, and the same bits. The fields are OP_BIATTN's, with
            // k = 0 and p0 = D of at most 64. OP_BIATTN reads each key row and each value row one time for each query
            // row and head, with 2 * D / 16 loads. The loads are its cost, and the rows of a large image do not stay
            // in the cache of the card.
            //
            // Here one subgroup takes a tile of VISION_ATTN_TILE query rows of one head. It reads the key rows and the
            // value rows of 8 keys with D / 16 block reads each, one time for the tile. The arithmetic of each query
            // row is OP_BIATTN's, in key order: the dot in 16 lanes, the online softmax, and the sums of the values.
            // The max, the sum and the two exponentials of query row i of the tile are in lane i of one vector.
            __global const float* Q = (__global const float*)task.a;
            __global const float* Kr = (__global const float*)task.b;
            __global const float* V = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint N = task.n, H = task.m & 0xFFFFu, KV = task.m >> 16, D = task.p0;
            uint per = D / 16, group = H / KV;
            float scale = task.layer ? as_float(task.layer) : 1.0f;
            uint tiles = (N + VISION_ATTN_TILE - 1) / VISION_ATTN_TILE;
            // The tiles of one head are next to each other, so the subgroups that run at one time read the same rows.
            for (uint item = gsg; item < H * tiles; item += n_gsg) {
                uint h = item / tiles, t0 = (item % tiles) * VISION_ATTN_TILE, g = h / group;
                float qv[VISION_ATTN_TILE][4], acc[VISION_ATTN_TILE][4];
                _Pragma("unroll") for (uint i = 0; i < VISION_ATTN_TILE; i++) {
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
                    // The 8 keys from y0 hold the keys from s0. The last 8 keys of the plane start before s0 when N is
                    // no multiple of 8, so no read goes past the plane. Fewer than 8 keys are read one row at a time.
                    uint y0 = N >= 8 ? min(s0, N - 8) : 0;
                    // Lane i of sc[j] is the score of query row i with key y0 + j.
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
                            _Pragma("unroll") for (uint i = 0; i < VISION_ATTN_TILE; i++) {
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
                            _Pragma("unroll") for (uint i = 0; i < VISION_ATTN_TILE; i++) {
                                float ci = sub_group_broadcast(corr, i), pi = sub_group_broadcast(p, i);
                                _Pragma("unroll") for (uint r = 0; r < 4; r++) if (r < per) acc[i][r] = acc[i][r] * ci + pi * vb[j][r];
                            }
                        }
                    }
                }
                _Pragma("unroll") for (uint i = 0; i < VISION_ATTN_TILE; i++) {
                    if (t0 + i < N) {
                        float li = sub_group_broadcast(l, i);
                        __global float* out = C + ((ulong)(t0 + i) * H + h) * D;
                        _Pragma("unroll") for (uint r = 0; r < 4; r++) if (r < per) out[lane + 16 * r] = acc[i][r] / li;
                    }
                }
            }
        }
