// The ops of the LFM3 engine (milestone 24 step 5). megakernel.cl includes this file inside the loop of
// mk_persistent_interpreter when MK_LFM3 is defined, so these branches see its locals, as the ops of qwen3tts_ops.cl do.
// The LFM3 binary also holds the speech ops: its OP_GEGLU is SwiGLU, and OP_RMSNORM takes eps and a partial rotation.
//
// Activations are f32 and token-major: row t of a tensor is one position. The weights are BF16, or F32 with FLAG_F32.
// Each op gives the same bits for a position in any split of a feed, so a decode step equals the same position in a
// prefill.

        else if (op == OP_SHORT_CONV) {
            // The gated short convolution of LFM3 after in_proj (modeling_lfm3.py:371-373, 422): with u = B * x,
            // c[t, d] = C[t, d] * the sum over the taps j of w[d, j] * u[t - (taps - 1) + j, d]. a = the in_proj rows
            // [n][3 * D], in the chunks B, C and x; b = w [D][taps]; x = the state [taps - 1][D], the last inputs u
            // of the quantum before, oldest first; c = [n][D]. n = positions, k = D (a multiple of 16), layer = taps
            // (2 to 8). The op reads the state and writes the last taps - 1 values of u into it.
            // A lane takes one channel of a chunk of SHORT_CONV_CHUNK positions, and sums its taps from the oldest
            // input with fma. Only chunk 0 reads the old state, so it writes the new state after its reads.
            __global const float* BCX = (__global const float*)task.a;
            __global const ushort* Wb = (__global const ushort*)task.b;
            __global const float* Wf = (__global const float*)task.b;
            __global float* S = (__global float*)task.x;
            __global float* Y = (__global float*)task.c;
            uint T = task.n, D = task.k, taps = task.layer, hist = taps - 1;
            bool wf32 = (flags & FLAG_F32) != 0;
            uint groups = D / 16, chunks = (T + SHORT_CONV_CHUNK - 1) / SHORT_CONV_CHUNK;
            for (uint item = gsg; item < groups * chunks; item += n_gsg) {
                uint ch = item / groups, d = (item % groups) * 16 + lane;
                uint t0 = ch * SHORT_CONV_CHUNK, t1 = min(T, t0 + SHORT_CONV_CHUNK);
                float w[SHORT_CONV_MAX_TAPS], win[SHORT_CONV_MAX_TAPS], old[SHORT_CONV_MAX_TAPS];
                for (uint j = 0; j < taps; j++) {
                    ulong at = (ulong)d * taps + j;
                    w[j] = wf32 ? Wf[at] : bf16_to_f32(Wb[at]);
                }
                // The window before position t0: the inputs t0 - hist .. t0 - 1, from the state before position 0.
                for (uint j = 0; j < hist; j++) {
                    int s = (int)t0 - (int)hist + (int)j;
                    __global const float* row = BCX + (ulong)max(s, 0) * 3 * D;
                    win[j] = s >= 0 ? row[d] * row[2 * D + d] : S[(ulong)(hist + s) * D + d];
                    old[j] = win[j];
                }
                for (uint t = t0; t < t1; t++) {
                    __global const float* row = BCX + (ulong)t * 3 * D;
                    win[hist] = row[d] * row[2 * D + d];
                    float acc = 0.0f;
                    for (uint j = 0; j < taps; j++) acc = fma(w[j], win[j], acc);
                    Y[(ulong)t * D + d] = row[D + d] * acc;
                    for (uint j = 0; j < hist; j++) win[j] = win[j + 1];
                }
                if (ch == 0) {
                    // The new state: the inputs T - hist .. T - 1. With T < hist its first rows come from the old state.
                    for (uint j = 0; j < hist; j++) {
                        int s = (int)T - (int)hist + (int)j;
                        __global const float* row = BCX + (ulong)max(s, 0) * 3 * D;
                        S[(ulong)j * D + d] = s >= 0 ? row[d] * row[2 * D + d] : old[hist + s];
                    }
                }
            }
        }
        else if (op == OP_CONTROL) {
            // One control item on the residual (engine-abi "Control vectors"), in place: a = n rows h of k values;
            // b = the vector v [k] of the layer; p0 = 1 add (h + s * v), 2 ablate (h - s * (h.u) * u), or 3 clamp
            // (h + (s - h.u) * u), with u = v / |v|; m = the bits of s. A work-group takes a row, and wg_sum adds the
            // dot products, so the bits of a row do not depend on the count of rows. A vector of zeros skips the row.
            __global float* H = (__global float*)task.a;
            __global const float* V = (__global const float*)task.b;
            uint K = task.k, how = task.p0;
            float s = as_float(task.m);
            for (uint r = wg_id; r < task.n; r += num_wgs) {
                __global float* h = H + (ulong)r * K;
                float hv = 0.0f, vv = 0.0f;
                for (uint i = lid; i < K; i += lsize) {
                    hv = fma(h[i], V[i], hv);
                    vv = fma(V[i], V[i], vv);
                }
                hv = wg_sum(hv, slm_reduce, lane, sg_id, num_sgs);
                vv = wg_sum(vv, slm_reduce, lane, sg_id, num_sgs);
                if (vv == 0.0f) continue;
                float norm = sqrt(vv), along = hv / norm;
                float coef = how == 1 ? s : how == 2 ? -s * along / norm : (s - along) / norm;
                for (uint i = lid; i < K; i += lsize) h[i] = fma(coef, V[i], h[i]);
            }
        }
        else if (op == OP_HEAD_GATE) {
            // The sigmoid gate of each head (modeling_lfm3.py:270-271): c[t, h * D + j] = a[t, h * D + j] *
            // sigmoid(b[t * p0 + h]). n = positions, m = heads, k = D, p0 = the stride of the rows of gates (0: heads).
            // a may be c.
            __global const float* A = (__global const float*)task.a;
            __global const float* G = (__global const float*)task.b;
            __global float* C = (__global float*)task.c;
            uint H = task.m, D = task.k, stride = task.p0 ? task.p0 : task.m;
            ulong total = (ulong)task.n * H * D;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                ulong t = i / ((ulong)H * D);
                uint h = (uint)((i / D) % H);
                float g = G[t * stride + h];
                C[i] = A[i] * (1.0f / (1.0f + exp(-g)));
            }
        }
        else if (op == OP_ATTN_STREAM) {
            // Multi-token attention with no score buffer: the partial sums of each chunk of keys, for n tokens. The
            // fields are OP_ATTN_SCORES'. scratch gets [chunk][t][h][max, sum, o[k]], then the scratch of each team
            // (ATTN_STREAM_TEAM_FLOATS). k is a multiple of 32 up to 128, and a ring has at least ATTN_STREAM_KEYS
            // rows. Work items: (chunk, head, tile of tokens), each for a team of 4 subgroups; subgroup mem takes
            // the key pair mem of each round and the output values 32 * mem. Each team of a work-group runs the
            // same count of items and rounds, so each subgroup meets each barrier.
            __global const float* Q  = (__global const float*)task.a;
            __global const float* Kc = (__global const float*)task.b;
            uint D = task.k, H = task.m & 0xFFFFu, KV = task.m >> 16, N = task.n, p0 = task.p0;
            uint window = (uint)(task.x & 0xFFFFFFFFul);
            ulong max_seq = task.x >> 32;
            uint ring = task.layer;
            __global const float* Vc = Kc + kv_plane_rows(max_seq, ring) * KV * D;
            uint key_rows = p0 + N, k_height = kv_surface_rows(key_rows, ring);
            uint t_tiles = (N + ATTN_TILE - 1) / ATTN_TILE;
            uint first = attn_stream_first(N, p0, window, token_meta, lane);
            uint span_keys = (key_rows + ATTN_STREAM_KEYS - 1) / ATTN_STREAM_KEYS * ATTN_STREAM_KEYS - first;
            uint rounds = attn_stream_rounds(H, N, span_keys), ck = ATTN_STREAM_KEYS * rounds;
            uint chunks = (span_keys + ck - 1) / ck;
            ulong part_floats = ((ulong)chunks * N * H * (D + 2) + 15) / 16 * 16;
            uint mem = sg_id & 3u, team = (sg_id >> 2) * num_wgs + wg_id, n_teams = (num_sgs >> 2) * num_wgs;
            // The team's probabilities of one round, [ATTN_TILE tokens][ATTN_STREAM_KEYS keys]; then the max and
            // the sum of each token over the key pair of each subgroup.
            __global float* tprob = scratch + part_floats + (ulong)team * ATTN_STREAM_TEAM_FLOATS;
            __global float* tmax = tprob + ATTN_TILE * ATTN_STREAM_KEYS;
            __global float* tsum = tmax + 4 * ATTN_TILE;
            uint total = chunks * H * t_tiles, turns = (total + n_teams - 1) / n_teams;
            for (uint turn = 0; turn < turns; turn++) {
                uint item = team + turn * n_teams;
                bool active = item < total;
                uint tt = item % t_tiles, h = (item / t_tiles) % H, c = item / (t_tiles * H);
                uint t0 = tt * ATTN_TILE, kvh = h / (H / KV), c_lo = first + c * ck;
                uint2 span = attn_tile_keys(t0, N, p0, window, token_meta, lane);
                // The key tiles [j_lo, j_hi) of the tile's span in this chunk. A team past the last item has none.
                uint j_lo = active ? max(span.x, c_lo) : 0u, j_hi = active ? min(span.y, c_lo + ck) : 0u;
                // Every token of the tile reads each key of [all_lo, all_hi] that is not in [gap_lo, gap_hi).
                uint all_lo = 0, all_hi = 0xFFFFFFFFu, gap_lo = 0xFFFFFFFFu, gap_hi = 0;
                for (uint r = 0; r < ATTN_TILE; r++) {
                    uint4 rk = attn_token_keys(min(t0 + r, N - 1), p0, window, token_meta);
                    all_lo = max(all_lo, rk.x);
                    all_hi = min(all_hi, rk.y);
                    if (rk.z < rk.w) { gap_lo = min(gap_lo, rk.z); gap_hi = max(gap_hi, rk.w); }
                }
                // The running max and sum of each token, and the output of this subgroup's 32 values before the
                // division by the sum.
                float run_max[ATTN_TILE], run_sum[ATTN_TILE];
                for (uint r = 0; r < ATTN_TILE; r++) { run_max[r] = -INFINITY; run_sum[r] = 0.0f; }
                float8 o[2][2];
                for (uint a = 0; a < 2; a++) for (uint b = 0; b < 2; b++) o[a][b] = 0.0f;
                // The cache row of the first key of the round, stepped with the round: no modulo in the loop.
                uint row0 = sub_group_broadcast((uint)kv_row(c_lo, ring), 0);
                for (uint rd = 0; rd < rounds; rd++) {
                    uint base = c_lo + rd * ATTN_STREAM_KEYS, jp = base + 32 * mem;
                    bool mine = jp < j_hi && jp + 32 > j_lo;
                    // Step 1: the scores of this subgroup's key pair, as OP_ATTN_SCORES takes them. A key that a
                    // token must not read gets the score -INFINITY.
                    float8 sc[2][2];
                    for (uint a = 0; a < 2; a++) for (uint b = 0; b < 2; b++) sc[a][b] = 0.0f;
                    if (mine) {
                        uint krow[2];
                        for (uint nt = 0; nt < 2; nt++) {
                            krow[nt] = row0 + 32 * mem + 16 * nt;
                            if (ring && krow[nt] >= ring) krow[nt] -= ring;
                        }
                        for (uint k0 = 0; k0 < D; k0 += 16) {
                            short8 qh[2], qm[2], ql[2];
                            _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) {
                                uint qv[8];
                                intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)Q, H * D * 4, N, H * D * 4, (int2)(h * D + k0, t0 + 8 * mt), qv);
                                split3_rows(qv, &qh[mt], &qm[mt], &ql[mt]);
                            }
                            _Pragma("unroll") for (uint nt = 0; nt < 2; nt++) {
                                uint kv[16];
                                int kr = (int)krow[nt];
                                intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)Kc, KV * D * 4, k_height, KV * D * 4, (int2)(kvh * D + k0, kr), kv);
                                intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void*)Kc, KV * D * 4, k_height, KV * D * 4, (int2)(kvh * D + k0 + 8, kr), kv + 8);
                                int8 kh, km, kl;
                                split3_pairs(kv, &kh, &km, &kl);
                                _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) sc[mt][nt] = dpas_split6(qh[mt], qm[mt], ql[mt], kh, km, kl, sc[mt][nt]);
                            }
                        }
                        for (uint nt = 0; nt < 2; nt++) {
                            uint jt = jp + 16 * nt, j = jt + lane;
                            bool whole = jt >= all_lo && jt + 15 <= all_hi && (gap_hi <= jt || gap_lo >= jt + 16);
                            if (!whole) {
                                for (uint mt = 0; mt < 2; mt++) {
                                    for (uint r = 0; r < 8; r++) {
                                        uint4 rk = attn_token_keys(min(t0 + 8 * mt + r, N - 1), p0, window, token_meta);
                                        bool valid = j >= rk.x && j <= rk.y && !(j >= rk.z && j < rk.w);
                                        sc[mt][nt][r] = valid ? sc[mt][nt][r] : -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            float m = mine ? sub_group_reduce_max(fmax(sc[mt][0][r], sc[mt][1][r])) : -INFINITY;
                            if (lane == 0) tmax[mem * ATTN_TILE + 8 * mt + r] = m;
                        }
                    }
                    barrier(CLK_GLOBAL_MEM_FENCE);
                    // Step 2: the new max of each token over the round, and the probabilities before the division
                    // by their sum, exp(score - max), for the team.
                    float new_max[ATTN_TILE];
                    for (uint r = 0; r < ATTN_TILE; r++) {
                        float m = fmax(fmax(tmax[r], tmax[ATTN_TILE + r]), fmax(tmax[2 * ATTN_TILE + r], tmax[3 * ATTN_TILE + r]));
                        new_max[r] = fmax(run_max[r], m);
                    }
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            float sum = 0.0f;
                            if (mine) {
                                __global float* prow = tprob + (8 * mt + r) * ATTN_STREAM_KEYS + 32 * mem;
                                float m = new_max[8 * mt + r];
                                for (uint nt = 0; nt < 2; nt++) {
                                    float s = sc[mt][nt][r];
                                    float pr = s > -INFINITY ? exp(s - m) : 0.0f;
                                    prow[16 * nt + lane] = pr;
                                    sum += pr;
                                }
                                sum = sub_group_reduce_add(sum);
                            }
                            if (lane == 0) tsum[mem * ATTN_TILE + 8 * mt + r] = sum;
                        }
                    }
                    barrier(CLK_GLOBAL_MEM_FENCE);
                    // Step 3: the old sums at the new max, then the sums of the round against this subgroup's V.
                    if (active && 32 * mem < D) {
                        float8 corr[2];
                        for (uint mt = 0; mt < 2; mt++) {
                            for (uint r = 0; r < 8; r++) {
                                uint tr = 8 * mt + r;
                                float cr = run_max[tr] == new_max[tr] ? 1.0f : exp(run_max[tr] - new_max[tr]);
                                float sum = ((tsum[tr] + tsum[ATTN_TILE + tr]) + tsum[2 * ATTN_TILE + tr]) + tsum[3 * ATTN_TILE + tr];
                                run_sum[tr] = run_sum[tr] * cr + sum;
                                run_max[tr] = new_max[tr];
                                corr[mt][r] = cr;
                            }
                        }
                        for (uint mt = 0; mt < 2; mt++) for (uint nt = 0; nt < 2; nt++) o[mt][nt] *= corr[mt];
                        for (uint ti = 0; ti < ATTN_STREAM_KEYS / 16; ti++) {
                            uint jt = base + 16 * ti;
                            if (jt < j_lo || jt >= j_hi) continue;
                            uint vy = row0 + 16 * ti;
                            if (ring && vy >= ring) vy -= ring;
                            short8 ph[2], pm[2], pl[2];
                            _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) {
                                uint pv[8];
                                intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)tprob, ATTN_STREAM_KEYS * 4, ATTN_TILE, ATTN_STREAM_KEYS * 4, (int2)(16 * ti, 8 * mt), pv);
                                split3_rows(pv, &ph[mt], &pm[mt], &pl[mt]);
                            }
                            _Pragma("unroll") for (uint nt = 0; nt < 2; nt++) {
                                uint vv[16];
                                intel_sub_group_2d_block_read_32b_16r16x1c((__global void*)Vc, KV * D * 4, k_height, KV * D * 4, (int2)(kvh * D + 32 * mem + 16 * nt, (int)vy), vv);
                                int8 vh, vm, vl;
                                split3_pairs(vv, &vh, &vm, &vl);
                                _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) o[mt][nt] = dpas_split6(ph[mt], pm[mt], pl[mt], vh, vm, vl, o[mt][nt]);
                            }
                        }
                    }
                    row0 += ATTN_STREAM_KEYS;
                    if (ring && row0 >= ring) row0 -= ring;
                }
                if (active && 32 * mem < D) {
                    __global float* part = scratch + (((ulong)c * N + t0) * H + h) * (D + 2);
                    ulong part_stride = (ulong)H * (D + 2);
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            uint t = t0 + 8 * mt + r;
                            if (t < N) {
                                __global float* pp = part + (ulong)(8 * mt + r) * part_stride;
                                if (mem == 0 && lane == 0) { pp[0] = run_max[8 * mt + r]; pp[1] = run_sum[8 * mt + r]; }
                                pp[2 + 32 * mem + lane] = o[mt][0][r];
                                pp[2 + 32 * mem + 16 + lane] = o[mt][1][r];
                            }
                        }
                    }
                }
            }
        }
        else if (op == OP_ATTN_STREAM_SUM) {
            // Merges OP_ATTN_STREAM's chunks from scratch into c[t, h, :] in chunk order, as OP_ATTN_COMBINE merges
            // the chunks of one token; fields as OP_ATTN_STREAM's. A chunk with the sum 0 holds no key of its
            // token. Work items: (token, head, 16 output values), one value per lane.
            __global float* O = (__global float*)task.c;
            uint D = task.k, H = task.m & 0xFFFFu, N = task.n, p0 = task.p0;
            uint window = (uint)(task.x & 0xFFFFFFFFul);
            uint first = attn_stream_first(N, p0, window, token_meta, lane);
            uint span_keys = (p0 + N + ATTN_STREAM_KEYS - 1) / ATTN_STREAM_KEYS * ATTN_STREAM_KEYS - first;
            uint ck = ATTN_STREAM_KEYS * attn_stream_rounds(H, N, span_keys), chunks = (span_keys + ck - 1) / ck;
            for (uint item = gsg; item < N * H * (D / 16); item += n_gsg) {
                uint d = (item % (D / 16)) * 16 + lane, th = item / (D / 16);
                float m = -INFINITY, l = 0.0f, o = 0.0f;
                for (uint c = 0; c < chunks; c++) {
                    __global const float* pp = scratch + ((ulong)c * N * H + th) * (D + 2);
                    float mc = pp[0], lc = pp[1];
                    if (lc > 0.0f) {
                        float nm = fmax(m, mc);
                        float a = exp(m - nm), b = exp(mc - nm);
                        l = l * a + lc * b;
                        o = o * a + pp[2 + d] * b;
                        m = nm;
                    }
                }
                O[(ulong)th * D + d] = o * (1.0f / l);
            }
        }
