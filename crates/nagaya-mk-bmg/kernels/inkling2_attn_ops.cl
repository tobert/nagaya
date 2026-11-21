// The attention ops of the Inkling 2 binary (MK_INKLING2). Each op is one `else if (op == ...)` branch of the task
// loop of megakernel.cl, through kernels/inkling2_ops.cl, so each op sees the locals of the loop. The opcodes must
// match src/protocol.rs. The host form is `attend`, `relative_logits` and `relative_bias` in
// crates/nagaya-mk-inkling2-bmg/src/host/ops.rs. The helpers are in kernels/inkling2_attn_helpers.cl.
//
// Bits and splits. The nonzero terms of one query position read only that position's query, relative states and
// sink, the keys and values that it reads, and the grid of chunks and rounds, which sits on absolute key positions.
// The merge visits the chunks in increasing key order and skips each chunk with no key of the position. Thus with
// the same chunk_keys, a position gets its terms in the same order in a block of any size or start, and in a decode
// step (a block of one position). The equal bits also need three things that the source cannot prove:
// - A dpas row does not read the other rows, and a zero product leaves the accumulator as it was. The other rows of
//   a tile, and the key tiles that another position of the tile reads, add probabilities of exactly 0.
// - Each value row below the key count of the block is finite, because 0 * NaN is NaN. The engine writes each row
//   before the key count passes it, and the surface height reads the rows past the key count as zeros.
// - A subgroup reduction gives the same bits for the same lanes in each place of a tile.
// What changes bits: another chunk_keys, other inputs (the projections before this op sum other blocks), and a sign
// of zero in an output whose terms are all zero. The card test `xe_attention` counts the values that another split
// changes. Until it shows 0, the rule of the engine is "another split gives a small gap" (engine-abi, "Bits and
// splits").

        else if (op == OP_INKLING2_ATTN) {
            // The partial sums of each chunk of keys for n query positions p0 .. p0 + n - 1, as OP_ATTN_STREAM takes
            // them, with the relative bias of each score. The fields:
            // - a: the queries [n][24][128] f32, after the clip and the log scale.
            // - b: the key plane, rows of [8][128] f32. The value plane follows it at m rows (cells) or at
            //   ring + KV_RING_SPILL rows (a ring). Key position j is row j, or row j % ring in a ring.
            // - c: the relative states [n][24][16] f32, after the clip and the log scale.
            // - x: proj [16][1024] bf16, the distance contiguous, as the checkpoint holds it.
            // - m: the rows of a cell plane (the slot's positions); a ring does not read it.
            // - n: 1 to 512 positions. p0: the position of the first. k: the keys of one chunk, a multiple of 128.
            // - layer: the rows of a ring, or 0 for cells. A ring is a sliding layer: a query reads 1,024 keys.
            // The keys and values of the block are in the planes before the op runs. A ring reads only rows
            // j % ring, so the spill rows of the block must be in place.
            // scratch gets [chunk][t][h][max, sum, o[128]], then INKLING2_ATTN_TEAM_FLOATS for each team.
            // Work items: (chunk, head, tile of 16 positions), each for a team of 4 subgroups. Each team of a
            // work-group runs the same count of items and rounds, so each subgroup meets each barrier.
            __global const float*  Q    = (__global const float*)task.a;
            __global const float*  Kc   = (__global const float*)task.b;
            __global const float*  REL  = (__global const float*)task.c;
            __global const ushort* PROJ = (__global const ushort*)task.x;
            const uint H = INKLING2_HEADS, KV = INKLING2_KV_HEADS, D = INKLING2_HEAD_DIM;
            uint N = task.n, p0 = task.p0, ck = task.k, ring = task.layer;
            __global const float* Vc = Kc + kv_plane_rows((ulong)task.m, ring) * KV * D;
            uint key_rows = p0 + N, k_height = kv_surface_rows(key_rows, ring);
            uint t_tiles = (N + 15) / 16;
            uint2 cuts = inkling2_attn_chunks(N, p0, ck, ring);
            uint c_first = cuts.x, chunks = cuts.y;
            // The rounds of each item: enough for the keys of the block in the chunk that holds the most of them.
            uint rounds = (min(ck, key_rows - c_first * ck) + INKLING2_ATTN_KEYS - 1) / INKLING2_ATTN_KEYS;
            ulong part_floats = ((ulong)chunks * N * H * (D + 2) + 15) / 16 * 16;
            uint mem = sg_id & 3u, team = (sg_id >> 2) * num_wgs + wg_id, n_teams = (num_sgs >> 2) * num_wgs;
            // The team's probabilities of one round, [16 positions][128 keys]; then the max and the sum of each
            // position over the key pair of each subgroup.
            __global float* tprob = scratch + part_floats + (ulong)team * INKLING2_ATTN_TEAM_FLOATS;
            __global float* tmax = tprob + 16 * INKLING2_ATTN_KEYS;
            __global float* tsum = tmax + 4 * 16;
            uint total = chunks * H * t_tiles, turns = (total + n_teams - 1) / n_teams;
            for (uint turn = 0; turn < turns; turn++) {
                uint item = team + turn * n_teams;
                bool active = item < total;
                uint tt = item % t_tiles, h = (item / t_tiles) % H, c = item / (t_tiles * H);
                uint t0 = tt * 16, kvh = h / (H / KV), c_lo = (c_first + c) * ck;
                uint t_last = min(t0 + 15, N - 1);
                // The 16-aligned key tiles [j_lo, j_hi) that the positions of the tile read, in this chunk. A team
                // past the last item has none.
                uint span_lo = inkling2_lowest_key(p0 + t0, ring) / 16 * 16, span_hi = (p0 + t_last) / 16 * 16 + 16;
                uint j_lo = active ? max(span_lo, c_lo) : 0u, j_hi = active ? min(span_hi, c_lo + ck) : 0u;
                float run_max[16], run_sum[16];
                for (uint r = 0; r < 16; r++) { run_max[r] = -INFINITY; run_sum[r] = 0.0f; }
                float8 o[2][2];
                for (uint a = 0; a < 2; a++) for (uint b = 0; b < 2; b++) o[a][b] = 0.0f;
                // The ring row of the first key of the round, stepped with the round: no modulo in the loop. A
                // round never crosses the end of a ring, because the ring and the grid are multiples of 128.
                uint row0 = sub_group_broadcast(active ? (uint)kv_row(c_lo, ring) : 0u, 0);
                for (uint rd = 0; rd < rounds; rd++) {
                    uint base = c_lo + rd * INKLING2_ATTN_KEYS, jp = base + 32 * mem;
                    bool mine = jp < j_hi && jp + 32 > j_lo;
                    // Step 1: the sums q . k of this subgroup's key pair, then the scale, the bias and the mask. A
                    // key that a position does not read gets the score -INFINITY.
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
                        for (uint mt = 0; mt < 2; mt++) {
                            for (uint r = 0; r < 8; r++) {
                                uint t = t0 + 8 * mt + r, pos = p0 + t;
                                bool row_in = t < N;
                                uint lowest = inkling2_lowest_key(pos, ring);
                                // Lane i holds relative state i of this position and head.
                                float relv = row_in ? REL[((ulong)t * H + h) * INKLING2_REL_STATES + lane] : 0.0f;
                                for (uint nt = 0; nt < 2; nt++) {
                                    uint jt = jp + 16 * nt, j = jt + lane;
                                    int dist = (int)pos - (int)j;
                                    // The tile gets the bias when a lane has a distance of 0 to 1,023. The test is
                                    // the same for each lane, so each lane calls the broadcasts of the bias.
                                    int far = (int)pos - (int)jt;
                                    float bias = 0.0f;
                                    if (row_in && far >= 0 && far - 15 < INKLING2_REL_EXTENT) {
                                        uint at = (uint)clamp(dist, 0, INKLING2_REL_EXTENT - 1);
                                        float b = inkling2_rel_bias(relv, PROJ, at);
                                        bias = (dist >= 0 && dist < INKLING2_REL_EXTENT) ? b : 0.0f;
                                    }
                                    bool valid = row_in && j <= pos && j >= lowest;
                                    float s = inkling2_score(sc[mt][nt][r], bias);
                                    sc[mt][nt][r] = valid ? s : -INFINITY;
                                }
                            }
                        }
                    }
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            float m = mine ? sub_group_reduce_max(fmax(sc[mt][0][r], sc[mt][1][r])) : -INFINITY;
                            if (lane == 0) tmax[mem * 16 + 8 * mt + r] = m;
                        }
                    }
                    barrier(CLK_GLOBAL_MEM_FENCE);
                    // Step 2: the new max of each position over the round, and the probabilities before the division
                    // by their sum, exp(score - max), for the team.
                    float new_max[16];
                    for (uint r = 0; r < 16; r++) {
                        float m = fmax(fmax(tmax[r], tmax[16 + r]), fmax(tmax[32 + r], tmax[48 + r]));
                        new_max[r] = fmax(run_max[r], m);
                    }
                    for (uint mt = 0; mt < 2; mt++) {
                        for (uint r = 0; r < 8; r++) {
                            float sum = 0.0f;
                            if (mine) {
                                __global float* prow = tprob + (8 * mt + r) * INKLING2_ATTN_KEYS + 32 * mem;
                                float m = new_max[8 * mt + r];
                                for (uint nt = 0; nt < 2; nt++) {
                                    float s = sc[mt][nt][r];
                                    float pr = s > -INFINITY ? exp(s - m) : 0.0f;
                                    prow[16 * nt + lane] = pr;
                                    sum += pr;
                                }
                                sum = sub_group_reduce_add(sum);
                            }
                            if (lane == 0) tsum[mem * 16 + 8 * mt + r] = sum;
                        }
                    }
                    barrier(CLK_GLOBAL_MEM_FENCE);
                    // Step 3: the old sums at the new max, then the sums of the round against this subgroup's 32
                    // values of V. A subgroup whose key pair is out of the span wrote no probabilities, so the loop
                    // reads only the key tiles of [j_lo, j_hi).
                    if (active) {
                        float8 corr[2];
                        for (uint mt = 0; mt < 2; mt++) {
                            for (uint r = 0; r < 8; r++) {
                                uint tr = 8 * mt + r;
                                float cr = run_max[tr] == new_max[tr] ? 1.0f : exp(run_max[tr] - new_max[tr]);
                                float sum = ((tsum[tr] + tsum[16 + tr]) + tsum[32 + tr]) + tsum[48 + tr];
                                run_sum[tr] = run_sum[tr] * cr + sum;
                                run_max[tr] = new_max[tr];
                                corr[mt][r] = cr;
                            }
                        }
                        for (uint mt = 0; mt < 2; mt++) for (uint nt = 0; nt < 2; nt++) o[mt][nt] *= corr[mt];
                        for (uint ti = 0; ti < INKLING2_ATTN_KEYS / 16; ti++) {
                            uint jt = base + 16 * ti;
                            if (jt < j_lo || jt >= j_hi) continue;
                            uint vy = row0 + 16 * ti;
                            if (ring && vy >= ring) vy -= ring;
                            short8 ph[2], pm[2], pl[2];
                            _Pragma("unroll") for (uint mt = 0; mt < 2; mt++) {
                                uint pv[8];
                                intel_sub_group_2d_block_read_32b_8r16x1c((__global void*)tprob, INKLING2_ATTN_KEYS * 4, 16, INKLING2_ATTN_KEYS * 4, (int2)(16 * ti, 8 * mt), pv);
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
                    // The next round writes the probabilities after its first barrier, so each subgroup has read
                    // them by then.
                    row0 += INKLING2_ATTN_KEYS;
                    if (ring && row0 >= ring) row0 -= ring;
                }
                if (active) {
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
        else if (op == OP_INKLING2_ATTN_SUM) {
            // c[t, h, :] = the merge of OP_INKLING2_ATTN's chunks from scratch, in increasing key order, with the
            // sink of each head. a: the sinks [n][24] f32. n, p0, k and layer are those of OP_INKLING2_ATTN. The sink
            // is one more score with a value of 0, so the merge starts from (max = sink, sum = 1, out = 0), once for
            // each position and head. A chunk with the sum 0 holds no key of its position. Work items: (position,
            // head, 16 output values), one value for each lane.
            __global const float* S = (__global const float*)task.a;
            __global float*       O = (__global float*)task.c;
            const uint H = INKLING2_HEADS, D = INKLING2_HEAD_DIM;
            uint N = task.n;
            uint chunks = inkling2_attn_chunks(N, task.p0, task.k, task.layer).y;
            for (uint item = gsg; item < N * H * (D / 16); item += n_gsg) {
                uint d = (item % (D / 16)) * 16 + lane, th = item / (D / 16);
                float m = S[th], l = 1.0f, o = 0.0f;
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
