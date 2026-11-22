// The layer-part ops of the Inkling 2 binary (MK_INKLING2): opcodes 150 to 159. inkling2_ops.cl includes this file
// inside the task loop, so each op sees the locals of the loop. The builders are in src/tasks.rs, and the host form
// of each op is in crates/nagaya-mk-inkling2-bmg/src/host/ops.rs.
//
// Activations are f32 and position-major. Each op computes a position from its own inputs and the state only, in a
// fixed order, so a position has the same bits in a block of any size. Each sum keeps the order of the host `dot`
// (inkling2_layer_helpers.cl), with no fused multiply-add.

        else if (op == OP_ATTN_RESIDUAL) {
#pragma OPENCL FP_CONTRACT OFF
            // The attention residual of each position (host `attention_residual`): the rows are the block outputs
            // and the partial row, each with the RMS norm of no weight. Query q gives the sum of the normed rows,
            // weighted by the softmax of q . row * k^-0.5. The messages of a cycle are the same op with 4 queries.
            // a = the block outputs, plane j at a + j * p0 * k; b = the partial rows [n][k]; x = the queries, f32
            // [queries][k]; c = the outputs, plane q at c + q * p0 * k. m = blocks (0 to 4) | queries (1 to 4) << 16.
            // n = positions, k = the width (a multiple of 16), p0 = the rows of a plane, layer = the bits of k^-0.5.
            // Each row is 16-byte aligned.
            // A team of queries * rows subgroups takes a position, and a work-group takes a chunk of `per` positions.
            // Subgroup (q, j) of a team takes the norm of row j, then the score of query q on that row. When a block
            // has fewer chunks than work-groups, `splits` work-groups take each chunk. Each one takes the scores
            // again and writes its part of the outputs. Each value has one order of its sums wherever it runs.
            __global const float* Bo = (__global const float*)task.a;
            __global const float* Xp = (__global const float*)task.b;
            __global const float* Qr = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint K = task.k, nb = task.m & 0xFFFFu, nq = task.m >> 16, rows = nb + 1u;
            ulong plane = (ulong)task.p0 * K;
            float scaling = as_float(task.layer);
            uint team = nq * rows, per = num_sgs / team;
            uint chunks = (task.n + per - 1u) / per;
            uint splits = chunks >= num_wgs ? 1u : num_wgs / chunks;
            uint groups = num_wgs / splits, group = wg_id / splits, part = wg_id % splits;
            // slm_reduce holds the norm factors [per][rows], and slm_reduce2 the scores, then the weights,
            // [per][nq][rows].
            // A chunk of one position first copies its rows and the queries into local memory, when they fit, with
            // block reads of all subgroups: then each sum waits for local memory, not for global memory.
            __local float* rs = &slm_part[0][0];
            uint bpr = K / INKLING2_BLOCK;
            bool staged = K % INKLING2_BLOCK == 0u && (rows + nq) * K <= INKLING2_SLM_FLOATS;
            for (uint ch = group; group < groups && ch < chunks; ch += groups) {
                uint t0 = ch * per, tn = min(per, task.n - t0);
                bool one = tn == 1u && staged;
                if (one) {
                    uint nblk = (rows + nq) * bpr;
                    for (uint b0 = sg_id; b0 < nblk; b0 += INKLING2_STAGE * num_sgs) {
                        uint8 v[INKLING2_STAGE];
#pragma unroll
                        for (uint u = 0; u < INKLING2_STAGE; u++) {
                            uint b = min(b0 + u * num_sgs, nblk - 1u), j = b / bpr;
                            __global const float* src = j < nb    ? Bo + j * plane + (ulong)t0 * K
                                                        : j == nb ? Xp + (ulong)t0 * K
                                                                  : Qr + (ulong)(j - rows) * K;
                            src += (b % bpr) * INKLING2_BLOCK;
                            v[u] = intel_sub_group_block_read8((__global const uint*)src);
                        }
#pragma unroll
                        for (uint u = 0; u < INKLING2_STAGE; u++) {
                            uint b = b0 + u * num_sgs;
                            if (b < nblk) intel_sub_group_block_write8((__local uint*)(rs + b * INKLING2_BLOCK), v[u]);
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (sg_id < team) {
                        uint q = sg_id / rows, j = sg_id % rows;
                        __local const float* r = rs + j * K;
                        float f = inkling2_rms_factor_block_slm(r, K, lane);
                        float acc = inkling2_scaled_dot_block_slm(0.0f, rs + (rows + q) * K, true, r, f, K, lane);
                        acc = inkling2_lane_sum(acc);
                        if (lane == 0) {
                            if (q == 0) slm_reduce[j] = f;
                            slm_reduce2[q * rows + j] = acc * scaling;
                        }
                    }
                } else if (sg_id < tn * team) {
                    uint p = sg_id / team, s = sg_id % team, q = s / rows, j = s % rows, t = t0 + p;
                    __global const float* r = j < nb ? Bo + j * plane + (ulong)t * K : Xp + (ulong)t * K;
                    float f = inkling2_rms_factor_block(r, K, lane);
                    float acc = inkling2_scaled_dot_block(0.0f, Qr + (ulong)q * K, true, r, f, K, lane);
                    acc = inkling2_lane_sum(acc);
                    if (lane == 0) {
                        if (q == 0) slm_reduce[p * rows + j] = f;
                        slm_reduce2[(p * nq + q) * rows + j] = acc * scaling;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // The softmax of each query of each position: one work item each.
                if (lid < tn * nq) {
                    __local float* sc = slm_reduce2 + lid * rows;
                    float pr[5];
                    float top = -INFINITY;
                    for (uint j = 0; j < rows; j++) top = fmax(top, sc[j]);
                    float total = 0.0f;
                    for (uint j = 0; j < rows; j++) {
                        pr[j] = exp(sc[j] - top);
                        total = total + pr[j];
                    }
                    for (uint j = 0; j < rows; j++) sc[j] = pr[j] / total;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                uint values = tn * nq * K;
                for (uint e = part * lsize + lid; e < values; e += splits * lsize) {
                    uint i = e % K, pq = e / K, p = pq / nq, q = pq % nq, t = t0 + p;
                    __local const float* wq = slm_reduce2 + pq * rows;
                    float v = 0.0f;
                    if (one) {
                        for (uint j = 0; j < rows; j++) v = v + wq[j] * (rs[j * K + i] * slm_reduce[j]);
                    } else {
                        for (uint j = 0; j < rows; j++) {
                            __global const float* r = j < nb ? Bo + j * plane + (ulong)t * K : Xp + (ulong)t * K;
                            v = v + wq[j] * (r[i] * slm_reduce[p * rows + j]);
                        }
                    }
                    C[q * plane + (ulong)t * K + i] = v;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        else if (op == OP_MESSAGE_GATE_DOWN) {
#pragma OPENCL FP_CONTRACT OFF
            // The first step of the message gate (host `message_gate`): g = D [norm(h), norm(m)] of each position,
            // with the RMS norm of no weight. a = h [n][k]; b = the messages m [n][k]; x = the bf16 weights D
            // [64][2k]; c = g, f32 [n][64]. n = positions, k = the width (a multiple of 16). Each row is 16-byte
            // aligned.
            // A work-group takes a tile of INKLING2_GATE_TILE positions: subgroup 2p + 0 or 1 takes the norm of h or
            // m of position p, then the rows of D for each position. When a block has fewer tiles than work-groups,
            // `splits` work-groups share each tile. Each one takes the norms again and a part of the rows.
            __global const float* Hh = (__global const float*)task.a;
            __global const float* Mm = (__global const float*)task.b;
            __global const ushort* Wd = (__global const ushort*)task.x;
            __global float* Gs = (__global float*)task.c;
            uint K = task.k, K2 = 2u * K;
            uint tiles = (task.n + INKLING2_GATE_TILE - 1u) / INKLING2_GATE_TILE;
            uint splits = tiles >= num_wgs ? 1u : min(num_wgs / tiles, (uint)INKLING2_GATE_RANK);
            uint groups = num_wgs / splits, group = wg_id / splits, part = wg_id % splits;
            // A tile of one position first copies h, m and the rows of D of its work-group into local memory with
            // block reads of all subgroups: then each sum waits for local memory, not for global memory.
            uint mine = (INKLING2_GATE_RANK - part + splits - 1u) / splits, bpr = K / INKLING2_BLOCK;
            bool fits = K % INKLING2_BLOCK == 0u && (2u + mine) * K <= INKLING2_SLM_FLOATS;
            __local float* xs = &slm_part[0][0];
            __local const ushort* ws = (__local const ushort*)(xs + K2);
            for (uint tile = group; group < groups && tile < tiles; tile += groups) {
                uint t0 = tile * INKLING2_GATE_TILE, tn = min((uint)INKLING2_GATE_TILE, task.n - t0);
                if (tn == 1u && fits) {
                    // Blocks 0 to 2 bpr - 1 are h and m; then each row of D is bpr blocks of 256 bf16 values.
                    uint nblk = (2u + mine) * bpr;
                    for (uint b0 = sg_id; b0 < nblk; b0 += INKLING2_STAGE * num_sgs) {
                        uint8 v[INKLING2_STAGE];
#pragma unroll
                        for (uint u = 0; u < INKLING2_STAGE; u++) {
                            uint b = min(b0 + u * num_sgs, nblk - 1u), j = b / bpr, o = (b % bpr) * INKLING2_BLOCK;
                            __global const uint* src = j == 0u ? (__global const uint*)(Hh + (ulong)t0 * K)
                                : j == 1u ? (__global const uint*)(Mm + (ulong)t0 * K)
                                          : (__global const uint*)(Wd + (ulong)(part + splits * (j - 2u)) * K2);
                            v[u] = intel_sub_group_block_read8(src + o);
                        }
#pragma unroll
                        for (uint u = 0; u < INKLING2_STAGE; u++) {
                            uint b = b0 + u * num_sgs;
                            if (b < nblk) intel_sub_group_block_write8((__local uint*)xs + b * INKLING2_BLOCK, v[u]);
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (sg_id < 2u) {
                        float f = inkling2_rms_factor_block_slm(xs + sg_id * K, K, lane);
                        if (lane == 0) slm_reduce[sg_id] = f;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (uint j = sg_id; j < mine; j += num_sgs) {
                        // The values of h, then those of m: the order of the host over [norm(h), norm(m)].
                        __local const ushort* w = ws + j * K2;
                        float acc = inkling2_scaled_dot_block_slm(0.0f, w, false, xs, slm_reduce[0], K, lane);
                        acc = inkling2_scaled_dot_block_slm(acc, w + K, false, xs + K, slm_reduce[1], K, lane);
                        acc = inkling2_lane_sum(acc);
                        if (lane == 0) Gs[(ulong)t0 * INKLING2_GATE_RANK + part + splits * j] = acc;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    continue;
                }
                if (sg_id < 2u * tn) {
                    __global const float* x = (sg_id & 1u ? Mm : Hh) + (ulong)(t0 + sg_id / 2u) * K;
                    float f = inkling2_rms_factor_block(x, K, lane);
                    if (lane == 0) slm_reduce[sg_id] = f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // Subgroup (r, p) takes row r of D for position p; the values of h come first, then those of m, as
                // in the host. A tile reads each row of D once for each of its positions, mostly from the cache.
                uint items = (INKLING2_GATE_RANK - part + splits - 1u) / splits * tn;
                for (uint it = sg_id; it < items; it += num_sgs) {
                    uint r = part + splits * (it / tn), p = it % tn;
                    __global const ushort* w = Wd + (ulong)r * K2;
                    ulong row = (ulong)(t0 + p) * K;
                    float acc = inkling2_scaled_dot_block(0.0f, w, false, Hh + row, slm_reduce[2u * p], K, lane);
                    acc = inkling2_scaled_dot_block(acc, w + K, false, Mm + row, slm_reduce[2u * p + 1u], K, lane);
                    acc = inkling2_lane_sum(acc);
                    if (lane == 0) Gs[(ulong)(t0 + p) * INKLING2_GATE_RANK + r] = acc;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        else if (op == OP_MESSAGE_GATE) {
#pragma OPENCL FP_CONTRACT OFF
            // The second step of the message gate (host `message_gate`): c = (1 - tanh(F g)) * h + tanh(G g) * m,
            // with g from OP_MESSAGE_GATE_DOWN. a = h [n][k]; b = the messages m [n][k]; c = [n][k], which may be a.
            // x = the bf16 weights of the gate, one after the other: D [64][2k], F [k][64] and G [k][64]. g is f32
            // [n][64] at p0 | layer << 32. n = positions, k = the width (a multiple of 16). Each row is 16-byte
            // aligned.
            // A subgroup takes the output i of a tile of up to INKLING2_GATE_TILE2 positions. Lane l holds the values
            // l, l + 16, l + 32 and l + 48 of row i of F and of G, so each sum has the order of the host `dot`. Lane
            // p then takes the output of position p.
            __global const float* Hh = (__global const float*)task.a;
            __global const float* Mm = (__global const float*)task.b;
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            __global const ushort* Wf = (__global const ushort*)task.x + (ulong)INKLING2_GATE_RANK * 2 * K;
            __global const ushort* Wg = Wf + (ulong)K * INKLING2_GATE_RANK;
            __global const float* Gs = (__global const float*)((ulong)task.p0 | (ulong)task.layer << 32);
            if (task.n < INKLING2_GATE_TILE2) {
                // A few positions: work item (t, i) takes output i of position t. g goes to local memory first, and
                // inkling2_serial_dot gives each sum the order of the host `dot` (measured faster here than a sum
                // over the lanes in a decode step: 6.2 us against 9.7 us, 2026-11-22).
                __local float* gl = &slm_part[0][0];
                for (uint e = lid; e < task.n * INKLING2_GATE_RANK; e += lsize) gl[e] = Gs[e];
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint e = gsg * 16u + lane; e < task.n * K; e += n_gsg * 16u) {
                    uint t = e / K, i = e % K;
                    __local const float* g = gl + t * INKLING2_GATE_RANK;
                    float forget = tanh(inkling2_serial_dot(Wf + (ulong)i * INKLING2_GATE_RANK, g, INKLING2_GATE_RANK));
                    float gain = tanh(inkling2_serial_dot(Wg + (ulong)i * INKLING2_GATE_RANK, g, INKLING2_GATE_RANK));
                    C[e] = (1.0f - forget) * Hh[e] + gain * Mm[e];
                }
            }
            uint tiles = task.n < INKLING2_GATE_TILE2 ? 0u : (task.n + INKLING2_GATE_TILE2 - 1u) / INKLING2_GATE_TILE2;
            for (uint item = gsg; item < K * tiles; item += n_gsg) {
                uint i = item % K, t0 = (item / K) * INKLING2_GATE_TILE2;
                uint tn = min((uint)INKLING2_GATE_TILE2, task.n - t0);
                ulong wrow = (ulong)i * INKLING2_GATE_RANK;
                float4 wf = as_float4(convert_uint4(intel_sub_group_block_read_us4(Wf + wrow)) << 16);
                float4 wg = as_float4(convert_uint4(intel_sub_group_block_read_us4(Wg + wrow)) << 16);
                float mine_f = 0.0f, mine_g = 0.0f;
                for (uint p = 0; p < tn; p++) {
                    float4 gv = as_float4(intel_sub_group_block_read4(
                        (__global const uint*)(Gs + (ulong)(t0 + p) * INKLING2_GATE_RANK)));
                    float fs = 0.0f, gs = 0.0f;
#pragma unroll
                    for (uint u = 0; u < 4u; u++) {
                        fs = fs + wf[u] * gv[u];
                        gs = gs + wg[u] * gv[u];
                    }
                    fs = inkling2_lane_sum(fs);
                    gs = inkling2_lane_sum(gs);
                    if (lane == p) {
                        mine_f = fs;
                        mine_g = gs;
                    }
                }
                if (lane < tn) {
                    ulong at = (ulong)(t0 + lane) * K + i;
                    float forget = tanh(mine_f);
                    float gain = tanh(mine_g);
                    C[at] = (1.0f - forget) * Hh[at] + gain * Mm[at];
                }
            }
        }
        else if (op == OP_HEAD_CLIP) {
#pragma OPENCL FP_CONTRACT OFF
            // The clip or the norm of each head, for n positions of m heads of k values (host `scaled_l2_clip`,
            // `l2_clip`, `rms_norm_unweighted` and `log_scale`). a = [n][m * k]; c = the same, and it may be a. b =
            // the bf16 scale of each value [m * k], or 0 for no scale. x = the bits of the L2 bound. layer = the mode
            // (bits 0 to 7: 0 is the L2 clip, 1 is the RMS norm of no weight) | 256 for the log scale of a full
            // layer. p0 = the position of row 0; with a token table, the position of row t is its entry. k is a
            // multiple of 16. A subgroup takes one head of one position.
            __global const float* A = (__global const float*)task.a;
            __global const ushort* Sc = (__global const ushort*)task.b;
            __global float* C = (__global float*)task.c;
            uint H = task.m, D = task.k, mode = task.layer & 0xFFu;
            bool logs = (task.layer & 0x100u) != 0;
            float bound = as_float((uint)task.x);
            ulong heads = (ulong)task.n * H;
            for (ulong item = gsg; item < heads; item += n_gsg) {
                uint t = (uint)(item / H), h = (uint)(item % H);
                __global const float* xin = A + item * D;
                __global float* y = C + item * D;
                float ss = 0.0f;
                for (uint i = lane; i < D; i += 16u) {
                    float v = Sc ? xin[i] * bf16_to_f32(Sc[h * D + i]) : xin[i];
                    ss = ss + v * v;
                }
                ss = inkling2_lane_sum(ss);
                float f = mode == 0u ? fmin(bound / (sqrt(ss) + 1e-6f), 1.0f)
                                     : 1.0f / sqrt(ss / (float)D + INKLING2_EPS);
                float ls = logs ? inkling2_log_scale(token_meta ? token_meta[t].x : task.p0 + t) : 1.0f;
                for (uint i = lane; i < D; i += 16u) {
                    float v = Sc ? xin[i] * bf16_to_f32(Sc[h * D + i]) : xin[i];
                    v = v * f;
                    y[i] = logs ? v * ls : v;
                }
            }
        }
        else if (op == OP_OUTPUT_GATE) {
#pragma OPENCL FP_CONTRACT OFF
            // The gate of the attention output (host `Model::attention`, the sigmoid of the query input): c[i] = a[i]
            // * sigmoid(b[i]) for n rows of k. a may be c.
            __global const float* A = (__global const float*)task.a;
            __global const float* G = (__global const float*)task.b;
            __global float* C = (__global float*)task.c;
            ulong total = (ulong)task.n * task.k;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) C[i] = A[i] * inkling2_sigmoid(G[i]);
        }
        else if (op == OP_RESIDUAL_CONV) {
#pragma OPENCL FP_CONTRACT OFF
            // The short convolution of kernel 4 with its residual, for n positions of k channels (host `short_conv`,
            // `dynamic_conv` and `conv_push`). layer 0 has fixed taps: (w . window + x) / sqrt(5), with b = the bf16
            // taps [k][4] and tap 3 on the current row. layer 1 has dynamic taps: the tap of lag j of channel i is
            // b[j][i] + the tap j of its head of 128, and the output is (sum + x) / sqrt(9). b = the bf16 taps [4][k]
            // by lag, and lag 0 is the current row. a = the inputs, a row of stride sa for each position; in the
            // dynamic mode, the 4 taps of each head follow the k channels of the row. x = the state, 3 rows of stride
            // ss, oldest first. m = sa | ss << 16 (0: k). c = [n][k]; FLAG_ACCUMULATE adds the result to c. p0 = the
            // position of row 0; at 0 the state reads as zeros. k is a multiple of 16, and of 128 in the dynamic
            // mode. The op writes the last 3 inputs into the state.
            // A lane takes one channel of a chunk of INKLING2_CONV_CHUNK positions. Only chunk 0 reads the old state,
            // so it writes the new state after its reads.
            __global const float* A = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global float* S = (__global float*)task.x;
            __global float* Y = (__global float*)task.c;
            uint T = task.n, Kc = task.k;
            uint sa = (task.m & 0xFFFFu) ? (task.m & 0xFFFFu) : Kc, ss = (task.m >> 16) ? (task.m >> 16) : Kc;
            bool dynamic = task.layer == 1u, fresh = task.p0 == 0u;
            uint groups = Kc / 16u, chunks = (T + INKLING2_CONV_CHUNK - 1u) / INKLING2_CONV_CHUNK;
            for (uint item = gsg; item < groups * chunks; item += n_gsg) {
                uint ch = item / groups, d = (item % groups) * 16u + lane;
                uint t0 = ch * INKLING2_CONV_CHUNK, t1 = min(T, t0 + INKLING2_CONV_CHUNK);
                float w[4], win[4], old[3];
                for (uint j = 0; j < 4u; j++) w[j] = bf16_to_f32(dynamic ? W[(ulong)j * Kc + d] : W[(ulong)d * 4 + j]);
                // The window before position t0: the inputs t0 - 3 to t0 - 1, from the state before position 0.
                for (uint j = 0; j < 3u; j++) {
                    int s = (int)t0 - 3 + (int)j;
                    win[j] = s >= 0 ? A[(ulong)s * sa + d] : (fresh ? 0.0f : S[(ulong)(3 + s) * ss + d]);
                    old[j] = win[j];
                }
                for (uint t = t0; t < t1; t++) {
                    __global const float* row = A + (ulong)t * sa;
                    float xv = row[d];
                    win[3] = xv;
                    float sum;
                    if (dynamic) {
                        __global const float* taps = row + Kc + (d / 128u) * 4u;
                        sum = 0.0f;
                        sum = sum + xv * (w[0] + taps[0]);
                        sum = sum + win[2] * (w[1] + taps[1]);
                        sum = sum + win[1] * (w[2] + taps[2]);
                        sum = sum + win[0] * (w[3] + taps[3]);
                    } else {
                        sum = w[0] * win[0] + w[1] * win[1] + w[2] * win[2] + w[3] * xv;
                    }
                    float out = (sum + xv) / (dynamic ? INKLING2_SQRT9 : INKLING2_SQRT5);
                    __global float* y = Y + (ulong)t * Kc + d;
                    *y = accumulate ? *y + out : out;
                    for (uint j = 0; j < 3u; j++) win[j] = win[j + 1];
                }
                if (ch == 0) {
                    // The new state: the inputs T - 3 to T - 1. With T < 3 its first rows come from the old state.
                    for (uint j = 0; j < 3u; j++) {
                        int s = (int)T - 3 + (int)j;
                        S[(ulong)j * ss + d] = s >= 0 ? A[(ulong)s * sa + d] : old[3 + s];
                    }
                }
            }
        }
        else if (op == OP_SIGMOID_ROUTER) {
#pragma OPENCL FP_CONTRACT OFF
            // The router of each position (host `route`): the score of expert e is sigmoid(a[t][e]), the selection
            // is the 8 largest score + b[e], and the weight of a pick is score / (the sum of the 8 scores + 1e-20) *
            // 8. a = the logits f32 [n][128]; b = the bias f32 [128]; c = MoERouteDecision [n], with the picks in
            // ascending expert order, so OP_EXPERT_SUM adds them in the order of the reference. The entropy and the
            // log-sum-exp are 0. n = positions, layer = the layer of the decisions.
            // A subgroup takes a position, and lane l holds the experts l, l + 16, and so on. The order of the keys
            // is that of f32::total_cmp, and equal keys go to the lower index, as in the host.
            __global const float* L = (__global const float*)task.a;
            __global const float* Bi = (__global const float*)task.b;
            __global MoERouteDecision* R = (__global MoERouteDecision*)task.c;
            for (uint t = gsg; t < task.n; t += n_gsg) {
                __global const float* lg = L + (ulong)t * NUM_EXPERTS;
                int key[NUM_EXPERTS / 16];
                for (uint j = 0; j < NUM_EXPERTS / 16; j++) {
                    uint e = j * 16u + lane;
                    key[j] = inkling2_total_key(inkling2_sigmoid(lg[e]) + Bi[e]);
                }
                uint taken = 0, pick[TOP_K];
                for (uint p = 0; p < TOP_K; p++) {
                    // The best key of this lane that is not taken, with the lower index first. A lane with none
                    // offers no index, so a real key always wins a tie with it.
                    int best = INT_MIN;
                    uint at = 0xFFFFFFFFu;
                    for (uint j = 0; j < NUM_EXPERTS / 16; j++) {
                        if ((taken >> j) & 1u) continue;
                        if (at == 0xFFFFFFFFu || key[j] > best) { best = key[j]; at = j * 16u + lane; }
                    }
                    int top = sub_group_reduce_max(best);
                    uint e = sub_group_reduce_min(best == top ? at : 0xFFFFFFFFu);
                    if ((e & 15u) == lane) taken |= 1u << (e >> 4);
                    pick[p] = e;
                }
                if (lane == 0) {
                    for (uint p = 1; p < TOP_K; p++) {
                        uint e = pick[p], q = p;
                        for (; q > 0 && pick[q - 1] > e; q--) pick[q] = pick[q - 1];
                        pick[q] = e;
                    }
                    float score[TOP_K];
                    float total = 0.0f;
                    for (uint p = 0; p < TOP_K; p++) {
                        score[p] = inkling2_sigmoid(lg[pick[p]]);
                        total = total + score[p];
                    }
                    total = total + 1e-20f;
                    for (uint p = 0; p < TOP_K; p++) {
                        R[t].top_k_indices[p] = pick[p];
                        R[t].top_k_weights[p] = score[p] / total * 8.0f;
                    }
                    R[t].router_entropy = 0.0f;
                    R[t].router_lse = 0.0f;
                    R[t].layer_idx = task.layer;
                    R[t].token = t;
                    for (uint i = 0; i < 4u; i++) R[t].reserved[i] = 0u;
                }
            }
        }
