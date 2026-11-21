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
            // A work-group takes a position: subgroup j takes the norm of row j, then subgroup q * rows + j the
            // score of query q on row j, then each work item some values of each output.
            __global const float* Bo = (__global const float*)task.a;
            __global const float* Xp = (__global const float*)task.b;
            __global const float* Qr = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint K = task.k, nb = task.m & 0xFFFFu, nq = task.m >> 16, rows = nb + 1u;
            ulong plane = (ulong)task.p0 * K;
            float scaling = as_float(task.layer);
            for (uint t = wg_id; t < task.n; t += num_wgs) {
                for (uint j = sg_id; j < rows; j += num_sgs) {
                    __global const float* r = j < nb ? Bo + j * plane + (ulong)t * K : Xp + (ulong)t * K;
                    float f = inkling2_sg_rms_factor(r, K, lane);
                    if (lane == 0) slm_reduce[j] = f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint s = sg_id; s < nq * rows; s += num_sgs) {
                    uint q = s / rows, j = s % rows;
                    __global const float* r = j < nb ? Bo + j * plane + (ulong)t * K : Xp + (ulong)t * K;
                    __global const float* qv = Qr + (ulong)q * K;
                    float f = slm_reduce[j];
                    float acc = 0.0f;
                    for (uint i = lane; i < K; i += 16u) acc = acc + qv[i] * (r[i] * f);
                    acc = inkling2_lane_sum(acc);
                    if (lane == 0) slm_reduce2[s] = acc * scaling;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint q = 0; q < nq; q++) {
                    // Each work item computes the same softmax, so each has the same weights.
                    float p[5];
                    float top = -INFINITY;
                    for (uint j = 0; j < rows; j++) top = fmax(top, slm_reduce2[q * rows + j]);
                    float total = 0.0f;
                    for (uint j = 0; j < rows; j++) {
                        p[j] = exp(slm_reduce2[q * rows + j] - top);
                        total = total + p[j];
                    }
                    for (uint j = 0; j < rows; j++) p[j] = p[j] / total;
                    __global float* o = C + q * plane + (ulong)t * K;
                    for (uint i = lid; i < K; i += lsize) {
                        float v = 0.0f;
                        for (uint j = 0; j < rows; j++) {
                            __global const float* r = j < nb ? Bo + j * plane + (ulong)t * K : Xp + (ulong)t * K;
                            v = v + p[j] * (r[i] * slm_reduce[j]);
                        }
                        o[i] = v;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        else if (op == OP_MESSAGE_GATE) {
#pragma OPENCL FP_CONTRACT OFF
            // The message gate of each position (host `message_gate`): c = (1 - tanh(F g)) * h + tanh(G g) * m, with
            // g = D [norm(h), norm(m)] and the RMS norm of no weight. a = h [n][k]; b = the messages m [n][k]; c =
            // [n][k], which may be a. x = the bf16 weights of the gate, one after the other: D [64][2k], F [k][64]
            // and G [k][64]. n = positions, k = the width (a multiple of 16). A work-group takes a position:
            // subgroups 0 and 1 take the two norms, subgroup r the value g[r], and each work item some outputs.
            __global const float* Hh = (__global const float*)task.a;
            __global const float* Mm = (__global const float*)task.b;
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            __global const ushort* Wd = (__global const ushort*)task.x;
            __global const ushort* Wf = Wd + (ulong)64 * 2 * K;
            __global const ushort* Wg = Wf + (ulong)K * 64;
            for (uint t = wg_id; t < task.n; t += num_wgs) {
                __global const float* h = Hh + (ulong)t * K;
                __global const float* mm = Mm + (ulong)t * K;
                if (sg_id < 2u) {
                    float f = inkling2_sg_rms_factor(sg_id == 0 ? h : mm, K, lane);
                    if (lane == 0) slm_reduce[sg_id] = f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                float fh = slm_reduce[0], fm = slm_reduce[1];
                for (uint r = sg_id; r < 64u; r += num_sgs) {
                    __global const ushort* w = Wd + (ulong)r * 2 * K;
                    float s = 0.0f;
                    for (uint i = lane; i < 2u * K; i += 16u) {
                        float v = i < K ? h[i] * fh : mm[i - K] * fm;
                        s = s + bf16_to_f32(w[i]) * v;
                    }
                    s = inkling2_lane_sum(s);
                    if (lane == 0) slm_reduce2[r] = s;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                __global float* o = C + (ulong)t * K;
                for (uint i = lid; i < K; i += lsize) {
                    float forget = tanh(inkling2_serial_dot(Wf + (ulong)i * 64, slm_reduce2, 64u));
                    float gain = tanh(inkling2_serial_dot(Wg + (ulong)i * 64, slm_reduce2, 64u));
                    o[i] = (1.0f - forget) * h[i] + gain * mm[i];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
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
