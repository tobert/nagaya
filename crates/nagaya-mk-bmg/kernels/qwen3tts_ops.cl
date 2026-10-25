// The ops of the speech engine (Qwen3-TTS 12 Hz, milestone 17). megakernel.cl includes this file inside the loop of
// mk_persistent_interpreter when MK_QWEN3TTS is defined, so these branches see its locals: task, op, flags,
// accumulate, the work-group and subgroup ids, and the local arrays. The Gemma 4 binary does not hold them.
//
// Activations are f32 and time-major: row t of a tensor is one token or one time step, of k (or m) values. The talker
// and the code predictor have BF16 weights and use the ops of megakernel.cl. The codec decoder has F32 weights, so
// its convolutions take F32 weights here, and a 1 x 1 convolution is its linear layer.

        else if (op == OP_SAMPLE) {
            // One id from the logits a[0, m) (engine-abi "Speech", spec 11.2), in the order of the reference's
            // processors: the repetition penalty over the history, the suppressed ids, temperature, top-k, then a
            // draw. b is the sampler state, uint [seed_lo, seed_hi, draws, count, history[count]]; c gets the id.
            // k = top_k (0: all); p0 = the first suppressed id (m: none); layer = the id kept in the suppressed range,
            // | min_new << 16: while the history holds fewer than min_new ids, that id is suppressed too.
            // x = temperature bits | repetition penalty bits << 32 (0 means 1.0). FLAG_HISTORY appends the id to the
            // history. Work-group 0 samples, with the row in local memory; m is at most 4096.
            if (wg_id == 0) {
                __global const float* Lg = (__global const float*)task.a;
                __global uint* St = (__global uint*)task.b;
                __global uint* Out = (__global uint*)task.c;
                uint V = task.m, topk = task.k, sup = task.p0;
                uint keep = task.layer & 0xFFFFu, min_new = task.layer >> 16;
                float temp = as_float((uint)(task.x & 0xFFFFFFFFul));
                float pen = as_float((uint)(task.x >> 32));
                if (temp == 0.0f) temp = 1.0f;
                if (pen == 0.0f) pen = 1.0f;
                __local float* s = slm_part[0];
                __local uint* mark = (__local uint*)slm_part[1];
                __local float* t = slm_part[1] + 256;
                uint count = St[3];
                for (uint i = lid; i < V; i += lsize) s[i] = Lg[i];
                for (uint i = lid; i < 128; i += lsize) mark[i] = 0;
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint i = lid; i < count; i += lsize) {
                    uint id = St[4 + i];
                    if (id < V) atomic_or(&mark[id / 32], 1u << (id % 32));
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint i = lid; i < V; i += lsize) {
                    float v = s[i];
                    if (mark[i / 32] & (1u << (i % 32))) v = v < 0.0f ? v * pen : v / pen;
                    if (i >= sup && i != keep) v = -INFINITY;
                    if (i == keep && count < min_new) v = -INFINITY;
                    s[i] = v / temp;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // The k-th largest value: a bitonic sort of a copy, largest first, over a power of two.
                uint P = 1;
                while (P < V) P <<= 1;
                for (uint i = lid; i < P; i += lsize) t[i] = i < V ? s[i] : -INFINITY;
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint size = 2; size <= P; size <<= 1) {
                    for (uint stride = size / 2; stride > 0; stride >>= 1) {
                        for (uint i = lid; i < P; i += lsize) {
                            uint j = i ^ stride;
                            if (j > i) {
                                float x0 = t[i], x1 = t[j];
                                bool down = (i & size) == 0;
                                if (down ? x0 < x1 : x0 > x1) {
                                    t[i] = x1;
                                    t[j] = x0;
                                }
                            }
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
                // The reference removes the ids below the k-th value; ties with it stay.
                float thresh = (topk > 0 && topk < V) ? t[topk - 1] : -INFINITY;
                float mx = t[0];
                float sum = 0.0f;
                for (uint i = lid; i < V; i += lsize) {
                    float v = s[i];
                    float p = (v >= thresh && v > -INFINITY) ? exp(v - mx) : 0.0f;
                    s[i] = p;
                    sum += p;
                }
                sum = wg_sum(sum, slm_reduce, lane, sg_id, num_sgs);
                if (lid == 0) {
                    uint draw = St[2];
                    float target = rng_uniform(St[0], St[1], draw) * sum, acc = 0.0f;
                    uint pick = V, last = 0;
                    for (uint i = 0; i < V; i++) {
                        float p = s[i];
                        if (p > 0.0f) {
                            last = i;
                            acc += p;
                            if (acc > target) {
                                pick = i;
                                break;
                            }
                        }
                    }
                    // The sum in order can fall short of the reduced sum by a rounding; the last id then wins.
                    if (pick == V) pick = last;
                    Out[0] = pick;
                    St[2] = draw + 1;
                    if (flags & FLAG_HISTORY) {
                        St[4 + count] = pick;
                        St[3] = count + 1;
                    }
                }
            }
        }
        else if (op == OP_EMBED) {
            // c[t, :] = row ids[t * stride] of the table b, for n rows of k values. a holds the ids (uint), m is their
            // stride (0: 1). The id NO_EVENT gives a row of zeros, or no change with FLAG_ACCUMULATE. The table is
            // BF16, or F32 with FLAG_F32.
            __global const uint* Ids = (__global const uint*)task.a;
            __global float* C = (__global float*)task.c;
            uint K = task.k, stride = task.m ? task.m : 1;
            bool f32 = (flags & FLAG_F32) != 0;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                uint t = (uint)(i / K), j = (uint)(i % K);
                uint id = Ids[(ulong)t * stride];
                float v = 0.0f;
                if (id != NO_EVENT) {
                    ulong at = (ulong)id * K + j;
                    v = f32 ? ((__global const float*)task.b)[at] : bf16_to_f32(((__global const ushort*)task.b)[at]);
                }
                C[i] = accumulate ? C[i] + v : v;
            }
        }
        else if (op == OP_MASK_ROWS) {
            // c[t, :] = 0 where ids[t] is NO_EVENT, for n rows of k values; a holds the ids. The prompt of a speech
            // request has positions with no text, and the text projection of those rows must be zero.
            __global const uint* Ids = (__global const uint*)task.a;
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                if (Ids[i / K] == NO_EVENT) C[i] = 0.0f;
            }
        }
        else if (op == OP_BIAS) {
            // c[t, r] += b[r] for n rows of k values. The bias is BF16, or F32 with FLAG_F32.
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            bool f32 = (flags & FLAG_F32) != 0;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                uint r = (uint)(i % K);
                C[i] += f32 ? ((__global const float*)task.b)[r] : bf16_to_f32(((__global const ushort*)task.b)[r]);
            }
        }
        else if (op == OP_SILU_MUL) {
            // c[i] = silu(a[i]) * b[i] for n rows of k values; b == 0 gives silu(a[i]). The codec's MLP is a 1 x 1
            // convolution for the gate and one for the up rows, so their product is this op.
            __global const float* A = (__global const float*)task.a;
            __global const float* B = (__global const float*)task.b;
            __global float* C = (__global float*)task.c;
            ulong total = (ulong)task.n * task.k;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                float g = A[i];
                float v = g / (1.0f + exp(-g));
                C[i] = B ? v * B[i] : v;
            }
        }
        else if (op == OP_CONV1D) {
            // A causal convolution over time, time-major: c[t, o] (+)= bias[o] + the sum over the inputs i of the group
            // of o and the taps j of w[o, i, j] * a[t - (kk - 1 - j) * dil, i]. Steps before 0 are zero, as the
            // reference pads the start of a decode. n = input steps, m = outputs, k = inputs, p0 = groups (0: 1),
            // layer = kk | dil << 8 | skip << 16, b = w (F32, [m][k / groups][kk]), x = bias (F32) or 0. The first
            // skip input steps are state from the quantum before: they give context, and c gets rows for t >= skip
            // only, at row t - skip. Thus a decode in quanta gives the bits of one decode (engine-abi "Speech").
            __global const float* A = (__global const float*)task.a;
            __global const float* W = (__global const float*)task.b;
            __global const float* Bias = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint T = task.n, M = task.m, Kin = task.k, groups = task.p0 ? task.p0 : 1;
            uint kk = task.layer & 0xFFu, dil = (task.layer >> 8) & 0xFFu, skip = task.layer >> 16;
            uint cin = Kin / groups, cout = M / groups;
            if (cin == 1) {
                // Depthwise: one lane for each output.
                ulong total = (ulong)(T - skip) * M;
                for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                    uint t = (uint)(i / M) + skip, o = (uint)(i % M), g = o / cout;
                    float acc = Bias ? Bias[o] : 0.0f;
                    for (uint j = 0; j < kk; j++) {
                        int src = (int)t - (int)((kk - 1 - j) * dil);
                        if (src >= 0) acc += W[(ulong)o * kk + j] * A[(ulong)src * Kin + g];
                    }
                    C[i] = accumulate ? C[i] + acc : acc;
                }
            } else {
                // One subgroup for each output, its lanes over the inputs.
                ulong total = (ulong)(T - skip) * M;
                for (ulong item = gsg; item < total; item += n_gsg) {
                    uint t = (uint)(item / M) + skip, o = (uint)(item % M), g = o / cout;
                    float acc = 0.0f;
                    for (uint j = 0; j < kk; j++) {
                        int src = (int)t - (int)((kk - 1 - j) * dil);
                        if (src < 0) continue;
                        __global const float* x = A + (ulong)src * Kin + (ulong)g * cin;
                        __global const float* w = W + (ulong)o * cin * kk + j;
                        for (uint i = lane; i < cin; i += 16) acc += w[(ulong)i * kk] * x[i];
                    }
                    acc = sub_group_reduce_add(acc);
                    if (lane == 0) {
                        if (Bias) acc += Bias[o];
                        C[item] = accumulate ? C[item] + acc : acc;
                    }
                }
            }
        }
        else if (op == OP_CONVT1D) {
            // A transposed convolution over time with stride s and kk taps, cut to n * s steps as the reference's
            // causal form cuts kk - s steps at the end: c[u, o] = bias[o] + the sum over t, j with t * s + j == u and
            // over the inputs i of w[i, o, j] * a[t, i]. n = input steps, m = outputs, k = inputs,
            // layer = kk | s << 8 | skip << 16, b = w (F32, [k][m][kk]), x = bias (F32) or 0. The first skip input
            // steps are state from the quantum before: c gets the steps u >= skip * s only, at row u - skip * s.
            __global const float* A = (__global const float*)task.a;
            __global const float* W = (__global const float*)task.b;
            __global const float* Bias = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint T = task.n, M = task.m, Kin = task.k;
            uint kk = task.layer & 0xFFu, st = (task.layer >> 8) & 0xFFu, skip = task.layer >> 16;
            ulong total = (ulong)(T - skip) * st * M;
            for (ulong item = gsg; item < total; item += n_gsg) {
                uint u = (uint)(item / M) + skip * st, o = (uint)(item % M);
                float acc = 0.0f;
                for (uint q = 0; q * st < kk; q++) {
                    int t = (int)(u / st) - (int)q;
                    uint j = u - (uint)t * st;
                    if (t < 0 || j >= kk) continue;
                    __global const float* x = A + (ulong)t * Kin;
                    __global const float* w = W + (ulong)o * kk + j;
                    for (uint i = lane; i < Kin; i += 16) acc += w[(ulong)i * M * kk] * x[i];
                }
                acc = sub_group_reduce_add(acc);
                if (lane == 0) C[item] = acc + (Bias ? Bias[o] : 0.0f);
            }
        }
        else if (op == OP_LAYERNORM) {
            // c[r] = (a[r] - mean) / sqrt(var + eps) * w + bias for n rows of k values; the variance is the biased one.
            // b = w (F32), x = bias (F32), p0 = the bits of eps. One work-group for each row.
            __global const float* A = (__global const float*)task.a;
            __global const float* W = (__global const float*)task.b;
            __global const float* Bias = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            float eps = as_float(task.p0);
            for (uint r = wg_id; r < task.n; r += num_wgs) {
                __global const float* x = A + (ulong)r * K;
                __global float* y = C + (ulong)r * K;
                float s = 0.0f;
                for (uint i = lid; i < K; i += lsize) s += x[i];
                float mean = wg_sum(s, slm_reduce, lane, sg_id, num_sgs) / (float)K;
                float q = 0.0f;
                for (uint i = lid; i < K; i += lsize) {
                    float d = x[i] - mean;
                    q += d * d;
                }
                float inv = rsqrt(wg_sum(q, slm_reduce, lane, sg_id, num_sgs) / (float)K + eps);
                for (uint i = lid; i < K; i += lsize) y[i] = (x[i] - mean) * inv * W[i] + Bias[i];
            }
        }
        else if (op == OP_GELU) {
            // c[i] = gelu(a[i]) for n rows of k values, with erf, as PyTorch's nn.GELU() by default.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            ulong total = (ulong)task.n * task.k;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                float v = A[i];
                C[i] = 0.5f * v * (1.0f + erf(v * 0.7071067811865476f));
            }
        }
        else if (op == OP_SNAKE) {
            // The SnakeBeta activation of the codec: c[t, j] = a + sin(a * exp(alpha[j]))^2 / (exp(beta[j]) + 1e-9) for
            // n rows of k channels. b = alpha (F32), x = beta (F32).
            __global const float* A = (__global const float*)task.a;
            __global const float* Al = (__global const float*)task.b;
            __global const float* Be = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                uint j = (uint)(i % K);
                float v = A[i];
                float sn = sin(v * exp(Al[j]));
                C[i] = v + (1.0f / (exp(Be[j]) + 0.000000001f)) * sn * sn;
            }
        }
        else if (op == OP_SCALE_ADD) {
            // c[t, j] += b[j] * a[t, j] for n rows of k channels: a layer scale or ConvNeXt's gamma, then the residual
            // add. b is F32.
            __global const float* A = (__global const float*)task.a;
            __global const float* G = (__global const float*)task.b;
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) C[i] += G[i % K] * A[i];
        }
        else if (op == OP_PCM) {
            // c[i] = the signed 16-bit sample of a[i] for n samples: clamp to [-1, 1] as the reference's decoder does,
            // then round(v * 32767) (spec 11.8). c holds shorts.
            __global const float* A = (__global const float*)task.a;
            __global short* C = (__global short*)task.c;
            for (ulong i = (ulong)gsg * 16 + lane; i < task.n; i += (ulong)n_gsg * 16) {
                float v = clamp(A[i], -1.0f, 1.0f);
                C[i] = (short)rint(v * 32767.0f);
            }
        }
