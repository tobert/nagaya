// The ops of the embedding engine (EmbeddingGemma 2). megakernel.cl includes this file inside the loop of
// mk_persistent_interpreter when MK_EMBEDDINGGEMMA2 is defined, so these branches see its locals, as the ops of
// qwen3tts_ops.cl do. The embedding binary also holds the speech ops, but its OP_GEGLU is GeGLU with gelu_tanh.
//
// Activations are f32 and token-major: row t of a tensor is one token. The weights are BF16 and use the ops of
// megakernel.cl.

        else if (op == OP_BIATTN) {
            // Attention in both directions with grouped heads: c[t, h * D + j] = the sum over the keys s of
            // softmax_s(scale * q[t, h] . k[s, g]) * v[s, g * D + j], with g = h / (H / KV). a = q rows [n][H * D],
            // b = k rows [n][KV * D], x = v rows [n][KV * D], c = out rows [n][H * D]. n = tokens, m = H | KV << 16,
            // k = the window (0: every key; else the keys s with |t - s| <= window), p0 = D (a multiple of 16, at most
            // 512), layer = the bits of scale (0: 1.0). One subgroup for each query row and head, with an online
            // softmax in key order.
            __global const float* Q = (__global const float*)task.a;
            __global const float* Kr = (__global const float*)task.b;
            __global const float* V = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint N = task.n, H = task.m & 0xFFFFu, KV = task.m >> 16, D = task.p0, W = task.k;
            uint per = D / 16, group = H / KV;
            float scale = task.layer ? as_float(task.layer) : 1.0f;
            for (uint item = gsg; item < N * H; item += n_gsg) {
                uint h = item % H, t = item / H, g = h / group;
                __global const float* q = Q + ((ulong)t * H + h) * D;
                float qv[32], acc[32];
                for (uint r = 0; r < per; r++) {
                    qv[r] = q[lane + 16 * r];
                    acc[r] = 0.0f;
                }
                uint lo = (W != 0 && t > W) ? t - W : 0;
                uint hi = W != 0 ? min(N, t + W + 1) : N;
                float mx = -INFINITY, l = 0.0f;
                for (uint s = lo; s < hi; s++) {
                    ulong row = ((ulong)s * KV + g) * D;
                    float dot = 0.0f;
                    for (uint r = 0; r < per; r++) dot += qv[r] * Kr[row + lane + 16 * r];
                    dot = sub_group_reduce_add(dot) * scale;
                    float mnew = fmax(mx, dot);
                    float corr = exp(mx - mnew), p = exp(dot - mnew);
                    l = l * corr + p;
                    for (uint r = 0; r < per; r++) acc[r] = acc[r] * corr + p * V[row + lane + 16 * r];
                    mx = mnew;
                }
                __global float* out = C + ((ulong)t * H + h) * D;
                for (uint r = 0; r < per; r++) out[lane + 16 * r] = acc[r] / l;
            }
        }
        else if (op == OP_GELU_MUL) {
            // c[t, j] = gelu_tanh(a[t, j]) * b[t * m + j] for n rows of k values, with m = 0 meaning k: the gate of the
            // per-layer input. The rows of b are those of one layer in the per-layer inputs of every layer.
            __global const float* A = (__global const float*)task.a;
            __global const float* B = (__global const float*)task.b;
            __global float* C = (__global float*)task.c;
            uint K = task.k, stride = task.m ? task.m : task.k;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                ulong t = i / K, j = i % K;
                C[i] = gelu_tanh(A[i]) * B[t * stride + j];
            }
        }
        else if (op == OP_SCALE) {
            // c[i] = a[i] * as_float(p0) for n rows of k values; a may be c. It is the scale of the embeddings, of the
            // per-layer projection, and the layer scalar.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            float s = as_float(task.p0);
            ulong total = (ulong)task.n * task.k;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) C[i] = A[i] * s;
        }
