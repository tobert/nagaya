// The ops of the image engine (SDXL, milestone 18 step 7). megakernel.cl includes this file inside the loop of
// mk_persistent_interpreter when MK_SDXL is defined, so these branches see its locals: task, op, flags, accumulate,
// the work-group and subgroup ids, and the local arrays. The image binary also defines MK_QWEN3TTS, so it has the
// ops of qwen3tts_ops.cl, such as OP_GELU, OP_SILU_MUL, OP_BIAS and OP_LAYERNORM. The other binaries hold none of these.
//
// Activations are f32 and pixel-major: row (image, y, x) of a tensor holds the channels of one pixel. Thus a pixel is
// a token for attention, and a 1 x 1 convolution is a linear layer over rows. The UNet has F16 weights (FLAG_F16),
// and the stock VAE has F32 weights (FLAG_F32).

// The weight at index I of W: F16 with FLAG_F16, F32 with FLAG_F32, else BF16.
#define SDXL_W(W, I) ((flags & FLAG_F16) ? vload_half((I), (__global const half*)(W)) \
    : (flags & FLAG_F32) ? ((__global const float*)(W))[(I)] : bf16_to_f32(((__global const ushort*)(W))[(I)]))

        else if (op == OP_CONV2D) {
            // c[b, yo, xo, o] (+)= bias[o] + the sum over the inputs i and the taps ky, kx of
            // w[o, i, ky, kx] * a[b, yo * s + ky - pad, xo * s + kx - pad, i], with zero outside the image.
            // n = H | W << 16 (the input), m = outputs, k = inputs, p0 = images (0: 1),
            // layer = kh | kw << 8 | s << 16 | pad << 24, b = w ([m][k][kh][kw], as PyTorch), x = bias or 0.
            // A 1 x 1 kernel with s 1 and pad 0 is a linear layer; s 2 is the down sampler of SDXL.
            // One subgroup for each output value, its lanes over the inputs.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint H = task.n & 0xFFFFu, Wd = task.n >> 16, M = task.m, Kin = task.k;
            uint B = task.p0 ? task.p0 : 1;
            uint kh = task.layer & 0xFFu, kw = (task.layer >> 8) & 0xFFu;
            uint st = (task.layer >> 16) & 0xFFu, pad = task.layer >> 24;
            uint Ho = (H + 2 * pad - kh) / st + 1, Wo = (Wd + 2 * pad - kw) / st + 1;
            ulong taps = (ulong)kh * kw;
            ulong total = (ulong)B * Ho * Wo * M;
            for (ulong item = gsg; item < total; item += n_gsg) {
                uint o = (uint)(item % M);
                ulong pix = item / M;
                uint xo = (uint)(pix % Wo), yo = (uint)((pix / Wo) % Ho), b = (uint)(pix / ((ulong)Wo * Ho));
                float acc = 0.0f;
                for (uint ky = 0; ky < kh; ky++) {
                    int y = (int)(yo * st + ky) - (int)pad;
                    if (y < 0 || y >= (int)H) continue;
                    for (uint kx = 0; kx < kw; kx++) {
                        int x = (int)(xo * st + kx) - (int)pad;
                        if (x < 0 || x >= (int)Wd) continue;
                        __global const float* in = A + (((ulong)b * H + y) * Wd + x) * Kin;
                        ulong w0 = (ulong)o * Kin * taps + ky * kw + kx;
                        for (uint i = lane; i < Kin; i += 16) acc += SDXL_W(task.b, w0 + (ulong)i * taps) * in[i];
                    }
                }
                acc = sub_group_reduce_add(acc);
                if (lane == 0) {
                    if (task.x) acc += SDXL_W(task.x, o);
                    C[item] = accumulate ? C[item] + acc : acc;
                }
            }
        }
        else if (op == OP_GROUPNORM) {
            // For each image b and group g: the mean and the biased variance over the pixels of the image and the
            // k / groups channels of the group, then c = (a - mean) * rsqrt(var + eps) * w[ch] + bias[ch].
            // FLAG_SILU applies silu after it. n = pixels of an image, k = channels, m = images (0: 1),
            // layer = groups, p0 = the bits of eps, b = w, x = bias. One work-group for each image and group.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint P = task.n, K = task.k, B = task.m ? task.m : 1, G = task.layer, cg = K / G;
            float eps = as_float(task.p0);
            ulong count = (ulong)P * cg;
            for (uint item = wg_id; item < B * G; item += num_wgs) {
                uint b = item / G, g = item % G;
                __global const float* x = A + (ulong)b * P * K + (ulong)g * cg;
                __global float* y = C + (ulong)b * P * K + (ulong)g * cg;
                float s = 0.0f;
                for (ulong e = lid; e < count; e += lsize) s += x[(e / cg) * K + e % cg];
                float mean = wg_sum(s, slm_reduce, lane, sg_id, num_sgs) / (float)count;
                float q = 0.0f;
                for (ulong e = lid; e < count; e += lsize) {
                    float d = x[(e / cg) * K + e % cg] - mean;
                    q += d * d;
                }
                float inv = rsqrt(wg_sum(q, slm_reduce, lane, sg_id, num_sgs) / (float)count + eps);
                for (ulong e = lid; e < count; e += lsize) {
                    ulong at = (e / cg) * K + e % cg;
                    uint ch = g * cg + (uint)(e % cg);
                    float v = (x[at] - mean) * inv * SDXL_W(task.b, ch) + SDXL_W(task.x, ch);
                    if (flags & FLAG_SILU) v = v / (1.0f + exp(-v));
                    y[at] = v;
                }
            }
        }
        else if (op == OP_SDPA) {
            // Attention with no mask: c[b, t, h * D + j] = the sum over the keys s of
            // softmax_s(q[b, t, h] . k[b, s, h] / sqrt(D)) * v[b, s, h * D + j]. a = q, b = k, x = v, c = out, each in
            // rows of k values (heads * D). n = query rows, m = key rows, p0 = D (a multiple of 16, at most 512),
            // layer = images (0: 1). FLAG_CAUSAL keeps the keys s <= t. One subgroup for each image, query row and
            // head, with an online softmax.
            __global const float* Q = (__global const float*)task.a;
            __global const float* Kt = (__global const float*)task.b;
            __global const float* V = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint Nq = task.n, Nk = task.m, R = task.k, D = task.p0, heads = R / D;
            uint B = task.layer ? task.layer : 1, per = D / 16;
            float scale = rsqrt((float)D);
            ulong total = (ulong)B * Nq * heads;
            for (ulong item = gsg; item < total; item += n_gsg) {
                uint h = (uint)(item % heads);
                ulong bt = item / heads;
                uint b = (uint)(bt / Nq);
                __global const float* q = Q + bt * R + (ulong)h * D;
                float qv[32], acc[32];
                for (uint r = 0; r < per; r++) {
                    qv[r] = q[lane + 16 * r] * scale;
                    acc[r] = 0.0f;
                }
                float mx = -INFINITY, l = 0.0f;
                uint keys = (flags & FLAG_CAUSAL) ? min(Nk, (uint)(bt % Nq) + 1) : Nk;
                for (uint s = 0; s < keys; s++) {
                    ulong row = ((ulong)b * Nk + s) * R + (ulong)h * D;
                    float dot = 0.0f;
                    for (uint r = 0; r < per; r++) dot += qv[r] * Kt[row + lane + 16 * r];
                    dot = sub_group_reduce_add(dot);
                    float mnew = fmax(mx, dot);
                    float corr = exp(mx - mnew), p = exp(dot - mnew);
                    l = l * corr + p;
                    for (uint r = 0; r < per; r++) acc[r] = acc[r] * corr + p * V[row + lane + 16 * r];
                    mx = mnew;
                }
                __global float* out = C + bt * R + (ulong)h * D;
                for (uint r = 0; r < per; r++) out[lane + 16 * r] = acc[r] / l;
            }
        }
        else if (op == OP_UPSAMPLE2X) {
            // The nearest up sampler: c[b, y, x, ch] = a[b, y / 2, x / 2, ch], for an output of 2H by 2W.
            // n = H | W << 16 (the input), k = channels, layer = images (0: 1).
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint H = task.n & 0xFFFFu, Wd = task.n >> 16, K = task.k, B = task.layer ? task.layer : 1;
            ulong total = (ulong)B * 4 * H * Wd * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                uint ch = (uint)(i % K);
                ulong pix = i / K;
                uint x = (uint)(pix % (2 * Wd)), y = (uint)((pix / (2 * Wd)) % (2 * H));
                uint b = (uint)(pix / (4ul * H * Wd));
                C[i] = A[(((ulong)b * H + y / 2) * Wd + x / 2) * K + ch];
            }
        }
        else if (op == OP_QUICK_GELU) {
            // c[i] = a[i] * sigmoid(1.702 * a[i]) for n rows of k values: the activation of CLIP ViT-L's text encoder.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            ulong total = (ulong)task.n * task.k;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                float v = A[i];
                C[i] = v / (1.0f + exp(-1.702f * v));
            }
        }
        else if (op == OP_GEGLU_HALF) {
            // c[t, j] = a[t, j] * gelu(a[t, k + j]) for n rows: a has rows of 2k values, c rows of k. GELU uses erf. This
            // is the GEGLU of the feed-forward of SDXL, after its projection to 2k values.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint K = task.k;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                ulong t = i / K, j = i % K;
                float h = A[t * 2 * K + j], g = A[t * 2 * K + K + j];
                C[i] = h * 0.5f * g * (1.0f + erf(g * 0.7071067811865476f));
            }
        }
        else if (op == OP_COPY_COLS) {
            // c[t, p0 + j] = a[t, j] for n rows of k values: a has rows of k values, c has rows of m values. A skip of the
            // UNet joins two tensors along the channels this way, because a row holds the channels of one pixel.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint K = task.k, M = task.m, off = task.p0;
            ulong total = (ulong)task.n * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                ulong t = i / K, j = i % K;
                C[t * M + off + j] = A[i];
            }
        }
