// The ops of the image engine (SDXL). megakernel.cl includes this file inside the loop of
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
            // A 1 x 1 kernel with s 1 and pad 0 is a linear layer; s 2 is the down sampler of SDXL. FLAG_PAD_END puts
            // the pad at the bottom and the right only, as the down sampler of the VAE encoder does: lo = 0.
            // One subgroup for each output value, its lanes over the inputs.
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint H = task.n & 0xFFFFu, Wd = task.n >> 16, M = task.m, Kin = task.k;
            uint B = task.p0 ? task.p0 : 1;
            uint kh = task.layer & 0xFFu, kw = (task.layer >> 8) & 0xFFu;
            uint st = (task.layer >> 16) & 0xFFu, pad = task.layer >> 24;
            uint lo = (flags & FLAG_PAD_END) ? 0u : pad;
            uint Ho = (H + lo + pad - kh) / st + 1, Wo = (Wd + lo + pad - kw) / st + 1;
            ulong taps = (ulong)kh * kw;
            ulong total = (ulong)B * Ho * Wo * M;
            for (ulong item = gsg; item < total; item += n_gsg) {
                uint o = (uint)(item % M);
                ulong pix = item / M;
                uint xo = (uint)(pix % Wo), yo = (uint)((pix / Wo) % Ho), b = (uint)(pix / ((ulong)Wo * Ho));
                float acc = 0.0f;
                for (uint ky = 0; ky < kh; ky++) {
                    int y = (int)(yo * st + ky) - (int)lo;
                    if (y < 0 || y >= (int)H) continue;
                    for (uint kx = 0; kx < kw; kx++) {
                        int x = (int)(xo * st + kx) - (int)lo;
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
        else if (op == OP_ADD_IMAGE_ROW) {
            // c[b, p, j] += a[b, j] for each image b of its n pixels of k values: the time embedding of a resnet, which
            // differs by image because the pooled text differs. layer = images (0: 1).
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint K = task.k, P = task.n, B = task.layer ? task.layer : 1;
            ulong total = (ulong)B * P * K;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                ulong b = i / ((ulong)P * K);
                C[i] += A[b * K + i % K];
            }
        }
        else if (op == OP_GEMM) {
            // A convolution or a linear layer on the matrix units (XMX): c[row, o] (+)= bias[o] + the sum over the taps
            // and the inputs of w[o, tap, i] * a[the pixel of the tap, i], with zero outside the image. Its K runs over
            // the taps, then the inputs: K = kernel * kernel * k. a = x (f32, pixel-major); b = the weight planes, each
            // [m][K] BF16, plane p at b + p * m * K * 2 bytes: 2 planes give an F16 weight exactly, 3 an F32 weight.
            // n = H | W << 16 (the input), m = outputs (a multiple of 16), k = inputs (a multiple of 64),
            // p0 = images (0: 1), layer = kernel | s << 8 | pad << 16 | planes << 24, x = bias (F32, or F16 with
            // FLAG_F16) or 0. A linear layer is images of 1 by 1 with kernel 1. FLAG_PAD_END puts the pad at the bottom
            // and the right only, as OP_CONV2D does.
            // A work-group stages 32 output rows by 64 of K in local memory, split into three BF16 parts as OP_MATMUL
            // does, so the products keep f32 precision: they drop only the terms below 2^-16 of the whole (2^-24 with
            // 3 planes). Each subgroup takes 16 outputs of the rows.
            __global const float* X = (__global const float*)task.a;
            __global const ushort* Wt = (__global const ushort*)task.b;
            __global float* Y = (__global float*)task.c;
            uint H = task.n & 0xFFFFu, Wd = task.n >> 16, M = task.m, Cin = task.k, B = task.p0 ? task.p0 : 1;
            uint kk = task.layer & 0xFFu, st = (task.layer >> 8) & 0xFFu, pad = (task.layer >> 16) & 0xFFu;
            uint planes = task.layer >> 24;
            uint lo = (flags & FLAG_PAD_END) ? 0u : pad;
            uint Ho = (H + lo + pad - kk) / st + 1, Wo = (Wd + lo + pad - kk) / st + 1;
            uint N = B * Ho * Wo, K = kk * kk * Cin;
            ulong plane = (ulong)M * K;
            uint row_tiles = M / 16, tok_tiles = (N + XMX_TOKENS - 1) / XMX_TOKENS;
            uint blocks = (row_tiles + num_sgs - 1) / num_sgs;
            for (uint wi = wg_id; wi < blocks * tok_tiles; wi += num_wgs) {
                uint tile = (wi % blocks) * num_sgs + sg_id, t0 = (wi / blocks) * XMX_TOKENS;
                uint r0 = tile * 16, tn = min((uint)XMX_TOKENS, N - t0);
                bool live = tile < row_tiles;
                __global const ushort* w0 = Wt + (ulong)(live ? r0 + lane : 0) * K;
                float8 acc[4];
                for (uint mt = 0; mt < 4; mt++) acc[mt] = 0.0f;
                for (uint kc = 0; kc < K; kc += XKC) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                    // A chunk of 64 lies in one tap, because the inputs are a multiple of 64.
                    uint tap = kc / Cin, i0 = kc % Cin, ky = tap / kk, kx = tap % kk;
                    for (uint i = lid; i < XMX_TOKENS * XKC; i += lsize) {
                        uint t = i / XKC, k = i % XKC, row = t0 + t;
                        float v = 0.0f;
                        if (row < N) {
                            uint xo = row % Wo, yo = (row / Wo) % Ho, b = row / (Wo * Ho);
                            int y = (int)(yo * st + ky) - (int)lo, x = (int)(xo * st + kx) - (int)lo;
                            if (y >= 0 && y < (int)H && x >= 0 && x < (int)Wd) {
                                v = X[(((ulong)b * H + y) * Wd + x) * Cin + i0 + k];
                            }
                        }
                        ulong p = bf16_split3(v);
                        uint at = ((k / 16) * 4 + t / 8) * 128 + (t % 8) * 16 + k % 16;
                        slm_x[at] = (ushort)p;
                        slm_x[XMX_TOKENS * XKC + at] = (ushort)(p >> 16);
                        slm_x[2 * XMX_TOKENS * XKC + at] = (ushort)(p >> 32);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if (!live) continue;
                    for (uint ks = 0; ks < XKC / 16; ks++) {
                        __global const uint* wk = (__global const uint*)(w0 + kc + 16 * ks);
                        int8 bh = as_int8(vload8(0, wk));
                        int8 bm = as_int8(vload8(0, (__global const uint*)(w0 + plane + kc + 16 * ks)));
                        int8 bl = planes == 3 ? as_int8(vload8(0, (__global const uint*)(w0 + 2 * plane + kc + 16 * ks))) : (int8)(0);
                        for (uint mt = 0; mt < 4; mt++) {
                            __local const ushort* a = slm_x + (ks * 4 + mt) * 128;
                            short8 ah = as_short8(intel_sub_group_block_read_us8(a));
                            short8 am = as_short8(intel_sub_group_block_read_us8(a + XMX_TOKENS * XKC));
                            short8 al = as_short8(intel_sub_group_block_read_us8(a + 2 * XMX_TOKENS * XKC));
                            if (planes == 3) {
                                acc[mt] = dpas_split6(ah, am, al, bh, bm, bl, acc[mt]);
                            } else {
                                // The F16 weight is bh + bm exactly. Smallest terms first; al * bm is below 2^-24.
                                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(al, bh, acc[mt]);
                                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(am, bm, acc[mt]);
                                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(ah, bm, acc[mt]);
                                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(am, bh, acc[mt]);
                                acc[mt] = intel_sub_group_bf16_bf16_matrix_mad_k16(ah, bh, acc[mt]);
                            }
                        }
                    }
                }
                if (!live) continue;
                float bias = task.x ? SDXL_W(task.x, r0 + lane) : 0.0f;
                for (uint mt = 0; mt < 4; mt++) {
                    for (uint r = 0; r < 8; r++) {
                        uint t = 8 * mt + r;
                        if (t < tn) {
                            __global float* y = Y + (ulong)(t0 + t) * M + r0 + lane;
                            float v = acc[mt][r] + bias;
                            *y = accumulate ? *y + v : v;
                        }
                    }
                }
            }
        }
        else if (op == OP_ATTN_PREP) {
            // The keys and the values of OP_ATTN_XMX, split into three BF16 planes (bf16_split3) for each image b and
            // head h: the key planes [3][Nkp][D], then the value planes transposed, [3][D][Nkp]. Nkp is Nk rounded up to
            // 16; the rows past Nk are zero. a = K rows, x = V rows, each [b][Nk][heads * D]; c = the planes, the block
            // of (b, h) at ((b * heads + h) * 6 * Nkp * D) ushorts. n = Nk, k = heads * D, p0 = D, layer = images.
            __global const float* Kt = (__global const float*)task.a;
            __global const float* V = (__global const float*)task.x;
            __global ushort* P = (__global ushort*)task.c;
            uint Nk = task.n, R = task.k, D = task.p0, heads = R / D, B = task.layer ? task.layer : 1;
            uint Nkp = (Nk + 15) / 16 * 16;
            ulong plane = (ulong)Nkp * D;
            ulong total = (ulong)B * heads * Nkp * D;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                uint d = (uint)(i % D);
                uint s = (uint)((i / D) % Nkp);
                ulong bh = i / plane;
                uint h = (uint)(bh % heads), b = (uint)(bh / heads);
                float kv = 0.0f, vv = 0.0f;
                if (s < Nk) {
                    ulong at = ((ulong)b * Nk + s) * R + (ulong)h * D + d;
                    kv = Kt[at];
                    vv = V[at];
                }
                __global ushort* base = P + bh * 6 * plane;
                ulong pk = bf16_split3(kv), pv = bf16_split3(vv);
                ulong kat = (ulong)s * D + d, vat = (ulong)d * Nkp + s;
                for (uint p = 0; p < 3; p++) {
                    base[p * plane + kat] = (ushort)(pk >> (16 * p));
                    base[(3 + p) * plane + vat] = (ushort)(pv >> (16 * p));
                }
            }
        }
        else if (op == OP_ATTN_XMX) {
            // Attention with no mask on the matrix units, as OP_SDPA computes it: a = Q rows [b][Nq][heads * D], b = the
            // planes of OP_ATTN_PREP, c = out rows. n = Nq, m = Nk, k = heads * D, p0 = D (16, 32, 48 or 64),
            // layer = images. One subgroup takes 8 query rows of one image and head. For each 16 keys it gets the
            // scores with dpas (the dpas A tile of the probabilities is their own layout: lane = key, element = row),
            // keeps an online softmax per row, and adds the probabilities times the values with dpas. Every operand is
            // in three BF16 parts (dpas_split6), so the result keeps f32 precision.
            __global const float* Q = (__global const float*)task.a;
            __global const ushort* P = (__global const ushort*)task.b;
            __global float* C = (__global float*)task.c;
            uint Nq = task.n, Nk = task.m, R = task.k, D = task.p0, heads = R / D, B = task.layer ? task.layer : 1;
            uint Nkp = (Nk + 15) / 16 * 16, kd = D / 16;
            ulong plane = (ulong)Nkp * D;
            float scale = rsqrt((float)D);
            uint qtiles = (Nq + 7) / 8;
            ulong total = (ulong)B * heads * qtiles;
            for (ulong item = gsg; item < total; item += n_gsg) {
                uint qt = (uint)(item % qtiles);
                ulong bh = item / qtiles;
                uint h = (uint)(bh % heads), b = (uint)(bh / heads);
                uint q0 = qt * 8;
                __global const ushort* base = P + bh * 6 * plane;
                // The query tile as dpas A operands: lane = d within its 16, element r = query row.
                short8 qh[4], qm[4], ql[4];
                for (uint ks = 0; ks < kd; ks++) {
                    for (uint r = 0; r < 8; r++) {
                        uint row = q0 + r;
                        float v = row < Nq ? Q[((ulong)b * Nq + row) * R + (ulong)h * D + ks * 16 + lane] * scale : 0.0f;
                        ulong p = bf16_split3(v);
                        qh[ks][r] = (short)p;
                        qm[ks][r] = (short)(p >> 16);
                        ql[ks][r] = (short)(p >> 32);
                    }
                }
                float8 acc[4];
                for (uint nt = 0; nt < 4; nt++) acc[nt] = 0.0f;
                float mrow[8], lrow[8];
                for (uint r = 0; r < 8; r++) {
                    mrow[r] = -INFINITY;
                    lrow[r] = 0.0f;
                }
                for (uint k0 = 0; k0 < Nkp; k0 += 16) {
                    float8 s = 0.0f;
                    for (uint ks = 0; ks < kd; ks++) {
                        __global const uint* kp = (__global const uint*)(base + (ulong)(k0 + lane) * D + ks * 16);
                        int8 kh = as_int8(vload8(0, kp));
                        int8 km = as_int8(vload8(0, kp + plane / 2));
                        int8 kl = as_int8(vload8(0, kp + plane));
                        s = dpas_split6(qh[ks], qm[ks], ql[ks], kh, km, kl, s);
                    }
                    bool key = k0 + lane < Nk;
                    short8 ph, pm, pl;
                    for (uint r = 0; r < 8; r++) {
                        float sr = key ? s[r] : -INFINITY;
                        float mnew = fmax(mrow[r], sub_group_reduce_max(sr));
                        float corr = exp(mrow[r] - mnew);
                        float p = key ? exp(sr - mnew) : 0.0f;
                        lrow[r] = lrow[r] * corr + sub_group_reduce_add(p);
                        mrow[r] = mnew;
                        for (uint nt = 0; nt < kd; nt++) acc[nt][r] *= corr;
                        ulong sp = bf16_split3(p);
                        ph[r] = (short)sp;
                        pm[r] = (short)(sp >> 16);
                        pl[r] = (short)(sp >> 32);
                    }
                    for (uint nt = 0; nt < kd; nt++) {
                        __global const uint* vp = (__global const uint*)(base + 3 * plane + (ulong)(nt * 16 + lane) * Nkp + k0);
                        int8 vh = as_int8(vload8(0, vp));
                        int8 vm = as_int8(vload8(0, vp + plane / 2));
                        int8 vl = as_int8(vload8(0, vp + plane));
                        acc[nt] = dpas_split6(ph, pm, pl, vh, vm, vl, acc[nt]);
                    }
                }
                for (uint nt = 0; nt < kd; nt++) {
                    for (uint r = 0; r < 8; r++) {
                        uint row = q0 + r;
                        if (row < Nq) C[((ulong)b * Nq + row) * R + (ulong)h * D + nt * 16 + lane] = acc[nt][r] / lrow[r];
                    }
                }
            }
        }
