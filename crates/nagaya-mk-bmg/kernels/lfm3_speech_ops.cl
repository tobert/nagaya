// The speech ops of the LFM3 engine (milestone 24 step 8): the attention of the depth decoder, and the polar form
// and the inverse FFT of the LiquidCodec decoder. megakernel.cl includes this file inside the loop of
// mk_persistent_interpreter when MK_LFM3 is defined, so these branches see its locals, as the ops of lfm3_ops.cl do.
//
// Activations are f32 and row-major. The host adds the frames of the inverse FFT and divides by the envelope.

        else if (op == OP_LFM3_GROUP_ATTN) {
            // Attention in both directions inside groups of rows, with grouped heads: c[t, h * D + j] = the sum over
            // the keys s of the group of t of softmax_s(scale * q[t, h] . k[s, g]) * v[s, g * D + j], with
            // g = h / (H / KV). The depth decoder attends over the 12 codebooks of one frame
            // (modeling_lfm3_omni.py:274-281). a = q rows [n][H * D], b = k rows [n][KV * D], x = v rows [n][KV * D],
            // c = out rows [n][H * D]. n = rows, m = H | KV << 16, k = the rows of a group (0: all n), p0 = D (a
            // multiple of 16, at most 512), layer = the bits of scale (0: 1.0). One subgroup for each row and head,
            // with an online softmax in key order.
            __global const float* Q = (__global const float*)task.a;
            __global const float* Kr = (__global const float*)task.b;
            __global const float* V = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint N = task.n, H = task.m & 0xFFFFu, KV = task.m >> 16, D = task.p0;
            uint G = task.k ? task.k : N;
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
                uint lo = t / G * G, hi = min(N, lo + G);
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
        else if (op == OP_LFM3_POLAR) {
            // The complex bins of each row from its log magnitudes and its angles (modeling_liquid_codec.py:228-229):
            // c[t, f] = exp(a[t, f]) * cos(a[t, F + f]) and c[t, F + f] = exp(a[t, f]) * sin(a[t, F + f]). a = rows
            // [n][m], c = [n][2 * F]. n = rows, k = F, m = the stride of a (at least 2 * F).
            __global const float* A = (__global const float*)task.a;
            __global float* C = (__global float*)task.c;
            uint F = task.k, stride = task.m;
            ulong total = (ulong)task.n * F;
            for (ulong i = (ulong)gsg * 16 + lane; i < total; i += (ulong)n_gsg * 16) {
                ulong t = i / F, f = i % F;
                float mag = exp(A[t * stride + f]), ang = A[t * stride + F + f];
                C[t * 2 * F + f] = mag * cos(ang);
                C[t * 2 * F + F + f] = mag * sin(ang);
            }
        }
        else if (op == OP_LFM3_IRFFT) {
            // The inverse real FFT of each row of bins, times the window (torch.fft.irfft with norm "backward",
            // modeling_liquid_codec.py:200-202): c[t, i] = w[i] / N * (re[0] + (-1)^i re[F - 1] + 2 * the sum over
            // the bins 0 < f < F - 1 of (re[f] cos(2 pi f i / N) - im[f] sin(2 pi f i / N))). The imaginary parts of
            // bins 0 and F - 1 do not count, as in irfft. a = the bins [n][2 * F] (re, then im), b = w [N] (F32),
            // x = the table [N][2] of cos and sin of 2 pi j / N (F32), c = [n][N]. n = rows, k = N (even), p0 = F
            // (N / 2 + 1). One subgroup for each output sample: the lanes split the bins, and a reduction sums them.
            __global const float* S = (__global const float*)task.a;
            __global const float* W = (__global const float*)task.b;
            __global const float* T = (__global const float*)task.x;
            __global float* C = (__global float*)task.c;
            uint Nf = task.k, F = task.p0;
            ulong total = (ulong)task.n * Nf;
            for (ulong item = gsg; item < total; item += n_gsg) {
                ulong t = item / Nf;
                uint i = (uint)(item % Nf);
                __global const float* re = S + t * 2 * F;
                __global const float* im = re + F;
                float acc = 0.0f;
                for (uint f = 1 + lane; f < F - 1; f += 16) {
                    uint j = (uint)(((ulong)f * i) % Nf);
                    acc += re[f] * T[2 * j] - im[f] * T[2 * j + 1];
                }
                acc = 2.0f * sub_group_reduce_add(acc);
                if (lane == 0) {
                    float edge = re[0] + ((i & 1u) ? -re[F - 1] : re[F - 1]);
                    C[t * Nf + i] = (edge + acc) / (float)Nf * W[i];
                }
            }
        }
