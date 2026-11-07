// The ops of the audio encoder of LFM3-Omni-2B (milestone 24 step 7). megakernel.cl includes this file inside the loop
// of mk_persistent_interpreter when MK_LFM3 is defined, after lfm3_ops.cl, so these branches see its locals.
// The opcodes 110 to 119 belong to the audio encoder; the opcodes must match src/protocol.rs.
//
// The encoder keeps the plane of the downsampler channel-last: row t of a tensor is one time step, and it holds the
// mel rows h, each of the channels. The 1 x 1 convolutions of the downsampler are then plain OP_MATMUL rows.

#define OP_SUBSAMPLE        110

        else if (op == OP_SUBSAMPLE) {
            // One 3 x 3 convolution with a stride of 2 and a pad of 1 of the audio downsampler
            // (modeling_lfm3_omni.py:374-404), over the plane of time and mel:
            // c[to, ho, ch] = bias[ch] + the sum over i, j of w[ch, 3 i + j] * a[2 to - 1 + j, 2 ho - 1 + i, ci],
            // with zeros outside the plane. i steps in mel (the height of the reference), and j steps in time.
            // a = [n][k][p0]: n time rows, k mel rows, p0 channels in. With p0 = 1, each channel reads channel 0
            // (conv_in); with p0 = m, channel ch reads channel ch (the depthwise conv_mid and conv_end).
            // b = w, BF16 [m][9]; x = bias, BF16 [m]; c = [ceil(n / 2)][ceil(k / 2)][m]; layer = 1 applies silu.
            // A lane sums the taps with fma in the order of i, then j, and then adds the bias.
            __global const float* A = (__global const float*)task.a;
            __global const ushort* W = (__global const ushort*)task.b;
            __global const ushort* Bias = (__global const ushort*)task.x;
            __global float* C = (__global float*)task.c;
            uint T = task.n, H = task.k, M = task.m, CI = task.p0;
            uint TO = (T + 1) / 2, HO = (H + 1) / 2;
            bool silu = task.layer == 1u;
            ulong total = (ulong)TO * HO * M;
            for (ulong e = (ulong)gsg * 16 + lane; e < total; e += (ulong)n_gsg * 16) {
                uint ch = (uint)(e % M);
                uint ho = (uint)((e / M) % HO);
                uint to = (uint)(e / ((ulong)M * HO));
                uint ci = CI == 1u ? 0u : ch;
                float acc = 0.0f;
                for (uint i = 0; i < 3; i++) {
                    int h = 2 * (int)ho - 1 + (int)i;
                    for (uint j = 0; j < 3; j++) {
                        int t = 2 * (int)to - 1 + (int)j;
                        float v = (h >= 0 && h < (int)H && t >= 0 && t < (int)T)
                            ? A[((ulong)t * H + (uint)h) * CI + ci] : 0.0f;
                        acc = fma(bf16_to_f32(W[(ulong)ch * 9 + 3 * i + j]), v, acc);
                    }
                }
                float y = acc + bf16_to_f32(Bias[ch]);
                C[e] = silu ? y / (1.0f + exp(-y)) : y;
            }
        }
