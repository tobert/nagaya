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
