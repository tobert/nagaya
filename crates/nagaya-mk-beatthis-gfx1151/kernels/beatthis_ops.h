// The task kinds of the Beat This! plan (nagaya-family-beatthis::plan::Task) as tile functions in a subset of C, for
// every GPU target: HIP now, CUDA and OpenCL C later. Each function computes one output value or one output row whole,
// so a launch maps one thread to one tile, and no two threads write one value.
//
// The includer defines, before it includes this file:
//   BT_FN       the qualifier of a device function, such as `__device__ static inline` (HIP, CUDA) or `static inline`
//               (OpenCL C);
//   BT_G        the qualifier of a pointer to global memory: empty (HIP, CUDA) or `__global` (OpenCL C);
//   bt_expf, bt_erff, bt_sinf, bt_cosf, bt_sqrtf   the f32 math functions of the target;
//   BT_L        the qualifier of a pointer to the shared memory of a workgroup: empty (HIP, CUDA) or `__local`;
//   BT_BARRIER() a barrier of the workgroup that also orders its shared memory.
// The functions named `_tile` run on a whole workgroup. The kernel declares their shared arrays and passes them in,
// as OpenCL C asks, and passes the workgroup's ids, so no function here reads a thread id.
// The build must not contract a multiply and an add (`-ffp-contract=off`), so the bits are those of this source on
// one target. Each target has its own bits (spec 8.5): its math functions are its own.
//
// Layout and meaning: `docs/spec.md` 11.19 and the doc of each Task kind in nagaya-family-beatthis/src/plan.rs.

#ifndef BEATTHIS_OPS_H
#define BEATTHIS_OPS_H

// sum of a[i] * b[i]: lane i % 8 holds the terms of its indexes in order, and the lanes join as
// ((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7)), as the CPU kernel does.
BT_FN float bt_dot(const BT_G float* a, const BT_G float* b, int n) {
    float l[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int k = 0; k < 8; k++) {
            l[k] += a[i + k] * b[i + k];
        }
    }
    for (int k = 0; i < n; i++, k++) {
        l[k] += a[i] * b[i];
    }
    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
}

// Affine: y[r, c] = x[r, c] * scale[c] + shift[c], at value index t.
BT_FN void bt_affine(const BT_G float* x, BT_G float* y, int cols, const BT_G float* scale, const BT_G float* shift,
                     long t) {
    int c = (int)(t % cols);
    y[t] = x[t] * scale[c] + shift[c];
}

// GELU in place, torch's exact form and order: x * 0.5 * (1 + erf(x / sqrt(2))).
BT_FN void bt_gelu(BT_G float* x, long t) {
    float v = x[t];
    x[t] = v * 0.5f * (1.0f + bt_erff(v * 0.70710678118654752f));
}

// RMSNorm of one row on a workgroup of 32 threads: y = x / max(|x|, 1e-12) * sqrt(dim) * gamma, in that order.
// Thread lid sums the squares of the values lid, lid + 32, ... in order; the 32 sums join in a fixed tree in shared
// memory (part: 32 values). Then each thread writes the values lid, lid + 32, ...
#define BT_RMS_THREADS 32
BT_FN void bt_rms_norm_tile(const BT_G float* x, BT_G float* y, int dim, const BT_G float* gamma, long r, int lid,
                            BT_L float* part) {
    const BT_G float* xr = x + r * dim;
    BT_G float* yr = y + r * dim;
    float sum = 0.0f;
    for (int i = lid; i < dim; i += BT_RMS_THREADS) {
        sum += xr[i] * xr[i];
    }
    part[lid] = sum;
    BT_BARRIER();
    for (int step = BT_RMS_THREADS / 2; step > 0; step /= 2) {
        if (lid < step) {
            part[lid] = part[lid] + part[lid + step];
        }
        BT_BARRIER();
    }
    float norm = bt_sqrtf(part[0]);
    if (norm < 1e-12f) {
        norm = 1e-12f;
    }
    float scale = bt_sqrtf((float)dim);
    for (int i = lid; i < dim; i += BT_RMS_THREADS) {
        yr[i] = xr[i] / norm * scale * gamma[i];
    }
}

// AddInto: acc[t] += x[t].
BT_FN void bt_add_into(BT_G float* acc, const BT_G float* x, long t) {
    acc[t] += x[t];
}

// The position of row r along the axis of a grid of a by b rows: axis 0 is A (frames), axis 1 is B (frequencies).
BT_FN int bt_position(long r, int b, int axis) {
    return axis == 0 ? (int)(r / b) : (int)(r % b);
}

// The row of the token at position pos in the sequence of row r.
BT_FN long bt_token(long r, int b, int axis, int pos) {
    return axis == 0 ? (long)pos * b + r % b : (r / b) * b + pos;
}

// RoPE: row r of qk[rows, 2 * width] gets q and k of qkv[rows, 3 * width], each pair (2i, 2i + 1) of each head turned
// by position * freqs[i]. One tile is one row.
BT_FN void bt_rope(const BT_G float* qkv, BT_G float* qk, int b, int axis, int heads, int hd, const BT_G float* freqs,
                   long r) {
    int width = heads * hd;
    float p = (float)bt_position(r, b, axis);
    const BT_G float* src = qkv + r * 3 * width;
    BT_G float* dst = qk + r * 2 * width;
    for (int part = 0; part < 2; part++) {
        for (int h = 0; h < heads; h++) {
            int base = part * width + h * hd;
            for (int i = 0; i < hd / 2; i++) {
                float angle = p * freqs[i];
                float c = bt_cosf(angle);
                float s = bt_sinf(angle);
                float x0 = src[base + 2 * i];
                float x1 = src[base + 2 * i + 1];
                dst[base + 2 * i] = x0 * c + -x1 * s;
                dst[base + 2 * i + 1] = x1 * c + x0 * s;
            }
        }
    }
}

// Tiled linear. A workgroup of 16 by 16 threads computes a tile of 64 rows by 64 outputs of y = x weight^T + bias;
// thread (ty, tx) holds the 4 by 4 values at rows row0 + 4 ty + i and outputs o0 + 4 tx + j. The k dimension comes in
// slices of 16 through shared memory, stored k-major (xs and ws: 16 rows of 68 values, the last 4 a pad), so the 4
// values that a thread reads at one k are side by side and load as one vector. Each value sums its k in order from 0
// to din - 1 in one accumulator, then adds the bias: which thread computes a value changes no bit. A value past the
// rows or the outputs is not written.
#define BT_LIN_TILE 64
#define BT_LIN_K 16
#define BT_LIN_ROW 68
BT_FN void bt_linear_tile(const BT_G float* x, BT_G float* y, long rows, int din, int dout, const BT_G float* weight,
                          const BT_G float* bias, long tile_r, int tile_o, int tx, int ty, BT_L float* xs,
                          BT_L float* ws) {
    long r0 = tile_r * BT_LIN_TILE;
    int o0 = tile_o * BT_LIN_TILE;
    float acc[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    int me = ty * 16 + tx;
    for (int k0 = 0; k0 < din; k0 += BT_LIN_K) {
        for (int l = me; l < BT_LIN_TILE * BT_LIN_K; l += 256) {
            int rr = l / BT_LIN_K;
            int kk = l % BT_LIN_K;
            int k = k0 + kk;
            long r = r0 + rr;
            int o = o0 + rr;
            xs[kk * BT_LIN_ROW + rr] = (r < rows && k < din) ? x[r * din + k] : 0.0f;
            ws[kk * BT_LIN_ROW + rr] = (o < dout && k < din) ? weight[(long)o * din + k] : 0.0f;
        }
        BT_BARRIER();
        int kn = din - k0 < BT_LIN_K ? din - k0 : BT_LIN_K;
        for (int kk = 0; kk < kn; kk++) {
            const BT_L float* xk = xs + kk * BT_LIN_ROW + 4 * ty;
            const BT_L float* wk = ws + kk * BT_LIN_ROW + 4 * tx;
            float xv[4] = {xk[0], xk[1], xk[2], xk[3]};
            float wv[4] = {wk[0], wk[1], wk[2], wk[3]};
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += xv[i] * wv[j];
                }
            }
        }
        BT_BARRIER();
    }
    for (int i = 0; i < 4; i++) {
        long r = r0 + 4 * ty + i;
        if (r >= rows) {
            continue;
        }
        for (int j = 0; j < 4; j++) {
            int o = o0 + 4 * tx + j;
            if (o < dout) {
                y[r * dout + o] = bias ? acc[i][j] + bias[o] : acc[i][j];
            }
        }
    }
}

#define BT_CONV_PAD 17

// Tiled conv as an implicit GEMM. Its shared rows are 17 values (BT_CONV_PAD), row-major. Each row of the product is one frame and output frequency (t, o), each column one
// output channel, and k runs over (i, j, ci) as the prepared kernel [cout, kf, kt, cin] lays them out. The tile and
// its sums are those of bt_linear_tile: one accumulator for each value, k in order. A tap past the first or the last
// frame reads a zero, so it adds a zero product.
BT_FN void bt_conv_tile(const BT_G float* x, BT_G float* y, int frames, int f_in, int cin, int cout, int kf, int kt,
                        int stride, const BT_G float* kernel, long tile_r, int tile_o, int tx, int ty, BT_L float* xs,
                        BT_L float* ws) {
    int f_out = (f_in - kf) / stride + 1;
    long rows = (long)frames * f_out;
    int din = kf * kt * cin;
    int half = kt / 2;
    long r0 = tile_r * BT_LIN_TILE;
    int o0 = tile_o * BT_LIN_TILE;
    float acc[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    int me = ty * 16 + tx;
    for (int k0 = 0; k0 < din; k0 += BT_LIN_K) {
        for (int l = me; l < BT_LIN_TILE * BT_LIN_K; l += 256) {
            int rr = l / BT_LIN_K;
            int kk = l % BT_LIN_K;
            int k = k0 + kk;
            long r = r0 + rr;
            float xv = 0.0f;
            if (r < rows && k < din) {
                int t = (int)(r / f_out);
                int o = (int)(r % f_out);
                int ci = k % cin;
                int ij = k / cin;
                int jj = ij % kt;
                int ii = ij / kt;
                int tt = t + jj - half;
                if (tt >= 0 && tt < frames) {
                    xv = x[((long)tt * f_in + (long)o * stride + ii) * cin + ci];
                }
            }
            xs[rr * BT_CONV_PAD + kk] = xv;
            int co = o0 + rr;
            ws[rr * BT_CONV_PAD + kk] = (co < cout && k < din) ? kernel[(long)co * din + k] : 0.0f;
        }
        BT_BARRIER();
        int kn = din - k0 < BT_LIN_K ? din - k0 : BT_LIN_K;
        for (int kk = 0; kk < kn; kk++) {
            float xv[4];
            float wv[4];
            for (int i = 0; i < 4; i++) {
                xv[i] = xs[(ty + 16 * i) * BT_CONV_PAD + kk];
                wv[i] = ws[(tx + 16 * i) * BT_CONV_PAD + kk];
            }
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += xv[i] * wv[j];
                }
            }
        }
        BT_BARRIER();
    }
    for (int i = 0; i < 4; i++) {
        long r = r0 + ty + 16 * i;
        if (r >= rows) {
            continue;
        }
        for (int j = 0; j < 4; j++) {
            int co = o0 + tx + 16 * j;
            if (co < cout) {
                y[r * cout + co] = acc[i][j];
            }
        }
    }
}

// Tiled attention. A workgroup of 64 threads holds 128 queries of one sequence and one head: thread lid has the
// queries at positions qtile * 128 + lid and qtile * 128 + 64 + lid, so each read of shared memory serves two queries.
// The keys and values come in tiles of 64 through shared memory (ks and vs: 64 rows of BT_ATT_HD values). Each query
// keeps a running max, a running sum and its weighted sum of v (online softmax). The keys of a tile go in groups of 8:
// the 8 scores first, then one new max for the group, one rescale by exp(m - m') when it grew, then the 8 exps and
// the 8 terms of the sums, in key order. The head dim is fixed at 32, so each loop unrolls and each array stays in
// registers; the launcher refuses another head dim. The dot of q and k keeps 8 lanes joined in pairs, as bt_dot does.
#define BT_ATT_TILE 64
#define BT_ATT_Q 2
#define BT_ATT_HD 32
#define BT_ATT_GROUP 8
BT_FN void bt_attention_tile(const BT_G float* qkv, const BT_G float* qk, const BT_G float* gates, BT_G float* out, int a,
                             int b, int axis, int heads, long seq, int h, int qtile, int lid, BT_L float* ks,
                             BT_L float* vs) {
    int width = heads * BT_ATT_HD;
    int len = axis == 0 ? a : b;
    float scale = 1.0f / bt_sqrtf((float)BT_ATT_HD);
    int valid[BT_ATT_Q];
    long r[BT_ATT_Q];
    float q[BT_ATT_Q][BT_ATT_HD];
    float acc[BT_ATT_Q][BT_ATT_HD];
    float m[BT_ATT_Q];
    float l[BT_ATT_Q];
    for (int u = 0; u < BT_ATT_Q; u++) {
        int pos = qtile * BT_ATT_TILE * BT_ATT_Q + u * BT_ATT_TILE + lid;
        valid[u] = pos < len;
        r[u] = axis == 0 ? (long)pos * b + seq : seq * b + pos;
        for (int d = 0; d < BT_ATT_HD; d++) {
            q[u][d] = valid[u] ? qk[r[u] * 2 * width + h * BT_ATT_HD + d] : 0.0f;
            acc[u][d] = 0.0f;
        }
        m[u] = -3.402823466e38f;
        l[u] = 0.0f;
    }
    for (int k0 = 0; k0 < len; k0 += BT_ATT_TILE) {
        int kp = k0 + lid;
        if (kp < len) {
            long kr = axis == 0 ? (long)kp * b + seq : seq * b + kp;
            for (int d = 0; d < BT_ATT_HD; d++) {
                ks[lid * BT_ATT_HD + d] = qk[kr * 2 * width + width + h * BT_ATT_HD + d];
                vs[lid * BT_ATT_HD + d] = qkv[kr * 3 * width + 2 * width + h * BT_ATT_HD + d];
            }
        }
        BT_BARRIER();
        int kn = len - k0 < BT_ATT_TILE ? len - k0 : BT_ATT_TILE;
        for (int g0 = 0; g0 < kn; g0 += BT_ATT_GROUP) {
            int gn = kn - g0 < BT_ATT_GROUP ? kn - g0 : BT_ATT_GROUP;
            float sc[BT_ATT_Q][BT_ATT_GROUP];
            for (int jj = 0; jj < BT_ATT_GROUP; jj++) {
                const BT_L float* kj = ks + (g0 + (jj < gn ? jj : 0)) * BT_ATT_HD;
                for (int u = 0; u < BT_ATT_Q; u++) {
                    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f, l4 = 0.0f, l5 = 0.0f, l6 = 0.0f, l7 = 0.0f;
                    for (int d = 0; d < BT_ATT_HD; d += 8) {
                        l0 += q[u][d] * kj[d];
                        l1 += q[u][d + 1] * kj[d + 1];
                        l2 += q[u][d + 2] * kj[d + 2];
                        l3 += q[u][d + 3] * kj[d + 3];
                        l4 += q[u][d + 4] * kj[d + 4];
                        l5 += q[u][d + 5] * kj[d + 5];
                        l6 += q[u][d + 6] * kj[d + 6];
                        l7 += q[u][d + 7] * kj[d + 7];
                    }
                    sc[u][jj] = (((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))) * scale;
                }
            }
            for (int u = 0; u < BT_ATT_Q; u++) {
                float mg = m[u];
                for (int jj = 0; jj < gn; jj++) {
                    mg = sc[u][jj] > mg ? sc[u][jj] : mg;
                }
                if (mg > m[u]) {
                    float shrink = bt_expf(m[u] - mg);
                    l[u] *= shrink;
                    for (int d = 0; d < BT_ATT_HD; d++) {
                        acc[u][d] *= shrink;
                    }
                    m[u] = mg;
                }
            }
            for (int jj = 0; jj < gn; jj++) {
                const BT_L float* vj = vs + (g0 + jj) * BT_ATT_HD;
                for (int u = 0; u < BT_ATT_Q; u++) {
                    float e = bt_expf(sc[u][jj] - m[u]);
                    l[u] += e;
                    for (int d = 0; d < BT_ATT_HD; d++) {
                        acc[u][d] += e * vj[d];
                    }
                }
            }
        }
        BT_BARRIER();
    }
    for (int u = 0; u < BT_ATT_Q; u++) {
        if (valid[u]) {
            float gate = 1.0f / (1.0f + bt_expf(-gates[r[u] * heads + h]));
            for (int d = 0; d < BT_ATT_HD; d++) {
                out[r[u] * width + h * BT_ATT_HD + d] = acc[u][d] / l[u] * gate;
            }
        }
    }
}

#endif
