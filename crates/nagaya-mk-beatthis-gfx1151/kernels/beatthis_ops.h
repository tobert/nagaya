// The task kinds of the Beat This! plan (nagaya-family-beatthis::plan::Task) as tile functions in a subset of C, for
// every GPU target: HIP now, CUDA and OpenCL C later. Each function computes one output value or one output row whole,
// so a launch maps one thread to one tile, and no two threads write one value.
//
// The includer defines, before it includes this file:
//   BT_FN       the qualifier of a device function, such as `__device__ static inline` (HIP, CUDA) or `static inline`
//               (OpenCL C);
//   BT_G        the qualifier of a pointer to global memory: empty (HIP, CUDA) or `__global` (OpenCL C);
//   bt_expf, bt_erff, bt_sinf, bt_cosf, bt_sqrtf   the f32 math functions of the target;
//   bt_fma(a, b, c)  a * b + c with one rounding, as IEEE 754 fusedMultiplyAdd: fmaf (HIP, CUDA) or fma (OpenCL C);
//   BT_HOSTDEV  the qualifier of a function that the host calls too: `__host__ __device__ static inline` (HIP,
//               CUDA) or `static inline` (OpenCL C);
//   BT_L        the qualifier of a pointer to the shared memory of a workgroup: empty (HIP, CUDA) or `__local`;
//   BT_BARRIER() a barrier of the workgroup that also orders its shared memory.
// The functions named `_tile` run on a whole workgroup. The kernel declares their shared arrays and passes them in,
// as OpenCL C asks, and passes the workgroup's ids, so no function here reads a thread id.
// The build must not contract a multiply and an add (`-ffp-contract=off`), so the bits are those of this source on
// one target: a multiply and an add fuse only where the source writes bt_fma, which rounds once on every target. Each target has its own bits (spec 8.5): its math functions are its own.
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
// slices of 32 through shared memory, stored k-major (xs and ws: 32 rows of 68 values, the last 4 a pad), so the 4
// values that a thread reads at one k are side by side. Each thread loads its 8 values of x and of weight for the next
// slice into registers while it computes the current one, so the reads of global memory overlap the sums. Each value
// sums its k in order from 0 to din - 1 in one accumulator, then adds the bias: which thread or tile computes a value
// changes no bit. A value past the rows or the outputs is not written.
#define BT_LIN_TILE 64
#define BT_LIN_K 32
#define BT_LIN_ROW 68
#define BT_LIN_EACH (BT_LIN_TILE * BT_LIN_K / 256)
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
    float px[BT_LIN_EACH];
    float pw[BT_LIN_EACH];
    for (int e = 0; e < BT_LIN_EACH; e++) {
        int l = me + 256 * e;
        int rr = l / BT_LIN_K;
        int k = l % BT_LIN_K;
        long r = r0 + rr;
        int o = o0 + rr;
        px[e] = (r < rows && k < din) ? x[r * din + k] : 0.0f;
        pw[e] = (o < dout && k < din) ? weight[(long)o * din + k] : 0.0f;
    }
    for (int k0 = 0; k0 < din; k0 += BT_LIN_K) {
        BT_BARRIER();
        for (int e = 0; e < BT_LIN_EACH; e++) {
            int l = me + 256 * e;
            int rr = l / BT_LIN_K;
            int kk = l % BT_LIN_K;
            xs[kk * BT_LIN_ROW + rr] = px[e];
            ws[kk * BT_LIN_ROW + rr] = pw[e];
        }
        BT_BARRIER();
        int k1 = k0 + BT_LIN_K;
        if (k1 < din) {
            for (int e = 0; e < BT_LIN_EACH; e++) {
                int l = me + 256 * e;
                int rr = l / BT_LIN_K;
                int k = k1 + l % BT_LIN_K;
                long r = r0 + rr;
                int o = o0 + rr;
                px[e] = (r < rows && k < din) ? x[r * din + k] : 0.0f;
                pw[e] = (o < dout && k < din) ? weight[(long)o * din + k] : 0.0f;
            }
        }
        int kn = din - k0 < BT_LIN_K ? din - k0 : BT_LIN_K;
        for (int kk = 0; kk < kn; kk++) {
            const BT_L float* xk = xs + kk * BT_LIN_ROW + 4 * ty;
            const BT_L float* wk = ws + kk * BT_LIN_ROW + 4 * tx;
            float xv[4] = {xk[0], xk[1], xk[2], xk[3]};
            float wv[4] = {wk[0], wk[1], wk[2], wk[3]};
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i][j] = bt_fma(xv[i], wv[j], acc[i][j]);
                }
            }
        }
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
#define BT_CONV_K 16

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
    for (int k0 = 0; k0 < din; k0 += BT_CONV_K) {
        for (int l = me; l < BT_LIN_TILE * BT_CONV_K; l += 256) {
            int rr = l / BT_CONV_K;
            int kk = l % BT_CONV_K;
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
        int kn = din - k0 < BT_CONV_K ? din - k0 : BT_CONV_K;
        for (int kk = 0; kk < kn; kk++) {
            float xv[4];
            float wv[4];
            for (int i = 0; i < 4; i++) {
                xv[i] = xs[(ty + 16 * i) * BT_CONV_PAD + kk];
                wv[i] = ws[(tx + 16 * i) * BT_CONV_PAD + kk];
            }
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i][j] = bt_fma(xv[i], wv[j], acc[i][j]);
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

// Tiled attention, split over the keys. A workgroup of 64 threads holds 128 queries of one sequence and one head
// (thread lid has positions qtile * 128 + lid and qtile * 128 + 64 + lid) and one part of the keys: part p of `parts`
// takes the keys [p * per, (p + 1) * per) with per = ceil(len / parts). The keys and values come in tiles of 64 through
// shared memory. Each query keeps a running max, a running sum and its weighted sum of v (online softmax), with the
// keys of a tile in groups of 8: the 8 scores, one new max, one rescale by exp(m - m'), then the 8 terms, in key order.
// The part writes (max, sum, acc[32]) of each query to `partial`; bt_attention_merge joins the parts in order.
// `parts` is bt_attention_parts(len), a function of the length alone, so the bits do not depend on the card.
// The head dim is fixed at 32, so each loop unrolls and each array stays in registers; the launcher refuses another.
#define BT_ATT_TILE 64
#define BT_ATT_Q 2
#define BT_ATT_HD 32
#define BT_ATT_GROUP 8
#define BT_ATT_KEYS_PER_PART 256
#define BT_ATT_PART (BT_ATT_HD + 2)

BT_HOSTDEV int bt_attention_parts(int len) {
    return (len + BT_ATT_KEYS_PER_PART - 1) / BT_ATT_KEYS_PER_PART;
}

// The values of `partial` for a grid of rows and heads: parts * rows * heads * (32 + 2).
BT_HOSTDEV long bt_attention_partial_floats(int a, int b, int axis, int heads) {
    int len = axis == 0 ? a : b;
    return (long)bt_attention_parts(len) * a * b * heads * BT_ATT_PART;
}

BT_FN void bt_attention_tile(const BT_G float* qkv, const BT_G float* qk, BT_G float* partial, int a, int b, int axis,
                             int heads, long seq, int h, int qtile, int part, int lid, BT_L float* ks, BT_L float* vs) {
    int width = heads * BT_ATT_HD;
    int len = axis == 0 ? a : b;
    int parts = bt_attention_parts(len);
    int per = (len + parts - 1) / parts;
    int kb = part * per;
    int ke = kb + per < len ? kb + per : len;
    long rows = (long)a * b;
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
    for (int k0 = kb; k0 < ke; k0 += BT_ATT_TILE) {
        int kp = k0 + lid;
        if (kp < ke) {
            long kr = axis == 0 ? (long)kp * b + seq : seq * b + kp;
            for (int d = 0; d < BT_ATT_HD; d++) {
                ks[lid * BT_ATT_HD + d] = qk[kr * 2 * width + width + h * BT_ATT_HD + d];
                vs[lid * BT_ATT_HD + d] = qkv[kr * 3 * width + 2 * width + h * BT_ATT_HD + d];
            }
        }
        BT_BARRIER();
        int kn = ke - k0 < BT_ATT_TILE ? ke - k0 : BT_ATT_TILE;
        for (int g0 = 0; g0 < kn; g0 += BT_ATT_GROUP) {
            int gn = kn - g0 < BT_ATT_GROUP ? kn - g0 : BT_ATT_GROUP;
            float sc[BT_ATT_Q][BT_ATT_GROUP];
            for (int jj = 0; jj < BT_ATT_GROUP; jj++) {
                const BT_L float* kj = ks + (g0 + (jj < gn ? jj : 0)) * BT_ATT_HD;
                for (int u = 0; u < BT_ATT_Q; u++) {
                    float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f, l4 = 0.0f, l5 = 0.0f, l6 = 0.0f, l7 = 0.0f;
                    for (int d = 0; d < BT_ATT_HD; d += 8) {
                        l0 = bt_fma(q[u][d], kj[d], l0);
                        l1 = bt_fma(q[u][d + 1], kj[d + 1], l1);
                        l2 = bt_fma(q[u][d + 2], kj[d + 2], l2);
                        l3 = bt_fma(q[u][d + 3], kj[d + 3], l3);
                        l4 = bt_fma(q[u][d + 4], kj[d + 4], l4);
                        l5 = bt_fma(q[u][d + 5], kj[d + 5], l5);
                        l6 = bt_fma(q[u][d + 6], kj[d + 6], l6);
                        l7 = bt_fma(q[u][d + 7], kj[d + 7], l7);
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
                        acc[u][d] = bt_fma(e, vj[d], acc[u][d]);
                    }
                }
            }
        }
        BT_BARRIER();
    }
    for (int u = 0; u < BT_ATT_Q; u++) {
        if (valid[u]) {
            BT_G float* p = partial + ((part * rows + r[u]) * heads + h) * BT_ATT_PART;
            p[0] = m[u];
            p[1] = l[u];
            for (int d = 0; d < BT_ATT_HD; d++) {
                p[2 + d] = acc[u][d];
            }
        }
    }
}

// The join of the parts of one query and head, t = row * heads + h, in part order: M = the max of the maxes, then the
// sum and acc of each part scaled by exp(m_p - M), added in order. Then acc / sum times the gate. A part with no keys
// has the max -FLT_MAX and adds zeros.
BT_FN void bt_attention_merge(const BT_G float* partial, const BT_G float* gates, BT_G float* out, int a, int b,
                              int axis, int heads, long t) {
    int len = axis == 0 ? a : b;
    int parts = bt_attention_parts(len);
    long rows = (long)a * b;
    long r = t / heads;
    int h = (int)(t % heads);
    float mx = -3.402823466e38f;
    for (int p = 0; p < parts; p++) {
        float mp = partial[((p * rows + r) * heads + h) * BT_ATT_PART];
        mx = mp > mx ? mp : mx;
    }
    float l = 0.0f;
    float acc[BT_ATT_HD];
    for (int d = 0; d < BT_ATT_HD; d++) {
        acc[d] = 0.0f;
    }
    for (int p = 0; p < parts; p++) {
        const BT_G float* q = partial + ((p * rows + r) * heads + h) * BT_ATT_PART;
        float f = bt_expf(q[0] - mx);
        l += q[1] * f;
        for (int d = 0; d < BT_ATT_HD; d++) {
            acc[d] += q[2 + d] * f;
        }
    }
    float gate = 1.0f / (1.0f + bt_expf(-gates[r * heads + h]));
    int width = heads * BT_ATT_HD;
    for (int d = 0; d < BT_ATT_HD; d++) {
        out[r * width + h * BT_ATT_HD + d] = acc[d] / l * gate;
    }
}

#endif
