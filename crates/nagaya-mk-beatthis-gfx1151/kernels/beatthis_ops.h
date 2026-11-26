// The task kinds of the Beat This! plan (nagaya-family-beatthis::plan::Task) as tile functions in a subset of C, for
// every GPU target: HIP now, CUDA and OpenCL C later. Each function computes one output value or one output row whole,
// so a launch maps one thread to one tile, and no two threads write one value.
//
// The includer defines, before it includes this file:
//   BT_FN       the qualifier of a device function, such as `__device__ static inline` (HIP, CUDA) or `static inline`
//               (OpenCL C);
//   BT_G        the qualifier of a pointer to global memory: empty (HIP, CUDA) or `__global` (OpenCL C);
//   bt_expf, bt_erff, bt_sinf, bt_cosf, bt_sqrtf   the f32 math functions of the target.
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

// Conv, channels last: the value t of y[frames, f_out, cout]. The sum runs over (i, j) in order, each term a dot over
// the input channels. A tap past the first or last frame reads zeros and is left out, as on the CPU.
BT_FN void bt_conv(const BT_G float* x, BT_G float* y, int frames, int f_in, int cin, int cout, int kf, int kt,
                   int stride, const BT_G float* kernel, long t) {
    int f_out = (f_in - kf) / stride + 1;
    int co = (int)(t % cout);
    long rest = t / cout;
    int o = (int)(rest % f_out);
    int frame = (int)(rest / f_out);
    int half = kt / 2;
    float acc = 0.0f;
    for (int i = 0; i < kf; i++) {
        for (int j = 0; j < kt; j++) {
            int tt = frame + j - half;
            if (tt < 0 || tt >= frames) {
                continue;
            }
            const BT_G float* xs = x + ((long)tt * f_in + (long)o * stride + i) * cin;
            const BT_G float* ks = kernel + (((long)co * kf + i) * kt + j) * cin;
            acc += bt_dot(ks, xs, cin);
        }
    }
    y[t] = acc;
}

// GELU in place, torch's exact form and order: x * 0.5 * (1 + erf(x / sqrt(2))).
BT_FN void bt_gelu(BT_G float* x, long t) {
    float v = x[t];
    x[t] = v * 0.5f * (1.0f + bt_erff(v * 0.70710678118654752f));
}

// RMSNorm of row r: y = x / max(|x|, 1e-12) * sqrt(dim) * gamma, in that order.
BT_FN void bt_rms_norm(const BT_G float* x, BT_G float* y, int dim, const BT_G float* gamma, long r) {
    const BT_G float* xr = x + r * dim;
    BT_G float* yr = y + r * dim;
    float norm = bt_sqrtf(bt_dot(xr, xr, dim));
    if (norm < 1e-12f) {
        norm = 1e-12f;
    }
    float scale = bt_sqrtf((float)dim);
    for (int i = 0; i < dim; i++) {
        yr[i] = xr[i] / norm * scale * gamma[i];
    }
}

// Linear: the value t of y[rows, dout] = dot(weight[o], x[r]) + bias[o]. `bias` may be 0.
BT_FN void bt_linear(const BT_G float* x, BT_G float* y, int din, int dout, const BT_G float* weight,
                     const BT_G float* bias, long t) {
    int o = (int)(t % dout);
    long r = t / dout;
    float d = bt_dot(weight + (long)o * din, x + r * din, din);
    y[t] = bias ? d + bias[o] : d;
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

// Attention of row r and head h: out[r, h] = sigmoid(gate) * sum over the tokens of softmax(q k / sqrt(hd)) * v. Two
// passes over the keys: the max, then exp(s - max) with its sum and the weighted sum of v, divided at the end. Thus
// a thread keeps no array of scores. The head dim is at most 64 (Beat This! has 32).
BT_FN void bt_attention(const BT_G float* qkv, const BT_G float* qk, const BT_G float* gates, BT_G float* out, int a,
                        int b, int axis, int heads, int hd, long t) {
    int h = (int)(t % heads);
    long r = t / heads;
    int width = heads * hd;
    int len = axis == 0 ? a : b;
    float scale = 1.0f / bt_sqrtf((float)hd);
    const BT_G float* q = qk + r * 2 * width + h * hd;
    float max = -3.402823466e38f;
    for (int pos = 0; pos < len; pos++) {
        const BT_G float* k = qk + bt_token(r, b, axis, pos) * 2 * width + width + h * hd;
        float s = bt_dot(q, k, hd) * scale;
        if (s > max) {
            max = s;
        }
    }
    float acc[64];
    for (int d = 0; d < hd; d++) {
        acc[d] = 0.0f;
    }
    float total = 0.0f;
    for (int pos = 0; pos < len; pos++) {
        long tok = bt_token(r, b, axis, pos);
        const BT_G float* k = qk + tok * 2 * width + width + h * hd;
        float e = bt_expf(bt_dot(q, k, hd) * scale - max);
        total += e;
        const BT_G float* v = qkv + tok * 3 * width + 2 * width + h * hd;
        for (int d = 0; d < hd; d++) {
            acc[d] += e * v[d];
        }
    }
    float gate = 1.0f / (1.0f + bt_expf(-gates[r * heads + h]));
    BT_G float* o = out + r * width + h * hd;
    for (int d = 0; d < hd; d++) {
        o[d] = acc[d] / total * gate;
    }
}

#endif
