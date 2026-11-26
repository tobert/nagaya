// The HIP form of the Beat This! task kinds: one thread for each tile of kernels/beatthis_ops.h. Each launcher queues
// its kernel on the stream and returns the error of the launch, or 0.

#include <hip/hip_runtime.h>

#define BT_FN __device__ static inline
#define BT_G
#define bt_expf expf
#define bt_erff erff
#define bt_sinf sinf
#define bt_cosf cosf
#define bt_sqrtf sqrtf
#include "beatthis_ops.h"

namespace {

constexpr int THREADS = 256;

unsigned blocks(long n) {
    return static_cast<unsigned>((n + THREADS - 1) / THREADS);
}

__device__ long tile() {
    return static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__global__ void affine_k(const float* x, float* y, long n, int cols, const float* scale, const float* shift) {
    long t = tile();
    if (t < n) bt_affine(x, y, cols, scale, shift, t);
}

__global__ void conv_k(const float* x, float* y, long n, int frames, int f_in, int cin, int cout, int kf, int kt,
                       int stride, const float* kernel) {
    long t = tile();
    if (t < n) bt_conv(x, y, frames, f_in, cin, cout, kf, kt, stride, kernel, t);
}

__global__ void gelu_k(float* x, long n) {
    long t = tile();
    if (t < n) bt_gelu(x, t);
}

__global__ void rms_norm_k(const float* x, float* y, long rows, int dim, const float* gamma) {
    long t = tile();
    if (t < rows) bt_rms_norm(x, y, dim, gamma, t);
}

__global__ void linear_k(const float* x, float* y, long n, int din, int dout, const float* weight, const float* bias) {
    long t = tile();
    if (t < n) bt_linear(x, y, din, dout, weight, bias, t);
}

__global__ void add_into_k(float* acc, const float* x, long n) {
    long t = tile();
    if (t < n) bt_add_into(acc, x, t);
}

__global__ void rope_k(const float* qkv, float* qk, long rows, int b, int axis, int heads, int hd, const float* freqs) {
    long t = tile();
    if (t < rows) bt_rope(qkv, qk, b, axis, heads, hd, freqs, t);
}

__global__ void attention_k(const float* qkv, const float* qk, const float* gates, float* out, long n, int a, int b,
                            int axis, int heads, int hd) {
    long t = tile();
    if (t < n) bt_attention(qkv, qk, gates, out, a, b, axis, heads, hd, t);
}

int launched() {
    return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" {

int bt_hip_affine(const float* x, float* y, long n, int cols, const float* scale, const float* shift, hipStream_t s) {
    affine_k<<<blocks(n), THREADS, 0, s>>>(x, y, n, cols, scale, shift);
    return launched();
}

int bt_hip_conv(const float* x, float* y, long n, int frames, int f_in, int cin, int cout, int kf, int kt, int stride,
                const float* kernel, hipStream_t s) {
    conv_k<<<blocks(n), THREADS, 0, s>>>(x, y, n, frames, f_in, cin, cout, kf, kt, stride, kernel);
    return launched();
}

int bt_hip_gelu(float* x, long n, hipStream_t s) {
    gelu_k<<<blocks(n), THREADS, 0, s>>>(x, n);
    return launched();
}

int bt_hip_rms_norm(const float* x, float* y, long rows, int dim, const float* gamma, hipStream_t s) {
    rms_norm_k<<<blocks(rows), THREADS, 0, s>>>(x, y, rows, dim, gamma);
    return launched();
}

int bt_hip_linear(const float* x, float* y, long n, int din, int dout, const float* weight, const float* bias,
                  hipStream_t s) {
    linear_k<<<blocks(n), THREADS, 0, s>>>(x, y, n, din, dout, weight, bias);
    return launched();
}

int bt_hip_add_into(float* acc, const float* x, long n, hipStream_t s) {
    add_into_k<<<blocks(n), THREADS, 0, s>>>(acc, x, n);
    return launched();
}

int bt_hip_attention(const float* qkv, float* qk, const float* gates, float* out, int a, int b, int axis, int heads,
                     int hd, const float* freqs, hipStream_t s) {
    if (hd > 64) {
        return -1;
    }
    long rows = static_cast<long>(a) * b;
    rope_k<<<blocks(rows), THREADS, 0, s>>>(qkv, qk, rows, b, axis, heads, hd, freqs);
    int e = launched();
    if (e != 0) {
        return e;
    }
    long n = rows * heads;
    attention_k<<<blocks(n), THREADS, 0, s>>>(qkv, qk, gates, out, n, a, b, axis, heads, hd);
    return launched();
}

}  // extern "C"
