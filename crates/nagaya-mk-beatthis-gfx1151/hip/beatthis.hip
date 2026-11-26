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
#define BT_L
#define BT_BARRIER() __syncthreads()
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

__global__ void __launch_bounds__(BT_RMS_THREADS) rms_norm_tile_k(const float* x, float* y, int dim,
                                                                   const float* gamma) {
    __shared__ float part[BT_RMS_THREADS];
    bt_rms_norm_tile(x, y, dim, gamma, blockIdx.x, threadIdx.x, part);
}

__global__ void __launch_bounds__(256) linear_tile_k(const float* x, float* y, long rows, int din, int dout,
                                                     const float* weight, const float* bias) {
    __shared__ float xs[BT_LIN_TILE * BT_LIN_PAD];
    __shared__ float ws[BT_LIN_TILE * BT_LIN_PAD];
    bt_linear_tile(x, y, rows, din, dout, weight, bias, blockIdx.y, blockIdx.x, threadIdx.x, threadIdx.y, xs, ws);
}

__global__ void __launch_bounds__(64) attention_tile_k(const float* qkv, const float* qk, const float* gates, float* out,
                                                       int a, int b, int axis, int heads, int qtiles) {
    __shared__ float ks[BT_ATT_TILE * BT_ATT_HD];
    __shared__ float vs[BT_ATT_TILE * BT_ATT_HD];
    // blockIdx.x = (seq * heads + h) * qtiles + qtile
    long g = blockIdx.x;
    int qtile = static_cast<int>(g % qtiles);
    long rest = g / qtiles;
    int h = static_cast<int>(rest % heads);
    long seq = rest / heads;
    bt_attention_tile(qkv, qk, gates, out, a, b, axis, heads, seq, h, qtile, threadIdx.x, ks, vs);
}

__global__ void add_into_k(float* acc, const float* x, long n) {
    long t = tile();
    if (t < n) bt_add_into(acc, x, t);
}

__global__ void rope_k(const float* qkv, float* qk, long rows, int b, int axis, int heads, int hd, const float* freqs) {
    long t = tile();
    if (t < rows) bt_rope(qkv, qk, b, axis, heads, hd, freqs, t);
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
    rms_norm_tile_k<<<static_cast<unsigned>(rows), BT_RMS_THREADS, 0, s>>>(x, y, dim, gamma);
    return launched();
}

int bt_hip_linear(const float* x, float* y, long rows, int din, int dout, const float* weight, const float* bias,
                  hipStream_t s) {
    dim3 grid(static_cast<unsigned>((dout + BT_LIN_TILE - 1) / BT_LIN_TILE),
              static_cast<unsigned>((rows + BT_LIN_TILE - 1) / BT_LIN_TILE));
    linear_tile_k<<<grid, dim3(16, 16), 0, s>>>(x, y, rows, din, dout, weight, bias);
    return launched();
}

int bt_hip_add_into(float* acc, const float* x, long n, hipStream_t s) {
    add_into_k<<<blocks(n), THREADS, 0, s>>>(acc, x, n);
    return launched();
}

int bt_hip_attention(const float* qkv, float* qk, const float* gates, float* out, int a, int b, int axis, int heads,
                     int hd, const float* freqs, hipStream_t s) {
    // The tiled attention unrolls a head dim of 32 (kernels/beatthis_ops.h). Another head dim is a defect of the plan.
    if (hd != BT_ATT_HD) {
        return -1;
    }
    long rows = static_cast<long>(a) * b;
    rope_k<<<blocks(rows), THREADS, 0, s>>>(qkv, qk, rows, b, axis, heads, hd, freqs);
    int e = launched();
    if (e != 0) {
        return e;
    }
    int len = axis == 0 ? a : b;
    long seqs = axis == 0 ? b : a;
    int qtiles = (len + BT_ATT_TILE - 1) / BT_ATT_TILE;
    long groups = seqs * heads * qtiles;
    attention_tile_k<<<static_cast<unsigned>(groups), BT_ATT_TILE, 0, s>>>(qkv, qk, gates, out, a, b, axis, heads,
                                                                          qtiles);
    return launched();
}

}  // extern "C"
