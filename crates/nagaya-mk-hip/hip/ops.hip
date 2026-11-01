// The row and element ops of the Qwen-Image engine on gfx1151 (milestone 22 step 4).
//
// Each tensor is bf16 in memory, row-major, with a row stride in elements. Each op computes in f32. An op rounds to
// bf16 where the reference rounds a tensor that it keeps, so the gates compare like with like (spec 11.9c).
// Each launcher returns the code of hipGetLastError after the launch.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef uint16_t bf16_t;

__device__ __forceinline__ float load(const bf16_t* p) {
    return __uint_as_float(static_cast<uint32_t>(*p) << 16);
}

// Round to the nearest bf16, ties to even. A NaN stays a quiet NaN.
__device__ __forceinline__ bf16_t round_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<bf16_t>((bits >> 16) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<bf16_t>(bits >> 16);
}

__device__ __forceinline__ float rnd(float x) {
    return __uint_as_float(static_cast<uint32_t>(round_bf16(x)) << 16);
}

constexpr int kThreads = 256;

// The sum of `v` over the block. Each thread gets the sum.
__device__ float block_sum(float v) {
    __shared__ float partial[kThreads / 32];
    __shared__ float total;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_xor(v, offset);
    }
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) s += partial[w];
        total = s;
    }
    __syncthreads();
    float result = total;
    __syncthreads();
    return result;
}

// out = x / sqrt(mean(x^2) + eps) * (w + offset). One block for each row.
__global__ void rmsnorm_kernel(const bf16_t* x, int64_t x_stride, const bf16_t* w, float offset, bf16_t* out,
                               int64_t out_stride, int dim, float eps) {
    const bf16_t* row = x + blockIdx.x * x_stride;
    bf16_t* o = out + blockIdx.x * out_stride;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = load(row + i);
        sum += v * v;
    }
    float inv = rsqrtf(block_sum(sum) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        o[i] = round_bf16(load(row + i) * inv * (load(w + i) + offset));
    }
}

// out = layer_norm(x) * (1 + scale), with no affine weight. The norm rounds to bf16 before the scale, as the
// reference keeps it. One block for each row.
__global__ void layernorm_modulate_kernel(const bf16_t* x, int64_t x_stride, const bf16_t* scale, bf16_t* out,
                                          int64_t out_stride, int dim, float eps) {
    const bf16_t* row = x + blockIdx.x * x_stride;
    bf16_t* o = out + blockIdx.x * out_stride;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += load(row + i);
    float mean = block_sum(sum) / dim;
    float var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = load(row + i) - mean;
        var += d * d;
    }
    float inv = rsqrtf(block_sum(var) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normed = rnd((load(row + i) - mean) * inv);
        float factor = rnd(1.0f + load(scale + i));
        o[i] = round_bf16(normed * factor);
    }
}

// h = h + tanh(gate) * y, each step rounded to bf16 as the reference does. One thread for each element.
__global__ void gate_add_kernel(bf16_t* h, int64_t h_stride, const bf16_t* y, int64_t y_stride, const bf16_t* gate,
                                int rows, int dim) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(rows) * dim) return;
    int64_t r = index / dim;
    int c = static_cast<int>(index % dim);
    float g = rnd(tanhf(load(gate + c)));
    float added = rnd(g * load(y + r * y_stride + c));
    bf16_t* p = h + r * h_stride + c;
    *p = round_bf16(load(p) + added);
}

// out[r, j] = silu(x[r, j]) * x[r, half + j]: the SwiGLU of a fused gate and up projection.
__global__ void silu_mul_kernel(const bf16_t* x, int64_t x_stride, int half, bf16_t* out, int64_t out_stride,
                                int rows) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(rows) * half) return;
    int64_t r = index / half;
    int j = static_cast<int>(index % half);
    float g = load(x + r * x_stride + j);
    float u = load(x + r * x_stride + half + j);
    float s = rnd(g / (1.0f + expf(-g)));
    out[r * out_stride + j] = round_bf16(s * u);
}

// x = gelu(x) with the tanh form, in place.
__global__ void gelu_tanh_kernel(bf16_t* x, int64_t stride, int rows, int dim) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(rows) * dim) return;
    bf16_t* p = x + (index / dim) * stride + index % dim;
    float v = load(p);
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    *p = round_bf16(0.5f * v * (1.0f + tanhf(inner)));
}

// x = silu(x), in place.
__global__ void silu_kernel(bf16_t* x, int64_t n) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    float v = load(x + index);
    x[index] = round_bf16(v / (1.0f + expf(-v)));
}

// For each row and head: RMSNorm over the head with weight `w`, a round to bf16, then the rotation of each pair by
// the angle of its row. `pairs` is 1 for the pairs (2i, 2i+1), and 0 for the pairs (i, i + head_dim / 2).
// `cs` holds (cos, sin) in f32 for each row and pair. One block of head_dim / 2 threads for each row and head.
__global__ void head_norm_rope_kernel(bf16_t* x, int64_t stride, int heads, int head_dim, const bf16_t* w, float eps,
                                      const float* cs, int pairs) {
    int r = blockIdx.x / heads;
    int h = blockIdx.x % heads;
    bf16_t* head = x + static_cast<int64_t>(r) * stride + static_cast<int64_t>(h) * head_dim;
    int half = head_dim / 2;
    int i = threadIdx.x;
    int a = pairs ? 2 * i : i;
    int b = pairs ? 2 * i + 1 : i + half;
    float xa = 0.0f, xb = 0.0f;
    if (i < half) {
        xa = load(head + a);
        xb = load(head + b);
    }
    __shared__ float partial[32];
    float v = xa * xa + xb * xb;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) v += __shfl_xor(v, offset);
    if (threadIdx.x % warpSize == 0) partial[threadIdx.x / warpSize] = v;
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < static_cast<int>((blockDim.x + warpSize - 1) / warpSize); ++k) sum += partial[k];
    float inv = rsqrtf(sum / head_dim + eps);
    if (i < half) {
        float na = w ? rnd(rnd(xa * inv) * load(w + a)) : rnd(xa * inv);
        float nb = w ? rnd(rnd(xb * inv) * load(w + b)) : rnd(xb * inv);
        float c = cs[(static_cast<int64_t>(r) * half + i) * 2];
        float s = cs[(static_cast<int64_t>(r) * half + i) * 2 + 1];
        head[a] = round_bf16(na * c - nb * s);
        head[b] = round_bf16(nb * c + na * s);
    }
}

// out[r] = table[ids[r]]: the rows of an embedding table.
__global__ void gather_rows_kernel(const bf16_t* table, const int32_t* ids, bf16_t* out, int dim) {
    int r = blockIdx.x;
    const bf16_t* src = table + static_cast<int64_t>(ids[r]) * dim;
    bf16_t* dst = out + static_cast<int64_t>(r) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

unsigned blocks(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

}  // namespace

extern "C" {

int nmk_rmsnorm(const void* x, int64_t x_stride, const void* w, float offset, void* out, int64_t out_stride,
                int rows, int dim, float eps, void* stream) {
    if (rows > 0) {
        hipLaunchKernelGGL(rmsnorm_kernel, dim3(rows), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<const bf16_t*>(x), x_stride, static_cast<const bf16_t*>(w), offset,
                           static_cast<bf16_t*>(out), out_stride, dim, eps);
    }
    return hipGetLastError();
}

int nmk_layernorm_modulate(const void* x, int64_t x_stride, const void* scale, void* out, int64_t out_stride, int rows,
                           int dim, float eps, void* stream) {
    if (rows > 0) {
        hipLaunchKernelGGL(layernorm_modulate_kernel, dim3(rows), dim3(kThreads), 0,
                           static_cast<hipStream_t>(stream), static_cast<const bf16_t*>(x), x_stride,
                           static_cast<const bf16_t*>(scale), static_cast<bf16_t*>(out), out_stride, dim, eps);
    }
    return hipGetLastError();
}

int nmk_gate_add(void* h, int64_t h_stride, const void* y, int64_t y_stride, const void* gate, int rows, int dim,
                 void* stream) {
    int64_t n = static_cast<int64_t>(rows) * dim;
    if (n > 0) {
        hipLaunchKernelGGL(gate_add_kernel, dim3(blocks(n)), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<bf16_t*>(h), h_stride, static_cast<const bf16_t*>(y), y_stride,
                           static_cast<const bf16_t*>(gate), rows, dim);
    }
    return hipGetLastError();
}

int nmk_silu_mul(const void* x, int64_t x_stride, int half, void* out, int64_t out_stride, int rows, void* stream) {
    int64_t n = static_cast<int64_t>(rows) * half;
    if (n > 0) {
        hipLaunchKernelGGL(silu_mul_kernel, dim3(blocks(n)), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<const bf16_t*>(x), x_stride, half, static_cast<bf16_t*>(out), out_stride, rows);
    }
    return hipGetLastError();
}

int nmk_gelu_tanh(void* x, int64_t stride, int rows, int dim, void* stream) {
    int64_t n = static_cast<int64_t>(rows) * dim;
    if (n > 0) {
        hipLaunchKernelGGL(gelu_tanh_kernel, dim3(blocks(n)), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<bf16_t*>(x), stride, rows, dim);
    }
    return hipGetLastError();
}

int nmk_silu(void* x, int64_t n, void* stream) {
    if (n > 0) {
        hipLaunchKernelGGL(silu_kernel, dim3(blocks(n)), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<bf16_t*>(x), n);
    }
    return hipGetLastError();
}

int nmk_head_norm_rope(void* x, int64_t stride, int rows, int heads, int head_dim, const void* w, float eps,
                       const float* cs, int pairs, void* stream) {
    if (head_dim % 2 != 0 || head_dim / 2 > 1024) return static_cast<int>(hipErrorInvalidValue);
    if (rows > 0) {
        int threads = ((head_dim / 2 + 31) / 32) * 32;
        hipLaunchKernelGGL(head_norm_rope_kernel, dim3(rows * heads), dim3(threads), 0,
                           static_cast<hipStream_t>(stream), static_cast<bf16_t*>(x), stride, heads, head_dim,
                           static_cast<const bf16_t*>(w), eps, cs, pairs);
    }
    return hipGetLastError();
}

int nmk_gather_rows(const void* table, const int32_t* ids, void* out, int rows, int dim, void* stream) {
    if (rows > 0) {
        hipLaunchKernelGGL(gather_rows_kernel, dim3(rows), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<const bf16_t*>(table), ids, static_cast<bf16_t*>(out), dim);
    }
    return hipGetLastError();
}

}  // extern "C"
