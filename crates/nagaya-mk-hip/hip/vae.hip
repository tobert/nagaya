// The ops of the Qwen-Image VAE decoder on gfx1151 (milestone 22 step 4).
//
// The decoder keeps each image channels-last: one row of channels for each pixel, in row order. A 3 by 3 conv is an
// im2col copy and one GEMM. Each op computes in f32 and rounds to bf16 where the reference keeps a bf16 tensor.
// Each launcher returns the code of hipGetLastError after the launch.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef uint16_t bf16_t;

constexpr int kThreads = 256;

__device__ __forceinline__ float load(const bf16_t* p) {
    return __uint_as_float(static_cast<uint32_t>(*p) << 16);
}

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

__device__ float block_sum(float v) {
    __shared__ float partial[kThreads / 32];
    __shared__ float total;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) v += __shfl_xor(v, offset);
    if (threadIdx.x % warpSize == 0) partial[threadIdx.x / warpSize] = v;
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

// The rows of a 3 by 3 conv with a padding of 1, for the output rows [row0, row0 + rows) of a `height` by `width`
// output. The source is `height / up` by `width / up`: `up` is 2 when a nearest 2x upsample comes first. Each output
// row of `cols` holds (ky, kx, c) in that order. One thread for each output value.
__global__ void im2col3x3_kernel(const bf16_t* x, int channels, int height, int width, int up, int row0, int rows,
                                 bf16_t* out) {
    int64_t cols = 9LL * channels;
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(rows) * width * cols) return;
    int64_t pixel = index / cols;
    int col = static_cast<int>(index % cols);
    int tap = col / channels;
    int c = col % channels;
    int py = row0 + static_cast<int>(pixel / width) + tap / 3 - 1;
    int px = static_cast<int>(pixel % width) + tap % 3 - 1;
    bf16_t value = 0;
    if (py >= 0 && py < height && px >= 0 && px < width) {
        int src_width = width / up;
        value = x[(static_cast<int64_t>(py / up) * src_width + px / up) * channels + c];
    }
    out[index] = value;
}

// The RMS norm of the VAE over the channels of each pixel (vae.py:187-219): x / max(||x||, 1e-12), times
// sqrt(channels), times gamma, each rounded to bf16, then SiLU when `silu` is 1. One block for each pixel.
__global__ void vae_norm_kernel(const bf16_t* x, const bf16_t* gamma, bf16_t* out, int channels, int silu) {
    const bf16_t* row = x + static_cast<int64_t>(blockIdx.x) * channels;
    bf16_t* o = out + static_cast<int64_t>(blockIdx.x) * channels;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        float v = load(row + i);
        sum += v * v;
    }
    float norm = fmaxf(sqrtf(block_sum(sum)), 1e-12f);
    float scale = sqrtf(static_cast<float>(channels));
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        float v = rnd(rnd(rnd(load(row + i) / norm) * scale) * load(gamma + i));
        if (silu) v = v / (1.0f + expf(-v));
        o[i] = round_bf16(v);
    }
}

// dst += the DupUp3D shortcut of src (vae.py:93-134): each output pixel (y, x) and channel c reads the source pixel
// (y / 2, x / 2) and channel f / repeats, with f = ((c * factor_t + t) * 2 + y % 2) * 2 + x % 2. The first chunk
// keeps the last time index, t = factor_t - 1. One thread for each output value.
__global__ void dup_up_add_kernel(const bf16_t* src, int src_channels, int src_height, int src_width, int factor_t,
                                  bf16_t* dst, int dst_channels) {
    int width = 2 * src_width;
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= 4LL * src_height * src_width * dst_channels) return;
    int64_t pixel = index / dst_channels;
    int c = static_cast<int>(index % dst_channels);
    int y = static_cast<int>(pixel / width);
    int x = static_cast<int>(pixel % width);
    int repeats = dst_channels * factor_t * 4 / src_channels;
    int f = ((c * factor_t + factor_t - 1) * 2 + y % 2) * 2 + x % 2;
    int ci = f / repeats;
    float s = load(src + (static_cast<int64_t>(y / 2) * src_width + x / 2) * src_channels + ci);
    dst[index] = round_bf16(load(dst + index) + s);
}

// out = x * scale + shift for each channel, each step rounded to bf16 (pipeline.py:842-852). One thread for each value.
__global__ void channel_affine_kernel(const bf16_t* x, const bf16_t* scale, const bf16_t* shift, bf16_t* out,
                                      int64_t n, int channels) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int c = static_cast<int>(index % channels);
    out[index] = round_bf16(rnd(load(x + index) * load(scale + c)) + load(shift + c));
}

// The bytes of the decoded image (vae.py:1328, image_processor.py:143, :223-235): x clamps to -1..1, u =
// bf16(x / 2 + 0.5) clamps to 0..1, and the byte is u * 255 in f32, rounded half to even. One thread for each value.
__global__ void to_bytes_kernel(const bf16_t* x, uint8_t* out, int64_t n) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    float v = fminf(fmaxf(load(x + index), -1.0f), 1.0f);
    float u = fminf(fmaxf(rnd(v * 0.5f + 0.5f), 0.0f), 1.0f);
    out[index] = static_cast<uint8_t>(rintf(u * 255.0f));
}

unsigned blocks(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

}  // namespace

extern "C" {

int nmk_im2col3x3(const void* x, int channels, int height, int width, int up, int row0, int rows, void* out,
                  void* stream) {
    if ((up != 1 && up != 2) || height % up != 0 || width % up != 0 || row0 < 0 || row0 + rows > height) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    int64_t n = static_cast<int64_t>(rows) * width * 9 * channels;
    if (n > 0) {
        hipLaunchKernelGGL(im2col3x3_kernel, dim3(blocks(n)), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<const bf16_t*>(x), channels, height, width, up, row0, rows,
                           static_cast<bf16_t*>(out));
    }
    return hipGetLastError();
}

int nmk_vae_norm(const void* x, const void* gamma, void* out, int pixels, int channels, int silu, void* stream) {
    if (pixels > 0) {
        hipLaunchKernelGGL(vae_norm_kernel, dim3(pixels), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<const bf16_t*>(x), static_cast<const bf16_t*>(gamma),
                           static_cast<bf16_t*>(out), channels, silu);
    }
    return hipGetLastError();
}

int nmk_dup_up_add(const void* src, int src_channels, int src_height, int src_width, int factor_t, void* dst,
                   int dst_channels, void* stream) {
    if ((factor_t != 1 && factor_t != 2) || (dst_channels * factor_t * 4) % src_channels != 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    int64_t n = 4LL * src_height * src_width * dst_channels;
    if (n > 0) {
        hipLaunchKernelGGL(dup_up_add_kernel, dim3(blocks(n)), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<const bf16_t*>(src), src_channels, src_height, src_width, factor_t,
                           static_cast<bf16_t*>(dst), dst_channels);
    }
    return hipGetLastError();
}

int nmk_channel_affine(const void* x, const void* scale, const void* shift, void* out, int64_t n, int channels,
                       void* stream) {
    if (n > 0) {
        hipLaunchKernelGGL(channel_affine_kernel, dim3(blocks(n)), dim3(kThreads), 0,
                           static_cast<hipStream_t>(stream), static_cast<const bf16_t*>(x),
                           static_cast<const bf16_t*>(scale), static_cast<const bf16_t*>(shift),
                           static_cast<bf16_t*>(out), n, channels);
    }
    return hipGetLastError();
}

int nmk_to_bytes(const void* x, void* out, int64_t n, void* stream) {
    if (n > 0) {
        hipLaunchKernelGGL(to_bytes_kernel, dim3(blocks(n)), dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                           static_cast<const bf16_t*>(x), static_cast<uint8_t*>(out), n);
    }
    return hipGetLastError();
}

}  // extern "C"
