// The softmax of the attention scores of the Qwen-Image engine on gfx1151 (milestone 22 step 4).
//
// The scores come from a hipBLASLt GEMM in f32, already scaled. Each row of probabilities goes out in bf16 for the
// GEMM with the values. A masked key gets the probability 0.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef uint16_t bf16_t;

constexpr int kThreads = 256;

__device__ __forceinline__ bf16_t round_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<bf16_t>((bits >> 16) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<bf16_t>(bits >> 16);
}

// The reduction of `v` over the block with `op`. Each thread gets the result.
template <typename Op>
__device__ float block_reduce(float v, Op op, float identity) {
    __shared__ float partial[kThreads / 32];
    __shared__ float total;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v = op(v, __shfl_xor(v, offset));
    }
    if (threadIdx.x % warpSize == 0) partial[threadIdx.x / warpSize] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = identity;
        for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) t = op(t, partial[w]);
        total = t;
    }
    __syncthreads();
    float result = total;
    __syncthreads();
    return result;
}

// One block for each row of scores. Row `q` of each matrix may see the keys `k <= q + causal_offset` when `causal`
// is 1, and every key when it is 0.
__global__ void softmax_kernel(const float* scores, bf16_t* probabilities, int lq, int lk, int causal,
                               int causal_offset) {
    int64_t row = blockIdx.x;
    int q = static_cast<int>(row % lq);
    const float* s = scores + row * lk;
    bf16_t* p = probabilities + row * lk;
    int visible = causal ? min(lk, q + causal_offset + 1) : lk;
    float m = -INFINITY;
    for (int k = threadIdx.x; k < visible; k += blockDim.x) m = fmaxf(m, s[k]);
    m = block_reduce(m, [](float a, float b) { return fmaxf(a, b); }, -INFINITY);
    float sum = 0.0f;
    for (int k = threadIdx.x; k < visible; k += blockDim.x) sum += expf(s[k] - m);
    sum = block_reduce(sum, [](float a, float b) { return a + b; }, 0.0f);
    float inv = 1.0f / sum;
    for (int k = threadIdx.x; k < lk; k += blockDim.x) {
        p[k] = k < visible ? round_bf16(expf(s[k] - m) * inv) : static_cast<bf16_t>(0);
    }
}

}  // namespace

extern "C" {

// `matrices` score matrices of `lq` rows and `lk` columns each, packed. Each row must see at least one key.
int nmk_softmax(const float* scores, void* probabilities, int matrices, int lq, int lk, int causal, int causal_offset,
                void* stream) {
    if (causal && causal_offset < 0) return static_cast<int>(hipErrorInvalidValue);
    int64_t rows = static_cast<int64_t>(matrices) * lq;
    if (rows > 0) {
        hipLaunchKernelGGL(softmax_kernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                           static_cast<hipStream_t>(stream), scores, static_cast<bf16_t*>(probabilities), lq, lk,
                           causal, causal_offset);
    }
    return hipGetLastError();
}

}  // extern "C"
