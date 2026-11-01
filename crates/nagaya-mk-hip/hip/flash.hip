// Flash attention on gfx1151 with the WMMA matrix instructions of RDNA 3.5 (milestone 22, the speed of a step).
//
// The scores stay in registers: each wave holds 16 queries, walks the keys in blocks of 16, keeps the running max
// and sum of each row (the online softmax), and adds each block's probabilities times its values into the output.
// The probabilities round to bf16 before the product with the values, as the GEMM path does.
//
// The layouts of a WMMA 16x16x16 on a wave of 32 lanes (gfx11): lane l holds row l % 16 of A and column l % 16 of B,
// each as 16 bf16 values along k; lanes 16 to 31 repeat lanes 0 to 15. Element j of the 8 floats of C or D in lane l
// is row 2 j + l / 16, column l % 16. The probe kernel below checks these layouts.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

typedef uint16_t bf16_t;
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ bf16_t round_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<bf16_t>((bits >> 16) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<bf16_t>(bits >> 16);
}

__device__ __forceinline__ v8f wmma(v16s a, v16s b, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// The probe: C = A B for one 16x16x16 product, A and B row-major, C row-major in f32. One wave of 32 lanes.
__global__ void wmma_probe_kernel(const bf16_t* a, const bf16_t* b, float* c) {
    int l = threadIdx.x;
    v16s fa, fb;
    for (int k = 0; k < 16; ++k) {
        fa[k] = static_cast<short>(a[(l % 16) * 16 + k]);
        fb[k] = static_cast<short>(b[k * 16 + (l % 16)]);
    }
    v8f acc = {};
    acc = wmma(fa, fb, acc);
    for (int j = 0; j < 8; ++j) {
        c[(2 * j + l / 16) * 16 + (l % 16)] = acc[j];
    }
}

constexpr int kDim = 128;
constexpr int kWaves = 4;
constexpr int kQueries = 16 * kWaves;

// One workgroup: one head and 64 queries, 4 waves of 16. The keys go in blocks of 16. Row `r` may see the key `k`
// when `k < lk` and, with `causal`, `k <= r + causal_offset`.
__global__ void __launch_bounds__(128) flash_kernel(const bf16_t* q, int64_t q_stride, const bf16_t* k, int64_t k_stride,
                                                    const bf16_t* v, int64_t v_stride, bf16_t* out, int64_t out_stride,
                                                    int lq, int lk, int group, float scale, int causal,
                                                    int causal_offset) {
    __shared__ bf16_t vt[16][kDim];
    __shared__ bf16_t pt[kWaves][16][16];
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int h = blockIdx.y;
    const int kvh = h / group;
    const int q_block = blockIdx.x * kQueries;
    const int q0 = q_block + wave * 16;

    // The queries of this wave as A fragments: lane l holds row l % 16.
    v16s qa[kDim / 16];
    {
        const int row = q0 + lane % 16;
        for (int c = 0; c < kDim / 16; ++c) {
            for (int e = 0; e < 16; ++e) {
                qa[c][e] = row < lq ? static_cast<short>(q[row * q_stride + h * kDim + c * 16 + e]) : 0;
            }
        }
    }
    v8f o[kDim / 16];
    float m[8], l[8];
    for (int d = 0; d < kDim / 16; ++d) o[d] = v8f{};
    for (int j = 0; j < 8; ++j) {
        m[j] = -INFINITY;
        l[j] = 0.0f;
    }
    // The same key blocks for each wave, so the barriers of the shared tiles match.
    int key_end = lk;
    if (causal) key_end = min(lk, q_block + kQueries - 1 + causal_offset + 1);
    for (int kb = 0; kb < key_end; kb += 16) {
        // The scores of 16 queries and 16 keys: lane l holds key kb + l % 16 as a B column.
        v8f s = {};
        const int key = kb + lane % 16;
        for (int c = 0; c < kDim / 16; ++c) {
            v16s kf;
            for (int e = 0; e < 16; ++e) {
                kf[e] = key < lk ? static_cast<short>(k[key * k_stride + kvh * kDim + c * 16 + e]) : 0;
            }
            s = wmma(qa[c], kf, s);
        }
        // The online softmax of each row j of this lane: row 2 j + lane / 16, key kb + lane % 16.
        for (int j = 0; j < 8; ++j) {
            const int row = q0 + 2 * j + lane / 16;
            const bool seen = key < lk && row < lq && (!causal || key <= row + causal_offset);
            float x = seen ? s[j] * scale : -INFINITY;
            float top = x;
            for (int w = 8; w > 0; w /= 2) top = fmaxf(top, __shfl_xor(top, w));
            const float m_new = fmaxf(m[j], top);
            float p = 0.0f, corr = 1.0f;
            if (m_new != -INFINITY) {
                p = seen ? expf(x - m_new) : 0.0f;
                corr = m[j] == -INFINITY ? 0.0f : expf(m[j] - m_new);
            }
            float sum = p;
            for (int w = 8; w > 0; w /= 2) sum += __shfl_xor(sum, w);
            l[j] = l[j] * corr + sum;
            m[j] = m_new;
            for (int d = 0; d < kDim / 16; ++d) o[d][j] *= corr;
            pt[wave][2 * j + lane / 16][lane % 16] = round_bf16(p);
        }
        // The values of the block, shared by the 4 waves: thread t loads 16 values of key t / 8.
        {
            const int t = threadIdx.x;
            const int vkey = kb + t / 8;
            const int chunk = t % 8;
            for (int e = 0; e < 16; ++e) {
                vt[t / 8][chunk * 16 + e] = vkey < lk ? v[vkey * v_stride + kvh * kDim + chunk * 16 + e] : 0;
            }
        }
        __syncthreads();
        // The probabilities as an A fragment (lane l: row l % 16), times the values (lane l: column l % 16).
        v16s pa;
        for (int e = 0; e < 16; ++e) pa[e] = static_cast<short>(pt[wave][lane % 16][e]);
        for (int d = 0; d < kDim / 16; ++d) {
            v16s vb;
            for (int e = 0; e < 16; ++e) vb[e] = static_cast<short>(vt[e][d * 16 + lane % 16]);
            o[d] = wmma(pa, vb, o[d]);
        }
        __syncthreads();
    }
    for (int j = 0; j < 8; ++j) {
        const int row = q0 + 2 * j + lane / 16;
        if (row >= lq) continue;
        const float inv = l[j] > 0.0f ? 1.0f / l[j] : 0.0f;
        for (int d = 0; d < kDim / 16; ++d) {
            out[row * out_stride + h * kDim + d * 16 + lane % 16] = round_bf16(o[d][j] * inv);
        }
    }
}

}  // namespace

extern "C" {

// Attention of head size 128 for `heads` query heads, `group` of them on each key head. Each row must see a key.
int nmk_flash(const void* q, int64_t q_stride, const void* k, int64_t k_stride, const void* v, int64_t v_stride,
              void* out, int64_t out_stride, int lq, int lk, int heads, int group, float scale, int causal,
              int causal_offset, void* stream) {
    if (lq <= 0 || lk <= 0 || heads <= 0 || group <= 0 || heads % group != 0) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    dim3 grid((lq + kQueries - 1) / kQueries, heads);
    hipLaunchKernelGGL(flash_kernel, grid, dim3(32 * kWaves), 0, static_cast<hipStream_t>(stream),
                       static_cast<const bf16_t*>(q), q_stride, static_cast<const bf16_t*>(k), k_stride,
                       static_cast<const bf16_t*>(v), v_stride, static_cast<bf16_t*>(out), out_stride, lq, lk, group,
                       scale, causal, causal_offset);
    return hipGetLastError();
}

int nmk_wmma_probe(const void* a, const void* b, float* c, void* stream) {
    hipLaunchKernelGGL(wmma_probe_kernel, dim3(1), dim3(32), 0, static_cast<hipStream_t>(stream),
                       static_cast<const bf16_t*>(a), static_cast<const bf16_t*>(b), c);
    return hipGetLastError();
}

}  // extern "C"
