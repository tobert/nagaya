// The hipBLASLt shim of nagaya-mk-hip: a flat C interface over the matmul calls and the index calls of
// hipblaslt_ext. Rust never mirrors a hipBLASLt struct. Each function returns 0, or a hipblasStatus_t code.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

extern "C" {

// The shape of one GEMM: D = alpha * op(A) op(B) + beta * C, in column-major order.
// The type codes are 0 for bf16 and 1 for f32. The epilogue is 0 for none.
struct nmk_gemm_shape {
    int32_t m, n, k;
    int32_t trans_a, trans_b;
    int64_t lda, ldb, ldc;
    int32_t batch;
    int64_t stride_a, stride_b, stride_c;
    int32_t type_ab, type_cd;
    int32_t epilogue;
};

// A prepared GEMM: its descriptors and the algorithm of one solution index.
struct nmk_gemm {
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t a, b, c, d;
    hipblasLtMatmulAlgo_t algo;
    size_t workspace;
};

}  // extern "C"

namespace {

constexpr int kBadArgument = 1000;

hipDataType data_type(int32_t code) {
    return code == 0 ? HIP_R_16BF : HIP_R_32F;
}

int status(hipblasStatus_t s) {
    return static_cast<int>(s);
}

#define NMK_TRY(call)                         \
    do {                                      \
        hipblasStatus_t s_ = (call);          \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {   \
            return status(s_);                \
        }                                     \
    } while (0)

int layout(hipblasLtMatrixLayout_t* out, hipDataType type, int64_t rows, int64_t cols, int64_t ld, int32_t batch,
           int64_t stride) {
    NMK_TRY(hipblasLtMatrixLayoutCreate(out, type, rows, cols, ld));
    if (batch > 1) {
        NMK_TRY(hipblasLtMatrixLayoutSetAttribute(*out, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        NMK_TRY(hipblasLtMatrixLayoutSetAttribute(*out, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                                                  sizeof(stride)));
    }
    return 0;
}

// Make the descriptors of a shape. The caller destroys them.
int describe(const nmk_gemm_shape* s, nmk_gemm* g) {
    if (s->m <= 0 || s->n <= 0 || s->k <= 0 || s->batch <= 0 || s->epilogue != 0) {
        return kBadArgument;
    }
    hipblasOperation_t ta = s->trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t tb = s->trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    NMK_TRY(hipblasLtMatmulDescCreate(&g->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    NMK_TRY(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    NMK_TRY(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    hipDataType ab = data_type(s->type_ab);
    hipDataType cd = data_type(s->type_cd);
    // op(A) is m by k, so A is stored k by m when it is transposed.
    int rc = layout(&g->a, ab, s->trans_a ? s->k : s->m, s->trans_a ? s->m : s->k, s->lda, s->batch, s->stride_a);
    if (rc != 0) return rc;
    rc = layout(&g->b, ab, s->trans_b ? s->n : s->k, s->trans_b ? s->k : s->n, s->ldb, s->batch, s->stride_b);
    if (rc != 0) return rc;
    rc = layout(&g->c, cd, s->m, s->n, s->ldc, s->batch, s->stride_c);
    if (rc != 0) return rc;
    return layout(&g->d, cd, s->m, s->n, s->ldc, s->batch, s->stride_c);
}

void destroy(nmk_gemm* g) {
    if (g->a) hipblasLtMatrixLayoutDestroy(g->a);
    if (g->b) hipblasLtMatrixLayoutDestroy(g->b);
    if (g->c) hipblasLtMatrixLayoutDestroy(g->c);
    if (g->d) hipblasLtMatrixLayoutDestroy(g->d);
    if (g->desc) hipblasLtMatmulDescDestroy(g->desc);
}

}  // namespace

extern "C" {

int nmk_blaslt_create(void** out) {
    hipblasLtHandle_t handle = nullptr;
    NMK_TRY(hipblasLtCreate(&handle));
    *out = handle;
    return 0;
}

int nmk_blaslt_version(int* out) {
    hipblasLtHandle_t handle = nullptr;
    NMK_TRY(hipblasLtCreate(&handle));
    int version = 0;
    hipblasStatus_t s = hipblasLtGetVersion(handle, &version);
    hipblasLtDestroy(handle);
    NMK_TRY(s);
    *out = version;
    return 0;
}

// The heuristic candidates of a shape, best first: their solution indices and workspace sizes.
int nmk_gemm_candidates(void* handle, const nmk_gemm_shape* s, uint64_t max_workspace, int32_t max, int32_t* indices,
                        uint64_t* workspaces, int32_t* count) {
    nmk_gemm g{};
    int rc = describe(s, &g);
    if (rc != 0) {
        destroy(&g);
        return rc;
    }
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasStatus_t st = hipblasLtMatmulPreferenceCreate(&pref);
    if (st == HIPBLAS_STATUS_SUCCESS) {
        st = hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
                                                   sizeof(max_workspace));
    }
    std::vector<hipblasLtMatmulHeuristicResult_t> results(max > 0 ? max : 1);
    int returned = 0;
    if (st == HIPBLAS_STATUS_SUCCESS) {
        st = hipblasLtMatmulAlgoGetHeuristic(static_cast<hipblasLtHandle_t>(handle), g.desc, g.a, g.b, g.c, g.d, pref,
                                             max, results.data(), &returned);
    }
    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    destroy(&g);
    NMK_TRY(st);
    int32_t n = 0;
    for (int i = 0; i < returned; ++i) {
        if (results[i].state != HIPBLAS_STATUS_SUCCESS) continue;
        indices[n] = hipblaslt_ext::getIndexFromAlgo(results[i].algo);
        workspaces[n] = results[i].workspaceSize;
        ++n;
    }
    *count = n;
    return 0;
}

// The kernel name of a solution index, cut to fit `len` with a NUL at the end.
int nmk_gemm_kernel_name(void* handle, int32_t index, char* buf, uint64_t len) {
    std::vector<int> wanted{index};
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    NMK_TRY(hipblaslt_ext::getAlgosFromIndex(static_cast<hipblasLtHandle_t>(handle), wanted, results));
    if (results.empty() || len == 0) return kBadArgument;
    std::string name = hipblaslt_ext::getKernelNameFromAlgo(static_cast<hipblasLtHandle_t>(handle), results[0].algo);
    size_t n = name.size() < len - 1 ? name.size() : len - 1;
    std::memcpy(buf, name.data(), n);
    buf[n] = 0;
    return 0;
}

// Prepare a shape with the algorithm of one solution index. The index must support the shape.
int nmk_gemm_prepare(void* handle, const nmk_gemm_shape* s, int32_t index, uint64_t max_workspace, void** out,
                     uint64_t* workspace) {
    auto* g = new (std::nothrow) nmk_gemm{};
    if (!g) return kBadArgument;
    int rc = describe(s, g);
    if (rc == 0) {
        std::vector<int> wanted{index};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        hipblasStatus_t st =
            hipblaslt_ext::getAlgosFromIndex(static_cast<hipblasLtHandle_t>(handle), wanted, results);
        if (st != HIPBLAS_STATUS_SUCCESS) {
            rc = status(st);
        } else if (results.empty()) {
            rc = kBadArgument;
        } else {
            g->algo = results[0].algo;
            float one = 1.0f;
            size_t bytes = max_workspace;
            st = hipblaslt_ext::matmulIsAlgoSupported(static_cast<hipblasLtHandle_t>(handle), g->desc, &one, g->a,
                                                      g->b, &one, g->c, g->d, g->algo, bytes);
            if (st != HIPBLAS_STATUS_SUCCESS) {
                rc = status(st);
            } else if (bytes > max_workspace) {
                rc = kBadArgument;
            } else {
                g->workspace = bytes;
            }
        }
    }
    if (rc != 0) {
        destroy(g);
        delete g;
        return rc;
    }
    *out = g;
    *workspace = g->workspace;
    return 0;
}

int nmk_gemm_run(void* handle, void* gemm, float alpha, const void* a, const void* b, float beta, const void* c,
                 void* d, void* workspace, uint64_t workspace_bytes, void* stream) {
    auto* g = static_cast<nmk_gemm*>(gemm);
    if (workspace_bytes < g->workspace) return kBadArgument;
    NMK_TRY(hipblasLtMatmul(static_cast<hipblasLtHandle_t>(handle), g->desc, &alpha, a, g->a, b, g->b, &beta, c, g->c,
                            d, g->d, &g->algo, workspace, workspace_bytes, static_cast<hipStream_t>(stream)));
    return 0;
}

void nmk_gemm_free(void* gemm) {
    auto* g = static_cast<nmk_gemm*>(gemm);
    if (!g) return;
    destroy(g);
    delete g;
}

}  // extern "C"
