// SPDX-License-Identifier: MIT
// The rounding rule gdn-bf16-v1 (docs/park.md): a held context's GDN recurrent state rounded in place to bf16 (round to
// nearest, ties to even, on the f32 bit pattern), kept as f32 with the low 16 bits zero. Its own translation unit with
// one kernel (two kernels in one TU, and LDS reached through a non-inlined call, have hung this GPU silently; k_stage and
// k_mega keep TUs of their own): a plain grid-stride loop, no LDS, no barriers, no float atomics. 108 MiB read and
// written once, about a millisecond.
#include "engine.h"

namespace mk {

using namespace model;

namespace {

constexpr uint32_t kRoundBlock = 256, kRoundGrid = 1024;
constexpr uint32_t kNoBad = 0xffffffffu;

// Each element: its rounded bits, written back; or, if it is not finite or rounds to infinity, left as it was and its
// index offered to *bad (the lowest such index wins: atomicMin on an integer, so the report doesn't depend on the
// schedule) and *n_bad counted.
__global__ void __launch_bounds__(kRoundBlock) k_round_bf16(uint32_t *s, uint32_t n, uint32_t *bad, uint32_t *n_bad) {
  for (uint32_t i = blockIdx.x * kRoundBlock + threadIdx.x; i < n; i += gridDim.x * kRoundBlock) {
    const uint32_t u = s[i];
    const uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    if ((u & 0x7f800000u) == 0x7f800000u || (r & 0x7f800000u) == 0x7f800000u) {
      atomicMin(bad, i);
      atomicAdd(n_bad, 1u);
      continue;
    }
    s[i] = r;
  }
}

}  // namespace

void Engine::round_gdn_bf16(Context *c) {
  if (c->poisoned) MK_DIE("round_gdn_bf16 of a context poisoned by a spin-timeout (%u tokens): its state is half-written", c->n);
  const uint64_t n = 36ull * kGdnVHeads * kGdnKDim * kGdnVDim;
  static_assert(36ull * kGdnVHeads * kGdnKDim * kGdnVDim < kNoBad, "the element index fits 32 bits below the sentinel");
  uint32_t *flag = nullptr;  // [0] the lowest bad index, [1] how many
  MK_CHECK(hipMalloc(&flag, 2 * sizeof(uint32_t)));
  const uint32_t init[2] = {kNoBad, 0};
  MK_CHECK(hipMemcpy(flag, init, sizeof(init), hipMemcpyHostToDevice));
  k_round_bf16<<<kRoundGrid, kRoundBlock>>>(reinterpret_cast<uint32_t *>(c->gdn_state), uint32_t(n), flag, flag + 1);
  MK_CHECK(hipGetLastError());
  uint32_t got[2];
  MK_CHECK(hipMemcpy(got, flag, sizeof(got), hipMemcpyDeviceToHost));  // the null stream: waits for the kernel
  MK_CHECK(hipFree(flag));
  if (got[0] == kNoBad && got[1] == 0) return;
  if (got[0] == kNoBad || got[1] == 0) MK_DIE("round_gdn_bf16: inconsistent report (index %u, count %u)", got[0], got[1]);
  // The offending element was left as it was: read it back for the report.
  uint32_t u = 0;
  MK_CHECK(hipMemcpy(&u, reinterpret_cast<uint32_t *>(c->gdn_state) + got[0], sizeof(u), hipMemcpyDeviceToHost));
  float f;
  memcpy(&f, &u, sizeof(f));
  const uint32_t i = got[0], v = i % kGdnVDim, k = (i / kGdnVDim) % kGdnKDim, h = (i / (kGdnVDim * kGdnKDim)) % kGdnVHeads,
                 gl = i / (kGdnVDim * kGdnKDim * kGdnVHeads);
  // Every 4th model layer (3, 7, ..., 47) is attention; GDN layer gl is model layer gl + gl / 3.
  MK_DIE("round_gdn_bf16: gdn_state[%u][%u][%u][%u] (GDN layer %u = model layer %u, value head %u, index %u) is 0x%08x (%g): %s"
         " (%u such elements; a context of %u tokens). A state is never NaN or inf.",
         gl, h, k, v, gl, gl + gl / 3, h, i, u, double(f),
         (u & 0x7f800000u) == 0x7f800000u ? "not finite" : "rounds to infinity in bf16", got[1], c->n);
}

}  // namespace mk
