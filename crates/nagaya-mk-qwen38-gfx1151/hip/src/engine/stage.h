// SPDX-License-Identifier: MIT
// A pass is a list of stages. A stage is one op with its arguments. Two ways to run the list,
// one device function between them:
//   - unfused (M3): one launch of k_stage per stage;
//   - fused (M4): one cooperative launch of the megakernel, which walks the list with a grid
//     barrier between stages.
// run_stage calls the op with (wg, nwg); every op's output bits depend only on its work items
// (src/ops/common.h), so both give the same bits, and any missing barrier shows up as a
// difference.
#pragma once
#include "common.h"
#include "qmm.h"
#include "qmmh.h"
#include "small.h"
#include "engine_ops.h"
#include "gdn.h"
#include "moe.h"
#include "attn.h"
#include "attn_mask.h"

namespace qwen38 {

// Stage::sched: no grid barrier after this stage (only a workgroup one, for the LDS the ops share), and the rotation.
constexpr uint32_t kSchedNoBarrier = 1u << 31;
// kSchedTail (docs/narrow.md N3): this stage runs as the tail of the grid barrier before it: the last workgroup to arrive
// there runs it alone, as op(wg 0, nwg 1), while the others wait, and its release is the barrier after the stage
// (mega.hip, device.h grid_arrive_hold / grid_tail_release). Set only by the schedule (src/schedule.rs, enable_tail), on a stage that is
// alone in its level and last-of-level. k_stage ignores it. The rotation is the bits below.
constexpr uint32_t kSchedTail = 1u << 30;
constexpr uint32_t kSchedRotMask = kSchedTail - 1;

enum StageOp : uint32_t {
  kStQmm = 1,
  kStRms,
  kStSiluScale,
  kStHcMix,
  kStHcCombine,
  kStEmbed,
  kStRepeat4,
  kStPleGate,
  kStPleConv,
  kStAdd,
  kStGdnConv,
  kStGdnCore,
  kStMoeRouter,
  kStMoeGroup,
  kStMoeGu,
  kStMoeDown,
  kStSwiglu,
  kStMoeCombine,
  kStAttnPrep,
  kStAttnSelect,
  kStAttnCore,
  kStQmmH,  // qmm on f16 WMMA (qmmh.h, PLAN.md Speed S2); QmmArgs as kStQmm
  kStQmmKsum,  // the split-K sum after a kStQmmH stage with ksplit > 1 (qmmh.h op_qmm_ksum); the same QmmArgs
  kStHcCombineRms,  // op_hc_combine_rms: a hc_combine and the rms of its h as one stage (small.h; engine fuse_hc)
  // The indexer's selection with an attention cell mask (attn_mask.h; lab only). Only k_stage (engine.hip) runs it:
  // run_stage below doesn't know it, so the megakernel traps on it (kTrapBadOp), and the host refuses a fused pass on a
  // masked context before that (Engine::run_pass).
  kStAttnSelectMasked,
  // The fused pass's split of kStAttnSelect (attn.h op_attn_score, op_attn_emit; goal 08 I1): scores over (token, block
  // range) items, then after a barrier each token's selection. Both take AttnSelectArgs (asel). A pass within 2051
  // visible cells (dense) builds only the emit stage. The masked lab path keeps the composed kStAttnSelectMasked.
  kStAttnScore,
  kStAttnEmit,
  // The pooled block-key cache's writer (attn.h op_attn_pool; goal 08 I2): the blocks a pass completes, pooled once per
  // QSA layer and segment, between prep and score. Takes AttnSelectArgs (asel).
  kStAttnPool,
};

struct Stage {
  uint32_t op;
  uint32_t layer;  // for the event ring / debugging
  uint32_t sub;    // an op's own sub-stage index (multi-stage ops)
  // The fused pass's schedule (src/schedule.rs; 0 = the plain schedule): kSchedNoBarrier, and in the low bits the rotation,
  // the items dealt to the stages before this one in its barrier-free segment. k_mega runs the stage as workgroup
  // (blockIdx + rotation) % grid, so a segment's small stages start on different workgroups. Every op's bits depend
  // only on its work items, never on which workgroup runs them (src/ops/common.h), so the rotation is schedule only.
  uint32_t sched;
  union {
    QmmArgs qmm;
    RmsArgs rms;
    SiluScaleArgs silu;
    HcMixArgs mix;
    HcCombineArgs comb;
    HcCombineRmsArgs combrms;
    EmbedArgs embed;
    Repeat4Args rep;
    PleGateArgs pgate;
    PleConvArgs pconv;
    AddArgs add;
    GdnConvArgs gconv;
    GdnCoreArgs gcore;
    MoeRouterArgs router;
    MoeGroupArgs group;
    MoeFfnArgs ffn;
    SwigluArgs swiglu;
    MoeCombineArgs mcomb;
    AttnPrepArgs aprep;
    AttnSelectArgs asel;
    AttnSelectMaskedArgs aselm;  // kStAttnSelectMasked (unfused only)
    AttnAttendArgs aatt;
    uint8_t raw[512];
  };
};

// Always inlined into its kernel (k_stage, k_mega). Called as a real function, with qmm's 36 KiB LDS tile
// reachable from it, k_stage never finished even an embed stage on one workgroup; inlined, or with a 9 KiB
// tile, it ran in ~2 ms (tests/engine/probe_stage1.hip, 2026-09-27; PLAN.md Problems). Likely the compiler's
// LDS lowering for callees, and plausibly run 2's k_stage hang too.
__device__ __attribute__((always_inline)) inline void run_stage(const Stage &s, uint32_t wg, uint32_t nwg) {
  switch (s.op) {
    case kStQmm: op_qmm(s.qmm, wg, nwg); break;
    case kStRms: op_rms(s.rms, wg, nwg); break;
    case kStSiluScale: op_silu_scale(s.silu, wg, nwg); break;
    case kStHcMix: op_hc_mix(s.mix, wg, nwg); break;
    case kStHcCombine: op_hc_combine(s.comb, wg, nwg); break;
    case kStHcCombineRms: op_hc_combine_rms(s.combrms, wg, nwg); break;
    case kStEmbed: op_embed(s.embed, wg, nwg); break;
    case kStRepeat4: op_repeat4(s.rep, wg, nwg); break;
    case kStPleGate: op_ple_gate(s.pgate, wg, nwg); break;
    case kStPleConv: op_ple_conv(s.pconv, wg, nwg); break;
    case kStAdd: op_add(s.add, wg, nwg); break;
    case kStGdnConv: op_gdn_conv(s.gconv, wg, nwg); break;
    case kStGdnCore: op_gdn_core(s.gcore, wg, nwg); break;
    case kStMoeRouter: op_moe_router(s.router, wg, nwg); break;
    case kStMoeGroup: op_moe_group(s.group, wg, nwg); break;
    case kStMoeGu: op_moe_gu(s.ffn, wg, nwg); break;
    case kStMoeDown: op_moe_down(s.ffn, wg, nwg); break;
    case kStSwiglu: op_swiglu(s.swiglu, wg, nwg); break;
    case kStMoeCombine: op_moe_combine(s.mcomb, wg, nwg); break;
    case kStAttnPrep: op_attn_prep(s.aprep, wg, nwg); break;
    case kStAttnSelect: op_attn_select(s.asel, wg, nwg); break;
    case kStAttnScore: op_attn_score(s.asel, wg, nwg); break;
    case kStAttnEmit: op_attn_emit(s.asel, wg, nwg); break;
    case kStAttnPool: op_attn_pool(s.asel, wg, nwg); break;
    case kStAttnCore: op_attn_attend(s.aatt, wg, nwg); break;
    case kStQmmH: op_qmmh(s.qmm, wg, nwg); break;
    case kStQmmKsum: op_qmm_ksum(s.qmm, wg, nwg); break;
    default: HIP_TRAP(kTrapBadOp, kTrapFileStage);
  }
}

// The qmm stage op for a weight type: kStQmmH (f16 WMMA) when the pass asks for it and the type has that
// path, else kStQmm (exact f32). Host side, at pass build.
inline uint32_t qmm_op(uint32_t type, bool fast) { return fast && qmmh_type(type) ? kStQmmH : kStQmm; }

}  // namespace qwen38
