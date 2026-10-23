// SPDX-License-Identifier: MIT
// The fused pass's schedule: fewer grid barriers, same bits (PLAN.md Speed, "The barrier schedule").
//
// k_mega puts a grid barrier after every stage. Most of a decode step's ~1500 stages are a few microseconds of work,
// many of them on a few workgroups while the rest wait, and a barrier costs ~1.2-4.4 us of wake-up plus exit spread.
// Two stages need a barrier between them only if one touches what the other writes. So the host derives each stage's
// footprint (the byte ranges it reads and writes) from its arguments, and between two stages it checks every hazard:
//   RAW  b reads what a writes,   WAR  b writes what a reads,   WAW  both write the same bytes.
// Ranges are the hull of what the op can touch (rows x stride), never smaller; a stage whose footprint isn't derived
// here (the attention stages, the GDN core, anything with row maps) is a fence and conflicts with everything.
// Weights and other constant tables aren't listed: no stage writes them.
//
// Siblings (goal 04 S): a pass of several segments (engine.h Segment) runs each state stage once per segment, tagged
// Stage::sub = segment + 1. The GDN core and the three attention stages are fences, but two of them with the same op
// and layer and distinct nonzero subs are compared by footprints derived from their arguments, so the segments' cores
// (or preps, selects, attends) share one level. Siblings whose footprints overlap stop the program: two segments
// would share state (forks of one parent writing the same K/V cells), which the pass's refusal should have caught.
// An attend's K/V reach (cells [0, p0 + t), the cells its select may pick) isn't in its arguments: it comes from its
// segment's prep in the same window (schedule_stages); a sibling attend without one stops the program.
//
// The scheduler, per window (the stages between two layer ends, so the per-layer capture points stay where they
// are): each stage's level is one more than the highest level of any earlier stage it conflicts with (0 if none).
// Stable-sorting a window by level keeps every conflicting pair in program order (the later stage has the higher
// level), so each stage reads the same values it did before: the same bits. Stages of one level are pairwise
// independent and run as one segment with no grid barrier between them (only a workgroup barrier, for the LDS the
// ops share); the level count is the window's longest dependency chain, the fewest barriers any order can have.
// Within a segment, stage i's work items are dealt starting where stage i - 1's ended (Stage::sched's rotation), so
// a segment's small stages land on different workgroups.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "rt.h"
#include "stage.h"

namespace mk {

struct Access {
  uintptr_t lo, hi;  // bytes [lo, hi)
  bool write;
};
struct Footprint {
  bool fence = false;
  std::vector<Access> acc;
  // A segment's state stage (Stage::sub != 0): op, layer and sub for the sibling rule; acc is derived for it although
  // it's a fence. unbounded: an attend whose K/V reach wasn't given (no prep of its segment in the window).
  uint32_t op = 0, layer = 0, sub = 0;
  bool unbounded = false;
};

namespace sched_detail {
inline void rd(Footprint &f, const void *p, uint64_t bytes) {
  if (!p || !bytes) return;
  f.acc.push_back({uintptr_t(p), uintptr_t(p) + bytes, false});
}
inline void wr(Footprint &f, const void *p, uint64_t bytes) {
  if (!p) MK_DIE("stage footprint: a null output");
  if (!bytes) return;
  f.acc.push_back({uintptr_t(p), uintptr_t(p) + bytes, true});
}
// `rows` rows of `cols` floats at a stride of `ld` floats: the hull.
inline uint64_t rows_bytes(uint32_t rows, uint64_t ld, uint64_t cols) {
  return rows == 0 ? 0 : sizeof(float) * ((uint64_t(rows) - 1) * ld + cols);
}
inline uint64_t fl(uint64_t n) { return sizeof(float) * n; }
}  // namespace sched_detail

// The four state ops whose per-segment stages can be siblings.
inline bool sibling_op(uint32_t op) { return op == kStGdnCore || op == kStAttnPrep || op == kStAttnSelect || op == kStAttnPool ||
         op == kStAttnScore || op == kStAttnEmit || op == kStAttnCore;
}

// kv_cells: for a segment's attend, the cells its K/V reads reach (its prep's p0 + t); 0 if unknown.
inline Footprint stage_footprint(const Stage &s, uint64_t kv_cells = 0) {
  using namespace sched_detail;
  using namespace model;
  Footprint f;
  if (sibling_op(s.op) && s.sub) {
    f.fence = true;
    f.op = s.op;
    f.layer = s.layer;
    f.sub = s.sub;
    constexpr uint64_t kv_cell = 2 * kKvHeads * kHeadDim, q_row = uint64_t(kHeads) * kHeadDim;
    switch (s.op) {
      case kStGdnCore: {
        const GdnCoreArgs &a = s.gcore;
        const uint64_t st = uint64_t(kGdnVHeads) * kGdnKDim * kGdnVDim;
        rd(f, a.qkv, fl(uint64_t(a.t) * kGdnConvDim));
        rd(f, a.z, fl(uint64_t(a.t) * kGdnValDim));
        rd(f, a.a, fl(uint64_t(a.t) * kGdnVHeads));
        rd(f, a.b, fl(uint64_t(a.t) * kGdnVHeads));
        rd(f, a.state_in, fl(st));
        wr(f, a.y, fl(uint64_t(a.t) * kGdnValDim));
        wr(f, a.state_out, fl(st));
        break;
      }
      case kStAttnPrep: {
        const AttnPrepArgs &a = s.aprep;
        rd(f, a.q_raw, fl(uint64_t(a.t) * kHeads * 2 * kHeadDim));
        rd(f, a.k_raw, fl(uint64_t(a.t) * kKvHeads * kHeadDim));
        rd(f, a.v_raw, fl(uint64_t(a.t) * kKvHeads * kHeadDim));
        rd(f, a.iq_raw, fl(uint64_t(a.t) * kIdxHeads * kIdxDim));
        rd(f, a.ik_raw, fl(uint64_t(a.t) * kIdxDim));
        wr(f, a.q_out, fl(a.t * q_row));
        wr(f, a.gate_out, fl(a.t * q_row));
        wr(f, a.iq_out, fl(uint64_t(a.t) * kIdxHeads * kIdxDim));
        wr(f, a.kv_cache + uint64_t(a.p0) * kv_cell, fl(uint64_t(a.t) * kv_cell));
        wr(f, a.ik_cache + uint64_t(a.p0) * kIdxDim, fl(uint64_t(a.t) * kIdxDim));
        break;
      }
      case kStAttnSelect: {
        const AttnSelectArgs &a = s.asel;
        rd(f, a.iq_out, fl(uint64_t(a.t) * kIdxHeads * kIdxDim));
        rd(f, a.ik_cache, fl((uint64_t(a.p0) + a.t) * kIdxDim));
        wr(f, a.block_scores, fl(uint64_t(a.t) * a.max_nb));
        if (a.block_scores_out) wr(f, a.block_scores_out, fl(uint64_t(a.t) * a.max_nb));
        wr(f, a.picked, sizeof(int32_t) * a.t * a.block_topk_max);
        wr(f, a.sel, sizeof(int32_t) * a.t * a.max_sel);
        wr(f, a.sel_count, sizeof(uint32_t) * a.t);
        break;
      }
      case kStAttnPool: {  // I2: blocks [pool_b0, (p0 + t) / ratio) of the cache, from the raw keys of their cells
        const AttnSelectArgs &a = s.asel;
        const uint64_t nb1 = (uint64_t(a.p0) + a.t) / a.ratio;
        if (!a.pooled) MK_DIE("stage footprint: a pool stage without a cache");
        if (a.pool_b0 > nb1) MK_DIE("stage footprint: a pool stage from block %u past its pass's %lu", a.pool_b0, (unsigned long)nb1);
        rd(f, a.ik_cache + uint64_t(a.pool_b0) * a.ratio * kIdxDim, fl((nb1 - a.pool_b0) * a.ratio * kIdxDim));
        wr(f, a.pooled + uint64_t(a.pool_b0) * kIdxDim, fl((nb1 - a.pool_b0) * kIdxDim));
        break;
      }
      case kStAttnScore: {  // I1: the scores alone. block_scores_out belongs to the composed op; the engine passes none.
        const AttnSelectArgs &a = s.asel;
        if (a.block_scores_out) MK_DIE("stage footprint: a score stage with block_scores_out");
        rd(f, a.iq_out, fl(uint64_t(a.t) * kIdxHeads * kIdxDim));
        if (a.pooled) rd(f, a.pooled, fl((uint64_t(a.p0) + a.t) / a.ratio * kIdxDim));  // I2: every block's key from the cache
        else rd(f, a.ik_cache, fl((uint64_t(a.p0) + a.t) * kIdxDim));
        wr(f, a.block_scores, fl(uint64_t(a.t) * a.max_nb));
        break;
      }
      case kStAttnEmit: {  // the selection alone: reads the scores (a dense pass reads none, nobody writes them)
        const AttnSelectArgs &a = s.asel;
        if (a.block_scores_out) MK_DIE("stage footprint: an emit stage with block_scores_out");
        rd(f, a.block_scores, fl(uint64_t(a.t) * a.max_nb));
        wr(f, a.picked, sizeof(int32_t) * a.t * a.block_topk_max);
        wr(f, a.sel, sizeof(int32_t) * a.t * a.max_sel);
        wr(f, a.sel_count, sizeof(uint32_t) * a.t);
        break;
      }
      case kStAttnCore: {
        const AttnAttendArgs &a = s.aatt;
        rd(f, a.q_out, fl(a.t * q_row));
        rd(f, a.gate_out, fl(a.t * q_row));
        if (kv_cells) rd(f, a.kv_cache, fl(kv_cells * kv_cell));
        else f.unbounded = true;
        rd(f, a.sel, sizeof(int32_t) * a.t * a.max_sel);
        rd(f, a.sel_count, sizeof(uint32_t) * a.t);
        wr(f, a.attn_out, fl(a.t * q_row));
        break;
      }
    }
    return f;
  }
  switch (s.op) {
    case kStQmm:
    case kStQmmH:
    case kStQmmKsum: {
      const QmmArgs &a = s.qmm;
      if (a.x_row || a.y_row) {  // row maps (the MoE's): not derived here
        f.fence = true;
        break;
      }
      const uint32_t ks = a.ksplit > 1 ? a.ksplit : 1;
      if (s.op == kStQmm && (ks > 1 || a.xin != kXinPlain)) MK_DIE("stage footprint: an exact qmm with split-K or fused staging");
      if (s.op == kStQmmKsum) {
        if (ks < 2 || !a.part) MK_DIE("stage footprint: a split-K sum without partials");
        rd(f, a.part, fl(uint64_t(ks) * a.t * a.n));
        wr(f, a.y, rows_bytes(a.t, a.ldy, a.n));
        break;
      }
      if (a.xin == kXinPlain) {
        rd(f, a.x, rows_bytes(a.t, a.ldx, a.k));
      } else {
        if (a.xks < 2 || !a.xp) MK_DIE("stage footprint: fused staging without partials");
        rd(f, a.xp, fl(uint64_t(a.xks) * a.t * a.k));
        if (a.xin == kXinSwigluSum) {
          if (!a.xp2) MK_DIE("stage footprint: swiglu staging without up partials");
          rd(f, a.xp2, fl(uint64_t(a.xks) * a.t * a.k));
        } else if (a.xin != kXinSiluSum) {
          MK_DIE("stage footprint: unknown xin %u", a.xin);
        }
      }
      if (a.yout != kYoutPlain) {  // hc up + hc_mix (docs/narrow.md N1): reads xn (and z), writes mixed (and inject), never y
        if (s.op != kStQmmH) MK_DIE("stage footprint: yout on op %u (f16 qmm only)", s.op);
        if (a.yout != kYoutHcMix) MK_DIE("stage footprint: unknown yout %u", a.yout);
        if (ks > 1) MK_DIE("stage footprint: yout with split-K");
        rd(f, a.mix_xn, fl(uint64_t(a.t) * 4 * a.mix_e));
        rd(f, a.mix_z, fl(uint64_t(a.t) * 4));
        wr(f, a.mix_out, fl(uint64_t(a.t) * a.mix_e));
        if (a.mix_z) wr(f, a.mix_inject, fl(uint64_t(a.t) * 4));
        break;
      }
      if (ks > 1) {
        if (!a.part) MK_DIE("stage footprint: split-K without partials");
        wr(f, a.part, fl(uint64_t(ks) * a.t * a.n));
      } else {
        wr(f, a.y, rows_bytes(a.t, a.ldy, a.n));
      }
      break;
    }
    case kStRms: {
      const RmsArgs &a = s.rms;
      rd(f, a.x, rows_bytes(a.t, a.ldx, uint64_t(a.n_groups) * a.d));
      wr(f, a.y, rows_bytes(a.t, a.ldy, uint64_t(a.n_groups) * a.d));
      break;
    }
    case kStSiluScale:
      rd(f, s.silu.x, fl(s.silu.n));
      wr(f, s.silu.y, fl(s.silu.n));
      break;
    case kStHcMix: {
      const HcMixArgs &a = s.mix;
      rd(f, a.xn, fl(uint64_t(a.t) * 4 * a.e));
      rd(f, a.up, fl(uint64_t(a.t) * 4 * a.e));
      wr(f, a.mixed, fl(uint64_t(a.t) * a.e));
      if (a.z) {  // inject is written only with z (small.h op_hc_mix)
        rd(f, a.z, fl(uint64_t(a.t) * 4));
        wr(f, a.inject, fl(uint64_t(a.t) * 4));
      }
      break;
    }
    case kStHcCombine: {
      const HcCombineArgs &a = s.comb;
      rd(f, a.h, fl(uint64_t(a.t) * 4 * a.e));
      wr(f, a.h, fl(uint64_t(a.t) * 4 * a.e));
      rd(f, a.out, fl(uint64_t(a.t) * a.e));
      rd(f, a.inject, fl(uint64_t(a.t) * 4));
      break;
    }
    case kStHcCombineRms: {  // op_hc_combine_rms: the combine's accesses, and the rms's write of xn (gamma is a constant table)
      const HcCombineRmsArgs &a = s.combrms;
      if (a.r.x != a.c.h || a.r.t != a.c.t || a.r.n_groups != 4 || a.r.d != a.c.e || a.r.ldx != 4 * a.c.e || !a.r.gamma_per_group)
        MK_DIE("stage footprint: a fused combine + rms whose rms doesn't read exactly the combine's h");
      rd(f, a.c.h, fl(uint64_t(a.c.t) * 4 * a.c.e));
      wr(f, a.c.h, fl(uint64_t(a.c.t) * 4 * a.c.e));
      rd(f, a.c.out, fl(uint64_t(a.c.t) * a.c.e));
      rd(f, a.c.inject, fl(uint64_t(a.c.t) * 4));
      wr(f, a.r.y, rows_bytes(a.r.t, a.r.ldy, uint64_t(4) * a.c.e));
      break;
    }
    case kStEmbed:
      rd(f, s.embed.ids, sizeof(int32_t) * s.embed.t);
      wr(f, s.embed.out, fl(uint64_t(s.embed.t) * s.embed.k));
      break;
    case kStRepeat4:
      rd(f, s.rep.e, fl(uint64_t(s.rep.t) * s.rep.d));
      wr(f, s.rep.h, fl(uint64_t(s.rep.t) * 4 * s.rep.d));
      break;
    case kStPleGate: {
      const PleGateArgs &a = s.pgate;
      rd(f, a.kn, fl(uint64_t(a.t) * 4 * a.e));
      rd(f, a.qn, fl(uint64_t(a.t) * 4 * a.e));
      rd(f, a.pv, fl(uint64_t(a.t) * a.e));
      wr(f, a.gv, fl(uint64_t(a.t) * 4 * a.e));
      break;
    }
    case kStPleConv: {
      const PleConvArgs &a = s.pconv;
      rd(f, a.x, fl(uint64_t(a.t) * a.c));
      rd(f, a.state, fl(uint64_t(9) * a.c));
      wr(f, a.state, fl(uint64_t(9) * a.c));
      wr(f, a.y, fl(uint64_t(a.t) * a.c));
      break;
    }
    case kStAdd:
      rd(f, s.add.h, fl(s.add.n));
      wr(f, s.add.h, fl(s.add.n));
      rd(f, s.add.a, fl(s.add.n));
      rd(f, s.add.b, fl(s.add.n));
      break;
    case kStGdnConv: {
      const GdnConvArgs &a = s.gconv;
      const uint64_t st = uint64_t(kGdnConvDim) * (kGdnConv - 1);
      rd(f, a.x, fl(uint64_t(a.t) * kGdnConvDim));
      rd(f, a.state_in, fl(st));
      wr(f, a.y, fl(uint64_t(a.t) * kGdnConvDim));
      wr(f, a.state_out, fl(st));
      break;
    }
    case kStMoeRouter:
      rd(f, s.router.logits, fl(uint64_t(s.router.t) * kExperts));
      wr(f, s.router.topk_idx, sizeof(int32_t) * s.router.t * kTopK);
      wr(f, s.router.topk_w, fl(uint64_t(s.router.t) * kTopK));
      break;
    case kStMoeGroup: {
      const MoeGroupArgs &a = s.group;
      rd(f, a.topk_idx, sizeof(int32_t) * a.t * kTopK);
      wr(f, a.count, sizeof(uint32_t) * kExperts);
      wr(f, a.offset, sizeof(uint32_t) * (kExperts + 1));
      wr(f, a.assign, sizeof(uint32_t) * a.t * kTopK);
      wr(f, a.token, sizeof(uint32_t) * a.t * kTopK);
      wr(f, a.item_gu, sizeof(uint32_t) * (kExperts + 1));
      wr(f, a.item_dn, sizeof(uint32_t) * (kExperts + 1));
      break;
    }
    case kStMoeGu:
    case kStMoeDown: {
      // MoeFfnArgs carries no T: the group's arrays and the activations at their largest (kMaxPassTokens).
      const MoeFfnArgs &a = s.ffn;
      const uint64_t maxa = uint64_t(kMaxPassTokens) * kTopK;
      rd(f, a.count, sizeof(uint32_t) * kExperts);
      rd(f, a.offset, sizeof(uint32_t) * (kExperts + 1));
      rd(f, a.assign, sizeof(uint32_t) * maxa);
      rd(f, a.token, sizeof(uint32_t) * maxa);
      rd(f, a.item_gu, sizeof(uint32_t) * (kExperts + 1));
      rd(f, a.item_dn, sizeof(uint32_t) * (kExperts + 1));
      if (s.op == kStMoeGu) {
        rd(f, a.x, fl(uint64_t(kMaxPassTokens) * kEmbd));
        wr(f, a.gu, fl(maxa * 2 * kFf));
      } else {
        rd(f, a.h, fl(maxa * kFf));
        wr(f, a.y, fl(maxa * kEmbd));
      }
      break;
    }
    case kStSwiglu: {
      const SwigluArgs &a = s.swiglu;
      rd(f, a.g, rows_bytes(a.rows, a.ldg, a.cols));
      rd(f, a.u, rows_bytes(a.rows, a.ldu, a.cols));
      wr(f, a.out, rows_bytes(a.rows, a.ldo, a.cols));
      break;
    }
    case kStMoeCombine: {
      const MoeCombineArgs &a = s.mcomb;
      rd(f, a.y, fl(uint64_t(a.t) * kTopK * kEmbd));
      rd(f, a.topk_idx, sizeof(int32_t) * a.t * kTopK);
      rd(f, a.topk_w, fl(uint64_t(a.t) * kTopK));
      rd(f, a.shared, fl(uint64_t(a.t) * kEmbd));
      rd(f, a.shared_gate, fl(a.t));
      wr(f, a.out, fl(uint64_t(a.t) * kEmbd));
      break;
    }
    default:  // kStGdnCore, kStAttnPrep, kStAttnSelect, kStAttnCore, and anything new: a fence until derived
      f.fence = true;
  }
  return f;
}

inline bool accesses_overlap(const Footprint &a, const Footprint &b) {
  for (const Access &p : a.acc)
    for (const Access &q : b.acc)
      if ((p.write || q.write) && p.lo < q.hi && q.lo < p.hi) return true;
  return false;
}
inline bool siblings(const Footprint &a, const Footprint &b) {
  return a.sub && b.sub && a.sub != b.sub && a.op == b.op && a.layer == b.layer;
}
// Siblings are independent unless they share bytes, and then the program stops (two segments share state).
inline void check_siblings(const Footprint &a, const Footprint &b) {
  if (a.unbounded || b.unbounded)
    MK_DIE("schedule: an attend of segment %u (layer %u) without its segment's prep in the window: its K/V reach is unknown",
           a.unbounded ? a.sub - 1 : b.sub - 1, a.layer);
  if (accesses_overlap(a, b))
    MK_DIE("schedule: segments %u and %u share state: their op %u stages in layer %u touch the same bytes", a.sub - 1, b.sub - 1,
           a.op, a.layer);
}
inline bool footprints_conflict(const Footprint &a, const Footprint &b) {
  if (siblings(a, b)) {
    check_siblings(a, b);
    return false;
  }
  if (a.fence || b.fence) return true;
  return accesses_overlap(a, b);
}

// A stage's work items as its op deals them (wg, wg + nwg, ...), for the rotation only: a wrong count costs balance,
// never bits.
inline uint32_t stage_items(const Stage &s) {
  auto cdiv = [](uint64_t a, uint64_t b) { return uint32_t((a + b - 1) / b); };
  switch (s.op) {
    case kStQmmH:
      return cdiv(s.qmm.n, QmmhDense::kRows) * cdiv(s.qmm.t, QmmhDense::kToks) * (s.qmm.ksplit > 1 ? s.qmm.ksplit : 1);
    case kStQmm: return cdiv(s.qmm.n, kQmmRows) * cdiv(s.qmm.t, kQmmToks);
    case kStQmmKsum: return cdiv(uint64_t(s.qmm.t) * s.qmm.n, kMkBlock);
    case kStRms: return s.rms.t * s.rms.n_groups;
    case kStSiluScale: return cdiv(s.silu.n, kMkBlock);
    case kStHcMix: return cdiv(uint64_t(s.mix.t) * s.mix.e, kMkBlock);
    case kStHcCombine: return cdiv(uint64_t(s.comb.t) * 4 * s.comb.e, kMkBlock);
    case kStHcCombineRms: return s.combrms.c.t * 4;
    case kStSwiglu: return cdiv(uint64_t(s.swiglu.rows) * s.swiglu.cols, kMkBlock);
    case kStMoeRouter: return s.router.t;
    case kStMoeCombine: return cdiv(uint64_t(s.mcomb.t) * model::kEmbd, kMkBlock);
    case kStGdnConv: return cdiv(model::kGdnConvDim, kMkBlock);
    case kStPleConv: return cdiv(s.pconv.c, kMkBlock);
    case kStGdnCore: return model::kGdnVHeads;
    case kStAttnPrep: return cdiv(uint64_t(s.aprep.t) * kAttnPrepUnits, kWavesPerBlock);
    case kStAttnSelect: return s.asel.t;
    case kStAttnPool: return cdiv(attn_pool_blocks(s.asel), kWavesPerBlock);
    case kStAttnScore: return s.asel.t * attn_score_ranges(s.asel);
    case kStAttnEmit: return s.asel.t;
    case kStAttnCore: return s.aatt.t == 1 ? model::kHeads : s.aatt.t * (model::kHeads / kAttnHeadsPerWave);
    default: return 1;
  }
}

// Reorders S within each window (window_starts: ascending stage indices where a new window begins; 0 is implied)
// and sets every stage's sched. Returns the grid barriers the pass will run (a tail stage's two halves count as the
// two barriers they replace: callers that count the executed ones subtract the tails).
// enable_tail (docs/narrow.md N3): also marks kSchedTail on a stage that is alone in its level, whose op is on the tail
// allowlist (the fused combine + rms), with at most kTailMaxItems work items (one runner does them all, serially), and
// that isn't the pass's first stage. The stage before it is then the last of its level (its barrier is the tail's first
// half); the schedule stops the program if that doesn't hold.
constexpr uint32_t kTailMaxItems = 4;
// The allowlist: the fused combine + rms (its items are bounded by kTailMaxItems), and moe_group, which op_moe_group already
// does in one workgroup (`if (wg != 0) return`, no use of nwg; stage_items is 1 at every T), so op(a, 0, 1) on the runner is
// the same work and bits. It is allowed at T = 1 only (docs/narrow.md N3b): the barrier it saves is the same at any T, but
// the serial per-assignment loops grow with T and the T > 1 passes stay as they were.
inline bool tail_op(const Stage &s) {
  return s.op == kStHcCombineRms || (s.op == kStMoeGroup && s.group.t == 1);
}
inline uint32_t schedule_stages(std::vector<Stage> &S, const std::vector<uint32_t> &window_starts, bool enable_tail = false) {
  const uint32_t n = uint32_t(S.size());
  std::vector<uint32_t> bounds{0};
  for (uint32_t w : window_starts) {
    if (w < bounds.back() || w > n) MK_DIE("schedule: window start %u out of order or past %u stages", w, n);
    if (w > bounds.back()) bounds.push_back(w);
  }
  if (bounds.back() != n) bounds.push_back(n);
  std::vector<Stage> out;
  out.reserve(n);
  uint32_t barriers = 0;
  for (size_t wi = 0; wi + 1 < bounds.size(); ++wi) {
    const uint32_t b = bounds[wi], m = bounds[wi + 1] - b;
    // A segment's attend reaches the K/V cells [0, p0 + t) of its prep (same layer and segment, same K/V buffer).
    std::vector<Footprint> fp(m);
    for (uint32_t i = 0; i < m; ++i) {
      const Stage &s = S[b + i];
      uint64_t kv_cells = 0;
      if (s.op == kStAttnCore && s.sub)
        for (uint32_t j = 0; j < m; ++j) {
          const Stage &p = S[b + j];
          if (p.op == kStAttnPrep && p.sub == s.sub && p.layer == s.layer && p.aprep.kv_cache == s.aatt.kv_cache && p.aprep.t == s.aatt.t)
            kv_cells = uint64_t(p.aprep.p0) + p.aprep.t;
        }
      fp[i] = stage_footprint(s, kv_cells);
    }
    // Every pair of siblings is checked, not only those the level loop below reaches.
    for (uint32_t i = 0; i < m; ++i)
      for (uint32_t j = 0; j < i; ++j)
        if (siblings(fp[j], fp[i])) check_siblings(fp[j], fp[i]);
    std::vector<uint32_t> level(m, 0);
    for (uint32_t i = 1; i < m; ++i)
      for (uint32_t j = 0; j < i; ++j)
        if (level[j] + 1 > level[i] && footprints_conflict(fp[j], fp[i])) level[i] = level[j] + 1;
    std::vector<uint32_t> order(m);
    for (uint32_t i = 0; i < m; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return level[x] < level[y]; });
    uint64_t dealt = 0;
    for (uint32_t k = 0; k < m; ++k) {
      Stage s = S[b + order[k]];
      const bool last_of_level = k + 1 == m || level[order[k + 1]] != level[order[k]];
      if (dealt > kSchedRotMask) MK_DIE("schedule: rotation %lu overflows", (unsigned long)dealt);
      s.sched = uint32_t(dealt) | (last_of_level ? 0u : kSchedNoBarrier);
      if (enable_tail && tail_op(s) && stage_items(s) <= kTailMaxItems && !out.empty()) {
        const bool first_of_level = k == 0 || level[order[k - 1]] != level[order[k]];  // k == 0: the window's first
        if (first_of_level && last_of_level) {  // alone in its level
          // The barrier before it is the tail's first half: the stage before must end its level (it does, by the
          // sort), carry a grid barrier, and not be a tail itself (its barrier would be its own release).
          if ((out.back().sched & kSchedNoBarrier) || (out.back().sched & kSchedTail))
            MK_DIE("schedule: tail stage %zu: the stage before it isn't the last of its level (sched %x)", out.size(), out.back().sched);
          s.sched |= kSchedTail;
        }
      }
      dealt = last_of_level ? 0 : dealt + stage_items(s);
      barriers += last_of_level;
      out.push_back(s);
    }
  }
  S.swap(out);
  return barriers;
}

}  // namespace mk
