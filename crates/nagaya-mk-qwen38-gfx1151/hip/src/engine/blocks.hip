// SPDX-License-Identifier: MIT
// The model's three blocks as stage lists: mixed [T, E] in, out [T, E] out.
#include "engine.h"

namespace mk {

using namespace model;

static Stage mkst(uint32_t op, uint32_t layer) {
  Stage s;
  memset(&s, 0, sizeof s);
  s.op = op;
  s.layer = layer;
  return s;
}
static Stage qmm_st(uint32_t il, const Tensor &w, uint32_t n, uint32_t k, const float *x, uint32_t t, float *y) {
  if (w.ne[0] != k || (w.ne[1] != n && !(n == 1 && w.ne[1] == 1)))
    MK_DIE("layer %u: qmm weight is %lux%lu, want K=%u N=%u", il, (unsigned long)w.ne[0], (unsigned long)w.ne[1], k, n);
  Stage s = mkst(qmm_op(w.type, fast_qmm_enabled()), il);
  s.qmm = QmmArgs{w.type, w.d, w.row_bytes, n, k, t, x, k, nullptr, y, n, nullptr, bulky_weight(w.row_bytes, n)};
  return s;
}

// GDN layer index among the 36: layers before il minus the attention layers before it.
static uint32_t gdn_index(uint32_t il) { return il - (il + 1) / 4; }

// The state stages run once per segment (engine.h Segment): the stage a pass of that segment alone would run, its
// pointers offset to the segment's rows (r0, the tokens before it in the pass) and its context. With several segments
// Stage::sub is the segment + 1; with one it stays 0 and the list is a one-context pass's.
void Engine::add_gdn(std::vector<Stage> &S, const Segment *seg, uint32_t ns, uint32_t il, uint32_t t, const float *mixed,
                     float *out) {
  const Layer &L = layers_[il];
  const uint32_t g = gdn_index(il);
  float *qkv = scratch("gdn_qkv", size_t(kMaxPassTokens) * kGdnConvDim);
  float *qkvc = scratch("gdn_qkvc", size_t(kMaxPassTokens) * kGdnConvDim);
  float *z = scratch("gdn_z", size_t(kMaxPassTokens) * kGdnValDim);
  float *a = scratch("gdn_a", size_t(kMaxPassTokens) * kGdnVHeads);
  float *b = scratch("gdn_b", size_t(kMaxPassTokens) * kGdnVHeads);
  float *core = scratch("gdn_core", size_t(kMaxPassTokens) * kGdnValDim);
  S.push_back(qmm_st(il, L.qkv, kGdnConvDim, kEmbd, mixed, t, qkv));
  S.push_back(qmm_st(il, L.zgate, kGdnValDim, kEmbd, mixed, t, z));
  S.push_back(qmm_st(il, L.alpha, kGdnVHeads, kEmbd, mixed, t, a));
  S.push_back(qmm_st(il, L.beta, kGdnVHeads, kEmbd, mixed, t, b));
  // Every segment's conv, then every segment's core.
  for (uint32_t i = 0, r0 = 0; i < ns; r0 += seg[i].t, ++i) {
    float *conv_state = seg[i].c->gdn_conv + size_t(g) * kGdnConvDim * (kGdnConv - 1);  // [10240][3], in place
    Stage cv = mkst(kStGdnConv, il);
    cv.sub = ns > 1 ? i + 1 : 0;
    cv.gconv = GdnConvArgs{qkv + size_t(r0) * kGdnConvDim, conv_state, L.conv1d.f32(), qkvc + size_t(r0) * kGdnConvDim,
                           conv_state, seg[i].t};
    S.push_back(cv);
  }
  for (uint32_t i = 0, r0 = 0; i < ns; r0 += seg[i].t, ++i) {
    float *state = seg[i].c->gdn_state + size_t(g) * kGdnVHeads * kGdnKDim * kGdnVDim;  // in place
    Stage co = mkst(kStGdnCore, il);
    co.sub = ns > 1 ? i + 1 : 0;
    co.gcore = GdnCoreArgs{qkvc + size_t(r0) * kGdnConvDim, z + size_t(r0) * kGdnValDim, a + size_t(r0) * kGdnVHeads,
                           b + size_t(r0) * kGdnVHeads, L.ssm_a.f32(), L.dt_bias.f32(), L.ssm_norm.f32(), state,
                           core + size_t(r0) * kGdnValDim, state, seg[i].t};
    S.push_back(co);
  }
  S.push_back(qmm_st(il, L.ssm_out, kEmbd, kGdnValDim, core, t, out));
}

void Engine::add_moe(std::vector<Stage> &S, uint32_t il, uint32_t t, const float *mixed, float *out) {
  const Layer &L = layers_[il];
  const uint32_t A = t * kTopK;
  float *logits = scratch("moe_logits", size_t(kMaxPassTokens) * kExperts);
  int32_t *tidx = reinterpret_cast<int32_t *>(scratch("moe_tidx", size_t(kMaxPassTokens) * kTopK));
  float *tw = scratch("moe_tw", size_t(kMaxPassTokens) * kTopK);
  uint32_t *cnt = reinterpret_cast<uint32_t *>(scratch("moe_cnt", kExperts));
  uint32_t *off = reinterpret_cast<uint32_t *>(scratch("moe_off", kExperts + 1));
  uint32_t *asg = reinterpret_cast<uint32_t *>(scratch("moe_asg", size_t(kMaxPassTokens) * kTopK));
  uint32_t *tok = reinterpret_cast<uint32_t *>(scratch("moe_tok", size_t(kMaxPassTokens) * kTopK));
  uint32_t *igu = reinterpret_cast<uint32_t *>(scratch("moe_igu", kExperts + 1));
  uint32_t *idn = reinterpret_cast<uint32_t *>(scratch("moe_idn", kExperts + 1));
  float *gu = scratch("moe_gu", size_t(kMaxPassTokens) * kTopK * 2 * kFf);
  float *h = scratch("moe_h", size_t(kMaxPassTokens) * kTopK * kFf);
  float *y = scratch("moe_y", size_t(kMaxPassTokens) * kTopK * kEmbd);
  float *sg = scratch("moe_sg", size_t(kMaxPassTokens) * kFfShared);
  float *su = scratch("moe_su", size_t(kMaxPassTokens) * kFfShared);
  float *sh = scratch("moe_sh", size_t(kMaxPassTokens) * kFfShared);
  float *sd = scratch("moe_sd", size_t(kMaxPassTokens) * kEmbd);
  float *sgt = scratch("moe_sgt", kMaxPassTokens);

  S.push_back(qmm_st(il, L.gate_inp, kExperts, kEmbd, mixed, t, logits));
  Stage r = mkst(kStMoeRouter, il);
  r.router = MoeRouterArgs{logits, tidx, tw, t};
  S.push_back(r);
  Stage gr = mkst(kStMoeGroup, il);
  const uint32_t fast = fast_qmm_enabled() ? 1u : 0u;  // group and experts must agree on the tile
  gr.group = MoeGroupArgs{tidx, t, cnt, off, asg, tok, igu, idn, fast};
  S.push_back(gr);
  if (L.gate_exps.ne[0] != kEmbd || L.gate_exps.ne[1] != kFf || L.up_exps.ne[0] != kEmbd || L.up_exps.ne[1] != kFf ||
      L.down_exps.ne[0] != kFf || L.down_exps.ne[1] != kEmbd)
    MK_DIE("layer %u: expert tensors have unexpected shapes", il);
  MoeFfnArgs f{L.gate_exps.type, L.up_exps.type, L.down_exps.type, L.gate_exps.d, L.up_exps.d, L.down_exps.d,
               L.gate_exps.row_bytes, L.up_exps.row_bytes, L.down_exps.row_bytes, mixed, gu, h, y,
               cnt, off, asg, tok, igu, idn, fast};
  Stage fg = mkst(kStMoeGu, il);
  fg.ffn = f;
  S.push_back(fg);
  Stage sw = mkst(kStSwiglu, il);
  sw.swiglu = SwigluArgs{gu, gu + kFf, 2 * kFf, 2 * kFf, h, kFf, A, kFf};
  S.push_back(sw);
  Stage fd = mkst(kStMoeDown, il);
  fd.ffn = f;
  S.push_back(fd);
  // the shared expert
  S.push_back(qmm_st(il, L.gate_shexp, kFfShared, kEmbd, mixed, t, sg));
  S.push_back(qmm_st(il, L.up_shexp, kFfShared, kEmbd, mixed, t, su));
  Stage ss = mkst(kStSwiglu, il);
  ss.swiglu = SwigluArgs{sg, su, kFfShared, kFfShared, sh, kFfShared, t, kFfShared};
  S.push_back(ss);
  S.push_back(qmm_st(il, L.down_shexp, kEmbd, kFfShared, sh, t, sd));
  S.push_back(qmm_st(il, L.gate_inp_shexp, 1, kEmbd, mixed, t, sgt));
  Stage cb = mkst(kStMoeCombine, il);
  cb.mcomb = MoeCombineArgs{y, tidx, tw, sd, sgt, out, t};
  S.push_back(cb);
}

// inv_freq[i] = 1 / theta^(2i / 64), in f32 like Qwen4ExpTextRotaryEmbedding's default init.
const float *inv_freq_table() {
  static float *d = nullptr;
  if (!d) {
    float h[kRopeDims / 2];
    for (uint32_t i = 0; i < kRopeDims / 2; ++i) h[i] = 1.0f / powf(kRopeTheta, float(2 * i) / float(kRopeDims));
    MK_CHECK(hipMalloc(&d, sizeof h));
    MK_CHECK(hipMemcpy(d, h, sizeof h, hipMemcpyHostToDevice));
  }
  return d;
}

void Engine::add_attn(std::vector<Stage> &S, const Segment *seg, uint32_t ns, uint32_t il, uint32_t t, const float *mixed,
                      float *out) {
  const Layer &L = layers_[il];
  const uint32_t ai = il / 4;
  const uint32_t max_sel = kIdxBudget + kIdxRatio - 1, topk = kIdxBudget / kIdxRatio;
  float *q = scratch("att_q", size_t(kMaxPassTokens) * kHeads * 2 * kHeadDim);
  float *k = scratch("att_k", size_t(kMaxPassTokens) * kKvHeads * kHeadDim);
  float *v = scratch("att_v", size_t(kMaxPassTokens) * kKvHeads * kHeadDim);
  float *iq = scratch("att_iq", size_t(kMaxPassTokens) * kIdxHeads * kIdxDim);
  float *ik = scratch("att_ik", size_t(kMaxPassTokens) * kIdxDim);
  float *qo = scratch("att_qo", size_t(kMaxPassTokens) * kHeads * kHeadDim);
  float *go = scratch("att_go", size_t(kMaxPassTokens) * kHeads * kHeadDim);
  float *iqo = scratch("att_iqo", size_t(kMaxPassTokens) * kIdxHeads * kIdxDim);
  // Sized for the largest context (8 MiB each), so contexts of every capacity share them; [T][max_nb] per pass.
  float *scores = scratch("att_scores", size_t(kMaxPassTokens) * (kMaxContext / kIdxRatio));
  int32_t *picked = reinterpret_cast<int32_t *>(scratch("att_picked", size_t(kMaxPassTokens) * topk));
  int32_t *sel = reinterpret_cast<int32_t *>(scratch("att_sel", size_t(kMaxPassTokens) * kAttnMaxSelMasked));
  uint32_t *sel_n = reinterpret_cast<uint32_t *>(scratch("att_seln", kMaxPassTokens));
  float *ao = scratch("att_out", size_t(kMaxPassTokens) * kHeads * kHeadDim);

  S.push_back(qmm_st(il, L.q, kHeads * 2 * kHeadDim, kEmbd, mixed, t, q));
  S.push_back(qmm_st(il, L.k, kKvHeads * kHeadDim, kEmbd, mixed, t, k));
  S.push_back(qmm_st(il, L.v, kKvHeads * kHeadDim, kEmbd, mixed, t, v));
  S.push_back(qmm_st(il, L.idx_q, kIdxHeads * kIdxDim, kEmbd, mixed, t, iq));
  S.push_back(qmm_st(il, L.idx_k, kIdxDim, kEmbd, mixed, t, ik));
  // Per segment: its context's K/V and index keys (a layer's stride is its capacity), p0 = its n, and its rows; the
  // block scores [t][max_nb] at that context's width, packed after the segments before it. Every prep, then every
  // pool (the pooled block keys the pass completes, goal 08 I2: only for a segment that completes a block; its own level,
  // after the prep that wrote their raw keys and before the score that reads them), then every
  // score (only for a segment whose pass reaches past 2051 visible cells: within them every token is dense, emits
  // cells 0..n_vis-1 and reads no score), then every select, then every attend. No stage writes the scores' near-tie
  // copy (block_scores_out).
  std::vector<Stage> prep, pool, score, select, attend;
  size_t soff = 0;
  for (uint32_t i = 0, r0 = 0; i < ns; r0 += seg[i].t, ++i) {
    const Context *c = seg[i].c;
    const uint32_t ti = seg[i].t, p0 = c->n, max_nb = c->max / kIdxRatio, sub = ns > 1 ? i + 1 : 0;
    if (c->max > kMaxContext) MK_DIE("context of capacity %u past %u", c->max, kMaxContext);
    float *kv = c->kv + size_t(ai) * c->max * 2 * kKvHeads * kHeadDim;
    float *ikc = c->idx_keys + size_t(ai) * c->max * kIdxDim;
    // The pooled-key cache of this layer: block b at b * 128. Its watermark can't pass the blocks this context already
    // completed (cells below n), or a read would trust keys nobody pooled.
    float *pkc = c->idx_pooled + size_t(ai) * max_nb * kIdxDim;
    if (c->pooled_n > p0 / kIdxRatio)
      MK_DIE("attention: pooled-key watermark %u past the context's %u complete blocks (%u tokens)", c->pooled_n, p0 / kIdxRatio, p0);
    const size_t r = r0;
    if (soff + size_t(ti) * max_nb > size_t(kMaxPassTokens) * (kMaxContext / kIdxRatio))
      MK_DIE("attention: block scores of the segments past their buffer");
    Stage pr = mkst(kStAttnPrep, il);
    pr.sub = sub;
    pr.aprep = AttnPrepArgs{q + r * kHeads * 2 * kHeadDim, k + r * kKvHeads * kHeadDim, v + r * kKvHeads * kHeadDim,
                            iq + r * kIdxHeads * kIdxDim, ik + r * kIdxDim, L.q_norm.f32(), L.k_norm.f32(),
                            L.idx_q_norm.f32(), inv_freq_table(), ti, p0, qo + r * kHeads * kHeadDim,
                            go + r * kHeads * kHeadDim, iqo + r * kIdxHeads * kIdxDim, kv, ikc};
    prep.push_back(pr);
    const AttnSelectArgs sa{iqo + r * kIdxHeads * kIdxDim, ikc, L.idx_k_norm.f32(), inv_freq_table(), ti, p0, kIdxBudget,
                            kIdxRatio, scores + soff, nullptr, picked + r * topk, sel + r * max_sel, sel_n + r,
                            max_nb, max_sel, topk, pkc, c->pooled_n};
    // The dense rule below equals attn_select_emit's select_count == nb only while the budget is a whole number of blocks.
    static_assert(kIdxBudget % kIdxRatio == 0, "dense rule p0 + t <= budget + ratio - 1 assumes budget % ratio == 0");
    const bool dense = uint64_t(p0) + ti <= uint64_t(kIdxBudget) + kIdxRatio - 1;  // every token has select_count == nb
    Stage se = mkst(kStAttnEmit, il);
    se.sub = sub;
    se.asel = sa;
    if (!dense) {
      Stage sc = se;
      sc.op = kStAttnScore;
      score.push_back(sc);
    }
    select.push_back(se);
    if ((uint64_t(p0) + ti) / kIdxRatio > c->pooled_n) {  // a block completes in this pass
      Stage po = se;
      po.op = kStAttnPool;
      po.asel = sa;
      pool.push_back(po);
      host_stats().pool_blocks += ((uint64_t(p0) + ti) / kIdxRatio - c->pooled_n);
    }
    Stage at = mkst(kStAttnCore, il);
    at.sub = sub;
    at.aatt = AttnAttendArgs{qo + r * kHeads * kHeadDim, go + r * kHeads * kHeadDim, kv, sel + r * max_sel, sel_n + r, ti,
                             max_sel, ao + r * kHeads * kHeadDim};
    attend.push_back(at);
    soff += size_t(ti) * max_nb;
  }
  for (const auto *L3 : {&prep, &pool, &score, &select, &attend}) S.insert(S.end(), L3->begin(), L3->end());
  S.push_back(qmm_st(il, L.o, kEmbd, kHeads * kHeadDim, ao, t, out));
}

// The pool stages that bring a context's pooled-key cache from block b0 up to its n / 4 (one per QSA layer): restore_n's
// one-off launch, the same stage a pass builds (add_attn). t = 0: the stage pools [b0, p0 / ratio).
std::vector<Stage> Engine::pool_stages(const Context *c, uint32_t b0) {
  std::vector<Stage> out;
  const uint32_t max_nb = c->max / kIdxRatio;
  if (b0 > c->n / kIdxRatio) MK_DIE("pool_stages: from block %u past the context's %u complete blocks", b0, c->n / kIdxRatio);
  for (uint32_t il = 3, ai = 0; il < kLayers; il += 4, ++ai) {
    const Layer &L = layers_[il];
    AttnSelectArgs sa{};
    sa.ik_cache = c->idx_keys + size_t(ai) * c->max * kIdxDim;
    sa.idxk_gamma = L.idx_k_norm.f32();
    sa.inv_freq = inv_freq_table();
    sa.t = 0;
    sa.p0 = c->n;
    sa.budget = kIdxBudget;
    sa.ratio = kIdxRatio;
    sa.max_nb = max_nb;
    sa.pooled = c->idx_pooled + size_t(ai) * max_nb * kIdxDim;
    sa.pool_b0 = b0;
    Stage po = mkst(kStAttnPool, il);
    po.asel = sa;
    out.push_back(po);
    host_stats().pool_blocks += c->n / kIdxRatio - b0;
  }
  return out;
}

}  // namespace mk
