//! The schedule of a fused pass: fewer grid barriers, the same bits. It was mk's `stage_sched.h` (mk `555a2a3`).
//!
//! The megakernel puts a grid barrier after each stage. Most stages of a decode step are a few microseconds of work,
//! and a barrier costs about 1.2 to 4.4 us. Two stages need a barrier between them only if one touches what the other
//! writes. Thus the schedule derives the footprint of each stage (the byte ranges that it reads and writes) from its
//! arguments. Between two stages it checks each hazard:
//!
//! - RAW: the second reads what the first writes.
//! - WAR: the second writes what the first reads.
//! - WAW: both write the same bytes.
//!
//! A range is the hull of what the op can touch (rows times stride), never smaller. A stage with no derived footprint
//! is a fence, and it conflicts with each stage. Weights and other constant tables are not in a footprint: no stage
//! writes them.
//!
//! Siblings: a pass of several segments runs each state stage once for each segment, with `Stage::sub` = segment + 1.
//! The state stages are fences. Two of them with the same op and layer and different nonzero `sub` are siblings, and
//! the schedule compares their footprints. Siblings whose footprints overlap are an error: two segments share state,
//! and the refusal of the pass must have caught that. The K/V reach of an attend (cells `[0, p0 + t)`) is not in its
//! arguments. It comes from the prep of its segment in the same window, and a sibling attend with no prep is an error.
//!
//! The schedule works on windows: the stages between two layer ends. Thus the capture points of each layer stay where
//! they are. The level of a stage is one more than the highest level of an earlier stage that it conflicts with, or 0.
//! A stable sort of the window by level keeps each conflicting pair in program order, so each stage reads the same
//! values as before: the same bits. The stages of one level are independent, and they run as one segment with no grid
//! barrier between them. The level count is the longest chain of dependencies: the fewest barriers that any order has.
//! In a segment, the work items of stage i start where those of stage i - 1 stopped (the rotation of `Stage::sched`).
//!
//! This module has no unsafe code. It reads the stages through the accessors of `stage.rs`.

use crate::stage::*;

pub(crate) mod call;

/// One access of a stage: the bytes `[lo, hi)`, read or written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Access {
    pub lo: u64,
    pub hi: u64,
    pub write: bool,
}

/// The footprint of a stage: its accesses, or a fence.
#[derive(Clone, Debug, Default)]
pub(crate) struct Footprint {
    pub fence: bool,
    pub acc: Vec<Access>,
    /// For a state stage of a segment (`sub != 0`): its op, layer and sub, for the sibling rule. Its accesses are
    /// derived, although it is a fence.
    pub op: u32,
    pub layer: u32,
    pub sub: u32,
    /// An attend whose K/V reach is unknown: no prep of its segment is in the window.
    pub unbounded: bool,
}

/// A stage or a list of stages that the schedule refuses. Each one is a defect of the pass build.
#[derive(Debug)]
pub(crate) struct ScheduleError(String);

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ScheduleError {}

fn refuse<T>(message: String) -> Result<T, ScheduleError> {
    Err(ScheduleError(message))
}

/// The address of a pointer of a stage. The schedule compares addresses, and never reads through them.
fn addr<T>(pointer: *const T) -> u64 {
    pointer as usize as u64
}

/// The bytes of `n` floats (or of `n` 32-bit integers).
fn fl(n: u64) -> u64 {
    4 * n
}

/// The hull of `rows` rows of `cols` floats at a stride of `ld` floats, in bytes.
fn rows_bytes(rows: u32, ld: u64, cols: u64) -> u64 {
    if rows == 0 {
        0
    } else {
        4 * ((u64::from(rows) - 1) * ld + cols)
    }
}

impl Footprint {
    /// A read of `bytes` at `at`. A null pointer or no bytes adds nothing.
    fn rd(&mut self, at: u64, bytes: u64) {
        if at != 0 && bytes != 0 {
            self.acc.push(Access {
                lo: at,
                hi: at + bytes,
                write: false,
            });
        }
    }

    /// A write of `bytes` at `at`. No bytes adds nothing. A null output is an error.
    fn wr(&mut self, at: u64, bytes: u64) -> Result<(), ScheduleError> {
        if at == 0 {
            return refuse("stage footprint: a null output".to_owned());
        }
        if bytes != 0 {
            self.acc.push(Access {
                lo: at,
                hi: at + bytes,
                write: true,
            });
        }
        Ok(())
    }
}

/// The state ops whose stages of different segments can be siblings.
fn sibling_op(op: u32) -> bool {
    matches!(
        op,
        ST_GDN_CORE | ST_ATTN_PREP | ST_ATTN_SELECT | ST_ATTN_POOL | ST_ATTN_SCORE | ST_ATTN_EMIT | ST_ATTN_CORE
    )
}

/// The footprint of a stage. `kv_cells` is the K/V reach of a segment's attend (its prep's `p0 + t`), or 0.
pub(crate) fn stage_footprint(s: &Stage, kv_cells: u64) -> Result<Footprint, ScheduleError> {
    let mut f = Footprint::default();
    if sibling_op(s.op) && s.sub != 0 {
        f.fence = true;
        f.op = s.op;
        f.layer = s.layer;
        f.sub = s.sub;
        state_footprint(s, kv_cells, &mut f)?;
        return Ok(f);
    }
    let (experts, top_k, embd) = (u64::from(EXPERTS), u64::from(TOP_K), u64::from(EMBD));
    match s.op {
        ST_QMM | ST_QMM_H | ST_QMM_KSUM => qmm_footprint(s, &mut f)?,
        ST_RMS => {
            let a = s.rms();
            let cols = u64::from(a.n_groups) * u64::from(a.d);
            f.rd(addr(a.x), rows_bytes(a.t, a.ldx.into(), cols));
            f.wr(addr(a.y), rows_bytes(a.t, a.ldy.into(), cols))?;
        }
        ST_SILU_SCALE => {
            let a = s.silu();
            f.rd(addr(a.x), fl(a.n.into()));
            f.wr(addr(a.y), fl(a.n.into()))?;
        }
        ST_HC_MIX => {
            let a = s.mix();
            let (t, e) = (u64::from(a.t), u64::from(a.e));
            f.rd(addr(a.xn), fl(t * 4 * e));
            f.rd(addr(a.up), fl(t * 4 * e));
            f.wr(addr(a.mixed), fl(t * e))?;
            if !a.z.is_null() {
                // The op writes inject only with z (small.h op_hc_mix).
                f.rd(addr(a.z), fl(t * 4));
                f.wr(addr(a.inject), fl(t * 4))?;
            }
        }
        ST_HC_COMBINE => {
            let a = s.comb();
            let (t, e) = (u64::from(a.t), u64::from(a.e));
            f.rd(addr(a.h), fl(t * 4 * e));
            f.wr(addr(a.h), fl(t * 4 * e))?;
            f.rd(addr(a.out), fl(t * e));
            f.rd(addr(a.inject), fl(t * 4));
        }
        ST_HC_COMBINE_RMS => {
            // The accesses of the combine, and the write of xn by the rms. Gamma is a constant table.
            let HcCombineRmsArgs { c, r } = *s.combrms();
            if r.x.cast_mut() != c.h
                || r.t != c.t
                || r.n_groups != 4
                || r.d != c.e
                || r.ldx != 4 * c.e
                || r.gamma_per_group == 0
            {
                return refuse(
                    "stage footprint: a fused combine + rms whose rms doesn't read exactly the combine's h".to_owned(),
                );
            }
            let (t, e) = (u64::from(c.t), u64::from(c.e));
            f.rd(addr(c.h), fl(t * 4 * e));
            f.wr(addr(c.h), fl(t * 4 * e))?;
            f.rd(addr(c.out), fl(t * e));
            f.rd(addr(c.inject), fl(t * 4));
            f.wr(addr(r.y), rows_bytes(r.t, r.ldy.into(), 4 * e))?;
        }
        ST_EMBED => {
            let a = s.embed();
            f.rd(addr(a.ids), fl(a.t.into()));
            f.wr(addr(a.out), fl(u64::from(a.t) * u64::from(a.k)))?;
        }
        ST_REPEAT4 => {
            let a = s.rep();
            let (t, d) = (u64::from(a.t), u64::from(a.d));
            f.rd(addr(a.e), fl(t * d));
            f.wr(addr(a.h), fl(t * 4 * d))?;
        }
        ST_PLE_GATE => {
            let a = s.pgate();
            let (t, e) = (u64::from(a.t), u64::from(a.e));
            f.rd(addr(a.kn), fl(t * 4 * e));
            f.rd(addr(a.qn), fl(t * 4 * e));
            f.rd(addr(a.pv), fl(t * e));
            f.wr(addr(a.gv), fl(t * 4 * e))?;
        }
        ST_PLE_CONV => {
            let a = s.pconv();
            let (t, c) = (u64::from(a.t), u64::from(a.c));
            f.rd(addr(a.x), fl(t * c));
            f.rd(addr(a.state), fl(9 * c));
            f.wr(addr(a.state), fl(9 * c))?;
            f.wr(addr(a.y), fl(t * c))?;
        }
        ST_ADD => {
            let a = s.add();
            let n = u64::from(a.n);
            f.rd(addr(a.h), fl(n));
            f.wr(addr(a.h), fl(n))?;
            f.rd(addr(a.a), fl(n));
            f.rd(addr(a.b), fl(n));
        }
        ST_GDN_CONV => {
            let a = s.gconv();
            let (t, conv_dim) = (u64::from(a.t), u64::from(GDN_CONV_DIM));
            let state = conv_dim * (u64::from(GDN_CONV) - 1);
            f.rd(addr(a.x), fl(t * conv_dim));
            f.rd(addr(a.state_in), fl(state));
            f.wr(addr(a.y), fl(t * conv_dim))?;
            f.wr(addr(a.state_out), fl(state))?;
        }
        ST_MOE_ROUTER => {
            let a = s.router();
            let t = u64::from(a.t);
            f.rd(addr(a.logits), fl(t * experts));
            f.wr(addr(a.topk_idx), fl(t * top_k))?;
            f.wr(addr(a.topk_w), fl(t * top_k))?;
        }
        ST_MOE_GROUP => {
            let a = s.group();
            let t = u64::from(a.t);
            f.rd(addr(a.topk_idx), fl(t * top_k));
            f.wr(addr(a.count), fl(experts))?;
            f.wr(addr(a.offset), fl(experts + 1))?;
            f.wr(addr(a.assign), fl(t * top_k))?;
            f.wr(addr(a.token), fl(t * top_k))?;
            f.wr(addr(a.item_gu), fl(experts + 1))?;
            f.wr(addr(a.item_dn), fl(experts + 1))?;
        }
        ST_MOE_GU | ST_MOE_DOWN => {
            // MoeFfnArgs holds no T: the group's arrays and the activations at their largest (MAX_PASS_TOKENS).
            let a = s.ffn();
            let tokens = u64::from(MAX_PASS_TOKENS);
            let most = tokens * top_k;
            f.rd(addr(a.count), fl(experts));
            f.rd(addr(a.offset), fl(experts + 1));
            f.rd(addr(a.assign), fl(most));
            f.rd(addr(a.token), fl(most));
            f.rd(addr(a.item_gu), fl(experts + 1));
            f.rd(addr(a.item_dn), fl(experts + 1));
            if s.op == ST_MOE_GU {
                f.rd(addr(a.x), fl(tokens * embd));
                f.wr(addr(a.gu), fl(most * 2 * u64::from(FF)))?;
            } else {
                f.rd(addr(a.h), fl(most * u64::from(FF)));
                f.wr(addr(a.y), fl(most * embd))?;
            }
        }
        ST_SWIGLU => {
            let a = s.swiglu();
            let cols = u64::from(a.cols);
            f.rd(addr(a.g), rows_bytes(a.rows, a.ldg.into(), cols));
            f.rd(addr(a.u), rows_bytes(a.rows, a.ldu.into(), cols));
            f.wr(addr(a.out), rows_bytes(a.rows, a.ldo.into(), cols))?;
        }
        ST_MOE_COMBINE => {
            let a = s.mcomb();
            let t = u64::from(a.t);
            f.rd(addr(a.y), fl(t * top_k * embd));
            f.rd(addr(a.topk_idx), fl(t * top_k));
            f.rd(addr(a.topk_w), fl(t * top_k));
            f.rd(addr(a.shared), fl(t * embd));
            f.rd(addr(a.shared_gate), fl(t));
            f.wr(addr(a.out), fl(t * embd))?;
        }
        // The state ops of a one-segment pass, the masked select, and each new op: a fence until it is derived.
        _ => f.fence = true,
    }
    Ok(f)
}

/// The accesses of a state stage of a segment: a GDN core or an attention stage.
fn state_footprint(s: &Stage, kv_cells: u64, f: &mut Footprint) -> Result<(), ScheduleError> {
    let (heads, kv_heads, head_dim) = (u64::from(HEADS), u64::from(KV_HEADS), u64::from(HEAD_DIM));
    let (idx_heads, idx_dim) = (u64::from(IDX_HEADS), u64::from(IDX_DIM));
    let kv_cell = 2 * kv_heads * head_dim;
    let q_row = heads * head_dim;
    match s.op {
        ST_GDN_CORE => {
            let a = s.gcore();
            let t = u64::from(a.t);
            let (conv_dim, val_dim, v_heads) =
                (u64::from(GDN_CONV_DIM), u64::from(GDN_VAL_DIM), u64::from(GDN_V_HEADS));
            let state = v_heads * u64::from(GDN_K_DIM) * u64::from(GDN_V_DIM);
            f.rd(addr(a.qkv), fl(t * conv_dim));
            f.rd(addr(a.z), fl(t * val_dim));
            f.rd(addr(a.a), fl(t * v_heads));
            f.rd(addr(a.b), fl(t * v_heads));
            f.rd(addr(a.state_in), fl(state));
            f.wr(addr(a.y), fl(t * val_dim))?;
            f.wr(addr(a.state_out), fl(state))?;
        }
        ST_ATTN_PREP => {
            let a = s.aprep();
            let t = u64::from(a.t);
            f.rd(addr(a.q_raw), fl(t * heads * 2 * head_dim));
            f.rd(addr(a.k_raw), fl(t * kv_heads * head_dim));
            f.rd(addr(a.v_raw), fl(t * kv_heads * head_dim));
            f.rd(addr(a.iq_raw), fl(t * idx_heads * idx_dim));
            f.rd(addr(a.ik_raw), fl(t * idx_dim));
            f.wr(addr(a.q_out), fl(t * q_row))?;
            f.wr(addr(a.gate_out), fl(t * q_row))?;
            f.wr(addr(a.iq_out), fl(t * idx_heads * idx_dim))?;
            let p0 = u64::from(a.p0);
            f.wr(addr(a.kv_cache) + fl(p0 * kv_cell), fl(t * kv_cell))?;
            f.wr(addr(a.ik_cache) + fl(p0 * idx_dim), fl(t * idx_dim))?;
        }
        ST_ATTN_SELECT => {
            let a = s.asel();
            let t = u64::from(a.t);
            f.rd(addr(a.iq_out), fl(t * idx_heads * idx_dim));
            f.rd(addr(a.ik_cache), fl((u64::from(a.p0) + t) * idx_dim));
            f.wr(addr(a.block_scores), fl(t * u64::from(a.max_nb)))?;
            if !a.block_scores_out.is_null() {
                f.wr(addr(a.block_scores_out), fl(t * u64::from(a.max_nb)))?;
            }
            f.wr(addr(a.picked), fl(t * u64::from(a.block_topk_max)))?;
            f.wr(addr(a.sel), fl(t * u64::from(a.max_sel)))?;
            f.wr(addr(a.sel_count), fl(t))?;
        }
        ST_ATTN_POOL => {
            // Blocks [pool_b0, (p0 + t) / ratio) of the cache, from the raw keys of their cells.
            let a = s.asel();
            let ratio = u64::from(a.ratio);
            let nb1 = (u64::from(a.p0) + u64::from(a.t)) / ratio;
            if a.pooled.is_null() {
                return refuse("stage footprint: a pool stage without a cache".to_owned());
            }
            let b0 = u64::from(a.pool_b0);
            if b0 > nb1 {
                return refuse(format!(
                    "stage footprint: a pool stage from block {b0} past its pass's {nb1}"
                ));
            }
            f.rd(
                addr(a.ik_cache) + fl(b0 * ratio * idx_dim),
                fl((nb1 - b0) * ratio * idx_dim),
            );
            f.wr(addr(a.pooled) + fl(b0 * idx_dim), fl((nb1 - b0) * idx_dim))?;
        }
        ST_ATTN_SCORE => {
            // The scores alone. block_scores_out belongs to the composed op, and the engine gives none here.
            let a = s.asel();
            let t = u64::from(a.t);
            if !a.block_scores_out.is_null() {
                return refuse("stage footprint: a score stage with block_scores_out".to_owned());
            }
            f.rd(addr(a.iq_out), fl(t * idx_heads * idx_dim));
            let cells = u64::from(a.p0) + t;
            if a.pooled.is_null() {
                f.rd(addr(a.ik_cache), fl(cells * idx_dim));
            } else {
                // Each block's key comes from the cache.
                f.rd(addr(a.pooled), fl(cells / u64::from(a.ratio) * idx_dim));
            }
            f.wr(addr(a.block_scores), fl(t * u64::from(a.max_nb)))?;
        }
        ST_ATTN_EMIT => {
            // The selection alone: it reads the scores. A dense pass reads scores that no stage writes.
            let a = s.asel();
            let t = u64::from(a.t);
            if !a.block_scores_out.is_null() {
                return refuse("stage footprint: an emit stage with block_scores_out".to_owned());
            }
            f.rd(addr(a.block_scores), fl(t * u64::from(a.max_nb)));
            f.wr(addr(a.picked), fl(t * u64::from(a.block_topk_max)))?;
            f.wr(addr(a.sel), fl(t * u64::from(a.max_sel)))?;
            f.wr(addr(a.sel_count), fl(t))?;
        }
        ST_ATTN_CORE => {
            let a = s.aatt();
            let t = u64::from(a.t);
            f.rd(addr(a.q_out), fl(t * q_row));
            f.rd(addr(a.gate_out), fl(t * q_row));
            if kv_cells != 0 {
                f.rd(addr(a.kv_cache), fl(kv_cells * kv_cell));
            } else {
                f.unbounded = true;
            }
            f.rd(addr(a.sel), fl(t * u64::from(a.max_sel)));
            f.rd(addr(a.sel_count), fl(t));
            f.wr(addr(a.attn_out), fl(t * q_row))?;
        }
        op => unreachable!("sibling_op does not name op {op}"),
    }
    Ok(())
}

/// The footprint of an exact qmm, an f16 qmm, or a split-K sum.
fn qmm_footprint(s: &Stage, f: &mut Footprint) -> Result<(), ScheduleError> {
    let a = s.qmm();
    if !a.x_row.is_null() || !a.y_row.is_null() {
        // The row maps of the MoE are not derived here.
        f.fence = true;
        return Ok(());
    }
    let ks = if a.ksplit > 1 { a.ksplit } else { 1 };
    let (t, n, k) = (u64::from(a.t), u64::from(a.n), u64::from(a.k));
    if s.op == ST_QMM && (ks > 1 || a.xin != XIN_PLAIN) {
        return refuse("stage footprint: an exact qmm with split-K or fused staging".to_owned());
    }
    if s.op == ST_QMM_KSUM {
        if ks < 2 || a.part.is_null() {
            return refuse("stage footprint: a split-K sum without partials".to_owned());
        }
        f.rd(addr(a.part), fl(u64::from(ks) * t * n));
        return f.wr(addr(a.y), rows_bytes(a.t, a.ldy.into(), n));
    }
    if a.xin == XIN_PLAIN {
        f.rd(addr(a.x), rows_bytes(a.t, a.ldx.into(), k));
    } else {
        if a.xks < 2 || a.xp.is_null() {
            return refuse("stage footprint: fused staging without partials".to_owned());
        }
        let xks = u64::from(a.xks);
        f.rd(addr(a.xp), fl(xks * t * k));
        if a.xin == XIN_SWIGLU_SUM {
            if a.xp2.is_null() {
                return refuse("stage footprint: swiglu staging without up partials".to_owned());
            }
            f.rd(addr(a.xp2), fl(xks * t * k));
        } else if a.xin != XIN_SILU_SUM {
            return refuse(format!("stage footprint: unknown xin {}", a.xin));
        }
    }
    if a.yout != YOUT_PLAIN {
        // hc up + hc_mix (docs/narrow.md N1): it reads xn (and z), writes mixed (and inject), and never writes y.
        if s.op != ST_QMM_H {
            return refuse(format!("stage footprint: yout on op {} (f16 qmm only)", s.op));
        }
        if a.yout != YOUT_HC_MIX {
            return refuse(format!("stage footprint: unknown yout {}", a.yout));
        }
        if ks > 1 {
            return refuse("stage footprint: yout with split-K".to_owned());
        }
        let e = u64::from(a.mix_e);
        f.rd(addr(a.mix_xn), fl(t * 4 * e));
        f.rd(addr(a.mix_z), fl(t * 4));
        f.wr(addr(a.mix_out), fl(t * e))?;
        if !a.mix_z.is_null() {
            f.wr(addr(a.mix_inject), fl(t * 4))?;
        }
        return Ok(());
    }
    if ks > 1 {
        if a.part.is_null() {
            return refuse("stage footprint: split-K without partials".to_owned());
        }
        f.wr(addr(a.part), fl(u64::from(ks) * t * n))
    } else {
        f.wr(addr(a.y), rows_bytes(a.t, a.ldy.into(), n))
    }
}

fn accesses_overlap(a: &Footprint, b: &Footprint) -> bool {
    a.acc
        .iter()
        .any(|p| b.acc.iter().any(|q| (p.write || q.write) && p.lo < q.hi && q.lo < p.hi))
}

fn siblings(a: &Footprint, b: &Footprint) -> bool {
    a.sub != 0 && b.sub != 0 && a.sub != b.sub && a.op == b.op && a.layer == b.layer
}

/// Siblings are independent unless they share bytes. Then two segments share state, and that is an error.
fn check_siblings(a: &Footprint, b: &Footprint) -> Result<(), ScheduleError> {
    if a.unbounded || b.unbounded {
        let sub = if a.unbounded { a.sub } else { b.sub };
        return refuse(format!(
            "schedule: an attend of segment {} (layer {}) without its segment's prep in the window: its K/V reach is \
             unknown",
            sub - 1,
            a.layer
        ));
    }
    if accesses_overlap(a, b) {
        return refuse(format!(
            "schedule: segments {} and {} share state: their op {} stages in layer {} touch the same bytes",
            a.sub - 1,
            b.sub - 1,
            a.op,
            a.layer
        ));
    }
    Ok(())
}

/// True when the second stage must wait for the first: a hazard, or a fence.
pub(crate) fn footprints_conflict(a: &Footprint, b: &Footprint) -> Result<bool, ScheduleError> {
    if siblings(a, b) {
        check_siblings(a, b)?;
        return Ok(false);
    }
    Ok(a.fence || b.fence || accesses_overlap(a, b))
}

/// The work items of a stage as its op deals them, for the rotation only. A wrong count costs balance, never bits.
pub(crate) fn stage_items(s: &Stage) -> u32 {
    let cdiv = |a: u64, b: u32| a.div_ceil(u64::from(b)) as u32;
    match s.op {
        ST_QMM_H => {
            let q = s.qmm();
            let ks = if q.ksplit > 1 { q.ksplit } else { 1 };
            cdiv(q.n.into(), QMMH_ROWS) * cdiv(q.t.into(), QMMH_TOKS) * ks
        }
        ST_QMM => cdiv(s.qmm().n.into(), QMM_ROWS) * cdiv(s.qmm().t.into(), QMM_TOKS),
        ST_QMM_KSUM => cdiv(u64::from(s.qmm().t) * u64::from(s.qmm().n), BLOCK_THREADS),
        ST_RMS => s.rms().t * s.rms().n_groups,
        ST_SILU_SCALE => cdiv(s.silu().n.into(), BLOCK_THREADS),
        ST_HC_MIX => cdiv(u64::from(s.mix().t) * u64::from(s.mix().e), BLOCK_THREADS),
        ST_HC_COMBINE => cdiv(u64::from(s.comb().t) * 4 * u64::from(s.comb().e), BLOCK_THREADS),
        ST_HC_COMBINE_RMS => s.combrms().c.t * 4,
        ST_SWIGLU => cdiv(u64::from(s.swiglu().rows) * u64::from(s.swiglu().cols), BLOCK_THREADS),
        ST_MOE_ROUTER => s.router().t,
        ST_MOE_COMBINE => cdiv(u64::from(s.mcomb().t) * u64::from(EMBD), BLOCK_THREADS),
        ST_GDN_CONV => cdiv(GDN_CONV_DIM.into(), BLOCK_THREADS),
        ST_PLE_CONV => cdiv(s.pconv().c.into(), BLOCK_THREADS),
        ST_GDN_CORE => GDN_V_HEADS,
        ST_ATTN_PREP => cdiv(u64::from(s.aprep().t) * u64::from(ATTN_PREP_UNITS), WAVES_PER_BLOCK),
        ST_ATTN_SELECT | ST_ATTN_EMIT => s.asel().t,
        ST_ATTN_POOL => {
            // attn_pool_blocks (ops/attn.h). The footprint refuses a first block past the pass's blocks.
            let a = s.asel();
            let blocks = ((a.p0 + a.t) / a.ratio).wrapping_sub(a.pool_b0);
            cdiv(blocks.into(), WAVES_PER_BLOCK)
        }
        ST_ATTN_SCORE => {
            // attn_score_ranges (ops/attn.h): the ranges of SCORE_RANGE blocks for each token.
            let a = s.asel();
            a.t * ((a.p0 + a.t) / a.ratio).div_ceil(SCORE_RANGE)
        }
        ST_ATTN_CORE => {
            let t = s.aatt().t;
            if t == 1 {
                HEADS
            } else {
                t * (HEADS / ATTN_HEADS_PER_WAVE)
            }
        }
        _ => 1,
    }
}

/// The most work items that one runner does alone as a tail stage, one after the other.
const TAIL_MAX_ITEMS: u32 = 4;

/// The ops that can be a tail (docs/narrow.md N3). The fused combine + rms has at most `TAIL_MAX_ITEMS` items.
/// `op_moe_group` already runs in one workgroup (`if (wg != 0) return`, no use of `nwg`), so `op(a, 0, 1)` on the
/// runner is the same work and bits. It is a tail at T = 1 only (N3b): its loops grow with T.
fn tail_op(s: &Stage) -> bool {
    s.op == ST_HC_COMBINE_RMS || (s.op == ST_MOE_GROUP && s.group().t == 1)
}

/// The schedule of a pass: the order of its stages and the `sched` word of each one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// `order[k]` is the index in build order of the k-th stage to run.
    pub order: Vec<u32>,
    /// `sched[k]` is the `Stage::sched` of the k-th stage to run.
    pub sched: Vec<u32>,
    /// The grid barriers of the pass. A tail stage's two halves count as the two barriers that they replace.
    pub barriers: u32,
}

/// The plan with no schedule (`QWEN38_ELIDE=0`): each stage in build order, with a grid barrier after it.
pub(crate) fn plain(n: u32) -> Plan {
    Plan {
        order: (0..n).collect(),
        sched: vec![0; n as usize],
        barriers: n,
    }
}

/// The schedule of `stages`. `window_starts` holds the ascending indices where a window starts; 0 is implied.
///
/// With `enable_tail` (docs/narrow.md N3), a stage gets `SCHED_TAIL` when it is alone in its level, its op is on the
/// tail list, it has at most `TAIL_MAX_ITEMS` items, and it is not the first stage of the pass. The stage before it
/// must then be the last of its level, with a grid barrier, and not a tail. Anything else is an error.
pub(crate) fn plan(stages: &[Stage], window_starts: &[u32], enable_tail: bool) -> Result<Plan, ScheduleError> {
    let n = u32::try_from(stages.len()).map_err(|_| ScheduleError(format!("schedule: {} stages", stages.len())))?;
    let mut bounds = vec![0u32];
    for &w in window_starts {
        let last = *bounds.last().expect("bounds starts with 0");
        if w < last || w > n {
            return refuse(format!("schedule: window start {w} out of order or past {n} stages"));
        }
        if w > last {
            bounds.push(w);
        }
    }
    if *bounds.last().expect("bounds starts with 0") != n {
        bounds.push(n);
    }
    let mut out = Plan {
        order: Vec::with_capacity(stages.len()),
        sched: Vec::with_capacity(stages.len()),
        barriers: 0,
    };
    for window in bounds.windows(2) {
        let (start, end) = (window[0] as usize, window[1] as usize);
        let w = &stages[start..end];
        let mut fp = Vec::with_capacity(w.len());
        for s in w {
            fp.push(stage_footprint(s, kv_reach(s, w))?);
        }
        // Check each pair of siblings, not only the pairs that the level loop below reaches.
        for i in 0..w.len() {
            for j in 0..i {
                if siblings(&fp[j], &fp[i]) {
                    check_siblings(&fp[j], &fp[i])?;
                }
            }
        }
        let mut level = vec![0u32; w.len()];
        for i in 1..w.len() {
            for j in 0..i {
                if level[j] + 1 > level[i] && footprints_conflict(&fp[j], &fp[i])? {
                    level[i] = level[j] + 1;
                }
            }
        }
        // A stable sort keeps program order in a level.
        let mut order: Vec<usize> = (0..w.len()).collect();
        order.sort_by_key(|&i| level[i]);
        let mut dealt: u64 = 0;
        for k in 0..order.len() {
            let s = &w[order[k]];
            let last_of_level = k + 1 == order.len() || level[order[k + 1]] != level[order[k]];
            if dealt > u64::from(SCHED_ROT_MASK) {
                return refuse(format!("schedule: rotation {dealt} overflows"));
            }
            let mut sched = dealt as u32 | if last_of_level { 0 } else { SCHED_NO_BARRIER };
            if enable_tail && tail_op(s) && stage_items(s) <= TAIL_MAX_ITEMS && !out.sched.is_empty() {
                // k == 0: the first stage of the window.
                let first_of_level = k == 0 || level[order[k - 1]] != level[order[k]];
                if first_of_level && last_of_level {
                    // The barrier before it is the first half of the tail. The stage before must end its level (the
                    // sort makes it so), have a grid barrier, and not be a tail: its barrier would be its own release.
                    let before = *out.sched.last().expect("not empty");
                    if before & (SCHED_NO_BARRIER | SCHED_TAIL) != 0 {
                        return refuse(format!(
                            "schedule: tail stage {}: the stage before it isn't the last of its level (sched {before:x})",
                            out.sched.len()
                        ));
                    }
                    sched |= SCHED_TAIL;
                }
            }
            dealt = if last_of_level {
                0
            } else {
                dealt + u64::from(stage_items(s))
            };
            out.barriers += u32::from(last_of_level);
            out.order.push((start + order[k]) as u32);
            out.sched.push(sched);
        }
    }
    Ok(out)
}

/// The K/V reach of a segment's attend: the cells `[0, p0 + t)` of its prep in the window `w` (the same layer, segment,
/// K/V buffer and token count). The last such prep counts. 0 for any other stage, or when no prep matches.
fn kv_reach(s: &Stage, w: &[Stage]) -> u64 {
    if s.op != ST_ATTN_CORE || s.sub == 0 {
        return 0;
    }
    let mut cells = 0;
    for p in w {
        if p.op == ST_ATTN_PREP
            && p.sub == s.sub
            && p.layer == s.layer
            && p.aprep().kv_cache.cast_const() == s.aatt().kv_cache
            && p.aprep().t == s.aatt().t
        {
            cells = u64::from(p.aprep().p0) + u64::from(p.aprep().t);
        }
    }
    cells
}

/// Reorder `stages` by their plan, set the `sched` word of each one, and give the grid barriers of the pass. It is the
/// form of mk's `schedule_stages`, for the tests. The engine applies the plan itself (`engine.hip` `Engine::build`).
#[cfg(test)]
pub(crate) fn schedule_stages(
    stages: &mut Vec<Stage>,
    window_starts: &[u32],
    enable_tail: bool,
) -> Result<u32, ScheduleError> {
    let plan = plan(stages, window_starts, enable_tail)?;
    let reordered = plan
        .order
        .iter()
        .zip(&plan.sched)
        .map(|(&index, &sched)| Stage {
            sched,
            ..stages[index as usize]
        })
        .collect();
    *stages = reordered;
    Ok(plan.barriers)
}

#[cfg(test)]
mod tests;
