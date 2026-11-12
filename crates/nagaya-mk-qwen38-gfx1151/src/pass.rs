//! The stage list of a pass, built in Rust (milestone 15 step 6).
//!
//! mk's `engine.hip` `build`, `build_stages`, `split_k`, `fuse_prologues`, `fuse_hc` and `fuse_hc_mix`, and
//! `blocks.hip` `add_gdn`, `add_attn`, `add_moe` and `pool_stages` built the stages in C++ (mk 555a2a3). This module
//! builds the same stages, field for field. The schedule (`schedule.rs`) then orders them. The stages name device
//! buffers: the weights in the arena, the context buffers, and the named scratch buffers of the pass.

// The port keeps mk's argument lists and its comparisons, so that each line reads as the C++ line it replaces.
#![allow(clippy::too_many_arguments, clippy::int_plus_one, clippy::ptr_eq)]
// A build without the feature hip runs no pass, so the rope table has no reader there.
#![cfg_attr(not(feature = "hip"), allow(dead_code))]

use std::collections::HashMap;

use crate::context::Qwen38Context;
use crate::schedule::{self, call::pass_plan};
use crate::stage::*;

/// `model.h`: the shapes of Qwen3.8 that the stages name.
pub(crate) const LAYERS: u32 = 48;
pub(crate) const EMBD: u32 = 2560;
const HC: u32 = 4;
const HC_DIM: u32 = HC * EMBD;
const HC_RANK: u32 = 320;
pub(crate) const VOCAB: u32 = 248_320;
const RMS_EPS: f32 = 1e-6;
const ROPE_DIMS: u32 = 64;
const ROPE_THETA: f32 = 1e7;
const TOP_K: u32 = 10;
const FF: u32 = 640;
const FF_SHARED: u32 = 640;
const PLE_LAYER: u32 = 1;
pub(crate) const MAX_PASS_TOKENS: u32 = 256;
pub(crate) const MAX_CONTEXT: u32 = 32_768;
pub(crate) const MAX_SEGMENTS: u32 = 16;
/// `attn_mask.h` `kAttnMaxSelMasked`: the stride of the selection buffer, which keeps its masked size.
const ATTN_MAX_SEL_MASKED: u32 = IDX_BUDGET + IDX_RATIO - 1 + 2 * 4 * (IDX_RATIO - 1);
/// `qmmh.h` `kMaxKsplit`, the most K slices of a split.
const MAX_KSPLIT: u32 = 16;
/// `qmmh.h` `kQmmhChunkUnits`: 64 values per K chunk.
const QMMH_CHUNK_UNITS: u32 = 2;
/// `engine.hip` `kFuseMaxT`: the fused staging and the fused hc_post + norm run for at most 16 tokens.
const FUSE_MAX_T: u32 = 16;
/// The floats of each split-K partial buffer: the largest split, 16 x 320 = 8 x 640 = 5120, for each token.
const PART_FLOATS: usize = 16 * MAX_PASS_TOKENS as usize * 640;

/// The quant type ids of `quant.h` that the build reads.
const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q6_K: u32 = 14;
const IQ4_NL: u32 = 20;
const IQ3_S: u32 = 21;
const IQ4_XS: u32 = 23;
const IQ3_S_TILES: u32 = 121;
const Q8_0_TILES: u32 = 108;
const Q6_K_TILES: u32 = 114;
const IQ4_NL_TILES: u32 = 120;

/// One tensor of the arena in its final layout.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tensor {
    pub kind: u32,
    pub ne: [u64; 3],
    pub row_bytes: u64,
    pub d: *const u8,
}

impl Tensor {
    /// The tensor's values as f32. A tensor of another type stops the program.
    fn f32(&self) -> *const f32 {
        assert_eq!(self.kind, F32, "tensor is type {}, not F32", self.kind);
        self.d as *const f32
    }
}

/// The tensors of one layer (`engine.h` `Layer`). A tensor that the layer kind does not have stays absent.
#[derive(Default)]
struct Layer {
    t: HashMap<&'static str, Tensor>,
}

impl Layer {
    fn get(&self, name: &str) -> Tensor {
        *self
            .t
            .get(name)
            .unwrap_or_else(|| panic!("the layer has no tensor {name}"))
    }
}

/// Every layer's tensors by role, from the arena's names (`engine.hip` `wire`).
pub(crate) struct Weights {
    layers: Vec<Layer>,
    tok_embd: Tensor,
    out_hc_norm: Tensor,
    out_hc_down: Tensor,
    out_hc_up: Tensor,
    output: Tensor,
}

fn is_attn_layer(il: u32) -> bool {
    il % 4 == 3
}

impl Weights {
    /// Wire the roles from `tensors`, the arena's tensors by name. A missing tensor is an error that names it.
    pub(crate) fn new(tensors: &HashMap<String, Tensor>) -> Result<Weights, String> {
        let get = |name: &str| tensors.get(name).copied().ok_or_else(|| format!("no tensor {name}"));
        let mut layers = Vec::with_capacity(LAYERS as usize);
        for il in 0..LAYERS {
            let b = format!("blk.{il}.");
            let mut layer = Layer::default();
            let mut put = |role: &'static str, name: String| -> Result<(), String> {
                layer.t.insert(role, get(&name)?);
                Ok(())
            };
            for (s, side) in ["attn", "ffn"].iter().enumerate() {
                let roles: [[&'static str; 4]; 2] = [
                    ["hc_norm0", "hc_down0", "hc_up0", "hc_inject0"],
                    ["hc_norm1", "hc_down1", "hc_up1", "hc_inject1"],
                ];
                put(roles[s][0], format!("{b}hc_{side}_norm.weight"))?;
                put(roles[s][1], format!("{b}hc_{side}_down.weight"))?;
                put(roles[s][2], format!("{b}hc_{side}_up.weight"))?;
                put(roles[s][3], format!("{b}hc_{side}_inject.weight"))?;
            }
            if is_attn_layer(il) {
                for (role, name) in [
                    ("q", "attn_q.weight"),
                    ("k", "attn_k.weight"),
                    ("v", "attn_v.weight"),
                    ("o", "attn_output.weight"),
                    ("q_norm", "attn_q_norm.weight"),
                    ("k_norm", "attn_k_norm.weight"),
                    ("idx_q", "indexer.q_proj.weight"),
                    ("idx_k", "indexer.k_proj.weight"),
                    ("idx_q_norm", "indexer.q_norm.weight"),
                    ("idx_k_norm", "indexer.k_norm.weight"),
                ] {
                    put(role, format!("{b}{name}"))?;
                }
            } else {
                for (role, name) in [
                    ("qkv", "attn_qkv.weight"),
                    ("zgate", "attn_gate.weight"),
                    ("alpha", "ssm_alpha.weight"),
                    ("beta", "ssm_beta.weight"),
                    ("ssm_a", "ssm_a"),
                    ("dt_bias", "ssm_dt.bias"),
                    ("conv1d", "ssm_conv1d.weight"),
                    ("ssm_norm", "ssm_norm.weight"),
                    ("ssm_out", "ssm_out.weight"),
                ] {
                    put(role, format!("{b}{name}"))?;
                }
            }
            for (role, name) in [
                ("gate_inp", "ffn_gate_inp.weight"),
                ("gate_inp_shexp", "ffn_gate_inp_shexp.weight"),
                ("gate_shexp", "ffn_gate_shexp.weight"),
                ("up_shexp", "ffn_up_shexp.weight"),
                ("down_shexp", "ffn_down_shexp.weight"),
                ("gate_exps", "ffn_gate_exps.weight"),
                ("up_exps", "ffn_up_exps.weight"),
                ("down_exps", "ffn_down_exps.weight"),
            ] {
                put(role, format!("{b}{name}"))?;
            }
            if il == PLE_LAYER {
                for (role, name) in [
                    ("ple_key", "ple_key.weight"),
                    ("ple_value", "ple_value.weight"),
                    ("ple_norm_key", "ple_norm_key.weight"),
                    ("ple_norm_query", "ple_norm_query.weight"),
                    ("ple_norm_conv", "ple_norm_conv.weight"),
                    ("ple_conv1d", "ple_conv1d.weight"),
                ] {
                    put(role, format!("{b}{name}"))?;
                }
            }
            layers.push(layer);
        }
        Ok(Weights {
            layers,
            tok_embd: get("token_embd.weight")?,
            out_hc_norm: get("output_hc_norm.weight")?,
            out_hc_down: get("output_hc_down.weight")?,
            out_hc_up: get("output_hc_up.weight")?,
            output: get("output.weight")?,
        })
    }
}

/// The named device buffers of a pass (`engine.hip` `ScratchPool`): one per name, made at its first ask and kept for the
/// engine's life. A later ask of a name for more floats than its buffer holds stops the program.
pub(crate) trait Scratch {
    fn get(&mut self, name: &str, floats: usize) -> *mut f32;
}

/// One segment of a pass: `t` tokens that continue the context `c`.
#[derive(Clone, Copy)]
pub(crate) struct Seg {
    pub c: *mut Qwen38Context,
    pub t: u32,
}

impl Seg {
    fn ctx(&self) -> &Qwen38Context {
        // SAFETY: the worker gives live contexts of this engine.
        unsafe { &*self.c }
    }
}

/// The knobs of the pass build, from the environment (`engine.hip` `fuse_enabled` and the rest).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Knobs {
    pub fast_qmm: bool,
    /// `MK_FUSE`: the fused staging and the fused hc_post + norm at decode.
    pub fuse: bool,
    /// `MK_FUSE_HC`: the combine + rms fusion; needs `fuse`.
    pub fuse_hc: bool,
    /// `MK_FUSE_HCMIX`: the hc up + hc_mix fusion; needs `fuse`.
    pub fuse_hcmix: bool,
    /// `MK_ELIDE`: the barrier schedule.
    pub elide: bool,
    /// `MK_TAIL`: the barrier tail.
    pub tail: bool,
    /// `MK_DEBUG_LAYERS`: build the first n layers and no head (bring-up), or 0 for all of them.
    pub debug_layers: u32,
}

/// The value of a 0-or-1 knob: unset or 1 is on, 0 is off, and any other value is an error that names it.
pub(crate) fn switch(name: &str, value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => Err(format!("{name}={other}: want 0 or 1")),
    }
}

impl Knobs {
    /// The knobs from `var`, one environment lookup. A bad value is an error that refuses the pass.
    pub(crate) fn from(fast_qmm: bool, var: impl Fn(&str) -> Option<String>) -> Result<Knobs, String> {
        let on = |name: &str| switch(name, var(name).as_deref());
        let debug_layers = match var("MK_DEBUG_LAYERS") {
            None => 0,
            Some(value) => match value.parse::<u32>() {
                Ok(n) if (1..=LAYERS).contains(&n) => n,
                _ => return Err(format!("MK_DEBUG_LAYERS={value} out of 1..{LAYERS}")),
            },
        };
        Ok(Knobs {
            fast_qmm,
            fuse: on("MK_FUSE")?,
            fuse_hc: on("MK_FUSE_HC")?,
            fuse_hcmix: on("MK_FUSE_HCMIX")?,
            elide: on("MK_ELIDE")?,
            tail: on("MK_TAIL")?,
            debug_layers,
        })
    }
}

/// The fixed device buffers of the pass: the token ids, the head's rows of a pass of several segments, and the rope
/// table of the attention prep.
#[derive(Clone, Copy)]
pub(crate) struct PassBuffers {
    pub tokens: *const i32,
    pub head_rows: *const u32,
    pub inv_freq: *const f32,
}

/// The rope table `inv_freq[i] = 1 / theta^(2i / 64)`, in f32 as `Qwen4ExpTextRotaryEmbedding`'s default init
/// (`blocks.hip` `inv_freq_table`).
pub(crate) fn inv_freq() -> [f32; (ROPE_DIMS / 2) as usize] {
    std::array::from_fn(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / ROPE_DIMS as f32))
}

pub(crate) fn st(op: u32, layer: u32) -> Stage {
    Stage {
        op,
        layer,
        sub: 0,
        sched: 0,
        args: StageArgs { raw: [0; 512] },
    }
}

fn qmmh_type(t: u32) -> bool {
    matches!(
        t,
        Q8_0 | Q8_0_TILES | Q6_K | IQ4_NL | IQ4_XS | IQ3_S | IQ3_S_TILES | Q6_K_TILES | IQ4_NL_TILES
    )
}

fn qmm_op(t: u32, fast: bool) -> u32 {
    if fast && qmmh_type(t) { ST_QMM_H } else { ST_QMM }
}

/// `engine.h` `bulky_weight`: a matrix of 8 MiB or more.
fn bulky_weight(row_bytes: u64, rows: u32) -> u32 {
    u32::from(row_bytes * u64::from(rows) >= 8 << 20)
}

fn qmmh_slice_units(units: u32, ksplit: u32) -> u32 {
    units.div_ceil(ksplit).div_ceil(QMMH_CHUNK_UNITS) * QMMH_CHUNK_UNITS
}

fn qmmh_ksplit_ok(k: u32, ksplit: u32) -> bool {
    ksplit >= 1 && k.is_multiple_of(32) && (ksplit - 1) * qmmh_slice_units(k / 32, ksplit) < k / 32
}

/// `qmmh.h` `qmmh_ksplit`: about 80 items at one token tile for a matrix of fewer than 20 row groups, with at least
/// 8 units for each slice and at most 16 slices. The split depends on the matrix shape alone.
pub(crate) fn qmmh_ksplit(n: u32, k: u32) -> u32 {
    let groups = n.div_ceil(QMMH_ROWS);
    let units = k / 32;
    if groups >= 20 || units < 16 {
        return 1;
    }
    let mut ks = 80u32.div_ceil(groups).min(16).min(units / 8);
    while ks > 1 && !qmmh_ksplit_ok(k, ks) {
        ks -= 1;
    }
    ks
}

fn qmm_args(w: &Tensor, n: u32, k: u32, t: u32, x: *const f32, ldx: u32, y: *mut f32, ldy: u32) -> QmmArgs {
    QmmArgs {
        r#type: w.kind,
        w: w.d,
        ldw: w.row_bytes,
        n,
        k,
        t,
        x,
        ldx,
        x_row: std::ptr::null(),
        y,
        ldy,
        y_row: std::ptr::null(),
        wbulk: bulky_weight(w.row_bytes, n),
        ksplit: 0,
        part: std::ptr::null_mut(),
        xin: XIN_PLAIN,
        xks: 0,
        xp: std::ptr::null(),
        xp2: std::ptr::null(),
        xscale: 0.0,
        yout: YOUT_PLAIN,
        mix_xn: std::ptr::null(),
        mix_z: std::ptr::null(),
        mix_out: std::ptr::null_mut(),
        mix_inject: std::ptr::null_mut(),
        mix_e: 0,
    }
}

/// `engine.hip` `qmm_stage`: a matmul of the mixer and the head, with the whole-tile check.
#[allow(clippy::too_many_arguments)]
fn qmm_stage(layer: u32, w: &Tensor, n: u32, k: u32, x: *const f32, ldx: u32, t: u32, y: *mut f32, ldy: u32) -> Stage {
    assert_eq!(w.ne[0], u64::from(k), "qmm: weight has K={}, want {k}", w.ne[0]);
    let tiled = matches!(w.kind, Q8_0_TILES | IQ3_S_TILES | Q6_K_TILES | IQ4_NL_TILES);
    assert!(
        !tiled || w.ne[1] * w.ne[2] == u64::from(n),
        "qmm over {n} rows of a tiled matrix with {}",
        w.ne[1] * w.ne[2]
    );
    let mut s = st(qmm_op(w.kind, KNOB_FAST.with(|f| f.get())), layer);
    s.args.qmm = qmm_args(w, n, k, t, x, ldx, y, ldy);
    s
}

/// `blocks.hip` `qmm_st`: a matmul of a block, with the shape check and no whole-tile check.
fn qmm_st(il: u32, w: &Tensor, n: u32, k: u32, x: *const f32, t: u32, y: *mut f32) -> Stage {
    assert!(
        w.ne[0] == u64::from(k) && (w.ne[1] == u64::from(n) || (n == 1 && w.ne[1] == 1)),
        "layer {il}: qmm weight is {}x{}, want K={k} N={n}",
        w.ne[0],
        w.ne[1]
    );
    let mut s = st(qmm_op(w.kind, KNOB_FAST.with(|f| f.get())), il);
    s.args.qmm = qmm_args(w, n, k, t, x, k, y, n);
    s
}

thread_local! {
    /// The fast-path knob of the pass that this thread builds. The build sets it first.
    static KNOB_FAST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

#[allow(clippy::too_many_arguments)]
fn rms_stage(
    layer: u32,
    x: *const f32,
    y: *mut f32,
    g: &Tensor,
    t: u32,
    groups: u32,
    d: u32,
    per_group: bool,
) -> Stage {
    let mut s = st(ST_RMS, layer);
    s.args.rms = RmsArgs {
        x,
        ldx: groups * d,
        y,
        ldy: groups * d,
        gamma: g.f32(),
        t,
        n_groups: groups,
        d,
        gamma_per_group: u32::from(per_group),
        eps: RMS_EPS,
    };
    s
}

/// The stages of a pass, in the order of the schedule.
pub(crate) struct Built {
    pub stages: Vec<Stage>,
}

/// The builder of one pass.
pub(crate) struct Builder<'a> {
    pub w: &'a Weights,
    pub scratch: &'a mut dyn Scratch,
    pub buffers: PassBuffers,
    pub knobs: Knobs,
}

const MP: usize = MAX_PASS_TOKENS as usize;

impl Builder<'_> {
    fn scratch(&mut self, name: &str, floats: usize) -> *mut f32 {
        self.scratch.get(name, floats)
    }

    /// The whole build of a pass (`engine.hip` `build`): the stages, the split-K pass, the fusions and the schedule.
    pub(crate) fn build(&mut self, seg: &[Seg], t: u32, want_logits: bool) -> Built {
        KNOB_FAST.with(|f| f.set(self.knobs.fast_qmm));
        let mut ends = Vec::new();
        let mut s = self.build_stages(seg, t, want_logits, &mut ends);
        let elide = self.knobs.elide;
        let part = self.scratch("qmm_part", PART_FLOATS);
        split_k(&mut s, &mut ends, part, PART_FLOATS, elide);
        if t <= FUSE_MAX_T && self.knobs.fuse {
            let part2 = self.scratch("qmm_part2", PART_FLOATS);
            fuse_prologues(&mut s, &mut ends, part2, PART_FLOATS);
            if self.knobs.fuse_hc {
                fuse_hc(&mut s, &mut ends);
            }
        }
        // The fold lives in the f16 matmul's epilogue (qmmh.h); the exact engine (MK_FAST_QMM=0) has only kStQmm
        // stages, so its passes keep their hc_mix stages.
        if self.knobs.fuse && self.knobs.fuse_hcmix && self.knobs.fast_qmm {
            fuse_hc_mix(&mut s, &mut ends);
        }
        let plan = pass_plan(&s, &ends, elide, self.knobs.tail);
        let n = s.len();
        assert!(
            plan.order.len() == n && plan.sched.len() == n,
            "schedule: the plan has {} stages, the pass {n}",
            plan.order.len()
        );
        let mut seen = vec![false; n];
        let mut out = Vec::with_capacity(n);
        for (k, &index) in plan.order.iter().enumerate() {
            let index = index as usize;
            assert!(
                index < n && !seen[index],
                "schedule: the plan names stage {index} twice or past {n} stages"
            );
            seen[index] = true;
            let mut stage = s[index];
            stage.sched = plan.sched[k];
            out.push(stage);
        }
        Built { stages: out }
    }

    /// `engine.hip` `build_stages`.
    fn build_stages(&mut self, seg: &[Seg], t: u32, want_logits: bool, ends: &mut Vec<u32>) -> Vec<Stage> {
        let mut s: Vec<Stage> = Vec::new();
        let e = self.scratch("e", MP * EMBD as usize);
        let h = self.scratch("h", MP * HC_DIM as usize);
        let xn = self.scratch("xn", MP * HC_DIM as usize);
        let down = self.scratch("down", MP * HC_RANK as usize);
        let act = self.scratch("act", MP * HC_RANK as usize);
        let up = self.scratch("up", MP * HC_DIM as usize);
        let z4 = self.scratch("z4", MP * 4);
        let inj = self.scratch("inj", MP * 4);
        let mixed = self.scratch("mixed", MP * EMBD as usize);
        let out = self.scratch("out", MP * EMBD as usize);

        let mut stage = st(ST_EMBED, 0);
        stage.args.embed = EmbedArgs {
            w: self.w.tok_embd.d,
            ldw: self.w.tok_embd.row_bytes,
            k: EMBD,
            ids: self.buffers.tokens,
            t,
            out: e,
        };
        s.push(stage);
        let mut stage = st(ST_REPEAT4, 0);
        stage.args.rep = Repeat4Args { e, h, t, d: EMBD };
        s.push(stage);

        let w = self.w;
        let hc_pre = |s: &mut Vec<Stage>, il: u32, side: usize, inject: bool| {
            let (gn, wd, wu) = if il < LAYERS {
                let l = &w.layers[il as usize];
                let roles = [["hc_norm0", "hc_down0", "hc_up0"], ["hc_norm1", "hc_down1", "hc_up1"]];
                (l.get(roles[side][0]), l.get(roles[side][1]), l.get(roles[side][2]))
            } else {
                (w.out_hc_norm, w.out_hc_down, w.out_hc_up)
            };
            s.push(rms_stage(il, h, xn, &gn, t, 4, EMBD, true));
            s.push(qmm_stage(il, &wd, HC_RANK, HC_DIM, xn, HC_DIM, t, down, HC_RANK));
            let mut a = st(ST_SILU_SCALE, il);
            a.args.silu = SiluScaleArgs {
                x: down,
                y: act,
                n: t * HC_RANK,
                scale: 1.0 / HC as f32,
            };
            s.push(a);
            s.push(qmm_stage(il, &wu, HC_DIM, HC_RANK, act, HC_RANK, t, up, HC_DIM));
            if inject {
                let inj_w = w.layers[il as usize].get(["hc_inject0", "hc_inject1"][side]);
                s.push(qmm_stage(il, &inj_w, 4, HC_DIM, xn, HC_DIM, t, z4, 4));
            }
            let mut m = st(ST_HC_MIX, il);
            m.args.mix = HcMixArgs {
                xn,
                up,
                mixed,
                z: if inject { z4 } else { std::ptr::null() },
                inject: inj,
                t,
                e: EMBD,
            };
            s.push(m);
        };
        let hc_post = |s: &mut Vec<Stage>, il: u32| {
            let mut cmb = st(ST_HC_COMBINE, il);
            cmb.args.comb = HcCombineArgs {
                h,
                out,
                inject: inj,
                t,
                e: EMBD,
            };
            s.push(cmb);
        };

        let n_layers = if self.knobs.debug_layers > 0 {
            assert!(
                !want_logits,
                "MK_DEBUG_LAYERS builds no head; ask for layers, not logits"
            );
            eprintln!("engine: MK_DEBUG_LAYERS={} (bring-up)", self.knobs.debug_layers);
            self.knobs.debug_layers
        } else {
            LAYERS
        };
        for il in 0..n_layers {
            if il == PLE_LAYER {
                let l = &w.layers[il as usize];
                let pemb = self.scratch("ple_emb", MP * EMBD as usize);
                let pk = self.scratch("ple_k", MP * HC_DIM as usize);
                let kn = self.scratch("ple_kn", MP * HC_DIM as usize);
                let pv = self.scratch("ple_v", MP * EMBD as usize);
                let gv = self.scratch("ple_gv", MP * HC_DIM as usize);
                let gvn = self.scratch("ple_gvn", MP * HC_DIM as usize);
                let py = self.scratch("ple_y", MP * HC_DIM as usize);
                s.push(qmm_stage(
                    il,
                    &l.get("ple_key"),
                    HC_DIM,
                    EMBD,
                    pemb,
                    EMBD,
                    t,
                    pk,
                    HC_DIM,
                ));
                s.push(rms_stage(il, pk, kn, &l.get("ple_norm_key"), t, 4, EMBD, true));
                s.push(qmm_stage(il, &l.get("ple_value"), EMBD, EMBD, pemb, EMBD, t, pv, EMBD));
                s.push(rms_stage(il, h, xn, &l.get("ple_norm_query"), t, 4, EMBD, true));
                let mut g = st(ST_PLE_GATE, il);
                g.args.pgate = PleGateArgs {
                    kn,
                    qn: xn,
                    pv,
                    gv,
                    t,
                    e: EMBD,
                };
                s.push(g);
                s.push(rms_stage(il, gv, gvn, &l.get("ple_norm_conv"), t, 4, EMBD, true));
                let ns = seg.len() as u32;
                let mut r0 = 0usize;
                for (i, sg) in seg.iter().enumerate() {
                    let mut cv = st(ST_PLE_CONV, il);
                    cv.sub = if ns > 1 { i as u32 + 1 } else { 0 };
                    cv.args.pconv = PleConvArgs {
                        x: gvn.wrapping_add(r0 * HC_DIM as usize),
                        state: sg.ctx().ple_conv,
                        w: l.get("ple_conv1d").f32(),
                        y: py.wrapping_add(r0 * HC_DIM as usize),
                        t: sg.t,
                        c: HC_DIM,
                    };
                    s.push(cv);
                    r0 += sg.t as usize;
                }
                let mut ad = st(ST_ADD, il);
                ad.args.add = AddArgs {
                    h,
                    a: gv,
                    b: py,
                    n: t * HC_DIM,
                };
                s.push(ad);
            }
            hc_pre(&mut s, il, 0, true);
            if is_attn_layer(il) {
                self.add_attn(&mut s, seg, il, t, mixed, out);
            } else {
                self.add_gdn(&mut s, seg, il, t, mixed, out);
            }
            hc_post(&mut s, il);
            hc_pre(&mut s, il, 1, true);
            self.add_moe(&mut s, il, t, mixed, out);
            hc_post(&mut s, il);
            ends.push(s.len() as u32);
        }
        if n_layers < LAYERS {
            return s;
        }
        hc_pre(&mut s, LAYERS, 0, false);
        if want_logits {
            let logits = self.scratch("logits", MAX_SEGMENTS as usize * VOCAB as usize);
            let ns = seg.len() as u32;
            if ns == 1 {
                let x = mixed.wrapping_add((t as usize - 1) * EMBD as usize);
                s.push(qmm_stage(LAYERS, &w.output, VOCAB, EMBD, x, EMBD, 1, logits, VOCAB));
            } else {
                let mut hd = qmm_stage(LAYERS, &w.output, VOCAB, EMBD, mixed, EMBD, ns, logits, VOCAB);
                hd.args.qmm.x_row = self.buffers.head_rows;
                s.push(hd);
            }
        }
        s
    }

    /// `blocks.hip` `add_gdn`.
    fn add_gdn(&mut self, s: &mut Vec<Stage>, seg: &[Seg], il: u32, t: u32, mixed: *const f32, out: *mut f32) {
        let l = &self.w.layers[il as usize];
        let g = (il - (il + 1) / 4) as usize;
        let qkv = self.scratch("gdn_qkv", MP * GDN_CONV_DIM as usize);
        let qkvc = self.scratch("gdn_qkvc", MP * GDN_CONV_DIM as usize);
        let z = self.scratch("gdn_z", MP * GDN_VAL_DIM as usize);
        let a = self.scratch("gdn_a", MP * GDN_V_HEADS as usize);
        let b = self.scratch("gdn_b", MP * GDN_V_HEADS as usize);
        let core = self.scratch("gdn_core", MP * GDN_VAL_DIM as usize);
        s.push(qmm_st(il, &l.get("qkv"), GDN_CONV_DIM, EMBD, mixed, t, qkv));
        s.push(qmm_st(il, &l.get("zgate"), GDN_VAL_DIM, EMBD, mixed, t, z));
        s.push(qmm_st(il, &l.get("alpha"), GDN_V_HEADS, EMBD, mixed, t, a));
        s.push(qmm_st(il, &l.get("beta"), GDN_V_HEADS, EMBD, mixed, t, b));
        let ns = seg.len() as u32;
        let sub = |i: usize| if ns > 1 { i as u32 + 1 } else { 0 };
        let mut r0 = 0usize;
        for (i, sg) in seg.iter().enumerate() {
            let conv_state = sg
                .ctx()
                .gdn_conv
                .wrapping_add(g * GDN_CONV_DIM as usize * (GDN_CONV as usize - 1));
            let mut cv = st(ST_GDN_CONV, il);
            cv.sub = sub(i);
            cv.args.gconv = GdnConvArgs {
                x: qkv.wrapping_add(r0 * GDN_CONV_DIM as usize),
                state_in: conv_state,
                w: l.get("conv1d").f32(),
                y: qkvc.wrapping_add(r0 * GDN_CONV_DIM as usize),
                state_out: conv_state,
                t: sg.t,
            };
            s.push(cv);
            r0 += sg.t as usize;
        }
        let mut r0 = 0usize;
        for (i, sg) in seg.iter().enumerate() {
            let state = sg
                .ctx()
                .gdn_state
                .wrapping_add(g * (GDN_V_HEADS * GDN_K_DIM * GDN_V_DIM) as usize);
            let mut co = st(ST_GDN_CORE, il);
            co.sub = sub(i);
            co.args.gcore = GdnCoreArgs {
                qkv: qkvc.wrapping_add(r0 * GDN_CONV_DIM as usize),
                z: z.wrapping_add(r0 * GDN_VAL_DIM as usize),
                a: a.wrapping_add(r0 * GDN_V_HEADS as usize),
                b: b.wrapping_add(r0 * GDN_V_HEADS as usize),
                ssm_a: l.get("ssm_a").f32(),
                dt_bias: l.get("dt_bias").f32(),
                gamma: l.get("ssm_norm").f32(),
                state_in: state,
                y: core.wrapping_add(r0 * GDN_VAL_DIM as usize),
                state_out: state,
                t: sg.t,
            };
            s.push(co);
            r0 += sg.t as usize;
        }
        s.push(qmm_st(il, &l.get("ssm_out"), EMBD, GDN_VAL_DIM, core, t, out));
    }

    /// `blocks.hip` `add_moe`.
    fn add_moe(&mut self, s: &mut Vec<Stage>, il: u32, t: u32, mixed: *const f32, out: *mut f32) {
        let l = &self.w.layers[il as usize];
        let a_rows = t * TOP_K;
        let e = EXPERTS as usize;
        let tk = TOP_K as usize;
        let logits = self.scratch("moe_logits", MP * e);
        let tidx = self.scratch("moe_tidx", MP * tk) as *mut i32;
        let tw = self.scratch("moe_tw", MP * tk);
        let cnt = self.scratch("moe_cnt", e) as *mut u32;
        let off = self.scratch("moe_off", e + 1) as *mut u32;
        let asg = self.scratch("moe_asg", MP * tk) as *mut u32;
        let tok = self.scratch("moe_tok", MP * tk) as *mut u32;
        let igu = self.scratch("moe_igu", e + 1) as *mut u32;
        let idn = self.scratch("moe_idn", e + 1) as *mut u32;
        let gu = self.scratch("moe_gu", MP * tk * 2 * FF as usize);
        let hh = self.scratch("moe_h", MP * tk * FF as usize);
        let y = self.scratch("moe_y", MP * tk * EMBD as usize);
        let sg = self.scratch("moe_sg", MP * FF_SHARED as usize);
        let su = self.scratch("moe_su", MP * FF_SHARED as usize);
        let sh = self.scratch("moe_sh", MP * FF_SHARED as usize);
        let sd = self.scratch("moe_sd", MP * EMBD as usize);
        let sgt = self.scratch("moe_sgt", MP);

        s.push(qmm_st(il, &l.get("gate_inp"), EXPERTS, EMBD, mixed, t, logits));
        let mut r = st(ST_MOE_ROUTER, il);
        r.args.router = MoeRouterArgs {
            logits,
            topk_idx: tidx,
            topk_w: tw,
            t,
        };
        s.push(r);
        let fast = u32::from(self.knobs.fast_qmm);
        let mut gr = st(ST_MOE_GROUP, il);
        gr.args.group = MoeGroupArgs {
            topk_idx: tidx,
            t,
            count: cnt,
            offset: off,
            assign: asg,
            token: tok,
            item_gu: igu,
            item_dn: idn,
            fast,
        };
        s.push(gr);
        let (ge, ue, de) = (l.get("gate_exps"), l.get("up_exps"), l.get("down_exps"));
        assert!(
            ge.ne[0] == u64::from(EMBD)
                && ge.ne[1] == u64::from(FF)
                && ue.ne[0] == u64::from(EMBD)
                && ue.ne[1] == u64::from(FF)
                && de.ne[0] == u64::from(FF)
                && de.ne[1] == u64::from(EMBD),
            "layer {il}: expert tensors have unexpected shapes"
        );
        let f = MoeFfnArgs {
            type_gate: ge.kind,
            type_up: ue.kind,
            type_down: de.kind,
            gate: ge.d,
            up: ue.d,
            down: de.d,
            ld_gate: ge.row_bytes,
            ld_up: ue.row_bytes,
            ld_down: de.row_bytes,
            x: mixed,
            gu,
            h: hh,
            y,
            count: cnt,
            offset: off,
            assign: asg,
            token: tok,
            item_gu: igu,
            item_dn: idn,
            fast,
        };
        let mut fg = st(ST_MOE_GU, il);
        fg.args.ffn = f;
        s.push(fg);
        let mut sw = st(ST_SWIGLU, il);
        sw.args.swiglu = SwigluArgs {
            g: gu,
            u: gu.wrapping_add(FF as usize),
            ldg: 2 * FF,
            ldu: 2 * FF,
            out: hh,
            ldo: FF,
            rows: a_rows,
            cols: FF,
        };
        s.push(sw);
        let mut fd = st(ST_MOE_DOWN, il);
        fd.args.ffn = f;
        s.push(fd);
        s.push(qmm_st(il, &l.get("gate_shexp"), FF_SHARED, EMBD, mixed, t, sg));
        s.push(qmm_st(il, &l.get("up_shexp"), FF_SHARED, EMBD, mixed, t, su));
        let mut ss = st(ST_SWIGLU, il);
        ss.args.swiglu = SwigluArgs {
            g: sg,
            u: su,
            ldg: FF_SHARED,
            ldu: FF_SHARED,
            out: sh,
            ldo: FF_SHARED,
            rows: t,
            cols: FF_SHARED,
        };
        s.push(ss);
        s.push(qmm_st(il, &l.get("down_shexp"), EMBD, FF_SHARED, sh, t, sd));
        s.push(qmm_st(il, &l.get("gate_inp_shexp"), 1, EMBD, mixed, t, sgt));
        let mut cb = st(ST_MOE_COMBINE, il);
        cb.args.mcomb = MoeCombineArgs {
            y,
            topk_idx: tidx,
            topk_w: tw,
            shared: sd,
            shared_gate: sgt,
            out,
            t,
        };
        s.push(cb);
    }

    /// `blocks.hip` `add_attn`.
    fn add_attn(&mut self, s: &mut Vec<Stage>, seg: &[Seg], il: u32, t: u32, mixed: *const f32, out: *mut f32) {
        let l = &self.w.layers[il as usize];
        let ai = (il / 4) as usize;
        let max_sel = IDX_BUDGET + IDX_RATIO - 1;
        let topk = IDX_BUDGET / IDX_RATIO;
        let q = self.scratch("att_q", MP * (HEADS * 2 * HEAD_DIM) as usize);
        let k = self.scratch("att_k", MP * (KV_HEADS * HEAD_DIM) as usize);
        let v = self.scratch("att_v", MP * (KV_HEADS * HEAD_DIM) as usize);
        let iq = self.scratch("att_iq", MP * (IDX_HEADS * IDX_DIM) as usize);
        let ik = self.scratch("att_ik", MP * IDX_DIM as usize);
        let qo = self.scratch("att_qo", MP * (HEADS * HEAD_DIM) as usize);
        let go = self.scratch("att_go", MP * (HEADS * HEAD_DIM) as usize);
        let iqo = self.scratch("att_iqo", MP * (IDX_HEADS * IDX_DIM) as usize);
        let scores = self.scratch("att_scores", MP * (MAX_CONTEXT / IDX_RATIO) as usize);
        let picked = self.scratch("att_picked", MP * topk as usize) as *mut i32;
        let sel = self.scratch("att_sel", MP * ATTN_MAX_SEL_MASKED as usize) as *mut i32;
        let sel_n = self.scratch("att_seln", MP) as *mut u32;
        let ao = self.scratch("att_out", MP * (HEADS * HEAD_DIM) as usize);

        s.push(qmm_st(il, &l.get("q"), HEADS * 2 * HEAD_DIM, EMBD, mixed, t, q));
        s.push(qmm_st(il, &l.get("k"), KV_HEADS * HEAD_DIM, EMBD, mixed, t, k));
        s.push(qmm_st(il, &l.get("v"), KV_HEADS * HEAD_DIM, EMBD, mixed, t, v));
        s.push(qmm_st(il, &l.get("idx_q"), IDX_HEADS * IDX_DIM, EMBD, mixed, t, iq));
        s.push(qmm_st(il, &l.get("idx_k"), IDX_DIM, EMBD, mixed, t, ik));
        let (mut prep, mut pool, mut score, mut select, mut attend) = (vec![], vec![], vec![], vec![], vec![]);
        let mut soff = 0usize;
        let ns = seg.len() as u32;
        let mut r = 0usize;
        for (i, sg) in seg.iter().enumerate() {
            let c = sg.ctx();
            let (ti, p0, max_nb) = (sg.t, c.n, c.max / IDX_RATIO);
            let sub = if ns > 1 { i as u32 + 1 } else { 0 };
            assert!(c.max <= MAX_CONTEXT, "context of capacity {} past {MAX_CONTEXT}", c.max);
            let kv =
                c.kv.wrapping_add(ai * c.max as usize * (2 * KV_HEADS * HEAD_DIM) as usize);
            let ikc = c.idx_keys.wrapping_add(ai * c.max as usize * IDX_DIM as usize);
            let pkc = c.idx_pooled.wrapping_add(ai * max_nb as usize * IDX_DIM as usize);
            assert!(
                c.pooled_n <= p0 / IDX_RATIO,
                "attention: pooled-key watermark {} past the context's {} complete blocks ({p0} tokens)",
                c.pooled_n,
                p0 / IDX_RATIO
            );
            assert!(
                soff + ti as usize * max_nb as usize <= MP * (MAX_CONTEXT / IDX_RATIO) as usize,
                "attention: block scores of the segments past their buffer"
            );
            let mut pr = st(ST_ATTN_PREP, il);
            pr.sub = sub;
            pr.args.aprep = AttnPrepArgs {
                q_raw: q.wrapping_add(r * (HEADS * 2 * HEAD_DIM) as usize),
                k_raw: k.wrapping_add(r * (KV_HEADS * HEAD_DIM) as usize),
                v_raw: v.wrapping_add(r * (KV_HEADS * HEAD_DIM) as usize),
                iq_raw: iq.wrapping_add(r * (IDX_HEADS * IDX_DIM) as usize),
                ik_raw: ik.wrapping_add(r * IDX_DIM as usize),
                q_gamma: l.get("q_norm").f32(),
                k_gamma: l.get("k_norm").f32(),
                idxq_gamma: l.get("idx_q_norm").f32(),
                inv_freq: self.buffers.inv_freq,
                t: ti,
                p0,
                q_out: qo.wrapping_add(r * (HEADS * HEAD_DIM) as usize),
                gate_out: go.wrapping_add(r * (HEADS * HEAD_DIM) as usize),
                iq_out: iqo.wrapping_add(r * (IDX_HEADS * IDX_DIM) as usize),
                kv_cache: kv,
                ik_cache: ikc,
            };
            prep.push(pr);
            let sa = AttnSelectArgs {
                iq_out: iqo.wrapping_add(r * (IDX_HEADS * IDX_DIM) as usize),
                ik_cache: ikc,
                idxk_gamma: l.get("idx_k_norm").f32(),
                inv_freq: self.buffers.inv_freq,
                t: ti,
                p0,
                budget: IDX_BUDGET,
                ratio: IDX_RATIO,
                block_scores: scores.wrapping_add(soff),
                block_scores_out: std::ptr::null_mut(),
                picked: picked.wrapping_add(r * topk as usize),
                sel: sel.wrapping_add(r * max_sel as usize),
                sel_count: sel_n.wrapping_add(r),
                max_nb,
                max_sel,
                block_topk_max: topk,
                pooled: pkc,
                pool_b0: c.pooled_n,
            };
            // Every token has select_count == nb while the pass stays within budget + ratio - 1 visible cells.
            let dense = u64::from(p0) + u64::from(ti) <= u64::from(IDX_BUDGET) + u64::from(IDX_RATIO) - 1;
            let mut se = st(ST_ATTN_EMIT, il);
            se.sub = sub;
            se.args.asel = sa;
            if !dense {
                let mut sc = se;
                sc.op = ST_ATTN_SCORE;
                score.push(sc);
            }
            select.push(se);
            if (u64::from(p0) + u64::from(ti)) / u64::from(IDX_RATIO) > u64::from(c.pooled_n) {
                let mut po = se;
                po.op = ST_ATTN_POOL;
                pool.push(po);
            }
            let mut at = st(ST_ATTN_CORE, il);
            at.sub = sub;
            at.args.aatt = AttnAttendArgs {
                q_out: qo.wrapping_add(r * (HEADS * HEAD_DIM) as usize),
                gate_out: go.wrapping_add(r * (HEADS * HEAD_DIM) as usize),
                kv_cache: kv,
                sel: sel.wrapping_add(r * max_sel as usize),
                sel_count: sel_n.wrapping_add(r),
                t: ti,
                max_sel,
                attn_out: ao.wrapping_add(r * (HEADS * HEAD_DIM) as usize),
            };
            attend.push(at);
            soff += ti as usize * max_nb as usize;
            r += ti as usize;
        }
        for list in [prep, pool, score, select, attend] {
            s.extend(list);
        }
        s.push(qmm_st(il, &l.get("o"), EMBD, HEADS * HEAD_DIM, ao, t, out));
    }

    /// `blocks.hip` `pool_stages`: one pool stage for each attention layer that brings the context's pooled-key cache
    /// from block `b0` up to its `n / 4` (the restore of a context).
    pub(crate) fn pool_stages(&self, c: &Qwen38Context, b0: u32) -> Vec<Stage> {
        let max_nb = c.max / IDX_RATIO;
        assert!(
            b0 <= c.n / IDX_RATIO,
            "pool_stages: from block {b0} past the context's {} complete blocks",
            c.n / IDX_RATIO
        );
        let mut out = Vec::new();
        for (ai, il) in (3..LAYERS).step_by(4).enumerate() {
            let l = &self.w.layers[il as usize];
            let mut po = st(ST_ATTN_POOL, il);
            po.args.asel = AttnSelectArgs {
                iq_out: std::ptr::null(),
                ik_cache: c.idx_keys.wrapping_add(ai * c.max as usize * IDX_DIM as usize),
                idxk_gamma: l.get("idx_k_norm").f32(),
                inv_freq: self.buffers.inv_freq,
                t: 0,
                p0: c.n,
                budget: IDX_BUDGET,
                ratio: IDX_RATIO,
                block_scores: std::ptr::null_mut(),
                block_scores_out: std::ptr::null_mut(),
                picked: std::ptr::null_mut(),
                sel: std::ptr::null_mut(),
                sel_count: std::ptr::null_mut(),
                max_nb,
                max_sel: 0,
                block_topk_max: 0,
                pooled: c.idx_pooled.wrapping_add(ai * max_nb as usize * IDX_DIM as usize),
                pool_b0: b0,
            };
            out.push(po);
        }
        out
    }
}

/// Map the old layer ends through `at`, the new index of each old stage.
fn remap(ends: &mut [u32], at: &[u32]) {
    for e in ends.iter_mut() {
        *e = at[*e as usize];
    }
}

/// `engine.hip` `split_k`: every f16 qmm stage on a narrow matrix gets `qmmh_ksplit(n, k)` K slices, writing partials
/// to `part`, and a sum stage right after it. With `own_ranges` (the schedule), each split gets its own partial range,
/// dealt round the buffer in whole 256-byte lines.
fn split_k(s: &mut Vec<Stage>, ends: &mut [u32], part: *mut f32, part_floats: usize, own_ranges: bool) {
    let mut out = Vec::with_capacity(s.len() + s.len() / 4);
    let mut at = vec![0u32; s.len() + 1];
    let mut next = 0usize;
    for (i, stage) in s.iter().enumerate() {
        at[i] = out.len() as u32;
        let mut st = *stage;
        let ks = if st.op == ST_QMM_H && st.qmm().x_row.is_null() && st.qmm().y_row.is_null() {
            qmmh_ksplit(st.qmm().n, st.qmm().k)
        } else {
            1
        };
        if ks > 1 {
            let q = *st.qmm();
            assert!(
                ks as usize * q.t as usize * q.n as usize <= part_floats,
                "split-K partials for {}x{} x {} tokens (ksplit {ks}) exceed the buffer",
                q.n,
                q.k,
                q.t
            );
            assert!(
                ks <= MAX_KSPLIT,
                "split-K: ksplit {ks} for {}x{} > the sum's {MAX_KSPLIT}",
                q.n,
                q.k
            );
            st.args.qmm.ksplit = ks;
            st.args.qmm.part = part;
            if own_ranges {
                let need = (ks as usize * q.t as usize * q.n as usize).div_ceil(64) * 64;
                if next + need > part_floats {
                    next = 0;
                }
                st.args.qmm.part = part.wrapping_add(next);
                next += need;
            }
            out.push(st);
            st.op = ST_QMM_KSUM;
        }
        out.push(st);
    }
    at[s.len()] = out.len() as u32;
    remap(ends, &at);
    *s = out;
}

/// `engine.hip` `fuse_prologues`: at decode, the split-K sums and the activations between two f16 matmuls move into
/// the consumer's staging, which computes the same values from the producer's partials.
fn fuse_prologues(s: &mut Vec<Stage>, ends: &mut [u32], part2: *mut f32, part_floats: usize) {
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    let mut at = vec![0u32; n + 1];
    let qh = |i: usize| i < n && s[i].op == ST_QMM_H;
    let split = |i: usize| qh(i) && s[i].qmm().ksplit > 1 && i + 1 < n && s[i + 1].op == ST_QMM_KSUM;
    let plain_consumer = |i: usize, x: *const f32, k: u32| {
        qh(i) && {
            let q = s[i].qmm();
            q.x == x && q.k == k && q.ldx == k && q.x_row.is_null() && q.ksplit <= 1 && q.xin == XIN_PLAIN
        }
    };
    let no_layer_end_inside = |b: usize, last: usize| {
        for &e in ends.iter() {
            assert!(
                !(e as usize > b && e as usize <= last),
                "fused staging: layer end {e} inside the fused stages {b}..{last}"
            );
        }
    };
    let mut i = 0usize;
    let mut keep = |i: &mut usize, j: usize, stage: Stage, out: &mut Vec<Stage>| {
        while *i <= j {
            at[*i] = out.len() as u32;
            *i += 1;
        }
        out.push(stage);
    };
    while i < n {
        // A: QmmH(split, y=D) Ksum SiluScale(D -> X) QmmH(x=X)  ->  QmmH(split) QmmH(xin=silu-sum)
        if split(i)
            && i + 3 < n
            && s[i + 2].op == ST_SILU_SCALE
            && s[i + 2].silu().x == s[i].qmm().y as *const f32
            && s[i + 2].silu().n == s[i].qmm().t * s[i].qmm().n
            && plain_consumer(i + 3, s[i + 2].silu().y, s[i].qmm().n)
        {
            let mut c = s[i + 3];
            c.args.qmm.xin = XIN_SILU_SUM;
            c.args.qmm.xks = s[i].qmm().ksplit;
            c.args.qmm.xp = s[i].qmm().part;
            c.args.qmm.xscale = s[i + 2].silu().scale;
            c.args.qmm.x = std::ptr::null();
            let b = i;
            no_layer_end_inside(b, b + 3);
            keep(&mut i, b, s[b], &mut out);
            keep(&mut i, b + 3, c, &mut out);
            continue;
        }
        // B: the shared expert's gate and up, their swiglu, and the down matmul.
        if split(i) && split(i + 2) && i + 5 < n && s[i + 4].op == ST_SWIGLU {
            let (g, u, w) = (*s[i].qmm(), *s[i + 2].qmm(), *s[i + 4].swiglu());
            if g.n == u.n
                && g.t == u.t
                && g.ksplit == u.ksplit
                && w.g == g.y as *const f32
                && w.u == u.y as *const f32
                && w.ldg == g.n
                && w.ldu == u.n
                && w.rows == g.t
                && w.cols == g.n
                && w.ldo == g.n
                && plain_consumer(i + 5, w.out, g.n)
            {
                assert!(
                    u.ksplit as usize * u.t as usize * u.n as usize <= part_floats,
                    "fused staging: up partials exceed part2"
                );
                let (gs, mut us, mut c) = (s[i], s[i + 2], s[i + 5]);
                us.args.qmm.part = part2;
                c.args.qmm.xin = XIN_SWIGLU_SUM;
                c.args.qmm.xks = g.ksplit;
                c.args.qmm.xp = g.part;
                c.args.qmm.xp2 = part2;
                c.args.qmm.x = std::ptr::null();
                let b = i;
                no_layer_end_inside(b, b + 5);
                keep(&mut i, b, gs, &mut out);
                keep(&mut i, b + 2, us, &mut out);
                keep(&mut i, b + 5, c, &mut out);
                continue;
            }
        }
        let (j, stage) = (i, s[i]);
        keep(&mut i, j, stage, &mut out);
    }
    at[n] = out.len() as u32;
    remap(ends, &at);
    *s = out;
}

/// `engine.hip` `fuse_hc`: a combine followed directly by the rms over the same h becomes one stage at decode. A
/// layer end between the pair moves to after the fused stage.
fn fuse_hc(s: &mut Vec<Stage>, ends: &mut [u32]) {
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    let mut at = vec![0u32; n + 1];
    let mut i = 0usize;
    while i < n {
        at[i] = out.len() as u32;
        if s[i].op == ST_HC_COMBINE && i + 1 < n && s[i + 1].op == ST_RMS {
            let (c, r) = (*s[i].comb(), *s[i + 1].rms());
            if r.x == c.h as *const f32
                && r.t == c.t
                && r.n_groups == 4
                && r.d == c.e
                && r.ldx == 4 * c.e
                && r.gamma_per_group == 1
            {
                let mut f = s[i];
                f.op = ST_HC_COMBINE_RMS;
                f.sub = 0;
                f.args.combrms = HcCombineRmsArgs { c, r };
                out.push(f);
                i += 1;
                at[i] = out.len() as u32;
                i += 1;
                continue;
            }
        }
        out.push(s[i]);
        i += 1;
    }
    at[n] = out.len() as u32;
    remap(ends, &at);
    *s = out;
}

/// `engine.hip` `fuse_hc_mix`: the hc up matmul moves to its hc_mix's slot, and its epilogue does hc_mix. The checks
/// stop the program when a stage between them touches `up`, writes `xn`, or writes the matmul's input.
fn fuse_hc_mix(s: &mut Vec<Stage>, ends: &mut [u32]) {
    let n = s.len();
    let mut out = Vec::with_capacity(n);
    let mut at = vec![0u32; n + 1];
    let mut producer_of: Vec<Option<usize>> = vec![None; n];
    let mut dropped = vec![false; n];
    let overlaps = |alo: u64, ahi: u64, blo: u64, bhi: u64| alo < bhi && blo < ahi;
    let footprint =
        |stage: &Stage| schedule::stage_footprint(stage, 0).unwrap_or_else(|error| panic!("mk's pass build: {error}"));
    for m in 0..n {
        if s[m].op != ST_HC_MIX {
            continue;
        }
        let x = *s[m].mix();
        let j = (0..m)
            .rev()
            .find(|&j| (s[j].op == ST_QMM_H || s[j].op == ST_QMM) && s[j].qmm().y as *const f32 == x.up)
            .unwrap_or_else(|| panic!("fuse_hc_mix: no matmul writes the up of hc_mix stage {m}"));
        let u = *s[j].qmm();
        assert!(
            s[j].op == ST_QMM_H,
            "fuse_hc_mix: hc_mix stage {m}: its up matmul is an exact qmm (op {}); MK_FUSE_HCMIX=0 for that",
            s[j].op
        );
        assert!(
            !(s[j].layer != s[m].layer
                || u.n != 4 * x.e
                || u.ldy != u.n
                || u.t != x.t
                || u.ksplit > 1
                || !u.x_row.is_null()
                || !u.y_row.is_null()
                || u.yout != YOUT_PLAIN
                || (u.r#type != Q8_0 && u.r#type != Q8_0_TILES)
                || !x.e.is_multiple_of(16)),
            "fuse_hc_mix: hc_mix stage {m}: the up matmul at {j} is not a plain Q8_0 4E-row projection of its layer"
        );
        for &e in ends.iter() {
            assert!(
                !(e as usize >= j && e as usize <= m),
                "fuse_hc_mix: layer end {e} inside the stages {j}..{m}"
            );
        }
        let fu = footprint(&s[j]);
        let span = 4 * u64::from(x.t) * u64::from(x.e) * 4;
        let (xlo, ulo) = (x.xn as u64, x.up as u64);
        let (xhi, uhi) = (xlo + span, ulo + span);
        for (k, stage) in s.iter().enumerate().take(m).skip(j + 1) {
            let fk = footprint(stage);
            assert!(
                !fk.fence,
                "fuse_hc_mix: a fence stage (op {}) between the up matmul and its hc_mix",
                stage.op
            );
            for a in &fk.acc {
                assert!(
                    !overlaps(a.lo, a.hi, ulo, uhi),
                    "fuse_hc_mix: stage {k} (op {}) touches up",
                    stage.op
                );
                assert!(
                    !(a.write && overlaps(a.lo, a.hi, xlo, xhi)),
                    "fuse_hc_mix: stage {k} (op {}) writes xn",
                    stage.op
                );
                if a.write {
                    for r in &fu.acc {
                        assert!(
                            r.write || !overlaps(a.lo, a.hi, r.lo, r.hi),
                            "fuse_hc_mix: stage {k} (op {}) writes the up matmul's input",
                            stage.op
                        );
                    }
                }
            }
        }
        for (k, stage) in s.iter().enumerate().skip(m + 1) {
            let fk = footprint(stage);
            if fk.fence {
                continue;
            }
            let (mut rewritten, mut reads) = (false, false);
            for a in &fk.acc {
                if overlaps(a.lo, a.hi, ulo, uhi) {
                    if a.write {
                        rewritten = true;
                    } else {
                        reads = true;
                    }
                }
            }
            assert!(
                !(reads && !rewritten),
                "fuse_hc_mix: stage {k} (op {}) reads up after its hc_mix {m}",
                stage.op
            );
            if rewritten {
                break;
            }
        }
        producer_of[m] = Some(j);
        dropped[j] = true;
    }
    for i in 0..n {
        at[i] = out.len() as u32;
        if dropped[i] {
            continue;
        }
        if s[i].op == ST_HC_MIX
            && let Some(j) = producer_of[i]
        {
            let x = *s[i].mix();
            let mut f = s[j];
            f.args.qmm.yout = YOUT_HC_MIX;
            f.args.qmm.mix_xn = x.xn;
            f.args.qmm.mix_z = x.z;
            f.args.qmm.mix_out = x.mixed;
            f.args.qmm.mix_inject = if x.z.is_null() { std::ptr::null_mut() } else { x.inject };
            f.args.qmm.mix_e = x.e;
            f.args.qmm.y = std::ptr::null_mut();
            f.args.qmm.ldy = 0;
            out.push(f);
            continue;
        }
        out.push(s[i]);
    }
    at[n] = out.len() as u32;
    remap(ends, &at);
    *s = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_depends_on_the_matrix_shape() {
        // The narrow matrices of qmmh.h's comment: hc down (5 groups), the shared expert (10), attention k/v (8).
        assert_eq!(qmmh_ksplit(320, 10_240), 16);
        assert_eq!(qmmh_ksplit(640, 2560), 8);
        assert_eq!(qmmh_ksplit(512, 2560), 10);
        // Wide matrices and short rows take no split.
        assert_eq!(qmmh_ksplit(10_240, 320), 1);
        assert_eq!(qmmh_ksplit(64, 256), 1);
        assert_eq!(qmmh_ksplit(2560, 640), 1);
    }

    #[test]
    fn a_knob_takes_0_or_1_and_refuses_the_rest() {
        assert_eq!(switch("MK_FUSE", None), Ok(true));
        assert_eq!(switch("MK_FUSE", Some("1")), Ok(true));
        assert_eq!(switch("MK_FUSE", Some("0")), Ok(false));
        assert_eq!(
            switch("MK_FUSE", Some("yes")),
            Err("MK_FUSE=yes: want 0 or 1".to_owned())
        );
        let knobs = Knobs::from(true, |name| (name == "MK_ELIDE").then(|| "0".to_owned())).unwrap();
        assert!(!knobs.elide && knobs.fuse && knobs.tail && knobs.debug_layers == 0);
        for bad in ["0", "49", "3x"] {
            let error = Knobs::from(true, |name| (name == "MK_DEBUG_LAYERS").then(|| bad.to_owned())).unwrap_err();
            assert!(error.contains("out of 1..48"), "{error}");
        }
        let error = Knobs::from(true, |name| (name == "MK_TAIL").then(|| "2".to_owned())).unwrap_err();
        assert_eq!(error, "MK_TAIL=2: want 0 or 1");
    }

    #[test]
    fn the_rope_table_is_mk_table() {
        let table = inv_freq();
        assert_eq!(table[0], 1.0);
        assert_eq!(table[1], 1.0 / 1e7f32.powf(2.0 / 64.0));
        assert_eq!(table[31], 1.0 / 1e7f32.powf(62.0 / 64.0));
    }
}
