//! The Rust mirror of the reference's stage (`hip/src/engine/stage.h`) and of the argument types of its ops.
//!
//! A pass is a list of stages, and a stage is one op with its arguments. The engine builds the list in C++. The Rust
//! schedule (`schedule.rs`) reads it through these `#[repr(C)]` types. `hip/src/engine/stage_layout.def` holds each
//! size and offset. The C++ build and the Rust build check the two sides against it, so a difference stops a build.
//! The constants (op codes, schedule bits, model sizes) come from the same file through `build.rs`.
//!
//! The schedule only compares addresses. It never reads through a pointer of a stage.

// The generated constants and checks. Some constants have no reader in the Rust code yet: step 6 reads them.
#[allow(dead_code)]
mod layout {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/stage_layout.rs"));
}
pub(crate) use layout::*;

/// `QmmArgs` (`ops/qmm.h`): `y[t, n] = W x[t, k]`, a weight row in a GGUF block format.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct QmmArgs {
    pub r#type: u32,
    pub w: *const u8,
    pub ldw: u64,
    pub n: u32,
    pub k: u32,
    pub t: u32,
    pub x: *const f32,
    pub ldx: u32,
    pub x_row: *const u32,
    pub y: *mut f32,
    pub ldy: u32,
    pub y_row: *const u32,
    pub wbulk: u32,
    pub ksplit: u32,
    pub part: *mut f32,
    pub xin: u32,
    pub xks: u32,
    pub xp: *const f32,
    pub xp2: *const f32,
    pub xscale: f32,
    pub yout: u32,
    pub mix_xn: *const f32,
    pub mix_z: *const f32,
    pub mix_out: *mut f32,
    pub mix_inject: *mut f32,
    pub mix_e: u32,
}

/// `RmsArgs` (`ops/small.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RmsArgs {
    pub x: *const f32,
    pub ldx: u32,
    pub y: *mut f32,
    pub ldy: u32,
    pub gamma: *const f32,
    pub t: u32,
    pub n_groups: u32,
    pub d: u32,
    pub gamma_per_group: u32,
    pub eps: f32,
}

/// `SiluScaleArgs` (`ops/small.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SiluScaleArgs {
    pub x: *const f32,
    pub y: *mut f32,
    pub n: u32,
    pub scale: f32,
}

/// `HcMixArgs` (`ops/small.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct HcMixArgs {
    pub xn: *const f32,
    pub up: *const f32,
    pub mixed: *mut f32,
    pub z: *const f32,
    pub inject: *mut f32,
    pub t: u32,
    pub e: u32,
}

/// `HcCombineArgs` (`ops/small.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct HcCombineArgs {
    pub h: *mut f32,
    pub out: *const f32,
    pub inject: *const f32,
    pub t: u32,
    pub e: u32,
}

/// `HcCombineRmsArgs` (`ops/small.h`): a combine and the rms of its `h` as one stage.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct HcCombineRmsArgs {
    pub c: HcCombineArgs,
    pub r: RmsArgs,
}

/// `EmbedArgs` (`engine/engine_ops.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EmbedArgs {
    pub w: *const u8,
    pub ldw: u64,
    pub k: u32,
    pub ids: *const i32,
    pub t: u32,
    pub out: *mut f32,
}

/// `Repeat4Args` (`engine/engine_ops.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Repeat4Args {
    pub e: *const f32,
    pub h: *mut f32,
    pub t: u32,
    pub d: u32,
}

/// `PleGateArgs` (`engine/engine_ops.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PleGateArgs {
    pub kn: *const f32,
    pub qn: *const f32,
    pub pv: *const f32,
    pub gv: *mut f32,
    pub t: u32,
    pub e: u32,
}

/// `PleConvArgs` (`engine/engine_ops.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PleConvArgs {
    pub x: *const f32,
    pub state: *mut f32,
    pub w: *const f32,
    pub y: *mut f32,
    pub t: u32,
    pub c: u32,
}

/// `AddArgs` (`engine/engine_ops.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AddArgs {
    pub h: *mut f32,
    pub a: *const f32,
    pub b: *const f32,
    pub n: u32,
}

/// `GdnConvArgs` (`ops/gdn.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnConvArgs {
    pub x: *const f32,
    pub state_in: *const f32,
    pub w: *const f32,
    pub y: *mut f32,
    pub state_out: *mut f32,
    pub t: u32,
}

/// `GdnCoreArgs` (`ops/gdn.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnCoreArgs {
    pub qkv: *const f32,
    pub z: *const f32,
    pub a: *const f32,
    pub b: *const f32,
    pub ssm_a: *const f32,
    pub dt_bias: *const f32,
    pub gamma: *const f32,
    pub state_in: *const f32,
    pub y: *mut f32,
    pub state_out: *mut f32,
    pub t: u32,
}

/// `MoeRouterArgs` (`ops/moe.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeRouterArgs {
    pub logits: *const f32,
    pub topk_idx: *mut i32,
    pub topk_w: *mut f32,
    pub t: u32,
}

/// `MoeGroupArgs` (`ops/moe.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeGroupArgs {
    pub topk_idx: *const i32,
    pub t: u32,
    pub count: *mut u32,
    pub offset: *mut u32,
    pub assign: *mut u32,
    pub token: *mut u32,
    pub item_gu: *mut u32,
    pub item_dn: *mut u32,
    pub fast: u32,
}

/// `MoeFfnArgs` (`ops/moe.h`): the experts' gate and up, or their down.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeFfnArgs {
    pub type_gate: u32,
    pub type_up: u32,
    pub type_down: u32,
    pub gate: *const u8,
    pub up: *const u8,
    pub down: *const u8,
    pub ld_gate: u64,
    pub ld_up: u64,
    pub ld_down: u64,
    pub x: *const f32,
    pub gu: *mut f32,
    pub h: *mut f32,
    pub y: *mut f32,
    pub count: *const u32,
    pub offset: *const u32,
    pub assign: *const u32,
    pub token: *const u32,
    pub item_gu: *const u32,
    pub item_dn: *const u32,
    pub fast: u32,
}

/// `SwigluArgs` (`ops/moe.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SwigluArgs {
    pub g: *const f32,
    pub u: *const f32,
    pub ldg: u32,
    pub ldu: u32,
    pub out: *mut f32,
    pub ldo: u32,
    pub rows: u32,
    pub cols: u32,
}

/// `MoeCombineArgs` (`ops/moe.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeCombineArgs {
    pub y: *const f32,
    pub topk_idx: *const i32,
    pub topk_w: *const f32,
    pub shared: *const f32,
    pub shared_gate: *const f32,
    pub out: *mut f32,
    pub t: u32,
}

/// `AttnPrepArgs` (`ops/attn.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttnPrepArgs {
    pub q_raw: *const f32,
    pub k_raw: *const f32,
    pub v_raw: *const f32,
    pub iq_raw: *const f32,
    pub ik_raw: *const f32,
    pub q_gamma: *const f32,
    pub k_gamma: *const f32,
    pub idxq_gamma: *const f32,
    pub inv_freq: *const f32,
    pub t: u32,
    pub p0: u32,
    pub q_out: *mut f32,
    pub gate_out: *mut f32,
    pub iq_out: *mut f32,
    pub kv_cache: *mut f32,
    pub ik_cache: *mut f32,
}

/// `AttnSelectArgs` (`ops/attn.h`): the select, score, emit and pool stages.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttnSelectArgs {
    pub iq_out: *const f32,
    pub ik_cache: *const f32,
    pub idxk_gamma: *const f32,
    pub inv_freq: *const f32,
    pub t: u32,
    pub p0: u32,
    pub budget: u32,
    pub ratio: u32,
    pub block_scores: *mut f32,
    pub block_scores_out: *mut f32,
    pub picked: *mut i32,
    pub sel: *mut i32,
    pub sel_count: *mut u32,
    pub max_nb: u32,
    pub max_sel: u32,
    pub block_topk_max: u32,
    pub pooled: *mut f32,
    pub pool_b0: u32,
}

/// `AttnAttendArgs` (`ops/attn.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttnAttendArgs {
    pub q_out: *const f32,
    pub gate_out: *const f32,
    pub kv_cache: *const f32,
    pub sel: *const i32,
    pub sel_count: *const u32,
    pub t: u32,
    pub max_sel: u32,
    pub attn_out: *mut f32,
}

/// The anonymous union of `Stage`. The op of the stage names the member that holds its arguments. The masked select
/// (`AttnSelectMaskedArgs`) has no mirror: no Rust code reads it, and `raw` keeps the size.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union StageArgs {
    pub qmm: QmmArgs,
    pub rms: RmsArgs,
    pub silu: SiluScaleArgs,
    pub mix: HcMixArgs,
    pub comb: HcCombineArgs,
    pub combrms: HcCombineRmsArgs,
    pub embed: EmbedArgs,
    pub rep: Repeat4Args,
    pub pgate: PleGateArgs,
    pub pconv: PleConvArgs,
    pub add: AddArgs,
    pub gconv: GdnConvArgs,
    pub gcore: GdnCoreArgs,
    pub router: MoeRouterArgs,
    pub group: MoeGroupArgs,
    pub ffn: MoeFfnArgs,
    pub swiglu: SwigluArgs,
    pub mcomb: MoeCombineArgs,
    pub aprep: AttnPrepArgs,
    pub asel: AttnSelectArgs,
    pub aatt: AttnAttendArgs,
    pub raw: [u8; 512],
}

/// `Stage` (`engine/stage.h`): one op with its arguments.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Stage {
    pub op: u32,
    /// The layer, for the event ring and for the sibling rule of the schedule.
    pub layer: u32,
    /// For a state stage of a pass of several segments: the segment plus one. Else 0.
    pub sub: u32,
    /// The schedule's word: `SCHED_NO_BARRIER`, `SCHED_TAIL` and the rotation in the low bits.
    pub sched: u32,
    pub args: StageArgs,
}

/// Each accessor reads the union member of one op.
///
/// SAFETY of each accessor: every member of `StageArgs` holds only integers, floats and raw pointers, so each bit
/// pattern is a valid value. The engine fills a new stage with zeros before it writes the member (`engine.hip` `st`),
/// and the Rust constructor `Stage::new` does the same, so the bytes of each member are initialized.
macro_rules! accessor {
    ($name:ident, $ty:ty) => {
        pub(crate) fn $name(&self) -> &$ty {
            // SAFETY: see the comment of the accessors above.
            unsafe { &self.args.$name }
        }
    };
}

impl Stage {
    /// A stage of `op` in `layer`, with every other byte zero, as `engine.hip` `st` makes it.
    #[cfg(test)]
    pub(crate) fn new(op: u32, layer: u32) -> Stage {
        Stage {
            op,
            layer,
            sub: 0,
            sched: 0,
            args: StageArgs { raw: [0; 512] },
        }
    }

    accessor!(qmm, QmmArgs);
    accessor!(rms, RmsArgs);
    accessor!(silu, SiluScaleArgs);
    accessor!(mix, HcMixArgs);
    accessor!(comb, HcCombineArgs);
    accessor!(combrms, HcCombineRmsArgs);
    accessor!(embed, EmbedArgs);
    accessor!(rep, Repeat4Args);
    accessor!(pgate, PleGateArgs);
    accessor!(pconv, PleConvArgs);
    accessor!(add, AddArgs);
    accessor!(gconv, GdnConvArgs);
    accessor!(gcore, GdnCoreArgs);
    accessor!(router, MoeRouterArgs);
    accessor!(group, MoeGroupArgs);
    accessor!(ffn, MoeFfnArgs);
    accessor!(swiglu, SwigluArgs);
    accessor!(mcomb, MoeCombineArgs);
    accessor!(aprep, AttnPrepArgs);
    accessor!(asel, AttnSelectArgs);
    accessor!(aatt, AttnAttendArgs);
}

impl std::fmt::Debug for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stage")
            .field("op", &self.op)
            .field("layer", &self.layer)
            .field("sub", &self.sub)
            .field("sched", &format_args!("{:#x}", self.sched))
            .finish_non_exhaustive()
    }
}
