//! Task descriptors and opcodes shared between the Rust host and `kernels/megakernel.cu`.

#![allow(dead_code)]

/// Maximum tasks in one ring, including the trailing `OP_HALT`.
pub const MAX_TASKS_PER_CARD: usize = 2048;

/// Task `i` waits on event `i` and signals event `i + 1`.
pub const NUM_SCOREBOARD_EVENTS: usize = MAX_TASKS_PER_CARD + 1;

/// Device bytes of the task ring copy read by the kernel.
pub const RING_DEV_BYTES: usize = MAX_TASKS_PER_CARD * 64;

/// Words of `u64` written by the profile cubin per task per thread block:
/// `t_start`, `t_after_wait`, and `t_end` from `%globaltimer` (nanoseconds).
pub const PROFILE_WORDS: usize = 3;

/// Event sentinel meaning "do not wait" or "do not signal".
pub const NO_EVENT: u32 = u32::MAX;

/// Inkling 2 MoE constants.
pub const NUM_EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
pub const MAX_PLAN_TOKENS: usize = 1024;

/// Bytes of `MoEPlan`: `count[128]`, `offset[129]`, `n_active`, `active[128]`,
/// then `MAX_PLAN_TOKENS * TOP_K` slots and weights.
pub const MOE_PLAN_BYTES: usize = (NUM_EXPERTS + NUM_EXPERTS + 1 + 1 + NUM_EXPERTS) * 4 + MAX_PLAN_TOKENS * TOP_K * 8;

// Core opcodes (must match `kernels/megakernel.cu`).
pub const OP_NOP: u16 = 0;
pub const OP_RMSNORM: u16 = 1;
pub const OP_MATMUL: u16 = 2;
pub const OP_MOE_PLAN: u16 = 7;
pub const OP_MOE_GATHER: u16 = 20;
pub const OP_COPY: u16 = 21;
pub const OP_EMBED: u16 = 41;
pub const OP_BIAS: u16 = 43;
pub const OP_LAYERNORM: u16 = 47;
pub const OP_GELU: u16 = 48;
pub const OP_SCALE_ADD: u16 = 50;
pub const OP_HALT: u16 = 255;

// Inkling 2 opcodes (matching `docs/engine-abi.md` "Inkling 2").
pub const OP_EXPERT_UP: u16 = 142;
pub const OP_EXPERT_DOWN: u16 = 143;
pub const OP_EXPERT_SUM: u16 = 144;
pub const OP_ATTN_RESIDUAL: u16 = 150;
pub const OP_MESSAGE_GATE: u16 = 151;
pub const OP_HEAD_CLIP: u16 = 152;
pub const OP_OUTPUT_GATE: u16 = 153;
pub const OP_RESIDUAL_CONV: u16 = 154;
pub const OP_SIGMOID_ROUTER: u16 = 155;
pub const OP_INKLING2_ATTN: u16 = 160;
pub const OP_INKLING2_ATTN_SUM: u16 = 161;
pub const OP_INKLING2_VISION_EMBED: u16 = 180;
pub const OP_INKLING2_VISION_ROPE: u16 = 181;
pub const OP_INKLING2_VISION_ATTN: u16 = 182;
pub const OP_INKLING2_CONTROL: u16 = 190;

// Flags.
pub const FLAG_ACCUMULATE: u16 = 1 << 0;
pub const FLAG_SOFTCAP_30: u16 = 1 << 2;
pub const FLAG_F32: u16 = 1 << 8;
pub const FLAG_MXFP8: u16 = 1 << 15;

/// 64-byte task descriptor shared with `kernels/megakernel.cu`.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MkTaskDescriptor {
    pub opcode: u16,
    pub flags: u16,
    pub layer: u32,
    pub wait_event: u32,
    pub signal_event: u32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,
    pub p0: u32,
    pub p1: u32,
    pub p2: u32,
    pub p3: u32,
}

impl Default for MkTaskDescriptor {
    fn default() -> Self {
        Self {
            opcode: OP_NOP,
            flags: 0,
            layer: 0,
            wait_event: NO_EVENT,
            signal_event: NO_EVENT,
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            p0: 0,
            p1: 0,
            p2: 0,
            p3: 0,
        }
    }
}

/// Per-block execution statistics written at the end of a megakernel launch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockStats {
    pub tasks_run: u32,
    pub pad: u32,
    pub wait_ns: u64,
    pub exec_ns: u64,
}
