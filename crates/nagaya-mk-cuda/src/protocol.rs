//! Task descriptors and opcodes shared between the Rust host and `kernels/megakernel.cu`.
//!
//! `build.rs` includes this file and writes `mk_protocol.h` from it: the opcodes, the flags, the sizes and the offset
//! of each field. The kernel includes that header and checks its own structs against it.

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

/// Define `u16` constants and a table of their names and values. `build.rs` writes the table into the header that
/// `kernels/megakernel.cu` includes, so the opcodes and the flags have one source.
macro_rules! define_u16_table {
    ($table:ident; $($(#[$meta:meta])* $name:ident = $value:expr;)*) => {
        $($(#[$meta])* pub const $name: u16 = $value;)*
        /// The name and the value of each constant above.
        pub const $table: &[(&str, u16)] = &[$((stringify!($name), $name)),*];
    };
}

define_u16_table! {
    OPCODES;
    // Core opcodes.
    OP_NOP = 0;
    OP_RMSNORM = 1;
    OP_MATMUL = 2;
    OP_MOE_PLAN = 7;
    OP_MOE_GATHER = 20;
    OP_COPY = 21;
    OP_EMBED = 41;
    OP_BIAS = 43;
    OP_LAYERNORM = 47;
    OP_GELU = 48;
    OP_SCALE_ADD = 50;
    OP_HALT = 255;

    // Inkling 2 opcodes (matching `docs/engine-abi.md` "Inkling 2").
    OP_EXPERT_UP = 142;
    OP_EXPERT_DOWN = 143;
    OP_EXPERT_SUM = 144;
    OP_ATTN_RESIDUAL = 150;
    OP_MESSAGE_GATE = 151;
    OP_HEAD_CLIP = 152;
    OP_OUTPUT_GATE = 153;
    OP_RESIDUAL_CONV = 154;
    OP_SIGMOID_ROUTER = 155;
    OP_INKLING2_ATTN = 160;
    OP_INKLING2_ATTN_SUM = 161;
    OP_INKLING2_VISION_EMBED = 180;
    OP_INKLING2_VISION_ROPE = 181;
    OP_INKLING2_VISION_ATTN = 182;
    OP_INKLING2_CONTROL = 190;
}

define_u16_table! {
    FLAGS;
    FLAG_ACCUMULATE = 1 << 0;
    FLAG_SOFTCAP_30 = 1 << 2;
    FLAG_F32 = 1 << 8;
    FLAG_MXFP8 = 1 << 15;
}

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
