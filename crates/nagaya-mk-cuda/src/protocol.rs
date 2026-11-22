//! Task descriptors and opcodes shared between the Rust host and `kernels/megakernel.cu`.
//!
//! `build.rs` includes this file and writes `mk_protocol.h` from it: the opcodes, the flags, the sizes and the offset
//! of each field. The kernel includes that header and checks its own structs against it.

/// Maximum tasks in one ring, including the trailing `OP_HALT`.
pub const MAX_TASKS_PER_CARD: usize = 2048;

/// Task `i` waits on event `i` and signals event `i + 1`.
pub const NUM_SCOREBOARD_EVENTS: usize = MAX_TASKS_PER_CARD + 1;

/// Device bytes of the task ring copy read by the kernel.
pub const RING_DEV_BYTES: usize = MAX_TASKS_PER_CARD * 128;

/// Words of `u64` written by the profile cubin per task per thread block:
/// `t_start`, `t_after_wait`, and `t_end` from `%globaltimer` (nanoseconds).
pub const PROFILE_WORDS: usize = 3;

/// Threads in a block of the producer shape: warp 0 is the producer, and warps 1 to 8 are the consumers. A block of
/// this width runs the producer shape. A block of another width runs the flat shape, where every thread is a
/// consumer. Only the probes use the flat shape.
pub const PRODUCER_BLOCK_THREADS: u32 = 288;

/// The consumers of a block of the producer shape. They do the work of the items.
pub const CONSUMER_THREADS: u32 = 256;

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

    // Probe opcodes. They measure the GB10 in `tests/cuda_probe.rs`. An engine does not use them. Each has its own
    // comment in `kernels/probe_ops.cuh`.
    // `OP_PROBE_READ`: every block streams items of `PROBE_ITEM_BYTES` from `a`, in the variant `k`.
    // `OP_PROBE_L2`: every block reads its own slice of `n` bytes at `a`, `k` times.
    // `OP_PROBE_TIMER`: block 0 reads `%globaltimer` `n` times, and writes four words to `c`.
    // `OP_PROBE_MMA_TILE`: warp 0 of block 0 runs one MMA tile of the kind `k` and writes its result to `c`.
    // `OP_PROBE_MMA_RATE`: every warp runs `n` MMAs of the kind `k` in each of `PROBE_MMA_CHAINS` chains.
    OP_PROBE_READ = 230;
    OP_PROBE_L2 = 231;
    OP_PROBE_TIMER = 232;
    OP_PROBE_MMA_TILE = 233;
    OP_PROBE_MMA_RATE = 234;
    // `OP_REDUCE_PROBE`: K-split sums. It tests the merge groups of `tests/cuda_merge.rs`. The vector of group `g` has
    // `n` floats at `a + g * n` floats. An item sums `kt` floats of it, and `group_items` items make a group. The
    // item writes its partial to `f`, and the item that arrives last in its group sums the partials in index order
    // and writes the sum to `c[g]`. `f` holds one float for each item, then one `u32` counter for each group. The
    // counters start at 0.
    OP_REDUCE_PROBE = 235;

    // Test opcodes. They gate the scoreboard in `tests/cuda_smoke.rs`. An engine does not use them.
    // `OP_SELFTEST_FILL`: block `n` writes `n` words at `a + n * n` words, after a delay of `k` ns in block `kt`.
    // `OP_SELFTEST_CHECK`: each block reads the words of all blocks and writes its count of wrong words to `d[block]`.
    // Both use the seed `group_items` for the value of each word.
    OP_SELFTEST_FILL = 250;
    OP_SELFTEST_CHECK = 251;
}

define_u16_table! {
    PROBE_READ_VARIANTS;
    // The variants of `OP_PROBE_READ` in `k`. The two prefetch variants read with `PROBE_READ_NC`, and `kt` is the
    // distance of the prefetch in items. `PROBE_READ_SMEM_BULK` copies into shared memory, and `kt` is the count of
    // stages of `PROBE_STAGE_BYTES` each.
    PROBE_READ_PLAIN = 0;
    PROBE_READ_NC = 1;
    PROBE_READ_PREFETCH_LINE = 2;
    PROBE_READ_PREFETCH_BULK = 3;
    PROBE_READ_SMEM_BULK = 4;
}

define_u16_table! {
    PROBE_MMA_KINDS;
    // The kinds of MMA in `k` of `OP_PROBE_MMA_TILE` and `OP_PROBE_MMA_RATE`.
    PROBE_MMA_BF16 = 0;
    PROBE_MMA_NVFP4 = 1;
    PROBE_MMA_MXFP8 = 2;
}

/// Bytes of one item of `OP_PROBE_READ`.
pub const PROBE_ITEM_BYTES: usize = 32 * 1024;

/// Bytes of one stage of shared memory in the variant `PROBE_READ_SMEM_BULK`.
pub const PROBE_STAGE_BYTES: usize = 16 * 1024;

/// Independent accumulators of each warp in `OP_PROBE_MMA_RATE`.
pub const PROBE_MMA_CHAINS: usize = 8;

/// Dynamic shared bytes that `OP_PROBE_READ` needs in the variant `PROBE_READ_SMEM_BULK` with `stages` stages: the
/// stages, then one 8-byte barrier for each stage, rounded to 256 bytes.
pub fn probe_smem_bytes(stages: usize) -> usize {
    (stages * PROBE_STAGE_BYTES + stages * 8).next_multiple_of(256)
}

define_u16_table! {
    FLAGS;
    FLAG_ACCUMULATE = 1 << 0;
    FLAG_SOFTCAP_30 = 1 << 2;
    FLAG_F32 = 1 << 8;
    FLAG_MXFP8 = 1 << 15;
}

/// The task descriptor of `docs/engine-abi.md` "The CUDA interpreter". It is 128 bytes, inline in the ring, and it
/// is shared with `kernels/megakernel.cu`.
///
/// A task waits on up to two pairs of an event and a target before its first work item. Each block signals the event
/// `signal` one time, after its last item of the task. It adds the count of its items. The pointers `a` to `f`, the
/// words `n` to `layer` and the parameters `p0` to `p3` have one meaning for each op. The probe and test ops use `n`, `k`, `kt` and
/// `group_items` as their four integer parameters.
#[repr(C, align(128))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MkTaskDescriptor {
    pub opcode: u16,
    pub flags: u16,
    /// The count of work items. `u32::MAX` means that the count comes from the plan in `d`.
    pub items: u32,
    pub wait0_event: u32,
    pub wait0_target: u32,
    pub wait1_event: u32,
    pub wait1_target: u32,
    pub signal: u32,
    pub rows: u32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,
    pub e: u64,
    pub f: u64,
    pub n: u32,
    pub k: u32,
    pub kt: u32,
    pub group_items: u32,
    pub item_bytes: u32,
    pub scale_bytes: u32,
    pub pos0: u32,
    pub layer: u32,
    pub p0: f32,
    pub p1: f32,
    pub p2: f32,
    pub p3: f32,
}

impl Default for MkTaskDescriptor {
    fn default() -> Self {
        Self {
            opcode: OP_NOP,
            flags: 0,
            items: 0,
            wait0_event: NO_EVENT,
            wait0_target: 0,
            wait1_event: NO_EVENT,
            wait1_target: 0,
            signal: NO_EVENT,
            rows: 0,
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            e: 0,
            f: 0,
            n: 0,
            k: 0,
            kt: 0,
            group_items: 0,
            item_bytes: 0,
            scale_bytes: 0,
            pos0: 0,
            layer: 0,
            p0: 0.0,
            p1: 0.0,
            p2: 0.0,
            p3: 0.0,
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

/// The value that `OP_SELFTEST_FILL` writes for word `index` of block `owner` under `seed`. It is never zero.
pub fn selftest_value(seed: u32, owner: u32, index: u32) -> u32 {
    (seed.wrapping_mul(0x9E37_79B1) ^ owner.wrapping_mul(0x85EB_CA6B) ^ index.wrapping_mul(0xC2B2_AE35)) | 1
}
