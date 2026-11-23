//! Task descriptors and opcodes shared between the Rust host and `kernels/megakernel.cu`.
//!
//! `build.rs` includes this file and writes `mk_protocol.h` from it: the opcodes, the flags, the sizes and the offset
//! of each field. The kernel includes that header and checks its own structs against it.

/// Maximum tasks in one ring, including the trailing `OP_HALT`.
pub const MAX_TASKS_PER_CARD: usize = 2048;

/// Task `i` waits on event `i` and signals event `i + 1`.
pub const NUM_SCOREBOARD_EVENTS: usize = MAX_TASKS_PER_CARD + 1;

/// The claim words follow the events in the scoreboard: the word of the task at place `i` of the ring is the word
/// `CLAIM_BASE + i`. The kernel adds 1 to it for each claim of a later item (`run_items`). A launch zeroes the events
/// and the claim words together. The scoreboard has `SCOREBOARD_WORDS` words.
pub const CLAIM_BASE: usize = NUM_SCOREBOARD_EVENTS;

/// Words in the scoreboard: the events, then one claim word for each task of the ring.
pub const SCOREBOARD_WORDS: usize = NUM_SCOREBOARD_EVENTS + MAX_TASKS_PER_CARD;

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
    // The opcodes of one Inkling 2 run on the GB10 are in lanes of ten numbers. A lane owns its numbers, its kernel
    // file and the stub of its ops. A number that no constant names is free for its lane. `run_op` in
    // `kernels/megakernel.cu` traps on an opcode that it does not know, and on an opcode that is reserved and not
    // yet built.
    OP_NOP = 0;
    OP_COPY = 21;

    // The dense lane D, 40 to 49 (`kernels/matmul_bf16.cuh`, `kernels/ops_dense.cuh`).
    // `OP_MATMUL_BF16`: `y[r, n] = sum_k x[r, k] * W[n, k]` for 1 to 512 rows. `a` = `x` (`f32`, `rows` rows of `k`),
    // `b` = `W` repacked by `repack.rs`, `c` = `y` (`f32`, `rows` rows of `n`), `f` = merge counters, then partials
    // (`MatmulShape::scratch_floats`), `n`, `k`, `kt`, `rows`, `items` and `group_items` as `repack.rs` gives them.
    // The head of the model is this op with `FLAG_NORM` and `FLAG_SOFTCAP_30`, so it has no opcode of its own.
    OP_MATMUL_BF16 = 40;
    // `OP_PROJ`: the joined matmul of the q, relative, sink, gate, k and v rows (and the taps of the dynamic
    // convolutions on a full layer), with a norm prologue and an epilogue for each segment. The rows of the matrix are
    // in `src/proj.rs`, and the fields of the descriptor and the slots of the table are in `kernels/ops_dense.cuh`.
    OP_PROJ = 44;
    // `OP_OUT_MIX`: the output matmul, its short convolution, the residual and the mix of the MLP input
    // (`src/outmix.rs`, `kernels/ops_out_mix.cuh`).
    OP_OUT_MIX = 45;

    // The MoE lane M, 50 to 69 (`kernels/quant.cuh`, `kernels/experts.cuh`, `kernels/ops_moe.cuh`). Each op has its
    // own comment in its kernel file. `OP_EXPERT_NVFP4` and `OP_EXPERT_MXFP8` also run the up and the down matmul
    // of the experts, with `FLAG_UP` or `FLAG_DOWN`.
    OP_QUANT_NVFP4 = 51;
    OP_QUANT_MXFP8 = 52;
    OP_EXPERT_NVFP4 = 53;
    OP_EXPERT_MXFP8 = 54;
    // `OP_ROUTE`: the router matmul, the sigmoid, the top 8 and the list of pairs (reserved for lane M).
    OP_ROUTE = 55;

    // The row lane R, 70 to 79 (`kernels/ops_rows.cuh`).
    // `OP_EMBED`: the embedding of the tokens, the first norm and the row factor.
    OP_EMBED = 70;
    // `OP_MIX`: the attention residual and the message mix over the planes.
    OP_MIX = 71;
    // `OP_GATE_DOWN`: the down matmul of the message gates.
    OP_GATE_DOWN = 72;
    // `OP_GATE_UP`: the forget and gain matmuls of the message gates, and their blend.
    OP_GATE_UP = 73;
    // `OP_MIX_OUT`: the output residual over the planes.
    OP_MIX_OUT = 74;

    // The attention lane A, 80 to 89 (`kernels/ops_attn.cuh`).
    // `OP_ATTN`: the attention of one layer over the cache. The fields are in `src/attn.rs` and in
    // `kernels/ops_attn.cuh`.
    OP_ATTN = 80;

    OP_HALT = 255;

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
    // `OP_PROBE_SMEM`: every block fills its dynamic shared memory with a pattern, runs `n` iterations of the variant
    // `k` (`PROBE_SMEM_*`) and writes a `u32` checksum to `c[block]`. For `PROBE_SMEM_MMA_*`, `kt` is the count of
    // k16 steps for each iteration.
    OP_PROBE_SMEM = 237;
    // `OP_REDUCE_PROBE`: K-split sums. It tests the merge groups of `tests/cuda_merge.rs`. The vector of group `g` has
    // `n` floats at `a + g * n` floats. An item sums `kt` floats of it, and `group_items` items make a group. The
    // item writes its partial to `f`, and the item that arrives last in its group sums the partials in index order
    // and writes the sum to `c[g]`. `f` holds one float for each item, then one `u32` counter for each group. The
    // counters start at 0.
    OP_REDUCE_PROBE = 235;
    // `OP_ROWS_PROBE`: the probe of the row finisher (`kernels/ops_rows.cuh`). `rows` rows of `group_items * n` floats
    // are at `a`. Item `i` writes the sum of squares of its `n` floats of each row to a partial in `f`, and the item
    // that arrives last in the group sums the partials in the order of the tiles. It writes the sums and the row
    // factors to `c`, and the normed rows to `e`. `f` holds the counter, the count of finishers, 2 free words, then
    // the partials.
    OP_ROWS_PROBE = 236;

    // Test opcodes. They gate the scoreboard in `tests/cuda_smoke.rs`. An engine does not use them.
    // `OP_SELFTEST_FILL`: block `n` writes `n` words at `a + n * n` words, after a delay of `k` ns in block `kt`.
    // `OP_SELFTEST_CHECK`: each block reads the words of all blocks and writes its count of wrong words to `d[block]`.
    // Both use the seed `group_items` for the value of each word. `OP_SELFTEST_CHECK` writes its counts to `e`.
    // `OP_SELFTEST_TABLE`: item `i` writes the `TABLE_POINTERS` words of its task table to `c + i * TABLE_POINTERS` words.
    OP_SELFTEST_FILL = 250;
    OP_SELFTEST_CHECK = 251;
    OP_SELFTEST_TABLE = 252;
    // `OP_SELFTEST_CLAIM`: item `i` adds 1 to the word `i` of the counts at `c`, and it writes the block that ran it to
    // the word `i` at `e`. An item with `i % k == 0` first spins `n` ns, so that its block is slow. `k = 0` has no spin.
    // An item at or beyond `items` does nothing. It gates the claim of the later items in `tests/cuda_claim.rs`.
    OP_SELFTEST_CLAIM = 253;
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
    // `PROBE_READ_SMEM_L2`: every block reads the same `n` items, `group_items` times, through bulk copies into
    // `kt` stages. A block starts at item `blockIdx.x * 5 % n` when `rows` is 1, and at item 0 when `rows` is 0.
    PROBE_READ_SMEM_L2 = 5;
}

define_u16_table! {
    PROBE_SMEM_VARIANTS;
    // The variants of `OP_PROBE_SMEM` in `k`. Loads and stores of one block: v4 and v2 words, with the lanes in
    // order, so that no bank conflicts. `PROBE_SMEM_MMA_FED` runs the 12 MMAs of one step of the prefill (4
    // chains, three terms) with the fragments loaded from shared memory, and `PROBE_SMEM_MMA_REGS` runs the same MMAs
    // with the fragments in registers.
    PROBE_SMEM_LD_V4 = 0;
    PROBE_SMEM_LD_V2 = 1;
    PROBE_SMEM_ST_V4 = 2;
    PROBE_SMEM_ST_V2 = 3;
    PROBE_SMEM_MMA_FED = 4;
    PROBE_SMEM_MMA_REGS = 5;
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

/// The rows of the weight that one work item of an expert op covers: its column tile.
pub const EXPERT_TILE_ROWS: usize = 64;

/// The most rows that one expert takes in one pair: two `n8` tiles of the MMA.
pub const EXPERT_MAX_ROWS: usize = 16;

/// The values of K in one group of a repacked weight. A group is the unit of the layout and of the loop: 2 MMAs of
/// `m16n8k64` for NVFP4, and 4 MMAs of `m16n8k32` for MXFP8. K is a multiple of this number.
pub const EXPERT_K_GROUP: usize = 128;

/// The bytes of one group record of an NVFP4 weight (16 rows by `EXPERT_K_GROUP` values): 2 tiles of 512 bytes, then
/// the 128 bytes of the scales in the order of the scale fragment.
pub const EXPERT_NVFP4_RECORD_BYTES: usize = 2 * 512 + 128;

/// The bytes of one group record of an MXFP8 weight: 4 tiles of 512 bytes, then 64 bytes of scales.
pub const EXPERT_MXFP8_RECORD_BYTES: usize = 4 * 512 + 64;

/// The dynamic shared bytes that an expert op needs, at offset 0: the partial sums of the second half of K.
pub const EXPERT_SMEM_BYTES: usize = 4 * 2 * 32 * 4 * 4;

/// One pair of an expert op: an expert and the rows that it takes. `x_row[i]` is the row of the quantized rows that
/// the row `i` of the expert reads, `y_row[i]` is the row of the output that it writes, and `weight[i]` is the route
/// weight of the row. Only the first `count` entries count. A pair with `count` 0 is empty, and the op does nothing
/// for it. The device reads the list through `e` of the descriptor (see [`ExpertPairHeader`]). The expert ops do not
/// use `weight` yet.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpertPair {
    pub expert: u32,
    pub count: u32,
    pub x_row: [u16; EXPERT_MAX_ROWS],
    pub y_row: [u16; EXPERT_MAX_ROWS],
    pub weight: [f32; EXPERT_MAX_ROWS],
}

/// The bytes of the header of a pair list.
pub const EXPERT_PAIR_HEADER_BYTES: usize = 16;

/// The header of a pair list, at the address `e` of the descriptor. The pairs follow it: pair `i` is at
/// `e + EXPERT_PAIR_HEADER_BYTES + i * size_of::<ExpertPair>()`.
///
/// `pairs` is the live count of pairs. The count of items of the task is a static bound, because the host does not
/// know the live count when it builds the ring (a router writes the list on the device). Item `i` takes the pair
/// `i / tiles`. A pair at or after `pairs` is not live: the op does nothing for its items, and they count as done for
/// the signal, so the waits on the task stay exact. The list must have room for `items / tiles` pairs. The op traps
/// when `pairs` is larger than that.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertPairHeader {
    pub pairs: u32,
    pub reserved: [u32; 3],
}

/// The static bound of the pairs of an expert op for `rows` rows of tokens, 8 experts for each row. A pair holds at
/// most 16 rows, so the live count is at most `8 * rows` (each pair has one row), and at most `8 * rows / 16` plus one
/// pair for each of the 128 experts (each expert wastes at most one part of a pair). The count is exact at 1 row.
/// The bound of the items is this number times the tiles of one pair.
pub fn expert_pair_bound(rows: usize) -> usize {
    (TOP_K * rows).min(TOP_K * rows / EXPERT_MAX_ROWS + NUM_EXPERTS)
}

/// The keys of one chunk of `OP_ATTN`. A chunk is the unit of the split over the SMs and of the merge in index order.
/// The chunks sit on absolute positions, so the bits of a row do not depend on the block that holds it. The number is
/// a part of the engine identity (design rule 8). It is a multiple of 64, the keys of one step of the kernel.
pub const ATTN_CHUNK_KEYS: usize = 256;

/// The most rows of `OP_ATTN` in the split mode (one item for each chunk, with a merge group). A task with more rows
/// runs in the whole mode (one item for each tile of rows, and it merges its chunks itself). The two modes give the
/// same bits.
pub const ATTN_SPLIT_ROWS: usize = 16;

/// The dynamic shared bytes that `OP_ATTN` needs, at offset 0.
pub const ATTN_SMEM_BYTES: usize = 56 * 1024;

/// The floats of one row of a chunk partial in the scratch of `OP_ATTN`: the maximum, the sum, two spare words, and 128
/// values.
pub const ATTN_PARTIAL_ROW_FLOATS: usize = 132;

/// Pointers in a task table.
pub const TABLE_POINTERS: usize = 16;

/// The task table of a task: 16 device pointers that stay the same for a slot and a run. The descriptor has room for
/// the pointers `a` to `f` only, and a fused op of wave B needs more: planes of the residual, norm weights, the weight
/// and the state of a convolution, biases, and an output for each segment. The host writes the table into device
/// memory at the load. `d` of the descriptor holds its address, or 0 when the task has no table. The address is
/// 128-byte aligned. The producer warp copies the table to shared memory with the descriptor, one task ahead.
///
/// An op gives each slot a meaning, and names the slot in its own kernel file and in this file (for example
/// [`EXPERT_TABLE_AMAX`]). A slot that an op does not use stays 0. Words that are not pointers are allowed too: the
/// kernel reads a slot as a `u64`.
#[repr(C, align(128))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MkTaskTable {
    pub ptr: [u64; TABLE_POINTERS],
}

/// The slot of the pointer to the amax of each expert (`f32`, 128 of them) in the table of `OP_EXPERT_NVFP4`.
pub const EXPERT_TABLE_AMAX: usize = 0;

/// The slots of the table of `OP_ATTN`: the relative states (`f32`, `rows` x 24 x 16), the sinks (`f32`, `rows` x 24),
/// the gate before its sigmoid (`f32`, `rows` x 3,072) and the relative projection (`bf16`, 16 x 1,024).
pub const ATTN_TABLE_REL: usize = 0;
pub const ATTN_TABLE_SINKS: usize = 1;
pub const ATTN_TABLE_GATE: usize = 2;
pub const ATTN_TABLE_PROJ: usize = 3;

/// The numbers of the dense lane (`kernels/matmul_bf16.cuh`, `kernels/ops_dense.cuh`). `build.rs` writes each one into
/// the header as `MK_<name>`.
pub const DENSE_CONSTANTS: &[(&str, usize)] = &[
    // The most rows of one `OP_MATMUL_BF16` task: a block of a prefill.
    ("MATMUL_MAX_ROWS", MATMUL_MAX_ROWS),
    // The slot of the pointer to the norm weight (`bf16`, `k` values) in the table of a matmul with `FLAG_NORM`.
    ("MATMUL_TABLE_NORM_WEIGHT", MATMUL_TABLE_NORM_WEIGHT),
    ("PROJ_TABLE_NORM_Q", PROJ_TABLE_NORM_Q),
    ("PROJ_TABLE_NORM_K", PROJ_TABLE_NORM_K),
    ("PROJ_TABLE_NORM_V", PROJ_TABLE_NORM_V),
    ("PROJ_TABLE_X_K", PROJ_TABLE_X_K),
    ("PROJ_TABLE_X_V", PROJ_TABLE_X_V),
    ("PROJ_TABLE_RF_K", PROJ_TABLE_RF_K),
    ("PROJ_TABLE_RF_V", PROJ_TABLE_RF_V),
    ("PROJ_TABLE_Q_SCALE", PROJ_TABLE_Q_SCALE),
    ("PROJ_TABLE_REL_SCALE", PROJ_TABLE_REL_SCALE),
    ("PROJ_TABLE_K_CONV", PROJ_TABLE_K_CONV),
    ("PROJ_TABLE_V_CONV", PROJ_TABLE_V_CONV),
    ("PROJ_TABLE_K_STATE", PROJ_TABLE_K_STATE),
    ("PROJ_TABLE_V_STATE", PROJ_TABLE_V_STATE),
    ("PROJ_TABLE_K_OUT", PROJ_TABLE_K_OUT),
    ("PROJ_TABLE_V_OUT", PROJ_TABLE_V_OUT),
    ("OUT_MIX_TABLE_STACK_RF", OUT_MIX_TABLE_STACK_RF),
    ("OUT_MIX_TABLE_CONV", OUT_MIX_TABLE_CONV),
    ("OUT_MIX_TABLE_STATE", OUT_MIX_TABLE_STATE),
    ("OUT_MIX_TABLE_NORM", OUT_MIX_TABLE_NORM),
    ("OUT_MIX_TABLE_OUT", OUT_MIX_TABLE_OUT),
    ("OUT_MIX_TABLE_QUERY", OUT_MIX_TABLE_QUERY),
];

/// The most rows of one `OP_MATMUL_BF16` task. The op loops over tiles of 16 rows.
pub const MATMUL_MAX_ROWS: usize = 512;

/// The slot of the table of `OP_MATMUL_BF16` that holds the norm weight.
pub const MATMUL_TABLE_NORM_WEIGHT: usize = 0;

// The slots of the table of `OP_PROJ`. All 15 are pointers of the op, and no slot may be 0. A sliding layer has one input
// row and one norm weight, so it gives the same pointer for q, k and v. The first input is `a` of the descriptor and its
// row factors are `e`.
/// The norm weights (`bf16`, 2304 values) of the q, rel, sink and gate rows, of the k rows and of the v rows.
pub const PROJ_TABLE_NORM_Q: usize = MATMUL_TABLE_NORM_WEIGHT;
pub const PROJ_TABLE_NORM_K: usize = 1;
pub const PROJ_TABLE_NORM_V: usize = 2;
/// The input rows (`f32`) of the k rows and of the v rows, and their row factors (`f32`, one for each row).
pub const PROJ_TABLE_X_K: usize = 3;
pub const PROJ_TABLE_X_V: usize = 4;
pub const PROJ_TABLE_RF_K: usize = 5;
pub const PROJ_TABLE_RF_V: usize = 6;
/// The scale of the query (`bf16`, `[24][128]`) and of the relative state (`bf16`, `[24][16]`).
pub const PROJ_TABLE_Q_SCALE: usize = 7;
pub const PROJ_TABLE_REL_SCALE: usize = 8;
/// The convolution weights (`bf16`): `[1024][4]` on a sliding layer, `[4][1024]` on a full layer.
pub const PROJ_TABLE_K_CONV: usize = 9;
pub const PROJ_TABLE_V_CONV: usize = 10;
/// The conv states (`f32`, `[3][1024]`, the oldest row first). The op reads them and writes them after the block.
pub const PROJ_TABLE_K_STATE: usize = 11;
pub const PROJ_TABLE_V_STATE: usize = 12;
/// The planes of the cache of the key and of the value (`f32` or `e4m3`, the layout of `src/attn.rs`).
pub const PROJ_TABLE_K_OUT: usize = 13;
pub const PROJ_TABLE_V_OUT: usize = 14;

// The slots of the table of `OP_OUT_MIX` (`src/outmix.rs`, `kernels/ops_out_mix.cuh`). Slots 0 to 4 are the stack of the mix
// as `OP_MIX` has it: the planes of the blocks, then `r` at the slot `depth - 1`.
/// The row factors of the rows of the stack: the slots 5 to 9.
pub const OUT_MIX_TABLE_STACK_RF: usize = 5;
/// The weight of the attention convolution (`bf16`, `[2304][4]`) and its state (`f32`, `[3][2304]`, oldest row first).
pub const OUT_MIX_TABLE_CONV: usize = 10;
pub const OUT_MIX_TABLE_STATE: usize = 11;
/// The norm weight of `h_e` (`f32`, 2304; 0 without `FLAG_NORM`).
pub const OUT_MIX_TABLE_NORM: usize = 12;
/// The outputs (`f32`): `e` for each row, `h_e` for each row, then the row factor of each `e`.
pub const OUT_MIX_TABLE_OUT: usize = 13;
/// The query of the mix of the MLP input (`f32`, 2304).
pub const OUT_MIX_TABLE_QUERY: usize = 14;

define_u16_table! {
    FLAGS;
    // Each flag is one bit, and no two flags share a bit. An op reads the flags that it knows, and it traps on a flag
    // that it does not know.
    FLAG_ACCUMULATE = 1 << 0;
    FLAG_SOFTCAP_30 = 1 << 2;
    // A matmul applies a norm to the rows of `x` while it loads them: `w * (v * rf)`, with the row factor `rf` that
    // the task before it wrote. The bit 4 was a norm with no weight. No engine set it, and it is free: an op traps on it.
    FLAG_NORM = 1 << 3;
    // The pair list of the task is in `e`, and the count of its items is a bound (see `expert_pair_bound`).
    FLAG_PAIRS = 1 << 5;
    // An expert op does the up matmul of the experts (`FLAG_UP`) or the down matmul (`FLAG_DOWN`).
    FLAG_UP = 1 << 6;
    FLAG_DOWN = 1 << 7;
    FLAG_F32 = 1 << 8;
    // `OP_PROJ` of a full layer: the dynamic taps of the convolutions, and the log scale of the query.
    FLAG_FULL_LAYER = 1 << 9;
    // `OP_MATMUL_BF16` and `OP_PROJ` in the block form, for a prefill (`kernels/matmul_block.cuh`): an item is up to 128
    // columns and 32 rows with all of K. The bits of each row equal those of the decode form. `items` is the count
    // of groups of tiles times the count of blocks of 32 rows.
    FLAG_BLOCK_FORM = 1 << 10;
    // The cache of the attention holds bf16 values (2 bytes each), rounded to nearest even from the f32 value, with a unit
    // scale. A cache with `FLAG_F32` holds f32 values, and a cache with neither flag holds e4m3 values.
    FLAG_BF16 = 1 << 11;
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
    /// The count of work items. A task whose live count comes from the plan on the device gets a static bound here,
    /// and an item beyond the live count does nothing (see [`ExpertPairHeader`]). `u32::MAX` is not a count: the
    /// kernel traps on it.
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
    /// The address of the task table (`MkTaskTable`), or 0 for a task with no table.
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
