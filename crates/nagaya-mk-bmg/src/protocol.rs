//! Structures shared between the host and `kernels/megakernel.cl`. Field meanings per op are
//! documented in the kernel; keep both sides in step.

#![allow(dead_code)]

/// Task ring capacity per card, including the trailing `OP_HALT`.
pub const MAX_TASKS_PER_CARD: usize = 2048;
/// Task `i` waits on event `i` and signals event `i + 1`.
pub const NUM_SCOREBOARD_EVENTS: usize = MAX_TASKS_PER_CARD + 1;

pub const OP_NOP: u16 = 0;
pub const OP_RMSNORM: u16 = 1;
pub const OP_MATMUL: u16 = 2;
pub const OP_GEGLU: u16 = 3;
pub const OP_ATTN: u16 = 5;
pub const OP_ROUTER: u16 = 6;
pub const OP_MOE_PLAN: u16 = 7;
pub const OP_MOE_GATE_UP: u16 = 8;
pub const OP_MOE_DOWN: u16 = 9;
pub const OP_MEAN_POOL_L2: u16 = 11;
pub const OP_ROUTER_LOGITS: u16 = 12;
pub const OP_ATTN_OUT: u16 = 13;
pub const OP_FFN_OUT: u16 = 14;
pub const OP_ATTN_COMBINE: u16 = 15;
pub const OP_ATTN_SCORES: u16 = 16;
pub const OP_ATTN_SOFTMAX: u16 = 17;
pub const OP_ATTN_PV: u16 = 18;
pub const OP_ATTN_PV_SUM: u16 = 19;
pub const OP_MOE_GATHER: u16 = 20;
pub const OP_COPY: u16 = 21;
pub const OP_HALT: u16 = 255;

// The ops of the speech binary (`kernels/qwen3tts_ops.cl`, `MK_QWEN3TTS`); the Gemma 4 binary does not hold them.
pub const OP_SAMPLE: u16 = 40;
pub const OP_EMBED: u16 = 41;
pub const OP_MASK_ROWS: u16 = 42;
pub const OP_BIAS: u16 = 43;
pub const OP_SILU_MUL: u16 = 44;
pub const OP_CONV1D: u16 = 45;
pub const OP_CONVT1D: u16 = 46;
pub const OP_LAYERNORM: u16 = 47;
pub const OP_GELU: u16 = 48;
pub const OP_SNAKE: u16 = 49;
pub const OP_SCALE_ADD: u16 = 50;
pub const OP_PCM: u16 = 51;

// The ops of the image binary (`kernels/sdxl_ops.cl`, `MK_SDXL`). It also holds the ops of the speech binary.
pub const OP_CONV2D: u16 = 60;
pub const OP_GROUPNORM: u16 = 61;
pub const OP_SDPA: u16 = 62;
pub const OP_UPSAMPLE2X: u16 = 63;
pub const OP_QUICK_GELU: u16 = 64;
pub const OP_GEGLU_HALF: u16 = 65;
pub const OP_COPY_COLS: u16 = 66;
pub const OP_ADD_IMAGE_ROW: u16 = 67;
pub const OP_GEMM: u16 = 68;
pub const OP_ATTN_PREP: u16 = 69;
pub const OP_ATTN_XMX: u16 = 70;

// The ops of the embedding binary (`kernels/embeddinggemma2_ops.cl`, `MK_EMBEDDINGGEMMA2`). It also holds the ops of the
// speech binary, and its `OP_GEGLU` is GeGLU.
pub const OP_BIATTN: u16 = 80;
pub const OP_GELU_MUL: u16 = 81;
pub const OP_SCALE: u16 = 82;

// The ops of the LFM3 binary (`kernels/lfm3_ops.cl`, `MK_LFM3`). It also holds the ops of the speech binary, and its
// `OP_RMSNORM` with `FLAG_ROPE` reads the rotated pairs from `layer`.
pub const OP_SHORT_CONV: u16 = 90;
pub const OP_HEAD_GATE: u16 = 91;
pub const OP_CONTROL: u16 = 92;
pub const OP_ATTN_STREAM: u16 = 93;
pub const OP_ATTN_STREAM_SUM: u16 = 94;

/// The subgroups of one launch of the megakernel: 32 work-groups of 1,024 work items, in subgroups of 16
/// (`level_zero::launch_megakernel`). `OP_ATTN_STREAM` keeps one score buffer for each subgroup.
pub const MEGAKERNEL_SUBGROUPS: usize = 2048;

/// The operation of one control item (engine-abi "Control vectors"). The value is the `p0` of `OP_CONTROL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ControlOp {
    /// `h + a * v`: push the residual toward the vector.
    Add = 1,
    /// `h - a * (h . u) * u`, with `u` the unit vector: remove the part along the vector, or weaken it below 1.
    Ablate = 2,
    /// `h + (a - h . u) * u`: set the part along the vector to `a`.
    Clamp = 3,
}
// The ops of the LFM3 vision tower (`kernels/lfm3_vision_ops.cl`, `MK_LFM3`): opcodes 100 to 109. The LFM3 binary also
// holds the ops of the embedding binary. `OP_VISION_ATTN` gives the bits of `OP_BIATTN` with every key.
pub const OP_VISION_ROPE: u16 = 100;
pub const OP_VISION_CONV: u16 = 101;
pub const OP_VISION_ATTN: u16 = 102;

// The speech ops of the LFM3 binary (`kernels/lfm3_speech_ops.cl`): opcodes 120 to 139.
pub const OP_LFM3_GROUP_ATTN: u16 = 120;
pub const OP_LFM3_POLAR: u16 = 121;
pub const OP_LFM3_IRFFT: u16 = 122;

// The ops of the audio encoder of LFM3-Omni-2B in the LFM3 binary (`kernels/lfm3_audio_ops.cl`): opcodes 110 to 119.
pub const OP_SUBSAMPLE: u16 = 110;

// The ops of the Inkling 2 binary (`kernels/inkling2_ops.cl`, `MK_INKLING2`): opcodes 140 to 169.
// The layer-part ops of the Inkling 2 binary (`kernels/inkling2_layer_ops.cl`): opcodes 150 to 159.
/// The attention residual: the softmax mix of the normed block outputs and the partial row, for 1 to 4 queries.
pub const OP_ATTN_RESIDUAL: u16 = 150;
/// The message gate of a cycle: `(1 - forget) * h + gain * m`.
pub const OP_MESSAGE_GATE: u16 = 151;
/// The L2 clip, or the RMS norm of no weight, of each head, with the log scale of a full layer.
pub const OP_HEAD_CLIP: u16 = 152;
/// The sigmoid gate of the attention output, value by value.
pub const OP_OUTPUT_GATE: u16 = 153;
/// The short convolution of kernel 4 with its residual, with fixed or dynamic taps.
pub const OP_RESIDUAL_CONV: u16 = 154;
/// The router of sigmoid scores with a bias: 8 picks in ascending expert order.
pub const OP_SIGMOID_ROUTER: u16 = 155;
/// The partial sums of the attention of Inkling 2 for each chunk of keys, with the relative bias
/// (`kernels/inkling2_attn_ops.cl`).
pub const OP_INKLING2_ATTN: u16 = 160;
/// The merge of the chunks of `OP_INKLING2_ATTN`, with the sink of each head.
pub const OP_INKLING2_ATTN_SUM: u16 = 161;
/// The `bf16` bits of the repacked NVFP4 planes of an expert: the check of the expansion that the expert ops use.
pub const OP_EXPAND_NVFP4: u16 = 140;
/// The `bf16` bits of the repacked MXFP8 planes of an expert.
pub const OP_EXPAND_MXFP8: u16 = 141;
/// The gate and up rows of the experts of a plan, with the softcapped SwiGLU of Inkling 2.
pub const OP_EXPERT_UP: u16 = 142;
/// The down rows of the experts of a plan, times the weight of each pick, into the slot of each pick.
pub const OP_EXPERT_DOWN: u16 = 143;
/// The sum of the 8 slots of each token, in the order of the picks.
pub const OP_EXPERT_SUM: u16 = 144;
/// `OP_EXPERT_UP` and `OP_EXPERT_DOWN`: the slab holds MXFP8, not NVFP4.
pub const FLAG_MXFP8: u16 = 1 << 15;
// The ops of the vision tower of Inkling 2 (`kernels/inkling2_vision_ops.cl`): opcodes 180 to 189.
/// The class row and the bilinear position embedding of one image.
pub const OP_INKLING2_VISION_EMBED: u16 = 180;
/// The rotary embedding in two axes of the queries and the keys of one image, from a table of the host.
pub const OP_INKLING2_VISION_ROPE: u16 = 181;
/// The attention in both directions over the rows of one image: `OP_VISION_ATTN` under the opcode of this binary.
pub const OP_INKLING2_VISION_ATTN: u16 = 182;

pub const FLAG_ACCUMULATE: u16 = 1 << 0;
pub const FLAG_SOFTCAP_30: u16 = 1 << 2;
pub const FLAG_ROPE_GLOBAL: u16 = 1 << 3;
pub const FLAG_ROPE: u16 = 1 << 5;
// The flags of the speech binary.
/// `OP_RMSNORM` with `FLAG_ROPE`: theta 1e6 over the whole head (Qwen3).
pub const FLAG_ROPE_FULL: u16 = 1 << 6;
/// `OP_SAMPLE`: append the id to the history of the sampler.
pub const FLAG_HISTORY: u16 = 1 << 7;
/// The weights, the table or the bias are F32, not BF16.
pub const FLAG_F32: u16 = 1 << 8;
/// `OP_RMSNORM`: scale and rotate the rows, with no normalization.
pub const FLAG_NO_NORM: u16 = 1 << 9;
// The flags of the image binary.
/// The weights and the bias are F16.
pub const FLAG_F16: u16 = 1 << 10;
/// `OP_GROUPNORM`: apply silu after the norm.
pub const FLAG_SILU: u16 = 1 << 11;
/// `OP_SDPA`: query row `t` attends to the key rows `s <= t`, as the text encoders of CLIP do.
pub const FLAG_CAUSAL: u16 = 1 << 12;
/// `OP_CONV2D` and `OP_GEMM`: the pad is at the bottom and the right only, as in the down sampler of the VAE encoder.
pub const FLAG_PAD_END: u16 = 1 << 13;
/// `OP_MATMUL` and `OP_GEGLU`: one row takes the path of 2 to 8 rows, so a row has the same bits for 1 to 8 rows
/// (engine-abi "LFM3", "Batched passes").
pub const FLAG_TILED: u16 = 1 << 14;

pub const NO_EVENT: u32 = u32::MAX;
pub const NUM_EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
/// Most tokens one block may hold; sized by `MoEPlan` in the kernel.
pub const MAX_PLAN_TOKENS: usize = 1024;

/// Bytes of the kernel's `MoEPlan`: count[128], offset[129], n_active, active[128], then
/// MAX_PLAN_TOKENS * TOP_K slots and weights.
pub const MOE_PLAN_BYTES: usize = (NUM_EXPERTS + NUM_EXPERTS + 1 + 1 + NUM_EXPERTS) * 4 + MAX_PLAN_TOKENS * TOP_K * 8;

/// 64-byte task descriptor in host USM. Before running, each work-group waits for
/// `scoreboard[wait_event]` to reach the work-group count; after, it adds 1 to
/// `scoreboard[signal_event]`. `NO_EVENT` skips either step.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct MkTaskDescriptor {
    pub opcode: u16,
    pub flags: u16,
    pub layer: u32,
    pub wait_event: u32,
    pub signal_event: u32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub p0: u32,
    pub x: u64,
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
            m: 0,
            n: 0,
            k: 0,
            p0: 0,
            x: 0,
        }
    }
}

const _: () = assert!(std::mem::size_of::<MkTaskDescriptor>() == 64);

/// One token's routing, written by `OP_ROUTER`.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct MoERouteDecision {
    pub top_k_indices: [u32; 8],
    /// Renormalized top-8 router probabilities times `per_expert_scale`
    pub top_k_weights: [f32; 8],
    pub router_entropy: f32,
    pub router_lse: f32,
    pub layer_idx: u32,
    pub token: u32,
    pub reserved: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<MoERouteDecision>() == 128);

/// Per-Xe2-core execution statistics written at the end of each launch
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct XeCoreStats {
    pub core_id: u32,
    pub tasks_executed: u32,
    pub spin_polls: u32,
    pub last_opcode: u32,
    pub xmx_tiles_executed: u32,
    pub reserved: [u32; 11],
}

const _: () = assert!(std::mem::size_of::<XeCoreStats>() == 64);
