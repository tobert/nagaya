//! The decoder of LiquidCodec on the card (milestone 24 step 8): its weights, its buffers, the state of a slot, the
//! ring of one frame, and the overlap-add on the host.
//!
//! The ring follows `LiquidCodecModel.decode` (modeling_liquid_codec.py:293-357) for one frame of 12 codes. The host
//! turns the codes into the 48 latents of the quantizer. `project_out` and `from_quantizer` give one row of 512
//! values, and the row repeats 6 times, one for each STFT frame (`nearest-exact`). An LFM3 stack of 8 layers with
//! sandwich norms runs on the 6 rows, with sliding attention of 30 positions on layers 2, 4 and 6
//! (configuration_liquid_codec.py:110-130). `lin` gives 641 log magnitudes and 641 angles for each row, and the card
//! makes the inverse FFT of each row times the window. The host adds the frames and divides by the envelope.

use nagaya_mk_bmg::protocol::{FLAG_ACCUMULATE, MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::safetensors::{Dtype, SafeTensorsView};
use nagaya_mk_bmg::tasks;

use crate::codes::CODEBOOKS;
use crate::spec::SLOT_TOKENS;
use crate::weights::{Upload, f32_bytes, rounded};

/// The file of the codec in the folder of Omni.
pub const WEIGHTS_FILE: &str = "audio_tokenizer/model.safetensors";

pub const HIDDEN: usize = 512;
pub const HEADS: usize = 8;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const INTERMEDIATE: usize = 2304;
pub const TAPS: usize = 7;
pub const EPS: f32 = 1e-5;
/// A sliding layer reads the keys `q - 29` to `q` (`decoder_window`).
pub const WINDOW: usize = 30;
/// The layers of the decoder: attention on 2, 4 and 6, and a gated short convolution on the others.
pub const ATTENTION: [bool; 8] = [false, false, true, false, true, false, true, false];

/// The FSQ levels of each codebook (`fsq_levels`), and the latents of a frame: 4 for each codebook.
pub const LEVELS: [u32; 4] = [8, 7, 6, 6];
pub const LATENTS: usize = CODEBOOKS * LEVELS.len();
/// The width of the quantizer (`quantizer_dim`).
pub const QUANTIZER: usize = 256;

/// The STFT frames of one codec frame (`stft_frame_stack`), the FFT size, the hop, and the bins of a frame.
pub const STACK: usize = 6;
pub const N_FFT: usize = 1280;
pub const HOP: usize = 320;
pub const BINS: usize = N_FFT / 2 + 1;
/// The samples of one frame at 24 kHz.
pub const FRAME_SAMPLES: usize = STACK * HOP;
/// The samples that the reference cuts from each end of the overlap-add (modeling_liquid_codec.py:324).
pub const TRIM: usize = (N_FFT - HOP) / 2;

/// The rows of the gate of the heads on the card: 8, padded with zero rows to 16, because a matmul of several rows
/// takes rows in 16s.
pub const GATE_ROWS: usize = 16;
/// The rows of `lin` on the card: 1,282, padded with zero rows to 1,296 for the same reason.
pub const LIN_ROWS: usize = 1296;

/// The rows of the ring of each attention layer of a slot. It holds a frame and the window before it, in rows of 32.
pub const RING: usize = 128;
/// The rows past the ring that a frame may write before `OP_COPY` moves them: the kernel's `KV_RING_SPILL`.
pub const SPILL: usize = 1024;
/// The positions of the codec in a slot: 6 for each position of the slot. The bound of the score buffer.
pub const MAX_POSITIONS: usize = STACK * SLOT_TOKENS;

const _: () = assert!(RING.is_multiple_of(32) && WINDOW + STACK + 32 <= RING && STACK <= SPILL);

/// The weights of one layer of the decoder on the card. A field of the other operator is 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct Layer {
    pub operator_norm: u64,
    pub post_operator_norm: u64,
    pub ffn_norm: u64,
    pub post_ffn_norm: u64,
    pub w13: u64,
    pub w2: u64,
    pub in_proj: u64,
    pub conv: u64,
    pub out_proj: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    /// `[GATE_ROWS][512]`: the 8 rows of `gate_proj`, then zeros.
    pub gate: u64,
}

/// The weights of the decoder on the card.
#[derive(Clone, Debug)]
pub struct Weights {
    pub project_out: u64,
    pub project_out_bias: u64,
    pub from_quantizer: u64,
    pub from_quantizer_bias: u64,
    pub layers: Vec<Layer>,
    pub norm: u64,
    /// `[LIN_ROWS][512]` and its bias of `LIN_ROWS`, zeros past 1,282.
    pub lin: u64,
    pub lin_bias: u64,
    /// The synthesis window of the checkpoint, F32 `[1280]`.
    pub window: u64,
    /// The table of the inverse FFT, F32 `[1280][2]` (`tasks::irfft_table`).
    pub table: u64,
}

/// The BF16 tensors of the decoder with their shapes. The window is F32 (`WINDOW_TENSOR`).
pub fn tensors() -> Vec<(String, Vec<usize>)> {
    let (h, i) = (HIDDEN, INTERMEDIATE);
    let mut all = vec![
        ("quantizer.project_out.weight".to_owned(), vec![QUANTIZER, LATENTS]),
        ("quantizer.project_out.bias".to_owned(), vec![QUANTIZER]),
        ("from_quantizer.weight".to_owned(), vec![h, QUANTIZER]),
        ("from_quantizer.bias".to_owned(), vec![h]),
        ("decoder.lfm.embedding_norm.weight".to_owned(), vec![h]),
        ("decoder.lin.weight".to_owned(), vec![2 * BINS, h]),
        ("decoder.lin.bias".to_owned(), vec![2 * BINS]),
    ];
    for (l, attention) in ATTENTION.iter().enumerate() {
        let mut layer = vec![
            ("operator_norm.weight", vec![h]),
            ("post_operator_norm.weight", vec![h]),
            ("ffn_norm.weight", vec![h]),
            ("post_ffn_norm.weight", vec![h]),
            ("feed_forward.w1.weight", vec![i, h]),
            ("feed_forward.w3.weight", vec![i, h]),
            ("feed_forward.w2.weight", vec![h, i]),
        ];
        if *attention {
            layer.extend([
                ("self_attn.q_proj.weight", vec![HEADS * HEAD_DIM, h]),
                ("self_attn.k_proj.weight", vec![KV_HEADS * HEAD_DIM, h]),
                ("self_attn.v_proj.weight", vec![KV_HEADS * HEAD_DIM, h]),
                ("self_attn.o_proj.weight", vec![h, HEADS * HEAD_DIM]),
                ("self_attn.q_norm.weight", vec![HEAD_DIM]),
                ("self_attn.k_norm.weight", vec![HEAD_DIM]),
                ("self_attn.gate_proj.weight", vec![HEADS, h]),
            ]);
        } else {
            layer.extend([
                ("conv.in_proj.weight", vec![3 * h, h]),
                ("conv.conv.weight", vec![h, 1, TAPS]),
                ("conv.out_proj.weight", vec![h, h]),
            ]);
        }
        all.extend(
            layer
                .into_iter()
                .map(|(n, s)| (format!("decoder.lfm.layers.{l}.{n}"), s)),
        );
    }
    all
}

/// The synthesis window of the checkpoint.
pub const WINDOW_TENSOR: &str = "decoder.istft.window";

/// Check that `view` holds each tensor of the decoder with its dtype and shape. The message names the tensor.
pub fn check(view: &SafeTensorsView<'_>) -> Result<(), String> {
    let mut want: Vec<(String, Dtype, Vec<usize>)> = tensors().into_iter().map(|(n, s)| (n, Dtype::BF16, s)).collect();
    want.push((WINDOW_TENSOR.to_owned(), Dtype::F32, vec![N_FFT]));
    for (name, dtype, shape) in want {
        let meta = view
            .tensors
            .get(&name)
            .ok_or_else(|| format!("{WEIGHTS_FILE} has no tensor {name}"))?;
        if meta.dtype != dtype || meta.shape != shape {
            return Err(format!(
                "tensor {name} of {WEIGHTS_FILE} is {:?} {:?}, and the codec needs {dtype:?} {shape:?}",
                meta.dtype, meta.shape
            ));
        }
    }
    Ok(())
}

/// Load the decoder of `view` into `sink`. A file that fails `check` uploads nothing.
pub fn load(sink: &mut dyn Upload, view: &SafeTensorsView<'_>) -> Result<Weights, String> {
    check(view)?;
    let bytes = |name: &str| -> &[u8] { view.tensor_bytes(name).expect("check found each tensor") };
    // Each allocation of `size` bytes or more gets the tensors one after the other, and zeros after them.
    let mut put = |names: &[String], size: usize| -> u64 {
        let parts: Vec<&[u8]> = names.iter().map(|n| bytes(n)).collect();
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let at = sink.alloc(total.max(size));
        if sink.writes() {
            let mut offset = 0u64;
            for part in parts {
                sink.write(at + offset, part);
                offset += part.len() as u64;
            }
        }
        at
    };
    let one = |name: &str| vec![name.to_owned()];
    let project_out = put(&one("quantizer.project_out.weight"), 0);
    let project_out_bias = put(&one("quantizer.project_out.bias"), 0);
    let from_quantizer = put(&one("from_quantizer.weight"), 0);
    let from_quantizer_bias = put(&one("from_quantizer.bias"), 0);
    let mut layers = Vec::with_capacity(ATTENTION.len());
    for (l, attention) in ATTENTION.iter().enumerate() {
        let n = |name: &str| vec![format!("decoder.lfm.layers.{l}.{name}")];
        let mut layer = Layer {
            operator_norm: put(&n("operator_norm.weight"), 0),
            post_operator_norm: put(&n("post_operator_norm.weight"), 0),
            ffn_norm: put(&n("ffn_norm.weight"), 0),
            post_ffn_norm: put(&n("post_ffn_norm.weight"), 0),
            w13: put(
                &[
                    format!("decoder.lfm.layers.{l}.feed_forward.w1.weight"),
                    format!("decoder.lfm.layers.{l}.feed_forward.w3.weight"),
                ],
                0,
            ),
            w2: put(&n("feed_forward.w2.weight"), 0),
            ..Layer::default()
        };
        if *attention {
            layer.q = put(&n("self_attn.q_proj.weight"), 0);
            layer.k = put(&n("self_attn.k_proj.weight"), 0);
            layer.v = put(&n("self_attn.v_proj.weight"), 0);
            layer.o = put(&n("self_attn.o_proj.weight"), 0);
            layer.q_norm = put(&n("self_attn.q_norm.weight"), 0);
            layer.k_norm = put(&n("self_attn.k_norm.weight"), 0);
            layer.gate = put(&n("self_attn.gate_proj.weight"), GATE_ROWS * HIDDEN * 2);
        } else {
            layer.in_proj = put(&n("conv.in_proj.weight"), 0);
            layer.conv = put(&n("conv.conv.weight"), 0);
            layer.out_proj = put(&n("conv.out_proj.weight"), 0);
        }
        layers.push(layer);
    }
    let norm = put(&one("decoder.lfm.embedding_norm.weight"), 0);
    let lin = put(&one("decoder.lin.weight"), LIN_ROWS * HIDDEN * 2);
    let lin_bias = put(&one("decoder.lin.bias"), LIN_ROWS * 2);
    let window = put(&one(WINDOW_TENSOR), 0);
    let table_values = tasks::irfft_table(N_FFT);
    let table = sink.alloc(table_values.len() * 4);
    if sink.writes() {
        sink.write(table, &f32_bytes(&table_values));
    }
    Ok(Weights {
        project_out,
        project_out_bias,
        from_quantizer,
        from_quantizer_bias,
        layers,
        norm,
        lin,
        lin_bias,
        window,
        table,
    })
}

/// The buffers of the ring of one frame. Each row of 512 values is one STFT frame.
#[derive(Clone, Copy, Debug)]
pub struct Bufs {
    /// The 48 latents of the frame, which the host writes.
    pub latents: u64,
    /// The row after `project_out`: `[256]`.
    pub quantized: u64,
    /// The row after `from_quantizer`: `[512]`.
    pub embedding: u64,
    pub x: u64,
    pub h: u64,
    pub bcx: u64,
    pub y: u64,
    pub q: u64,
    pub gates: u64,
    pub o: u64,
    pub g: u64,
    /// The scores of the attention, for keys up to `MAX_POSITIONS`.
    pub scores: u64,
    /// The rows of `lin`: `[6][LIN_ROWS]`.
    pub spec: u64,
    /// The bins of each row: `[6][2 * 641]`, the real parts and then the imaginary parts.
    pub bins: u64,
    /// The frames of the inverse FFT times the window: `[6][1280]`.
    pub frames: u64,
}

impl Bufs {
    /// The buffers, each from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64) -> Self {
        let rows = |values: usize| STACK * values * 4;
        Self {
            latents: z(LATENTS * 4),
            quantized: z(QUANTIZER * 4),
            embedding: z(HIDDEN * 4),
            x: z(rows(HIDDEN)),
            h: z(rows(HIDDEN)),
            bcx: z(rows(3 * HIDDEN)),
            y: z(rows(HIDDEN)),
            q: z(rows(HEADS * HEAD_DIM)),
            gates: z(rows(GATE_ROWS)),
            o: z(rows(HIDDEN)),
            g: z(rows(INTERMEDIATE)),
            scores: z(tasks::score_floats(HEADS as u32, STACK as u32, (MAX_POSITIONS - STACK) as u32) * 4),
            spec: z(rows(LIN_ROWS)),
            bins: z(rows(2 * BINS)),
            frames: z(rows(N_FFT)),
        }
    }

    /// The device bytes of the buffers, with no card.
    pub fn bytes() -> usize {
        let mut total = 0usize;
        Self::new(&mut |bytes| {
            total += rounded(bytes);
            0
        });
        total
    }
}

/// The device memory of the codec state of one slot: the conv inputs `[6][512]` of each conv layer, and the key
/// plane of each attention layer with its value plane after it, `RING + SPILL` rows each.
#[derive(Clone, Debug)]
pub struct Slot {
    pub layers: Vec<u64>,
}

impl Slot {
    /// The memory of one slot, from `z`, which gives zeroed device memory of a byte count.
    pub fn new(z: &mut dyn FnMut(usize) -> u64) -> Self {
        Self {
            layers: ATTENTION
                .iter()
                .map(|attention| {
                    if *attention {
                        z(2 * (RING + SPILL) * KV_HEADS * HEAD_DIM * 4)
                    } else {
                        z((TAPS - 1) * HIDDEN * 4)
                    }
                })
                .collect(),
        }
    }

    /// The device bytes of one slot, with no card.
    pub fn bytes() -> usize {
        let mut total = 0usize;
        Self::new(&mut |bytes| {
            total += rounded(bytes);
            0
        });
        total
    }

    /// The conv layers, by index, with the bytes of each state.
    pub fn conv_states(&self) -> impl Iterator<Item = (u64, usize)> + '_ {
        ATTENTION
            .iter()
            .zip(&self.layers)
            .filter(|(attention, _)| !**attention)
            .map(|(_, at)| (*at, (TAPS - 1) * HIDDEN * 4))
    }

    /// The address of the key row (`value` false) or the value row of codec position `pos` of attention layer `l`.
    pub fn row(&self, l: usize, value: bool, pos: usize) -> u64 {
        assert!(ATTENTION[l], "layer {l} of the codec is a conv layer");
        let plane = if value { RING + SPILL } else { 0 };
        self.layers[l] + ((plane + pos % RING) * KV_HEADS * HEAD_DIM * 4) as u64
    }
}

/// The 48 latents of one frame: 4 for each codebook, `((code / base) % level - level / 2) / (level / 2)`
/// (modeling_liquid_codec.py:182-189).
pub fn latents(codes: &[u32]) -> Vec<f32> {
    assert_eq!(codes.len(), CODEBOOKS, "a frame has 12 codes");
    let mut out = Vec::with_capacity(LATENTS);
    for code in codes {
        assert!(
            *code < 2016,
            "code {code} is not a code of the codec; the end code makes no audio"
        );
        let mut base = 1u32;
        for level in LEVELS {
            let half = (level / 2) as i32;
            out.push((((code / base) % level) as i32 - half) as f32 / half as f32);
            base *= level;
        }
    }
    out
}

fn accumulate(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    task.flags |= FLAG_ACCUMULATE;
    task
}

/// The tasks of one frame at codec position `pos0` of `slot`. The latents are in `bufs.latents`. After the ring,
/// `bufs.frames` holds the 6 frames of the inverse FFT times the window.
pub fn frame(w: &Weights, bufs: &Bufs, slot: &Slot, pos0: usize) -> Vec<MkTaskDescriptor> {
    assert!(
        pos0.is_multiple_of(STACK) && pos0 + STACK <= MAX_POSITIONS,
        "codec position {pos0} is not the start of a frame below {MAX_POSITIONS}"
    );
    let n = STACK as u32;
    let (h, i) = (HIDDEN as u32, INTERMEDIATE as u32);
    let (heads, kv, d) = (HEADS as u32, KV_HEADS as u32, HEAD_DIM as u32);
    let mut t = vec![
        tasks::matmul(
            bufs.latents,
            w.project_out,
            bufs.quantized,
            1,
            QUANTIZER as u32,
            LATENTS as u32,
            false,
        ),
        tasks::bias(w.project_out_bias, bufs.quantized, 1, QUANTIZER as u32, false),
        tasks::matmul(
            bufs.quantized,
            w.from_quantizer,
            bufs.embedding,
            1,
            h,
            QUANTIZER as u32,
            false,
        ),
        tasks::bias(w.from_quantizer_bias, bufs.embedding, 1, h, false),
    ];
    for r in 0..STACK {
        t.push(tasks::copy(bufs.embedding, bufs.x + (r * HIDDEN * 4) as u64, h));
    }
    for (l, lw) in w.layers.iter().enumerate() {
        t.push(tasks::rmsnorm(bufs.x, lw.operator_norm, bufs.h, n, h, EPS, false));
        if ATTENTION[l] {
            let (kr, vr) = (slot.row(l, false, pos0), slot.row(l, true, pos0));
            t.push(tasks::matmul(bufs.h, lw.q, bufs.q, n, heads * d, h, false));
            t.push(tasks::matmul(bufs.h, lw.k, kr, n, kv * d, h, false));
            t.push(tasks::matmul(bufs.h, lw.v, vr, n, kv * d, h, false));
            t.push(tasks::matmul(
                bufs.h,
                lw.gate,
                bufs.gates,
                n,
                GATE_ROWS as u32,
                h,
                false,
            ));
            // The queries carry the scale head_dim^-0.5, because the attention ops do not scale.
            let scale = (HEAD_DIM as f32).powf(-0.5);
            t.push(tasks::qk_rope(
                bufs.q,
                lw.q_norm,
                n * heads,
                heads,
                d,
                pos0 as u32,
                true,
                true,
                EPS,
                scale,
            ));
            t.push(tasks::qk_rope(
                kr,
                lw.k_norm,
                n * kv,
                kv,
                d,
                pos0 as u32,
                true,
                true,
                EPS,
                1.0,
            ));
            if pos0 % RING + STACK > RING {
                // The rows past the end of the ring move to its start, bit for bit, before attention reads them.
                let spill = ((pos0 % RING + STACK - RING) * KV_HEADS * HEAD_DIM) as u32;
                for value in [false, true] {
                    let start = slot.row(l, value, 0);
                    t.push(tasks::copy(
                        start + (RING * KV_HEADS * HEAD_DIM * 4) as u64,
                        start,
                        spill,
                    ));
                }
            }
            let mut attend = tasks::attend_block(
                bufs.q,
                slot.layers[l],
                bufs.scores,
                bufs.y,
                n,
                heads,
                kv,
                d,
                pos0 as u32,
                MAX_POSITIONS as u32,
                WINDOW as u32,
            );
            for task in &mut attend {
                task.layer = RING as u32;
            }
            t.extend(attend);
            t.push(tasks::head_gate(
                bufs.y,
                bufs.gates,
                bufs.y,
                n,
                heads,
                d,
                GATE_ROWS as u32,
            ));
            t.push(tasks::matmul(bufs.y, lw.o, bufs.o, n, h, heads * d, false));
        } else {
            t.push(tasks::matmul(bufs.h, lw.in_proj, bufs.bcx, n, 3 * h, h, false));
            t.push(tasks::short_conv(
                bufs.bcx,
                lw.conv,
                slot.layers[l],
                bufs.y,
                n,
                h,
                TAPS as u32,
                false,
            ));
            t.push(tasks::matmul(bufs.y, lw.out_proj, bufs.o, n, h, h, false));
        }
        t.push(accumulate(tasks::rmsnorm(
            bufs.o,
            lw.post_operator_norm,
            bufs.x,
            n,
            h,
            EPS,
            false,
        )));
        t.push(tasks::rmsnorm(bufs.x, lw.ffn_norm, bufs.h, n, h, EPS, false));
        t.push(tasks::swiglu(bufs.h, lw.w13, bufs.g, n, i, h));
        t.push(tasks::matmul(bufs.g, lw.w2, bufs.o, n, h, i, false));
        t.push(accumulate(tasks::rmsnorm(
            bufs.o,
            lw.post_ffn_norm,
            bufs.x,
            n,
            h,
            EPS,
            false,
        )));
    }
    t.push(tasks::rmsnorm(bufs.x, w.norm, bufs.h, n, h, EPS, false));
    t.push(tasks::matmul(bufs.h, w.lin, bufs.spec, n, LIN_ROWS as u32, h, false));
    t.push(tasks::bias(w.lin_bias, bufs.spec, n, LIN_ROWS as u32, false));
    t.push(tasks::polar(bufs.spec, bufs.bins, n, BINS as u32, LIN_ROWS as u32));
    t.push(tasks::irfft(bufs.bins, w.window, w.table, bufs.frames, n, N_FFT as u32));
    assert!(t.len() < MAX_TASKS_PER_CARD, "the frame has {} tasks", t.len());
    t
}

/// The overlap-add of the frames of the inverse FFT for one slot, with the stream state of the reference
/// (modeling_liquid_codec.py:324-357). The samples that a later frame still adds to wait for it.
#[derive(Clone, Debug, Default)]
pub struct Overlap {
    /// The sums of the last `2 * TRIM` samples, and of their envelope.
    tail: Option<(Vec<f32>, Vec<f32>)>,
    started: bool,
}

impl Overlap {
    /// The square of the window at each sample of the overlap-add of `count` frames.
    fn envelope(window: &[f32], count: usize) -> Vec<f32> {
        let mut e = vec![0.0f32; (count - 1) * HOP + N_FFT];
        for f in 0..count {
            for (i, w) in window.iter().enumerate() {
                e[f * HOP + i] += w * w;
            }
        }
        e
    }

    /// Add the 6 frames of one codec frame (each `N_FFT` samples, after the window), and give the samples that are
    /// whole: 1,440 for the first frame after a clear, and 1,920 after it.
    pub fn add(&mut self, frames: &[f32], window: &[f32]) -> Vec<f32> {
        assert_eq!(
            frames.len(),
            STACK * N_FFT,
            "a codec frame gives 6 frames of the inverse FFT"
        );
        assert_eq!(window.len(), N_FFT, "the window has 1,280 samples");
        let mut audio = vec![0.0f32; (STACK - 1) * HOP + N_FFT];
        for f in 0..STACK {
            for i in 0..N_FFT {
                audio[f * HOP + i] += frames[f * N_FFT + i];
            }
        }
        let mut envelope = Self::envelope(window, STACK);
        if let Some((a, e)) = self.tail.take() {
            for i in 0..2 * TRIM {
                audio[i] += a[i];
                envelope[i] += e[i];
            }
        }
        let start = if self.started { 0 } else { TRIM };
        self.started = true;
        let keep = audio.len() - 2 * TRIM;
        self.tail = Some((audio[keep..].to_vec(), envelope[keep..].to_vec()));
        (start..keep).map(|i| audio[i] / envelope[i]).collect()
    }

    /// The last samples of the stream: 480 after a frame, or none when no frame came. The state starts again.
    pub fn finish(&mut self) -> Vec<f32> {
        let out = match self.tail.take() {
            Some((a, e)) => (0..TRIM).map(|i| a[i] / e[i]).collect(),
            None => Vec::new(),
        };
        *self = Self::default();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nagaya_mk_bmg::protocol::{OP_ATTN_SCORES, OP_COPY, OP_HEAD_GATE, OP_LFM3_IRFFT, OP_LFM3_POLAR, OP_SHORT_CONV};

    fn weights() -> Weights {
        Weights {
            project_out: 0x1000,
            project_out_bias: 0x2000,
            from_quantizer: 0x3000,
            from_quantizer_bias: 0x4000,
            layers: vec![Layer::default(); 8],
            norm: 0x5000,
            lin: 0x6000,
            lin_bias: 0x7000,
            window: 0x8000,
            table: 0x9000,
        }
    }

    fn alloc() -> impl FnMut(usize) -> u64 {
        let mut next = 1u64 << 40;
        move |bytes| {
            let at = next;
            next += rounded(bytes) as u64;
            at
        }
    }

    #[test]
    fn a_frame_runs_five_conv_layers_three_sliding_layers_and_the_inverse_fft() {
        let (w, b, s) = (weights(), Bufs::new(&mut alloc()), Slot::new(&mut alloc()));
        let t = frame(&w, &b, &s, 6);
        let count = |op: u16| t.iter().filter(|x| x.opcode == op).count();
        assert_eq!(count(OP_SHORT_CONV), 5);
        assert_eq!(count(OP_ATTN_SCORES), 3);
        assert_eq!(
            count(OP_COPY),
            6,
            "the row repeats 6 times, and no ring wraps at position 6"
        );
        let scores: Vec<&MkTaskDescriptor> = t.iter().filter(|x| x.opcode == OP_ATTN_SCORES).collect();
        assert!(
            scores
                .iter()
                .all(|x| x.layer == 128 && x.x == (MAX_POSITIONS as u64) << 32 | 30 && x.p0 == 6)
        );
        let gates: Vec<&MkTaskDescriptor> = t.iter().filter(|x| x.opcode == OP_HEAD_GATE).collect();
        assert!(
            gates.iter().all(|x| x.p0 == 16 && x.m == 8),
            "the gate rows are padded to 16"
        );
        let last = &t[t.len() - 2..];
        assert_eq!((last[0].opcode, last[0].k, last[0].m), (OP_LFM3_POLAR, 641, 1296));
        assert_eq!(
            (last[1].opcode, last[1].c, last[1].k, last[1].p0),
            (OP_LFM3_IRFFT, b.frames, 1280, 641)
        );
    }

    #[test]
    fn a_frame_that_crosses_the_end_of_the_ring_moves_its_spill() {
        let (w, b, s) = (weights(), Bufs::new(&mut alloc()), Slot::new(&mut alloc()));
        // Positions 126 to 131 cross the ring of 128 rows: 4 rows of 128 values spill in each plane of 3 layers.
        let t = frame(&w, &b, &s, 126);
        let spills: Vec<&MkTaskDescriptor> = t.iter().filter(|x| x.opcode == OP_COPY && x.a != b.embedding).collect();
        assert_eq!(spills.len(), 6);
        assert!(spills.iter().all(|x| x.n == 4 * 128));
        assert_eq!(s.row(2, false, 130), s.layers[2] + 2 * 128 * 4);
        assert_eq!(s.row(2, true, 0), s.layers[2] + (128 + 1024) * 128 * 4);
    }

    #[test]
    fn the_latents_are_the_digits_of_each_code() {
        let mut codes = vec![0u32; 12];
        codes[1] = 2015;
        codes[2] = 1 + 8 * 2 + 56 * 3 + 336 * 4;
        let l = latents(&codes);
        assert_eq!(&l[..4], &[-1.0, -1.0, -1.0, -1.0]);
        // 2015 = 7 + 8 * (6 + 7 * (5 + 6 * 5)): the top digit of each level.
        assert_eq!(&l[4..8], &[3.0 / 4.0, 3.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]);
        assert_eq!(&l[8..12], &[-3.0 / 4.0, -1.0 / 3.0, 0.0, 1.0 / 3.0]);
    }

    fn hann() -> Vec<f32> {
        (0..N_FFT)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32)
            .collect()
    }

    #[test]
    fn the_stream_gives_1440_then_1920_samples_and_480_at_the_end() {
        let window = hann();
        let mut overlap = Overlap::default();
        let frames: Vec<f32> = (0..STACK * N_FFT).map(|i| ((i % 97) as f32 - 48.0) / 50.0).collect();
        let sizes: Vec<usize> = (0..3).map(|_| overlap.add(&frames, &window).len()).collect();
        assert_eq!(sizes, vec![1440, 1920, 1920]);
        assert_eq!(overlap.finish().len(), 480);
        assert_eq!(overlap.finish().len(), 0, "a finished stream holds nothing");
        assert_eq!(
            overlap.add(&frames, &window).len(),
            1440,
            "a finished stream starts again"
        );
    }

    #[test]
    fn a_frame_of_the_window_alone_gives_ones_inside_the_stream() {
        // Frames that are the window give the envelope, so each sample of the stream is 1.
        let window = hann();
        let frames: Vec<f32> = (0..STACK).flat_map(|_| window.iter().map(|w| w * w)).collect();
        let mut overlap = Overlap::default();
        let mut audio = Vec::new();
        for _ in 0..3 {
            audio.extend(overlap.add(&frames, &window));
        }
        audio.extend(overlap.finish());
        assert_eq!(audio.len(), 3 * FRAME_SAMPLES);
        assert!(audio.iter().all(|v| (v - 1.0).abs() < 1e-6), "{:?}", &audio[..4]);
    }
}
