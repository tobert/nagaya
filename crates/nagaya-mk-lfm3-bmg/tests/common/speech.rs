//! The CPU forms of the speech path of Omni (milestone 24 step 8): the depth decoder, the codebook table of the
//! backbone, and the decoder of LiquidCodec. They follow Liquid's remote code of `LFM3-Omni-2B` at `21d75123`:
//! `Lfm3OmniDepthDecoder` (modeling_lfm3_omni.py:230-281), `get_audio_out_embeds` (:576-597), `Lfm3Model`
//! (modeling_lfm3.py:445-765), and `LiquidCodecModel.decode` (modeling_liquid_codec.py:141-357).
//!
//! The forms sum in f64 and keep f32 values between the steps, so their gap to the oracle is the rounding of the
//! reference. They need no card. `speech_cpu_forms.rs` checks them against the oracle of `tools/lfm3_reference.py`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors, bf16_to_f32};

/// The codebooks of a frame, and the codes of each codebook (`config.json`).
pub const CODEBOOKS: usize = 12;
pub const CODES: usize = 2017;
/// The end code (`audio_eos_code`).
pub const END_CODE: u32 = 2016;
/// The samples of one frame at 24 kHz, and the STFT frames of one codec frame (`audio_tokenizer/config.json`).
pub const FRAME_SAMPLES: usize = 1920;
pub const STACK: usize = 6;
pub const N_FFT: usize = 1280;
pub const HOP: usize = 320;
pub const BINS: usize = N_FFT / 2 + 1;
/// The samples that the reference cuts from each end of the overlap-add (modeling_liquid_codec.py:324).
pub const TRIM: usize = (N_FFT - HOP) / 2;

/// The tensors of a safetensors file whose names start with `prefix`, as f32, with the prefix cut.
pub fn tensors(path: &Path, prefix: &str) -> HashMap<String, Vec<f32>> {
    let map = MappedSafeTensors::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let view = map.view().unwrap();
    let mut out = HashMap::new();
    for (name, meta) in &view.tensors {
        let Some(short) = name.strip_prefix(prefix) else {
            continue;
        };
        let bytes = view.tensor_bytes(name).unwrap();
        let values: Vec<f32> = match meta.dtype {
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
                .collect(),
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            other => panic!("tensor {name} is {other:?}; the speech path holds BF16 and F32"),
        };
        out.insert(short.to_owned(), values);
    }
    out
}

fn take(t: &HashMap<String, Vec<f32>>, name: &str) -> Vec<f32> {
    t.get(name).unwrap_or_else(|| panic!("no tensor {name}")).clone()
}

/// The rows `y[t] = x[t] . w[r] + b[r]` of `n` rows of `k` against `m` rows of `w`, summed in f64, in parallel.
pub fn linear(x: &[f32], n: usize, w: &[f32], m: usize, k: usize, b: Option<&[f32]>) -> Vec<f32> {
    assert_eq!(x.len(), n * k, "the input has {} values, not {n} x {k}", x.len());
    assert_eq!(w.len(), m * k, "the weights have {} values, not {m} x {k}", w.len());
    let mut y = vec![0.0f32; n * m];
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get());
    let per = m.div_ceil(threads).max(1);
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..m)
            .step_by(per)
            .map(|r0| {
                s.spawn(move || {
                    let r1 = (r0 + per).min(m);
                    let mut out = vec![0.0f32; n * (r1 - r0)];
                    for r in r0..r1 {
                        let row = &w[r * k..(r + 1) * k];
                        for t in 0..n {
                            let xt = &x[t * k..(t + 1) * k];
                            let mut acc = 0.0f64;
                            for (a, c) in xt.iter().zip(row) {
                                acc += *a as f64 * *c as f64;
                            }
                            let v = acc as f32 + b.map_or(0.0, |b| b[r]);
                            out[t * (r1 - r0) + r - r0] = v;
                        }
                    }
                    (r0, out)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (r0, part) in parts {
        let width = part.len() / n;
        for t in 0..n {
            y[t * m + r0..t * m + r0 + width].copy_from_slice(&part[t * width..(t + 1) * width]);
        }
    }
    y
}

/// RMSNorm of each row of `k` values with the weights `w`: f32 at the end, as `Lfm3RMSNorm` in f32 is.
pub fn rmsnorm(x: &[f32], w: &[f32], k: usize, eps: f32) -> Vec<f32> {
    x.chunks(k)
        .flat_map(|row| {
            let ss: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let inv = 1.0 / (ss / k as f64 + eps as f64).sqrt();
            row.iter()
                .zip(w)
                .map(move |(v, g)| (*v as f64 * inv) as f32 * g)
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    (x as f64 / (1.0 + (-x as f64).exp())) as f32
}

fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-x as f64).exp())) as f32
}

/// The kind of one layer of an LFM3 stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Conv,
    Attention,
}

/// The weights of one layer of an LFM3 stack.
pub struct Layer {
    kind: Kind,
    operator_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post: Option<(Vec<f32>, Vec<f32>)>,
    w1: Vec<f32>,
    w3: Vec<f32>,
    w2: Vec<f32>,
    in_proj: Vec<f32>,
    conv: Vec<f32>,
    out_proj: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: Vec<f32>,
}

/// The numbers of an LFM3 stack: the depth trunk, or the decoder of the codec.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub taps: usize,
    /// 0: every key of the call, in both directions. Else causal, with this window.
    pub window: usize,
    pub theta: f32,
    pub eps: f32,
}

/// The depth trunk: 3 full attention layers in both directions over the 12 codebooks (`depth_config`).
pub const DEPTH: Shape = Shape {
    hidden: 1024,
    heads: 32,
    kv_heads: 8,
    head_dim: 32,
    intermediate: 2816,
    taps: 7,
    window: 0,
    theta: 1e6,
    eps: 1e-5,
};

/// The decoder of LiquidCodec: 8 layers, sliding attention of 30 positions on layers 2, 4 and 6
/// (configuration_liquid_codec.py:110-130).
pub const CODEC: Shape = Shape {
    hidden: 512,
    heads: 8,
    kv_heads: 2,
    head_dim: 64,
    intermediate: 2304,
    taps: 7,
    window: 30,
    theta: 1e6,
    eps: 1e-5,
};

/// The state of a stack between calls: the conv inputs of each conv layer, and the keys and values of each attention
/// layer at every position so far.
#[derive(Default, Clone)]
pub struct State {
    conv: HashMap<usize, Vec<f32>>,
    keys: HashMap<usize, Vec<f32>>,
    values: HashMap<usize, Vec<f32>>,
    pub positions: usize,
}

/// An LFM3 stack with its final norm.
pub struct Stack {
    pub shape: Shape,
    layers: Vec<Layer>,
    norm: Vec<f32>,
}

impl Stack {
    /// The stack of `t` with the names under `prefix` (`layers.<l>....` and `embedding_norm.weight`).
    pub fn load(t: &HashMap<String, Vec<f32>>, prefix: &str, shape: Shape, kinds: &[Kind], sandwich: bool) -> Self {
        let layers = kinds
            .iter()
            .enumerate()
            .map(|(l, kind)| {
                let n = |name: &str| format!("{prefix}layers.{l}.{name}");
                let get = |name: &str| take(t, &n(name));
                let maybe = |name: &str| t.get(&n(name)).cloned().unwrap_or_default();
                Layer {
                    kind: *kind,
                    operator_norm: get("operator_norm.weight"),
                    ffn_norm: get("ffn_norm.weight"),
                    post: sandwich.then(|| (get("post_operator_norm.weight"), get("post_ffn_norm.weight"))),
                    w1: get("feed_forward.w1.weight"),
                    w3: get("feed_forward.w3.weight"),
                    w2: get("feed_forward.w2.weight"),
                    in_proj: maybe("conv.in_proj.weight"),
                    conv: maybe("conv.conv.weight"),
                    out_proj: maybe("conv.out_proj.weight"),
                    q: maybe("self_attn.q_proj.weight"),
                    k: maybe("self_attn.k_proj.weight"),
                    v: maybe("self_attn.v_proj.weight"),
                    o: maybe("self_attn.o_proj.weight"),
                    q_norm: maybe("self_attn.q_norm.weight"),
                    k_norm: maybe("self_attn.k_norm.weight"),
                    gate: maybe("self_attn.gate_proj.weight"),
                }
            })
            .collect();
        Self {
            shape,
            layers,
            norm: take(t, &format!("{prefix}embedding_norm.weight")),
        }
    }

    /// The norm and the rotation of `rows` head rows of `d` values, `per` rows for each position from `pos0`
    /// (modeling_lfm3.py:72-124, 204-210). The angle is f32, as the reference makes it.
    fn norm_rope(&self, x: &mut [f32], w: &[f32], per: usize, pos0: usize) {
        let d = self.shape.head_dim;
        let half = d / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.shape.theta.powf((2 * i) as f32 / d as f32))
            .collect();
        for (r, row) in x.chunks_mut(d).enumerate() {
            let normed = rmsnorm(row, w, d, self.shape.eps);
            let pos = (pos0 + r / per) as f32;
            for i in 0..half {
                let ang = pos * inv[i];
                let (c, s) = ((ang as f64).cos(), (ang as f64).sin());
                let (c, s) = (c as f32 as f64, s as f32 as f64);
                let (a, b) = (normed[i] as f64, normed[i + half] as f64);
                row[i] = (a * c - b * s) as f32;
                row[i + half] = (b * c + a * s) as f32;
            }
        }
    }

    fn conv(&self, l: usize, a: &[f32], n: usize, state: &mut State) -> Vec<f32> {
        let (h, taps) = (self.shape.hidden, self.shape.taps);
        let layer = &self.layers[l];
        let bcx = linear(a, n, &layer.in_proj, 3 * h, h, None);
        let hist = state.conv.entry(l).or_insert_with(|| vec![0.0; (taps - 1) * h]);
        let mut u = hist.clone();
        for t in 0..n {
            let row = &bcx[t * 3 * h..(t + 1) * 3 * h];
            u.extend((0..h).map(|c| row[c] * row[2 * h + c]));
        }
        let mut y = vec![0.0f32; n * h];
        for t in 0..n {
            let row = &bcx[t * 3 * h..(t + 1) * 3 * h];
            for c in 0..h {
                let mut acc = 0.0f64;
                for j in 0..taps {
                    acc += layer.conv[c * taps + j] as f64 * u[(t + j) * h + c] as f64;
                }
                y[t * h + c] = row[h + c] * acc as f32;
            }
        }
        *hist = u[n * h..].to_vec();
        linear(&y, n, &layer.out_proj, h, h, None)
    }

    fn attention(&self, l: usize, a: &[f32], n: usize, pos0: usize, state: &mut State) -> Vec<f32> {
        let s = self.shape;
        let (h, heads, kv, d) = (s.hidden, s.heads, s.kv_heads, s.head_dim);
        let layer = &self.layers[l];
        let mut q = linear(a, n, &layer.q, heads * d, h, None);
        let mut k = linear(a, n, &layer.k, kv * d, h, None);
        let v = linear(a, n, &layer.v, kv * d, h, None);
        let gates = linear(a, n, &layer.gate, heads, h, None);
        self.norm_rope(&mut q, &layer.q_norm, heads, pos0);
        self.norm_rope(&mut k, &layer.k_norm, kv, pos0);
        let keys = state.keys.entry(l).or_default();
        keys.extend_from_slice(&k);
        let values = state.values.entry(l).or_default();
        values.extend_from_slice(&v);
        let (keys, values) = (state.keys[&l].clone(), state.values[&l].clone());
        let total = keys.len() / (kv * d);
        let scale = (d as f64).powf(-0.5);
        let group = heads / kv;
        let mut y = vec![0.0f32; n * heads * d];
        for t in 0..n {
            let p = pos0 + t;
            let (lo, hi) = if s.window == 0 {
                (pos0, pos0 + n)
            } else {
                ((p + 1).saturating_sub(s.window), p + 1)
            };
            assert!(hi <= total, "the keys stop at {total}");
            for hh in 0..heads {
                let g = hh / group;
                let qv = &q[(t * heads + hh) * d..(t * heads + hh + 1) * d];
                let scores: Vec<f64> = (lo..hi)
                    .map(|j| {
                        let kr = &keys[(j * kv + g) * d..(j * kv + g + 1) * d];
                        qv.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale
                    })
                    .collect();
                let most = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|x| (x - most).exp()).collect();
                let sum: f64 = e.iter().sum();
                let gate = sigmoid(gates[t * heads + hh]) as f64;
                for i in 0..d {
                    let mut acc = 0.0f64;
                    for (jj, j) in (lo..hi).enumerate() {
                        acc += e[jj] * values[(j * kv + g) * d + i] as f64;
                    }
                    y[(t * heads + hh) * d + i] = ((acc / sum) as f32 as f64 * gate) as f32;
                }
            }
        }
        linear(&y, n, &layer.o, h, heads * d, None)
    }

    /// Run `n` rows of `x` at the positions from `state.positions`, and give the final norm of each row.
    pub fn forward(&self, x: &[f32], n: usize, state: &mut State) -> Vec<f32> {
        let s = self.shape;
        let h = s.hidden;
        let pos0 = state.positions;
        let mut x = x.to_vec();
        for (l, layer) in self.layers.iter().enumerate() {
            let a = rmsnorm(&x, &layer.operator_norm, h, s.eps);
            let mut out = match layer.kind {
                Kind::Conv => self.conv(l, &a, n, state),
                Kind::Attention => self.attention(l, &a, n, pos0, state),
            };
            if let Some((post, _)) = &layer.post {
                out = rmsnorm(&out, post, h, s.eps);
            }
            for (xi, oi) in x.iter_mut().zip(&out) {
                *xi += *oi;
            }
            let a = rmsnorm(&x, &layer.ffn_norm, h, s.eps);
            let g1 = linear(&a, n, &layer.w1, s.intermediate, h, None);
            let g3 = linear(&a, n, &layer.w3, s.intermediate, h, None);
            let g: Vec<f32> = g1.iter().zip(&g3).map(|(a, b)| silu(*a) * b).collect();
            let mut f = linear(&g, n, &layer.w2, h, s.intermediate, None);
            if let Some((_, post)) = &layer.post {
                f = rmsnorm(&f, post, h, s.eps);
            }
            for (xi, fi) in x.iter_mut().zip(&f) {
                *xi += *fi;
            }
        }
        state.positions += n;
        rmsnorm(&x, &self.norm, h, s.eps)
    }
}

/// The depth decoder of Omni with its codebook tables.
pub struct Depth {
    linear_w: Vec<f32>,
    linear_b: Vec<f32>,
    mask: Vec<f32>,
    /// The table of each codebook `[2017][1024]`, which is also its head.
    tables: Vec<Vec<f32>>,
    norms: Vec<Vec<f32>>,
    pub trunk: Stack,
    /// The codebook table of the backbone `[12 * 2017][2048]`, for the embedding of a fed frame.
    codebook: Vec<f32>,
}

impl Depth {
    pub fn load(omni: &Path) -> Self {
        let t = tensors(&omni.join("model.safetensors"), "model.");
        let p = "depth_decoder.";
        Self {
            linear_w: take(&t, &format!("{p}depth_linear.weight")),
            linear_b: take(&t, &format!("{p}depth_linear.bias")),
            mask: take(&t, &format!("{p}mask_embedding")),
            tables: (0..CODEBOOKS)
                .map(|g| take(&t, &format!("{p}depth_embeddings.{g}.embedding.weight")))
                .collect(),
            norms: (0..CODEBOOKS)
                .map(|g| take(&t, &format!("{p}depth_embeddings.{g}.embedding_norm.weight")))
                .collect(),
            trunk: Stack::load(&t, &format!("{p}model."), DEPTH, &[Kind::Attention; 3], false),
            codebook: take(&t, "codebook_embedding.weight"),
        }
    }

    /// The input of the trunk for one frame: the projected hidden state of each codebook, plus the code embedding of a
    /// revealed codebook or the mask embedding of a masked one (modeling_lfm3_omni.py:266-268).
    pub fn input(&self, hidden: &[f32], codes: &[u32], masked: &[bool]) -> Vec<f32> {
        let d = DEPTH.hidden;
        let mut x = linear(hidden, 1, &self.linear_w, CODEBOOKS * d, 2048, Some(&self.linear_b));
        for g in 0..CODEBOOKS {
            let add = if masked[g] {
                &self.mask[..]
            } else {
                &self.tables[g][codes[g] as usize * d..(codes[g] as usize + 1) * d]
            };
            for (xi, ai) in x[g * d..(g + 1) * d].iter_mut().zip(add) {
                *xi += *ai;
            }
        }
        x
    }

    /// The logits `[12][2017]` of the trunk output of one frame: the norm of each codebook, then its table.
    pub fn logits(&self, trunk: &[f32]) -> Vec<f32> {
        let d = DEPTH.hidden;
        (0..CODEBOOKS)
            .flat_map(|g| {
                let normed = rmsnorm(&trunk[g * d..(g + 1) * d], &self.norms[g], d, DEPTH.eps);
                linear(&normed, 1, &self.tables[g], CODES, d, None)
            })
            .collect()
    }

    /// The trunk output and the logits of one frame.
    pub fn run(&self, hidden: &[f32], codes: &[u32], masked: &[bool]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let input = self.input(hidden, codes, masked);
        let trunk = self.trunk.forward(&input, CODEBOOKS, &mut State::default());
        let logits = self.logits(&trunk);
        (input, trunk, logits)
    }

    /// The embedding of a fed frame: the sum of the 12 rows of the codebook table (modeling_lfm3_omni.py:596-597).
    pub fn frame_embedding(&self, codes: &[u32]) -> Vec<f32> {
        let h = 2048;
        let mut sum = vec![0.0f64; h];
        for (g, code) in codes.iter().enumerate() {
            let row = (g * CODES + *code as usize) * h;
            for (s, v) in sum.iter_mut().zip(&self.codebook[row..row + h]) {
                *s += *v as f64;
            }
        }
        sum.into_iter().map(|v| v as f32).collect()
    }
}

/// The decoder of LiquidCodec.
pub struct Codec {
    project_out_w: Vec<f32>,
    project_out_b: Vec<f32>,
    from_w: Vec<f32>,
    from_b: Vec<f32>,
    pub stack: Stack,
    lin_w: Vec<f32>,
    lin_b: Vec<f32>,
    pub window: Vec<f32>,
}

/// The FSQ levels and the base of each digit of a code (configuration_liquid_codec.py, `fsq_levels`).
pub const LEVELS: [u32; 4] = [8, 7, 6, 6];

/// The 48 latents of one frame: 4 for each codebook, `((code / base) % level - level / 2) / (level / 2)`
/// (modeling_liquid_codec.py:182-189).
pub fn latents(codes: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(codes.len() * 4);
    for code in codes {
        let mut base = 1u32;
        for level in LEVELS {
            let half = (level / 2) as i32;
            out.push((((code / base) % level) as i32 - half) as f32 / half as f32);
            base *= level;
        }
    }
    out
}

/// The overlap-add of the inverse FFT frames, with the stream state of the reference (modeling_liquid_codec.py:
/// 324-357): the samples past the last frame wait for the next call.
#[derive(Default)]
pub struct Overlap {
    audio: Option<Vec<f64>>,
    envelope: Option<Vec<f64>>,
    started: bool,
}

impl Overlap {
    /// Add `frames` (each `N_FFT` samples after the window) and give the samples that are whole. With `last`, give the
    /// rest too.
    pub fn add(&mut self, frames: &[f32], window: &[f32], last: bool) -> Vec<f32> {
        let count = frames.len() / N_FFT;
        let tail = 2 * TRIM;
        let (mut audio, mut envelope) = if count == 0 {
            (vec![0.0f64; tail], vec![0.0f64; tail])
        } else {
            let len = (count - 1) * HOP + N_FFT;
            let (mut a, mut e) = (vec![0.0f64; len], vec![0.0f64; len]);
            for f in 0..count {
                for i in 0..N_FFT {
                    a[f * HOP + i] += frames[f * N_FFT + i] as f64;
                    e[f * HOP + i] += (window[i] as f64) * (window[i] as f64);
                }
            }
            (a, e)
        };
        if let (Some(old_a), Some(old_e)) = (self.audio.take(), self.envelope.take()) {
            for i in 0..tail {
                audio[i] += old_a[i];
                envelope[i] += old_e[i];
            }
        }
        if !self.started {
            audio.drain(..TRIM);
            envelope.drain(..TRIM);
            self.started = true;
        }
        if last {
            audio.truncate(audio.len() - TRIM);
            envelope.truncate(envelope.len() - TRIM);
        } else {
            let keep = audio.len() - tail;
            self.audio = Some(audio.split_off(keep));
            self.envelope = Some(envelope.split_off(keep));
        }
        audio.iter().zip(&envelope).map(|(a, e)| (a / e) as f32).collect()
    }
}

impl Codec {
    pub fn load(omni: &Path) -> Self {
        let t = tensors(&omni.join("audio_tokenizer/model.safetensors"), "");
        let kinds: Vec<Kind> = (0..8)
            .map(|l| {
                if [2, 4, 6].contains(&l) {
                    Kind::Attention
                } else {
                    Kind::Conv
                }
            })
            .collect();
        Self {
            project_out_w: take(&t, "quantizer.project_out.weight"),
            project_out_b: take(&t, "quantizer.project_out.bias"),
            from_w: take(&t, "from_quantizer.weight"),
            from_b: take(&t, "from_quantizer.bias"),
            stack: Stack::load(&t, "decoder.lfm.", CODEC, &kinds, true),
            lin_w: take(&t, "decoder.lin.weight"),
            lin_b: take(&t, "decoder.lin.bias"),
            window: take(&t, "decoder.istft.window"),
        }
    }

    /// The embedding of each frame `[n][512]`, after the quantizer and `from_quantizer`.
    pub fn embeddings(&self, codes: &[u32]) -> Vec<f32> {
        let n = codes.len() / CODEBOOKS;
        let lat = latents(codes);
        let q = linear(&lat, n, &self.project_out_w, 256, 48, Some(&self.project_out_b));
        linear(&q, n, &self.from_w, 512, 256, Some(&self.from_b))
    }

    /// The windowed inverse FFT frames `[6n][1280]` of `n` frames of codes, from the stack state `state`.
    pub fn frames(&self, codes: &[u32], state: &mut State) -> Vec<f32> {
        let n = codes.len() / CODEBOOKS;
        let emb = self.embeddings(codes);
        let h = CODEC.hidden;
        let mut rows = Vec::with_capacity(n * STACK * h);
        for f in 0..n {
            for _ in 0..STACK {
                rows.extend_from_slice(&emb[f * h..(f + 1) * h]);
            }
        }
        let out = self.stack.forward(&rows, n * STACK, state);
        let spec = linear(&out, n * STACK, &self.lin_w, 2 * BINS, h, Some(&self.lin_b));
        irfft_frames(&spec, n * STACK, &self.window)
    }

    /// The audio of a whole clip of codes, as `decode` with no stream gives it.
    pub fn decode(&self, codes: &[u32]) -> Vec<f32> {
        let frames = self.frames(codes, &mut State::default());
        let mut overlap = Overlap::default();
        overlap.add(&frames, &self.window, true)
    }

    /// The audio of a clip in calls of `chunk` frames, as the stream of the reference gives it.
    pub fn decode_stream(&self, codes: &[u32], chunk: usize) -> (Vec<f32>, Vec<usize>) {
        let mut state = State::default();
        let mut overlap = Overlap::default();
        let mut audio = Vec::new();
        let mut sizes = Vec::new();
        let n = codes.len() / CODEBOOKS;
        for start in (0..n).step_by(chunk) {
            let end = (start + chunk).min(n);
            let frames = self.frames(&codes[start * CODEBOOKS..end * CODEBOOKS], &mut state);
            let piece = overlap.add(&frames, &self.window, end == n);
            sizes.push(piece.len());
            audio.extend(piece);
        }
        (audio, sizes)
    }
}

/// The inverse real FFT of each row of `n` rows of `2 * BINS` values (the log magnitudes, then the angles), times
/// the window (modeling_liquid_codec.py:200-202, 228-229). The sum is f64, with no FFT.
pub fn irfft_frames(spec: &[f32], n: usize, window: &[f32]) -> Vec<f32> {
    let tw: Vec<(f64, f64)> = (0..N_FFT)
        .map(|m| {
            let a = 2.0 * std::f64::consts::PI * m as f64 / N_FFT as f64;
            (a.cos(), a.sin())
        })
        .collect();
    let threads = std::thread::available_parallelism().map_or(8, |p| p.get());
    let per = n.div_ceil(threads).max(1);
    let parts: Vec<Vec<f32>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .step_by(per)
            .map(|t0| {
                let tw = &tw;
                s.spawn(move || {
                    let mut out = Vec::new();
                    for t in t0..(t0 + per).min(n) {
                        let row = &spec[t * 2 * BINS..(t + 1) * 2 * BINS];
                        let (re, im): (Vec<f64>, Vec<f64>) = (0..BINS)
                            .map(|k| {
                                let mag = (row[k] as f64).exp() as f32 as f64;
                                let ang = row[BINS + k] as f64;
                                ((mag * ang.cos()) as f32 as f64, (mag * ang.sin()) as f32 as f64)
                            })
                            .unzip();
                        for i in 0..N_FFT {
                            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                            let mut acc = re[0] + sign * re[BINS - 1];
                            for k in 1..BINS - 1 {
                                let (c, s) = tw[(k * i) % N_FFT];
                                acc += 2.0 * (re[k] * c - im[k] * s);
                            }
                            out.push((acc / N_FFT as f64 * window[i] as f64) as f32);
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.concat()
}

/// The codes of `frames` frames of a tensor `[frames][12]` of the oracle.
pub fn codes_of(values: &[u32], frames: usize) -> Vec<u32> {
    values[..frames * CODEBOOKS].to_vec()
}
