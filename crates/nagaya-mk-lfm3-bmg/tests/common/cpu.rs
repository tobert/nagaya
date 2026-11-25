//! A CPU form of the interpreter for the ops that the audio encoder uses. It runs the tasks of
//! `audio::encode` on host memory, so the host test checks the graph, the addresses and the layout against the oracle
//! with no card. Each op follows the comment of its kernel. The sums of the dot products and the norms are in f64,
//! so the form is close to the f32 reference of transformers. It does not model the bits of the card: the card test
//! prints the gap between the card and this form.
#![allow(dead_code)]

use std::path::Path;

use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_lfm3_bmg::audio::{self, AudioBufs, AudioWeights, Clip, Encoder, MAX_CLIP_POSITIONS, Stage};
use nagaya_mk_lfm3_bmg::weights::{Upload, rounded};

/// The first address of the host memory, so that address 0 is never a buffer.
const BASE: u64 = 1 << 32;

/// Host memory that stands in for the device memory of a card.
pub struct CpuCard {
    mem: Vec<u8>,
}

impl Default for CpuCard {
    fn default() -> Self {
        Self::new()
    }
}

impl Upload for CpuCard {
    fn alloc(&mut self, bytes: usize) -> u64 {
        let at = BASE + self.mem.len() as u64;
        self.mem.resize(self.mem.len() + rounded(bytes), 0);
        at
    }

    fn write(&mut self, at: u64, data: &[u8]) {
        let o = self.offset(at, data.len());
        self.mem[o..o + data.len()].copy_from_slice(data);
    }

    fn writes(&self) -> bool {
        true
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The error function in f64: the Taylor series below 2.5, and the continued fraction of erfc above.
fn erf(x: f64) -> f64 {
    let a = x.abs();
    let y = if a < 2.5 {
        let (mut term, mut sum, mut n) = (a, a, 0.0f64);
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -a * a / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        let mut k = a;
        for n in (1..=80).rev() {
            k = a + (n as f64 / 2.0) / k;
        }
        1.0 - (-a * a).exp() / (std::f64::consts::PI.sqrt() * k)
    };
    if x < 0.0 { -y } else { y }
}

impl CpuCard {
    pub fn new() -> Self {
        Self { mem: Vec::new() }
    }

    fn offset(&self, at: u64, bytes: usize) -> usize {
        assert!(at >= BASE, "address {at:#x} is not in the host memory");
        let o = (at - BASE) as usize;
        assert!(
            o + bytes <= self.mem.len(),
            "bytes {o}..{} are past the {} bytes of the host memory",
            o + bytes,
            self.mem.len()
        );
        o
    }

    /// `n` f32 values at `at`.
    pub fn f32s(&self, at: u64, n: usize) -> Vec<f32> {
        let o = self.offset(at, n * 4);
        self.mem[o..o + n * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// `n` BF16 or F32 weights at `at`, as f32.
    fn weights(&self, at: u64, n: usize, f32_weights: bool) -> Vec<f32> {
        if f32_weights {
            return self.f32s(at, n);
        }
        let o = self.offset(at, n * 2);
        self.mem[o..o + n * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16(u16::from_le_bytes(*b)))
            .collect()
    }

    pub fn put(&mut self, at: u64, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write(at, &bytes);
    }

    /// Run the tasks in order. An op that this form does not hold panics.
    pub fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        for task in tasks {
            self.step(task);
        }
    }

    fn step(&mut self, t: &MkTaskDescriptor) {
        let acc = t.flags & FLAG_ACCUMULATE != 0;
        let (n, m, k) = (t.n as usize, t.m as usize, t.k as usize);
        match t.opcode {
            OP_SUBSAMPLE => {
                let (cin, silu_on) = (t.p0 as usize, t.layer == 1);
                let x = self.f32s(t.a, n * k * cin);
                let w = self.weights(t.b, m * 9, false);
                let b = self.weights(t.x, m, false);
                let (to_n, ho_n) = (n.div_ceil(2), k.div_ceil(2));
                let mut y = vec![0.0f32; to_n * ho_n * m];
                for to in 0..to_n {
                    for ho in 0..ho_n {
                        for ch in 0..m {
                            let ci = if cin == 1 { 0 } else { ch };
                            let mut s = 0.0f64;
                            for i in 0..3 {
                                let h = (2 * ho + i) as isize - 1;
                                for j in 0..3 {
                                    let tt = (2 * to + j) as isize - 1;
                                    if h >= 0 && (h as usize) < k && tt >= 0 && (tt as usize) < n {
                                        let v = x[(tt as usize * k + h as usize) * cin + ci];
                                        s += w[ch * 9 + 3 * i + j] as f64 * v as f64;
                                    }
                                }
                            }
                            let v = (s + b[ch] as f64) as f32;
                            y[(to * ho_n + ho) * m + ch] = if silu_on { silu(v) } else { v };
                        }
                    }
                }
                self.put(t.c, &y);
            }
            OP_MATMUL => {
                let x = self.f32s(t.a, n * k);
                let w = self.weights(t.b, m * k, false);
                let mut y = if acc { self.f32s(t.c, n * m) } else { vec![0.0; n * m] };
                rows_parallel(&mut y, m, |row, out| {
                    let xr = &x[row * k..][..k];
                    for (r, o) in out.iter_mut().enumerate() {
                        let d: f64 = w[r * k..][..k].iter().zip(xr).map(|(a, b)| *a as f64 * *b as f64).sum();
                        *o = if acc { *o + d as f32 } else { d as f32 };
                    }
                });
                self.put(t.c, &y);
            }
            OP_GEGLU => {
                // SwiGLU in the LFM3 binary: silu(x . w[r]) * (x . w[m + r]).
                let x = self.f32s(t.a, n * k);
                let w = self.weights(t.b, 2 * m * k, false);
                let mut y = vec![0.0f32; n * m];
                rows_parallel(&mut y, m, |row, out| {
                    let xr = &x[row * k..][..k];
                    for (r, o) in out.iter_mut().enumerate() {
                        let dot = |r: usize| -> f32 {
                            w[r * k..][..k]
                                .iter()
                                .zip(xr)
                                .map(|(a, b)| *a as f64 * *b as f64)
                                .sum::<f64>() as f32
                        };
                        *o = silu(dot(r)) * dot(m + r);
                    }
                });
                self.put(t.c, &y);
            }
            OP_BIAS => {
                let b = self.weights(t.b, k, t.flags & FLAG_F32 != 0);
                let mut y = self.f32s(t.c, n * k);
                for (i, v) in y.iter_mut().enumerate() {
                    *v += b[i % k];
                }
                self.put(t.c, &y);
            }
            OP_SILU_MUL => {
                let a = self.f32s(t.a, n * k);
                let b = (t.b != 0).then(|| self.f32s(t.b, n * k));
                let y: Vec<f32> = a
                    .iter()
                    .enumerate()
                    .map(|(i, v)| silu(*v) * b.as_ref().map_or(1.0, |b| b[i]))
                    .collect();
                self.put(t.c, &y);
            }
            OP_GELU => {
                let a = self.f32s(t.a, n * k);
                let y: Vec<f32> = a
                    .iter()
                    .map(|v| {
                        let v = *v as f64;
                        (0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2))) as f32
                    })
                    .collect();
                self.put(t.c, &y);
            }
            OP_LAYERNORM => {
                let a = self.f32s(t.a, n * k);
                let w = self.f32s(t.b, k);
                let b = self.f32s(t.x, k);
                let eps = f32::from_bits(t.p0) as f64;
                let mut y = vec![0.0f32; n * k];
                for r in 0..n {
                    let x = &a[r * k..][..k];
                    let mean = x.iter().map(|v| *v as f64).sum::<f64>() / k as f64;
                    let var = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / k as f64;
                    let inv = 1.0 / (var + eps).sqrt();
                    for i in 0..k {
                        y[r * k + i] = ((x[i] as f64 - mean) * inv * w[i] as f64 + b[i] as f64) as f32;
                    }
                }
                self.put(t.c, &y);
            }
            OP_RMSNORM => self.rmsnorm(t),
            OP_COPY => {
                let v = self.f32s(t.a, n);
                self.put(t.c, &v);
            }
            OP_SHORT_CONV => {
                let (d, taps) = (k, t.layer as usize);
                let hist = taps - 1;
                let bcx = self.f32s(t.a, n * 3 * d);
                let w = self.weights(t.b, d * taps, t.flags & FLAG_F32 != 0);
                let state = self.f32s(t.x, hist * d);
                let u = |s: isize, c: usize| -> f32 {
                    if s < 0 {
                        state[(hist as isize + s) as usize * d + c]
                    } else {
                        let row = &bcx[s as usize * 3 * d..][..3 * d];
                        row[c] * row[2 * d + c]
                    }
                };
                let mut y = vec![0.0f32; n * d];
                for p in 0..n {
                    for c in 0..d {
                        let mut s = 0.0f64;
                        for j in 0..taps {
                            s += w[c * taps + j] as f64 * u(p as isize - hist as isize + j as isize, c) as f64;
                        }
                        y[p * d + c] = bcx[p * 3 * d + d + c] * s as f32;
                    }
                }
                let next: Vec<f32> = (0..hist * d)
                    .map(|i| u(n as isize - hist as isize + (i / d) as isize, i % d))
                    .collect();
                self.put(t.c, &y);
                self.put(t.x, &next);
            }
            OP_ATTN_SCORES => {
                let (h_n, kv_n, d, p0) = (m & 0xFFFF, m >> 16, k, t.p0 as usize);
                assert_eq!(t.layer, 0, "the CPU form has no ring");
                let window = (t.x & 0xFFFF_FFFF) as usize;
                let row_f = row_floats(p0, n);
                let q = self.f32s(t.a, n * h_n * d);
                let keys = self.f32s(t.b, (p0 + n) * kv_n * d);
                let mut s = self.f32s(t.c, h_n * n * row_f);
                for h in 0..h_n {
                    let kvh = h / (h_n / kv_n);
                    for tt in 0..n {
                        let (lo, hi) = key_range(p0 + tt, window);
                        for j in lo..=hi {
                            let qr = &q[(tt * h_n + h) * d..][..d];
                            let kr = &keys[(j * kv_n + kvh) * d..][..d];
                            s[(h * n + tt) * row_f + j] =
                                qr.iter().zip(kr).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
                        }
                    }
                }
                self.put(t.c, &s);
            }
            OP_ATTN_SOFTMAX => {
                let (h_n, p0) = (m & 0xFFFF, t.p0 as usize);
                let window = (t.x & 0xFFFF_FFFF) as usize;
                let row_f = row_floats(p0, n);
                let mut s = self.f32s(t.c, h_n * n * row_f);
                for h in 0..h_n {
                    for tt in 0..n {
                        let (lo, hi) = key_range(p0 + tt, window);
                        let row = &mut s[(h * n + tt) * row_f..][..row_f];
                        let top = row[lo..=hi].iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b)) as f64;
                        let sum: f64 = row[lo..=hi].iter().map(|v| (*v as f64 - top).exp()).sum();
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = if (lo..=hi).contains(&j) {
                                ((*v as f64 - top).exp() / sum) as f32
                            } else {
                                0.0
                            };
                        }
                    }
                }
                self.put(t.c, &s);
            }
            OP_ATTN_PV => {
                let (h_n, kv_n, d, p0) = (m & 0xFFFF, m >> 16, k, t.p0 as usize);
                let max_seq = (t.x >> 32) as usize;
                let row_f = row_floats(p0, n);
                let s = self.f32s(t.a, h_n * n * row_f);
                let vals = self.f32s(t.b + (max_seq * kv_n * d * 4) as u64, (p0 + n) * kv_n * d);
                let mut o = vec![0.0f32; n * h_n * d];
                for h in 0..h_n {
                    let kvh = h / (h_n / kv_n);
                    for tt in 0..n {
                        for e in 0..d {
                            let sum: f64 = (0..p0 + n)
                                .map(|j| s[(h * n + tt) * row_f + j] as f64 * vals[(j * kv_n + kvh) * d + e] as f64)
                                .sum();
                            o[(tt * h_n + h) * d + e] = sum as f32;
                        }
                    }
                }
                self.put(t.c, &o);
            }
            // OP_ATTN_PV writes the whole sum here, so the sum of the segments of the card has nothing to add.
            OP_ATTN_PV_SUM => {}
            other => panic!("the CPU form has no op {other}"),
        }
    }

    fn rmsnorm(&mut self, t: &MkTaskDescriptor) {
        let (n, k) = (t.n as usize, t.k as usize);
        let lo = (t.x & 0xFFFF_FFFF) as u32;
        let eps = if lo == 0 { 1e-6 } else { f32::from_bits(lo) as f64 };
        let hi = (t.x >> 32) as u32;
        let scale = if hi == 0 { 1.0 } else { f32::from_bits(hi) };
        let x = self.f32s(t.a, n * k);
        let w = if t.b == 0 {
            vec![1.0; k]
        } else {
            self.weights(t.b, k, t.flags & FLAG_F32 != 0)
        };
        let no_norm = t.flags & FLAG_NO_NORM != 0;
        let inv = |row: &[f32]| -> f32 {
            if no_norm {
                1.0
            } else {
                (1.0 / (row.iter().map(|v| *v as f64 * *v as f64).sum::<f64>() / k as f64 + eps).sqrt()) as f32
            }
        };
        if t.flags & FLAG_ROPE != 0 {
            assert_eq!(t.flags & FLAG_ROPE_GLOBAL, 0, "the CPU form has no global rotation");
            let theta: f64 = if t.flags & FLAG_ROPE_FULL != 0 { 1e6 } else { 1e4 };
            let half = k / 2;
            let pairs = if t.layer == 0 { half } else { t.layer as usize };
            let heads = t.m as usize;
            let mut y = vec![0.0f32; n * k];
            for r in 0..n {
                let row = &x[r * k..][..k];
                let iv = inv(row);
                let pos = (t.p0 as usize + r / heads) as f64;
                for i in 0..half {
                    let x0 = row[i] * iv * w[i];
                    let x1 = row[i + half] * iv * w[i + half];
                    let (a, b) = if i < pairs {
                        let ang = pos / theta.powf(2.0 * i as f64 / k as f64);
                        let (sn, cs) = (ang.sin(), ang.cos());
                        (
                            (x0 as f64 * cs - x1 as f64 * sn) as f32,
                            (x1 as f64 * cs + x0 as f64 * sn) as f32,
                        )
                    } else {
                        (x0, x1)
                    };
                    y[r * k + i] = a * scale;
                    y[r * k + i + half] = b * scale;
                }
            }
            self.put(t.c, &y);
        } else {
            let mut y = if t.flags & FLAG_ACCUMULATE != 0 {
                self.f32s(t.c, n * k)
            } else {
                vec![0.0; n * k]
            };
            let acc = t.flags & FLAG_ACCUMULATE != 0;
            for r in 0..n {
                let row = &x[r * k..][..k];
                let iv = inv(row);
                for i in 0..k {
                    let v = row[i] * iv * w[i];
                    y[r * k + i] = if acc { y[r * k + i] + v } else { v };
                }
            }
            self.put(t.c, &y);
        }
    }
}

/// The audio encoder of Omni in host memory, with its buffers for the longest clip.
pub struct CpuEncoder {
    pub cpu: CpuCard,
    pub e: Encoder,
    pub w: AudioWeights,
    pub bufs: AudioBufs,
}

/// The names of the stages that [`CpuEncoder::stages`] gives, as the oracle names them.
pub const STAGES: [&str; 3] = ["downsample", "encoder", "rows"];

impl CpuEncoder {
    /// Load the encoder of the Omni folder `dir`.
    pub fn load(dir: &Path) -> Self {
        let e = Encoder::read(dir).unwrap().expect("Omni has an audio encoder");
        let map = MappedSafeTensors::open(&dir.join("model.safetensors")).unwrap();
        let mut cpu = CpuCard::new();
        let w = audio::load(&mut cpu, &e, &map.view().unwrap()).unwrap();
        let bufs = AudioBufs::new(&mut |bytes| cpu.alloc(bytes), &e, MAX_CLIP_POSITIONS);
        Self { cpu, e, w, bufs }
    }

    /// Run the tasks of the clip `mel` (`[frames][mel]`) in three parts, and give the rows after each stage.
    pub fn stages(&mut self, mel: &[f32]) -> [Vec<f32>; 3] {
        let frames = mel.len() / self.e.mel;
        let p = audio::positions(frames);
        let out = self.cpu.alloc(p * self.e.out * 4);
        self.cpu.put(self.bufs.mel, mel);
        let clip = Clip {
            mel: self.bufs.mel,
            frames,
            out,
        };
        let all = audio::encode(&self.e, &self.w, &self.bufs, clip, Stage::Rows);
        let down = audio::encode(&self.e, &self.w, &self.bufs, clip, Stage::Downsample).len();
        let enc = audio::encode(&self.e, &self.w, &self.bufs, clip, Stage::Encoder).len();
        self.cpu.run(&all[..down]);
        let a = self.cpu.f32s(self.bufs.x, p * self.e.hidden);
        self.cpu.run(&all[down..enc]);
        let b = self.cpu.f32s(self.bufs.h, p * self.e.hidden);
        self.cpu.run(&all[enc..]);
        let c = self.cpu.f32s(out, p * self.e.out);
        [a, b, c]
    }
}

/// The mel rows `[frames][bins]` of an oracle tensor that holds the bins first.
pub fn mel_rows(golden: &super::Tensor) -> Vec<f32> {
    let (bins, frames) = (golden.shape[0], golden.shape[1]);
    let v = golden.f32();
    (0..frames * bins).map(|i| v[(i % bins) * frames + i / bins]).collect()
}

/// The floats of a score row of a block of `n` tokens at cache rows `p0..` (`attn_row_floats`, megakernel.cl).
fn row_floats(p0: usize, n: usize) -> usize {
    (p0 + n).div_ceil(16) * 16 + 16
}

/// The keys that position `pos` reads: the window of `window` keys up to itself, or all keys with a window of 0.
fn key_range(pos: usize, window: usize) -> (usize, usize) {
    let lo = if window > 0 && pos + 1 > window {
        pos + 1 - window
    } else {
        0
    };
    (lo, pos)
}

/// Run `f(row, out)` for each row of `width` values of `y`, on the threads of the host.
fn rows_parallel(y: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = y.len() / width;
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(rows.max(1));
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (c, chunk) in y.chunks_mut(per * width).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (i, out) in chunk.chunks_mut(width).enumerate() {
                    f(c * per + i, out);
                }
            });
        }
    });
}
