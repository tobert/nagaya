//! The ops of the speech binary on a B70 card (milestone 17 step 5), each against a plain CPU form on random inputs.
//! The Gemma 4 binary keeps its bytes, so the speech ops do not change the Gemma 4 engine.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_qwen3tts_ops -- --test-threads=1`.
#![cfg(feature = "xe")]

use std::ffi::c_void;

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{KERNEL_ENTRY, QWEN3TTS_KERNEL};

/// The SHA-256 of the Gemma 4 binary of milestone 15s, and the ocloc that makes it.
const GEMMA4_SHA256: &str = "a93ed27853c9835c512af1495ef1ada21cf74b0b8a2a7cbd3b7764d6304c5f2d";
const GEMMA4_OCLOC: &str = "26.35.39758";

/// A small random source: xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// A value in [-1, 1).
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

/// The BF16 bits of `x`, rounded to the nearest even.
fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// A buffer in device memory, which the test fills and reads through host memory.
struct Buf {
    ptr: *mut u8,
    bytes: usize,
}

impl Buf {
    fn zeros(card: &B70Card, bytes: usize) -> Self {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        Self { ptr, bytes }
    }

    fn of<T: Copy>(card: &B70Card, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::zeros(card, bytes);
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        // SAFETY: `stage` holds `bytes` bytes of host USM, and `data` is `bytes` bytes long.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, stage, bytes) };
        card.memcpy_sync(buf.ptr as *mut c_void, stage as *const c_void, bytes);
        card.free(stage as *mut c_void);
        buf
    }

    fn read<T: Copy + Default>(&self, card: &B70Card, n: usize) -> Vec<T> {
        let bytes = n * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes.max(64));
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        card.memcpy_sync(stage as *mut c_void, self.ptr as *const c_void, bytes);
        let mut out = vec![T::default(); n];
        // SAFETY: `stage` holds `bytes` bytes, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, bytes) };
        card.free(stage as *mut c_void);
        out
    }

    fn at(&self) -> u64 {
        self.ptr as u64
    }
}

struct Card {
    cluster: L0Cluster,
    ring: Ring,
}

impl Card {
    fn open() -> Self {
        let cluster = L0Cluster::init(QWEN3TTS_KERNEL, KERNEL_ENTRY);
        assert!(!cluster.cards.is_empty(), "no Level Zero card");
        let ring = Ring::new(&cluster.cards[0], 8 << 20);
        Self { cluster, ring }
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each test sizes its buffers for its tasks.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }
}

fn task(opcode: u16) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode,
        ..Default::default()
    }
}

/// The largest absolute difference, relative to the largest absolute value of `want`.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1e-30f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[test]
fn the_gemma4_binary_keeps_its_bytes() {
    if nagaya_mk_bmg::OCLOC_VERSION != GEMMA4_OCLOC {
        eprintln!(
            "skipped: the hash is of ocloc {GEMMA4_OCLOC}, and this build has ocloc {}",
            nagaya_mk_bmg::OCLOC_VERSION
        );
        return;
    }
    use sha2::{Digest, Sha256};
    let digest: String = Sha256::digest(nagaya_mk_bmg::GEMMA4_KERNEL)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest, GEMMA4_SHA256,
        "an op of the speech binary changed the Gemma 4 binary"
    );
}

#[test]
fn the_glu_of_the_speech_binary_is_swiglu() {
    let mut card = Card::open();
    let mut rng = Rng(7);
    let (m, k) = (48usize, 64usize);
    let w: Vec<u16> = rng.vec(2 * m * k).into_iter().map(bf16).collect();
    let wb = Buf::of(card.card(), &w);
    for n in [1usize, 3] {
        let x = rng.vec(n * k);
        let xb = Buf::of(card.card(), &x);
        let yb = Buf::zeros(card.card(), n * m * 4);
        card.run(&[MkTaskDescriptor {
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ..task(OP_GEGLU)
        }]);
        let got: Vec<f32> = yb.read(card.card(), n * m);
        let mut want = vec![0.0f32; n * m];
        for t in 0..n {
            for r in 0..m {
                let dot = |row: usize| -> f32 { (0..k).map(|i| from_bf16(w[row * k + i]) * x[t * k + i]).sum() };
                want[t * m + r] = silu(dot(r)) * dot(m + r);
            }
        }
        let err = rel(&got, &want);
        assert!(err < 1e-5, "{n} tokens: relative error {err}");
    }
}

#[test]
fn rmsnorm_takes_eps_f32_weights_a_full_rope_a_scale_and_no_norm() {
    let mut card = Card::open();
    let mut rng = Rng(11);
    // Rows of 2048 (a work-group each) and of 512 (a subgroup each), with eps 1e-5 and F32 weights.
    for k in [2048usize, 512] {
        let n = 3;
        let x = rng.vec(n * k);
        let w = rng.vec(k);
        let (xb, wb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &w),
            Buf::zeros(card.card(), n * k * 4),
        );
        card.run(&[MkTaskDescriptor {
            flags: FLAG_F32,
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            n: n as u32,
            k: k as u32,
            x: 1e-5f32.to_bits() as u64,
            ..task(OP_RMSNORM)
        }]);
        let got: Vec<f32> = yb.read(card.card(), n * k);
        let mut want = vec![0.0f32; n * k];
        for r in 0..n {
            let ss: f32 = x[r * k..(r + 1) * k].iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / k as f32 + 1e-5).sqrt();
            for i in 0..k {
                want[r * k + i] = x[r * k + i] * inv * w[i];
            }
        }
        let err = rel(&got, &want);
        assert!(err < 1e-5, "rows of {k}: relative error {err}");
    }
    // Head rows: Qwen3's q norm and full RoPE at theta 1e6 with a scale, then the codec's RoPE at 1e4 with no norm.
    let rope = |x: &[f32], d: usize, pos: f32, theta: f32| -> Vec<f32> {
        let h = d / 2;
        let mut y = vec![0.0f32; d];
        for i in 0..h {
            let ang = pos / theta.powf((2 * i) as f32 / d as f32);
            let (s, c) = ang.sin_cos();
            y[i] = x[i] * c - x[i + h] * s;
            y[i + h] = x[i + h] * c + x[i] * s;
        }
        y
    };
    for (d, heads, theta, flags, scale, weights) in [
        (
            128usize,
            4usize,
            1e6f32,
            FLAG_ROPE | FLAG_ROPE_FULL,
            128f32.powf(-0.5),
            true,
        ),
        (64usize, 3usize, 1e4f32, FLAG_ROPE | FLAG_NO_NORM, 0.125f32, false),
    ] {
        let tokens = 2;
        let n = tokens * heads;
        let p0 = 7u32;
        let x = rng.vec(n * d);
        let w: Vec<u16> = rng.vec(d).into_iter().map(bf16).collect();
        let (xb, wb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &w),
            Buf::zeros(card.card(), n * d * 4),
        );
        card.run(&[MkTaskDescriptor {
            flags,
            a: xb.at(),
            b: if weights { wb.at() } else { 0 },
            c: yb.at(),
            m: heads as u32,
            n: n as u32,
            k: d as u32,
            p0,
            x: (scale.to_bits() as u64) << 32,
            ..task(OP_RMSNORM)
        }]);
        let got: Vec<f32> = yb.read(card.card(), n * d);
        let mut want = Vec::with_capacity(n * d);
        for r in 0..n {
            let row = &x[r * d..(r + 1) * d];
            let inv = if weights {
                1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-6).sqrt()
            } else {
                1.0
            };
            let normed: Vec<f32> = (0..d)
                .map(|i| row[i] * inv * if weights { from_bf16(w[i]) } else { 1.0 })
                .collect();
            let pos = (p0 as usize + r / heads) as f32;
            want.extend(rope(&normed, d, pos, theta).into_iter().map(|v| v * scale));
        }
        let err = rel(&got, &want);
        assert!(err < 1e-5, "heads of {d}: relative error {err}");
    }
}

/// Softmax attention of one query row against `keys` rows, by the CPU.
fn attend(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    kv: usize,
    kvh: usize,
    d: usize,
    keys: std::ops::Range<usize>,
) -> Vec<f32> {
    let scores: Vec<f32> = keys
        .clone()
        .map(|j| (0..d).map(|i| q[i] * kc[(j * kv + kvh) * d + i]).sum())
        .collect();
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
    let z: f32 = e.iter().sum();
    let mut o = vec![0.0f32; d];
    for (p, j) in e.iter().zip(keys) {
        for i in 0..d {
            o[i] += p / z * vc[(j * kv + kvh) * d + i];
        }
    }
    o
}

#[test]
fn one_token_attention_takes_heads_of_128_and_of_64() {
    let mut card = Card::open();
    let mut rng = Rng(13);
    for (d, h, kv, p0, window) in [
        (128usize, 16usize, 8usize, 37usize, 0usize),
        (64, 16, 16, 40, 0),
        (64, 16, 16, 40, 9),
    ] {
        let max_seq = 64usize;
        let q = rng.vec(h * d);
        let kc = rng.vec(max_seq * kv * d);
        let vc = rng.vec(max_seq * kv * d);
        let cache: Vec<f32> = kc.iter().chain(vc.iter()).copied().collect();
        let (qb, cb, ob) = (
            Buf::of(card.card(), &q),
            Buf::of(card.card(), &cache),
            Buf::zeros(card.card(), h * d * 4),
        );
        let x = ((max_seq as u64) << 32) | window as u64;
        card.run(&[
            MkTaskDescriptor {
                a: qb.at(),
                b: cb.at(),
                m: (h | kv << 16) as u32,
                k: d as u32,
                p0: p0 as u32,
                x,
                ..task(OP_ATTN)
            },
            MkTaskDescriptor {
                c: ob.at(),
                m: h as u32,
                k: d as u32,
                p0: p0 as u32,
                x,
                ..task(OP_ATTN_COMBINE)
            },
        ]);
        let got: Vec<f32> = ob.read(card.card(), h * d);
        let lo = if window > 0 && p0 + 1 > window {
            p0 + 1 - window
        } else {
            0
        };
        let mut want = Vec::with_capacity(h * d);
        for head in 0..h {
            want.extend(attend(
                &q[head * d..(head + 1) * d],
                &kc,
                &vc,
                kv,
                head / (h / kv),
                d,
                lo..p0 + 1,
            ));
        }
        let err = rel(&got, &want);
        assert!(err < 1e-5, "head size {d}, window {window}: relative error {err}");
    }
}

#[test]
fn block_attention_takes_heads_of_64_with_a_window() {
    let mut card = Card::open();
    let mut rng = Rng(17);
    let (d, h, n, p0, window) = (64usize, 16usize, 29usize, 0usize, 9usize);
    let max_seq = 64usize;
    let q = rng.vec(n * h * d);
    let kc = rng.vec(max_seq * h * d);
    let vc = rng.vec(max_seq * h * d);
    let cache: Vec<f32> = kc.iter().chain(vc.iter()).copied().collect();
    let row_f = (p0 + n).div_ceil(16) * 16 + 16;
    let (qb, cb, sb, ob) = (
        Buf::of(card.card(), &q),
        Buf::of(card.card(), &cache),
        Buf::zeros(card.card(), h * n * row_f * 4),
        Buf::zeros(card.card(), n * h * d * 4),
    );
    let x = ((max_seq as u64) << 32) | window as u64;
    let base = MkTaskDescriptor {
        m: (h | h << 16) as u32,
        n: n as u32,
        k: d as u32,
        p0: p0 as u32,
        x,
        ..Default::default()
    };
    card.run(&[
        MkTaskDescriptor {
            opcode: OP_ATTN_SCORES,
            a: qb.at(),
            b: cb.at(),
            c: sb.at(),
            ..base
        },
        MkTaskDescriptor {
            opcode: OP_ATTN_SOFTMAX,
            c: sb.at(),
            ..base
        },
        MkTaskDescriptor {
            opcode: OP_ATTN_PV,
            a: sb.at(),
            b: cb.at(),
            c: ob.at(),
            ..base
        },
        MkTaskDescriptor {
            opcode: OP_ATTN_PV_SUM,
            c: ob.at(),
            ..base
        },
    ]);
    let got: Vec<f32> = ob.read(card.card(), n * h * d);
    let mut want = Vec::with_capacity(n * h * d);
    for t in 0..n {
        let pos = p0 + t;
        let lo = (pos + 1).saturating_sub(window);
        for head in 0..h {
            let qrow = &q[(t * h + head) * d..(t * h + head + 1) * d];
            want.extend(attend(qrow, &kc, &vc, h, head, d, lo..pos + 1));
        }
    }
    let err = rel(&got, &want);
    assert!(err < 1e-5, "relative error {err}");
}

#[test]
fn embed_gathers_rows_and_skips_absent_ids() {
    let mut card = Card::open();
    let mut rng = Rng(19);
    let (rows, k, n, stride) = (40usize, 24usize, 5usize, 3usize);
    let table = rng.vec(rows * k);
    let table16: Vec<u16> = table.iter().map(|v| bf16(*v)).collect();
    let mut ids = vec![0u32; n * stride];
    for t in 0..n {
        ids[t * stride] = (t * 7 % rows) as u32;
    }
    ids[2 * stride] = NO_EVENT;
    let base = rng.vec(n * k);
    for f32_table in [false, true] {
        for accumulate in [false, true] {
            let tb = if f32_table {
                Buf::of(card.card(), &table)
            } else {
                Buf::of(card.card(), &table16)
            };
            let (ib, cb) = (Buf::of(card.card(), &ids), Buf::of(card.card(), &base));
            let mut flags = if f32_table { FLAG_F32 } else { 0 };
            if accumulate {
                flags |= FLAG_ACCUMULATE;
            }
            card.run(&[MkTaskDescriptor {
                flags,
                a: ib.at(),
                b: tb.at(),
                c: cb.at(),
                m: stride as u32,
                n: n as u32,
                k: k as u32,
                ..task(OP_EMBED)
            }]);
            let got: Vec<f32> = cb.read(card.card(), n * k);
            let mut want = vec![0.0f32; n * k];
            for t in 0..n {
                let id = ids[t * stride];
                for j in 0..k {
                    let v = if id == NO_EVENT {
                        0.0
                    } else if f32_table {
                        table[id as usize * k + j]
                    } else {
                        from_bf16(table16[id as usize * k + j])
                    };
                    want[t * k + j] = if accumulate { base[t * k + j] + v } else { v };
                }
            }
            assert_eq!(got, want, "f32 table {f32_table}, accumulate {accumulate}");
        }
    }
}

#[test]
fn the_elementwise_ops_match_their_cpu_forms() {
    let mut card = Card::open();
    let mut rng = Rng(23);
    let (n, k) = (7usize, 40usize);
    let a = rng.vec(n * k);
    let b = rng.vec(n * k);
    let ch = rng.vec(k);
    let ch2 = rng.vec(k);
    let ids: Vec<u32> = (0..n).map(|t| if t % 3 == 1 { NO_EVENT } else { t as u32 }).collect();
    let bias16: Vec<u16> = ch.iter().map(|v| bf16(*v)).collect();
    let cases: Vec<(&str, MkTaskDescriptor, Vec<f32>)> = Vec::new();
    drop(cases);
    let run = |card: &mut Card, t: MkTaskDescriptor, c: &[f32], extra: &[(&str, Buf)]| -> Vec<f32> {
        let _ = extra;
        let cb = Buf::of(card.card(), c);
        card.run(&[MkTaskDescriptor { c: cb.at(), ..t }]);
        cb.read(card.card(), c.len())
    };
    let ab = Buf::of(card.card(), &a);
    let bb = Buf::of(card.card(), &b);
    let chb = Buf::of(card.card(), &ch);
    let ch2b = Buf::of(card.card(), &ch2);
    let ib = Buf::of(card.card(), &ids);
    let biasb = Buf::of(card.card(), &bias16);
    let dims = MkTaskDescriptor {
        n: n as u32,
        k: k as u32,
        ..Default::default()
    };
    // OP_MASK_ROWS
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_MASK_ROWS,
            a: ib.at(),
            ..dims
        },
        &a,
        &[],
    );
    let want: Vec<f32> = (0..n * k)
        .map(|i| if ids[i / k] == NO_EVENT { 0.0 } else { a[i] })
        .collect();
    assert_eq!(got, want, "mask rows");
    // OP_BIAS, BF16 and F32
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_BIAS,
            b: biasb.at(),
            ..dims
        },
        &a,
        &[],
    );
    let want: Vec<f32> = (0..n * k).map(|i| a[i] + from_bf16(bias16[i % k])).collect();
    assert_eq!(got, want, "bias bf16");
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_BIAS,
            flags: FLAG_F32,
            b: chb.at(),
            ..dims
        },
        &a,
        &[],
    );
    let want: Vec<f32> = (0..n * k).map(|i| a[i] + ch[i % k]).collect();
    assert_eq!(got, want, "bias f32");
    // OP_SILU_MUL, with and without b
    let zeros = vec![0.0f32; n * k];
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_SILU_MUL,
            a: ab.at(),
            b: bb.at(),
            ..dims
        },
        &zeros,
        &[],
    );
    let want: Vec<f32> = (0..n * k).map(|i| silu(a[i]) * b[i]).collect();
    assert!(rel(&got, &want) < 1e-6, "silu mul {}", rel(&got, &want));
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_SILU_MUL,
            a: ab.at(),
            ..dims
        },
        &zeros,
        &[],
    );
    let want: Vec<f32> = a.iter().map(|v| silu(*v)).collect();
    assert!(rel(&got, &want) < 1e-6, "silu {}", rel(&got, &want));
    // OP_GELU (erf)
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_GELU,
            a: ab.at(),
            ..dims
        },
        &zeros,
        &[],
    );
    let want: Vec<f32> = a.iter().map(|v| 0.5 * v * (1.0 + libm_erf(v / 2f32.sqrt()))).collect();
    assert!(rel(&got, &want) < 1e-6, "gelu {}", rel(&got, &want));
    // OP_SNAKE
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_SNAKE,
            a: ab.at(),
            b: chb.at(),
            x: ch2b.at(),
            ..dims
        },
        &zeros,
        &[],
    );
    let want: Vec<f32> = (0..n * k)
        .map(|i| {
            let s = (a[i] * ch[i % k].exp()).sin();
            a[i] + 1.0 / (ch2[i % k].exp() + 1e-9) * s * s
        })
        .collect();
    assert!(rel(&got, &want) < 1e-5, "snake {}", rel(&got, &want));
    // OP_SCALE_ADD
    let got = run(
        &mut card,
        MkTaskDescriptor {
            opcode: OP_SCALE_ADD,
            a: ab.at(),
            b: chb.at(),
            ..dims
        },
        &b,
        &[],
    );
    let want: Vec<f32> = (0..n * k).map(|i| b[i] + ch[i % k] * a[i]).collect();
    assert!(rel(&got, &want) < 1e-6, "scale add {}", rel(&got, &want));
    // OP_PCM
    let samples: Vec<f32> = (0..300).map(|i| (i as f32 - 150.0) / 120.0).collect();
    let sb = Buf::of(card.card(), &samples);
    let pb = Buf::zeros(card.card(), samples.len() * 2);
    card.run(&[MkTaskDescriptor {
        opcode: OP_PCM,
        a: sb.at(),
        c: pb.at(),
        n: samples.len() as u32,
        ..Default::default()
    }]);
    let got: Vec<i16> = pb.read(card.card(), samples.len());
    let want: Vec<i16> = samples
        .iter()
        .map(|v| (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16)
        .collect();
    assert_eq!(got, want, "pcm");
}

/// erf by Abramowitz and Stegun 7.1.26 is too coarse for 1e-6; this is the series and continued fraction in f64.
fn libm_erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

#[test]
fn conv1d_matches_a_causal_convolution() {
    let mut card = Card::open();
    let mut rng = Rng(29);
    // (steps, inputs, outputs, taps, dilation, groups, skip): a skip gives the tail of the same convolution.
    for (t, cin, cout, kk, dil, groups, skip) in [
        (13usize, 24usize, 20usize, 7usize, 3usize, 1usize, 0usize),
        (11, 32, 32, 7, 1, 32, 0),
        (5, 64, 48, 1, 1, 1, 0),
        (9, 16, 16, 3, 1, 1, 0),
        (22, 24, 20, 7, 3, 1, 18),
        (10, 32, 32, 7, 1, 32, 6),
    ] {
        let x = rng.vec(t * cin);
        let w = rng.vec(cout * (cin / groups) * kk);
        let bias = rng.vec(cout);
        let (xb, wb, bb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &w),
            Buf::of(card.card(), &bias),
            Buf::zeros(card.card(), t * cout * 4),
        );
        card.run(&[MkTaskDescriptor {
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            x: bb.at(),
            m: cout as u32,
            n: t as u32,
            k: cin as u32,
            p0: groups as u32,
            layer: (kk | dil << 8 | skip << 16) as u32,
            ..task(OP_CONV1D)
        }]);
        let got: Vec<f32> = yb.read(card.card(), (t - skip) * cout);
        let (ci, co) = (cin / groups, cout / groups);
        let mut want = vec![0.0f32; t * cout];
        for s in 0..t {
            for o in 0..cout {
                let g = o / co;
                let mut acc = bias[o];
                for j in 0..kk {
                    let back = (kk - 1 - j) * dil;
                    if s < back {
                        continue;
                    }
                    for i in 0..ci {
                        acc += w[(o * ci + i) * kk + j] * x[(s - back) * cin + g * ci + i];
                    }
                }
                want[s * cout + o] = acc;
            }
        }
        let err = rel(&got, &want[skip * cout..]);
        assert!(
            err < 1e-5,
            "conv {t}x{cin}->{cout} k{kk} d{dil} g{groups} skip {skip}: relative error {err}"
        );
    }
}

#[test]
fn convt1d_matches_a_cut_transposed_convolution() {
    let mut card = Card::open();
    let mut rng = Rng(31);
    // (input steps, inputs, outputs, taps, stride, skip)
    for (t, cin, cout, kk, s, skip) in [
        (5usize, 16usize, 12usize, 16usize, 8usize, 0usize),
        (4, 32, 32, 2, 2, 0),
        (3, 24, 8, 10, 5, 0),
        (6, 16, 12, 16, 8, 1),
    ] {
        let x = rng.vec(t * cin);
        let w = rng.vec(cin * cout * kk);
        let bias = rng.vec(cout);
        let (xb, wb, bb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &w),
            Buf::of(card.card(), &bias),
            Buf::zeros(card.card(), t * s * cout * 4),
        );
        card.run(&[MkTaskDescriptor {
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            x: bb.at(),
            m: cout as u32,
            n: t as u32,
            k: cin as u32,
            layer: (kk | s << 8 | skip << 16) as u32,
            ..task(OP_CONVT1D)
        }]);
        let got: Vec<f32> = yb.read(card.card(), (t - skip) * s * cout);
        let mut want = vec![0.0f32; t * s * cout];
        for (u, row) in want.chunks_mut(cout).enumerate() {
            row.copy_from_slice(&bias);
            for src in 0..t {
                let Some(j) = u.checked_sub(src * s) else { continue };
                if j >= kk {
                    continue;
                }
                for o in 0..cout {
                    for i in 0..cin {
                        row[o] += w[(i * cout + o) * kk + j] * x[src * cin + i];
                    }
                }
            }
        }
        let err = rel(&got, &want[skip * s * cout..]);
        assert!(
            err < 1e-5,
            "transposed conv k{kk} s{s} skip {skip}: relative error {err}"
        );
    }
}

#[test]
fn layernorm_matches_its_cpu_form() {
    let mut card = Card::open();
    let mut rng = Rng(37);
    let (n, k) = (5usize, 1024usize);
    let x: Vec<f32> = rng.vec(n * k).into_iter().map(|v| v * 3.0 + 0.5).collect();
    let w = rng.vec(k);
    let bias = rng.vec(k);
    let (xb, wb, bb, yb) = (
        Buf::of(card.card(), &x),
        Buf::of(card.card(), &w),
        Buf::of(card.card(), &bias),
        Buf::zeros(card.card(), n * k * 4),
    );
    card.run(&[MkTaskDescriptor {
        a: xb.at(),
        b: wb.at(),
        c: yb.at(),
        x: bb.at(),
        n: n as u32,
        k: k as u32,
        p0: 1e-6f32.to_bits(),
        ..task(OP_LAYERNORM)
    }]);
    let got: Vec<f32> = yb.read(card.card(), n * k);
    let mut want = vec![0.0f32; n * k];
    for r in 0..n {
        let row = &x[r * k..(r + 1) * k];
        let mean = row.iter().sum::<f32>() / k as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / k as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for i in 0..k {
            want[r * k + i] = (row[i] - mean) * inv * w[i] + bias[i];
        }
    }
    let err = rel(&got, &want);
    assert!(err < 1e-5, "relative error {err}");
}

/// The probabilities that the sampler draws from: the reference's processors on the CPU.
fn processed(logits: &[f32], history: &[u32], pen: f32, sup: usize, keep: usize, temp: f32, topk: usize) -> Vec<f64> {
    let mut s: Vec<f32> = logits.to_vec();
    let mut seen = vec![false; s.len()];
    for id in history {
        seen[*id as usize] = true;
    }
    for (i, v) in s.iter_mut().enumerate() {
        if seen[i] {
            *v = if *v < 0.0 { *v * pen } else { *v / pen };
        }
        if i >= sup && i != keep {
            *v = f32::NEG_INFINITY;
        }
        *v /= temp;
    }
    let mut sorted = s.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let thresh = sorted[topk - 1];
    let mx = sorted[0];
    let e: Vec<f64> = s
        .iter()
        .map(|v| {
            if *v >= thresh && *v > f32::NEG_INFINITY {
                ((v - mx) as f64).exp()
            } else {
                0.0
            }
        })
        .collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|p| p / z).collect()
}

#[test]
fn the_sampler_draws_from_the_processed_distribution_and_keeps_its_history() {
    let mut card = Card::open();
    let mut rng = Rng(41);
    let v = 3072usize;
    let logits: Vec<f32> = rng.vec(v).into_iter().map(|x| x * 4.0).collect();
    let history = [5u32, 77, 2100, 77];
    let (pen, temp, topk, sup, keep) = (1.05f32, 0.9f32, 50usize, 2048usize, 2150usize);
    let want = processed(&logits, &history, pen, sup, keep, temp, topk);
    let lb = Buf::of(card.card(), &logits);
    let draws = 4000usize;
    let mut state = vec![0u32; 4 + draws + history.len()];
    state[0] = 0x1234_5678;
    state[1] = 0x9abc_def0;
    state[3] = history.len() as u32;
    state[4..4 + history.len()].copy_from_slice(&history);
    let sb = Buf::of(card.card(), &state);
    let ob = Buf::zeros(card.card(), 4 * draws);
    // Each task samples once, so the ring is the draw count; the ring holds at most 2047 tasks.
    let x = (temp.to_bits() as u64) | ((pen.to_bits() as u64) << 32);
    let mut picks = Vec::with_capacity(draws);
    for chunk in 0..draws / 1000 {
        let tasks: Vec<MkTaskDescriptor> = (0..1000)
            .map(|i| MkTaskDescriptor {
                a: lb.at(),
                b: sb.at(),
                c: ob.at() + 4 * (chunk * 1000 + i) as u64,
                m: v as u32,
                k: topk as u32,
                p0: sup as u32,
                layer: keep as u32,
                x,
                ..task(OP_SAMPLE)
            })
            .collect();
        card.run(&tasks);
    }
    picks.extend(ob.read::<u32>(card.card(), draws));
    let state_after: Vec<u32> = sb.read(card.card(), 4);
    assert_eq!(state_after[2], draws as u32, "the draw count");
    assert_eq!(
        state_after[3],
        history.len() as u32,
        "with no FLAG_HISTORY the history keeps its length"
    );
    let mut counts = vec![0usize; v];
    for p in &picks {
        assert!(
            want[*p as usize] > 0.0,
            "the sampler drew id {p}, which has probability 0"
        );
        counts[*p as usize] += 1;
    }
    // Total variation distance between the draws and the distribution: with 4000 draws over 50 ids it is about 0.05.
    let tv: f64 = (0..v)
        .map(|i| (counts[i] as f64 / draws as f64 - want[i]).abs())
        .sum::<f64>()
        / 2.0;
    assert!(tv < 0.1, "total variation distance {tv}");

    // The same seed gives the same draws, and FLAG_HISTORY appends each id.
    let run_seeded = |card: &mut Card| -> (Vec<u32>, Vec<u32>) {
        let mut st = vec![0u32; 4 + 64];
        st[0] = 42;
        let sb = Buf::of(card.card(), &st);
        let ob = Buf::zeros(card.card(), 4 * 16);
        let tasks: Vec<MkTaskDescriptor> = (0..16)
            .map(|i| MkTaskDescriptor {
                flags: FLAG_HISTORY,
                a: lb.at(),
                b: sb.at(),
                c: ob.at() + 4 * i as u64,
                m: v as u32,
                k: topk as u32,
                p0: sup as u32,
                layer: keep as u32 | 2 << 16,
                x,
                ..task(OP_SAMPLE)
            })
            .collect();
        card.run(&tasks);
        (ob.read(card.card(), 16), sb.read(card.card(), 4 + 16))
    };
    let (first, state1) = run_seeded(&mut card);
    let (second, _) = run_seeded(&mut card);
    assert_eq!(first, second, "the same seed gives the same draws");
    assert_eq!(state1[3], 16, "the history holds each draw");
    assert_eq!(&state1[4..20], &first[..], "the history holds the ids in order");
    // While the history holds fewer than two ids, the kept id of the suppressed range is suppressed too.
    assert_ne!(first[0], keep as u32);
    assert_ne!(first[1], keep as u32);
}
