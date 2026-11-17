//! The ops of the embedding binary on a B70 card (milestone 19 step 5), each against a plain CPU form on random inputs.
//! The Gemma 4, speech and image binaries keep their bytes, so the embedding ops do not change those engines.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_embeddinggemma2_ops -- --test-threads=1`.
#![cfg(feature = "xe")]

use std::ffi::c_void;

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{EMBEDDINGGEMMA2_KERNEL, KERNEL_ENTRY};

/// The SHA-256 of each binary, and the ocloc that makes them. The speech and image binaries hold the `-I` path of the
/// kernels, so their hashes are of one checkout: a worktree gives other bytes.
const BINARIES: [(&str, &str); 3] = [
    (
        "gemma4",
        "de22b4f6c295c7827d161827f5485dbc71f42dccab054fa00f80790ed8ff2158",
    ),
    (
        "qwen3tts",
        "6be745b921c712f06a3348e2ca021040493a10dcfa61b598e796b80215437f69",
    ),
    (
        "sdxl",
        "7a8f675f3416cc66aefff9651c561586fa75c5af7acf29f9ca51bd9e46101b9a",
    ),
];
const BINARIES_OCLOC: &str = "26.35.39758";

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

    fn read(&self, card: &B70Card, n: usize) -> Vec<f32> {
        let bytes = n * 4;
        assert!(bytes <= self.bytes.max(64));
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        card.memcpy_sync(stage as *mut c_void, self.ptr as *const c_void, bytes);
        let mut out = vec![0.0f32; n];
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
        let cluster = L0Cluster::init_test(EMBEDDINGGEMMA2_KERNEL, KERNEL_ENTRY);
        let ring = Ring::new(&cluster.cards[0], 1);
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

    fn buf(&self, data: &[f32]) -> Buf {
        Buf::of(self.card(), data)
    }
}

/// The largest absolute difference, relative to the largest absolute value of `want`.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1e-30f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

#[test]
fn the_other_binaries_keep_their_bytes() {
    if nagaya_mk_bmg::OCLOC_VERSION != BINARIES_OCLOC {
        eprintln!(
            "skipped: the hashes are of ocloc {BINARIES_OCLOC}, and this build has ocloc {}",
            nagaya_mk_bmg::OCLOC_VERSION
        );
        return;
    }
    use sha2::{Digest, Sha256};
    let kernels = [
        nagaya_mk_bmg::GEMMA4_KERNEL,
        nagaya_mk_bmg::QWEN3TTS_KERNEL,
        nagaya_mk_bmg::SDXL_KERNEL,
    ];
    for ((name, want), kernel) in BINARIES.iter().zip(kernels) {
        let digest: String = Sha256::digest(kernel).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(&digest, want, "the embedding ops changed the {name} binary");
    }
}

/// Attention in both directions on the CPU, as `OP_BIATTN` computes it.
#[allow(clippy::too_many_arguments)]
fn biattn_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    kv: usize,
    d: usize,
    window: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * heads * d];
    for t in 0..n {
        for h in 0..heads {
            let g = h / (heads / kv);
            let keys: Vec<usize> = (0..n).filter(|&s| window == 0 || t.abs_diff(s) <= window).collect();
            let scores: Vec<f32> = keys
                .iter()
                .map(|&s| {
                    let qr = &q[(t * heads + h) * d..][..d];
                    let kr = &k[(s * kv + g) * d..][..d];
                    scale * qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            let max = scores.iter().fold(f32::NEG_INFINITY, |m, s| m.max(*s));
            let p: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = p.iter().sum();
            for j in 0..d {
                out[(t * heads + h) * d + j] = keys
                    .iter()
                    .zip(&p)
                    .map(|(&s, p)| p * v[(s * kv + g) * d + j])
                    .sum::<f32>()
                    / sum;
            }
        }
    }
    out
}

#[test]
fn biattn_attends_both_ways_within_its_window_with_grouped_heads() {
    let mut card = Card::open();
    let mut rng = Rng(5);
    // The sliding layers of EmbeddingGemma 2 (4 heads on 2 KV heads of 256), and its full layers (on 1 KV head of 512).
    for (n, heads, kv, d, window, scale) in [
        (37usize, 4usize, 2usize, 256usize, 5usize, 1.0f32),
        (21, 4, 1, 512, 0, 0.5),
        (1, 4, 2, 256, 3, 1.0),
        (9, 2, 2, 16, 20, 1.0),
    ] {
        let q = rng.vec(n * heads * d);
        let k = rng.vec(n * kv * d);
        let v = rng.vec(n * kv * d);
        let (qb, kb, vb) = (card.buf(&q), card.buf(&k), card.buf(&v));
        let yb = Buf::zeros(card.card(), n * heads * d * 4);
        card.run(&[tasks::biattn(
            qb.at(),
            kb.at(),
            vb.at(),
            yb.at(),
            n as u32,
            heads as u32,
            kv as u32,
            d as u32,
            window as u32,
            scale,
        )]);
        let got = yb.read(card.card(), n * heads * d);
        let want = biattn_cpu(&q, &k, &v, n, heads, kv, d, window, scale);
        let err = rel(&got, &want);
        assert!(
            err < 1e-5,
            "n {n}, heads {heads}/{kv}, d {d}, window {window}: relative error {err}"
        );
    }
}

#[test]
fn the_glu_of_the_embedding_binary_is_geglu_with_gelu_tanh() {
    let mut card = Card::open();
    let mut rng = Rng(7);
    let (m, k) = (48usize, 64usize);
    let w: Vec<u16> = rng.vec(2 * m * k).into_iter().map(bf16).collect();
    let wb = Buf::of(card.card(), &w);
    for n in [1usize, 3, 20] {
        let x = rng.vec(n * k);
        let xb = card.buf(&x);
        let yb = Buf::zeros(card.card(), n * m * 4);
        card.run(&[MkTaskDescriptor {
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            opcode: OP_GEGLU,
            ..Default::default()
        }]);
        let got = yb.read(card.card(), n * m);
        let mut want = vec![0.0f32; n * m];
        for t in 0..n {
            for r in 0..m {
                let dot = |row: usize| -> f32 { (0..k).map(|i| from_bf16(w[row * k + i]) * x[t * k + i]).sum() };
                want[t * m + r] = gelu_tanh(dot(r)) * dot(m + r);
            }
        }
        let err = rel(&got, &want);
        assert!(err < 1e-5, "{n} tokens: relative error {err}");
    }
}

#[test]
fn gelu_mul_gates_with_rows_of_a_wider_tensor() {
    let mut card = Card::open();
    let mut rng = Rng(9);
    let (n, k) = (5usize, 48usize);
    for stride in [0usize, 3 * k] {
        let b_cols = if stride == 0 { k } else { stride };
        let a = rng.vec(n * k);
        let b = rng.vec(n * b_cols);
        let (ab, bb) = (card.buf(&a), card.buf(&b));
        let yb = Buf::zeros(card.card(), n * k * 4);
        // With a stride, the gate reads the second block of k values of each row of b.
        let offset = if stride == 0 { 0 } else { k };
        card.run(&[tasks::gelu_mul(
            ab.at(),
            bb.at() + 4 * offset as u64,
            yb.at(),
            n as u32,
            k as u32,
            stride as u32,
        )]);
        let got = yb.read(card.card(), n * k);
        let want: Vec<f32> = (0..n * k)
            .map(|i| gelu_tanh(a[i]) * b[(i / k) * b_cols + offset + i % k])
            .collect();
        let err = rel(&got, &want);
        assert!(err < 1e-6, "stride {stride}: relative error {err}");
    }
}

#[test]
fn scale_multiplies_in_place() {
    let mut card = Card::open();
    let mut rng = Rng(13);
    let (n, k) = (7usize, 512usize);
    let x = rng.vec(n * k);
    let xb = card.buf(&x);
    card.run(&[tasks::scale(xb.at(), xb.at(), n as u32, k as u32, 22.627417)]);
    let got = xb.read(card.card(), n * k);
    let want: Vec<f32> = x.iter().map(|v| v * 22.627417).collect();
    assert_eq!(got, want, "a scale in f32 is exact to the bit");
}

#[test]
fn qk_rope_rotates_the_whole_head_at_both_thetas_far_into_the_input() {
    let mut card = Card::open();
    let mut rng = Rng(17);
    let rope = |x: &[f32], d: usize, pos: f32, theta: f32| -> Vec<f32> {
        let h = d / 2;
        let mut y = vec![0.0f32; d];
        for i in 0..h {
            let inv = 1.0 / theta.powf((2 * i) as f32 / d as f32);
            let (s, c) = (pos * inv).sin_cos();
            y[i] = x[i] * c - x[i + h] * s;
            y[i + h] = x[i + h] * c + x[i] * s;
        }
        y
    };
    // The sliding layers rotate heads of 256 at theta 1e4; the full layers rotate heads of 512 at 1e6.
    for (d, heads, full, theta) in [(256usize, 2usize, false, 1e4f32), (512, 1, true, 1e6)] {
        let tokens = 3;
        let p0 = 2045u32;
        let n = tokens * heads;
        let x = rng.vec(n * d);
        let w: Vec<u16> = rng.vec(d).into_iter().map(bf16).collect();
        let (xb, wb) = (card.buf(&x), Buf::of(card.card(), &w));
        card.run(&[tasks::qk_rope(
            xb.at(),
            wb.at(),
            n as u32,
            heads as u32,
            d as u32,
            p0,
            full,
            true,
            1e-6,
            1.0,
        )]);
        let got = xb.read(card.card(), n * d);
        let mut want = Vec::with_capacity(n * d);
        for r in 0..n {
            let row = &x[r * d..(r + 1) * d];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / d as f32 + 1e-6).sqrt();
            let normed: Vec<f32> = (0..d).map(|i| row[i] * inv * from_bf16(w[i])).collect();
            want.extend(rope(&normed, d, (p0 + (r / heads) as u32) as f32, theta));
        }
        let err = rel(&got, &want);
        assert!(err < 1e-3, "heads of {d}: relative error {err}");
    }
}
