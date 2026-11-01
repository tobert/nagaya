//! Attention on gfx1151 against a CPU form in f64 (milestone 22 step 4). The probabilities round to bf16 before the
//! values, so a result may differ from the CPU form by about one bf16 step of the output scale.
//! Stop `nagaya.service` first (spec 3.1). Run: `cargo test -p nagaya-mk-hip --features hip --test hip_attention`.
#![cfg(feature = "hip")]

use nagaya_mk_hip::attention::{Attention, AttentionShape, Mask};
use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{Gemms, lineage};
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::plan::PlanTable;

struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bf16::from_f32(self.f() * scale)).collect()
    }
}

fn v(h: u16) -> f64 {
    f64::from(bf16::to_f32(h))
}

/// The CPU form: `out[q, h] = softmax(q_h k_g^T / sqrt(d)) v_g` with `g = h / (heads / kv_heads)`.
fn attention_cpu(s: &AttentionShape, q: &[u16], k: &[u16], val: &[u16]) -> Vec<f64> {
    let d = s.head_dim;
    let group = s.heads / s.kv_heads;
    let mut out = vec![0.0; s.lq * s.heads * d];
    for h in 0..s.heads {
        let g = h / group;
        for i in 0..s.lq {
            let visible = match s.mask {
                Mask::None => s.lk,
                Mask::Causal { offset } => (i + offset + 1).min(s.lk),
            };
            let scores: Vec<f64> = (0..visible)
                .map(|j| {
                    (0..d)
                        .map(|t| v(q[i * s.q_stride + h * d + t]) * v(k[j * s.k_stride + g * d + t]))
                        .sum::<f64>()
                        / (d as f64).sqrt()
                })
                .collect();
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for t in 0..d {
                out[(i * s.heads + h) * d + t] = (0..visible)
                    .map(|j| weights[j] / sum * v(val[j * s.v_stride + g * d + t]))
                    .sum();
            }
        }
    }
    out
}

struct Run {
    out: Vec<u16>,
    want: Vec<f64>,
}

fn run(s: AttentionShape, capacity: usize, seed: u64) -> Run {
    let mut rng = Rng(seed);
    let q = rng.vec(s.lq * s.q_stride, 2.0);
    let k = rng.vec(s.lk * s.k_stride, 2.0);
    let val = rng.vec(s.lk * s.v_stride, 1.0);
    let out = launch(&s, capacity, &q, &k, &val);
    Run {
        out,
        want: attention_cpu(&s, &q, &k, &val),
    }
}

fn launch(s: &AttentionShape, capacity: usize, q: &[u16], k: &[u16], val: &[u16]) -> Vec<u16> {
    let stream = Stream::new().unwrap();
    let mut gemms = Gemms::new(PlanTable::in_memory(&lineage().unwrap()), 32 << 20).unwrap();
    let attention = Attention::new(capacity).unwrap();
    let (qd, kd, vd) = (
        DeviceBuffer::from_slice(q).unwrap(),
        DeviceBuffer::from_slice(k).unwrap(),
        DeviceBuffer::from_slice(val).unwrap(),
    );
    let od = DeviceBuffer::from_slice(&vec![0x7fc0u16; s.lq * s.out_stride]).unwrap();
    // SAFETY: the buffers hold the rows of the shape.
    unsafe {
        attention
            .run(&mut gemms, s, qd.ptr(), kd.ptr(), vd.ptr(), od.ptr(), &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();
    od.to_vec::<u16>().unwrap()
}

fn compare(s: &AttentionShape, r: &Run, what: &str) {
    let width = s.heads * s.head_dim;
    let scale = r.want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut worst = 0.0f64;
    for i in 0..s.lq {
        for c in 0..width {
            let got = v(r.out[i * s.out_stride + c]);
            assert!(got.is_finite(), "{what}: row {i} col {c} is not written");
            worst = worst.max((got - r.want[i * width + c]).abs());
        }
        for c in width..s.out_stride {
            assert_eq!(
                r.out[i * s.out_stride + c],
                0x7fc0,
                "{what}: the padding of row {i} stays"
            );
        }
    }
    eprintln!("{what}: worst error {:.2e} of scale {scale:.3}", worst);
    assert!(worst / scale < 8e-3, "{what}: worst error {worst} of scale {scale}");
}

/// The encoder: 8 query heads over 2 key heads, causal, from one fused q, k and v row.
fn encoder(lq: usize) -> AttentionShape {
    let d = 128;
    AttentionShape {
        lq,
        lk: lq,
        heads: 8,
        kv_heads: 2,
        head_dim: d,
        q_stride: 12 * d,
        k_stride: 12 * d,
        v_stride: 12 * d,
        out_stride: 8 * d + 64,
        mask: Mask::Causal { offset: 0 },
    }
}

#[test]
fn grouped_causal_attention_matches_the_cpu_form() {
    let s = encoder(37);
    compare(&s, &run(s, 1 << 20, 3), "encoder");
}

#[test]
fn a_step_attends_from_the_latent_to_every_key() {
    let d = 128;
    let s = AttentionShape {
        lq: 50,
        lk: 13 + 50,
        heads: 4,
        kv_heads: 4,
        head_dim: d,
        q_stride: 3 * 4 * d,
        k_stride: 4 * d,
        v_stride: 4 * d,
        out_stride: 4 * d,
        mask: Mask::None,
    };
    compare(&s, &run(s, 1 << 20, 5), "step");
}

#[test]
fn a_small_scratch_takes_one_head_in_each_pass() {
    let s = encoder(29);
    compare(&s, &run(s, 29 * 29, 7), "one head in each pass");
    compare(&s, &run(s, 3 * 29 * 29, 7), "three heads, then one");
}

#[test]
fn a_query_never_reads_a_masked_key() {
    // The condition rows of image_begin: causal among themselves, with an offset of 2.
    let mut s = encoder(21);
    s.lk = 24;
    s.mask = Mask::Causal { offset: 2 };
    let mut rng = Rng(11);
    let q = rng.vec(s.lq * s.q_stride, 2.0);
    let k = rng.vec(s.lk * s.k_stride, 2.0);
    let val = rng.vec(s.lk * s.v_stride, 1.0);
    let first = launch(&s, 1 << 20, &q, &k, &val);
    compare(
        &s,
        &Run {
            out: first.clone(),
            want: attention_cpu(&s, &q, &k, &val),
        },
        "offset 2",
    );
    // Row 0 sees keys 0 to 2. Change key and value 3 to 23 for every head: row 0 must keep its bits.
    let (mut k2, mut v2) = (k.clone(), val.clone());
    for j in 3..s.lk {
        for t in 0..s.k_stride {
            k2[j * s.k_stride + t] = bf16::from_f32(9.0);
            v2[j * s.v_stride + t] = bf16::from_f32(-9.0);
        }
    }
    let second = launch(&s, 1 << 20, &q, &k2, &v2);
    assert_eq!(first[..s.out_stride], second[..s.out_stride], "row 0 read a masked key");
}
