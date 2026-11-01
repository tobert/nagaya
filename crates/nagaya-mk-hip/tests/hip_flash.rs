//! Flash attention on gfx1151 (`hip/flash.hip`). Stop `nagaya.service` first (spec 3.1).
//! Run: `cargo test -p nagaya-mk-hip --features hip --test hip_flash -- --test-threads=1 --nocapture`.
#![cfg(feature = "hip")]

use std::ffi::{c_int, c_void};

use nagaya_mk_hip::attention::{AttentionShape, Mask, flash};
use nagaya_mk_hip::bf16;
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};

unsafe extern "C" {
    fn nmk_wmma_probe(a: *const c_void, b: *const c_void, c: *mut f32, stream: *mut c_void) -> c_int;
}

/// The layouts of a WMMA product that the flash kernel assumes: one 16x16x16 product against the CPU. The sum inside
/// the instruction is not exact in f32 (a sum of -43 came back as -43.000004), so each value may differ by 1e-3 of its
/// size; a wrong layout gives other values altogether.
#[test]
fn the_wmma_layouts_are_the_layouts_that_the_kernel_assumes() {
    let a: Vec<f32> = (0..256).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i * 5) % 13) as f32 - 6.0).collect();
    let ad = DeviceBuffer::from_slice(&a.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<u16>>()).unwrap();
    let bd = DeviceBuffer::from_slice(&b.iter().map(|&x| bf16::from_f32(x)).collect::<Vec<u16>>()).unwrap();
    let cd = DeviceBuffer::new(256 * 4).unwrap();
    let stream = Stream::new().unwrap();
    // SAFETY: the buffers hold 256 values each.
    let code = unsafe { nmk_wmma_probe(ad.ptr(), bd.ptr(), cd.ptr().cast(), stream.raw()) };
    assert_eq!(code, 0);
    stream.synchronize().unwrap();
    let got = cd.to_vec::<f32>().unwrap();
    for r in 0..16 {
        for c in 0..16 {
            let want: f32 = (0..16).map(|k| a[r * 16 + k] * b[k * 16 + c]).sum();
            let error = (got[r * 16 + c] - want).abs();
            assert!(
                error <= 1e-3 * want.abs().max(1.0),
                "row {r} column {c}: {} and {want}",
                got[r * 16 + c]
            );
        }
    }
}

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

/// The CPU form in f64, as in `hip_attention.rs`.
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

fn case(s: AttentionShape, seed: u64, what: &str) {
    let mut rng = Rng(seed);
    let q = rng.vec(s.lq * s.q_stride, 2.0);
    let k = rng.vec(s.lk * s.k_stride, 2.0);
    let val = rng.vec(s.lk * s.v_stride, 1.0);
    let stream = Stream::new().unwrap();
    let (qd, kd, vd) = (
        DeviceBuffer::from_slice(&q).unwrap(),
        DeviceBuffer::from_slice(&k).unwrap(),
        DeviceBuffer::from_slice(&val).unwrap(),
    );
    let od = DeviceBuffer::from_slice(&vec![0x7fc0u16; s.lq * s.out_stride]).unwrap();
    // SAFETY: the buffers hold the rows of the shape.
    unsafe { flash(&s, qd.ptr(), kd.ptr(), vd.ptr(), od.ptr(), &stream).unwrap() };
    stream.synchronize().unwrap();
    let out = od.to_vec::<u16>().unwrap();
    let want = attention_cpu(&s, &q, &k, &val);
    let width = s.heads * s.head_dim;
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut worst = 0.0f64;
    for i in 0..s.lq {
        for c in 0..width {
            let got = v(out[i * s.out_stride + c]);
            assert!(got.is_finite(), "{what}: row {i} col {c} is not written");
            worst = worst.max((got - want[i * width + c]).abs());
        }
        for c in width..s.out_stride {
            assert_eq!(
                out[i * s.out_stride + c],
                0x7fc0,
                "{what}: the padding of row {i} stays"
            );
        }
    }
    eprintln!("{what}: worst error {worst:.2e} of scale {scale:.3}");
    assert!(worst / scale < 8e-3, "{what}: worst error {worst} of scale {scale}");
}

fn shape(lq: usize, lk: usize, heads: usize, kv_heads: usize, mask: Mask) -> AttentionShape {
    let d = 128;
    let stride = (heads + 2 * kv_heads) * d;
    AttentionShape {
        lq,
        lk,
        heads,
        kv_heads,
        head_dim: d,
        q_stride: stride,
        k_stride: stride,
        v_stride: stride,
        out_stride: heads * d + 64,
        mask,
    }
}

#[test]
fn flash_matches_the_cpu_form_for_grouped_causal_heads() {
    case(shape(37, 37, 8, 2, Mask::Causal { offset: 0 }), 3, "encoder");
}

#[test]
fn flash_matches_the_cpu_form_for_a_step_over_every_key() {
    case(shape(50, 13 + 50, 4, 4, Mask::None), 5, "step");
}

#[test]
fn flash_keeps_a_causal_offset() {
    case(shape(21, 24, 4, 4, Mask::Causal { offset: 2 }), 7, "offset 2");
}

#[test]
fn flash_runs_many_workgroups_and_key_blocks() {
    case(shape(300, 347, 4, 4, Mask::None), 11, "300 queries over 347 keys");
    case(shape(200, 200, 4, 1, Mask::Causal { offset: 0 }), 13, "causal over 200");
}
