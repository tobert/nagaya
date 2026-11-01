//! The ops of the VAE decoder on gfx1151, each against a CPU form (milestone 22 step 4). Stop `nagaya.service` first
//! (spec 3.1). Run: `cargo test -p nagaya-mk-hip --features hip --test hip_vae -- --test-threads=1`.
#![cfg(feature = "hip")]

use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{Gemms, lineage};
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::plan::PlanTable;
use nagaya_mk_hip::vae::{self, Conv3x3};

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

fn v(h: u16) -> f32 {
    bf16::to_f32(h)
}

fn r(x: f32) -> f32 {
    bf16::round(x)
}

fn upload<T: Copy>(data: &[T]) -> DeviceBuffer {
    DeviceBuffer::from_slice(data).unwrap()
}

fn gemms() -> Gemms {
    Gemms::new(PlanTable::in_memory(&lineage().unwrap()), 32 << 20).unwrap()
}

/// The CPU form of a 3 by 3 conv with a padding of 1 over a source that `up` repeats.
#[allow(clippy::too_many_arguments)]
fn conv_cpu(
    x: &[u16],
    cin: usize,
    height: usize,
    width: usize,
    up: usize,
    w: &[u16],
    bias: &[u16],
    cout: usize,
) -> Vec<f64> {
    let src_width = width / up;
    let mut out = vec![0.0; height * width * cout];
    for y in 0..height {
        for x0 in 0..width {
            for o in 0..cout {
                let mut sum = f64::from(v(bias[o]));
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (py, px) = (y as i64 + ky as i64 - 1, x0 as i64 + kx as i64 - 1);
                        if py < 0 || px < 0 || py >= height as i64 || px >= width as i64 {
                            continue;
                        }
                        let s = ((py as usize / up) * src_width + px as usize / up) * cin;
                        for c in 0..cin {
                            let wi = ((o * 3 + ky) * 3 + kx) * cin + c;
                            sum += f64::from(v(w[wi])) * f64::from(v(x[s + c]));
                        }
                    }
                }
                out[(y * width + x0) * cout + o] = sum;
            }
        }
    }
    out
}

fn rel(got: &[u16], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let worst = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((f64::from(v(*g)) - w).abs()));
    worst / scale
}

#[allow(clippy::too_many_arguments)]
fn conv_case(
    cin: usize,
    cout: usize,
    height: usize,
    width: usize,
    up: usize,
    residual: bool,
    scratch: usize,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let x = rng.vec((height / up) * (width / up) * cin, 1.0);
    let w = rng.vec(cout * 9 * cin, 0.5);
    let bias = rng.vec(cout, 1.0);
    let res = rng.vec(height * width * cout, 2.0);
    let stream = Stream::new().unwrap();
    let mut gemms = gemms();
    let (xd, wd, bd) = (upload(&x), upload(&w), upload(&bias));
    let out = upload(&res);
    let scratch = DeviceBuffer::new(scratch).unwrap();
    let conv = Conv3x3 {
        weight: wd.ptr(),
        bias: bd.ptr(),
        inputs: cin,
        outputs: cout,
    };
    let residual = residual.then_some(out.ptr().cast_const());
    // SAFETY: the buffers hold the image, the weight, the bias and the output.
    unsafe {
        conv.run(
            &mut gemms,
            xd.ptr(),
            height,
            width,
            up,
            residual,
            out.ptr(),
            &scratch,
            &stream,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let mut want = conv_cpu(&x, cin, height, width, up, &w, &bias, cout);
    if residual.is_some() {
        for (w, r) in want.iter_mut().zip(&res) {
            *w += f64::from(v(*r));
        }
    }
    let error = rel(&out.to_vec::<u16>().unwrap(), &want);
    eprintln!(
        "conv {cin}->{cout} {height}x{width} up {up} residual {}: {error:.2e}",
        residual.is_some()
    );
    assert!(error < 4e-3, "relative error {error}");
}

#[test]
fn a_conv_with_a_bias_matches_the_cpu_form() {
    conv_case(24, 40, 13, 17, 1, false, 64 << 20, 3);
}

#[test]
fn a_conv_after_a_nearest_upsample_reads_each_source_pixel_four_times() {
    conv_case(16, 24, 14, 10, 2, false, 64 << 20, 5);
}

#[test]
fn a_conv_adds_a_residual_and_runs_in_bands() {
    // The scratch holds 3 rows of the copy, so the 11 rows go in bands of 3, 3, 3 and 2.
    let (cin, width) = (8, 9);
    conv_case(cin, 12, 11, width, 1, true, 3 * width * 9 * cin * 2, 7);
}

#[test]
fn the_vae_norm_divides_by_the_norm_and_scales_by_the_root_of_the_channels() {
    let (pixels, channels) = (33, 288);
    let mut rng = Rng(11);
    let x = rng.vec(pixels * channels, 3.0);
    let gamma = rng.vec(channels, 1.5);
    let stream = Stream::new().unwrap();
    for silu in [false, true] {
        let (xd, gd) = (upload(&x), upload(&gamma));
        let out = upload(&vec![0u16; pixels * channels]);
        // SAFETY: the buffers hold the pixels.
        unsafe { vae::norm(xd.ptr(), gd.ptr(), out.ptr(), pixels, channels, silu, &stream).unwrap() };
        stream.synchronize().unwrap();
        let got = out.to_vec::<u16>().unwrap();
        for p in 0..pixels {
            let row = &x[p * channels..(p + 1) * channels];
            let norm = row.iter().map(|&h| f64::from(v(h)).powi(2)).sum::<f64>().sqrt() as f32;
            for c in 0..channels {
                let mut want = r(r(r(v(row[c]) / norm) * (channels as f32).sqrt()) * v(gamma[c]));
                if silu {
                    want /= 1.0 + (-want).exp();
                }
                let (g, w) = (got[p * channels + c], bf16::from_f32(want));
                assert!(
                    g.abs_diff(w) <= 1,
                    "silu {silu} pixel {p} channel {c}: {} vs {}",
                    v(g),
                    v(w)
                );
            }
        }
    }
}

/// The CPU form of DupUp3D with `first_chunk`, from the steps of `vae.py:112-134`.
fn dup_up_cpu(src: &[u16], cin: usize, h: usize, w: usize, factor_t: usize, cout: usize) -> Vec<u16> {
    let repeats = cout * factor_t * 4 / cin;
    let mut out = vec![0u16; 4 * h * w * cout];
    for y in 0..2 * h {
        for x in 0..2 * w {
            for c in 0..cout {
                // repeat_interleave, then the view (out, t, sy, sx), then the last t.
                let flat = ((c * factor_t + factor_t - 1) * 2 + y % 2) * 2 + x % 2;
                out[(y * 2 * w + x) * cout + c] = src[((y / 2) * w + x / 2) * cin + flat / repeats];
            }
        }
    }
    out
}

#[test]
fn dup_up_add_adds_the_shortcut_of_each_block_form() {
    let (h, w) = (3, 5);
    let stream = Stream::new().unwrap();
    // (in, out, factor_t), the forms of the decoder's up blocks: block 0 keeps its channels with a time upsample (8
    // repeats, so the time index reads the same channel); block 2 halves them with a time upsample (4 repeats, so
    // the kept time index picks the channel); block 3 halves them with no time upsample.
    for (cin, cout, factor_t) in [(16, 16, 2), (16, 8, 2), (16, 8, 1), (12, 12, 2)] {
        let mut rng = Rng(13 + cin as u64);
        let src = rng.vec(h * w * cin, 1.0);
        let dst = rng.vec(4 * h * w * cout, 1.0);
        let (sd, dd) = (upload(&src), upload(&dst));
        // SAFETY: the buffers hold both images.
        unsafe { vae::dup_up_add(sd.ptr(), cin, h, w, factor_t, dd.ptr(), cout, &stream).unwrap() };
        stream.synchronize().unwrap();
        let got = dd.to_vec::<u16>().unwrap();
        let shortcut = dup_up_cpu(&src, cin, h, w, factor_t, cout);
        for i in 0..got.len() {
            assert_eq!(
                got[i],
                bf16::from_f32(v(dst[i]) + v(shortcut[i])),
                "{cin}->{cout} t{factor_t}: value {i}"
            );
        }
    }
}

#[test]
fn the_latent_affine_and_the_bytes_round_as_the_reference_does() {
    let channels = 4;
    let x: Vec<u16> = [-3.0, -1.0, -0.5, 0.0, 0.25, 1.0, 2.0, 0.7]
        .iter()
        .map(|&f| bf16::from_f32(f))
        .collect();
    let scale: Vec<u16> = [2.0, 0.5, 1.0, 3.0].iter().map(|&f| bf16::from_f32(f)).collect();
    let shift: Vec<u16> = [0.5, -1.0, 0.0, 0.125].iter().map(|&f| bf16::from_f32(f)).collect();
    let stream = Stream::new().unwrap();
    let (xd, sd, hd) = (upload(&x), upload(&scale), upload(&shift));
    let out = upload(&vec![0u16; x.len()]);
    let bytes = upload(&vec![0u8; x.len()]);
    // SAFETY: the buffers hold the values.
    unsafe {
        vae::channel_affine(xd.ptr(), sd.ptr(), hd.ptr(), out.ptr(), x.len(), channels, &stream).unwrap();
        vae::to_bytes(xd.ptr(), bytes.ptr(), x.len(), &stream).unwrap();
    }
    stream.synchronize().unwrap();
    let got = out.to_vec::<u16>().unwrap();
    for i in 0..x.len() {
        let c = i % channels;
        let want = r(r(v(x[i]) * v(scale[c])) + v(shift[c]));
        assert_eq!(v(got[i]), want, "affine {i}");
    }
    // -3 and -1 clamp to 0; -0.5 gives 0.25 * 255 = 63.75, so 64; 0 gives 127.5, a tie that rounds to the even 128;
    // 0.25 gives 0.625 * 255 = 159.375; 1 and 2 give 255; 0.7 is 0.69921875 in bf16, so u rounds to 0.8515625 and
    // gives 217.15.
    assert_eq!(bytes.to_vec::<u8>().unwrap(), [0, 0, 64, 128, 159, 255, 255, 217]);
}
