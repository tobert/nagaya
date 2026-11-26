//! Each kernel of `src` against the frozen copy in `reference/kernels.rs`, bit for bit. A faster kernel must give the
//! bits of the copy for each shape and thread count (spec 8.5). The shapes have tails that are not multiples of 8 or
//! of a block, and the values have zeros and negative zeros, so a changed order of a sum or a changed first term
//! shows here.

#[allow(dead_code)]
#[path = "reference/kernels.rs"]
mod reference;

mod levels;

use levels::each_level;
use nagaya_family_beatthis::plan::Axis;
use nagaya_mk_beatthis_cpu::kernels;

/// Values in [-1, 1) from a linear congruential generator, with each seventh value a zero of either sign.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match i % 7 {
                3 => 0.0,
                5 => -0.0,
                _ => (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0,
            }
        })
        .collect()
}

fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!(
            "{what}: value {i} is {:e}, and the reference gives {:e}",
            got[i], want[i]
        );
    }
}

const THREADS: [usize; 4] = [1, 3, 8, 16];

#[test]
fn linear_gives_the_bits_of_the_reference() {
    each_level("linear_gives_the_bits_of_the_reference", |level| {
        // (rows, din, dout): odd rows and outputs, inputs not a multiple of 8, and the shapes of the plan.
        for (rows, din, dout) in [
            (1, 1, 1),
            (7, 37, 11),
            (13, 64, 9),
            (33, 128, 512),
            (9, 512, 2),
            (21, 96, 384),
        ] {
            let mut x = values(rows * din, 1);
            // A row of zeros against negative weights: each product is a zero of either sign, and the sum must start
            // at +0 as the reference does.
            x[..din].fill(-0.0);
            let w = values(dout * din, 2);
            let b = values(dout, 3);
            for bias in [None, Some(b.as_slice())] {
                let mut want = vec![0.0f32; rows * dout];
                reference::linear(&x, &mut want, din, dout, &w, bias, 1);
                for threads in THREADS {
                    let mut y = vec![f32::NAN; rows * dout];
                    kernels::linear(&x, &mut y, din, dout, &w, bias, threads);
                    same(
                        &y,
                        &want,
                        &format!("linear {rows}x{din}x{dout}, threads {threads}, {level:?}"),
                    );
                }
            }
        }
    });
}

#[test]
fn conv_gives_the_bits_of_the_reference() {
    each_level("conv_gives_the_bits_of_the_reference", |level| {
        // (frames, f_in, cin, cout, kf, kt, stride): the stem, the blocks, and odd shapes.
        for (frames, f_in, cin, cout, kf, kt, stride) in [
            (5, 8, 3, 4, 2, 3, 2),
            (11, 128, 1, 32, 4, 3, 4),
            (9, 32, 32, 64, 2, 3, 2),
            (7, 8, 13, 5, 2, 3, 2),
            (1, 4, 9, 3, 2, 3, 2),
        ] {
            let f_out = (f_in - kf) / stride + 1;
            let x = values(frames * f_in * cin, 4);
            let k = values(cout * kf * kt * cin, 5);
            let mut want = vec![0.0f32; frames * f_out * cout];
            reference::conv(&x, &mut want, frames, f_in, cin, cout, kf, kt, stride, &k, 1);
            for threads in THREADS {
                let mut y = vec![f32::NAN; want.len()];
                kernels::conv(&x, &mut y, frames, f_in, cin, cout, kf, kt, stride, &k, threads);
                same(
                    &y,
                    &want,
                    &format!("conv {frames} {f_in} {cin} {cout}, threads {threads}, {level:?}"),
                );
            }
        }
    });
}

#[test]
fn attention_gives_the_bits_of_the_reference() {
    each_level("attention_gives_the_bits_of_the_reference", |level| {
        // (a, b, heads, hd): odd lengths, a head dim that is not a multiple of 8, and the shapes of the plan.
        for (a, b, heads, hd) in [
            (6, 4, 2, 8),
            (9, 5, 3, 12),
            (37, 3, 1, 32),
            (19, 1, 4, 32),
            (5, 7, 2, 6),
        ] {
            let width = heads * hd;
            let qkv: Vec<f32> = values(a * b * 3 * width, 10).iter().map(|v| v * 4.0).collect();
            let gates = values(a * b * heads, 11);
            let freqs: Vec<f32> = (0..hd / 2)
                .map(|k| 1.0 / 10_000f32.powf(2.0 * k as f32 / hd as f32))
                .collect();
            for along in [Axis::A, Axis::B] {
                let mut want = vec![0.0f32; a * b * width];
                reference::attention(&qkv, &gates, &mut want, a, b, along, heads, hd, &freqs, 1);
                for threads in THREADS {
                    let mut out = vec![f32::NAN; want.len()];
                    kernels::attention(&qkv, &gates, &mut out, a, b, along, heads, hd, &freqs, threads);
                    same(
                        &out,
                        &want,
                        &format!("attention {a}x{b} {heads}x{hd} {along:?}, threads {threads}, {level:?}"),
                    );
                }
            }
        }
    });
}

#[test]
fn the_elementwise_kernels_give_the_bits_of_the_reference() {
    each_level("the_elementwise_kernels_give_the_bits_of_the_reference", |level| {
        for (rows, dim) in [(1, 1), (3, 10), (17, 128), (5, 2051)] {
            let x: Vec<f32> = values(rows * dim, 6).iter().map(|v| v * 6.0).collect();
            let scale = values(dim, 7);
            let shift = values(dim, 8);
            let gamma = values(dim, 9);
            for threads in THREADS {
                let what = format!("{rows}x{dim}, threads {threads}, {level:?}");
                let (mut y, mut want) = (vec![f32::NAN; rows * dim], vec![0.0f32; rows * dim]);
                kernels::affine(&x, &mut y, dim, &scale, &shift, threads);
                reference::affine(&x, &mut want, dim, &scale, &shift, 1);
                same(&y, &want, &format!("affine {what}"));

                kernels::rms_norm(&x, &mut y, dim, &gamma, threads);
                reference::rms_norm(&x, &mut want, dim, &gamma, 1);
                same(&y, &want, &format!("rms_norm {what}"));

                let (mut g, mut want) = (x.clone(), x.clone());
                kernels::gelu(&mut g, threads);
                reference::gelu(&mut want, 1);
                same(&g, &want, &format!("gelu {what}"));

                let (mut acc, mut want) = (x.clone(), x.clone());
                kernels::add_into(&mut acc, &scale.repeat(rows), threads);
                reference::add_into(&mut want, &scale.repeat(rows), 1);
                same(&acc, &want, &format!("add_into {what}"));
            }
        }
    });
}

/// `exps` against `libm::expf` on a sweep of the bit patterns of f32, on each pattern near a bound of a branch of
/// `expf`, and on the specials. The sweep holds each sign, NaN, the infinities, and the subnormals.
#[test]
fn exps_gives_the_bits_of_libm() {
    each_level("exps_gives_the_bits_of_libm", |level| {
        let mut xs: Vec<f32> = (0..=u32::MAX).step_by(251).map(f32::from_bits).collect();
        for bound in [
            0x3900_0000u32,
            0x3eb1_7218,
            0x3f85_1592,
            0x42ae_ac50,
            0x42b1_7218,
            0x42cf_f1b5,
            0x7f80_0000,
        ] {
            for sign in [0, 0x8000_0000u32] {
                for d in 0..64 {
                    xs.push(f32::from_bits((bound | sign) - 32 + d));
                }
            }
        }
        xs.extend([
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
        ]);
        // Each value at each place in a block of 4, and in the tail.
        for start in 0..5 {
            for shift in [0.0f32, 3.5, -20.0] {
                let mut got = xs[start..].to_vec();
                kernels::exps(&mut got, shift);
                for (i, (&x, &g)) in xs[start..].iter().zip(&got).enumerate() {
                    // black_box keeps the compiler from folding `x - 0.0` to `x`, which would keep a signaling NaN.
                    let want = libm::expf(x - std::hint::black_box(shift));
                    assert!(
                        g.to_bits() == want.to_bits(),
                        "{level:?}: exps of {x:e} ({:#010x}) - {shift}: {g:e} ({:#010x}), and libm gives {want:e} ({:#010x}) (value {i})",
                        x.to_bits(),
                        g.to_bits(),
                        want.to_bits()
                    );
                }
            }
        }
    });
}
