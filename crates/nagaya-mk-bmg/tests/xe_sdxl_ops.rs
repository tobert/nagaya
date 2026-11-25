//! The ops of the image binary on a B70 card, each against a plain CPU form on random inputs.
//! One test also runs the real `conv_in` of the SDXL UNet on the input of the oracle (`tools/sdxl_golden.sh`).
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_sdxl_ops -- --test-threads=1`.
#![cfg(feature = "xe")]

use std::ffi::c_void;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{KERNEL_ENTRY, SDXL_KERNEL};

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

/// The F16 bits of `x`, rounded to the nearest even. The test values are normal F16 numbers.
fn f16(x: f32) -> u16 {
    let u = x.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xFF) as i32 - 127 + 15;
    if x == 0.0 {
        return sign;
    }
    assert!((1..31).contains(&exp), "{x} is not a normal F16 number");
    let mant = u & 0x7F_FFFF;
    let mut half = ((exp as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1FFF;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
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
        let cluster = L0Cluster::init_test(SDXL_KERNEL, KERNEL_ENTRY);
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

/// The shape of a convolution: images, input height and width, inputs, outputs, kernel, stride, padding. With `end`,
/// the padding is at the bottom and the right only (`FLAG_PAD_END`).
#[derive(Clone, Copy, Debug)]
struct Conv {
    b: usize,
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
    end: bool,
}

impl Conv {
    /// The padding at the top and the left.
    fn start(&self) -> usize {
        if self.end { 0 } else { self.pad }
    }

    fn out(&self) -> (usize, usize) {
        (
            (self.h + self.start() + self.pad - self.kernel) / self.stride + 1,
            (self.w + self.start() + self.pad - self.kernel) / self.stride + 1,
        )
    }

    fn task(&self) -> MkTaskDescriptor {
        MkTaskDescriptor {
            n: (self.h | self.w << 16) as u32,
            m: self.cout as u32,
            k: self.cin as u32,
            p0: self.b as u32,
            layer: (self.kernel | self.kernel << 8 | self.stride << 16 | self.pad << 24) as u32,
            ..task(OP_CONV2D)
        }
    }
}

/// The plain convolution, pixel-major, with weights [cout][cin][k][k].
fn conv_cpu(s: Conv, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let (ho, wo) = s.out();
    let mut y = vec![0.0f32; s.b * ho * wo * s.cout];
    for b in 0..s.b {
        for yo in 0..ho {
            for xo in 0..wo {
                for o in 0..s.cout {
                    let mut acc = bias.map_or(0.0, |bias| bias[o]);
                    for ky in 0..s.kernel {
                        for kx in 0..s.kernel {
                            let yy = (yo * s.stride + ky) as isize - s.start() as isize;
                            let xx = (xo * s.stride + kx) as isize - s.start() as isize;
                            if yy < 0 || xx < 0 || yy >= s.h as isize || xx >= s.w as isize {
                                continue;
                            }
                            let at = ((b * s.h + yy as usize) * s.w + xx as usize) * s.cin;
                            for i in 0..s.cin {
                                acc += w[((o * s.cin + i) * s.kernel + ky) * s.kernel + kx] * x[at + i];
                            }
                        }
                    }
                    y[((b * ho + yo) * wo + xo) * s.cout + o] = acc;
                }
            }
        }
    }
    y
}

#[test]
fn conv2d_with_f16_weights_pads_strides_and_takes_a_bias() {
    let mut card = Card::open();
    let mut rng = Rng(3);
    for s in [
        // A 3 x 3 convolution of the UNet, and the down sampler: 3 x 3 with stride 2.
        Conv {
            b: 2,
            h: 7,
            w: 5,
            cin: 20,
            cout: 24,
            kernel: 3,
            stride: 1,
            pad: 1,
            end: false,
        },
        Conv {
            b: 2,
            h: 8,
            w: 6,
            cin: 33,
            cout: 16,
            kernel: 3,
            stride: 2,
            pad: 1,
            end: false,
        },
    ] {
        let x = rng.vec(s.b * s.h * s.w * s.cin);
        let w16: Vec<u16> = rng.vec(s.cout * s.cin * 9).into_iter().map(f16).collect();
        let b16: Vec<u16> = rng.vec(s.cout).into_iter().map(f16).collect();
        let w: Vec<f32> = w16.iter().map(|h| from_f16(*h)).collect();
        let bias: Vec<f32> = b16.iter().map(|h| from_f16(*h)).collect();
        let (ho, wo) = s.out();
        let out = s.b * ho * wo * s.cout;
        let (xb, wb, bb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &w16),
            Buf::of(card.card(), &b16),
            Buf::zeros(card.card(), out * 4),
        );
        card.run(&[MkTaskDescriptor {
            flags: FLAG_F16,
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            x: bb.at(),
            ..s.task()
        }]);
        let got: Vec<f32> = yb.read(card.card(), out);
        let err = rel(&got, &conv_cpu(s, &x, &w, Some(&bias)));
        assert!(err < 1e-5, "{s:?}: relative error {err}");
    }
}

#[test]
fn a_1x1_conv2d_with_f32_weights_is_a_linear_layer_and_can_accumulate() {
    let mut card = Card::open();
    let mut rng = Rng(5);
    let s = Conv {
        b: 1,
        h: 9,
        w: 1,
        cin: 40,
        cout: 32,
        kernel: 1,
        stride: 1,
        pad: 0,
        end: false,
    };
    let x = rng.vec(9 * 40);
    let w = rng.vec(32 * 40);
    let start = rng.vec(9 * 32);
    let (xb, wb, yb) = (
        Buf::of(card.card(), &x),
        Buf::of(card.card(), &w),
        Buf::of(card.card(), &start),
    );
    card.run(&[MkTaskDescriptor {
        flags: FLAG_F32 | FLAG_ACCUMULATE,
        a: xb.at(),
        b: wb.at(),
        c: yb.at(),
        ..s.task()
    }]);
    let got: Vec<f32> = yb.read(card.card(), 9 * 32);
    let want: Vec<f32> = conv_cpu(s, &x, &w, None)
        .iter()
        .zip(&start)
        .map(|(y, c)| y + c)
        .collect();
    let err = rel(&got, &want);
    assert!(err < 1e-5, "relative error {err}");
}

/// The down sampler of the VAE encoder as diffusers runs it (`downsampling.py:139-145`): one row of zeros at the bottom
/// and one column of zeros at the right (`F.pad(x, (0, 1, 0, 1))`), then the convolution with no padding. This CPU form
/// pads the input itself, so it does not share the arithmetic of `Conv::start`.
fn pad_end_cpu(s: Conv, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    assert!(s.end && s.pad == 1);
    let (h, wd) = (s.h + 1, s.w + 1);
    let mut padded = vec![0.0f32; s.b * h * wd * s.cin];
    for b in 0..s.b {
        for y in 0..s.h {
            for xx in 0..s.w {
                let from = ((b * s.h + y) * s.w + xx) * s.cin;
                let to = ((b * h + y) * wd + xx) * s.cin;
                padded[to..to + s.cin].copy_from_slice(&x[from..from + s.cin]);
            }
        }
    }
    let plain = Conv {
        h,
        w: wd,
        pad: 0,
        end: false,
        ..s
    };
    conv_cpu(plain, &padded, w, Some(bias))
}

/// The down samplers of the VAE encoder. An image of `h` by `w` gives `h / 2` by `w / 2`.
fn pad_end_shapes() -> [Conv; 2] {
    let s = Conv {
        b: 2,
        h: 8,
        w: 6,
        cin: 64,
        cout: 32,
        kernel: 3,
        stride: 2,
        pad: 1,
        end: true,
    };
    // An odd size keeps the last row and column of the input out of each output, as PyTorch does.
    [s, Conv { h: 9, w: 5, ..s }]
}

/// `OP_CONV2D` with `FLAG_PAD_END` pads the bottom and the right only, as the down sampler of the VAE encoder does.
#[test]
fn conv2d_with_the_pad_at_the_end_gives_the_down_sampler_of_the_vae_encoder() {
    let mut card = Card::open();
    let mut rng = Rng(53);
    for s in pad_end_shapes() {
        let x = rng.vec(s.b * s.h * s.w * s.cin);
        let w = rng.vec(s.cout * s.cin * 9);
        let bias = rng.vec(s.cout);
        let (ho, wo) = s.out();
        assert_eq!((ho, wo), (s.h / 2, s.w / 2));
        let out = s.b * ho * wo * s.cout;
        let (xb, wb, bb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &w),
            Buf::of(card.card(), &bias),
            Buf::zeros(card.card(), out * 4),
        );
        card.run(&[nagaya_mk_bmg::tasks::pad_end(MkTaskDescriptor {
            flags: FLAG_F32,
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            x: bb.at(),
            ..s.task()
        })]);
        let got: Vec<f32> = yb.read(card.card(), out);
        let err = rel(&got, &pad_end_cpu(s, &x, &w, &bias));
        assert!(err < 1e-5, "{s:?}: relative error {err}");
    }
}

/// `OP_GEMM` with `FLAG_PAD_END` pads the bottom and the right only. The engine runs the down samplers of the VAE
/// encoder this way, with 3 planes for their F32 weights.
#[test]
fn gemm_with_the_pad_at_the_end_gives_the_down_sampler_of_the_vae_encoder() {
    use nagaya_mk_bmg::tasks::{gemm, gemm_planes, pad_end};
    let mut card = Card::open();
    let mut rng = Rng(59);
    for s in pad_end_shapes() {
        let x = rng.vec(s.b * s.h * s.w * s.cin);
        let w = rng.vec(s.cout * s.cin * 9);
        let bias = rng.vec(s.cout);
        let (ho, wo) = s.out();
        let out = s.b * ho * wo * s.cout;
        let wp = gemm_planes(&w, s.cout, s.cin, 3, 3);
        let (xb, wb, bb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &wp),
            Buf::of(card.card(), &bias),
            Buf::zeros(card.card(), out * 4),
        );
        card.run(&[pad_end(gemm(
            xb.at(),
            wb.at(),
            bb.at(),
            yb.at(),
            s.b as u32,
            s.h as u32,
            s.w as u32,
            s.cin as u32,
            s.cout as u32,
            (3, 2, 1),
            3,
            false,
            false,
        ))]);
        let got: Vec<f32> = yb.read(card.card(), out);
        let err = rel(&got, &pad_end_cpu(s, &x, &w, &bias));
        assert!(err < 1e-5, "{s:?}: relative error {err}");
    }
}

/// The plain group norm, pixel-major.
// The shape of the op is its arguments, as in the task descriptor.
#[allow(clippy::too_many_arguments)]
fn groupnorm_cpu(x: &[f32], b: usize, p: usize, k: usize, g: usize, eps: f32, w: &[f32], bias: &[f32]) -> Vec<f32> {
    let cg = k / g;
    let mut y = vec![0.0f32; x.len()];
    for img in 0..b {
        for grp in 0..g {
            let at = |pix: usize, c: usize| (img * p + pix) * k + grp * cg + c;
            let n = (p * cg) as f64;
            let mean = (0..p)
                .flat_map(|pix| (0..cg).map(move |c| (pix, c)))
                .map(|(pix, c)| x[at(pix, c)] as f64)
                .sum::<f64>()
                / n;
            let var = (0..p)
                .flat_map(|pix| (0..cg).map(move |c| (pix, c)))
                .map(|(pix, c)| (x[at(pix, c)] as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let inv = 1.0 / (var + eps as f64).sqrt();
            for pix in 0..p {
                for c in 0..cg {
                    let ch = grp * cg + c;
                    y[at(pix, c)] = ((x[at(pix, c)] as f64 - mean) * inv * w[ch] as f64 + bias[ch] as f64) as f32;
                }
            }
        }
    }
    y
}

#[test]
fn groupnorm_normalizes_each_group_of_each_image_and_can_apply_silu() {
    let mut card = Card::open();
    let mut rng = Rng(9);
    // The UNet: 32 groups, eps 1e-5, F16 weights. The VAE: eps 1e-6, F32 weights. A group of 10 channels too.
    for (b, p, k, g, eps, flags) in [
        (2usize, 35usize, 64usize, 32usize, 1e-5f32, FLAG_F16),
        (1, 130, 128, 32, 1e-6, FLAG_F32 | FLAG_SILU),
        (2, 12, 40, 4, 1e-5, FLAG_F32),
    ] {
        let x: Vec<f32> = rng.vec(b * p * k).into_iter().map(|v| 3.0 * v + 0.5).collect();
        let (w, bias): (Vec<f32>, Vec<f32>) = (rng.vec(k), rng.vec(k));
        let (wb, bb, w_used, bias_used) = if flags & FLAG_F16 != 0 {
            let (w16, b16): (Vec<u16>, Vec<u16>) = (
                w.iter().map(|v| f16(*v)).collect(),
                bias.iter().map(|v| f16(*v)).collect(),
            );
            let used = (
                w16.iter().map(|h| from_f16(*h)).collect(),
                b16.iter().map(|h| from_f16(*h)).collect(),
            );
            (Buf::of(card.card(), &w16), Buf::of(card.card(), &b16), used.0, used.1)
        } else {
            (
                Buf::of(card.card(), &w),
                Buf::of(card.card(), &bias),
                w.clone(),
                bias.clone(),
            )
        };
        let (xb, yb) = (Buf::of(card.card(), &x), Buf::zeros(card.card(), x.len() * 4));
        card.run(&[MkTaskDescriptor {
            flags,
            a: xb.at(),
            b: wb.at(),
            c: yb.at(),
            x: bb.at(),
            n: p as u32,
            k: k as u32,
            m: b as u32,
            layer: g as u32,
            p0: eps.to_bits(),
            ..task(OP_GROUPNORM)
        }]);
        let got: Vec<f32> = yb.read(card.card(), x.len());
        let mut want = groupnorm_cpu(&x, b, p, k, g, eps, &w_used, &bias_used);
        if flags & FLAG_SILU != 0 {
            want.iter_mut().for_each(|v| *v /= 1.0 + (-*v).exp());
        }
        let err = rel(&got, &want);
        assert!(
            err < 1e-5,
            "{b} images, {p} pixels, {k} channels, {g} groups: relative error {err}"
        );
    }
}

/// Plain attention with no mask, in rows of heads * d values.
// The shape of the op is its arguments, as in the task descriptor.
#[allow(clippy::too_many_arguments)]
fn sdpa_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    nq: usize,
    nk: usize,
    heads: usize,
    d: usize,
    causal: bool,
) -> Vec<f32> {
    let r = heads * d;
    let mut out = vec![0.0f32; b * nq * r];
    for img in 0..b {
        for t in 0..nq {
            for h in 0..heads {
                let qrow = &q[(img * nq + t) * r + h * d..][..d];
                let keys = if causal { nk.min(t + 1) } else { nk };
                let scores: Vec<f64> = (0..keys)
                    .map(|s| {
                        let krow = &k[(img * nk + s) * r + h * d..][..d];
                        qrow.iter()
                            .zip(krow)
                            .map(|(a, b)| (*a as f64) * (*b as f64))
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f64 = e.iter().sum();
                for j in 0..d {
                    let acc: f64 = (0..keys).map(|s| e[s] * v[(img * nk + s) * r + h * d + j] as f64).sum();
                    out[(img * nq + t) * r + h * d + j] = (acc / sum) as f32;
                }
            }
        }
    }
    out
}

#[test]
fn sdpa_attends_over_the_keys_of_its_own_image_for_each_head() {
    let mut card = Card::open();
    let mut rng = Rng(13);
    // Cross attention to 77 text rows with heads of 64, self attention, the one head of 512 of the VAE, and the causal
    // self attention of CLIP's text encoders.
    for (b, nq, nk, heads, d, causal) in [
        (2usize, 33usize, 77usize, 5usize, 64usize, false),
        (2, 40, 40, 2, 64, false),
        (1, 24, 24, 1, 512, false),
        (2, 77, 77, 3, 64, true),
    ] {
        let r = heads * d;
        let scale = 2.0;
        let q: Vec<f32> = rng.vec(b * nq * r).into_iter().map(|v| v * scale).collect();
        let k: Vec<f32> = rng.vec(b * nk * r).into_iter().map(|v| v * scale).collect();
        let v = rng.vec(b * nk * r);
        let (qb, kb, vb, yb) = (
            Buf::of(card.card(), &q),
            Buf::of(card.card(), &k),
            Buf::of(card.card(), &v),
            Buf::zeros(card.card(), b * nq * r * 4),
        );
        card.run(&[MkTaskDescriptor {
            flags: if causal { FLAG_CAUSAL } else { 0 },
            a: qb.at(),
            b: kb.at(),
            x: vb.at(),
            c: yb.at(),
            n: nq as u32,
            m: nk as u32,
            k: r as u32,
            p0: d as u32,
            layer: b as u32,
            ..task(OP_SDPA)
        }]);
        let got: Vec<f32> = yb.read(card.card(), b * nq * r);
        let err = rel(&got, &sdpa_cpu(&q, &k, &v, b, nq, nk, heads, d, causal));
        assert!(
            err < 1e-5,
            "{b} images, {nq} by {nk}, {heads} heads of {d}, causal {causal}: relative error {err}"
        );
    }
}

#[test]
fn upsample2x_repeats_each_pixel_in_a_2_by_2_block() {
    let mut card = Card::open();
    let mut rng = Rng(17);
    let (b, h, w, k) = (2usize, 3usize, 5usize, 7usize);
    let x = rng.vec(b * h * w * k);
    let (xb, yb) = (Buf::of(card.card(), &x), Buf::zeros(card.card(), 4 * x.len() * 4));
    card.run(&[MkTaskDescriptor {
        a: xb.at(),
        c: yb.at(),
        n: (h | w << 16) as u32,
        k: k as u32,
        layer: b as u32,
        ..task(OP_UPSAMPLE2X)
    }]);
    let got: Vec<f32> = yb.read(card.card(), 4 * x.len());
    let mut want = vec![0.0f32; 4 * x.len()];
    for img in 0..b {
        for y in 0..2 * h {
            for xx in 0..2 * w {
                for c in 0..k {
                    want[((img * 2 * h + y) * 2 * w + xx) * k + c] = x[((img * h + y / 2) * w + xx / 2) * k + c];
                }
            }
        }
    }
    assert_eq!(got, want);
}

#[test]
fn quick_gelu_geglu_half_and_copy_cols_match_their_cpu_forms() {
    let mut card = Card::open();
    let mut rng = Rng(19);
    let (n, k) = (5usize, 48usize);
    let erf_gelu = |g: f32| 0.5 * g * (1.0 + erf(g / std::f32::consts::SQRT_2));

    let x: Vec<f32> = rng.vec(n * k).into_iter().map(|v| 4.0 * v).collect();
    let (xb, yb) = (Buf::of(card.card(), &x), Buf::zeros(card.card(), n * k * 4));
    card.run(&[MkTaskDescriptor {
        a: xb.at(),
        c: yb.at(),
        n: n as u32,
        k: k as u32,
        ..task(OP_QUICK_GELU)
    }]);
    let want: Vec<f32> = x.iter().map(|v| v / (1.0 + (-1.702 * v).exp())).collect();
    let err = rel(&yb.read(card.card(), n * k), &want);
    assert!(err < 1e-6, "quick gelu: relative error {err}");

    let x: Vec<f32> = rng.vec(n * 2 * k).into_iter().map(|v| 4.0 * v).collect();
    let (xb, yb) = (Buf::of(card.card(), &x), Buf::zeros(card.card(), n * k * 4));
    card.run(&[MkTaskDescriptor {
        a: xb.at(),
        c: yb.at(),
        n: n as u32,
        k: k as u32,
        ..task(OP_GEGLU_HALF)
    }]);
    let mut want = vec![0.0f32; n * k];
    for t in 0..n {
        for j in 0..k {
            want[t * k + j] = x[t * 2 * k + j] * erf_gelu(x[t * 2 * k + k + j]);
        }
    }
    let err = rel(&yb.read(card.card(), n * k), &want);
    assert!(err < 1e-5, "geglu half: relative error {err}");

    // Join rows of 48 into rows of 80 at offset 20; the other values stay.
    let (m, off) = (80usize, 20usize);
    let x = rng.vec(n * k);
    let start = rng.vec(n * m);
    let (xb, yb) = (Buf::of(card.card(), &x), Buf::of(card.card(), &start));
    card.run(&[MkTaskDescriptor {
        a: xb.at(),
        c: yb.at(),
        n: n as u32,
        k: k as u32,
        m: m as u32,
        p0: off as u32,
        ..task(OP_COPY_COLS)
    }]);
    let mut want = start.clone();
    for t in 0..n {
        want[t * m + off..t * m + off + k].copy_from_slice(&x[t * k..(t + 1) * k]);
    }
    assert_eq!(yb.read::<f32>(card.card(), n * m), want);
}

#[test]
fn add_image_row_adds_the_row_of_each_image_to_its_pixels() {
    let mut card = Card::open();
    let mut rng = Rng(23);
    let (b, p, k) = (3usize, 7usize, 40usize);
    let rows = rng.vec(b * k);
    let start = rng.vec(b * p * k);
    let (rb, yb) = (Buf::of(card.card(), &rows), Buf::of(card.card(), &start));
    card.run(&[MkTaskDescriptor {
        a: rb.at(),
        c: yb.at(),
        n: p as u32,
        k: k as u32,
        layer: b as u32,
        ..task(OP_ADD_IMAGE_ROW)
    }]);
    let mut want = start.clone();
    for (i, v) in want.iter_mut().enumerate() {
        *v += rows[(i / (p * k)) * k + i % k];
    }
    assert_eq!(yb.read::<f32>(card.card(), b * p * k), want);
}

#[test]
fn gemm_on_the_matrix_units_gives_the_convolution_in_f32_precision() {
    use nagaya_mk_bmg::tasks::{gemm, gemm_planes};
    let mut card = Card::open();
    let mut rng = Rng(29);
    // A 3 x 3 convolution with F16 weights (2 planes), the down sampler with F32 weights (3 planes) that accumulates,
    // and a linear layer of 37 rows with more than 64 subgroups of outputs.
    for (s, f16_weights, accumulate) in [
        (
            Conv {
                b: 2,
                h: 7,
                w: 5,
                cin: 64,
                cout: 32,
                kernel: 3,
                stride: 1,
                pad: 1,
                end: false,
            },
            true,
            false,
        ),
        (
            Conv {
                b: 1,
                h: 9,
                w: 6,
                cin: 128,
                cout: 48,
                kernel: 3,
                stride: 2,
                pad: 1,
                end: false,
            },
            false,
            true,
        ),
        (
            Conv {
                b: 37,
                h: 1,
                w: 1,
                cin: 192,
                cout: 1040,
                kernel: 1,
                stride: 1,
                pad: 0,
                end: false,
            },
            true,
            false,
        ),
    ] {
        let x: Vec<f32> = rng.vec(s.b * s.h * s.w * s.cin).into_iter().map(|v| 8.0 * v).collect();
        // The test's F16 conversion takes normal numbers only, so the weights stay away from zero.
        let raw: Vec<f32> = rng
            .vec(s.cout * s.cin * s.kernel * s.kernel)
            .into_iter()
            .map(|v| v.signum() * (0.01 + v.abs()))
            .collect();
        let w: Vec<f32> = if f16_weights {
            raw.iter().map(|v| from_f16(f16(*v))).collect()
        } else {
            raw
        };
        let (ho, wo) = s.out();
        let out = s.b * ho * wo * s.cout;
        let start = rng.vec(out);
        let planes = if f16_weights { 2 } else { 3 };
        let wp = gemm_planes(&w, s.cout, s.cin, s.kernel, planes);
        let (bias_buf, bias): (Buf, Vec<f32>) = if f16_weights {
            let b16: Vec<u16> = rng.vec(s.cout).into_iter().map(f16).collect();
            let used = b16.iter().map(|h| from_f16(*h)).collect();
            (Buf::of(card.card(), &b16), used)
        } else {
            let b = rng.vec(s.cout);
            (Buf::of(card.card(), &b), b)
        };
        let (xb, wb, yb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &wp),
            Buf::of(card.card(), &start),
        );
        card.run(&[gemm(
            xb.at(),
            wb.at(),
            bias_buf.at(),
            yb.at(),
            s.b as u32,
            s.h as u32,
            s.w as u32,
            s.cin as u32,
            s.cout as u32,
            (s.kernel as u32, s.stride as u32, s.pad as u32),
            planes as u32,
            f16_weights,
            accumulate,
        )]);
        let got: Vec<f32> = yb.read(card.card(), out);
        let mut want = conv_cpu(s, &x, &w, Some(&bias));
        if accumulate {
            want.iter_mut().zip(&start).for_each(|(w, s)| *w += s);
        }
        // The bound of the other ops: f32 sums of many terms in another order differ by about 1e-6.
        let err = rel(&got, &want);
        assert!(err < 1e-5, "{s:?}, {planes} planes: relative error {err}");
    }
}

/// F32 weights need their third plane: with positive weights and inputs nothing cancels, so dropping it costs about
/// 2^-17 of each product, above the bound.
#[test]
fn gemm_with_three_planes_keeps_f32_weights_past_16_bits() {
    use nagaya_mk_bmg::tasks::{gemm, gemm_planes};
    let mut card = Card::open();
    let mut rng = Rng(41);
    let (rows, cin, cout) = (8usize, 128usize, 16usize);
    let x: Vec<f32> = rng.vec(rows * cin).into_iter().map(|v| 0.75 + 0.25 * v).collect();
    let w: Vec<f32> = rng.vec(cout * cin).into_iter().map(|v| 0.75 + 0.25 * v).collect();
    let wp = gemm_planes(&w, cout, cin, 1, 3);
    let (xb, wb, yb) = (
        Buf::of(card.card(), &x),
        Buf::of(card.card(), &wp),
        Buf::zeros(card.card(), rows * cout * 4),
    );
    card.run(&[gemm(
        xb.at(),
        wb.at(),
        0,
        yb.at(),
        rows as u32,
        1,
        1,
        cin as u32,
        cout as u32,
        (1, 1, 0),
        3,
        false,
        false,
    )]);
    let got: Vec<f32> = yb.read(card.card(), rows * cout);
    let want: Vec<f32> = (0..rows * cout)
        .map(|i| {
            let (t, o) = (i / cout, i % cout);
            (0..cin)
                .map(|j| x[t * cin + j] as f64 * w[o * cin + j] as f64)
                .sum::<f64>() as f32
        })
        .collect();
    let err = rel(&got, &want);
    assert!(err < 1e-6, "relative error {err}");
}

#[test]
fn gemm_planes_sum_to_the_weight() {
    use nagaya_mk_bmg::tasks::gemm_planes;
    let mut rng = Rng(31);
    let raw: Vec<f32> = rng
        .vec(4 * 8 * 9)
        .into_iter()
        .map(|v| v.signum() * (0.01 + v.abs()))
        .collect();
    let f16s: Vec<f32> = raw.iter().map(|v| from_f16(f16(*v))).collect();
    let two = gemm_planes(&f16s, 4, 8, 3, 2);
    let three = gemm_planes(&raw, 4, 8, 3, 3);
    let k = 9 * 8;
    for o in 0..4 {
        for i in 0..8 {
            for tap in 0..9 {
                let at = o * k + tap * 8 + i;
                let sum2: f32 = (0..2).map(|p| f32::from_bits((two[p * 4 * k + at] as u32) << 16)).sum();
                assert_eq!(
                    sum2,
                    f16s[(o * 8 + i) * 9 + tap],
                    "two planes hold an F16 weight exactly"
                );
                let sum3: f64 = (0..3)
                    .map(|p| f32::from_bits((three[p * 4 * k + at] as u32) << 16) as f64)
                    .sum();
                let want = raw[(o * 8 + i) * 9 + tap] as f64;
                assert!(
                    (sum3 - want).abs() <= want.abs() * 2f64.powi(-23),
                    "three planes hold an F32 weight"
                );
            }
        }
    }
}

#[test]
fn attention_on_the_matrix_units_gives_sdpa() {
    use nagaya_mk_bmg::tasks::{attn_planes_len, attn_prep, attn_xmx};
    let mut card = Card::open();
    let mut rng = Rng(37);
    // Self attention with rows that are not whole tiles, cross attention to 77 keys, and heads of 16 values.
    for (b, nq, nk, heads, d) in [
        (2usize, 37usize, 37usize, 3usize, 64usize),
        (2, 40, 77, 5, 64),
        (1, 21, 50, 2, 16),
    ] {
        let r = heads * d;
        let q: Vec<f32> = rng.vec(b * nq * r).into_iter().map(|v| 3.0 * v).collect();
        let k: Vec<f32> = rng.vec(b * nk * r).into_iter().map(|v| 3.0 * v).collect();
        let v = rng.vec(b * nk * r);
        let (qb, kb, vb, yb) = (
            Buf::of(card.card(), &q),
            Buf::of(card.card(), &k),
            Buf::of(card.card(), &v),
            Buf::zeros(card.card(), b * nq * r * 4),
        );
        let pb = Buf::zeros(card.card(), attn_planes_len(b, nk, heads, d) * 2);
        card.run(&[
            attn_prep(kb.at(), vb.at(), pb.at(), b as u32, nk as u32, heads as u32, d as u32),
            attn_xmx(
                qb.at(),
                pb.at(),
                yb.at(),
                b as u32,
                nq as u32,
                nk as u32,
                heads as u32,
                d as u32,
            ),
        ]);
        let got: Vec<f32> = yb.read(card.card(), b * nq * r);
        let err = rel(&got, &sdpa_cpu(&q, &k, &v, b, nq, nk, heads, d, false));
        assert!(
            err < 1e-5,
            "{b} images, {nq} by {nk}, {heads} heads of {d}: relative error {err}"
        );
    }
}

/// The error function, by Abramowitz and Stegun 7.1.26 in f64; good to 1.5e-7.
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let y = 1.0 - poly * (-x * x).exp();
    (if x < 0.0 { -y } else { y }) as f32
}

/// The bytes and the shape of one tensor of a safetensors file. It reads the header, then only that tensor.
fn tensor(path: &PathBuf, name: &str) -> (Vec<u8>, String, Vec<usize>) {
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    let mut len = [0u8; 8];
    file.read_exact(&mut len).unwrap();
    let header_len = u64::from_le_bytes(len);
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    let entry = &header[name];
    assert!(entry.is_object(), "{} has no tensor {name}", path.display());
    let start = entry["data_offsets"][0].as_u64().unwrap();
    let end = entry["data_offsets"][1].as_u64().unwrap();
    let shape = entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut bytes = vec![0u8; (end - start) as usize];
    file.seek(SeekFrom::Start(8 + header_len + start)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    (bytes, entry["dtype"].as_str().unwrap().to_owned(), shape)
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    let (whole, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "a tensor of f32 has 4 bytes for each value");
    whole.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// NCHW to pixel-major: [b][c][h][w] to [b][h][w][c].
fn to_pixels(x: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for img in 0..b {
        for ch in 0..c {
            for p in 0..h * w {
                y[(img * h * w + p) * c + ch] = x[(img * c + ch) * h * w + p];
            }
        }
    }
    y
}

/// The paths of the checkpoint and of the oracle, or `None` when either is absent. `NAGAYA_REQUIRE_MODELS=1` makes
/// absence a failure (see AGENTS.md).
fn golden() -> Option<(PathBuf, PathBuf)> {
    let require = std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1");
    let model = std::env::var_os("SDXL_MODEL_DIR").map(PathBuf::from);
    let gold = std::env::var_os("SDXL_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/sdxl-golden"));
    let unet = model
        .as_ref()
        .map(|m| m.join("unet/diffusion_pytorch_model.fp16.safetensors"));
    let parts = gold.join("parts.safetensors");
    match (unet, parts.is_file()) {
        (Some(unet), true) if unet.is_file() => Some((unet, parts)),
        _ => {
            assert!(
                !require,
                "NAGAYA_REQUIRE_MODELS is set: set SDXL_MODEL_DIR, and run tools/sdxl_golden.sh"
            );
            eprintln!(
                "skipped: the SDXL checkpoint or the oracle is absent; set SDXL_MODEL_DIR, run tools/sdxl_golden.sh"
            );
            None
        }
    }
}

#[test]
fn the_conv_in_of_the_unet_gives_the_oracle() {
    let Some((unet, parts)) = golden() else { return };
    let (w, wdt, wshape) = tensor(&unet, "conv_in.weight");
    let (bias, bdt, _) = tensor(&unet, "conv_in.bias");
    assert_eq!((wdt.as_str(), bdt.as_str()), ("F16", "F16"));
    assert_eq!(wshape, [320, 4, 3, 3]);
    let (input, _, ishape) = tensor(&parts, "unet.in");
    let (want, _, wshape_out) = tensor(&parts, "unet.conv_in");
    let (b, h, wd) = (ishape[0], ishape[2], ishape[3]);
    assert_eq!(wshape_out, [b, 320, h, wd]);
    let x = to_pixels(&f32s(&input), b, 4, h, wd);
    let want = to_pixels(&f32s(&want), b, 320, h, wd);
    let mut card = Card::open();
    let s = Conv {
        b,
        h,
        w: wd,
        cin: 4,
        cout: 320,
        kernel: 3,
        stride: 1,
        pad: 1,
        end: false,
    };
    let (xb, wb, bb, yb) = (
        Buf::of(card.card(), &x),
        Buf::of(card.card(), &w),
        Buf::of(card.card(), &bias),
        Buf::zeros(card.card(), want.len() * 4),
    );
    card.run(&[MkTaskDescriptor {
        flags: FLAG_F16,
        a: xb.at(),
        b: wb.at(),
        c: yb.at(),
        x: bb.at(),
        ..s.task()
    }]);
    let got: Vec<f32> = yb.read(card.card(), want.len());
    let err = rel(&got, &want);
    assert!(err < 1e-5, "relative error {err} against the oracle's unet.conv_in");
}
