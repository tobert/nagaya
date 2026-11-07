//! The speech ops of the LFM3 binary on a B70 card (milestone 24 step 8): the attention of the depth decoder in
//! groups, the polar form, and the inverse real FFT with the window. Each test compares the kernel with the CPU form
//! of `common/lfm3_speech.rs`, which `lfm3_speech_cpu_forms.rs` checks on the host.
//!
//! The tolerances are not measured yet, so each test prints its gap. The bounds only catch a wrong op: a wrong group,
//! sign, bin or window gives a relative error of 1e-1 or more. Each op must give the same bits on each run.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_lfm3_speech_ops -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::ffi::c_void;

use common::lfm3::{first_other_bits, rel, values};
use common::lfm3_speech::*;
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{KERNEL_ENTRY, LFM3_KERNEL};

/// The bound of a wrong op. A right op on the card gives about 1e-6.
const DEFECT_BOUND: f32 = 1e-3;

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
    bufs: Vec<Buf>,
}

impl Card {
    fn open() -> Self {
        let cluster = L0Cluster::init(LFM3_KERNEL, KERNEL_ENTRY);
        assert!(!cluster.cards.is_empty(), "no Level Zero card");
        let ring = Ring::new(&cluster.cards[0], 8 << 20);
        Self {
            cluster,
            ring,
            bufs: Vec::new(),
        }
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each test sizes its buffers for its tasks.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }

    /// A buffer that holds `data`, as an address. The card frees it at the end of the test.
    fn put<T: Copy>(&mut self, data: &[T]) -> u64 {
        let buf = Buf::of(self.card(), data);
        let at = buf.at();
        self.bufs.push(buf);
        at
    }

    /// A buffer of `n` zero floats, as an index into the buffers.
    fn out(&mut self, n: usize) -> usize {
        let buf = Buf::zeros(self.card(), n * 4);
        self.bufs.push(buf);
        self.bufs.len() - 1
    }

    fn read(&self, index: usize, n: usize) -> Vec<f32> {
        self.bufs[index].read(self.card(), n)
    }
}

impl Drop for Card {
    fn drop(&mut self) {
        for buf in self.bufs.drain(..) {
            self.cluster.cards[0].free(buf.ptr as *mut c_void);
        }
    }
}

/// Run `task` twice into the buffer `out` of `n` floats, check that both runs give the same bits, and give the output.
fn twice(card: &mut Card, task: MkTaskDescriptor, out: usize, n: usize) -> Vec<f32> {
    card.run(&[task]);
    let first = card.read(out, n);
    card.run(&[task]);
    let again = card.read(out, n);
    assert_eq!(
        first_other_bits(&first, &again),
        None,
        "op {} gave other bits on a second run",
        task.opcode
    );
    first
}

#[test]
fn group_attention_matches_its_cpu_form_in_each_group() {
    let mut card = Card::open();
    // The depth decoder: 4 frames of 12 codebooks, 32 heads of 32 over 8 key heads. Then a head of 64 in one group.
    for (n, heads, kv, d, group, seed) in [(48usize, 32, 8, 32, 12, 600u64), (12, 8, 2, 64, 12, 610)] {
        let scale = (d as f32).powf(-0.5);
        let q = values(seed, n * heads * d, 2.0);
        let k = values(seed + 1, n * kv * d, 2.0);
        let v = values(seed + 2, n * kv * d, 1.0);
        let (qa, ka, va) = (card.put(&q), card.put(&k), card.put(&v));
        let out = card.out(n * heads * d);
        let task = tasks::group_attn(
            qa,
            ka,
            va,
            card.bufs[out].at(),
            n as u32,
            heads as u32,
            kv as u32,
            d as u32,
            group as u32,
            scale,
        );
        let got = twice(&mut card, task, out, n * heads * d);
        let want = group_attn(&q, &k, &v, n, heads, kv, d, group, scale);
        let err = rel(&got, &want);
        println!("group attention of {n} rows, {heads} heads of {d}: relative error {err:.3e}");
        assert!(err < DEFECT_BOUND, "relative error {err}");
    }
}

#[test]
fn the_polar_form_matches_its_cpu_form_and_skips_the_pad() {
    let mut card = Card::open();
    // The rows of `lin` of the codec: 6 rows of 1,296 values, 641 log magnitudes and 641 angles, then 14 pad values.
    let (n, f, stride) = (6usize, 641usize, 1296usize);
    let mut a = values(700, n * stride, 3.0);
    for t in 0..n {
        for i in 0..f {
            a[t * stride + f + i] *= std::f32::consts::PI / 3.0;
        }
        for i in 2 * f..stride {
            a[t * stride + i] = f32::NAN;
        }
    }
    let input = card.put(&a);
    let out = card.out(n * 2 * f);
    let task = tasks::polar(input, card.bufs[out].at(), n as u32, f as u32, stride as u32);
    let got = twice(&mut card, task, out, n * 2 * f);
    let want = polar(&a, n, f, stride);
    assert!(got.iter().all(|v| v.is_finite()), "the op read the pad");
    let err = rel(&got, &want);
    println!("polar form of {n} rows of {f} bins: relative error {err:.3e}");
    assert!(err < DEFECT_BOUND, "relative error {err}");
}

#[test]
fn the_inverse_fft_matches_its_cpu_form_with_the_window() {
    let mut card = Card::open();
    // The 6 STFT frames of one codec frame: 641 bins of 1,280 samples, and the Hann window of the codec.
    let (n, nf) = (6usize, 1280usize);
    let f = nf / 2 + 1;
    let bins = values(800, n * 2 * f, 4.0);
    let window = hann(nf);
    let table = tasks::irfft_table(nf);
    let (ba, wa, ta) = (card.put(&bins), card.put(&window), card.put(&table));
    let out = card.out(n * nf);
    let task = tasks::irfft(ba, wa, ta, card.bufs[out].at(), n as u32, nf as u32);
    let got = twice(&mut card, task, out, n * nf);
    let want = irfft(&bins, n, nf, &window);
    let err = rel(&got, &want);
    println!("inverse FFT of {n} rows of {nf} samples: relative error {err:.3e}");
    assert!(err < DEFECT_BOUND, "relative error {err}");
    // The imaginary parts of the first and the last bin do not count, as in irfft.
    let mut edges = bins.clone();
    for t in 0..n {
        edges[t * 2 * f + f] += 5.0;
        edges[t * 2 * f + 2 * f - 1] -= 5.0;
    }
    let ea = card.put(&edges);
    let task = tasks::irfft(ea, wa, ta, card.bufs[out].at(), n as u32, nf as u32);
    let moved = twice(&mut card, task, out, n * nf);
    assert_eq!(first_other_bits(&moved, &got), None, "the edge bins changed the output");
}
