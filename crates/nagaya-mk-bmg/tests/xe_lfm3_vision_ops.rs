//! The ops of the LFM3 vision tower on a B70 card: the norm and the axial rotation of the head
//! rows, the gated depthwise convolution over the patch grid, and the attention in both directions of the embedding
//! binary, which the LFM3 binary now holds. Each test compares the kernel with the CPU forms of
//! `common/lfm3_vision.rs`. The engine test `vision_cpu.rs` of `nagaya-mk-lfm3-bmg` checks those forms against
//! transformers on the host. Each test prints its gap, so the card run measures it.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_lfm3_vision_ops -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::ffi::c_void;

use common::lfm3::{bf16, first_other_bits, from_bf16, rel, values};
use common::lfm3_vision::*;
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{KERNEL_ENTRY, LFM3_KERNEL};

/// The sizes of the tower of LFM3-Omni-2B: heads of 64 values, 16 query heads and 8 key heads, 1,024 channels.
const D: usize = 64;
const HEADS: usize = 16;
const KV: usize = 8;
const CHANNELS: usize = 1024;
const EPS: f32 = 1e-5;
const THETA: f32 = 1e6;

/// The grids of the cases, down by across: a small square, the small image of the oracle, and a wide one.
const GRIDS: [(usize, usize); 3] = [(4, 4), (14, 22), (6, 40)];

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
        buf.write(card, data);
        buf
    }

    fn write<T: Copy>(&self, card: &B70Card, data: &[T]) {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes.max(64));
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        // SAFETY: `stage` holds `bytes` bytes of host USM, and `data` is `bytes` bytes long.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, stage, bytes) };
        card.memcpy_sync(self.ptr as *mut c_void, stage as *const c_void, bytes);
        card.free(stage as *mut c_void);
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

    fn free(self, card: &B70Card) {
        card.free(self.ptr as *mut c_void);
    }
}

struct Card {
    cluster: L0Cluster,
    ring: Ring,
}

impl Card {
    fn open() -> Self {
        let cluster = L0Cluster::init_test(LFM3_KERNEL, KERNEL_ENTRY);
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

    fn buf<T: Copy>(&self, data: &[T]) -> Buf {
        Buf::of(self.card(), data)
    }
}

/// BF16 values near 1, as the norm weights of a head.
fn norm_weights(seed: u64, d: usize) -> Vec<u16> {
    values(seed, d, 0.5).into_iter().map(|v| bf16(v + 1.0)).collect()
}

/// The rotation is the norm of each head row and the turn of its pairs by the coordinates of its patch. The kernel
/// computes the frequencies with `pow` and the angles with `cos` and `sin` of the card, so its bits differ from the
/// CPU form in the last places.
#[test]
fn vision_rope_turns_each_head_by_the_coordinates_of_its_patch() {
    let mut card = Card::open();
    for (case, (gh, gw)) in GRIDS.iter().enumerate() {
        for heads in [HEADS, KV] {
            let rows = gh * gw * heads;
            let x = values(700 + case as u64, rows * D, 2.0);
            let w = norm_weights(710 + case as u64, D);
            let xb = card.buf(&x);
            let wb = card.buf(&w);
            card.run(&[tasks::vision_rope(
                xb.at(),
                wb.at(),
                rows as u32,
                heads as u32,
                D as u32,
                *gh as u32,
                *gw as u32,
                EPS,
                THETA,
            )]);
            let got = xb.read(card.card(), rows * D);
            let wf: Vec<f32> = w.iter().map(|b| from_bf16(*b)).collect();
            let want = norm_axial(&x, &wf, heads, D, *gh, *gw, THETA, EPS);
            let err = rel(&got, &want);
            println!("vision_rope {gh} by {gw}, {heads} heads: relative error {err:.3e}");
            // A wrong axis or a wrong order of the patches gives an error near 1.
            assert!(err < 1e-3, "{gh} by {gw}, {heads} heads: relative error {err}");
            for b in [xb, wb] {
                b.free(card.card());
            }
        }
    }
}

/// The convolution sums its 9 taps down then across with fma, as the CPU form does, so it gives the bits of the form.
#[test]
fn vision_conv_matches_its_cpu_form_to_the_bit() {
    let mut card = Card::open();
    for (case, (gh, gw)) in GRIDS.iter().enumerate() {
        // Two frames, so the edges of one frame do not read the other.
        let n = 2 * gh * gw;
        let bcx = values(800 + case as u64, n * 3 * CHANNELS, 1.0);
        let w: Vec<u16> = values(810 + case as u64, CHANNELS * 9, 0.5)
            .into_iter()
            .map(bf16)
            .collect();
        let (xb, wb) = (card.buf(&bcx), card.buf(&w));
        let yb = Buf::zeros(card.card(), n * CHANNELS * 4);
        card.run(&[tasks::vision_conv(
            xb.at(),
            wb.at(),
            yb.at(),
            n as u32,
            CHANNELS as u32,
            *gh as u32,
            *gw as u32,
        )]);
        let got = yb.read(card.card(), n * CHANNELS);
        let wf: Vec<f32> = w.iter().map(|b| from_bf16(*b)).collect();
        let mut want = vision_conv(&bcx[..n / 2 * 3 * CHANNELS], &wf, CHANNELS, *gh, *gw);
        want.extend(vision_conv(&bcx[n / 2 * 3 * CHANNELS..], &wf, CHANNELS, *gh, *gw));
        println!("vision_conv {gh} by {gw}: relative error {:.3e}", rel(&got, &want));
        if let Some(i) = first_other_bits(&got, &want) {
            panic!(
                "{gh} by {gw}: row {} channel {} is {} on the card and {} on the CPU",
                i / CHANNELS,
                i % CHANNELS,
                got[i],
                want[i]
            );
        }
        for b in [xb, wb, yb] {
            b.free(card.card());
        }
    }
}

/// The attention of the embedding binary in the LFM3 binary, at the shapes of the tower: each patch attends to every
/// patch of its image, with 2 query heads for each key head and the scale 1/8.
#[test]
fn biattn_attends_to_every_patch_of_an_image() {
    let mut card = Card::open();
    for (case, (gh, gw)) in GRIDS.iter().enumerate() {
        let n = gh * gw;
        let q = values(900 + case as u64, n * HEADS * D, 1.0);
        let k = values(910 + case as u64, n * KV * D, 1.0);
        let v = values(920 + case as u64, n * KV * D, 1.0);
        let (qb, kb, vb) = (card.buf(&q), card.buf(&k), card.buf(&v));
        let yb = Buf::zeros(card.card(), n * HEADS * D * 4);
        let scale = (D as f32).powf(-0.5);
        card.run(&[tasks::biattn(
            qb.at(),
            kb.at(),
            vb.at(),
            yb.at(),
            n as u32,
            HEADS as u32,
            KV as u32,
            D as u32,
            0,
            scale,
        )]);
        let got = yb.read(card.card(), n * HEADS * D);
        let want = biattn(&q, &k, &v, HEADS, KV, D, n, scale as f64);
        let err = rel(&got, &want);
        println!("biattn {gh} by {gw}: relative error {err:.3e}");
        assert!(err < 1e-4, "{gh} by {gw}: relative error {err}");
        for b in [qb, kb, vb, yb] {
            b.free(card.card());
        }
    }
}

/// The attention of the tower gives the bits of `OP_BIATTN` with every key. The grids are a small one, one whose
/// count of patches is no multiple of the tile, the two images of the oracle, and a large one. The heads have 64 values
/// and 16 values, and the scales are 1/8 and 1.
#[test]
fn vision_attn_gives_the_bits_of_biattn() {
    let mut card = Card::open();
    let grids = [(2usize, 2usize), (2, 5), (14, 22), (18, 26), (48, 64)];
    for (case, (gh, gw)) in grids.iter().enumerate() {
        for (d, scale) in [(D, (D as f32).powf(-0.5)), (16, 1.0)] {
            let n = gh * gw;
            let seed = 1000 + 10 * case as u64 + d as u64;
            // Large scores, so a new max of the softmax is rare after the first keys, and small ones.
            let q = values(seed, n * HEADS * d, if case % 2 == 0 { 1.0 } else { 6.0 });
            let k = values(seed + 1, n * KV * d, 1.0);
            let v = values(seed + 2, n * KV * d, 1.0);
            let (qb, kb, vb) = (card.buf(&q), card.buf(&k), card.buf(&v));
            let old = Buf::zeros(card.card(), n * HEADS * d * 4);
            let new = Buf::zeros(card.card(), n * HEADS * d * 4);
            let (n32, h32, kv32, d32) = (n as u32, HEADS as u32, KV as u32, d as u32);
            card.run(&[
                tasks::biattn(qb.at(), kb.at(), vb.at(), old.at(), n32, h32, kv32, d32, 0, scale),
                tasks::vision_attn(qb.at(), kb.at(), vb.at(), new.at(), n32, h32, kv32, d32, scale),
            ]);
            let want = old.read(card.card(), n * HEADS * d);
            let got = new.read(card.card(), n * HEADS * d);
            assert!(want.iter().any(|x| *x != 0.0), "{gh} by {gw}: OP_BIATTN wrote nothing");
            if let Some(i) = first_other_bits(&got, &want) {
                panic!(
                    "{gh} by {gw}, heads of {d}: row {} head {} value {} is {:e} and OP_BIATTN gives {:e}",
                    i / (HEADS * d),
                    i / d % HEADS,
                    i % d,
                    got[i],
                    want[i]
                );
            }
            for b in [qb, kb, vb, old, new] {
                b.free(card.card());
            }
        }
    }
}

/// A measurement, not a gate: the time of the ops of the tower by the size of the grid, on random values. A layer of
/// the tower with attention runs the attention one time, and 4 products of the size of `q`. A conv layer runs the
/// convolution one time and the product `in_proj`. Run it alone, with `--ignored --nocapture`.
#[test]
#[ignore = "a measurement"]
fn the_time_of_each_op_of_the_tower_by_the_size_of_the_grid() {
    let mut card = Card::open();
    for (gh, gw) in [(16usize, 16usize), (48, 64), (60, 80), (76, 100)] {
        let n = gh * gw;
        let q = values(1, n * HEADS * D, 1.0);
        let k = values(2, n * KV * D, 1.0);
        let v = values(3, n * KV * D, 1.0);
        let x = values(4, n * CHANNELS, 1.0);
        let w: Vec<u16> = values(5, 3 * CHANNELS * CHANNELS, 0.05).into_iter().map(bf16).collect();
        let taps: Vec<u16> = values(6, CHANNELS * 9, 0.5).into_iter().map(bf16).collect();
        let (qb, kb, vb, xb, wb, tb) = (
            card.buf(&q),
            card.buf(&k),
            card.buf(&v),
            card.buf(&x),
            card.buf(&w),
            card.buf(&taps),
        );
        let yb = Buf::zeros(card.card(), n * HEADS * D * 4);
        let bcx = Buf::zeros(card.card(), n * 3 * CHANNELS * 4);
        let (n32, c32) = (n as u32, CHANNELS as u32);
        let scale = (D as f32).powf(-0.5);
        let ops = [
            (
                "OP_BIATTN",
                tasks::biattn(
                    qb.at(),
                    kb.at(),
                    vb.at(),
                    yb.at(),
                    n32,
                    HEADS as u32,
                    KV as u32,
                    D as u32,
                    0,
                    scale,
                ),
            ),
            (
                "attention of the tower",
                tasks::vision_attn(
                    qb.at(),
                    kb.at(),
                    vb.at(),
                    yb.at(),
                    n32,
                    HEADS as u32,
                    KV as u32,
                    D as u32,
                    scale,
                ),
            ),
            (
                "product q",
                tasks::matmul(xb.at(), wb.at(), yb.at(), n32, c32, c32, false),
            ),
            (
                "product in_proj",
                tasks::matmul(xb.at(), wb.at(), bcx.at(), n32, 3 * c32, c32, false),
            ),
            (
                "convolution",
                tasks::vision_conv(bcx.at(), tb.at(), yb.at(), n32, c32, gh as u32, gw as u32),
            ),
        ];
        let mut line = format!("{gh} by {gw}, {n} patches:");
        for (name, op) in &ops {
            // The first run of a buffer pays for its pages.
            card.run(std::slice::from_ref(op));
            let started = std::time::Instant::now();
            card.run(std::slice::from_ref(op));
            line += &format!(" {name} {:.1} ms;", started.elapsed().as_secs_f64() * 1e3);
        }
        println!("{line}");
        for b in [qb, kb, vb, xb, wb, tb, yb, bcx] {
            b.free(card.card());
        }
    }
}
