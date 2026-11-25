//! The matrix multiply and SwiGLU of a batched decode on a B70 card. With `FLAG_TILED`, one row
//! takes the path of 2 to 8 rows. Thus a row gives the same bits for any count of rows from 1 to 8, and a context's
//! bits do not depend on the other segments of its pass (engine-abi "LFM3", "Batched passes").
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_lfm3_batch_ops -- --test-threads=1`.
#![cfg(feature = "xe")]

use std::ffi::c_void;
use std::time::Instant;

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::*;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{KERNEL_ENTRY, LFM3_KERNEL};

/// The sizes of LFM3-1.7B (`config.json`): the width, the width of the MLP, and the K and V rows of a layer.
const HIDDEN: u32 = 2048;
const MLP: u32 = 7680;
const KV_ROWS: u32 = 512;
/// The most rows of the path of 2 to 8 rows.
const ROWS: usize = 8;

/// A buffer in device memory.
struct Buf {
    ptr: *mut u8,
    bytes: usize,
}

impl Buf {
    fn of<T: Copy>(card: &B70Card, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data).max(64);
        let ptr = card.alloc_device_usm::<u8>(bytes);
        card.fill_zero_sync(ptr as *mut c_void, bytes);
        let stage = card.alloc_host_usm::<u8>(bytes);
        // SAFETY: `stage` holds at least the bytes of `data`.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, stage, std::mem::size_of_val(data)) };
        card.memcpy_sync(ptr as *mut c_void, stage as *const c_void, std::mem::size_of_val(data));
        card.free(stage as *mut c_void);
        Self { ptr, bytes }
    }

    fn zeros(card: &B70Card, floats: usize) -> Self {
        Self::of(card, &vec![0.0f32; floats])
    }

    fn read(&self, card: &B70Card, n: usize) -> Vec<f32> {
        assert!(n * 4 <= self.bytes);
        let stage = card.alloc_host_usm::<u8>(n * 4);
        card.memcpy_sync(stage as *mut c_void, self.ptr as *const c_void, n * 4);
        let mut out = vec![0.0f32; n];
        // SAFETY: `stage` holds `n` floats, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, n * 4) };
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
}

/// `n` values from a fixed generator, in [-1, 1).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The BF16 bits of each value, with the low bits cut, as the weights of a checkpoint hold them.
fn bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
}

fn tiled(mut task: MkTaskDescriptor) -> MkTaskDescriptor {
    task.flags |= FLAG_TILED;
    task
}

/// The outputs of `op` for the first `n` of 8 rows, with `FLAG_TILED`. `width` is the floats of one output row.
fn rows_of(
    card: &mut Card,
    op: &dyn Fn(u64, u64, u32) -> MkTaskDescriptor,
    x: &Buf,
    width: usize,
    n: usize,
) -> Vec<f32> {
    let y = Buf::zeros(card.card(), ROWS * width);
    card.run(&[tiled(op(x.at(), y.at(), n as u32))]);
    y.read(card.card(), n * width)
}

/// Each row gives the bits of the run of 8 rows, for each count of rows from 1 to 8.
fn check_rows(card: &mut Card, name: &str, op: &dyn Fn(u64, u64, u32) -> MkTaskDescriptor, k: u32, width: usize) {
    let x = Buf::of(card.card(), &values(7, ROWS * k as usize));
    let all = rows_of(card, op, &x, width, ROWS);
    for n in 1..ROWS {
        let some = rows_of(card, op, &x, width, n);
        let differ = some
            .iter()
            .zip(&all)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differ, 0,
            "{name}: {differ} values of {n} rows differ from the run of {ROWS} rows"
        );
    }
}

#[test]
fn a_tiled_matmul_gives_a_row_the_same_bits_for_1_to_8_rows() {
    let mut card = Card::open();
    for (m, k) in [
        (HIDDEN, HIDDEN),
        (3 * HIDDEN, HIDDEN),
        (KV_ROWS, HIDDEN),
        (16, HIDDEN),
        (HIDDEN, MLP),
    ] {
        let w = Buf::of(card.card(), &bf16(&values(u64::from(m * k), (m * k) as usize)));
        let at = w.at();
        check_rows(
            &mut card,
            &format!("matmul {m} by {k}"),
            &|x, y, n| tasks::matmul(x, at, y, n, m, k, false),
            k,
            m as usize,
        );
    }
}

#[test]
fn a_tiled_swiglu_gives_a_row_the_same_bits_for_1_to_8_rows() {
    let mut card = Card::open();
    let w = Buf::of(card.card(), &bf16(&values(11, (2 * MLP * HIDDEN) as usize)));
    let at = w.at();
    check_rows(
        &mut card,
        "swiglu",
        &|x, y, n| tasks::swiglu(x, at, y, n, MLP, HIDDEN),
        HIDDEN,
        MLP as usize,
    );
}

/// The time of the matrix multiplies of 24 layers for one row, on the path of one row and on the tiled path. This is
/// a measurement, not a gate: run it with `--ignored --nocapture`.
#[test]
#[ignore]
fn time_one_row_on_both_paths() {
    let mut card = Card::open();
    let c = card.card();
    let w_in = Buf::of(c, &bf16(&values(1, (3 * HIDDEN * HIDDEN) as usize)));
    let w_out = Buf::of(c, &bf16(&values(2, (HIDDEN * HIDDEN) as usize)));
    let w13 = Buf::of(c, &bf16(&values(3, (2 * MLP * HIDDEN) as usize)));
    let w2 = Buf::of(c, &bf16(&values(4, (HIDDEN * MLP) as usize)));
    let x = Buf::of(c, &values(5, MLP as usize));
    let y = Buf::zeros(c, (3 * HIDDEN).max(MLP) as usize);
    let layer = |flag: u16| {
        let mut t = vec![
            tasks::matmul(x.at(), w_in.at(), y.at(), 1, 3 * HIDDEN, HIDDEN, false),
            tasks::matmul(x.at(), w_out.at(), y.at(), 1, HIDDEN, HIDDEN, false),
            tasks::swiglu(x.at(), w13.at(), y.at(), 1, MLP, HIDDEN),
            tasks::matmul(x.at(), w2.at(), y.at(), 1, HIDDEN, MLP, false),
        ];
        for task in &mut t {
            task.flags |= flag;
        }
        t
    };
    for (name, flag) in [("one row", 0), ("tiled", FLAG_TILED)] {
        let ring: Vec<MkTaskDescriptor> = (0..24).flat_map(|_| layer(flag)).collect();
        card.run(&ring);
        let reps = 50;
        let start = Instant::now();
        for _ in 0..reps {
            card.run(&ring);
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        println!("{name}: {ms:.3} ms for the matmuls of 24 layers");
    }
}
