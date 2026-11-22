//! GPU gates of the dense ops of wave B lane D (milestone 37 step 9), against the host forms of the B70 crate, on real
//! weights of Inkling 2 Mini. A test skips when `INKLING2_MODEL_DIR` is absent, and fails with `NAGAYA_REQUIRE_MODELS`.
//!
//! - The norm prologue of `OP_MATMUL_BF16` (`FLAG_NORM`, `FLAG_NORM_UNWEIGHTED`): the matmul applies `w * (x * rf)` while
//!   it loads `x`. The oracle is `ops::rms_norm` then the matmul.
//! - The head: `FLAG_NORM | FLAG_SOFTCAP_30` over 200,064 rows.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_dense -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::fs::File;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use common::gpu::{GRIDS, download, open, open_profile, run_with_watchdog, upload};
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::mma_model::dot_bits;
use nagaya_mk_cuda::protocol::{
    FLAG_NORM, FLAG_NORM_UNWEIGHTED, FLAG_SOFTCAP_30, MATMUL_TABLE_NORM_WEIGHT, MkTaskTable, PROFILE_WORDS,
};
use nagaya_mk_cuda::repack::{MatmulShape, repack_bf16};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::safetensors::SafeTensorsHeader;
use nagaya_mk_inkling2_bmg::host::ops;

const MODEL_VAR: &str = "INKLING2_MODEL_DIR";
const CHECKPOINT: &str = "model-00001-of-00001.safetensors";

/// The read peak of the GB10 that the probe measured, in GB/s.
const PEAK_GBPS: f64 = 252.0;

/// The bound of the accuracy of a dense matmul, in epsilons of `f32` times the sum of the absolute terms (wave A).
const ACCURACY_BOUND_EPS: f64 = 4.0;

const HIDDEN: usize = 2304;

/// The head holds 200,064 of the 201,024 rows of the unembedding: the 200,058 rows of the vocabulary, rounded up to the
/// tile of 16. The padded rows are not read.
const HEAD_ROWS: usize = 200_064;

struct Checkpoint {
    header: SafeTensorsHeader,
    file: Mutex<File>,
}

/// The checkpoint, or `None` when the model is absent and the test skips.
fn checkpoint() -> Option<&'static Checkpoint> {
    static CELL: OnceLock<Option<Checkpoint>> = OnceLock::new();
    CELL.get_or_init(|| {
        let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
        let Some(dir) = std::env::var_os(MODEL_VAR) else {
            assert!(!require, "{MODEL_VAR} is not set and NAGAYA_REQUIRE_MODELS is set");
            eprintln!("cuda_dense: skipped, {MODEL_VAR} is not set");
            return None;
        };
        let path = PathBuf::from(dir).join(CHECKPOINT);
        if !path.exists() {
            assert!(
                !require,
                "{} is absent and NAGAYA_REQUIRE_MODELS is set",
                path.display()
            );
            eprintln!("cuda_dense: skipped, {} is absent", path.display());
            return None;
        }
        let header = SafeTensorsHeader::read(&path).unwrap_or_else(|e| panic!("{e}"));
        let file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        Some(Checkpoint {
            header,
            file: Mutex::new(file),
        })
    })
    .as_ref()
}

impl Checkpoint {
    /// The bytes of a tensor, with its dtype and shape checked.
    fn bytes(&self, name: &str, dtype: &str, shape: &[usize]) -> Vec<u8> {
        let meta = self
            .header
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no tensor {name}"));
        assert_eq!((meta.dtype.as_str(), &meta.shape[..]), (dtype, shape), "{name}");
        self.header
            .read_tensor(&mut self.file.lock().unwrap(), name)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// A `bf16` tensor widened to `f32` (exact).
    fn bf16(&self, name: &str, shape: &[usize]) -> Vec<f32> {
        self.bytes(name, "BF16", shape)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| ops::bf16_to_f32(u16::from_le_bytes(*b)))
            .collect()
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `[-scale, scale]` with all 24 bits of mantissa random.
    fn value(&mut self, scale: f32) -> f32 {
        let r = self.next();
        ((r >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
    }
}

/// `rows` rows of `k` values of a residual: a scale of 1 to 8 for each row, so that the row factors differ.
fn make_rows(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 1.0 + (r % 8) as f32;
        out.extend((0..k).map(|_| rng.value(scale)));
    }
    out
}

/// The row factor, the way `ops::rms_norm` computes it. The producer task of a row writes it.
fn row_factor(x: &[f32]) -> f32 {
    let mean = ops::dot(x, x) / x.len() as f32;
    1.0 / (mean + ops::EPS).sqrt()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

/// Nanoseconds from the first block that passes the wait of ring task `task` to the last block that ends it.
fn task_ns(profile: &[u64], blocks: usize, task: usize) -> u64 {
    let slots = &profile[task * blocks * PROFILE_WORDS..(task + 1) * blocks * PROFILE_WORDS];
    let begin = slots.chunks(PROFILE_WORDS).map(|s| s[1]).min().unwrap();
    let end = slots.chunks(PROFILE_WORDS).map(|s| s[2]).max().unwrap();
    end - begin
}

/// One matmul `W[n, 2304]` on the GPU, with the buffers for passes of up to `max_rows` rows.
struct Mm<'d> {
    dev: &'d CudaDevice,
    shape: MatmulShape,
    /// The `bf16` bytes of the first `n` rows of `W`, as the checkpoint stores them.
    w: Vec<u8>,
    packed: CUdeviceptr,
    x: CUdeviceptr,
    rf: CUdeviceptr,
    table: CUdeviceptr,
    y: CUdeviceptr,
    ring: Ring<'d>,
    shared: u32,
    max_rows: usize,
}

impl<'d> Mm<'d> {
    /// `norm_w` is the norm weight (`None` when the pass has none), `x` holds `max_rows` rows and `rf` their factors.
    fn new(
        dev: &'d CudaDevice,
        w: Vec<u8>,
        n: usize,
        x: &[f32],
        rf: &[f32],
        norm_w: Option<&[f32]>,
        profile: bool,
    ) -> Self {
        let shape = MatmulShape::new(n, HIDDEN).unwrap();
        let max_rows = x.len() / HIDDEN;
        let packed = upload(dev, &repack_bf16(&w, &shape).unwrap());
        let mut table = MkTaskTable::default();
        if let Some(nw) = norm_w {
            let bf: Vec<u16> = nw.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            table.ptr[MATMUL_TABLE_NORM_WEIGHT] = upload(dev, &bf);
        }
        let ring = Ring::new(dev, shape.scratch_floats(max_rows), profile).unwrap();
        ring.zero_device().unwrap();
        Self {
            dev,
            shape,
            w,
            packed,
            x: upload(dev, x),
            rf: upload(dev, rf),
            table: upload(dev, &[table]),
            y: dev.alloc_device(max_rows * n * 4).unwrap(),
            ring,
            shared: dev.max_dynamic_shared_bytes().unwrap(),
            max_rows,
        }
    }

    fn task(&self, rows: usize, row0: usize, flags: u16) -> Task {
        let shape = &self.shape;
        let x = self.x + (row0 * HIDDEN * 4) as u64;
        let mut desc = shape
            .descriptor(rows, x, self.packed, self.y, self.ring.scratch())
            .unwrap();
        desc.flags = flags;
        if flags & (FLAG_NORM | FLAG_NORM_UNWEIGHTED) != 0 {
            desc.e = self.rf + (row0 * 4) as u64;
            desc.d = self.table;
        }
        Task::with_items(desc)
            .reads(x, rows * HIDDEN * 4)
            .reads(self.rf + (row0 * 4) as u64, rows * 4)
            .reads(self.packed, shape.packed_bytes())
            .writes(self.y, rows * shape.n * 4)
            .writes(self.ring.scratch(), shape.scratch_floats(rows) * 4)
    }

    fn run(&mut self, rows: usize, row0: usize, flags: u16, grid: u32) -> Vec<f32> {
        self.dev.memset_zero(self.y, self.max_rows * self.shape.n * 4).unwrap();
        let task = self.task(rows, row0, flags);
        self.ring.set(&[task]);
        run_with_watchdog(&self.ring, grid, 288, self.shared);
        download(self.dev, self.y, rows * self.shape.n)
    }

    /// The `f32` values of the `bf16` weight of column `c`.
    fn column(&self, c: usize) -> Vec<f32> {
        (0..HIDDEN)
            .map(|i| {
                let at = (c * HIDDEN + i) * 2;
                ops::bf16_to_f32(u16::from_le_bytes([self.w[at], self.w[at + 1]]))
            })
            .collect()
    }

    fn free(self) {
        for p in [self.packed, self.x, self.rf, self.table, self.y] {
            self.dev.free_device(p).unwrap();
        }
        self.ring.free().unwrap();
    }
}

/// The normed row that the host computes: `w * (v * rf)` with the weight, `v * rf` without.
fn host_normed(row: &[f32], weight: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0f32; row.len()];
    match weight {
        Some(w) => ops::rms_norm(row, w, &mut out),
        None => {
            out.copy_from_slice(row);
            ops::rms_norm_unweighted(&mut out);
        }
    }
    out
}

/// The sum of the exact products of `normed` and the column, in `f64`, and the sum of their absolute values.
fn exact(normed: &[f32], column: &[f32]) -> (f64, f64) {
    let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
    for (a, b) in normed.iter().zip(column) {
        let term = f64::from(*a) * f64::from(*b);
        sum += term;
        magnitude += term.abs();
    }
    (sum, magnitude)
}

/// The columns that a gate checks: the first and the last tiles, and tiles between.
fn columns_of(n: usize, count: usize) -> Vec<usize> {
    let tiles = n / 16;
    let mut tile_list: Vec<usize> = vec![0, 1, tiles - 1];
    tile_list.extend((0..count).map(|i| (i * 977 + 3) % tiles));
    tile_list.sort_unstable();
    tile_list.dedup();
    tile_list
        .iter()
        .flat_map(|t| (0..16).map(move |c| t * 16 + c))
        .collect()
}

const NORM_SHAPES: [(&str, &str, usize); 2] = [
    ("wq_du", "model.llm.layers.0.attn.wq_du.weight", 3072),
    ("wk_dv", "model.llm.layers.0.attn.wk_dv.weight", 1024),
];

fn attn_norm(ck: &Checkpoint) -> Vec<f32> {
    ck.bf16("model.llm.layers.0.attn_norm.weight", &[HIDDEN])
}

#[test]
fn the_norm_prologue_equals_the_host_norm_then_the_matmul() {
    let Some(ck) = checkpoint() else { return };
    let dev = open();
    let weight = attn_norm(ck);
    for (name, tensor, n) in NORM_SHAPES {
        let w = ck.bytes(tensor, "BF16", &[n, HIDDEN]);
        let rows = 40;
        let x = make_rows(rows, HIDDEN, 11);
        let rf: Vec<f32> = x.chunks(HIDDEN).map(row_factor).collect();
        let mut mm = Mm::new(&dev, w, n, &x, &rf, Some(&weight), false);
        let columns = columns_of(n, 6);
        for (label, flags, norm_w) in [
            ("weighted", FLAG_NORM, Some(&weight[..])),
            ("unweighted", FLAG_NORM_UNWEIGHTED, None),
        ] {
            let y = mm.run(rows, 0, flags, 48);
            let mut worst = 0.0f64;
            let mut matvec_worst = 0.0f64;
            for r in 0..rows {
                let normed = host_normed(&x[r * HIDDEN..(r + 1) * HIDDEN], norm_w);
                let mut host_out = vec![0f32; n];
                ops::matvec_bf16(&mm.w, &normed, &mut host_out);
                for &c in &columns {
                    let column = mm.column(c);
                    let (sum, magnitude) = exact(&normed, &column);
                    let got = y[r * n + c];
                    let error = (f64::from(got) - sum).abs() / magnitude / f64::from(f32::EPSILON);
                    assert!(
                        error <= ACCURACY_BOUND_EPS,
                        "{name} {label}: row {r} column {c}: the GPU gives {got:e}, the f64 sum is {sum:e}, the error is {error:.2} epsilons of {magnitude:e}"
                    );
                    worst = worst.max(error);
                    // The bits: the norm is the same f32 operations, so the model of the MMA gives the bits.
                    let want = dot_bits(&column, &normed, mm.shape.kt);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{name} {label}: row {r} column {c}: the GPU gives {got:e}, the host model gives {want:e}"
                    );
                    let host = host_out[c];
                    matvec_worst =
                        matvec_worst.max((f64::from(host) - sum).abs() / magnitude / f64::from(f32::EPSILON));
                }
            }
            eprintln!(
                "cuda_dense norm: {name:<6} {label:<10} {rows} rows x {} columns: worst error {worst:.3} epsilons of the sum of |terms| \
                 against f64 (bound {ACCURACY_BOUND_EPS}); the host matvec is {matvec_worst:.3}; the bits equal the host model of the MMA",
                columns.len()
            );
        }
        mm.free();
    }
}

#[test]
fn the_bits_of_a_normed_row_do_not_depend_on_the_rows_or_the_grid() {
    let Some(ck) = checkpoint() else { return };
    let dev = open();
    let weight = attn_norm(ck);
    for (name, tensor, n) in NORM_SHAPES {
        let w = ck.bytes(tensor, "BF16", &[n, HIDDEN]);
        let x = make_rows(512, HIDDEN, 12);
        let rf: Vec<f32> = x.chunks(HIDDEN).map(row_factor).collect();
        let mut mm = Mm::new(&dev, w, n, &x, &rf, Some(&weight), false);
        for flags in [FLAG_NORM, FLAG_NORM_UNWEIGHTED] {
            let full = mm.run(512, 0, flags, 48);
            for rows in [1, 7, 16, 17, 100] {
                assert_eq!(
                    bits(&mm.run(rows, 0, flags, 48)),
                    bits(&full[..rows * n]),
                    "{name} flags {flags}: {rows} rows differ from the first rows of 512"
                );
            }
            for (rows, row0) in [(12, 500), (1, 311), (17, 33)] {
                assert_eq!(
                    bits(&mm.run(rows, row0, flags, 48)),
                    bits(&full[row0 * n..(row0 + rows) * n]),
                    "{name} flags {flags}: {rows} rows from row {row0} differ"
                );
            }
            for grid in &GRIDS[1..] {
                assert_eq!(
                    bits(&mm.run(512, 0, flags, *grid)),
                    bits(&full),
                    "{name} flags {flags}: grid {grid} differs from grid 48"
                );
            }
        }
        eprintln!(
            "cuda_dense norm: {name:<6} the bits of both norms are equal at 1 to 512 rows and at grids {GRIDS:?}"
        );
        mm.free();
    }
}

/// The head: the first 200,064 rows of the unembedding, and the final norm.
fn head(ck: &Checkpoint) -> (Vec<u8>, Vec<f32>) {
    let all = ck.bytes("model.llm.unembed.weight", "BF16", &[201_024, HIDDEN]);
    (
        all[..HEAD_ROWS * HIDDEN * 2].to_vec(),
        ck.bf16("model.llm.norm.weight", &[HIDDEN]),
    )
}

/// The bound of the softcap, in absolute value: `tanhf` of the GPU and `tanh` of the host differ in the last bits. A
/// value near the cap of 30 has an ulp of 1.9e-6. The gate measured 3.8e-6 (2 ulp) at most over 16 rows of 688 columns;
/// the bound is 4 ulp.
const SOFTCAP_BOUND: f64 = 8e-6;

#[test]
fn the_head_equals_the_host_norm_matvec_and_softcap() {
    let Some(ck) = checkpoint() else { return };
    let dev = open();
    let (w, weight) = head(ck);
    let rows = 16;
    let x = make_rows(rows, HIDDEN, 13);
    let rf: Vec<f32> = x.chunks(HIDDEN).map(row_factor).collect();
    let mut mm = Mm::new(&dev, w, HEAD_ROWS, &x, &rf, Some(&weight), false);
    let flags = FLAG_NORM | FLAG_SOFTCAP_30;
    let columns = columns_of(HEAD_ROWS, 40);
    let y = mm.run(rows, 0, flags, 48);
    let (mut worst_eps, mut worst_cap) = (0.0f64, 0.0f64);
    for r in 0..rows {
        let normed = host_normed(&x[r * HIDDEN..(r + 1) * HIDDEN], Some(&weight));
        let mut host_out = vec![0f32; HEAD_ROWS];
        ops::matvec_bf16(&mm.w, &normed, &mut host_out);
        for &c in &columns {
            let column = mm.column(c);
            let raw = dot_bits(&column, &normed, mm.shape.kt);
            let want = ops::softcap(raw, ops::LOGIT_SOFTCAP);
            let got = y[r * HEAD_ROWS + c];
            let diff = (f64::from(got) - f64::from(want)).abs();
            assert!(
                diff <= SOFTCAP_BOUND,
                "row {r} column {c}: the GPU gives {got:e}, the host gives {want:e}; the bound is {SOFTCAP_BOUND:e}"
            );
            worst_cap = worst_cap.max(diff);
            // The host matvec and the f64 sum: the raw result and the softcap differ by their own rounding.
            let (sum, magnitude) = exact(&normed, &column);
            let capped = f64::from(ops::LOGIT_SOFTCAP) * (sum / f64::from(ops::LOGIT_SOFTCAP)).tanh();
            let error = (f64::from(got) - capped).abs() / magnitude.max(1.0) / f64::from(f32::EPSILON);
            worst_eps = worst_eps.max(error);
            let host = ops::softcap(host_out[c], ops::LOGIT_SOFTCAP);
            assert!(
                (f64::from(host) - capped).abs() <= 1e-4,
                "the host matvec disagrees with f64 at row {r} column {c}"
            );
        }
    }
    eprintln!(
        "cuda_dense head: {rows} rows x {} columns of {HEAD_ROWS}: worst difference to the host softcap of the model of the MMA \
         {worst_cap:.3e} (bound {SOFTCAP_BOUND:e}); worst error to the f64 softcap {worst_eps:.3} epsilons of max(1, sum of |terms|)",
        columns.len()
    );
    mm.free();
}

#[test]
fn the_bits_of_a_head_row_do_not_depend_on_the_rows_or_the_grid() {
    let Some(ck) = checkpoint() else { return };
    let dev = open();
    let (w, weight) = head(ck);
    let x = make_rows(100, HIDDEN, 14);
    let rf: Vec<f32> = x.chunks(HIDDEN).map(row_factor).collect();
    let mut mm = Mm::new(&dev, w, HEAD_ROWS, &x, &rf, Some(&weight), false);
    let flags = FLAG_NORM | FLAG_SOFTCAP_30;
    let full = mm.run(100, 0, flags, 48);
    for rows in [1, 7, 16, 17] {
        assert_eq!(
            bits(&mm.run(rows, 0, flags, 48)),
            bits(&full[..rows * HEAD_ROWS]),
            "{rows} rows differ from the first rows of 100"
        );
    }
    assert_eq!(
        bits(&mm.run(1, 63, flags, 48)),
        bits(&full[63 * HEAD_ROWS..64 * HEAD_ROWS]),
        "row 63 alone differs"
    );
    for grid in &GRIDS[1..] {
        assert_eq!(
            bits(&mm.run(100, 0, flags, *grid)),
            bits(&full),
            "grid {grid} differs from grid 48"
        );
    }
    eprintln!("cuda_dense head: the bits of the head are equal at 1 to 100 rows and at grids {GRIDS:?}");
    mm.free();
}

/// The head at one row must read at this share of the peak (the brief of the lane, design rule 1).
const HEAD_GATE: f64 = 0.90;

#[test]
fn the_speed_of_the_head_at_one_row() {
    let Some(ck) = checkpoint() else { return };
    let dev = open_profile();
    let blocks = dev.sm_count() as usize;
    let (w, weight) = head(ck);
    let bytes = HEAD_ROWS * HIDDEN * 2;
    let flags = FLAG_NORM | FLAG_SOFTCAP_30;
    let mut at_one = 0.0f64;
    for rows in [1usize, 16] {
        let x = make_rows(rows, HIDDEN, 15);
        let rf: Vec<f32> = x.chunks(HIDDEN).map(row_factor).collect();
        let mut mm = Mm::new(&dev, w.clone(), HEAD_ROWS, &x, &rf, Some(&weight), true);
        let tasks = 6;
        let list: Vec<Task> = (0..tasks).map(|_| mm.task(rows, 0, flags)).collect();
        mm.ring.set(&list);
        run_with_watchdog(&mm.ring, blocks as u32, 288, mm.shared);
        let profile = mm.ring.read_profile().unwrap();
        let spans: Vec<u64> = (1..tasks)
            .map(|t| task_ns(&profile, blocks, mm.ring.ring_index(t)))
            .collect();
        let (typical, best) = (median(spans.clone()), *spans.iter().min().unwrap());
        let gbps = |ns: u64| bytes as f64 / ns as f64;
        eprintln!(
            "cuda_dense speed: head {HEAD_ROWS} rows, norm and softcap, {rows:>2} rows: {:.1} us (best {:.1} us), W {:.1} MB, {:.1} GB/s \
             ({:.1}% of {PEAK_GBPS}), best {:.1} GB/s",
            typical as f64 / 1e3,
            best as f64 / 1e3,
            bytes as f64 / 1e6,
            gbps(typical),
            gbps(typical) / PEAK_GBPS * 100.0,
            gbps(best)
        );
        if rows == 1 {
            at_one = gbps(typical);
        }
        mm.free();
    }
    assert!(
        at_one >= HEAD_GATE * PEAK_GBPS,
        "the head at one row reads {at_one:.1} GB/s, under {:.0}% of {PEAK_GBPS} GB/s",
        HEAD_GATE * 100.0
    );
}
