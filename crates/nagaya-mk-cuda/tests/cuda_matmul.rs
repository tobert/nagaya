//! GPU gates of `OP_MATMUL_BF16`, with real weights of Inkling 2 Mini: `wq_du`,
//! `wo_ud` and `wk_dv` of layer 0, and the head. Each test reads the model folder from `INKLING2_MODEL_DIR`. It skips
//! when the variable or the folder is absent, and with `NAGAYA_REQUIRE_MODELS` set an absence fails the test.
//!
//! The gates:
//! - the accuracy against a host `f64` sum of the exact products;
//! - the bits of a row do not depend on the count of rows or on its place in the pass (design rule 4);
//! - the bits do not depend on the grid width, or on the block that merges (design rule 5);
//! - the speed of each shape, and the head at one row against the measured peak (design rule 1).
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-cuda --features cuda
//!  --test cuda_matmul -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::mma_model::dot_bits;
use nagaya_mk_cuda::protocol::PROFILE_WORDS;
use nagaya_mk_cuda::repack::{MatmulShape, repack_bf16};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_cuda::safetensors::SafeTensorsHeader;
use nagaya_mk_cuda::{INKLING2_CUBIN, INKLING2_PROFILE_CUBIN, test_cards};

const MODEL_VAR: &str = "INKLING2_MODEL_DIR";
const CHECKPOINT: &str = "model-00001-of-00001.safetensors";

/// The matrices of the gate: a name, the tensor, `N` and `K`.
const MATRICES: [(&str, &str, usize, usize); 4] = [
    ("wq_du", "model.llm.layers.0.attn.wq_du.weight", 3072, 2304),
    ("wo_ud", "model.llm.layers.0.attn.wo_ud.weight", 2304, 3072),
    ("wk_dv", "model.llm.layers.0.attn.wk_dv.weight", 1024, 2304),
    ("head", "model.llm.unembed.weight", 201_024, 2304),
];

/// The read peak of the GB10 that the probe measured (`docs/engine-abi.md`, "The facts of the GB10"), in GB/s.
const PEAK_GBPS: f64 = 252.0;

/// The head at one row must read at this share of the peak.
const HEAD_GATE: f64 = 0.90;

/// The bound of the accuracy, in units of `f32::EPSILON` times the sum of the absolute terms of the result. The
/// largest error that the gate measured is printed by `the_matmul_is_accurate_against_an_f64_sum`.
const ACCURACY_BOUND_EPS: f64 = 4.0;

/// The bound for a coherent row, where every term has one sign. The tensor core cuts each sum toward zero, so the
/// error of such a sum adds up in one direction: up to 18 cuts of one ulp in a warp, and then the tree. The lo term
/// alone is 48 epsilons, so the bound still shows a missing term.
const COHERENT_BOUND_EPS: f64 = 24.0;

/// The grid widths of the gate. 7 divides no count of items below, and 1 runs every item in one block.
const GRIDS: [u32; 4] = [48, 24, 7, 1];

struct Loaded {
    name: &'static str,
    shape: MatmulShape,
    /// The bytes of `W` as the checkpoint stores them.
    w: Vec<u8>,
    packed: Vec<u8>,
}

/// The matrices, read and repacked one time for all the tests of the process. `None` means that the model is absent
/// and the test skips.
fn matrices() -> Option<&'static [Loaded]> {
    static CELL: OnceLock<Option<Vec<Loaded>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
        let dir = match std::env::var_os(MODEL_VAR) {
            Some(dir) => PathBuf::from(dir),
            None => {
                assert!(!require, "{MODEL_VAR} is not set and NAGAYA_REQUIRE_MODELS is set");
                eprintln!("cuda_matmul: skipped, {MODEL_VAR} is not set");
                return None;
            }
        };
        let path = dir.join(CHECKPOINT);
        if !path.exists() {
            assert!(
                !require,
                "{} is absent and NAGAYA_REQUIRE_MODELS is set",
                path.display()
            );
            eprintln!("cuda_matmul: skipped, {} is absent", path.display());
            return None;
        }
        let header = SafeTensorsHeader::read(&path).unwrap_or_else(|e| panic!("{e}"));
        let mut file = File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        let mut out = Vec::new();
        for (name, tensor, n, k) in MATRICES {
            let meta = &header.tensors[tensor];
            assert_eq!(
                (meta.dtype.as_str(), &meta.shape[..]),
                ("BF16", &[n, k][..]),
                "{tensor}"
            );
            let w = header.read_tensor(&mut file, tensor).unwrap_or_else(|e| panic!("{e}"));
            let shape = MatmulShape::new(n, k).unwrap();
            let packed = repack_bf16(&w, &shape).unwrap();
            out.push(Loaded { name, shape, w, packed });
        }
        Some(out)
    })
    .as_deref()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `rows` rows of `k` floats. Each value has all 24 bits of mantissa random and an exponent from 2^-4 to 2^3, so that
/// the mid and the lo term matter and the rows differ.
fn make_x(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    (0..rows * k)
        .map(|_| {
            let r = rng.next();
            let sign = ((r >> 40) as u32 & 1) << 31;
            let exponent = 123 + ((r >> 41) as u32 % 8);
            f32::from_bits(sign | (exponent << 23) | (r as u32 & 0x007F_FFFF))
        })
        .collect()
}

fn bf16_to_f64(w: &[u8], index: usize) -> f64 {
    f64::from(f32::from_bits(
        u32::from(u16::from_le_bytes([w[index * 2], w[index * 2 + 1]])) << 16,
    ))
}

/// The sum of the exact products in `f64`, and the sum of their absolute values, for each row and each column in
/// `columns`.
fn reference(m: &Loaded, x: &[f32], rows: usize, columns: &[usize]) -> Vec<(f64, f64)> {
    let k = m.shape.k;
    let mut out = Vec::with_capacity(rows * columns.len());
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for &c in columns {
            let (mut sum, mut magnitude) = (0.0f64, 0.0f64);
            for (i, &xv) in xr.iter().enumerate() {
                let term = f64::from(xv) * bf16_to_f64(&m.w, c * k + i);
                sum += term;
                magnitude += term.abs();
            }
            out.push((sum, magnitude));
        }
    }
    out
}

/// The columns that the accuracy gate checks: all of a small matrix, and tiles spread over the head, with the first
/// and the last.
fn gate_columns(n: usize) -> Vec<usize> {
    if n <= 4096 {
        return (0..n).collect();
    }
    let tiles = n / 16;
    let mut chosen: Vec<usize> = vec![0, 1, 2, tiles - 3, tiles - 2, tiles - 1];
    chosen.extend((0..tiles).step_by(331));
    chosen.sort_unstable();
    chosen.dedup();
    chosen.iter().flat_map(|t| t * 16..t * 16 + 16).collect()
}

fn open() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU")
}

fn upload<T: Copy>(dev: &CudaDevice, data: &[T]) -> CUdeviceptr {
    let bytes = std::mem::size_of_val(data);
    let ptr = dev.alloc_device(bytes).expect("allocate");
    unsafe { dev.copy_htod(ptr, data.as_ptr().cast::<c_void>(), bytes) }.expect("upload");
    ptr
}

fn download<T: Copy + Default>(dev: &CudaDevice, ptr: CUdeviceptr, count: usize) -> Vec<T> {
    let mut out = vec![T::default(); count];
    unsafe {
        dev.copy_dtoh(
            out.as_mut_ptr().cast::<c_void>(),
            ptr,
            std::mem::size_of_val(out.as_slice()),
        )
    }
    .expect("download");
    out
}

/// Run the ring in a thread with a watchdog, so that a hang fails the test.
fn run_with_watchdog(ring: &Ring<'_>, grid: u32, shared: u32, seconds: u64) {
    let done = Arc::new(AtomicBool::new(false));
    std::thread::scope(|scope| {
        let flag = Arc::clone(&done);
        scope.spawn(move || {
            for _ in 0..seconds * 100 {
                std::thread::sleep(std::time::Duration::from_millis(10));
                if flag.load(Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_matmul: the ring did not finish within {seconds} s at grid {grid}; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run_grid(grid, DEFAULT_BLOCK_THREADS, shared).unwrap() };
        done.store(true, Ordering::SeqCst);
    });
}

/// The buffers of one matrix on the GPU: `copies` copies of the repacked weight, 16 rows of `x`, `y` and the scratch.
struct Rig<'d> {
    dev: &'d CudaDevice,
    shape: MatmulShape,
    weights: Vec<CUdeviceptr>,
    x: CUdeviceptr,
    y: CUdeviceptr,
    ring: Ring<'d>,
    shared: u32,
    max_rows: usize,
    /// Run the block form (`FLAG_BLOCK_FORM`) instead of the decode form.
    block: bool,
}

impl<'d> Rig<'d> {
    fn new(dev: &'d CudaDevice, m: &Loaded, x: &[f32], copies: usize, profile: bool) -> Self {
        Self::with_rows(dev, m, x, copies, profile, 16)
    }

    /// A rig for passes of up to `max_rows` rows. `x` holds at least that many rows.
    fn with_rows(dev: &'d CudaDevice, m: &Loaded, x: &[f32], copies: usize, profile: bool, max_rows: usize) -> Self {
        let shape = m.shape;
        let weights = (0..copies).map(|_| upload(dev, &m.packed)).collect();
        let x = upload(dev, x);
        let y = dev.alloc_device(max_rows * shape.n * 4).unwrap();
        let ring = Ring::new(dev, shape.scratch_floats(max_rows), profile).unwrap();
        // The counters of the merge groups start at 0, and the op leaves them there.
        ring.zero_device().unwrap();
        let shared = dev.max_dynamic_shared_bytes().unwrap();
        Self {
            dev,
            shape,
            weights,
            x,
            y,
            ring,
            shared,
            max_rows,
            block: false,
        }
    }

    /// The task of `rows` rows of `x`, from the row `row0`, with the copy `copy` of the weight.
    fn task(&self, rows: usize, row0: usize, copy: usize) -> Task {
        let shape = &self.shape;
        let w = self.weights[copy % self.weights.len()];
        let x = self.x + (row0 * shape.k * 4) as u64;
        let desc = if self.block {
            shape.descriptor_block(rows, x, w, self.y).unwrap()
        } else {
            shape.descriptor(rows, x, w, self.y, self.ring.scratch()).unwrap()
        };
        Task::with_items(desc)
            .reads(x, rows * shape.k * 4)
            .reads(w, shape.packed_bytes())
            .writes(self.y, rows * shape.n * 4)
            .writes(self.ring.scratch(), shape.scratch_floats(rows) * 4)
    }

    /// One launch of one task: the `rows` rows of `y`.
    fn run(&mut self, rows: usize, row0: usize, grid: u32) -> Vec<f32> {
        self.dev.memset_zero(self.y, self.max_rows * self.shape.n * 4).unwrap();
        let task = self.task(rows, row0, 0);
        self.ring.set(&[task]);
        run_with_watchdog(&self.ring, grid, self.shared, 120);
        download(self.dev, self.y, rows * self.shape.n)
    }

    /// The counters of the merge groups, read from the scratch.
    fn counters(&self) -> Vec<u32> {
        download(self.dev, self.ring.scratch(), self.shape.tiles())
    }

    fn free(self) {
        for w in &self.weights {
            self.dev.free_device(*w).unwrap();
        }
        self.dev.free_device(self.x).unwrap();
        self.dev.free_device(self.y).unwrap();
        self.ring.free().unwrap();
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// A row of `x` that makes the terms of column `column` all have one sign, with a value whose mid term and lo term are
/// both nonzero: `1 + 3 * 2^-10 + 3 * 2^-19`. Then the sum of the absolute terms equals the sum, and the lo term alone
/// moves the result by 48 epsilons. A random row cannot show a missing term: its terms cancel, and the sum of the
/// absolute terms is large.
fn coherent_row(m: &Loaded, column: usize) -> Vec<f32> {
    let value = 1.0 + 3.0 * 2f32.powi(-10) + 3.0 * 2f32.powi(-19);
    (0..m.shape.k)
        .map(|i| {
            if bf16_to_f64(&m.w, column * m.shape.k + i) < 0.0 {
                -value
            } else {
                value
            }
        })
        .collect()
}

#[test]
fn the_matmul_is_accurate_against_an_f64_sum() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    let mut worst_all = 0.0f64;
    for m in matrices {
        // Rows 0 to 15 are random. Rows 16 to 31 are coherent with the columns `coherent[r]`.
        let coherent: Vec<usize> = (0..16).map(|r| (r * 997) % m.shape.n).collect();
        let mut x = make_x(16, m.shape.k, 1);
        for &c in &coherent {
            x.extend(coherent_row(m, c));
        }
        let mut rig = Rig::new(&dev, m, &x, 1, false);
        let columns = gate_columns(m.shape.n);
        let want = reference(m, &x[..16 * m.shape.k], 16, &columns);
        let worst = std::cell::Cell::new(0.0f64);
        let check = |what: &str, bound: f64, row: usize, c: usize, got: f32, sum: f64, magnitude: f64| {
            let error = (f64::from(got) - sum).abs() / magnitude / f64::from(f32::EPSILON);
            assert!(
                error <= bound,
                "{}: {what} row {row} column {c}: the GPU gives {got:e}, the f64 sum is {sum:e}, the error is {error:.2} \
                 epsilons of the sum of the absolute terms {magnitude:e}; the bound is {bound}",
                m.name
            );
            worst.set(worst.get().max(error));
        };
        for (rows, row0) in [(16, 0), (1, 7)] {
            let y = rig.run(rows, row0, 48);
            for r in 0..rows {
                for (ci, &c) in columns.iter().enumerate() {
                    let (sum, magnitude) = want[(row0 + r) * columns.len() + ci];
                    check(
                        "random",
                        ACCURACY_BOUND_EPS,
                        row0 + r,
                        c,
                        y[r * m.shape.n + c],
                        sum,
                        magnitude,
                    );
                }
            }
        }
        let y = rig.run(16, 16, 48);
        let worst_random = worst.get();
        for (r, &c) in coherent.iter().enumerate() {
            let (sum, magnitude) = reference(m, &x[(16 + r) * m.shape.k..(17 + r) * m.shape.k], 1, &[c])[0];
            assert!(
                (sum.abs() - magnitude).abs() <= magnitude * 1e-12,
                "the coherent row is not coherent"
            );
            check(
                "coherent",
                COHERENT_BOUND_EPS,
                16 + r,
                c,
                y[r * m.shape.n + c],
                sum,
                magnitude,
            );
        }
        eprintln!(
            "cuda_matmul accuracy: {:<6} {} columns x 16 random rows, and 16 coherent rows, against f64: worst error {worst_random:.3} \
             and {:.3} epsilons of the sum of |terms| (bounds {ACCURACY_BOUND_EPS} and {COHERENT_BOUND_EPS})",
            m.name,
            columns.len(),
            worst.get()
        );
        worst_all = worst_all.max(worst.get());
        rig.free();
    }
    eprintln!(
        "cuda_matmul accuracy: worst of all {worst_all:.3} epsilons (epsilon = {:e})",
        f32::EPSILON
    );
}

#[test]
fn the_bits_of_a_row_do_not_depend_on_the_count_of_rows() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        let n = m.shape.n;
        let x = make_x(16, m.shape.k, 2);
        let mut rig = Rig::new(&dev, m, &x, 1, false);
        let full = rig.run(16, 0, 48);
        // The same run again: the merging block can change from run to run, and the bits must not.
        assert_eq!(
            bits(&full),
            bits(&rig.run(16, 0, 48)),
            "{}: two runs of 16 rows differ",
            m.name
        );
        assert!(
            rig.counters().iter().all(|c| *c == 0),
            "{}: a merge counter is not 0 after the run",
            m.name
        );
        for r in 0..16 {
            let alone = rig.run(1, r, 48);
            assert_eq!(
                bits(&alone),
                bits(&full[r * n..(r + 1) * n]),
                "{}: row {r} alone differs from row {r} of 16 rows",
                m.name
            );
        }
        // Rows 11 to 15 sit at the places 0 to 4 of a pass of 5 rows: another lane, and one MMA, not two.
        let five = rig.run(5, 11, 48);
        for i in 0..5 {
            assert_eq!(
                bits(&five[i * n..(i + 1) * n]),
                bits(&full[(11 + i) * n..(12 + i) * n]),
                "{}: row {} in a pass of 5 rows differs from row {} of 16 rows",
                m.name,
                i,
                11 + i
            );
        }
        // 8 and 9 rows: the border of the second MMA.
        for rows in [8, 9] {
            let part = rig.run(rows, 0, 48);
            assert_eq!(
                bits(&part),
                bits(&full[..rows * n]),
                "{}: a pass of {rows} rows differs",
                m.name
            );
        }
        eprintln!(
            "cuda_matmul rows: {:<6} 16 rows = 16 single rows = 5 rows (11..16) = 8 and 9 rows, bit for bit",
            m.name
        );
        rig.free();
    }
}

#[test]
fn the_bits_do_not_depend_on_the_grid() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        let x = make_x(16, m.shape.k, 3);
        let mut rig = Rig::new(&dev, m, &x, 1, false);
        for rows in [16, 1] {
            let first = rig.run(rows, 0, GRIDS[0]);
            for &grid in &GRIDS[1..] {
                for run in 0..2 {
                    let got = rig.run(rows, 0, grid);
                    assert_eq!(
                        bits(&got),
                        bits(&first),
                        "{}: {rows} rows at grid {grid}, run {run} differ from grid 48",
                        m.name
                    );
                }
            }
            assert!(
                rig.counters().iter().all(|c| *c == 0),
                "{}: a merge counter is not 0",
                m.name
            );
        }
        eprintln!(
            "cuda_matmul grid: {:<6} the bits at grids {GRIDS:?} are equal, for 16 rows and 1 row",
            m.name
        );
        rig.free();
    }
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

#[test]
fn the_speed_of_each_shape_and_the_head_at_the_peak() {
    let Some(matrices) = matrices() else { return };
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN).expect("open the GPU");
    let blocks = dev.sm_count() as usize;
    eprintln!(
        "cuda_matmul speed: the weights are read from DRAM: each task of a chain reads another copy of W, 400 MB or more in all"
    );
    eprintln!("cuda_matmul speed: GB/s counts the bytes of W only; the peak is {PEAK_GBPS} GB/s");
    let mut head_one_row = 0.0f64;
    for m in matrices {
        let bytes = m.shape.packed_bytes();
        let copies = (400_000_000usize.div_ceil(bytes)).clamp(1, 128);
        let tasks = if copies == 1 { 6 } else { copies * 2 };
        let x = make_x(16, m.shape.k, 4);
        for rows in [1usize, 16] {
            let mut rig = Rig::new(&dev, m, &x, copies, true);
            let list: Vec<Task> = (0..tasks).map(|i| rig.task(rows, 0, i)).collect();
            rig.ring.set(&list);
            run_with_watchdog(&rig.ring, blocks as u32, rig.shared, 120);
            let profile = rig.ring.read_profile().unwrap();
            let spans: Vec<u64> = (1..tasks)
                .map(|t| task_ns(&profile, blocks, rig.ring.ring_index(t)))
                .collect();
            let typical = median(spans.clone());
            let best = *spans.iter().min().unwrap();
            let gbps = |ns: u64| bytes as f64 / ns as f64;
            eprintln!(
                "cuda_matmul speed: {:<6} N {:>6} K {} {rows:>2} rows: {:>8.1} us (best {:>8.1} us), W {:>7.2} MB, {:>6.1} GB/s ({:>5.1}% of peak), best {:>6.1} GB/s",
                m.name,
                m.shape.n,
                m.shape.k,
                typical as f64 / 1e3,
                best as f64 / 1e3,
                bytes as f64 / 1e6,
                gbps(typical),
                gbps(typical) / PEAK_GBPS * 100.0,
                gbps(best)
            );
            if m.name == "head" && rows == 1 {
                head_one_row = gbps(typical);
                // The profile of the last task: the time of each block, and the spread.
                let last = rig.ring.ring_index(tasks - 1);
                let slots = &profile[last * blocks * PROFILE_WORDS..(last + 1) * blocks * PROFILE_WORDS];
                let exec: Vec<u64> = slots.chunks(PROFILE_WORDS).map(|s| s[2] - s[1]).collect();
                let begin = slots.chunks(PROFILE_WORDS).map(|s| s[1]).collect::<Vec<_>>();
                let end = slots.chunks(PROFILE_WORDS).map(|s| s[2]).collect::<Vec<_>>();
                let (lo, hi) = (*exec.iter().min().unwrap(), *exec.iter().max().unwrap());
                eprintln!(
                    "cuda_matmul profile: head, 1 row, one task: block time min {:.1} us, median {:.1} us, max {:.1} us, spread {:.2}%",
                    lo as f64 / 1e3,
                    median(exec.clone()) as f64 / 1e3,
                    hi as f64 / 1e3,
                    (hi - lo) as f64 / hi as f64 * 100.0
                );
                eprintln!(
                    "cuda_matmul profile: head: the blocks start within {:.2} us and end within {:.2} us; {} items, {} or {} for each block",
                    (begin.iter().max().unwrap() - begin.iter().min().unwrap()) as f64 / 1e3,
                    (end.iter().max().unwrap() - end.iter().min().unwrap()) as f64 / 1e3,
                    m.shape.items(),
                    m.shape.items() / blocks,
                    m.shape.items().div_ceil(blocks)
                );
            }
            rig.free();
        }
    }
    assert!(
        head_one_row >= HEAD_GATE * PEAK_GBPS,
        "the head at one row reads {head_one_row:.1} GB/s, which is under {:.0}% of the peak of {PEAK_GBPS} GB/s",
        HEAD_GATE * 100.0
    );
}

// ---------------------------------------------------------------------------------------------------------------
// The host model: the bits of the matmul.
// ---------------------------------------------------------------------------------------------------------------
//
// The model of the bits is `nagaya_mk_cuda::mma_model` (the rule of the tensor core is in its header).

/// The bits that the op gives for the result of row `row` of `x` and column `column` of `W`: the model of the
/// order of the sums in `kernels/matmul_bf16.cuh`.
fn host_matmul(m: &Loaded, x: &[f32], column: usize) -> f32 {
    let k = m.shape.k;
    let w: Vec<f32> = (0..k).map(|i| bf16_to_f64(&m.w, column * k + i) as f32).collect();
    dot_bits(&w, x, m.shape.kt)
}

#[test]
fn the_bits_equal_the_host_model_of_the_mma_and_the_order_of_the_sums() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        let k = m.shape.k;
        let x = make_x(16, k, 6);
        let mut rig = Rig::new(&dev, m, &x, 1, false);
        // The columns cover each lane of a tile, and tiles from the start to the end of the matrix.
        let tiles = m.shape.tiles();
        let columns: Vec<usize> = (0..32).map(|i| (i * 977 % tiles) * 16 + (i % 16)).collect();
        for (rows, row0) in [(16, 0), (3, 12)] {
            let y = rig.run(rows, row0, 48);
            for r in 0..rows {
                let xr = &x[(row0 + r) * k..(row0 + r + 1) * k];
                for &c in &columns {
                    let want = host_matmul(m, xr, c);
                    let got = y[r * m.shape.n + c];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{}: row {} column {c}, {rows} rows: the GPU gives {got:e}, the host model gives {want:e}",
                        m.name,
                        row0 + r
                    );
                }
            }
        }
        eprintln!(
            "cuda_matmul model: {:<6} {} columns x 19 rows: the bits equal the host model",
            m.name,
            columns.len()
        );
        rig.free();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Row tiles: a pass of up to 512 rows (wave B, lane D).
// ---------------------------------------------------------------------------------------------------------------

/// The counts of rows of the row tile gates: one row, a part of a tile, one tile, one tile and one row, a few tiles,
/// and a block of a prefill.
const BLOCK_ROWS: [usize; 6] = [1, 7, 16, 17, 100, 512];

#[test]
fn the_bits_of_a_row_do_not_depend_on_the_count_of_rows_up_to_a_block() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        let n = m.shape.n;
        let x = make_x(512, m.shape.k, 7);
        let mut rig = Rig::with_rows(&dev, m, &x, 1, false, 512);
        let full = rig.run(512, 0, 48);
        assert_eq!(
            bits(&full),
            bits(&rig.run(512, 0, 48)),
            "{}: two runs of 512 rows differ",
            m.name
        );
        assert!(
            rig.counters().iter().all(|c| *c == 0),
            "{}: a merge counter is not 0",
            m.name
        );
        for rows in BLOCK_ROWS {
            let part = rig.run(rows, 0, 48);
            assert_eq!(
                bits(&part),
                bits(&full[..rows * n]),
                "{}: a pass of {rows} rows differs from the first {rows} rows of 512 rows",
                m.name
            );
        }
        // A short pass at the end of the block, and one that starts inside a tile: the place of a row in a pass does
        // not change its bits.
        for (rows, row0) in [(12, 500), (1, 511), (17, 33), (100, 250)] {
            let part = rig.run(rows, row0, 48);
            assert_eq!(
                bits(&part),
                bits(&full[row0 * n..(row0 + rows) * n]),
                "{}: {rows} rows from row {row0} differ from the same rows of 512 rows",
                m.name
            );
        }
        eprintln!(
            "cuda_matmul tiles: {:<6} the rows at counts {BLOCK_ROWS:?} and at other places equal the rows of 512, bit for bit",
            m.name
        );
        rig.free();
    }
}

#[test]
fn the_bits_of_a_block_do_not_depend_on_the_grid() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        // The head at one block would take long: it runs 100 rows at the narrow grids.
        let rows = if m.name == "head" { 100 } else { 512 };
        let x = make_x(rows, m.shape.k, 8);
        let mut rig = Rig::with_rows(&dev, m, &x, 1, false, rows);
        let first = rig.run(rows, 0, GRIDS[0]);
        for &grid in &GRIDS[1..] {
            let got = rig.run(rows, 0, grid);
            assert_eq!(
                bits(&got),
                bits(&first),
                "{}: {rows} rows at grid {grid} differ from grid 48",
                m.name
            );
            assert!(
                rig.counters().iter().all(|c| *c == 0),
                "{}: a merge counter is not 0",
                m.name
            );
        }
        eprintln!(
            "cuda_matmul tiles: {:<6} {rows} rows: the bits at grids {GRIDS:?} are equal",
            m.name
        );
        rig.free();
    }
}

#[test]
fn the_bits_of_rows_over_16_equal_the_host_model() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        let k = m.shape.k;
        let x = make_x(512, k, 9);
        let mut rig = Rig::with_rows(&dev, m, &x, 1, false, 512);
        let tiles = m.shape.tiles();
        let columns: Vec<usize> = (0..16).map(|i| (i * 977 % tiles) * 16 + (i % 16)).collect();
        let y = rig.run(512, 0, 48);
        // Rows of the first tile, of the second, of the middle and of the last tile, with the border rows.
        for r in [0, 7, 15, 16, 17, 100, 255, 256, 495, 496, 504, 511] {
            for &c in &columns {
                let want = host_matmul(m, &x[r * k..(r + 1) * k], c);
                let got = y[r * m.shape.n + c];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{}: row {r} column {c} of 512 rows: the GPU gives {got:e}, the host model gives {want:e}",
                    m.name
                );
            }
        }
        eprintln!(
            "cuda_matmul tiles: {:<6} 12 rows x {} columns of 512 rows equal the host model",
            m.name,
            columns.len()
        );
        rig.free();
    }
}

#[test]
fn the_speed_of_a_block_of_512_rows() {
    let Some(matrices) = matrices() else { return };
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN).expect("open the GPU");
    let blocks = dev.sm_count() as usize;
    for m in matrices {
        let bytes = m.shape.packed_bytes();
        let x = make_x(512, m.shape.k, 10);
        let mut rig = Rig::with_rows(&dev, m, &x, 1, true, 512);
        let tasks = 5;
        let list: Vec<Task> = (0..tasks).map(|_| rig.task(512, 0, 0)).collect();
        rig.ring.set(&list);
        run_with_watchdog(&rig.ring, blocks as u32, rig.shared, 120);
        let profile = rig.ring.read_profile().unwrap();
        let spans: Vec<u64> = (1..tasks)
            .map(|t| task_ns(&profile, blocks, rig.ring.ring_index(t)))
            .collect();
        let typical = median(spans);
        let flops = 2.0 * 512.0 * m.shape.n as f64 * m.shape.k as f64;
        eprintln!(
            "cuda_matmul tiles speed: {:<6} N {:>6} K {} 512 rows: {:>9.1} us, {:>5.1} TFLOP/s of f32-equivalent work ({:>5.1} of bf16 with the three terms; the peak is 127), W {:.1} GB/s",
            m.name,
            m.shape.n,
            m.shape.k,
            typical as f64 / 1e3,
            flops / typical as f64 / 1e3,
            3.0 * flops / typical as f64 / 1e3,
            bytes as f64 / typical as f64
        );
        rig.free();
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The block form (`FLAG_BLOCK_FORM`, lane F): the bits of each row equal those of the decode form.
// ---------------------------------------------------------------------------------------------------------------

/// The counts of rows of the gate of the block form: more than a tile of 16 rows, one block of 32 rows, a part of the
/// next, a few blocks, and a block of a prefill.
const FORM_ROWS: [usize; 6] = [17, 32, 33, 100, 511, 512];

/// The rows and the columns that the host model checks in the block form.
const FORM_MODEL_ROWS: [usize; 8] = [0, 16, 17, 31, 32, 33, 100, 511];

#[test]
fn the_block_form_equals_the_decode_form_and_the_host_model() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        let n = m.shape.n;
        let k = m.shape.k;
        // The head has 201,024 columns: the decode form takes long, so it runs fewer counts.
        let counts: &[usize] = if m.name == "head" { &[33, 100] } else { &FORM_ROWS };
        let x = make_x(512, k, 11);
        let mut decode = Rig::with_rows(&dev, m, &x, 1, false, 512);
        let mut block = Rig::with_rows(&dev, m, &x, 1, false, 512);
        block.block = true;
        for &rows in counts {
            let want = decode.run(rows, 0, 48);
            let got = block.run(rows, 0, 48);
            assert_eq!(
                bits(&got),
                bits(&want),
                "{}: {rows} rows: the block form differs from the decode form",
                m.name
            );
        }
        // The host model of the order of the sums (`dot_bits`) gives the same bits as both forms.
        let rows = 512.min(if m.name == "head" { 100 } else { 512 });
        let y = block.run(rows, 0, 48);
        let tiles = m.shape.tiles();
        let columns: Vec<usize> = (0..16).map(|i| (i * 977 % tiles) * 16 + (i % 16)).collect();
        for r in FORM_MODEL_ROWS.into_iter().filter(|r| *r < rows) {
            for &c in &columns {
                let want = host_matmul(m, &x[r * k..(r + 1) * k], c);
                let got = y[r * n + c];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{}: row {r} column {c}: the block form gives {got:e}, the host model gives {want:e}",
                    m.name
                );
            }
        }
        eprintln!(
            "cuda_matmul block: {:<6} rows {counts:?}: equal to the decode form; {} rows x {} columns equal to the host model",
            m.name,
            FORM_MODEL_ROWS.iter().filter(|r| **r < rows).count(),
            columns.len()
        );
        decode.free();
        block.free();
    }
}

/// `wo` (N 2304, K 3072) is the matmul of `OP_OUT_MIX` when a big block splits it into a block-form matmul and the rest.
/// The raw sums must keep the bits of the decode form (the fused op) at every count of rows, including the counts below
/// 17, and at every grid.
#[test]
fn the_block_form_of_wo_equals_the_decode_form_at_1_to_512_rows_and_every_grid() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    let m = matrices.iter().find(|m| m.name == "wo_ud").expect("wo is in the list");
    let x = make_x(512, m.shape.k, 21);
    let mut decode = Rig::with_rows(&dev, m, &x, 1, false, 512);
    let mut block = Rig::with_rows(&dev, m, &x, 1, false, 512);
    block.block = true;
    for rows in [1usize, 2, 7, 16, 17, 33, 64, 65, 100, 129, 512] {
        let want = decode.run(rows, 0, 48);
        for grid in GRIDS {
            let got = block.run(rows, 0, grid);
            assert_eq!(
                bits(&got),
                bits(&want),
                "wo: {rows} rows at grid {grid}: the block form differs from the decode form"
            );
        }
    }
    eprintln!("cuda_matmul block: wo equals the decode form at 11 row counts and grids {GRIDS:?}");
    decode.free();
    block.free();
}

#[test]
fn the_bits_of_a_block_form_row_do_not_depend_on_the_count_of_rows_or_the_grid() {
    let Some(matrices) = matrices() else { return };
    let dev = open();
    for m in matrices {
        let n = m.shape.n;
        let rows = if m.name == "head" { 100 } else { 512 };
        let x = make_x(rows, m.shape.k, 12);
        let mut rig = Rig::with_rows(&dev, m, &x, 1, false, rows);
        rig.block = true;
        let first = rig.run(rows, 0, GRIDS[0]);
        for &grid in &GRIDS[1..] {
            assert_eq!(
                bits(&rig.run(rows, 0, grid)),
                bits(&first),
                "{}: {rows} rows at grid {grid} differ from grid 48",
                m.name
            );
        }
        // A part of the rows, from the start and from the middle of the block.
        for count in FORM_ROWS.into_iter().filter(|c| *c <= rows) {
            assert_eq!(
                bits(&rig.run(count, 0, 48)),
                bits(&first[..count * n]),
                "{}: {count} rows differ from the first {count} rows of {rows}",
                m.name
            );
        }
        for (count, row0) in [(12, rows - 12), (1, rows - 1), (17, 33), (40, 31)] {
            assert_eq!(
                bits(&rig.run(count, row0, 48)),
                bits(&first[row0 * n..(row0 + count) * n]),
                "{}: {count} rows from row {row0} differ",
                m.name
            );
        }
        eprintln!(
            "cuda_matmul block: {:<6} {rows} rows: the bits at grids {GRIDS:?} and at other counts and places of rows are equal",
            m.name
        );
        rig.free();
    }
}

#[test]
fn the_speed_of_the_block_form() {
    let Some(matrices) = matrices() else { return };
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN).expect("open the GPU");
    let blocks = dev.sm_count() as usize;
    for m in matrices {
        let bytes = m.shape.packed_bytes();
        // The head at 512 rows takes 100 ms in the decode form: it runs the decode form at 33 rows only.
        let counts: &[usize] = if m.name == "head" {
            &[33, 512]
        } else {
            &[17, 33, 64, 128, 256, 512]
        };
        for &rows in counts {
            let x = make_x(512, m.shape.k, 13);
            for block in [false, true] {
                if m.name == "head" && !block && rows == 512 {
                    continue;
                }
                let mut rig = Rig::with_rows(&dev, m, &x, 1, true, 512);
                rig.block = block;
                let tasks = 5;
                let list: Vec<Task> = (0..tasks).map(|_| rig.task(rows, 0, 0)).collect();
                rig.ring.set(&list);
                run_with_watchdog(&rig.ring, blocks as u32, rig.shared, 120);
                let profile = rig.ring.read_profile().unwrap();
                let spans: Vec<u64> = (1..tasks)
                    .map(|t| task_ns(&profile, blocks, rig.ring.ring_index(t)))
                    .collect();
                let typical = median(spans);
                let flops = 2.0 * rows as f64 * m.shape.n as f64 * m.shape.k as f64;
                eprintln!(
                    "cuda_matmul block speed: {:<6} N {:>5} K {} {rows:>3} rows {:<6}: {:>8.1} us, {:>5.1} TFLOP/s (f32-equivalent; the bf16 peak with 3 terms is 127/3 = 42), W {:>6.1} GB/s",
                    m.name,
                    m.shape.n,
                    m.shape.k,
                    if block { "block" } else { "decode" },
                    typical as f64 / 1e3,
                    flops / typical as f64 / 1e3,
                    bytes as f64 / typical as f64
                );
                rig.free();
            }
        }
    }
}
