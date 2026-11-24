//! Helpers of the long-context gates of `OP_ATTN`: the seeded layers of Inkling 2 Mini, the pool of keys and values, the
//! planes, the host form of a row and the launch of one task. They follow `cuda_attn.rs`, which stays as it is.

#![allow(dead_code)]
#![cfg(feature = "cuda")]

pub mod merge_model;

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use nagaya_mk_cuda::attn::{
    self, AttnShape, CacheFormat, GROUP, HEAD_DIM, HEADS, KV_HEADS, REL_EXTENT, REL_STATES, plane_bytes, row_of,
    row_offset, sliding_ring_rows,
};
use nagaya_mk_cuda::driver::{CUdeviceptr, CudaDevice};
use nagaya_mk_cuda::protocol::{ATTN_CHUNK_KEYS, MkTaskTable};
use nagaya_mk_cuda::ring::{DEFAULT_BLOCK_THREADS, Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, INKLING2_PROFILE_CUBIN, test_cards};
use nagaya_mk_inkling2_bmg::host::ops;
use nagaya_mk_inkling2_bmg::host::weights::Checkpoint;
use nagaya_mk_inkling2_sm121::cache::{pack_row, unpack_row};

pub const MODEL_VAR: &str = "INKLING2_MODEL_DIR";
pub const CHECKPOINT: &str = "model-00001-of-00001.safetensors";
/// The read peak of the GB10 that the probe measured, in GB/s (`docs/engine-abi.md`, "The facts of the GB10").
pub const PEAK_GBPS: f64 = 252.0;
/// A decode step of a full run at 65,536 positions must read the cache at this share of the peak.
pub const SPEED_GATE: f64 = 0.80;
/// The distinct (key, value) rows of the pool. Position `j` reads the row `j % POOL`.
pub const POOL: usize = 1024;
pub const GRIDS: [u32; 4] = [48, 24, 7, 1];
pub const WINDOW: usize = 1024;
pub const MAX_BLOCK: usize = 512;

/// The bound of the accuracy of an f32 cache against the host form, relative to the largest output of the row. The
/// worst error that the gate measured is printed by each test.
pub const ACCURACY_BOUND: f64 = 4e-6;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A normal value, by the sum of 12 uniform values.
    fn normal(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..12 {
            s += (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        }
        s - 6.0
    }
}

/// Run `f(i)` for each `i` in `0..n` on all the threads, and return the results in order.
pub fn parallel<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(n.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= n {
                            return mine;
                        }
                        mine.push((i, f(i)));
                    }
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, v)| v).collect()
}

/// The weights of one layer that the attention inputs need, widened to f32 (exact).
pub struct Layer {
    pub full: bool,
    pub wq: Vec<f32>,
    pub wr: Vec<f32>,
    pub sink: Vec<f32>,
    pub gate: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub q_scale: Vec<f32>,
    pub rel_scale: Vec<f32>,
    pub proj: Vec<f32>,
    /// The projection as the checkpoint holds it, `bf16`, for the device.
    proj_bf16: Vec<u8>,
    /// The pool of keys and values: `POOL` positions of `KV_HEADS` rows each.
    pub keys: Vec<[f32; HEAD_DIM]>,
    pub values: Vec<[f32; HEAD_DIM]>,
}

pub fn widen(ck: &Checkpoint, name: &str, rows: usize, cols: usize) -> Vec<f32> {
    let m = ck.matrix(name, rows, cols).unwrap_or_else(|e| panic!("{e}"));
    let mut out = vec![0f32; rows * cols];
    ops::widen_bf16_row(ck.of(&m), &mut out);
    out
}

/// `out[i] = W x_i` for each row `x_i` of `xs`, on all the threads.
pub fn matvec_rows(w: &[f32], n: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    parallel(xs.len(), |i| {
        let mut out = vec![0f32; n];
        ops::matvec(w, &xs[i], &mut out);
        out
    })
}

/// The layer `l`, or `None` when the model is absent. The test skips then, and with `NAGAYA_REQUIRE_MODELS` it fails.
pub fn layer(l: usize) -> Option<&'static Layer> {
    static CELLS: [OnceLock<Option<Layer>>; 2] = [OnceLock::new(), OnceLock::new()];
    let cell = &CELLS[usize::from(l == 5)];
    cell.get_or_init(|| {
        let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
        let Some(dir) = std::env::var_os(MODEL_VAR).map(PathBuf::from) else {
            assert!(!require, "{MODEL_VAR} is not set and NAGAYA_REQUIRE_MODELS is set");
            eprintln!("cuda_long: skipped, {MODEL_VAR} is not set");
            return None;
        };
        let path = dir.join(CHECKPOINT);
        if !path.exists() {
            assert!(
                !require,
                "{} is absent and NAGAYA_REQUIRE_MODELS is set",
                path.display()
            );
            eprintln!("cuda_long: skipped, {} is absent", path.display());
            return None;
        }
        let ck = Checkpoint::open(&path).unwrap_or_else(|e| panic!("{e}"));
        let p = |name: &str| format!("model.llm.layers.{l}.attn.{name}");
        let qd = HEADS * HEAD_DIM;
        let kvd = KV_HEADS * HEAD_DIM;
        let hidden = ops::HIDDEN;
        let proj_name = p("rel_logits_proj.proj");
        let proj_matrix = ck
            .matrix(&proj_name, REL_STATES, REL_EXTENT)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut layer = Layer {
            full: l % 6 == 5,
            wq: widen(&ck, &p("wq_du.weight"), qd, hidden),
            wr: widen(&ck, &p("wr_du.weight"), HEADS * REL_STATES, hidden),
            sink: widen(&ck, &p("attn_sink_proj.weight"), HEADS, hidden),
            gate: widen(&ck, &p("gate_attn.weight"), qd, hidden),
            wk: widen(&ck, &p("wk_dv.weight"), kvd, hidden),
            wv: widen(&ck, &p("wv_dv.weight"), kvd, hidden),
            q_scale: ck.f32s(&p("q_l2_clip.scale"), &[HEADS, HEAD_DIM]).unwrap(),
            rel_scale: ck.f32s(&p("rel_l2_clip.scale"), &[HEADS, REL_STATES]).unwrap(),
            proj: ck.f32s(&proj_name, &[REL_STATES, REL_EXTENT]).unwrap(),
            proj_bf16: ck.of(&proj_matrix).to_vec(),
            keys: Vec::new(),
            values: Vec::new(),
        };
        // The pool of keys and values: random hidden rows through the projections, then the clip of a key head and the
        // norm of a value head (`Inkling2LoopKeyValueProjection`).
        let xs: Vec<Vec<f32>> = (0..POOL)
            .map(|i| {
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15
                    ^ (i as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D)
                    ^ ((l as u64) << 56));
                (0..hidden).map(|_| rng.normal()).collect()
            })
            .collect();
        let ks = matvec_rows(&layer.wk, kvd, &xs);
        let vs = matvec_rows(&layer.wv, kvd, &xs);
        for (mut k, mut v) in ks.into_iter().zip(vs) {
            for head in k.as_chunks_mut::<HEAD_DIM>().0.iter_mut() {
                ops::l2_clip(head, ops::QK_CLIP);
            }
            for head in v.as_chunks_mut::<HEAD_DIM>().0.iter_mut() {
                ops::rms_norm_unweighted(head);
            }
            layer.keys.extend_from_slice(k.as_chunks::<HEAD_DIM>().0);
            layer.values.extend_from_slice(v.as_chunks::<HEAD_DIM>().0);
        }
        Some(layer)
    })
    .as_ref()
}

/// The inputs of `rows` query rows at the positions `pos0 ..`: the queries (clipped and, in a full run, log scaled), the
/// relative states, the sinks and the gates before their sigmoid.
#[derive(Clone)]
pub struct Rows {
    pub q: Vec<f32>,
    pub rel: Vec<f32>,
    pub sinks: Vec<f32>,
    pub gate: Vec<f32>,
}

pub fn make_rows(layer: &Layer, pos0: usize, rows: usize, seed: u64) -> Rows {
    let hidden = ops::HIDDEN;
    let xs: Vec<Vec<f32>> = (0..rows)
        .map(|i| {
            let mut rng =
                Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            (0..hidden).map(|_| rng.normal()).collect()
        })
        .collect();
    let q = matvec_rows(&layer.wq, HEADS * HEAD_DIM, &xs);
    let rel = matvec_rows(&layer.wr, HEADS * REL_STATES, &xs);
    let sinks = matvec_rows(&layer.sink, HEADS, &xs);
    let gate = matvec_rows(&layer.gate, HEADS * HEAD_DIM, &xs);
    let mut out = Rows {
        q: Vec::new(),
        rel: Vec::new(),
        sinks: Vec::new(),
        gate: Vec::new(),
    };
    for i in 0..rows {
        let (mut q, mut rel) = (q[i].clone(), rel[i].clone());
        ops::scaled_l2_clip(&mut q, &layer.q_scale, HEAD_DIM, ops::QK_CLIP);
        ops::scaled_l2_clip(&mut rel, &layer.rel_scale, REL_STATES, ops::REL_CLIP);
        if layer.full {
            let s = ops::log_scale(pos0 + i);
            q.iter_mut().chain(rel.iter_mut()).for_each(|v| *v *= s);
        }
        out.q.extend(q);
        out.rel.extend(rel);
        out.sinks.extend(&sinks[i]);
        out.gate.extend(&gate[i]);
    }
    out
}

/// The format of a cache of the layer: the pool rows, as the cache holds them (the values of an e4m3 cache are the
/// unpacked values), and the packed bytes of each.
pub struct Pool {
    pub format: CacheFormat,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub packed_keys: Vec<Vec<u8>>,
    pub packed_values: Vec<Vec<u8>>,
}

pub fn make_pool(layer: &Layer, format: CacheFormat) -> Pool {
    let n = format.row_bytes();
    let pack = |rows: &[[f32; HEAD_DIM]]| -> (Vec<Vec<f32>>, Vec<Vec<u8>>) {
        let mut vals = Vec::new();
        let mut bytes = Vec::new();
        for row in rows {
            let mut b = vec![0u8; n];
            pack_row(format, row, &mut b).unwrap();
            vals.push(unpack_row(format, &b).unwrap());
            bytes.push(b);
        }
        (vals, bytes)
    };
    let (keys, packed_keys) = pack(&layer.keys);
    let (values, packed_values) = pack(&layer.values);
    Pool {
        format,
        keys,
        values,
        packed_keys,
        packed_values,
    }
}

/// The bytes of the two planes with the positions 0 ..= `last` written, as a decode step at `last` finds them. Position
/// `j` holds the pool row `j % POOL` of each KV head. A row that no position wrote is zero.
pub fn build_planes(pool: &Pool, window: usize, plane_rows: usize, last: usize) -> (Vec<u8>, Vec<u8>) {
    let bytes = plane_bytes(plane_rows, pool.format);
    let mut k = vec![0u8; bytes];
    let mut v = vec![0u8; bytes];
    let n = pool.format.row_bytes();
    let first = if window == 0 {
        0
    } else {
        (last + 1).saturating_sub(plane_rows)
    };
    for j in first..=last {
        for kvh in 0..KV_HEADS {
            let at = row_offset(kvh, row_of(j, window, plane_rows), plane_rows, pool.format);
            k[at..at + n].copy_from_slice(&pool.packed_keys[(j % POOL) * KV_HEADS + kvh]);
            v[at..at + n].copy_from_slice(&pool.packed_values[(j % POOL) * KV_HEADS + kvh]);
        }
    }
    (k, v)
}

/// The host form of the attention of one row at `pos`, for the 24 heads: the output of `attention`, before `wo`.
pub fn host_row(layer: &Layer, pool: &Pool, rows: &Rows, row: usize, pos: usize, window: usize) -> Vec<f32> {
    let rel = &rows.rel[row * HEADS * REL_STATES..(row + 1) * HEADS * REL_STATES];
    let mut profiles = vec![0f32; HEADS * REL_EXTENT];
    ops::relative_logits(rel, &layer.proj, &mut profiles);
    let lowest = attn::lowest_key(pos, window);
    let heads = parallel(HEADS, |h| {
        let kvh = h / GROUP;
        let profile = &profiles[h * REL_EXTENT..(h + 1) * REL_EXTENT];
        let ks: Vec<&[f32]> = (lowest..=pos)
            .map(|j| &pool.keys[(j % POOL) * KV_HEADS + kvh][..])
            .collect();
        let vs: Vec<&[f32]> = (lowest..=pos)
            .map(|j| &pool.values[(j % POOL) * KV_HEADS + kvh][..])
            .collect();
        let bias: Vec<f32> = (lowest..=pos)
            .map(|j| ops::relative_bias(profile, (pos - j) as i64))
            .collect();
        let q = &rows.q[row * HEADS * HEAD_DIM + h * HEAD_DIM..row * HEADS * HEAD_DIM + (h + 1) * HEAD_DIM];
        let mut out = vec![0f32; HEAD_DIM];
        ops::attend(q, &ks, &vs, &bias, rows.sinks[row * HEADS + h], &mut out);
        let g = &rows.gate[row * HEADS * HEAD_DIM + h * HEAD_DIM..];
        for (o, g) in out.iter_mut().zip(g) {
            *o *= ops::sigmoid(*g);
        }
        out
    });
    heads.concat()
}
pub fn upload<T: Copy>(dev: &CudaDevice, data: &[T]) -> CUdeviceptr {
    let bytes = std::mem::size_of_val(data);
    let ptr = dev.alloc_device(bytes.max(16)).expect("allocate");
    unsafe { dev.copy_htod(ptr, data.as_ptr().cast::<c_void>(), bytes) }.expect("upload");
    ptr
}

pub fn download<T: Copy + Default>(dev: &CudaDevice, ptr: CUdeviceptr, count: usize) -> Vec<T> {
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

/// The planes and the projection of a case on the GPU.
pub struct Planes {
    pub k: CUdeviceptr,
    pub v: CUdeviceptr,
    pub proj: CUdeviceptr,
    pub format: CacheFormat,
    pub window: usize,
    pub plane_rows: usize,
}

impl Planes {
    pub fn new(dev: &CudaDevice, layer: &Layer, pool: &Pool, window: usize, plane_rows: usize, last: usize) -> Self {
        let (k, v) = build_planes(pool, window, plane_rows, last);
        Self {
            k: upload(dev, &k),
            v: upload(dev, &v),
            proj: upload(dev, &layer.proj_bf16),
            format: pool.format,
            window,
            plane_rows,
        }
    }

    pub fn free(self, dev: &CudaDevice) {
        for p in [self.k, self.v, self.proj] {
            dev.free_device(p).unwrap();
        }
    }
}

/// Run the watchdog around a launch, so that a hang fails the test and does not block the GPU.
pub fn run_with_watchdog(ring: &Ring<'_>, grid: u32, shared: u32, seconds: u64) {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for _ in 0..seconds * 100 {
                std::thread::sleep(std::time::Duration::from_millis(10));
                if done.load(Ordering::SeqCst) {
                    return;
                }
            }
            eprintln!("cuda_attn: the ring did not finish within {seconds} s at grid {grid}; it hangs");
            std::process::exit(2);
        });
        unsafe { ring.run_grid(grid, DEFAULT_BLOCK_THREADS, shared).unwrap() };
        done.store(true, Ordering::SeqCst);
    });
}

/// The result of one launch: the output, the wall time of the launch, and the time of the task in the kernel.
pub struct Run {
    pub out: Vec<f32>,
    pub wall_ns: u64,
    pub kernel_ns: u64,
}

/// Run one `OP_ATTN` task over `planes` for `rows` rows at `pos0`, with a grid of `grid` blocks. `launches` launches
/// follow the first one, and the times are the least of all.
#[allow(clippy::too_many_arguments)]
pub fn run_attn(
    dev: &CudaDevice,
    planes: &Planes,
    input: &Rows,
    rows: usize,
    pos0: usize,
    grid: u32,
    launches: usize,
    profile: bool,
) -> Run {
    run_attn_after(dev, planes, input, rows, pos0, grid, launches, profile, None)
}

/// Like [`run_attn`]. With `before` the ring first runs another task on the same scratch: the rows of its input, their
/// count and the position of the first. The ring is not zeroed between the two, as in a pass.
#[allow(clippy::too_many_arguments)]
pub fn run_attn_after(
    dev: &CudaDevice,
    planes: &Planes,
    input: &Rows,
    rows: usize,
    pos0: usize,
    grid: u32,
    launches: usize,
    profile: bool,
    before: Option<(&Rows, usize, usize)>,
) -> Run {
    let shape = AttnShape::new(rows, pos0, planes.window, planes.plane_rows, planes.format).unwrap();
    let before_shape =
        before.map(|(_, r, p)| AttnShape::new(r, p, planes.window, planes.plane_rows, planes.format).unwrap());
    let scratch_floats = shape
        .scratch_floats()
        .max(before_shape.as_ref().map_or(0, |s| s.scratch_floats()));
    let q = upload(dev, &input.q);
    let rel = upload(dev, &input.rel);
    let sinks = upload(dev, &input.sinks);
    let gate = upload(dev, &input.gate);
    let out = dev.alloc_device(rows * HEADS * HEAD_DIM * 4).unwrap();
    dev.memset_zero(out, rows * HEADS * HEAD_DIM * 4).unwrap();
    let table_host: [MkTaskTable; 1] = [attn::table(rel, sinks, gate, planes.proj)];
    let table = upload(dev, &table_host);
    let mut ring = Ring::new(dev, scratch_floats, profile).unwrap();
    ring.zero_device().unwrap();
    let shared = dev.max_dynamic_shared_bytes().unwrap();
    if let (Some((rows_in, r, p)), Some(bshape)) = (before, before_shape) {
        let bq = upload(dev, &rows_in.q);
        let brel = upload(dev, &rows_in.rel);
        let bsinks = upload(dev, &rows_in.sinks);
        let bgate = upload(dev, &rows_in.gate);
        let bout = dev.alloc_device(r * HEADS * HEAD_DIM * 4).unwrap();
        let btable = upload(dev, &[attn::table(brel, bsinks, bgate, planes.proj)]);
        let _ = p;
        let desc = bshape.descriptor(&attn::AttnBuffers {
            q: bq,
            k: planes.k,
            v: planes.v,
            out: bout,
            scratch: ring.scratch(),
            table: btable,
        });
        ring.set(&[Task::with_items(desc)
            .reads(bq, rows_in.q.len() * 4)
            .writes(bout, r * HEADS * HEAD_DIM * 4)]);
        run_with_watchdog(&ring, grid, shared, 120);
        for ptr in [bq, brel, bsinks, bgate, bout, btable] {
            dev.free_device(ptr).unwrap();
        }
    }
    let desc = shape.descriptor(&attn::AttnBuffers {
        q,
        k: planes.k,
        v: planes.v,
        out,
        scratch: ring.scratch(),
        table,
    });
    let task = Task::with_items(desc)
        .reads(q, input.q.len() * 4)
        .writes(out, rows * HEADS * HEAD_DIM * 4);
    ring.set(&[task]);
    let (mut wall, mut kernel) = (u64::MAX, u64::MAX);
    for _ in 0..=launches {
        let t0 = std::time::Instant::now();
        run_with_watchdog(&ring, grid, shared, 120);
        wall = wall.min(t0.elapsed().as_nanos() as u64);
        if profile {
            let times = ring.read_profile().unwrap();
            let g = ring.last_grid() as usize;
            // Task 0 is the attention. Each block has [start, after_wait, end].
            let start = (0..g).map(|b| times[b * 3 + 1]).min().unwrap();
            let end = (0..g).map(|b| times[b * 3 + 2]).max().unwrap();
            kernel = kernel.min(end - start);
        }
    }
    let result = download::<f32>(dev, out, rows * HEADS * HEAD_DIM);
    ring.free().unwrap();
    for p in [q, rel, sinks, gate, out, table] {
        dev.free_device(p).unwrap();
    }
    Run {
        out: result,
        wall_ns: wall,
        kernel_ns: kernel,
    }
}

pub fn open() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU")
}

/// Open the GPU with the cubin that holds only the cache format `format`.
pub fn open_cache(format: CacheFormat) -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), format.cubin()).expect("open the GPU with the cubin of the format")
}

/// Open the GPU with the profile cubin of the cache format `format`.
pub fn open_profile_cache(format: CacheFormat) -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), format.profile_cubin()).expect("open the GPU with the profile cubin")
}

pub fn open_profile() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN).expect("open the GPU with the profile cubin")
}

pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The worst error of `got` against `want`, relative to the largest `|want|` of the row of 3,072 values, and the largest
/// error in absolute terms.
pub fn worst_error(got: &[f32], want: &[f32]) -> (f64, f64) {
    let scale = want.iter().fold(0f64, |m, v| m.max(f64::from(v.abs()))).max(1e-30);
    let abs = got
        .iter()
        .zip(want)
        .fold(0f64, |m, (g, w)| m.max((f64::from(*g) - f64::from(*w)).abs()));
    (abs / scale, abs)
}

pub fn plane_rows_for(window: usize, last: usize) -> usize {
    if window == 0 {
        (last + 1).next_multiple_of(ATTN_CHUNK_KEYS)
    } else {
        sliding_ring_rows(window, MAX_BLOCK)
    }
}

/// Like [`host_row`], with the scores, the weights and the sums in `f64`. It tells the error of the `f32` host form from
/// the error of the device. The inputs are the same `f32` values.
pub fn exact_row(layer: &Layer, pool: &Pool, rows: &Rows, row: usize, pos: usize, window: usize) -> Vec<f32> {
    let rel = &rows.rel[row * HEADS * REL_STATES..(row + 1) * HEADS * REL_STATES];
    let mut profiles = vec![0f32; HEADS * REL_EXTENT];
    ops::relative_logits(rel, &layer.proj, &mut profiles);
    let lowest = attn::lowest_key(pos, window);
    let heads = parallel(HEADS, |h| {
        let kvh = h / GROUP;
        let profile = &profiles[h * REL_EXTENT..(h + 1) * REL_EXTENT];
        let q = &rows.q[row * HEADS * HEAD_DIM + h * HEAD_DIM..row * HEADS * HEAD_DIM + (h + 1) * HEAD_DIM];
        let scaling = (HEAD_DIM as f64).powf(-0.5);
        let sink = f64::from(rows.sinks[row * HEADS + h]);
        let scores: Vec<f64> = (lowest..=pos)
            .map(|j| {
                let k = &pool.keys[(j % POOL) * KV_HEADS + kvh];
                let dot: f64 = q.iter().zip(k).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
                dot * scaling + f64::from(ops::relative_bias(profile, (pos - j) as i64))
            })
            .collect();
        let max = scores.iter().copied().fold(sink, f64::max);
        let total: f64 = scores.iter().map(|s| (s - max).exp()).sum::<f64>() + (sink - max).exp();
        let mut sums = vec![0f64; HEAD_DIM];
        for (s, j) in scores.iter().zip(lowest..=pos) {
            let p = (s - max).exp() / total;
            let v = &pool.values[(j % POOL) * KV_HEADS + kvh];
            for (a, x) in sums.iter_mut().zip(v) {
                *a += p * f64::from(*x);
            }
        }
        let g = &rows.gate[row * HEADS * HEAD_DIM + h * HEAD_DIM..];
        sums.iter()
            .zip(g)
            .map(|(s, g)| (*s * f64::from(ops::sigmoid(*g))) as f32)
            .collect::<Vec<f32>>()
    });
    heads.concat()
}
