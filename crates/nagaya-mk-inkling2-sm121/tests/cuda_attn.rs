//! GPU gates of `OP_ATTN`, with the real projection weights of Inkling 2 Mini.
//!
//! The inputs come from the host: a pool of random hidden rows goes through the real `wq`, `wr`, `sink`, `gate_attn`, `wk`
//! and `wv` of one layer, then the clips, the log scale and the norm of the layer (`host/forward.rs`, `attention`). The
//! short convolution of the keys and values is left out, because the attention does not read it. A layer 5 is a full
//! run, and a layer 0 is a sliding run. The oracle is `ops::attend`, `ops::relative_logits` and `ops::relative_bias`
//! of the B70 crate.
//!
//! The gates:
//! - the accuracy of a decode row against the host form, at 1, 1,024, 4,096 and 65,536 positions;
//! - the bits of a row do not depend on the count of rows, on its place in the block, or on the grid (design rules 4
//!   and 5), in both modes of the op;
//! - the e4m3 cache: the f32 form of the op on the same packed cache gives the same bits (the expansion is exact), and
//!   the gap to the f32 cache is reported;
//! - the speed: the cache read of a decode step against the measured peak, and the time of a block of 512 rows.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_attn -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

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

const MODEL_VAR: &str = "INKLING2_MODEL_DIR";
const CHECKPOINT: &str = "model-00001-of-00001.safetensors";
/// The read peak of the GB10 that the probe measured, in GB/s (`docs/engine-abi.md`, "The facts of the GB10").
const PEAK_GBPS: f64 = 252.0;
/// A decode step of a full run at 65,536 positions must read the cache at this share of the peak.
const SPEED_GATE: f64 = 0.80;
/// The distinct (key, value) rows of the pool. Position `j` reads the row `j % POOL`.
const POOL: usize = 1024;
const GRIDS: [u32; 4] = [48, 24, 7, 1];
const WINDOW: usize = 1024;
const MAX_BLOCK: usize = 512;

/// The bound of the accuracy of an f32 cache against the host form, relative to the largest output of the row. The
/// worst error that the gate measured is printed by each test.
const ACCURACY_BOUND: f64 = 4e-6;

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
fn parallel<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
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
struct Layer {
    full: bool,
    wq: Vec<f32>,
    wr: Vec<f32>,
    sink: Vec<f32>,
    gate: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    q_scale: Vec<f32>,
    rel_scale: Vec<f32>,
    proj: Vec<f32>,
    /// The projection as the checkpoint holds it, `bf16`, for the device.
    proj_bf16: Vec<u8>,
    /// The pool of keys and values: `POOL` positions of `KV_HEADS` rows each.
    keys: Vec<[f32; HEAD_DIM]>,
    values: Vec<[f32; HEAD_DIM]>,
}

fn widen(ck: &Checkpoint, name: &str, rows: usize, cols: usize) -> Vec<f32> {
    let m = ck.matrix(name, rows, cols).unwrap_or_else(|e| panic!("{e}"));
    let mut out = vec![0f32; rows * cols];
    ops::widen_bf16_row(ck.of(&m), &mut out);
    out
}

/// `out[i] = W x_i` for each row `x_i` of `xs`, on all the threads.
fn matvec_rows(w: &[f32], n: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    parallel(xs.len(), |i| {
        let mut out = vec![0f32; n];
        ops::matvec(w, &xs[i], &mut out);
        out
    })
}

/// The layer `l`, or `None` when the model is absent. The test skips then, and with `NAGAYA_REQUIRE_MODELS` it fails.
fn layer(l: usize) -> Option<&'static Layer> {
    static CELLS: [OnceLock<Option<Layer>>; 2] = [OnceLock::new(), OnceLock::new()];
    let cell = &CELLS[usize::from(l == 5)];
    cell.get_or_init(|| {
        let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
        let Some(dir) = std::env::var_os(MODEL_VAR).map(PathBuf::from) else {
            assert!(!require, "{MODEL_VAR} is not set and NAGAYA_REQUIRE_MODELS is set");
            eprintln!("cuda_attn: skipped, {MODEL_VAR} is not set");
            return None;
        };
        let path = dir.join(CHECKPOINT);
        if !path.exists() {
            assert!(
                !require,
                "{} is absent and NAGAYA_REQUIRE_MODELS is set",
                path.display()
            );
            eprintln!("cuda_attn: skipped, {} is absent", path.display());
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
struct Rows {
    q: Vec<f32>,
    rel: Vec<f32>,
    sinks: Vec<f32>,
    gate: Vec<f32>,
}

fn make_rows(layer: &Layer, pos0: usize, rows: usize, seed: u64) -> Rows {
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
struct Pool {
    format: CacheFormat,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    packed_keys: Vec<Vec<u8>>,
    packed_values: Vec<Vec<u8>>,
}

fn make_pool(layer: &Layer, format: CacheFormat) -> Pool {
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
fn build_planes(pool: &Pool, window: usize, plane_rows: usize, last: usize) -> (Vec<u8>, Vec<u8>) {
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
fn host_row(layer: &Layer, pool: &Pool, rows: &Rows, row: usize, pos: usize, window: usize) -> Vec<f32> {
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

fn upload<T: Copy>(dev: &CudaDevice, data: &[T]) -> CUdeviceptr {
    let bytes = std::mem::size_of_val(data);
    let ptr = dev.alloc_device(bytes.max(16)).expect("allocate");
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

/// The planes and the projection of a case on the GPU.
struct Planes {
    k: CUdeviceptr,
    v: CUdeviceptr,
    proj: CUdeviceptr,
    format: CacheFormat,
    window: usize,
    plane_rows: usize,
}

impl Planes {
    fn new(dev: &CudaDevice, layer: &Layer, pool: &Pool, window: usize, plane_rows: usize, last: usize) -> Self {
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

    fn free(self, dev: &CudaDevice) {
        for p in [self.k, self.v, self.proj] {
            dev.free_device(p).unwrap();
        }
    }
}

/// Run the watchdog around a launch, so that a hang fails the test and does not block the GPU.
fn run_with_watchdog(ring: &Ring<'_>, grid: u32, shared: u32, seconds: u64) {
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
struct Run {
    out: Vec<f32>,
    wall_ns: u64,
    kernel_ns: u64,
}

/// Run one `OP_ATTN` task over `planes` for `rows` rows at `pos0`, with a grid of `grid` blocks. `launches` launches
/// follow the first one, and the times are the least of all.
#[allow(clippy::too_many_arguments)]
fn run_attn(
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
fn run_attn_after(
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

fn open() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU")
}

/// Open the GPU with the cubin that holds only the cache format `format`.
fn open_cache(format: CacheFormat) -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), format.cubin()).expect("open the GPU with the cubin of the format")
}

/// Open the GPU with the profile cubin of the cache format `format`.
fn open_profile_cache(format: CacheFormat) -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), format.profile_cubin()).expect("open the GPU with the profile cubin")
}

fn open_profile() -> CudaDevice {
    CudaDevice::open_by_pci(&test_cards::card(), INKLING2_PROFILE_CUBIN).expect("open the GPU with the profile cubin")
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The worst error of `got` against `want`, relative to the largest `|want|` of the row of 3,072 values, and the largest
/// error in absolute terms.
fn worst_error(got: &[f32], want: &[f32]) -> (f64, f64) {
    let scale = want.iter().fold(0f64, |m, v| m.max(f64::from(v.abs()))).max(1e-30);
    let abs = got
        .iter()
        .zip(want)
        .fold(0f64, |m, (g, w)| m.max((f64::from(*g) - f64::from(*w)).abs()));
    (abs / scale, abs)
}

fn plane_rows_for(window: usize, last: usize) -> usize {
    if window == 0 {
        (last + 1).next_multiple_of(ATTN_CHUNK_KEYS)
    } else {
        sliding_ring_rows(window, MAX_BLOCK)
    }
}

#[test]
fn a_decode_row_matches_the_host_form_at_each_context_length() {
    decode_accuracy(None, CacheFormat::F32);
}

/// The same gate on the cubin of each cache format.
#[test]
fn a_decode_row_matches_the_host_form_in_each_cache_format() {
    for format in [CacheFormat::F32, CacheFormat::E4m3, CacheFormat::Bf16] {
        decode_accuracy(Some(format), format);
    }
}

/// `cubin` is the format of the cubin, or `None` for the default cubin. `format` is the format of the cache.
fn decode_accuracy(cubin: Option<CacheFormat>, format: CacheFormat) {
    let (Some(full), Some(slide)) = (layer(5), layer(0)) else {
        return;
    };
    let dev = cubin.map_or_else(open, open_cache);
    let mut worst = 0f64;
    // A full run at 1, 1,024, 4,096 and 65,536 positions, and a sliding run before and after its ring wraps.
    let cases: [(&Layer, usize, usize); 7] = [
        (full, 0, 1),
        (full, 0, 1024),
        (full, 0, 4096),
        (full, 0, 65_536),
        (slide, WINDOW, 1),
        (slide, WINDOW, 5000),
        (slide, WINDOW, 70_000),
    ];
    let pools = [make_pool(full, format), make_pool(slide, format)];
    for (layer, window, positions) in cases {
        let pool = &pools[usize::from(!layer.full)];
        let last = positions - 1;
        let planes = Planes::new(&dev, layer, pool, window, plane_rows_for(window, last), last);
        let input = make_rows(layer, last, 1, 11);
        let want = host_row(layer, pool, &input, 0, last, window);
        let run = run_attn(&dev, &planes, &input, 1, last, 48, 0, false);
        let (rel, abs) = worst_error(&run.out, &want);
        eprintln!(
            "cuda_attn: decode {format:?}, window {window:4}, {positions:6} positions: worst error {rel:.2e} of the row maximum ({abs:.2e} absolute)"
        );
        assert!(
            rel <= ACCURACY_BOUND,
            "error {rel:e} at {positions} positions, window {window}"
        );
        worst = worst.max(rel);
        planes.free(&dev);
    }
    eprintln!("cuda_attn: decode {format:?}: the worst error of all cases is {worst:.2e}, bound {ACCURACY_BOUND:.0e}");
}

/// The bits of every row of a block equal the bits of the same row run alone as a decode step, at every grid.
fn check_rows_and_grids(
    dev: &CudaDevice,
    layer: &Layer,
    pool: &Pool,
    window: usize,
    pos0: usize,
    rows: usize,
    sample: &[usize],
) {
    let last = pos0 + rows - 1;
    let planes = Planes::new(dev, layer, pool, window, plane_rows_for(window, last), last);
    let input = make_rows(layer, pos0, rows, 23);
    let reference = run_attn(dev, &planes, &input, rows, pos0, 48, 0, false).out;
    let width = HEADS * HEAD_DIM;
    for grid in GRIDS {
        let again = run_attn(dev, &planes, &input, rows, pos0, grid, 0, false).out;
        assert_eq!(
            bits(&again),
            bits(&reference),
            "{rows} rows from {pos0}: grid {grid} changes the bits"
        );
    }
    for &t in sample {
        let one = Rows {
            q: input.q[t * width..(t + 1) * width].to_vec(),
            rel: input.rel[t * HEADS * REL_STATES..(t + 1) * HEADS * REL_STATES].to_vec(),
            sinks: input.sinks[t * HEADS..(t + 1) * HEADS].to_vec(),
            gate: input.gate[t * width..(t + 1) * width].to_vec(),
        };
        let alone = run_attn(dev, &planes, &one, 1, pos0 + t, 48, 0, false).out;
        assert_eq!(
            bits(&alone),
            bits(&reference[t * width..(t + 1) * width]),
            "row {t} of {rows} rows from {pos0} differs from the same position as a decode step"
        );
    }
    planes.free(dev);
}

#[test]
fn the_bits_of_a_row_do_not_depend_on_the_rows_or_on_the_grid() {
    bits_do_not_depend_on_rows_or_grid(None, CacheFormat::F32);
}

/// The same gate on the cubin of each cache format.
#[test]
fn the_bits_of_a_row_do_not_depend_on_the_rows_or_on_the_grid_in_each_cache_format() {
    for format in [CacheFormat::F32, CacheFormat::E4m3, CacheFormat::Bf16] {
        bits_do_not_depend_on_rows_or_grid(Some(format), format);
    }
}

fn bits_do_not_depend_on_rows_or_grid(cubin: Option<CacheFormat>, format: CacheFormat) {
    let (Some(full), Some(slide)) = (layer(5), layer(0)) else {
        return;
    };
    let dev = &cubin.map_or_else(open, open_cache);
    let full_pool = make_pool(full, format);
    let slide_pool = make_pool(slide, format);
    // The split mode: 1, 7 and 16 rows. A block of 7 rows is 21 rows of the MMA, and 16 rows are 3 tiles.
    for rows in [7usize, 16] {
        let sample: Vec<usize> = (0..rows).collect();
        check_rows_and_grids(dev, full, &full_pool, 0, 4000, rows, &sample);
        check_rows_and_grids(dev, slide, &slide_pool, WINDOW, 5000, rows, &sample);
    }
    // The whole mode: a block of 17 rows, and a block of 512 rows from a position that is not on a chunk.
    check_rows_and_grids(dev, full, &full_pool, 0, 1000, 17, &[0, 1, 8, 16]);
    check_rows_and_grids(dev, full, &full_pool, 0, 3000, 512, &[0, 1, 255, 256, 300, 511]);
    check_rows_and_grids(dev, slide, &slide_pool, WINDOW, 3000, 512, &[0, 1, 255, 511]);
    eprintln!(
        "cuda_attn: {format:?}: the bits are equal for 7, 16, 17 and 512 rows against a decode step, and at grids {GRIDS:?}"
    );
}

/// A call of few rows leaves partials in the scratch where a call of more rows has its counters. The counters sit at
/// a place that does not depend on the rows, so the second call gives the bits of a fresh ring.
#[test]
fn a_call_after_a_call_of_other_rows_on_the_same_scratch_gives_the_bits_of_a_fresh_ring() {
    let Some(full) = layer(5) else {
        return;
    };
    let pool = make_pool(full, CacheFormat::F32);
    let dev = open();
    let planes = Planes::new(&dev, full, &pool, 0, plane_rows_for(0, 4100), 4100);
    for (first, second) in [(1usize, 16usize), (1, 7), (7, 16), (16, 1)] {
        let pos0 = 3000;
        let a = make_rows(full, pos0, first, 31);
        let b = make_rows(full, pos0 + first, second, 32);
        let fresh = run_attn(&dev, &planes, &b, second, pos0 + first, 48, 0, false).out;
        let after = run_attn_after(
            &dev,
            &planes,
            &b,
            second,
            pos0 + first,
            48,
            0,
            false,
            Some((&a, first, pos0)),
        )
        .out;
        assert_eq!(
            bits(&after),
            bits(&fresh),
            "{second} rows after {first} rows on the same scratch give other bits than a fresh ring"
        );
    }
    planes.free(&dev);
}

#[test]
fn a_block_of_rows_matches_the_host_form() {
    block_accuracy(None, CacheFormat::F32);
}

/// The same gate on the cubin of each cache format.
#[test]
fn a_block_of_rows_matches_the_host_form_in_each_cache_format() {
    for format in [CacheFormat::F32, CacheFormat::E4m3, CacheFormat::Bf16] {
        block_accuracy(Some(format), format);
    }
}

fn block_accuracy(cubin: Option<CacheFormat>, format: CacheFormat) {
    let (Some(full), Some(slide)) = (layer(5), layer(0)) else {
        return;
    };
    let dev = cubin.map_or_else(open, open_cache);
    let pools = [make_pool(full, format), make_pool(slide, format)];
    let mut worst = 0f64;
    // A block from position 0 has the causal mask of its own rows. A block later also reads the cache before it.
    for (layer, window, pos0, rows, sample) in [
        (full, 0usize, 0usize, 512usize, vec![0usize, 1, 2, 63, 255, 256, 511]),
        (full, 0, 8192, 512, vec![0, 1, 130, 511]),
        (slide, WINDOW, 0, 512, vec![0, 1, 511]),
        (slide, WINDOW, 8192, 512, vec![0, 1, 130, 511]),
        (full, 0, 100, 40, vec![0, 3, 39]),
    ] {
        let pool = &pools[usize::from(!layer.full)];
        let last = pos0 + rows - 1;
        let planes = Planes::new(&dev, layer, pool, window, plane_rows_for(window, last), last);
        let input = make_rows(layer, pos0, rows, 31);
        let run = run_attn(&dev, &planes, &input, rows, pos0, 48, 0, false);
        let width = HEADS * HEAD_DIM;
        for &t in &sample {
            let want = host_row(layer, pool, &input, t, pos0 + t, window);
            let (rel, _) = worst_error(&run.out[t * width..(t + 1) * width], &want);
            assert!(
                rel <= ACCURACY_BOUND,
                "row {t} of {rows} rows from {pos0}, window {window}: error {rel:e}"
            );
            worst = worst.max(rel);
        }
        planes.free(&dev);
    }
    eprintln!(
        "cuda_attn: blocks {format:?}: the worst error of the sampled rows is {worst:.2e}, bound {ACCURACY_BOUND:.0e}"
    );
}

#[test]
fn an_e4m3_cache_gives_the_bits_of_the_f32_form_on_the_same_packed_cache() {
    let (Some(full), Some(slide)) = (layer(5), layer(0)) else {
        return;
    };
    let dev = open();
    let mut worst_gap = 0f64;
    for (layer, window, pos0, rows) in [
        (full, 0usize, 4095usize, 1usize),
        (full, 0, 20_000, 1),
        (full, 0, 3000, 16),
        (full, 0, 1000, 64),
        (slide, WINDOW, 5000, 1),
        (slide, WINDOW, 3000, 40),
    ] {
        let f32_pool = make_pool(layer, CacheFormat::F32);
        let e4_pool = make_pool(layer, CacheFormat::E4m3);
        let last = pos0 + rows - 1;
        let plane_rows = plane_rows_for(window, last);
        let input = make_rows(layer, pos0, rows, 41);
        // The e4m3 cache, and the f32 cache that holds the values of the e4m3 cache.
        let packed = Planes::new(&dev, layer, &e4_pool, window, plane_rows, last);
        let expanded_pool = Pool {
            format: CacheFormat::F32,
            keys: e4_pool.keys.clone(),
            values: e4_pool.values.clone(),
            packed_keys: e4_pool
                .keys
                .iter()
                .map(|r| r.iter().flat_map(|v| v.to_le_bytes()).collect())
                .collect(),
            packed_values: e4_pool
                .values
                .iter()
                .map(|r| r.iter().flat_map(|v| v.to_le_bytes()).collect())
                .collect(),
        };
        let expanded = Planes::new(&dev, layer, &expanded_pool, window, plane_rows, last);
        let exact = Planes::new(&dev, layer, &f32_pool, window, plane_rows, last);
        let got = run_attn(&dev, &packed, &input, rows, pos0, 48, 0, false).out;
        let reference = run_attn(&dev, &expanded, &input, rows, pos0, 48, 0, false).out;
        assert_eq!(
            bits(&got),
            bits(&reference),
            "the e4m3 form differs from the f32 form on the same values: {rows} rows from {pos0}, window {window}"
        );
        // The gap to the f32 cache of the unpacked values, row by row.
        let wide = run_attn(&dev, &exact, &input, rows, pos0, 48, 0, false).out;
        let width = HEADS * HEAD_DIM;
        let gap = (0..rows)
            .map(|t| worst_error(&got[t * width..(t + 1) * width], &wide[t * width..(t + 1) * width]).0)
            .fold(0f64, f64::max);
        // The e4m3 form against the host form on the same values.
        let want = host_row(layer, &e4_pool, &input, 0, pos0, window);
        let (rel, _) = worst_error(&got[..width], &want);
        assert!(
            rel <= ACCURACY_BOUND,
            "the e4m3 form against the host form: error {rel:e}"
        );
        eprintln!(
            "cuda_attn: e4m3 {rows:3} rows from {pos0:6}, window {window:4}: bits equal to the f32 form; gap to the f32 cache {gap:.2e}; error of row 0 against the host form {rel:.2e}"
        );
        worst_gap = worst_gap.max(gap);
        for p in [packed, expanded, exact] {
            p.free(&dev);
        }
    }
    eprintln!("cuda_attn: e4m3: the largest gap to the f32 cache is {worst_gap:.2e} of the row maximum");
}

/// A pool of f32 rows that hold the values of `pool` (a bf16 or an e4m3 pool): the expansion is exact.
fn expanded(pool: &Pool) -> Pool {
    let bytes = |rows: &[Vec<f32>]| -> Vec<Vec<u8>> {
        rows.iter()
            .map(|r| r.iter().flat_map(|v| v.to_le_bytes()).collect())
            .collect()
    };
    Pool {
        format: CacheFormat::F32,
        keys: pool.keys.clone(),
        values: pool.values.clone(),
        packed_keys: bytes(&pool.keys),
        packed_values: bytes(&pool.values),
    }
}

/// A bf16 cache gives the bits of the f32 form on the same values (the expansion is exact, and the terms that the f32 form
/// adds are zero). The two forms run on the cubin of their own format, one after the other. The gap to the f32 cache of the
/// unrounded values and the error against the host form are reported.
#[test]
fn a_bf16_cache_gives_the_bits_of_the_f32_form_on_the_same_values() {
    let (Some(full), Some(slide)) = (layer(5), layer(0)) else {
        return;
    };
    let mut worst_gap = 0f64;
    let mut worst_rel = 0f64;
    for (layer, window, pos0, rows) in [
        (full, 0usize, 4095usize, 1usize),
        (full, 0, 20_000, 1),
        (full, 0, 3000, 16),
        (full, 0, 1000, 64),
        (slide, WINDOW, 5000, 1),
        (slide, WINDOW, 3000, 40),
    ] {
        let wide_pool = make_pool(layer, CacheFormat::F32);
        let bf_pool = make_pool(layer, CacheFormat::Bf16);
        let same_values = expanded(&bf_pool);
        let last = pos0 + rows - 1;
        let plane_rows = plane_rows_for(window, last);
        let input = make_rows(layer, pos0, rows, 41);
        let run = |format: CacheFormat, pool: &Pool| {
            let dev = open_cache(format);
            let planes = Planes::new(&dev, layer, pool, window, plane_rows, last);
            let out = run_attn(&dev, &planes, &input, rows, pos0, 48, 0, false).out;
            planes.free(&dev);
            out
        };
        let got = run(CacheFormat::Bf16, &bf_pool);
        let reference = run(CacheFormat::F32, &same_values);
        assert_eq!(
            bits(&got),
            bits(&reference),
            "the bf16 form differs from the f32 form on the same values: {rows} rows from {pos0}, window {window}"
        );
        let wide = run(CacheFormat::F32, &wide_pool);
        let width = HEADS * HEAD_DIM;
        let gap = (0..rows)
            .map(|t| worst_error_of(&got[t * width..(t + 1) * width], &wide[t * width..(t + 1) * width]))
            .fold(0f64, f64::max);
        let want = host_row(layer, &bf_pool, &input, 0, pos0, window);
        let (rel, _) = worst_error(&got[..width], &want);
        assert!(
            rel <= ACCURACY_BOUND,
            "the bf16 form against the host form: error {rel:e}"
        );
        eprintln!(
            "cuda_attn: bf16 {rows:3} rows from {pos0:6}, window {window:4}: bits equal to the f32 form; gap to the f32 cache {gap:.2e}; error of row 0 against the host form {rel:.2e}"
        );
        worst_gap = worst_gap.max(gap);
        worst_rel = worst_rel.max(rel);
    }
    eprintln!(
        "cuda_attn: bf16: the largest gap to the f32 cache is {worst_gap:.2e} of the row maximum; the worst error against the host form is {worst_rel:.2e}"
    );
}

fn worst_error_of(got: &[f32], want: &[f32]) -> f64 {
    worst_error(got, want).0
}

/// The bytes that a decode step reads from the cache: both planes over the visible positions of the 8 KV heads.
fn decode_bytes(format: CacheFormat, positions: usize, window: usize) -> usize {
    let visible = positions - attn::lowest_key(positions - 1, window);
    2 * visible * KV_HEADS * format.row_bytes()
}

#[test]
fn the_cache_read_of_a_decode_step_holds_the_speed_gate_at_65536_positions() {
    let Some(full) = layer(5) else { return };
    let positions = 65_536usize;
    let mut table = Vec::new();
    for format in [CacheFormat::F32, CacheFormat::E4m3, CacheFormat::Bf16] {
        let dev = open_profile_cache(format);
        let pool = make_pool(full, format);
        let planes = Planes::new(&dev, full, &pool, 0, plane_rows_for(0, positions - 1), positions - 1);
        let input = make_rows(full, positions - 1, 1, 7);
        let run = run_attn(&dev, &planes, &input, 1, positions - 1, 48, 20, true);
        let bytes = decode_bytes(format, positions, 0);
        let gbps = bytes as f64 / run.kernel_ns as f64;
        let wall = bytes as f64 / run.wall_ns as f64;
        eprintln!(
            "cuda_attn: speed {format:?}: {positions} positions, {:.1} MB, kernel {:.3} ms = {gbps:.1} GB/s ({:.0}% of {PEAK_GBPS}), wall {:.3} ms = {wall:.1} GB/s",
            bytes as f64 / 1e6,
            run.kernel_ns as f64 / 1e6,
            100.0 * gbps / PEAK_GBPS,
            run.wall_ns as f64 / 1e6
        );
        table.push((format, gbps));
        planes.free(&dev);
    }
    // The floor is on the f32 cache. The e4m3 and bf16 numbers are reported: they are bound by the same pipeline.
    assert!(
        table[0].1 >= SPEED_GATE * PEAK_GBPS,
        "an f32 decode step reads {:.1} GB/s, under {:.1} GB/s",
        table[0].1,
        SPEED_GATE * PEAK_GBPS
    );
}

#[test]
fn the_time_of_the_decode_steps_and_the_blocks_of_a_run_is_reported() {
    let (Some(full), Some(slide)) = (layer(5), layer(0)) else {
        return;
    };
    let dev = open_profile();
    let pools = [make_pool(full, CacheFormat::F32), make_pool(slide, CacheFormat::F32)];
    let e4 = [make_pool(full, CacheFormat::E4m3), make_pool(slide, CacheFormat::E4m3)];
    let bf = [make_pool(full, CacheFormat::Bf16), make_pool(slide, CacheFormat::Bf16)];
    let dev_bf = open_profile_cache(CacheFormat::Bf16);
    eprintln!("cuda_attn: time table (kernel time of the task, least of 5 launches)");
    for (pools, name) in [(&pools, "f32"), (&e4, "e4m3"), (&bf, "bf16")] {
        let dev = if name == "bf16" { &dev_bf } else { &dev };
        for (layer, window, pos0, rows) in [
            (full, 0usize, 1023usize, 1usize),
            (full, 0, 4095, 1),
            (full, 0, 16_383, 1),
            (slide, WINDOW, 5000, 1),
            (slide, WINDOW, 5000, 16),
            (full, 0, 4000, 16),
            (full, 0, 0, 512),
            (full, 0, 8192, 512),
            (slide, WINDOW, 0, 512),
            (slide, WINDOW, 8192, 512),
        ] {
            let pool = &pools[usize::from(!layer.full)];
            let last = pos0 + rows - 1;
            let planes = Planes::new(dev, layer, pool, window, plane_rows_for(window, last), last);
            let input = make_rows(layer, pos0, rows, 5);
            let run = run_attn(dev, &planes, &input, rows, pos0, 48, 4, true);
            let shape = AttnShape::new(rows, pos0, window, planes.plane_rows, pool.format).unwrap();
            let read = decode_bytes(pool.format, last + 1, window);
            eprintln!(
                "cuda_attn: {name:4} {} {rows:3} rows at {pos0:6}: {:8.1} us, {} items, cache {:6.1} MB = {:6.1} GB/s",
                if window == 0 { "full   " } else { "sliding" },
                run.kernel_ns as f64 / 1e3,
                shape.items(),
                read as f64 / 1e6,
                read as f64 / run.kernel_ns as f64
            );
            planes.free(dev);
        }
    }
}
