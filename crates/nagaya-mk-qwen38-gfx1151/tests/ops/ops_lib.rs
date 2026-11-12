//! The loader of the op libraries and the calls that most op tests share. It ports `tests/ops/mkops.py` of mk.

#![allow(dead_code)]

use crate::inputs::Qtype;
use libloading::Library;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Mutex;

/// The directory of the op libraries. `QWEN38_OPS_LIBDIR` selects another build, for a mutation build of an op.
pub fn libdir() -> PathBuf {
    if let Some(dir) = std::env::var_os("QWEN38_OPS_LIBDIR") {
        return PathBuf::from(dir);
    }
    match option_env!("NAGAYA_MK_OPS_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            panic!("build.rs did not build the op libraries. Build with the feature hip, or set QWEN38_OPS_LIBDIR.")
        }
    }
}

struct Loaded {
    library: &'static Library,
    sunk: bool,
}

static LIBRARIES: Mutex<Option<HashMap<String, Loaded>>> = Mutex::new(None);

/// The library `libqwen38_<op>.so`. The first call loads it. With `trap_sink` the library reports its device traps
/// (`rt/trap.h`): the call installs the trap record and the SIGABRT handler. That touches the GPU.
pub fn lib_with(op: &str, trap_sink: bool) -> &'static Library {
    let mut guard = LIBRARIES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    let entry = map.entry(op.to_owned()).or_insert_with(|| {
        let path = libdir().join(format!("libqwen38_{op}.so"));
        let library = unsafe { Library::new(&path) }.unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
        Loaded {
            library: Box::leak(Box::new(library)),
            sunk: false,
        }
    });
    if trap_sink && !entry.sunk {
        let install: libloading::Symbol<unsafe extern "C" fn() -> i32> =
            unsafe { entry.library.get(b"mk_trap_sink_install\0") }.expect("mk_trap_sink_install");
        assert_eq!(unsafe { install() }, 0, "the trap sink of {op} did not install");
        entry.sunk = true;
    }
    entry.library
}

/// The library `libqwen38_<op>.so`, loaded as a raw load leaves it: with no trap sink, and marked so that `lib` does not
/// install the sink later. It is the state of a library that a program loads with no knowledge of the sink.
pub fn lib_raw(op: &str) -> &'static Library {
    let library = lib_with(op, false);
    let mut guard = LIBRARIES.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.as_mut().expect("loaded").get_mut(op).expect("loaded").sunk = true;
    library
}

/// The library `libqwen38_<op>.so`, with its trap sink.
pub fn lib(op: &str) -> &'static Library {
    lib_with(op, true)
}

/// Look up `name` in `library` as a function of the given type.
#[macro_export]
macro_rules! sym {
    ($lib:expr, $name:literal, fn($($arg:ty),* $(,)?) -> $ret:ty) => {{
        let library: &::libloading::Library = $lib;
        let symbol: ::libloading::Symbol<unsafe extern "C" fn($($arg),*) -> $ret> =
            unsafe { library.get(concat!($name, "\0").as_bytes()) }
                .unwrap_or_else(|error| panic!("symbol {}: {error}", $name));
        symbol
    }};
}

pub fn ptr<T>(values: &[T]) -> *const c_void {
    values.as_ptr().cast()
}

pub fn ptr_mut<T>(values: &mut [T]) -> *mut c_void {
    values.as_mut_ptr().cast()
}

pub fn nbytes<T>(values: &[T]) -> u64 {
    std::mem::size_of_val(values) as u64
}

/// The pointer for an optional buffer: null for none.
pub fn opt_ptr<T>(values: Option<&[T]>) -> *const c_void {
    values.map_or(std::ptr::null(), ptr)
}

/// The byte size for an optional buffer: 0 for none.
pub fn opt_nbytes<T>(values: Option<&[T]>) -> u64 {
    values.map_or(0, nbytes)
}

pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// True when both slices have the same f32 bits.
pub fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The number of values that differ in bits, and the first one.
pub fn bit_diff(a: &[f32], b: &[f32]) -> (usize, Option<usize>) {
    assert_eq!(a.len(), b.len(), "two outputs of different sizes");
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
    (bad.len(), bad.first().copied())
}

/// Fail with the count of values that differ and the first one.
#[track_caller]
pub fn assert_same(a: &[f32], b: &[f32], what: &str) {
    let (count, first) = bit_diff(a, b);
    if let Some(i) = first {
        panic!(
            "{what}: {count} of {} values differ, first at {i}: {} vs {}",
            a.len(),
            a[i],
            b[i]
        );
    }
}

/// The relative L2 error of `a` against `b`, in f64.
pub fn relerr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x) - f64::from(*y);
        num += d * d;
        den += f64::from(*y) * f64::from(*y);
    }
    (num / den).sqrt()
}

pub fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// The quantization type as the number the op libraries take.
pub fn tcode(qtype: Qtype) -> u32 {
    qtype as u32
}

/// The dequantized values of `rows` rows of `k` values: `qwen38_dequant`.
pub fn dequant(qtype: Qtype, raw: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    let f = sym!(
        lib("qmm"),
        "qwen38_dequant",
        fn(u32, *const c_void, u64, u32, u32, *mut c_void, u64) -> i32
    );
    let rc = unsafe {
        f(
            tcode(qtype),
            ptr(raw),
            nbytes(raw),
            rows as u32,
            k as u32,
            ptr_mut(&mut out),
            nbytes(&out),
        )
    };
    assert_eq!(rc, 0, "qwen38_dequant");
    out
}

/// Which kernel `qmm` calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// `qwen38_qmm`: the exact f32 kernel of the engine.
    Exact,
    /// `qwen38_qmm_oracle`: the kernel of run 2, which is the oracle of the exact kernel.
    Oracle,
    /// `qwen38_qmmh`: the f16 WMMA kernel of the engine.
    Fast,
}

/// The options of one matmul call. `y_rows` and `ldy` default to the token count and `n`.
#[derive(Clone, Default)]
pub struct QmmOptions<'a> {
    pub grid: u32,
    pub x_row: Option<&'a [u32]>,
    pub y_row: Option<&'a [u32]>,
    pub y_rows: Option<usize>,
    pub ldy: Option<usize>,
}

/// `y[y_row[t], :n] = W x[x_row[t]]`, where a missing map is the identity. The output starts as NaN, so a value that
/// the kernel does not write stays NaN. It returns the whole `[y_rows, ldy]` buffer. It ports `mkops._qmm`.
pub fn qmm_with(
    kernel: Kernel,
    qtype: Qtype,
    raw: &[u8],
    n: usize,
    k: usize,
    x: &[f32],
    options: &QmmOptions,
) -> Vec<f32> {
    assert_eq!(x.len() % k, 0);
    let x_rows = x.len() / k;
    let t = options.x_row.map_or(x_rows, <[u32]>::len);
    if let Some(map) = options.x_row {
        assert!(
            map.iter().all(|&r| (r as usize) < x_rows),
            "a row map names a row that x does not have"
        );
    }
    let y_rows = options.y_rows.unwrap_or(t);
    let ldy = options.ldy.unwrap_or(n);
    let mut y = vec![f32::NAN; y_rows * ldy];
    let (library, name) = match kernel {
        Kernel::Exact => ("qmm", "qwen38_qmm"),
        Kernel::Oracle => ("qmm_oracle", "qwen38_qmm_oracle"),
        Kernel::Fast => ("qmmh", "qwen38_qmmh"),
    };
    let lib = lib(library);
    let f = match kernel {
        Kernel::Exact => sym!(
            lib,
            "qwen38_qmm",
            fn(
                u32,
                *const c_void,
                u64,
                u32,
                u32,
                *const c_void,
                u64,
                u32,
                *const c_void,
                u64,
                u32,
                *mut c_void,
                u64,
                u32,
                *const c_void,
                u64,
                u32,
                u32,
            ) -> i32
        ),
        Kernel::Oracle => sym!(
            lib,
            "qwen38_qmm_oracle",
            fn(
                u32,
                *const c_void,
                u64,
                u32,
                u32,
                *const c_void,
                u64,
                u32,
                *const c_void,
                u64,
                u32,
                *mut c_void,
                u64,
                u32,
                *const c_void,
                u64,
                u32,
                u32,
            ) -> i32
        ),
        Kernel::Fast => sym!(
            lib,
            "qwen38_qmmh",
            fn(
                u32,
                *const c_void,
                u64,
                u32,
                u32,
                *const c_void,
                u64,
                u32,
                *const c_void,
                u64,
                u32,
                *mut c_void,
                u64,
                u32,
                *const c_void,
                u64,
                u32,
                u32,
            ) -> i32
        ),
    };
    let rc = unsafe {
        f(
            tcode(qtype),
            ptr(raw),
            nbytes(raw),
            n as u32,
            k as u32,
            ptr(x),
            nbytes(x),
            x_rows as u32,
            opt_ptr(options.x_row),
            opt_nbytes(options.x_row),
            t as u32,
            ptr_mut(&mut y),
            nbytes(&y),
            y_rows as u32,
            opt_ptr(options.y_row),
            opt_nbytes(options.y_row),
            ldy as u32,
            options.grid,
        )
    };
    assert_eq!(rc, 0, "{name}");
    y
}

/// The exact kernel with a grid size and no maps.
pub fn qmm(qtype: Qtype, raw: &[u8], n: usize, k: usize, x: &[f32], grid: u32) -> Vec<f32> {
    qmm_with(
        Kernel::Exact,
        qtype,
        raw,
        n,
        k,
        x,
        &QmmOptions {
            grid,
            ..Default::default()
        },
    )
}

pub fn qmm_oracle(qtype: Qtype, raw: &[u8], n: usize, k: usize, x: &[f32], grid: u32) -> Vec<f32> {
    qmm_with(
        Kernel::Oracle,
        qtype,
        raw,
        n,
        k,
        x,
        &QmmOptions {
            grid,
            ..Default::default()
        },
    )
}

pub fn qmm_fast(qtype: Qtype, raw: &[u8], n: usize, k: usize, x: &[f32], grid: u32) -> Vec<f32> {
    qmm_with(
        Kernel::Fast,
        qtype,
        raw,
        n,
        k,
        x,
        &QmmOptions {
            grid,
            ..Default::default()
        },
    )
}

/// `x W^T` in f64, with `w` as `[n, k]` and `x` as `[t, k]`, and the sum of the absolute products, as `[t, n]`.
pub fn matmul_f64(x: &[f32], w: &[f32], t: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0f64; t * n];
    let mut mag = vec![0f64; t * n];
    for i in 0..t {
        for j in 0..n {
            let (mut sum, mut abs) = (0f64, 0f64);
            for l in 0..k {
                let p = f64::from(x[i * k + l]) * f64::from(w[j * k + l]);
                sum += p;
                abs += p.abs();
            }
            y[i * n + j] = sum;
            mag[i * n + j] = abs;
        }
    }
    (y, mag)
}
