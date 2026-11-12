//! The byte-size checks and the device traps of the op ABIs. It ports `test_abi_sizes.py`, `abi_cases.py` and
//! `test_traps.py` of mk.
//!
//! An op ABI takes the byte size of each layout-sized buffer beside the buffer, and it refuses a call whose size is not
//! what the layout needs. It refuses on the host, before the first HIP call. The refusal aborts the process, so each
//! mismatch runs in a child process that sees no GPU. A refusal that came before the first HIP call names the buffer with
//! the two sizes. A check that is missing goes as far as `hipMalloc` and stops with another message, so the test fails.
//!
//! The expected sizes below come from the formats (bytes in a block), not from the helpers of the kernels.

use crate::child;
use crate::inputs::Qtype;
use crate::ops_lib::{lib, lib_raw, lib_with, ptr, ptr_mut, tcode};
use crate::sym;
use libloading::Library;
use std::collections::BTreeMap;
use std::ffi::c_void;

/// The byte sizes of one call. `size` adds the delta of the buffer that the test names, and records the names in order.
struct Sizes {
    delta: Option<(String, i64)>,
    names: Vec<&'static str>,
}

impl Sizes {
    fn new(delta: Option<(&str, i64)>) -> Sizes {
        Sizes {
            delta: delta.map(|(name, d)| (name.to_owned(), d)),
            names: Vec::new(),
        }
    }

    fn size(&mut self, name: &'static str, bytes: usize) -> u64 {
        self.names.push(name);
        let delta = match &self.delta {
            Some((wanted, d)) if wanted == name => *d,
            _ => 0,
        };
        (bytes as u64).wrapping_add(delta as u64)
    }
}

/// A case: the symbol and a builder. The builder makes the buffers and the sizes. It calls the symbol only
/// when `call` is true, and it gives the result of the call as an f64.
type Builder = Box<dyn Fn(&mut Sizes, bool) -> Option<f64> + Send + Sync>;

struct Case {
    symbol: &'static str,
    build: Builder,
}

/// The bytes of the tiled layout of `n` rows of `k` values.
type TiledBytes = fn(usize, usize) -> usize;

/// The library with no trap sink: installing the sink touches the GPU, and the child of a refusal sees no GPU.
fn nosink(op: &str) -> &'static Library {
    lib_with(op, false)
}

fn row_bytes(qtype: Qtype, k: usize) -> usize {
    qtype.row_bytes(k)
}

/// The bytes of IQ3_S tiles: 16 rows by one superblock take 1792 bytes, which is 112 bytes for each row and superblock.
fn iq3s_tiled(n: usize, k: usize) -> usize {
    112 * (k / 256) * n
}

type Relayout = unsafe extern "C" fn(*const c_void, u64, u32, u32, *mut c_void, u64) -> i32;
type TileDequant = unsafe extern "C" fn(*const c_void, u64, u32, u32, *mut c_void, u64) -> i32;

fn relayout_case(symbol: &'static str, qtype: Qtype, tiled: TiledBytes) -> Case {
    Case {
        symbol,
        build: Box::new(move |s, call| {
            let (n, k) = (16usize, 256usize);
            let raw = vec![0u8; row_bytes(qtype, k) * n];
            let mut out = vec![0u8; tiled(n, k)];
            let (raw_bytes, out_bytes) = (s.size("raw", raw.len()), s.size("tiled_out", out.len()));
            call.then(|| {
                let f: libloading::Symbol<Relayout> =
                    unsafe { nosink("tiles").get(format!("{symbol}\0").as_bytes()) }.unwrap();
                f_call_relayout(&f, &raw, raw_bytes, n, k, &mut out, out_bytes)
            })
        }),
    }
}

fn f_call_relayout(
    f: &Relayout,
    raw: &[u8],
    raw_bytes: u64,
    n: usize,
    k: usize,
    out: &mut [u8],
    out_bytes: u64,
) -> f64 {
    f64::from(unsafe { f(ptr(raw), raw_bytes, n as u32, k as u32, ptr_mut(out), out_bytes) })
}

fn dequant_case(symbol: &'static str, tiled: TiledBytes) -> Case {
    Case {
        symbol,
        build: Box::new(move |s, call| {
            let (n, k) = (16usize, 256usize);
            let tiles = vec![0u8; tiled(n, k)];
            let mut out = vec![0f32; n * k];
            let (tiles_bytes, out_bytes) = (s.size("tiled", tiles.len()), s.size("out", 4 * out.len()));
            call.then(|| {
                let f: libloading::Symbol<TileDequant> =
                    unsafe { nosink("tiles").get(format!("{symbol}\0").as_bytes()) }.unwrap();
                f64::from(unsafe {
                    f(
                        ptr(&tiles),
                        tiles_bytes,
                        n as u32,
                        k as u32,
                        ptr_mut(&mut out),
                        out_bytes,
                    )
                })
            })
        }),
    }
}

type TileQmm =
    unsafe extern "C" fn(*const c_void, u64, u32, u32, *const c_void, u64, u32, *mut c_void, u64, u32, u32) -> i32;
type TileQmmSplit =
    unsafe extern "C" fn(*const c_void, u64, u32, u32, *const c_void, u64, u32, *mut c_void, u64, u32, u32, u32) -> i32;

fn tile_qmm_case(symbol: &'static str, qtype: Qtype, ksplit: bool) -> Case {
    Case {
        symbol,
        build: Box::new(move |s, call| {
            let (n, k, t) = (16usize, 256usize, 3usize);
            let raw = vec![0u8; row_bytes(qtype, k) * n];
            let x = vec![0f32; t * k];
            let mut y = vec![0f32; t * n];
            let (raw_bytes, x_bytes, y_bytes) = (
                s.size("raw", raw.len()),
                s.size("x", 4 * x.len()),
                s.size("y", 4 * y.len()),
            );
            call.then(|| {
                let library = nosink("tiles");
                // The mode is 1: tiles.
                if ksplit {
                    let f: libloading::Symbol<TileQmmSplit> =
                        unsafe { library.get(format!("{symbol}\0").as_bytes()) }.unwrap();
                    f64::from(unsafe {
                        f(
                            ptr(&raw),
                            raw_bytes,
                            n as u32,
                            k as u32,
                            ptr(&x),
                            x_bytes,
                            t as u32,
                            ptr_mut(&mut y),
                            y_bytes,
                            1,
                            1,
                            0,
                        )
                    })
                } else {
                    let f: libloading::Symbol<TileQmm> =
                        unsafe { library.get(format!("{symbol}\0").as_bytes()) }.unwrap();
                    f64::from(unsafe {
                        f(
                            ptr(&raw),
                            raw_bytes,
                            n as u32,
                            k as u32,
                            ptr(&x),
                            x_bytes,
                            t as u32,
                            ptr_mut(&mut y),
                            y_bytes,
                            1,
                            0,
                        )
                    })
                }
            })
        }),
    }
}

type QmmFn = unsafe extern "C" fn(
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
) -> i32;

/// The call of `qwen38_qmm`, `qwen38_qmmh` and `qwen38_qmm_oracle` on Q8_0, with row maps or without.
fn qmm_case(library: &'static str, symbol: &'static str, maps: bool) -> Case {
    Case {
        symbol,
        build: Box::new(move |s, call| {
            let (n, k) = (16usize, 64usize);
            let (x_rows, t, y_rows, ldy) = if maps {
                (3usize, 2usize, 4usize, 20usize)
            } else {
                (2, 2, 2, n)
            };
            let w = vec![0u8; row_bytes(Qtype::Q8_0, k) * n];
            let x = vec![0f32; x_rows * k];
            let mut y = vec![0f32; y_rows * ldy];
            let x_row: Vec<u32> = (0..t as u32).collect();
            let y_row: Vec<u32> = (0..t as u32).collect();
            let w_bytes = s.size("w", w.len());
            let x_bytes = s.size("x", 4 * x.len());
            let x_row_bytes = s.size("x_row", if maps { 4 * x_row.len() } else { 0 });
            let y_bytes = s.size("y", 4 * y.len());
            let y_row_bytes = s.size("y_row", if maps { 4 * y_row.len() } else { 0 });
            call.then(|| {
                let f: libloading::Symbol<QmmFn> =
                    unsafe { nosink(library).get(format!("{symbol}\0").as_bytes()) }.unwrap();
                let (xr, yr) = if maps {
                    (ptr(&x_row), ptr(&y_row))
                } else {
                    (std::ptr::null(), std::ptr::null())
                };
                f64::from(unsafe {
                    f(
                        tcode(Qtype::Q8_0),
                        ptr(&w),
                        w_bytes,
                        n as u32,
                        k as u32,
                        ptr(&x),
                        x_bytes,
                        x_rows as u32,
                        xr,
                        x_row_bytes,
                        t as u32,
                        ptr_mut(&mut y),
                        y_bytes,
                        y_rows as u32,
                        yr,
                        y_row_bytes,
                        ldy as u32,
                        0,
                    )
                })
            })
        }),
    }
}

type BenchFn = unsafe extern "C" fn(u32, *const c_void, u64, u32, u32, *const c_void, u64, u32, u32, u32) -> f64;

fn bench_case(library: &'static str, symbol: &'static str) -> Case {
    Case {
        symbol,
        build: Box::new(move |s, call| {
            let (n, k, t) = (16usize, 64usize, 2usize);
            let w = vec![0u8; row_bytes(Qtype::Q8_0, k) * n];
            let x = vec![0f32; t * k];
            let (w_bytes, x_bytes) = (s.size("w", w.len()), s.size("x", 4 * x.len()));
            call.then(|| {
                let f: libloading::Symbol<BenchFn> =
                    unsafe { nosink(library).get(format!("{symbol}\0").as_bytes()) }.unwrap();
                unsafe {
                    f(
                        tcode(Qtype::Q8_0),
                        ptr(&w),
                        w_bytes,
                        n as u32,
                        k as u32,
                        ptr(&x),
                        x_bytes,
                        t as u32,
                        1,
                        1,
                    )
                }
            })
        }),
    }
}

fn split_case() -> Case {
    Case {
        symbol: "qwen38_qmmh_split",
        build: Box::new(|s, call| {
            let (n, k, t) = (16usize, 256usize, 2usize);
            let w = vec![0u8; row_bytes(Qtype::Q8_0, k) * n];
            let x = vec![0f32; t * k];
            let mut y = vec![0f32; t * n];
            let (w_bytes, x_bytes, y_bytes) =
                (s.size("w", w.len()), s.size("x", 4 * x.len()), s.size("y", 4 * y.len()));
            call.then(|| {
                let f = sym!(
                    nosink("qmmh"),
                    "qwen38_qmmh_split",
                    fn(u32, *const c_void, u64, u32, u32, *const c_void, u64, u32, *mut c_void, u64, u32, u32) -> i32
                );
                f64::from(unsafe {
                    f(
                        tcode(Qtype::Q8_0),
                        ptr(&w),
                        w_bytes,
                        n as u32,
                        k as u32,
                        ptr(&x),
                        x_bytes,
                        t as u32,
                        ptr_mut(&mut y),
                        y_bytes,
                        2,
                        0,
                    )
                })
            })
        }),
    }
}

type XinFn = unsafe extern "C" fn(
    u32,
    *const c_void,
    u64,
    u32,
    u32,
    u32,
    *const c_void,
    u64,
    *const c_void,
    u64,
    u32,
    u32,
    f32,
    *mut c_void,
    u64,
    *mut c_void,
    u64,
    *mut c_void,
    u64,
    *mut c_void,
    u64,
    u32,
) -> i32;

fn xin_case(kind: u32) -> Case {
    Case {
        symbol: "qwen38_qmmh_xin",
        build: Box::new(move |s, call| {
            let (n, k, t, xks) = (16usize, 64usize, 2usize, 2usize);
            let w = vec![0u8; row_bytes(Qtype::Q8_0, k) * n];
            let parts = vec![0f32; xks * t * k];
            let parts2 = vec![0f32; if kind == 2 { xks * t * k } else { 0 }];
            let (mut y_fused, mut y_ref) = (vec![0f32; t * n], vec![0f32; t * n]);
            let (mut x_fused, mut x_ref) = (vec![0f32; t * k], vec![0f32; t * k]);
            let w_bytes = s.size("w", w.len());
            let parts_bytes = s.size("parts", 4 * parts.len());
            let parts2_bytes = s.size("parts2", 4 * parts2.len());
            let y_fused_bytes = s.size("y_fused", 4 * y_fused.len());
            let y_ref_bytes = s.size("y_ref", 4 * y_ref.len());
            let x_fused_bytes = s.size("x_fused", 4 * x_fused.len());
            let x_ref_bytes = s.size("x_ref", 4 * x_ref.len());
            call.then(|| {
                let f: libloading::Symbol<XinFn> = unsafe { nosink("qmmh").get(b"qwen38_qmmh_xin\0") }.unwrap();
                let parts2_ptr = if kind == 2 { ptr(&parts2) } else { std::ptr::null() };
                f64::from(unsafe {
                    f(
                        tcode(Qtype::Q8_0),
                        ptr(&w),
                        w_bytes,
                        n as u32,
                        k as u32,
                        t as u32,
                        ptr(&parts),
                        parts_bytes,
                        parts2_ptr,
                        parts2_bytes,
                        xks as u32,
                        kind,
                        0.25,
                        ptr_mut(&mut y_fused),
                        y_fused_bytes,
                        ptr_mut(&mut y_ref),
                        y_ref_bytes,
                        ptr_mut(&mut x_fused),
                        x_fused_bytes,
                        ptr_mut(&mut x_ref),
                        x_ref_bytes,
                        0,
                    )
                })
            })
        }),
    }
}

const NE: usize = 512;
const KFF: usize = 640;
const EMB: usize = 2560;

type MoeDownFn = unsafe extern "C" fn(
    u32,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *mut c_void,
    u64,
    u32,
    u32,
) -> i32;
type MoeDownBenchFn = unsafe extern "C" fn(
    u32,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *mut c_void,
    u64,
    u32,
    u32,
    u32,
) -> f64;

fn moe_counts() -> Vec<u32> {
    let mut counts = vec![0u32; NE];
    counts[3] = 2;
    counts[5] = 1;
    counts
}

fn moe_down_case(reps: bool) -> Case {
    Case {
        symbol: if reps {
            "qwen38_moe_down_bench"
        } else {
            "qwen38_moe_down"
        },
        build: Box::new(move |s, call| {
            let qtype = Qtype::Iq4Nl;
            let counts = moe_counts();
            let a: usize = counts.iter().map(|&c| c as usize).sum();
            // A zero buffer of this size takes no memory until it is written: the refusal comes before any read.
            let raw = vec![0u8; row_bytes(qtype, KFF) * EMB * NE];
            let assign: Vec<u32> = (0..a as u32).collect();
            let h = vec![0f32; a * KFF];
            let mut y = vec![0f32; a * EMB];
            let raw_bytes = s.size("raw", raw.len());
            let counts_bytes = s.size("counts", 4 * counts.len());
            let assign_bytes = s.size("assign", 4 * assign.len());
            let h_bytes = s.size("h", 4 * h.len());
            let y_bytes = s.size("y", 4 * y.len());
            call.then(|| {
                let library = nosink("moe");
                if reps {
                    let f: libloading::Symbol<MoeDownBenchFn> =
                        unsafe { library.get(b"qwen38_moe_down_bench\0") }.unwrap();
                    unsafe {
                        f(
                            tcode(qtype),
                            ptr(&raw),
                            raw_bytes,
                            ptr(&counts),
                            counts_bytes,
                            ptr(&assign),
                            assign_bytes,
                            ptr(&h),
                            h_bytes,
                            ptr_mut(&mut y),
                            y_bytes,
                            1,
                            0,
                            1,
                        )
                    }
                } else {
                    let f: libloading::Symbol<MoeDownFn> = unsafe { library.get(b"qwen38_moe_down\0") }.unwrap();
                    f64::from(unsafe {
                        f(
                            tcode(qtype),
                            ptr(&raw),
                            raw_bytes,
                            ptr(&counts),
                            counts_bytes,
                            ptr(&assign),
                            assign_bytes,
                            ptr(&h),
                            h_bytes,
                            ptr_mut(&mut y),
                            y_bytes,
                            1,
                            0,
                        )
                    })
                }
            })
        }),
    }
}

type MoeGuFn = unsafe extern "C" fn(
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    u32,
    *mut c_void,
    u64,
    u32,
    u32,
) -> i32;
type MoeGuBenchFn = unsafe extern "C" fn(
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    u32,
    *mut c_void,
    u64,
    u32,
    u32,
    u32,
) -> f64;

fn moe_gu_case(reps: bool) -> Case {
    Case {
        symbol: if reps { "qwen38_moe_gu_bench" } else { "qwen38_moe_gu" },
        build: Box::new(move |s, call| {
            let counts = moe_counts();
            let a: usize = counts.iter().map(|&c| c as usize).sum();
            let raw = vec![0u8; row_bytes(Qtype::Iq3S, EMB) * KFF * NE];
            let x = vec![0f32; a * EMB];
            let mut gu = vec![0f32; a * 2 * KFF];
            let raw_bytes = s.size("raw", raw.len());
            let counts_bytes = s.size("counts", 4 * counts.len());
            let x_bytes = s.size("x", 4 * x.len());
            let gu_bytes = s.size("gu", 4 * gu.len());
            call.then(|| {
                let library = nosink("moe");
                if reps {
                    let f: libloading::Symbol<MoeGuBenchFn> = unsafe { library.get(b"qwen38_moe_gu_bench\0") }.unwrap();
                    unsafe {
                        f(
                            ptr(&raw),
                            raw_bytes,
                            ptr(&counts),
                            counts_bytes,
                            ptr(&x),
                            x_bytes,
                            a as u32,
                            ptr_mut(&mut gu),
                            gu_bytes,
                            1,
                            0,
                            1,
                        )
                    }
                } else {
                    let f: libloading::Symbol<MoeGuFn> = unsafe { library.get(b"qwen38_moe_gu\0") }.unwrap();
                    f64::from(unsafe {
                        f(
                            ptr(&raw),
                            raw_bytes,
                            ptr(&counts),
                            counts_bytes,
                            ptr(&x),
                            x_bytes,
                            a as u32,
                            ptr_mut(&mut gu),
                            gu_bytes,
                            1,
                            0,
                        )
                    })
                }
            })
        }),
    }
}

type MoeFn = unsafe extern "C" fn(
    *const c_void,
    u64,
    u32,
    *const c_void,
    u64,
    *const c_void,
    u64,
    u32,
    *const c_void,
    u64,
    u32,
    *const c_void,
    u64,
    u32,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *const c_void,
    u64,
    *mut c_void,
    u64,
    *mut c_void,
    u64,
    *mut c_void,
    u64,
    u32,
    u32,
) -> i32;

fn moe_case() -> Case {
    Case {
        symbol: "qwen38_moe",
        build: Box::new(|s, call| {
            let (t, top) = (2usize, 10usize);
            let x = vec![0f32; t * EMB];
            let router = vec![0f32; NE * EMB];
            let shexp_gate = vec![0f32; EMB];
            let gate = vec![0u8; row_bytes(Qtype::Iq3S, EMB) * KFF * NE];
            let up = vec![0u8; row_bytes(Qtype::Iq3S, EMB) * KFF * NE];
            let down = vec![0u8; row_bytes(Qtype::Iq4Nl, KFF) * EMB * NE];
            let sh_gate = vec![0u8; row_bytes(Qtype::Q8_0, EMB) * KFF];
            let sh_up = vec![0u8; row_bytes(Qtype::Q8_0, EMB) * KFF];
            let sh_down = vec![0u8; row_bytes(Qtype::Q8_0, KFF) * EMB];
            let mut out = vec![0f32; t * EMB];
            let mut idx = vec![0i32; t * top];
            let mut w = vec![0f32; t * top];
            let x_bytes = s.size("x", 4 * x.len());
            let router_bytes = s.size("router", 4 * router.len());
            let shexp_gate_bytes = s.size("shexp_gate", 4 * shexp_gate.len());
            let gate_bytes = s.size("gate", gate.len());
            let up_bytes = s.size("up", up.len());
            let down_bytes = s.size("down", down.len());
            let sh_gate_bytes = s.size("sh_gate", sh_gate.len());
            let sh_up_bytes = s.size("sh_up", sh_up.len());
            let sh_down_bytes = s.size("sh_down", sh_down.len());
            let out_bytes = s.size("out", 4 * out.len());
            let idx_bytes = s.size("topk_idx", 4 * idx.len());
            let w_bytes = s.size("topk_w", 4 * w.len());
            call.then(|| {
                let f: libloading::Symbol<MoeFn> = unsafe { nosink("moe").get(b"qwen38_moe\0") }.unwrap();
                f64::from(unsafe {
                    f(
                        ptr(&x),
                        x_bytes,
                        t as u32,
                        ptr(&router),
                        router_bytes,
                        ptr(&shexp_gate),
                        shexp_gate_bytes,
                        tcode(Qtype::Iq3S),
                        ptr(&gate),
                        gate_bytes,
                        tcode(Qtype::Iq3S),
                        ptr(&up),
                        up_bytes,
                        tcode(Qtype::Iq4Nl),
                        ptr(&down),
                        down_bytes,
                        ptr(&sh_gate),
                        sh_gate_bytes,
                        ptr(&sh_up),
                        sh_up_bytes,
                        ptr(&sh_down),
                        sh_down_bytes,
                        ptr_mut(&mut out),
                        out_bytes,
                        ptr_mut(&mut idx),
                        idx_bytes,
                        ptr_mut(&mut w),
                        w_bytes,
                        0,
                        0,
                    )
                })
            })
        }),
    }
}

type DequantFn = unsafe extern "C" fn(u32, *const c_void, u64, u32, u32, *mut c_void, u64) -> i32;

fn plain_dequant_case() -> Case {
    Case {
        symbol: "qwen38_dequant",
        build: Box::new(|s, call| {
            let (n, k) = (4usize, 64usize);
            let w = vec![0u8; row_bytes(Qtype::Q8_0, k) * n];
            let mut out = vec![0f32; n * k];
            let (w_bytes, out_bytes) = (s.size("w", w.len()), s.size("out", 4 * out.len()));
            call.then(|| {
                let f: libloading::Symbol<DequantFn> = unsafe { nosink("qmm").get(b"qwen38_dequant\0") }.unwrap();
                f64::from(unsafe {
                    f(
                        tcode(Qtype::Q8_0),
                        ptr(&w),
                        w_bytes,
                        n as u32,
                        k as u32,
                        ptr_mut(&mut out),
                        out_bytes,
                    )
                })
            })
        }),
    }
}

/// Every case, by name, in the order of mk's `abi_cases.py`.
fn cases() -> BTreeMap<String, Case> {
    let mut map = BTreeMap::new();
    let same = |qtype: Qtype| -> TiledBytes {
        match qtype {
            Qtype::Q8_0 => |n, k| 34 * (k / 32) * n,
            Qtype::Q6K => |n, k| 210 * (k / 256) * n,
            Qtype::Iq4Nl => |n, k| 18 * (k / 32) * n,
            _ => unreachable!(),
        }
    };
    let tiled_types: [(Qtype, &'static str, TiledBytes); 4] = [
        (Qtype::Iq3S, "iq3s", iq3s_tiled),
        (Qtype::Q8_0, "q8", same(Qtype::Q8_0)),
        (Qtype::Q6K, "q6", same(Qtype::Q6K)),
        (Qtype::Iq4Nl, "iq4nl", same(Qtype::Iq4Nl)),
    ];
    let leak = |text: String| -> &'static str { Box::leak(text.into_boxed_str()) };
    for (qtype, short, tiled) in tiled_types {
        map.insert(
            format!("qwen38_relayout_{short}"),
            relayout_case(leak(format!("qwen38_relayout_{short}")), qtype, tiled),
        );
        map.insert(
            format!("qwen38_tile_dequant_{short}"),
            dequant_case(leak(format!("qwen38_tile_dequant_{short}")), tiled),
        );
    }
    for (qtype, short, ksplit) in [
        (Qtype::Iq3S, "iq3s", false),
        (Qtype::Q8_0, "q8", true),
        (Qtype::Q6K, "q6", true),
    ] {
        map.insert(
            format!("qwen38_tile_qmm_{short}"),
            tile_qmm_case(leak(format!("qwen38_tile_qmm_{short}")), qtype, ksplit),
        );
    }
    map.insert("qwen38_dequant".to_owned(), plain_dequant_case());
    for (library, symbol) in [
        ("qmm", "qwen38_qmm"),
        ("qmmh", "qwen38_qmmh"),
        ("qmm_oracle", "qwen38_qmm_oracle"),
    ] {
        map.insert(symbol.to_owned(), qmm_case(library, symbol, false));
        map.insert(format!("{symbol}_maps"), qmm_case(library, symbol, true));
    }
    for (library, symbol) in [
        ("qmm", "qwen38_qmm_bench"),
        ("qmmh", "qwen38_qmmh_bench"),
        ("qmm_oracle", "qwen38_qmm_oracle_bench"),
    ] {
        map.insert(symbol.to_owned(), bench_case(library, symbol));
    }
    map.insert("qwen38_qmmh_split".to_owned(), split_case());
    map.insert("mk_qmmh_xin_silu".to_owned(), xin_case(1));
    map.insert("mk_qmmh_xin_swiglu".to_owned(), xin_case(2));
    map.insert("qwen38_moe_down".to_owned(), moe_down_case(false));
    map.insert("qwen38_moe_down_bench".to_owned(), moe_down_case(true));
    map.insert("qwen38_moe_gu".to_owned(), moe_gu_case(false));
    map.insert("qwen38_moe_gu_bench".to_owned(), moe_gu_case(true));
    map.insert("qwen38_moe".to_owned(), moe_case());
    map
}

/// The byte-size arguments of a case, in order. Nothing runs.
fn buffers(case: &Case) -> Vec<&'static str> {
    let mut sizes = Sizes::new(None);
    (case.build)(&mut sizes, false);
    sizes.names
}

/// Run a case with the delta on one buffer. In a child process, the call aborts when the size is refused.
fn call(name: &str, delta: Option<(&str, i64)>) -> f64 {
    let cases = cases();
    let case = cases.get(name).unwrap_or_else(|| panic!("no case {name}"));
    let mut sizes = Sizes::new(delta);
    (case.build)(&mut sizes, true).expect("the builder calls the symbol")
}

fn run_child(case: &str, buffer: &str, delta: i64) -> std::process::Output {
    let mut env: Vec<(&str, String)> = child::NO_GPU.iter().map(|(k, v)| (*k, (*v).to_owned())).collect();
    env.push(("NAGAYA_ABI_CASE", case.to_owned()));
    env.push(("NAGAYA_ABI_BUFFER", buffer.to_owned()));
    env.push(("NAGAYA_ABI_DELTA", delta.to_string()));
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    child::run("abi::child_case", &env)
}

/// The child: it makes one call with one wrong size. The parent reads its stderr. Without the variables it does nothing.
#[test]
#[ignore = "child of every_wrong_buffer_size_is_refused_before_any_launch"]
fn child_case() {
    let (Ok(case), Ok(buffer), Ok(delta)) = (
        std::env::var("NAGAYA_ABI_CASE"),
        std::env::var("NAGAYA_ABI_BUFFER"),
        std::env::var("NAGAYA_ABI_DELTA"),
    ) else {
        panic!("the variables NAGAYA_ABI_CASE, NAGAYA_ABI_BUFFER and NAGAYA_ABI_DELTA are not set");
    };
    let rc = call(&case, Some((&buffer, delta.parse().expect("a delta"))));
    println!("returned {rc}");
}

/// Take the first `device trap` or refusal line out of a stderr text.
fn refusal(stderr: &str, symbol: &str, buffer: &str) -> Option<(u64, u64)> {
    let prefix = format!("{symbol}: buffer '{buffer}' is ");
    let line = stderr.lines().find(|line| line.contains(&prefix))?;
    let rest = &line[line.find(&prefix)? + prefix.len()..];
    let (given, rest) = rest.split_once(" B, the layout needs ")?;
    let need = rest.strip_suffix(" B")?;
    Some((given.parse().ok()?, need.trim().parse().ok()?))
}

#[test]
fn every_wrong_buffer_size_is_refused_before_any_launch() {
    let cases = cases();
    let mut jobs = Vec::new();
    for (name, case) in &cases {
        for buffer in buffers(case) {
            for delta in [-1i64, 1] {
                jobs.push((name.clone(), case.symbol, buffer, delta));
            }
        }
    }
    assert!(jobs.len() > 100, "{} mismatches only", jobs.len());
    // Each child spends about 0.7 s loading the HIP code objects: run eight at a time.
    let queue = std::sync::Mutex::new(jobs.iter().collect::<Vec<_>>());
    let failures = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                loop {
                    let Some((name, symbol, buffer, delta)) = queue.lock().unwrap().pop() else {
                        break;
                    };
                    let output = run_child(name, buffer, *delta);
                    let stderr = child::stderr(&output);
                    let label = format!("{name}/{buffer}{delta:+}");
                    if child::signal(&output) != Some(6) {
                        failures.lock().unwrap().push(format!(
                            "{label}: not aborted ({:?}): {}",
                            output.status,
                            tail(&stderr)
                        ));
                    } else if let Some((given, need)) = refusal(&stderr, symbol, buffer) {
                        if given != need.wrapping_add(*delta as u64) {
                            failures
                                .lock()
                                .unwrap()
                                .push(format!("{label}: the message says given {given}, needs {need}"));
                        }
                    } else {
                        failures
                            .lock()
                            .unwrap()
                            .push(format!("{label}: no size refusal naming the buffer: {}", tail(&stderr)));
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(
        failures.is_empty(),
        "{} of {} mismatches:\n{}",
        failures.len(),
        jobs.len(),
        failures.join("\n")
    );
}

fn tail(text: &str) -> String {
    let start = text.len().saturating_sub(300);
    let start = (start..text.len())
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(text.len());
    text[start..].to_owned()
}

/// The incident: a bench sized an IQ3_S tile pool at the 110 bytes of the GGUF for each row and superblock, and not the 112
/// of the tiles, and wrote past it. The pool of 16 rows and one superblock is 1792 bytes, and 110 bytes make 1760.
#[test]
fn iq3s_tile_pool_at_110_bytes_is_refused() {
    let output = run_child("qwen38_relayout_iq3s", "tiled_out", 110 * 16 - 112 * 16);
    assert_eq!(child::signal(&output), Some(6), "{}", tail(&child::stderr(&output)));
    let stderr = child::stderr(&output);
    assert!(
        stderr.contains("buffer 'tiled_out' is 1760 B, the layout needs 1792 B"),
        "{}",
        tail(&stderr)
    );
}

/// A case that declares no byte size would pass the test above with no check.
#[test]
fn every_case_names_its_buffers() {
    let mut total = 0;
    let all = cases();
    for (name, case) in &all {
        let names = buffers(case);
        assert!(!names.is_empty(), "{name} names no buffer");
        total += names.len();
    }
    // The cases and the buffers of mk's `abi_cases.py` at mk commit 555a2a3. A case that goes missing is a check that goes.
    assert_eq!((all.len(), total), (29, 110));
}

// ---- test_traps.py ----

/// The traps of `rt.h`, with the names of `status_name`, in the order of the enum.
const TRAPS: [(u32, &str); 8] = [
    (16, "trap-activation-range"),
    (17, "trap-bad-args"),
    (18, "trap-moe-route"),
    (19, "trap-moe-group"),
    (20, "trap-moe-total"),
    (21, "trap-attn-select"),
    (22, "trap-attn-no-selection"),
    (23, "trap-bad-op"),
];
const FILES: [(u32, &str); 8] = [
    (1, "qmmh.h"),
    (2, "qmm.h"),
    (3, "qmm_oracle.h"),
    (4, "small.h"),
    (5, "moe.h"),
    (6, "moe.hip"),
    (7, "attn.h"),
    (8, "stage.h"),
];

/// The report line of `hip_trap_format` for a status, a site and a workgroup.
fn format_trap(status: u32, site: u32, wg: u32) -> String {
    let f = sym!(
        lib_with("qmm", false),
        "hip_trap_format",
        fn(u32, u32, u32, *mut u8, usize) -> usize
    );
    let mut buffer = [0u8; 256];
    let n = unsafe { f(status, site, wg, buffer.as_mut_ptr(), buffer.len()) };
    String::from_utf8(buffer[..n].to_vec()).expect("ASCII")
}

#[test]
fn every_trap_status_has_its_own_name_and_report() {
    let mut names = std::collections::BTreeSet::new();
    for (status, name) in TRAPS {
        let line = format_trap(status, (1 << 16) | 408, 7);
        assert_eq!(line, format!("device trap: {name} at site qmmh.h:408 (wg 7)\n"));
        names.insert(name);
    }
    assert_eq!(names.len(), TRAPS.len(), "two statuses share a name");
}

#[test]
fn every_trap_file_has_a_name_and_the_site_splits_into_file_and_line() {
    for (file, name) in FILES {
        assert_eq!(
            format_trap(17, (file << 16) | 65535, 0),
            format!("device trap: trap-bad-args at site {name}:65535 (wg 0)\n")
        );
    }
    assert_eq!(
        format_trap(17, (99 << 16) | 3, 0),
        "device trap: trap-bad-args at site unknown-file:3 (wg 0)\n"
    );
}

/// ok, aborted and spin-timeout are the own returns of the engine, not traps: the handler says nothing about them.
#[test]
fn only_traps_are_reported() {
    for status in [0u32, 1, 2, 15, 24, 1000] {
        assert_eq!(format_trap(status, (1 << 16) | 1, 0), "", "status {status}");
    }
}

/// The fields of a line `device trap: <status> at site <file>:<line> (wg <n>)`.
fn trap_report(stderr: &str) -> Option<(String, String, u32, u32)> {
    let line = stderr.lines().find(|line| line.starts_with("device trap: "))?;
    let rest = line.strip_prefix("device trap: ")?;
    let (status, rest) = rest.split_once(" at site ")?;
    let (site, rest) = rest.split_once(" (wg ")?;
    let (file, line_number) = site.rsplit_once(':')?;
    Some((
        status.to_owned(),
        file.to_owned(),
        line_number.parse().ok()?,
        rest.strip_suffix(')')?.parse().ok()?,
    ))
}

/// Run a child that must trap, and check what its abort names.
#[track_caller]
fn assert_trap(test: &str, status: &str, file: &str, what: &str) {
    let output = child::run(test, &[]);
    let (stdout, stderr) = (child::stdout(&output), child::stderr(&output));
    assert!(
        !output.status.success() && !stdout.contains("NO TRAP"),
        "{what}: no trap: {:?} {}",
        output.status,
        tail(&stdout)
    );
    let Some((got_status, got_file, line, _wg)) = trap_report(&stderr) else {
        panic!(
            "{what}: the abort said nothing about which trap fired:\n{}",
            tail(&stderr)
        );
    };
    assert_eq!(
        (got_status.as_str(), got_file.as_str()),
        (status, file),
        "{what}: expected {status} in {file}, the trap report says {got_status} at {got_file}:{line}"
    );
}

fn qmm_fast_with_activation(position: usize, value: f32, rows: usize) {
    let raw = crate::inputs::raw_rows(Qtype::Q8_0, 71, 16, 2560);
    let mut x = vec![0f32; rows * 2560];
    x[position] = value;
    let _ = crate::ops_lib::qmm_fast(Qtype::Q8_0, &raw, 16, 2560, &x, 0);
    println!("NO TRAP");
}

#[test]
fn qmmh_activation_beyond_f16_reports_its_status() {
    assert_trap(
        "abi::child_qmmh_activation",
        "trap-activation-range",
        "qmmh.h",
        "qmmh activation 70000",
    );
}

#[test]
#[ignore = "child of qmmh_activation_beyond_f16_reports_its_status"]
fn child_qmmh_activation() {
    qmm_fast_with_activation(2560 + 7, 70000.0, 2);
}

#[test]
fn qmmh_nan_activation_reports_the_same_status() {
    assert_trap(
        "abi::child_qmmh_nan",
        "trap-activation-range",
        "qmmh.h",
        "qmmh NaN activation",
    );
}

#[test]
#[ignore = "child of qmmh_nan_activation_reports_the_same_status"]
fn child_qmmh_nan() {
    qmm_fast_with_activation(100, f32::NAN, 1);
}

#[test]
fn moe_router_with_nan_logits_reports_its_status() {
    assert_trap(
        "abi::child_moe_route_nan",
        "trap-moe-route",
        "moe.h",
        "moe router on NaN logits",
    );
}

#[test]
#[ignore = "child of moe_router_with_nan_logits_reports_its_status"]
fn child_moe_route_nan() {
    let t = 2usize;
    let logits = vec![f32::NAN; t * 512];
    let mut idx = vec![0i32; t * 10];
    let mut w = vec![0f32; t * 10];
    let mut count = vec![0u32; 512];
    let (mut offset, mut item_gu, mut item_dn) = (vec![0u32; 513], vec![0u32; 513], vec![0u32; 513]);
    let (mut assign, mut token) = (vec![0u32; t * 10], vec![0u32; t * 10]);
    let f = sym!(
        lib("moe"),
        "qwen38_moe_route",
        fn(
            *const c_void,
            u32,
            u32,
            u32,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *mut c_void,
        ) -> i32
    );
    unsafe {
        f(
            ptr(&logits),
            t as u32,
            0,
            0,
            ptr_mut(&mut idx),
            ptr_mut(&mut w),
            ptr_mut(&mut count),
            ptr_mut(&mut offset),
            ptr_mut(&mut assign),
            ptr_mut(&mut token),
            ptr_mut(&mut item_gu),
            ptr_mut(&mut item_dn),
        );
    }
    println!("NO TRAP");
}

#[test]
fn attention_selection_with_a_nan_score_reports_its_status() {
    assert_trap(
        "abi::child_attn_select_nan",
        "trap-attn-select",
        "attn.h",
        "attention selection on a NaN block score",
    );
}

#[test]
#[ignore = "child of attention_selection_with_a_nan_score_reports_its_status"]
fn child_attn_select_nan() {
    let (t, p0, budget, ratio) = (1usize, 4095usize, 2048usize, 4usize);
    let nb = (p0 + t) / ratio;
    let mut scores: Vec<f32> = (0..nb).map(|i| i as f32 / (nb - 1) as f32).collect();
    scores[17] = f32::NAN;
    let mut sel = vec![0i32; t * (budget + ratio - 1)];
    let mut count = vec![0u32; t];
    let f = sym!(
        lib("attn"),
        "qwen38_attn_select_emit",
        fn(u32, u32, u32, u32, *const c_void, *mut c_void, *mut c_void, u32) -> i32
    );
    unsafe {
        f(
            t as u32,
            p0 as u32,
            budget as u32,
            ratio as u32,
            ptr(&scores),
            ptr_mut(&mut sel),
            ptr_mut(&mut count),
            0,
        )
    };
    println!("NO TRAP");
}

/// A library that loads with no sink must trap all the same: recording is never a condition. `lib_raw` leaves the state
/// that a raw load leaves, and the report must be absent.
#[test]
fn a_trap_without_a_sink_still_traps() {
    let output = child::run("abi::child_trap_without_sink", &[]);
    assert!(
        !output.status.success() && !child::stdout(&output).contains("NO TRAP"),
        "no trap: {:?}",
        output.status
    );
    assert!(
        !child::stderr(&output).contains("device trap:"),
        "a report without a sink"
    );
}

#[test]
#[ignore = "child of a_trap_without_a_sink_still_traps"]
fn child_trap_without_sink() {
    let _ = lib_raw("qmmh");
    qmm_fast_with_activation(2560 + 7, 70000.0, 2);
}

/// Each library links its own copy of the rails, because mkrt is static, so a library that first loads with no sink must
/// get the sink when the program asks for it later. This is the check of mk's kaibo job 3, goal 03: the sink was installed
/// only at the first load, and a library first loaded without it never reported its traps in that process.
#[test]
fn a_library_first_loaded_without_the_sink_gets_it_when_asked_later() {
    assert_trap(
        "abi::child_late_sink",
        "trap-activation-range",
        "qmmh.h",
        "a library first loaded with no sink",
    );
}

#[test]
#[ignore = "child of a_library_first_loaded_without_the_sink_gets_it_when_asked_later"]
fn child_late_sink() {
    let _ = crate::ops_lib::lib_with("qmmh", false);
    qmm_fast_with_activation(2560 + 7, 70000.0, 2);
}
