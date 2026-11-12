//! mk's contexts in Rust: the device buffers of one context, and its parts to and from host memory (milestone 15
//! step 6).
//!
//! mk's `engine.hip` held `new_context`, `free_context`, `state_read`, `state_write`, `state_write_cells` and
//! `restore_n` in C++ (mk 555a2a3). They are here now, over the thin runtime calls `hip_rt_*` of `rt.cpp`. The pass
//! still reads and advances a context in C++, so `Qwen38Context` is `struct Context` of `engine.h`, field for field. The
//! layout test (`layout.rs`) compares the two. The checks of each call are in the worker (`worker.rs`), as before.
//! A call here that gets input that the worker refuses is a defect, so it stops the program.
//!
//! The functions have the C types of the binding's calls (`binding.rs`), so the stub binding of the tests stays.

// A build without the feature hip makes no context, so only the tests read the layout and the part sizes.
#![cfg_attr(not(feature = "hip"), allow(dead_code))]

use std::ffi::c_void;

use crate::spec;

/// The attention layers, which hold K/V and index keys.
const ATTENTION_LAYERS: u64 = 12;
/// The index keys of 4 cells pool into one block (`model.h` `kIdxRatio`).
const IDX_RATIO: u32 = 4;

/// `struct Context` of `engine.h`: everything that a pass continues from.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct Qwen38Context {
    /// The tokens so far.
    pub n: u32,
    /// The K/V capacity in tokens.
    pub max: u32,
    /// f32 [36][48][128][128].
    pub gdn_state: *mut f32,
    /// f32 [36][3][10240].
    pub gdn_conv: *mut f32,
    /// f32 [9][10240].
    pub ple_conv: *mut f32,
    /// f32 [12][max][2][2][256].
    pub kv: *mut f32,
    /// f32 [12][max][128], the raw index keys.
    pub idx_keys: *mut f32,
    /// f32 [12][max / 4][128], the pooled block keys. Blocks [0, pooled_n) are valid.
    pub idx_pooled: *mut f32,
    pub pooled_n: u32,
    /// The last two token ids.
    pub ple_hist: [i32; 2],
    /// A pass that extended it stopped on a spin-timeout. Its state is half-written; free it.
    pub poisoned: bool,
}

/// The `hipMemcpyKind` values of the runtime calls.
#[derive(Clone, Copy)]
#[repr(u32)]
pub(crate) enum Copy {
    HostToDevice = 1,
    DeviceToHost = 2,
}

/// The bytes of one cell of one attention layer in a cells part: part 3 (`kv`) or part 4 (`idx_keys`).
fn cell_bytes(part: u32) -> u64 {
    match part {
        3 => spec::KV_ROW_BYTES / ATTENTION_LAYERS,
        4 => spec::IDX_ROW_BYTES / ATTENTION_LAYERS,
        other => panic!("context cell part {other}: want 3 (kv) or 4 (idx_keys)"),
    }
}

/// The bytes of a part for `n` tokens (`engine.hip` `state_bytes(part, n)`). An unknown part stops the program.
pub(crate) unsafe extern "C" fn part_bytes(part: u32, n: u32) -> u64 {
    bytes_of(part, n)
}

/// The bytes of a part that does not depend on the token count (`engine.hip` `state_bytes(part)`).
pub(crate) unsafe extern "C" fn state_bytes(part: u32) -> u64 {
    fixed_bytes_of(part)
}

fn bytes_of(part: u32, n: u32) -> u64 {
    match part {
        0 => spec::GDN_STATE_BYTES,
        1 => spec::GDN_CONV_BYTES,
        2 => spec::PLE_CONV_BYTES,
        3 | 4 => ATTENTION_LAYERS * u64::from(n) * cell_bytes(part),
        5 => spec::PLE_HIST_BYTES,
        other => panic!(
            "context state part {other}: want 0 (gdn_state), 1 (gdn_conv), 2 (ple_conv), 3 (kv), 4 (idx_keys) or 5 \
             (ple_hist)"
        ),
    }
}

fn fixed_bytes_of(part: u32) -> u64 {
    assert!(
        part != 3 && part != 4,
        "context state part {part}: its size depends on the token count"
    );
    bytes_of(part, 0)
}

/// The context at `c`, which `ctx_new` made.
unsafe fn context<'a>(c: *mut c_void) -> &'a mut Qwen38Context {
    assert!(!c.is_null(), "a context call got no context");
    unsafe { &mut *(c as *mut Qwen38Context) }
}

pub(crate) unsafe extern "C" fn ctx_n(c: *mut c_void) -> u32 {
    unsafe { context(c) }.n
}

pub(crate) unsafe extern "C" fn ctx_max(c: *mut c_void) -> u32 {
    unsafe { context(c) }.max
}

/// Every context owns its K/V: the engine has no fork.
pub(crate) unsafe extern "C" fn ctx_owns_kv(_c: *mut c_void) -> bool {
    true
}

pub(crate) unsafe extern "C" fn ctx_poisoned(c: *mut c_void) -> bool {
    unsafe { context(c) }.poisoned
}

#[cfg(feature = "hip")]
pub(crate) use linked::*;

#[cfg(feature = "hip")]
mod linked {
    use std::ffi::c_void;

    use super::{ATTENTION_LAYERS, Copy, IDX_RATIO, Qwen38Context, cell_bytes, context, part_bytes};
    use crate::binding::rt;
    use crate::ple::PLE_EOS;
    use crate::spec;

    /// A new device buffer of `bytes`.
    unsafe fn device(bytes: u64) -> *mut f32 {
        let p = unsafe { rt::hip_rt_malloc(bytes) };
        assert!(!p.is_null(), "hip_rt_malloc({bytes}) gave no buffer");
        p as *mut f32
    }

    /// A new empty context of `max_tokens` (`engine.hip` `new_context`). The fixed state is zero. The pooled keys are
    /// NaN (0xff bytes), so a block read before it was pooled scores as a poisoned one; nothing reads a block past
    /// `pooled_n`. A capacity that the worker refuses stops the program.
    pub(crate) unsafe extern "C" fn ctx_new(_engine: *mut c_void, max_tokens: u32) -> *mut c_void {
        assert!(
            max_tokens > 0 && max_tokens <= spec::MAX_CONTEXT && max_tokens.is_multiple_of(spec::MAX_PASS_TOKENS),
            "context capacity {max_tokens}: want a multiple of {} in 1..={}",
            spec::MAX_PASS_TOKENS,
            spec::MAX_CONTEXT
        );
        let max = u64::from(max_tokens);
        let pooled_bytes = u64::from(max_tokens / IDX_RATIO) * spec::IDX_ROW_BYTES;
        let c = unsafe {
            Qwen38Context {
                n: 0,
                max: max_tokens,
                gdn_state: device(spec::GDN_STATE_BYTES),
                gdn_conv: device(spec::GDN_CONV_BYTES),
                ple_conv: device(spec::PLE_CONV_BYTES),
                kv: device(max * spec::KV_ROW_BYTES),
                idx_keys: device(max * spec::IDX_ROW_BYTES),
                idx_pooled: device(pooled_bytes),
                pooled_n: 0,
                ple_hist: [PLE_EOS as i32; 2],
                poisoned: false,
            }
        };
        unsafe {
            rt::hip_rt_memset(c.idx_pooled.cast(), 0xff, pooled_bytes);
            rt::hip_rt_memset(c.gdn_state.cast(), 0, spec::GDN_STATE_BYTES);
            rt::hip_rt_memset(c.gdn_conv.cast(), 0, spec::GDN_CONV_BYTES);
            rt::hip_rt_memset(c.ple_conv.cast(), 0, spec::PLE_CONV_BYTES);
        }
        Box::into_raw(Box::new(c)) as *mut c_void
    }

    /// Free a context and its device buffers (`engine.hip` `free_context`).
    pub(crate) unsafe extern "C" fn ctx_free(_engine: *mut c_void, c: *mut c_void) {
        if c.is_null() {
            return;
        }
        let c = unsafe { Box::from_raw(c as *mut Qwen38Context) };
        for p in [c.gdn_state, c.gdn_conv, c.ple_conv, c.kv, c.idx_keys, c.idx_pooled] {
            unsafe { rt::hip_rt_free(p.cast()) };
        }
    }

    /// The device buffer of a fixed part: 0 gdn_state, 1 gdn_conv, 2 ple_conv.
    fn fixed(c: &Qwen38Context, part: u32) -> *mut f32 {
        match part {
            0 => c.gdn_state,
            1 => c.gdn_conv,
            2 => c.ple_conv,
            other => panic!("context state part {other}: want 0 (gdn_state), 1 (gdn_conv) or 2 (ple_conv)"),
        }
    }

    /// Read one part into `dst` (`engine.hip` `state_read`). A cells part gives cells 0..n-1 of each attention layer
    /// as one compact block: a layer's cells are the capacity apart on the device.
    pub(crate) unsafe extern "C" fn state_read(_engine: *mut c_void, c: *mut c_void, part: u32, dst: *mut c_void) {
        let c = unsafe { context(c) };
        assert!(!c.poisoned, "state read of a poisoned context ({} tokens)", c.n);
        match part {
            3 | 4 => {
                if c.n == 0 {
                    return;
                }
                let cell = cell_bytes(part);
                let src = if part == 3 { c.kv } else { c.idx_keys };
                let (n, max) = (u64::from(c.n), u64::from(c.max));
                unsafe {
                    rt::hip_rt_memcpy_2d(
                        dst,
                        cell * n,
                        src.cast(),
                        cell * max,
                        cell * n,
                        ATTENTION_LAYERS,
                        Copy::DeviceToHost as u32,
                    )
                };
            }
            5 => unsafe { std::ptr::copy_nonoverlapping(c.ple_hist.as_ptr(), dst as *mut i32, 2) },
            _ => unsafe {
                rt::hip_rt_memcpy(
                    dst,
                    fixed(c, part).cast(),
                    part_bytes(part, 0),
                    Copy::DeviceToHost as u32,
                )
            },
        }
    }

    /// Write a fixed part, or the PLE history (`engine.hip` `state_write`). The worker checks the rules first.
    pub(crate) unsafe extern "C" fn state_write(_engine: *mut c_void, c: *mut c_void, part: u32, src: *const c_void) {
        let c = unsafe { context(c) };
        assert!(!c.poisoned, "state write into a poisoned context ({} tokens)", c.n);
        match part {
            3 | 4 => panic!("state write of part {part} (cells): use the cells write with the cell count"),
            5 => {
                assert_eq!(c.n, 0, "a ple_hist write into a context of {} tokens", c.n);
                unsafe { std::ptr::copy_nonoverlapping(src as *const i32, c.ple_hist.as_mut_ptr(), 2) };
            }
            _ => unsafe {
                rt::hip_rt_memcpy(
                    fixed(c, part).cast(),
                    src,
                    part_bytes(part, 0),
                    Copy::HostToDevice as u32,
                )
            },
        }
    }

    /// Write `n` cells of a cells part into cells 0..n-1 of an empty context (`engine.hip` `state_write_cells`).
    pub(crate) unsafe extern "C" fn cells_write(
        _engine: *mut c_void,
        c: *mut c_void,
        part: u32,
        n: u32,
        src: *const c_void,
    ) {
        let c = unsafe { context(c) };
        assert!(!c.poisoned, "a cells write into a poisoned context ({} tokens)", c.n);
        assert_eq!(c.n, 0, "a cells write into a context of {} tokens", c.n);
        assert!(
            n > 0 && n <= c.max,
            "a cells write of {n} cells into a context of capacity {}",
            c.max
        );
        let cell = cell_bytes(part);
        let dst = if part == 3 { c.kv } else { c.idx_keys };
        let (n, max) = (u64::from(n), u64::from(c.max));
        unsafe {
            rt::hip_rt_memcpy_2d(
                dst.cast(),
                cell * max,
                src,
                cell * n,
                cell * n,
                ATTENTION_LAYERS,
                Copy::HostToDevice as u32,
            )
        };
    }

    /// Set the token count of a restored empty context to `n` (`engine.hip` `restore_n`). The pooled keys are derived
    /// state, not stored: the worker then pools every complete block of the written index keys (`host.rs`
    /// `pool_context`), so `pooled_n == n / 4`.
    pub(crate) unsafe extern "C" fn set_n(_engine: *mut c_void, c: *mut c_void, n: u32) {
        let context = unsafe { context(c) };
        assert!(!context.poisoned, "set_n({n}) on a poisoned context");
        assert_eq!(context.n, 0, "set_n({n}) on a context of {} tokens", context.n);
        assert!(
            n > 0 && n <= context.max,
            "set_n({n}) past the capacity {}",
            context.max
        );
        context.n = n;
        context.pooled_n = n / IDX_RATIO;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_part_sizes_are_the_declared_ones() {
        unsafe {
            assert_eq!(state_bytes(0), 113_246_208);
            assert_eq!(state_bytes(1), 4_423_680);
            assert_eq!(state_bytes(2), 368_640);
            assert_eq!(part_bytes(3, 5), 5 * 49_152);
            assert_eq!(part_bytes(4, 5), 5 * 6_144);
            assert_eq!(state_bytes(5), 8);
        }
        assert!(
            spec::parts_problem(
                |part| unsafe { state_bytes(part) },
                |part, n| unsafe { part_bytes(part, n) }
            )
            .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "depends on the token count")]
    fn a_cells_part_has_no_size_without_a_token_count() {
        fixed_bytes_of(3);
    }

    #[test]
    #[should_panic(expected = "context state part 6")]
    fn a_part_past_5_stops_the_program() {
        bytes_of(6, 0);
    }
}
