//! The C ABI of mk's engine, linked statically from `hip/` (spec 12.1, AGENTS.md milestone 15).
//!
//! `build.rs` compiles the engine into `libmkengine.a` under the feature `hip`. A build without `hip` links no mk
//! code, and `linked` returns an error. The engine is never closed: mk's HIP teardown segfaults in the process (spec
//! 3.3).
//!
//! The C++ side is a thin extern layer, `hip/src/engine/extern.hip`: each call calls one `Engine` method and checks
//! nothing. The checks and the result rules of a pass are here, in the methods of `Binding` (milestone 15 step 4).

use std::ffi::{c_char, c_int, c_void};

use crate::spec;
use crate::stage::Stage;
use crate::table::MkWeightTable;

/// `struct Segment` of `engine.h`: `t` tokens at `tokens` that continue `context`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MkSegment {
    pub context: *mut c_void,
    pub tokens: *const i32,
    pub t: u32,
}

/// `mk_text_sink`: gives `len` bytes of text, with no NUL, to the buffer `out`.
pub(crate) type TextSink = unsafe extern "C" fn(*mut c_void, *const c_char, u64);

/// `void *mk_engine_open(const mk_weight_table *t)`. The table comes from the Rust host (`table.rs`).
pub(crate) type EngineOpen = unsafe extern "C" fn(*const MkWeightTable) -> *mut c_void;

/// `void mk_relayout(uint32_t tile, const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k)`.
pub(crate) type Relayout = unsafe extern "C" fn(u32, *const u8, *mut u8, u32, u32);

/// `void mk_rt_host_register(void *p, uint64_t n)`.
pub(crate) type RtHostRegister = unsafe extern "C" fn(*mut c_void, u64);

/// `void *mk_rt_host_malloc(uint64_t n)`.
pub(crate) type RtHostMalloc = unsafe extern "C" fn(u64) -> *mut c_void;

/// `void mk_rt_host_free(void *p)`.
pub(crate) type RtHostFree = unsafe extern "C" fn(*mut c_void);

/// `void mk_engine_recover_spin_timeouts(void *e, bool on)`.
pub(crate) type RecoverSpinTimeouts = unsafe extern "C" fn(*mut c_void, bool);

/// The context call `new`, in Rust (`context.rs`).
pub(crate) type CtxNew = unsafe extern "C" fn(*mut c_void, u32) -> *mut c_void;

/// The context call `free`, in Rust (`context.rs`).
pub(crate) type CtxFree = unsafe extern "C" fn(*mut c_void, *mut c_void);

/// The context call `n`, in Rust (`context.rs`).
pub(crate) type CtxN = unsafe extern "C" fn(*mut c_void) -> u32;

/// The context call `max`, in Rust (`context.rs`).
pub(crate) type CtxMax = unsafe extern "C" fn(*mut c_void) -> u32;

/// The context call `owns_kv`, in Rust (`context.rs`).
pub(crate) type CtxOwnsKv = unsafe extern "C" fn(*mut c_void) -> bool;

/// The context call `poisoned`, in Rust (`context.rs`).
pub(crate) type CtxPoisoned = unsafe extern "C" fn(*mut c_void) -> bool;

/// The context call `state_bytes`, in Rust (`context.rs`).
pub(crate) type StateBytes = unsafe extern "C" fn(u32) -> u64;

/// The context call `part_bytes`, in Rust (`context.rs`).
pub(crate) type PartBytes = unsafe extern "C" fn(u32, u32) -> u64;

/// The context call `state_read`, in Rust (`context.rs`).
pub(crate) type StateRead = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void);

/// The context call `state_write`, in Rust (`context.rs`).
pub(crate) type StateWrite = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const c_void);

/// The context call `cells_write`, in Rust (`context.rs`).
pub(crate) type CellsWrite = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, *const c_void);

/// The context call `set_n`, in Rust (`context.rs`).
pub(crate) type SetN = unsafe extern "C" fn(*mut c_void, *mut c_void, u32);

/// `void mk_ctx_round_gdn_bf16(void *e, void *c)`.
pub(crate) type RoundGdn = unsafe extern "C" fn(*mut c_void, *mut c_void);

/// `void mk_last_error(void *e, mk_text_sink put, void *out)`.
pub(crate) type LastError = unsafe extern "C" fn(*mut c_void, TextSink, *mut c_void);

/// `void mk_segments_problem(void *e, const Segment *s, uint32_t n, mk_text_sink put, void *out)`.
pub(crate) type SegmentsProblem = unsafe extern "C" fn(*mut c_void, *const MkSegment, u32, TextSink, *mut c_void);

/// `double mk_pass_stages(void *e, const Segment *s, uint32_t n, bool want_logits, float *logits, const float *ple,
/// const Stage *stages, uint32_t n_stages)`: one fused pass of the stages that the Rust host built.
pub(crate) type PassStages =
    unsafe extern "C" fn(*mut c_void, *const MkSegment, u32, bool, *mut f32, *const f32, *const Stage, u32) -> f64;

/// `uint32_t mk_build(void *e, const Segment *s, uint32_t n, bool want_logits, Stage *out, uint32_t cap)`: the C++
/// build of the same pass, for the live comparison.
pub(crate) type BuildCpp = unsafe extern "C" fn(*mut c_void, *const MkSegment, u32, bool, *mut Stage, u32) -> u32;

/// `float *mk_scratch(void *e, const char *name, uint64_t floats)`.
pub(crate) type ScratchCpp = unsafe extern "C" fn(*mut c_void, *const c_char, u64) -> *mut f32;

/// `void mk_pass_buffers(void *e, const int32_t **tokens, const uint32_t **head_rows, const float **inv_freq)`.
pub(crate) type PassBuffersCpp = unsafe extern "C" fn(*mut c_void, *mut *const i32, *mut *const u32, *mut *const f32);

/// `hipError_t hipMemGetInfo(size_t *free, size_t *total)`, from the HIP runtime that the engine links.
pub(crate) type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;

/// The calls that this binding uses.
pub(crate) struct Binding {
    pub engine_open: EngineOpen,
    pub relayout: Relayout,
    pub rt_host_register: RtHostRegister,
    pub rt_host_malloc: RtHostMalloc,
    pub rt_host_free: RtHostFree,
    pub recover_spin_timeouts: RecoverSpinTimeouts,
    pub ctx_new: CtxNew,
    pub ctx_free: CtxFree,
    pub ctx_n: CtxN,
    pub ctx_max: CtxMax,
    pub ctx_owns_kv: CtxOwnsKv,
    pub ctx_poisoned: CtxPoisoned,
    pub state_bytes: StateBytes,
    pub part_bytes: PartBytes,
    pub state_read: StateRead,
    pub state_write: StateWrite,
    pub cells_write: CellsWrite,
    pub set_n: SetN,
    pub round_gdn: RoundGdn,
    pub last_error: LastError,
    pub segments_problem: SegmentsProblem,
    pub pass_stages: PassStages,
    pub build_cpp: BuildCpp,
    pub scratch_cpp: ScratchCpp,
    pub pass_buffers_cpp: PassBuffersCpp,
    pub mem_get_info: MemGetInfo,
}

/// Why a pass did not finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PassStop {
    /// The binding or mk's segment check refused the pass before it ran. Nothing changed.
    Refused(String),
    /// The pass ran and stopped on a spin-timeout. mk poisoned each of its contexts.
    Stopped(String),
}

/// The binding's `mk_text_sink`: append the text to the `Vec<u8>` at `out`.
unsafe extern "C" fn put_text(out: *mut c_void, text: *const c_char, len: u64) {
    // SAFETY: the binding gives a pointer to its own `Vec<u8>` as `out`, and mk gives `len` readable bytes at `text`.
    let out = unsafe { &mut *(out as *mut Vec<u8>) };
    if len > 0 {
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(text as *const u8, len as usize) });
    }
}

/// The segments of a pass: `lens[i]` tokens of `tokens` continue `contexts[i]`, in order.
///
/// A count that differs is a defect of the caller, so it stops the program.
pub(crate) fn segments(contexts: &[*mut c_void], lens: &[u32], tokens: &[i32]) -> Vec<MkSegment> {
    assert_eq!(contexts.len(), lens.len(), "a pass needs one length for each context");
    let total: usize = lens.iter().map(|len| *len as usize).sum();
    assert_eq!(total, tokens.len(), "the lengths of a pass must add to its token count");
    let mut start = 0;
    contexts
        .iter()
        .zip(lens)
        .map(|(context, len)| {
            let segment = MkSegment {
                context: *context,
                tokens: tokens[start..].as_ptr(),
                t: *len,
            };
            start += *len as usize;
            segment
        })
        .collect()
}

impl Binding {
    /// The last pass's error, whole. It is empty after a clean pass.
    ///
    /// # Safety
    ///
    /// `engine` is an engine of this binding.
    pub(crate) unsafe fn last_error(&self, engine: *mut c_void) -> String {
        let mut text = Vec::new();
        unsafe { (self.last_error)(engine, put_text, (&mut text as *mut Vec<u8>).cast()) };
        String::from_utf8_lossy(&text).into_owned()
    }

    /// The first problem of `segments` as one pass, or `None`.
    ///
    /// The binding refuses a segment count outside `1..=MAX_SEGMENTS` itself, and asks mk for the other rules.
    ///
    /// # Safety
    ///
    /// `engine` is an engine of this binding, and each segment names a live context of it.
    pub(crate) unsafe fn segments_problem(&self, engine: *mut c_void, segments: &[MkSegment]) -> Option<String> {
        if segments.is_empty() || segments.len() > spec::MAX_SEGMENTS as usize {
            return Some(format!(
                "{} segments: a pass takes 1..={} segments",
                segments.len(),
                spec::MAX_SEGMENTS
            ));
        }
        let mut text = Vec::new();
        unsafe {
            (self.segments_problem)(
                engine,
                segments.as_ptr(),
                segments.len() as u32,
                put_text,
                (&mut text as *mut Vec<u8>).cast(),
            )
        };
        (!text.is_empty()).then(|| String::from_utf8_lossy(&text).into_owned())
    }

    /// Run one pass of `segments`, and give its wall time in milliseconds. mk runs each pass fused. `ple` holds the
    /// PLE rows of the pass's tokens, `ple::EMBD` values for each token.
    ///
    /// mk stops the program on segments that its check refuses and on a poisoned context. Thus both checks run
    /// first, and a problem refuses the pass. `logits` takes one row of `VOCAB` values for each segment. A buffer of another size stops the program,
    /// because mk would write past its end. After the pass, a last error means that mk stopped the pass and poisoned
    /// its contexts.
    ///
    /// # Safety
    ///
    /// `engine` is an engine of this binding, and each segment names a live context of it and `t` readable tokens.
    pub(crate) unsafe fn pass_segments(
        &self,
        engine: *mut c_void,
        segments: &[MkSegment],
        logits: Option<&mut [f32]>,
        ple: &[f32],
        build: impl FnOnce() -> Vec<Stage>,
    ) -> Result<f64, PassStop> {
        let tokens: usize = segments.iter().map(|segment| segment.t as usize).sum();
        assert_eq!(
            ple.len(),
            tokens * crate::ple::EMBD,
            "the PLE rows must hold EMBD values for each token of the pass"
        );
        if let Some(logits) = &logits {
            assert_eq!(
                logits.len(),
                segments.len() * spec::VOCAB as usize,
                "the logits buffer must hold one row of VOCAB values for each segment"
            );
        }
        if let Some(problem) = unsafe { self.segments_problem(engine, segments) } {
            return Err(PassStop::Refused(problem));
        }
        // mk stops the program on a poisoned context, before its segment check (engine.hip run_pass).
        for (index, segment) in segments.iter().enumerate() {
            if unsafe { (self.ctx_poisoned)(segment.context) } {
                return Err(PassStop::Refused(format!(
                    "segment {index}: the context is poisoned; free it"
                )));
            }
        }
        let (want_logits, logits) = match logits {
            Some(logits) => (true, logits.as_mut_ptr()),
            None => (false, std::ptr::null_mut()),
        };
        // The build reads the contexts, so it runs after the checks.
        let stages = build();
        let ms = unsafe {
            (self.pass_stages)(
                engine,
                segments.as_ptr(),
                segments.len() as u32,
                want_logits,
                logits,
                ple.as_ptr(),
                stages.as_ptr(),
                stages.len() as u32,
            )
        };
        let error = unsafe { self.last_error(engine) };
        if error.is_empty() {
            Ok(ms)
        } else {
            Err(PassStop::Stopped(error))
        }
    }
}

/// The facts of the engine build, from `build.rs`. The identity holds them (spec 12.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Build {
    /// The SHA-256 of `libmkengine.a`, as 64 lowercase hex digits.
    pub library_sha256: &'static str,
    /// The first line of `clang++ --version`, the compiler of every unit.
    pub compiler: &'static str,
    /// The output of `hipconfig --version`, the HIP version of the build.
    pub hip: &'static str,
}

/// The calls of the engine and the runtime that the Rust host makes itself (`context.rs`).
#[cfg(feature = "hip")]
pub(crate) mod rt {
    pub(crate) use super::linked::{
        mk_ctx_pool, mk_rt_free, mk_rt_malloc, mk_rt_memcpy, mk_rt_memcpy_2d, mk_rt_memset,
    };
}

#[cfg(feature = "hip")]
mod linked {
    use std::ffi::{c_int, c_void};

    use std::ffi::c_char;

    use super::{MkSegment, TextSink};
    use crate::stage::Stage;
    use crate::table::MkWeightTable;

    // The calls of the extern layer `hip/src/engine/extern.hip`, the runtime calls `mk_rt_*` of
    // `hip/src/rt/rt.cpp`, and one call of the HIP runtime. A test compares the lists of calls.
    unsafe extern "C" {
        pub(super) fn mk_engine_open(table: *const MkWeightTable) -> *mut c_void;
        pub(super) fn mk_relayout(tile: u32, src: *const u8, dst: *mut u8, rows: u32, k: u32);
        pub(super) fn mk_rt_host_register(p: *mut c_void, n: u64);
        pub(super) fn mk_rt_host_malloc(n: u64) -> *mut c_void;
        pub(super) fn mk_rt_host_free(p: *mut c_void);
        pub(super) fn mk_engine_recover_spin_timeouts(e: *mut c_void, on: bool);
        pub(crate) fn mk_ctx_pool(e: *mut c_void, c: *const c_void);
        pub(crate) fn mk_rt_malloc(n: u64) -> *mut c_void;
        pub(crate) fn mk_rt_free(p: *mut c_void);
        pub(crate) fn mk_rt_memset(p: *mut c_void, value: i32, n: u64);
        pub(crate) fn mk_rt_memcpy(dst: *mut c_void, src: *const c_void, n: u64, kind: u32);
        pub(crate) fn mk_rt_memcpy_2d(
            dst: *mut c_void,
            dpitch: u64,
            src: *const c_void,
            spitch: u64,
            width: u64,
            height: u64,
            kind: u32,
        );
        pub(super) fn mk_ctx_round_gdn_bf16(e: *mut c_void, c: *mut c_void);
        pub(super) fn mk_last_error(e: *mut c_void, put: TextSink, out: *mut c_void);
        pub(super) fn mk_segments_problem(e: *mut c_void, s: *const MkSegment, n: u32, put: TextSink, out: *mut c_void);
        pub(super) fn mk_pass_stages(
            e: *mut c_void,
            s: *const MkSegment,
            n: u32,
            want_logits: bool,
            logits: *mut f32,
            ple: *const f32,
            stages: *const Stage,
            n_stages: u32,
        ) -> f64;
        pub(super) fn mk_build(
            e: *mut c_void,
            s: *const MkSegment,
            n: u32,
            want_logits: bool,
            out: *mut Stage,
            cap: u32,
        ) -> u32;
        pub(super) fn mk_scratch(e: *mut c_void, name: *const c_char, floats: u64) -> *mut f32;
        pub(super) fn mk_pass_buffers(
            e: *mut c_void,
            tokens: *mut *const i32,
            head_rows: *mut *const u32,
            inv_freq: *mut *const f32,
        );
        pub(super) fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    }

    /// The facts that `build.rs` gave.
    pub(super) const BUILD: super::Build = super::Build {
        library_sha256: env!("MK_ENGINE_BUILD_SHA256"),
        compiler: env!("MK_ENGINE_BUILD_COMPILER"),
        hip: env!("MK_ENGINE_BUILD_HIP"),
    };
}

/// Bind every call to the engine that this binary links, and give the facts of its build.
#[cfg(feature = "hip")]
pub(crate) fn linked() -> Result<(Binding, Build), String> {
    use crate::context;
    use linked::*;
    Ok((
        Binding {
            engine_open: mk_engine_open,
            relayout: mk_relayout,
            rt_host_register: mk_rt_host_register,
            rt_host_malloc: mk_rt_host_malloc,
            rt_host_free: mk_rt_host_free,
            recover_spin_timeouts: mk_engine_recover_spin_timeouts,
            ctx_new: context::ctx_new,
            ctx_free: context::ctx_free,
            ctx_n: context::ctx_n,
            ctx_max: context::ctx_max,
            ctx_owns_kv: context::ctx_owns_kv,
            ctx_poisoned: context::ctx_poisoned,
            state_bytes: context::state_bytes,
            part_bytes: context::part_bytes,
            state_read: context::state_read,
            state_write: context::state_write,
            cells_write: context::cells_write,
            set_n: context::set_n,
            round_gdn: mk_ctx_round_gdn_bf16,
            last_error: mk_last_error,
            segments_problem: mk_segments_problem,
            pass_stages: mk_pass_stages,
            build_cpp: mk_build,
            scratch_cpp: mk_scratch,
            pass_buffers_cpp: mk_pass_buffers,
            mem_get_info: hipMemGetInfo,
        },
        BUILD,
    ))
}

/// A build without the feature `hip` links no engine.
#[cfg(not(feature = "hip"))]
pub(crate) fn linked() -> Result<(Binding, Build), String> {
    Err(NOT_LINKED.to_owned())
}

/// The error of `linked` in a build without the feature `hip`.
#[cfg(not(feature = "hip"))]
pub(crate) const NOT_LINKED: &str =
    "this build links no mk engine; build nagaya-mk-qwen38-gfx1151 with the feature hip";

/// A binding to stub calls and a stub context, for tests that need a poisoned context. mk poisons a context only
/// on a spin timeout, and no test can force that.
#[cfg(test)]
pub(crate) mod stub {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

    use super::{
        Binding, BuildCpp, CtxFree, CtxMax, CtxNew, EngineOpen, LastError, MemGetInfo, PartBytes, PassBuffersCpp,
        PassStages, RecoverSpinTimeouts, Relayout, RoundGdn, RtHostFree, RtHostMalloc, RtHostRegister, ScratchCpp,
        SegmentsProblem, StateBytes, StateRead,
    };

    /// What the stub calls know about one context. A test passes a pointer to this in place of mk's context.
    pub(crate) struct StubContext {
        pub poisoned: AtomicBool,
        pub n: AtomicU32,
        /// The count of calls that would change the context in mk.
        pub changes: AtomicUsize,
        /// When true, a state write poisons the context, as an mk write that fails would.
        pub write_poisons: bool,
    }

    impl StubContext {
        pub(crate) fn new(poisoned: bool) -> Self {
            StubContext {
                poisoned: AtomicBool::new(poisoned),
                n: AtomicU32::new(0),
                changes: AtomicUsize::new(0),
                write_poisons: false,
            }
        }
    }

    unsafe fn context<'a>(pointer: *mut c_void) -> &'a StubContext {
        unsafe { &*(pointer as *const StubContext) }
    }

    unsafe extern "C" fn ctx_n(pointer: *mut c_void) -> u32 {
        unsafe { context(pointer) }.n.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn ctx_owns_kv(_pointer: *mut c_void) -> bool {
        true
    }

    unsafe extern "C" fn ctx_poisoned(pointer: *mut c_void) -> bool {
        unsafe { context(pointer) }.poisoned.load(Ordering::SeqCst)
    }

    unsafe extern "C" fn state_write(_engine: *mut c_void, pointer: *mut c_void, _part: u32, _source: *const c_void) {
        let context = unsafe { context(pointer) };
        context.changes.fetch_add(1, Ordering::SeqCst);
        if context.write_poisons {
            context.poisoned.store(true, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn cells_write(
        _engine: *mut c_void,
        pointer: *mut c_void,
        _part: u32,
        _n: u32,
        _source: *const c_void,
    ) {
        unsafe { context(pointer) }.changes.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn set_n(_engine: *mut c_void, pointer: *mut c_void, n: u32) {
        let context = unsafe { context(pointer) };
        context.changes.fetch_add(1, Ordering::SeqCst);
        context.n.store(n, Ordering::SeqCst);
    }

    /// A call that a stub test must not reach.
    unsafe extern "C" fn never() {
        std::process::abort();
    }

    /// A binding whose calls on a context read a `StubContext`. Every other call aborts the process.
    pub(crate) fn binding() -> Binding {
        let never: unsafe extern "C" fn() = never;
        // The transmutes below only name the call types. The stub never calls them, and `never` takes no argument.
        unsafe {
            Binding {
                engine_open: std::mem::transmute::<unsafe extern "C" fn(), EngineOpen>(never),
                relayout: std::mem::transmute::<unsafe extern "C" fn(), Relayout>(never),
                rt_host_register: std::mem::transmute::<unsafe extern "C" fn(), RtHostRegister>(never),
                rt_host_malloc: std::mem::transmute::<unsafe extern "C" fn(), RtHostMalloc>(never),
                rt_host_free: std::mem::transmute::<unsafe extern "C" fn(), RtHostFree>(never),
                ctx_new: std::mem::transmute::<unsafe extern "C" fn(), CtxNew>(never),
                ctx_free: std::mem::transmute::<unsafe extern "C" fn(), CtxFree>(never),
                ctx_n,
                ctx_max: std::mem::transmute::<unsafe extern "C" fn(), CtxMax>(never),
                ctx_owns_kv,
                ctx_poisoned,
                state_read: std::mem::transmute::<unsafe extern "C" fn(), StateRead>(never),
                state_write,
                cells_write,
                set_n,
                recover_spin_timeouts: std::mem::transmute::<unsafe extern "C" fn(), RecoverSpinTimeouts>(never),
                round_gdn: std::mem::transmute::<unsafe extern "C" fn(), RoundGdn>(never),
                part_bytes: std::mem::transmute::<unsafe extern "C" fn(), PartBytes>(never),
                state_bytes: std::mem::transmute::<unsafe extern "C" fn(), StateBytes>(never),
                last_error: std::mem::transmute::<unsafe extern "C" fn(), LastError>(never),
                segments_problem: std::mem::transmute::<unsafe extern "C" fn(), SegmentsProblem>(never),
                pass_stages: std::mem::transmute::<unsafe extern "C" fn(), PassStages>(never),
                build_cpp: std::mem::transmute::<unsafe extern "C" fn(), BuildCpp>(never),
                scratch_cpp: std::mem::transmute::<unsafe extern "C" fn(), ScratchCpp>(never),
                pass_buffers_cpp: std::mem::transmute::<unsafe extern "C" fn(), PassBuffersCpp>(never),
                mem_get_info: std::mem::transmute::<unsafe extern "C" fn(), MemGetInfo>(never),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::ffi::c_void;
    use std::path::Path;

    use super::stub::{self, StubContext};
    use super::{Binding, MkSegment, PassStop, TextSink, segments};
    use crate::spec;
    use crate::stage::Stage;

    /// The names of the `mk_` calls that `text` defines or declares: each `mk_<name>(` that follows `fn ` in Rust, or
    /// that starts a definition at the left margin in C++.
    fn calls(text: &str, rust: bool) -> Vec<String> {
        let mut names: Vec<String> = text
            .lines()
            .filter_map(|line| {
                let line = if rust {
                    let line = line.trim_start();
                    line.strip_prefix("pub(super) fn ")
                        .or_else(|| line.strip_prefix("pub(crate) fn "))?
                } else {
                    line
                };
                if !rust && (line.starts_with(' ') || line.starts_with("//")) {
                    return None;
                }
                let start = line.find("mk_")?;
                let end = start + line[start..].find('(')?;
                let name = &line[start..end];
                name.bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                    .then(|| name.to_owned())
            })
            .collect();
        names.sort();
        names
    }

    /// What the stub calls of a pass saw and give, on this test's thread.
    #[derive(Default)]
    struct PassStub {
        /// The text that the stub `mk_segments_problem` gives.
        problem: String,
        /// The text that the stub `mk_last_error` gives.
        error: String,
        /// The count of calls of `mk_segments_problem`.
        checks: usize,
        /// The arguments of each `mk_pass_segments` call: the segment count, `want_logits`, and whether `logits` is
        /// null.
        passes: Vec<(u32, bool, bool)>,
    }

    thread_local! {
        static PASS: RefCell<PassStub> = RefCell::new(PassStub::default());
    }

    /// Give `text` to mk's sink, as the extern layer does.
    unsafe fn give(put: TextSink, out: *mut c_void, text: &str) {
        unsafe { put(out, text.as_ptr().cast(), text.len() as u64) };
    }

    unsafe extern "C" fn segments_problem(
        _engine: *mut c_void,
        _segments: *const MkSegment,
        _n: u32,
        put: TextSink,
        out: *mut c_void,
    ) {
        let problem = PASS.with_borrow_mut(|stub| {
            stub.checks += 1;
            stub.problem.clone()
        });
        unsafe { give(put, out, &problem) };
    }

    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" fn pass_stages(
        _engine: *mut c_void,
        _segments: *const MkSegment,
        n: u32,
        want_logits: bool,
        logits: *mut f32,
        _ple: *const f32,
        _stages: *const Stage,
        _n_stages: u32,
    ) -> f64 {
        PASS.with_borrow_mut(|stub| stub.passes.push((n, want_logits, logits.is_null())));
        12.5
    }

    unsafe extern "C" fn last_error(_engine: *mut c_void, put: TextSink, out: *mut c_void) {
        let error = PASS.with_borrow(|stub| stub.error.clone());
        unsafe { give(put, out, &error) };
    }

    /// A binding whose pass calls are the stubs above, with a new `PassStub` for this thread.
    fn pass_binding(problem: &str, error: &str) -> Binding {
        PASS.set(PassStub {
            problem: problem.to_owned(),
            error: error.to_owned(),
            ..PassStub::default()
        });
        Binding {
            segments_problem,
            pass_stages,
            last_error,
            ..stub::binding()
        }
    }

    /// `count` segments of one token each, on distinct stub contexts that are not poisoned. The test leaks them.
    fn some_segments(count: usize, tokens: &[i32]) -> Vec<MkSegment> {
        let contexts: Vec<*mut c_void> = (0..count)
            .map(|_| Box::leak(Box::new(StubContext::new(false))) as *mut StubContext as *mut c_void)
            .collect();
        segments(&contexts, &vec![1; count], &tokens[..count])
    }

    #[test]
    fn a_pass_on_a_poisoned_context_is_refused_before_mk() {
        // Engine::run_pass stops the program on a poisoned context before its own segment check.
        let binding = pass_binding("", "");
        let tokens = [7i32; 2];
        let poisoned = Box::leak(Box::new(StubContext::new(true))) as *mut StubContext as *mut c_void;
        let mut made = some_segments(2, &tokens);
        made[1].context = poisoned;
        let result = unsafe {
            binding.pass_segments(
                std::ptr::null_mut(),
                &made,
                None,
                &[0.0; 2 * crate::ple::EMBD],
                Vec::new,
            )
        };
        assert!(
            matches!(&result, Err(PassStop::Refused(message)) if message.contains("segment 1") && message.contains("poisoned")),
            "{result:?}"
        );
        PASS.with_borrow(|stub| {
            // mk's check comes first: it refuses a null context, which the poison read would follow.
            assert_eq!(stub.checks, 1);
            assert!(stub.passes.is_empty(), "a poisoned context reached mk's pass");
        });
    }

    #[test]
    fn a_pass_of_no_segments_or_of_too_many_is_refused_before_mk() {
        let tokens = [7i32; 17];
        for count in [0, spec::MAX_SEGMENTS as usize + 1] {
            let binding = pass_binding("", "");
            let segments = some_segments(count, &tokens);
            let result = unsafe {
                binding.pass_segments(
                    std::ptr::null_mut(),
                    &segments,
                    None,
                    &vec![0.0; count * crate::ple::EMBD],
                    Vec::new,
                )
            };
            assert!(
                matches!(&result, Err(PassStop::Refused(message)) if message.contains("1..=16 segments")),
                "{count} segments: {result:?}"
            );
            PASS.with_borrow(|stub| {
                assert_eq!(stub.checks, 0, "{count} segments reached mk's check");
                assert!(stub.passes.is_empty(), "{count} segments reached mk's pass");
            });
        }
    }

    #[test]
    fn a_pass_runs_mk_check_first_and_a_problem_refuses_it() {
        let binding = pass_binding("segment 0: context full", "");
        let tokens = [7i32; 1];
        let result = unsafe {
            binding.pass_segments(
                std::ptr::null_mut(),
                &some_segments(1, &tokens),
                None,
                &[0.0; crate::ple::EMBD],
                Vec::new,
            )
        };
        assert_eq!(result, Err(PassStop::Refused("segment 0: context full".to_owned())));
        PASS.with_borrow(|stub| {
            assert_eq!(stub.checks, 1);
            assert!(stub.passes.is_empty(), "a refused pass reached mk");
        });
    }

    #[test]
    fn a_pass_with_a_last_error_stopped_and_the_error_is_whole() {
        // The old C ABI cut the error at the caller's buffer of 1,024 bytes. The sink takes it whole.
        let error = "x".repeat(5000);
        let binding = pass_binding("", &error);
        let tokens = [7i32; 2];
        let result = unsafe {
            binding.pass_segments(
                std::ptr::null_mut(),
                &some_segments(2, &tokens),
                None,
                &[0.0; 2 * crate::ple::EMBD],
                Vec::new,
            )
        };
        assert_eq!(result, Err(PassStop::Stopped(error)));
        PASS.with_borrow(|stub| assert_eq!(stub.passes.len(), 1));
    }

    #[test]
    fn a_clean_pass_gives_its_time_and_asks_for_logits_only_with_a_buffer() {
        let tokens = [7i32; 2];
        let binding = pass_binding("", "");
        let result = unsafe {
            binding.pass_segments(
                std::ptr::null_mut(),
                &some_segments(2, &tokens),
                None,
                &[0.0; 2 * crate::ple::EMBD],
                Vec::new,
            )
        };
        assert_eq!(result, Ok(12.5));
        let mut logits = vec![0f32; 2 * spec::VOCAB as usize];
        let result = unsafe {
            binding.pass_segments(
                std::ptr::null_mut(),
                &some_segments(2, &tokens),
                Some(&mut logits),
                &[0.0; 2 * crate::ple::EMBD],
                Vec::new,
            )
        };
        assert_eq!(result, Ok(12.5));
        PASS.with_borrow(|stub| assert_eq!(stub.passes, [(2, false, true), (2, true, false)]));
    }

    #[test]
    #[should_panic(expected = "one row of VOCAB values for each segment")]
    fn a_logits_buffer_of_another_size_stops_the_program() {
        let binding = pass_binding("", "");
        let tokens = [7i32; 2];
        let mut logits = vec![0f32; spec::VOCAB as usize];
        let _ = unsafe {
            binding.pass_segments(
                std::ptr::null_mut(),
                &some_segments(2, &tokens),
                Some(&mut logits),
                &[0.0; 2 * crate::ple::EMBD],
                Vec::new,
            )
        };
    }

    #[test]
    fn the_segments_take_their_tokens_in_order() {
        let tokens = [10i32, 11, 12, 13, 14, 15];
        let contexts = [2usize as *mut c_void, 3usize as *mut c_void, 4usize as *mut c_void];
        let made = segments(&contexts, &[1, 3, 2], &tokens);
        let starts: Vec<i32> = made.iter().map(|segment| unsafe { *segment.tokens }).collect();
        assert_eq!(starts, [10, 11, 14]);
        assert_eq!(made.iter().map(|segment| segment.t).collect::<Vec<_>>(), [1, 3, 2]);
        assert_eq!(made.iter().map(|segment| segment.context).collect::<Vec<_>>(), contexts);
    }

    #[test]
    #[should_panic(expected = "must add to its token count")]
    fn segments_whose_lengths_do_not_add_to_the_tokens_stop_the_program() {
        segments(&[std::ptr::dangling_mut::<c_void>()], &[3], &[1, 2]);
    }

    #[test]
    fn the_last_error_of_a_clean_pass_is_empty() {
        let binding = pass_binding("", "");
        assert_eq!(unsafe { binding.last_error(std::ptr::null_mut()) }, "");
        let binding = pass_binding("", "spin-timeout at stage 3");
        assert_eq!(
            unsafe { binding.last_error(std::ptr::null_mut()) },
            "spin-timeout at stage 3"
        );
    }

    #[test]
    fn the_extern_layer_and_the_binding_have_the_same_calls() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let layer = std::fs::read_to_string(root.join("hip/src/engine/extern.hip")).expect("read extern.hip");
        let runtime = std::fs::read_to_string(root.join("hip/src/rt/rt.cpp")).expect("read rt.cpp");
        let binding = std::fs::read_to_string(root.join("src/binding.rs")).expect("read binding.rs");
        let mut layer = calls(&layer, false);
        // rt.cpp also holds the trap calls of the op tests. The host's runtime calls are the ones named mk_rt_.
        layer.extend(
            calls(&runtime, false)
                .into_iter()
                .filter(|name| name.starts_with("mk_rt_")),
        );
        layer.sort();
        assert!(
            layer.len() >= 10,
            "the extern layer defines {} calls: {layer:?}",
            layer.len()
        );
        assert_eq!(
            layer,
            calls(&binding, true),
            "extern.hip and the declarations in binding.rs differ"
        );
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn a_build_without_hip_links_no_engine() {
        let error = super::linked().err().expect("a build without hip linked an engine");
        assert_eq!(error, super::NOT_LINKED);
        assert!(error.contains("feature hip"), "{error}");
    }

    #[cfg(feature = "hip")]
    #[test]
    fn a_hip_build_gives_the_facts_of_its_build() {
        let (_binding, build) = super::linked().expect("a hip build links the engine");
        assert_eq!(build.library_sha256.len(), 64, "{}", build.library_sha256);
        assert!(
            build
                .library_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "{}",
            build.library_sha256
        );
        assert!(build.compiler.contains("clang version"), "{}", build.compiler);
        assert!(!build.hip.is_empty());
        // The part sizes are pure host code, so this call needs no device.
        assert!(unsafe { (_binding.state_bytes)(0) } > 0);
    }
}
