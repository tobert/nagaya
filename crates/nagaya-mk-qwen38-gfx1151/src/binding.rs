//! The C ABI of mk's engine, linked statically from `hip/` (spec 12.1, AGENTS.md milestone 15).
//!
//! `build.rs` compiles the engine into `libmkengine.a` under the feature `hip`. A build without `hip` links no mk
//! code, and `linked` returns an error. The engine is never closed: mk's HIP teardown segfaults in the process (spec
//! 3.3).

use std::ffi::{c_char, c_int, c_void};

use crate::table::MkTensorTable;

/// `void *mk_engine_open(const mk_tensor_table *t)`. The table comes from the Rust reader (`table.rs`).
pub(crate) type EngineOpen = unsafe extern "C" fn(*const MkTensorTable) -> *mut c_void;

/// `void *mk_ctx_new(void *e, uint32_t max_tokens)`.
pub(crate) type CtxNew = unsafe extern "C" fn(*mut c_void, u32) -> *mut c_void;

/// `void mk_ctx_free(void *e, void *c)`.
pub(crate) type CtxFree = unsafe extern "C" fn(*mut c_void, *mut c_void);

/// `uint32_t mk_ctx_n(void *c)`.
pub(crate) type CtxN = unsafe extern "C" fn(*mut c_void) -> u32;

/// `uint32_t mk_ctx_max(void *c)`.
pub(crate) type CtxMax = unsafe extern "C" fn(*mut c_void) -> u32;

/// `int mk_ctx_owns_kv(void *c)`.
pub(crate) type CtxOwnsKv = unsafe extern "C" fn(*mut c_void) -> c_int;

/// `int mk_ctx_poisoned(void *c)`.
pub(crate) type CtxPoisoned = unsafe extern "C" fn(*mut c_void) -> c_int;

/// `void mk_ctx_state_read(void *e, void *c, uint32_t part, void *dst)`.
pub(crate) type StateRead = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void);

/// `void mk_ctx_state_write(void *e, void *c, uint32_t part, const void *src)`.
pub(crate) type StateWrite = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const c_void);

/// `void mk_ctx_cells_write(void *e, void *c, uint32_t part, uint32_t n, const void *src)`.
pub(crate) type CellsWrite = unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, *const c_void);

/// `void mk_ctx_set_n(void *e, void *c, uint32_t n)`.
pub(crate) type SetN = unsafe extern "C" fn(*mut c_void, *mut c_void, u32);

/// `void mk_engine_recover_spin_timeouts(void *e, int on)`.
pub(crate) type RecoverSpinTimeouts = unsafe extern "C" fn(*mut c_void, c_int);

/// `void mk_ctx_round_gdn_bf16(void *e, void *c)`.
pub(crate) type RoundGdn = unsafe extern "C" fn(*mut c_void, *mut c_void);

/// `uint64_t mk_ctx_part_bytes(uint32_t part, uint32_t n)`.
pub(crate) type PartBytes = unsafe extern "C" fn(u32, u32) -> u64;

/// `uint64_t mk_ctx_state_bytes(uint32_t part)`.
pub(crate) type StateBytes = unsafe extern "C" fn(u32) -> u64;

/// `void mk_engine_arena(void *e, int32_t *fd, uint64_t *bytes)`.
pub(crate) type Arena = unsafe extern "C" fn(*mut c_void, *mut i32, *mut u64);

/// `hipError_t hipMemGetInfo(size_t *free, size_t *total)`, from the HIP runtime that the engine links.
pub(crate) type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;

/// `uint32_t mk_last_error(void *e, char *why, uint32_t cap)`.
pub(crate) type LastError = unsafe extern "C" fn(*mut c_void, *mut c_char, u32) -> u32;

/// `int mk_segments_problem(void *e, uint32_t n, void *const *ctxs, const uint32_t *lens, char *why, uint32_t cap)`.
pub(crate) type SegmentsProblem =
    unsafe extern "C" fn(*mut c_void, u32, *const *mut c_void, *const u32, *mut c_char, u32) -> c_int;

/// `double mk_pass_segments(void *e, uint32_t n, void *const *ctxs, const uint32_t *lens, const int32_t *tokens,
/// int fused, float *logits, float *hidden)`.
#[allow(clippy::type_complexity)]
pub(crate) type PassSegments = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *const *mut c_void,
    *const u32,
    *const i32,
    c_int,
    *mut f32,
    *mut f32,
) -> f64;

/// The calls that this binding uses.
pub(crate) struct Binding {
    pub engine_open: EngineOpen,
    pub ctx_new: CtxNew,
    pub ctx_free: CtxFree,
    pub ctx_n: CtxN,
    pub ctx_max: CtxMax,
    pub ctx_owns_kv: CtxOwnsKv,
    pub ctx_poisoned: CtxPoisoned,
    pub state_read: StateRead,
    pub state_write: StateWrite,
    pub cells_write: CellsWrite,
    pub set_n: SetN,
    pub recover_spin_timeouts: RecoverSpinTimeouts,
    pub round_gdn: RoundGdn,
    pub part_bytes: PartBytes,
    pub state_bytes: StateBytes,
    pub arena: Arena,
    pub last_error: LastError,
    pub segments_problem: SegmentsProblem,
    pub pass_segments: PassSegments,
    pub mem_get_info: MemGetInfo,
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

#[cfg(feature = "hip")]
mod linked {
    use std::ffi::{c_char, c_int, c_void};

    // The declarations of mk `hip/src/engine/capi.hip`, and one call of the HIP runtime.
    unsafe extern "C" {
        pub(super) fn mk_engine_open(table: *const crate::table::MkTensorTable) -> *mut c_void;
        pub(super) fn mk_ctx_new(e: *mut c_void, max_tokens: u32) -> *mut c_void;
        pub(super) fn mk_ctx_free(e: *mut c_void, c: *mut c_void);
        pub(super) fn mk_ctx_n(c: *mut c_void) -> u32;
        pub(super) fn mk_ctx_max(c: *mut c_void) -> u32;
        pub(super) fn mk_ctx_owns_kv(c: *mut c_void) -> c_int;
        pub(super) fn mk_ctx_poisoned(c: *mut c_void) -> c_int;
        pub(super) fn mk_ctx_state_read(e: *mut c_void, c: *mut c_void, part: u32, dst: *mut c_void);
        pub(super) fn mk_ctx_state_write(e: *mut c_void, c: *mut c_void, part: u32, src: *const c_void);
        pub(super) fn mk_ctx_cells_write(e: *mut c_void, c: *mut c_void, part: u32, n: u32, src: *const c_void);
        pub(super) fn mk_ctx_set_n(e: *mut c_void, c: *mut c_void, n: u32);
        pub(super) fn mk_engine_recover_spin_timeouts(e: *mut c_void, on: c_int);
        pub(super) fn mk_ctx_round_gdn_bf16(e: *mut c_void, c: *mut c_void);
        pub(super) fn mk_ctx_part_bytes(part: u32, n: u32) -> u64;
        pub(super) fn mk_ctx_state_bytes(part: u32) -> u64;
        pub(super) fn mk_engine_arena(e: *mut c_void, fd: *mut i32, bytes: *mut u64);
        pub(super) fn mk_last_error(e: *mut c_void, why: *mut c_char, cap: u32) -> u32;
        pub(super) fn mk_segments_problem(
            e: *mut c_void,
            n: u32,
            ctxs: *const *mut c_void,
            lens: *const u32,
            why: *mut c_char,
            cap: u32,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub(super) fn mk_pass_segments(
            e: *mut c_void,
            n: u32,
            ctxs: *const *mut c_void,
            lens: *const u32,
            tokens: *const i32,
            fused: c_int,
            logits: *mut f32,
            hidden: *mut f32,
        ) -> f64;
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
    use linked::*;
    Ok((
        Binding {
            engine_open: mk_engine_open,
            ctx_new: mk_ctx_new,
            ctx_free: mk_ctx_free,
            ctx_n: mk_ctx_n,
            ctx_max: mk_ctx_max,
            ctx_owns_kv: mk_ctx_owns_kv,
            ctx_poisoned: mk_ctx_poisoned,
            state_read: mk_ctx_state_read,
            state_write: mk_ctx_state_write,
            cells_write: mk_ctx_cells_write,
            set_n: mk_ctx_set_n,
            recover_spin_timeouts: mk_engine_recover_spin_timeouts,
            round_gdn: mk_ctx_round_gdn_bf16,
            part_bytes: mk_ctx_part_bytes,
            state_bytes: mk_ctx_state_bytes,
            arena: mk_engine_arena,
            last_error: mk_last_error,
            segments_problem: mk_segments_problem,
            pass_segments: mk_pass_segments,
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
    use std::ffi::{c_int, c_void};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

    use super::{
        Arena, Binding, CtxFree, CtxMax, CtxNew, EngineOpen, LastError, MemGetInfo, PartBytes, PassSegments,
        RecoverSpinTimeouts, RoundGdn, SegmentsProblem, StateBytes, StateRead,
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

    unsafe extern "C" fn ctx_owns_kv(_pointer: *mut c_void) -> c_int {
        1
    }

    unsafe extern "C" fn ctx_poisoned(pointer: *mut c_void) -> c_int {
        c_int::from(unsafe { context(pointer) }.poisoned.load(Ordering::SeqCst))
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
                arena: std::mem::transmute::<unsafe extern "C" fn(), Arena>(never),
                last_error: std::mem::transmute::<unsafe extern "C" fn(), LastError>(never),
                segments_problem: std::mem::transmute::<unsafe extern "C" fn(), SegmentsProblem>(never),
                pass_segments: std::mem::transmute::<unsafe extern "C" fn(), PassSegments>(never),
                mem_get_info: std::mem::transmute::<unsafe extern "C" fn(), MemGetInfo>(never),
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
