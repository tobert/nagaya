//! The C ABI of `libmkengine.so`, loaded at run time.
//!
//! The library is loaded from `MK_ENGINE_LIB` and is never closed. mk's HIP teardown segfaults in the process (spec
//! 3.3), so the engine stays open until the process exits. The loaded library is leaked for the same reason.

use std::ffi::{c_char, c_int, c_void};
use std::path::Path;

use libloading::Library;

/// `void *mk_engine_open(const char *s1, const char *s2, const char *s3)`.
pub(crate) type EngineOpen = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> *mut c_void;

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

/// `hipError_t hipMemGetInfo(size_t *free, size_t *total)`. HIP is a dependency of the mk library, so the loader
/// finds it through the mk library handle.
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
    #[allow(dead_code)]
    library: &'static Library,
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

/// Load the library and bind every call.
pub(crate) fn load(path: &Path) -> Result<Binding, String> {
    if !path.is_file() {
        return Err(format!("the mk engine library {} is missing", path.display()));
    }
    let library: &'static Library = Box::leak(Box::new(
        unsafe { Library::new(path) }.map_err(|error| format!("load {}: {error}", path.display()))?,
    ));
    // Each symbol lives as long as the leaked library.
    unsafe {
        Ok(Binding {
            library,
            engine_open: symbol(library, b"mk_engine_open\0")?,
            ctx_new: symbol(library, b"mk_ctx_new\0")?,
            ctx_free: symbol(library, b"mk_ctx_free\0")?,
            ctx_n: symbol(library, b"mk_ctx_n\0")?,
            ctx_max: symbol(library, b"mk_ctx_max\0")?,
            ctx_owns_kv: symbol(library, b"mk_ctx_owns_kv\0")?,
            ctx_poisoned: symbol(library, b"mk_ctx_poisoned\0")?,
            state_read: symbol(library, b"mk_ctx_state_read\0")?,
            state_write: symbol(library, b"mk_ctx_state_write\0")?,
            cells_write: symbol(library, b"mk_ctx_cells_write\0")?,
            set_n: symbol(library, b"mk_ctx_set_n\0")?,
            recover_spin_timeouts: symbol(library, b"mk_engine_recover_spin_timeouts\0")?,
            round_gdn: symbol(library, b"mk_ctx_round_gdn_bf16\0")?,
            part_bytes: symbol(library, b"mk_ctx_part_bytes\0")?,
            state_bytes: symbol(library, b"mk_ctx_state_bytes\0")?,
            arena: symbol(library, b"mk_engine_arena\0")?,
            last_error: symbol(library, b"mk_last_error\0")?,
            segments_problem: symbol(library, b"mk_segments_problem\0")?,
            pass_segments: symbol(library, b"mk_pass_segments\0")?,
            mem_get_info: symbol(library, b"hipMemGetInfo\0")?,
        })
    }
}

/// Copy one function pointer out of the library.
unsafe fn symbol<T: Copy>(library: &'static Library, name: &[u8]) -> Result<T, String> {
    let symbol = unsafe { library.get::<T>(name) }
        .map_err(|error| format!("bind {}: {error}", String::from_utf8_lossy(&name[..name.len() - 1])))?;
    Ok(*symbol)
}

/// A binding to stub calls and a stub context, for tests that need a poisoned context. mk poisons a context only
/// on a spin timeout, and no test can force that.
#[cfg(test)]
pub(crate) mod stub {
    use std::ffi::{c_int, c_void};
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

    use libloading::Library;

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
        let library: &'static Library = Box::leak(Box::new(Library::from(libloading::os::unix::Library::this())));
        let never: unsafe extern "C" fn() = never;
        // The transmutes below only name the call types. The stub never calls them, and `never` takes no argument.
        unsafe {
            Binding {
                library,
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
    use super::load;

    #[test]
    fn a_missing_library_is_an_error() {
        let result = load(std::path::Path::new("/nonexistent/libmkengine.so"));
        assert!(result.is_err(), "a missing library loaded");
        let error = result.err().unwrap_or_default();
        assert!(error.contains("is missing"), "{error}");
    }
}
