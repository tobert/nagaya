//! The C ABI of the Qwen3.8 engine, linked statically from `hip/` (spec 12.1, AGENTS.md milestone 15).
//!
//! `build.rs` compiles the engine into `libqwen38engine.a` under the feature `hip`. A build without `hip` links no engine
//! code, and `linked` returns an error. The engine is never closed: the HIP teardown of the engine segfaults in the process (spec
//! 3.3).
//!
//! The C++ side is the runtime layer: the launches of `hip/src/engine/extern.hip` and the runtime calls `hip_rt_*` of
//! `hip/src/rt/rt.cpp`. Each call checks nothing. The Rust host makes every decision (milestone 15 step 6): the
//! arena (`arena.rs`), the contexts (`context.rs`), the PLE rows (`ple.rs`), the stages (`pass.rs`) and the pass
//! (`host.rs`). `Binding` is a table of the calls, so the worker's stub tests can replace them.

use std::ffi::{c_char, c_int, c_void};

use crate::stage::Stage;

/// One segment of a pass: `t` tokens at `tokens` that continue the context `context` (`context.rs` `Qwen38Context`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Qwen38Segment {
    pub context: *mut c_void,
    pub tokens: *const i32,
    pub t: u32,
}

/// `struct LaunchResult` of `engine.h`: how a launch sequence ended.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Qwen38Launch {
    /// The control block's status (`rt.h` `Status`), 0 when the launch was clean.
    pub status: u32,
    pub site: u32,
    pub wg: u32,
    /// 1 when the watchdog asked the kernel to stop: its deadline or the memory floor.
    pub watchdog_stopped: u32,
}

/// `void *qwen38_engine_open(double watchdog_deadline_s)`.
pub(crate) type EngineOpen = unsafe extern "C" fn(f64) -> *mut c_void;

/// `void qwen38_relayout(uint32_t tile, const uint8_t *src, uint8_t *dst, uint32_t rows, uint32_t k)`.
pub(crate) type Relayout = unsafe extern "C" fn(u32, *const u8, *mut u8, u32, u32);

/// `void hip_rt_host_register(void *p, uint64_t n)`.
pub(crate) type RtHostRegister = unsafe extern "C" fn(*mut c_void, u64);

/// `void hip_rt_host_unregister(void *p)`.
pub(crate) type RtHostUnregister = unsafe extern "C" fn(*mut c_void);

/// `void *hip_rt_host_malloc(uint64_t n)`.
pub(crate) type RtHostMalloc = unsafe extern "C" fn(u64) -> *mut c_void;

/// `void hip_rt_host_free(void *p)`.
pub(crate) type RtHostFree = unsafe extern "C" fn(*mut c_void);

/// `void *hip_rt_malloc(uint64_t n)`.
pub(crate) type RtMalloc = unsafe extern "C" fn(u64) -> *mut c_void;

/// `void hip_rt_memset(void *p, int value, uint64_t n)`.
pub(crate) type RtMemset = unsafe extern "C" fn(*mut c_void, i32, u64);

/// `void hip_rt_memcpy(void *dst, const void *src, uint64_t n, uint32_t kind)`.
pub(crate) type RtMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, u64, u32);

/// `void qwen38_launch_fused(void *e, const Stage *ds, uint32_t n, int grid, uint32_t skip, LaunchResult *out)`.
pub(crate) type LaunchFused = unsafe extern "C" fn(*mut c_void, *const Stage, u32, i32, u32, *mut Qwen38Launch);

/// `void qwen38_launch_each(void *e, const Stage *ds, uint32_t n, LaunchResult *out)`.
pub(crate) type LaunchEach = unsafe extern "C" fn(*mut c_void, *const Stage, u32, *mut Qwen38Launch);

/// `int qwen38_max_coop_grid()`.
pub(crate) type MaxCoopGrid = unsafe extern "C" fn() -> i32;

/// `const char *qwen38_status_name(uint32_t status)`.
pub(crate) type StatusName = unsafe extern "C" fn(u32) -> *const c_char;

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

/// `void qwen38_ctx_round_gdn_bf16(void *e, void *c)`.
pub(crate) type RoundGdn = unsafe extern "C" fn(*mut c_void, *mut c_void);

/// `hipError_t hipMemGetInfo(size_t *free, size_t *total)`, from the HIP runtime that the engine links.
pub(crate) type MemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;

/// The calls that this binding uses.
pub(crate) struct Binding {
    pub engine_open: EngineOpen,
    pub relayout: Relayout,
    pub rt_host_register: RtHostRegister,
    pub rt_host_unregister: RtHostUnregister,
    pub rt_host_malloc: RtHostMalloc,
    pub rt_host_free: RtHostFree,
    pub rt_malloc: RtMalloc,
    pub rt_memset: RtMemset,
    pub rt_memcpy: RtMemcpy,
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
    pub launch_fused: LaunchFused,
    pub launch_each: LaunchEach,
    pub max_coop_grid: MaxCoopGrid,
    pub status_name: StatusName,
    pub mem_get_info: MemGetInfo,
}

/// Why a pass did not finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PassStop {
    /// The binding or the segment check of the engine refused the pass before it ran. Nothing changed.
    Refused(String),
    /// The pass ran and stopped on a spin-timeout. The engine poisoned each of its contexts.
    Stopped(String),
}

/// The segments of a pass: `lens[i]` tokens of `tokens` continue `contexts[i]`, in order.
///
/// A count that differs is a defect of the caller, so it stops the program.
pub(crate) fn segments(contexts: &[*mut c_void], lens: &[u32], tokens: &[i32]) -> Vec<Qwen38Segment> {
    assert_eq!(contexts.len(), lens.len(), "a pass needs one length for each context");
    let total: usize = lens.iter().map(|len| *len as usize).sum();
    assert_eq!(total, tokens.len(), "the lengths of a pass must add to its token count");
    let mut start = 0;
    contexts
        .iter()
        .zip(lens)
        .map(|(context, len)| {
            let segment = Qwen38Segment {
                context: *context,
                tokens: tokens[start..].as_ptr(),
                t: *len,
            };
            start += *len as usize;
            segment
        })
        .collect()
}

/// The facts of the engine build, from `build.rs`. The identity holds them (spec 12.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Build {
    /// The SHA-256 of `libqwen38engine.a`, as 64 lowercase hex digits.
    pub library_sha256: &'static str,
    /// The first line of `clang++ --version`, the compiler of every unit.
    pub compiler: &'static str,
    /// The output of `hipconfig --version`, the HIP version of the build.
    pub hip: &'static str,
}

/// The calls of the engine and the runtime that the Rust host makes itself (`context.rs`).
/// The runtime calls that the Rust context code makes itself (`context.rs`).
#[cfg(feature = "hip")]
pub(crate) mod rt {
    pub(crate) use super::linked::{hip_rt_free, hip_rt_malloc, hip_rt_memcpy, hip_rt_memcpy_2d, hip_rt_memset};
}

#[cfg(feature = "hip")]
mod linked {
    use std::ffi::{c_char, c_int, c_void};

    use super::Qwen38Launch;
    use crate::stage::Stage;

    // The calls of the extern layer `hip/src/engine/extern.hip`, the runtime calls `hip_rt_*` of
    // `hip/src/rt/rt.cpp`, and one call of the HIP runtime. A test compares the lists of calls.
    unsafe extern "C" {
        pub(super) fn qwen38_engine_open(watchdog_deadline_s: f64) -> *mut c_void;
        pub(super) fn qwen38_relayout(tile: u32, src: *const u8, dst: *mut u8, rows: u32, k: u32);
        pub(super) fn qwen38_ctx_round_gdn_bf16(e: *mut c_void, c: *mut c_void);
        pub(super) fn qwen38_launch_fused(
            e: *mut c_void,
            ds: *const Stage,
            n: u32,
            grid: i32,
            skip: u32,
            out: *mut Qwen38Launch,
        );
        pub(super) fn qwen38_launch_each(e: *mut c_void, ds: *const Stage, n: u32, out: *mut Qwen38Launch);
        pub(super) fn qwen38_max_coop_grid() -> i32;
        pub(super) fn qwen38_status_name(status: u32) -> *const c_char;
        pub(super) fn hip_rt_host_register(p: *mut c_void, n: u64);
        pub(super) fn hip_rt_host_unregister(p: *mut c_void);
        pub(super) fn hip_rt_host_malloc(n: u64) -> *mut c_void;
        pub(super) fn hip_rt_host_free(p: *mut c_void);
        pub(crate) fn hip_rt_malloc(n: u64) -> *mut c_void;
        pub(crate) fn hip_rt_free(p: *mut c_void);
        pub(crate) fn hip_rt_memset(p: *mut c_void, value: i32, n: u64);
        pub(crate) fn hip_rt_memcpy(dst: *mut c_void, src: *const c_void, n: u64, kind: u32);
        pub(crate) fn hip_rt_memcpy_2d(
            dst: *mut c_void,
            dpitch: u64,
            src: *const c_void,
            spitch: u64,
            width: u64,
            height: u64,
            kind: u32,
        );
        pub(super) fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    }

    /// The facts that `build.rs` gave.
    pub(super) const BUILD: super::Build = super::Build {
        library_sha256: env!("QWEN38_BUILD_SHA256"),
        compiler: env!("QWEN38_BUILD_COMPILER"),
        hip: env!("QWEN38_BUILD_HIP"),
    };
}

/// Bind every call to the engine that this binary links, and give the facts of its build.
#[cfg(feature = "hip")]
pub(crate) fn linked() -> Result<(Binding, Build), String> {
    use crate::context;
    use linked::*;
    Ok((
        Binding {
            engine_open: qwen38_engine_open,
            relayout: qwen38_relayout,
            rt_host_register: hip_rt_host_register,
            rt_host_unregister: hip_rt_host_unregister,
            rt_host_malloc: hip_rt_host_malloc,
            rt_host_free: hip_rt_host_free,
            rt_malloc: hip_rt_malloc,
            rt_memset: hip_rt_memset,
            rt_memcpy: hip_rt_memcpy,
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
            round_gdn: qwen38_ctx_round_gdn_bf16,
            launch_fused: qwen38_launch_fused,
            launch_each: qwen38_launch_each,
            max_coop_grid: qwen38_max_coop_grid,
            status_name: qwen38_status_name,
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
    "this build links no Qwen3.8 engine; build nagaya-mk-qwen38-gfx1151 with the feature hip";

/// A binding to stub calls and a stub context, for tests that need a poisoned context. The engine poisons a context only
/// on a spin timeout, and no test can force that.
#[cfg(test)]
pub(crate) mod stub {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

    use super::{
        Binding, CtxFree, CtxMax, CtxNew, EngineOpen, LaunchEach, LaunchFused, MaxCoopGrid, MemGetInfo, PartBytes,
        Relayout, RoundGdn, RtHostFree, RtHostMalloc, RtHostRegister, RtHostUnregister, RtMalloc, RtMemcpy, RtMemset,
        StateBytes, StateRead, StatusName,
    };

    /// What the stub calls know about one context. A test passes a pointer to this in place of a context of the engine.
    pub(crate) struct StubContext {
        pub poisoned: AtomicBool,
        pub n: AtomicU32,
        /// The count of calls that would change the context of the engine.
        pub changes: AtomicUsize,
        /// When true, a state write poisons the context, as a failed write of the engine would.
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
                rt_host_unregister: std::mem::transmute::<unsafe extern "C" fn(), RtHostUnregister>(never),
                rt_host_malloc: std::mem::transmute::<unsafe extern "C" fn(), RtHostMalloc>(never),
                rt_host_free: std::mem::transmute::<unsafe extern "C" fn(), RtHostFree>(never),
                rt_malloc: std::mem::transmute::<unsafe extern "C" fn(), RtMalloc>(never),
                rt_memset: std::mem::transmute::<unsafe extern "C" fn(), RtMemset>(never),
                rt_memcpy: std::mem::transmute::<unsafe extern "C" fn(), RtMemcpy>(never),
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
                round_gdn: std::mem::transmute::<unsafe extern "C" fn(), RoundGdn>(never),
                part_bytes: std::mem::transmute::<unsafe extern "C" fn(), PartBytes>(never),
                state_bytes: std::mem::transmute::<unsafe extern "C" fn(), StateBytes>(never),
                launch_fused: std::mem::transmute::<unsafe extern "C" fn(), LaunchFused>(never),
                launch_each: std::mem::transmute::<unsafe extern "C" fn(), LaunchEach>(never),
                max_coop_grid: std::mem::transmute::<unsafe extern "C" fn(), MaxCoopGrid>(never),
                status_name: std::mem::transmute::<unsafe extern "C" fn(), StatusName>(never),
                mem_get_info: std::mem::transmute::<unsafe extern "C" fn(), MemGetInfo>(never),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::path::Path;

    use super::segments;

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
                let start = line.find("qwen38_").or_else(|| line.find("hip_rt_"))?;
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
    fn the_extern_layer_and_the_binding_have_the_same_calls() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let layer = std::fs::read_to_string(root.join("hip/src/engine/extern.hip")).expect("read extern.hip");
        let runtime = std::fs::read_to_string(root.join("hip/src/rt/rt.cpp")).expect("read rt.cpp");
        let binding = std::fs::read_to_string(root.join("src/binding.rs")).expect("read binding.rs");
        let mut layer = calls(&layer, false);
        // rt.cpp also holds the trap calls of the op tests. The host's runtime calls are the ones named hip_rt_.
        layer.extend(
            calls(&runtime, false)
                .into_iter()
                .filter(|name| name.starts_with("hip_rt_")),
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
