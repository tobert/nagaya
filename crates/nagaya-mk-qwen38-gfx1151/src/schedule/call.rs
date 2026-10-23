//! The two calls of the schedule that mk's engine makes through the tensor table (`hip/src/engine/tensor_table.h`).
//!
//! The engine builds the stages of a pass in C++, and the Rust schedule orders them. An error of the schedule is a
//! defect of the pass build, so each call stops the program with the message. A panic cannot unwind out of an
//! `extern "C"` call: it stops the program after it prints the message.

use super::{Plan, plain, plan, stage_footprint};
use crate::stage::Stage;

/// `struct mk_access` of `tensor_table.h`: one access of a footprint.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MkAccess {
    pub lo: u64,
    pub hi: u64,
    pub write: u32,
}

/// A slice of `n` values at `pointer`. A null pointer is the empty slice only when `n` is 0.
///
/// # Safety
///
/// When `n` is not 0, `pointer` names `n` readable values that stay valid for the call.
unsafe fn slice<'a, T>(pointer: *const T, n: u32, what: &str) -> &'a [T] {
    if n == 0 {
        return &[];
    }
    assert!(!pointer.is_null(), "mk's engine gave {n} {what} at a null pointer");
    // SAFETY: the caller's contract.
    unsafe { std::slice::from_raw_parts(pointer, n as usize) }
}

/// The plan of a pass: the schedule with `elide`, else each stage in build order with a barrier after it.
pub(crate) fn pass_plan(stages: &[Stage], window_starts: &[u32], elide: bool, tail: bool) -> Plan {
    if !elide {
        return plain(stages.len() as u32);
    }
    plan(stages, window_starts, tail).unwrap_or_else(|error| panic!("mk's pass build: {error}"))
}

/// The engine's `schedule`: the plan of the `n` stages at `stages`. It writes the build index of the k-th stage to run
/// to `order[k]`, and its `Stage::sched` to `sched[k]`. It gives the grid barriers of the pass.
///
/// # Safety
///
/// `stages` names `n` stages, `window_starts` names `n_windows` values, and `order` and `sched` name `n` writable
/// values each. Each one stays valid for the call.
pub(crate) unsafe extern "C" fn schedule_call(
    stages: *const Stage,
    n: u32,
    window_starts: *const u32,
    n_windows: u32,
    elide: u32,
    tail: u32,
    order: *mut u32,
    sched: *mut u32,
) -> u32 {
    // SAFETY: the caller's contract.
    let (stages, window_starts) = unsafe { (slice(stages, n, "stages"), slice(window_starts, n_windows, "windows")) };
    let plan = pass_plan(stages, window_starts, elide != 0, tail != 0);
    if n != 0 {
        assert!(
            !order.is_null() && !sched.is_null(),
            "mk's engine gave no room for the schedule"
        );
        // SAFETY: the caller's contract: n writable values at each pointer.
        let (order, sched) = unsafe {
            (
                std::slice::from_raw_parts_mut(order, n as usize),
                std::slice::from_raw_parts_mut(sched, n as usize),
            )
        };
        order.copy_from_slice(&plan.order);
        sched.copy_from_slice(&plan.sched);
    }
    plan.barriers
}

/// The engine's `footprint`: the footprint of the stage at `stage`, as `stage_footprint` with no K/V reach. It writes
/// at most `cap` accesses to `out` and gives their count, and it writes the fence flag to `fence`.
///
/// # Safety
///
/// `stage` names one stage, `out` names `cap` writable accesses, and `fence` names one writable value.
pub(crate) unsafe extern "C" fn footprint_call(
    stage: *const Stage,
    out: *mut MkAccess,
    cap: u32,
    fence: *mut u32,
) -> u32 {
    assert!(
        !stage.is_null() && !fence.is_null(),
        "mk's engine asked for a footprint with a null pointer"
    );
    // SAFETY: the caller's contract.
    let stage = unsafe { &*stage };
    let footprint = stage_footprint(stage, 0).unwrap_or_else(|error| panic!("mk's pass build: {error}"));
    let count = footprint.acc.len();
    assert!(
        count <= cap as usize,
        "a footprint of op {} has {count} accesses, and mk's engine has room for {cap}",
        stage.op
    );
    if count != 0 {
        assert!(!out.is_null(), "mk's engine gave no room for the footprint");
        // SAFETY: the caller's contract: cap writable accesses at `out`, and count <= cap.
        let out = unsafe { std::slice::from_raw_parts_mut(out, count) };
        for (slot, access) in out.iter_mut().zip(&footprint.acc) {
            *slot = MkAccess {
                lo: access.lo,
                hi: access.hi,
                write: u32::from(access.write),
            };
        }
    }
    // SAFETY: the caller's contract.
    unsafe { *fence = u32::from(footprint.fence) };
    count as u32
}
