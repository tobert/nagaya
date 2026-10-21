//! The thread calls of spec 3.2a: pin the calling thread to a CPU mask and set its scheduling policy.
//!
//! Each function acts on the calling thread. Linux takes pid 0 as the calling thread (`sched_setaffinity(2)`).

use std::io;
use std::mem::size_of;

/// The CPU number that a mask can hold. It is the size of `cpu_set_t` in bits.
pub(crate) const MASK_CPUS: usize = libc::CPU_SETSIZE as usize;

/// The calling thread. Linux takes pid 0 for it.
const THIS_THREAD: libc::pid_t = 0;

/// Pin the calling thread to the CPUs in `cpus`. The list must not be empty.
pub fn pin_current_thread(cpus: &[u32]) -> io::Result<()> {
    if cpus.is_empty() {
        return Err(invalid("A CPU mask needs at least one CPU."));
    }
    // SAFETY: an all-zero `cpu_set_t` is an empty mask.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for &cpu in cpus {
        if cpu as usize >= MASK_CPUS {
            return Err(invalid(format!(
                "CPU {cpu} is past the {MASK_CPUS} CPUs that one mask holds. Remove it from the list."
            )));
        }
        // SAFETY: the CPU number is below the mask size, which the check above shows.
        unsafe { libc::CPU_SET(cpu as usize, &mut mask) };
    }
    // SAFETY: `mask` is a valid `cpu_set_t`, and its size is the size that the call reads.
    let result = unsafe { libc::sched_setaffinity(THIS_THREAD, size_of::<libc::cpu_set_t>(), &mask) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// The CPUs that the calling thread may run on, in increasing order.
pub fn current_affinity() -> io::Result<Vec<u32>> {
    // SAFETY: an all-zero `cpu_set_t` is an empty mask, and the call fills it.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    // SAFETY: `mask` is a valid, writable `cpu_set_t` of the size that the call is given.
    let result = unsafe { libc::sched_getaffinity(THIS_THREAD, size_of::<libc::cpu_set_t>(), &mut mask) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: each index is below the mask size, so `CPU_ISSET` reads inside `mask`.
    Ok((0..MASK_CPUS)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .map(|cpu| cpu as u32)
        .collect())
}

/// Set the calling thread to `SCHED_BATCH` (`sched_setscheduler(2)`). The CPU worker pool uses this (spec 3.2).
pub fn set_batch_policy() -> io::Result<()> {
    let param = libc::sched_param { sched_priority: 0 };
    // SAFETY: `param` is a valid `sched_param`. `SCHED_BATCH` takes priority 0.
    let result = unsafe { libc::sched_setscheduler(THIS_THREAD, libc::SCHED_BATCH, &param) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// The scheduling policy of the calling thread, as `sched_getscheduler(2)` gives it.
pub fn current_policy() -> io::Result<i32> {
    // SAFETY: the call reads no memory that Rust owns.
    let policy = unsafe { libc::sched_getscheduler(THIS_THREAD) };
    if policy < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(policy)
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
