//! Show that every thread of the engine takes the CPU set of the thread that opens it (spec 3.2a, milestone 15 step 6).
//!
//! The nagaya pins an engine's group thread to its CPU set before it opens the engine. A thread that the HIP runtime
//! started before `main` would keep the whole mask. The example pins the main thread to two CPUs, opens the engine
//! and runs the probe passes of `open`, and then reads the mask of each thread of the process with
//! `sched_getaffinity`. It exits with 1 when a thread has another mask. The example never closes the engine (spec 3.3).

use std::mem::size_of;

/// The CPUs that the thread with this id may run on.
fn mask_of(tid: i32) -> Vec<usize> {
    // SAFETY: an all-zero `cpu_set_t` is an empty mask, and the call fills it.
    let mut mask: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::sched_getaffinity(tid, size_of::<libc::cpu_set_t>(), &mut mask) };
    assert_eq!(result, 0, "sched_getaffinity failed for thread {tid}");
    (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect()
}

fn main() {
    let pinned = [2usize, 3];
    // SAFETY: an all-zero `cpu_set_t` is an empty mask.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for cpu in pinned {
        unsafe { libc::CPU_SET(cpu, &mut set) };
    }
    let result = unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) };
    assert_eq!(result, 0, "sched_setaffinity failed");
    let _engine = nagaya_mk_qwen38_gfx1151::open().expect("open the engine");
    let mut bad = 0;
    let mut threads = 0;
    for entry in std::fs::read_dir("/proc/self/task").expect("read /proc/self/task") {
        let entry = entry.expect("a task entry");
        let tid: i32 = entry
            .file_name()
            .to_str()
            .expect("a task id")
            .parse()
            .expect("a task id");
        let name = std::fs::read_to_string(entry.path().join("comm")).unwrap_or_default();
        let mask = mask_of(tid);
        threads += 1;
        let same = mask == pinned;
        if !same {
            bad += 1;
        }
        println!(
            "thread {tid} {:<16} mask {mask:?}{}",
            name.trim(),
            if same { "" } else { "  OTHER MASK" }
        );
    }
    println!("{threads} threads, {bad} with another mask");
    std::process::exit(i32::from(bad > 0));
}
