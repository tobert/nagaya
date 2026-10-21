//! The evidence on failure (spec 9.6): the machine's counters before and after a quantum that fails.

use std::path::Path;

use nagaya::evidence::{Counters, EvidenceError, Machine};
use nagaya_telemetry::testing::log_lines as logged;

const PID: u32 = 4242;
const GIB: u64 = 1 << 30;

/// Write a fixed tree of the files that the machine reads, under `root`.
fn tree(root: &Path, kfd_ms: Option<u64>, available_kib: u64, gtt: Option<u64>, vmstat: &[(&str, u64)]) {
    let write = |path: &str, text: String| {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write(
        "proc/meminfo",
        format!(
            "MemTotal:       99999999 kB\nMemFree:        1 kB\nMemAvailable:   {available_kib} kB\nBuffers: 5 kB\n"
        ),
    );
    let mut text = String::from("nr_free_pages 1\n");
    for (name, value) in vmstat {
        text.push_str(&format!("{name} {value}\n"));
    }
    text.push_str("other 11\n");
    write("proc/vmstat", text);
    if let Some(ms) = kfd_ms {
        write(
            &format!("sys/class/kfd/kfd/proc/{PID}/stats_0/evicted_ms"),
            format!("{ms}\n"),
        );
    }
    if let Some(bytes) = gtt {
        write("sys/class/drm/card1/device/mem_info_gtt_used", format!("{bytes}\n"));
    }
}

fn full_vmstat(stall: u64, migrate: u64, swap: u64, scan: u64) -> Vec<(&'static str, u64)> {
    vec![
        ("compact_stall", stall),
        ("pgmigrate_success", migrate),
        ("pswpout", swap),
        ("pgscan_direct", scan),
    ]
}

fn machine(root: &Path) -> Machine {
    Machine::at(root, PID)
}

#[test]
fn read_gives_the_counters_of_spec_9_6() {
    let root = tempfile::tempdir().unwrap();
    tree(
        root.path(),
        Some(1000),
        46 * 1024 * 1024,
        Some(9 * GIB),
        &full_vmstat(1, 2, 3, 4),
    );
    let counters = machine(root.path()).read().unwrap();
    assert_eq!(counters.kfd_evicted_ms, Some(1000));
    assert_eq!(counters.mem_available_bytes, 46 * GIB);
    assert_eq!(counters.gtt_used_bytes, Some(9 * GIB));
    assert_eq!(counters.compact_stall, 1);
    assert_eq!(counters.pgmigrate_success, 2);
    assert_eq!(counters.pswpout, 3);
    assert_eq!(counters.pgscan_direct, 4);
}

#[test]
fn a_process_with_no_gpu_queues_has_no_kfd_counter() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), None, 1024, Some(0), &full_vmstat(0, 0, 0, 0));
    assert_eq!(machine(root.path()).read().unwrap().kfd_evicted_ms, None);
}

#[test]
fn a_machine_with_no_gtt_file_has_no_gtt_counter() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), None, 1024, None, &full_vmstat(0, 0, 0, 0));
    assert_eq!(machine(root.path()).read().unwrap().gtt_used_bytes, None);
}

#[test]
fn the_kfd_counter_sums_the_gpus_of_the_process() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), Some(10), 1024, None, &full_vmstat(0, 0, 0, 0));
    let second = root.path().join(format!("sys/class/kfd/kfd/proc/{PID}/stats_1"));
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(second.join("evicted_ms"), "5\n").unwrap();
    assert_eq!(machine(root.path()).read().unwrap().kfd_evicted_ms, Some(15));
}

#[test]
fn a_missing_vmstat_counter_stops_with_its_name() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), None, 1024, None, &[("compact_stall", 1), ("pswpout", 3)]);
    let error = machine(root.path()).read().unwrap_err();
    assert!(
        matches!(&error, EvidenceError::Missing { what, .. } if what.contains("pgmigrate_success") && what.contains("pgscan_direct")),
        "{error}"
    );
}

#[test]
fn a_meminfo_with_no_mem_available_stops_with_its_name() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), None, 1024, None, &full_vmstat(0, 0, 0, 0));
    std::fs::write(root.path().join("proc/meminfo"), "MemTotal: 5 kB\n").unwrap();
    let error = machine(root.path()).read().unwrap_err();
    assert!(error.to_string().contains("MemAvailable"), "{error}");
}

#[test]
fn a_file_that_does_not_parse_names_the_file() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), Some(1), 1024, None, &full_vmstat(0, 0, 0, 0));
    std::fs::write(
        root.path()
            .join(format!("sys/class/kfd/kfd/proc/{PID}/stats_0/evicted_ms")),
        "many\n",
    )
    .unwrap();
    let error = machine(root.path()).read().unwrap_err();
    assert!(error.to_string().contains("evicted_ms"), "{error}");
}

fn counters(root: &Path, kfd: Option<u64>, available_gib: u64, gtt_gib: u64, vm: [u64; 4]) -> Counters {
    tree(
        root,
        kfd,
        available_gib * 1024 * 1024,
        Some(gtt_gib * GIB),
        &full_vmstat(vm[0], vm[1], vm[2], vm[3]),
    );
    machine(root).read().unwrap()
}

#[test]
fn the_change_names_each_counter_with_its_movement() {
    let root = tempfile::tempdir().unwrap();
    let before = counters(
        root.path(),
        Some(1000),
        47,
        10,
        [567_581, 910_558_411, 13_999_534, 1_374_303_607],
    );
    let after = counters(
        root.path(),
        Some(1413),
        39,
        17,
        [567_590, 910_558_411 + 52_000, 13_999_534, 1_374_303_607 + 7],
    );
    let text = after.change_since(&before).unwrap().to_string();
    assert_eq!(
        text,
        "KFD evicted 413 ms; MemAvailable 47.0 -> 39.0 GiB; GTT 10.0 -> 17.0 GiB; \
         compact_stall +9; pgmigrate_success +52000; pswpout +0; pgscan_direct +7"
    );
}

#[test]
fn the_change_says_so_when_the_process_has_no_gpu_queues() {
    let root = tempfile::tempdir().unwrap();
    let before = counters(root.path(), None, 47, 10, [0, 0, 0, 0]);
    let after = counters(root.path(), None, 47, 10, [0, 0, 0, 0]);
    let text = after.change_since(&before).unwrap().to_string();
    assert!(text.starts_with("KFD evicted n/a; "), "{text}");
}

#[test]
fn a_counter_that_goes_back_is_an_error() {
    let root = tempfile::tempdir().unwrap();
    let before = counters(root.path(), Some(1000), 47, 10, [5, 5, 5, 5]);
    let after = counters(root.path(), Some(900), 47, 10, [5, 5, 5, 5]);
    let error = after.change_since(&before).unwrap_err();
    assert!(
        matches!(&error, EvidenceError::WentBack { counter, .. } if counter == "KFD evicted_ms"),
        "{error}"
    );
    let after = counters(root.path(), Some(1000), 47, 10, [5, 4, 5, 5]);
    let error = after.change_since(&before).unwrap_err();
    assert!(error.to_string().contains("pgmigrate_success"), "{error}");
}

#[test]
fn a_failed_quantum_logs_the_failure_and_the_change() {
    let root = tempfile::tempdir().unwrap();
    let machine = machine(root.path());
    tree(
        root.path(),
        Some(1000),
        47 * 1024 * 1024,
        Some(10 * GIB),
        &full_vmstat(1, 1, 1, 1),
    );
    let watch = machine.watch();
    tree(
        root.path(),
        Some(1413),
        39 * 1024 * 1024,
        Some(17 * GIB),
        &full_vmstat(10, 1, 1, 8),
    );
    let mut line = String::new();
    let lines = logged(|| line = watch.failed("engine qwen: spin-timeout at stage 9"));

    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines[0].starts_with("quantum failed: engine qwen: spin-timeout at stage 9; "),
        "{}",
        lines[0]
    );
    assert!(lines[0].contains("KFD evicted 413 ms"), "{}", lines[0]);
    assert!(lines[0].contains("compact_stall +9"), "{}", lines[0]);
    assert!(lines[0].ends_with("level=error"), "{}", lines[0]);
    assert!(line.contains("MemAvailable 47.0 -> 39.0 GiB"), "{line}");
}

#[test]
fn evidence_that_cannot_be_read_never_hides_the_failure() {
    let root = tempfile::tempdir().unwrap();
    // The tree is empty, so the read before the quantum fails.
    let machine = machine(root.path());
    let watch = machine.watch();
    let mut line = String::new();
    let lines = logged(|| line = watch.failed("engine qwen: spin-timeout"));

    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines[0].starts_with("quantum failed: engine qwen: spin-timeout; evidence unavailable ("),
        "{}",
        lines[0]
    );
    assert!(line.contains("evidence unavailable"), "{line}");
}

#[test]
fn a_failure_in_the_read_after_the_quantum_is_named_too() {
    let root = tempfile::tempdir().unwrap();
    let machine = machine(root.path());
    tree(root.path(), Some(1000), 1024, None, &full_vmstat(1, 1, 1, 1));
    let watch = machine.watch();
    std::fs::remove_file(root.path().join("proc/vmstat")).unwrap();
    let line = watch.failed("engine qwen: spin-timeout");
    assert!(
        line.contains("evidence unavailable") && line.contains("vmstat"),
        "{line}"
    );
}

#[test]
fn the_real_machine_reads() {
    // The fixed trees above copy the formats of /proc. This test reads the real files of a Linux machine.
    let first = Machine::real().read().expect("a Linux machine has the files");
    assert!(first.mem_available_bytes > 0);
    let second = Machine::real().read().unwrap();
    let change = second.change_since(&first).expect("the counters only grow");
    assert!(change.to_string().contains("MemAvailable"), "{change}");
}

#[test]
fn the_gtt_counter_sums_the_devices() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), None, 1024, Some(3 * GIB), &full_vmstat(0, 0, 0, 0));
    let second = root.path().join("sys/class/drm/card2/device");
    std::fs::create_dir_all(&second).unwrap();
    std::fs::write(second.join("mem_info_gtt_used"), format!("{}\n", 4 * GIB)).unwrap();
    // A connector entry has no such file. It must not stop the read.
    std::fs::create_dir_all(root.path().join("sys/class/drm/card1-DP-1")).unwrap();
    assert_eq!(machine(root.path()).read().unwrap().gtt_used_bytes, Some(7 * GIB));
}
