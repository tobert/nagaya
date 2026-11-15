//! A quantum that fails logs the change of the machine's counters (spec 9.6). The group thread calls
//! `Machine::watch` before each quantum that is not decode, and `Watch::failed` when the quantum fails.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nagaya::evidence::Machine;
use nagaya::jobs::{GroupJobs, JobBody, JobSpec, Ran};
use nagaya_abi::{Engine, Kind};
use nagaya_sched::Class;
use nagaya_telemetry::testing::log_lines;

const PID: u32 = 4242;

fn tree(root: &Path, stall: u64) {
    let write = |path: &str, text: String| {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write("proc/meminfo", "MemAvailable:   1048576 kB\n".to_owned());
    write(
        "proc/vmstat",
        format!("compact_stall {stall}\npgmigrate_success 0\npswpout 0\npgscan_direct 0\n"),
    );
}

/// A body that moves a counter of the machine during its quantum, and then fails or succeeds.
struct Moves {
    root: PathBuf,
    stall_after: u64,
    outcome: fn() -> Ran,
    runs: Arc<AtomicUsize>,
}

impl JobBody for Moves {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Ran {
        self.runs.fetch_add(1, Ordering::SeqCst);
        tree(&self.root, self.stall_after);
        (self.outcome)()
    }
}

fn spec(first: Kind, body: Moves) -> JobSpec {
    JobSpec {
        class: Class::Interactive,
        priority: 0,
        deadline: None,
        first,
        engine: 0,
        body: Box::new(body),
    }
}

fn run_all(jobs: &mut GroupJobs) {
    let mut guard = 0;
    while !jobs.idle() {
        jobs.run_one(&mut []);
        guard += 1;
        assert!(guard < 20, "the jobs never ended");
    }
}

fn failure_lines(root: &Path, first: Kind, outcome: fn() -> Ran, files: bool) -> Vec<String> {
    if files {
        tree(root, 10);
    }
    let runs = Arc::new(AtomicUsize::new(0));
    let body = Moves {
        root: root.to_owned(),
        stall_after: 15,
        outcome,
        runs: Arc::clone(&runs),
    };
    let lines = log_lines(|| {
        let mut jobs = GroupJobs::new(0, ["toy"]).machine(Machine::at(root, PID));
        jobs.add(spec(first, body));
        run_all(&mut jobs);
    });
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    lines
        .into_iter()
        .filter(|line| line.starts_with("quantum failed") || line.starts_with("job refused"))
        .collect()
}

#[test]
fn a_failed_quantum_logs_the_change_of_the_counters() {
    let root = tempfile::tempdir().unwrap();
    let lines = failure_lines(root.path(), Kind::Prefill, || Ran::Failed, true);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("prefill quantum"), "{}", lines[0]);
    assert!(lines[0].contains("engine toy"), "{}", lines[0]);
    assert!(lines[0].contains("compact_stall +5"), "{}", lines[0]);
    assert!(lines[0].ends_with("level=error"), "{}", lines[0]);
}

#[test]
fn a_machine_without_the_files_logs_the_failure_and_says_that_evidence_is_unavailable() {
    let root = tempfile::tempdir().unwrap();
    let lines = failure_lines(root.path(), Kind::Prefill, || Ran::Failed, false);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("evidence unavailable"), "{}", lines[0]);
}

#[test]
fn a_quantum_that_succeeds_logs_no_failure() {
    let root = tempfile::tempdir().unwrap();
    let lines = failure_lines(root.path(), Kind::Prefill, || Ran::Done, true);
    assert!(lines.is_empty(), "{lines:?}");
}

/// A job that finds no free slot is refused (spec 9.6, "A refused job is not a failed quantum"). The job ends, and the
/// log holds no failure line. It holds one line at the level `info`.
#[test]
fn a_refused_job_ends_and_logs_one_info_line_and_no_failure() {
    let root = tempfile::tempdir().unwrap();
    let lines = failure_lines(root.path(), Kind::Copy, || Ran::Refused, true);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("job refused: copy quantum"), "{}", lines[0]);
    assert!(
        lines[0].contains("engine toy") && !lines[0].ends_with("level=error"),
        "{}",
        lines[0]
    );
}

#[test]
fn a_decode_quantum_reads_no_counters_and_a_failure_of_it_says_so() {
    let root = tempfile::tempdir().unwrap();
    let lines = failure_lines(root.path(), Kind::Decode, || Ran::Failed, true);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("decode quantum"), "{}", lines[0]);
    assert!(
        lines[0].contains("evidence unavailable (not read before a decode quantum)"),
        "{}",
        lines[0]
    );
    assert!(!lines[0].contains("compact_stall"), "{}", lines[0]);
}
