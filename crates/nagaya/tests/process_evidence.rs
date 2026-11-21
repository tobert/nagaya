//! A pass that fails on the group thread logs the change of the machine's counters (spec 9.6). The engine is the fake
//! engine. The job is a body that runs a pass that the fake refuses, and the machine has a fixed tree of files.

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::evidence::Machine;
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::Start;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, Kind, Limits, OutputsWanted, Segment, SegmentInputs, SlotLimit, State, Tier};
use nagaya_sched::Class;
use nagaya_telemetry::testing::log_lines;

const PID: u32 = 7;

fn fake() -> FakeAlways {
    FakeAlways::with_slots(vec![SlotLimit {
        bytes: 256,
        count: 1,
        tokens: 4096,
    }])
}

struct Simple;

impl Factory for Simple {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(fake().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(fake()))
    }
}

fn profile() -> PlacementProfile {
    PlacementProfile {
        engines: vec![EnginePlacement {
            name: "one".to_owned(),
            kind: "simple".to_owned(),
            devices: vec![0],
            config: toml::Table::new(),
            family: toml::Table::new(),
            lab: toml::Table::new(),
            share: 1,
            residency: Default::default(),
            pinned: false,
        }],
        budgets: vec![
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 4096,
            },
            BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Host)],
                bytes: 4096,
            },
        ],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

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

/// A body that runs a pass the fake refuses, because it wants a logit row that no input holds.
struct BadPass {
    root: std::path::PathBuf,
    done: mpsc::Sender<()>,
    /// What the body reports after the pass.
    outcome: Ran,
}

impl JobBody for BadPass {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let segment = Segment {
            control: Vec::new(),
            state: State::Slot(0),
            inputs: SegmentInputs::Tokens(vec![1]),
            positions: vec![0],
            meta: Vec::new(),
            outputs: OutputsWanted { logit_rows: vec![5] },
            layer_loop: None,
            cycles: None,
        };
        let error = engines[0].pass(&[segment], &[]).expect_err("the fake refuses the pass");
        assert!(error.message.contains("row"), "{error:?}");
        // The kernel counted five compactions stalls while the pass ran.
        tree(&self.root, 15);
        let _ = self.done.send(());
        self.outcome.clone()
    }
}

fn failure_lines(root: &Path, outcome: Ran) -> Vec<String> {
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("simple", Box::new(Simple));
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1024,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1024,
    };
    let (done_tx, done_rx) = mpsc::channel();
    let lines = log_lines(|| {
        let mut process = Start::new(profile(), registry, inventory)
            .store(directory.path())
            .heartbeat(Duration::from_millis(10))
            .timeout(Duration::from_secs(5))
            .machine(Machine::at(root, PID))
            .start()
            .unwrap();
        process.wait_opens(Duration::from_secs(5)).unwrap();
        process
            .submit_job(
                0,
                JobSpec {
                    class: Class::Interactive,
                    priority: 0,
                    deadline: None,
                    first: Kind::Prefill,
                    engine: 0,
                    body: Box::new(BadPass {
                        root: root.to_owned(),
                        done: done_tx,
                        outcome,
                    }),
                },
            )
            .unwrap();
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        // The stop joins the group thread, so the line is written when it returns.
        process.stop(Duration::from_secs(5), &SignalFlag::new());
    });
    lines
        .into_iter()
        .filter(|line| line.starts_with("quantum failed"))
        .collect()
}

#[test]
fn a_fake_pass_that_fails_logs_the_change_of_the_counters() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), 10);
    let lines = failure_lines(root.path(), Ran::Failed);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines[0].contains("prefill quantum of a job request on engine one"),
        "{}",
        lines[0]
    );
    assert!(lines[0].contains("compact_stall +5"), "{}", lines[0]);
}

#[test]
fn a_machine_without_the_files_still_logs_the_failure() {
    let root = tempfile::tempdir().unwrap();
    // The body writes the tree during the pass, so the read before the quantum finds no file.
    let lines = failure_lines(root.path(), Ran::Failed);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].contains("evidence unavailable"), "{}", lines[0]);
}

/// A refused job is not a failed quantum (spec 9.6): the process logs no failure line for it. The live service of
/// 2026-11-15 still logged the line after the first repair, because this driver had its own check.
#[test]
fn a_refused_job_logs_no_failure_in_the_process() {
    let root = tempfile::tempdir().unwrap();
    tree(root.path(), 10);
    let lines = failure_lines(root.path(), Ran::Refused);
    assert!(lines.is_empty(), "{lines:?}");
}
