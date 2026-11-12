//! Decode jobs of a batching engine on a device thread share passes (milestone 27 step 5, engine-abi "LFM3", "Batched
//! passes"). `chat` with stages spans both cards, so the cards are one group with a device thread for each card, as on
//! b70host. `batcher` on card 0 has a feed rule and 8 segments, so its decode quanta join on the thread of card 0. A
//! second test gives `batcher` `segments_exact` and no feed rule, as the Qwen3.8 engine has (spec 5.7, "The first joins
//! of an engine with `segments_exact`"); its decode quanta join the same way.

mod feed_support;

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::process::Start;
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Dtype, Engine, FeedRule, Identity, Kind, Limits, Output, OutputsWanted, PassError, PassOut, PassStatus,
    Segment, SegmentInputs, Stages, State, Tap, Tier,
};
use nagaya_sched::{Class, Item};

/// The segment count of each pass of `batcher`, with the name of the thread that ran it.
type Passes = Arc<Mutex<Vec<(usize, String)>>>;

/// An engine with 8 segments, and a feed rule or `segments_exact`. For each logit row it gives one output `logits`:
/// the slot and the position.
struct Batcher {
    fake: FakeAlways,
    passes: Passes,
    exact: bool,
}

/// The limits of `batcher`: `segments_exact` and no feed rule when `exact` is true, else a feed rule.
fn batcher_limits(exact: bool) -> Limits {
    let mut limits = with_close(FakeAlways::new().limits());
    limits.feed = (!exact).then(|| FeedRule {
        block: u64::from(limits.max_pass_inputs),
    });
    limits.segments_exact = exact;
    limits.max_segments = 8;
    limits
}

impl Engine for Batcher {
    fn identity(&self) -> Identity {
        self.fake.identity()
    }

    fn limits(&self) -> Limits {
        batcher_limits(self.exact)
    }

    fn pass(&mut self, segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
        self.passes.lock().unwrap().push((segments.len(), thread));
        let mut outputs = Vec::new();
        for segment in segments {
            let State::Slot(slot) = segment.state;
            for position in &segment.outputs.logit_rows {
                let bytes = [slot as f32, *position as f32]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                outputs.push(Output {
                    name: "logits".to_owned(),
                    dtype: Dtype::F32,
                    bytes,
                });
            }
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: 1,
            outputs,
        })
    }

    fn last_error(&self) -> Option<String> {
        None
    }
}

/// `chat` with `close`, which an engine on demand needs (spec 3.3a). It forwards the rest.
struct Closing(Box<dyn Engine + Send>);

fn with_close(mut limits: Limits) -> Limits {
    limits.capabilities.push(Capability::Close);
    limits
}

impl Engine for Closing {
    fn identity(&self) -> Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        with_close(self.0.limits())
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        self.0.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.0.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        self.0.coverage()
    }

    fn stages(&mut self) -> Option<&mut dyn Stages> {
        self.0.stages()
    }
}

/// The factory of both engines, with the passes of `batcher` and whether it declares `segments_exact`.
struct Fakes(Passes, bool);

impl Factory for Fakes {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(match config["name"].as_str().unwrap() {
            "chat" => with_close(feed_support::engine(2).limits()),
            _ => batcher_limits(self.1),
        })
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(match config["name"].as_str().unwrap() {
            "chat" => Box::new(Closing(Box::new(feed_support::engine(2)))),
            _ => Box::new(Batcher {
                fake: FakeAlways::new(),
                passes: Arc::clone(&self.0),
                exact: self.1,
            }),
        })
    }
}

fn engine(name: &str, devices: Vec<u32>, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("name".to_owned(), toml::Value::String(name.to_owned()));
    EnginePlacement {
        name: name.to_owned(),
        kind: "fakes".to_owned(),
        devices,
        config,
        family: toml::Table::new(),
        lab: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        pinned: false,
    }
}

const BATCHER: usize = 1;

/// A job of `steps` decode quanta of `batcher` on its own slot. It writes the rows that it got, and it tells `done`.
struct Decoder {
    slot: u32,
    position: u64,
    steps: usize,
    seen: Arc<Mutex<Vec<(f32, f32)>>>,
    done: Sender<u32>,
}

impl Decoder {
    fn segment(&self) -> Segment {
        Segment {
            control: Vec::new(),
            state: State::Slot(self.slot),
            inputs: SegmentInputs::Tokens(vec![7]),
            positions: vec![self.position],
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: vec![self.position],
            },
        }
    }

    fn took(&mut self, out: PassOut) -> Ran {
        assert_eq!(out.outputs.len(), 1, "a job gets its own row only");
        let v: Vec<f32> = out.outputs[0]
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        self.seen.lock().unwrap().push((v[0], v[1]));
        self.position += 1;
        self.steps -= 1;
        if self.steps > 0 {
            Ran::More(vec![Kind::Decode])
        } else {
            let _ = self.done.send(self.slot);
            Ran::Done
        }
    }
}

impl JobBody for Decoder {
    fn unavailable(&mut self, engine: &str, _state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is not ready, and this test keeps it ready");
    }

    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let out = engines[BATCHER].pass(&[self.segment()], &[]).expect("a pass alone");
        self.took(out)
    }

    fn item(&self, kind: Kind) -> Item {
        Item::Block(kind)
    }

    fn joinable(&self, kind: Kind) -> bool {
        kind == Kind::Decode
    }

    fn begin_pass(&mut self, _kind: Kind, _engines: &mut [Box<dyn Engine>]) -> Result<Vec<Segment>, Ran> {
        Ok(vec![self.segment()])
    }

    fn end_pass(&mut self, _kind: Kind, result: Result<PassOut, PassError>, _engines: &mut [Box<dyn Engine>]) -> Ran {
        self.took(result.expect("the batch ran"))
    }
}

#[test]
fn decode_jobs_of_a_batching_engine_share_passes_on_its_device_thread() {
    decode_jobs_share_passes(false);
}

#[test]
fn decode_jobs_of_an_engine_with_segments_exact_share_passes_on_its_device_thread() {
    decode_jobs_share_passes(true);
}

/// Run 4 decode jobs of `batcher` on the thread of card 0, and check that they shared a pass and each got its rows.
fn decode_jobs_share_passes(exact: bool) {
    let directory = tempfile::tempdir().unwrap();
    let passes = Passes::default();
    let mut registry = Registry::new();
    registry.register("fakes", Box::new(Fakes(Arc::clone(&passes), exact)));
    let budget = |tier: Tier, bytes| BudgetSpec {
        tiers: vec![TierSpec::from(tier)],
        bytes,
    };
    let profile = PlacementProfile {
        engines: vec![
            engine("chat", vec![0, 1], Residency::OnDemand),
            engine("batcher", vec![0], Residency::Resident),
        ],
        budgets: vec![
            budget(Tier::Device(0), 1 << 20),
            budget(Tier::Device(1), 1 << 20),
            budget(Tier::Host, 1 << 20),
        ],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    };
    let inventory = Inventory {
        devices: (0..2)
            .map(|id| Device {
                id,
                name: format!("card-{id}"),
                pci_address: format!("0000:0{id}:00.0"),
                memory_bytes: 1 << 20,
                unified: false,
                ignored: false,
            })
            .collect(),
        host_bytes: 1 << 20,
    };
    let mut process = Start::new(profile, registry, inventory)
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(5))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(5)).unwrap();
    let (done, finished) = mpsc::channel();
    let mut seen = Vec::new();
    const JOBS: u32 = 4;
    const STEPS: usize = 6;
    for slot in 0..JOBS {
        let rows = Arc::new(Mutex::new(Vec::new()));
        seen.push(Arc::clone(&rows));
        let body = Decoder {
            slot,
            position: 10 * u64::from(slot),
            steps: STEPS,
            seen: rows,
            done: done.clone(),
        };
        let spec = JobSpec {
            class: Class::Interactive,
            priority: 0,
            deadline: None,
            first: Kind::Decode,
            engine: BATCHER,
            body: Box::new(body),
        };
        process.submit_job(0, spec).unwrap();
    }
    for _ in 0..JOBS {
        finished.recv_timeout(Duration::from_secs(5)).expect("each job ended");
    }
    for (slot, rows) in seen.iter().enumerate() {
        let want: Vec<(f32, f32)> = (0..STEPS).map(|i| (slot as f32, (10 * slot + i) as f32)).collect();
        assert_eq!(*rows.lock().unwrap(), want, "job {slot} got its own rows, in order");
    }
    let passes = passes.lock().unwrap().clone();
    let rows: usize = passes.iter().map(|(n, _)| n).sum();
    assert_eq!(rows, JOBS as usize * STEPS);
    assert!(
        passes.iter().all(|(_, thread)| thread == "nagaya-dev-0-0"),
        "{passes:?}"
    );
    assert!(
        passes.iter().any(|(n, _)| *n == JOBS as usize),
        "the jobs shared a pass: {passes:?}"
    );
    process.stop(Duration::from_secs(5), &nagaya::signal::SignalFlag::new());
}
