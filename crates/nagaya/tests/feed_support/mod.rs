//! Helpers for the tests of the fake with a feed rule (engine-abi D1, D11): a factory, a profile on one or two
//! devices, a started process with a council service, and a direct run of the fake by the rule.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::council::{Clock, Service, ServiceConfig};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeFeed;
use nagaya_abi::{
    Engine, Limits, OutputsWanted, PassError, PassOut, Residual, SegmentInputs, SlotLimit, StageHandle, StageOut,
    Stages, State, Tier,
};
use nagaya_council::family::{Boundary, Family};
use nagaya_store::{ContentId, Snapshot, snapshot_from_engine};
use nagaya_text_toy::ToyFamily;

/// The name of the engine in the profile.
pub const ENGINE: &str = "feed";

/// The feed block of the fake.
pub const BLOCK: u64 = 8;

/// The most tokens of one pass: more than a block, so a feed that breaks the rule fits a pass.
pub const PASS_INPUTS: u32 = 64;

/// The factory of the fake with a feed rule, in one stage or in two.
#[derive(Clone)]
pub struct FeedFactory {
    pub stages: usize,
    /// The devices that each `open` got, in order.
    pub opened_on: Arc<Mutex<Vec<Vec<GroupDevice>>>>,
    /// The time that each stage quantum takes, in milliseconds. A whole pass of the fake of one stage takes it too.
    pub delay_ms: Arc<AtomicU64>,
    /// The stage, the start and the end of each stage quantum.
    pub log: StageLog,
    /// The stage whose next quantum panics, or `NO_STAGE`.
    pub panic_stage: Arc<AtomicUsize>,
    /// The stage whose next quantum returns an error, or `NO_STAGE`.
    pub fail_stage: Arc<AtomicUsize>,
    /// The stage whose next quantum panics as a defect would, with no plan, or `NO_STAGE`.
    pub surprise_stage: Arc<AtomicUsize>,
    /// The stage and the thread name of each stage pass and each part, slot or coverage call of a stage handle.
    pub threads: ThreadLog,
    /// The segment count of each pass, logged once a pass: by a whole pass, or by stage 0 of a staged pass.
    pub sizes: Arc<Mutex<Vec<usize>>>,
    /// The profile's `devices` list of the engine. `None` gives `0` to `stages - 1`.
    pub devices: Option<Vec<u32>>,
}

/// The stage and the thread name of each call of a stage handle.
pub type ThreadLog = Arc<Mutex<Vec<(usize, String)>>>;

/// Write the stage and the name of the current thread to the log.
fn log_thread(log: &ThreadLog, stage: usize) {
    let name = std::thread::current().name().unwrap_or("unnamed").to_owned();
    log.lock().unwrap().push((stage, name));
}

/// The value of `panic_stage` when no stage panics.
pub const NO_STAGE: usize = usize::MAX;

/// The stage, the start and the end of each stage quantum.
pub type StageLog = Arc<Mutex<Vec<(usize, Instant, Instant)>>>;

impl FeedFactory {
    pub fn new(stages: usize) -> Self {
        Self {
            stages,
            opened_on: Arc::default(),
            delay_ms: Arc::default(),
            log: Arc::default(),
            panic_stage: Arc::new(AtomicUsize::new(NO_STAGE)),
            fail_stage: Arc::new(AtomicUsize::new(NO_STAGE)),
            surprise_stage: Arc::new(AtomicUsize::new(NO_STAGE)),
            threads: Arc::default(),
            sizes: Arc::default(),
            devices: None,
        }
    }
}

/// The fake, with a delay and a log on each stage quantum.
pub struct Traced {
    inner: FakeFeed,
    delay_ms: Arc<AtomicU64>,
    log: StageLog,
    panic_stage: Arc<AtomicUsize>,
    fail_stage: Arc<AtomicUsize>,
    surprise_stage: Arc<AtomicUsize>,
    threads: ThreadLog,
    sizes: Arc<Mutex<Vec<usize>>>,
}

impl Engine for Traced {
    fn identity(&self) -> nagaya_abi::Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        self.inner.limits()
    }

    fn pass(&mut self, segments: &[nagaya_abi::Segment], taps: &[nagaya_abi::Tap]) -> Result<PassOut, PassError> {
        self.sizes.lock().unwrap().push(segments.len());
        std::thread::sleep(Duration::from_millis(self.delay_ms.load(Ordering::SeqCst)));
        self.inner.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        self.inner.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.inner.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        self.inner.coverage()
    }

    fn stages(&mut self) -> Option<&mut dyn Stages> {
        self.inner.stages()?;
        Some(self)
    }
}

impl Stages for Traced {
    fn take_handles(&mut self) -> Vec<Box<dyn StageHandle + Send>> {
        let handles = self
            .inner
            .stages()
            .map(|stages| stages.take_handles())
            .unwrap_or_default();
        handles
            .into_iter()
            .enumerate()
            .map(|(stage, inner)| {
                Box::new(TracedStage {
                    stage,
                    inner,
                    delay_ms: Arc::clone(&self.delay_ms),
                    log: Arc::clone(&self.log),
                    panic_stage: Arc::clone(&self.panic_stage),
                    fail_stage: Arc::clone(&self.fail_stage),
                    surprise_stage: Arc::clone(&self.surprise_stage),
                    threads: Arc::clone(&self.threads),
                    sizes: Arc::clone(&self.sizes),
                }) as Box<dyn StageHandle + Send>
            })
            .collect()
    }
}

struct TracedStage {
    stage: usize,
    inner: Box<dyn StageHandle + Send>,
    delay_ms: Arc<AtomicU64>,
    log: StageLog,
    panic_stage: Arc<AtomicUsize>,
    fail_stage: Arc<AtomicUsize>,
    surprise_stage: Arc<AtomicUsize>,
    threads: ThreadLog,
    sizes: Arc<Mutex<Vec<usize>>>,
}

impl StageHandle for TracedStage {
    fn stage_pass(&mut self, segments: &[nagaya_abi::Segment], residual: &mut Residual) -> Result<StageOut, PassError> {
        if self.stage == 0 {
            self.sizes.lock().unwrap().push(segments.len());
        }
        if self.panic_stage.load(Ordering::SeqCst) == self.stage {
            panic!("stage {} breaks on purpose", self.stage);
        }
        if self.surprise_stage.load(Ordering::SeqCst) == self.stage {
            panic!("stage {} breaks", self.stage);
        }
        if self.fail_stage.load(Ordering::SeqCst) == self.stage {
            return Err(PassError {
                kind: nagaya_abi::PassErrorKind::Refused,
                message: format!("stage {} failed on purpose", self.stage),
            });
        }
        log_thread(&self.threads, self.stage);
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(self.delay_ms.load(Ordering::SeqCst)));
        let out = self.inner.stage_pass(segments, residual);
        self.log.lock().unwrap().push((self.stage, start, Instant::now()));
        out
    }

    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        log_thread(&self.threads, self.stage);
        self.inner.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        log_thread(&self.threads, self.stage);
        self.inner.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        log_thread(&self.threads, self.stage);
        self.inner.coverage()
    }
}

pub fn slots() -> Vec<SlotLimit> {
    vec![SlotLimit {
        bytes: 8 * 4096,
        count: 3,
        tokens: 4096,
    }]
}

/// The fake of `stages` stages over the toy vocabulary, with the speaking bias of the toy family.
pub fn engine(stages: usize) -> FakeFeed {
    let engine = match stages {
        1 => FakeFeed::new(BLOCK, slots()).with_max_pass_inputs(PASS_INPUTS),
        2 => FakeFeed::two_stages(BLOCK, slots()),
        _ => panic!("the fake has one stage or two"),
    };
    engine
        .with_family("toy")
        .with_vocab(gogi_toy::VOCAB)
        .with_logit_bias(nagaya_text_toy::speaking_bias())
}

impl Factory for FeedFactory {
    fn plan(&self, _config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        if devices.len() != self.stages {
            return Err(PlanError {
                message: format!(
                    "the fake of {} stages needs {} devices, not {}",
                    self.stages,
                    self.stages,
                    devices.len()
                ),
            });
        }
        Ok(engine(self.stages).limits())
    }

    fn open(&self, _config: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        self.opened_on.lock().unwrap().push(devices.to_vec());
        Ok(Box::new(Traced {
            inner: engine(self.stages),
            delay_ms: Arc::clone(&self.delay_ms),
            log: Arc::clone(&self.log),
            panic_stage: Arc::clone(&self.panic_stage),
            fail_stage: Arc::clone(&self.fail_stage),
            surprise_stage: Arc::clone(&self.surprise_stage),
            threads: Arc::clone(&self.threads),
            sizes: Arc::clone(&self.sizes),
        }))
    }
}

pub fn profile(stages: usize) -> PlacementProfile {
    let mut budgets: Vec<BudgetSpec> = (0..stages as u32)
        .map(|device| BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Device(device))],
            bytes: 4096,
        })
        .collect();
    budgets.push(BudgetSpec {
        tiers: vec![TierSpec::from(Tier::Host)],
        bytes: 1 << 20,
    });
    PlacementProfile {
        engines: vec![EnginePlacement {
            name: ENGINE.to_owned(),
            kind: "feed".to_owned(),
            devices: (0..stages as u32).collect(),
            config: toml::Table::new(),
            family: toml::Table::new(),
            lab: toml::Table::new(),
            share: 1,
            pin_identity: false,
            residency: Default::default(),
            pinned: false,
        }],
        budgets,
        store: Some(StoreSpec {
            nvme: 1 << 24,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
    }
}

pub fn inventory(stages: usize) -> Inventory {
    Inventory {
        devices: (0..stages as u32)
            .map(|id| Device {
                id,
                name: format!("device-{id}"),
                pci_address: format!("0000:0{id}:00.0"),
                memory_bytes: 1 << 20,
                unified: false,
                ignored: false,
            })
            .collect(),
        host_bytes: 1 << 22,
    }
}

/// A started process with the fake, and a council service of the toy family on it.
pub struct FeedFixture {
    pub service: Service,
    pub process: Process,
    pub directory: tempfile::TempDir,
    pub factory: FeedFactory,
}

impl FeedFixture {
    pub fn new(stages: usize) -> Self {
        Self::with(FeedFactory::new(stages), Duration::from_secs(10))
    }

    /// A fixture of one stage whose service has this family, not the toy family.
    pub fn with_family(family: Arc<dyn Family>) -> Self {
        Self::build(FeedFactory::new(1), Duration::from_secs(10), None, None, family)
    }

    /// A fixture with this factory and this watchdog timeout.
    pub fn with(factory: FeedFactory, timeout: Duration) -> Self {
        Self::with_sets(factory, timeout, None)
    }

    /// A fixture with this factory, this watchdog timeout and these CPU sets.
    pub fn with_sets(factory: FeedFactory, timeout: Duration, sets: Option<nagaya::cpu_sets::CpuSets>) -> Self {
        Self::build(factory, timeout, sets, None, Arc::new(ToyFamily::new()))
    }

    /// A fixture with this factory, this watchdog timeout and this machine tree.
    pub fn with_machine(factory: FeedFactory, timeout: Duration, machine: nagaya::evidence::Machine) -> Self {
        Self::build(factory, timeout, None, Some(machine), Arc::new(ToyFamily::new()))
    }

    fn build(
        factory: FeedFactory,
        timeout: Duration,
        sets: Option<nagaya::cpu_sets::CpuSets>,
        machine: Option<nagaya::evidence::Machine>,
        family: Arc<dyn Family>,
    ) -> Self {
        abort_on_a_panic_with_no_plan();
        let stages = factory.stages;
        let directory = tempfile::tempdir().unwrap();
        let mut registry = Registry::new();
        registry.register("feed", Box::new(factory.clone()));
        let mut profile = profile(stages);
        if let Some(devices) = &factory.devices {
            profile.engines[0].devices = devices.clone();
        }
        let mut start = Start::new(profile, registry, inventory(stages));
        if let Some(sets) = sets {
            start = start.cpu_sets(sets);
        }
        if let Some(machine) = machine {
            start = start.machine(machine);
        }
        let mut process = start
            .store(directory.path())
            .heartbeat(Duration::from_millis(10))
            .timeout(timeout)
            .start()
            .unwrap();
        process.wait_opens(Duration::from_secs(10)).unwrap();
        let clock = Arc::new(AtomicU64::new(1_000_000));
        let read: Clock = Arc::new(move || clock.fetch_add(60_000, Ordering::SeqCst));
        let service = Service::attach_with_clock(
            process.ports(),
            family,
            directory.path(),
            ServiceConfig::new("toy", ENGINE),
            read,
        )
        .unwrap();
        Self {
            service,
            process,
            directory,
            factory,
        }
    }
}

/// Abort the test binary when a thread of the process panics with no plan.
///
/// The process turns such a panic into an event (spec 8.5), and a call of the service then waits for a reply that
/// never comes. Thus a defect that panics would hang the test, not fail it. A planned panic says "on purpose". It
/// makes other threads of its process panic after it, so the hook stops in a binary after a planned panic.
fn abort_on_a_panic_with_no_plan() {
    static HOOK: std::sync::Once = std::sync::Once::new();
    static PLANNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            previous(info);
            let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
            let message = info.payload_as_str().unwrap_or("");
            if message.contains("on purpose") {
                PLANNED.store(true, Ordering::SeqCst);
            }
            if thread.starts_with("nagaya-") && !PLANNED.load(Ordering::SeqCst) {
                eprintln!("thread `{thread}` panicked with no plan, so the test binary aborts: {message}");
                std::process::abort();
            }
        }));
    });
}

/// A direct run of the fake by the feed rule, with no process: the oracle of the tests.
pub struct Direct {
    pub engine: FakeFeed,
    pub rows: u64,
}

impl Direct {
    pub fn new() -> Self {
        Self {
            engine: engine(1),
            rows: 0,
        }
    }

    /// Feed tokens in one pass, and give the logit row of the last token.
    pub fn pass(&mut self, tokens: &[u32]) -> Vec<u8> {
        let to = self.rows + tokens.len() as u64;
        let out = self
            .engine
            .pass(
                &[nagaya_abi::Segment {
                    control: Vec::new(),
                    state: State::Slot(0),
                    inputs: SegmentInputs::Tokens(tokens.to_vec()),
                    positions: (self.rows..to).collect(),
                    meta: Vec::new(),
                    outputs: OutputsWanted {
                        logit_rows: vec![to - 1],
                    },
                }],
                &[],
            )
            .unwrap();
        self.rows = to;
        out.outputs[0].bytes.clone()
    }

    /// Feed `tokens` by the rule: each span between `boundaries` in passes of the block from its start.
    pub fn by_rule(tokens: &[u32], boundaries: &[Boundary]) -> Self {
        let mut direct = Self::new();
        let mut start = 0;
        for boundary in boundaries {
            for chunk in tokens[start..boundary.tokens].chunks(BLOCK as usize) {
                direct.pass(chunk);
            }
            start = boundary.tokens;
        }
        direct
    }

    /// The snapshot of slot 0, as the store keeps it.
    pub fn snapshot(&mut self) -> Snapshot {
        let rows: Vec<(usize, u64, u64)> = (0..self.engine.limits().parts.len())
            .map(|part| (part, 0, self.rows))
            .collect();
        snapshot_from_engine(&mut self.engine, State::Slot(0), &rows).unwrap()
    }

    pub fn content_id(&mut self) -> ContentId {
        self.snapshot().content_id().unwrap()
    }
}

/// The tokens and the boundaries of a context, as the toy family renders it for a build.
pub fn rendering(context: &gogi_proto::context::Context) -> (Vec<u32>, Vec<Boundary>) {
    let rendering = ToyFamily::new().render(context, false).unwrap();
    (rendering.tokens[..rendering.end].to_vec(), rendering.boundaries)
}
