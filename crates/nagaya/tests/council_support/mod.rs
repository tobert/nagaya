//! Helpers for the tests of the council service: a fake engine with a context term, a profile, and a started process.

#![allow(dead_code)]

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nagaya::council::{Clock, Service, ServiceConfig};
use nagaya_text_toy::ToyFamily;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{OpenedEngine, Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Control, ControlItem, Coverage, Engine, Identity, Kind, Lifetime, Limits, MandatoryTerm, Parts,
    PassError, PassOut, Seal, Segment, SlotLimit, Slots, Tap, Tier,
};

/// What the engines of a factory saw of control vectors: the items of each pass, and each write into the table.
#[derive(Clone, Debug, Default)]
pub struct ControlLog(Arc<Mutex<Logged>>);

/// The items of each pass, the taps of each pass, and each write as its entry and its first value.
#[derive(Clone, Debug, Default)]
pub struct Logged {
    pub passes: Vec<(Vec<ControlItem>, Vec<String>)>,
    /// The segment count of each pass, in the order of `passes`.
    pub sizes: Vec<usize>,
    pub writes: Vec<(u32, f32)>,
}

impl ControlLog {
    /// What the log holds now.
    pub fn take(&self) -> Logged {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// The name of the engine in the profile.
pub const ENGINE: &str = "toy";

/// The tokens that a scripted engine favors: from the position `base`, the token at `base + k` is `tokens[k]`.
#[derive(Clone, Debug, Default)]
pub struct Script(Arc<Mutex<Option<Scripted>>>);

/// The first position of a script, and its tokens.
type Scripted = (u64, Vec<u32>);

impl Script {
    /// Favor `tokens` from the position `base`.
    pub fn set(&self, base: usize, tokens: Vec<u32>) {
        *self.0.lock().unwrap() = Some((base as u64, tokens));
    }

    /// The logit that a scripted token gets. The fake's logits lie between -6 and 6, so the token wins by far.
    pub const LOGIT: f32 = 40.0;

    fn apply(&self, positions: &[u64], outputs: &mut [nagaya_abi::Output], vocab: usize) {
        let script = self.0.lock().unwrap();
        let Some((base, tokens)) = script.as_ref() else {
            return;
        };
        let rows = outputs.iter_mut().filter(|output| output.name == "logits");
        for (position, output) in positions.iter().zip(rows) {
            // The row at a position gives the token after it.
            let Some(&token) = (position + 1).checked_sub(*base).and_then(|k| tokens.get(k as usize)) else {
                continue;
            };
            assert!((token as usize) < vocab, "a script names a token of the vocabulary");
            let at = token as usize * 4;
            output.bytes[at..at + 4].copy_from_slice(&Self::LOGIT.to_le_bytes());
        }
    }
}

/// The segments that an engine saw, in order. A test reads the inputs and the `meta` that the jobs sent.
#[derive(Clone, Debug, Default)]
pub struct Seen(Arc<Mutex<Vec<Segment>>>);

impl Seen {
    /// The segments so far. The log is empty after the call.
    pub fn take(&self) -> Vec<Segment> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

/// A fake engine that declares a mandatory term for each context.
pub struct WithContextTerm {
    inner: FakeAlways,
    bytes: u64,
    delay: Duration,
    script: Script,
    vocab: usize,
    grain: Option<u64>,
    seal: bool,
    close: bool,
    log: ControlLog,
    seen: Option<Seen>,
    segments: u32,
}

impl Control for WithContextTerm {
    fn control_write(&mut self, index: u32, bytes: &[u8]) -> Result<(), String> {
        let first = f32::from_le_bytes(bytes[..4].try_into().unwrap());
        self.log.0.lock().unwrap().writes.push((index, first));
        self.inner
            .control()
            .expect("the fake has a table")
            .control_write(index, bytes)
    }
}

impl Engine for WithContextTerm {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        let mut limits = self.inner.limits();
        limits.checkpoint_grain = self.grain;
        limits.max_segments = self.segments;
        if self.close {
            limits.capabilities.push(Capability::Close);
        }
        if self.bytes > 0 {
            limits.memory.mandatory.push(MandatoryTerm {
                name: "context".to_owned(),
                tier: Tier::Host,
                bytes: self.bytes,
                lifetime: Lifetime::Context,
                kinds: vec![Kind::Prefill, Kind::Read],
            });
        }
        limits
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        std::thread::sleep(self.delay);
        if let Some(segment) = segments.first() {
            let taps = taps.iter().map(|tap| tap.name.clone()).collect();
            let mut log = self.log.0.lock().unwrap();
            log.passes.push((segment.control.clone(), taps));
            log.sizes.push(segments.len());
        }
        if let Some(seen) = &self.seen {
            seen.0.lock().unwrap().extend(segments.iter().cloned());
        }
        // The fake reads tokens. A segment with rows feeds each token mixed with a hash of its rows and meta, so two
        // images of one size give two states, as on a card.
        let tokens: Vec<Segment> = segments
            .iter()
            .cloned()
            .map(|mut segment| {
                if let nagaya_abi::SegmentInputs::TokensAndRows { tokens, rows } = segment.inputs {
                    let hash = blake3::Hasher::new().update(&rows).update(&segment.meta).finalize();
                    let mix = u32::from_le_bytes(hash.as_bytes()[..4].try_into().unwrap());
                    segment.inputs = nagaya_abi::SegmentInputs::Tokens(tokens.iter().map(|t| t ^ mix).collect());
                    segment.meta = Vec::new();
                }
                segment
            })
            .collect();
        let segments = tokens.as_slice();
        let mut out = self.inner.pass(segments, taps)?;
        assert!(
            segments.len() <= self.segments as usize,
            "a pass holds at most the engine's segments"
        );
        if let [segment] = segments {
            self.script
                .apply(&segment.outputs.logit_rows, &mut out.outputs, self.vocab);
        } else if !segments.is_empty() {
            // A joined pass gives only `logits`, one for each logit row, as the Qwen3.8 engine does (engine-abi "Logit
            // outputs"). The fake also gives a `state` output for each segment, which the process's split refuses, so
            // this engine drops it. A joined pass carries no tap (council `joinable_unit`).
            assert!(taps.is_empty(), "a joined pass carries no tap");
            out.outputs.retain(|output| output.name == "logits");
            let rows: usize = segments.iter().map(|segment| segment.outputs.logit_rows.len()).sum();
            assert_eq!(
                out.outputs.len(),
                rows,
                "a joined pass gives one `logits` for each logit row"
            );
            let mut rest = out.outputs.as_mut_slice();
            for segment in segments {
                let (mine, after) = std::mem::take(&mut rest).split_at_mut(segment.outputs.logit_rows.len());
                self.script.apply(&segment.outputs.logit_rows, mine, self.vocab);
                rest = after;
            }
        }
        Ok(out)
    }

    fn last_error(&self) -> Option<String> {
        self.inner.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.inner.parts()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.inner.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        if !self.seal {
            return None;
        }
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }

    fn control(&mut self) -> Option<&mut dyn Control> {
        if self.inner.control().is_some() {
            Some(self)
        } else {
            None
        }
    }
}

/// The factory of [`WithContextTerm`] engines.
pub struct TermFactory {
    pub bytes: u64,
    pub slots: Vec<SlotLimit>,
    pub vocab: Option<usize>,
    /// The time that each pass takes.
    pub delay: Duration,
    /// The tokens that the engine favors.
    pub script: Script,
    /// The checkpoint grain that the engine declares (spec 6.13).
    pub grain: Option<u64>,
    /// The cap on interval copies in the `[store]` table of the profile.
    pub interval_copies: u32,
    /// True when the engine speaks the toy language: its logits get the speaking bias of `nagaya-text-toy`.
    pub speaking: bool,
    /// False for an engine with no `seal` capability: a seal changes no bits.
    pub seal: bool,
    /// True for an engine that declares `close`, so that it can be on demand (spec 3.3a).
    pub close: bool,
    /// The entries of the control table of the engine, or `None` for an engine with no table (engine-abi "Control
    /// vectors").
    pub control: Option<u32>,
    /// True for an engine with the tap `lens` (spec 13.1). It needs the control table.
    pub lens: bool,
    /// What the engines saw of control vectors.
    pub log: ControlLog,
    /// The most segments in a pass. The toy has 1, so its council jobs never join. With more, the engine declares
    /// `segments_exact` (the fake's), so its decode quanta join (spec 5.7).
    pub segments: u32,
    /// The log of the segments of each pass, or none.
    pub seen: Option<Seen>,
}

impl TermFactory {
    /// A factory of engines with the toy vocabulary, roomy slots and no term.
    pub fn toy() -> Self {
        Self {
            bytes: 0,
            slots: roomy_slots(3),
            vocab: Some(gogi_toy::VOCAB),
            delay: Duration::ZERO,
            script: Script::default(),
            grain: None,
            interval_copies: 0,
            speaking: false,
            seal: true,
            close: false,
            control: None,
            lens: false,
            log: ControlLog::default(),
            seen: None,
            segments: 1,
        }
    }

    fn engine(&self) -> WithContextTerm {
        let mut inner = FakeAlways::with_slots(self.slots.clone());
        if let Some(vocab) = self.vocab {
            inner = inner.with_vocab(vocab);
        }
        if self.speaking {
            inner = inner.with_logit_bias(nagaya_text_toy::speaking_bias());
        }
        if let Some(tables) = self.control {
            inner = inner.with_control(tables);
        }
        if self.lens {
            inner = inner.with_lens();
        }
        WithContextTerm {
            inner,
            bytes: self.bytes,
            delay: self.delay,
            script: self.script.clone(),
            vocab: self.vocab.unwrap_or(0),
            grain: self.grain,
            seal: self.seal,
            close: self.close,
            log: self.log.clone(),
            seen: self.seen.clone(),
            segments: self.segments,
        }
    }
}

impl Factory for TermFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.engine().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(self.engine()))
    }
}

/// Slots that are big enough for the toy contexts of the tests.
pub fn roomy_slots(count: u32) -> Vec<SlotLimit> {
    vec![SlotLimit {
        bytes: 16384,
        count,
        tokens: 4096,
    }]
}

pub fn profile(host_budget: u64, nvme: u64, host_store: Option<u64>) -> PlacementProfile {
    profile_with_copies(host_budget, nvme, host_store, 0)
}

/// The profile with a cap on interval copies (spec 6.13).
pub fn profile_with_copies(host_budget: u64, nvme: u64, host_store: Option<u64>, copies: u32) -> PlacementProfile {
    PlacementProfile {
        engines: vec![EnginePlacement {
            name: ENGINE.to_owned(),
            kind: "term".to_owned(),
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
                bytes: host_budget,
            },
        ],
        store: Some(StoreSpec {
            nvme,
            host: host_store,
            interval_copies: copies,
            objects: None,
        }),
    }
}

pub fn inventory() -> Inventory {
    Inventory {
        devices: vec![Device {
            id: 0,
            name: "device-0".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 1024,
            unified: false,
            ignored: false,
        }],
        host_bytes: 1 << 20,
    }
}

/// Start a process with one engine and wait until it opens.
pub fn start(
    directory: &Path,
    factory: TermFactory,
    host_budget: u64,
    host_store: Option<u64>,
) -> (Process, Vec<OpenedEngine>) {
    let copies = factory.interval_copies;
    // The control table is a device term of the engine, so the device budget grows by it.
    let table = factory.engine().limits().control.map_or(0, |limit| limit.table_bytes());
    let mut registry = Registry::new();
    registry.register("term", Box::new(factory));
    let mut profile = profile_with_copies(host_budget, 1 << 24, host_store, copies);
    profile.budgets[0].bytes += table;
    let mut process = Start::new(profile, registry, inventory())
        .store(directory)
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(10))
        .start()
        .unwrap();
    let opened = process.wait_opens(Duration::from_secs(10)).unwrap();
    (process, opened)
}

/// A started process with one engine, and a service on it. The process outlives the service.
pub struct Fixture {
    pub service: Service,
    pub process: Process,
    pub directory: tempfile::TempDir,
    /// The clock of the service, in milliseconds.
    pub clock: Arc<AtomicU64>,
}

/// Each reading of the clock is a minute later. The eviction value of the store has a resolution of about a second,
/// so a smaller step would leave the entries tied.
pub const TICK_MS: u64 = 60_000;

impl Fixture {
    pub fn new() -> Self {
        Self::with(0, 1 << 20, None, ServiceConfig::new("toy", ENGINE))
    }

    pub fn with(term: u64, host_budget: u64, host_store: Option<u64>, config: ServiceConfig) -> Self {
        let factory = TermFactory {
            bytes: term,
            ..TermFactory::toy()
        };
        Self::with_factory(factory, host_budget, host_store, config)
    }

    /// A fixture whose engines come from `factory`.
    pub fn with_factory(
        factory: TermFactory,
        host_budget: u64,
        host_store: Option<u64>,
        config: ServiceConfig,
    ) -> Self {
        Self::with_family(factory, Arc::new(ToyFamily::new()), host_budget, host_store, config)
    }

    /// A fixture whose engines come from `factory`, and whose service uses `family`.
    pub fn with_family(
        factory: TermFactory,
        family: Arc<dyn nagaya_council::family::Family>,
        host_budget: u64,
        host_store: Option<u64>,
        config: ServiceConfig,
    ) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let (process, _) = start(directory.path(), factory, host_budget, host_store);
        let clock = Arc::new(AtomicU64::new(1_000_000));
        let reader = Arc::clone(&clock);
        let read: Clock = Arc::new(move || reader.fetch_add(TICK_MS, Ordering::SeqCst));
        let service = Service::attach_with_clock(process.ports(), family, directory.path(), config, read).unwrap();
        Self {
            service,
            process,
            directory,
            clock,
        }
    }

    /// A reader of the clock of the service. Each reading is a tick later.
    pub fn reader(&self) -> Clock {
        let reader = Arc::clone(&self.clock);
        Arc::new(move || reader.fetch_add(TICK_MS, Ordering::SeqCst))
    }

    /// Attach another service to the same process and store directory, with the same clock.
    pub fn attach_again(&self) -> Service {
        Service::attach_with_clock(
            self.process.ports(),
            Arc::new(ToyFamily::new()),
            self.directory.path(),
            ServiceConfig::new("toy", ENGINE),
            self.reader(),
        )
        .unwrap()
    }

    /// Move the clock forward.
    pub fn advance(&self, ms: u64) {
        self.clock.fetch_add(ms, Ordering::SeqCst);
    }

    /// The entries of the store.
    pub fn entries(&self) -> usize {
        let ports = self.process.ports();
        let store = ports.store().unwrap().lock().unwrap();
        store.entries().len()
    }
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}

/// A direct run of the fake engine, with no process and no store: the oracle of the council tests.
pub struct Direct {
    pub engine: FakeAlways,
    /// The rows of slot 0.
    pub rows: usize,
}

impl Direct {
    /// An engine with one slot that holds `context` as a build leaves it: fed in passes of 64 tokens, and sealed at
    /// each boundary.
    pub fn built(context: &gogi_proto::context::Context) -> Self {
        use nagaya_council::family::Family;
        let rendering = ToyFamily::new().render(context, false).unwrap();
        let mut direct = Self::empty();
        let mut position = 0;
        for boundary in &rendering.boundaries {
            for from in (position..boundary.tokens).step_by(64) {
                let to = boundary.tokens.min(from + 64);
                direct.feed(&rendering.tokens[from..to], false);
            }
            position = boundary.tokens;
            direct.engine.seal().unwrap().seal(nagaya_abi::State::Slot(0)).unwrap();
        }
        direct
    }

    /// An engine with one empty slot.
    pub fn empty() -> Self {
        Self {
            engine: FakeAlways::with_slots(vec![SlotLimit {
                bytes: 16384,
                count: 1,
                tokens: 4096,
            }])
            .with_vocab(gogi_toy::VOCAB),
            rows: 0,
        }
    }

    /// Feed tokens in one pass. With `row`, give the logit row of the last token.
    pub fn feed(&mut self, tokens: &[u32], row: bool) -> Option<Vec<f32>> {
        use nagaya_abi::{OutputsWanted, SegmentInputs, State};
        let from = self.rows as u64;
        let to = from + tokens.len() as u64;
        let out = self
            .engine
            .pass(
                &[Segment {
                    control: Vec::new(),
                    state: State::Slot(0),
                    inputs: SegmentInputs::Tokens(tokens.to_vec()),
                    positions: (from..to).collect(),
                    meta: Vec::new(),
                    outputs: OutputsWanted {
                        logit_rows: if row { vec![to - 1] } else { Vec::new() },
                    },
                }],
                &[],
            )
            .unwrap();
        self.rows += tokens.len();
        row.then(|| {
            let bytes = &out.outputs.iter().find(|output| output.name == "logits").unwrap().bytes;
            nagaya_council::read::logit_row(bytes, gogi_toy::VOCAB).unwrap()
        })
    }

    /// Feed tokens in passes of at most 64, and give the row of the last token.
    pub fn feed_all(&mut self, tokens: &[u32]) -> Vec<f32> {
        let chunks: Vec<&[u32]> = tokens.chunks(64).collect();
        let mut row = None;
        for (i, chunk) in chunks.iter().enumerate() {
            row = self.feed(chunk, i + 1 == chunks.len());
        }
        row.unwrap()
    }
}

/// Post a JSON body to gogi-fake, and give the JSON of its 200 response.
pub fn post(server: &gogi_fake::Server, path: &str, body: serde_json::Value) -> serde_json::Value {
    use std::io::{Read as _, Write as _};
    let addr = server.url().trim_start_matches("http://").to_string();
    let mut c = std::net::TcpStream::connect(&addr).unwrap();
    let b = body.to_string();
    write!(
        c,
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    )
    .unwrap();
    let mut raw = String::new();
    c.read_to_string(&mut raw).unwrap();
    let (head, body) = raw.split_once("\r\n\r\n").unwrap();
    assert!(head.contains(" 200 "), "{head}\n{body}");
    serde_json::from_str(body).unwrap()
}

/// The F16 bits of `value`, which must be exact in F16: a normal number with at most 11 significant bits, or zero.
pub fn f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent), "{value} is a normal F16 number");
    assert_eq!(bits & 0x1fff, 0, "{value} has at most 11 significant bits");
    sign | ((exponent as u16) << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// Write a J-Lens file of the fit tool's format into `path` (`docs/lab.md`, "The rules of the J-Lens"): the F16
/// matrix `layers.<l>` of [width, width] for each matrix, with `weight_hash` in its metadata. Give its bytes.
pub fn lens_file(path: &Path, weight_hash: &str, matrices: &[Vec<f32>], width: usize) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut names: Vec<String> = (0..matrices.len()).map(|l| format!("layers.{l}")).collect();
    names.sort();
    for name in &names {
        let l: usize = name["layers.".len()..].parse().unwrap();
        assert_eq!(matrices[l].len(), width * width, "a matrix is [width, width]");
        let start = data.len();
        for value in &matrices[l] {
            data.extend(f16_bits(*value).to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F16", "shape": [width, width], "data_offsets": [start, data.len()]}),
        );
    }
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({"model": "toy", "weight_hash": weight_hash, "nagaya.lens": "jlens"}),
    );
    let text = serde_json::Value::Object(header).to_string();
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend(text.as_bytes());
    bytes.extend(data);
    std::fs::write(path, &bytes).unwrap();
    bytes
}
