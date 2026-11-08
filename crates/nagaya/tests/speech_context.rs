//! Generations of a speech context in the process (spec 11.8a), with a fake speech engine that holds state as the
//! real one does: one row of state for each position, read and written through a cells part.
//!
//! The fake's row is a hash of the row before it and of the position. The codes of a frame are a hash of the last row,
//! the seed and the frame. Thus a line gives the same codes only from the same state, fed or restored.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use gogi_proto::generation::GenerationDone;
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::council::Schedule;
use nagaya::http::{Council, router};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::speech::codes::CodeBook;
use nagaya::speech::{
    ContextItem, ContextRendering, SpeechConfig, SpeechContext, SpeechError, SpeechFamily, SpeechService, Spoken, Told,
};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Dtype, Engine, Identity, Lifetime, Limits, MandatoryTerm, PartError, PartId, PartKind, PartLimits,
    Parts, PassError, PassErrorKind, PassOut, PromptPair, Segment, SlotId, SlotLimit, Slots, Speech, SpeechOut,
    SpeechPosition, SpeechSampler, State, Tap, Tier, WriteResult,
};
use nagaya_store::objects::ObjectStore;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "fake-voice";
/// The samples of one fake frame.
const FRAME: usize = 2;
const ROOM: u64 = 256;

type Log = Arc<Mutex<Vec<String>>>;

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^ (x >> 29)
}

fn words(position: &SpeechPosition) -> Vec<u64> {
    match position {
        SpeechPosition::Pair(p) => vec![0, u64::from(p.text.unwrap_or(7)), u64::from(p.codec.unwrap_or(9))],
        SpeechPosition::Frame(codes) => std::iter::once(1).chain(codes.iter().map(|&c| u64::from(c))).collect(),
    }
}

#[derive(Default)]
struct Line {
    seed: u64,
    frames: usize,
    total: usize,
}

#[derive(Default)]
struct SlotState {
    rows: Vec<u64>,
    line: Option<Line>,
}

/// A speech engine with one `u64` row of state for each position.
struct Voice {
    log: Log,
    slots: Vec<SlotState>,
    /// While it is set, a frames call waits.
    hold: Arc<AtomicBool>,
    /// The top_k and the temperature of the last feed with a sampler.
    sampled: Arc<Mutex<Option<(u32, f32)>>>,
}

impl Voice {
    fn push(&mut self, slot: SlotId, position: &SpeechPosition) {
        let rows = &mut self.slots[slot as usize].rows;
        let last = rows.last().copied().unwrap_or(1);
        rows.push(words(position).into_iter().fold(last, mix));
    }
}

impl Speech for Voice {
    fn speech_feed(
        &mut self,
        slot: SlotId,
        positions: &[SpeechPosition],
        sampler: Option<&SpeechSampler>,
    ) -> Result<(), PassError> {
        let s = &mut self.slots[slot as usize];
        assert!(
            s.line.as_ref().is_none_or(|line| line.frames == line.total),
            "a feed during a line"
        );
        self.log
            .lock()
            .unwrap()
            .push(format!("feed:{slot}:{}:{}", positions.len(), sampler.is_some()));
        for position in positions {
            self.push(slot, position);
        }
        if let Some(sampler) = sampler {
            *self.sampled.lock().unwrap() = Some((sampler.top_k, sampler.temperature));
        }
        let s = &mut self.slots[slot as usize];
        s.line = sampler.map(|sampler| Line {
            seed: sampler.seed,
            frames: 0,
            // The state decides the length of the line: 2 to 4 frames.
            total: 2 + (mix(*s.rows.last().unwrap(), sampler.seed) % 3) as usize,
        });
        Ok(())
    }

    fn speech_frames(&mut self, slot: SlotId, max: u32) -> Result<SpeechOut, PassError> {
        while self.hold.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let s = &mut self.slots[slot as usize];
        let line = s.line.as_mut().expect("frames of a line");
        let n = (line.total - line.frames).min(max as usize).min(2);
        let mut out = SpeechOut::default();
        for _ in 0..n {
            let last = *s.rows.last().unwrap();
            let codes: Vec<u32> = (0..16)
                .map(|g| (mix(mix(last, line.seed), (line.frames * 16 + g) as u64) % 2048) as u32)
                .collect();
            out.audio.extend((0..FRAME).map(|i| codes[i] as i16));
            out.codes.extend(&codes);
            line.frames += 1;
            let frame = SpeechPosition::Frame(codes);
            let rows = &mut s.rows;
            rows.push(words(&frame).into_iter().fold(last, mix));
        }
        out.frames = n as u32;
        out.end = line.frames == line.total;
        Ok(out)
    }

    fn speech_clear(&mut self, slot: SlotId) {
        self.slots[slot as usize] = SlotState::default();
    }
}

impl Parts for Voice {
    fn part_read(&mut self, _state: State, _part: PartId, _out: &mut [u8]) -> Result<(), PartError> {
        Err(PartError::Refused)
    }

    fn part_write(&mut self, _state: State, _part: PartId, _bytes: &[u8]) -> WriteResult {
        WriteResult::Refused
    }

    fn cells_read(&mut self, state: State, _part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let State::Slot(slot) = state;
        let rows = &self.slots[slot as usize].rows;
        let (from, n) = (from as usize, n as usize);
        if from + n > rows.len() || out.len() != n * 8 {
            return Err(PartError::Refused);
        }
        for (k, row) in rows[from..from + n].iter().enumerate() {
            out[k * 8..k * 8 + 8].copy_from_slice(&row.to_le_bytes());
        }
        Ok(())
    }

    fn cells_write(&mut self, state: State, _part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        let State::Slot(slot) = state;
        let s = &mut self.slots[slot as usize];
        if s.line.is_some() || bytes.len() != n as usize * 8 {
            return WriteResult::Refused;
        }
        let end = (from + n) as usize;
        if s.rows.len() < end {
            s.rows.resize(end, 0);
        }
        for k in 0..n as usize {
            s.rows[from as usize + k] = u64::from_le_bytes(bytes[k * 8..k * 8 + 8].try_into().unwrap());
        }
        WriteResult::Ok
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let State::Slot(slot) = state;
        self.slots[slot as usize].rows.truncate(n as usize);
        self.log.lock().unwrap().push(format!("restore:{slot}:{n}"));
        WriteResult::Ok
    }
}

impl Slots for Voice {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.slots[slot as usize].rows.len() as u64
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        self.speech_clear(slot);
        WriteResult::Ok
    }
}

fn limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities = vec![
        Capability::Speech,
        Capability::Close,
        Capability::Parts,
        Capability::Slots,
    ];
    limits.parts = vec![PartLimits {
        name: "kv".to_owned(),
        kind: PartKind::Cells,
        dtype: Dtype::U64,
        shape: vec![1],
        block_format: None,
        stage: 0,
    }];
    limits.slots = vec![SlotLimit {
        bytes: 0,
        count: 2,
        tokens: ROOM,
    }];
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes: 100,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic = Vec::new();
    limits
}

impl Engine for Voice {
    fn identity(&self) -> Identity {
        let mut identity = FakeAlways::new().identity();
        identity.model = MODEL.to_owned();
        identity
    }

    fn limits(&self) -> Limits {
        limits()
    }

    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        Err(PassError {
            kind: PassErrorKind::Refused,
            message: "a speech engine takes no pass".to_owned(),
        })
    }

    fn last_error(&self) -> Option<String> {
        None
    }

    fn speech(&mut self) -> Option<&mut dyn Speech> {
        Some(self)
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        Some(self)
    }
}

struct Voices(Log, Arc<AtomicBool>, Arc<Mutex<Option<(u32, f32)>>>);

impl Factory for Voices {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Voice {
            log: Arc::clone(&self.0),
            slots: vec![SlotState::default(), SlotState::default()],
            hold: Arc::clone(&self.1),
            sampled: Arc::clone(&self.2),
        }))
    }
}

/// A family that renders a voice and a line as one text position for each character, and audio as its frames. A line
/// after audio starts with the codec end, as the Qwen3-TTS family does.
struct Letters;

fn chars(text: &str) -> impl Iterator<Item = SpeechPosition> + '_ {
    text.chars().map(|c| {
        SpeechPosition::Pair(PromptPair {
            text: Some(c as u32),
            codec: None,
        })
    })
}

impl SpeechFamily for Letters {
    fn context(
        &self,
        voice: Option<&str>,
        items: &[ContextItem],
        language: &str,
    ) -> Result<ContextRendering, SpeechError> {
        let mut positions: Vec<SpeechPosition> = Vec::new();
        let mut ends = Vec::new();
        if let Some(voice) = voice {
            positions.extend(chars(voice));
            ends.push(positions.len());
        }
        let mut after_audio = false;
        for item in items {
            match item {
                ContextItem::Line(text) => {
                    if after_audio {
                        positions.push(SpeechPosition::Pair(PromptPair {
                            text: None,
                            codec: Some(2150),
                        }));
                    }
                    positions.extend(chars(language));
                    positions.extend(chars(text));
                    after_audio = false;
                }
                ContextItem::Audio(frames) => {
                    positions.extend(frames.iter().map(|f| SpeechPosition::Frame(f.to_vec())));
                    after_audio = true;
                }
            }
            ends.push(positions.len());
        }
        Ok(ContextRendering { positions, ends })
    }
}

struct Fixture {
    process: Process,
    log: Log,
    hold: Arc<AtomicBool>,
    sampled: Arc<Mutex<Option<(u32, f32)>>>,
    directory: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self::with_host(1 << 20)
    }

    /// A fixture whose host budget is `host` bytes: the ledger charges the copy of each snapshot there.
    fn with_host(host: u64) -> Self {
        Self::with(host, Residency::Resident)
    }

    /// A fixture whose speech engine is on demand, so it starts unloaded (spec 3.3a).
    fn on_demand() -> Self {
        Self::with(1 << 20, Residency::OnDemand)
    }

    fn with(host: u64, residency: Residency) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let log: Log = Arc::default();
        let hold = Arc::new(AtomicBool::new(false));
        let mut registry = Registry::new();
        let sampled = Arc::default();
        registry.register(
            "voices",
            Box::new(Voices(Arc::clone(&log), Arc::clone(&hold), Arc::clone(&sampled))),
        );
        let profile = PlacementProfile {
            engines: vec![EnginePlacement {
                name: "voice".to_owned(),
                kind: "voices".to_owned(),
                devices: vec![0],
                config: toml::Table::new(),
                family: toml::Table::new(),
                lab: toml::Table::new(),
                share: 1,
                pin_identity: false,
                residency,
                pinned: false,
            }],
            budgets: vec![
                BudgetSpec {
                    tiers: vec![TierSpec::from(Tier::Device(0))],
                    bytes: 1000,
                },
                BudgetSpec {
                    tiers: vec![TierSpec::from(Tier::Host)],
                    bytes: host,
                },
            ],
            store: Some(StoreSpec {
                nvme: 1 << 20,
                host: None,
                interval_copies: 0,
                objects: None,
            }),
        };
        let inventory = Inventory {
            devices: vec![Device {
                id: 0,
                name: "device-0".to_owned(),
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 4096,
                unified: false,
                ignored: false,
            }],
            host_bytes: 1 << 20,
        };
        let mut process = Start::new(profile, registry, inventory)
            .store(directory.path())
            .heartbeat(Duration::from_millis(10))
            .timeout(Duration::from_secs(5))
            .start()
            .unwrap();
        process.wait_opens(Duration::from_secs(5)).unwrap();
        Self {
            process,
            log,
            hold,
            sampled,
            directory,
        }
    }

    /// A council that serves the speech model with a code book and an object store, and no text model.
    fn council(&self) -> Arc<Council> {
        let dir = self.directory.path();
        let service = self.service().with_codes(CodeBook::open(dir).unwrap());
        let objects = ObjectStore::open(&dir.join("objects"), 1 << 24).unwrap();
        Arc::new(
            Council::new(Vec::new(), None)
                .unwrap()
                .with_speech(vec![service])
                .with_objects(objects),
        )
    }

    fn service(&self) -> SpeechService {
        let mut config = SpeechConfig::new(MODEL, "voice", 2);
        config.context_positions = ROOM;
        config.snapshot_bytes = 8;
        SpeechService::new(config, Arc::new(Letters), self.process.ports()).unwrap()
    }

    /// Speak the last line of a context. Give the audio and the result.
    fn speak(&self, context: &SpeechContext) -> (Vec<i16>, Spoken) {
        let told = self.service().generate(context).expect("a generation starts");
        let mut audio = Vec::new();
        for item in told.iter() {
            match item.expect("a generation") {
                Told::Audio(samples) => audio.extend(samples),
                Told::End(spoken) => return (audio, spoken),
            }
        }
        panic!("the generation sent no end");
    }

    fn take_log(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

const VOICE: &str = "a calm ship";

fn first(line: &str, snap_voice: bool) -> SpeechContext {
    SpeechContext {
        voice: Some(VOICE.to_owned()),
        items: vec![ContextItem::Line(line.to_owned())],
        snaps: vec![snap_voice, false],
        language: "en".to_owned(),
        seed: 5,
        temperature: None,
        top_k: None,
        schedule: Schedule::default(),
    }
}

/// The context after `spoken`: its reply's audio, then the next line.
fn next(context: &SpeechContext, spoken: &Spoken, line: &str) -> SpeechContext {
    let mut next = context.clone();
    next.items.push(ContextItem::Audio(spoken.codes.clone()));
    next.items.push(ContextItem::Line(line.to_owned()));
    next.snaps.extend([false, false]);
    next.seed = 9;
    next
}

#[test]
fn a_first_line_feeds_the_voice_and_the_line_and_holds_the_head() {
    let fixture = Fixture::new();
    let context = first("hello", true);
    let (audio, spoken) = fixture.speak(&context);
    let voice = VOICE.len();
    let total = voice + "en".len() + "hello".len();
    assert_eq!((spoken.prompt, spoken.kept, spoken.fed), (total, 0, total));
    assert!((2..=4).contains(&spoken.codes.len()), "{} frames", spoken.codes.len());
    assert_eq!(audio.len(), spoken.codes.len() * FRAME);
    assert!(spoken.head.is_some(), "the head after the line is held");
    assert_eq!(
        spoken.held.iter().map(|(message, _, _)| *message).collect::<Vec<_>>(),
        [0],
        "the marked voice"
    );
    let log = fixture.take_log();
    assert_eq!(
        log,
        [
            format!("feed:0:{voice}:false"),
            format!("feed:0:{}:true", total - voice)
        ],
        "a feed to the marked voice with no sampler, then the line"
    );
}

#[test]
fn the_next_line_starts_from_the_head_and_speaks_as_a_whole_feed() {
    let held = Fixture::new();
    let one = first("hello", false);
    let (_, spoken) = held.speak(&one);
    let two = next(&one, &spoken, "world");
    held.take_log();
    let (from_head, again) = held.speak(&two);
    let head_rows = spoken.prompt + spoken.codes.len();
    assert_eq!(
        again.kept, head_rows,
        "the build starts from the head of the line before"
    );
    assert_eq!(
        again.fed,
        1 + "en".len() + "world".len(),
        "it feeds the codec end and the line"
    );
    assert_eq!(held.take_log()[0], format!("restore:0:{head_rows}"));

    // A process with an empty store feeds the whole context, and the line has the same codes.
    let fresh = Fixture::new();
    let (whole, alone) = fresh.speak(&two);
    assert_eq!(alone.kept, 0);
    assert_eq!(
        alone.codes, again.codes,
        "the same state gives the same codes, restored or fed"
    );
    assert_eq!(whole, from_head);
}

#[test]
fn a_held_voice_starts_another_first_line() {
    let fixture = Fixture::new();
    fixture.speak(&first("hello", true));
    let (_, other) = fixture.speak(&first("goodbye", false));
    assert_eq!(other.kept, VOICE.len(), "the build starts from the marked voice");
}

#[test]
fn a_context_that_does_not_end_with_a_line_is_refused() {
    let fixture = Fixture::new();
    let (_, spoken) = fixture.speak(&first("hello", false));
    let mut context = first("hello", false);
    context.items.push(ContextItem::Audio(spoken.codes));
    context.snaps.push(false);
    let error = fixture.service().generate(&context).unwrap_err();
    assert!(
        matches!(&error, SpeechError::Invalid { param: "messages", .. }),
        "{error:?}"
    );
}

#[test]
fn an_unmarked_voice_feeds_with_the_line_and_holds_only_the_head() {
    let fixture = Fixture::new();
    let (_, spoken) = fixture.speak(&first("hello", false));
    assert!(spoken.held.is_empty(), "{:?}", spoken.held);
    assert_eq!(fixture.take_log(), [format!("feed:0:{}:true", spoken.prompt)]);
}

// ---- the council routes of a speech model (spec 11.8a)

/// Send a request through the router. Give the status, the content type, the location and the whole body.
fn send(
    council: &Arc<Council>,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, String, String, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        if body.is_some() {
            request = request.header("content-type", "application/json");
        }
        let body = body.map_or_else(Body::empty, |value| Body::from(value.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let header = |name| {
            response
                .headers()
                .get(name)
                .map(|value: &axum::http::HeaderValue| value.to_str().unwrap().to_owned())
                .unwrap_or_default()
        };
        let (kind, location) = (header("content-type"), header("location"));
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24)
            .await
            .unwrap()
            .to_vec();
        (status, kind, location, bytes)
    })
}

fn json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(bytes)))
}

fn messages(lines: &[Value]) -> Value {
    let mut all = vec![json!({"role": "system", "content": VOICE, "snap": true})];
    all.extend(lines.iter().cloned());
    json!({"model": MODEL, "context": {"thinking": false, "messages": all}, "max_tokens": 1, "sample": {"seed": 5},
           "output": {"mime": "audio/wav"}})
}

fn generate(council: &Arc<Council>, body: Value) -> (StatusCode, Value) {
    let (status, _, _, bytes) = send(council, "POST", "/council/v2/generations", Some(body));
    (status, json(&bytes))
}

/// The status and `error.param` of a refused generation.
fn refused(council: &Arc<Council>, body: Value) -> (StatusCode, String) {
    let (status, v) = generate(council, body);
    (status, v["error"]["param"].as_str().unwrap_or("").to_owned())
}

#[test]
fn identity_lists_the_speech_model_as_a_writer_of_wav() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, _, _, bytes) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    let v = json(&bytes);
    let model = v["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model"] == MODEL)
        .expect("the speech model");
    assert_eq!(model["media"]["outputs"], json!(["audio/wav"]));
    assert_eq!(model["capabilities"], json!(["generate"]));
}

#[test]
fn a_generation_writes_a_wav_and_the_next_line_starts_from_its_head() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, done) = generate(&council, messages(&[json!({"role": "user", "content": "hello"})]));
    assert_eq!(status, StatusCode::OK, "{done}");
    let output = &done["outputs"][0];
    let id = output["id"].as_str().unwrap();
    let (status, kind, _, wav) = send(&council, "GET", &format!("/council/v2/objects/{id}"), None);
    assert_eq!((status, kind.as_str()), (StatusCode::OK, "audio/wav"));
    assert_eq!(
        format!("blake3:{}", blake3::hash(&wav).to_hex()),
        id,
        "the bytes hash to the id"
    );
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(
        u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize,
        wav.len() - 44,
        "the true data size"
    );
    assert_eq!(done["usage"]["media"]["written"]["objects"], json!(1));
    assert!(done["head"].is_string(), "{done}");

    // The client appends the reply, as `done` gives it, and the next line.
    let reply: GenerationDone = serde_json::from_value(done.clone()).unwrap();
    let reply = serde_json::to_value(reply.as_message().unwrap()).unwrap();
    let (status, next) = generate(
        &council,
        messages(&[
            json!({"role": "user", "content": "hello"}),
            reply,
            json!({"role": "user", "content": "world"}),
        ]),
    );
    assert_eq!(status, StatusCode::OK, "{next}");
    let frames = (wav.len() - 44) / 2 / FRAME;
    let head_rows = done["usage"]["prompt"].as_u64().unwrap() as usize + frames;
    assert_eq!(next["usage"]["kept"], json!(head_rows), "{}", next["usage"]);
    assert_eq!(next["usage"]["media"]["read"]["objects"], json!(1));
}

#[test]
/// The server cannot tell audio that it did not write from audio whose codes it dropped, so both are a 404 that
/// names the part (spec 11.8a, e3f99db).
fn audio_that_the_model_did_not_write_and_unheld_audio_are_a_404() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let bytes = b"RIFF not ours".to_vec();
    let id = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("PUT")
            .uri(format!("/council/v2/objects/{id}"))
            .header("content-type", "audio/wav")
            .body(Body::from(bytes))
            .unwrap();
        assert_eq!(
            router(Arc::clone(&council)).oneshot(request).await.unwrap().status(),
            StatusCode::CREATED
        );
    });
    let with = |id: &str| {
        messages(&[
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": [{"type": "object", "id": id, "mime": "audio/wav"}]}),
            json!({"role": "user", "content": "world"}),
        ])
    };
    let part = "messages[2].content[0]".to_owned();
    assert_eq!(refused(&council, with(&id)), (StatusCode::NOT_FOUND, part.clone()));
    let unheld = format!("blake3:{}", blake3::hash(b"never put").to_hex());
    assert_eq!(refused(&council, with(&unheld)), (StatusCode::NOT_FOUND, part));
    assert!(council.failure().is_none(), "{:?}", council.failure());
}

#[test]
fn a_generation_names_each_member_that_a_speech_model_does_not_take() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let line = json!({"role": "user", "content": "hello"});
    let mut no_output = messages(std::slice::from_ref(&line));
    no_output.as_object_mut().unwrap().remove("output");
    assert_eq!(
        refused(&council, no_output),
        (StatusCode::BAD_REQUEST, "output".to_owned())
    );
    let mut option = messages(std::slice::from_ref(&line));
    option["output"]["options"] = json!({"seconds": 3});
    assert_eq!(
        refused(&council, option),
        (StatusCode::BAD_REQUEST, "output.options.seconds".to_owned())
    );
    let mut warm = messages(std::slice::from_ref(&line));
    warm["sample"]["min_p"] = json!(0.5);
    assert_eq!(
        refused(&council, warm),
        (StatusCode::BAD_REQUEST, "sample.min_p".to_owned())
    );
    let mut image = messages(std::slice::from_ref(&line));
    image["output"]["mime"] = json!("image/png");
    assert_eq!(
        refused(&council, image),
        (StatusCode::BAD_REQUEST, "output.mime".to_owned())
    );
    let ends_with_audio = messages(&[line.clone(), json!({"role": "assistant", "content": "spoken"})]);
    assert_eq!(refused(&council, ends_with_audio).0, StatusCode::BAD_REQUEST);
    assert!(council.failure().is_none(), "{:?}", council.failure());
}

/// The events of a stream body, as name and data.
fn events(body: &[u8]) -> Vec<(String, Value)> {
    let text = String::from_utf8_lossy(body);
    let mut out = Vec::new();
    for block in text.split("\n\n").filter(|b| !b.trim().is_empty()) {
        let name = block
            .lines()
            .find_map(|l| l.strip_prefix("event: "))
            .unwrap_or("")
            .to_owned();
        let data: String = block.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
        out.push((name, serde_json::from_str(&data).unwrap_or(Value::Null)));
    }
    out
}

#[test]
fn a_stream_names_a_pending_id_that_streams_the_audio_and_then_moves_to_the_object() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let mut body = messages(&[json!({"role": "user", "content": "hello"})]);
    body["stream"] = json!(true);
    fixture.hold.store(true, Ordering::SeqCst);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let (stream_events, during, after) = runtime.block_on(async {
        use http_body_util::BodyExt;
        let request = Request::builder()
            .method("POST")
            .uri("/council/v2/generations")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(&council)).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut sse = response.into_body();
        let mut seen = Vec::new();
        let pending = loop {
            let frame = sse.frame().await.expect("an event").unwrap();
            seen.extend(frame.into_data().unwrap());
            if let Some((_, output)) = events(&seen).into_iter().find(|(name, _)| name == "output") {
                break output["pending"].as_str().unwrap().to_owned();
            }
        };
        // The line waits in the engine, so the pending id streams its header first.
        let request = Request::builder()
            .uri(format!("/council/v2/objects/{pending}"))
            .body(Body::empty())
            .unwrap();
        let response = router(Arc::clone(&council)).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut reading = response.into_body();
        let first = reading.frame().await.expect("the header").unwrap().into_data().unwrap();
        fixture.hold.store(false, Ordering::SeqCst);
        let mut during = first.to_vec();
        while let Some(frame) = reading.frame().await {
            during.extend(frame.unwrap().into_data().unwrap());
        }
        while let Some(frame) = sse.frame().await {
            seen.extend(frame.unwrap().into_data().unwrap());
        }
        (events(&seen), during, pending)
    });
    let pending = after;
    let done = &stream_events
        .iter()
        .find(|(name, _)| name == "done")
        .expect("a done event")
        .1;
    let id = done["outputs"][0]["id"].as_str().unwrap();
    let (_, _, _, wav) = send(&council, "GET", &format!("/council/v2/objects/{id}"), None);
    assert_eq!(&during[4..8], [0xff; 4], "the stream's header has the unknown sizes");
    assert_eq!(
        during[44..],
        wav[44..],
        "the stream has the samples of the sealed object"
    );
    assert_eq!(during[8..40], wav[8..40]);
    let (status, _, location, _) = send(&council, "GET", &format!("/council/v2/objects/{pending}"), None);
    assert_eq!(status, StatusCode::PERMANENT_REDIRECT);
    assert_eq!(location, format!("/council/v2/objects/{id}"));
}

// ---- the paths that the kaibo review of 9033ed3 found (job-6)

#[test]
fn a_stop_during_a_line_cancels_it_frees_its_slot_and_is_no_internal_error() {
    let mut fixture = Fixture::new();
    let council = fixture.council();
    fixture.hold.store(true, Ordering::SeqCst);
    let log = Arc::clone(&fixture.log);
    let hold = Arc::clone(&fixture.hold);
    let (status, failure) = std::thread::scope(|scope| {
        let request = scope.spawn(|| {
            let line = messages(&[json!({"role": "user", "content": "hello"})]);
            send(&council, "POST", "/council/v2/generations", Some(line)).0
        });
        // The job is in its frames, which wait.
        while !log.lock().unwrap().iter().any(|line| line.ends_with(":true")) {
            std::thread::sleep(Duration::from_millis(2));
        }
        let release = scope.spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            hold.store(false, Ordering::SeqCst);
        });
        fixture
            .process
            .stop(Duration::from_millis(50), &nagaya::signal::SignalFlag::new());
        release.join().unwrap();
        (request.join().unwrap(), council.failure())
    });
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(failure.is_none(), "a stop is no internal error: {failure:?}");
    assert_eq!(
        council.speech_service(MODEL).unwrap().free_slots(),
        2,
        "the slot came back"
    );
}

#[test]
fn audio_whose_codes_are_gone_is_a_404_that_names_the_part() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (_, done) = generate(&council, messages(&[json!({"role": "user", "content": "hello"})]));
    let id = done["outputs"][0]["id"].as_str().unwrap();
    let file = fixture
        .directory
        .path()
        .join("speech-codes")
        .join(id.trim_start_matches("blake3:"));
    std::fs::remove_file(&file).unwrap();
    let reply = json!({"role": "assistant", "content": [{"type": "object", "id": id, "mime": "audio/wav"}]});
    let next = messages(&[
        json!({"role": "user", "content": "hello"}),
        reply,
        json!({"role": "user", "content": "x"}),
    ]);
    assert_eq!(
        refused(&council, next),
        (StatusCode::NOT_FOUND, "messages[2].content[0]".to_owned())
    );
}

#[test]
fn an_empty_line_is_refused_and_a_batch_line_speaks() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let empty = messages(&[json!({"role": "user", "content": "  "})]);
    assert_eq!(
        refused(&council, empty),
        (StatusCode::BAD_REQUEST, "messages[1].content".to_owned())
    );
    let mut batch = messages(&[json!({"role": "user", "content": "hello"})]);
    batch["class"] = json!("batch");
    assert_eq!(generate(&council, batch).0, StatusCode::OK);
}

#[test]
fn a_context_past_the_slot_is_context_full() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let long = "a".repeat(ROOM as usize);
    let (status, v) = generate(&council, messages(&[json!({"role": "user", "content": long})]));
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
    assert_eq!(v["error"]["type"], json!("context_full"));
    assert_eq!(v["error"]["context_tokens"], json!(ROOM));
}

#[test]
fn a_refused_copy_of_a_snapshot_is_busy_and_no_internal_error() {
    // The voice of 11 positions is 88 bytes of state, past a host budget of 16 bytes.
    let fixture = Fixture::with_host(16);
    let council = fixture.council();
    let (status, v) = generate(&council, messages(&[json!({"role": "user", "content": "hello"})]));
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{v}");
    assert!(council.failure().is_none(), "{:?}", council.failure());
    assert_eq!(council.speech_service(MODEL).unwrap().free_slots(), 2);
}

/// The gogi suite against the speech model, served in process over HTTP: a model with no text replies (gogi
/// `media.text_replies`).
#[test]
fn the_gogi_suite_passes_for_the_speech_model() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = runtime.spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(listener, router(council)).await.unwrap();
    });
    let url = format!("http://{address}");
    let report = std::thread::spawn(move || {
        gogi_conformance::run(
            gogi_conformance::Opts {
                url,
                restart: None,
                model: Some(MODEL.to_owned()),
            },
            true,
        )
    })
    .join()
    .unwrap();
    server.abort();
    assert_eq!(report.failures(), 0, "{}", report.summary());
}

#[test]
fn temperature_0_samples_greedily_and_top_k_and_temperature_reach_the_engine() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let line = || messages(&[json!({"role": "user", "content": "hello"})]);
    let mut greedy = line();
    greedy["sample"]["temperature"] = json!(0);
    assert_eq!(generate(&council, greedy).0, StatusCode::OK);
    assert_eq!(
        *fixture.sampled.lock().unwrap(),
        Some((1, 1.0)),
        "temperature 0 is top_k 1"
    );
    let mut warm = line();
    warm["sample"]["temperature"] = json!(0.5);
    warm["sample"]["top_k"] = json!(7);
    assert_eq!(generate(&council, warm).0, StatusCode::OK);
    assert_eq!(*fixture.sampled.lock().unwrap(), Some((7, 0.5)));
    assert_eq!(generate(&council, line()).0, StatusCode::OK);
    assert_eq!(
        *fixture.sampled.lock().unwrap(),
        Some((50, 0.9)),
        "the checkpoint's sampler"
    );
    let mut none = line();
    none["sample"]["top_k"] = json!(0);
    assert_eq!(
        refused(&council, none),
        (StatusCode::BAD_REQUEST, "sample.top_k".to_owned())
    );
}

#[test]
fn an_unloaded_speech_model_is_unloaded_and_its_generation_is_a_503() {
    let fixture = Fixture::on_demand();
    let council = fixture.council();
    let status = |council: &Arc<Council>| {
        let (_, _, _, bytes) = send(council, "GET", "/council/v2/identity", None);
        let v = json(&bytes);
        let model = v["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["model"] == MODEL)
            .cloned();
        model.expect("the speech model")["status"].clone()
    };
    assert_eq!(status(&council), json!("unloaded"));
    let (code, body) = generate(&council, messages(&[json!({"role": "user", "content": "hello"})]));
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["type"], json!("model_unavailable"), "{body}");
    assert_eq!(body["error"]["model"], json!(MODEL), "{body}");
    assert_eq!(body["error"]["status"], json!("unloaded"), "{body}");
    assert!(fixture.take_log().is_empty(), "nothing ran");
    // Any route that names the model gives the same 503 first (gogi "models": a build naming a model that is not
    // ready).
    let build =
        json!({"model": MODEL, "context": {"thinking": false, "messages": [{"role": "user", "content": "hello"}]}});
    let (code, _, _, bytes) = send(&council, "POST", "/council/v2/contexts", Some(build));
    let body = json(&bytes);
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["type"], json!("model_unavailable"), "{body}");
    assert_eq!(body["error"]["model"], json!(MODEL), "{body}");
    assert_eq!(body["error"]["status"], json!("unloaded"), "{body}");

    fixture.process.load("voice").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while fixture.process.ports().engine_state("voice") != Some(nagaya::notify::EngineState::Ready) {
        assert!(
            std::time::Instant::now() < deadline,
            "the speech engine did not load in 5 s"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(status(&council), json!("ready"));
    // The served model shows the hashes of its engine from its open, before any line (spec 3.3a).
    let (_, _, _, bytes) = send(&council, "GET", "/council/v2/identity", None);
    let models = json(&bytes)["models"].clone();
    let model = models
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model"] == MODEL)
        .unwrap()
        .clone();
    assert_eq!(
        model["weight_hash"],
        json!(FakeAlways::new().identity().checkpoint_digest),
        "{model}"
    );
    let (code, body) = generate(&council, messages(&[json!({"role": "user", "content": "hello"})]));
    assert_eq!(code, StatusCode::OK, "{body}");
}
