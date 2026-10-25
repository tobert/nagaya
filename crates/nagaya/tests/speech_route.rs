//! The speech route `POST /v1/audio/speech` in the process, with a fake speech engine and a fake family (spec 11.8).
//! The speech engine is on demand, and it shares its device with a resident engine, so a request swaps it in.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::http::speech::wav_header;
use nagaya::http::{Council, router};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::speech::{Chunk, SpeechConfig, SpeechError, SpeechFamily, SpeechRequest, SpeechService};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, Identity, Lifetime, Limits, MandatoryTerm, Parts, PassError, PassErrorKind, PassOut,
    PromptPair, Seal, Segment, SlotId, SlotLimit, Slots, Speech, SpeechOut, SpeechSampler, Tap, Tier,
};
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "fake-speech";
/// The samples of one fake frame.
const FRAME: usize = 4;
/// The most frames of one `speech_frames` call of the fake.
const QUANTUM: usize = 3;
/// The most prompt positions that the fake takes.
const MAX_PROMPT: usize = 12;

type Log = Arc<Mutex<Vec<String>>>;

/// A speech engine that makes one frame for each prompt position. Sample `i` of frame `f` is `seed + 10 f + i`.
struct FakeSpeech {
    log: Log,
    hold: Arc<AtomicBool>,
    slots: Vec<Option<(usize, usize, u64)>>,
}

impl Speech for FakeSpeech {
    fn speech_begin(&mut self, slot: SlotId, prompt: &[PromptPair], sampler: &SpeechSampler) -> Result<(), PassError> {
        if prompt.len() > MAX_PROMPT {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: format!(
                    "the prompt has {} positions, and the engine takes 1 to {MAX_PROMPT}",
                    prompt.len()
                ),
            });
        }
        assert!(self.slots[slot as usize].is_none(), "slot {slot} holds a request");
        self.log.lock().unwrap().push(format!("begin:{slot}"));
        self.slots[slot as usize] = Some((prompt.len(), 0, sampler.seed));
        Ok(())
    }

    fn speech_frames(&mut self, slot: SlotId, max: u32) -> Result<SpeechOut, PassError> {
        self.log.lock().unwrap().push(format!("frames:{slot}"));
        while self.hold.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let (total, done, seed) = self.slots[slot as usize].expect("a request in the slot");
        let n = (total - done).min(QUANTUM).min(max as usize);
        let mut audio = Vec::new();
        for f in done..done + n {
            audio.extend((0..FRAME).map(|i| (seed as i64 + 10 * f as i64 + i as i64) as i16));
        }
        self.slots[slot as usize] = Some((total, done + n, seed));
        Ok(SpeechOut {
            frames: n as u32,
            codes: vec![0; n * 16],
            audio,
            end: done + n == total,
        })
    }

    fn speech_clear(&mut self, slot: SlotId) {
        if self.slots[slot as usize].take().is_some() {
            self.log.lock().unwrap().push(format!("clear:{slot}"));
        }
    }
}

fn limits(bytes: u64, speech: bool) -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Close);
    if speech {
        limits.capabilities = vec![Capability::Speech, Capability::Close];
        limits.slots = vec![SlotLimit {
            bytes: 0,
            count: 2,
            tokens: 64,
        }];
    }
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic = Vec::new();
    limits
}

impl Engine for FakeSpeech {
    fn identity(&self) -> Identity {
        let mut identity = FakeAlways::new().identity();
        identity.model = MODEL.to_owned();
        identity
    }

    fn limits(&self) -> Limits {
        limits(600, true)
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
}

/// A resident engine with no speech, as the chat engine of the machine.
struct Chat(FakeAlways);

impl Engine for Chat {
    fn identity(&self) -> Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        limits(600, false)
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.0.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.0.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.0.coverage()
    }

    fn seal(&mut self) -> Option<&mut dyn Seal> {
        self.0.seal()
    }
}

struct Factories {
    log: Log,
    hold: Arc<AtomicBool>,
}

impl Factory for Factories {
    fn plan(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits(600, config["speech"].as_bool().unwrap()))
    }

    fn open(&self, config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        if config["speech"].as_bool().unwrap() {
            self.log.lock().unwrap().push("open:voice".to_owned());
            Ok(Box::new(FakeSpeech {
                log: Arc::clone(&self.log),
                hold: Arc::clone(&self.hold),
                slots: vec![None, None],
            }))
        } else {
            self.log.lock().unwrap().push("open:chat".to_owned());
            Ok(Box::new(Chat(FakeAlways::new())))
        }
    }
}

/// A family that gives one prompt position for each character of the text.
struct Letters;

impl SpeechFamily for Letters {
    fn prompt(&self, text: &str, _instructions: &str, language: &str) -> Result<Vec<PromptPair>, SpeechError> {
        if !matches!(language, "auto" | "english") {
            return Err(SpeechError::Invalid {
                param: "language",
                message: format!("`{language}` is not a language of the engine"),
            });
        }
        Ok(text
            .chars()
            .map(|c| PromptPair {
                text: Some(c as u32),
                codec: None,
            })
            .collect())
    }
}

fn engine(name: &str, speech: bool, residency: Residency) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("speech".to_owned(), toml::Value::Boolean(speech));
    EnginePlacement {
        name: name.to_owned(),
        kind: "fakes".to_owned(),
        devices: vec![0],
        config,
        family: toml::Table::new(),
        share: 1,
        pin_identity: false,
        residency,
        idle_unload_s: None,
    }
}

struct Fixture {
    process: Process,
    log: Log,
    hold: Arc<AtomicBool>,
    _directory: tempfile::TempDir,
}

impl Fixture {
    /// A process with a resident chat engine and an on-demand speech engine, which do not fit on the device together.
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let log: Log = Arc::default();
        let hold = Arc::new(AtomicBool::new(false));
        let mut registry = Registry::new();
        registry.register(
            "fakes",
            Box::new(Factories {
                log: Arc::clone(&log),
                hold: Arc::clone(&hold),
            }),
        );
        let profile = PlacementProfile {
            engines: vec![
                engine("chat", false, Residency::Resident),
                engine("voice", true, Residency::OnDemand),
            ],
            budgets: vec![
                BudgetSpec {
                    tiers: vec![TierSpec::from(Tier::Device(0))],
                    bytes: 1000,
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
            host_bytes: 4096,
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
            _directory: directory,
        }
    }

    fn service(&self, slots: u32, timeout: Duration) -> SpeechService {
        let mut config = SpeechConfig::new(MODEL, "voice", slots);
        config.timeout = timeout;
        SpeechService::new(config, Arc::new(Letters), self.process.ports()).unwrap()
    }

    fn council(&self) -> Arc<Council> {
        let service = self.service(2, Duration::from_secs(5));
        Arc::new(Council::new(Vec::new(), None).unwrap().with_speech(vec![service]))
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

/// Send one speech request through the router. Give the status, the content type and the whole body.
fn send(council: &Arc<Council>, body: Value) -> (StatusCode, String, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let kind = response
            .headers()
            .get("content-type")
            .map_or_else(String::new, |v| v.to_str().unwrap().to_owned());
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, kind, bytes.to_vec())
    })
}

fn request(input: &str) -> Value {
    json!({"model": MODEL, "input": input, "instructions": "A calm voice.", "voice": "alloy", "seed": 100})
}

/// The samples that the fake makes for a text of `frames` characters with `seed`.
fn samples(frames: usize, seed: i64) -> Vec<u8> {
    (0..frames)
        .flat_map(|f| (0..FRAME).map(move |i| (seed + 10 * f as i64 + i as i64) as i16))
        .flat_map(|s| s.to_le_bytes())
        .collect()
}

#[test]
fn a_request_swaps_the_engine_in_and_streams_wav_of_every_quantum() {
    let f = Fixture::new();
    let council = f.council();
    let (status, kind, body) = send(&council, request("Hello, ship."));
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(kind, "audio/wav");
    assert_eq!(&body[..44], &wav_header()[..]);
    assert_eq!(
        &body[44..],
        &samples(12, 100)[..],
        "the audio of 12 frames in quanta of 3"
    );
    assert_eq!(
        f.log(),
        [
            "open:chat",
            "open:voice",
            "begin:0",
            "frames:0",
            "frames:0",
            "frames:0",
            "frames:0",
            "clear:0"
        ]
    );
    // The header: RIFF, PCM, mono, 24 kHz, 16 bits, and the two sizes unknown.
    assert_eq!(&body[0..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(body[4..8].try_into().unwrap()), u32::MAX);
    assert_eq!(u32::from_le_bytes(body[24..28].try_into().unwrap()), 24_000);
    assert_eq!(u16::from_le_bytes(body[34..36].try_into().unwrap()), 16);
    assert_eq!(u32::from_le_bytes(body[40..44].try_into().unwrap()), u32::MAX);
}

#[test]
fn pcm_has_no_header_and_the_same_seed_gives_the_same_bytes() {
    let f = Fixture::new();
    let council = f.council();
    let mut pcm = request("Ship.");
    pcm["response_format"] = json!("pcm");
    let (status, kind, first) = send(&council, pcm.clone());
    assert_eq!((status, kind.as_str()), (StatusCode::OK, "audio/pcm"));
    assert_eq!(first, samples(5, 100));
    let (_, _, again) = send(&council, pcm.clone());
    assert_eq!(again, first);
    // With no seed, the seed is the hash of the request id, so two requests differ.
    pcm.as_object_mut().unwrap().remove("seed");
    let (_, _, one) = send(&council, pcm.clone());
    let (_, _, two) = send(&council, pcm);
    assert_ne!(one, two);
}

#[test]
fn each_rule_of_the_request_gives_an_openai_error_that_names_its_param() {
    let f = Fixture::new();
    let council = f.council();
    let with = |key: &str, value: Value| {
        let mut body = request("Ship.");
        body[key] = value;
        body
    };
    let mut no_model = request("Ship.");
    no_model.as_object_mut().unwrap().remove("model");
    let mut no_instructions = request("Ship.");
    no_instructions.as_object_mut().unwrap().remove("instructions");
    let cases: Vec<(Value, StatusCode, Option<&str>, Option<&str>)> = vec![
        (no_model, StatusCode::BAD_REQUEST, Some("model"), None),
        (
            with("model", json!("tts-1")),
            StatusCode::NOT_FOUND,
            Some("model"),
            Some("model_not_found"),
        ),
        (with("input", json!("  ")), StatusCode::BAD_REQUEST, Some("input"), None),
        (
            with("input", json!("a".repeat(4097))),
            StatusCode::BAD_REQUEST,
            Some("input"),
            None,
        ),
        (no_instructions, StatusCode::BAD_REQUEST, Some("instructions"), None),
        (
            with("response_format", json!("mp3")),
            StatusCode::BAD_REQUEST,
            Some("response_format"),
            None,
        ),
        (with("speed", json!(1.5)), StatusCode::BAD_REQUEST, Some("speed"), None),
        (
            with("language", json!("klingon")),
            StatusCode::BAD_REQUEST,
            Some("language"),
            None,
        ),
        (with("seed", json!(-1)), StatusCode::BAD_REQUEST, Some("seed"), None),
        // The engine refuses a prompt past its limit, before any audio.
        (
            with("input", json!("a".repeat(13))),
            StatusCode::BAD_REQUEST,
            Some("input"),
            None,
        ),
    ];
    for (body, status, param, code) in cases {
        let (got, kind, bytes) = send(&council, body.clone());
        assert_eq!(got, status, "{body}: {}", String::from_utf8_lossy(&bytes));
        assert!(kind.starts_with("application/json"), "{kind}");
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        let error = &error["error"];
        assert!(error["message"].as_str().is_some_and(|m| !m.is_empty()), "{error}");
        assert!(error["type"].is_string(), "{error}");
        assert_eq!(error["param"].as_str(), param, "{error}");
        assert_eq!(error["code"].as_str(), code, "{error}");
    }
    // `speed` 1.0 and the voice of any name are fine.
    let (status, _, _) = send(&council, with("speed", json!(1.0)));
    assert_eq!(status, StatusCode::OK);
    // A refused request gives its slot back: both slots are free.
    assert_eq!(council.speech_service(MODEL).unwrap().free_slots(), 2);
}

fn speak(service: &SpeechService, input: &str) -> Result<Vec<Chunk>, SpeechError> {
    let request = SpeechRequest {
        input: input.to_owned(),
        instructions: "A calm voice.".to_owned(),
        language: "auto".to_owned(),
        seed: 1,
    };
    let chunks = service.start(&request)?;
    let mut out = Vec::new();
    for item in chunks {
        let end = matches!(item, Ok(Chunk::End));
        out.push(item?);
        if end {
            break;
        }
    }
    Ok(out)
}

#[test]
fn a_full_wait_for_a_slot_is_busy_and_a_late_job_times_out() {
    let f = Fixture::new();
    let service = Arc::new(f.service(1, Duration::from_secs(1)));
    f.hold.store(true, Ordering::SeqCst);
    let held = {
        let service = Arc::clone(&service);
        std::thread::spawn(move || speak(&service, "Ship."))
    };
    // The first request holds the only slot while its engine holds its frames. The second request starts only when
    // the first job is in that quantum, so the first job cannot expire before the hold and free the slot.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !f.log().contains(&"frames:0".to_owned()) {
        assert!(std::time::Instant::now() < deadline, "the first job made no frames");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(service.free_slots(), 0);
    let error = speak(&service, "Ship.").unwrap_err();
    assert!(matches!(error, SpeechError::Busy(_)), "{error:?}");
    f.hold.store(false, Ordering::SeqCst);
    // The hold outlasted the timeout of the first request too, so its job expired after its first quantum.
    let error = held.join().unwrap().unwrap_err();
    assert!(matches!(error, SpeechError::Timeout(_)), "{error:?}");
    // The expired job gave its slot back, and the next job clears the slot on the engine before it begins.
    assert_eq!(speak(&service, "Hi.").unwrap().last(), Some(&Chunk::End));
    assert_eq!(service.free_slots(), 1);
}

#[test]
fn a_client_that_goes_away_stops_the_job_and_frees_its_slot() {
    let f = Fixture::new();
    let service = f.service(1, Duration::from_secs(5));
    let request = SpeechRequest {
        input: "Twelve chars".to_owned(),
        instructions: "A calm voice.".to_owned(),
        language: "auto".to_owned(),
        seed: 1,
    };
    // The engine holds the first quantum while the client goes away.
    f.hold.store(true, Ordering::SeqCst);
    let chunks = service.start(&request).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !f.log().contains(&"frames:0".to_owned()) {
        assert!(std::time::Instant::now() < deadline, "the job made no frames");
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(chunks);
    f.hold.store(false, Ordering::SeqCst);
    while service.free_slots() != 1 {
        assert!(std::time::Instant::now() < deadline, "the slot did not come back");
        std::thread::sleep(Duration::from_millis(5));
    }
    // The job made one quantum of its four, and it cleared its slot on the engine.
    let log = f.log();
    assert_eq!(log[2..], ["begin:0", "frames:0", "clear:0"], "{log:?}");
    assert_eq!(speak(&service, "Hi.").unwrap().last(), Some(&Chunk::End));
}
