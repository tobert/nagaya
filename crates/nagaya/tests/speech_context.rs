//! Generations of a speech context in the process (spec 11.8a), with a fake speech engine that holds state as the
//! real one does: one row of state for each position, read and written through a cells part.
//!
//! The fake's row is a hash of the row before it and of the position. The codes of a frame are a hash of the last row,
//! the seed and the frame. Thus a line gives the same codes only from the same state, fed or restored.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya::speech::{
    ContextItem, ContextRendering, SpeechConfig, SpeechContext, SpeechError, SpeechFamily, SpeechService, Spoken, Told,
};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Dtype, Engine, Identity, Lifetime, Limits, MandatoryTerm, PartError, PartId, PartKind, PartLimits,
    Parts, PassError, PassErrorKind, PassOut, PromptPair, Segment, SlotId, SlotLimit, Slots, Speech, SpeechOut,
    SpeechPosition, SpeechSampler, State, Tap, Tier, WriteResult,
};

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

struct Voices(Log);

impl Factory for Voices {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Voice {
            log: Arc::clone(&self.0),
            slots: vec![SlotState::default(), SlotState::default()],
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
    fn prompt(&self, text: &str, _instructions: &str, _language: &str) -> Result<Vec<PromptPair>, SpeechError> {
        Ok(text
            .chars()
            .map(|c| PromptPair {
                text: Some(c as u32),
                codec: None,
            })
            .collect())
    }

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
    _directory: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let log: Log = Arc::default();
        let mut registry = Registry::new();
        registry.register("voices", Box::new(Voices(Arc::clone(&log))));
        let profile = PlacementProfile {
            engines: vec![EnginePlacement {
                name: "voice".to_owned(),
                kind: "voices".to_owned(),
                devices: vec![0],
                config: toml::Table::new(),
                family: toml::Table::new(),
                share: 1,
                pin_identity: false,
                residency: Residency::Resident,
                idle_unload_s: None,
            }],
            budgets: vec![
                BudgetSpec {
                    tiers: vec![TierSpec::from(Tier::Device(0))],
                    bytes: 1000,
                },
                BudgetSpec {
                    tiers: vec![TierSpec::from(Tier::Host)],
                    bytes: 1 << 20,
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
            _directory: directory,
        }
    }

    fn service(&self) -> SpeechService {
        SpeechService::new(
            SpeechConfig::new(MODEL, "voice", 2),
            Arc::new(Letters),
            self.process.ports(),
        )
        .unwrap()
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
        spoken.held.iter().map(|(message, _)| *message).collect::<Vec<_>>(),
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
