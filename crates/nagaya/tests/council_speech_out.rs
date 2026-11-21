//! Generations with an `output` of `audio/wav` of a served text model that speaks (spec 11.11, "The generation modes"
//! and "The speech loop of a generation"). The engine is a fake with the capability `speech_frames`, over the toy
//! vocabulary and the toy family. Four toy bytes stand for the markers of LFM3, so no card and no model folder is
//! needed.

mod council_support;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ENGINE, Script, inventory, profile_with_copies};
use gogi_proto::generation::{Finish, GenerationDone, GenerationRequest, StreamEvent, TokenEvent};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::council::error::ServiceError;
use nagaya::council::speech_out::{SpeechOut, Spoken};
use nagaya::council::{Clock, Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya::process::{Process, Start};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Coverage, Engine, FrameDraw, FrameOut, Identity, Limits, OutputsWanted, Parts, PassError,
    PassErrorKind, PassOut, Seal, Segment, SegmentInputs, SlotId, Slots, SpeechFrames, State, Tap,
};
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};
use tower::ServiceExt;

/// The toy bytes that stand for `<|audio_start|>`, `<|mixed_start|>`, `<|text_end|>` and `<|audio_out|>`.
const AUDIO_START: u32 = b'{' as u32;
const MIXED_START: u32 = b'}' as u32;
const TEXT_END: u32 = b'|' as u32;
const AUDIO_OUT: u32 = b'~' as u32;

/// The end code of LFM3.
const END_CODE: u32 = 2016;
const CODEBOOKS: usize = 12;

fn speech() -> SpeechOut {
    SpeechOut {
        audio_start: AUDIO_START,
        mixed_start: MIXED_START,
        text_end: TEXT_END,
        audio_out: AUDIO_OUT,
        n_text: 6,
        n_audio: 12,
    }
}

/// What the fake engine says and saw.
#[derive(Debug, Default)]
struct Plan {
    /// The text that the engine favors after a marker: the token after `k` text tokens is `text[k]`.
    text: Vec<u32>,
    /// The text that the engine favors with no marker, from the position `plain.0`.
    plain: Option<(u64, Vec<u32>)>,
    /// The frame index at which the engine gives the end frame.
    end_at: u64,
    /// Each frame call: its seed, its frame, the length of its history, and whether it asked for logits.
    calls: Vec<(u64, u64, usize, bool)>,
    /// The tokens of the slot after its last marker, after each pass and frame.
    after_marker: Vec<u32>,
    /// The last marker of the slot.
    marker: Option<u32>,
    /// The codec calls: the count of `codec` calls and of `codec_clear` calls.
    codecs: usize,
    clears: usize,
}

type Shared = Arc<Mutex<Plan>>;

/// A fake engine whose frames come from a hash of the seed, the frame and the codebook.
struct Speaking {
    inner: FakeAlways,
    plan: Shared,
    /// The tokens of each slot, by position.
    tokens: HashMap<SlotId, Vec<u32>>,
    /// The frames that the codec of each slot decoded since its clear.
    codec_frames: HashMap<SlotId, usize>,
}

fn code(seed: u64, frame: u64, g: usize) -> u32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(&frame.to_le_bytes());
    hasher.update(&[g as u8]);
    let hash = hasher.finalize();
    u32::from_le_bytes(hash.as_bytes()[..4].try_into().unwrap()) % END_CODE
}

impl Speaking {
    fn slot_of(segment: &Segment) -> SlotId {
        let State::Slot(slot) = segment.state;
        slot
    }

    /// Record the tokens of a segment, and favor the scripted token on each logit row.
    fn record_and_favor(&mut self, segment: &Segment, out: &mut PassOut) {
        let slot = Self::slot_of(segment);
        let SegmentInputs::Tokens(tokens) = &segment.inputs else {
            panic!("the council feeds tokens");
        };
        let history = self.tokens.entry(slot).or_default();
        history.truncate(segment.positions[0] as usize);
        history.extend(tokens);
        let mut plan = self.plan.lock().unwrap();
        let marker = history.iter().rposition(|t| *t == AUDIO_START || *t == MIXED_START);
        plan.after_marker = marker.map_or_else(Vec::new, |m| history[m + 1..].to_vec());
        plan.marker = marker.map(|m| history[m]);
        let rows = out.outputs.iter_mut().filter(|output| output.name == "logits");
        for (position, output) in segment.outputs.logit_rows.iter().zip(rows) {
            let p = *position as usize;
            let favored = match marker.filter(|m| *m <= p) {
                Some(m) => {
                    let k = history[m + 1..=p].iter().filter(|t| **t != AUDIO_OUT).count();
                    plan.text.get(k).copied()
                }
                None => plan
                    .plain
                    .as_ref()
                    .and_then(|(base, text)| (p as u64 + 1).checked_sub(*base).and_then(|k| text.get(k as usize)))
                    .copied(),
            };
            if let Some(token) = favored {
                let at = token as usize * 4;
                output.bytes[at..at + 4].copy_from_slice(&Script::LOGIT.to_le_bytes());
            }
        }
    }
}

impl Engine for Speaking {
    fn identity(&self) -> Identity {
        self.inner.identity()
    }

    fn limits(&self) -> Limits {
        let mut limits = self.inner.limits();
        limits.capabilities.push(Capability::SpeechFrames);
        limits
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let mut out = self.inner.pass(segments, taps)?;
        if let Some(segment) = segments.first() {
            self.record_and_favor(segment, &mut out);
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
        self.inner.seal()
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.inner.slots()
    }

    fn frames(&mut self) -> Option<&mut dyn SpeechFrames> {
        Some(self)
    }
}

impl SpeechFrames for Speaking {
    fn codebooks(&self) -> u32 {
        CODEBOOKS as u32
    }

    fn frame(&mut self, slot: SlotId, draw: &FrameDraw<'_>, logits: bool) -> Result<FrameOut, PassError> {
        let end = {
            let mut plan = self.plan.lock().unwrap();
            plan.calls.push((draw.seed, draw.frame, draw.history.len(), logits));
            draw.frame >= plan.end_at
        };
        let codes: Vec<u32> = if end {
            vec![END_CODE; CODEBOOKS]
        } else {
            (0..CODEBOOKS).map(|g| code(draw.seed, draw.frame, g)).collect()
        };
        let n = self.inner.slots().unwrap().slot_high(slot);
        // A frame takes no control list (spec 11.12, "Omni").
        let segment = Segment {
            control: Vec::new(),
            state: State::Slot(slot),
            inputs: SegmentInputs::Tokens(vec![AUDIO_OUT]),
            positions: vec![n],
            meta: Vec::new(),
            outputs: OutputsWanted {
                logit_rows: if logits { vec![n] } else { Vec::new() },
            },
            layer_loop: None,
            cycles: None,
        };
        let mut out = self.inner.pass(std::slice::from_ref(&segment), &[])?;
        self.record_and_favor(&segment, &mut out);
        let logits = logits.then(|| {
            let row = out.outputs.iter().find(|o| o.name == "logits").expect("a row");
            row.bytes
                .chunks(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        });
        Ok(FrameOut { codes, end, logits })
    }

    fn codec(&mut self, slot: SlotId, codes: &[u32]) -> Result<Vec<f32>, PassError> {
        if codes.iter().any(|c| *c >= END_CODE) {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: "the end frame makes no audio".to_owned(),
            });
        }
        self.plan.lock().unwrap().codecs += 1;
        let frames = self.codec_frames.entry(slot).or_default();
        let count = if *frames == 0 { 1440 } else { 1920 };
        *frames += 1;
        Ok(vec![codes[0] as f32 / END_CODE as f32 - 0.5; count])
    }

    fn codec_clear(&mut self, slot: SlotId) -> Vec<f32> {
        self.plan.lock().unwrap().clears += 1;
        match self.codec_frames.insert(slot, 0) {
            Some(n) if n > 0 => vec![0.25; 480],
            _ => Vec::new(),
        }
    }
}

struct SpeakingFactory(Shared);

impl SpeakingFactory {
    fn engine(&self) -> Speaking {
        Speaking {
            inner: FakeAlways::with_slots(council_support::roomy_slots(2)).with_vocab(gogi_toy::VOCAB),
            plan: Arc::clone(&self.0),
            tokens: HashMap::new(),
            codec_frames: HashMap::new(),
        }
    }
}

impl Factory for SpeakingFactory {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(self.engine().limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(self.engine()))
    }
}

struct Rig {
    service: Arc<Service>,
    plan: Shared,
    process: Process,
    directory: tempfile::TempDir,
}

fn config() -> ServiceConfig {
    let mut config = ServiceConfig::new("toy", ENGINE);
    config.speech = Some(speech());
    // A reply with audio takes the whole slot (spec 11.11, "The frame limit"); the fake's slots hold 1,024 tokens.
    config.context_tokens = 1024;
    config
}

fn rig_with(config: ServiceConfig) -> Rig {
    let plan: Shared = Arc::default();
    let directory = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register("term", Box::new(SpeakingFactory(Arc::clone(&plan))));
    let profile = profile_with_copies(1 << 20, 1 << 24, None, 0);
    let mut process = Start::new(profile, registry, inventory())
        .store(directory.path())
        .heartbeat(Duration::from_millis(10))
        .timeout(Duration::from_secs(10))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(10)).unwrap();
    let clock: Clock = Arc::new(|| 1_000_000);
    let service = Service::attach_with_clock(
        process.ports(),
        Arc::new(ToyFamily::new()),
        directory.path(),
        config,
        clock,
    )
    .unwrap();
    Rig {
        service: Arc::new(service),
        plan,
        process,
        directory,
    }
}

fn rig() -> Rig {
    rig_with(config())
}

fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn request(mode: Option<&str>, max_tokens: u32, seed: u64) -> GenerationRequest {
    let mut output = json!({"mime": "audio/wav"});
    if let Some(mode) = mode {
        output["options"] = json!({"mode": mode});
    }
    serde_json::from_value(json!({
        "model": "toy",
        "context": {"thinking": false, "messages": [
            {"role": "system", "content": "Respond with interleaved text and audio."},
            {"role": "user", "content": "Say hello.", "snap": true}
        ]},
        "max_tokens": max_tokens,
        "sample": {"temperature": 0, "seed": seed},
        "output": output
    }))
    .unwrap()
}

fn speak(rig: &Rig, request: &GenerationRequest) -> Result<(Spoken, Vec<TokenEvent>), ServiceError> {
    let plan = rig.service.prepare_speech(request)?;
    let (tx, rx) = std::sync::mpsc::sync_channel(4096);
    let spoken = rig.service.speak(plan, Some(tx), None)?;
    Ok((spoken, rx.try_iter().collect()))
}

/// The pattern of the reply in the slot: `t` for a text token, `a` for a frame.
fn pattern(rig: &Rig) -> String {
    let plan = rig.plan.lock().unwrap();
    plan.after_marker
        .iter()
        .map(|t| if *t == AUDIO_OUT { 'a' } else { 't' })
        .collect()
}

/// The interleaved reply of the oracle (`target/lfm3-golden/facts.json`, `mixed.pattern`), without its last token,
/// `<|im_end|>`, which the nagaya does not sample after the end frame (spec 11.11, "After the end frame").
const ORACLE_PATTERN: &str = "ttttttaaaaaaaaaaaattttttaaaaaaaaaaaataaaaaaaaaaaaaaaaaaaaaaa";

#[test]
fn an_interleaved_reply_has_blocks_of_6_tokens_and_12_frames_then_its_audio_tail() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = bytes("Hello, nagay");
        plan.text.push(TEXT_END);
        plan.end_at = 46;
    }
    let (spoken, events) = speak(&rig, &request(None, 64, 7)).unwrap();
    assert_eq!(pattern(&rig), ORACLE_PATTERN);
    assert_eq!(
        rig.plan.lock().unwrap().marker,
        Some(MIXED_START),
        "the default mode is interleaved"
    );
    // The toy family reads the byte that stands for <|text_end|> as text. The parser of LFM3 ends at the token.
    assert_eq!(spoken.done.content, "Hello, nagay|");
    assert_eq!(spoken.done.finish, Finish::EndTurn);
    assert_eq!(spoken.done.usage.completion, 13, "the 12 bytes and <|text_end|>");
    assert_eq!(events.len(), 13, "one token event for each text token");
    assert_eq!(
        events.iter().map(|e| e.n).collect::<Vec<_>>(),
        (1..=13).collect::<Vec<_>>()
    );
    assert_eq!((spoken.frames, spoken.ended), (47, true));
    assert_eq!(spoken.codes.len(), 46 * CODEBOOKS, "the end frame is not in the codes");
    assert_eq!(
        spoken.pcm.len(),
        46 * 1920,
        "1,920 samples for each frame that makes audio"
    );
    assert!(spoken.done.head.is_none(), "a reply with audio holds no head");
}

#[test]
fn a_reply_in_speech_mode_is_frames_until_the_end_frame_then_text_until_a_stop() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = vec![gogi_toy::END];
        plan.end_at = 5;
    }
    let (spoken, events) = speak(&rig, &request(Some("speech"), 64, 7)).unwrap();
    // The prompt ends with <|audio_start|>, and the last token, the stop, is not fed.
    assert_eq!(pattern(&rig), "aaaaaa");
    assert_eq!(rig.plan.lock().unwrap().marker, Some(AUDIO_START));
    assert_eq!(spoken.done.content, "");
    assert_eq!(spoken.done.finish, Finish::EndTurn);
    assert_eq!((spoken.done.usage.completion, events.len()), (1, 1));
    assert_eq!(spoken.pcm.len(), 5 * 1920);
    assert_eq!(spoken.codes.len(), 5 * CODEBOOKS);
}

#[test]
fn each_frame_gets_its_index_the_seed_and_the_codes_before_it() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = vec![gogi_toy::END];
        plan.end_at = 4;
    }
    let (spoken, _) = speak(&rig, &request(Some("speech"), 64, 99)).unwrap();
    let calls = rig.plan.lock().unwrap().calls.clone();
    assert_eq!(
        calls,
        (0..5)
            .map(|f| (99, f, f as usize * CODEBOOKS, true))
            .collect::<Vec<_>>(),
        "each frame asks for its logit row"
    );
    assert!(spoken.codes.iter().all(|c| *c < END_CODE));
    let want: Vec<u32> = (0..4)
        .flat_map(|f| (0..CODEBOOKS).map(move |g| code(99, f, g)))
        .collect();
    assert_eq!(spoken.codes, want);
}

#[test]
fn the_same_seed_gives_the_same_codes_and_audio_and_another_seed_others() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = vec![gogi_toy::END];
        plan.end_at = 6;
    }
    let (a, _) = speak(&rig, &request(Some("speech"), 64, 3)).unwrap();
    let (b, _) = speak(&rig, &request(Some("speech"), 64, 3)).unwrap();
    let (c, _) = speak(&rig, &request(Some("speech"), 64, 4)).unwrap();
    assert_eq!((&a.codes, &a.pcm), (&b.codes, &b.pcm));
    assert_ne!(a.codes, c.codes);
    assert_eq!((a.done.seed, c.done.seed), (Some(3), Some(4)));
}

#[test]
fn max_tokens_bounds_the_text_and_ends_the_audio_with_the_last_samples_of_the_codec() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = bytes("abcdefghijklmnopqrstuvwxyz");
        plan.end_at = 1000;
    }
    let (spoken, events) = speak(&rig, &request(None, 8, 7)).unwrap();
    // 6 tokens, 12 frames, then 2 tokens: the 8th is sampled and not fed.
    assert_eq!(pattern(&rig), format!("tttttt{}t", "a".repeat(12)));
    assert_eq!(spoken.done.finish, Finish::MaxTokens);
    assert_eq!(spoken.done.content, "abcdefgh");
    assert_eq!((spoken.done.usage.completion, events.len()), (8, 8));
    assert_eq!((spoken.frames, spoken.ended), (12, false));
    assert_eq!(
        spoken.pcm.len(),
        12 * 1920,
        "the codec gives its last 480 samples at the cut"
    );
    assert_eq!(
        rig.plan.lock().unwrap().clears,
        2,
        "a clear at frame 0 and one at the cut"
    );
}

#[test]
fn a_stop_token_before_the_text_end_ends_the_reply_and_its_audio() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        let mut text = bytes("abcdefgh");
        text.push(gogi_toy::END);
        plan.text = text;
        plan.end_at = 1000;
    }
    let (spoken, _) = speak(&rig, &request(None, 64, 7)).unwrap();
    assert_eq!(pattern(&rig), format!("tttttt{}tt", "a".repeat(12)));
    assert_eq!(spoken.done.finish, Finish::EndTurn);
    assert_eq!(spoken.done.content, "abcdefgh");
    assert_eq!(spoken.pcm.len(), 12 * 1920);
}

#[test]
fn a_full_slot_ends_a_reply_of_speech_with_context_full() {
    let mut config = config();
    config.context_tokens = 200;
    let rig = rig_with(config);
    rig.plan.lock().unwrap().end_at = 100_000;
    let request = request(Some("speech"), 64, 7);
    let prompt = ToyFamily::new().render(&request.context, true).unwrap().tokens.len() + 1;
    use nagaya_council::family::Family;
    let (spoken, _) = speak(&rig, &request).unwrap();
    assert_eq!(spoken.done.finish, Finish::ContextFull);
    assert_eq!(spoken.frames, 200 - prompt, "the frames fill the slot");
    assert_eq!(spoken.pcm.len(), spoken.frames * 1920);
}

#[test]
fn a_mode_that_is_not_known_or_another_option_is_a_400_that_names_it() {
    let rig = rig();
    for (options, param) in [
        (json!({"mode": "sing"}), "output.options.mode"),
        (json!({"mode": 3}), "output.options.mode"),
        (json!({"voice": "warm"}), "output.options.voice"),
    ] {
        let mut request = request(None, 8, 1);
        request.output.as_mut().unwrap().options = options.as_object().unwrap().clone();
        match rig.service.prepare_speech(&request) {
            Err(ServiceError::Invalid(invalid)) => assert_eq!(invalid.param.as_deref(), Some(param)),
            Err(other) => panic!("{options}: {other:?}"),
            Ok(_) => panic!("{options}: the plan has no error"),
        }
    }
    let mut open = request(None, 8, 1);
    open.context
        .messages
        .push(serde_json::from_value(json!({"role": "assistant", "content": "Hel", "continue": true})).unwrap());
    match rig.service.prepare_speech(&open) {
        Err(ServiceError::Invalid(invalid)) => assert_eq!(invalid.param.as_deref(), Some("continue")),
        other => panic!("a continuation with an output: {:?}", other.err()),
    }
}

#[test]
fn a_text_reply_of_a_model_that_speaks_never_picks_a_marker() {
    let rig = rig();
    let request: GenerationRequest = serde_json::from_value(json!({
        "model": "toy",
        "context": {"thinking": false, "messages": [{"role": "user", "content": "Say hello."}]},
        "max_tokens": 3,
        "sample": {"temperature": 0}
    }))
    .unwrap();
    use nagaya_council::family::Family;
    let prompt = ToyFamily::new().render(&request.context, true).unwrap().tokens.len() as u64;
    // The engine favors each marker by far. Greedy then picks the favored token unless the sampler drops it.
    rig.plan.lock().unwrap().plain = Some((prompt, vec![AUDIO_START, MIXED_START, b'x' as u32]));
    let done = rig.service.generate(&request, None).unwrap();
    assert!(
        !done.content.contains('{') && !done.content.contains('}'),
        "{:?}",
        done.content
    );
    assert!(
        done.content.ends_with('x'),
        "the favored text after the markers: {:?}",
        done.content
    );
}

#[test]
fn the_text_of_a_reply_with_audio_never_picks_a_marker_either() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = vec![MIXED_START, AUDIO_START, gogi_toy::END];
        plan.end_at = 1000;
    }
    let (spoken, events) = speak(&rig, &request(None, 2, 7)).unwrap();
    assert!(!events.is_empty());
    assert!(
        events.iter().all(|e| e.id != MIXED_START && e.id != AUDIO_START),
        "{events:?}"
    );
    assert!(!spoken.done.content.contains(['{', '}']), "{:?}", spoken.done.content);
}

#[test]
fn a_model_with_no_speech_path_takes_no_plan() {
    let rig = rig_with(ServiceConfig::new("toy", ENGINE));
    assert!(matches!(
        rig.service.prepare_speech(&request(None, 8, 1)),
        Err(ServiceError::Internal(_))
    ));
}

// ---- the route

fn council(rig: &Rig) -> Arc<Council> {
    let service = Service::attach_with_clock(
        rig.process.ports(),
        Arc::new(ToyFamily::new()),
        rig.directory.path(),
        config(),
        Arc::new(|| 1_000_000),
    )
    .unwrap();
    let objects = nagaya_store::objects::ObjectStore::open(&rig.directory.path().join("objects"), 1 << 30).unwrap();
    Arc::new(
        Council::new(
            vec![Served {
                service,
                aliases: Vec::new(),
                device: None,
            }],
            None,
        )
        .unwrap()
        .with_objects(objects),
    )
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let request = Request::builder().method(method).uri(path).body(body).unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, bytes.to_vec())
    })
}

fn body(value: GenerationRequest) -> Value {
    serde_json::to_value(value).unwrap()
}

#[test]
fn identity_lists_audio_wav_as_an_output_of_a_model_that_speaks() {
    let rig = rig();
    let council = council(&rig);
    let (status, bytes) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    let identity: Value = serde_json::from_slice(&bytes).unwrap();
    let media = &identity["models"][0]["media"];
    assert_eq!(media["outputs"], json!(["audio/wav"]), "{identity}");
    assert!(media.get("text_replies").is_none(), "its replies hold text: {media}");
}

#[test]
fn a_reply_holds_its_text_and_one_sealed_wav_whose_object_has_all_its_frames() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = bytes("Hello, nagay");
        plan.text.push(TEXT_END);
        plan.end_at = 46;
    }
    let council = council(&rig);
    let (status, bytes) = send(
        &council,
        "POST",
        "/council/v2/generations",
        Some(body(request(None, 64, 7))),
    );
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
    let done: GenerationDone = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        done.content, "Hello, nagay|",
        "the toy family reads <|text_end|> as a byte"
    );
    let [output] = done.outputs.as_slice() else {
        panic!("one output: {:?}", done.outputs)
    };
    assert_eq!((output.index, output.mime.as_str()), (0, "audio/wav"));
    let data = 46 * 1920 * 2;
    assert_eq!(output.bytes, 44 + data as u64);
    let written = done.usage.media.as_ref().unwrap().written;
    assert_eq!(
        (written.objects, written.bytes, written.audio_ms),
        (1, 44 + data as u64, Some(46 * 80))
    );
    let (status, wav) = send(
        &council,
        "GET",
        &format!("/council/v2/objects/{}", output.id.as_str()),
        None,
    );
    assert_eq!(status, StatusCode::OK);
    assert_eq!(wav.len(), 44 + data);
    assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize, data);
    assert_eq!(gogi_proto::content_id(&wav), output.id.as_str());
    // The service keeps the codes of the output under its content id.
    let id = nagaya_store::ContentId::from_hex(output.id.as_str().trim_start_matches("blake3:")).unwrap();
    let served = council.text_model("toy").unwrap();
    assert_eq!(
        served.service.output_codes(&id).map(|codes| codes.len()),
        Some(46 * CODEBOOKS)
    );
    // The reply's message holds the output after its text (gogi "Outputs").
    let message = done.as_message().unwrap();
    assert_eq!(message.content.unwrap().parts().len(), 2);
}

#[test]
fn a_stream_sends_the_output_event_then_the_tokens_then_done_and_the_pending_id_reads_the_audio() {
    let rig = rig();
    {
        let mut plan = rig.plan.lock().unwrap();
        plan.text = bytes("Hi");
        plan.text.push(TEXT_END);
        plan.end_at = 3;
    }
    let council = council(&rig);
    let mut request = request(None, 64, 7);
    request.stream = Some(true);
    let (status, bytes) = send(&council, "POST", "/council/v2/generations", Some(body(request)));
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(bytes).unwrap();
    let events: Vec<StreamEvent> = text
        .split("\n\n")
        .filter(|chunk| !chunk.trim().is_empty())
        .map(|chunk| {
            let mut name = "";
            let mut data = String::new();
            for line in chunk.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    name = rest.trim();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data.push_str(rest.trim_start());
                }
            }
            StreamEvent::parse(name, &data).unwrap().expect("a known event")
        })
        .collect();
    let names: Vec<&str> = events.iter().map(|e| e.name()).collect();
    assert_eq!(names, ["output", "token", "token", "token", "done"], "{text}");
    let (StreamEvent::Output(output), StreamEvent::Done(done)) = (&events[0], &events[4]) else {
        panic!("{text}");
    };
    assert_eq!(done.outputs[0].bytes, 44 + 3 * 1920 * 2);
    // After the seal the pending id is a redirect to the object.
    let (status, _) = send(
        &council,
        "GET",
        &format!("/council/v2/objects/{}", output.pending),
        None,
    );
    assert_eq!(status, StatusCode::PERMANENT_REDIRECT);
}

/// A reply with audio names the record of its context with its reply: the text, then the object part of its WAV
/// (gogi "The reply in a record", spec 11.13). The record of the request's context is held too.
///
/// Falsified by a route that seals the reply and names no record: `done.context` is none.
#[test]
fn a_reply_with_audio_names_the_record_with_its_text_and_its_output() {
    for stream in [false, true] {
        let rig = rig();
        {
            let mut plan = rig.plan.lock().unwrap();
            plan.text = bytes("Hi");
            plan.text.push(TEXT_END);
            plan.end_at = 3;
        }
        let council = council(&rig);
        let mut request = request(None, 64, 7);
        request.stream = Some(stream);
        let (status, bytes) = send(&council, "POST", "/council/v2/generations", Some(body(request.clone())));
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
        let done: GenerationDone = if stream {
            let text = String::from_utf8(bytes).unwrap();
            let last = text
                .split("\n\n")
                .filter(|chunk| !chunk.trim().is_empty())
                .last()
                .unwrap();
            assert!(last.contains("event: done"), "{last}");
            let data: String = last.lines().filter_map(|line| line.strip_prefix("data:")).collect();
            serde_json::from_str(data.trim_start()).unwrap()
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        let with_reply = gogi_proto::with_reply(&request.context, &done);
        assert_eq!(
            done.context,
            Some(gogi_proto::context_id(&with_reply)),
            "stream {stream}"
        );
        let read = |id: &gogi_proto::ContextId| {
            let (status, bytes) = send(&council, "GET", &format!("/council/v2/contexts/{id}"), None);
            assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
            serde_json::from_slice::<Value>(&bytes).unwrap()
        };
        let record = read(done.context.as_ref().unwrap());
        let last = record["context"]["messages"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .clone();
        assert_eq!(
            last["content"],
            json!([{"type": "text", "text": done.content},
                   {"type": "object", "id": done.outputs[0].id.as_str(), "mime": "audio/wav"}]),
            "{last}"
        );
        assert_eq!(last["snap"], json!(true));
        let asked = read(&gogi_proto::context_id(&request.context));
        assert_eq!(asked["context"], serde_json::to_value(&request.context).unwrap());
    }
}

#[test]
fn a_mode_that_is_not_known_is_a_400_at_the_route() {
    let rig = rig();
    let council = council(&rig);
    let mut request = request(Some("sing"), 8, 1);
    let (status, bytes) = send(&council, "POST", "/council/v2/generations", Some(body(request.clone())));
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(error["error"]["param"], "output.options.mode", "{error}");
    request.stream = Some(true);
    let (status, _) = send(&council, "POST", "/council/v2/generations", Some(body(request)));
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a stream checks before its output event"
    );
}

/// A reply with audio follows the control list of its context, as a text reply does (spec 11.12). This engine has no
/// table, so a list is a 400 that names `control`, and no reply speaks without the list it asked for.
#[test]
fn a_control_list_for_a_model_with_no_table_is_a_400_for_a_reply_with_audio_too() {
    let rig = rig();
    let mut steered = request(None, 8, 1);
    steered.context.control = serde_json::from_value(json!([
        {"vector": format!("blake3:{}", "ab".repeat(32)), "op": "ablate", "strength": 1.0}
    ]))
    .unwrap();
    match rig.service.prepare_speech(&steered) {
        Err(ServiceError::Invalid(invalid)) => assert_eq!(invalid.param.as_deref(), Some("control")),
        other => panic!("a control list with no table: {:?}", other.err()),
    }
}

/// A reply with audio refuses a layer loop (spec 11.16). A frame feeds its position as a plain pass, so the extra
/// runs of the loop would hold no state there.
///
/// Falsified by a service that takes the loop: the plan of the speech is made.
#[test]
fn a_reply_with_audio_refuses_a_layer_loop() {
    let rig = rig();
    let mut looped = request(None, 8, 1);
    looped.context.layer_loop = Some(gogi_proto::layer_loop::Loop {
        start: 0,
        end: 2,
        count: 2,
    });
    match rig.service.prepare_speech(&looped) {
        Err(ServiceError::Invalid(invalid)) => {
            assert_eq!(invalid.param.as_deref(), Some("loop"));
            assert!(invalid.message.contains("audio"), "{}", invalid.message);
        }
        other => panic!("a loop with audio: {:?}", other.err()),
    }
}
