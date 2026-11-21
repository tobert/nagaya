//! The speech of a served text model that also speaks, such as LFM3 Omni (spec 11.11, "The generation modes" and
//! "The speech loop of a generation").
//!
//! A generation with an `output` of `audio/wav` samples text tokens with the sampler of spec 11.2, and makes frames of
//! audio through the capability `speech_frames` of the engine. In `interleaved` mode the text comes in blocks of 6
//! tokens, then 12 frames, in turn. In `speech` mode the reply is frames until the end frame. The job gives each piece
//! of audio to a sink as PCM, and the reply keeps the codes of its frames.

use std::sync::Arc;
use std::sync::mpsc::SyncSender;
use std::time::Instant;

use gogi_proto::Invalid;
use gogi_proto::context::Usage;
use gogi_proto::generation::{Finish, GenerationDone, GenerationRequest, Phase, ReplyStats, TokenEvent};
use nagaya_abi::{Capability, Engine, FrameDraw, FrameOut, Kind, SlotId};
use nagaya_council::build::BoundaryRecipe;
use nagaya_council::family::{MediaInput, ParsedReply, Rendering, ReplyParser, ReplyStart};
use nagaya_council::sampler::{Settings, pick_except};

use super::build::{ContextBody, Got, JobControl, Then, Want};
use super::error::ServiceError;
use super::generations::{PartStats, fresh_seed, tail_text, tool_calls};
use super::{MediaRead, Schedule, Service};

/// The samples of one second of the audio of LFM3.
pub const RATE: u32 = 24_000;

/// The speech path of a served text model (spec 11.11): the marker tokens and the shape of an interleaved reply. A
/// family library that speaks gives it to the service config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechOut {
    /// The marker that opens a reply of speech: `<|audio_start|>`.
    pub audio_start: u32,
    /// The marker that opens an interleaved reply: `<|mixed_start|>`.
    pub mixed_start: u32,
    /// The token that ends the text of an interleaved reply: `<|text_end|>`.
    pub text_end: u32,
    /// The placeholder of a frame in the slot: `<|audio_out|>`.
    pub audio_out: u32,
    /// The text tokens of each block of an interleaved reply (`interleaved_n_text`).
    pub n_text: usize,
    /// The frames of each block of an interleaved reply (`interleaved_n_audio`).
    pub n_audio: usize,
}

impl SpeechOut {
    /// The tokens that the text sampler never picks: the two markers (spec 11.11).
    pub fn markers(&self) -> [u32; 2] {
        [self.audio_start, self.mixed_start]
    }
}

/// The mode of a reply that speaks (spec 11.11, "The generation modes").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechMode {
    /// Text in blocks, then frames, in turn. The default.
    Interleaved,
    /// Frames only, until the end frame.
    Speech,
}

impl SpeechMode {
    /// The mode of `output.options`: its `mode`, `interleaved` when it has none. Another member, or another mode, is
    /// a 400 that names it.
    pub fn of(options: &serde_json::Map<String, serde_json::Value>, model: &str) -> Result<Self, Invalid> {
        let mut mode = SpeechMode::Interleaved;
        for (key, value) in options {
            if key != "mode" {
                return Err(Invalid::param(
                    format!("output.options.{key}"),
                    format!("nothing ran: {model} takes the output option `mode` only, not `{key}`"),
                ));
            }
            mode = match value.as_str() {
                Some("interleaved") => SpeechMode::Interleaved,
                Some("speech") => SpeechMode::Speech,
                _ => {
                    return Err(Invalid::param(
                        "output.options.mode",
                        format!("nothing ran: output.options.mode is {value}; give \"interleaved\" or \"speech\""),
                    ));
                }
            };
        }
        Ok(mode)
    }
}

/// The receiver of the audio of a reply, as PCM bytes while the frames decode. It gives false when its reader went
/// away, and the reply then ends as cancelled. It runs on the group thread, so it must not wait.
pub type AudioSink = Arc<dyn Fn(&[u8]) -> bool + Send + Sync>;

/// The signed 16-bit sample of each f32 sample: clamped to [-1, 1], then `round(v × 32767)` with ties to even, as the
/// codec op of Qwen3-TTS does (spec 11.11, "PCM").
pub fn pcm16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|v| (v.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16)
        .collect()
}

/// The little-endian bytes of PCM samples.
pub fn pcm_bytes(pcm: &[i16]) -> Vec<u8> {
    pcm.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// The draw of one frame that the work asks of its slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameWant {
    /// The seed of the request.
    pub seed: u64,
    /// The index of the frame in the output, from 0.
    pub frame: u64,
    /// The codes of the frames of the output so far, without the end frame.
    pub history: Vec<u32>,
    /// True when the work wants the logit row of the fed frame.
    pub logits: bool,
    /// The token that stands for the frame in the tokens of the slot.
    pub placeholder: u32,
}

/// What one frame gave: the frame, and the samples of the codec that are whole after it.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameGot {
    pub out: FrameOut,
    pub samples: Vec<f32>,
}

/// Make one frame in the slot, and decode it. The first frame of an output clears the codec of the slot first, so a
/// window that a cancelled reply left is gone. An end frame gives the last samples of the codec. `vocab` is the width
/// of a logit row.
pub(crate) fn frame(
    engine: &mut dyn Engine,
    slot: SlotId,
    want: &FrameWant,
    vocab: usize,
) -> Result<FrameGot, ServiceError> {
    let last_error = |engine: &mut dyn Engine| engine.last_error().unwrap_or_default();
    let Some(speech) = engine.frames() else {
        return Err(ServiceError::internal(
            "the engine lost its capability speech_frames; a reply with audio needs it",
        ));
    };
    if want.frame == 0 {
        speech.codec_clear(slot);
    }
    let draw = FrameDraw {
        seed: want.seed,
        frame: want.frame,
        history: &want.history,
    };
    let out = match speech.frame(slot, &draw, want.logits) {
        Ok(out) => out,
        Err(error) => {
            let message = last_error(engine);
            return Err(ServiceError::internal(format!(
                "frame {} of slot {slot} failed: {} ({:?}): {message}",
                want.frame, error.message, error.kind
            )));
        }
    };
    let codebooks = speech.codebooks() as usize;
    if out.codes.len() != codebooks {
        return Err(ServiceError::internal(format!(
            "frame {} of slot {slot} gave {} codes, and the engine declares {codebooks} codebooks",
            want.frame,
            out.codes.len()
        )));
    }
    match &out.logits {
        Some(row) if row.len() != vocab => {
            return Err(ServiceError::internal(format!(
                "frame {} of slot {slot} gave a logit row of {} values, and the vocabulary has {vocab}",
                want.frame,
                row.len()
            )));
        }
        None if want.logits => {
            return Err(ServiceError::internal(format!(
                "frame {} of slot {slot} gave no logit row, and the job asked for one",
                want.frame
            )));
        }
        _ => {}
    }
    let samples = if out.end {
        speech.codec_clear(slot)
    } else {
        match speech.codec(slot, &out.codes) {
            Ok(samples) => samples,
            Err(error) => {
                let message = last_error(engine);
                return Err(ServiceError::internal(format!(
                    "the codec of slot {slot} failed at frame {}: {} ({:?}): {message}",
                    want.frame, error.message, error.kind
                )));
            }
        }
    };
    Ok(FrameGot { out, samples })
}

/// The last samples of the codec of the slot, at the end of an output that has no end frame.
pub(crate) fn codec_end(engine: &mut dyn Engine, slot: SlotId) -> Result<Vec<f32>, ServiceError> {
    let speech = engine
        .frames()
        .ok_or_else(|| ServiceError::internal("the engine lost its capability speech_frames"))?;
    Ok(speech.codec_clear(slot))
}

/// Where the work of a reply with audio is.
#[derive(Debug, PartialEq, Eq)]
enum Stage {
    Building,
    Running,
    /// The reply ended, and the codec gives its last samples.
    Closing,
    Done,
}

/// What a reply with audio gives back.
pub struct Spoken {
    /// The reply, with no output: the caller seals the WAV and adds it.
    pub done: GenerationDone,
    /// The PCM of all the frames of the output.
    pub pcm: Vec<i16>,
    /// The codes of the frames of the output, 12 for each frame, without the end frame.
    pub codes: Vec<u32>,
    /// The frames that the reply fed, the end frame too.
    pub frames: usize,
    /// True when the reply had an end frame.
    pub ended: bool,
}

/// The work of a reply with audio after its build: feed the generation prompt and the marker, then text tokens and
/// frames in the order of the mode, until the reply ends (spec 11.11, "The speech loop of a generation").
pub(crate) struct SpeakThen {
    speech: SpeechOut,
    mode: SpeechMode,
    /// The tokens of the prompt: the context, the generation prompt and the marker.
    prompt: Vec<u32>,
    /// The tokens that the build makes.
    end: usize,
    settings: Settings,
    max_tokens: usize,
    context_tokens: usize,
    parser: Option<Box<dyn ReplyParser>>,
    events: Option<SyncSender<TokenEvent>>,
    sink: Option<AudioSink>,
    started: Instant,
    stage: Stage,
    recipes: Vec<BoundaryRecipe>,
    sampled: Vec<u32>,
    reasoning: u64,
    stats: [PartStats; 2],
    /// The frames fed, the end frame too.
    frames: usize,
    codes: Vec<u32>,
    pcm: Vec<i16>,
    /// True while an interleaved reply makes the frames of its block.
    speaking: bool,
    /// The steps left in the block of an interleaved reply.
    left: usize,
    /// True after the text of an interleaved reply ended with `<|text_end|>`.
    text_done: bool,
    /// True after the end frame.
    audio_done: bool,
    parsed: Option<ParsedReply>,
    finish: Option<Finish>,
    next_phase: Option<Phase>,
}

impl SpeakThen {
    /// True when the next step of the reply is a frame.
    fn next_is_frame(&self) -> bool {
        !self.audio_done && (self.mode == SpeechMode::Speech || self.speaking)
    }

    /// The positions of the reply so far: its text tokens and its frames.
    fn reply_positions(&self) -> usize {
        self.sampled.len() + self.frames
    }

    /// Count one step of an interleaved reply, and turn to the frames or to the text at the end of a block. The
    /// reference counts the same way (`modeling_lfm3_omni.py`, `slots_left`).
    fn count_step(&mut self) {
        if self.mode != SpeechMode::Interleaved || self.text_done || self.audio_done {
            return;
        }
        self.left -= 1;
        if self.left == 0 {
            self.speaking = !self.speaking;
            self.left = if self.speaking {
                self.speech.n_audio
            } else {
                self.speech.n_text
            };
        }
    }

    /// Finish the parser after the last text token of the reply. `cut` is true when a limit cut the text.
    fn finish_parser(&mut self, cut: bool) -> String {
        let Some(parser) = self.parser.take() else {
            return String::new();
        };
        self.next_phase = Some(parser.phase());
        let parsed = parser.finish(cut);
        let phase = self.next_phase.unwrap_or(Phase::Content);
        // The tail of the parser joins the last token event (gogi "Streams").
        let tail = tail_text(&parsed, phase).unwrap_or_default();
        self.parsed = Some(parsed);
        tail
    }

    /// The next want after a step, when the reply goes on. `row` is the logit row after the step, when it has one.
    fn go_on(&mut self, row: Option<Vec<f32>>) -> Result<Want, ServiceError> {
        if self.next_is_frame() {
            return Ok(self.frame_want());
        }
        let row = row.ok_or_else(|| ServiceError::internal("a text token of a reply with audio has no logit row"))?;
        self.sample(&row)
    }

    fn frame_want(&self) -> Want {
        Want::Frame(FrameWant {
            seed: self.settings.seed,
            frame: self.frames as u64,
            history: self.codes.clone(),
            // A text token can follow any frame, the end frame too (spec 11.11).
            logits: true,
            placeholder: self.speech.audio_out,
        })
    }

    /// Sample the next text token from a logit row, and give it to the parser and to the stream.
    fn sample(&mut self, row: &[f32]) -> Result<Want, ServiceError> {
        let position = self.prompt.len() + self.reply_positions();
        let chosen = pick_except(row, &self.settings, position as u64, &self.speech.markers())
            .map_err(|error| ServiceError::internal(format!("the sample at position {position} failed: {error}")))?;
        let parser = self
            .parser
            .as_mut()
            .ok_or_else(|| ServiceError::internal("a reply with audio sampled text after its text ended"))?;
        let step = parser.push(chosen.token)?;
        let ended = parser.ended();
        self.sampled.push(chosen.token);
        match step.phase {
            Phase::Reasoning => {
                self.reasoning += 1;
                self.stats[0].push(chosen.p, chosen.h);
            }
            Phase::Content => self.stats[1].push(chosen.p, chosen.h),
            Phase::Call | Phase::Unknown => {}
        }
        let n = self.sampled.len();
        let text_end = chosen.token == self.speech.text_end;
        // The text ends at a stop token or at `<|text_end|>`. In an interleaved reply before the end frame, the audio
        // goes on after `<|text_end|>` (spec 11.11).
        let tail_of_audio = text_end && self.mode == SpeechMode::Interleaved && !self.audio_done;
        let mut piece = step.piece;
        let mut end = None;
        if ended || text_end {
            piece.push_str(&self.finish_parser(false));
            if tail_of_audio {
                self.text_done = true;
                self.speaking = true;
            } else {
                end = Some(None);
            }
        } else if self.prompt.len() + self.reply_positions() >= self.context_tokens {
            piece.push_str(&self.finish_parser(true));
            end = Some(Some(Finish::ContextFull));
        } else if n == self.max_tokens {
            piece.push_str(&self.finish_parser(true));
            end = Some(Some(Finish::MaxTokens));
        }
        if let Some(events) = &self.events {
            let seconds = self.started.elapsed().as_secs_f64();
            let event = TokenEvent {
                id: chosen.token,
                piece,
                phase: step.phase,
                p: chosen.p,
                h: chosen.h,
                n: n as u64,
                tps: (seconds > 0.0).then(|| n as f64 / seconds),
            };
            // A closed stream cancels the reply (gogi "Streams").
            if events.send(event).is_err() {
                return Ok(Want::Cancel);
            }
        }
        self.count_step();
        if let Some(cut) = end {
            return Ok(self.close(cut));
        }
        Ok(Want::Feed {
            tokens: vec![chosen.token],
            logits: !self.next_is_frame(),
            kind: Kind::Decode,
        })
    }

    /// Take a frame and its samples, and give the next want.
    fn framed(&mut self, got: FrameGot) -> Result<Want, ServiceError> {
        self.frames += 1;
        if !self.push_audio(&got.samples) {
            return Ok(Want::Cancel);
        }
        if got.out.end {
            self.audio_done = true;
        } else {
            self.codes.extend_from_slice(&got.out.codes);
        }
        self.count_step();
        if self.audio_done && self.parser.is_none() {
            return Ok(self.end(None));
        }
        if self.prompt.len() + self.reply_positions() >= self.context_tokens {
            let tail = self.finish_parser(true);
            if !tail.is_empty() {
                tracing::debug!("a reply with audio cut on a frame drops the parser tail {tail:?} from its events");
            }
            return Ok(self.close(Some(Finish::ContextFull)));
        }
        self.go_on(got.out.logits)
    }

    /// Give PCM to the sink, and keep it. False when the reader of the sink went away.
    fn push_audio(&mut self, samples: &[f32]) -> bool {
        let pcm = pcm16(samples);
        let alive = match &self.sink {
            Some(sink) if !pcm.is_empty() => sink(&pcm_bytes(&pcm)),
            _ => true,
        };
        self.pcm.extend(pcm);
        alive
    }

    /// End the reply. An output whose codec has an open window gives its last samples first.
    fn close(&mut self, cut: Option<Finish>) -> Want {
        self.set_finish(cut);
        if self.frames > 0 && !self.audio_done {
            self.stage = Stage::Closing;
            return Want::CodecEnd;
        }
        self.stage = Stage::Done;
        Want::Done
    }

    /// End a reply whose audio is complete.
    fn end(&mut self, cut: Option<Finish>) -> Want {
        self.set_finish(cut);
        self.stage = Stage::Done;
        Want::Done
    }

    /// The finish of the reply. A reply whose text ended its turn ends its turn, whatever cut its audio (gogi
    /// "Generations", `finish`).
    fn set_finish(&mut self, cut: Option<Finish>) {
        if self.parser.is_some() {
            self.finish_parser(cut.is_some());
        }
        let calls = self.parsed.as_ref().is_some_and(|parsed| !parsed.calls.is_empty());
        let ended_turn = cut.is_none() || (self.text_done && self.mode == SpeechMode::Interleaved);
        self.finish = Some(match cut {
            Some(cut) if !ended_turn => cut,
            _ if calls => Finish::ToolUse,
            _ => Finish::EndTurn,
        });
    }
}

impl Then for SpeakThen {
    const NEEDS_STATE: bool = true;
    const REQUEST: &'static str = "generation";

    fn next(&mut self, got: Got) -> Result<Want, ServiceError> {
        match (&self.stage, got) {
            (Stage::Building, Got::Built(recipes)) => {
                self.stage = Stage::Running;
                self.recipes = recipes;
                Ok(Want::Feed {
                    tokens: self.prompt[self.end..].to_vec(),
                    logits: !self.next_is_frame(),
                    kind: Kind::Prefill,
                })
            }
            (Stage::Running, Got::Fed(row)) => self.go_on(row),
            (Stage::Running, Got::Frame(got)) => self.framed(got),
            (Stage::Closing, Got::Samples(samples)) => {
                self.stage = Stage::Done;
                if !self.push_audio(&samples) {
                    return Ok(Want::Cancel);
                }
                Ok(Want::Done)
            }
            (_, got) => Err(ServiceError::internal(format!(
                "a reply with audio got {got:?} where it did not expect it"
            ))),
        }
    }
}

/// A checked generation with audio, ready to run.
pub struct SpeechPlan {
    then: SpeakThen,
    tokens: Arc<[u32]>,
    recipes: Vec<BoundaryRecipe>,
    cuts: Vec<super::build::Cut>,
    room: usize,
    schedule: Schedule,
    /// The root recipe of the context and its control list (spec 11.12). The text passes take the list, and a frame
    /// takes none.
    root: nagaya_store::RecipeId,
    control: Option<JobControl>,
    /// The rows of each media span of the prompt (spec 11.11).
    media: Arc<[MediaInput]>,
    /// The objects that the request reads, for `usage.media.read`.
    read: MediaRead,
    /// The mode of the reply.
    pub mode: SpeechMode,
}

impl Service {
    /// The speech path of this served model, when it speaks.
    pub fn speech_out(&self) -> Option<&SpeechOut> {
        self.config.speech.as_ref()
    }

    /// Check a generation with an `output` of `audio/wav`, and plan its job. Each check runs before anything runs,
    /// so a failed check is a status, not an `error` event (spec 11.11, "The stream").
    pub fn prepare_speech(&self, request: &GenerationRequest) -> Result<SpeechPlan, ServiceError> {
        request.validate()?;
        let Some(speech) = self.config.speech.clone() else {
            return Err(ServiceError::internal(format!(
                "{} writes text only, and a reply with audio reached it",
                self.config.model
            )));
        };
        let output = request
            .output
            .as_ref()
            .ok_or_else(|| ServiceError::internal("a reply with audio has an output"))?;
        if output.mime != WAV {
            return Err(Invalid::param(
                "output.mime",
                format!(
                    "nothing ran: output.mime is {}, and {} writes text and {}",
                    output.mime, self.config.model, WAV
                ),
            )
            .into());
        }
        let mode = SpeechMode::of(&output.options, &self.config.model)?;
        if !self.shared.limits.capabilities.contains(&Capability::SpeechFrames) {
            return Err(ServiceError::internal(format!(
                "the engine `{}` has no capability speech_frames, and its family speaks; serve the engine of the \
                 family",
                self.config.engine
            )));
        }
        let read = self.check_objects(&gogi_proto::objects::objects_of(&request.context.messages))?;
        if !request.decisions.is_empty() {
            return Err(Invalid::param(
                "decisions",
                "riding decisions need the ride capability, which this model does not have",
            )
            .into());
        }
        let context = &request.context;
        if context.open().is_some() {
            return Err(Invalid::param(
                "continue",
                "nothing ran: a reply with an output continues no message, because an open message holds text only; \
                 send the open message without `output`, or close it",
            )
            .into());
        }
        let context_tokens = self.config.context_tokens;
        let max_tokens = request.max_tokens as usize;
        if max_tokens > context_tokens {
            return Err(Invalid::param("max_tokens", format!("max_tokens is 1 to {context_tokens}")).into());
        }
        let sample = request.sample.clone().unwrap_or_default();
        let defaults = self.config.sample;
        let settings = Settings {
            temperature: sample.temperature.unwrap_or(defaults.temperature),
            top_p: sample.top_p.unwrap_or(defaults.top_p),
            top_k: sample.top_k.unwrap_or(defaults.top_k),
            min_p: sample.min_p.unwrap_or(defaults.min_p),
            seed: sample.seed.unwrap_or_else(fresh_seed),
        };
        settings
            .validate()
            .map_err(|error| Invalid::param("sample", error.to_string()))?;
        let rendering = self.family.render(context, true)?;
        let mut prompt = rendering.tokens.clone();
        // The marker of the mode follows the whole generation prompt (spec 11.11, "The marker of a mode").
        prompt.push(match mode {
            SpeechMode::Interleaved => speech.mixed_start,
            SpeechMode::Speech => speech.audio_start,
        });
        if prompt.len() >= context_tokens {
            return Err(ServiceError::ContextFull {
                prompt_tokens: prompt.len() as u64,
                context_tokens: context_tokens as u64,
            });
        }
        let built = Rendering {
            tokens: rendering.tokens[..rendering.end].to_vec(),
            end: rendering.end,
            boundaries: rendering.boundaries.clone(),
            control_text: Vec::new(),
            media: rendering.media.clone(),
        };
        let steer = self.steer(context, "")?;
        // A frame feeds its position as a plain pass, so it would leave the extra runs of a loop with no state there
        // (engine-abi "Layer loops"). Refuse the loop until a frame carries it.
        if steer.layer_loop.is_some() {
            return Err(Invalid::param(
                "loop",
                "nothing ran: a reply with audio does not run under a layer loop yet; send the loop with a text reply, \
                 or no loop with an output",
            )
            .into());
        }
        // A frame runs the default count of cycles, so another count would leave its position with the state of
        // another model (spec 11.17).
        if let (Some(count), Some(limit)) = (steer.cycles, self.cycles())
            && count != limit.default
        {
            return Err(Invalid::param(
                "cycles",
                format!(
                    "nothing ran: a reply with audio runs {} cycles only, the default; send cycles with a text reply, \
                     or no cycles with an output",
                    limit.default
                ),
            )
            .into());
        }
        let root = steer.root;
        let recipes = super::plan_recipes(root, &built)?;
        let cuts = self.cuts(root, context, &built)?;
        let media = self.media_rows(&rendering)?;
        let end = built.end;
        let parser = self.family.reply_parser(ReplyStart::of(context), &context.tools)?;
        let then = SpeakThen {
            mode,
            prompt,
            end,
            settings,
            max_tokens,
            context_tokens,
            parser: Some(parser),
            events: None,
            sink: None,
            started: Instant::now(),
            stage: Stage::Building,
            recipes: Vec::new(),
            sampled: Vec::new(),
            reasoning: 0,
            stats: [PartStats::default(); 2],
            frames: 0,
            codes: Vec::new(),
            pcm: Vec::new(),
            speaking: false,
            left: speech.n_text,
            text_done: false,
            audio_done: false,
            parsed: None,
            finish: None,
            next_phase: None,
            speech,
        };
        Ok(SpeechPlan {
            then,
            tokens: built.tokens[..end].into(),
            recipes,
            cuts,
            // A frame takes a position of the slot, so the reply may fill the slot (spec 11.11, "The frame limit").
            room: context_tokens,
            schedule: Schedule::of(request.class, request.priority),
            root,
            control: steer.control,
            media,
            read,
            mode,
        })
    }

    /// Run a planned reply with audio. Each text token goes to `events`, and the PCM of each frame goes to `sink`, as
    /// they come. A closed stream or a sink whose reader went away cancels the reply.
    pub fn speak(
        &self,
        plan: SpeechPlan,
        events: Option<SyncSender<TokenEvent>>,
        sink: Option<AudioSink>,
    ) -> Result<Spoken, ServiceError> {
        let SpeechPlan {
            mut then,
            tokens,
            recipes,
            cuts,
            room,
            schedule,
            root,
            control,
            media,
            read,
            ..
        } = plan;
        then.events = events;
        then.sink = sink;
        then.started = Instant::now();
        let prompt = then.prompt.len();
        let seed = then.settings.seed;
        let shared = Arc::clone(&self.shared);
        // `prepare_speech` refused each count but the default, so the reply runs the default count of cycles.
        let job = |tx| {
            ContextBody::with(shared, tokens, recipes.clone(), room, then, tx)
                .media(media)
                .cuts(cuts)
                .control(root, control, None, self.cycles().map(|limit| limit.default))
        };
        let finished = self.run_context(job, schedule, None)?;
        let then = finished.then;
        if then.stage != Stage::Done {
            return Err(ServiceError::internal(
                "a reply with audio ended before its work was done",
            ));
        }
        let held = super::held_boundaries(&finished.outcome);
        let kept = finished.outcome.kept as u64;
        let snapshots = self.infos(&held);
        let parsed = then
            .parsed
            .clone()
            .ok_or_else(|| ServiceError::internal("a done reply with audio has no parsed text"))?;
        let ids = call_ids(&then.prompt, &then.sampled, parsed.calls.len());
        let finish = then.finish.expect("a done reply has a finish");
        let done = GenerationDone {
            model: self.config.model.clone(),
            content: parsed.content.clone(),
            reasoning: parsed.reasoning.clone(),
            tool_calls: tool_calls(&parsed, &ids),
            finish,
            usage: Usage {
                prompt: prompt as u64,
                kept,
                fed: prompt as u64 - kept,
                completion: then.sampled.len() as u64,
                reasoning: then.reasoning,
                // The route adds the counts of the output that it writes.
                media: read,
            },
            // A reply with audio holds no head (spec 11.11, "No head").
            head: None,
            context: None,
            snapshots,
            stats: Some(ReplyStats {
                reasoning: then.stats[0].finish(),
                content: then.stats[1].finish(),
            }),
            decisions: Vec::new(),
            seed: Some(seed),
            phase: then.next_phase.filter(|_| finish == Finish::MaxTokens),
            outputs: Vec::new(),
        };
        Ok(Spoken {
            done,
            pcm: then.pcm,
            codes: then.codes,
            frames: then.frames,
            ended: then.audio_done,
        })
    }

    /// Keep the codes of an output under its content id, for a later `assistant` audio part (spec 11.11, "Codes").
    /// `held` says which outputs the object store still holds: the codes of the others go.
    pub fn keep_output_codes(
        &self,
        id: nagaya_store::ContentId,
        codes: Vec<u32>,
        held: &dyn Fn(&nagaya_store::ContentId) -> bool,
    ) {
        let mut book = self.output_codes.0.lock().expect("the code lock is poisoned");
        book.retain(|kept, _| held(kept));
        book.insert(id, codes);
    }

    /// The codes of an output that this service wrote, when it still holds them.
    pub fn output_codes(&self, id: &nagaya_store::ContentId) -> Option<Vec<u32>> {
        self.output_codes
            .0
            .lock()
            .expect("the code lock is poisoned")
            .get(id)
            .cloned()
    }
}

/// The media type of the audio output.
pub const WAV: &str = "audio/wav";

/// The id of each call of a reply with audio, as the text path names them: a hash of the prompt and the text.
fn call_ids(prompt: &[u32], sampled: &[u32], count: usize) -> Vec<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"nagaya call v1");
    hasher.update(&(prompt.len() as u64).to_le_bytes());
    for token in prompt.iter().chain(sampled) {
        hasher.update(&token.to_le_bytes());
    }
    let hash = hasher.finalize();
    let stem: String = hash.as_bytes()[..8].iter().map(|byte| format!("{byte:02x}")).collect();
    (0..count).map(|index| format!("call_{stem}_{index}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_is_clamped_and_rounded_with_ties_to_even() {
        let pcm = pcm16(&[0.0, 1.0, -1.0, 2.0, -3.0, 0.5, -0.25, 0.1]);
        assert_eq!(pcm, vec![0, 32767, -32767, 32767, -32767, 16384, -8192, 3277]);
    }

    #[test]
    fn a_mode_is_interleaved_by_default_and_another_mode_or_member_is_a_400_that_names_it() {
        let map = |v: serde_json::Value| v.as_object().unwrap().clone();
        assert_eq!(
            SpeechMode::of(&map(serde_json::json!({})), "m").unwrap(),
            SpeechMode::Interleaved
        );
        assert_eq!(
            SpeechMode::of(&map(serde_json::json!({"mode": "speech"})), "m").unwrap(),
            SpeechMode::Speech
        );
        let error = SpeechMode::of(&map(serde_json::json!({"mode": "sing"})), "m").unwrap_err();
        assert_eq!(error.param.as_deref(), Some("output.options.mode"));
        assert!(error.message.contains("sing"), "{}", error.message);
        let error = SpeechMode::of(&map(serde_json::json!({"voice": "x"})), "m").unwrap_err();
        assert_eq!(error.param.as_deref(), Some("output.options.voice"));
    }
}
