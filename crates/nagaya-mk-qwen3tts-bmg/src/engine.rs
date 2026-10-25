//! The speech engine on one B70 card (engine-abi "Speech"): the weights, the slots, and the ring that runs each pass.

use std::collections::BTreeMap;
use std::path::Path;

use nagaya_abi::{
    Engine, Identity, Limits, PartError, PartId, Parts, PassError, PassErrorKind, PassOut, PromptPair, Segment, SlotId,
    Slots, Speech, SpeechOut, SpeechPosition, SpeechSampler, State, Tap, WriteResult,
};
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_bmg::{IGC_REVISION, KERNEL_ENTRY, OCLOC_VERSION, QWEN3TTS_KERNEL};

use crate::device::{Device, OnCard};
use crate::model::{self, Sampling, Slot, Tasks, Work};
use crate::plan::{SCRATCH_FLOATS, limits, weights_bytes};
use crate::spec::*;
use crate::weights::{Weights, f32_values, load};

/// The request in a slot.
#[derive(Clone, Copy, Debug)]
struct Request {
    /// The positions of the prompt.
    prompt: usize,
    /// The frames whose 16 codes are on the card. Codebook 0 of the next frame is there too.
    frames: usize,
    /// The frames that the codec decoded.
    decoded: usize,
    sampling: Sampling,
    end: bool,
}

/// What a forced run gives: the logits at each step, and the audio of all the frames (`tests/xe_oracle.rs`).
pub struct Forced {
    /// The talker's logits of codebook 0: after the prompt, then after each frame.
    pub talker_logits: Vec<Vec<f32>>,
    /// The logits of codebooks 1 to 15 of each frame, 15 rows of 2048.
    pub cp_logits: Vec<Vec<f32>>,
    /// The talker's hidden state after its norm, at the same steps as `talker_logits`.
    pub talker_hidden: Vec<Vec<f32>>,
    /// The audio of the frames, as floats before the PCM conversion.
    pub audio: Vec<f32>,
    /// The PCM of the frames.
    pub pcm: Vec<i16>,
}

/// The speech engine.
pub struct Qwen3TtsEngine {
    config: Config,
    limits: Limits,
    identity: Identity,
    last_error: Option<String>,
    cluster: L0Cluster,
    ring: Ring,
    device: Device,
    weights: Weights,
    work: Work,
    slots: Vec<(Slot, Option<Request>)>,
    /// The rows of the talker KV that each slot holds while it has no line: after a restore (`set_n`).
    held: Vec<usize>,
}

// The engine holds USM pointers of its card, and one thread at a time calls it.
unsafe impl Send for Qwen3TtsEngine {}

impl Drop for Qwen3TtsEngine {
    /// Free the weights, the slots and the ring (engine-abi "`close` in Rust"). The card destroys its context after
    /// this call, when the cluster drops.
    fn drop(&mut self) {
        let card = &self.cluster.cards[0];
        self.ring.free(card);
        self.device.free_all(card);
    }
}

fn refused(message: String) -> PassError {
    PassError {
        kind: PassErrorKind::Refused,
        message,
    }
}

/// The BLAKE3 digest of the files of the checkpoint that the engine reads: the two configs and the two tensor files.
fn checkpoint_digest(model_dir: &Path, maps: [&MappedSafeTensors; 2]) -> Result<String, String> {
    let mut all = blake3::Hasher::new();
    for name in ["config.json", "speech_tokenizer/config.json"] {
        let path = model_dir.join(name);
        let bytes = std::fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        all.update(blake3::hash(&bytes).as_bytes());
    }
    for map in maps {
        // Each chunk's hash is independent, so the threads share the work.
        let chunks: Vec<&[u8]> = map.as_bytes().chunks(256 << 20).collect();
        let hashes: Vec<blake3::Hash> = std::thread::scope(|scope| {
            let threads: Vec<_> = chunks
                .iter()
                .map(|chunk| scope.spawn(move || blake3::hash(chunk)))
                .collect();
            threads
                .into_iter()
                .map(|t| t.join().expect("a hash thread panicked"))
                .collect()
        });
        for hash in hashes {
            all.update(hash.as_bytes());
        }
    }
    Ok(format!("blake3:{}", all.finalize().to_hex()))
}

fn sha256_hex(parts: &[&[u8]]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    format!("sha256:{:x}", hasher.finalize())
}

impl Qwen3TtsEngine {
    /// Open the card at the PCI address `pci`, and upload the checkpoint in `config.model_dir`.
    pub fn open(config: Config, pci: &str) -> Result<Self, String> {
        config.check()?;
        check_config(&config.model_dir)?;
        let limits = limits(&config, weights_bytes(&config)?);
        let dir = &config.model_dir;
        let model_map = MappedSafeTensors::open(&dir.join("model.safetensors"))?;
        let codec_map = MappedSafeTensors::open(&dir.join("speech_tokenizer/model.safetensors"))?;
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|error| {
                format!(
                    "cannot read {}: {error}; the checkpoint holds its tokenizer",
                    path.display()
                )
            })
        };
        // The tokenizer is the two files that the family library reads (spec 11.8).
        let tokenizer_hash = sha256_hex(&[&read("vocab.json")?, &read("merges.txt")?]);
        let checkpoint_digest = checkpoint_digest(dir, [&model_map, &codec_map])?;

        let mut cluster = L0Cluster::init(QWEN3TTS_KERNEL, KERNEL_ENTRY);
        let index = cluster.cards_at(&[pci.to_owned()])?[0];
        // The engine keeps its card. A drop of the other cards destroys their contexts.
        let card = cluster.cards.swap_remove(index);
        cluster.cards.clear();
        cluster.cards.push(card);
        let card = &cluster.cards[0];
        let driver = cluster.driver_version();

        let mut device = Device::new(card);
        let weights = load(
            &mut OnCard {
                device: &mut device,
                card,
            },
            &model_map.view()?,
            &codec_map.view()?,
        )?;
        let mut alloc = |bytes: usize| device.zeros(card, bytes);
        let work = Work::new(&mut alloc, &config);
        let slots: Vec<(Slot, Option<Request>)> = (0..config.slots)
            .map(|_| (Slot::new(&mut alloc, &config), None))
            .collect();
        device.write(card, work.pad_id, &TTS_PAD.to_le_bytes());
        let ring = Ring::new(card, SCRATCH_FLOATS);

        let mut numbers = BTreeMap::new();
        numbers.insert(
            "kernel.megakernel_qwen3tts_bmg.bin".to_owned(),
            sha256_hex(&[QWEN3TTS_KERNEL]),
        );
        numbers.insert("ocloc".to_owned(), OCLOC_VERSION.to_owned());
        numbers.insert("igc".to_owned(), IGC_REVISION.to_owned());
        numbers.insert("driver".to_owned(), driver.to_string());
        numbers.insert("frames_per_quantum".to_owned(), config.frames_per_quantum.to_string());
        numbers.insert("canary".to_owned(), "not run".to_owned());
        let identity = Identity {
            model: MODEL.to_owned(),
            checkpoint_digest,
            family: "qwen3tts".to_owned(),
            tokenizer_hash,
            template: String::new(),
            numbers,
            plan_table: format!(
                "ocloc {OCLOC_VERSION} igc {IGC_REVISION} bmg-g31; level-zero driver {driver}; devices {:04x}:{:04x}",
                card.vendor_id, card.device_id
            ),
            bit_settings: Vec::new(),
            canary_hash: String::new(),
        };
        let mut engine = Self {
            config,
            limits,
            identity,
            last_error: None,
            cluster,
            ring,
            device,
            weights,
            work,
            held: vec![0; slots.len()],
            slots,
        };
        engine.run(model::pad_row(&engine.weights, &engine.work));
        Ok(engine)
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn run(&mut self, tasks: Tasks) {
        self.ring.set(&tasks.0);
        // SAFETY: each task points at buffers of this engine on its card, sized by the config for the pass.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }

    fn read_f32(&self, at: u64, n: usize) -> Vec<f32> {
        f32_values(&self.device.read(self.card(), at, n * 4))
    }

    fn read_u32(&self, at: u64, n: usize) -> Vec<u32> {
        self.device
            .read(self.card(), at, n * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    /// Check a prompt and put its ids on the card.
    fn put_prompt(&self, prompt: &[PromptPair]) -> Result<(), PassError> {
        if prompt.is_empty() || prompt.len() > self.config.max_prompt as usize {
            return Err(refused(format!(
                "the prompt has {} positions, and the engine takes 1 to {}; give a shorter text",
                prompt.len(),
                self.config.max_prompt
            )));
        }
        for (at, pair) in prompt.iter().enumerate() {
            if pair.text.is_some_and(|id| id as usize >= TEXT_VOCAB)
                || pair.codec.is_some_and(|id| id as usize >= CODEC_VOCAB)
            {
                return Err(refused(format!(
                    "prompt position {at} has the ids {pair:?}, outside the vocabularies of {TEXT_VOCAB} text ids \
                     and {CODEC_VOCAB} codec ids"
                )));
            }
        }
        let (text, codec) = model::prompt_ids(prompt);
        self.device.write(self.card(), self.work.ids_text, &text);
        self.device.write(self.card(), self.work.ids_codec, &codec);
        Ok(())
    }

    /// The rows of the talker KV that slot `index` holds: a line's prompt and fed frames, or the rows of a restore.
    fn rows(&self, index: usize) -> usize {
        match self.slots[index].1 {
            Some(line) => line.prompt + line.frames,
            None => self.held[index],
        }
    }

    /// Check the ids of feed positions against the vocabularies.
    fn check_positions(&self, positions: &[SpeechPosition]) -> Result<(), PassError> {
        if positions.is_empty() {
            return Err(refused("the feed has 0 positions; give at least one".to_owned()));
        }
        for (at, p) in positions.iter().enumerate() {
            let fits = match p {
                SpeechPosition::Pair(pair) => {
                    !pair.text.is_some_and(|id| id as usize >= TEXT_VOCAB)
                        && !pair.codec.is_some_and(|id| id as usize >= CODEC_VOCAB)
                }
                SpeechPosition::Frame(codes) => {
                    codes.len() == CODEBOOKS
                        && (codes[0] as usize) < CODEC_VOCAB
                        && codes[1..].iter().all(|&c| (c as usize) < CP_VOCAB)
                }
            };
            if !fits {
                return Err(refused(format!(
                    "feed position {at} is {p:?}, outside the vocabularies of {TEXT_VOCAB} text ids, {CODEC_VOCAB} \
                     codec ids and {CP_VOCAB} codes, or not {CODEBOOKS} codes"
                )));
            }
        }
        Ok(())
    }

    /// The device address of layer `l`'s K (`kv` 0) or V (`kv` 1) row `r` in a slot's KV.
    fn kv_row(&self, index: usize, l: usize, kv: usize, r: usize) -> u64 {
        let max_seq = self.config.max_positions() as usize;
        let kvd = KV_HEADS * HEAD_DIM;
        self.slots[index].0.kv + (((l * 2 + kv) * max_seq + r) * kvd * 4) as u64
    }

    fn slot_index(&self, slot: SlotId) -> Result<usize, PassError> {
        if (slot as usize) < self.slots.len() {
            Ok(slot as usize)
        } else {
            Err(refused(format!(
                "slot {slot} does not exist; the engine has {} slots",
                self.slots.len()
            )))
        }
    }

    /// Run the prompt and the given frames with no sampling, and give every logit row and the audio of the frames.
    /// The codec decodes in quanta of `frames_per_quantum`. The oracle checks the passes with it (spec 8.3).
    pub fn forced(&mut self, prompt: &[PromptPair], codes: &[[u32; CODEBOOKS]]) -> Result<Forced, PassError> {
        if codes.len() > self.config.max_frames as usize {
            return Err(refused(format!(
                "a forced run of {} frames is more than max_frames {}",
                codes.len(),
                self.config.max_frames
            )));
        }
        self.put_prompt(prompt)?;
        let flat: Vec<u8> = codes.iter().flatten().flat_map(|c| c.to_le_bytes()).collect();
        let n = prompt.len();
        let (w, k, s) = (&self.weights, &self.work, &self.slots[0].0);
        self.device.write(self.card(), s.codes, &flat);
        for (at, bytes) in s.codec_state(&self.config) {
            self.card().fill_zero_sync(at as *mut std::ffi::c_void, bytes);
        }
        let tasks = model::prefill(w, k, s, &self.config, n, None);
        self.run(tasks);
        let mut out = Forced {
            talker_logits: vec![self.read_f32(self.work.logits, CODEC_VOCAB)],
            cp_logits: Vec::new(),
            talker_hidden: vec![self.read_f32(self.slots[0].0.hidden, HIDDEN)],
            audio: Vec::new(),
            pcm: Vec::new(),
        };
        for frame in 0..codes.len() {
            let tasks = model::frame(
                &self.weights,
                &self.work,
                &self.slots[0].0,
                &self.config,
                n,
                frame,
                None,
            );
            self.run(tasks);
            out.cp_logits
                .push(self.read_f32(self.work.cp_logits, (CODEBOOKS - 1) * CP_VOCAB));
            out.talker_logits.push(self.read_f32(self.work.logits, CODEC_VOCAB));
            out.talker_hidden.push(self.read_f32(self.slots[0].0.hidden, HIDDEN));
        }
        let quantum = self.config.frames_per_quantum as usize;
        let mut done = 0;
        while done < codes.len() {
            let n = quantum.min(codes.len() - done);
            let tasks = model::codec(&self.weights, &self.work, &self.slots[0].0, &self.config, done, n, done);
            self.run(tasks);
            out.audio.extend(self.read_f32(self.work.wav, n * SAMPLES_PER_FRAME));
            out.pcm.extend(self.read_pcm(n));
            done += n;
        }
        Ok(out)
    }

    /// The rows of a prompt that the talker takes, before its first layer. The oracle checks them (spec 8.3).
    pub fn prompt_rows(&mut self, prompt: &[PromptPair]) -> Result<Vec<f32>, PassError> {
        self.put_prompt(prompt)?;
        self.run(model::prompt_rows(&self.weights, &self.work, prompt.len()));
        Ok(self.read_f32(self.work.x, prompt.len() * HIDDEN))
    }

    fn read_pcm(&self, frames: usize) -> Vec<i16> {
        self.device
            .read(self.card(), self.work.pcm, frames * SAMPLES_PER_FRAME * 2)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect()
    }
}

impl Speech for Qwen3TtsEngine {
    fn speech_feed(
        &mut self,
        slot: SlotId,
        positions: &[SpeechPosition],
        sampler: Option<&SpeechSampler>,
    ) -> Result<(), PassError> {
        let index = self.slot_index(slot)?;
        if self.slots[index].1.is_some_and(|line| !line.end) {
            return Err(refused(format!(
                "slot {slot} speaks a line that has not ended; make its frames to the end, or clear the slot"
            )));
        }
        if let Some(sampler) = sampler
            && (!(sampler.temperature > 0.0 && sampler.temperature.is_finite())
                || !(sampler.repetition_penalty > 0.0 && sampler.repetition_penalty.is_finite()))
        {
            return Err(refused(format!(
                "the sampler has temperature {} and repetition penalty {}; give positive finite numbers",
                sampler.temperature, sampler.repetition_penalty
            )));
        }
        self.check_positions(positions)?;
        let base = self.rows(index);
        let room = self.config.max_positions() as usize;
        if base + positions.len() >= room {
            return Err(refused(format!(
                "slot {slot} holds {base} positions, and {} more leave no room for a frame in its {room}; start a \
                 shorter context",
                positions.len()
            )));
        }
        let sampling = sampler.map(|sampler| {
            let s = &self.slots[index].0;
            let state = model::sampler_state(sampler.seed);
            self.device.write(self.card(), s.state, &state);
            // The code predictor draws from its own stream, so the draws of codebook 0 do not move with it.
            let cp_state = model::sampler_state(sampler.seed ^ 0x9e37_79b9_7f4a_7c15);
            self.device.write(self.card(), s.cp_state, &cp_state);
            for (at, bytes) in s.codec_state(&self.config) {
                self.card().fill_zero_sync(at as *mut std::ffi::c_void, bytes);
            }
            Sampling {
                temperature: sampler.temperature,
                top_k: sampler.top_k,
                penalty: sampler.repetition_penalty,
            }
        });
        // A feed longer than the work buffers runs in chunks. With a sampler, the last chunk samples codebook 0 of the
        // line's first frame.
        let chunk = self.config.max_prompt as usize;
        let mut at = base;
        let count = positions.len().div_ceil(chunk);
        for (k, part) in positions.chunks(chunk).enumerate() {
            let (text, codec, cp, pad) = model::feed_ids(part, chunk);
            self.device.write(self.card(), self.work.ids_text, &text);
            self.device.write(self.card(), self.work.ids_codec, &codec);
            self.device.write(self.card(), self.work.ids_cp, &cp);
            self.device.write(self.card(), self.work.ids_pad, &pad);
            let sample = sampling.filter(|_| k + 1 == count);
            let tasks = model::feed(
                &self.weights,
                &self.work,
                &self.slots[index].0,
                &self.config,
                at,
                part.len(),
                sample,
            );
            self.run(tasks);
            at += part.len();
        }
        let Some(sampling) = sampling else {
            self.slots[index].1 = None;
            self.held[index] = at;
            return Ok(());
        };
        let first = self.read_u32(self.slots[index].0.code(0, 0), 1)[0];
        self.held[index] = 0;
        self.slots[index].1 = Some(Request {
            prompt: at,
            frames: 0,
            decoded: 0,
            sampling,
            end: first == CODEC_EOS,
        });
        Ok(())
    }

    fn speech_frames(&mut self, slot: SlotId, max: u32) -> Result<SpeechOut, PassError> {
        let index = self.slot_index(slot)?;
        let Some(mut request) = self.slots[index].1 else {
            return Err(refused(format!("slot {slot} holds no request; begin one first")));
        };
        // Frame f writes its input at row `prompt + f`, so the rows of the slot bound a line too.
        let positions_left = self.config.max_positions() as usize - (request.prompt + request.frames);
        let room = (self.config.max_frames as usize - request.frames).min(positions_left);
        let want = (max.min(self.config.frames_per_quantum) as usize).min(room);
        let first = request.frames;
        while !request.end && request.frames < first + want {
            let tasks = model::frame(
                &self.weights,
                &self.work,
                &self.slots[index].0,
                &self.config,
                request.prompt,
                request.frames,
                Some(request.sampling),
            );
            self.run(tasks);
            request.frames += 1;
            let next = self.read_u32(self.slots[index].0.code(request.frames, 0), 1)[0];
            // A request ends at end-of-speech, or when it has no room for another frame.
            request.end = next == CODEC_EOS
                || request.frames == self.config.max_frames as usize
                || request.prompt + request.frames + 1 >= self.config.max_positions() as usize;
        }
        let n = request.frames - first;
        let mut out = SpeechOut {
            frames: n as u32,
            codes: Vec::new(),
            audio: Vec::new(),
            end: request.end,
        };
        if n > 0 {
            let s = &self.slots[index].0;
            out.codes = self.read_u32(s.code(first, 0), n * CODEBOOKS);
            let tasks = model::codec(&self.weights, &self.work, s, &self.config, first, n, request.decoded);
            self.run(tasks);
            out.audio = self.read_pcm(n);
            request.decoded += n;
        }
        self.slots[index].1 = Some(request);
        Ok(out)
    }

    fn speech_clear(&mut self, slot: SlotId) {
        if let Some((_, request)) = self.slots.get_mut(slot as usize) {
            *request = None;
            self.held[slot as usize] = 0;
        }
    }
}

/// The bytes of one row of the cells part `talker_kv`: the K and V of every layer at one position.
pub const KV_ROW_BYTES: usize = LAYERS * 2 * KV_HEADS * HEAD_DIM * 4;

impl Parts for Qwen3TtsEngine {
    fn part_read(&mut self, _state: State, _part: PartId, _out: &mut [u8]) -> Result<(), PartError> {
        // The engine's only part is a cells part.
        Err(PartError::Refused)
    }

    fn part_write(&mut self, _state: State, _part: PartId, _bytes: &[u8]) -> WriteResult {
        WriteResult::Refused
    }

    /// Rows `from..from + n` of `talker_kv`: for each row, the K and V of each layer, as `[layer][k, v][head][dim]`.
    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let State::Slot(slot) = state;
        let (from, n) = (from as usize, n as usize);
        let Ok(index) = self.slot_index(slot) else {
            return Err(PartError::Refused);
        };
        if part != 0 || from + n > self.rows(index) || out.len() != n * KV_ROW_BYTES {
            return Err(PartError::Refused);
        }
        let kvd = KV_HEADS * HEAD_DIM * 4;
        for l in 0..LAYERS {
            for kv in 0..2 {
                let plane = self.device.read(self.card(), self.kv_row(index, l, kv, from), n * kvd);
                for r in 0..n {
                    let at = r * KV_ROW_BYTES + (l * 2 + kv) * kvd;
                    out[at..at + kvd].copy_from_slice(&plane[r * kvd..(r + 1) * kvd]);
                }
            }
        }
        Ok(())
    }

    /// Write rows of `talker_kv` into a slot with no line, as a restore does (6.8).
    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        let State::Slot(slot) = state;
        let (from, n) = (from as usize, n as usize);
        let Ok(index) = self.slot_index(slot) else {
            return WriteResult::Refused;
        };
        if part != 0
            || self.slots[index].1.is_some()
            || from + n > self.config.max_positions() as usize
            || bytes.len() != n * KV_ROW_BYTES
        {
            return WriteResult::Refused;
        }
        let kvd = KV_HEADS * HEAD_DIM * 4;
        for l in 0..LAYERS {
            for kv in 0..2 {
                let mut plane = vec![0u8; n * kvd];
                for r in 0..n {
                    let at = r * KV_ROW_BYTES + (l * 2 + kv) * kvd;
                    plane[r * kvd..(r + 1) * kvd].copy_from_slice(&bytes[at..at + kvd]);
                }
                self.device.write(self.card(), self.kv_row(index, l, kv, from), &plane);
            }
        }
        WriteResult::Ok
    }

    /// Set the rows that a slot with no line holds, after a write of its cells.
    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let State::Slot(slot) = state;
        let Ok(index) = self.slot_index(slot) else {
            return WriteResult::Refused;
        };
        if self.slots[index].1.is_some() || n as usize >= self.config.max_positions() as usize {
            return WriteResult::Refused;
        }
        self.held[index] = n as usize;
        WriteResult::Ok
    }
}

impl Slots for Qwen3TtsEngine {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.slot_index(slot).map_or(0, |index| self.rows(index) as u64)
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        if self.slot_index(slot).is_err() {
            return WriteResult::Refused;
        }
        self.speech_clear(slot);
        WriteResult::Ok
    }
}

impl Engine for Qwen3TtsEngine {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// A speech engine takes no pass (engine-abi "Speech").
    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        let error = refused("a speech engine takes no pass; use speech_feed and speech_frames".to_owned());
        self.last_error = Some(error.message.clone());
        Err(error)
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
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
