//! The speech path of Omni on the card (milestone 24 step 8; engine-abi "The speech calls of LFM3"): the depth
//! decoder and the sampler of `frame`, the feed of a frame, and the codec of each slot.

use nagaya_abi::{FrameDraw, FrameOut, PassError, Segment, SegmentInputs, SlotId, SpeechFrames, State};
use nagaya_mk_bmg::level_zero::B70Card;
use nagaya_mk_bmg::protocol::OP_EMBED;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_bmg::tasks;

use super::{Lfm3Engine, refused};
use crate::codes::{self, CODEBOOKS, CODES, END_CODE, SCHEDULE};
use crate::device::{Device, OnCard};
use crate::frames::{AUDIO_OUT_TOKEN, check_configs, term_bytes};
use crate::model::{self, Run};
use crate::spec::WEIGHTS_FILE;
use crate::weights::{f32_bytes, f32_values};
use crate::{codec, depth};

/// What a probe of the depth decoder gives: the input of the trunk and its output `[12][1024]`, and the logits of
/// each codebook `[12][2017]`.
pub struct DepthProbe {
    pub input: Vec<f32>,
    pub trunk: Vec<f32>,
    pub logits: Vec<f32>,
}

/// The speech state of one slot.
struct SpeechSlot {
    /// The final norm of the last position of the slot: `[2048]`.
    hidden: u64,
    /// The token count of the slot when a pass or a feed wrote `hidden`. Any other change of the slot clears it.
    hidden_at: Option<u64>,
    codec: codec::Slot,
    /// The frames that the codec decoded since its last clear.
    frames: usize,
    overlap: codec::Overlap,
}

/// The speech path of the engine: the weights, the buffers and the state of each slot.
pub(super) struct Speech {
    depth: depth::Weights,
    codec: codec::Weights,
    depth_bufs: depth::Bufs,
    codec_bufs: codec::Bufs,
    /// The synthesis window of the codec, for the overlap-add on the host.
    window: Vec<f32>,
    slots: Vec<SpeechSlot>,
    /// The device bytes of the speech path.
    pub bytes: u64,
}

impl Speech {
    /// Upload the depth decoder and the codec of the folder `dir` of Omni, and give each of `slots` slots its state.
    /// The device bytes must equal the term of the plan.
    pub fn open(dir: &std::path::Path, card: &B70Card, device: &mut Device, slots: usize) -> Result<Self, String> {
        check_configs(dir)?;
        let map = MappedSafeTensors::open(&dir.join(WEIGHTS_FILE))?;
        let codec_map = MappedSafeTensors::open(&dir.join(codec::WEIGHTS_FILE))?;
        let (view, codec_view) = (map.view()?, codec_map.view()?);
        let before = device.bytes;
        let mut sink = OnCard {
            device: &mut *device,
            card,
        };
        let depth = depth::load(&mut sink, &view)?;
        let codec = codec::load(&mut sink, &codec_view)?;
        let weights = device.bytes - before;
        let window = f32_values(
            codec_view
                .tensor_bytes(codec::WINDOW_TENSOR)
                .expect("codec::check found the window"),
        );
        let depth_bufs = depth::Bufs::new(&mut |bytes| device.zeros(card, bytes));
        let codec_bufs = codec::Bufs::new(&mut |bytes| device.zeros(card, bytes));
        let slots = (0..slots)
            .map(|_| SpeechSlot {
                hidden: device.zeros(card, depth::BACKBONE * 4),
                hidden_at: None,
                codec: codec::Slot::new(&mut |bytes| device.zeros(card, bytes)),
                frames: 0,
                overlap: codec::Overlap::default(),
            })
            .collect::<Vec<_>>();
        let bytes = (device.bytes - before) as u64;
        let planned = term_bytes(weights, slots.len());
        if bytes != planned {
            return Err(format!(
                "the speech path took {bytes} device bytes, and the plan counted {planned}"
            ));
        }
        Ok(Self {
            depth,
            codec,
            depth_bufs,
            codec_bufs,
            window,
            slots,
            bytes,
        })
    }

    /// Clear the hidden row of `slot`: the slot changed in another way than a pass or a feed.
    pub fn forget(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.hidden_at = None;
        }
    }

    /// The tasks that keep row `row` of the final norm in `h` as the hidden row of `slot`.
    pub fn keep_hidden(&self, slot: usize, h: u64, row: usize) -> nagaya_mk_bmg::protocol::MkTaskDescriptor {
        tasks::copy(
            h + (row * depth::BACKBONE * 4) as u64,
            self.slots[slot].hidden,
            depth::BACKBONE as u32,
        )
    }

    /// Take the hidden row that a pass or a feed wrote at token count `n` of `slot`.
    pub fn kept(&mut self, slot: usize, n: u64) {
        self.slots[slot].hidden_at = Some(n);
    }

    /// Clear the codec of `slot`, and give the last samples of its stream.
    pub fn clear_codec(&mut self, slot: usize) -> Vec<f32> {
        let s = &mut self.slots[slot];
        s.frames = 0;
        s.overlap.finish()
    }
}

impl Lfm3Engine {
    fn speech_ref(&self) -> Result<&Speech, PassError> {
        self.speech.as_ref().ok_or_else(|| {
            refused(format!(
                "{} has no speech path; only lfm3-omni speaks",
                self.b.model.name()
            ))
        })
    }

    fn speech_slot(&self, slot: SlotId) -> Result<usize, PassError> {
        let speech = self.speech_ref()?;
        let slot = slot as usize;
        if slot >= speech.slots.len() {
            return Err(refused(format!(
                "the engine has {} slots, not {slot}",
                speech.slots.len()
            )));
        }
        Ok(slot)
    }

    /// Run one pass of the depth decoder on the hidden row at `hidden`, with the codes `codes` of the codebooks that
    /// `revealed` names, and give the logits of each codebook of `picks`, in order.
    fn depth_pass(&mut self, hidden: u64, codes: &[u32], revealed: &[bool], picks: &[usize]) -> Vec<Vec<f32>> {
        let speech = self.speech.as_ref().expect("the caller checked the speech path");
        let ids: Vec<u8> = depth::ids(codes, revealed)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.device.write(self.card(), speech.depth_bufs.ids, &ids);
        let tasks = depth::pass(&speech.depth, &speech.depth_bufs, hidden, picks);
        let logits = speech.depth_bufs.logits;
        self.run(&tasks);
        let all = self.read_f32(logits, picks.len() * CODES);
        all.chunks(CODES).map(|row| row.to_vec()).collect()
    }

    /// The probe of the oracle (8.3): the input of the trunk, the trunk output and the logits `[12][2017]` of the
    /// depth decoder on the hidden row `hidden`, with the codes of the codebooks that `masked` does not hide.
    pub fn depth_probe(&mut self, hidden: &[f32], codes: &[u32], masked: &[bool]) -> Result<DepthProbe, String> {
        let speech = self.speech_ref().map_err(|e| e.message)?;
        if hidden.len() != depth::BACKBONE || codes.len() != CODEBOOKS || masked.len() != CODEBOOKS {
            return Err("a probe takes a row of 2,048 values, 12 codes and 12 masks".to_owned());
        }
        if let Some(code) = codes.iter().find(|c| **c as usize >= CODES) {
            return Err(format!("code {code} is past the codebook of {CODES} codes"));
        }
        let bufs = speech.depth_bufs;
        let probe = bufs.probe;
        self.device.write(self.card(), probe, &f32_bytes(hidden));
        let revealed: Vec<bool> = masked.iter().map(|m| !m).collect();
        // The input alone: the first three tasks of a pass.
        let ids: Vec<u8> = depth::ids(codes, &revealed)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.device.write(self.card(), bufs.ids, &ids);
        let first = depth::pass(&speech.depth, &bufs, probe, &[]);
        assert_eq!(first[2].opcode, OP_EMBED, "the third task of a pass adds the codes");
        self.run(&first[..3]);
        let input = self.read_f32(bufs.x, CODEBOOKS * depth::HIDDEN);
        let all: Vec<usize> = (0..CODEBOOKS).collect();
        let logits = self.depth_pass(probe, codes, &revealed, &all).concat();
        let trunk = self.read_f32(bufs.h, CODEBOOKS * depth::HIDDEN);
        Ok(DepthProbe { input, trunk, logits })
    }

    /// Feed one frame of codes into `slot` as its next position: the sum of the 12 rows of the codebook table, through
    /// the backbone. It keeps the hidden row of the position, and gives its logits when `logits` is true.
    pub fn feed_frame(&mut self, slot: SlotId, codes: &[u32], logits: bool) -> Result<Option<Vec<f32>>, PassError> {
        let index = self.speech_slot(slot)?;
        if codes.len() != CODEBOOKS {
            return Err(refused(format!("a frame has 12 codes, not {}", codes.len())));
        }
        if let Some(code) = codes.iter().find(|c| **c as usize >= CODES) {
            return Err(refused(format!("code {code} is past the codebook of {CODES} codes")));
        }
        let pos = self.held.n[index];
        let segment = Segment {
            state: State::Slot(slot),
            inputs: SegmentInputs::Tokens(vec![AUDIO_OUT_TOKEN]),
            positions: vec![pos],
            meta: Vec::new(),
            outputs: nagaya_abi::OutputsWanted {
                logit_rows: if logits { vec![pos] } else { Vec::new() },
            },
        };
        let block = self.check_block(&[segment])?;
        if block.pos0 == 0 {
            self.zero_conv(index);
        }
        let speech = self.speech.as_ref().expect("speech_slot checked the speech path");
        let ids: Vec<u8> = depth::frame_ids(codes).iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.write(self.card(), speech.depth_bufs.frame_ids, &ids);
        let rows: &[usize] = if logits { &[0] } else { &[] };
        let mut tasks = model::pass(
            &self.b,
            &self.weights,
            &self.bufs,
            Run {
                slot: &self.slots[index],
                pos0: block.pos0,
                n: 1,
                layers: self.b.layers.len(),
                logit_rows: Some(rows),
            },
        );
        assert_eq!(
            tasks[0].opcode, OP_EMBED,
            "the first task of a pass is the embedding of its tokens"
        );
        tasks.splice(
            0..1,
            depth::frame_embedding(&speech.depth, &speech.depth_bufs, self.bufs.x),
        );
        tasks.push(speech.keep_hidden(index, self.bufs.h, 0));
        self.run(&tasks);
        let out = logits.then(|| self.read_f32(self.bufs.logits, self.vocab));
        self.held.ran(&block);
        let n = self.held.n[index];
        self.speech.as_mut().expect("checked").kept(index, n);
        Ok(out)
    }

    /// The hidden row of the last position of `slot`, as a pass or a feed kept it. The probe of the oracle reads it.
    pub fn hidden_row(&self, slot: SlotId) -> Result<Vec<f32>, String> {
        let index = self.speech_slot(slot).map_err(|e| e.message)?;
        let s = &self.speech.as_ref().expect("checked").slots[index];
        if s.hidden_at != Some(self.held.n[index]) || self.held.n[index] == 0 {
            return Err(format!(
                "slot {slot} holds no hidden row of its last position; run a pass first"
            ));
        }
        Ok(self.read_f32(s.hidden, depth::BACKBONE))
    }
}

impl SpeechFrames for Lfm3Engine {
    fn codebooks(&self) -> u32 {
        CODEBOOKS as u32
    }

    /// Run the depth decoder on the hidden row of the last position of the slot, sample 12 codes, and feed the frame.
    fn frame(&mut self, slot: SlotId, draw: &FrameDraw<'_>, logits: bool) -> Result<FrameOut, PassError> {
        let result = (|| {
            let index = self.speech_slot(slot)?;
            let s = &self.speech.as_ref().expect("checked").slots[index];
            let n = self.held.n[index];
            if n == 0 || s.hidden_at != Some(n) {
                return Err(refused(format!(
                    "slot {slot} holds no hidden row of its last position; run a pass after a restore, a write or a \
                     clear"
                )));
            }
            if !draw.history.len().is_multiple_of(CODEBOOKS) {
                return Err(refused(format!(
                    "the history has {} codes; it holds 12 codes for each frame",
                    draw.history.len()
                )));
            }
            if let Some(code) = draw.history.iter().find(|c| **c >= END_CODE) {
                return Err(refused(format!(
                    "the history holds the code {code}; it holds the codes 0 to 2015 of the frames that made audio"
                )));
            }
            let hidden = s.hidden;
            let order = codes::order(draw.seed, draw.frame);
            let mut codes = vec![0u32; CODEBOOKS];
            let mut revealed = vec![false; CODEBOOKS];
            let mut start = 0;
            for size in SCHEDULE {
                let picks = &order[start..start + size];
                let rows = self.depth_pass(hidden, &codes, &revealed, picks);
                for (g, row) in picks.iter().zip(&rows) {
                    codes[*g] = codes::sample(row, draw.history, *g, draw.seed, draw.frame);
                }
                for g in picks {
                    revealed[*g] = true;
                }
                start += size;
            }
            let end = codes::ends(&codes);
            if end {
                codes = vec![END_CODE; CODEBOOKS];
            }
            let logits = self.feed_frame(slot, &codes, logits)?;
            Ok(FrameOut { codes, end, logits })
        })();
        result.map_err(|e| self.fail(e))
    }

    /// Decode one frame of codes in the codec of the slot, and give the samples that are whole.
    fn codec(&mut self, slot: SlotId, codes: &[u32]) -> Result<Vec<f32>, PassError> {
        let result = (|| {
            let index = self.speech_slot(slot)?;
            if codes.len() != CODEBOOKS {
                return Err(refused(format!("a frame has 12 codes, not {}", codes.len())));
            }
            if let Some(code) = codes.iter().find(|c| **c >= END_CODE) {
                return Err(refused(format!(
                    "code {code} makes no audio: the codec takes the codes 0 to 2015, and the end frame is not part \
                     of the audio"
                )));
            }
            let speech = self.speech.as_ref().expect("checked");
            let s = &speech.slots[index];
            let pos0 = s.frames * codec::STACK;
            if pos0 + codec::STACK > codec::MAX_POSITIONS {
                return Err(refused(format!(
                    "the codec of slot {slot} holds {} frames, the most of a slot",
                    s.frames
                )));
            }
            if s.frames == 0 {
                for (at, bytes) in s.codec.conv_states() {
                    self.device.zero(self.card(), at, bytes);
                }
            }
            let latents = codec::latents(codes);
            self.device
                .write(self.card(), speech.codec_bufs.latents, &f32_bytes(&latents));
            let tasks = codec::frame(&speech.codec, &speech.codec_bufs, &s.codec, pos0);
            let frames_at = speech.codec_bufs.frames;
            self.run(&tasks);
            let frames = self.read_f32(frames_at, codec::STACK * codec::N_FFT);
            let speech = self.speech.as_mut().expect("checked");
            let s = &mut speech.slots[index];
            s.frames += 1;
            Ok(s.overlap.add(&frames, &speech.window))
        })();
        result.map_err(|e| self.fail(e))
    }

    /// Give the last 480 samples of the stream of the slot, and clear its codec. A slot with no stream gives none.
    fn codec_clear(&mut self, slot: SlotId) -> Vec<f32> {
        match self.speech_slot(slot) {
            Ok(index) => self.speech.as_mut().expect("checked").clear_codec(index),
            Err(_) => Vec::new(),
        }
    }
}
