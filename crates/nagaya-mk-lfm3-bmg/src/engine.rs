//! The LFM3 engine on its card: the `Engine` trait, the parts, the slots and the coverage over the ring of `model`.
//! The speech path of Omni is in `speech`.

mod speech;

pub use speech::DepthProbe;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nagaya_abi::{
    BitClass, BitSetting, Control, ControlLimit, ControlOp, Coverage, CoverageSpan, Dtype, Engine, Identity, Limits,
    Output, PartError, PartId, Parts, PassError, PassErrorKind, PassOut, PassStatus, Segment, SlotId, Slots,
    SpeechFrames, State, Tap, WriteResult,
};
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView};
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{IGC_REVISION, KERNEL_ENTRY, LFM3_KERNEL, OCLOC_VERSION};
use nagaya_text_lfm3::{Kind as FamilyKind, Lfm3Family};

use crate::audio::{self, AudioBufs, AudioWeights, Clip, MAX_CLIP_POSITIONS, Stage};
use crate::device::{Device, OnCard};
use crate::media::{self, MAX_IMAGE_PATCHES, PATCH_ROW, SpanKind};
use crate::model::{self, Batch, Bufs, Run, SlotMemory, bufs_bytes, scratch_floats};
use crate::spec::{self, Backbone, Block, Config, FAMILY, Held, LayerKind, Model, Part, RING_ROWS, VisionBytes};
use crate::spec::{LENS_TAP, MAX_SEGMENTS, RESIDUAL_TAP};
use crate::vision::{self, HeldRows, Until, Vision, VisionBufs, VisionWeights};
use crate::weights::{self, Count, Weights, f32_values};

/// The vision tower of Omni on the card: its numbers, its weights and its buffers.
struct Tower {
    v: Vision,
    w: VisionWeights,
    bufs: VisionBufs,
    /// The image whose rows the output of the projector holds.
    held: HeldRows,
}

/// The LFM3 engine on one card.
pub struct Lfm3Engine {
    b: Backbone,
    config: Config,
    limits: Limits,
    identity: Identity,
    /// The ids of the family: the width of a logit row, and the bound of a token id.
    vocab: usize,
    last_error: Option<String>,
    cluster: L0Cluster,
    ring: Ring,
    device: Device,
    weights: Weights,
    bufs: Bufs,
    /// The vision tower of Omni. `LFM3-1.7B` has none.
    tower: Option<Tower>,
    slots: Vec<SlotMemory>,
    held: Held,
    /// The weights and the buffers of the audio encoder of Omni (engine-abi "The audio encoder of LFM3").
    audio: Option<(AudioWeights, AudioBufs)>,
    /// The depth decoder and the codec of Omni. The text model has none.
    speech: Option<speech::Speech>,
    /// The control table on the card and its limit (engine-abi "Control vectors").
    control: Option<(u64, ControlLimit)>,
}

// The engine holds USM pointers of its card, and one thread at a time calls it.
unsafe impl Send for Lfm3Engine {}

impl Drop for Lfm3Engine {
    /// Free the weights, the slots, the buffers and the ring (engine-abi "`close` in Rust"). The card destroys its
    /// context after this call, when the cluster drops.
    fn drop(&mut self) {
        let card = &self.cluster.cards[0];
        self.ring.free(card);
        self.device.free_all(card);
    }
}

/// The taps that a pass names (spec 13.1).
#[derive(Clone, Copy, Debug, Default)]
struct Taps {
    residual: bool,
    lens: bool,
}

fn refused(message: String) -> PassError {
    PassError {
        kind: PassErrorKind::Refused,
        message,
    }
}

/// The logical digest of the checkpoint (spec 12.2): BLAKE3 with the key `nagaya lfm3 checkpoint v1` over the
/// canonical `config.json` and each tensor of the file, in bytewise order of its name. A tensor adds its name, its
/// dtype, its shape and the BLAKE3 of its bytes. The tensors of Omni's other towers are in it too, so a later step
/// that runs them does not change the digest.
pub fn checkpoint_digest(config_text: &str, view: &SafeTensorsView<'_>) -> Result<String, String> {
    let config: serde_json::Value =
        serde_json::from_str(config_text).map_err(|e| format!("config.json is not JSON: {e}"))?;
    let mut names: Vec<&String> = view.tensors.keys().collect();
    names.sort();
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let mut hashes: Vec<[u8; 32]> = vec![[0; 32]; names.len()];
    let results: Vec<Vec<(usize, [u8; 32])>> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(name) = names.get(index) else {
                            return done;
                        };
                        let bytes = view.tensor_bytes(name).expect("the name came from the view");
                        done.push((index, *blake3::hash(bytes).as_bytes()));
                    }
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().expect("a hash thread panicked"))
            .collect()
    });
    for (index, hash) in results.into_iter().flatten() {
        hashes[index] = hash;
    }
    let mut hasher = blake3::Hasher::new_derive_key("nagaya lfm3 checkpoint v1");
    let canonical = canonical_json(&config);
    hasher.update(&(canonical.len() as u64).to_le_bytes());
    hasher.update(canonical.as_bytes());
    for (name, hash) in names.iter().zip(&hashes) {
        let meta = &view.tensors[*name];
        let dtype = format!("{:?}", meta.dtype);
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(dtype.len() as u64).to_le_bytes());
        hasher.update(dtype.as_bytes());
        hasher.update(&(meta.shape.len() as u64).to_le_bytes());
        for dim in &meta.shape {
            hasher.update(&(*dim as u64).to_le_bytes());
        }
        hasher.update(hash);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

/// JSON with the keys of each object in bytewise order and no whitespace.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let items: Vec<String> = keys
                .into_iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        serde_json::Value::String(key.clone()),
                        canonical_json(&map[key])
                    )
                })
                .collect();
            format!("{{{}}}", items.join(","))
        }
        serde_json::Value::Array(items) => {
            let items: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", items.join(","))
        }
        other => other.to_string(),
    }
}

/// The PCI revision of the device at `address`, from sysfs.
fn pci_revision(address: &str) -> Result<u32, String> {
    let path = format!("/sys/bus/pci/devices/{address}/revision");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let text = text.trim();
    u32::from_str_radix(text.trim_start_matches("0x"), 16).map_err(|e| format!("{path} holds `{text}`: {e}"))
}

impl Lfm3Engine {
    /// Open the card at the PCI address `pci`, and upload the backbone of the model folder of `config`.
    ///
    /// The open checks the config, the family files and every tensor of the backbone before it opens the card. Its
    /// device bytes must equal the plan.
    pub fn open(config: Config, pci: &str) -> Result<Self, String> {
        config.check()?;
        let dir = config.model_dir.clone();
        let b = Backbone::read(&dir)?;
        let family = Lfm3Family::from_dir(&dir)
            .map_err(|e| format!("cannot build the LFM3 family from `{}`: {e}", dir.display()))?;
        let want = match b.model {
            Model::Text => FamilyKind::Text,
            Model::Omni => FamilyKind::Omni,
        };
        if family.kind() != want {
            return Err(format!(
                "the family of `{}` is {:?}, and config.json names {:?}",
                dir.display(),
                family.kind(),
                b.model
            ));
        }
        let vocab = family.tokenizer().len();
        if vocab == 0 || vocab > b.vocab_rows {
            return Err(format!(
                "tokenizer.json has {vocab} ids, and the embedding table has {} rows",
                b.vocab_rows
            ));
        }
        let config_text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("cannot read {}: {e}", dir.join("config.json").display()))?;
        let map = MappedSafeTensors::open(&dir.join(spec::WEIGHTS_FILE))?;
        let view = map.view()?;
        weights::check(&b, &view)?;
        let mut count = Count::default();
        weights::load(&mut count, &b, &view)?;
        if let Some(e) = &b.audio {
            audio::load(&mut count, e, &view)?;
        }
        // Omni's vision tower: its numbers, its tensors and its bytes, before the card opens.
        let vision = match b.model {
            Model::Text => None,
            Model::Omni => {
                let v = Vision::parse(&config_text, b.hidden)
                    .map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?;
                let mut tower_count = Count::default();
                vision::load(&mut tower_count, &v, &view)?;
                let mut rules = vision::media_rules(&config_text)?;
                rules.audio_encoder = b.audio.is_some();
                // Omni loads its speech path with its codebook table, so an assistant audio part can feed its frames.
                rules.audio_out_token = Some(crate::frames::AUDIO_OUT_TOKEN);
                Some((v, tower_count.bytes, rules))
            }
        };
        // Each model takes latent rows at positions of its pad token (engine-abi "Latent rows of LFM3").
        let latent = media::LatentRule::of(&config_text, b.hidden)
            .map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?;
        let checkpoint_digest = checkpoint_digest(&config_text, &view)?;

        // Only the engine's card gets a context, so the open does not stall the work of another card.
        let cluster = L0Cluster::init_at(LFM3_KERNEL, KERNEL_ENTRY, &[pci.to_owned()])?;
        let card = &cluster.cards[0];
        let driver = cluster.driver_version();
        let mut device = Device::new(card);
        let weights = weights::load(
            &mut OnCard {
                device: &mut device,
                card,
            },
            &b,
            &view,
        )?;
        let audio_weights = match &b.audio {
            Some(e) => Some(audio::load(
                &mut OnCard {
                    device: &mut device,
                    card,
                },
                e,
                &view,
            )?),
            None => None,
        };
        if device.bytes != count.bytes {
            return Err(format!(
                "the weights take {} device bytes, and the plan counted {}",
                device.bytes, count.bytes
            ));
        }
        let weights_bytes = device.bytes as u64;
        let tower_weights = match &vision {
            Some((v, planned, _)) => {
                let w = vision::load(
                    &mut OnCard {
                        device: &mut device,
                        card,
                    },
                    v,
                    &view,
                )?;
                let took = device.bytes - weights_bytes as usize;
                if took != *planned {
                    return Err(format!(
                        "the vision tower takes {took} device bytes, and the plan counted {planned}"
                    ));
                }
                Some(w)
            }
            None => None,
        };
        let max_seq = config.max_seq();
        let bufs = Bufs::new(&mut |bytes| device.zeros(card, bytes), &b, config.max_block);
        let tower = match (&vision, tower_weights) {
            (Some((v, _, _)), Some(w)) => Some(Tower {
                bufs: VisionBufs::new(&mut |bytes| device.zeros(card, bytes), v, config.max_block),
                v: v.clone(),
                w,
                held: HeldRows::default(),
            }),
            _ => None,
        };
        let slots: Vec<SlotMemory> = config
            .seqs()
            .iter()
            .map(|seq| SlotMemory::new(&mut |bytes| device.zeros(card, bytes), &b, *seq, 0))
            .collect();
        let control = spec::control_limit(&b).map(|limit| {
            let bytes = limit.table_bytes() as usize;
            (device.zeros(card, bytes), limit)
        });
        let audio = match (&b.audio, audio_weights) {
            (Some(e), Some(w)) => Some((
                w,
                AudioBufs::new(&mut |bytes| device.zeros(card, bytes), e, MAX_CLIP_POSITIONS),
            )),
            (None, None) => None,
            _ => unreachable!("the audio weights load when the backbone has an audio encoder"),
        };
        let vision_bytes = vision.as_ref().map(|(v, w, _)| VisionBytes {
            weights: *w as u64,
            work: vision::bufs_bytes(v, config.max_block) as u64,
        });
        let mut limits = spec::limits(&b, &config, weights_bytes, vision_bytes);
        limits.check()?;
        let kv = config.slots_bytes(&b);
        let planned = weights_bytes
            + kv
            + bufs_bytes(&b, config.max_block) as u64
            + spec::audio_bufs_bytes(&b) as u64
            + vision_bytes.map_or(0, |v| v.weights + v.work)
            + control.map_or(0, |(_, limit)| limit.table_bytes());
        if device.bytes as u64 != planned {
            return Err(format!(
                "the engine took {} device bytes, and its limits declare {planned}",
                device.bytes
            ));
        }
        // Omni adds the term `speech` after the terms of the backbone (engine-abi "The speech calls of LFM3").
        let speech = match b.model {
            Model::Omni => Some(speech::Speech::open(&dir, card, &mut device, config.slots.len())?),
            Model::Text => None,
        };
        if let Some(speech) = &speech {
            crate::frames::declare(&mut limits, speech.bytes);
            limits.check()?;
        }
        let ring = Ring::new(card, scratch_floats(max_seq));

        let revision = pci_revision(pci)?;
        let mut numbers = BTreeMap::new();
        numbers.insert(
            "kernel.megakernel_lfm3_bmg.bin".to_owned(),
            format!("blake3:{}", blake3::hash(LFM3_KERNEL).to_hex()),
        );
        numbers.insert("ocloc".to_owned(), OCLOC_VERSION.to_owned());
        numbers.insert("igc".to_owned(), IGC_REVISION.to_owned());
        numbers.insert("driver".to_owned(), driver.to_string());
        numbers.insert("max_block".to_owned(), config.max_block.to_string());
        numbers.insert("ring_rows".to_owned(), RING_ROWS.to_string());
        numbers.insert("canary".to_owned(), "not run".to_owned());
        let identity = Identity {
            model: b.model.name().to_owned(),
            checkpoint_digest,
            family: FAMILY.to_owned(),
            tokenizer_hash: family.tokenizer_hash().to_owned(),
            template: family.template().to_owned(),
            numbers,
            plan_table: format!(
                "ocloc {OCLOC_VERSION} igc {IGC_REVISION} bmg-g31; level-zero driver {driver}; devices \
                 {:04x}:{:04x}:{revision:02x}",
                card.vendor_id, card.device_id
            ),
            // D11 rule 3: the block is a bit setting that defines the reference bits.
            bit_settings: vec![BitSetting {
                name: "feed.block".to_owned(),
                value: config.max_block.to_string(),
                class: BitClass::ReferenceDefining,
            }],
            canary_hash: String::new(),
            build: std::collections::BTreeMap::new(),
        };
        let mut held = Held::new(&b, &config.seqs());
        if let Some((_, _, rules)) = vision {
            held.media = rules;
        }
        held.media.latent = Some(latent);
        Ok(Self {
            b,
            config,
            limits,
            identity,
            vocab,
            last_error: None,
            cluster,
            ring,
            device,
            weights,
            bufs,
            tower,
            slots,
            held,
            audio,
            speech,
            control,
        })
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    /// The ids of the family: the width of each logit row.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each task points at the weights, the buffers or the slots of this engine, sized for the config.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }

    /// Zero the conv state of each conv layer of `slot`: the state before position 0.
    fn zero_conv(&self, slot: usize) {
        for l in self.b.layers_of(LayerKind::Conv) {
            self.device
                .zero(self.card(), self.slots[slot].layers[l], self.b.conv_state() * 4);
        }
    }

    /// Run a block from position `pos0` of `slot`, up to the end of `layers` layers, with the logit rows when given.
    /// The caller checked the block. A block from position 0 starts from zero conv states. A ring past the task limit
    /// of the card is refused before the slot changes.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        slot: usize,
        pos0: usize,
        tokens: &[u32],
        layers: usize,
        logit_rows: Option<&[usize]>,
        after_embed: &[MkTaskDescriptor],
        control: &[model::Control],
        tap: bool,
        lens: bool,
    ) -> Result<(), String> {
        let mut tasks = model::pass(
            &self.b,
            &self.weights,
            &self.bufs,
            Run {
                slot: &self.slots[slot],
                pos0,
                n: tokens.len(),
                layers,
                logit_rows,
                after_embed,
                control,
                tap,
                lens,
                layer_loop: None,
            },
        );
        // Omni keeps the final norm of the last position for `frame` (engine-abi "The speech calls of LFM3").
        if let Some(speech) = &self.speech
            && logit_rows.is_some()
            && layers == self.b.layers.len()
        {
            tasks.push(speech.keep_hidden(slot, self.bufs.h, tokens.len() - 1));
        }
        fits(&tasks, control.len())?;
        if pos0 == 0 {
            self.zero_conv(slot);
        }
        let bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.write(self.card(), self.bufs.ids, &bytes);
        self.run(&tasks);
        Ok(())
    }

    /// Run one decode row on the slot of each block in one ring (engine-abi "LFM3", "Batched passes").
    fn forward_batch(&mut self, blocks: &[Block], control: &[model::Control]) -> Result<(), String> {
        let mut tasks = {
            let rows: Vec<(&SlotMemory, usize)> = blocks.iter().map(|b| (&self.slots[b.slot], b.pos0)).collect();
            model::batch(
                &self.b,
                &self.weights,
                &self.bufs,
                Batch {
                    rows: &rows,
                    control,
                    layer_loop: None,
                },
            )
        };
        // Omni keeps the final norm of each row for `frame`, as a pass of one row does.
        if let Some(speech) = &self.speech {
            for (i, block) in blocks.iter().enumerate() {
                tasks.push(speech.keep_hidden(block.slot, self.bufs.h, i));
            }
        }
        fits(&tasks, control.len())?;
        for block in blocks {
            if block.pos0 == 0 {
                self.zero_conv(block.slot);
            }
        }
        let bytes: Vec<u8> = blocks.iter().flat_map(|b| b.tokens[0].to_le_bytes()).collect();
        self.device.write(self.card(), self.bufs.ids, &bytes);
        self.run(&tasks);
        Ok(())
    }

    /// The blocks of a batch: each segment is one decode token with its logit row, on a slot of its own, and all have
    /// one control list. A pass with a tap is one segment. Any break is refused before a slot changes.
    fn check_batch(&self, segments: &[Segment], taps: &[Tap]) -> Result<Vec<Block>, PassError> {
        let refuse = |why: String| refused(format!("nothing ran: {why}"));
        let most = (MAX_SEGMENTS as usize).min(self.config.max_block);
        if segments.len() > most {
            return Err(refuse(format!(
                "the pass has {} segments; the engine takes 1 to {most}",
                segments.len()
            )));
        }
        if !taps.is_empty() {
            return Err(refuse(format!(
                "a batch of {} segments names a tap; a pass with a tap has one segment",
                segments.len()
            )));
        }
        let mut blocks: Vec<Block> = Vec::with_capacity(segments.len());
        for (i, segment) in segments.iter().enumerate() {
            let block = self
                .held
                .check_pass(std::slice::from_ref(segment), self.config.max_block, self.vocab)
                .map_err(|why| refuse(format!("segment {i}: {why}")))?;
            if block.tokens.len() != 1 || !block.spans.is_empty() {
                return Err(refuse(format!(
                    "segment {i} has {} tokens; each segment of a batch is one decode token with no media",
                    block.tokens.len()
                )));
            }
            if block.logit_rows != [0] {
                return Err(refuse(format!(
                    "segment {i} asks for {} logit rows; each segment of a batch gives the logits of its token",
                    block.logit_rows.len()
                )));
            }
            if let Some(j) = blocks.iter().position(|b| b.slot == block.slot) {
                return Err(refuse(format!("segments {j} and {i} are both on slot {}", block.slot)));
            }
            if segment.control != segments[0].control {
                return Err(refuse(format!(
                    "segment {i} has another control list than segment 0; a batch has one list"
                )));
            }
            blocks.push(block);
        }
        Ok(blocks)
    }

    /// A pass of several segments: one decode row on each slot, in one ring.
    fn pass_batch(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let blocks = match self.check_batch(segments, taps) {
            Ok(blocks) => blocks,
            Err(error) => return Err(self.fail(error)),
        };
        let (control, _) = match self.check_control_and_taps(segments, taps) {
            Ok(found) => found,
            Err(error) => return Err(self.fail(error)),
        };
        let start = Instant::now();
        if let Err(why) = self.forward_batch(&blocks, &control) {
            return Err(self.fail(refused(format!("nothing ran: {why}"))));
        }
        let row = self.b.vocab_rows * 4;
        let outputs: Vec<Output> = (0..blocks.len())
            .map(|i| Output {
                name: "logits".to_owned(),
                dtype: Dtype::F32,
                bytes: self
                    .device
                    .read(self.card(), self.bufs.logits + (i * row) as u64, self.vocab * 4),
            })
            .collect();
        for block in &blocks {
            self.held.ran(block);
            if let Some(speech) = &mut self.speech {
                speech.kept(block.slot, self.held.n[block.slot]);
            }
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: start.elapsed().as_millis() as u64,
            outputs,
        })
    }

    /// Clear the hidden row of the last position of `slot`: the slot changed in another way than a pass or a feed.
    fn forget_hidden(&mut self, slot: usize) {
        if let Some(speech) = &mut self.speech {
            speech.forget(slot);
        }
    }

    /// Run the encoder of each media span of `block` on the card, from the rows of `segment`. Each one leaves the rows
    /// of the positions of the block in the media buffer, and the copies that this returns put them over the
    /// embedding rows of their placeholders. The caller checked the block.
    fn encode(&mut self, segment: &Segment, block: &Block) -> Vec<MkTaskDescriptor> {
        if block.spans.is_empty() {
            return Vec::new();
        }
        let nagaya_abi::SegmentInputs::TokensAndRows { rows, .. } = &segment.inputs else {
            unreachable!("check_pass gives spans only for a segment of tokens and rows");
        };
        let mut places = Vec::with_capacity(block.spans.len());
        for used in &block.spans {
            let bytes = &rows[used.rows_at..used.rows_at + used.span.bytes()];
            // Latent rows need no encoder. The host writes the rows of the block into the normed rows, which the pass
            // writes only after the embedding, and a copy puts them over the embedding rows of their positions.
            if let SpanKind::Latent { width } = used.span.kind {
                let row = width * 4;
                let at = (used.block_row * row) as u64;
                let taken = &bytes[used.take.start * row..used.take.end * row];
                self.device.write(self.card(), self.bufs.h + at, taken);
                places.push(tasks::copy(
                    self.bufs.h + at,
                    self.bufs.x + at,
                    (used.take.len() * width) as u32,
                ));
                continue;
            }
            let tasks = match used.span.kind {
                SpanKind::Image { grid } => {
                    // An image of more positions than a block is in several passes. The tower runs for the first one.
                    let held = self
                        .tower
                        .as_mut()
                        .expect("check_pass refuses an image when the model has no vision tower")
                        .held
                        .holds([grid[1], grid[2]], bytes);
                    let tower = self.tower.as_ref().expect("the tower held the image above");
                    let take = (used.take.start, used.take.end);
                    if held {
                        vec![vision::rows(&tower.v, &tower.bufs, take, used.block_row)]
                    } else {
                        self.device.write(self.card(), tower.bufs.pixels, bytes);
                        vision::tower(
                            &tower.v,
                            &tower.w,
                            &tower.bufs,
                            grid[1],
                            grid[2],
                            Until::Rows {
                                take,
                                block_row: used.block_row,
                            },
                        )
                    }
                }
                // The encoder writes every row of the clip into its own rows. A copy then takes the rows of this
                // block to their place in the media rows of the tower, as the tower writes an image's rows. A run of
                // a clip takes its rows from the run's first row of the clip (spec 11.11, "The video of a request").
                // The frames of an assistant audio part (spec 11.11, "Assistant audio"): each row is the sum of the 12
                // codebook rows of its frame, as the speech loop feeds a frame. The host turns the codes of the block's
                // frames into rows of the table, and 12 embeds of the block's frames sum them into the media rows.
                SpanKind::Latent { .. } => unreachable!("latent rows take their place above"),
                SpanKind::Codes => {
                    let speech = self
                        .speech
                        .as_ref()
                        .expect("check_pass refuses speech when the model has no speech path");
                    let tower = self
                        .tower
                        .as_ref()
                        .expect("Omni has a vision tower beside its speech path");
                    let codes = &bytes.as_chunks::<4>().0
                        [used.take.start * media::FRAME_CODES..used.take.end * media::FRAME_CODES];
                    let ids: Vec<u8> = codes
                        .iter()
                        .enumerate()
                        .flat_map(|(i, b)| {
                            let g = (i % media::FRAME_CODES) as u32;
                            (g * media::CODEBOOK_CODES + u32::from_le_bytes(*b)).to_le_bytes()
                        })
                        .collect();
                    // The pixel buffer of the tower holds the ids; no image of this block uses it.
                    self.device.write(self.card(), tower.bufs.pixels, &ids);
                    let row = (self.b.hidden * 4) as u64;
                    (0..media::FRAME_CODES)
                        .map(|g| {
                            tasks::embed(
                                tower.bufs.pixels + (g * 4) as u64,
                                speech.codebook(),
                                tower.bufs.media + used.block_row as u64 * row,
                                used.take.len() as u32,
                                self.b.hidden as u32,
                                media::FRAME_CODES as u32,
                                false,
                                g > 0,
                            )
                        })
                        .collect()
                }
                SpanKind::Audio { frames } | SpanKind::Run { frames, .. } => {
                    let start = match used.span.kind {
                        SpanKind::Run { start, .. } => start,
                        _ => 0,
                    };
                    let (w, bufs) = self
                        .audio
                        .as_ref()
                        .expect("check_pass refuses a clip when the model has no audio encoder");
                    let e = self.b.audio.as_ref().expect("the audio weights load with the encoder");
                    let media = self
                        .tower
                        .as_ref()
                        .expect("Omni has a vision tower beside its audio encoder")
                        .bufs
                        .media;
                    self.device.write(self.card(), bufs.mel, bytes);
                    let mut tasks = audio::encode(
                        e,
                        w,
                        bufs,
                        Clip {
                            mel: bufs.mel,
                            frames,
                            out: bufs.rows,
                        },
                        Stage::Rows,
                    );
                    let row = (self.b.hidden * 4) as u64;
                    tasks.push(tasks::copy(
                        bufs.rows + (start + used.take.start) as u64 * row,
                        media + used.block_row as u64 * row,
                        (used.take.len() * self.b.hidden) as u32,
                    ));
                    tasks
                }
            };
            self.run(&tasks);
            let tower = self
                .tower
                .as_ref()
                .expect("Omni has a vision tower, and its media rows hold each span");
            places.push(vision::place(
                &tower.v,
                &tower.bufs,
                self.bufs.x,
                used.block_row..used.block_row + used.take.len(),
            ));
        }
        places
    }

    /// The stages of the vision tower for one image of `grid_h` by `grid_w` patches: the patch embedding after its
    /// norm `[patches][hidden]`, the tower `[patches][hidden]`, and the rows of the projector `[patches / 4][out]`.
    /// `pixels` holds the patch rows in merge-block order. The probe of the oracle (8.3) uses it.
    pub fn vision_stages(&mut self, pixels: &[f32], grid_h: usize, grid_w: usize) -> Result<[Vec<f32>; 3], String> {
        let Some(tower) = self.tower.as_mut() else {
            return Err("this model has no vision tower".to_owned());
        };
        // The probe writes the output of the projector.
        tower.held.forget();
        let n = grid_h * grid_w;
        if grid_h < 2 || grid_w < 2 || !grid_h.is_multiple_of(2) || !grid_w.is_multiple_of(2) || n > MAX_IMAGE_PATCHES {
            return Err(format!(
                "an image of {grid_h} by {grid_w} patches does not fit the tower: each side is even, and it has at \
                 most {MAX_IMAGE_PATCHES} patches"
            ));
        }
        if pixels.len() != n * PATCH_ROW {
            return Err(format!("{} values are not {n} patch rows of {PATCH_ROW}", pixels.len()));
        }
        let (v, w, bufs) = (tower.v.clone(), tower.w.clone(), tower.bufs);
        self.device.write(self.card(), bufs.pixels, &weights::f32_bytes(pixels));
        let mut out = Vec::with_capacity(3);
        for (until, at, len) in [
            (Until::Patches, bufs.x, n * v.hidden),
            (Until::Tower, bufs.x, n * v.hidden),
            (Until::Projector, bufs.y, n / 4 * v.out),
        ] {
            let tasks = vision::tower(&v, &w, &bufs, grid_h, grid_w, until);
            self.run(&tasks);
            out.push(self.read_f32(at, len));
        }
        Ok(out.try_into().expect("three stages"))
    }

    /// The count of the images that the vision tower ran for. A gate reads it: an image in several passes is one run.
    pub fn tower_runs(&self) -> u64 {
        self.tower.as_ref().map_or(0, |tower| tower.held.runs())
    }

    fn read_f32(&self, at: u64, n: usize) -> Vec<f32> {
        f32_values(&self.device.read(self.card(), at, n * 4))
    }

    /// The residual stream after `layers` layers of `ids` from position 0 of `slot`, or the final norm of all layers
    /// with `norm`: `[n][hidden]`. The probe of the oracle (8.3) uses it. It empties the slot.
    pub fn hidden(&mut self, slot: usize, ids: &[u32], layers: usize, norm: bool) -> Result<Vec<f32>, String> {
        let segment = Segment {
            control: Vec::new(),
            state: State::Slot(slot as u32),
            inputs: nagaya_abi::SegmentInputs::Tokens(ids.to_vec()),
            positions: (0..ids.len() as u64).collect(),
            meta: Vec::new(),
            outputs: Default::default(),
        };
        if layers > self.b.layers.len() || norm && layers != self.b.layers.len() {
            return Err(format!(
                "a probe of {layers} layers with the norm {norm} does not exist"
            ));
        }
        self.held.clear(slot);
        self.held.check_pass(&[segment], self.config.max_block, self.vocab)?;
        self.forward(slot, 0, ids, layers, norm.then_some(&[][..]), &[], &[], false, false)
            .expect("a probe with no control item fits the ring");
        let at = if norm { self.bufs.h } else { self.bufs.x };
        let out = self.read_f32(at, ids.len() * self.b.hidden);
        self.held.clear(slot);
        self.zero_conv(slot);
        self.forget_hidden(slot);
        Ok(out)
    }

    /// The rows of the audio encoder of one clip after `upto`, from its mel rows `[frames][mel]`: `[positions][hidden]`
    /// of the encoder, or `[positions][out]` after the projector. The probe of the audio oracle uses it. It writes the
    /// rows of the projector to the residual buffer of the pass, as a span does, and it keeps the slots.
    pub fn audio_rows(&mut self, mel: &[f32], upto: Stage) -> Result<Vec<f32>, String> {
        let Some((w, bufs)) = &self.audio else {
            return Err(format!("{} has no audio encoder", self.b.model.name()));
        };
        let Some(e) = &self.b.audio else {
            unreachable!("the engine holds audio buffers only for a backbone with an audio encoder")
        };
        if mel.is_empty() || !mel.len().is_multiple_of(e.mel) {
            return Err(format!(
                "a clip has rows of {} mel values, and the probe got {} values",
                e.mel,
                mel.len()
            ));
        }
        let frames = mel.len() / e.mel;
        let positions = audio::positions(frames);
        if positions > bufs.positions || positions > self.config.max_block {
            return Err(format!(
                "a clip of {frames} frames takes {positions} positions, and the probe takes at most {}",
                bufs.positions.min(self.config.max_block)
            ));
        }
        let clip = Clip {
            mel: bufs.mel,
            frames,
            out: self.bufs.x,
        };
        let tasks = audio::encode(e, w, bufs, clip, upto);
        let (at, width) = match upto {
            Stage::Downsample => (bufs.x, e.hidden),
            Stage::Encoder => (bufs.h, e.hidden),
            Stage::Rows => (clip.out, e.out),
        };
        let bytes: Vec<u8> = mel.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.write(self.card(), bufs.mel, &bytes);
        self.run(&tasks);
        Ok(self.read_f32(at, positions * width))
    }

    fn fail(&mut self, error: PassError) -> PassError {
        self.last_error = Some(error.message.clone());
        error
    }

    fn refuse(&mut self, message: String) -> WriteResult {
        self.last_error = Some(message);
        WriteResult::Refused
    }

    fn refuse_read(&mut self, message: String) -> PartError {
        self.last_error = Some(message);
        PartError::Refused
    }

    /// Read the rows `rows` of a row part of `slot` into `out`.
    fn read_rows(&self, slot: usize, part: Part, rows: std::ops::Range<u64>, out: &mut [u8]) {
        let row = self.held.row_bytes() as usize;
        let mut at = 0usize;
        let value = matches!(part, Part::V(_));
        let runs = self.slots[slot].runs(&self.b, part.layer(), value, rows.start as usize..rows.end as usize);
        for (src, n) in runs {
            self.device.read_into(self.card(), src, &mut out[at..at + n * row]);
            at += n * row;
        }
    }

    /// The control items of the one segment of a pass, as tasks of the ring, and the taps that the pass names. A list
    /// for an engine with no table, an item outside the table, or an unknown tap is refused.
    fn check_control_and_taps(
        &self,
        segments: &[Segment],
        taps: &[Tap],
    ) -> Result<(Vec<model::Control>, Taps), PassError> {
        let mut named = Taps::default();
        for t in taps {
            match t.name.as_str() {
                RESIDUAL_TAP => named.residual = true,
                LENS_TAP => named.lens = true,
                other => {
                    return Err(refused(format!(
                        "nothing ran: the engine has no tap `{other}`; its taps are `{RESIDUAL_TAP}` and `{LENS_TAP}`"
                    )));
                }
            }
        }
        let tap = named;
        let items = segments.first().map_or(&[][..], |segment| &segment.control[..]);
        if items.is_empty() {
            return Ok((Vec::new(), tap));
        }
        let Some((table, limit)) = self.control else {
            return Err(refused(format!(
                "nothing ran: {} does not declare `control`, and the segment names {} control items",
                self.b.model.name(),
                items.len()
            )));
        };
        limit
            .check(items)
            .map_err(|message| refused(format!("nothing ran: {message}")))?;
        let control = items
            .iter()
            .map(|item| model::Control {
                vector: table + item.index as u64 * limit.entry_bytes(),
                op: match item.op {
                    ControlOp::Add => nagaya_mk_bmg::protocol::ControlOp::Add,
                    ControlOp::Ablate => nagaya_mk_bmg::protocol::ControlOp::Ablate,
                    ControlOp::Clamp => nagaya_mk_bmg::protocol::ControlOp::Clamp,
                },
                strength: item.strength,
                layers: item.layers.clone(),
            })
            .collect();
        Ok((control, tap))
    }

    fn check_block(&self, segments: &[Segment]) -> Result<Block, PassError> {
        self.held
            .check_pass(segments, self.config.max_block, self.vocab)
            .map_err(|message| refused(format!("nothing ran: {message}")))
    }
}

impl Engine for Lfm3Engine {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        if segments.len() > 1 {
            return self.pass_batch(segments, taps);
        }
        let block = match self.check_block(segments) {
            Ok(block) => block,
            Err(error) => return Err(self.fail(error)),
        };
        let (control, tap) = match self.check_control_and_taps(segments, taps) {
            Ok(found) => found,
            Err(error) => return Err(self.fail(error)),
        };
        if tap.lens && block.logit_rows.len() != 1 {
            let error = refused(format!(
                "nothing ran: the tap `{LENS_TAP}` reads one logit row, and the pass asks for {}",
                block.logit_rows.len()
            ));
            return Err(self.fail(error));
        }
        let start = Instant::now();
        let layers = self.b.layers.len();
        let places = self.encode(&segments[0], &block);
        if let Err(why) = self.forward(
            block.slot,
            block.pos0,
            &block.tokens,
            layers,
            Some(&block.logit_rows),
            &places,
            &control,
            tap.residual || tap.lens,
            tap.lens,
        ) {
            return Err(self.fail(refused(format!("nothing ran: {why}"))));
        }
        let row = self.b.vocab_rows * 4;
        let mut outputs: Vec<Output> = (0..block.logit_rows.len())
            .map(|i| Output {
                name: "logits".to_owned(),
                dtype: Dtype::F32,
                bytes: self
                    .device
                    .read(self.card(), self.bufs.logits + (i * row) as u64, self.vocab * 4),
            })
            .collect();
        // The residual tap (spec 13.1): for each logit row, the residual after each layer, in layer order.
        if tap.residual {
            let bytes = block.logit_rows.len() * self.b.layers.len() * self.b.hidden * 4;
            outputs.push(Output {
                name: RESIDUAL_TAP.to_owned(),
                dtype: Dtype::F32,
                bytes: self.device.read(self.card(), self.bufs.taps, bytes),
            });
        }
        // The lens (spec 13.1): the logit row of each layer, as wide as the logit row of the head.
        if tap.lens {
            let mut bytes = Vec::with_capacity(self.b.layers.len() * self.vocab * 4);
            for l in 0..self.b.layers.len() {
                let at = self.bufs.lens + (l * self.b.vocab_rows * 4) as u64;
                bytes.extend(self.device.read(self.card(), at, self.vocab * 4));
            }
            outputs.push(Output {
                name: LENS_TAP.to_owned(),
                dtype: Dtype::F32,
                bytes,
            });
        }
        self.held.ran(&block);
        if let Some(speech) = &mut self.speech {
            speech.kept(block.slot, self.held.n[block.slot]);
        }
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: start.elapsed().as_millis() as u64,
            outputs,
        })
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        Some(self)
    }

    fn frames(&mut self) -> Option<&mut dyn SpeechFrames> {
        if self.speech.is_some() { Some(self) } else { None }
    }

    fn control(&mut self) -> Option<&mut dyn Control> {
        if self.control.is_some() { Some(self) } else { None }
    }
}

impl Control for Lfm3Engine {
    /// Write one vector into its entry of the table. A wrong index, a wrong size or a value that is not finite is
    /// refused, and nothing changes.
    fn control_write(&mut self, index: u32, bytes: &[u8]) -> Result<(), String> {
        let (table, limit) = self.control.expect("the engine gives `control` only with a table");
        let refused = if index >= limit.tables {
            Some(format!(
                "the control table has {} entries, not entry {index}",
                limit.tables
            ))
        } else if bytes.len() as u64 != limit.entry_bytes() {
            Some(format!(
                "a vector is {} layers of {} f32 values, {} bytes, not {} bytes",
                limit.layers,
                limit.width,
                limit.entry_bytes(),
                bytes.len()
            ))
        } else {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .position(|b| !f32::from_le_bytes(*b).is_finite())
                .map(|at| format!("value {at} of the vector is not finite; the op needs finite values"))
        };
        if let Some(message) = refused {
            self.last_error = Some(message.clone());
            return Err(message);
        }
        self.device
            .write(self.card(), table + u64::from(index) * limit.entry_bytes(), bytes);
        Ok(())
    }
}

impl Parts for Lfm3Engine {
    /// Read a conv state whole, or the rows of a row part from its base: as many as `out` holds.
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let (slot, which, from) = match self.held.check_part_read(state, part, out.len()) {
            Ok(found) => found,
            Err(message) => return Err(self.refuse_read(message)),
        };
        match which {
            Part::Conv(l) => self.device.read_into(self.card(), self.slots[slot].layers[l], out),
            _ => {
                let rows = out.len() as u64 / self.held.row_bytes();
                self.read_rows(slot, which, from..from + rows, out);
            }
        }
        Ok(())
    }

    /// Write a conv state whole. The slot holds no token until `set_n`.
    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        let (slot, which) = match self.held.check_part_write(state, part, bytes.len()) {
            Ok(found) => found,
            Err(message) => return self.refuse(message),
        };
        self.device
            .write(self.card(), self.slots[slot].layers[which.layer()], bytes);
        self.held.wrote(slot, part, None);
        self.forget_hidden(slot);
        WriteResult::Ok
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let (slot, which) = match self.held.check_read(state, part, from, n, out.len()) {
            Ok(found) => found,
            Err(message) => return Err(self.refuse_read(message)),
        };
        self.read_rows(slot, which, from..from + n, out);
        Ok(())
    }

    /// Write the rows of positions `from` to `from + n`. A ring keeps the last of them. The slot holds no token until
    /// `set_n`, so a pass cannot read rows of two states.
    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        let (slot, which, kept) = match self.held.check_cells_write(state, part, from, n, bytes.len()) {
            Ok(found) => found,
            Err(message) => return self.refuse(message),
        };
        let row = self.held.row_bytes() as usize;
        let mut at = ((kept.start - from) as usize) * row;
        let value = matches!(which, Part::V(_));
        let runs = self.slots[slot].runs(&self.b, which.layer(), value, kept.start as usize..kept.end as usize);
        for (dst, rows) in runs {
            self.device.write(self.card(), dst, &bytes[at..at + rows * row]);
            at += rows * row;
        }
        self.held.wrote(slot, part, Some(from..from + n));
        self.forget_hidden(slot);
        WriteResult::Ok
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        match self.held.check_set_n(state, n) {
            Ok(slot) => {
                if n == 0 {
                    self.zero_conv(slot);
                }
                self.held.set_n(slot, n);
                self.forget_hidden(slot);
                WriteResult::Ok
            }
            Err(message) => self.refuse(message),
        }
    }
}

impl Slots for Lfm3Engine {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.held.n.get(slot as usize).copied().unwrap_or(0)
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        if slot as usize >= self.held.n.len() {
            return self.refuse(format!("the engine has no slot {slot}"));
        }
        self.held.clear(slot as usize);
        self.zero_conv(slot as usize);
        self.forget_hidden(slot as usize);
        // A clear also ends the stream of the codec of the slot, and its last samples go.
        if let Some(speech) = &mut self.speech {
            speech.clear_codec(slot as usize);
        }
        WriteResult::Ok
    }
}

impl Coverage for Lfm3Engine {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        self.held.coverage(state, part, bytes)
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        self.held.fits(slot as usize, coverage)
    }
}

/// A ring holds fewer than `MAX_TASKS_PER_CARD` tasks. A long control list can pass it, because each item adds a task
/// after each layer; such a pass is refused before it changes a slot.
fn fits(tasks: &[MkTaskDescriptor], items: usize) -> Result<(), String> {
    if tasks.len() < nagaya_mk_bmg::protocol::MAX_TASKS_PER_CARD {
        return Ok(());
    }
    Err(format!(
        "the pass needs {} tasks with {items} control items, and a ring holds fewer than {}; send fewer control items",
        tasks.len(),
        nagaya_mk_bmg::protocol::MAX_TASKS_PER_CARD
    ))
}
