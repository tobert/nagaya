//! The `Engine` trait on the Gemma 4 copy, with two stages (engine-abi D1, "How mkviolet maps").
//!
//! [`Loaded`] holds the cards and the weights. [`Gemma4`] is one engine on them: its slots, its work buffers and its
//! two stages. Several engines can share one `Loaded`, as the conformance gates need. Stage 0 runs layers 0-14 on
//! the first card of the group, and stage 1 runs layers 15-29 and the head on the second card. Each stage is a
//! [`StageHandle`] that moves to the thread of its card.

use std::collections::HashMap;
use std::ops::Range;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nagaya_abi::{
    Arena, ArenaFd, BitClass, BitSetting, Coverage, CoverageSpan, Dtype, Engine, Identity, Limits, Output, PartError,
    PartId, Parts, PassError, PassErrorKind, PassOut, PassStatus, Residual, Segment, SlotId, Slots, StageHandle,
    StageOut, Stages, State, Tap, WriteResult,
};

use crate::gemma4::{
    CARD_LAYERS, CardShard, CardWeights, EMBED_TENSOR, HIDDEN, NUM_LAYERS, VOCAB, check_config, kv_can_rewind,
    kv_high_after_load, kv_keep_from, kv_ring, run_first_stage, run_last_stage, verify_stored_head, verify_stored_slab,
};
use crate::level_zero::{B70Card, L0Cluster};
use crate::sharding::Checkpoint;
use crate::spec::{
    self, Block, Config, HEAD_PART, Lineage, SlotView, layer_of, layer_part, row_floats, stage_of_layer,
};
use crate::{MEGAKERNEL_BIN, MEGAKERNEL_ENTRY};

/// The cards of one group, the checkpoint and the weights on the cards. Engines on these cards share it.
pub struct Loaded {
    cluster: L0Cluster,
    ckpt: Checkpoint,
    /// The index in `cluster.cards` of the card of each stage.
    cards: [usize; 2],
    weights: [CardWeights; 2],
    /// The DMA-BUF fd of each slab of each stage, by part name (spec 10.2). `Loaded` owns them.
    slabs: [Vec<(String, OwnedFd)>; 2],
    /// True when a cold load exported the slabs. The driver closes the fd of an exported slab when it frees the slab.
    exported: bool,
    checkpoint_digest: String,
    tokenizer_hash: String,
    lineage: Lineage,
}

impl Drop for Loaded {
    /// Free the weight slabs on each card (engine-abi "`close` in Rust"). The cluster drops after this call, and a card
    /// destroys its context only when the context holds no slab.
    fn drop(&mut self) {
        if self.exported {
            // `zeMemFree` closes the fd that the export gave, so these fds must not close a second time.
            for slabs in &mut self.slabs {
                for (_, fd) in slabs.drain(..) {
                    let _ = fd.into_raw_fd();
                }
            }
        }
        for (stage, weights) in self.weights.iter().enumerate() {
            let card = &self.cluster.cards[self.cards[stage]];
            for layer in &weights.weights {
                card.free(layer.slab as *mut std::ffi::c_void);
            }
            if let Some((base, _)) = weights.head {
                card.free(base as *mut std::ffi::c_void);
            }
        }
    }
}

impl Loaded {
    /// Open the cards at the PCI addresses `pci` (stage 0, then stage 1), and upload the checkpoint in `model_dir`.
    ///
    /// The engine finds each card by its PCI address, because the order of `zeDeviceGet` can differ from the order
    /// of the profile (engine-abi "The factory gets its devices").
    pub fn load(model_dir: &Path, pci: &[String]) -> Result<Arc<Self>, String> {
        Self::load_with(model_dir, pci, None).map_err(|error| match error {
            LoadError::Mismatch(why) | LoadError::Other(why) => why,
        })
    }

    /// Open the cards as `load` does, and import the slabs of a stored arena (spec 10.2). `stored` holds the slab
    /// fds of each stage by part name, and `Loaded` owns them after a load.
    ///
    /// The spot check of each slab runs first: the first and last 64 bytes of each tensor (`verify_stored_slab`). A
    /// missing slab, an extra slab or a slab that differs from the checkpoint is a mismatch, and nothing stays
    /// imported.
    pub fn attach(
        model_dir: &Path,
        pci: &[String],
        stored: [Vec<(String, OwnedFd)>; 2],
    ) -> Result<Arc<Self>, LoadError> {
        Self::load_with(model_dir, pci, Some(stored))
    }

    fn load_with(
        model_dir: &Path,
        pci: &[String],
        stored: Option<[Vec<(String, OwnedFd)>; 2]>,
    ) -> Result<Arc<Self>, LoadError> {
        let other = LoadError::Other;
        if pci.len() != 2 {
            return Err(other(format!(
                "the Gemma 4 engine runs on two cards, but its group has {}",
                pci.len()
            )));
        }
        check_config(model_dir).map_err(other)?;
        let ckpt = Checkpoint::open_dir(model_dir).map_err(other)?;
        let tokenizer = model_dir.join("tokenizer.json");
        let bytes = std::fs::read(&tokenizer).map_err(|e| {
            other(format!(
                "cannot read {}: {e}; the checkpoint holds its tokenizer",
                tokenizer.display()
            ))
        })?;
        // The family library hashes the file in the same way (spec 12.3).
        let tokenizer_hash = nagaya_text_gemma4::tokenizer::sha256_of(&bytes);
        let cluster = L0Cluster::init(MEGAKERNEL_BIN, MEGAKERNEL_ENTRY);
        let found = cluster.cards_at(pci).map_err(other)?;
        let cards = [found[0], found[1]];
        if let Some(stored) = &stored {
            for stage in 0..2 {
                check_stored(&cluster.cards[cards[stage]], &ckpt, stage, &stored[stage])
                    .map_err(LoadError::Mismatch)?;
            }
        }
        let checkpoint_digest = checkpoint_digest(model_dir, &ckpt).map_err(other)?;
        let raw = |stage: usize| -> HashMap<String, i32> {
            stored.as_ref().map_or_else(HashMap::new, |stored| {
                stored[stage]
                    .iter()
                    .map(|(part, fd)| (part.clone(), fd.as_raw_fd()))
                    .collect()
            })
        };
        let w0 = CardWeights::upload(&cluster.cards[cards[0]], &ckpt, CARD_LAYERS[0].clone(), false, &raw(0))
            .map_err(other)?;
        let w1 = CardWeights::upload(&cluster.cards[cards[1]], &ckpt, CARD_LAYERS[1].clone(), true, &raw(1))
            .map_err(other)?;
        let exported = stored.is_none();
        let slabs = match stored {
            Some(stored) => stored,
            // A cold load owns the fds that the export of each new slab gave.
            None => [slab_fds(&w0), slab_fds(&w1)],
        };
        let mut devices = Vec::new();
        for &card in &cards {
            let card = &cluster.cards[card];
            devices.push((
                card.vendor_id,
                card.device_id,
                pci_revision(&card.pci_address).map_err(other)?,
            ));
        }
        let lineage = Lineage {
            kernel_sha256: nagaya_text_gemma4::tokenizer::sha256_of(MEGAKERNEL_BIN),
            ocloc: nagaya_mk_bmg::OCLOC_VERSION.to_owned(),
            igc: nagaya_mk_bmg::IGC_REVISION.to_owned(),
            driver: cluster.driver_version(),
            devices,
        };
        Ok(Arc::new(Self {
            cluster,
            ckpt,
            cards,
            weights: [w0, w1],
            slabs,
            exported,
            checkpoint_digest,
            tokenizer_hash,
            lineage,
        }))
    }

    /// A new fd for each slab of the arena, with the stage as its device (spec 10.2).
    pub fn arena_export(&self) -> Result<Vec<ArenaFd>, String> {
        let mut out = Vec::new();
        for (stage, slabs) in self.slabs.iter().enumerate() {
            for (part, fd) in slabs {
                out.push(ArenaFd {
                    device: stage,
                    part: part.clone(),
                    fd: fd
                        .try_clone()
                        .map_err(|error| format!("cannot duplicate the slab fd of `{part}`: {error}"))?,
                });
            }
        }
        Ok(out)
    }

    fn card(&self, stage: usize) -> &B70Card {
        &self.cluster.cards[self.cards[stage]]
    }

    fn embed_bytes(&self) -> &[u8] {
        self.ckpt
            .bf16(EMBED_TENSOR, &[VOCAB, HIDDEN])
            .expect("the checkpoint was checked at load")
    }

    /// The cluster of the cards, for the profiles and the tests.
    pub fn cluster(&self) -> &L0Cluster {
        &self.cluster
    }

    /// The device bytes of the weights on the card of each stage.
    pub fn weights_bytes(&self) -> [u64; 2] {
        [self.weights[0].vram_bytes as u64, self.weights[1].vram_bytes as u64]
    }

    /// The identity of an engine on these weights with a feed block of `max_block` tokens (spec 12.1).
    pub fn identity(&self, max_block: usize) -> Identity {
        Identity {
            model: spec::MODEL.to_owned(),
            checkpoint_digest: self.checkpoint_digest.clone(),
            family: spec::FAMILY.to_owned(),
            tokenizer_hash: self.tokenizer_hash.clone(),
            template: nagaya_text_gemma4::CHAT_TEMPLATE.to_owned(),
            numbers: self.lineage.numbers(max_block),
            plan_table: self.lineage.plan_table(),
            // D11 rule 3: the block is a bit setting that defines the reference bits.
            bit_settings: vec![BitSetting {
                name: "feed.block".to_owned(),
                value: max_block.to_string(),
                class: BitClass::ReferenceDefining,
            }],
            canary_hash: String::new(),
        }
    }
}

/// Why `Loaded::attach` did not load.
#[derive(Debug)]
pub enum LoadError {
    /// The stored arena does not match the checkpoint. Nothing is imported.
    Mismatch(String),
    /// Any other failure.
    Other(String),
}

/// The fds of the slabs of a cold load, by part name. The export of a new slab gave each fd, and nothing else owns it.
fn slab_fds(weights: &CardWeights) -> Vec<(String, OwnedFd)> {
    let mut out: Vec<(String, OwnedFd)> = weights
        .layers
        .clone()
        .zip(&weights.weights)
        // SAFETY: `alloc_exportable_vram_dmabuf` gave the fd, and no other value owns it.
        .map(|(l, w)| (layer_part(l), unsafe { OwnedFd::from_raw_fd(w.dmabuf_fd) }))
        .collect();
    if let Some(fd) = weights.head_fd {
        // SAFETY: as above, for the head slab.
        out.push((HEAD_PART.to_owned(), unsafe { OwnedFd::from_raw_fd(fd) }));
    }
    out
}

/// Check the stored slabs of one stage: the part names of the stage, each once, and the spot check of each slab.
fn check_stored(card: &B70Card, ckpt: &Checkpoint, stage: usize, stored: &[(String, OwnedFd)]) -> Result<(), String> {
    let mut expected: Vec<String> = CARD_LAYERS[stage].clone().map(layer_part).collect();
    if stage == 1 {
        expected.push(HEAD_PART.to_owned());
    }
    let mut names: Vec<String> = stored.iter().map(|(part, _)| part.clone()).collect();
    names.sort();
    expected.sort();
    if names != expected {
        return Err(format!(
            "stage {stage} has the stored slabs [{}], and it needs [{}]",
            names.join(", "),
            expected.join(", ")
        ));
    }
    for (part, fd) in stored {
        if part == HEAD_PART {
            verify_stored_head(card, ckpt, fd.as_raw_fd())?;
        } else {
            let l = CARD_LAYERS[stage]
                .clone()
                .find(|l| layer_part(*l) == *part)
                .expect("the names were checked");
            verify_stored_slab(card, ckpt, l, fd.as_raw_fd())?;
        }
    }
    Ok(())
}

/// The PCI revision of the device at `address`, from sysfs.
fn pci_revision(address: &str) -> Result<u32, String> {
    let path = format!("/sys/bus/pci/devices/{address}/revision");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let text = text.trim();
    u32::from_str_radix(text.trim_start_matches("0x"), 16).map_err(|e| format!("{path} holds `{text}`: {e}"))
}

/// The logical digest of the checkpoint (spec 12.2): BLAKE3 over the canonical config and each tensor's payload.
///
/// The canonical config is `config.json` with sorted keys and no whitespace. Then each tensor follows in bytewise
/// order of its name: the name, the dtype, the shape and the BLAKE3 of its bytes. A new packing on the card does not
/// change it, and a changed weight does.
pub fn checkpoint_digest(model_dir: &Path, ckpt: &Checkpoint) -> Result<String, String> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut names: Vec<&String> = ckpt.names().collect();
    names.sort();
    // Each tensor's payload hash is independent, so the threads share the work.
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
                        let (bytes, _, _) = ckpt.raw(name).expect("the name came from the checkpoint");
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
    let mut hasher = blake3::Hasher::new_derive_key("nagaya gemma4 checkpoint v1");
    let canonical = canonical_json(&config);
    hasher.update(&(canonical.len() as u64).to_le_bytes());
    hasher.update(canonical.as_bytes());
    for (name, hash) in names.iter().zip(&hashes) {
        let (_, dtype, shape) = ckpt.raw(name)?;
        let dtype = format!("{dtype:?}");
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(dtype.len() as u64).to_le_bytes());
        hasher.update(dtype.as_bytes());
        hasher.update(&(shape.len() as u64).to_le_bytes());
        for dim in shape {
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

fn refused(message: String) -> PassError {
    PassError {
        kind: PassErrorKind::Refused,
        message,
    }
}

/// One stage of an engine: the shard of its card, and the token count of each slot in this stage.
pub struct StageShard {
    loaded: Arc<Loaded>,
    stage: usize,
    shard: CardShard,
    max_block: usize,
    /// The token count of each slot.
    n: Vec<u64>,
    /// For each slot, the rows that each part of this stage got from `cells_write` since the last `set_n`.
    writes: Vec<HashMap<PartId, Range<u64>>>,
    last_error: Option<String>,
}

impl Drop for StageShard {
    fn drop(&mut self) {
        // No kernel runs: a stage runs its ring to the end inside each call.
        unsafe { self.shard.free(self.loaded.card(self.stage)) };
    }
}

impl StageShard {
    fn new(loaded: &Arc<Loaded>, stage: usize, config: &Config) -> Result<Self, String> {
        let shard = CardShard::with_weights(
            loaded.card(stage),
            &loaded.weights[stage],
            &config.slots,
            config.max_block,
        )?;
        Ok(Self {
            loaded: Arc::clone(loaded),
            stage,
            shard,
            max_block: config.max_block,
            n: vec![0; config.slots.len()],
            writes: vec![HashMap::new(); config.slots.len()],
            last_error: None,
        })
    }

    fn card(&self) -> &B70Card {
        self.loaded.card(self.stage)
    }

    fn highs(&self) -> Vec<usize> {
        (0..self.n.len()).map(|slot| self.shard.kv_high(slot)).collect()
    }

    fn check(&self, segments: &[Segment]) -> Result<Block, PassError> {
        let highs = self.highs();
        let view = SlotView {
            seqs: &self.shard.slot_seq,
            n: &self.n,
            high: &highs,
        };
        spec::check_pass(segments, &view, self.max_block).map_err(refused)
    }

    /// Check the residual of a block for this stage: room for its rows before the last stage, and its rows after the
    /// first stage.
    fn check_residual(&self, block: &Block, residual: &Residual) -> Result<(), PassError> {
        let rows = block.tokens.len();
        if residual.row_bytes() != HIDDEN * 4 {
            return Err(refused(format!(
                "a residual row has {} bytes; the engine writes {}",
                residual.row_bytes(),
                HIDDEN * 4
            )));
        }
        if self.stage == 0 && rows > residual.capacity() {
            return Err(refused(format!(
                "the block has {rows} tokens; the residual holds {} rows",
                residual.capacity()
            )));
        }
        if self.stage == 1 && residual.rows() != rows {
            return Err(refused(format!(
                "the residual holds {} rows, but the block has {rows} tokens",
                residual.rows()
            )));
        }
        Ok(())
    }

    /// Run this stage of a checked block. A stage changes the slot and cannot refuse after this point.
    fn run(&mut self, block: &Block, residual: &mut Residual) -> StageOut {
        self.shard.set_slot(block.slot);
        let meta = block.meta.as_deref();
        let n = block.tokens.len();
        let out = if self.stage == 0 {
            let out = residual.write(n).expect("the check fitted the rows to the residual");
            let us = unsafe {
                run_first_stage(
                    &self.shard,
                    self.card(),
                    self.loaded.embed_bytes(),
                    &block.tokens,
                    block.pos0,
                    meta,
                    out,
                )
            };
            StageOut::Residual {
                status: PassStatus::Ok,
                ms: (us / 1000) as u64,
            }
        } else {
            let us = unsafe {
                run_last_stage(
                    &self.shard,
                    self.card(),
                    residual.bytes(),
                    block.pos0,
                    meta,
                    &block.logit_rows,
                )
            };
            let outputs = (0..block.logit_rows.len())
                .map(|i| Output {
                    name: "logits".to_owned(),
                    dtype: Dtype::F32,
                    bytes: crate::gemma4::f32_bytes(self.shard.logits_row(i)).to_vec(),
                })
                .collect();
            StageOut::Last(PassOut {
                status: PassStatus::Ok,
                ms: (us / 1000) as u64,
                outputs,
            })
        };
        self.n[block.slot] = block.n_after();
        self.writes[block.slot].clear();
        out
    }

    /// The layer of a part of this stage, and true for a V plane.
    fn own(&self, part: PartId) -> Option<(usize, bool)> {
        layer_of(part).filter(|(l, _)| stage_of_layer(*l) == self.stage)
    }

    fn slot(&self, state: State) -> Option<usize> {
        let State::Slot(slot) = state;
        ((slot as usize) < self.n.len()).then_some(slot as usize)
    }

    fn refuse(&mut self, message: String) -> WriteResult {
        self.last_error = Some(message);
        WriteResult::Refused
    }

    fn refuse_read(&mut self, message: String) -> PartError {
        self.last_error = Some(message);
        PartError::Refused
    }

    /// The first position that part `part` holds in `slot`.
    fn base(&self, slot: usize, l: usize) -> u64 {
        spec::base(self.shard.slot_seq[slot], l, self.n[slot], self.shard.kv_high(slot))
    }

    /// Check a `set_n`, and give the ring high that it sets. The message says why the count does not fit.
    fn check_set_n(&self, slot: usize, n: u64) -> Result<usize, String> {
        let seq = self.shard.slot_seq[slot];
        if n > seq as u64 {
            return Err(format!("slot {slot} holds {seq} positions, not {n}"));
        }
        let writes = &self.writes[slot];
        if writes.is_empty() {
            // A lower count goes back, as a pass from that position does.
            let high = self.shard.kv_high(slot);
            if n > self.n[slot] {
                return Err(format!(
                    "slot {slot} holds {} tokens and got no rows; it cannot hold {n}",
                    self.n[slot]
                ));
            }
            if n < self.n[slot] && !kv_can_rewind(seq, high, n as usize) {
                return Err(format!("slot {slot} cannot go back to {n}: its rings lost the window"));
            }
            return Ok(high);
        }
        // A restore: every part of this stage got rows that end at n and fit the slot.
        let mut need = 0usize;
        for l in CARD_LAYERS[self.stage].clone() {
            for value in [false, true] {
                let part = spec::part_of(l, value);
                let Some(rows) = writes.get(&part) else {
                    return Err(format!("part {part} got no rows since the last set_n"));
                };
                if rows.end != n {
                    return Err(format!("part {part} got rows up to {}, not up to {n}", rows.end));
                }
                let span = CoverageSpan {
                    part,
                    start: rows.start,
                    end: rows.end,
                };
                spec::fits(seq, l, &span)?;
                if kv_ring(seq, l) > 0 {
                    need = need.max(kv_keep_from(seq, l, n as usize).max(rows.start as usize));
                }
            }
        }
        let ring = (0..NUM_LAYERS).map(|l| kv_ring(seq, l)).max().unwrap_or(0);
        Ok(if ring > 0 {
            kv_high_after_load(need, n as usize, ring)
        } else {
            n as usize
        })
    }

    fn apply_set_n(&mut self, slot: usize, n: u64, high: usize) {
        self.n[slot] = n;
        self.shard.set_kv_high(slot, high);
        self.writes[slot].clear();
    }

    fn clear(&mut self, slot: usize) {
        self.n[slot] = 0;
        self.shard.set_kv_high(slot, 0);
        self.writes[slot].clear();
    }
}

impl StageHandle for StageShard {
    fn stage_pass(&mut self, segments: &[Segment], residual: &mut Residual) -> Result<StageOut, PassError> {
        let block = self.check(segments)?;
        self.check_residual(&block, residual)?;
        Ok(self.run(&block, residual))
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
}

impl Parts for StageShard {
    /// Read the rows from the base of the part: as many as `out` holds.
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let (Some(slot), Some((l, _))) = (self.slot(state), self.own(part)) else {
            return Err(self.refuse_read(format!("stage {} has no part {part} or no {state:?}", self.stage)));
        };
        let row = (row_floats(l) * 4) as u64;
        if !(out.len() as u64).is_multiple_of(row) {
            return Err(self.refuse_read(format!("{} bytes are not whole rows of {row} bytes", out.len())));
        }
        let from = self.base(slot, l);
        self.cells_read(state, part, from, out.len() as u64 / row, out)
    }

    /// The engine has no fixed part, so a whole write is refused.
    fn part_write(&mut self, _state: State, part: PartId, _bytes: &[u8]) -> WriteResult {
        self.refuse(format!("part {part} is a row part; write its rows with cells_write"))
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let (Some(slot), Some((l, value))) = (self.slot(state), self.own(part)) else {
            return Err(self.refuse_read(format!("stage {} has no part {part} or no {state:?}", self.stage)));
        };
        let row = (row_floats(l) * 4) as u64;
        let base = self.base(slot, l);
        if out.len() as u64 != n * row || from < base || from + n > self.n[slot] {
            return Err(self.refuse_read(format!(
                "a read of rows {from} to {} of part {part} does not fit: slot {slot} holds rows {base} to {}",
                from + n,
                self.n[slot]
            )));
        }
        self.shard.set_slot(slot);
        self.shard
            .read_rows(self.card(), l, value, from as usize, (from + n) as usize, out);
        Ok(())
    }

    /// Write the rows of positions `from` to `from + n`. A ring keeps the last of them. The slot holds no token until
    /// `set_n`, so a pass cannot read rows of two states.
    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        let (Some(slot), Some((l, value))) = (self.slot(state), self.own(part)) else {
            return self.refuse(format!("stage {} has no part {part} or no {state:?}", self.stage));
        };
        let seq = self.shard.slot_seq[slot];
        let row = (row_floats(l) * 4) as u64;
        if bytes.len() as u64 != n * row || from + n > seq as u64 {
            return self.refuse(format!(
                "rows {from} to {} of part {part} do not fit slot {slot} of {seq} positions",
                from + n
            ));
        }
        if kv_ring(seq, l) == 0 && from != 0 {
            return self.refuse(format!(
                "part {part} in slot {slot} holds every position from 0; the write starts at {from}"
            ));
        }
        let kept = spec::written(seq, l, from..from + n);
        let skip = ((kept.start - from) * row) as usize;
        self.n[slot] = 0;
        self.shard.set_slot(slot);
        self.shard.write_rows(
            self.card(),
            l,
            value,
            kept.start as usize,
            kept.end as usize,
            &bytes[skip..],
        );
        self.writes[slot].insert(part, from..from + n);
        WriteResult::Ok
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let Some(slot) = self.slot(state) else {
            return self.refuse(format!("the engine has no {state:?}"));
        };
        match self.check_set_n(slot, n) {
            Ok(high) => {
                self.apply_set_n(slot, n, high);
                WriteResult::Ok
            }
            Err(message) => self.refuse(message),
        }
    }
}

impl Slots for StageShard {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.n.get(slot as usize).copied().unwrap_or(0)
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        if slot as usize >= self.n.len() {
            return self.refuse(format!("the engine has no slot {slot}"));
        }
        self.clear(slot as usize);
        WriteResult::Ok
    }
}

impl Coverage for StageShard {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        let (Some(slot), Some((l, _))) = (self.slot(state), layer_of(part)) else {
            return CoverageSpan { part, start: 0, end: 0 };
        };
        let start = self.base(slot, l);
        let rows = bytes.len() as u64 / (row_floats(l) * 4) as u64;
        CoverageSpan {
            part,
            start,
            end: start + rows,
        }
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        let seq = *self
            .shard
            .slot_seq
            .get(slot as usize)
            .ok_or_else(|| format!("the engine has no slot {slot}"))?;
        let (l, _) = layer_of(coverage.part).ok_or_else(|| format!("the engine has no part {}", coverage.part))?;
        spec::fits(seq, l, coverage)
    }
}

/// The Gemma 4 engine on two cards, with two stages.
pub struct Gemma4 {
    /// The cards and the weights. The arena export reads the slab fds here.
    loaded: Arc<Loaded>,
    limits: Limits,
    identity: Identity,
    /// The stages, until `take_handles` moves them out.
    stages: Option<Vec<StageShard>>,
    /// The residual of `Engine::pass`, which runs both stages on the calling thread.
    residual: Residual,
    last_error: Option<String>,
}

/// Open an engine with the slots and the block of `config` on loaded weights. Its limits equal `spec::plan`.
pub fn open(loaded: &Arc<Loaded>, config: &Config) -> Result<Gemma4, String> {
    config.check()?;
    let limits = spec::limits(config, loaded.weights_bytes());
    limits.check()?;
    let stages = vec![StageShard::new(loaded, 0, config)?, StageShard::new(loaded, 1, config)?];
    Ok(Gemma4 {
        loaded: Arc::clone(loaded),
        identity: loaded.identity(config.max_block),
        residual: Residual::new(HIDDEN * 4, config.max_block),
        limits,
        stages: Some(stages),
        last_error: None,
    })
}

impl Gemma4 {
    fn stages_mut(&mut self) -> Option<&mut Vec<StageShard>> {
        self.stages.as_mut()
    }

    fn stage_of_part(part: PartId) -> Option<usize> {
        layer_of(part).map(|(l, _)| stage_of_layer(l))
    }

    /// The token count of `slot` in the stage that holds `stage`, for the tests.
    pub fn slot_n(&self, stage: usize, slot: usize) -> Option<u64> {
        self.stages.as_ref().map(|stages| stages[stage].n[slot])
    }
}

impl Engine for Gemma4 {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// Run both stages in order on the calling thread. After `take_handles`, the pass is refused.
    fn pass(&mut self, segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        let Some(stages) = self.stages.as_mut() else {
            return Err(refused(
                "the stages moved to their handles; run the pass through each stage handle".to_owned(),
            ));
        };
        // Both stages check the pass before either one changes a slot, so a refusal changes nothing.
        let block = stages[0].check(segments)?;
        stages[1].check(segments)?;
        stages[0].check_residual(&block, &self.residual)?;
        let first = stages[0].run(&block, &mut self.residual);
        let StageOut::Residual { ms: ms0, .. } = first else {
            unreachable!("the first stage writes the residual");
        };
        match stages[1].run(&block, &mut self.residual) {
            StageOut::Last(mut out) => {
                out.ms += ms0;
                Ok(out)
            }
            StageOut::Residual { .. } => unreachable!("the last stage gives the outputs"),
        }
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone().or_else(|| {
            self.stages
                .as_ref()
                .and_then(|stages| stages.iter().find_map(|stage| stage.last_error.clone()))
        })
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        self.stages.as_ref()?;
        Some(self)
    }

    fn slots(&mut self) -> Option<&mut dyn Slots> {
        self.stages.as_ref()?;
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        self.stages.as_ref()?;
        Some(self)
    }

    fn stages(&mut self) -> Option<&mut dyn Stages> {
        Some(self)
    }

    fn arena(&mut self) -> Option<&mut dyn Arena> {
        Some(self)
    }
}

impl Arena for Gemma4 {
    fn arena_export(&mut self) -> Result<Vec<ArenaFd>, String> {
        self.loaded.arena_export()
    }
}

impl Stages for Gemma4 {
    fn take_handles(&mut self) -> Vec<Box<dyn StageHandle + Send>> {
        let Some(stages) = self.stages.take() else {
            return Vec::new();
        };
        stages
            .into_iter()
            .map(|stage| Box::new(stage) as Box<dyn StageHandle + Send>)
            .collect()
    }
}

/// Before `take_handles`, the engine sends each part call to the stage that holds the part.
impl Parts for Gemma4 {
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let stage = Self::stage_of_part(part).ok_or(PartError::Refused)?;
        self.stages_mut().ok_or(PartError::Refused)?[stage].part_read(state, part, out)
    }

    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        match (Self::stage_of_part(part), self.stages_mut()) {
            (Some(stage), Some(stages)) => stages[stage].part_write(state, part, bytes),
            _ => WriteResult::Refused,
        }
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        let stage = Self::stage_of_part(part).ok_or(PartError::Refused)?;
        self.stages_mut().ok_or(PartError::Refused)?[stage].cells_read(state, part, from, n, out)
    }

    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        match (Self::stage_of_part(part), self.stages_mut()) {
            (Some(stage), Some(stages)) => stages[stage].cells_write(state, part, from, n, bytes),
            _ => WriteResult::Refused,
        }
    }

    /// Both stages check the count before either one takes it, so a refusal changes nothing.
    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        let Some(stages) = self.stages.as_mut() else {
            return WriteResult::Refused;
        };
        let Some(slot) = stages[0].slot(state) else {
            return stages[0].refuse(format!("the engine has no {state:?}"));
        };
        let mut highs = Vec::with_capacity(stages.len());
        for stage in stages.iter_mut() {
            match stage.check_set_n(slot, n) {
                Ok(high) => highs.push(high),
                Err(message) => return stage.refuse(message),
            }
        }
        for (stage, high) in stages.iter_mut().zip(highs) {
            stage.apply_set_n(slot, n, high);
        }
        WriteResult::Ok
    }
}

impl Slots for Gemma4 {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.stages.as_ref().map_or(0, |stages| {
            stages.iter().map(|stage| stage.slot_high(slot)).max().unwrap_or(0)
        })
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        let Some(stages) = self.stages.as_mut() else {
            return WriteResult::Refused;
        };
        if slot as usize >= stages[0].n.len() {
            return stages[0].refuse(format!("the engine has no slot {slot}"));
        }
        for stage in stages.iter_mut() {
            stage.clear(slot as usize);
        }
        WriteResult::Ok
    }
}

impl Coverage for Gemma4 {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        let stages = self.stages.as_ref().expect("coverage before take_handles");
        stages[Self::stage_of_part(part).unwrap_or(0)].coverage(state, part, bytes)
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        let stages = self.stages.as_ref().ok_or("the stages moved to their handles")?;
        stages[0].fits(coverage, slot)
    }
}
