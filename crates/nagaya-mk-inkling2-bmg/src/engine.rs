//! The Inkling 2 engine on its card: the `Engine` trait, the parts, the slots and the coverage over the ring of
//! `model`. Every check of a call is a rule of `spec::Held`, so a refused call changes nothing.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nagaya_abi::{
    BitClass, BitSetting, Coverage, CoverageSpan, Dtype, Engine, Identity, Limits, Output, PartError, PartId, Parts,
    PassError, PassErrorKind, PassOut, PassStatus, Segment, SlotId, Slots, State, Tap, WriteResult,
};
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView};
use nagaya_mk_bmg::{IGC_REVISION, INKLING2_KERNEL, KERNEL_ENTRY, OCLOC_VERSION};

use crate::device::{Device, OnCard};
use crate::host::ops::HIDDEN;
use crate::host::weights::LOGITS;
use crate::model::{self, Bufs, LAUNCH_TASKS, SlotMemory, TaskRing};
use crate::spec::{self, Block, Config, FAMILY, HEAD_ROWS, Held, Model, Part, RING_ROWS, Run};
use crate::weights::{self, Count, Weights};

/// The Inkling 2 engine on one card.
pub struct Inkling2Engine {
    model: Model,
    config: Config,
    limits: Limits,
    identity: Identity,
    last_error: Option<String>,
    cluster: L0Cluster,
    ring: Ring,
    device: Device,
    weights: Weights,
    bufs: Bufs,
    slots: Vec<SlotMemory>,
    held: Held,
    /// The runs of a pass at the default count of cycles.
    runs: Vec<Run>,
    /// The most tasks of one launch. A test sets fewer to cut a pass at each run.
    launch_tasks: usize,
    /// The launches of the last ring.
    last_launches: usize,
}

// The engine holds USM pointers of its card, and one thread at a time calls it.
unsafe impl Send for Inkling2Engine {}

impl Drop for Inkling2Engine {
    /// Free the weights, the slots, the buffers and the ring (engine-abi "`close` in Rust"). The card destroys its
    /// context after this call, when the cluster drops.
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

/// The logical digest of the checkpoint (spec 12.2): BLAKE3 with the key `nagaya inkling2 checkpoint v1` over the
/// canonical `config.json` and each tensor of the file, in bytewise order of its name. A tensor adds its name, its
/// dtype, its shape and the BLAKE3 of its bytes. The tensors of the tower of images are in it too, so step 9 does
/// not change the digest. The two models share every tensor, so `config.json` makes their digests differ.
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
    let mut hasher = blake3::Hasher::new_derive_key("nagaya inkling2 checkpoint v1");
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

/// The PCI vendor id, device id and revision of the device at `address`, from sysfs. The plan table names them, and
/// a description reads them with no card (spec 11.1).
fn pci_ids(address: &str) -> Result<(u32, u32, u32), String> {
    let read = |name: &str| -> Result<u32, String> {
        let path = format!("/sys/bus/pci/devices/{address}/{name}");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
        let text = text.trim();
        u32::from_str_radix(text.trim_start_matches("0x"), 16).map_err(|e| format!("{path} holds `{text}`: {e}"))
    };
    Ok((read("vendor")?, read("device")?, read("revision")?))
}

/// The identity of an engine of `config` on the card at `pci` with this Level Zero driver (spec 12.1). The open and
/// the description both build it here, so the two give the same identity.
fn identity_of(
    model: Model,
    checkpoint_digest: String,
    facts: spec::TextFacts,
    config: &Config,
    pci: &str,
    driver: u32,
) -> Result<Identity, String> {
    let (vendor, device, revision) = pci_ids(pci)?;
    let mut numbers = BTreeMap::new();
    numbers.insert(
        "kernel.megakernel_inkling2_bmg.bin".to_owned(),
        format!("blake3:{}", blake3::hash(INKLING2_KERNEL).to_hex()),
    );
    numbers.insert("ocloc".to_owned(), OCLOC_VERSION.to_owned());
    numbers.insert("igc".to_owned(), IGC_REVISION.to_owned());
    numbers.insert("driver".to_owned(), driver.to_string());
    numbers.insert("max_block".to_owned(), config.max_block.to_string());
    numbers.insert("ring_rows".to_owned(), RING_ROWS.to_string());
    numbers.insert("cycles".to_owned(), spec::DEFAULT_CYCLES.to_string());
    numbers.insert("canary".to_owned(), "not run".to_owned());
    Ok(Identity {
        model: model.name().to_owned(),
        checkpoint_digest,
        family: FAMILY.to_owned(),
        tokenizer_hash: facts.tokenizer_hash,
        template: facts.template,
        numbers,
        plan_table: format!(
            "ocloc {OCLOC_VERSION} igc {IGC_REVISION} bmg-g31; level-zero driver {driver}; devices \
             {vendor:04x}:{device:04x}:{revision:02x}"
        ),
        // D11 rule 3: the block is a bit setting that defines the reference bits.
        bit_settings: vec![BitSetting {
            name: "feed.block".to_owned(),
            value: config.max_block.to_string(),
            class: BitClass::ReferenceDefining,
        }],
        canary_hash: String::new(),
        build: BTreeMap::new(),
    })
}

/// The identity and the limits that an open of `config` on the card at `pci` gives, with no load (spec 11.1). It
/// reads the model folder, every tensor for the digest, sysfs and the version of the Level Zero driver. It opens no
/// card.
pub fn describe(config: &Config, pci: &str) -> Result<(Identity, Limits), String> {
    config.check()?;
    let dir = &config.model_dir;
    let model = Model::read(dir)?;
    let config_text = String::from_utf8(read_file(dir, "config.json")?)
        .map_err(|e| format!("config.json of {} is not UTF-8: {e}", dir.display()))?;
    let facts = spec::text_facts(dir)?;
    let map = MappedSafeTensors::open(&dir.join(spec::WEIGHTS_FILE))?;
    let view = map.view()?;
    let limits = spec::limits(config, spec::weights_bytes_of(&view.tensors)?);
    let digest = checkpoint_digest(&config_text, &view)?;
    let driver = nagaya_mk_bmg::level_zero::first_driver_version();
    Ok((identity_of(model, digest, facts, config, pci, driver)?, limits))
}

/// The bytes of a file of the model folder.
fn read_file(dir: &std::path::Path, name: &str) -> Result<Vec<u8>, String> {
    let path = dir.join(name);
    std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

impl Inkling2Engine {
    /// Open the card at the PCI address `pci`, and upload the text model of the model folder of `config`.
    ///
    /// The open checks the config, `config.json` and every tensor before it opens the card. Its device bytes must
    /// equal the plan.
    pub fn open(config: Config, pci: &str) -> Result<Self, String> {
        config.check()?;
        let dir = config.model_dir.clone();
        let model = Model::read(&dir)?;
        let config_text = String::from_utf8(read_file(&dir, "config.json")?)
            .map_err(|e| format!("config.json of {} is not UTF-8: {e}", dir.display()))?;
        // The family library gives the tokenizer hash and the template, so the registry's check at the start compares
        // like with like (spec 11.3).
        let facts = spec::text_facts(&dir)?;
        let map = MappedSafeTensors::open(&dir.join(spec::WEIGHTS_FILE))?;
        let view = map.view()?;
        let planned_weights = spec::weights_bytes_of(&view.tensors)?;
        let mut count = Count::default();
        weights::load(&mut count, &view)?;
        if count.bytes as u64 != planned_weights {
            return Err(format!(
                "the load counts {} device bytes of weights, and the plan counts {planned_weights}",
                count.bytes
            ));
        }
        let checkpoint_digest = checkpoint_digest(&config_text, &view)?;

        // Only the engine's card gets a context, so the open does not stall the work of another card.
        let cluster = L0Cluster::init_at(INKLING2_KERNEL, KERNEL_ENTRY, &[pci.to_owned()])?;
        let card = &cluster.cards[0];
        let driver = cluster.driver_version();
        let mut device = Device::new(card);
        let weights = weights::load(
            &mut OnCard {
                device: &mut device,
                card,
            },
            &view,
        )?;
        if device.bytes as u64 != planned_weights {
            return Err(format!(
                "the weights take {} device bytes, and the plan counted {planned_weights}",
                device.bytes
            ));
        }
        let bufs = Bufs::new(&mut |bytes| device.zeros(card, bytes), config.max_block);
        let ids: Vec<u8> = Bufs::sink_ids_of(config.max_block)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        device.write(card, bufs.sink_ids, &ids);
        let slots: Vec<SlotMemory> = config
            .seqs()
            .iter()
            .map(|seq| SlotMemory::new(&mut |bytes| device.zeros(card, bytes), *seq, config.extra_runs))
            .collect();
        let limits = spec::limits(&config, planned_weights);
        limits.check()?;
        let planned = planned_weights + config.slots_bytes() + spec::bufs_bytes(config.max_block);
        if device.bytes as u64 != planned {
            return Err(format!(
                "the engine took {} device bytes, and its limits declare {planned}",
                device.bytes
            ));
        }
        let ring = Ring::new(card, spec::scratch_floats(config.max_seq()));

        let identity = identity_of(model, checkpoint_digest, facts, &config, pci, driver)?;
        let held = Held::new(&config.seqs(), config.extra_runs);
        Ok(Self {
            model,
            config,
            limits,
            identity,
            last_error: None,
            cluster,
            ring,
            device,
            weights,
            bufs,
            slots,
            held,
            runs: spec::runs(spec::DEFAULT_CYCLES),
            launch_tasks: LAUNCH_TASKS,
            last_launches: 0,
        })
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    /// The model of the folder.
    pub fn model(&self) -> Model {
        self.model
    }

    /// The most tasks of one launch. A test cuts a pass at each run with a small count; a cut changes no bit.
    pub fn set_launch_tasks(&mut self, tasks: usize) {
        assert!(
            (1..=LAUNCH_TASKS).contains(&tasks),
            "a launch holds 1 to {LAUNCH_TASKS} tasks, not {tasks}"
        );
        self.launch_tasks = tasks;
    }

    /// The count of launches of the last pass.
    pub fn launches(&self) -> usize {
        self.last_launches
    }

    fn run(&mut self, launches: &[Vec<MkTaskDescriptor>]) {
        for tasks in launches {
            self.ring.set(tasks);
            // SAFETY: each task points at the weights, the buffers or the slots of this engine, sized for the config.
            unsafe { self.ring.run(&self.cluster.cards[0]) };
        }
    }

    /// Zero each conv part of `slot`: the state before position 0.
    fn zero_conv(&self, slot: usize) {
        for at in &self.slots[slot].conv {
            self.device.zero(self.card(), *at, spec::CONV_BYTES as usize);
        }
    }

    /// Run a block from position `pos0` of `slot` up to the end of `upto` runs, with the logit rows when given. The
    /// caller checked the block.
    fn forward(&mut self, slot: usize, pos0: usize, tokens: &[u32], upto: usize, logit_rows: Option<&[usize]>) {
        let pass = model::pass(
            &self.weights,
            &self.bufs,
            TaskRing {
                slot: &self.slots[slot],
                pos0,
                n: tokens.len(),
                runs: &self.runs,
                upto,
                logit_rows,
            },
        );
        let launches = pass.launches(self.launch_tasks);
        self.last_launches = launches.len();
        let bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.write(self.card(), self.bufs.ids, &bytes);
        self.run(&launches);
    }

    fn read_f32(&self, at: u64, n: usize) -> Vec<f32> {
        weights_f32(&self.device.read(self.card(), at, n * 4))
    }

    /// The residual after `upto` runs of `ids` from position 0 of `slot`, `[n][2304]`, or the final norm after all
    /// runs with `norm`. The probe of the oracle (8.3) uses it. It empties the slot.
    pub fn residual(&mut self, slot: usize, ids: &[u32], upto: usize, norm: bool) -> Result<Vec<f32>, String> {
        if upto > self.runs.len() || norm && upto != self.runs.len() {
            return Err(format!("a probe of {upto} runs with the norm {norm} does not exist"));
        }
        let segment = probe_segment(slot, ids);
        self.held.clear(slot);
        let block = self.held.check_pass(&[segment], self.config.max_block)?;
        self.forward(slot, 0, &block.tokens, upto, norm.then_some(&[][..]));
        let at = if norm { self.bufs.e_norm } else { self.bufs.x };
        let out = self.read_f32(at, ids.len() * HIDDEN);
        self.held.clear(slot);
        self.zero_conv(slot);
        Ok(out)
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
        let row = spec::ROW_BYTES as usize;
        let mut at = 0usize;
        let value = matches!(part, Part::V(_));
        let runs = self.slots[slot].runs(part.state(), value, rows.start as usize..rows.end as usize);
        for (src, n) in runs {
            self.device.read_into(self.card(), src, &mut out[at..at + n * row]);
            at += n * row;
        }
    }

    fn check_block(&self, segments: &[Segment], taps: &[Tap]) -> Result<Block, PassError> {
        if let Some(tap) = taps.first() {
            return Err(refused(format!(
                "nothing ran: the engine has no tap `{}`; the tap `residual` comes in milestone 35 step 8",
                tap.name
            )));
        }
        self.held
            .check_pass(segments, self.config.max_block)
            .map_err(|message| refused(format!("nothing ran: {message}")))
    }
}

/// The `f32` values of little-endian bytes.
fn weights_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The segment of a probe: `ids` from position 0 of `slot`, with no logit row.
fn probe_segment(slot: usize, ids: &[u32]) -> Segment {
    Segment {
        control: Vec::new(),
        state: State::Slot(slot as u32),
        inputs: nagaya_abi::SegmentInputs::Tokens(ids.to_vec()),
        positions: (0..ids.len() as u64).collect(),
        meta: Vec::new(),
        outputs: Default::default(),
        layer_loop: None,
        cycles: None,
    }
}

impl Engine for Inkling2Engine {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        let block = match self.check_block(segments, taps) {
            Ok(block) => block,
            Err(error) => return Err(self.fail(error)),
        };
        let start = Instant::now();
        if block.zero_conv() {
            self.zero_conv(block.slot);
        }
        let upto = self.runs.len();
        self.forward(block.slot, block.pos0, &block.tokens, upto, Some(&block.logit_rows));
        let outputs: Vec<Output> = (0..block.logit_rows.len())
            .map(|i| Output {
                name: "logits".to_owned(),
                dtype: Dtype::F32,
                bytes: self
                    .device
                    .read(self.card(), self.bufs.logits + (i * HEAD_ROWS * 4) as u64, LOGITS * 4),
            })
            .collect();
        self.held.ran(&block);
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
}

impl Parts for Inkling2Engine {
    /// Read a conv part whole, or the rows of a row part from its base: as many as `out` holds.
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        let (slot, which, from) = match self.held.check_part_read(state, part, out.len()) {
            Ok(found) => found,
            Err(message) => return Err(self.refuse_read(message)),
        };
        match which {
            Part::Conv(s) => self.device.read_into(self.card(), self.slots[slot].conv[s], out),
            _ => {
                let rows = out.len() as u64 / spec::ROW_BYTES;
                self.read_rows(slot, which, from..from + rows, out);
            }
        }
        Ok(())
    }

    /// Write a conv part whole. The slot holds no token until `set_n`.
    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        let (slot, which) = match self.held.check_part_write(state, part, bytes.len()) {
            Ok(found) => found,
            Err(message) => return self.refuse(message),
        };
        self.device
            .write(self.card(), self.slots[slot].conv[which.state()], bytes);
        self.held.wrote(slot, part, None);
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
        let row = spec::ROW_BYTES as usize;
        let mut at = ((kept.start - from) as usize) * row;
        let value = matches!(which, Part::V(_));
        let runs = self.slots[slot].runs(which.state(), value, kept.start as usize..kept.end as usize);
        for (dst, rows) in runs {
            self.device.write(self.card(), dst, &bytes[at..at + rows * row]);
            at += rows * row;
        }
        self.held.wrote(slot, part, Some(from..from + n));
        WriteResult::Ok
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        match self.held.check_set_n(state, n) {
            Ok(slot) => {
                if n == 0 {
                    self.zero_conv(slot);
                }
                self.held.set_n(slot, n);
                WriteResult::Ok
            }
            Err(message) => self.refuse(message),
        }
    }
}

impl Slots for Inkling2Engine {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.held.n.get(slot as usize).copied().unwrap_or(0)
    }

    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        if slot as usize >= self.held.n.len() {
            return self.refuse(format!("the engine has no slot {slot}"));
        }
        self.held.clear(slot as usize);
        self.zero_conv(slot as usize);
        WriteResult::Ok
    }
}

impl Coverage for Inkling2Engine {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        self.held.coverage(state, part, bytes)
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        self.held.fits(slot as usize, coverage)
    }
}
