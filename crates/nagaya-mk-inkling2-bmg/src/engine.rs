//! The Inkling 2 engine on its card: the `Engine` trait, the parts, the slots and the coverage over the ring of
//! `model`, and the vision tower of `vision` for the images of a pass. Every check of a call is a rule of `spec::Held`, so a refused call changes nothing.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nagaya_abi::{
    BitClass, BitSetting, Control, ControlLimit, ControlOp, Coverage, CoverageSpan, Dtype, Engine, Identity, Limits,
    Output, PartError, PartId, Parts, PassError, PassErrorKind, PassOut, PassStatus, Segment, SlotId, Slots, State,
    Tap, WriteResult,
};
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView};
use nagaya_mk_bmg::{IGC_REVISION, INKLING2_KERNEL, KERNEL_ENTRY, OCLOC_VERSION};

use crate::device::{Device, OnCard};
use crate::host::ops::HIDDEN;
use crate::host::weights::LOGITS;
use crate::media::{Grid, HeldRows};
use crate::model::{self, Bufs, LAUNCH_TASKS, SlotMemory, TapBufs, TapRing, TaskRing};
use crate::spec::{self, Block, Config, FAMILY, HEAD_ROWS, Held, Model, Part, ROUTE_WORDS, Run, Source, Taps};
use crate::vision::{self, PassBufs};
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
    /// The buffers of the taps, in `work` (engine-abi "The lab of Inkling 2").
    tap_bufs: TapBufs,
    /// The control table on the card, the term `control`, and its limit.
    control: (u64, ControlLimit),
    /// The weights of the vision tower: the term `vision`.
    tower: vision::Weights,
    /// The buffers of the images of a pass, in `work`.
    images: PassBufs,
    /// The image whose projector rows `images.out` holds.
    held_rows: HeldRows,
    slots: Vec<SlotMemory>,
    held: Held,
    /// The runs of a pass at each count of cycles, from 1. A probe of the oracle runs the default.
    runs: Vec<Vec<Run>>,
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
    let mut numbers = spec::identity_numbers(config);
    numbers.insert(
        "kernel.megakernel_inkling2_bmg.bin".to_owned(),
        format!("blake3:{}", blake3::hash(INKLING2_KERNEL).to_hex()),
    );
    numbers.insert("ocloc".to_owned(), OCLOC_VERSION.to_owned());
    numbers.insert("igc".to_owned(), IGC_REVISION.to_owned());
    numbers.insert("driver".to_owned(), driver.to_string());
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
    spec::read_generation(dir)?;
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
    /// Open the card at the PCI address `pci`, and upload the text model and the vision tower of the model folder of
    /// `config`.
    ///
    /// The open checks the config, `config.json` and every tensor before it opens the card. Its device bytes must
    /// equal the plan.
    pub fn open(config: Config, pci: &str) -> Result<Self, String> {
        config.check()?;
        let dir = config.model_dir.clone();
        let model = Model::read(&dir)?;
        spec::read_generation(&dir)?;
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
        // The tower checks its tensors before the card opens too.
        let mut count = Count::default();
        vision::load(&mut count, &view)?;
        if count.bytes != vision::WEIGHTS_BYTES {
            return Err(format!(
                "the load counts {} device bytes of the vision tower, and the plan counts {}",
                count.bytes,
                vision::WEIGHTS_BYTES
            ));
        }
        let checkpoint_digest = checkpoint_digest(&config_text, &view)?;

        // The memory of this process on the card before the open, so the check after it sees what the open placed.
        let before = nagaya_mk_bmg::vram::card_memory(pci)?;
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
        let tower = vision::load(
            &mut OnCard {
                device: &mut device,
                card,
            },
            &view,
        )?;
        let bufs = Bufs::new(&mut |bytes| device.zeros(card, bytes), config.max_block);
        let images = PassBufs::new(&mut |bytes| device.zeros(card, bytes), config.max_block);
        let ids: Vec<u8> = Bufs::sink_ids_of(config.max_block)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        device.write(card, bufs.sink_ids, &ids);
        let tap_bufs = TapBufs::new(&mut |bytes| device.zeros(card, bytes), config.extra_runs);
        let control_limit = spec::control_limit();
        let control = (device.zeros(card, control_limit.table_bytes() as usize), control_limit);
        let slots: Vec<SlotMemory> = config
            .seqs()
            .iter()
            .map(|seq| SlotMemory::new(&mut |bytes| device.zeros(card, bytes), *seq, config.extra_runs))
            .collect();
        let limits = spec::limits(&config, planned_weights);
        limits.check()?;
        let planned = planned_weights
            + vision::WEIGHTS_BYTES as u64
            + config.slots_bytes()
            + spec::bufs_bytes(config.max_block)
            + PassBufs::bytes(config.max_block)
            + spec::tap_bytes(config.extra_runs)
            + control_limit.table_bytes();
        if device.bytes as u64 != planned {
            return Err(format!(
                "the engine took {} device bytes, and its limits declare {planned}",
                device.bytes
            ));
        }
        let mut ring = Ring::new(card, spec::scratch_floats(config.max_seq()));
        // Under memory pressure, the driver gives a buffer its memory only at its first use. The open writes each
        // other buffer, so it fills the ring too, and the check sees all of it.
        ring.zero_device(card);
        // The driver puts a buffer that does not fit in system memory, and then each pass runs 5 to 6 times slower.
        // Thus the open refuses, and frees what it took (engine-abi "Inkling 2", "The cost of a slot").
        // The device bytes of the ring are the scratch, the scoreboard, and the wake buffer and the device copy of the
        // tasks, which the card filled at its open. The tasks of the ring are host memory.
        let ring_bytes = spec::ring_device_bytes(config.max_seq());
        let check = || {
            nagaya_mk_bmg::vram::card_memory(pci)
                .and_then(|after| nagaya_mk_bmg::vram::check_open(pci, planned + ring_bytes, before, after))
        };
        let mut fits = check();
        if fits.is_ok() {
            // The driver places the state of a launch at the first launch, so a launch of the halt alone places it
            // before the second check. It comes after the first check, because the launch of an open that the card
            // does not hold fails with no device memory.
            ring.set(&[]);
            // SAFETY: the ring holds only its halt.
            unsafe { ring.run(card) };
            fits = check();
        }
        if let Err(message) = fits {
            ring.free(card);
            device.free_all(card);
            return Err(message);
        }

        let identity = identity_of(model, checkpoint_digest, facts, &config, pci, driver)?;
        let runs = (1..=usize::from(limits_cycles_max(&limits))).map(spec::runs).collect();
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
            tap_bufs,
            control,
            tower,
            images,
            held_rows: HeldRows::default(),
            slots,
            held,
            runs,
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

    /// The runs of a pass of `cycles` cycles.
    fn runs_of(&self, cycles: usize) -> &[Run] {
        &self.runs[cycles - 1]
    }

    /// The count of the images that the tower ran for. A gate reads it: an image in several blocks is one run.
    pub fn tower_runs(&self) -> u64 {
        self.held_rows.runs()
    }

    /// Run the tower of each image span of `block` that the engine does not hold, from the rows of `segment`, and put
    /// the projector rows of the block's positions into the media rows. A latent span puts its rows there with no
    /// tower. The copies that this returns put them over
    /// the normed embedding rows of the placeholders (engine-abi "The choices of the pass"). The caller checked the
    /// block.
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
            let places_rows = used.block_row..used.block_row + used.take.len();
            let Some(g) = used.span.grid() else {
                // Latent rows go into the media rows of their positions, with no tower and no norm (engine-abi "The
                // lab of Inkling 2", "Latent rows").
                let row = HIDDEN * 4;
                let at = self.images.media + (used.block_row * row) as u64;
                self.device
                    .write(self.card(), at, &bytes[used.take.start * row..used.take.end * row]);
                places.push(self.images.place(self.bufs.x, places_rows));
                continue;
            };
            let mut tasks = Vec::new();
            if !self.held_rows.holds(g, blake3::hash(bytes)) {
                let card = &self.cluster.cards[0];
                self.device.write(card, self.images.rows, bytes);
                self.device.write(card, self.images.axes, &vision::axes(g));
                tasks = vision::tower(
                    &self.tower,
                    &self.images.tower,
                    g,
                    self.images.rows,
                    self.images.axes,
                    self.images.out,
                )
                .tasks;
            }
            tasks.push(self.images.take(used.take.clone(), used.block_row));
            self.run(&[tasks]);
            places.push(self.images.place(self.bufs.x, places_rows));
        }
        places
    }

    /// Run a block of `cycles` cycles from position `pos0` of `slot` up to the end of `upto` runs, with the logit rows
    /// when given, and the tasks `after_embed` after the norm of the embedding. The caller checked the block.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        slot: usize,
        pos0: usize,
        tokens: &[u32],
        (cycles, upto): (usize, usize),
        logit_rows: Option<&[usize]>,
        after_embed: &[MkTaskDescriptor],
        (control, taps): (&[model::Control], Taps),
    ) {
        let tap_bufs = self.tap_bufs;
        let pass = model::pass(
            &self.weights,
            &self.bufs,
            TaskRing {
                slot: &self.slots[slot],
                pos0,
                n: tokens.len(),
                runs: self.runs_of(cycles),
                upto,
                logit_rows,
                after_embed,
                control,
                taps: taps.any().then_some(TapRing {
                    named: taps,
                    bufs: &tap_bufs,
                }),
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
    /// runs with `norm`. The probe of the oracle (8.3) uses it. It empties the slot. It returns its error, and it
    /// leaves `last_error` as it is.
    pub fn residual(&mut self, slot: usize, ids: &[u32], upto: usize, norm: bool) -> Result<Vec<f32>, String> {
        let all = self.runs_of(spec::DEFAULT_CYCLES).len();
        if upto > all || norm && upto != all {
            return Err(format!("a probe of {upto} runs with the norm {norm} does not exist"));
        }
        // The check comes before the clear, so a refused probe changes nothing.
        let block = self.held.check_probe(slot, ids, self.config.max_block)?;
        self.held.clear(slot);
        self.forward(
            slot,
            0,
            &block.tokens,
            (block.cycles, upto),
            norm.then_some(&[][..]),
            &[],
            (&[], Taps::default()),
        );
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

    fn check_block(&self, segments: &[Segment], taps: &[Tap]) -> Result<(Block, Taps), PassError> {
        let named = Taps::of(taps).map_err(|message| refused(format!("nothing ran: {message}")))?;
        let block = self
            .held
            .check_pass(segments, self.config.max_block)
            .map_err(|message| refused(format!("nothing ran: {message}")))?;
        Ok((block, named))
    }

    /// The control items of a block as tasks of the ring: the address of each vector in the table.
    fn control_of(&self, block: &Block) -> Vec<model::Control> {
        let (table, limit) = self.control;
        block
            .control
            .iter()
            .map(|item| model::Control {
                vector: table + u64::from(item.index) * limit.entry_bytes(),
                op: match item.op {
                    ControlOp::Add => nagaya_mk_bmg::protocol::ControlOp::Add,
                    ControlOp::Ablate => nagaya_mk_bmg::protocol::ControlOp::Ablate,
                    ControlOp::Clamp => nagaya_mk_bmg::protocol::ControlOp::Clamp,
                },
                strength: item.strength,
                layers: item.layers.clone(),
            })
            .collect()
    }

    /// The outputs of the taps of a pass of `block` that ran, in the order of `spec::TAPS` (engine-abi "The lab of
    /// Inkling 2").
    fn tap_outputs(&self, block: &Block, taps: Taps) -> Vec<Output> {
        let b = &self.tap_bufs;
        let rows = &block.logit_rows;
        let runs = self.runs_of(block.cycles).len();
        let row = HIDDEN * 4;
        let messages = (block.cycles - 1) * spec::CYCLE_LAYERS * 4;
        let f32s = |name: &str, bytes: Vec<u8>| Output {
            name: name.to_owned(),
            dtype: Dtype::F32,
            bytes,
        };
        let gather = |at: &dyn Fn(usize) -> u64, n: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(rows.len() * n);
            for i in 0..rows.len() {
                out.extend(self.device.read(self.card(), at(i), n));
            }
            out
        };
        let mut out = Vec::new();
        if taps.residual {
            out.push(f32s(spec::RESIDUAL_TAP, gather(&|i| b.residual_row(i, 0), runs * row)));
        }
        if taps.fold {
            let at = |i: usize| self.bufs.e_norm + (rows[i] * row) as u64;
            out.push(f32s(spec::FOLD_TAP, gather(&at, row)));
        }
        if taps.blocks {
            out.push(f32s(
                spec::BLOCKS_TAP,
                gather(&|i| b.block_row(i, 0), (block.cycles + 3) * row),
            ));
        }
        if taps.messages {
            let at = |i: usize| b.message_row(b.messages, i, 0, 0);
            out.push(f32s(spec::MESSAGES_TAP, gather(&at, messages * row)));
        }
        if taps.gates {
            let at = |i: usize| b.message_row(b.gates, i, 0, 0);
            out.push(f32s(spec::GATES_TAP, gather(&at, messages * row)));
        }
        if taps.routes {
            let words = gather(&|i| b.route_row(i, 0), runs * ROUTE_WORDS * 4);
            let (mut picks, mut weights) = (Vec::new(), Vec::new());
            for decision in words.as_chunks::<{ ROUTE_WORDS * 4 }>().0 {
                picks.extend_from_slice(&decision[..32]);
                weights.extend_from_slice(&decision[32..64]);
            }
            out.push(Output {
                name: spec::ROUTES_TAP.to_owned(),
                dtype: Dtype::U32,
                bytes: picks,
            });
            out.push(f32s(spec::ROUTE_WEIGHTS, weights));
        }
        out
    }
}

/// The grid and the patch rows of each image span of `block`, in the order of the spans. Latent spans have none.
fn image_rows<'a>(segment: &'a Segment, block: &Block) -> Vec<(Grid, &'a [u8])> {
    let nagaya_abi::SegmentInputs::TokensAndRows { rows, .. } = &segment.inputs else {
        return Vec::new();
    };
    block
        .spans
        .iter()
        .filter_map(|used| Some((used.span.grid()?, &rows[used.rows_at..used.rows_at + used.span.bytes()])))
        .collect()
}

/// True when each byte is zero: the bytes of a run that a count does not reach.
fn zeros(bytes: &[u8]) -> bool {
    bytes.iter().all(|b| *b == 0)
}

/// The most cycles of the limits.
fn limits_cycles_max(limits: &Limits) -> u8 {
    limits.cycles.expect("the engine declares `cycles`").max
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

impl Engine for Inkling2Engine {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// A pass that runs the tower of an image declares a bound from its count of tokens (`bound::pass_bound_ms`).
    /// The held rows of the last image tell which towers run, as in `encode`.
    fn pass_bound_ms(&self, segments: &[Segment], taps: &[Tap]) -> Result<Option<std::num::NonZeroU32>, String> {
        // A pass that the checks refuse runs nothing, and `pass` gives the refusal.
        let Ok((block, _)) = self.check_block(segments, taps) else {
            return Ok(None);
        };
        // The bound hashes each image, and the pass hashes it again: about 20 ms for the largest image, against 843 ms
        // for a later block of it (engine-abi "The bound of a pass with a tower").
        let images = image_rows(&segments[0], &block)
            .into_iter()
            .map(|(grid, rows)| (grid, blake3::hash(rows)));
        let towers: Vec<usize> = self.held_rows.towers(images).iter().map(|g| g.tokens()).collect();
        Ok(crate::bound::pass_bound_ms(&towers))
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.last_error = None;
        let (block, named) = match self.check_block(segments, taps) {
            Ok(found) => found,
            Err(error) => return Err(self.fail(error)),
        };
        let control = self.control_of(&block);
        let start = Instant::now();
        let places = self.encode(&segments[0], &block);
        if block.zero_conv() {
            self.zero_conv(block.slot);
        }
        let upto = self.runs_of(block.cycles).len();
        self.forward(
            block.slot,
            block.pos0,
            &block.tokens,
            (block.cycles, upto),
            Some(&block.logit_rows),
            &places,
            (&control, named),
        );
        let mut outputs: Vec<Output> = (0..block.logit_rows.len())
            .map(|i| Output {
                name: "logits".to_owned(),
                dtype: Dtype::F32,
                bytes: self
                    .device
                    .read(self.card(), self.bufs.logits + (i * HEAD_ROWS * 4) as u64, LOGITS * 4),
            })
            .collect();
        outputs.extend(self.tap_outputs(&block, named));
        self.held.ran(&block);
        Ok(PassOut {
            status: PassStatus::Ok,
            ms: start.elapsed().as_millis() as u64,
            outputs,
        })
    }

    /// Why the last call that returns a status failed, or `None` when it did not fail. Each such call clears it first.
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

    fn control(&mut self) -> Option<&mut dyn Control> {
        Some(self)
    }
}

impl Control for Inkling2Engine {
    /// Write one vector into its entry of the table. A wrong index, a wrong size or a value that is not finite is
    /// refused, and nothing changes.
    fn control_write(&mut self, index: u32, bytes: &[u8]) -> Result<(), String> {
        self.last_error = None;
        let (table, limit) = self.control;
        if let Err(message) = spec::check_control_write(&limit, index, bytes) {
            self.last_error = Some(message.clone());
            return Err(message);
        }
        self.device
            .write(self.card(), table + u64::from(index) * limit.entry_bytes(), bytes);
        Ok(())
    }
}

impl Parts for Inkling2Engine {
    /// Read a fixed part whole, or the rows of a row part from its base: as many as `out` holds. A run that the count
    /// of the slot does not reach reads as zeros, and the part `cycles` reads the count (engine-abi "Inkling 2").
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.last_error = None;
        let (slot, which, from, source) = match self.held.check_part_read(state, part, out.len()) {
            Ok(found) => found,
            Err(message) => return Err(self.refuse_read(message)),
        };
        match (which, source) {
            (_, Source::Zeros) => out.fill(0),
            (_, Source::Count(count)) => out.copy_from_slice(&count.to_le_bytes()),
            (Part::Conv(s), Source::Device) => self.device.read_into(self.card(), self.slots[slot].conv[s], out),
            (_, Source::Device) => {
                let rows = out.len() as u64 / spec::ROW_BYTES;
                self.read_rows(slot, which, from..from + rows, out);
            }
        }
        Ok(())
    }

    /// Write a fixed part whole: a conv part, or the count of cycles. The slot holds no token until `set_n`.
    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        self.last_error = None;
        let (slot, which) = match self.held.check_part_write(state, part, bytes) {
            Ok(found) => found,
            Err(message) => return self.refuse(message),
        };
        if which == Part::Cycles {
            let count = u32::from_le_bytes(bytes.try_into().expect("check_part_write took 4 bytes"));
            self.held.wrote_count(slot, count);
            return WriteResult::Ok;
        }
        self.device
            .write(self.card(), self.slots[slot].conv[which.state()], bytes);
        self.held.wrote(slot, part, None, zeros(bytes));
        WriteResult::Ok
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.last_error = None;
        let (slot, which, source) = match self.held.check_read(state, part, from, n, out.len()) {
            Ok(found) => found,
            Err(message) => return Err(self.refuse_read(message)),
        };
        match source {
            Source::Device => self.read_rows(slot, which, from..from + n, out),
            Source::Zeros | Source::Count(_) => out.fill(0),
        }
        Ok(())
    }

    /// Write the rows of positions `from` to `from + n`. A ring keeps the last of them. The slot holds no token until
    /// `set_n`, so a pass cannot read rows of two states.
    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.last_error = None;
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
        self.held.wrote(slot, part, Some(from..from + n), zeros(bytes));
        WriteResult::Ok
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.last_error = None;
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
        self.last_error = None;
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
