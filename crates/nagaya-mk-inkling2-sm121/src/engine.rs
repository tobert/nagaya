//! The Inkling 2 engine on one NVIDIA GB10 (`sm_121`) GPU: the `Engine` trait, the parts, the slots and the coverage
//! over the pass of `pass::step` (`cuda` feature; engine-abi "The GB10 Inkling 2 binding").
//!
//! Every check of a call is a rule of `held::Held`, and each check runs before the call changes anything. So a
//! refused call changes nothing. The engine adds no numeric step: a pass gives the bits of `Pass::run_slot`.
//!
//! Errors:
//! - A refused check changes nothing, and the call says `Refused`.
//! - A failed launch poisons its slot. The slot refuses each call that changes it until a clear.
//! - A sticky CUDA error (the 700 class) leaves the context of the GPU unusable. The call that met it gives the
//!   poison, and each later call of the engine panics, so that the process restarts (spec 8.5).

use std::collections::BTreeMap;
use std::path::Path;
use std::ptr;
use std::time::Instant;

use nagaya_abi::{
    Coverage, CoverageSpan, Dtype, Engine, Identity, Limits, Output, PartError, PartId, Parts, PassError,
    PassErrorKind, PassOut, PassStatus, Segment, SlotId, Slots, State, Tap, Tier, WriteResult,
};
use nagaya_mk_cuda::NVCC_VERSION;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::memory::Available;

use crate::health::{after_open_problem, check_logit_rows, is_sticky, weights_problem};
use crate::held::{Block, CONV_BYTES, Held, Part, Source, check_values};
use crate::identity::{self, TextFacts};
use crate::pass::step::{Pass, PassConfig};
use crate::pass::weights::DeviceWeights;
use crate::spec::{self, Config, FAMILY, LOGITS, Model, WEIGHTS_FILE};

/// The owner of the device, the weights and the pass.
///
/// `Pass<'d>` borrows the device and the weights, and the weights borrow the device. A struct cannot hold a value and
/// its borrow, so the owner puts the device and the weights on the heap and gives the pass the lifetime `'static`.
/// This is sound for these reasons:
/// - Each heap value has one address from `Box::into_raw` until `release`. A move of the owner moves the pointers only.
/// - The pointers are raw, so a move of the owner does not assert that a `Box` is unique.
/// - The fields are private. The pass leaves the owner only as a borrow of `&self` or `&mut self`, so no reference of
///   the pass outlives the owner.
/// - `release` drops the pass first, then the weights, then the device. Each one frees its buffers through the
///   device, so the device must live longest.
struct Owner {
    pass: Option<Pass<'static>>,
    weights: *mut DeviceWeights<'static>,
    device: *mut CudaDevice,
}

impl Owner {
    fn new(device: CudaDevice) -> Self {
        Self {
            pass: None,
            weights: ptr::null_mut(),
            device: Box::into_raw(Box::new(device)),
        }
    }

    fn device(&self) -> &CudaDevice {
        // SAFETY: `device` is a live `Box` pointer from `new` until `release` sets it to null. A call after `release`
        // is a defect, and `release` runs only in `Drop` and in `close`, which consume the owner.
        unsafe { self.device.as_ref().expect("the device was released") }
    }

    fn load_weights(&mut self, dir: &Path) -> Result<(), String> {
        // SAFETY: the device lives until `release`, and `release` drops the weights first (see the type).
        let device: &'static CudaDevice = unsafe { &*self.device };
        let weights = DeviceWeights::load(device, dir, |_| {})?;
        self.weights = Box::into_raw(Box::new(weights));
        Ok(())
    }

    fn open_pass_with_taps(&mut self, config: &PassConfig) -> Result<(), String> {
        // SAFETY: as in `open_pass`.
        let (device, weights): (&'static CudaDevice, &'static DeviceWeights<'static>) =
            unsafe { (&*self.device, self.weights.as_ref().expect("the weights are loaded")) };
        self.pass = Some(Pass::open_with_taps(device, weights, config)?);
        Ok(())
    }

    fn open_pass(&mut self, config: &PassConfig) -> Result<(), String> {
        // SAFETY: as in `load_weights`: the device and the weights live until `release` drops the pass first.
        let (device, weights): (&'static CudaDevice, &'static DeviceWeights<'static>) =
            unsafe { (&*self.device, self.weights.as_ref().expect("the weights are loaded")) };
        self.pass = Some(Pass::open(device, weights, config)?);
        Ok(())
    }

    /// The bytes of device memory that the weights hold.
    fn weights_bytes(&self) -> u64 {
        // SAFETY: the pointer is a live `Box` pointer after `load_weights`, until `release`.
        unsafe { self.weights.as_ref().expect("the weights are loaded") }.bytes() as u64
    }

    fn pass(&self) -> &Pass<'static> {
        self.pass.as_ref().expect("the pass is open")
    }

    fn pass_mut(&mut self) -> &mut Pass<'static> {
        self.pass.as_mut().expect("the pass is open")
    }

    /// Drop the pass, the weights and the device, in this order. The result is the count of device buffers that the
    /// pass and the weights left behind: 0 when each one freed its buffers.
    fn release(&mut self) -> usize {
        drop(self.pass.take());
        if !self.weights.is_null() {
            // SAFETY: the pointer came from `Box::into_raw` in `load_weights`, and it is set to null here.
            drop(unsafe { Box::from_raw(self.weights) });
            self.weights = ptr::null_mut();
        }
        let Some(device) = (unsafe { self.device.as_ref() }) else {
            return 0;
        };
        let left = device.live_buffers();
        // SAFETY: the pointer came from `Box::into_raw` in `new`, and it is set to null here.
        drop(unsafe { Box::from_raw(self.device) });
        self.device = ptr::null_mut();
        left
    }
}

impl Drop for Owner {
    /// An open that fails after it made the owner frees what it took. `release` is safe to call a second time.
    fn drop(&mut self) {
        self.release();
    }
}

/// The Inkling 2 engine on one GB10 GPU.
pub struct Inkling2Sm121Engine {
    config: Config,
    limits: Limits,
    identity: Identity,
    held: Held,
    last_error: Option<String>,
    /// The first sticky error of the driver. After it, each call that uses the device panics.
    dead: Option<String>,
    /// The time of the digest of the checkpoint at the open, in milliseconds.
    digest_ms: u64,
    owner: Owner,
}

impl Drop for Inkling2Sm121Engine {
    /// Free the pass, the weights and the device. A buffer that is still live after it is a defect of the engine, and
    /// the drop panics (crash before you corrupt). A dead context cannot free its buffers, so it is not checked.
    fn drop(&mut self) {
        let left = self.owner.release();
        if left != 0 && self.dead.is_none() && !std::thread::panicking() {
            panic!(
                "the engine left {left} device buffers after it freed the pass and the weights. This is a defect of the \
                 engine: report it"
            );
        }
    }
}

/// The identity of an engine of `config` on a device with the driver `driver` and `sms` multiprocessors (spec 12.1).
pub(crate) fn identity_of(
    model: Model,
    checkpoint_digest: String,
    facts: TextFacts,
    config: &Config,
    (driver, sms): (i32, u32),
) -> Identity {
    let mut numbers: BTreeMap<String, String> = spec::identity_numbers(config);
    numbers.insert("driver".to_owned(), driver.to_string());
    numbers.insert("canary".to_owned(), "not run".to_owned());
    Identity {
        model: model.name().to_owned(),
        checkpoint_digest,
        family: FAMILY.to_owned(),
        tokenizer_hash: facts.tokenizer_hash,
        template: facts.template,
        numbers,
        plan_table: format!("nvcc {NVCC_VERSION} sm_121a; cuda driver {driver}; {sms} multiprocessors"),
        // D11 rule 3: the block and the cache format define the reference bits.
        bit_settings: spec::bit_settings(config),
        canary_hash: String::new(),
        build: BTreeMap::new(),
    }
}

/// The bytes of device memory that `limits` declares: the weights, the slots and the work.
pub(crate) fn device_bytes(limits: &Limits) -> u64 {
    limits
        .memory
        .mandatory
        .iter()
        .filter(|term| matches!(term.tier, Tier::Device(_)))
        .map(|term| term.bytes)
        .sum()
}

/// The message of an open that the free memory of the GPU cannot hold. It names the bytes and what the reader can do.
fn memory_message(planned: u64, available: Available, total: u64, config: &Config) -> String {
    let advice = if config.cache == nagaya_mk_cuda::attn::CacheFormat::E4m3 {
        "Name fewer slots or slots of fewer positions in `slots`"
    } else {
        "Name fewer slots or slots of fewer positions in `slots`, or set `cache` to `bf16` or `e4m3`"
    };
    let label = available.count.label();
    let available_bytes = available.bytes;
    format!(
        "the engine needs {planned} bytes of GPU memory for its weights, its slots and its work, and the {label} is {available_bytes} \
         bytes, with {total} bytes in all on the GPU. {advice}, or stop the programs that hold GPU memory"
    )
}

impl Inkling2Sm121Engine {
    /// Open the GPU at the PCI address `pci`, and upload the weights of the model folder of `config`.
    ///
    /// The open checks the config and the plan first, then the available memory of the GPU against the planned bytes, and
    /// only then it reads the whole checkpoint. After its allocations it checks that the GPU still holds a floor of
    /// available memory. The open refuses with a message that names the bytes, and it frees what it took.
    pub fn open(config: Config, pci: &str) -> Result<Self, String> {
        config.check()?;
        let dir = config.model_dir.clone();
        // The plan reads the folder, the generation config and the header of the checkpoint. It gives the limits.
        let limits = spec::plan(&config)?;
        let model = Model::read(&dir)?;
        let device = CudaDevice::open_by_pci(pci, config.cache.cubin())?;
        let planned = device_bytes(&limits);
        let (before, total) = device.available_bytes()?;
        if planned > before.bytes {
            return Err(memory_message(planned, before, total as u64, &config));
        }
        let (driver, sms) = (device.driver_version(), device.sm_count());
        let config_path = dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("cannot read `{}`: {e}", config_path.display()))?;
        let facts = identity::text_facts(&dir)?;
        let digest_start = Instant::now();
        let digest = identity::checkpoint_digest_cached(&config_text, &dir.join(WEIGHTS_FILE))?;
        let digest_ms = digest_start.elapsed().as_millis() as u64;

        let mut owner = Owner::new(device);
        owner.load_weights(&dir)?;
        let planned_weights = limits
            .memory
            .mandatory
            .iter()
            .find(|term| term.name == "weights")
            .map_or(0, |term| term.bytes);
        if let Some(message) = weights_problem(owner.weights_bytes(), planned_weights) {
            return Err(message);
        }
        let extra_cycles = 1 + config.extra_runs / spec::CYCLE_LAYERS;
        let pass_config = PassConfig {
            seqs: config.seqs(),
            max_rows: config.max_block,
            max_cycles: extra_cycles,
            profile: false,
            format: config.cache,
        };
        if config.taps {
            owner.open_pass_with_taps(&pass_config)?;
        } else {
            owner.open_pass(&pass_config)?;
        }
        // The pre-check counted the planned bytes against the free bytes. After the allocations the GPU must still hold
        // a floor of free memory, because the free bytes of a unified memory are the memory of the machine.
        let counted = owner.weights_bytes() + owner.pass().slots.iter().map(|slot| slot.bytes() as u64).sum::<u64>();
        let (after, _) = owner.device().available_bytes()?;
        if let Some(message) = after_open_problem(counted, before.bytes, after.bytes, after.count) {
            return Err(message);
        }
        let identity = identity_of(model, digest, facts, &config, (driver, sms));
        let held = Held::new(&config.seqs(), config.extra_runs, config.cache);
        Ok(Self {
            config,
            limits,
            identity,
            held,
            last_error: None,
            dead: None,
            digest_ms,
            owner,
        })
    }

    /// The time that the open took to hash the checkpoint, in milliseconds.
    pub fn digest_millis(&self) -> u64 {
        self.digest_ms
    }

    /// The bytes of device memory that the weights hold.
    pub fn weights_bytes(&self) -> u64 {
        self.owner.weights_bytes()
    }

    /// The count of device buffers that the engine holds now. It is more than 0 while the engine is open.
    pub fn live_buffers(&self) -> usize {
        self.owner.device().live_buffers()
    }

    /// The free and the total bytes of device memory, as the driver reports them.
    pub fn memory_info(&self) -> Result<(usize, usize), String> {
        self.owner.device().memory_info()
    }

    /// Free the engine, and give the count of device buffers that it left behind: 0 when it freed each one. A test
    /// uses it to find a leak. The plain drop panics on a leak.
    pub fn close(mut self) -> usize {
        let left = self.owner.release();
        // The drop that follows finds the owner empty.
        left
    }

    /// Mark `slot` as poisoned, as a failed launch does. A test uses it to reach the state that a failed launch leaves,
    /// because a real failed launch needs a task that faults the GPU.
    #[doc(hidden)]
    pub fn poison_for_test(&mut self, slot: usize) {
        self.owner.pass_mut().slots[slot].poisoned = true;
    }

    /// Panic when the context of the GPU is dead.
    fn guard(&self) {
        if let Some(why) = &self.dead {
            panic!(
                "the GPU context of the engine is dead after a sticky CUDA error: {why}. The engine cannot run again. \
                 Restart the process: a planned restart (spec 8.5) opens the engine again"
            );
        }
    }

    /// Take a failure of the device: a sticky error kills the engine.
    fn note_failure(&mut self, message: &str) {
        if is_sticky(message) && self.dead.is_none() {
            self.dead = Some(message.to_owned());
        }
    }

    fn poisoned(&self, slot: usize) -> bool {
        self.owner.pass().slots[slot].poisoned
    }

    fn poison(&mut self, slot: usize) {
        self.owner.pass_mut().slots[slot].poisoned = true;
    }

    fn poisoned_message(slot: usize) -> String {
        format!(
            "slot {slot} is poisoned by a failed call. Clear the slot with slot_clear before the next call that changes it"
        )
    }

    fn refuse_pass(&mut self, message: String) -> PassError {
        self.last_error = Some(message.clone());
        PassError {
            kind: PassErrorKind::Refused,
            message,
        }
    }

    fn refuse(&mut self, message: String) -> WriteResult {
        self.last_error = Some(message);
        WriteResult::Refused
    }

    fn refuse_read(&mut self, message: String) -> PartError {
        self.last_error = Some(message);
        PartError::Refused
    }

    /// A device call failed after it may have changed `slot`: poison the slot, and keep the cause.
    fn fail_slot(&mut self, slot: usize, message: String) {
        self.note_failure(&message);
        self.poison(slot);
        self.last_error = Some(message);
    }

    /// The pass of a block that the checks passed. Its error says whether the call changed the slot.
    fn run_block(&mut self, block: &Block) -> Result<Vec<f32>, PassError> {
        let slot = block.slot;
        if block.zero_conv() {
            // A pass at position 0 starts from conv states of zeros (engine-abi "Inkling 2", "Position 0 and no way
            // back"). A restore can have written a conv state with no count.
            if let Err(message) = self.owner.pass_mut().slots[slot].clear() {
                self.fail_slot(slot, message.clone());
                return Err(PassError {
                    kind: PassErrorKind::Ran(PassStatus::Poison),
                    message,
                });
            }
        }
        let position = self.owner.pass().slots[slot].position;
        assert_eq!(
            position, block.pos0,
            "slot {slot} holds {position} positions in the pass, and the held state says {}. This is a defect of the \
             engine: report it",
            block.pos0
        );
        // The pass keeps the launches of the last call. Clear them, so that a launch in the list is a launch of this call.
        self.owner.pass_mut().launches.clear();
        let result = self
            .owner
            .pass_mut()
            .run_slot(slot, block.cycles, &block.tokens, &block.logit_rows);
        match result {
            Ok(logits) => Ok(logits),
            Err(message) => {
                // The slot changed when a launch ran, even when a later step failed. The pass poisons a slot itself
                // after a failed launch. A failure after a good launch, or after the zero of the conv state, is
                // another change that the pass does not flag.
                let launched = !self.owner.pass().launches.is_empty();
                let changed = launched || block.zero_conv();
                if changed || self.poisoned(slot) {
                    self.fail_slot(slot, message.clone());
                    Err(PassError {
                        kind: PassErrorKind::Ran(PassStatus::Poison),
                        message,
                    })
                } else {
                    Err(self.refuse_pass(message))
                }
            }
        }
    }
}

impl Engine for Inkling2Sm121Engine {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// A block that the model of time puts above `max_quantum_ms` declares a bound (`bound::pass_bound_ms`). A pass that
    /// the checks refuse runs nothing, so it declares none, and `pass` gives the refusal.
    fn pass_bound_ms(&self, segments: &[Segment], taps: &[Tap]) -> Result<Option<std::num::NonZeroU32>, String> {
        if !taps.is_empty() {
            return Ok(None);
        }
        let Ok(block) = self.held.check_pass(segments, self.config.max_block) else {
            return Ok(None);
        };
        Ok(crate::bound::pass_bound_ms(
            block.pos0,
            block.tokens.len(),
            block.cycles,
        ))
    }

    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError> {
        self.guard();
        self.last_error = None;
        if !taps.is_empty() {
            return Err(self.refuse_pass(format!(
                "the pass names {} taps; the engine declares no taps yet",
                taps.len()
            )));
        }
        let block = match self.held.check_pass(segments, self.config.max_block) {
            Ok(block) => block,
            Err(message) => return Err(self.refuse_pass(message)),
        };
        if self.poisoned(block.slot) {
            return Err(self.refuse_pass(Self::poisoned_message(block.slot)));
        }
        if let Err(message) = check_logit_rows(&block.logit_rows) {
            return Err(self.refuse_pass(message));
        }
        let start = Instant::now();
        let logits = self.run_block(&block)?;
        let outputs = logits
            .as_chunks::<LOGITS>()
            .0
            .iter()
            .map(|row| Output {
                name: "logits".to_owned(),
                dtype: Dtype::F32,
                bytes: row.iter().flat_map(|value| value.to_le_bytes()).collect(),
            })
            .collect();
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
}

impl Parts for Inkling2Sm121Engine {
    /// Read a fixed part whole, or the rows of a row part from its base: as many as `out` holds. A run that the count
    /// of the slot does not reach reads as zeros, and the part `cycles` reads the count (engine-abi "Inkling 2").
    fn part_read(&mut self, state: State, part: PartId, out: &mut [u8]) -> Result<(), PartError> {
        self.guard();
        self.last_error = None;
        let (slot, which, from, source) = match self.held.check_part_read(state, part, out.len()) {
            Ok(found) => found,
            Err(message) => return Err(self.refuse_read(message)),
        };
        let read = match (which, source) {
            (_, Source::Zeros) => {
                out.fill(0);
                Ok(())
            }
            (_, Source::Count(count)) => {
                out.copy_from_slice(&count.to_le_bytes());
                Ok(())
            }
            (Part::Conv(s), Source::Device) => self.owner.pass().slots[slot].read_conv(s).map(|bytes| {
                debug_assert_eq!(bytes.len() as u64, CONV_BYTES);
                out.copy_from_slice(&bytes);
            }),
            (_, Source::Device) => {
                let rows = out.len() as u64 / self.held.row_bytes();
                self.read_rows(slot, which, from, rows, out)
            }
        };
        read.map_err(|message| {
            self.fail_slot(slot, message);
            PartError::Failed
        })
    }

    /// Write a fixed part whole: a conv part, or the count of cycles. The slot holds no token until `set_n`.
    fn part_write(&mut self, state: State, part: PartId, bytes: &[u8]) -> WriteResult {
        self.guard();
        self.last_error = None;
        let (slot, which) = match self.held.check_part_write(state, part, bytes) {
            Ok(found) => found,
            Err(message) => return self.refuse(message),
        };
        if self.poisoned(slot) {
            return self.refuse(Self::poisoned_message(slot));
        }
        if which == Part::Cycles {
            let count = u32::from_le_bytes(bytes.try_into().expect("check_part_write took 4 bytes"));
            self.held.wrote_count(slot, count);
            return WriteResult::Ok;
        }
        if let Err(message) = self.owner.pass().slots[slot].write_conv(which.state(), bytes) {
            self.fail_slot(slot, message);
            return WriteResult::Failed;
        }
        self.held.wrote(slot, part, None, zeros(bytes));
        WriteResult::Ok
    }

    fn cells_read(&mut self, state: State, part: PartId, from: u64, n: u64, out: &mut [u8]) -> Result<(), PartError> {
        self.guard();
        self.last_error = None;
        let (slot, which, source) = match self.held.check_read(state, part, from, n, out.len()) {
            Ok(found) => found,
            Err(message) => return Err(self.refuse_read(message)),
        };
        let read = match source {
            Source::Device => self.read_rows(slot, which, from, n, out),
            Source::Zeros | Source::Count(_) => {
                out.fill(0);
                Ok(())
            }
        };
        read.map_err(|message| {
            self.fail_slot(slot, message);
            PartError::Failed
        })
    }

    /// Write the rows of positions `from` to `from + n`. A ring keeps the last of them. The slot holds no token until
    /// `set_n`, so a pass cannot read rows of two states.
    fn cells_write(&mut self, state: State, part: PartId, from: u64, n: u64, bytes: &[u8]) -> WriteResult {
        self.guard();
        self.last_error = None;
        let (slot, which, kept) = match self.held.check_cells_write(state, part, from, n, bytes.len()) {
            Ok(found) => found,
            Err(message) => return self.refuse(message),
        };
        if let Err(message) = check_values(self.held.format, bytes) {
            return self.refuse(message);
        }
        if self.poisoned(slot) {
            return self.refuse(Self::poisoned_message(slot));
        }
        let row = self.held.row_bytes() as usize;
        let kept_bytes = &bytes[(kept.start - from) as usize * row..(kept.end - from) as usize * row];
        let value = matches!(which, Part::V(_));
        let written = self.owner.pass().slots[slot].write_cells(
            which.state(),
            value,
            kept.start as usize,
            (kept.end - kept.start) as usize,
            kept_bytes,
        );
        if let Err(message) = written {
            self.fail_slot(slot, message);
            return WriteResult::Failed;
        }
        self.held.wrote(slot, part, Some(from..from + n), zeros(bytes));
        WriteResult::Ok
    }

    fn set_n(&mut self, state: State, n: u64) -> WriteResult {
        self.guard();
        self.last_error = None;
        let slot = match self.held.check_set_n(state, n) {
            Ok(slot) => slot,
            Err(message) => return self.refuse(message),
        };
        if n == 0 {
            // A count of 0 empties the slot: the conv states go to zeros, and the poison goes.
            if let Err(message) = self.owner.pass_mut().slots[slot].clear() {
                self.fail_slot(slot, message);
                return WriteResult::Failed;
            }
        } else {
            if self.poisoned(slot) {
                return self.refuse(Self::poisoned_message(slot));
            }
            if let Err(message) = self.owner.pass_mut().slots[slot].set_position(n as usize) {
                self.fail_slot(slot, message);
                return WriteResult::Failed;
            }
        }
        self.held.set_n(slot, n);
        WriteResult::Ok
    }
}

impl Inkling2Sm121Engine {
    /// Read the rows `from` to `from + n` of a row part of `slot` into `out`.
    fn read_rows(&self, slot: usize, part: Part, from: u64, n: u64, out: &mut [u8]) -> Result<(), String> {
        let value = matches!(part, Part::V(_));
        let bytes = self.owner.pass().slots[slot].read_cells(part.state(), value, from as usize, n as usize)?;
        out.copy_from_slice(&bytes);
        Ok(())
    }
}

/// True when every byte is zero: the bytes of an inactive run.
fn zeros(bytes: &[u8]) -> bool {
    bytes.iter().all(|byte| *byte == 0)
}

impl Slots for Inkling2Sm121Engine {
    fn slot_high(&self, slot: SlotId) -> u64 {
        self.held.n.get(slot as usize).copied().unwrap_or(0)
    }

    /// Empty a slot: its count of positions and cycles go to 0, its conv states go to zeros, and its poison goes. The
    /// planes of the cache stay as they are: a pass reads only the rows that it or a restore wrote.
    fn slot_clear(&mut self, slot: SlotId) -> WriteResult {
        self.guard();
        self.last_error = None;
        let slot = slot as usize;
        if slot >= self.held.n.len() {
            return self.refuse(format!("the engine has no slot {slot}"));
        }
        if let Err(message) = self.owner.pass_mut().slots[slot].clear() {
            self.fail_slot(slot, message);
            return WriteResult::Failed;
        }
        self.held.clear(slot);
        WriteResult::Ok
    }
}

impl Coverage for Inkling2Sm121Engine {
    fn coverage(&self, state: State, part: PartId, bytes: &[u8]) -> CoverageSpan {
        self.held.coverage(state, part, bytes)
    }

    fn fits(&self, coverage: &CoverageSpan, slot: SlotId) -> Result<(), String> {
        self.held.fits(slot as usize, coverage)
    }
}
