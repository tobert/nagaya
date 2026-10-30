//! The SDXL engine on its card: the `Image` capability of engine-abi "Image" over the graphs of `model`.

use std::collections::BTreeMap;

use nagaya_abi::{
    Engine, Identity, Image, ImageChunk, ImageJob, ImageSampler, ImageSpacing, ImageStepOut, Limits, PassError,
    PassErrorKind, PassOut, RgbImage, Segment, SlotId, Tap,
};
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_bmg::{IGC_REVISION, KERNEL_ENTRY, OCLOC_VERSION, SDXL_KERNEL};

use crate::device::{Device, OnCard};
use crate::host;
use crate::lora::{self, Target};
use crate::model::{self, Carve, Run, TextBufs, UnetBufs, UnetSize, VaeBufs};
use crate::noise::{self, Purpose};
use crate::plan::{MAX_LATENT_ROWS, arena_bytes, limits};
use crate::sampler::{self, History, Schedule};
use crate::spec::*;
use crate::weights::{Adapter, Weights, f32_bytes, f32_values, sources};
use std::collections::HashMap;

pub use crate::spec::MODEL;

/// A job in the slot.
struct Job {
    job: ImageJob,
    schedule: Schedule,
    size: UnetSize,
    /// The latent of each image, pixel-major: rows of 4 channels.
    latents: Vec<Vec<f32>>,
    history: Vec<History>,
    step: usize,
}

/// The SDXL engine.
pub struct SdxlEngine {
    limits: Limits,
    identity: Identity,
    last_error: Option<String>,
    cluster: L0Cluster,
    ring: Ring,
    device: Device,
    weights: Weights,
    text: TextBufs,
    unet: UnetBufs,
    vae: VaeBufs,
    job: Option<Job>,
    /// The device mark before the LoRA files of the job were uploaded; `None` when the job has no LoRA.
    lora_mark: Option<usize>,
}

// The engine holds USM pointers of its card, and one thread at a time calls it.
unsafe impl Send for SdxlEngine {}

impl Drop for SdxlEngine {
    /// Free the weights, the arena and the ring (engine-abi "`close` in Rust"). The card destroys its context after
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

/// The BLAKE3 digest of the files of the checkpoint that the engine reads.
fn checkpoint_digest(config: &Config) -> Result<String, String> {
    let mut all = blake3::Hasher::new();
    for (_, path) in sources(config)? {
        let map = MappedSafeTensors::open(&path)?;
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

/// The pixel-major rows of `chunk` chunks of both encoders, joined: each row holds its 768 values of encoder 1, then
/// its 1280 values of encoder 2.
fn joined(hidden1: &[f32], hidden2: &[f32], rows: std::ops::Range<usize>) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * CONTEXT);
    for r in rows {
        out.extend_from_slice(&hidden1[r * TEXT1.hidden..(r + 1) * TEXT1.hidden]);
        out.extend_from_slice(&hidden2[r * TEXT2.hidden..(r + 1) * TEXT2.hidden]);
    }
    out
}

impl SdxlEngine {
    /// Upload the LoRA files of a job, and set their deltas on the weights (spec 11.9a). Each file is parsed before any
    /// upload, so a bad file leaves the card as it was. A job with no LoRA changes nothing.
    fn attach_loras(&mut self, job: &ImageJob) -> Result<(), String> {
        if self.lora_mark.is_some() {
            return Err("the slot still holds the LoRAs of an earlier job; clear the slot first".to_owned());
        }
        if job.loras.is_empty() {
            return Ok(());
        }
        let targets: Vec<Target> = self
            .weights
            .modules()
            .iter()
            .filter_map(|(name, shape)| Target::of_weight(name, shape))
            .collect();
        let mut parsed = Vec::new();
        for input in &job.loras {
            let file = lora::parse(&input.file, input.weight, &targets)
                .map_err(|e| format!("nothing ran: the LoRA {} is refused: {e}", input.id))?;
            parsed.push(file);
        }
        let card = &self.cluster.cards[0];
        let mark = self.device.mark();
        let mut adapters: HashMap<String, Vec<Adapter>> = HashMap::new();
        let mut scratch_floats = 0;
        for module in parsed.iter().flat_map(|file| &file.modules) {
            // The scale goes into the up matrix, so the engine adds the delta with no further scale.
            let up: Vec<f32> = module.up.iter().map(|v| v * module.scale).collect();
            let down = self.device.upload(card, &f32_bytes(&module.down));
            let up = self.device.upload(card, &f32_bytes(&up));
            scratch_floats = scratch_floats.max(module.rank);
            adapters.entry(module.target.prefixed()).or_default().push(Adapter {
                down,
                up,
                rank: module.rank,
                kernel: module.target.kernel,
                scratch: 0,
            });
        }
        // The hidden values of the largest batch: a UNet step of each image, or the text rows of a prompt.
        let (h, w) = (job.height as usize / VAE_SCALE, job.width as usize / VAE_SCALE);
        let rows = (2 * job.count as usize * h * w).max(2 * MAX_CHUNKS * CHUNK_IDS);
        if !adapters.is_empty() {
            let scratch = self.device.zeros(card, rows * scratch_floats * 4);
            for list in adapters.values_mut() {
                for adapter in list {
                    adapter.scratch = scratch;
                }
            }
        }
        self.weights.set_adapters(adapters);
        self.lora_mark = Some(mark);
        Ok(())
    }

    /// Free the LoRA files of the job and their deltas. The job's base weights are not changed by a LoRA.
    fn release_loras(&mut self) {
        if let Some(mark) = self.lora_mark.take() {
            self.weights.set_adapters(HashMap::new());
            self.device.release_to(&self.cluster.cards[0], mark);
        }
    }

    /// Open the card at the PCI address `pci`, and upload the checkpoint of `config`.
    pub fn open(config: Config, pci: &str) -> Result<Self, String> {
        let dir = config.model_dir.clone();
        let mut tokenizer = Vec::new();
        for name in ["tokenizer/vocab.json", "tokenizer/merges.txt"] {
            let path = dir.join(name);
            tokenizer.extend(std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?);
        }
        let tokenizer_hash = format!("blake3:{}", blake3::hash(&tokenizer).to_hex());
        let checkpoint_digest = checkpoint_digest(&config)?;
        // A single file names the model; a folder is SDXL base.
        let model = match &config.checkpoint {
            Some(file) => file
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.trim_end_matches(".safetensors").to_owned())
                .ok_or_else(|| format!("the checkpoint {} has no file name", file.display()))?,
            None => MODEL.to_owned(),
        };

        // Only the engine's card gets a context, so the open does not stall the work of another card.
        let cluster = L0Cluster::init_at(SDXL_KERNEL, KERNEL_ENTRY, &[pci.to_owned()])?;
        let card = &cluster.cards[0];
        let driver = cluster.driver_version();

        let mut device = Device::new(card);
        let weights = Weights::load(
            &config,
            &mut OnCard {
                device: &mut device,
                card,
            },
        )?;
        // The plan checked the graphs at the admission; this checks them again for a file that changed since.
        model::check(&weights)?;
        let weights_bytes = device.bytes as u64;
        let arena = arena_bytes();
        let at = device.zeros(card, arena as usize);
        let mut carve = Carve::new(at, arena);
        let text = TextBufs::carve(&mut carve, 2 * MAX_CHUNKS);
        let (mut unet_carve, mut vae_carve) = (carve, carve);
        let unet = UnetBufs::carve(&mut unet_carve, MAX_IMAGES, MAX_LATENT_ROWS, MAX_CHUNKS * CHUNK_IDS);
        let vae = VaeBufs::carve(&mut vae_carve, MAX_IMAGE_PIXELS);
        let ring = Ring::new(card, 1);

        let mut numbers = BTreeMap::new();
        numbers.insert(
            "kernel.megakernel_sdxl_bmg.bin".to_owned(),
            format!("blake3:{}", blake3::hash(SDXL_KERNEL).to_hex()),
        );
        numbers.insert("ocloc".to_owned(), OCLOC_VERSION.to_owned());
        numbers.insert("igc".to_owned(), IGC_REVISION.to_owned());
        numbers.insert("driver".to_owned(), driver.to_string());
        numbers.insert("canary".to_owned(), "not run".to_owned());
        let identity = Identity {
            model,
            checkpoint_digest,
            family: "sdxl".to_owned(),
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
        Ok(Self {
            limits: limits(weights_bytes),
            identity,
            last_error: None,
            cluster,
            ring,
            device,
            weights,
            text,
            unet,
            vae,
            job: None,
            lora_mark: None,
        })
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn run(&mut self, run: &Run) {
        self.ring.set(&run.tasks);
        // SAFETY: each task points at the weights or at the arena of this engine, sized by the plan for its run.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }

    fn write_f32(&self, at: u64, values: &[f32]) {
        self.device.write(self.card(), at, &f32_bytes(values));
    }

    fn write_u32(&self, at: u64, values: &[u32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.device.write(self.card(), at, &bytes);
    }

    fn read_f32(&self, at: u64, n: usize) -> Vec<f32> {
        f32_values(&self.device.read(self.card(), at, n * 4))
    }

    /// The text context and the pooled text of the negative prompt and of the prompt (spec 11.9a): the hidden rows of
    /// both encoders with the token weights of A1111, `[2][77 * chunks][2048]`, and `[2][1280]`. The prompt with fewer
    /// chunks gets empty chunks, so both have one length.
    pub fn encode(&mut self, prompt: &[ImageChunk], negative: &[ImageChunk], clip_skip: usize) -> (Vec<f32>, Vec<f32>) {
        let n = prompt.len().max(negative.len());
        let pad = |chunks: &[ImageChunk]| {
            let mut out = chunks.to_vec();
            out.resize(n, host::empty_chunk());
            out
        };
        let seqs: Vec<ImageChunk> = pad(negative).into_iter().chain(pad(prompt)).collect();
        let ids1: Vec<u32> = seqs.iter().flat_map(|c| c.ids1.iter().copied()).collect();
        let ids2: Vec<u32> = seqs.iter().flat_map(|c| c.ids2.iter().copied()).collect();
        let pos: Vec<u32> = (0..2 * n).flat_map(|_| 0..CHUNK_IDS as u32).collect();
        self.write_u32(self.text.ids1, &ids1);
        self.write_u32(self.text.ids2, &ids2);
        self.write_u32(self.text.pos, &pos);
        let eos = [
            host::eos_position(&seqs[0].ids2),
            n * CHUNK_IDS + host::eos_position(&seqs[n].ids2),
        ];
        for second in [false, true] {
            for run in model::text_encoder(&self.weights, second, &self.text, 2 * n, clip_skip, eos) {
                self.run(&run);
            }
        }
        let rows = 2 * n * CHUNK_IDS;
        let mut hidden1 = self.read_f32(self.text.hidden1, rows * TEXT1.hidden);
        let mut hidden2 = self.read_f32(self.text.hidden2, rows * TEXT2.hidden);
        for (s, chunk) in seqs.iter().enumerate() {
            let r = s * CHUNK_IDS..(s + 1) * CHUNK_IDS;
            host::weight_chunk(
                &mut hidden1[r.start * TEXT1.hidden..r.end * TEXT1.hidden],
                &chunk.weights,
            );
            host::weight_chunk(
                &mut hidden2[r.start * TEXT2.hidden..r.end * TEXT2.hidden],
                &chunk.weights,
            );
        }
        let context = joined(&hidden1, &hidden2, 0..rows);
        let pooled = self.read_f32(self.text.pooled, 2 * POOLED);
        (context, pooled)
    }

    /// One forward pass of the UNet on the batch of `size`: the inputs `x_in` (pixel-major latents of the batch), the
    /// timestep, the context of the batch `[batch][ctx][2048]` and the add features `[batch][2816]`. It gives the noise
    /// prediction of the batch, and with `probe` the hidden state after each run of `model::UNET_RUNS`.
    pub fn unet_forward(
        &mut self,
        size: UnetSize,
        x_in: &[f32],
        timestep: f32,
        context: &[f32],
        add: &[f32],
        probe: bool,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let batch = size.batch();
        let t_proj: Vec<f32> = (0..batch)
            .flat_map(|_| host::timestep_features(timestep, TIME_PROJ))
            .collect();
        self.write_f32(self.unet.x_in, x_in);
        self.write_f32(self.unet.t_proj, &t_proj);
        self.write_f32(self.unet.ctx, context);
        self.write_f32(self.unet.add_proj, add);
        let runs = model::unet(&self.weights, &self.unet, &size);
        let mut probes = Vec::new();
        for (k, (run, out)) in runs.iter().enumerate() {
            self.run(run);
            if probe {
                let floats = match k {
                    0 => size.rows(0) * LEVELS[0],
                    1 => size.rows(1) * LEVELS[0],
                    2 => size.rows(2) * LEVELS[1],
                    3 | 4 => size.rows(2) * LEVELS[2],
                    5 => size.rows(1) * LEVELS[2],
                    6 => size.rows(0) * LEVELS[1],
                    7 => size.rows(0) * LEVELS[0],
                    _ => size.rows(0) * LATENT,
                };
                probes.push(self.read_f32(*out, floats));
            }
        }
        (self.read_f32(self.unet.eps, size.rows(0) * LATENT), probes)
    }

    /// Run each task of a UNet step of `size` as its own ring, on the inputs that the arena holds, and give the time
    /// of each opcode: its count of tasks and its milliseconds. It is for measurement; a step runs its tasks in runs.
    pub fn profile_unet(&mut self, size: UnetSize) -> BTreeMap<u16, (usize, f64)> {
        let runs = model::unet(&self.weights, &self.unet, &size);
        let mut out: BTreeMap<u16, (usize, f64)> = BTreeMap::new();
        for (run, _) in &runs {
            for task in &run.tasks {
                let single = Run { tasks: vec![*task] };
                let t = std::time::Instant::now();
                self.run(&single);
                let entry = out.entry(task.opcode).or_default();
                entry.0 += 1;
                entry.1 += t.elapsed().as_secs_f64() * 1000.0;
            }
        }
        out
    }

    /// The VAE decode of one latent of `h` by `w` rows (pixel-major, 4 channels): 3 values of each pixel, in [-1, 1]
    /// before the clamp.
    pub fn decode_latent(&mut self, latent: &[f32], h: usize, w: usize) -> Vec<f32> {
        let z: Vec<f32> = latent.iter().map(|v| v / VAE_SCALING_FACTOR).collect();
        self.write_f32(self.vae.z, &z);
        let run = model::vae(&self.weights, &self.vae, h, w);
        self.run(&run);
        self.read_f32(self.vae.image, 3 * h * w * VAE_SCALE * VAE_SCALE)
    }

    /// Check a job against the limits of the engine. The council checks the ranges of spec 11.9a before it.
    fn check(&self, job: &ImageJob) -> Result<(), PassError> {
        job_problem(job).map_or(Ok(()), |m| Err(refused(format!("nothing ran: {m}"))))
    }

    fn fail(&mut self, error: PassError) -> PassError {
        self.last_error = Some(error.message.clone());
        error
    }
}

/// The reason an image job is refused, or `None` when the engine can run it.
pub fn job_problem(job: &ImageJob) -> Option<String> {
    let pixels = job.width as usize * job.height as usize;
    let chunk_ok =
        |c: &ImageChunk| c.ids1.len() == CHUNK_IDS && c.ids2.len() == CHUNK_IDS && c.weights.len() == CHUNK_IDS;
    if job.count == 0 || job.count as usize > MAX_IMAGES {
        Some(format!(
            "the job asks for {} images; the engine makes 1 to {MAX_IMAGES}",
            job.count
        ))
    } else if !job.width.is_multiple_of(64) || !job.height.is_multiple_of(64) || job.width == 0 || job.height == 0 {
        Some(format!(
            "the size {} by {} is not a multiple of 64",
            job.width, job.height
        ))
    } else if pixels > MAX_IMAGE_PIXELS {
        Some(format!(
            "an image of {pixels} pixels is more than the {MAX_IMAGE_PIXELS} of the engine"
        ))
    } else if job.steps == 0 {
        Some("the job has 0 steps".to_owned())
    } else if !(2..=3).contains(&job.clip_skip) {
        Some(format!("clip_skip {} is not 2 or 3", job.clip_skip))
    } else if !job.cfg.is_finite() {
        Some("the CFG scale is not a finite number".to_owned())
    } else if job.spacing == ImageSpacing::Trailing
        && !matches!(job.sampler, ImageSampler::Euler | ImageSampler::EulerAncestral)
    {
        Some("trailing spacing needs euler or euler a; the DPM++ samplers take their sigmas".to_owned())
    } else if job.prompt.is_empty()
        || job.prompt.len() > MAX_CHUNKS
        || job.negative.len() > MAX_CHUNKS
        || !job.prompt.iter().chain(&job.negative).all(chunk_ok)
    {
        Some(format!(
            "a prompt has 1 to {MAX_CHUNKS} chunks of {CHUNK_IDS} ids and weights"
        ))
    } else {
        None
    }
}

impl Image for SdxlEngine {
    fn image_max_pixels(&self) -> u64 {
        (MAX_IMAGES * MAX_IMAGE_PIXELS) as u64
    }

    fn image_begin(&mut self, slot: SlotId, job: &ImageJob) -> Result<(), PassError> {
        if slot != 0 {
            return Err(self.fail(refused(format!("slot {slot} does not exist; the engine has slot 0"))));
        }
        if let Err(error) = self.check(job) {
            return Err(self.fail(error));
        }
        if let Err(message) = self.attach_loras(job) {
            return Err(self.fail(refused(message)));
        }
        let negative = if job.negative.is_empty() {
            vec![host::empty_chunk()]
        } else {
            job.negative.clone()
        };
        let (context, pooled) = self.encode(&job.prompt, &negative, job.clip_skip as usize);
        let images = job.count as usize;
        let (h, w) = (job.height as usize / VAE_SCALE, job.width as usize / VAE_SCALE);
        let ctx = context.len() / (2 * CONTEXT);
        let size = UnetSize { images, h, w, ctx };
        // The batch: each image with the negative prompt, then each image with the prompt.
        let side = |s: usize| &context[s * ctx * CONTEXT..(s + 1) * ctx * CONTEXT];
        let batch_ctx: Vec<f32> = (0..2 * images).flat_map(|b| side(b / images).iter().copied()).collect();
        let ids = host::time_ids(job.width as usize, job.height as usize);
        let add: Vec<f32> = (0..2 * images)
            .flat_map(|b| host::add_features(&pooled[(b / images) * POOLED..(b / images + 1) * POOLED], &ids))
            .collect();
        self.write_f32(self.unet.ctx, &batch_ctx);
        self.write_f32(self.unet.add_proj, &add);
        let schedule = Schedule::new(job.sampler, job.spacing, job.steps as usize);
        let latents = (0..images)
            .map(|i| {
                noise::normal(job.seed.wrapping_add(i as u64), 0, 0, Purpose::Start, h * w * LATENT)
                    .into_iter()
                    .map(|v| v * schedule.init_noise_sigma())
                    .collect()
            })
            .collect();
        self.job = Some(Job {
            job: job.clone(),
            schedule,
            size,
            latents,
            history: vec![History::default(); images],
            step: 0,
        });
        Ok(())
    }

    fn image_step(&mut self, slot: SlotId) -> Result<ImageStepOut, PassError> {
        let Some(mut job) = self.job.take().filter(|_| slot == 0) else {
            return Err(self.fail(refused(format!(
                "slot {slot} holds no image job; call image_begin first"
            ))));
        };
        let i = job.step;
        if i >= job.schedule.timesteps.len() {
            self.job = Some(job);
            return Err(self.fail(refused("the job ran all its steps; call image_decode".to_owned())));
        }
        let (images, size) = (job.size.images, job.size);
        let scale = job.schedule.input_scale(i);
        let x_in: Vec<f32> = (0..2 * images)
            .flat_map(|b| job.latents[b % images].iter().map(move |v| v * scale))
            .collect();
        let context = self.read_f32(self.unet.ctx, 2 * images * size.ctx * CONTEXT);
        let add = self.read_f32(self.unet.add_proj, 2 * images * ADD_IN);
        let (eps, _) = self.unet_forward(size, &x_in, job.schedule.timesteps[i], &context, &add, false);
        let n = size.h * size.w * LATENT;
        let cfg = job.job.cfg;
        for img in 0..images {
            let (neg, pos) = (
                &eps[img * n..(img + 1) * n],
                &eps[(images + img) * n..(images + img + 1) * n],
            );
            let guided: Vec<f32> = neg.iter().zip(pos).map(|(u, c)| u + cfg * (c - u)).collect();
            let draw = job
                .schedule
                .draws(i)
                .then(|| noise::normal(job.job.seed.wrapping_add(img as u64), 0, i as u32, Purpose::Step, n));
            sampler::step(
                &job.schedule,
                i,
                &mut job.latents[img],
                &guided,
                draw.as_deref(),
                &mut job.history[img],
            );
        }
        job.step += 1;
        let previews = if job.job.previews {
            job.latents.iter().map(|l| host::preview(l, size.w, size.h)).collect()
        } else {
            Vec::new()
        };
        let out = ImageStepOut {
            step: job.step as u32,
            steps: job.schedule.timesteps.len() as u32,
            previews,
        };
        self.job = Some(job);
        Ok(out)
    }

    fn image_decode(&mut self, slot: SlotId) -> Result<Vec<RgbImage>, PassError> {
        let Some(job) = self.job.take().filter(|_| slot == 0) else {
            return Err(self.fail(refused(format!(
                "slot {slot} holds no image job; call image_begin first"
            ))));
        };
        if job.step < job.schedule.timesteps.len() {
            let message = format!(
                "the job ran {} of its {} steps; run them first",
                job.step,
                job.schedule.timesteps.len()
            );
            self.job = Some(job);
            return Err(self.fail(refused(message)));
        }
        let (h, w) = (job.size.h, job.size.w);
        // The VAE decode has no LoRA, so the LoRA files of the job go here, before the decode.
        self.release_loras();
        let images = job
            .latents
            .iter()
            .map(|latent| RgbImage {
                width: (w * VAE_SCALE) as u32,
                height: (h * VAE_SCALE) as u32,
                rgb: host::rgb_bytes(&self.decode_latent(latent, h, w)),
            })
            .collect();
        self.job = Some(job);
        Ok(images)
    }

    fn image_clear(&mut self, _slot: SlotId) {
        self.job = None;
        self.release_loras();
    }
}

impl Engine for SdxlEngine {
    fn identity(&self) -> Identity {
        self.identity.clone()
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// An image engine takes no pass (engine-abi "Image").
    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        Err(self.fail(refused(
            "an image engine takes no pass; use image_begin, image_step and image_decode".to_owned(),
        )))
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    fn image(&mut self) -> Option<&mut dyn Image> {
        Some(self)
    }
}
