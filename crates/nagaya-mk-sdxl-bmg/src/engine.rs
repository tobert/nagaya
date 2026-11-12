//! The SDXL engine on its card: the `Image` capability of engine-abi "Image" over the graphs of `model`.

use std::collections::BTreeMap;

use nagaya_abi::{
    DecodedImage, Engine, Identity, Image, ImageChunk, ImageJob, ImagePrompt, ImageStepOut, Limits, PassError,
    PassErrorKind, PassOut, Segment, SlotId, Tap,
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
use crate::sampler::{self, History, Schedule, begin_index, image_start};
use crate::spec::*;
use crate::weights::{Adapter, Weights, f32_bytes, f32_values, sources};
use std::collections::HashMap;

pub use crate::job::job_problem;
pub use crate::spec::MODEL;

/// A job in the slot.
struct Job {
    job: ImageJob,
    schedule: Schedule,
    size: UnetSize,
    /// The latent of each image, pixel-major: rows of 4 channels.
    latents: Vec<Vec<f32>>,
    history: Vec<History>,
    /// The index of the next step of the schedule. It starts at `begin`.
    step: usize,
    /// The index of the first step that runs: 0 for text to image, and `steps - n` for a job with an image (spec 11.9b).
    begin: usize,
    /// The text context of the batch, `[2 * images][ctx][2048]`, and its add features, `[2 * images][2816]`. The host
    /// keeps them, because the VAE buffers share the arena with the UNet buffers, and the encode quantum overwrites them.
    context: Vec<f32>,
    add: Vec<f32>,
    /// The image latent of each image of a job with `init`, after the encode quantum: rows of 4 channels, already
    /// times the scaling factor (spec 11.9b). `None` before the encode and for text to image.
    image_latents: Option<Vec<Vec<f32>>>,
    /// The latent mask of an inpainting job: 1 repaints a latent pixel, and 0 keeps it (spec 11.9b, "The mask").
    /// `None` for a job with no mask.
    mask: Option<Vec<u8>>,
    /// The start noise of each image of an inpainting job with a 4-channel UNet, after the encode quantum. Each `known`
    /// of the blend reuses it (spec 11.9b, "The blend"). Empty for any other job.
    start_noise: Vec<Vec<f32>>,
    /// The masked image latent of each image of a job of a 9-channel UNet, after the encode quantum: rows of 4
    /// channels, times the scaling factor (spec 11.9b, "The 9-channel UNet"). `None` for a 4-channel UNet.
    masked_latents: Option<Vec<Vec<f32>>>,
    /// True after the encode quantum of a job with `init`. A 9-channel UNet at a strength of 1.0 encodes no image, so
    /// `image_latents` cannot tell.
    encoded: bool,
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
/// The phases of an open, written to stderr when `NAGAYA_OPEN_TIMES` is `1`. Each line has the wall clock in ms since
/// the Unix epoch, so it lines up with the work of another card.
struct Phases {
    on: bool,
    pci: String,
    last: std::time::Instant,
}

impl Phases {
    fn new(pci: &str) -> Self {
        let phases = Self {
            on: std::env::var("NAGAYA_OPEN_TIMES").is_ok_and(|value| value == "1"),
            pci: pci.to_owned(),
            last: std::time::Instant::now(),
        };
        phases.write("open starts", 0);
        phases
    }

    /// Write the end of a phase and its time.
    fn mark(&mut self, phase: &str) {
        let ms = self.last.elapsed().as_millis();
        self.last = std::time::Instant::now();
        self.write(phase, ms);
    }

    fn write(&self, phase: &str, ms: u128) {
        if self.on {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.as_millis());
            eprintln!("sdxl open {} at {now} ms: {phase} took {ms} ms", self.pci);
        }
    }
}

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
        let mut phases = Phases::new(pci);
        let dir = config.model_dir.clone();
        let mut tokenizer = Vec::new();
        for name in ["tokenizer/vocab.json", "tokenizer/merges.txt"] {
            let path = dir.join(name);
            tokenizer.extend(std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?);
        }
        let tokenizer_hash = format!("blake3:{}", blake3::hash(&tokenizer).to_hex());
        phases.mark("tokenizer");
        let checkpoint_digest = checkpoint_digest(&config)?;
        phases.mark("checkpoint digest");
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
        phases.mark("level zero init");

        let mut device = Device::new(card);
        phases.mark("staging buffer");
        let weights = Weights::load(
            &config,
            &mut OnCard {
                device: &mut device,
                card,
            },
        )?;
        phases.mark(&format!("weights: {}", device.times()));
        // The plan checked the graphs at the admission; this checks them again for a file that changed since.
        model::check(&weights)?;
        let in_channels = model::unet_in_channels(&weights)?;
        phases.mark("graph check");
        let weights_bytes = device.bytes as u64;
        let arena = arena_bytes(in_channels);
        let at = device.zeros(card, arena as usize);
        phases.mark("arena");
        let mut carve = Carve::new(at, arena);
        let text = TextBufs::carve(&mut carve, 2 * MAX_CHUNKS);
        let (mut unet_carve, mut vae_carve) = (carve, carve);
        let unet = UnetBufs::carve(
            &mut unet_carve,
            MAX_IMAGES,
            MAX_LATENT_ROWS,
            MAX_CHUNKS * CHUNK_IDS,
            in_channels,
        );
        let vae = VaeBufs::carve(&mut vae_carve, MAX_IMAGE_PIXELS);
        let ring = Ring::new(card, 1);
        phases.mark("ring");

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
            build: std::collections::BTreeMap::new(),
        };
        Ok(Self {
            limits: limits(weights_bytes, in_channels),
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

    /// The input channels of the UNet: 4, or 9 for inpainting (spec 11.9b).
    pub fn in_channels(&self) -> usize {
        self.unet.in_channels
    }

    /// One forward pass of the UNet on the batch of `size`: the inputs `x_in` (pixel-major rows of the batch, with
    /// `in_channels` values each), the
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
        assert_eq!(
            x_in.len(),
            size.rows(0) * self.unet.in_channels,
            "the UNet input has {} values for each latent row",
            self.unet.in_channels
        );
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

    /// The VAE encoder and `quant_conv` on one image of `h` by `w` pixels: the input values of
    /// `nagaya_text_sdxl::inputs::unit`, pixel-major. It gives the 8 moments of each latent pixel, pixel-major: the mean of 4
    /// channels, then their log variance (spec 11.9b).
    pub fn encode_image(&mut self, values: &[f32], h: usize, w: usize) -> Vec<f32> {
        assert_eq!(values.len(), 3 * h * w, "the encoder takes 3 values for each pixel");
        self.write_f32(self.vae.image, values);
        let run = model::vae_encoder(&self.weights, &self.vae, h, w);
        self.run(&run);
        self.read_f32(self.vae.moments, 2 * LATENT * (h / VAE_SCALE) * (w / VAE_SCALE))
    }

    /// The image latent of each image of the job in slot 0, after its encode quantum (spec 11.9b), or `None`. A
    /// 9-channel UNet at a strength of 1.0 encodes no image, so it gives `None`.
    pub fn image_latents(&self) -> Option<&[Vec<f32>]> {
        self.job.as_ref()?.image_latents.as_deref()
    }

    /// The masked image latent of each image of the job in slot 0, after its encode quantum, or `None`. Only a
    /// 9-channel UNet has them (spec 11.9b, "The 9-channel UNet").
    pub fn masked_latents(&self) -> Option<&[Vec<f32>]> {
        self.job.as_ref()?.masked_latents.as_deref()
    }

    /// The latents of `images` images from the encoder moments of one input: image `i` draws its own `z` from the seed
    /// plus `i`, with `purpose` (spec 11.9b, "The image latent").
    fn draw_latents(moments: &[f32], seed: u64, images: usize, purpose: Purpose) -> Vec<Vec<f32>> {
        (0..images)
            .map(|i| {
                host::image_latent(
                    moments,
                    &noise::normal(seed.wrapping_add(i as u64), 0, 0, purpose, moments.len() / 2),
                )
            })
            .collect()
    }

    /// The encode quantum of a job with `init` (spec 11.9b, "The VAE encode quantum"). The encoder runs once on the
    /// image, and each image draws its own latent. A 9-channel UNet also encodes the masked image, with the purpose
    /// `encode masked`. At a strength of 1.0 it encodes no image, because its start is the noise alone (inpaint.py:805).
    fn encode_quantum(&mut self, job: &mut Job) {
        let init = job
            .job
            .init
            .clone()
            .expect("the encode quantum runs for a job with an image");
        let (h, w) = (job.job.height as usize, job.job.width as usize);
        let (seed, images) = (job.job.seed, job.size.images);
        let n = LATENT * (h / VAE_SCALE) * (w / VAE_SCALE);
        let starts: Vec<Vec<f32>> = (0..images)
            .map(|i| noise::normal(seed.wrapping_add(i as u64), 0, 0, Purpose::Start, n))
            .collect();
        let nine = self.unet.in_channels == INPAINT_CHANNELS;
        let inpaint = job.mask.is_some();
        if nine && init.strength == 1.0 {
            let sigma = job.schedule.init_noise_sigma();
            job.latents = starts
                .iter()
                .map(|start| start.iter().map(|v| v * sigma).collect())
                .collect();
        } else {
            let moments = self.encode_image(&nagaya_text_sdxl::inputs::unit(&init.pixels), h, w);
            let latents = Self::draw_latents(&moments, seed, images, Purpose::Encode);
            job.latents = latents
                .iter()
                .zip(&starts)
                .map(|(latent, start)| image_start(&job.schedule, job.begin, latent, start, inpaint, init.strength))
                .collect();
            job.image_latents = Some(latents);
        }
        if nine {
            let blurred = init
                .mask
                .as_ref()
                .expect("job_problem refused a 9-channel job without a mask");
            let values = nagaya_text_sdxl::inputs::masked(&init.pixels, &nagaya_text_sdxl::inputs::binary(blurred));
            let moments = self.encode_image(&values, h, w);
            job.masked_latents = Some(Self::draw_latents(&moments, seed, images, Purpose::EncodeMasked));
        } else if inpaint {
            job.start_noise = starts;
        }
        job.encoded = true;
    }

    /// The latent of each image of the job in slot 0, or `None`. It is the latent after the last step that ran.
    pub fn latents(&self) -> Option<&[Vec<f32>]> {
        Some(&self.job.as_ref()?.latents)
    }

    /// Check a job against the limits of the engine. The council checks the ranges of spec 11.9a before it.
    fn check(&self, job: &ImageJob) -> Result<(), PassError> {
        job_problem(job, self.unet.in_channels).map_or(Ok(()), |m| Err(refused(format!("nothing ran: {m}"))))
    }

    fn fail(&mut self, error: PassError) -> PassError {
        self.last_error = Some(error.message.clone());
        error
    }
}

impl Image for SdxlEngine {
    fn image_max_pixels(&self) -> u64 {
        (MAX_IMAGES * MAX_IMAGE_PIXELS) as u64
    }

    fn image_begin(&mut self, slot: SlotId, job: &ImageJob) -> Result<Option<Vec<u8>>, PassError> {
        if slot != 0 {
            return Err(self.fail(refused(format!("slot {slot} does not exist; the engine has slot 0"))));
        }
        if let Err(error) = self.check(job) {
            return Err(self.fail(error));
        }
        if let Err(message) = self.attach_loras(job) {
            return Err(self.fail(refused(message)));
        }
        let ImagePrompt::Chunks {
            prompt,
            negative,
            clip_skip,
        } = &job.prompt
        else {
            unreachable!("check refused the tokens form");
        };
        let negative = if negative.is_empty() {
            vec![host::empty_chunk()]
        } else {
            negative.clone()
        };
        let (context, pooled) = self.encode(prompt, &negative, *clip_skip as usize);
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
        let schedule = Schedule::new(job.sampler, job.spacing, job.steps as usize);
        let mask = job.init.as_ref().and_then(|init| init.mask.as_ref()).map(|blurred| {
            nagaya_text_sdxl::inputs::latent_mask(&nagaya_text_sdxl::inputs::binary(blurred), job.width, job.height)
        });
        // A job with an image gets its start latents from the encode quantum, its first step.
        let (begin, latents) = match &job.init {
            Some(init) => (begin_index(job.steps, init.strength), Vec::new()),
            None => (
                0,
                (0..images)
                    .map(|i| {
                        noise::normal(job.seed.wrapping_add(i as u64), 0, 0, Purpose::Start, h * w * LATENT)
                            .into_iter()
                            .map(|v| v * schedule.init_noise_sigma())
                            .collect()
                    })
                    .collect(),
            ),
        };
        self.job = Some(Job {
            job: job.clone(),
            schedule,
            size,
            latents,
            history: vec![History::default(); images],
            step: begin,
            begin,
            context: batch_ctx,
            add,
            image_latents: None,
            mask,
            start_noise: Vec::new(),
            masked_latents: None,
            encoded: false,
        });
        Ok(None)
    }

    fn image_step(&mut self, slot: SlotId) -> Result<ImageStepOut, PassError> {
        let Some(mut job) = self.job.take().filter(|_| slot == 0) else {
            return Err(self.fail(refused(format!(
                "slot {slot} holds no image job; call image_begin first"
            ))));
        };
        if job.job.init.is_some() && !job.encoded {
            // The first step of a job with an image is the encode: step 0 of the n steps that run, with no previews
            // (engine-abi "Image inputs").
            self.encode_quantum(&mut job);
            let out = ImageStepOut {
                step: 0,
                steps: (job.schedule.timesteps.len() - job.begin) as u32,
                previews: Vec::new(),
            };
            self.job = Some(job);
            return Ok(out);
        }
        let i = job.step;
        if i >= job.schedule.timesteps.len() {
            self.job = Some(job);
            return Err(self.fail(refused("the job ran all its steps; call image_decode".to_owned())));
        }
        let (images, size) = (job.size.images, job.size);
        let scale = job.schedule.input_scale(i);
        // A 9-channel UNet also takes the latent mask and the masked latent of the image, in both halves of CFG.
        let x_in: Vec<f32> = (0..2 * images)
            .flat_map(|b| {
                let scaled: Vec<f32> = job.latents[b % images].iter().map(|v| v * scale).collect();
                match (&job.masked_latents, &job.mask) {
                    (Some(masked), Some(mask)) => host::inpaint_input(&scaled, mask, &masked[b % images]),
                    (None, _) => scaled,
                    (Some(_), None) => unreachable!("job_problem refused a 9-channel job without a mask"),
                }
            })
            .collect();
        let (eps, _) = self.unet_forward(size, &x_in, job.schedule.timesteps[i], &job.context, &job.add, false);
        let n = size.h * size.w * LATENT;
        let cfg = job.job.cfg;
        for img in 0..images {
            let (neg, pos) = (
                &eps[img * n..(img + 1) * n],
                &eps[(images + img) * n..(images + img + 1) * n],
            );
            let guided: Vec<f32> = neg.iter().zip(pos).map(|(u, c)| u + cfg * (c - u)).collect();
            // The key of a step draw is the index of the step in the whole schedule, also for a job with an image.
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
            // An inpainting job of a 4-channel UNet blends the kept latent pixels back after each step, before the
            // preview. A 9-channel UNet does no blend (inpaint.py:1648).
            if let (Some(mask), None) = (&job.mask, &job.masked_latents) {
                let image_latents = job.image_latents.as_ref().expect("the encode quantum ran first");
                let known = job.schedule.known(i, &image_latents[img], &job.start_noise[img]);
                sampler::blend(&mut job.latents[img], &known, mask);
            }
        }
        job.step += 1;
        let previews = if job.job.previews {
            job.latents.iter().map(|l| host::preview(l, size.w, size.h)).collect()
        } else {
            Vec::new()
        };
        // The steps of `progress` count only the steps that run (spec 11.9b).
        let out = ImageStepOut {
            step: (job.step - job.begin) as u32,
            steps: (job.schedule.timesteps.len() - job.begin) as u32,
            previews,
        };
        self.job = Some(job);
        Ok(out)
    }

    fn image_decode(&mut self, slot: SlotId) -> Result<Vec<DecodedImage>, PassError> {
        let Some(job) = self.job.take().filter(|_| slot == 0) else {
            return Err(self.fail(refused(format!(
                "slot {slot} holds no image job; call image_begin first"
            ))));
        };
        if job.step < job.schedule.timesteps.len() {
            let message = format!(
                "the job ran {} of its {} steps; run them first",
                job.step - job.begin,
                job.schedule.timesteps.len() - job.begin
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
            .map(|latent| DecodedImage {
                width: (w * VAE_SCALE) as u32,
                height: (h * VAE_SCALE) as u32,
                channels: 3,
                pixels: host::rgb_bytes(&self.decode_latent(latent, h, w)),
            })
            .collect();
        self.job = Some(job);
        Ok(images)
    }

    fn image_clear(&mut self, _slot: SlotId) {
        self.job = None;
        self.release_loras();
    }

    /// A 9-channel UNet refuses a job without an image and a mask (engine-abi "Image inputs").
    fn image_needs_mask(&self) -> bool {
        self.unet.in_channels == INPAINT_CHANNELS
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
