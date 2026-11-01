//! The Qwen-Image 2.1 engine on gfx1151: the `Engine` trait with the image calls (engine-abi "Image", spec 11.9c;
//! milestone 22 step 7).
//!
//! `image_begin` encodes the template ids, or takes the rows of a conditioning hit, and runs the condition rows
//! through the transformer once. Each `image_step` runs one step of each image, and `image_decode` decodes each
//! latent into RGBA. The engine holds one job at a time, in slot 0.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use nagaya_abi::{
    DecodedImage, Engine, Identity, Image, ImageConditioning, ImageJob, ImagePrompt, ImageSampler, ImageSpacing,
    ImageStepOut, Limits, PassError, PassErrorKind, PassOut, Segment, SlotId, Tap,
};
use nagaya_mk_hip::attention::Attention;
use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{Gemms, lineage};
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::plan::PlanTable;

use crate::encoder::Encoder;
use crate::noise::{Purpose, normal};
pub use crate::plan::Config;
use crate::plan::{MAX_IDS, MAX_IMAGES, MAX_PIXELS, MAX_ROWS, ROW_BYTES, limits};
use crate::safetensors::Safetensors;
use crate::sampler;
use crate::transformer::{Conditioned, LATENT, Transformer};
use crate::vae::{CHANNELS, SCALE, VaeDecoder, Z};

/// The model name of the identity.
pub const MODEL: &str = "qwen-image-2.1";
/// The workspace of each GEMM.
const WORKSPACE: u64 = 64 << 20;
/// The template of the family (spec 11.9c). The identity holds it, so a new template is a new recipe.
pub const TEMPLATE: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n\
{}<|im_end|>\n<|im_start|>assistant\n";

/// The BLAKE3 hash of a file, in chunks of 256 MiB on threads, as the SDXL engine hashes its checkpoint.
pub fn file_hash(path: &Path) -> Result<blake3::Hash, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    const CHUNK: u64 = 256 << 20;
    let chunks = size.div_ceil(CHUNK).max(1);
    let hashes: Vec<Result<blake3::Hash, String>> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..chunks)
            .map(|i| {
                let file = &file;
                scope.spawn(move || {
                    let start = i * CHUNK;
                    let mut bytes = vec![0u8; (size - start).min(CHUNK) as usize];
                    file.read_exact_at(&mut bytes, start)
                        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                    Ok(blake3::hash(&bytes))
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|t| t.join().expect("a hash thread panicked"))
            .collect()
    });
    let mut all = blake3::Hasher::new();
    for hash in hashes {
        all.update(hash?.as_bytes());
    }
    Ok(all.finalize())
}

/// The state of the job in slot 0.
struct Job {
    conditioned: Conditioned,
    /// The latent of each image, `[h * w, 64]` in bf16.
    latents: Vec<Vec<u16>>,
    sigmas: Vec<f32>,
    step: usize,
    h: usize,
    w: usize,
}

/// The Qwen-Image 2.1 engine.
pub struct QwenImageEngine {
    gemms: Gemms,
    stream: Stream,
    encoder: Encoder,
    transformer: Transformer,
    vae: VaeDecoder,
    identity: Identity,
    limits: Limits,
    conditioning: ImageConditioning,
    job: Option<Job>,
    last_error: Option<String>,
}

fn refused(message: String) -> PassError {
    PassError {
        kind: PassErrorKind::Refused,
        message,
    }
}

impl QwenImageEngine {
    /// Open the engine: load and check the encoder, the transformer with its LoRA, and the VAE decoder.
    pub fn open(config: &Config) -> Result<QwenImageEngine, String> {
        let started = std::time::Instant::now();
        let stream = Stream::new().map_err(|e| e.to_string())?;
        let lineage = lineage().map_err(|e| e.to_string())?;
        let table = match &config.plan_dir {
            Some(dir) => PlanTable::open(dir, &lineage).map_err(|e| e.to_string())?,
            None => PlanTable::in_memory(&lineage),
        };
        let mut gemms = Gemms::new(table, WORKSPACE).map_err(|e| e.to_string())?;
        let tokenizer_path = config.model.join("processor/tokenizer.json");
        let tokenizer =
            std::fs::read(&tokenizer_path).map_err(|e| format!("cannot read {}: {e}", tokenizer_path.display()))?;
        let tokenizer_hash = {
            use sha2::Digest;
            let digest = sha2::Sha256::digest(&tokenizer);
            format!(
                "sha256:{}",
                digest.iter().map(|b| format!("{b:02x}")).collect::<String>()
            )
        };
        let encoder_hash = file_hash(&config.encoder)?;
        let encoder = Encoder::load(&Safetensors::open(&config.encoder).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let lora_file = match &config.lora {
            Some((path, strength)) => Some((Safetensors::open(path).map_err(|e| e.to_string())?, *strength)),
            None => None,
        };
        let transformer = Transformer::load(
            &config.model.join("transformer"),
            lora_file.as_ref().map(|(f, s)| (f, *s)),
            &mut gemms,
            &stream,
        )
        .map_err(|e| e.to_string())?;
        let vae = VaeDecoder::load(&config.model.join("vae")).map_err(|e| e.to_string())?;
        // The checkpoint digest covers every weight file of the variant, as the SDXL engine's does.
        let mut checkpoint = blake3::Hasher::new();
        checkpoint.update(encoder_hash.as_bytes());
        let transformer_dir = config.model.join("transformer");
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&transformer_dir)
            .map_err(|e| format!("cannot list {}: {e}", transformer_dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        shards.sort();
        shards.push(config.model.join("vae/diffusion_pytorch_model.safetensors"));
        if let Some((path, _)) = &config.lora {
            shards.push(path.clone());
        }
        for shard in &shards {
            checkpoint.update(file_hash(shard)?.as_bytes());
        }
        let mut numbers = BTreeMap::new();
        if let Some((path, strength)) = &config.lora {
            numbers.insert("lora".to_owned(), path.display().to_string());
            numbers.insert("lora_strength".to_owned(), strength.to_string());
        }
        let facts = nagaya_mk_hip::build_facts().expect("a hip build has its build facts");
        numbers.insert("build.sha256".to_owned(), facts.sha256.to_owned());
        numbers.insert("build.compiler".to_owned(), facts.compiler.to_owned());
        numbers.insert("build.hip".to_owned(), facts.hip.to_owned());
        numbers.insert("build.hipblaslt".to_owned(), facts.hipblaslt.to_owned());
        numbers.insert("canary".to_owned(), "not run".to_owned());
        // The encoder digest (spec 6.16): the text tower, the tokenizer, the template, the cut, the build and the
        // lineage of the plan table. Not the transformer or its LoRA.
        let mut digest = blake3::Hasher::new_derive_key("nagaya encoder digest v1");
        for part in [
            encoder_hash.to_hex().as_str(),
            tokenizer_hash.as_str(),
            TEMPLATE,
            "cut 14",
            facts.sha256,
            facts.hipblaslt,
            lineage.as_str(),
        ] {
            digest.update(&(part.len() as u64).to_le_bytes());
            digest.update(part.as_bytes());
        }
        let encoder_digest = format!("blake3:{}", digest.finalize().to_hex());
        // The limits count what the plan counts from the headers, so the ledger's terms and the engine's agree.
        let weights = crate::plan::weights_bytes(config)?;
        let identity = Identity {
            model: MODEL.to_owned(),
            checkpoint_digest: format!("blake3:{}", checkpoint.finalize().to_hex()),
            family: "qwenimage21".to_owned(),
            tokenizer_hash,
            template: TEMPLATE.to_owned(),
            numbers,
            plan_table: lineage,
            bit_settings: Vec::new(),
            canary_hash: String::new(),
        };
        eprintln!(
            "qwenimage21: open in {:.1} s, {:.2} GiB of weights",
            started.elapsed().as_secs_f64(),
            weights as f64 / (1u64 << 30) as f64
        );
        Ok(QwenImageEngine {
            gemms,
            stream,
            encoder,
            transformer,
            vae,
            identity,
            limits: limits(weights),
            conditioning: ImageConditioning {
                row_bytes: ROW_BYTES as u32,
                max_rows: MAX_ROWS as u32,
                encoder_digest,
            },
            job: None,
            last_error: None,
        })
    }

    fn fail(&mut self, error: PassError) -> PassError {
        self.last_error = Some(error.message.clone());
        error
    }

    /// The checks of a job that the engine runs (spec 11.9c). The route checked the ranges first.
    fn check(job: &ImageJob) -> Result<(&[u32], usize), String> {
        let ImagePrompt::Tokens { ids, cut } = &job.prompt else {
            return Err("the qwenimage21 engine reads the tokens form of a prompt, not chunks".to_owned());
        };
        let cut = *cut as usize;
        if ids.len() > MAX_IDS || ids.len() <= cut {
            return Err(format!(
                "{} ids with a cut of {cut} make no conditioning of at most {MAX_IDS} ids",
                ids.len()
            ));
        }
        if job.sampler != ImageSampler::FlowEuler || job.spacing != ImageSpacing::Leading {
            return Err("the qwenimage21 engine runs the flow Euler sampler only".to_owned());
        }
        if job.cfg != 1.0 {
            return Err(format!(
                "the qwenimage21 engine runs a cfg of 1.0 only, not {}",
                job.cfg
            ));
        }
        if !job.loras.is_empty() || job.previews {
            return Err("the qwenimage21 engine takes no LoRA of a request and no previews".to_owned());
        }
        let (w, h) = (job.width as usize, job.height as usize);
        if w % 32 != 0 || h % 32 != 0 || !(512..=2048).contains(&w) || !(512..=2048).contains(&h) || w * h > MAX_PIXELS
        {
            return Err(format!("{w} by {h} is not a size of the qwenimage21 engine"));
        }
        if !(1..=MAX_IMAGES as u32).contains(&job.count) || !(1..=150).contains(&job.steps) {
            return Err(format!("{} images of {} steps is out of range", job.count, job.steps));
        }
        Ok((ids, cut))
    }

    /// A job of the oracle's size, which the route refuses: 256 by 256. The oracle gate uses it.
    pub fn begin_any_size(&mut self, job: &ImageJob) -> Result<Option<Vec<u8>>, PassError> {
        self.begin(job)
    }

    fn begin(&mut self, job: &ImageJob) -> Result<Option<Vec<u8>>, PassError> {
        let ImagePrompt::Tokens { ids, cut } = &job.prompt else {
            return Err(self.fail(refused(
                "the qwenimage21 engine reads the tokens form of a prompt".to_owned(),
            )));
        };
        let (ids, cut) = (ids.clone(), *cut as usize);
        let rows = ids.len() - cut;
        let attention_capacity = Encoder::attention_capacity(ids.len()).max(Transformer::attention_capacity(
            rows,
            (job.width as usize / SCALE) * (job.height as usize / SCALE),
        ));
        let attention = Attention::new(attention_capacity).map_err(|e| self.fail(refused(e.to_string())))?;
        let made;
        let conditioning = match &job.conditioning {
            Some(bytes) => {
                if bytes.len() != rows * ROW_BYTES {
                    return Err(self.fail(refused(format!(
                        "a conditioning of {} bytes is not {rows} rows of {ROW_BYTES}",
                        bytes.len()
                    ))));
                }
                made = None;
                DeviceBuffer::from_slice(bytes.as_slice()).map_err(|e| self.fail(refused(e.to_string())))?
            }
            None => {
                let hidden = self
                    .encoder
                    .encode(&mut self.gemms, &attention, &ids, &self.stream)
                    .map_err(|e| self.fail(refused(e.to_string())))?;
                let all: Vec<u8> = hidden.to_vec::<u8>().map_err(|e| self.fail(refused(e.to_string())))?;
                let rows_bytes = all[cut * ROW_BYTES..].to_vec();
                made = Some(rows_bytes.clone());
                DeviceBuffer::from_slice(&rows_bytes).map_err(|e| self.fail(refused(e.to_string())))?
            }
        };
        let conditioned = self
            .transformer
            .begin(&mut self.gemms, &attention, &conditioning, 0, rows, &self.stream)
            .map_err(|e| self.fail(refused(e.to_string())))?;
        let (h, w) = (job.height as usize / SCALE, job.width as usize / SCALE);
        let latents = (0..job.count)
            .map(|i| {
                // The noise fills [64, h, w] in order (spec 11.9c); a token row holds the 64 channels of a pixel.
                let noise = normal(job.seed.wrapping_add(u64::from(i)), 0, 0, Purpose::Start, Z * h * w);
                let mut tokens = vec![0u16; h * w * LATENT];
                for c in 0..Z {
                    for p in 0..h * w {
                        tokens[p * LATENT + c] = bf16::from_f32(noise[c * h * w + p]);
                    }
                }
                tokens
            })
            .collect();
        self.job = Some(Job {
            conditioned,
            latents,
            sigmas: sampler::sigmas(job.steps as usize, h * w),
            step: 0,
            h,
            w,
        });
        Ok(made)
    }

    /// Replace the latent of image `index` of the job, for the oracle gate: diffusers draws its own start noise.
    pub fn set_latent(&mut self, index: usize, latent: &[u16]) {
        let job = self.job.as_mut().expect("a job is in the slot");
        assert_eq!(
            job.latents[index].len(),
            latent.len(),
            "the latent has the size of the job"
        );
        job.latents[index] = latent.to_vec();
    }
}

impl Image for QwenImageEngine {
    fn image_max_pixels(&self) -> u64 {
        (MAX_IMAGES * MAX_PIXELS) as u64
    }

    fn image_conditioning(&self) -> Option<ImageConditioning> {
        Some(self.conditioning.clone())
    }

    fn image_begin(&mut self, slot: SlotId, job: &ImageJob) -> Result<Option<Vec<u8>>, PassError> {
        if slot != 0 {
            return Err(self.fail(refused(format!("the engine has one slot, not {slot}"))));
        }
        if let Err(message) = QwenImageEngine::check(job) {
            return Err(self.fail(refused(message)));
        }
        self.begin(job)
    }

    fn image_step(&mut self, slot: SlotId) -> Result<ImageStepOut, PassError> {
        let Some(mut job) = self.job.take().filter(|_| slot == 0) else {
            return Err(self.fail(refused(format!(
                "slot {slot} holds no image job; call image_begin first"
            ))));
        };
        let steps = job.sigmas.len() - 1;
        if job.step >= steps {
            self.job = Some(job);
            return Err(self.fail(refused("the job ran each step; decode it".to_owned())));
        }
        let (sigma, next) = (job.sigmas[job.step], job.sigmas[job.step + 1]);
        let rows = job.conditioned.rows();
        let attention = Attention::new(Transformer::attention_capacity(rows, job.h * job.w))
            .map_err(|e| self.fail(refused(e.to_string())))?;
        for latent in &mut job.latents {
            let x = DeviceBuffer::from_slice(latent).map_err(|e| self.fail(refused(e.to_string())))?;
            let v = self
                .transformer
                .step(
                    &mut self.gemms,
                    &attention,
                    &job.conditioned,
                    &x,
                    job.h,
                    job.w,
                    sampler::timestep(sigma),
                    &self.stream,
                )
                .map_err(|e| self.fail(refused(e.to_string())))?;
            let v: Vec<u16> = v.to_vec().map_err(|e| self.fail(refused(e.to_string())))?;
            sampler::euler(latent, &v, sigma, next);
        }
        job.step += 1;
        let out = ImageStepOut {
            step: job.step as u32,
            steps: steps as u32,
            previews: Vec::new(),
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
        if job.step + 1 < job.sigmas.len() {
            let message = format!(
                "the job ran {} of its {} steps; run them first",
                job.step,
                job.sigmas.len() - 1
            );
            self.job = Some(job);
            return Err(self.fail(refused(message)));
        }
        let mut images = Vec::new();
        for latent in &job.latents {
            let x = DeviceBuffer::from_slice(latent).map_err(|e| self.fail(refused(e.to_string())))?;
            let pixels = self
                .vae
                .decode(&mut self.gemms, &x, job.h, job.w, &self.stream)
                .map_err(|e| self.fail(refused(e.to_string())))?;
            images.push(DecodedImage {
                width: (job.w * SCALE) as u32,
                height: (job.h * SCALE) as u32,
                channels: CHANNELS as u32,
                pixels,
            });
        }
        self.job = Some(job);
        Ok(images)
    }

    fn image_clear(&mut self, _slot: SlotId) {
        self.job = None;
    }
}

impl Engine for QwenImageEngine {
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
