//! The Beat This! engine on one gfx1151 card (spec 11.19; engine-abi "Beats").

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

use nagaya_abi::{Beats, Engine, Identity, Limits, PassError, PassErrorKind, PassOut, Segment, Tap};
use nagaya_family_beatthis::chunks::CHUNK;
use nagaya_family_beatthis::mel::MELS;
use nagaya_family_beatthis::plan::{self, Config, Plan};
use nagaya_family_beatthis::weights::{self, Tensors};
use sha2::{Digest, Sha256};

use crate::gpu::Gpu;

/// The engine: the prepared weights on the card, and a plan for each chunk length that it met.
pub struct BeatThisGpuEngine {
    path: PathBuf,
    config: Config,
    gpu: Gpu,
    plans: HashMap<usize, Plan>,
    digest: String,
    last_error: Option<String>,
}

pub use nagaya_family_beatthis::canary::OpenFailure;

impl BeatThisGpuEngine {
    /// Read and prepare the checkpoint at `path`, upload it to the current card, and check the canary against its
    /// record (`crate::canary`). A checkpoint with no record, or other bits, does not open.
    pub fn open(path: &Path) -> Result<Self, OpenFailure> {
        let mut engine = Self::open_unchecked(path).map_err(OpenFailure::Other)?;
        let measured = engine.measure_canary().map_err(OpenFailure::Other)?;
        nagaya_family_beatthis::canary::verify(crate::canary::TABLE, crate::canary::TARGET, &engine.digest, &measured)
            .map_err(OpenFailure::Canary)?;
        Ok(engine)
    }

    /// The SHA-256 of the weights file, as the key of the canary records.
    pub fn checkpoint_digest(&self) -> &str {
        &self.digest
    }

    /// Run the canary chunks through the engine and give their hash.
    pub fn measure_canary(&mut self) -> Result<String, String> {
        use nagaya_family_beatthis::canary;
        let mut outs = Vec::with_capacity(canary::FRAMES.len());
        for (i, &frames) in canary::FRAMES.iter().enumerate() {
            let out = Beats::beats(self, &canary::input(i), frames as u32).map_err(|e| e.message)?;
            outs.push(out);
        }
        Ok(canary::hash(&outs))
    }

    /// Read, prepare and upload the checkpoint at `path`, with no canary. Only the card test that measures a record
    /// uses it; a server opens with `open`.
    pub fn open_unchecked(path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| format!("cannot read the weights {}: {e}", path.display()))?;
        let hp = Tensors::hyper_parameters(&bytes).map_err(|e| e.to_string())?;
        let config = Config::from_json(&hp).map_err(|e| e.to_string())?;
        let tensors = Tensors::parse(&bytes).map_err(|e| e.to_string())?;
        let prepared = weights::prepare(&plan::build(&config, CHUNK).weights, &tensors).map_err(|e| e.to_string())?;
        let gpu = Gpu::new(&prepared)?;
        let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
        Ok(Self {
            path: path.to_owned(),
            config,
            gpu,
            plans: HashMap::new(),
            digest,
            last_error: None,
        })
    }

    fn refuse(&mut self, message: String) -> PassError {
        self.last_error = Some(message.clone());
        PassError {
            kind: PassErrorKind::Refused,
            message,
        }
    }
}

impl Beats for BeatThisGpuEngine {
    fn beats_max_frames(&self) -> u32 {
        CHUNK as u32
    }

    fn beats(&mut self, rows: &[f32], frames: u32) -> Result<Vec<f32>, PassError> {
        let t = frames as usize;
        if t == 0 || t > CHUNK {
            return Err(self.refuse(format!(
                "nothing ran: a chunk has {t} frames, and the engine takes 1 to {CHUNK}"
            )));
        }
        if rows.len() != t * MELS {
            return Err(self.refuse(format!(
                "nothing ran: {t} frames need {} log-mel values, and the call has {}",
                t * MELS,
                rows.len()
            )));
        }
        let config = self.config;
        let p = self.plans.entry(t).or_insert_with(|| plan::build(&config, t));
        match self.gpu.run(p, rows, false, |_, _| {}) {
            Ok(out) => Ok(out),
            Err(message) => {
                // A launch that failed leaves the card in an unknown state, so the pass is not refused but failed.
                self.last_error = Some(message.clone());
                Err(PassError {
                    kind: PassErrorKind::Ran(nagaya_abi::PassStatus::Poison),
                    message,
                })
            }
        }
    }
}

impl Engine for BeatThisGpuEngine {
    fn identity(&self) -> Identity {
        let mut numbers = BTreeMap::new();
        for (k, v) in [
            ("stem_dim", self.config.stem_dim),
            ("transformer_dim", self.config.transformer_dim),
            ("n_layers", self.config.n_layers),
            ("head_dim", self.config.head_dim),
            ("ff_mult", self.config.ff_mult),
        ] {
            numbers.insert(k.to_owned(), v.to_string());
        }
        // The canary holds the bits (spec 8.1), so the build facts stay out of the digest (12.1).
        let mut build = BTreeMap::new();
        build.insert(
            "source_sha256".to_owned(),
            env!("NAGAYA_BEATTHIS_GFX1151_SOURCE_SHA256").to_owned(),
        );
        numbers.insert("target".to_owned(), "gfx1151".to_owned());
        Identity {
            model: self
                .path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            family: "beatthis".to_owned(),
            checkpoint_digest: self.digest.clone(),
            tokenizer_hash: String::new(),
            template: "logmel:22050:1024:441:128".to_owned(),
            numbers,
            plan_table: "nagaya-family-beatthis::plan".to_owned(),
            bit_settings: Vec::new(),
            canary_hash: nagaya_family_beatthis::canary::record(crate::canary::TABLE, &self.digest)
                .unwrap_or_default()
                .to_owned(),
            build,
        }
    }

    fn limits(&self) -> Limits {
        crate::limits(&self.config)
    }

    /// A beats engine takes no pass (engine-abi "Beats").
    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        Err(self.refuse("a beats engine takes no pass; use beats".to_owned()))
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    fn beats(&mut self) -> Option<&mut dyn Beats> {
        Some(self)
    }
}
