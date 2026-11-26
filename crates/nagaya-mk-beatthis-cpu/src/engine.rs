//! The Beat This! engine of the CPU group (spec 3.2c, 11.19; engine-abi "Beats").

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

use nagaya_abi::{
    Beats, BitClass, BitSetting, Capability, ChainLimits, Determinism, Engine, Identity, Inputs, Lifetime, Limits,
    MandatoryTerm, MemoryLimits, PassError, PassErrorKind, PassOut, Segment, Tap, Tier,
};
use nagaya_family_beatthis::chunks::CHUNK;
use nagaya_family_beatthis::mel::MELS;
use nagaya_family_beatthis::plan::{self, Config, Plan, Task};
use nagaya_family_beatthis::weights::{self, Tensors};
use sha2::{Digest, Sha256};

/// The bound of one chunk. A chunk of `final0` takes 1.4 s on 16 threads in a build of the tests (spec 11.19).
pub const MAX_QUANTUM_MS: u32 = 10_000;

/// The host bytes of the engine: its prepared weights, the buffers of a plan of one whole chunk, and the q and k
/// that its largest attention rotates. The weights are the same for each chunk length, and a shorter chunk needs
/// fewer bytes.
pub fn host_bytes(config: &Config) -> u64 {
    let p = plan::build(config, CHUNK);
    let weights: usize = p.weights.iter().map(|w| w.prep.len()).sum();
    let buffers: usize = p.buffers.iter().sum();
    let rotated = p
        .tasks
        .iter()
        .map(|t| match *t {
            Task::Attention {
                a, b, heads, head_dim, ..
            } => a * b * 2 * heads * head_dim,
            _ => 0,
        })
        .max()
        .unwrap_or(0);
    4 * (weights + buffers + rotated) as u64
}

/// The limits of the engine for a checkpoint of `config`. It runs in the CPU group, so it names no device.
pub fn limits(config: &Config) -> Limits {
    Limits {
        capabilities: vec![Capability::Beats, Capability::Close],
        parts: Vec::new(),
        inputs: Inputs::Rows,
        devices: Vec::new(),
        stages: Vec::new(),
        max_segments: 0,
        max_pass_inputs: 0,
        max_logit_rows: 0,
        slots: Vec::new(),
        max_quantum_ms: MAX_QUANTUM_MS,
        max_long_quantum_ms: None,
        memory: MemoryLimits {
            mandatory: vec![MandatoryTerm {
                name: "weights-and-scratch".to_owned(),
                tier: Tier::Host,
                bytes: host_bytes(config),
                lifetime: Lifetime::Engine,
                kinds: Vec::new(),
            }],
            elastic: Vec::new(),
        },
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
        segments_exact: false,
        determinism: Determinism::Always,
        bit_dependence: Vec::new(),
        taps: Vec::new(),
        checkpoint_grain: None,
        residual_row_bytes: 0,
        control: None,
        feed: None,
        layer_loops: None,
        cycles: None,
    }
}

/// The config of a converted checkpoint from its header.
pub fn config_of(path: &Path) -> Result<Config, String> {
    let hp = weights::read_hyper_parameters(path).map_err(|e| e.to_string())?;
    Config::from_json(&hp).map_err(|e| e.to_string())
}

/// The engine: the prepared weights of a checkpoint, and a plan for each chunk length that it met.
pub struct BeatThisEngine {
    path: PathBuf,
    config: Config,
    weights: Vec<Vec<f32>>,
    plans: HashMap<usize, Plan>,
    threads: usize,
    digest: String,
    last_error: Option<String>,
}

impl BeatThisEngine {
    /// Read and prepare the checkpoint at `path`, and run with `threads` threads.
    pub fn open(path: &Path, threads: u32) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| format!("cannot read the weights {}: {e}", path.display()))?;
        let hp = Tensors::hyper_parameters(&bytes).map_err(|e| e.to_string())?;
        let config = Config::from_json(&hp).map_err(|e| e.to_string())?;
        let tensors = Tensors::parse(&bytes).map_err(|e| e.to_string())?;
        let weights = weights::prepare(&plan::build(&config, CHUNK).weights, &tensors).map_err(|e| e.to_string())?;
        let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
        // The workers of the kernels start here, on the thread of the open (the group thread), and get its mask.
        crate::kernels::start_workers(threads.max(1) as usize);
        Ok(Self {
            path: path.to_owned(),
            config,
            weights,
            plans: HashMap::new(),
            threads: threads.max(1) as usize,
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

impl Beats for BeatThisEngine {
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
        // A plan of each length has the weights of every other, in the same order (plan::build).
        Ok(crate::run(p, &self.weights, rows, self.threads, |_, _| {}))
    }
}

impl Engine for BeatThisEngine {
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
        // No canary yet (spec 11.19), so the build facts stay in the digest (12.1).
        numbers.insert(
            "source_sha256".to_owned(),
            env!("NAGAYA_BEATTHIS_SOURCE_SHA256").to_owned(),
        );
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
            bit_settings: vec![BitSetting {
                name: "threads".to_owned(),
                value: self.threads.to_string(),
                class: BitClass::Policy,
            }],
            canary_hash: String::new(),
            build: BTreeMap::new(),
        }
    }

    fn limits(&self) -> Limits {
        limits(&self.config)
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
