//! The limits of the engine for a config, from `config.json` and the tensor headers. The plan reads no weight bytes
//! (spec 2.4), so it runs with no card.

use nagaya_abi::{
    Capability, ChainLimits, Determinism, Inputs, Kind, Lifetime, Limits, MandatoryTerm, MemoryLimits, SlotLimit, Tier,
};
use nagaya_mk_bmg::protocol::MAX_TASKS_PER_CARD;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;

use crate::model::{Slot, Work, bytes_of};
use crate::spec::*;
use crate::weights::{Count, load};

/// The longest quantum in milliseconds (D8): a budget for `frames_per_quantum` frames and their codec decode. It is
/// not measured yet.
pub const MAX_QUANTUM_MS: u32 = 2000;

/// The floats of the scratch of the ring. One-token attention keeps its chunks there.
pub const SCRATCH_FLOATS: usize = 8 << 20;

/// Bytes of host memory that one upload copy goes through.
pub const STAGING_BYTES: usize = 64 << 20;

impl Config {
    /// Check the numbers of the config. A zero or a quantum larger than a request is an error.
    pub fn check(&self) -> Result<(), String> {
        for (name, value) in [
            ("slots", self.slots),
            ("max_frames", self.max_frames),
            ("max_prompt", self.max_prompt),
            ("frames_per_quantum", self.frames_per_quantum),
        ] {
            if value == 0 {
                return Err(format!("the speech engine's {name} is 0; give a positive number"));
            }
        }
        if self.frames_per_quantum > self.max_frames {
            return Err(format!(
                "frames_per_quantum {} is more than max_frames {}; give a quantum that fits a request",
                self.frames_per_quantum, self.max_frames
            ));
        }
        Ok(())
    }
}

fn term(name: &str, tier: Tier, bytes: u64) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier,
        bytes,
        lifetime: Lifetime::Engine,
        kinds: vec![Kind::Prefill, Kind::Decode],
    }
}

/// The device bytes of the weights of the checkpoint in `config.model_dir`, from the tensor headers.
pub fn weights_bytes(config: &Config) -> Result<u64, String> {
    let model = MappedSafeTensors::open(&config.model_dir.join("model.safetensors"))?;
    let codec = MappedSafeTensors::open(&config.model_dir.join("speech_tokenizer/model.safetensors"))?;
    let mut count = Count::default();
    load(&mut count, &model.view()?, &codec.view()?)?;
    Ok(count.bytes as u64)
}

/// The device bytes of the ring and of the cluster on the card.
fn ring_bytes() -> u64 {
    let scoreboard = (MAX_TASKS_PER_CARD + 1) * 4;
    // `L0Cluster::init` takes the wake buffer and the device copy of the ring.
    let cluster = 128 + MAX_TASKS_PER_CARD * 64;
    (SCRATCH_FLOATS * 4 + scoreboard + cluster) as u64
}

/// The limits for a config, with the device bytes of the weights.
pub fn limits(config: &Config, weights: u64) -> Limits {
    let tier = Tier::Device(0);
    let slot = bytes_of(|z| Slot::new(z, config)) as u64;
    let work = bytes_of(|z| Work::new(z, config)) as u64;
    let mandatory = vec![
        term("weights", tier, weights),
        term("slots", tier, slot * config.slots as u64),
        term("work", tier, work + ring_bytes()),
        // The staging buffer of uploads, and the descriptors and statistics of the ring.
        term(
            "pinned",
            Tier::Host,
            (STAGING_BYTES + MAX_TASKS_PER_CARD * 64 + 32 * 64) as u64,
        ),
    ];
    Limits {
        // A drop frees the weights, the slots and the card's context.
        capabilities: vec![Capability::Speech, Capability::Close],
        parts: Vec::new(),
        inputs: Inputs::Tokens,
        devices: vec![0],
        stages: Vec::new(),
        max_segments: 0,
        max_pass_inputs: 0,
        max_logit_rows: 0,
        slots: vec![SlotLimit {
            bytes: slot,
            count: config.slots,
            tokens: config.max_positions() as u64,
        }],
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory,
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
        feed: None,
    }
}

/// The limits of a config, from `config.json` and the tensor headers.
pub fn plan(config: &Config) -> Result<Limits, String> {
    config.check()?;
    check_config(&config.model_dir)?;
    Ok(limits(config, weights_bytes(config)?))
}
