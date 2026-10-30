//! The limits of the engine for a config, from `config.json` and the tensor headers. The plan reads no weight bytes
//! (spec 2.4), so it runs with no card.

use nagaya_abi::{
    Capability, ChainLimits, Determinism, Inputs, Kind, Lifetime, Limits, MandatoryTerm, MemoryLimits, Tier,
};
use nagaya_mk_bmg::protocol::MAX_TASKS_PER_CARD;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;

use crate::model::bufs_bytes;
use crate::spec::*;
use crate::weights::{Count, load};

/// The longest quantum in milliseconds (D8): one input of `MAX_TOKENS` tokens. It took 601 ms on a B70 card on
/// 2026-10-30 (`tests/xe_embed.rs`), so the budget keeps a margin of about 3 times.
pub const MAX_QUANTUM_MS: u32 = 2000;

/// Bytes of host memory that one upload copy goes through.
pub const STAGING_BYTES: usize = 64 << 20;

fn term(name: &str, tier: Tier, bytes: u64) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier,
        bytes,
        lifetime: Lifetime::Engine,
        kinds: vec![Kind::Embed],
    }
}

/// The device bytes of the text tower of the checkpoint in `config.model_dir`, from the tensor headers. A checkpoint
/// that lacks a tensor is an error here, before any open.
pub fn weights_bytes(config: &Config) -> Result<u64, String> {
    let map = MappedSafeTensors::open(&config.model_dir.join(WEIGHTS_FILE))?;
    let mut count = Count::default();
    load(&mut count, &map.view()?)?;
    Ok(count.bytes as u64)
}

/// The device bytes of the ring and of the card's buffers: the scoreboard, the scratch, the wake buffer and the device
/// copy of the ring.
fn ring_bytes() -> u64 {
    ((MAX_TASKS_PER_CARD + 1) * 4 + 64 + 128 + MAX_TASKS_PER_CARD * 64) as u64
}

/// The limits for a config, with the device bytes of the weights.
pub fn limits(weights: u64) -> Limits {
    Limits {
        // A drop frees the weights, the buffers and the card's context.
        capabilities: vec![Capability::Embedding, Capability::Close],
        parts: Vec::new(),
        inputs: Inputs::Tokens,
        devices: vec![0],
        stages: Vec::new(),
        max_segments: 0,
        max_pass_inputs: 0,
        max_logit_rows: 0,
        // The engine holds no state: no part, no slot and no context (engine-abi "Embedding").
        slots: Vec::new(),
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory: vec![
                term("weights", Tier::Device(0), weights),
                term(
                    "activations",
                    Tier::Device(0),
                    bufs_bytes(MAX_TOKENS) as u64 + ring_bytes(),
                ),
                term(
                    "pinned",
                    Tier::Host,
                    (STAGING_BYTES + MAX_TASKS_PER_CARD * 64 + 32 * 64) as u64,
                ),
            ],
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
    check_config(&config.model_dir)?;
    Ok(limits(weights_bytes(config)?))
}
