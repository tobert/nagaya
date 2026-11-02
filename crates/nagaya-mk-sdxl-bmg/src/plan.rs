//! The limits of the engine for a config, from the tensor headers of the checkpoint. The plan reads no weight bytes
//! (spec 2.4), so it runs with no card.

use nagaya_abi::{
    Capability, ChainLimits, Determinism, Inputs, Kind, Lifetime, Limits, MandatoryTerm, MemoryLimits, SlotLimit, Tier,
};
use nagaya_mk_bmg::protocol::MAX_TASKS_PER_CARD;

use crate::model::{Carve, TextBufs, UnetBufs, VaeBufs};
use crate::spec::*;
use crate::weights::{Count, Weights};

/// The longest quantum in milliseconds (D8): one UNet step of 4 images of 1.5 MP, the VAE decode of them, or the VAE
/// encode of their image and masked image. It is not measured yet.
pub const MAX_QUANTUM_MS: u32 = 60_000;

/// Bytes of host memory that one upload copy goes through.
pub const STAGING_BYTES: usize = 64 << 20;

/// The most latent rows of all images of a job: 4 images of 1.5 MP.
pub const MAX_LATENT_ROWS: usize = MAX_IMAGES * MAX_IMAGE_PIXELS / (VAE_SCALE * VAE_SCALE);

/// The bytes of the activation arena: the buffers of the text encoders, then the larger of the buffers of the UNet
/// and of the VAE, which never run at once. The VAE encoder and decoder share the buffers of the VAE.
pub fn arena_bytes() -> u64 {
    let mut carve = Carve::new(0, u64::MAX / 2);
    TextBufs::carve(&mut carve, 2 * MAX_CHUNKS);
    let after_text = carve.used(0);
    let mut unet = carve;
    UnetBufs::carve(&mut unet, MAX_IMAGES, MAX_LATENT_ROWS, MAX_CHUNKS * CHUNK_IDS);
    let mut vae = carve;
    VaeBufs::carve(&mut vae, MAX_IMAGE_PIXELS);
    after_text + (unet.used(0) - after_text).max(vae.used(0) - after_text)
}

/// The device bytes of the weights of the checkpoint of `config`, from the tensor headers. A checkpoint that lacks a
/// tensor of a graph is an error here, before any open.
pub fn weights_bytes(config: &Config) -> Result<u64, String> {
    let mut count = Count::default();
    let weights = Weights::load(config, &mut count)?;
    crate::model::check(&weights)?;
    Ok(count.bytes)
}

fn term(name: &str, tier: Tier, bytes: u64) -> MandatoryTerm {
    MandatoryTerm {
        name: name.to_owned(),
        tier,
        bytes,
        lifetime: Lifetime::Engine,
        kinds: vec![Kind::Embed, Kind::Step],
    }
}

/// The limits for a config, with the device bytes of the weights.
pub fn limits(weights: u64) -> Limits {
    let ring = ((MAX_TASKS_PER_CARD + 1) * 4 + 128 + MAX_TASKS_PER_CARD * 64 + 256) as u64;
    Limits {
        // A drop frees the weights, the arena and the card's context.
        capabilities: vec![Capability::Image, Capability::Close],
        parts: Vec::new(),
        inputs: Inputs::Tokens,
        devices: vec![0],
        stages: Vec::new(),
        max_segments: 0,
        max_pass_inputs: 0,
        max_logit_rows: 0,
        // One job at a time. An image job holds no state in the store (spec 11.9).
        slots: vec![SlotLimit {
            bytes: 0,
            count: 1,
            tokens: (MAX_CHUNKS * CHUNK_IDS) as u64,
        }],
        max_quantum_ms: MAX_QUANTUM_MS,
        memory: MemoryLimits {
            mandatory: vec![
                term("weights", Tier::Device(0), weights),
                term("activations", Tier::Device(0), arena_bytes() + ring),
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

/// The limits of a config.
pub fn plan(config: &Config) -> Result<Limits, String> {
    Ok(limits(weights_bytes(config)?))
}
