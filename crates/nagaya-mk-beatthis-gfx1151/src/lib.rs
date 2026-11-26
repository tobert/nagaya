//! Beat This! on gfx1151: the plan of `nagaya-family-beatthis` run as one HIP launch for each task (spec 11.19).
//!
//! The kernels are the tile functions of `kernels/beatthis_ops.h`, a source in a subset of C that the CUDA and OpenCL
//! ports share. This crate holds the HIP launchers, the host loop and the engine. Its bits are its own: the same
//! identity and input give the same bits on gfx1151, and the tests check them against the oracle with a tolerance.
//! The limits need no ROCm, so a build without `hip` can plan the engine.

#[cfg(feature = "hip")]
pub mod engine;
#[cfg(feature = "hip")]
pub mod gpu;

use nagaya_abi::{Capability, ChainLimits, Determinism, Inputs, Lifetime, Limits, MandatoryTerm, MemoryLimits, Tier};
use nagaya_family_beatthis::chunks::CHUNK;
use nagaya_family_beatthis::plan::{self, Config, Task};

/// The bound of one chunk. A naive chunk of `final0` takes about 1 s on the card (spec 11.19).
pub const MAX_QUANTUM_MS: u32 = 10_000;

/// The device bytes of the engine: its prepared weights, the buffers of a plan of one whole chunk, and the rotated q
/// and k of its largest attention.
pub fn device_bytes(config: &Config) -> u64 {
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

/// The limits of the engine on one card, for a checkpoint of `config`.
pub fn limits(config: &Config) -> Limits {
    Limits {
        capabilities: vec![Capability::Beats, Capability::Close],
        parts: Vec::new(),
        inputs: Inputs::Rows,
        devices: vec![0],
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
                tier: Tier::Device(0),
                bytes: device_bytes(config),
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
