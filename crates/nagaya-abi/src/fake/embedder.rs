//! The fake embedding engine (engine-abi "Embedding"), for the tests of the embedding route with no card.
//!
//! Component `k` of the vector of one id is a hash of the id and `k`. The vector of a call is the mean over its ids,
//! with no normalization, as the engine gives it. Thus a vector depends only on its ids, and a change of one id changes
//! it.

use std::collections::BTreeMap;

use crate::Capability;
use crate::{
    ChainLimits, Determinism, Embedding, Engine, Identity, Inputs, Lifetime, Limits, MandatoryTerm, MemoryLimits,
    PassError, PassErrorKind, PassOut, Segment, Tap, Tier,
};

/// The fake embedding engine.
pub struct FakeEmbedder {
    limits: Limits,
    dimension: u32,
    max_tokens: u32,
    model: String,
    last_error: Option<String>,
}

impl FakeEmbedder {
    /// Make an engine with vectors of `dimension` components and at most `max_tokens` ids in a call. Its weights
    /// take `bytes` of device 0.
    pub fn new(dimension: u32, max_tokens: u32, bytes: u64) -> Self {
        assert!(dimension > 0, "a vector has at least one component");
        let limits = Limits {
            capabilities: vec![Capability::Embedding, Capability::Close],
            parts: Vec::new(),
            inputs: Inputs::Tokens,
            devices: vec![0],
            stages: Vec::new(),
            max_segments: 0,
            max_pass_inputs: 0,
            max_logit_rows: 0,
            slots: Vec::new(),
            max_quantum_ms: 1000,
            memory: MemoryLimits {
                mandatory: vec![MandatoryTerm {
                    name: "weights".to_owned(),
                    tier: Tier::Device(0),
                    bytes,
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
        };
        Self {
            limits,
            dimension,
            max_tokens,
            model: "fake-embedder".to_owned(),
            last_error: None,
        }
    }

    /// Give the engine another model name in its identity.
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_owned();
        self
    }
}

/// Component `k` of the vector of `id`: a value in [-1, 1) from a hash of both.
fn component(id: u32, k: u32) -> f32 {
    let mut x = (u64::from(id) << 32) | u64::from(k);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    ((x >> 40) as f32 / (1u64 << 23) as f32) - 1.0
}

impl Embedding for FakeEmbedder {
    fn embedding_max_tokens(&self) -> u32 {
        self.max_tokens
    }

    fn embedding_dimension(&self) -> u32 {
        self.dimension
    }

    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, PassError> {
        if ids.is_empty() || ids.len() > self.max_tokens as usize {
            let message = format!(
                "nothing ran: the call has {} ids; the fake takes 1 to {}",
                ids.len(),
                self.max_tokens
            );
            self.last_error = Some(message.clone());
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message,
            });
        }
        let mut sum = vec![0f64; self.dimension as usize];
        for &id in ids {
            for (k, value) in sum.iter_mut().enumerate() {
                *value += f64::from(component(id, k as u32));
            }
        }
        Ok(sum.into_iter().map(|value| (value / ids.len() as f64) as f32).collect())
    }
}

impl Engine for FakeEmbedder {
    fn identity(&self) -> Identity {
        Identity {
            model: self.model.clone(),
            family: "fake-embedder".to_owned(),
            checkpoint_digest: "fake:embedder".to_owned(),
            tokenizer_hash: "fake:bytes".to_owned(),
            template: "fake".to_owned(),
            numbers: BTreeMap::new(),
            plan_table: "fake".to_owned(),
            bit_settings: Vec::new(),
            canary_hash: "fake:0".to_owned(),
            build: std::collections::BTreeMap::new(),
        }
    }

    fn limits(&self) -> Limits {
        self.limits.clone()
    }

    /// An embedding engine takes no pass (engine-abi "Embedding").
    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        Err(PassError {
            kind: PassErrorKind::Refused,
            message: "an embedding engine takes no pass; use embed".to_owned(),
        })
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    fn embedding(&mut self) -> Option<&mut dyn Embedding> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vector_depends_only_on_its_ids() {
        let mut engine = FakeEmbedder::new(8, 16, 0);
        let a = engine.embed(&[2, 7, 9, 1]).unwrap();
        assert_eq!(a.len(), 8);
        assert_eq!(engine.embed(&[2, 7, 9, 1]).unwrap(), a);
        assert_ne!(engine.embed(&[2, 7, 8, 1]).unwrap(), a);
        assert!(engine.embed(&[]).is_err());
        assert!(engine.embed(&[1; 17]).is_err());
    }
}
