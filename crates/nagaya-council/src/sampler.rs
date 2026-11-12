//! The sampler of spec 11.2. One sampler serves all engines.
//!
//! The filters match `service/sampling.py` of the reference: top_k, top_p, min_p, then temperature. The draw keeps no state: it
//! is a hash of the seed, the position and the purpose. This crate never reads OS randomness. The caller gives
//! the seed, and it chooses one when the client sent none.

use std::fmt;

/// The resolved settings of one generation. The engine gives the defaults, and the caller resolves them.
///
/// `top_k` 0 means off. `top_p` 1.0 means off. `min_p` 0.0 means off. `temperature` 0.0 means greedy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u64,
    pub min_p: f64,
    pub seed: u64,
}

impl Settings {
    /// Check the ranges of gogi's `council-api.md`, "Generations": temperature at least 0 and finite, top_p above 0
    /// and at most 1, min_p at least 0 and below 1. A NaN fails each check.
    pub fn validate(&self) -> Result<(), SamplerError> {
        let bad = |what| Err(SamplerError::BadSetting(what));
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return bad("temperature must be finite and at least 0");
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return bad("top_p must be above 0 and at most 1");
        }
        if !(self.min_p >= 0.0 && self.min_p < 1.0) {
            return bad("min_p must be at least 0 and below 1");
        }
        Ok(())
    }
}

/// Why a sampler call failed. No call gives a token after an error.
#[derive(Debug, PartialEq)]
pub enum SamplerError {
    /// A setting is out of range. The text names the range.
    BadSetting(&'static str),
    /// The logits are empty. An engine with no vocabulary has a defect.
    EmptyLogits,
    /// A logit is NaN or infinite. This is a defect upstream, and sampling it would hide the defect.
    NonFiniteLogit { id: u32, value: f32 },
    /// The temperature is so small that a tempered logit overflows. Use temperature 0 for greedy.
    TemperatureOverflow,
    /// The uniform value is not in [0, 1).
    BadUniform(f64),
    /// Each token of the row is one that the reply never picks. A family with such a vocabulary has a defect.
    AllDropped,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSetting(what) => write!(f, "sample setting out of range: {what}"),
            Self::EmptyLogits => write!(f, "the logits are empty: the engine gave no vocabulary"),
            Self::NonFiniteLogit { id, value } => {
                write!(f, "logit {id} is {value}: the engine gave a logit that is not finite")
            }
            Self::TemperatureOverflow => {
                write!(
                    f,
                    "the temperature is too small: a tempered logit overflows; use temperature 0 for greedy"
                )
            }
            Self::BadUniform(u) => write!(f, "the uniform value {u} is not in [0, 1)"),
            Self::AllDropped => write!(f, "each token of the row is a token that the reply never picks"),
        }
    }
}

impl std::error::Error for SamplerError {}

/// Why a random value is drawn (spec 11.2). The number is part of the hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    Sample = 0,
    Resample = 1,
    /// The test of a draft.
    Draft = 2,
}

/// The survivors of the filters, in order of logit down and id up, with the probabilities that the draw uses.
#[derive(Clone, Debug, PartialEq)]
pub struct Survivors {
    pub ids: Vec<u32>,
    pub probs: Vec<f64>,
}

/// The chosen token. `p` is its probability, and `h` is the entropy in nats, both of the untempered softmax over
/// the whole vocabulary, in float64. The filters do not change them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub token: u32,
    pub p: f64,
    pub h: f64,
}

const DRAW_DOMAIN: &[u8] = b"nagaya draw v1";
const FORK_DOMAIN: &[u8] = b"nagaya fork v1";

/// The uniform value in [0, 1) for one token. It depends on the seed, the position and the purpose only.
///
/// The position is the token's position in the context, from 0.
pub fn draw_uniform(seed: u64, position: u64, purpose: Purpose) -> f64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DRAW_DOMAIN);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&position.to_le_bytes());
    hasher.update(&[purpose as u8]);
    let hash = hasher.finalize();
    let first = u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("a hash has 32 bytes"));
    (first >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// The seed of a fork: a hash of the parent seed and the branch number.
pub fn child_seed(parent: u64, branch: u64) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(FORK_DOMAIN);
    hasher.update(&parent.to_le_bytes());
    hasher.update(&branch.to_le_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("a hash has 32 bytes"))
}

fn widen(logits: &[f32]) -> Result<Vec<f64>, SamplerError> {
    if logits.is_empty() {
        return Err(SamplerError::EmptyLogits);
    }
    if let Some((id, &value)) = logits.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(SamplerError::NonFiniteLogit { id: id as u32, value });
    }
    Ok(logits.iter().map(|&x| f64::from(x)).collect())
}

/// Softmax in float64. The caller guarantees that `z` is finite and not empty.
fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.into_iter().map(|x| x / sum).collect()
}

/// The sort key of the order of the filters: logit down, then id up. It is a total order, so a cut at a tie is exact.
/// The logits come from f32 (`widen`), so the f32 bits order them. Minus zero becomes plus zero, because the two tie as
/// numbers. A sort of these keys is many times faster than a sort with a comparison of the logits.
fn order_key(logit: f64, id: u32) -> u64 {
    let x = logit as f32;
    debug_assert!(f64::from(x) == logit, "a logit of the sampler comes from f32");
    let x = if x == 0.0 { 0.0f32 } else { x };
    let bits = x.to_bits();
    let up = if bits >> 31 == 1 { !bits } else { bits | 0x8000_0000 };
    (u64::from(!up) << 32) | u64::from(id)
}

/// The survivors of the filters. `dropped` are tokens that the reply never picks: they leave the row before each
/// filter, as a logit of minus infinity does in the reference.
fn candidates(l: &[f64], s: &Settings, dropped: &[u32]) -> Result<Survivors, SamplerError> {
    let kept = |i: &u32| !dropped.contains(i);
    if s.temperature == 0.0 {
        // Greedy: the first maximum, which is the lowest id. No filter and no draw.
        let mut best: Option<usize> = None;
        for (i, &x) in l.iter().enumerate() {
            if kept(&(i as u32)) && best.is_none_or(|b| x > l[b]) {
                best = Some(i);
            }
        }
        let best = best.ok_or(SamplerError::AllDropped)?;
        return Ok(Survivors {
            ids: vec![best as u32],
            probs: vec![1.0],
        });
    }
    let mut keys: Vec<u64> = (0..l.len() as u32)
        .filter(kept)
        .map(|i| order_key(l[i as usize], i))
        .collect();
    if keys.is_empty() {
        return Err(SamplerError::AllDropped);
    }
    if s.top_k > 0 && (s.top_k as usize) < keys.len() {
        let k = s.top_k as usize;
        keys.select_nth_unstable(k - 1);
        keys.truncate(k);
    }
    keys.sort_unstable();
    let mut ids: Vec<u32> = keys.iter().map(|k| *k as u32).collect();
    let mut z: Vec<f64> = ids.iter().map(|&i| l[i as usize]).collect();
    if s.top_p < 1.0 {
        // Keep the smallest prefix whose running mass reaches top_p, and never fewer than one.
        let mut run = 0.0;
        let cum: Vec<f64> = softmax(&z)
            .into_iter()
            .map(|p| {
                run += p;
                run
            })
            .collect();
        let keep = (cum.partition_point(|&c| c < s.top_p) + 1).min(ids.len());
        ids.truncate(keep);
        z.truncate(keep);
    }
    if s.min_p > 0.0 {
        // The logits are sorted, so the passing ones are a prefix, and the first one passes.
        let floor = z[0] + s.min_p.ln();
        let keep = z.iter().filter(|&&x| x >= floor).count();
        ids.truncate(keep);
        z.truncate(keep);
    }
    let tempered: Vec<f64> = z.iter().map(|&x| x / s.temperature).collect();
    if tempered.iter().any(|x| !x.is_finite()) {
        return Err(SamplerError::TemperatureOverflow);
    }
    Ok(Survivors {
        ids,
        probs: softmax(&tempered),
    })
}

/// The survivors of top_k, top_p and min_p, with the probabilities of the draw. Greedy gives the argmax alone.
pub fn survivors(logits: &[f32], s: &Settings) -> Result<Survivors, SamplerError> {
    s.validate()?;
    candidates(&widen(logits)?, s, &[])
}

/// The probability of the token and the entropy (nats) of the untempered softmax over the whole vocabulary.
fn raw_stats(l: &[f64], token: u32) -> (f64, f64) {
    // One loop with the same sums in the same order as two passes, so the bits stay, and nothing is allocated.
    let max = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut sum, mut weighted) = (0.0, 0.0);
    for &x in l {
        let z = x - max;
        let e = z.exp();
        sum += e;
        weighted += e * z;
    }
    ((l[token as usize] - max).exp() / sum, sum.ln() - weighted / sum)
}

/// Pick a token with a given uniform value in [0, 1). Greedy settings ignore the value.
///
/// The pick is the inverse CDF over the survivors, in float64. With the same uniform value as the reference, it gives the same token.
pub fn pick_with(logits: &[f32], s: &Settings, u: f64) -> Result<Pick, SamplerError> {
    pick_with_except(logits, s, u, &[])
}

/// Pick a token with a given uniform value, and never one of `dropped`. The probability and the entropy stay those of
/// the whole row.
pub fn pick_with_except(logits: &[f32], s: &Settings, u: f64, dropped: &[u32]) -> Result<Pick, SamplerError> {
    s.validate()?;
    if !(0.0..1.0).contains(&u) {
        return Err(SamplerError::BadUniform(u));
    }
    let l = widen(logits)?;
    let surv = candidates(&l, s, dropped)?;
    let token = if s.temperature == 0.0 {
        surv.ids[0]
    } else {
        let mut run = 0.0;
        let cdf: Vec<f64> = surv
            .probs
            .iter()
            .map(|p| {
                run += p;
                run
            })
            .collect();
        let target = u * cdf[cdf.len() - 1];
        surv.ids[cdf.partition_point(|&c| c <= target).min(cdf.len() - 1)]
    };
    let (p, h) = raw_stats(&l, token);
    Ok(Pick { token, p, h })
}

/// Pick the token at a position of the context. The uniform value comes from the draw function with the purpose
/// `Sample`. Use `pick_with` for another purpose.
pub fn pick(logits: &[f32], s: &Settings, position: u64) -> Result<Pick, SamplerError> {
    pick_except(logits, s, position, &[])
}

/// Pick the token at a position of the context, and never one of `dropped` (spec 11.11, "The generation modes").
pub fn pick_except(logits: &[f32], s: &Settings, position: u64, dropped: &[u32]) -> Result<Pick, SamplerError> {
    pick_with_except(logits, s, draw_uniform(s.seed, position, Purpose::Sample), dropped)
}
