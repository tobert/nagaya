//! The oracle's fixture, metrics and tolerances (spec 8.3, milestone 11.5).
//!
//! The bench `benches/oracle.rs` and the test `tests/oracle.rs` both include this file. It needs no GPU.
//!
//! The fixture holds, for each prompt, the token ids and the top-64 logits of the last position from mk's CPU
//! reference. The bench runs the same ids through mk and calls `measure` and `breaches`.

#![allow(dead_code)]

use std::path::Path;

/// The size of the reference's top list.
pub const TOP: usize = 64;

/// The argmax must agree when the reference top-1 and top-2 logits are further apart than this.
pub const MARGIN: f64 = 0.25;

/// The reference top-8 ids must all be inside the engine's top list of this size.
pub const TOP8_INSIDE: usize = 16;

/// The qmm path that the engine runs. `MK_FAST_QMM=0` is exact. Unset or 1 is the f16 fast path, an approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QmmPath {
    Exact,
    Fast,
}

impl QmmPath {
    /// Read the value of `MK_FAST_QMM` as the binding reads it (`worker.rs` `fast_qmm_from`).
    pub fn from_setting(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("1") => Ok(Self::Fast),
            Some("0") => Ok(Self::Exact),
            Some(other) => Err(format!("MK_FAST_QMM={other}: want 0 or 1")),
        }
    }

    pub fn from_env() -> Self {
        let value = std::env::var("MK_FAST_QMM").ok();
        Self::from_setting(value.as_deref()).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact (MK_FAST_QMM=0, a gate)",
            Self::Fast => "fast (f16 qmm, measured only)",
        }
    }

    /// The largest absolute logit difference over the reference top-64. `None` means that it is not a gate.
    pub fn max_abs_diff(self) -> Option<f64> {
        (self == Self::Exact).then_some(1e-3)
    }

    /// The largest KL(reference || engine) over the reference top-64, both sides renormalised over those ids.
    pub fn max_kl(self) -> Option<f64> {
        (self == Self::Exact).then_some(1e-8)
    }

    /// The largest absolute difference of the two logsumexp values over the whole vocabulary.
    pub fn max_lse_diff(self) -> Option<f64> {
        (self == Self::Exact).then_some(1e-3)
    }
}

/// One prompt of the fixture.
#[derive(Clone, Debug)]
pub struct Prompt {
    pub name: String,
    pub source: String,
    pub text: String,
    pub ids: Vec<u32>,
    pub top_ids: Vec<usize>,
    pub top_logits: Vec<f64>,
    pub logsumexp: f64,
}

/// The fixture: the tool's facts and the prompts.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub layers: u64,
    pub vocab: usize,
    pub transformers: String,
    pub reference_sha: String,
    pub seconds: u64,
    pub prompts: Vec<Prompt>,
}

fn field<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value, String> {
    value.get(key).ok_or_else(|| format!("the field {key} is missing"))
}

fn text_of(value: &serde_json::Value, key: &str) -> Result<String, String> {
    let text = field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} is not a string"))?;
    if text.is_empty() {
        return Err(format!("{key} is empty"));
    }
    Ok(text.to_owned())
}

fn number_of(value: &serde_json::Value, key: &str) -> Result<f64, String> {
    field(value, key)?
        .as_f64()
        .ok_or_else(|| format!("{key} is not a number"))
}

fn list_of<T>(
    value: &serde_json::Value,
    key: &str,
    item: impl Fn(&serde_json::Value) -> Option<T>,
) -> Result<Vec<T>, String> {
    let array = field(value, key)?
        .as_array()
        .ok_or_else(|| format!("{key} is not a list"))?;
    array
        .iter()
        .map(|entry| item(entry).ok_or_else(|| format!("{key} holds a bad entry")))
        .collect()
}

/// Read a fixture from JSON text.
pub fn parse(text: &str) -> Result<Fixture, String> {
    let document: serde_json::Value = serde_json::from_str(text).map_err(|error| format!("bad JSON: {error}"))?;
    let layers = number_of(&document, "layers")? as u64;
    let vocab = number_of(&document, "vocab")? as usize;
    let transformers = text_of(&document, "transformers")?;
    let reference_sha = text_of(&document, "reference_sha")?;
    let seconds = number_of(&document, "seconds")? as u64;
    let prompts = field(&document, "prompts")?
        .as_array()
        .ok_or("prompts is not a list")?
        .iter()
        .map(|entry| {
            let name = text_of(entry, "name")?;
            let read = || -> Result<Prompt, String> {
                Ok(Prompt {
                    source: text_of(entry, "source")?,
                    text: text_of(entry, "text")?,
                    ids: list_of(entry, "ids", |v| v.as_u64().map(|id| id as u32))?,
                    top_ids: list_of(entry, "top_ids", |v| v.as_u64().map(|id| id as usize))?,
                    top_logits: list_of(entry, "top_logits", |v| v.as_f64())?,
                    logsumexp: number_of(entry, "logsumexp")?,
                    name: name.clone(),
                })
            };
            read().map_err(|error| format!("prompt {name}: {error}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Fixture {
        layers,
        vocab,
        transformers,
        reference_sha,
        seconds,
        prompts,
    })
}

/// Read the fixture file. It panics with the reason when the file is bad.
pub fn load(path: &Path) -> Fixture {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    parse(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The first fault of a fixture, or `None` when it is sound.
pub fn fixture_problem(fixture: &Fixture, max_tokens: usize) -> Option<String> {
    if fixture.prompts.is_empty() {
        return Some("the fixture has no prompt".to_owned());
    }
    for prompt in &fixture.prompts {
        let name = &prompt.name;
        if prompt.ids.is_empty() || prompt.ids.len() > max_tokens {
            return Some(format!(
                "{name}: {} tokens; the limit is {max_tokens}",
                prompt.ids.len()
            ));
        }
        if let Some(id) = prompt.ids.iter().find(|id| **id as usize >= fixture.vocab) {
            return Some(format!("{name}: token id {id} is outside the vocabulary"));
        }
        if prompt.top_ids.len() != TOP || prompt.top_logits.len() != TOP {
            return Some(format!("{name}: the top list does not hold {TOP} entries"));
        }
        if let Some(id) = prompt.top_ids.iter().find(|id| **id >= fixture.vocab) {
            return Some(format!("{name}: top id {id} is outside the vocabulary"));
        }
        let mut seen = prompt.top_ids.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != TOP {
            return Some(format!("{name}: a top id repeats"));
        }
        if prompt.top_logits.windows(2).any(|pair| pair[0] < pair[1]) {
            return Some(format!("{name}: the top logits are not sorted, largest first"));
        }
        if prompt.top_logits.iter().any(|v| !v.is_finite()) || !prompt.logsumexp.is_finite() {
            return Some(format!("{name}: a logit is not finite"));
        }
        if prompt.top_logits[0] > prompt.logsumexp {
            return Some(format!("{name}: a logit is above the logsumexp"));
        }
    }
    None
}

/// The measured gap of one prompt between the reference and the engine.
#[derive(Clone, Debug)]
pub struct Gap {
    pub finite: bool,
    pub margin: f64,
    pub argmax_agrees: bool,
    pub overlap_1: usize,
    pub overlap_8: usize,
    pub overlap_64: usize,
    /// How many of the reference top-8 ids are inside the engine's top `TOP8_INSIDE`.
    pub top8_inside: usize,
    pub max_abs_diff: f64,
    pub kl: f64,
    pub lse_diff: f64,
}

/// The ids of the k largest values, largest first.
fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
    order.truncate(k);
    order
}

fn overlap(a: &[usize], b: &[usize], k: usize) -> usize {
    a[..k].iter().filter(|id| b[..k].contains(id)).count()
}

/// Log-softmax of a list, over the list alone.
fn log_softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    values.iter().map(|v| v - max - sum.ln()).collect()
}

/// KL(p || q) of two log-probability lists over the same ids.
fn kl(p_log: &[f64], q_log: &[f64]) -> f64 {
    p_log.iter().zip(q_log).map(|(p, q)| p.exp() * (p - q)).sum()
}

/// The logsumexp of the engine's logits over the whole vocabulary.
fn logsumexp(logits: &[f32]) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    max + logits.iter().map(|v| (f64::from(*v) - max).exp()).sum::<f64>().ln()
}

/// Compare the engine's logits of the last position with the reference of one prompt.
pub fn measure(prompt: &Prompt, logits: &[f32]) -> Gap {
    let finite = logits.iter().all(|v| v.is_finite());
    let engine_top = top_k(logits, TOP);
    let at: Vec<f64> = prompt.top_ids.iter().map(|id| f64::from(logits[*id])).collect();
    Gap {
        finite,
        margin: prompt.top_logits[0] - prompt.top_logits[1],
        argmax_agrees: prompt.top_ids[0] == engine_top[0],
        overlap_1: overlap(&prompt.top_ids, &engine_top, 1),
        overlap_8: overlap(&prompt.top_ids, &engine_top, 8),
        overlap_64: overlap(&prompt.top_ids, &engine_top, TOP),
        top8_inside: prompt.top_ids[..8]
            .iter()
            .filter(|id| engine_top[..TOP8_INSIDE].contains(id))
            .count(),
        max_abs_diff: at
            .iter()
            .zip(&prompt.top_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max),
        kl: kl(&log_softmax(&prompt.top_logits), &log_softmax(&at)),
        lse_diff: logsumexp(logits) - prompt.logsumexp,
    }
}

/// The tolerances that a gap breaks on one qmm path. An empty list means that the gap is inside every gate.
pub fn breaches(gap: &Gap, path: QmmPath) -> Vec<String> {
    let mut list = Vec::new();
    if !gap.finite {
        list.push("the engine gave a non-finite logit".to_owned());
    }
    if gap.margin > MARGIN && !gap.argmax_agrees {
        list.push(format!(
            "the argmax differs, and the reference margin is {:.3}",
            gap.margin
        ));
    }
    if gap.top8_inside < 8 {
        list.push(format!(
            "only {}/8 of the reference top-8 are in the engine top-{TOP8_INSIDE}",
            gap.top8_inside
        ));
    }
    if let Some(limit) = path.max_abs_diff()
        && (gap.max_abs_diff.is_nan() || gap.max_abs_diff > limit)
    {
        list.push(format!("max abs diff {:.6} is past {limit:e}", gap.max_abs_diff));
    }
    if let Some(limit) = path.max_kl()
        && (gap.kl.is_nan() || gap.kl > limit)
    {
        list.push(format!("KL {:.3e} is past {limit:e}", gap.kl));
    }
    if let Some(limit) = path.max_lse_diff()
        && (gap.lse_diff.is_nan() || gap.lse_diff.abs() > limit)
    {
        list.push(format!("logsumexp diff {:.6} is past {limit:e}", gap.lse_diff));
    }
    list
}
