//! The oracle's fixture, metrics and tolerances (spec 8.3, milestone 11.5).
//!
//! The bench `benches/oracle.rs` and the test `tests/oracle.rs` both include this file. It needs no GPU.
//!
//! The fixture holds, for each prompt, the token ids and the top-64 logits of the last position from the CPU
//! reference. The bench runs the same ids through the reference and calls `measure` and `breaches`.

#![allow(dead_code)]

use std::path::Path;

/// The size of the reference's top list.
pub const TOP: usize = 64;

/// The argmax must agree when the reference top-1 and top-2 logits are further apart than this.
pub const MARGIN: f64 = 0.25;

/// The reference top-8 ids must all be inside the engine's top list of this size.
pub const TOP8_INSIDE: usize = 16;

/// The qmm path that the engine runs. `QWEN38_FAST_QMM=0` is exact. Unset or 1 is the f16 fast path, an approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QmmPath {
    Exact,
    Fast,
}

impl QmmPath {
    /// Read the value of `QWEN38_FAST_QMM` as the binding reads it (`worker.rs` `fast_qmm_from`).
    pub fn from_setting(value: Option<&str>) -> Result<Self, String> {
        match value {
            None | Some("1") => Ok(Self::Fast),
            Some("0") => Ok(Self::Exact),
            Some(other) => Err(format!("QWEN38_FAST_QMM={other}: want 0 or 1")),
        }
    }

    pub fn from_env() -> Self {
        let value = std::env::var("QWEN38_FAST_QMM").ok();
        Self::from_setting(value.as_deref()).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact (QWEN38_FAST_QMM=0, a gate)",
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

// ---- The oracle of the residual, the fold and an add (milestone 34) --------------------------------------------------
//
// `tools/oracle_fixture.py --residual` writes the JSON `tests/fixtures/oracle/qwen38_residual.json` and the full rows
// as little-endian f32 files. A test reads the JSON with `parse_residual`, and the rows with `read_rows`. An absent
// file skips the test, and one line on stderr says why. A file that exists and cannot be read or parsed fails the test.

use sha2::{Digest, Sha256};

/// The layers of the model, and so the rows of the tap `residual`.
pub const RESIDUAL_LAYERS: usize = 48;

/// The width of the residual: 4 streams of 2,560 values (engine-abi "The residual of the Qwen3.8 engine").
pub const RESIDUAL_WIDTH: usize = 10_240;

/// The width of the fold: the row that the head reads.
pub const FOLD_WIDTH: usize = 2_560;

/// The values that the JSON keeps from each row, at the indices that it names.
pub const SAMPLE: usize = 64;

/// The fixture of the residual, below the manifest directory of the crate.
pub const RESIDUAL_FIXTURE: &str = "tests/fixtures/oracle/qwen38_residual.json";

/// The variable of the folder of the full rows. The default is `target/qwen38-golden` of the workspace.
pub const GOLDEN_VARIABLE: &str = "QWEN38_GOLDEN_DIR";

/// The variable that makes an absent file fail the test (`AGENTS.md`, "Tests that need a model file").
pub const REQUIRE_VARIABLE: &str = "NAGAYA_REQUIRE_MODELS";

/// The record of the 4 streams after one layer.
#[derive(Clone, Debug)]
pub struct ResidualRow {
    pub layer: usize,
    pub stream_norms: Vec<f64>,
    pub sample: Vec<f32>,
    pub sha256: String,
}

/// The record of the row that the head reads.
#[derive(Clone, Debug)]
pub struct FoldRow {
    pub norm: f64,
    pub sample: Vec<f32>,
    pub sha256: String,
}

/// One run of the reference: a prompt, and the rows at its last position.
#[derive(Clone, Debug)]
pub struct ResidualRun {
    pub name: String,
    pub ids: Vec<u32>,
    pub add: bool,
    pub residual_file: String,
    pub fold_file: String,
    pub rows: Vec<ResidualRow>,
    pub fold: FoldRow,
    pub top_ids: Vec<usize>,
    pub top_logits: Vec<f64>,
    pub logsumexp: f64,
}

/// The add of the third run.
#[derive(Clone, Debug)]
pub struct AddSpec {
    pub formula: String,
    pub strength: f64,
    pub strength_f32: f32,
    pub layers: Vec<usize>,
    pub run: String,
}

/// The fixture of the residual.
#[derive(Clone, Debug)]
pub struct ResidualFixture {
    pub reference_sha: String,
    pub vocab: usize,
    pub sample_indices: Vec<usize>,
    pub fold_sample_indices: Vec<usize>,
    pub add: AddSpec,
    pub runs: Vec<ResidualRun>,
}

impl ResidualFixture {
    pub fn run(&self, name: &str) -> &ResidualRun {
        self.runs
            .iter()
            .find(|run| run.name == name)
            .unwrap_or_else(|| panic!("the fixture has no run {name}"))
    }
}

fn bool_of(value: &serde_json::Value, key: &str) -> Result<bool, String> {
    field(value, key)?
        .as_bool()
        .ok_or_else(|| format!("{key} is not true or false"))
}

fn exact_count(value: &serde_json::Value, key: &str, want: usize) -> Result<(), String> {
    let got = number_of(value, key)?;
    if got != want as f64 {
        return Err(format!(
            "{key} is {got}, but the engine has {want}; make the fixture again with the tool"
        ));
    }
    Ok(())
}

fn hash_of(value: &serde_json::Value, key: &str) -> Result<String, String> {
    let text = text_of(value, key)?;
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(format!("{key} is not 64 lowercase hex digits"));
    }
    Ok(text)
}

fn floats_of(value: &serde_json::Value, key: &str, count: usize) -> Result<Vec<f32>, String> {
    let list = list_of(value, key, |v| v.as_f64().map(|x| x as f32))?;
    if list.len() != count {
        return Err(format!("{key} holds {} values, not {count}", list.len()));
    }
    if list.iter().any(|x| !x.is_finite()) {
        return Err(format!("{key} holds a value that is not finite"));
    }
    Ok(list)
}

fn indices_of(value: &serde_json::Value, key: &str, width: usize) -> Result<Vec<usize>, String> {
    let list = list_of(value, key, |v| v.as_u64().map(|x| x as usize))?;
    if list.len() != SAMPLE {
        return Err(format!("{key} holds {} indices, not {SAMPLE}", list.len()));
    }
    if list.windows(2).any(|pair| pair[0] >= pair[1]) || list.iter().any(|i| *i >= width) {
        return Err(format!("{key} are not rising indices below {width}"));
    }
    Ok(list)
}

fn file_name_of(value: &serde_json::Value, key: &str) -> Result<String, String> {
    let name = text_of(value, key)?;
    if name.contains('/') || name.contains('\\') || name.starts_with('.') {
        return Err(format!("{key} {name} is not a plain file name"));
    }
    Ok(name)
}

fn parse_run(entry: &serde_json::Value) -> Result<ResidualRun, String> {
    let rows_json = field(entry, "rows")?.as_array().ok_or("rows is not a list")?;
    if rows_json.len() != RESIDUAL_LAYERS {
        return Err(format!(
            "rows holds {} layers, not {RESIDUAL_LAYERS}; make the fixture again with the tool",
            rows_json.len()
        ));
    }
    let mut rows = Vec::with_capacity(RESIDUAL_LAYERS);
    for (index, row) in rows_json.iter().enumerate() {
        let read = || -> Result<ResidualRow, String> {
            let layer = number_of(row, "layer")?;
            if layer != index as f64 {
                return Err(format!(
                    "layer {index} is missing: the list holds layer {layer} in its place; make the fixture again with the tool"
                ));
            }
            let stream_norms = list_of(row, "stream_norms", |v| v.as_f64())?;
            if stream_norms.len() != 4 || stream_norms.iter().any(|x| !x.is_finite() || *x <= 0.0) {
                return Err("stream_norms are not 4 positive numbers".to_owned());
            }
            Ok(ResidualRow {
                layer: index,
                stream_norms,
                sample: floats_of(row, "sample", SAMPLE)?,
                sha256: hash_of(row, "sha256")?,
            })
        };
        rows.push(read().map_err(|error| format!("layer {index}: {error}"))?);
    }
    let fold_json = field(entry, "fold")?;
    let fold = FoldRow {
        norm: number_of(fold_json, "norm").map_err(|error| format!("fold: {error}"))?,
        sample: floats_of(fold_json, "sample", SAMPLE).map_err(|error| format!("fold: {error}"))?,
        sha256: hash_of(fold_json, "sha256").map_err(|error| format!("fold: {error}"))?,
    };
    let top_ids = list_of(entry, "top_ids", |v| v.as_u64().map(|id| id as usize))?;
    let top_logits = list_of(entry, "top_logits", |v| v.as_f64())?;
    if top_ids.len() != TOP || top_logits.len() != TOP {
        return Err(format!("the top list does not hold {TOP} entries"));
    }
    let ids = list_of(entry, "ids", |v| v.as_u64().map(|id| id as u32))?;
    if ids.is_empty() {
        return Err("ids is empty".to_owned());
    }
    Ok(ResidualRun {
        name: text_of(entry, "name")?,
        ids,
        add: bool_of(entry, "add")?,
        residual_file: file_name_of(entry, "residual_file")?,
        fold_file: file_name_of(entry, "fold_file")?,
        rows,
        fold,
        top_ids,
        top_logits,
        logsumexp: number_of(entry, "logsumexp")?,
    })
}

/// Read the fixture of the residual from JSON text.
///
/// It refuses a dtype other than float32, a layer count, width or sample count that differs from the engine's, a
/// missing layer, a bad hash and a missing run. The runs must be `plain`, `chat` and `plain_add`.
pub fn parse_residual(text: &str) -> Result<ResidualFixture, String> {
    let document: serde_json::Value = serde_json::from_str(text).map_err(|error| format!("bad JSON: {error}"))?;
    let dtype = text_of(&document, "dtype")?;
    if dtype != "float32" {
        return Err(format!(
            "dtype is {dtype}, not float32; the rows are f32 bytes, so make the fixture again in float32"
        ));
    }
    exact_count(&document, "layers", RESIDUAL_LAYERS)?;
    exact_count(&document, "width", RESIDUAL_WIDTH)?;
    exact_count(&document, "fold_width", FOLD_WIDTH)?;
    exact_count(&document, "streams", 4)?;
    let add_json = field(&document, "add")?;
    let strength = number_of(add_json, "strength").map_err(|error| format!("add: {error}"))?;
    let add = AddSpec {
        formula: text_of(add_json, "formula").map_err(|error| format!("add: {error}"))?,
        strength,
        strength_f32: number_of(add_json, "strength_f32").map_err(|error| format!("add: {error}"))? as f32,
        layers: list_of(add_json, "layers", |v| v.as_u64().map(|l| l as usize))
            .map_err(|error| format!("add: {error}"))?,
        run: text_of(add_json, "run").map_err(|error| format!("add: {error}"))?,
    };
    if !strength.is_finite() || add.layers.is_empty() || add.layers.iter().any(|l| *l >= RESIDUAL_LAYERS) {
        return Err(format!(
            "add: the strength {strength} or the layers {:?} are not usable",
            add.layers
        ));
    }
    let runs = field(&document, "runs")?
        .as_array()
        .ok_or("runs is not a list")?
        .iter()
        .map(|entry| {
            let name = text_of(entry, "name")?;
            parse_run(entry).map_err(|error| format!("run {name}: {error}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    for want in ["plain", "chat", "plain_add"] {
        match runs.iter().filter(|run| run.name == want).count() {
            1 => {}
            0 => return Err(format!("the run {want} is missing")),
            _ => return Err(format!("the run {want} comes twice")),
        }
    }
    if runs.len() != 3 {
        return Err(format!("the fixture has {} runs, not 3", runs.len()));
    }
    if runs.iter().any(|run| run.add != (run.name == add.run)) {
        return Err(format!("only the run {} has the add", add.run));
    }
    Ok(ResidualFixture {
        reference_sha: text_of(&document, "reference_sha")?,
        vocab: number_of(&document, "vocab")? as usize,
        sample_indices: indices_of(&document, "sample_indices", RESIDUAL_WIDTH)?,
        fold_sample_indices: indices_of(&document, "fold_sample_indices", FOLD_WIDTH)?,
        add,
        runs,
    })
}

/// The full rows of one run, from its two files.
#[derive(Clone, Debug)]
pub struct Rows {
    /// 48 rows of 10,240 values, in layer order.
    pub residual: Vec<f32>,
    /// The 2,560 values that the head reads.
    pub fold: Vec<f32>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn le_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn norm(values: &[f32]) -> f64 {
    values.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt()
}

/// Check one row of a golden file against its record: the hash of the bytes, the sample at the named indices and the
/// norm. `what` names the row in the message.
fn row_problem(what: &str, bytes: &[u8], want_hash: &str, want_sample: &[f32], indices: &[usize]) -> Option<String> {
    let got = hex(&Sha256::digest(bytes));
    if got != want_hash {
        return Some(format!(
            "{what}: the SHA-256 of the row is {got}, but the fixture says {want_hash}; the file and the JSON come from different runs"
        ));
    }
    let values = le_values(bytes);
    for (k, index) in indices.iter().enumerate() {
        if values[*index].to_bits() != want_sample[k].to_bits() {
            return Some(format!(
                "{what}: the value at index {index} is {}, but the fixture says {}",
                values[*index], want_sample[k]
            ));
        }
    }
    None
}

/// Check the bytes of the two golden files of a run against the fixture. The first fault is a message.
pub fn rows_problem(fixture: &ResidualFixture, run: &ResidualRun, residual: &[u8], fold: &[u8]) -> Option<String> {
    let row_bytes = RESIDUAL_WIDTH * 4;
    if residual.len() != RESIDUAL_LAYERS * row_bytes {
        return Some(format!(
            "{} holds {} bytes, but {RESIDUAL_LAYERS} rows of {RESIDUAL_WIDTH} f32 values are {} bytes; make the file again with the tool",
            run.residual_file,
            residual.len(),
            RESIDUAL_LAYERS * row_bytes
        ));
    }
    if fold.len() != FOLD_WIDTH * 4 {
        return Some(format!(
            "{} holds {} bytes, but {FOLD_WIDTH} f32 values are {} bytes; make the file again with the tool",
            run.fold_file,
            fold.len(),
            FOLD_WIDTH * 4
        ));
    }
    for (layer, record) in run.rows.iter().enumerate() {
        let bytes = &residual[layer * row_bytes..(layer + 1) * row_bytes];
        let what = format!("{} layer {layer}", run.residual_file);
        if let Some(problem) = row_problem(&what, bytes, &record.sha256, &record.sample, &fixture.sample_indices) {
            return Some(problem);
        }
        let values = le_values(bytes);
        for (stream, want) in record.stream_norms.iter().enumerate() {
            let got = norm(&values[stream * FOLD_WIDTH..(stream + 1) * FOLD_WIDTH]);
            if (got - want).abs() > 1e-6 * want.abs() {
                return Some(format!(
                    "{what}: the norm of stream {stream} is {got}, but the fixture says {want}"
                ));
            }
        }
    }
    row_problem(
        &run.fold_file,
        fold,
        &run.fold.sha256,
        &run.fold.sample,
        &fixture.fold_sample_indices,
    )
}

/// Read a file. `Ok(None)` means that it is absent. Any other fault is an error, so only absence can skip a test.
pub fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, String> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("cannot read {}: {error}", path.display())),
    }
}

/// The folder of the full rows: `QWEN38_GOLDEN_DIR`, or `target/qwen38-golden` of the workspace.
pub fn golden_dir() -> std::path::PathBuf {
    match std::env::var_os(GOLDEN_VARIABLE) {
        Some(dir) if !dir.is_empty() => dir.into(),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qwen38-golden"),
    }
}

/// True when `NAGAYA_REQUIRE_MODELS=1`.
pub fn require_files() -> bool {
    std::env::var(REQUIRE_VARIABLE).is_ok_and(|value| value == "1")
}

/// Say that a test skips. It prints one line to stderr. When `require` is set, an absent file fails the test.
fn skip(test: &str, what: &str, path: &Path, require: bool) {
    if require {
        panic!(
            "{what} {} does not exist, and {REQUIRE_VARIABLE} is set; write it with tools/oracle_fixture.py --residual",
            path.display()
        );
    }
    eprintln!(
        "skipped: {test}: {what} {} does not exist; write it with tools/oracle_fixture.py --residual",
        path.display()
    );
}

/// The fixture of the residual, or `None` when its file is absent (the test then returns). A file that exists and
/// cannot be read or parsed fails the test.
pub fn residual_fixture(test: &str) -> Option<ResidualFixture> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(RESIDUAL_FIXTURE);
    let bytes = match read_optional(&path).unwrap_or_else(|error| panic!("{error}")) {
        Some(bytes) => bytes,
        None => {
            skip(test, "the fixture", &path, require_files());
            return None;
        }
    };
    let text = String::from_utf8(bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    Some(parse_residual(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display())))
}

/// The full rows of a run from `dir`, or `None` when a file is absent. A bad file fails the test.
pub fn read_rows(test: &str, dir: &Path, fixture: &ResidualFixture, run: &ResidualRun, require: bool) -> Option<Rows> {
    let residual_path = dir.join(&run.residual_file);
    let fold_path = dir.join(&run.fold_file);
    let residual = read_optional(&residual_path).unwrap_or_else(|error| panic!("{error}"));
    let fold = read_optional(&fold_path).unwrap_or_else(|error| panic!("{error}"));
    let (Some(residual), Some(fold)) = (residual, fold) else {
        let missing = if fixture_file_absent(&residual_path) {
            &residual_path
        } else {
            &fold_path
        };
        skip(test, "the oracle file", missing, require);
        return None;
    };
    if let Some(problem) = rows_problem(fixture, run, &residual, &fold) {
        panic!("{problem}");
    }
    Some(Rows {
        residual: le_values(&residual),
        fold: le_values(&fold),
    })
}

fn fixture_file_absent(path: &Path) -> bool {
    !path.exists()
}

/// The gap of one row from the reference, relative to the reference: `max|e - r| / rms(r)`. The sums run in `f64`.
///
/// A row that is not finite, a length that differs and a reference of zero are errors, not a large number: a gate must
/// fail on them.
pub fn relative_gap(engine: &[f32], reference: &[f32]) -> Result<f64, String> {
    if engine.len() != reference.len() || engine.is_empty() {
        return Err(format!(
            "the engine row holds {} values and the reference row holds {}; they must be equal and not empty",
            engine.len(),
            reference.len()
        ));
    }
    if let Some(at) = engine.iter().position(|x| !x.is_finite()) {
        return Err(format!("the engine row has a value that is not finite at index {at}"));
    }
    if let Some(at) = reference.iter().position(|x| !x.is_finite()) {
        return Err(format!(
            "the reference row has a value that is not finite at index {at}"
        ));
    }
    let rms = (reference.iter().map(|r| f64::from(*r) * f64::from(*r)).sum::<f64>() / reference.len() as f64).sqrt();
    if rms == 0.0 {
        return Err("the reference row is all zero, so the gap has no scale".to_owned());
    }
    let max = engine
        .iter()
        .zip(reference)
        .map(|(e, r)| (f64::from(*e) - f64::from(*r)).abs())
        .fold(0.0, f64::max);
    Ok(max / rms)
}

/// The gap of each layer of two sets of rows, in layer order. Each row has `RESIDUAL_WIDTH` values.
pub fn layer_gaps(engine: &[f32], reference: &[f32]) -> Result<Vec<f64>, String> {
    if engine.len() != RESIDUAL_LAYERS * RESIDUAL_WIDTH || reference.len() != engine.len() {
        return Err(format!(
            "each side must hold {RESIDUAL_LAYERS} rows of {RESIDUAL_WIDTH} values; the engine has {} and the reference has {}",
            engine.len(),
            reference.len()
        ));
    }
    engine
        .as_chunks::<RESIDUAL_WIDTH>()
        .0
        .iter()
        .zip(reference.as_chunks::<RESIDUAL_WIDTH>().0)
        .enumerate()
        .map(|(layer, (e, r))| relative_gap(e, r).map_err(|error| format!("layer {layer}: {error}")))
        .collect()
}

/// The layers whose gap is past `gate`. An empty list means that every layer is inside the gate. The gate comes from
/// the caller: the first card run measures it.
pub fn gap_breaches(gaps: &[f64], gate: f64) -> Vec<String> {
    gaps.iter()
        .enumerate()
        .filter(|(_, gap)| gap.is_nan() || **gap > gate)
        .map(|(layer, gap)| format!("layer {layer}: the gap {gap:e} is past the gate {gate:e}"))
        .collect()
}
