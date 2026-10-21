//! The pure parts of a decision (gogi "Decisions", "Answers"): the asked questions, and the answers of the reads and
//! their pool.
//!
//! The numbers come from `gogi_proto::math`, which every server shares (spec 11.1).

use std::collections::HashSet;

use gogi_proto::Invalid;
use gogi_proto::decision::{
    Pool, PoolEcho, PoolMethod, PoolWeights, PooledAnswer, PooledChoice, PooledNoul, PooledScore, QuestionId,
    ReadAnswer, ReadChoice, ReadNoul, ReadScore,
};
use gogi_proto::math::{self, MathError, Row, WeightSpec};
use gogi_proto::spec::{QuestionKind, Spec};
use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::family::Asked;

/// The asked questions of a decision, in spec order, each with the positions of its kept options.
///
/// `ask` names some of the spec's questions. With no `ask`, the decision asks every question. `options` narrows a
/// `choice` question to at least 2 of its options. Narrowing never adds or rewords a question (gogi "Decisions").
pub fn asked(spec: &Spec, ask: Option<&[String]>, options: Option<&Map<String, Value>>) -> Result<Vec<Asked>, Invalid> {
    let mut questions: Vec<usize> = match ask {
        None => (0..spec.questions.len())
            .filter(|&index| !matches!(spec.questions[index].kind, QuestionKind::Text { .. }))
            .collect(),
        Some(ids) => {
            let mut out = Vec::with_capacity(ids.len());
            for id in ids {
                let Some(index) = spec.questions.iter().position(|q| &q.id == id) else {
                    return Err(Invalid::param(
                        "ask",
                        format!("the spec has no question {id}; ask its questions"),
                    ));
                };
                if matches!(spec.questions[index].kind, QuestionKind::Text { .. }) {
                    return Err(Invalid::param(
                        "ask",
                        format!("question {id} is a text question, which needs the describe capability"),
                    ));
                }
                if out.contains(&index) {
                    return Err(Invalid::param(
                        "ask",
                        format!("ask repeats question {id}; name it once"),
                    ));
                }
                out.push(index);
            }
            out
        }
    };
    if questions.is_empty() {
        return Err(Invalid::param("ask", "the decision asks no question; ask at least one"));
    }
    questions.sort_unstable();
    let mut kept: Vec<Vec<usize>> = questions
        .iter()
        .map(|&index| (0..spec.questions[index].answer_keys().len()).collect())
        .collect();
    for (id, list) in options.into_iter().flatten() {
        let Some(index) = spec.questions.iter().position(|q| &q.id == id) else {
            return Err(Invalid::param("options", format!("the spec has no question {id}")));
        };
        let QuestionKind::Choice { options: choices } = &spec.questions[index].kind else {
            return Err(Invalid::param("options", format!("{id} is not a choice question")));
        };
        let Some(slot) = questions.iter().position(|&q| q == index) else {
            return Err(Invalid::param("options", format!("{id} is not an asked question")));
        };
        let names: Option<Vec<&str>> = list.as_array().and_then(|a| a.iter().map(Value::as_str).collect());
        let Some(names) = names else {
            return Err(Invalid::param(
                "options",
                format!("options.{id} is a list of option names"),
            ));
        };
        if names.len() < 2 || names.iter().collect::<HashSet<_>>().len() != names.len() {
            return Err(Invalid::param(
                "options",
                format!(
                    "options.{id} names {} options; narrow to at least 2 distinct options",
                    names.len()
                ),
            ));
        }
        let mut positions = Vec::with_capacity(names.len());
        for name in names {
            match choices.iter().position(|(option, _)| option == name) {
                Some(position) => positions.push(position),
                None => return Err(Invalid::param("options", format!("{id} has no option {name}"))),
            }
        }
        positions.sort_unstable();
        kept[slot] = positions;
    }
    Ok(questions
        .into_iter()
        .zip(kept)
        .map(|(question, options)| Asked { question, options })
        .collect())
}

/// The answers of a decision: of each read, and pooled over the reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Answers {
    /// The pooled answer of each asked question, in asked order.
    pub pooled: IndexMap<QuestionId, PooledAnswer>,
    /// The answers of each read, in read order.
    pub reads: Vec<IndexMap<QuestionId, ReadAnswer>>,
    /// The pool settings, with the weight of each read.
    pub pool: PoolEcho,
}

/// Compute the answers from the raw numbers of the reads.
///
/// `logprobs[r][a]` holds the log probability of each kept option of asked question `a` in read `r`, in the order of
/// `Asked::options`. A pooled answer of two or more reads has `leave_one_out`, keyed by the index of the context.
/// Only a decision with contexts has two or more reads.
pub fn answers(spec: &Spec, asked: &[Asked], logprobs: &[Vec<Vec<f64>>], pool: &Pool) -> Result<Answers, MathError> {
    let method = pool.method.unwrap_or(PoolMethod::Linear);
    let weights = pool.weights.unwrap_or(PoolWeights::Uniform);
    let spec_weights = match weights {
        PoolWeights::Uniform => WeightSpec::Uniform,
        PoolWeights::Mass => WeightSpec::Mass,
        PoolWeights::Given => WeightSpec::Given(pool.values.clone().unwrap_or_default()),
    };
    let mut pooled = IndexMap::new();
    let mut reads: Vec<IndexMap<QuestionId, ReadAnswer>> = vec![IndexMap::new(); logprobs.len()];
    let mut normalized = IndexMap::new();
    for (slot, a) in asked.iter().enumerate() {
        let question = &spec.questions[a.question];
        let all = question.answer_keys();
        let keys: Vec<String> = a.options.iter().map(|&o| all[o].clone()).collect();
        let raw: Vec<&[f64]> = logprobs.iter().map(|read| read[slot].as_slice()).collect();
        let rows = raw
            .iter()
            .map(|lp| Row::from_logprobs(lp))
            .collect::<Result<Vec<Row>, MathError>>()?;
        let p = math::pool(&rows, method, &spec_weights)?;
        normalized.insert(question.id.clone(), p.weights.clone());
        let map =
            |values: &[f64]| -> IndexMap<String, f64> { keys.iter().cloned().zip(values.iter().copied()).collect() };
        let legend = |levels: &[String]| -> IndexMap<String, String> {
            levels
                .iter()
                .enumerate()
                .map(|(k, level)| (k.to_string(), level.clone()))
                .collect()
        };
        for (r, row) in rows.iter().enumerate() {
            let answer = match &question.kind {
                QuestionKind::Choice { .. } => ReadAnswer::Choice(ReadChoice {
                    choice: keys[math::argmax(&row.probs)].clone(),
                    probabilities: map(&row.probs),
                    confidence: math::confidence(row.mass, &row.probs),
                    logprobs: map(raw[r]),
                    mass: row.mass,
                }),
                QuestionKind::Score { levels } => ReadAnswer::Score(ReadScore {
                    score: math::score(&row.probs),
                    legend: legend(levels),
                    probabilities: map(&row.probs),
                    confidence: math::confidence(row.mass, &row.probs),
                    logprobs: map(raw[r]),
                    mass: row.mass,
                }),
                QuestionKind::Noul { .. } => ReadAnswer::Noul(ReadNoul {
                    noul: row.probs[0],
                    logprobs: map(raw[r]),
                    mass: row.mass,
                }),
                QuestionKind::Text { .. } => unreachable!("`asked` never asks a text question"),
            };
            reads[r].insert(question.id.clone(), answer);
        }
        let loo = p.leave_one_out.as_ref();
        let key = |r: usize| r.to_string();
        let answer = match &question.kind {
            QuestionKind::Choice { .. } => PooledAnswer::Choice(PooledChoice {
                choice: keys[math::argmax(&p.probs)].clone(),
                probabilities: map(&p.probs),
                confidence: p.confidence(),
                agree: p.agree,
                spread: p.spread,
                leave_one_out: loo.map(|l| {
                    l.iter()
                        .enumerate()
                        .map(|(r, v)| (key(r), v.as_deref().map(map)))
                        .collect()
                }),
            }),
            QuestionKind::Score { levels } => PooledAnswer::Score(PooledScore {
                score: math::score(&p.probs),
                legend: legend(levels),
                probabilities: map(&p.probs),
                confidence: p.confidence(),
                agree: p.agree,
                spread: p.spread,
                leave_one_out: loo.map(|l| {
                    l.iter()
                        .enumerate()
                        .map(|(r, v)| (key(r), v.as_deref().map(map)))
                        .collect()
                }),
            }),
            QuestionKind::Noul { .. } => PooledAnswer::Noul(PooledNoul {
                noul: p.probs[0],
                agree: p.agree,
                spread: p.spread,
                leave_one_out: loo.map(|l| {
                    l.iter()
                        .enumerate()
                        .map(|(r, v)| (key(r), v.as_ref().map(|v| v[0])))
                        .collect()
                }),
            }),
            QuestionKind::Text { .. } => unreachable!("`asked` never asks a text question"),
        };
        pooled.insert(question.id.clone(), answer);
    }
    Ok(Answers {
        pooled,
        reads,
        pool: PoolEcho {
            method,
            weights,
            normalized,
        },
    })
}

/// The text of a case: a string verbatim, and an object or an array as JSON in request order (gogi "Decisions").
pub fn case_text(case: &Value) -> String {
    match case {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}
