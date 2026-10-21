//! The council reads of the Qwen3.8 family: the plans of a plain read and of a queue read, and the raw read.
//!
//! This is the port of mkviolet's `src/template.rs` and of the decision code of `src/council_server.rs`. Where gogi's
//! contract and mkviolet disagree, gogi wins. Each place has a comment with the word "gogi".

use std::collections::HashSet;
use std::sync::Arc;

use gogi_proto::canon::sha256_hex;
use gogi_proto::context::Context;
use gogi_proto::decision::{check_queue_schema, queue_call, queue_tool};
use gogi_proto::spec::{QuestionKind, Spec};
use nagaya_council::family::{
    Advance, Asked, Continues, ControlHit, DecisionInput, FamilyError, RawRead, ReadDone, ReadPlan, ReadStep, Rendering,
};
use serde_json::Value;

use super::messages::{message_json, tools_json, turn_json};
use super::{Qwen38Family, Site, refusal};
use crate::Tokenizer;
use crate::json::Json;

#[cfg(test)]
use super::HookFn;

/// The words that end the case turn of a plain read. The question and its letter follow, as in mkviolet.
const ANSWER: &str = "Answer to question";

// ---------------------------------------------------------------- the raw read

/// The tokens of a raw read. A letter is client text, so it is content and must be one token. A suffix is the
/// client's own structure: its marker text becomes the marker token (gogi "Raw reads").
pub(crate) fn raw_read(family: &Qwen38Family, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
    let mut ids: Vec<u32> = Vec::new();
    for letter in letters {
        let tokens = family.encode(&family.content(letter)?)?;
        let [id] = tokens[..] else {
            return Err(FamilyError::client_param(
                "letters",
                format!(
                    "letter {letter:?} is {} tokens. Each letter must be exactly one token.",
                    tokens.len()
                ),
            ));
        };
        if ids.contains(&id) {
            return Err(FamilyError::client_param(
                "letters",
                format!("letter {letter:?} is the same token as an earlier letter. Send distinct letters."),
            ));
        }
        ids.push(id);
    }
    family.tweak(Site::Raw, &mut ids);
    let mut tokens = Vec::with_capacity(suffixes.len());
    let mut control_text: Vec<ControlHit> = Vec::new();
    for (i, suffix) in suffixes.iter().enumerate() {
        let place = format!("suffix:{i}");
        let mut t = family.tokenizer().encode(suffix).map_err(|e| match refusal(e) {
            FamilyError::Client(invalid) => FamilyError::client_param("suffixes", invalid.message),
            other => other,
        })?;
        family.tweak(Site::Raw, &mut t);
        tokens.push(t);
        control_text.extend(family.hits(&place, suffix));
    }
    Ok(RawRead {
        letters: ids,
        suffixes: tokens,
        control_text,
    })
}

// ---------------------------------------------------------------- the pieces of the rendering

/// Text with its client pieces kept apart, so that the library can report the control text in them.
#[derive(Default)]
struct Layer {
    text: String,
    client: Vec<String>,
}

impl Layer {
    /// Text of the template.
    fn lit(&mut self, s: &str) {
        self.text.push_str(s);
    }

    /// Text of the client.
    fn content(&mut self, s: &str) {
        self.text.push_str(s);
        self.client.push(s.to_string());
    }
}

/// The letter of option `j`.
fn letter(j: usize) -> char {
    char::from(b'A' + u8::try_from(j).expect("an option position is below the letters"))
}

/// The spec layer: the text of the user turn that holds the spec. This is `spec_block` of mkviolet.
fn spec_layer(spec: &Spec) -> Result<Layer, FamilyError> {
    let mut l = Layer::default();
    if !spec.instructions.is_empty() {
        l.content(&spec.instructions);
        l.lit("\n\n");
    }
    l.lit("Answer each question below about the ");
    l.content(&spec.input_label);
    l.lit(" in the next message. Answer question N with the letter of one option, as \"Answer to question N: X\".\n");
    for (i, q) in spec.questions.iter().enumerate() {
        l.lit(&format!("\nQuestion {}. ", i + 1));
        match &q.kind {
            QuestionKind::Choice { options } => {
                l.content(&q.instructions);
                l.lit("\n");
                for (j, (option, means)) in options.iter().enumerate() {
                    l.lit(&format!("{}. ", letter(j)));
                    l.content(option);
                    l.lit(": ");
                    l.content(means);
                    l.lit("\n");
                }
            }
            QuestionKind::Score { levels } => {
                l.content(&q.instructions);
                l.lit(" The options run from lowest to highest.\n");
                for (j, level) in levels.iter().enumerate() {
                    l.lit(&format!("{}. ", letter(j)));
                    l.content(level);
                    l.lit("\n");
                }
            }
            QuestionKind::Noul { criteria } => {
                l.lit("Is this true? ");
                l.content(&q.instructions);
                if let Some(c) = criteria {
                    l.lit("\n");
                    l.content(c);
                }
                l.lit("\nA. yes\nB. no\n");
            }
            QuestionKind::Text { .. } => {
                return Err(FamilyError::internal(format!(
                    "question {} is a text question, which this server cannot ask. The spec should have been refused.",
                    q.id
                )));
            }
        }
    }
    Ok(l)
}

/// The suffix that asks question `number` (counting from 1): ` N (stem):`, where the stem is the first sentence of
/// the question. This puts the question's own words next to its answer (mkviolet `question_suffix`).
fn question_suffix(number: usize, instructions: &str) -> String {
    let stem = match instructions.find(". ") {
        Some(i) => &instructions[..=i],
        None => instructions,
    };
    format!(" {number} ({}):", stem.trim())
}

/// The control text of the pieces, found at `place`.
fn hits_of(family: &Qwen38Family, place: &str, pieces: &[String]) -> Vec<ControlHit> {
    pieces.iter().flat_map(|p| family.hits(place, p)).collect()
}

fn sha_of_escaped(tokenizer: &Tokenizer, escaped: &str) -> Result<String, FamilyError> {
    Ok(sha256_hex(tokenizer.unescape(escaped).map_err(refusal)?.as_bytes()))
}

// ---------------------------------------------------------------- the plan

/// The state of a plan between calls to `advance`.
struct Cursor {
    /// The steps given so far.
    given: usize,
    /// The number of letters of the last step, until the caller answers it.
    awaiting: Option<usize>,
    done: bool,
}

impl Cursor {
    fn new() -> Self {
        Cursor {
            given: 0,
            awaiting: None,
            done: false,
        }
    }

    /// Check a call to `advance`. It gives the letters of the step before, or none for the first call.
    fn accept<'a>(&mut self, last: Option<&'a [f64]>) -> Result<Option<&'a [f64]>, FamilyError> {
        if self.done {
            return Err(FamilyError::internal("advance was called after the plan was done"));
        }
        match (self.awaiting.take(), last) {
            (None, None) => Ok(None),
            (None, Some(_)) => Err(FamilyError::internal(
                "the first call to advance has no letter log probabilities",
            )),
            (Some(_), None) => Err(FamilyError::internal(
                "a later call to advance needs the letter log probabilities",
            )),
            (Some(want), Some(last)) if last.len() != want => Err(FamilyError::internal(format!(
                "the step has {want} letters; advance got {} log probabilities",
                last.len()
            ))),
            (Some(_), Some(last)) if last.iter().any(|x| x.is_nan()) => {
                Err(FamilyError::internal("a letter log probability is NaN"))
            }
            (Some(_), Some(last)) => Ok(Some(last)),
        }
    }

    fn step(&mut self, step: ReadStep) -> Advance {
        self.given += 1;
        self.awaiting = Some(step.letters.len());
        Advance::Step(step)
    }

    fn done(&mut self, rendered_sha256: String) -> Advance {
        self.done = true;
        Advance::Done(ReadDone { rendered_sha256 })
    }
}

/// Check the asked questions against the spec. The client is at fault for a question with no option to read, or with
/// more options than the model has letters. Any other fault is the caller's.
fn check_asked(family: &Qwen38Family, spec: &Spec, asked: &[Asked]) -> Result<(), FamilyError> {
    if asked.is_empty() {
        return Err(FamilyError::internal("a decision read asks at least one question"));
    }
    let letters = family.letters.len();
    let mut seen = HashSet::new();
    for a in asked {
        let Some(q) = spec.questions.get(a.question) else {
            return Err(FamilyError::internal(format!(
                "question {} is not in the spec, which has {}",
                a.question,
                spec.questions.len()
            )));
        };
        if !seen.insert(a.question) {
            return Err(FamilyError::internal(format!("question {} is asked twice", q.id)));
        }
        let keys = q.answer_keys().len();
        if keys == 0 || a.options.is_empty() {
            return Err(FamilyError::client_param(
                "ask",
                format!("question {} has no option to read", q.id),
            ));
        }
        if keys > letters {
            return Err(FamilyError::client_param(
                "spec_id",
                format!(
                    "question {} has {keys} options, and this model has letters for {letters}. Use a spec with at most {letters} options in a question.",
                    q.id
                ),
            ));
        }
        if a.options.windows(2).any(|w| w[0] >= w[1]) || a.options.iter().any(|&j| j >= keys) {
            return Err(FamilyError::internal(format!(
                "the options of question {} are not increasing positions below {keys}",
                q.id
            )));
        }
    }
    Ok(())
}

pub(crate) fn decision_read(
    family: &Qwen38Family,
    input: &DecisionInput<'_>,
) -> Result<Box<dyn ReadPlan>, FamilyError> {
    check_asked(family, input.spec, input.asked)?;
    match input.context {
        Some((c, rendering)) if queue_call(c).is_some() => queue_plan(family, input, c, rendering),
        _ => plain_plan(family, input),
    }
}

/// The tokens of the context that the read goes on from, or none.
fn context_tokens<'a>(input: &DecisionInput<'a>) -> &'a [u32] {
    match input.context {
        Some((_, r)) => &r.tokens[..r.end],
        None => &[],
    }
}

/// The messages and tools of the context of a read. A read with no context has none.
fn context_messages(input: &DecisionInput<'_>) -> (Vec<Json>, Option<Json>) {
    match input.context {
        Some((c, _)) => (c.messages.iter().map(message_json).collect(), tools_json(&c.tools)),
        None => (Vec::new(), None),
    }
}

// ---------------------------------------------------------------- a plain read

/// A read with no queue: each question is read after the same text.
struct PlainPlan {
    prefix: Vec<u32>,
    rendered_sha256: String,
    /// For each asked question: its tokens and its letters.
    steps: Vec<(Vec<u32>, Vec<u32>)>,
    cursor: Cursor,
    control_text: Vec<ControlHit>,
}

fn plain_plan(family: &Qwen38Family, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
    let spec = spec_layer(input.spec)?;
    // The label goes in the case turn too, as in mkviolet.
    let case_turn = format!("{}:\n{}", input.spec.input_label, input.case);
    let mut control_text = hits_of(family, "spec", &spec.client);
    control_text.extend(family.hits("spec", &input.spec.input_label));
    control_text.extend(family.hits("case", input.case));

    let (mut messages, tools) = context_messages(input);
    messages.push(turn_json("user", &spec.text));
    messages.push(turn_json("user", &case_turn));
    // A read closes the thinking, so the case turn ends with the prompt of the template for thinking off.
    let rendered = family.run(&messages, tools.as_ref(), false, true)?;
    let escaped = format!("{}{ANSWER}", rendered.escaped);
    let base = family.encode(&escaped)?;
    let context = context_tokens(input);
    if !base.starts_with(context) {
        return Err(FamilyError::internal(
            "the read does not extend the rendering of its context. The rendering is not the context's.",
        ));
    }
    let mut prefix = base[context.len()..].to_vec();
    family.tweak(Site::Prefix, &mut prefix);

    let mut steps = Vec::new();
    for a in input.asked {
        let q = &input.spec.questions[a.question];
        let suffix = family.content(&question_suffix(a.question + 1, &q.instructions))?;
        let ids = family.encode(&format!("{escaped}{suffix}"))?;
        // The suffix must go on from the tokens of the case, so that the model state is the same.
        if !ids.starts_with(&base) {
            return Err(FamilyError::internal(
                "the question suffix changes the tokens of the case. The rendering has a token border it cannot keep.",
            ));
        }
        let mut feed = ids[base.len()..].to_vec();
        family.tweak(Site::Feed, &mut feed);
        steps.push((feed, a.options.iter().map(|&j| family.letters[j]).collect()));
    }
    Ok(Box::new(PlainPlan {
        prefix,
        // gogi: the hash covers the text that the read was conditioned on. mkviolet hashes the label of the head.
        rendered_sha256: sha_of_escaped(family.tokenizer(), &escaped)?,
        steps,
        cursor: Cursor::new(),
        control_text,
    }))
}

impl ReadPlan for PlainPlan {
    fn prefix(&self) -> &[u32] {
        &self.prefix
    }

    fn control_text(&self) -> &[ControlHit] {
        &self.control_text
    }

    fn advance(&mut self, last: Option<&[f64]>) -> Result<Advance, FamilyError> {
        self.cursor.accept(last)?;
        let Some((feed, letters)) = self.steps.get(self.cursor.given) else {
            return Ok(self.cursor.done(self.rendered_sha256.clone()));
        };
        let step = ReadStep {
            question: self.cursor.given,
            continues: Continues::Prefix,
            feed: feed.clone(),
            letters: letters.clone(),
        };
        Ok(self.cursor.step(step))
    }
}

// ---------------------------------------------------------------- a queue read

struct QueueStep {
    /// The index in `asked`.
    slot: usize,
    /// The question id, escaped.
    id: String,
    options: Vec<usize>,
}

/// A read of a queue context: the case is the result of the queue call, and the model writes the next call.
struct QueuePlan {
    tokenizer: Arc<Tokenizer>,
    #[cfg(test)]
    hook: Option<HookFn>,
    prefix: Vec<u32>,
    /// All the text of the read so far, escaped.
    escaped: String,
    /// The tokens of that text.
    tokens: Vec<u32>,
    /// The token of each bare letter.
    bare_letters: Vec<u32>,
    /// The steps, in the order of the properties of `answers` in the queue tool's schema.
    steps: Vec<QueueStep>,
    cursor: Cursor,
    control_text: Vec<ControlHit>,
}

/// The names of the properties of `answers` in the queue tool's schema, in the order of the schema. The template
/// writes the keys of an object in the order it gets them, so this is the order that the model writes them in.
/// (mkviolet sorts by question id, because Gemma 4's template sorts keys.)
fn answer_keys(c: &Context) -> Vec<&str> {
    let Some(tool) = queue_tool(c) else {
        return Vec::new();
    };
    let props = tool.input_schema.get("properties").unwrap_or(&Value::Null);
    match &props["answers"]["properties"] {
        Value::Object(map) => map.keys().map(String::as_str).collect(),
        _ => Vec::new(),
    }
}

fn queue_plan(
    family: &Qwen38Family,
    input: &DecisionInput<'_>,
    c: &Context,
    rendering: &Rendering,
) -> Result<Box<dyn ReadPlan>, FamilyError> {
    let questions: Vec<(&str, usize)> = input
        .asked
        .iter()
        .map(|a| {
            let q = &input.spec.questions[a.question];
            (q.id.as_str(), q.answer_keys().len())
        })
        .collect();
    check_queue_schema(c, &questions)?;
    let schema_order = answer_keys(c);
    let mut order: Vec<usize> = (0..input.asked.len()).collect();
    order.sort_by_key(|&slot| {
        let id = input.spec.questions[input.asked[slot].question].id.as_str();
        schema_order
            .iter()
            .position(|k| *k == id)
            .expect("check_queue_schema found the property")
    });

    // The template matches a result to its call by order, so the call's id is not rendered.
    let (mut messages, tools) = context_messages(input);
    messages.push(turn_json("tool", input.case));
    let rendered = family.run(&messages, tools.as_ref(), false, true)?;
    let mut escaped = rendered.escaped;
    escaped.push_str("<tool_call>\n<function=");
    // The one caller takes this path only when `queue_call(c)` is `Some`, and that needs `c.queue` (gogi `decision.rs`).
    // `Context::check` has refused a queue that names no tool, as `DecisionRequest::validate` calls it.
    escaped.push_str(&family.content(c.queue.as_deref().expect("a queue context names its queue"))?);
    escaped.push_str(">\n<parameter=answers>\n");
    let tokens = family.encode(&escaped)?;
    let context = &rendering.tokens[..rendering.end];
    if !tokens.starts_with(context) {
        return Err(FamilyError::internal(
            "the queue read does not extend the rendering of its context. The rendering is not the context's.",
        ));
    }
    let mut prefix = tokens[context.len()..].to_vec();
    family.tweak(Site::Prefix, &mut prefix);

    let steps = order
        .into_iter()
        .map(|slot| {
            Ok(QueueStep {
                slot,
                id: family.content(&input.spec.questions[input.asked[slot].question].id)?,
                options: input.asked[slot].options.clone(),
            })
        })
        .collect::<Result<_, FamilyError>>()?;
    Ok(Box::new(QueuePlan {
        tokenizer: family.shared_tokenizer(),
        #[cfg(test)]
        hook: family.hook,
        prefix,
        escaped,
        tokens,
        bare_letters: family.bare_letters.clone(),
        steps,
        cursor: Cursor::new(),
        // gogi: the spec is not in a queue read, so only the case can spell control text.
        control_text: family.hits("case", input.case),
    }))
}

impl QueuePlan {
    #[cfg_attr(not(test), allow(unused_variables))]
    fn tweak(&self, site: Site, tokens: &mut [u32]) {
        #[cfg(test)]
        if let Some(hook) = self.hook {
            hook(site, tokens);
        }
    }
}

impl ReadPlan for QueuePlan {
    fn prefix(&self) -> &[u32] {
        &self.prefix
    }

    fn control_text(&self) -> &[ControlHit] {
        &self.control_text
    }

    fn advance(&mut self, last: Option<&[f64]>) -> Result<Advance, FamilyError> {
        let last = self.cursor.accept(last)?;
        let mut chosen = None;
        if let Some(last) = last {
            // The step before chose its likeliest letter. The first of equal letters wins.
            let before = &self.steps[self.cursor.given - 1];
            let best = (0..last.len()).fold(0, |b, i| if last[i] > last[b] { i } else { b });
            self.escaped.push(letter(before.options[best]));
            self.escaped.push('"');
            chosen = Some(self.bare_letters[before.options[best]]);
        }
        let Some(step) = self.steps.get(self.cursor.given) else {
            // The hash covers all the text the read was conditioned on, up to the last letter and its quote (gogi).
            let sha256 = sha_of_escaped(&self.tokenizer, &self.escaped)?;
            return Ok(self.cursor.done(sha256));
        };
        self.escaped
            .push_str(if self.cursor.given == 0 { "{\"" } else { ", \"" });
        self.escaped.push_str(&step.id);
        self.escaped.push_str("\": \"");
        let ids = self.tokenizer.encode_escaped(&self.escaped).map_err(refusal)?;
        if !ids.starts_with(&self.tokens) {
            return Err(FamilyError::internal(
                "the queue call changes the tokens before it. The rendering has a token border it cannot keep.",
            ));
        }
        let mut feed = ids[self.tokens.len()..].to_vec();
        // The chosen letter is the first token to feed, and it is the bare letter token that was read.
        if let Some(letter) = chosen
            && feed.first() != Some(&letter)
        {
            return Err(FamilyError::internal(
                "the letter that was read is not the token that the text gives. The queue call has a token border it cannot keep.",
            ));
        }
        self.tokens = ids;
        self.tweak(Site::Feed, &mut feed);
        let letters = step.options.iter().map(|&j| self.bare_letters[j]).collect();
        let step = ReadStep {
            question: step.slot,
            continues: Continues::PreviousStep,
            feed,
            letters,
        };
        Ok(self.cursor.step(step))
    }
}
