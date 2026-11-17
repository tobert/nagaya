//! The council reads of the Gemma 4 family: the plans of a plain read and of a queue read, and the raw read.
//!
//! This is the port of the template and the decision code of the first Gemma 4 engine. Where gogi's contract
//! and the first engine disagree, gogi wins. Each place has a comment with the word "gogi".

use std::collections::HashSet;

use gogi_proto::canon::content_id;
use gogi_proto::context::Context;
use gogi_proto::decision::{check_queue_schema, queue_call};
use gogi_proto::spec::{QuestionKind, Spec};
use nagaya_council::family::{
    Advance, Asked, Continues, ControlHit, DecisionInput, FamilyError, RawRead, ReadDone, ReadPlan, ReadStep, Rendering,
};
use nagaya_jinja::Json;

use super::messages::{message_json, tools_json};
use super::{Gemma4Family, client};
use crate::tokenizer::{BOS, check_text};

/// The words that end the case turn of a plain read. The question and its letter follow, as in the first Gemma 4
/// engine.
const ANSWER: &str = "Answer to question";

/// The quote of a string in a Gemma 4 call.
const QUOTE: &str = "<|\"|>";

// ---------------------------------------------------------------- the raw read

/// The tokens of a raw read. A letter is client text, so it is content and must be one token. A suffix is the
/// client's own structure: its marker text becomes the marker token (gogi "Raw reads").
pub(crate) fn raw_read(family: &Gemma4Family, letters: &[String], suffixes: &[String]) -> Result<RawRead, FamilyError> {
    let mut ids: Vec<u32> = Vec::new();
    for letter in letters {
        let tokens = family.content(letter).map_err(|e| param("letters", e))?;
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
    let mut tokens = Vec::with_capacity(suffixes.len());
    let mut control_text: Vec<ControlHit> = Vec::new();
    for (i, suffix) in suffixes.iter().enumerate() {
        check_text(suffix).map_err(|e| param("suffixes", client(e)))?;
        tokens.push(family.tpl(suffix)?);
        control_text.extend(family.hits(&format!("suffix:{i}"), suffix));
    }
    Ok(RawRead {
        letters: ids,
        suffixes: tokens,
        control_text,
    })
}

/// A client error that names the request field `name`.
fn param(name: &str, e: FamilyError) -> FamilyError {
    match e {
        FamilyError::Client(invalid) => FamilyError::client_param(name, invalid.message),
        other => other,
    }
}

// ---------------------------------------------------------------- the pieces of the rendering

/// Rendered pieces: template text with special tokens on, client text with them off.
struct Pieces<'a> {
    family: &'a Gemma4Family,
    tokens: Vec<u32>,
    text: String,
    /// The control text that the client text spells, with the place of each.
    hits: Vec<ControlHit>,
}

impl<'a> Pieces<'a> {
    fn new(family: &'a Gemma4Family) -> Self {
        Pieces {
            family,
            tokens: Vec::new(),
            text: String::new(),
            hits: Vec::new(),
        }
    }

    /// Text of the template.
    fn tpl(&mut self, s: &str) -> Result<(), FamilyError> {
        self.tokens.extend(self.family.tpl(s)?);
        self.text.push_str(s);
        Ok(())
    }

    /// Text of the client, found at `place`. Empty text adds nothing.
    fn content(&mut self, s: &str, place: &str) -> Result<(), FamilyError> {
        if s.is_empty() {
            return Ok(());
        }
        self.tokens.extend(self.family.content(s)?);
        self.text.push_str(s);
        self.hits.extend(self.family.hits(place, s));
        Ok(())
    }
}

/// The letter of option `j`.
fn letter(j: usize) -> char {
    char::from(b'A' + u8::try_from(j).expect("an option position is below the letters"))
}

/// The spec layer: the user turn that holds the spec.
fn spec_layer(p: &mut Pieces<'_>, spec: &Spec) -> Result<(), FamilyError> {
    p.tpl("<|turn>user\n")?;
    if !spec.instructions.is_empty() {
        p.content(&spec.instructions, "spec")?;
        p.tpl("\n\n")?;
    }
    p.tpl("Answer each question below about the ")?;
    p.content(&spec.input_label, "spec")?;
    p.tpl(" in the next message. Answer question N with the letter of one option, as \"Answer to question N: X\".\n")?;
    for (i, q) in spec.questions.iter().enumerate() {
        p.tpl(&format!("\nQuestion {}. ", i + 1))?;
        match &q.kind {
            QuestionKind::Choice { options } => {
                p.content(&q.instructions, "spec")?;
                p.tpl("\n")?;
                for (j, (option, means)) in options.iter().enumerate() {
                    p.tpl(&format!("{}. ", letter(j)))?;
                    p.content(option, "spec")?;
                    p.tpl(": ")?;
                    p.content(means, "spec")?;
                    p.tpl("\n")?;
                }
            }
            QuestionKind::Score { levels } => {
                p.content(&q.instructions, "spec")?;
                p.tpl(" The options run from lowest to highest.\n")?;
                for (j, level) in levels.iter().enumerate() {
                    p.tpl(&format!("{}. ", letter(j)))?;
                    p.content(level, "spec")?;
                    p.tpl("\n")?;
                }
            }
            QuestionKind::Noul { criteria } => {
                p.tpl("Is this true? ")?;
                p.content(&q.instructions, "spec")?;
                if let Some(c) = criteria {
                    p.tpl("\n")?;
                    p.content(c, "spec")?;
                }
                p.tpl("\nA. yes\nB. no\n")?;
            }
            QuestionKind::Text { .. } => {
                return Err(FamilyError::internal(format!(
                    "question {} is a text question, which this server cannot ask. The spec should have been refused.",
                    q.id
                )));
            }
        }
    }
    p.tpl("<turn|>\n")
}

/// The case turn, and the start of the answer with the thinking closed.
fn case_layer(p: &mut Pieces<'_>, label: &str, case: &str) -> Result<(), FamilyError> {
    p.tpl("<|turn>user\n")?;
    p.content(label, "spec")?;
    p.tpl(":\n")?;
    p.content(case, "case")?;
    p.tpl(&format!("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{ANSWER}"))
}

/// The tokens that ask question `number` (counting from 1): ` N (stem):`, where the stem is the first sentence of
/// the question. This puts the question's own words next to its answer.
fn question_suffix(family: &Gemma4Family, number: usize, instructions: &str) -> Result<Vec<u32>, FamilyError> {
    let stem = match instructions.find(". ") {
        Some(i) => &instructions[..=i],
        None => instructions,
    };
    let mut p = Pieces::new(family);
    p.tpl(&format!(" {number} ("))?;
    p.content(stem.trim(), "spec")?;
    p.tpl("):")?;
    Ok(p.tokens)
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

    fn done(&mut self, rendered_hash: String) -> Advance {
        self.done = true;
        Advance::Done(ReadDone { rendered_hash })
    }
}

/// Check the asked questions against the spec. The client is at fault for a question with no option to read, or with
/// more options than the model has letters. Any other fault is the caller's.
fn check_asked(family: &Gemma4Family, spec: &Spec, asked: &[Asked]) -> Result<(), FamilyError> {
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
    family: &Gemma4Family,
    input: &DecisionInput<'_>,
) -> Result<Box<dyn ReadPlan>, FamilyError> {
    check_asked(family, input.spec, input.asked)?;
    match input.context {
        Some((c, rendering)) if queue_call(c).is_some() => queue_plan(family, input, c, rendering),
        _ => plain_plan(family, input),
    }
}

/// The context of a read rendered again, with no generation prompt: its text and its tokens. The tokens must be the
/// tokens of the rendering that the caller holds.
fn context_again(family: &Gemma4Family, c: &Context, rendering: &Rendering) -> Result<String, FamilyError> {
    let messages: Vec<Json> = c.messages.iter().map(message_json).collect();
    let again = family.run(&messages, &tools_json(&c.tools), c.thinking, false)?;
    if again.tokens[..] != rendering.tokens[..rendering.end] {
        return Err(FamilyError::internal(
            "the rendering of the context is not the rendering that the family gives. The rendering is not the context's.",
        ));
    }
    Ok(again.text)
}

// ---------------------------------------------------------------- a plain read

/// A read with no queue: each question is read after the same text.
struct PlainPlan {
    prefix: Vec<u32>,
    rendered_hash: String,
    /// For each asked question: its tokens and its letters.
    steps: Vec<(Vec<u32>, Vec<u32>)>,
    cursor: Cursor,
    control_text: Vec<ControlHit>,
}

fn plain_plan(family: &Gemma4Family, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
    let mut p = Pieces::new(family);
    let context_text = match input.context {
        Some((c, rendering)) => context_again(family, c, rendering)?,
        None => {
            // A read with no context starts with `<bos>` alone, as in the first engine.
            p.tpl(BOS)?;
            String::new()
        }
    };
    spec_layer(&mut p, input.spec)?;
    case_layer(&mut p, &input.spec.input_label, input.case)?;
    let mut steps = Vec::new();
    for a in input.asked {
        let q = &input.spec.questions[a.question];
        let feed = question_suffix(family, a.question + 1, &q.instructions)?;
        steps.push((feed, a.options.iter().map(|&j| family.letters[j]).collect()));
    }
    // The spec layer reports the control text of each question in full, so a stem adds no hit of its own.
    let control_text = p.hits;
    Ok(Box::new(PlainPlan {
        prefix: p.tokens,
        // gogi: the hash covers the text that the read was conditioned on. The first engine hashed the id of the head.
        rendered_hash: content_id(format!("{context_text}{}", p.text).as_bytes()),
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
            return Ok(self.cursor.done(self.rendered_hash.clone()));
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
    /// The tokens of `{id}:<|"|>`, after the letter of the step before.
    key: Vec<u32>,
    /// The text of the same.
    key_text: String,
    options: Vec<usize>,
}

/// A read of a queue context: the case is the result of the queue call, and the model writes the next call.
struct QueuePlan {
    family_tokens: QueueTokens,
    prefix: Vec<u32>,
    /// All the text of the read so far.
    text: String,
    /// The steps, in the order of the question ids, as the template sorts the keys of a map.
    steps: Vec<QueueStep>,
    cursor: Cursor,
    control_text: Vec<ControlHit>,
}

/// The tokens that a queue plan needs after it leaves the family.
struct QueueTokens {
    /// The token of each bare letter.
    bare_letters: Vec<u32>,
    /// The tokens of `<|"|>`, and of `<|"|>,`.
    close: Vec<u32>,
    close_more: Vec<u32>,
}

fn queue_plan(
    family: &Gemma4Family,
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
    // The first engine sorted by question id, because Google's template sorts the keys of a map (`dictsort`).
    let mut order: Vec<usize> = (0..input.asked.len()).collect();
    order.sort_by(|&x, &y| {
        let id = |slot: usize| input.spec.questions[input.asked[slot].question].id.as_str();
        id(x).cmp(id(y))
    });

    let call_id = queue_call(c).expect("the caller checked the queue call").id.clone();
    let mut messages: Vec<Json> = c.messages.iter().map(message_json).collect();
    messages.push(Json::Object(vec![
        ("role".to_string(), Json::Str("tool".to_string())),
        ("tool_call_id".to_string(), Json::Str(call_id)),
        ("content".to_string(), Json::Str(input.case.to_string())),
    ]));
    let thinking = c.thinking;
    let case = family.run(&messages, &tools_json(&c.tools), thinking, true)?;
    let context = &rendering.tokens[..rendering.end];
    if !case.tokens.starts_with(context) {
        return Err(FamilyError::internal(
            "the queue read does not extend the rendering of its context. The rendering is not the context's.",
        ));
    }
    // With thinking on, the generation prompt after a tool result opens the thought channel. The read closes it.
    let queue = c.queue.as_deref().expect("a queue context names its queue");
    let mut p = Pieces::new(family);
    p.tpl(if thinking {
        "<channel|><|tool_call>call:"
    } else {
        "<|tool_call>call:"
    })?;
    p.content(queue, "queue")?;
    p.tpl("{answers:{")?;
    let mut prefix = case.tokens[context.len()..].to_vec();
    prefix.extend(&p.tokens);

    let steps = order
        .into_iter()
        .map(|slot| {
            let mut k = Pieces::new(family);
            k.content(&input.spec.questions[input.asked[slot].question].id, "spec")?;
            k.tpl(&format!(":{QUOTE}"))?;
            Ok(QueueStep {
                slot,
                key: k.tokens,
                key_text: k.text,
                options: input.asked[slot].options.clone(),
            })
        })
        .collect::<Result<_, FamilyError>>()?;
    Ok(Box::new(QueuePlan {
        family_tokens: QueueTokens {
            bare_letters: family.bare_letters.clone(),
            close: family.tpl(QUOTE)?,
            close_more: family.tpl(&format!("{QUOTE},"))?,
        },
        prefix,
        text: format!("{}{}", case.text, p.text),
        steps,
        cursor: Cursor::new(),
        // gogi: the spec is not in a queue read, so only the case can spell control text.
        control_text: family.hits("case", input.case),
    }))
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
        let mut feed = Vec::new();
        if let Some(last) = last {
            // The step before chose its likeliest letter. The first of equal letters wins.
            let before = &self.steps[self.cursor.given - 1];
            let best = (0..last.len()).fold(0, |b, i| if last[i] > last[b] { i } else { b });
            let more = self.cursor.given < self.steps.len();
            feed.push(self.family_tokens.bare_letters[before.options[best]]);
            feed.extend(if more {
                &self.family_tokens.close_more
            } else {
                &self.family_tokens.close
            });
            self.text.push(letter(before.options[best]));
            self.text.push_str(QUOTE);
            if more {
                self.text.push(',');
            }
        }
        let Some(step) = self.steps.get(self.cursor.given) else {
            // gogi: the hash covers all the text the read was conditioned on, up to the last letter and its quote.
            // The first engine hashed the text up to the call's opening brace.
            return Ok(self.cursor.done(content_id(self.text.as_bytes())));
        };
        feed.extend(&step.key);
        self.text.push_str(&step.key_text);
        let letters = step
            .options
            .iter()
            .map(|&j| self.family_tokens.bare_letters[j])
            .collect();
        let step = ReadStep {
            question: step.slot,
            continues: Continues::PreviousStep,
            feed,
            letters,
        };
        Ok(self.cursor.step(step))
    }
}
