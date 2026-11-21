//! The council reads of the Inkling 2 family: plain reads, queue reads, and raw reads (spec 11.3a).

use std::collections::HashSet;

use gogi_proto::context::Context;
use gogi_proto::decision::{check_queue_schema, queue_call, queue_tool};
use gogi_proto::spec::{QuestionKind, Spec};
use nagaya_council::family::{
    Advance, Asked, Continues, ControlHit, DecisionInput, FamilyError, RawRead, ReadDone, ReadPlan, ReadStep, Rendering,
};

use super::Inkling2Family;
use super::messages::{messages_json, tools_json};
use super::refusal;
use crate::tokenizer::{Tokenizer, check_text};

const ANSWER: &str = "Answer:";

pub(crate) fn raw_read(
    family: &Inkling2Family,
    letters: &[String],
    suffixes: &[String],
) -> Result<RawRead, FamilyError> {
    let mut ids: Vec<u32> = Vec::new();
    for letter in letters {
        let tokens = family.encode(&family.escaped(letter)?)?;
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
        let place = format!("suffix:{i}");
        let t = check_text(suffix)
            .and_then(|()| family.tokenizer().encode(suffix))
            .map_err(|e| match refusal(e) {
                FamilyError::Client(invalid) => FamilyError::client_param("suffixes", invalid.message),
                other => other,
            })?;
        tokens.push(t);
        control_text.extend(family.hits(&place, suffix));
    }
    Ok(RawRead {
        letters: ids,
        suffixes: tokens,
        control_text,
    })
}

#[derive(Default)]
struct Layer {
    text: String,
    question_client: Vec<String>,
}

impl Layer {
    fn lit(&mut self, s: &str) {
        self.text.push_str(s);
    }

    fn shared(&mut self, s: &str) {
        self.text.push_str(s);
    }

    fn content(&mut self, s: &str) {
        self.text.push_str(s);
        self.question_client.push(s.to_string());
    }
}

fn letter(j: usize) -> char {
    char::from(b'A' + u8::try_from(j).expect("an option position is below the letters"))
}

fn question_layer(spec: &Spec, question: usize) -> Result<Layer, FamilyError> {
    let q = &spec.questions[question];
    let mut l = Layer::default();
    if !spec.instructions.is_empty() {
        l.shared(&spec.instructions);
        l.lit("\n\n");
    }
    l.lit("Answer this question about the ");
    l.shared(&spec.input_label);
    l.lit(" with the letter of one option, as \"Answer: X\".\n\n");
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
                "question {} is a text question, which this server cannot ask.",
                q.id
            )));
        }
    }
    Ok(l)
}

fn hits_of(family: &Inkling2Family, place: &str, pieces: &[String]) -> Vec<ControlHit> {
    pieces.iter().flat_map(|p| family.hits(place, p)).collect()
}

fn hash_of_escaped(tokenizer: &Tokenizer, escaped: &str) -> Result<String, FamilyError> {
    Ok(gogi_proto::canon::content_id(
        tokenizer.unescape(escaped).map_err(refusal)?.as_bytes(),
    ))
}

fn check_asked(family: &Inkling2Family, spec: &Spec, asked: &[Asked]) -> Result<(), FamilyError> {
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

struct Cursor {
    given: usize,
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

    fn accept<'a>(&mut self, last: Option<&'a [f64]>) -> Result<Option<&'a [f64]>, FamilyError> {
        if self.done {
            return Err(FamilyError::internal("ReadPlan::advance called after Done"));
        }
        match (self.awaiting, last) {
            (None, Some(_)) => Err(FamilyError::internal("first ReadPlan::advance got logprobs")),
            (Some(_), None) => Err(FamilyError::internal("ReadPlan::advance missing logprobs")),
            (Some(want), Some(got)) if got.len() != want => Err(FamilyError::internal(format!(
                "ReadPlan::advance got {} logprobs, expected {want}",
                got.len()
            ))),
            (Some(_), Some(got)) if got.iter().any(|x| x.is_nan()) => {
                Err(FamilyError::internal("a letter log probability is NaN"))
            }
            (None, None) => Ok(None),
            (Some(_), Some(got)) => Ok(Some(got)),
        }
    }
}

pub(crate) fn decision_read(
    family: &Inkling2Family,
    input: &DecisionInput<'_>,
) -> Result<Box<dyn ReadPlan>, FamilyError> {
    check_asked(family, input.spec, input.asked)?;
    if let Some((context, rendering)) = input.context
        && queue_call(context).is_some()
    {
        return queue_read(family, context, rendering, input.spec, input.case, input.asked);
    }
    plain_read(family, input)
}

struct PlainStep {
    feed: Vec<u32>,
    letters: Vec<u32>,
}

struct PlainPlan {
    control_text: Vec<ControlHit>,
    steps: Vec<PlainStep>,
    cursor: Cursor,
    rendered_hash: String,
    room: usize,
}

impl ReadPlan for PlainPlan {
    fn prefix(&self) -> &[u32] {
        &[]
    }

    fn control_text(&self) -> &[ControlHit] {
        &self.control_text
    }

    fn room(&self) -> usize {
        self.room
    }

    fn advance(&mut self, last: Option<&[f64]>) -> Result<Advance, FamilyError> {
        self.cursor.accept(last)?;
        if self.cursor.given == self.steps.len() {
            self.cursor.done = true;
            return Ok(Advance::Done(ReadDone {
                rendered_hash: self.rendered_hash.clone(),
            }));
        }
        let q_idx = self.cursor.given;
        let step = &self.steps[q_idx];
        self.cursor.given += 1;
        self.cursor.awaiting = Some(step.letters.len());
        Ok(Advance::Step(ReadStep {
            question: q_idx,
            feed: step.feed.clone(),
            letters: step.letters.clone(),
            continues: Continues::Prefix,
        }))
    }
}

fn plain_read(family: &Inkling2Family, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
    let mut base_escaped = String::new();
    let base_tokens = match input.context {
        Some((ctx, rendering)) => {
            let msgs = messages_json(&ctx.messages)?;
            let tools = tools_json(&ctx.tools);
            let rendered = family.run(&msgs, &tools)?;
            if !rendered.tokens.starts_with(&rendering.tokens[..rendering.end]) {
                return Err(FamilyError::internal(
                    "the context rendering does not match the context tokens; render the context again",
                ));
            }
            base_escaped = rendered.escaped;
            rendered.tokens
        }
        None => Vec::new(),
    };
    let mut control_text = Vec::new();
    if !input.spec.instructions.is_empty() {
        control_text.extend(family.hits("spec", &input.spec.instructions));
    }
    control_text.extend(family.hits("spec", &input.spec.input_label));
    control_text.extend(family.hits("case", input.case));

    let escaped_case = family.escaped(input.case)?;
    let escaped_label = family.escaped(&input.spec.input_label)?;
    let mut steps = Vec::with_capacity(input.asked.len());
    let mut max_feed = 0usize;
    let mut hash_text = String::new();
    for asked in input.asked {
        let q = question_layer(input.spec, asked.question)?;
        control_text.extend(hits_of(family, "spec", &q.question_client));
        let escaped_q = family.escaped(&q.text)?;
        let step_escaped = format!(
            "<|message_user|><|content_text|>{escaped_q}<|end_message|><|message_model|><|content_text|>{escaped_label}:\n{escaped_case}<|end_message|><|message_model|><|content_text|>{ANSWER}"
        );
        let full_escaped = format!("{base_escaped}{step_escaped}");
        let full_tokens = family.encode(&full_escaped)?;
        if !full_tokens.starts_with(&base_tokens) {
            return Err(FamilyError::internal(
                "the question turn changed a token border of the context; check the context rendering",
            ));
        }
        let feed = full_tokens[base_tokens.len()..].to_vec();
        max_feed = max_feed.max(feed.len());
        let letters: Vec<u32> = asked.options.iter().map(|&opt| family.letters[opt]).collect();
        steps.push(PlainStep { feed, letters });
        if hash_text.is_empty() {
            hash_text = full_escaped;
        }
    }
    let rendered_hash = hash_of_escaped(family.tokenizer(), &hash_text)?;
    Ok(Box::new(PlainPlan {
        control_text,
        steps,
        cursor: Cursor::new(),
        rendered_hash,
        room: max_feed + 1,
    }))
}

struct QueuePlan {
    tokenizer: std::sync::Arc<Tokenizer>,
    prefix: Vec<u32>,
    control_text: Vec<ControlHit>,
    ids: Vec<String>,
    asked_indices: Vec<usize>,
    letter_sets: Vec<Vec<u32>>,
    queue_name: String,
    escaped_acc: String,
    tokens_acc: Vec<u32>,
    cursor: Cursor,
    room: usize,
}

impl ReadPlan for QueuePlan {
    fn prefix(&self) -> &[u32] {
        &self.prefix
    }

    fn control_text(&self) -> &[ControlHit] {
        &self.control_text
    }

    fn room(&self) -> usize {
        self.room
    }

    fn advance(&mut self, last: Option<&[f64]>) -> Result<Advance, FamilyError> {
        if let Some(scores) = self.cursor.accept(last)? {
            let best = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let chosen_tok = self.letter_sets[self.cursor.given - 1][best];
            let chosen_str = self
                .tokenizer
                .decode_whole(&[chosen_tok])
                .map_err(|e| FamilyError::internal(e.to_string()))?;
            self.escaped_acc.push_str(&chosen_str);
            self.tokens_acc.push(chosen_tok);
        }
        if self.cursor.given == self.ids.len() {
            self.cursor.done = true;
            self.escaped_acc.push('"');
            return Ok(Advance::Done(ReadDone {
                rendered_hash: hash_of_escaped(&self.tokenizer, &self.escaped_acc)?,
            }));
        }
        let idx = self.cursor.given;
        let qid_json = serde_json::to_string(&self.ids[idx]).map_err(|e| FamilyError::internal(e.to_string()))?;
        let escaped_qid = self.tokenizer.escape(&qid_json).map_err(refusal)?;
        let chunk = if idx == 0 {
            let escaped_name = self.tokenizer.escape(&self.queue_name).map_err(refusal)?;
            let name_json =
                serde_json::to_string(&self.queue_name).map_err(|e| FamilyError::internal(e.to_string()))?;
            let escaped_name_json = self.tokenizer.escape(&name_json).map_err(refusal)?;
            format!(
                "<|message_model|>{escaped_name}<|content_invoke_tool_json|>{{\"name\":{escaped_name_json},\"args\":{{\"answers\":{{{escaped_qid}:\""
            )
        } else {
            format!("\",{escaped_qid}:\"")
        };
        let next_escaped = format!("{}{chunk}", self.escaped_acc);
        let next_tokens = self.tokenizer.encode_escaped(&next_escaped).map_err(refusal)?;
        if !next_tokens.starts_with(&self.tokens_acc) {
            return Err(FamilyError::internal(
                "the queue read step changed a token border of the previous step",
            ));
        }
        let feed = next_tokens[self.tokens_acc.len()..].to_vec();
        self.escaped_acc = next_escaped;
        self.tokens_acc = next_tokens;
        self.cursor.given += 1;
        let letters = self.letter_sets[idx].clone();
        self.cursor.awaiting = Some(letters.len());
        Ok(Advance::Step(ReadStep {
            question: self.asked_indices[idx],
            feed,
            letters,
            continues: Continues::PreviousStep,
        }))
    }
}

fn queue_read(
    family: &Inkling2Family,
    context: &Context,
    rendering: &Rendering,
    spec: &Spec,
    case: &str,
    asked: &[Asked],
) -> Result<Box<dyn ReadPlan>, FamilyError> {
    let tool = queue_tool(context).ok_or_else(|| FamilyError::internal("queue context has no queue tool"))?;
    let call = queue_call(context).ok_or_else(|| FamilyError::internal("queue context has no queue call"))?;
    let pairs: Vec<(&str, usize)> = asked
        .iter()
        .map(|a| {
            let q = &spec.questions[a.question];
            (q.id.as_str(), q.answer_keys().len())
        })
        .collect();
    check_queue_schema(context, &pairs)?;
    let order: Vec<(usize, usize)> = tool.input_schema["properties"]["answers"]["properties"]
        .as_object()
        .map(|props| {
            props
                .keys()
                .filter_map(|k| {
                    asked
                        .iter()
                        .enumerate()
                        .find(|(_, a)| spec.questions[a.question].id == *k)
                        .map(|(asked_idx, a)| (asked_idx, a.question))
                })
                .collect()
        })
        .unwrap_or_else(|| asked.iter().enumerate().map(|(i, a)| (i, a.question)).collect());
    let mut control_text = family.hits("case", case);
    control_text.extend(rendering.control_text.iter().cloned());

    let mut msgs = messages_json(&context.messages)?;
    msgs.push(nagaya_jinja::Json::Object(vec![
        ("role".to_string(), nagaya_jinja::Json::Str("tool".to_string())),
        ("tool_call_id".to_string(), nagaya_jinja::Json::Str(call.id.clone())),
        ("content".to_string(), nagaya_jinja::Json::Str(case.to_string())),
    ]));
    let tools = tools_json(&context.tools);
    let rendered = family.run(&msgs, &tools)?;
    let ctx_len = rendering.end;
    if !rendered.tokens.starts_with(&rendering.tokens[..ctx_len]) {
        return Err(FamilyError::internal(
            "the queue read case turn changed a token border of the context",
        ));
    }
    let prefix = rendered.tokens[ctx_len..].to_vec();
    let ids: Vec<String> = order
        .iter()
        .map(|&(_, q_idx)| spec.questions[q_idx].id.clone())
        .collect();
    let asked_indices: Vec<usize> = order.iter().map(|&(asked_idx, _)| asked_idx).collect();
    let letter_sets: Vec<Vec<u32>> = order
        .iter()
        .map(|&(_, q_idx)| {
            let a = asked
                .iter()
                .find(|a| a.question == q_idx)
                .expect("checked by check_queue_schema");
            a.options.iter().map(|&opt| family.bare_letters[opt]).collect()
        })
        .collect();
    let queue_name = tool.name.clone();
    let room = ids.iter().map(|id| id.len() + 16).sum::<usize>() + queue_name.len() + 64;
    Ok(Box::new(QueuePlan {
        tokenizer: family.shared_tokenizer(),
        prefix,
        control_text,
        ids,
        asked_indices,
        letter_sets,
        queue_name,
        escaped_acc: rendered.escaped,
        tokens_acc: rendered.tokens,
        cursor: Cursor::new(),
        room,
    }))
}
