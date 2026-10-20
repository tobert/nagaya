//! The decision read of the toy family: the plans for a plain read and for a queue read.
//!
//! The tokens come from `gogi_toy::read`, so they equal the tokens of `gogi-fake`.

use std::collections::HashSet;

use gogi_proto::decision::{check_queue_schema, queue_call};
use gogi_proto::spec::Spec;
use gogi_toy::read;
use nagaya_council::family::{
    Advance, Asked, Continues, ControlHit, DecisionInput, FamilyError, ReadDone, ReadPlan, ReadStep,
};

use crate::{Site, ToyFamily, hits};

/// The most options a question has: one letter each, `A` to `Z`.
const LETTERS: usize = 26;

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

/// The letter token of each kept option.
fn letters(options: &[usize]) -> Vec<u32> {
    options.iter().map(|&j| read::option_letter(j)).collect()
}

pub(crate) fn decision_read(family: &ToyFamily, input: &DecisionInput<'_>) -> Result<Box<dyn ReadPlan>, FamilyError> {
    check_asked(input.spec, input.asked)?;
    let mut control_text: Vec<ControlHit> = hits("case", input.case);
    control_text.extend(hits("spec", &input.spec.instructions));
    let queue = match input.context {
        Some((c, _)) => queue_call(c),
        None => None,
    };
    if let (Some((c, rendering)), Some(call)) = (input.context, queue) {
        let asked: Vec<(&str, usize)> = input
            .asked
            .iter()
            .map(|a| {
                let q = &input.spec.questions[a.question];
                (q.id.as_str(), q.answer_keys().len())
            })
            .collect();
        check_queue_schema(c, &asked)?;
        let start = read::queue_start(c, &call.id, input.case);
        let context = &rendering.tokens[..rendering.end];
        if !start.starts_with(context) {
            return Err(FamilyError::internal(
                "the queue read does not extend the rendering of its context; the rendering is not the context's",
            ));
        }
        let mut prefix = start[context.len()..].to_vec();
        family.tweak(Site::Prefix, &mut prefix);
        let mut order: Vec<usize> = (0..input.asked.len()).collect();
        order.sort_by(|&x, &y| {
            let id = |slot: usize| &input.spec.questions[input.asked[slot].question].id;
            id(x).cmp(id(y))
        });
        let steps = order
            .into_iter()
            .map(|slot| QueueStep {
                slot,
                id: input.spec.questions[input.asked[slot].question].id.clone(),
                options: input.asked[slot].options.clone(),
            })
            .collect();
        return Ok(Box::new(QueuePlan {
            #[cfg(test)]
            hook: family.hook,
            prefix,
            tokens: start,
            steps,
            cursor: Cursor::new(),
            control_text,
        }));
    }
    let base: Vec<u32> = match input.context {
        Some((_, rendering)) => rendering.tokens[..rendering.end].to_vec(),
        None => read::empty_context(),
    };
    // With no context the prefix holds the whole text, `<bos>` too.
    let from = if input.context.is_some() { base.len() } else { 0 };
    let text = read::plain_text(&base, &input.spec.instructions, &input.spec.input_label, input.case);
    let mut prefix = text[from..].to_vec();
    family.tweak(Site::Prefix, &mut prefix);
    let mut steps = Vec::new();
    for a in input.asked {
        let q = &input.spec.questions[a.question];
        let mut feed = read::plain_question(&text, &q.id, &q.instructions)[text.len()..].to_vec();
        family.tweak(Site::Feed, &mut feed);
        steps.push((feed, letters(&a.options)));
    }
    Ok(Box::new(PlainPlan {
        prefix,
        rendered_sha256: read::rendered_sha256(&text),
        steps,
        cursor: Cursor::new(),
        control_text,
    }))
}

/// Check the asked questions against the spec. A question with no option to read is the client's mistake. The rest
/// are mistakes of the caller.
fn check_asked(spec: &Spec, asked: &[Asked]) -> Result<(), FamilyError> {
    if asked.is_empty() {
        return Err(FamilyError::internal("a decision read asks at least one question"));
    }
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
        if a.options.windows(2).any(|w| w[0] >= w[1]) || a.options.iter().any(|&j| j >= keys || j >= LETTERS) {
            return Err(FamilyError::internal(format!(
                "the options of question {} are not increasing positions below {}",
                q.id,
                keys.min(LETTERS)
            )));
        }
    }
    Ok(())
}

/// A read with no queue: each question is read after the same text.
struct PlainPlan {
    prefix: Vec<u32>,
    rendered_sha256: String,
    /// For each asked question: its tokens and its letters.
    steps: Vec<(Vec<u32>, Vec<u32>)>,
    cursor: Cursor,
    control_text: Vec<ControlHit>,
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

struct QueueStep {
    /// The index in `asked`.
    slot: usize,
    id: String,
    options: Vec<usize>,
}

/// A read of a queue context: the case is the queue call's result, and the model writes the call.
struct QueuePlan {
    #[cfg(test)]
    hook: crate::Hook,
    prefix: Vec<u32>,
    /// All the tokens of the read so far, the context's too.
    tokens: Vec<u32>,
    /// The steps, in the order of the question ids.
    steps: Vec<QueueStep>,
    cursor: Cursor,
    control_text: Vec<ControlHit>,
}

impl QueuePlan {
    #[cfg_attr(not(test), allow(unused_variables))]
    fn tweak(&self, site: Site, tokens: &mut [u32]) {
        #[cfg(test)]
        crate::apply(self.hook, site, tokens);
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
        let from = self.tokens.len();
        if let Some(last) = last {
            // The step before chose its likeliest letter. The first of equal letters wins.
            let before = &self.steps[self.cursor.given - 1];
            let best = (0..last.len()).fold(0, |b, i| if last[i] > last[b] { i } else { b });
            read::queue_letter(&mut self.tokens, before.options[best]);
        }
        let Some(step) = self.steps.get(self.cursor.given) else {
            return Ok(self.cursor.done(read::rendered_sha256(&self.tokens)));
        };
        read::queue_question(&mut self.tokens, &step.id);
        let mut feed = self.tokens[from..].to_vec();
        self.tweak(Site::Feed, &mut feed);
        let step = ReadStep {
            question: step.slot,
            continues: Continues::PreviousStep,
            feed,
            letters: letters(&step.options),
        };
        Ok(self.cursor.step(step))
    }
}
