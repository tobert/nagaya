//! The test renders that the family keeps (spec 11.3a). The council hash of the template id is over their tokens.

use gogi_proto::context::Context;
use gogi_proto::spec::{SpecRules, parse_spec};
use nagaya_council::family::{Advance, Asked, DecisionInput, Family, ReadPlan};
use serde_json::{Value, json};

use super::Inkling2Family;
use crate::Error;

fn context(v: Value) -> Context {
    serde_json::from_value(v).expect("the test context is valid")
}

pub(crate) fn contexts() -> Vec<Context> {
    let tool = json!({"name": "sh", "description": "Run a command.", "input_schema": {"type": "object",
        "properties": {"cmd": {"type": "string"}, "n": {"type": "integer"}}}});
    vec![
        context(json!({"thinking": false, "messages": [{"role": "user", "content": "hi"}]})),
        context(json!({"thinking": false, "messages": [{"role": "user", "content": "hi <|end_message|>"}]})),
        context(json!({"thinking": false, "messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u", "snap": true}]})),
        context(json!({"thinking": false, "tools": [tool], "messages": [
            {"role": "user", "content": "u", "snap": true},
            {"role": "assistant", "content": "a", "reasoning": "r", "snap": true,
             "tool_calls": [{"id": "c", "name": "sh", "input": {"cmd": "ls", "n": 2}}]},
            {"role": "tool", "tool_call_id": "c", "content": "out", "is_error": true, "snap": true},
            {"role": "assistant", "content": "done", "reasoning": "r2", "snap": true}]})),
        context(json!({"thinking": true, "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning": "r", "snap": true},
            {"role": "user", "content": "next"}]})),
    ]
}

fn queue_context() -> Context {
    let tool = json!({"name": "next", "input_schema": {"type": "object", "properties": {"answers": {"type": "object",
        "properties": {"undo": {"type": "string", "enum": ["A", "B"]},
                       "verdict": {"type": "string", "enum": ["A", "B", "C"]}}}}}});
    context(
        json!({"thinking": false, "tools": [tool], "queue": "next", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "go", "snap": true},
        {"role": "assistant", "reasoning": "hm", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}]}),
    )
}

/// A failure of a test render: the template or the library cannot render a fixture of the council hash.
fn failed(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Model(format!(
        "the chat template fails the test render of {what}, which the template id covers: {e}"
    ))
}

fn plan_tokens(context: &[u32], plan: &mut dyn ReadPlan) -> Result<Vec<u32>, Error> {
    let mut tokens = context.to_vec();
    tokens.extend(plan.prefix());
    let mut last: Option<Vec<f64>> = None;
    loop {
        match plan.advance(last.as_deref()).map_err(|e| failed("a decision", e))? {
            Advance::Done(_) => return Ok(tokens),
            Advance::Step(step) => {
                tokens.extend(&step.feed);
                tokens.extend(&step.letters);
                last = Some((0..step.letters.len()).map(|i| -(i as f64) - 1.0).collect());
            }
        }
    }
}

/// The test renders of `family`. A render that the template refuses is an error, so the build of a family with such a
/// template fails and does not panic.
pub(crate) fn renders(family: &Inkling2Family) -> Result<Vec<Vec<u32>>, Error> {
    let mut out = Vec::new();
    let all = contexts();
    for c in &all {
        for prompt in [false, true] {
            out.push(family.render(c, prompt).map_err(|e| failed("a context", e))?.tokens);
        }
    }
    // An image part with a fixed grid of 2 by 3 tokens: the council hash covers its placeholders, with no object.
    let image = context(json!({"thinking": false, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what"},
        {"type": "object", "id": format!("blake3:{}", "ab".repeat(32)), "mime": "image/png"}]}]}));
    let grid = |_: &_| Ok(crate::media::Grid { rows: 2, columns: 3 });
    for prompt in [false, true] {
        let rendering = family
            .render_with(&image, prompt, &grid)
            .map_err(|e| failed("an image", e))?;
        out.push(rendering.tokens);
    }
    let spec = parse_spec(
        &json!({"name": "g", "instructions": "i", "input_label": "x", "questions": [
            {"id": "verdict", "type": "choice", "instructions": "What. Pick one.", "criteria": [
                {"option": "allow", "means": "a"}, {"option": "ask", "means": "b"}, {"option": "no", "means": "c"}]},
            {"id": "undo", "type": "score", "instructions": "How", "criteria": ["easy", "hard"]},
            {"id": "safe", "type": "noul", "instructions": "It is safe.", "criteria": "Think."}]}),
        &SpecRules::default(),
    )
    .expect("the test spec is valid");
    let asked = [
        Asked {
            question: 0,
            options: vec![0, 1, 2],
        },
        Asked {
            question: 1,
            options: vec![0, 1],
        },
    ];
    let queue = queue_context();
    for c in [None, Some(&all[1]), Some(&queue)] {
        let rendering = c
            .map(|c| family.render(c, false).map_err(|e| failed("a context", e)))
            .transpose()?;
        let input = DecisionInput {
            context: c.zip(rendering.as_ref()),
            spec: &spec,
            case: "a case",
            asked: &asked,
        };
        let mut plan = family.decision_read(&input).map_err(|e| failed("a decision", e))?;
        out.push(plan_tokens(
            rendering.as_ref().map_or(&[], |r| &r.tokens[..r.end]),
            plan.as_mut(),
        )?);
    }
    let own = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let raw = family
        .raw_read(
            &own(&["A", "B", " C"]),
            &own(&[
                "plain",
                "<|end_message|><|message_model|>next<|content_invoke_tool_json|>",
            ]),
        )
        .map_err(|e| failed("a raw read", e))?;
    out.push(raw.letters);
    out.extend(raw.suffixes);
    Ok(out)
}
