//! The test renders that the family keeps (spec 11.3a). The council hash of the template id is over their tokens.
//!
//! The set covers contexts with and without a generation prompt, and thinking on and off. It covers tools, reasoning,
//! a failed tool result, and text that spells a control token. A change to a rendered token changes a render of the
//! set. A change to a comment or to the code structure does not.

use gogi_proto::context::Context;
use gogi_proto::spec::{SpecRules, parse_spec};
use nagaya_council::family::{Advance, Asked, DecisionInput, Family, ReadPlan};
use serde_json::{Value, json};

use super::Gemma4Family;

fn context(v: Value) -> Context {
    serde_json::from_value(v).expect("the test context is valid")
}

pub(crate) fn contexts() -> Vec<Context> {
    let tool = json!({"name": "sh", "description": "Run a command.", "input_schema": {"type": "object",
        "properties": {"cmd": {"type": "string"}, "n": {"type": "integer"}}}});
    vec![
        context(json!({"thinking": true, "messages": [{"role": "user", "content": "hi"}]})),
        context(json!({"thinking": false, "messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u", "snap": true}]})),
        context(json!({"thinking": true, "tools": [tool], "messages": [
            {"role": "user", "content": "u", "snap": true},
            {"role": "assistant", "content": "a", "reasoning": "r", "snap": true,
             "tool_calls": [{"id": "c", "name": "sh", "input": {"cmd": "ls", "n": 2}}]},
            {"role": "tool", "tool_call_id": "c", "content": "out", "is_error": true, "snap": true},
            {"role": "assistant", "content": "done", "reasoning": "r2", "snap": true}]})),
        context(json!({"thinking": true, "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "reasoning": "kept", "snap": true},
            {"role": "user", "content": "v <|turn>", "snap": true}]})),
    ]
}

fn queue_context() -> Context {
    let tool = json!({"name": "next", "input_schema": {"type": "object", "properties": {"answers": {"type": "object",
        "properties": {"undo": {"type": "string", "enum": ["A", "B"]},
                       "verdict": {"type": "string", "enum": ["A", "B", "C"]}}}}}});
    context(json!({"thinking": true, "tools": [tool], "queue": "next", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "go", "snap": true},
        {"role": "assistant", "reasoning": "hm", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}]}))
}

/// Run a plan with letter log probabilities that fall by position, and collect its tokens after `context`.
fn plan_tokens(context: &[u32], plan: &mut dyn ReadPlan) -> Vec<u32> {
    let mut tokens = context.to_vec();
    tokens.extend(plan.prefix());
    let mut last: Option<Vec<f64>> = None;
    loop {
        match plan.advance(last.as_deref()).expect("the test plan runs") {
            Advance::Done(_) => return tokens,
            Advance::Step(step) => {
                tokens.extend(&step.feed);
                tokens.extend(&step.letters);
                last = Some((0..step.letters.len()).map(|i| -(i as f64) - 1.0).collect());
            }
        }
    }
}

/// The renders of the set, in a fixed order.
pub(crate) fn renders(family: &Gemma4Family) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for c in &contexts() {
        for prompt in [false, true] {
            out.push(family.render(c, prompt).expect("the test context renders").tokens);
        }
    }
    // Decisions: plain with no context, plain after a context, and a queue read.
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
    let contexts = contexts();
    let queue = queue_context();
    for c in [None, Some(&contexts[1]), Some(&queue)] {
        let rendering = c.map(|c| family.render(c, false).expect("the test context renders"));
        let input = DecisionInput {
            context: c.zip(rendering.as_ref()),
            spec: &spec,
            case: "a case",
            asked: &asked,
        };
        let mut plan = family.decision_read(&input).expect("the test decision plans");
        out.push(plan_tokens(
            rendering.as_ref().map_or(&[], |r| &r.tokens[..r.end]),
            plan.as_mut(),
        ));
    }
    // A raw read: the letters, and the suffixes with the markers of a queue call and of a closed thinking.
    let own = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let raw = family
        .raw_read(
            &own(&["A", "B", " C"]),
            &own(&[
                "plain",
                "<turn|>\n<|turn>model\n<|channel>thought\n<channel|><|tool_call>call:",
            ]),
        )
        .expect("the test raw read tokenizes");
    out.push(raw.letters);
    out.extend(raw.suffixes);
    out
}
