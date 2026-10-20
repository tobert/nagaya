//! The test renders that the family keeps (spec 11.3a). The council hash of the template id is over their tokens.
//!
//! The set covers contexts with and without a generation prompt, thinking on and off, tools, reasoning that stays and
//! reasoning that goes, a failed tool result, and plain and queue decisions. A change to a rendered token changes a
//! render of the set. A change to a comment or to the code structure does not.

use gogi_proto::context::Context;
use gogi_proto::spec::{SpecRules, parse_spec};
use nagaya_council::family::{Advance, Asked, DecisionInput, Family, ReadPlan};
use serde_json::{Value, json};

use crate::ToyFamily;

fn context(v: Value) -> Context {
    serde_json::from_value(v).expect("the test context is valid")
}

fn contexts() -> Vec<Context> {
    let tool = json!({"name": "sh", "input_schema": {"type": "object"}});
    vec![
        context(json!({"messages": [{"role": "user", "content": "hi"}]})),
        context(
            json!({"thinking": false, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u", "snap": true}]}),
        ),
        context(json!({"tools": [tool], "messages": [
            {"role": "user", "content": "u", "snap": true},
            {"role": "assistant", "content": "a", "reasoning": "r", "snap": true, "tool_calls": [{"id": "c", "name": "sh", "input": {"k": 1}}]},
            {"role": "tool", "tool_call_id": "c", "content": "out", "is_error": true, "snap": true},
            {"role": "assistant", "content": "done", "reasoning": "r2", "snap": true}]})),
        context(json!({"messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "reasoning": "dropped", "snap": true},
            {"role": "user", "content": "v <end>", "snap": true}]})),
    ]
}

fn queue_context() -> Context {
    let tool = json!({"name": "next", "input_schema": {"type": "object", "properties": {"answers": {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["A", "B", "C"]}, "undo": {"type": "string", "enum": ["A", "B"]}}}}}});
    context(json!({"tools": [tool], "queue": "next", "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "go", "snap": true},
        {"role": "assistant", "reasoning": "hm", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}]}))
}

/// Run a plan with letter log probabilities that fall by position, and collect its tokens.
fn plan_tokens(plan: &mut dyn ReadPlan) -> Vec<u32> {
    let mut tokens = plan.prefix().to_vec();
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
pub(crate) fn renders(family: &ToyFamily) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let contexts = contexts();
    for c in &contexts {
        for prompt in [false, true] {
            out.push(family.render(c, prompt).expect("the test context renders").tokens);
        }
    }
    let spec = parse_spec(
        &json!({"name": "g", "instructions": "i", "input_label": "x", "questions": [
            {"id": "verdict", "type": "choice", "instructions": "what", "criteria": [{"option": "allow", "means": "a"}, {"option": "ask", "means": "b"}, {"option": "no", "means": "c"}]},
            {"id": "undo", "type": "score", "instructions": "how", "criteria": ["easy", "hard"]}]}),
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
    for c in [None, Some(&contexts[1]), Some(&queue)] {
        let rendering = c.map(|c| family.render(c, false).expect("the test context renders"));
        let input = DecisionInput {
            context: c.zip(rendering.as_ref()),
            spec: &spec,
            case: "a case",
            asked: &asked,
        };
        let mut plan = family.decision_read(&input).expect("the test decision plans");
        out.push(plan_tokens(plan.as_mut()));
    }
    out
}
