//! The test renders that the family keeps (spec 11.3a). The council hash of the template id is over their tokens.
//!
//! The set covers contexts with and without a generation prompt, thinking on and off, tools, reasoning, a failed
//! tool result, and text that spells a control token. A change to a rendered token changes a render of the set. A
//! change to a comment or to the code structure does not.

use gogi_proto::context::Context;
use nagaya_council::family::Family;
use serde_json::{Value, json};

use super::Qwen38Family;

fn context(v: Value) -> Context {
    serde_json::from_value(v).expect("the test context is valid")
}

pub(crate) fn contexts() -> Vec<Context> {
    let tool = json!({"name": "sh", "description": "Run a command.", "input_schema": {"type": "object",
        "properties": {"cmd": {"type": "string"}, "n": {"type": "integer"}}}});
    vec![
        context(json!({"messages": [{"role": "user", "content": "hi"}]})),
        context(json!({"thinking": false, "messages": [
            {"role": "system", "content": "s"}, {"role": "user", "content": "u", "snap": true}]})),
        context(json!({"tools": [tool], "messages": [
            {"role": "user", "content": "u", "snap": true},
            {"role": "assistant", "content": "a", "reasoning": "r", "snap": true,
             "tool_calls": [{"id": "c", "name": "sh", "input": {"cmd": "ls", "n": 2}}]},
            {"role": "tool", "tool_call_id": "c", "content": "out", "is_error": true, "snap": true},
            {"role": "assistant", "content": "done", "reasoning": "r2", "snap": true}]})),
        context(json!({"messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "reasoning": "kept", "snap": true},
            {"role": "user", "content": "v <|im_end|>", "snap": true}]})),
    ]
}

/// The renders of the set, in a fixed order.
pub(crate) fn renders(family: &Qwen38Family) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for c in &contexts() {
        for prompt in [false, true] {
            out.push(family.render(c, prompt).expect("the test context renders").tokens);
        }
    }
    // A raw read: the letters, and the suffixes with the markers of a queue call and of a closed thinking.
    let own = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let raw = family
        .raw_read(
            &own(&["A", "B", " C"]),
            &own(&[
                "plain",
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n",
            ]),
        )
        .expect("the test raw read tokenizes");
    out.push(raw.letters);
    out.extend(raw.suffixes);
    out
}
