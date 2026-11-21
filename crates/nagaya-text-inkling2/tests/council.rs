mod common;

use common::model::{absent, mini_dir};
use gogi_proto::context::Context;
use gogi_proto::spec::{SpecRules, parse_spec};
use nagaya_council::family::{Advance, Asked, DecisionInput, Family};
use nagaya_text_inkling2::{Files, Inkling2Family};
use serde_json::json;

#[test]
fn family_reports_unpadded_vocab_size_and_thinking() {
    if absent() {
        return;
    }
    let fam = Inkling2Family::from_dir(&mini_dir()).unwrap();
    assert_eq!(fam.vocab_size(), 200_058);
    assert!(fam.thinks());
    assert!(fam.template_id().starts_with("nagaya-council:"));
    assert!(fam.template_id().contains("+inkling2-chat:"));
}

#[test]
fn template_id_survives_comment_edit_and_changes_on_token_change() {
    if absent() {
        return;
    }
    let base = Files::read(&mini_dir()).unwrap();
    let fam_a = Inkling2Family::from_files(base.clone()).unwrap();

    let mut comment_edit = base.clone();
    comment_edit.template = format!("{}\n{{# comment #}}", comment_edit.template);
    let fam_b = Inkling2Family::from_files(comment_edit).unwrap();
    // Council hash stays identical because rendered tokens did not change (spec 11.3a).
    let council_a = fam_a.template_id().split('+').next().unwrap();
    let council_b = fam_b.template_id().split('+').next().unwrap();
    assert_eq!(council_a, council_b);
    assert_ne!(fam_a.template_id(), fam_b.template_id());

    let mut token_edit = base;
    token_edit.template = token_edit
        .template
        .replace("Thinking effort level: ", "Thinking effort: ");
    let fam_c = Inkling2Family::from_files(token_edit).unwrap();
    let council_c = fam_c.template_id().split('+').next().unwrap();
    assert_ne!(council_a, council_c);
}

#[test]
fn council_render_decisions_and_raw_read_work() {
    if absent() {
        return;
    }
    let fam = Inkling2Family::from_dir(&mini_dir()).unwrap();
    let ctx: Context = serde_json::from_value(json!({
        "thinking": true,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi <|end_message|>", "snap": true},
            {"role": "assistant", "content": "hello", "reasoning": "thought", "snap": true},
            {"role": "user", "content": "next"}
        ]
    }))
    .unwrap();
    let rendering = fam.render(&ctx, true).unwrap();
    // The generation prompt is `<|message_model|>`, the token that the reference writes first with a probability of
    // 1 - 6e-12 after "chat" (spec 11.18). The council needs a generation prompt for the first logit row.
    let message_model = fam.tokenizer().token_id("<|message_model|>").unwrap();
    assert_eq!(rendering.end + 1, rendering.tokens.len());
    assert_eq!(rendering.tokens.last(), Some(&message_model));
    let built = fam.render(&ctx, false).unwrap();
    assert_eq!(built.tokens, rendering.tokens[..rendering.end]);
    assert_eq!(built.end, built.tokens.len());
    assert_eq!(built.boundaries, rendering.boundaries);
    assert!(!rendering.boundaries.is_empty());
    assert_eq!(rendering.control_text.len(), 1);
    assert_eq!(rendering.control_text[0].token, "<|end_message|>");

    // Plain decision read
    let spec = parse_spec(
        &json!({
            "name": "g",
            "instructions": "Pick one.",
            "input_label": "item",
            "questions": [{
                "id": "v",
                "type": "choice",
                "instructions": "Which?",
                "criteria": [{"option": "a", "means": "one"}, {"option": "b", "means": "two"}]
            }]
        }),
        &SpecRules::default(),
    )
    .unwrap();
    let asked = [Asked {
        question: 0,
        options: vec![0, 1],
    }];
    let input = DecisionInput {
        context: Some((&ctx, &rendering)),
        spec: &spec,
        case: "the item",
        asked: &asked,
    };
    let mut plan = fam.decision_read(&input).unwrap();
    // spec 11.18, "Reads": the prefix is the case turn, a user message with the label and the case.
    let prefix = fam.detokenize(plan.prefix()).unwrap();
    assert_eq!(prefix, "<|message_user|><|content_text|>item:\nthe item<|end_message|>");
    assert!(plan.prefix_is_head());
    let Advance::Step(step) = plan.advance(None).unwrap() else {
        panic!("expected step");
    };
    assert_eq!(step.question, 0);
    assert_eq!(step.letters.len(), 2);
    let step_text = fam.detokenize(&step.feed).unwrap();
    assert!(
        step_text.starts_with("<|message_user|><|content_text|>"),
        "plain read user turn missing <|content_text|>: {step_text:?}"
    );
    let Advance::Done(done_a) = plan.advance(Some(&[-0.1, -2.0])).unwrap() else {
        panic!("expected done");
    };
    assert!(done_a.rendered_hash.starts_with("blake3:"));

    // Same read with different logprobs gives the exact same rendered_hash (conditioning text only).
    let mut plan_b = fam.decision_read(&input).unwrap();
    plan_b.advance(None).unwrap();
    let Advance::Done(done_b) = plan_b.advance(Some(&[-2.0, -0.1])).unwrap() else {
        panic!("expected done");
    };
    assert_eq!(done_a.rendered_hash, done_b.rendered_hash);

    // A read with no context renders with thinking off, so the template writes the effort 0 before the case turn.
    let alone = DecisionInput { context: None, ..input };
    let plan_alone = fam.decision_read(&alone).unwrap();
    assert_eq!(
        fam.detokenize(plan_alone.prefix()).unwrap(),
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
         <|message_user|><|content_text|>item:\nthe item<|end_message|>"
    );

    // NaN logprob is refused.
    let mut plan_nan = fam.decision_read(&input).unwrap();
    plan_nan.advance(None).unwrap();
    assert!(plan_nan.advance(Some(&[f64::NAN, -1.0])).is_err());

    // Queue read: prefix is delta only, step.question maps to `asked` index, narrowed options work.
    let q_spec = parse_spec(
        &json!({
            "name": "g",
            "instructions": "i",
            "input_label": "x",
            "questions": [
                {"id": "first", "type": "noul", "instructions": "One?"},
                {"id": "second", "type": "choice", "instructions": "Two?", "criteria": [
                    {"option": "a", "means": "1"}, {"option": "b", "means": "2"}, {"option": "c", "means": "3"}
                ]}
            ]
        }),
        &SpecRules::default(),
    )
    .unwrap();
    let q_ctx: Context = serde_json::from_value(json!({
        "thinking": false,
        "queue": "next",
        "tools": [{
            "name": "next",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "object",
                        "properties": {
                            "second": {"type": "string", "enum": ["A", "B", "C"]},
                            "first": {"type": "string", "enum": ["A", "B"]}
                        }
                    }
                }
            }
        }],
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}
        ]
    }))
    .unwrap();
    let q_rend = fam.render(&q_ctx, false).unwrap();
    let q_asked = [
        Asked {
            question: 0,
            options: vec![0, 1],
        },
        Asked {
            question: 1,
            options: vec![0, 2],
        },
    ];
    let q_input = DecisionInput {
        context: Some((&q_ctx, &q_rend)),
        spec: &q_spec,
        case: "case text",
        asked: &q_asked,
    };
    let mut q_plan = fam.decision_read(&q_input).unwrap();
    // Prefix is only the new tool result turn, not the whole context.
    assert!(q_plan.prefix().len() < q_rend.end);
    let Advance::Step(s0) = q_plan.advance(None).unwrap() else {
        panic!("expected step 0");
    };
    // Schema has "second" before "first", so step 0 reads asked[1].
    assert_eq!(s0.question, 1);
    assert_eq!(s0.letters.len(), 2);
    let Advance::Step(s1) = q_plan.advance(Some(&[-0.1, -1.0])).unwrap() else {
        panic!("expected step 1");
    };
    assert_eq!(s1.question, 0);
    assert!(matches!(q_plan.advance(Some(&[-0.1, -1.0])).unwrap(), Advance::Done(_)));

    // Open continuation renders without <|end_message|>
    let open_ctx: Context = serde_json::from_value(json!({
        "thinking": true,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "reasoning": "step 1", "content": "ans", "continue": true}
        ]
    }))
    .unwrap();
    let open_rend = fam.render(&open_ctx, false).unwrap();
    let open_text = fam.detokenize(&open_rend.tokens).unwrap();
    assert!(open_text.ends_with("<|content_text|>ans"));

    // Raw read
    let raw = fam
        .raw_read(&["A".to_string(), "B".to_string()], &["suffix".to_string()])
        .unwrap();
    assert_eq!(raw.letters.len(), 2);
    assert_eq!(raw.suffixes.len(), 1);
}

/// A context with thinking off renders the effort "none" (0), and one with thinking on renders the default of the
/// template (0.9). Milestone 35 question 4, agreed by Amy on 2026-11-21.
#[test]
fn thinking_chooses_the_effort_of_the_template() {
    if absent() {
        return;
    }
    let fam = Inkling2Family::from_dir(&mini_dir()).unwrap();
    let text = |thinking: bool| {
        let context: Context = serde_json::from_value(json!({
            "thinking": thinking,
            "messages": [{"role": "user", "content": "Name three colors.", "snap": true},
                         {"role": "assistant", "content": "Red, green, blue."},
                         {"role": "user", "content": "Two more."}]
        }))
        .unwrap();
        let rendering = fam.render(&context, true).unwrap();
        fam.tokenizer().decode_lossy(&rendering.tokens).unwrap()
    };
    let off = text(false);
    let on = text(true);
    assert!(off.contains("Thinking effort level: 0<|end_message|>"), "{off}");
    assert!(on.contains("Thinking effort level: 0.9<|end_message|>"), "{on}");
}

/// A latent position of Inkling 2 holds `<|endoftext|>`, the `pad_token_id` of `generation_config.json`, and its
/// record is the kind 5 with the width of a row (engine-abi "The lab of Inkling 2", "Latent rows").
#[test]
fn a_latent_span_holds_the_pad_token_and_a_record_of_kind_5() {
    if absent() {
        return;
    }
    let fam = Inkling2Family::from_dir(&mini_dir()).unwrap();
    let (token, record) = fam.latent_record(7, 3, 2304).expect("the family takes latent rows");
    assert_eq!(token, 199_999);
    let word = |at: usize| u32::from_le_bytes(record[at..at + 4].try_into().unwrap());
    assert_eq!(record.len(), 32);
    assert_eq!((word(0), word(4)), (5, 3), "the kind and the count");
    assert_eq!(
        u64::from_le_bytes(record[8..16].try_into().unwrap()),
        7,
        "the first position"
    );
    assert_eq!(
        (word(16), word(20), word(24), word(28)),
        (2304, 0, 0, 0),
        "the width and three 0"
    );
}
