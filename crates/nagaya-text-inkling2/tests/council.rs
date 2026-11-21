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
    assert_eq!(rendering.end, rendering.tokens.len());
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
    let Advance::Step(step) = plan.advance(None).unwrap() else {
        panic!("expected step");
    };
    assert_eq!(step.letters.len(), 2);
    let Advance::Done(done) = plan.advance(Some(&[-0.1, -2.0])).unwrap() else {
        panic!("expected done");
    };
    assert!(done.rendered_hash.starts_with("blake3:"));

    // Raw read
    let raw = fam
        .raw_read(&["A".to_string(), "B".to_string()], &["suffix".to_string()])
        .unwrap();
    assert_eq!(raw.letters.len(), 2);
    assert_eq!(raw.suffixes.len(), 1);
}
