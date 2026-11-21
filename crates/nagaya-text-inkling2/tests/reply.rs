mod common;

use common::model::{absent, mini_dir};
use gogi_proto::generation::Phase;
use nagaya_council::family::{Family, ReplyCall, ReplyStart};
use nagaya_text_inkling2::Inkling2Family;
use serde_json::Value;

#[test]
fn reply_parser_matches_reply_fixture() {
    if absent() {
        return;
    }
    let fam = Inkling2Family::from_dir(&mini_dir()).unwrap();
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(common::fixture("reply.json")).unwrap()).unwrap();

    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ending = case["ending"].as_str().unwrap();
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        let mut parser = fam.reply_parser(ReplyStart::prompt(true), &[]).unwrap();
        let mut steps = Vec::new();
        for &tok in &ids {
            steps.push(parser.push(tok).unwrap_or_else(|e| panic!("{name}: push {tok}: {e}")));
        }
        assert_eq!(parser.ended(), ending == "stop", "{name}: ended");
        let parsed = parser.finish(ending == "length");
        assert_eq!(parsed.ended, ending == "stop", "{name}: parsed.ended");

        if let Some(want) = case.get("parsed") {
            assert_eq!(parsed.content, want["content"].as_str().unwrap(), "{name}: content");
            assert_eq!(
                parsed.reasoning,
                want["reasoning"].as_str().unwrap(),
                "{name}: reasoning"
            );
            let want_calls = want["tool_calls"].as_array().unwrap();
            assert_eq!(parsed.calls.len(), want_calls.len(), "{name}: calls len");
            for (got, w) in parsed.calls.iter().zip(want_calls) {
                match got {
                    ReplyCall::Parsed { name: cname, input } => {
                        assert_eq!(cname, w["name"].as_str().unwrap(), "{name}: call name");
                        let w_args = w["arguments"].as_object().unwrap();
                        assert_eq!(input, w_args, "{name}: call args");
                    }
                    ReplyCall::Malformed { text, error } => {
                        panic!("{name}: unexpected malformed call {text:?}: {error}");
                    }
                }
            }
        } else {
            assert!(
                matches!(parsed.calls.as_slice(), [ReplyCall::Malformed { .. }]),
                "{name}: expected malformed call, got {:?}",
                parsed.calls
            );
        }
    }
}

#[test]
fn reply_parser_refuses_post_stop_and_out_of_vocab_and_oversized_int() {
    if absent() {
        return;
    }
    let fam = Inkling2Family::from_dir(&mini_dir()).unwrap();
    let tok = fam.tokenizer();
    let eos = tok.token_id("<|content_model_end_sampling|>").unwrap();

    // Out-of-vocab token
    let mut parser = fam.reply_parser(ReplyStart::prompt(true), &[]).unwrap();
    assert!(parser.push(fam.vocab_size() as u32).is_err());

    // Post-stop token
    let mut parser = fam.reply_parser(ReplyStart::prompt(true), &[]).unwrap();
    parser.push(eos).unwrap();
    assert!(parser.ended());
    assert!(parser.push(eos).is_err());

    // Oversized >64-bit integer in tool call becomes Malformed (spec 11.3).
    let mut parser = fam.reply_parser(ReplyStart::prompt(true), &[]).unwrap();
    let bad_call = r#"<|message_model|>sh<|content_invoke_tool_json|>{"name":"sh","args":{"n":1234567890123456789012345}}<|end_message|><|content_model_end_sampling|>"#;
    for id in fam.tokenize_special(bad_call).unwrap() {
        parser.push(id).unwrap();
    }
    let reply = parser.finish(false);
    assert!(matches!(reply.calls.as_slice(), [ReplyCall::Malformed { .. }]));

    // Open continuation in thinking starts in Reasoning phase.
    let parser = fam
        .reply_parser(
            ReplyStart {
                thinking: true,
                open: true,
            },
            &[],
        )
        .unwrap();
    assert_eq!(parser.phase(), Phase::Reasoning);
}
