//! The oracle fixture of the Gemma 4 family (spec 8.3, "The Gemma 4 oracle"), made by `tools/gemma4_oracle_fixture.py`.
//!
//! An engine of the family compares its logits with this fixture. These tests check that the fixture is whole, and
//! that its ids are the ids that this library gives.

mod common;

use common::model::{absent, family};
use common::{JsonExt, fixture};
use nagaya_jinja::Json;
use nagaya_text_gemma4::chat::Chat;
use std::sync::Arc;

fn float(j: &Json) -> f64 {
    match j {
        Json::Float(f) => *f,
        Json::Int(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    }
}

#[test]
fn the_fixture_holds_five_whole_prompts() {
    let oracle = fixture("oracle.json");
    assert_eq!(oracle.get("layers"), &Json::Int(30), "a fixture of all 30 layers");
    assert_eq!(oracle.get("vocab"), &Json::Int(262_144));
    let prompts = oracle.get("prompts").items();
    let names: Vec<&str> = prompts.iter().map(|p| p.get("name").str()).collect();
    assert_eq!(names, ["plain", "code", "chat", "long", "longer"]);
    for p in prompts {
        let logits: Vec<f64> = p.get("top_logits").items().iter().map(float).collect();
        assert_eq!(logits.len(), 64);
        assert_eq!(p.get("top_ids").ids().len(), 64);
        assert!(logits.windows(2).all(|w| w[0] >= w[1]), "the top logits fall");
        assert!(float(p.get("logsumexp")) >= logits[0], "{}", p.get("name").str());
        assert_eq!(p.get("ids").ids()[0], 2, "each prompt starts with <bos>");
    }
}

#[test]
fn the_ids_of_the_fixture_are_the_ids_of_this_library() {
    if absent() {
        return;
    }
    let oracle = fixture("oracle.json");
    let tok = family().tokenizer();
    assert_eq!(format!("sha256:{}", oracle.get("tokenizer_sha256").str()), tok.hash());
    for p in oracle.get("prompts").items() {
        let name = p.get("name").str();
        let ids = p.get("ids").ids();
        if name == "chat" {
            let chat = Chat::new(Arc::new(
                nagaya_text_gemma4::Tokenizer::from_file(&common::model::model_dir().join("tokenizer.json")).unwrap(),
            ))
            .unwrap();
            let user = Json::parse(r#"{"role": "user", "content": "Name three colors."}"#).unwrap();
            assert_eq!(
                chat.render(&[user], &[], false, true).unwrap().tokens,
                ids,
                "the chat prompt"
            );
        } else {
            let mut want = vec![2];
            want.extend(tok.encode(p.get("text").str()).unwrap());
            assert_eq!(want, ids, "{name}");
        }
    }
}
