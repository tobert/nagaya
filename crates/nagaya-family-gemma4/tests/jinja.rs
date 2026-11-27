//! Google's chat template in the nagaya's interpreter gives the text of transformers (spec 11.3).
//!
//! The fixture `chat.json` holds each case with its variables, and the text or the error of Python. This test needs no
//! checkpoint: it runs the interpreter alone.

mod common;

use common::{JsonExt, fixture};
use nagaya_family_gemma4::CHAT_TEMPLATE;
use nagaya_jinja::{Json, Template};

#[test]
fn the_template_renders_what_transformers_renders() {
    let template = Template::from_source(CHAT_TEMPLATE).expect("Google's template compiles");
    let cases = fixture("chat.json");
    let mut checked = 0;
    for case in cases.get("cases").items() {
        let name = case.get("name").str();
        let mut variables: Vec<(&str, Json)> = vec![("messages", case.get("messages").clone())];
        if !matches!(case.get("tools"), Json::Null) {
            variables.push(("tools", case.get("tools").clone()));
        }
        let Json::Object(more) = case.get("variables") else {
            panic!("{name}: the variables are not an object")
        };
        for (k, v) in more {
            variables.push((k.as_str(), v.clone()));
        }
        match (template.render(&variables), case.has("text")) {
            (Ok(text), true) => assert_eq!(text, case.get("text").str(), "case {name:?}"),
            (Err(e), false) => {
                // A message of `raise_exception` reaches the error. Another refusal has Python's own words.
                let python = case.get("error").str();
                if let Some(message) = python.strip_prefix("TemplateError: ") {
                    assert!(e.to_string().contains(message), "case {name:?}: {e}");
                }
            }
            (Ok(text), false) => panic!("case {name:?}: Python refused it, and the interpreter gave {text:?}"),
            (Err(e), true) => panic!("case {name:?}: Python rendered it, and the interpreter failed: {e}"),
        }
        checked += 1;
    }
    assert_eq!(checked, 26, "the fixture holds 26 cases");
}

#[test]
fn the_template_is_the_file_that_made_the_fixture() {
    use sha2::{Digest, Sha256};
    let digest: String = Sha256::digest(CHAT_TEMPLATE.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest, fixture("chat.json").get("header").get("template_sha256").str());
}
