//! `Model::parse` against a small config fixture, with no model folder and no GPU. Each key that the slot plan or the
//! ops depend on must match the value that the engine runs.

use nagaya_mk_inkling2_sm121::spec::Model;
use serde_json::{Value, json};

const CONFIG: &str = include_str!("fixtures/config.json");

/// Parse the fixture after `edit` changed it.
fn parse_with(edit: impl FnOnce(&mut Value)) -> Result<Model, String> {
    let mut root: Value = serde_json::from_str(CONFIG).unwrap();
    edit(&mut root);
    Model::parse(&root.to_string())
}

#[test]
fn the_fixture_is_mini_and_the_flag_selects_the_research_model() {
    assert_eq!(parse_with(|_| {}).unwrap(), Model::Mini);
    let research = parse_with(|root| root["text_config"]["loop_config"]["is_varloop"] = json!(true));
    assert_eq!(research.unwrap(), Model::Research);
    let bad_flag = parse_with(|root| root["text_config"]["loop_config"]["is_varloop"] = json!("yes"));
    assert!(bad_flag.unwrap_err().contains("is_varloop"));
}

#[test]
fn a_key_that_the_slot_plan_or_the_ops_depend_on_must_match() {
    let cases = [
        ("sliding_window_size", json!(2048)),
        ("sconv_kernel_size", json!(3)),
        ("n_routed_experts", json!(64)),
        ("intermediate_size", json!(1024)),
        (
            "local_layer_ids",
            json!([0, 1, 2, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]),
        ),
        ("hidden_size", json!(2048)),
    ];
    for (key, value) in cases {
        let error = parse_with(|root| root["text_config"][key] = value.clone()).unwrap_err();
        assert!(
            error.contains(&format!("text_config.{key}")),
            "the error for {key} does not name the key: {error}"
        );
        assert!(
            error.contains(&value.to_string()),
            "the error for {key} does not show the value: {error}"
        );
    }
}

#[test]
fn a_missing_key_is_an_error_that_names_the_key() {
    for key in [
        "sliding_window_size",
        "sconv_kernel_size",
        "n_routed_experts",
        "intermediate_size",
        "local_layer_ids",
    ] {
        let error = parse_with(|root| {
            root["text_config"].as_object_mut().unwrap().remove(key);
        })
        .unwrap_err();
        assert!(error.contains(&format!("text_config.{key}")), "{key}: {error}");
        assert!(error.contains("null"), "{key}: {error}");
    }
}

#[test]
fn the_error_says_what_to_do() {
    let error = parse_with(|root| root["text_config"]["sconv_kernel_size"] = json!(3)).unwrap_err();
    assert!(error.contains("expected 4"), "{error}");
    assert!(error.contains("another model"), "{error}");
}
