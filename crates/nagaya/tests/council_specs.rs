//! Specs are held by content and persist in the store directory (gogi "Specs").

use std::path::Path;

use gogi_proto::spec_id_of;
use nagaya::council::error::ServiceError;
use nagaya::council::specs::Specs;
use serde_json::{Value, json};

fn spec(name: &str) -> Value {
    json!({"name": name, "instructions": "Judge it.", "input_label": "Statement", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "What happens.",
         "criteria": [{"option": "allow", "means": "fine"}, {"option": "ask", "means": "not fine"}]}]})
}

fn open(directory: &Path) -> Specs {
    Specs::open(directory, Default::default()).unwrap()
}

#[test]
fn the_same_spec_gets_the_same_id_and_it_is_the_gogi_spec_id() {
    let directory = tempfile::tempdir().unwrap();
    let mut specs = open(directory.path());
    let first = specs.hold(spec("a")).unwrap();
    let again = specs.hold(spec("a")).unwrap();
    assert_eq!(first.spec_id.as_str(), spec_id_of(&spec("a")).unwrap());
    assert_eq!(first, again);
    assert_ne!(specs.hold(spec("b")).unwrap().spec_id, first.spec_id);
    assert_eq!(specs.list().len(), 2);
}

#[test]
fn a_held_spec_is_found_listed_and_dropped() {
    let directory = tempfile::tempdir().unwrap();
    let mut specs = open(directory.path());
    let held = specs.hold(spec("a")).unwrap();
    assert_eq!(specs.get(held.spec_id.as_str()).unwrap(), held);
    assert!(matches!(specs.get("sha256:00"), Err(ServiceError::NotFound(_))));
    specs.drop_spec(held.spec_id.as_str()).unwrap();
    assert!(matches!(
        specs.get(held.spec_id.as_str()),
        Err(ServiceError::NotFound(_))
    ));
    assert!(matches!(
        specs.drop_spec(held.spec_id.as_str()),
        Err(ServiceError::NotFound(_))
    ));
    assert!(specs.list().is_empty());
}

#[test]
fn specs_survive_a_restart_and_a_drop_survives_too() {
    let directory = tempfile::tempdir().unwrap();
    let (a, b) = {
        let mut specs = open(directory.path());
        (specs.hold(spec("a")).unwrap(), specs.hold(spec("b")).unwrap())
    };
    {
        let mut specs = open(directory.path());
        assert_eq!(specs.get(a.spec_id.as_str()).unwrap(), a);
        specs.drop_spec(b.spec_id.as_str()).unwrap();
    }
    let specs = open(directory.path());
    assert_eq!(specs.list(), [a]);
}

#[test]
fn a_spec_that_breaks_the_rules_is_refused_and_nothing_is_held() {
    let directory = tempfile::tempdir().unwrap();
    let mut specs = open(directory.path());
    let mut bad = spec("a");
    bad["extra"] = json!(1);
    let error = specs.hold(bad).unwrap_err();
    assert!(matches!(error, ServiceError::Invalid(_)), "{error:?}");
    let mut fraction = spec("a");
    fraction["questions"][0]["max_tokens"] = json!(1.5);
    assert!(matches!(specs.hold(fraction), Err(ServiceError::Invalid(_))));
    assert!(specs.list().is_empty());
    assert!(
        !directory.path().join("specs").read_dir().unwrap().any(|_| true),
        "no file"
    );
}

#[test]
fn a_file_whose_content_does_not_match_its_name_stops_the_open() {
    let directory = tempfile::tempdir().unwrap();
    let id = {
        let mut specs = open(directory.path());
        specs.hold(spec("a")).unwrap().spec_id
    };
    let other = serde_json::to_vec(&spec("b")).unwrap();
    let file = directory
        .path()
        .join("specs")
        .join(format!("{}.json", id.as_str().trim_start_matches("sha256:")));
    std::fs::write(&file, other).unwrap();
    let error = Specs::open(directory.path(), Default::default()).unwrap_err();
    assert!(matches!(error, ServiceError::Internal(_)), "{error:?}");
    assert!(error.to_string().contains("specs"), "{error}");
}

#[test]
fn a_spec_with_a_text_question_needs_describe() {
    let directory = tempfile::tempdir().unwrap();
    let mut specs = open(directory.path());
    let mut with_text = spec("a");
    with_text["questions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id": "effect", "type": "text", "instructions": "What it does.", "max_tokens": 8}));
    assert!(matches!(specs.hold(with_text), Err(ServiceError::Invalid(_))));
}

#[test]
fn a_file_that_breaks_the_rules_stops_the_open() {
    let directory = tempfile::tempdir().unwrap();
    open(directory.path());
    let mut bad = spec("a");
    bad["extra"] = json!(1);
    let id = spec_id_of(&bad).unwrap();
    let file = directory
        .path()
        .join("specs")
        .join(format!("{}.json", id.trim_start_matches("sha256:")));
    std::fs::write(&file, serde_json::to_vec(&bad).unwrap()).unwrap();
    let error = Specs::open(directory.path(), Default::default()).unwrap_err();
    assert!(matches!(error, ServiceError::Internal(_)), "{error:?}");
    assert!(error.to_string().contains("breaks the rules"), "{error}");
}
