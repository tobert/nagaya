//! Raw reads: letter log probabilities after a suffix, read after each context (gogi "Raw reads", spec 11.3c). The
//! engine is the fake engine over the toy vocabulary, and the family is the toy family.

mod council_support;

use council_support::{Direct, Fixture, post};
use gogi_fake::{Config, Server};
use gogi_proto::context::{Context, Usage};
use gogi_proto::math::log_sum_exp;
use gogi_proto::reads::{ReadsRequest, ReadsResponse};
use nagaya::council::error::ServiceError;
use nagaya_council::read::log_probs;
use serde_json::{Value, json};

fn ctx(messages: Vec<Value>) -> Value {
    json!({"thinking": true, "tools": [{"name": "sh", "input_schema": {"type": "object"}}], "messages": messages})
}

fn opening() -> Vec<Value> {
    vec![
        json!({"role": "system", "content": "You are a coding agent. Use the shell tool."}),
        json!({"role": "user", "content": "Delete the build directory, then rebuild.", "snap": true}),
    ]
}

fn other() -> Vec<Value> {
    vec![json!({"role": "user", "content": "Is it safe to run rm -rf build?", "snap": true})]
}

fn context(value: &Value) -> Context {
    serde_json::from_value(value.clone()).unwrap()
}

const SUFFIXES: [&str; 2] = ["\nAnswer: <call>", "\nVerdict: "];

fn request(contexts: &[Value], top: Option<usize>) -> ReadsRequest {
    let mut body = json!({"model": "toy", "contexts": contexts, "suffixes": SUFFIXES, "letters": ["A", "B", "C"]});
    if let Some(top) = top {
        body["top"] = json!(top);
    }
    serde_json::from_value(body).unwrap()
}

fn read(f: &Fixture, contexts: &[Value]) -> ReadsResponse {
    f.service.read(&request(contexts, Some(4))).unwrap()
}

/// The full-vocabulary log probabilities after a suffix, from a direct run.
fn direct(value: &Value, suffix: &str) -> Vec<f64> {
    let mut run = Direct::built(&context(value));
    log_probs(&run.feed_all(&gogi_toy::tokenize_suffix(suffix))).unwrap()
}

#[test]
fn a_read_gives_the_log_probabilities_of_a_direct_run_bit_for_bit() {
    let f = Fixture::new();
    let contexts = [ctx(opening()), ctx(other())];
    let got = read(&f, &contexts);
    assert_eq!(got.model, "toy");
    assert_eq!(got.reads.len(), 4, "one read for each context and suffix");
    let letters: Vec<u32> = b"ABC".iter().map(|&b| u32::from(b)).collect();
    for (i, r) in got.reads.iter().enumerate() {
        let (c, s) = (i / 2, i % 2);
        assert_eq!((r.context, r.suffix), (c as u64, s as u64), "context-major order");
        let lp = direct(&contexts[c], SUFFIXES[s]);
        let want: Vec<f64> = letters.iter().map(|&l| lp[l as usize]).collect();
        let have: Vec<f64> = r.logprobs.values().copied().collect();
        assert_eq!(have, want, "read {i}");
        assert_eq!(r.logprobs.keys().collect::<Vec<_>>(), ["A", "B", "C"]);
        assert_eq!(r.mass, log_sum_exp(&want));
        let top: Vec<(u32, f64)> = r.top.iter().map(|t| (t.id.unwrap(), t.logprob)).collect();
        assert_eq!(top, nagaya_council::read::top(&lp, 4));
    }
}

#[test]
fn each_read_names_the_head_of_its_context() {
    let f = Fixture::new();
    let contexts = [ctx(opening()), ctx(other())];
    let got = read(&f, &contexts);
    for (c, value) in contexts.iter().enumerate() {
        let built = f.service.build(&context(value), false, Default::default()).unwrap();
        for r in got.reads.iter().filter(|r| r.context == c as u64) {
            assert_eq!(r.snapshot, built.head, "context {c}");
        }
    }
}

#[test]
fn a_read_from_a_held_head_gives_the_same_numbers_and_feeds_only_the_suffixes() {
    let f = Fixture::new();
    let contexts = [ctx(opening())];
    let first = read(&f, &contexts);
    let again = read(&f, &contexts);
    assert_eq!(first.reads, again.reads, "a read replays bit for bit (gogi rule 1)");
    let suffix_tokens: u64 = SUFFIXES.iter().map(|s| gogi_toy::tokenize_suffix(s).len() as u64).sum();
    assert_eq!(again.usage.fed, suffix_tokens, "{:?}", again.usage);
    assert!(first.usage.fed > again.usage.fed);
}

#[test]
fn usage_and_signals_match_gogi_fake_for_the_same_requests() {
    let fake = Server::start(Config::default()).unwrap();
    let f = Fixture::new();
    for contexts in [
        vec![ctx(opening())],
        vec![ctx(opening()), ctx(other())],
        vec![ctx(other())],
    ] {
        let body = serde_json::to_value(request(&contexts, Some(4))).unwrap();
        let want = post(&fake, "/council/v2/reads", body);
        let got = f.service.read(&request(&contexts, Some(4))).unwrap();
        let usage: Usage = serde_json::from_value(want["usage"].clone()).unwrap();
        assert_eq!(got.usage, usage, "{contexts:?}");
        assert_eq!(
            serde_json::to_value(&got.signals).unwrap(),
            want["signals"],
            "{contexts:?}"
        );
    }
}

#[test]
fn top_has_a_default_and_a_limit() {
    let f = Fixture::new();
    let got = f.service.read(&request(&[ctx(other())], None)).unwrap();
    assert!(got.reads.iter().all(|r| r.top.len() == 8), "the default top is 8");
    let err = f.service.read(&request(&[ctx(other())], Some(65))).unwrap_err();
    let ServiceError::Invalid(invalid) = err else {
        panic!("{err:?}")
    };
    assert_eq!(invalid.param.as_deref(), Some("top"));
}

#[test]
fn a_request_that_breaks_the_contract_is_a_client_error_and_runs_nothing() {
    let f = Fixture::new();
    let mut bad = request(&[ctx(other())], Some(4));
    bad.letters = vec!["A".to_owned()];
    assert!(matches!(f.service.read(&bad), Err(ServiceError::Invalid(_))));
    let mut two = request(&[ctx(other())], Some(4));
    two.letters = vec!["AB".to_owned(), "C".to_owned()];
    let Err(ServiceError::Invalid(invalid)) = f.service.read(&two) else {
        panic!()
    };
    assert_eq!(invalid.param.as_deref(), Some("letters"), "a letter is one token");
    assert_eq!(f.entries(), 0, "nothing was built");
    assert_eq!(f.service.free_slots(), 3);
}
