//! Decisions: a case read after each context, with the reads pooled (gogi "Decisions", spec 11.3a, 11.3c). The
//! engine is the fake engine over the toy vocabulary, and the family is the toy family.

mod council_support;

use std::time::Duration;

use council_support::{Direct, Fixture, TermFactory, post};
use gogi_fake::{Config, Server};
use gogi_proto::context::{Context, Usage};
use gogi_proto::decision::{DecisionRequest, DecisionResponse, ReadAnswer};
use gogi_proto::math;
use gogi_proto::spec::{SpecRules, parse_spec};
use nagaya::council::ServiceConfig;
use nagaya::council::error::ServiceError;
use nagaya_council::decision::{asked, case_text};
use nagaya_council::family::{Advance, Continues, DecisionInput, Family};
use nagaya_council::read::log_probs;
use nagaya_text_toy::ToyFamily;
use serde_json::{Value, json};

fn spec_json() -> Value {
    json!({"name": "g", "instructions": "Judge the command.", "input_label": "Command", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "What happens?", "criteria": [
            {"option": "allow", "means": "fine"}, {"option": "ask", "means": "not clearly fine"},
            {"option": "report", "means": "harmful"}]},
        {"id": "undo", "type": "score", "instructions": "How hard is it to undo?", "criteria": ["easy", "hard"]},
        {"id": "fine", "type": "noul", "instructions": "Is it fine?"}]})
}

fn tools() -> Value {
    json!([{"name": "next_command", "input_schema": {"type": "object", "properties": {"answers": {"type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["A", "B", "C"]},
                       "undo": {"type": "string", "enum": ["A", "B"]},
                       "fine": {"type": "string", "enum": ["A", "B"]}}}}}}])
}

fn plain(content: &str) -> Value {
    json!({"thinking": true, "tools": tools(), "messages": [
        {"role": "system", "content": "You judge shell commands."},
        {"role": "user", "content": content, "snap": true}]})
}

fn queued(content: &str) -> Value {
    json!({"thinking": true, "tools": tools(), "queue": "next_command", "messages": [
        {"role": "system", "content": "You judge shell commands."},
        {"role": "user", "content": content, "snap": true},
        {"role": "assistant", "reasoning": "I get the first command.",
         "tool_calls": [{"id": "c0", "name": "next_command", "input": {}}], "snap": true}]})
}

fn context(value: &Value) -> Context {
    serde_json::from_value(value.clone()).unwrap()
}

fn request(spec_id: &str, contexts: &[Value], extra: Value) -> DecisionRequest {
    let mut body = json!({"model": "toy", "spec_id": spec_id, "case": "rm -rf build", "contexts": contexts});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    serde_json::from_value(body).unwrap()
}

fn held(f: &Fixture) -> String {
    f.service.hold_spec(spec_json()).unwrap().spec_id.to_string()
}

fn decide(f: &Fixture, contexts: &[Value], extra: Value) -> DecisionResponse {
    let id = held(f);
    f.service.decide(&request(&id, contexts, extra)).unwrap()
}

/// A direct run of the toy plan on the fake engine.
struct Run {
    /// The letter log probabilities of each asked question, in order.
    letters: Vec<Vec<f64>>,
    rendered_sha256: String,
    /// The tokens of the context, the prefix once, and each step.
    tokens: usize,
}

/// A direct run of the toy plan on the fake engine. Each step that continues from the prefix builds the context
/// again and feeds the prefix again.
fn direct(value: Option<&Value>, case: &str, ask: Option<&[String]>) -> Run {
    let family = ToyFamily::new();
    let spec = parse_spec(&spec_json(), &SpecRules::default()).unwrap();
    let asked = asked(&spec, ask, None).unwrap();
    let ctx = value.map(context);
    let rendering = ctx.as_ref().map(|c| family.render(c, false).unwrap());
    let input = DecisionInput {
        context: ctx.as_ref().zip(rendering.as_ref()),
        spec: &spec,
        case,
        asked: &asked,
    };
    let mut plan = family.decision_read(&input).unwrap();
    let fresh = || {
        let mut run = match &ctx {
            Some(c) => Direct::built(c),
            None => Direct::empty(),
        };
        let prefix = plan_prefix(&family, &input);
        if !prefix.is_empty() {
            for chunk in prefix.chunks(64) {
                run.feed(chunk, false);
            }
        }
        run
    };
    let mut run = fresh();
    let mut first = true;
    let mut out = vec![Vec::new(); asked.len()];
    let mut last: Option<Vec<f64>> = None;
    let mut tokens = run.rows;
    loop {
        match plan.advance(last.as_deref()).unwrap() {
            Advance::Done(done) => {
                return Run {
                    letters: out,
                    rendered_sha256: done.rendered_sha256,
                    tokens,
                };
            }
            Advance::Step(step) => {
                tokens += step.feed.len();
                if step.continues == Continues::Prefix && !first {
                    run = fresh();
                }
                first = false;
                let lp = log_probs(&run.feed_all(&step.feed)).unwrap();
                let letters: Vec<f64> = step.letters.iter().map(|&l| lp[l as usize]).collect();
                out[step.question] = letters.clone();
                last = Some(letters);
            }
        }
    }
}

fn plan_prefix(family: &ToyFamily, input: &DecisionInput<'_>) -> Vec<u32> {
    family.decision_read(input).unwrap().prefix().to_vec()
}

/// The raw letter log probabilities of each read of a response, in asked order.
fn raw(response: &DecisionResponse) -> Vec<Vec<Vec<f64>>> {
    response
        .reads
        .iter()
        .map(|read| {
            read.answers
                .values()
                .map(|answer| match answer {
                    ReadAnswer::Choice(a) => a.logprobs.values().copied().collect(),
                    ReadAnswer::Score(a) => a.logprobs.values().copied().collect(),
                    ReadAnswer::Noul(a) => a.logprobs.values().copied().collect(),
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_plain_decision_gives_the_numbers_of_a_direct_run_and_obeys_the_exact_relations() {
    let f = Fixture::new();
    let contexts = [plain("Commands wait in a queue."), plain("Judge each command alone.")];
    let r = decide(
        &f,
        &contexts,
        json!({"pool": {"method": "loglinear", "weights": "mass"}}),
    );
    let req = request(
        &held(&f),
        &contexts,
        json!({"pool": {"method": "loglinear", "weights": "mass"}}),
    );
    math::verify(&r, &req.expect()).unwrap();
    assert_eq!(r.reads.len(), 2);
    let mut tokens = 0;
    for (i, value) in contexts.iter().enumerate() {
        let run = direct(Some(value), "rm -rf build", None);
        assert_eq!(
            raw(&r)[i],
            run.letters,
            "read {i}: each question after the same prefix, bit for bit"
        );
        assert_eq!(r.reads[i].rendered_sha256, run.rendered_sha256);
        tokens += run.tokens as u64;
        assert_eq!(r.reads[i].context, Some(i as u64));
        let built = f.service.build(&context(value), false, Default::default()).unwrap();
        assert_eq!(r.reads[i].snapshot, built.head, "a read names the head of its context");
    }
    // The usage counts what the reads fed: the shared prefix once for each read, and each question.
    assert_eq!(
        r.usage,
        Usage {
            prompt: tokens,
            kept: 0,
            fed: tokens,
            completion: 0,
            reasoning: 0
        }
    );
    assert_eq!(r.answers.keys().collect::<Vec<_>>(), ["verdict", "undo", "fine"]);
    assert_eq!(r.identity.spec_id.as_ref().map(|id| id.to_string()), Some(held(&f)));
    assert_eq!(r.identity.template, ToyFamily::new().template_id());
}

#[test]
fn a_queue_decision_reads_each_question_after_the_letters_before_it() {
    let f = Fixture::new();
    let contexts = [queued("Commands wait in a queue.")];
    let r = decide(&f, &contexts, json!({}));
    math::verify(&r, &request(&held(&f), &contexts, json!({})).expect()).unwrap();
    let run = direct(Some(&contexts[0]), "rm -rf build", None);
    assert_eq!(raw(&r)[0], run.letters);
    assert_eq!(r.reads[0].rendered_sha256, run.rendered_sha256);
    assert_eq!(r.usage.fed, run.tokens as u64);
}

#[test]
fn a_decision_with_no_context_has_one_read_with_context_null() {
    let f = Fixture::new();
    let r = decide(&f, &[], json!({}));
    math::verify(&r, &request(&held(&f), &[], json!({})).expect()).unwrap();
    assert_eq!(r.reads.len(), 1);
    assert_eq!((r.reads[0].context, r.reads[0].snapshot.clone()), (None, None));
    let run = direct(None, "rm -rf build", None);
    assert_eq!(raw(&r)[0], run.letters);
    assert_eq!(r.usage.fed, run.tokens as u64);
}

#[test]
fn ask_and_options_narrow_the_read() {
    let f = Fixture::new();
    let contexts = [plain("Commands wait in a queue.")];
    let extra = json!({"ask": ["verdict"], "options": {"verdict": ["report", "allow"]}});
    let r = decide(&f, &contexts, extra.clone());
    math::verify(&r, &request(&held(&f), &contexts, extra).expect()).unwrap();
    let ReadAnswer::Choice(verdict) = &r.reads[0].answers["verdict"] else {
        panic!()
    };
    assert_eq!(verdict.logprobs.keys().collect::<Vec<_>>(), ["allow", "report"]);
    assert_eq!(r.answers.len(), 1);
}

#[test]
fn a_repeated_decision_gives_the_same_numbers_and_keeps_its_contexts() {
    let f = Fixture::new();
    let contexts = [plain("Commands wait in a queue.")];
    let first = decide(&f, &contexts, json!({}));
    let again = decide(&f, &contexts, json!({}));
    assert_eq!(raw(&first), raw(&again), "a decision replays bit for bit (gogi rule 1)");
    assert_eq!(first.usage.prompt, again.usage.prompt);
    let built = f
        .service
        .build(&context(&contexts[0]), true, Default::default())
        .unwrap();
    assert_eq!(
        again.usage.kept, built.usage.prompt,
        "the second read keeps the whole context"
    );
    assert!(again.usage.fed < first.usage.fed);
}

#[test]
fn usage_signals_and_hashes_match_gogi_fake_for_the_same_requests() {
    let fake = Server::start(Config::default()).unwrap();
    let f = Fixture::new();
    let fake_spec = post(&fake, "/council/v2/specs", spec_json());
    let id = held(&f);
    assert_eq!(fake_spec["spec_id"].as_str(), Some(id.as_str()));
    let cases: Vec<(Vec<Value>, Value)> = vec![
        (vec![plain("Commands wait in a queue.")], json!({})),
        (
            vec![plain("Commands wait in a queue."), plain("Another <end> view.")],
            json!({}),
        ),
        (vec![], json!({"case": {"command": "git push", "cwd": "/src"}})),
        (
            vec![queued("Commands wait in a queue.")],
            json!({"ask": ["undo", "verdict"]}),
        ),
        (
            vec![plain("x")],
            json!({"ask": ["verdict"], "options": {"verdict": ["ask", "report"]}}),
        ),
    ];
    for (contexts, extra) in cases {
        let req = request(&id, &contexts, extra);
        let want = post(&fake, "/council/v2/decisions", serde_json::to_value(&req).unwrap());
        let got = f.service.decide(&req).unwrap();
        let usage: Usage = serde_json::from_value(want["usage"].clone()).unwrap();
        // gogi-fake counts the shared text of a plain read once for each question, and the letter that ends a queue
        // read. The nagaya counts what it fed: the shared text once, and no last letter (spec 11.3c). The counts
        // agree for a plain read of one question.
        let queue = req.contexts.iter().any(|c| c.queue.is_some());
        let one = !queue && req.ask.as_ref().is_some_and(|a| a.len() == 1);
        if one {
            assert_eq!(got.usage, usage, "{req:?}");
        } else {
            assert_eq!(got.usage.kept, usage.kept, "{req:?}");
            assert!(got.usage.fed < usage.fed, "{req:?}");
        }
        assert_eq!(serde_json::to_value(&got.signals).unwrap(), want["signals"], "{req:?}");
        let hashes: Vec<&str> = want["reads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["rendered_sha256"].as_str().unwrap())
            .collect();
        let ours: Vec<&str> = got.reads.iter().map(|r| r.rendered_sha256.as_str()).collect();
        // A queue read writes the likeliest letters into its text, and the two models give other letters. The
        // direct run checks the hash of a queue read.
        if !queue {
            assert_eq!(ours, hashes, "{req:?}");
        }
        let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(
            keys(&serde_json::to_value(&got.answers).unwrap()),
            keys(&want["answers"]),
            "{req:?}"
        );
    }
    assert_eq!(case_text(&json!({"b": 1, "a": 2})), r#"{"b":1,"a":2}"#);
}

fn invalid_param(result: Result<DecisionResponse, ServiceError>) -> String {
    match result {
        Err(ServiceError::Invalid(invalid)) => invalid.param.unwrap_or_default(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn each_refused_decision_names_its_field_and_reads_nothing() {
    let f = Fixture::new();
    let id = held(&f);
    let one = [plain("Commands wait in a queue.")];
    assert_eq!(
        invalid_param(f.service.decide(&request(&id, &one, json!({"think": {"tokens": 4}})))),
        "think"
    );
    assert_eq!(
        invalid_param(f.service.decide(&request(&id, &one, json!({"ask": ["nope"]})))),
        "ask"
    );
    let mixed = [plain("a"), queued("b")];
    assert_eq!(
        invalid_param(f.service.decide(&request(&id, &mixed, json!({})))),
        "contexts"
    );
    let unknown = format!("sha256:{}", "0".repeat(64));
    assert!(matches!(
        f.service.decide(&request(&unknown, &one, json!({}))),
        Err(ServiceError::NotFound(_))
    ));
    let big = json!({"case": "x".repeat(65537)});
    assert!(matches!(
        f.service.decide(&request(&id, &one, big)),
        Err(ServiceError::TooLarge { param, .. }) if param == "case_bytes"
    ));
    // think with 0 tokens needs no capability.
    f.service
        .decide(&request(&id, &one, json!({"think": {"tokens": 0}})))
        .unwrap();
    assert_eq!(f.service.free_slots(), 3);
}

#[test]
fn a_context_past_the_decision_limit_is_too_large() {
    let mut config = ServiceConfig::new("toy", council_support::ENGINE);
    config.decision_context_tokens = 40;
    let f = Fixture::with(0, 1 << 20, None, config);
    let id = held(&f);
    let err = f
        .service
        .decide(&request(&id, &[plain("Commands wait in a queue.")], json!({})))
        .unwrap_err();
    assert!(
        matches!(&err, ServiceError::TooLarge { param, .. } if param == "decision_context_tokens"),
        "{err:?}"
    );
}

/// Kaibo review 2026-10-20, finding 8: a context that fits the decision limit can still fit no slot with its prefix.
/// The limit that breaks is then the slot, which identity gives as `context_tokens` (spec 4.3).
#[test]
fn a_context_that_fits_the_decision_limit_but_no_slot_names_the_slot_limit() {
    let f = Fixture::new();
    let id = held(&f);
    // 16384 bytes of cells in a slot hold 2048 rows. The decision limit is 3072 tokens.
    let long = plain(&"x".repeat(2100));
    let err = f.service.decide(&request(&id, &[long], json!({}))).unwrap_err();
    assert!(
        matches!(&err, ServiceError::TooLarge { param, .. } if param == "context_tokens"),
        "{err:?}"
    );
    assert_eq!(f.service.free_slots(), 3);
}

#[test]
fn a_decision_past_its_timeout_is_a_timeout_and_frees_its_slot() {
    let factory = TermFactory {
        delay: Duration::from_millis(20),
        ..TermFactory::toy()
    };
    let f = Fixture::with_factory(
        factory,
        1 << 20,
        None,
        ServiceConfig::new("toy", council_support::ENGINE),
    );
    let id = held(&f);
    let contexts = [plain("Commands wait in a queue."), plain("Another view.")];
    let err = f
        .service
        .decide(&request(&id, &contexts, json!({"timeout_ms": 30})))
        .unwrap_err();
    assert_eq!(err, ServiceError::Timeout);
    assert_eq!(f.service.free_slots(), 3, "an expired job gives its slot back");
    // The same decision with time to spare succeeds.
    f.service
        .decide(&request(&id, &contexts, json!({"timeout_ms": 60_000})))
        .unwrap();
}
