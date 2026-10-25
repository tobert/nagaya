//! The pure parts of a decision: the asked questions and the answers (gogi "Decisions", "Answers").

use gogi_proto::context::Usage;
use gogi_proto::decision::{
    DecisionIdentity, DecisionResponse, Expect, Pool, PoolMethod, PoolWeights, PooledAnswer, Read, ReadAnswer,
};
use gogi_proto::math;
use gogi_proto::spec::{Spec, SpecRules, parse_spec};
use nagaya_council::decision::{answers, asked};
use nagaya_council::family::Asked;
use serde_json::{Map, Value, json};

fn spec() -> Spec {
    let value = json!({"name": "g", "instructions": "i", "input_label": "x", "questions": [
        {"id": "verdict", "type": "choice", "instructions": "what", "criteria": [
            {"option": "allow", "means": "a"}, {"option": "ask", "means": "b"}, {"option": "no", "means": "c"}]},
        {"id": "undo", "type": "score", "instructions": "how", "criteria": ["easy", "hard", "never"]},
        {"id": "fine", "type": "noul", "instructions": "fine?"}]});
    parse_spec(&value, &SpecRules::default()).unwrap()
}

fn ids(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

fn options(value: Value) -> Map<String, Value> {
    value.as_object().unwrap().clone()
}

fn param(result: Result<Vec<Asked>, gogi_proto::Invalid>) -> String {
    result.unwrap_err().param.unwrap()
}

#[test]
fn with_no_ask_a_decision_asks_every_question_with_every_option() {
    let got = asked(&spec(), None, None).unwrap();
    assert_eq!(
        got,
        [
            Asked {
                question: 0,
                options: vec![0, 1, 2]
            },
            Asked {
                question: 1,
                options: vec![0, 1, 2]
            },
            Asked {
                question: 2,
                options: vec![0, 1]
            },
        ]
    );
}

#[test]
fn ask_names_questions_and_they_go_in_spec_order() {
    let got = asked(&spec(), Some(&ids(&["fine", "verdict"])), None).unwrap();
    assert_eq!(got.iter().map(|a| a.question).collect::<Vec<_>>(), [0, 2]);
}

#[test]
fn options_narrow_a_choice_and_keep_the_positions_of_the_spec() {
    let narrowed = options(json!({"verdict": ["no", "allow"]}));
    let got = asked(&spec(), None, Some(&narrowed)).unwrap();
    assert_eq!(got[0].options, [0, 2]);
}

#[test]
fn each_wrong_ask_or_option_is_a_client_error_that_names_its_field() {
    let s = spec();
    assert_eq!(param(asked(&s, Some(&ids(&["nope"])), None)), "ask");
    assert_eq!(param(asked(&s, Some(&ids(&["fine", "fine"])), None)), "ask");
    for bad in [
        json!({"nope": ["a", "b"]}),
        json!({"undo": ["0", "1"]}),
        json!({"verdict": ["allow"]}),
        json!({"verdict": ["allow", "allow"]}),
        json!({"verdict": ["allow", "maybe"]}),
        json!({"verdict": "allow"}),
    ] {
        assert_eq!(param(asked(&s, None, Some(&options(bad.clone())))), "options", "{bad}");
    }
    let only_undo = asked(
        &s,
        Some(&ids(&["undo"])),
        Some(&options(json!({"verdict": ["allow", "ask"]}))),
    );
    assert_eq!(
        param(only_undo),
        "options",
        "options name a question that the decision does not ask"
    );
}

/// A response with the answers of `logprobs`, as a server gives it.
fn response(logprobs: &[Vec<Vec<f64>>], named: bool, pool: &Pool) -> DecisionResponse {
    let s = spec();
    let a = asked(&s, None, None).unwrap();
    let got = answers(&s, &a, logprobs, pool).unwrap();
    DecisionResponse {
        model: "toy".to_owned(),
        answers: got.pooled,
        reads: got
            .reads
            .into_iter()
            .enumerate()
            .map(|(i, answers)| Read {
                context: named.then_some(i as u64),
                snapshot: None,
                described: None,
                answers,
                rendered_hash: gogi_proto::ContentId::parse(gogi_proto::content_id(b"x")).unwrap(),
                thought: None,
                tokens: None,
                ms: None,
            })
            .collect(),
        pool: got.pool,
        signals: None,
        identity: DecisionIdentity {
            model: "toy".to_owned(),
            weight_hash: "w".to_owned(),
            tokenizer_hash: "t".to_owned(),
            template: "p".to_owned(),
            engine: "e".to_owned(),
            spec_id: None,
        },
        usage: Usage::default(),
        queue_ms: None,
        ms: None,
    }
}

fn reads() -> Vec<Vec<Vec<f64>>> {
    vec![
        vec![vec![-1.0, -2.5, -3.0], vec![-0.7, -1.9, -4.0], vec![-0.2, -2.0]],
        vec![vec![-3.0, -0.4, -2.0], vec![-2.2, -0.9, -1.1], vec![-1.5, -0.6]],
    ]
}

#[test]
fn the_answers_obey_the_exact_relations_for_each_pool() {
    for method in [PoolMethod::Linear, PoolMethod::Loglinear] {
        for (weights, values) in [
            (PoolWeights::Uniform, None),
            (PoolWeights::Mass, None),
            (PoolWeights::Given, Some(vec![3.0, 1.0])),
        ] {
            let pool = Pool {
                method: Some(method),
                weights: Some(weights),
                values,
            };
            let r = response(&reads(), true, &pool);
            math::verify(
                &r,
                &Expect {
                    pool: pool.clone(),
                    contexts: 2,
                },
            )
            .unwrap();
        }
    }
}

#[test]
fn a_read_keeps_its_raw_logprobs_by_option_and_the_pool_names_each_context() {
    let r = response(&reads(), true, &Pool::default());
    let ReadAnswer::Choice(first) = &r.reads[0].answers["verdict"] else {
        panic!()
    };
    assert_eq!(first.logprobs.keys().collect::<Vec<_>>(), ["allow", "ask", "no"]);
    assert_eq!(first.logprobs["ask"], -2.5);
    assert_eq!(first.choice, "allow");
    let PooledAnswer::Choice(pooled) = &r.answers["verdict"] else {
        panic!()
    };
    assert!(!pooled.agree, "the reads have different top options");
    let loo = pooled.leave_one_out.as_ref().unwrap();
    assert_eq!(loo.keys().collect::<Vec<_>>(), ["0", "1"]);
    // Without read 0, the pool is read 1 alone.
    let ReadAnswer::Choice(second) = &r.reads[1].answers["verdict"] else {
        panic!()
    };
    assert_eq!(loo["0"].as_ref().unwrap(), &second.probabilities);
    let ReadAnswer::Noul(fine) = &r.reads[0].answers["fine"] else {
        panic!()
    };
    assert_eq!(fine.logprobs.keys().collect::<Vec<_>>(), ["yes", "no"]);
    let ReadAnswer::Score(undo) = &r.reads[0].answers["undo"] else {
        panic!()
    };
    assert_eq!(undo.legend.values().collect::<Vec<_>>(), ["easy", "hard", "never"]);
}

#[test]
fn one_read_with_no_context_has_no_leave_one_out() {
    let one = vec![reads()[0].clone()];
    let r = response(&one, false, &Pool::default());
    math::verify(
        &r,
        &Expect {
            pool: Pool::default(),
            contexts: 0,
        },
    )
    .unwrap();
    let PooledAnswer::Choice(pooled) = &r.answers["verdict"] else {
        panic!()
    };
    assert!(pooled.leave_one_out.is_none());
    assert!(pooled.agree);
}
