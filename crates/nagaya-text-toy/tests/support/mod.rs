//! Helpers for the tests: a client of gogi-fake, and a driver that runs a read plan on the toy model.

use std::io::{Read, Write};
use std::net::TcpStream;

use gogi_fake::Server;
use gogi_proto::context::Context;
use gogi_proto::spec::{Spec, SpecRules, parse_spec};
use gogi_toy::{State, log_softmax};
use nagaya_council::family::{Advance, Continues, ReadPlan};
use serde_json::Value;

/// The seed of the model that gogi-fake serves as `toy`.
pub const SEED: u64 = 0;

pub fn post(s: &Server, path: &str, body: Value) -> (u16, Value) {
    let addr = s.url().trim_start_matches("http://").to_string();
    let mut c = TcpStream::connect(&addr).unwrap();
    let b = body.to_string();
    write!(
        c,
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
        b.len()
    )
    .unwrap();
    let mut raw = String::new();
    c.read_to_string(&mut raw).unwrap();
    let (head, body) = raw.split_once("\r\n\r\n").unwrap();
    let code = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (code, serde_json::from_str(body).unwrap_or(Value::Null))
}

pub fn context(v: Value) -> Context {
    serde_json::from_value(v).unwrap()
}

pub fn spec(v: &Value) -> Spec {
    parse_spec(v, &SpecRules::default()).unwrap()
}

/// What a plan gave when a toy model ran it.
#[derive(Debug)]
pub struct Driven {
    /// For each step, the question and the letter log probabilities.
    pub steps: Vec<(usize, Vec<f64>)>,
    pub rendered_sha256: String,
}

/// Run a plan on the toy model of seed `SEED`. `context_tokens` are the tokens of the context's rendering.
pub fn drive(plan: &mut dyn ReadPlan, context_tokens: &[u32]) -> Driven {
    let after_prefix = State::after(SEED, context_tokens).fold(plan.prefix());
    let mut state = after_prefix;
    let mut steps = Vec::new();
    let mut last: Option<Vec<f64>> = None;
    loop {
        match plan.advance(last.as_deref()).unwrap() {
            Advance::Done(done) => {
                return Driven {
                    steps,
                    rendered_sha256: done.rendered_sha256,
                };
            }
            Advance::Step(step) => {
                let from = match step.continues {
                    Continues::Prefix => after_prefix,
                    Continues::PreviousStep => state,
                };
                state = from.fold(&step.feed);
                let lp = log_softmax(&state.logits());
                let letters: Vec<f64> = step.letters.iter().map(|&id| lp[id as usize]).collect();
                steps.push((step.question, letters.clone()));
                last = Some(letters);
            }
        }
    }
}

trait Fold {
    fn fold(self, tokens: &[u32]) -> State;
}

impl Fold for State {
    fn fold(self, tokens: &[u32]) -> State {
        tokens.iter().fold(self, |s, &t| s.step(t))
    }
}
