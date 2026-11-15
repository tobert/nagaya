//! The decoded head of a reply on the LFM3 engine (milestone 32 step 4, spec 6.14 "The decoded head"). The card gates:
//!
//! - the second turn by `parent` starts from the head of the first reply, so it feeds only its new tokens;
//! - the same turn twice from the same head gives the same reply with a seed, and the same head;
//! - a read from the decoded head and a read of the same context built from its text on a new nagaya both run. The
//!   test prints the largest absolute difference of their next-token log probabilities. That number goes into spec
//!   6.14, "The bits differ a little".
//!
//! The council gives log probabilities, not logits, so the test measures those: the top 64 tokens and the letters.
//!
//! It needs card 0 and `LFM3_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_lfm3_head -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod card_support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::Registry;
use nagaya::cpu_sets::for_machine_with_devices_ignoring;
use nagaya::http::{Council, router};
use nagaya::inventory::Probe;
use nagaya::lifecycle::{self, Options, Ready};
use nagaya::probe::FileProbe;
use nagaya::profile::PlacementProfile;
use nagaya::serving::{FamilyRegistry, council};
use nagaya::signal::SignalFlag;
use serde_json::{Value, json};
use tower::ServiceExt;

/// The iGPU of b70host drives the display. The b70host config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

/// The served model.
const MODEL: &str = "lfm3";

fn directory(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the model folder")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the model folder");
            None
        }
    }
}

/// `LFM3-1.7B` resident on device 0, with one slot, as `tests/xe_lfm3.rs` puts it.
fn profile(dir: &Path) -> PlacementProfile {
    let card = card_support::device();
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"{MODEL}\"\nkind = \"mk-lfm3-bmg\"\ndevices = [{card}]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = 1\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\nobjects = 4294967296\n"
    ))
    .expect("the profile parses")
}

/// A nagaya with the model on card 0, and a new store.
struct Nagaya {
    ready: Ready,
    council: Arc<Council>,
    _store: tempfile::TempDir,
}

impl Nagaya {
    fn start(dir: &Path) -> Self {
        let probe = FileProbe::new("/");
        let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
        let ignore = vec![IGNORE.to_owned()];
        inventory.ignore(&ignore).expect("the machine has the iGPU");
        let store = tempfile::tempdir().unwrap();
        let profile = profile(dir);
        let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
        let start = Instant::now();
        let ready = lifecycle::start_with(
            profile.clone(),
            Registry::compiled_in(),
            inventory,
            store.path(),
            None,
            Options::new(Duration::from_secs(600)).cpu_sets(sets),
        )
        .expect("the process opens the engine");
        println!("opened in {} ms", start.elapsed().as_millis());
        let ports = ready.process.ports();
        let council = Arc::new(
            council(
                &FamilyRegistry::compiled_in(),
                &profile,
                &ready.opened,
                &ports,
                store.path(),
                &[],
                None,
            )
            .expect("the process serves the model")
            .with_loads(ports.clone()),
        );
        Self {
            ready,
            council,
            _store: store,
        }
    }

    /// Stop the process, so the next one gets the card.
    fn stop(mut self) {
        self.council.drain();
        let stopped = lifecycle::stop(&mut self.ready, None, Duration::from_secs(30), &SignalFlag::new());
        println!("stop: {}", stopped.log_line());
    }

    fn send(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let request = Request::builder().method(method).uri(path);
            let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
            let response = router(Arc::clone(&self.council))
                .oneshot(request.body(body).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            assert_eq!(status, StatusCode::OK, "{method} {path}: {value}");
            value
        })
    }

    fn generate(&self, context: &Value, sample: &Value) -> Value {
        let started = Instant::now();
        let body = json!({"model": MODEL, "context": context, "max_tokens": 96, "sample": sample});
        let done = self.send("POST", "/council/v2/generations", Some(body));
        println!(
            "generation in {} ms: finish {}, usage {}, head {}",
            started.elapsed().as_millis(),
            done["finish"],
            done["usage"],
            done["head"]
        );
        done
    }

    /// Read the next token after `context` and the suffix: the log probability of the top 64 tokens and of the
    /// letters, by token id or letter, and the usage of the read.
    fn read(&self, context: &Value) -> (BTreeMap<String, f64>, Value) {
        let body = json!({"model": MODEL, "contexts": [context], "suffixes": ["\nVerdict: "],
                          "letters": ["A", "B", "C", "D"], "top": 64});
        let response = self.send("POST", "/council/v2/reads", Some(body));
        let read = &response["reads"][0];
        let mut logprobs = BTreeMap::new();
        for top in read["top"].as_array().expect("a read lists its top tokens") {
            logprobs.insert(format!("id {}", top["id"]), top["logprob"].as_f64().unwrap());
        }
        for (letter, logprob) in read["logprobs"].as_object().expect("a read gives its letters") {
            logprobs.insert(format!("letter {letter}"), logprob.as_f64().unwrap());
        }
        (logprobs, response["usage"].clone())
    }
}

fn count(value: &Value) -> u64 {
    value.as_u64().unwrap_or_else(|| panic!("{value} is not a count"))
}

#[test]
fn a_turn_by_parent_starts_from_the_decoded_head_on_card_0() {
    let Some(dir) = directory("LFM3_MODEL_DIR") else {
        return;
    };
    let first = json!({"role": "user", "content": "Name three colors.", "snap": true});
    let second = json!({"role": "user", "content": "Now name three animals."});
    let greedy = json!({"temperature": 0});
    let seeded = json!({"temperature": 0.7, "seed": 11});

    let nagaya = Nagaya::start(&dir);
    let one = nagaya.generate(&json!({"thinking": false, "messages": [first]}), &greedy);
    assert_eq!(one["finish"], "end_turn", "the first reply must end its turn: {one}");
    let head = one["head"].as_str().unwrap_or_else(|| {
        panic!(
            "the first reply holds no decoded head. The family rendered the reply with other tokens than the model \
             sampled, for example with white space removed (spec 6.14). Use a prompt whose reply renders as it was \
             sampled: {one}"
        )
    });
    let head_tokens = one["snapshots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|snapshot| snapshot["id"] == head)
        .map(|snapshot| count(&snapshot["tokens"]))
        .expect("the response lists its head");
    let parent = one["context"].as_str().expect("a reply names its record");

    // Gate 1: the second turn by `parent` starts from the head, so it feeds only its new tokens.
    let turn = json!({"parent": parent, "messages": [second]});
    let two = nagaya.generate(&turn, &seeded);
    let usage = &two["usage"];
    assert_eq!(
        count(&usage["kept"]),
        head_tokens,
        "the turn starts from the decoded head: {two}"
    );
    assert_eq!(count(&usage["fed"]), count(&usage["prompt"]) - head_tokens);
    println!(
        "turn 2 by parent: the head holds {head_tokens} tokens, and the turn fed {} tokens: its new message and the \
         generation prompt",
        usage["fed"]
    );

    // Gate 2: the same turn again, with the same seed, gives the same reply and the same head. The id of a head is
    // the hash of its bytes.
    let again = nagaya.generate(&turn, &seeded);
    for member in ["content", "reasoning", "finish", "head", "context"] {
        assert_eq!(
            again[member], two[member],
            "{member} of the same turn from the same head"
        );
    }
    assert!(count(&again["usage"]["kept"]) >= head_tokens);

    // Gate 3: the next-token log probabilities from the decoded head, and from a build of the text on a new nagaya.
    let (from_head, head_usage) = nagaya.read(&turn);
    assert!(count(&head_usage["kept"]) >= head_tokens, "{head_usage}");
    let record = nagaya.send("GET", &format!("/council/v2/contexts/{parent}"), None);
    let mut whole = record["context"].clone();
    whole["messages"]
        .as_array_mut()
        .expect("a record holds its messages")
        .push(turn["messages"][0].clone());
    nagaya.stop();

    let fresh = Nagaya::start(&dir);
    let (from_text, text_usage) = fresh.read(&whole);
    assert_eq!(
        count(&text_usage["kept"]),
        0,
        "a new store holds no prefix: {text_usage}"
    );
    let mut largest: (f64, &str) = (0.0, "");
    let mut shared = 0;
    for (name, logprob) in &from_head {
        if let Some(other) = from_text.get(name) {
            shared += 1;
            let difference = (logprob - other).abs();
            if difference > largest.0 {
                largest = (difference, name);
            }
        }
    }
    assert!(shared > 0, "the two reads share no token: {from_head:?} {from_text:?}");
    println!(
        "decoded head against a build of the text: the largest absolute difference of the next-token log \
         probabilities is {:e}, at {}, over {shared} tokens that both reads list",
        largest.0, largest.1
    );
    // A turn on the state that the rule made runs too.
    let three = fresh.generate(&whole, &seeded);
    assert!(count(&three["usage"]["completion"]) > 0, "{three}");
    fresh.stop();
}
