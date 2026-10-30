//! The engine on a B70 card against sentence-transformers 6.1.0 in f32 (spec 8.3, 11.10): the oracle of
//! `tools/embeddinggemma2_golden.sh`. The gate is 1e-4 per value of each normalized vector, at each cut.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test --release -p nagaya-mk-embeddinggemma2-bmg --features xe --test xe_oracle -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use nagaya_abi::Embedding;
use nagaya_mk_embeddinggemma2_bmg::{Config, EmbeddingGemma2Engine};

/// The gate of spec 11.10 on each value of a normalized vector.
const GATE: f64 = 1e-4;

fn open() -> EmbeddingGemma2Engine {
    let card = common::cards()[1].clone();
    EmbeddingGemma2Engine::open(
        Config {
            model_dir: common::model_dir(),
        },
        &card,
    )
    .expect("open EmbeddingGemma 2 on card 1")
}

#[test]
fn the_hidden_states_of_one_input_follow_the_reference_layer_by_layer() {
    if common::absent() || common::golden_absent("golden.safetensors") {
        return;
    }
    let golden = common::golden("golden.safetensors");
    let ids = golden["ascii.query.ids"].ids();
    let mut engine = open();
    let probes = engine.probes(&ids).expect("the probes run");
    let report = |name: &str, got: &[f32], want: &str| -> f64 {
        let err = common::rel(got, &golden[want].f32());
        println!("{name}: relative error {err:.3e}");
        err
    };
    assert!(report("scaled embeddings", &probes.embed, "parts.embed") < 1e-7);
    assert!(report("per-layer inputs", &probes.ple, "parts.ple") < 1e-5);
    let mut worst = 0.0f64;
    for (l, layer) in probes.layers.iter().enumerate() {
        worst = worst.max(report(&format!("layer {l}"), layer, &format!("parts.layer{l}")));
    }
    let out = report("last output", &probes.out, "parts.token_embeddings");
    assert!(worst < 1e-4, "a layer is {worst:.3e} from the reference");
    assert!(out < 1e-4, "the last output is {out:.3e} from the reference");
}

#[test]
fn every_case_and_task_is_within_the_gate_at_every_cut() {
    if common::absent() || common::golden_absent("cases.json") || common::golden_absent("golden.safetensors") {
        return;
    }
    let golden = common::golden("golden.safetensors");
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(common::golden_dir().join("cases.json")).unwrap()).unwrap();
    let mut engine = open();
    let (mut worst, mut lowest) = (0.0f64, 1.0f64);
    let (mut worst_bf16, mut lowest_bf16) = (0.0f64, 1.0f64);
    let mut failures = Vec::new();
    for case in cases["cases"].as_array().unwrap() {
        let key = format!("{}.{}", case["case"].as_str().unwrap(), case["task"].as_str().unwrap());
        let ids = golden[&format!("{key}.ids")].ids();
        assert_eq!(ids.len() as u64, case["tokens"].as_u64().unwrap(), "{key}");
        let start = std::time::Instant::now();
        let mean = engine.embed(&ids).expect("the input embeds");
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(mean.len(), 768);
        let mut line = format!("{key:32} {:5} tokens {ms:8.1} ms", ids.len());
        for dims in [768usize, 512, 256, 128] {
            let got = common::normalized(&mean[..dims]);
            let want = golden[&format!("{key}.v{dims}")].f32();
            let err = common::max_abs(&got, &want);
            let cos = common::cosine(&got, &want);
            worst = worst.max(err);
            lowest = lowest.min(cos);
            line += &format!("  {dims}: {err:.2e} cos {cos:.9}");
            if err >= GATE {
                failures.push(format!("{key} at {dims}: {err:.3e}"));
            }
        }
        if let Some(bf16) = golden.get(&format!("{key}.bf16")) {
            let got = common::normalized(&mean);
            worst_bf16 = worst_bf16.max(common::max_abs(&got, &bf16.f32()));
            lowest_bf16 = lowest_bf16.min(common::cosine(&got, &bf16.f32()));
        }
        println!("{line}");
    }
    println!("against f32: largest difference {worst:.3e}, lowest cosine {lowest:.9}");
    println!("against bf16: largest difference {worst_bf16:.3e}, lowest cosine {lowest_bf16:.9}");
    assert!(failures.is_empty(), "past the gate of {GATE}: {failures:#?}");
}
