//! The beats route with two engines in one process: `mk-beatthis-cpu` in the CPU group and `mk-beatthis-gfx1151` on
//! the card, each a served beats model (spec 11.19). Both give upstream's beat times for the oracle's 10 s clip.
//!
//! A card test. Stop `nagaya.service` first (spec 3.1):
//! `cargo test -p nagaya --features hip --test beats_gpu_route -- --test-threads=1`.
#![cfg(feature = "hip")]

#[path = "../../nagaya-family-beatthis/tests/common/mod.rs"]
mod golden;

use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::Registry;
use nagaya::http::{Council, router};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::Start;
use nagaya::profile::{BudgetSpec, CpuSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::Tier;
use nagaya_store::objects::ObjectStore;
use serde_json::{Value, json};
use tower::ServiceExt;

fn engine(name: &str, kind: &str, devices: Vec<u32>, weights: &str) -> EnginePlacement {
    let mut config = toml::Table::new();
    config.insert("weights".into(), toml::Value::String(weights.into()));
    EnginePlacement {
        name: name.to_owned(),
        kind: kind.to_owned(),
        devices,
        config,
        family: toml::Table::new(),
        lab: toml::Table::new(),
        share: 1,
        residency: Residency::Resident,
        pinned: true,
    }
}

fn wav(samples: &[f32]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 22_050,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut out = Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut out, spec).unwrap();
    for &s in samples {
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
    out.into_inner()
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Body, mime: &str) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", mime)
            .body(body)
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

#[test]
fn the_cpu_and_the_gpu_engine_both_give_upstreams_beats() {
    let Some(g) = golden::golden() else { return };
    if golden::weights("small0").is_none() {
        return;
    }
    let Some(dir_of_models) = std::env::var_os(golden::MODEL_VARIABLE) else {
        return;
    };
    let weights = std::path::PathBuf::from(dir_of_models)
        .join("small0.safetensors")
        .display()
        .to_string();
    let dir = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(nagaya::beatthis::KIND, Box::new(nagaya::beatthis::BeatThisFactory));
    registry.register(
        nagaya::beatthis_gfx1151::KIND,
        Box::new(nagaya::beatthis_gfx1151::BeatThisGfx1151Factory),
    );
    let profile = PlacementProfile {
        engines: vec![
            engine("beats-cpu", nagaya::beatthis::KIND, vec![], &weights),
            engine("beats-gpu", nagaya::beatthis_gfx1151::KIND, vec![0], &weights),
        ],
        budgets: vec![BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Device(0)), TierSpec::from(Tier::Host)],
            bytes: 8 << 30,
        }],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
        cpu: Some(CpuSpec { threads: 4 }),
    };
    let inventory = Inventory {
        devices: vec![Device {
            id: 0,
            name: "gfx1151".to_owned(),
            pci_address: "0000:00:00.0".to_owned(),
            memory_bytes: 16 << 30,
            unified: true,
            ignored: false,
        }],
        host_bytes: 16 << 30,
    };
    let mut process = Start::new(profile.clone(), registry, inventory)
        .store(dir.path())
        .heartbeat(Duration::from_millis(50))
        .timeout(Duration::from_secs(30))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(60)).unwrap();
    let services = nagaya::beats::services(&profile, &process.ports()).unwrap();
    assert_eq!(services.len(), 2, "one service for each engine");
    let objects = ObjectStore::open(&dir.path().join("objects"), 1 << 26).unwrap();
    let council = Arc::new(
        Council::new(Vec::new(), None)
            .unwrap()
            .with_beats(services)
            .with_object_store(Arc::new(Mutex::new(objects)))
            .with_loads(process.ports()),
    );
    let (wave, _) = g.f32("click97_34.wave");
    let bytes = wav(&wave);
    let id = format!("blake3:{}", nagaya_store::content_id(&bytes).hex());
    let (status, body) = send(
        &council,
        "PUT",
        &format!("/council/v2/objects/{id}"),
        Body::from(bytes),
        "audio/wav",
    );
    assert!(status.is_success(), "{body}");
    let (want, _) = g.f64("small0.click97_34.beats_s");
    for model in ["beats-cpu", "beats-gpu"] {
        let request = json!({"model": model, "audio": id});
        let (status, body) = send(
            &council,
            "POST",
            "/nagaya/v1/analysis/beats",
            Body::from(request.to_string()),
            "application/json",
        );
        assert_eq!(status, StatusCode::OK, "{model}: {body}");
        let got: Vec<f64> = body["beats"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(got, want, "{model}");
        eprintln!("{model}: {} beats in {} ms", got.len(), body["usage"]["ms"]);
    }
}
