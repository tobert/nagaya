//! `POST /nagaya/v1/analysis/beats` end to end (spec 11.19): a process with the CPU group, the real Beat This! engine
//! with `small0`, an `audio/wav` object, and the route. The beat times must equal upstream's for the oracle's clip.
//!
//! The weights come from `BEATTHIS_MODEL_DIR` and the clip from `BEATTHIS_GOLDEN_DIR` (`tools/beatthis_golden.sh`).
//! Without them each test skips; `NAGAYA_REQUIRE_MODELS=1` makes them fail.

#[path = "../../nagaya-family-beatthis/tests/common/mod.rs"]
mod golden;

use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::Registry;
use nagaya::http::{Council, router};
use nagaya::inventory::Inventory;
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, CpuSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::Tier;
use nagaya_store::objects::ObjectStore;
use serde_json::{Value, json};
use tower::ServiceExt;

struct Rig {
    _process: Process,
    council: Arc<Council>,
    _dir: tempfile::TempDir,
}

fn rig() -> Option<Rig> {
    let dir_of_models = std::env::var_os(golden::MODEL_VARIABLE).map(std::path::PathBuf::from);
    golden::weights("small0")?;
    let weights = dir_of_models.unwrap().join("small0.safetensors");
    let dir = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(nagaya::beatthis::KIND, Box::new(nagaya::beatthis::BeatThisFactory));
    let mut config = toml::Table::new();
    config.insert("weights".into(), toml::Value::String(weights.display().to_string()));
    let profile = PlacementProfile {
        engines: vec![EnginePlacement {
            name: "beatthis".to_owned(),
            kind: nagaya::beatthis::KIND.to_owned(),
            devices: vec![],
            config,
            family: toml::Table::new(),
            lab: toml::Table::new(),
            share: 1,
            residency: Residency::Resident,
            pinned: true,
        }],
        budgets: vec![BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Host)],
            bytes: 4 << 30,
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
        devices: vec![],
        host_bytes: 8 << 30,
    };
    let mut process = Start::new(profile, registry, inventory)
        .store(dir.path())
        .heartbeat(Duration::from_millis(50))
        .timeout(Duration::from_secs(30))
        .start()
        .unwrap();
    process.wait_opens(Duration::from_secs(60)).unwrap();
    let services = nagaya::beats::services(&profile_of(&process), &process.ports()).unwrap();
    let objects = ObjectStore::open(&dir.path().join("objects"), 1 << 26).unwrap();
    let council = Arc::new(
        Council::new(Vec::new(), None)
            .unwrap()
            .with_beats(services)
            .with_object_store(Arc::new(Mutex::new(objects)))
            .with_loads(process.ports()),
    );
    Some(Rig {
        _process: process,
        council,
        _dir: dir,
    })
}

/// The profile of the rig, for the services: one engine named `beatthis`.
fn profile_of(_process: &Process) -> PlacementProfile {
    PlacementProfile::parse(
        r#"
[[engines]]
name = "beatthis"
kind = "mk-beatthis-cpu"
devices = []
share = 1
[engines.config]
weights = "unused"
[[budgets]]
tiers = ["host"]
bytes = 1
[cpu]
threads = 1
"#,
    )
    .unwrap()
}

fn wav(samples: &[f32], rate: u32, channels: u16) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut out = Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut out, spec).unwrap();
    for &s in samples {
        for _ in 0..channels {
            w.write_sample(s).unwrap();
        }
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

/// Put a WAV object, and give its id.
fn put(rig: &Rig, bytes: Vec<u8>) -> String {
    let id = format!("blake3:{}", nagaya_store::content_id(&bytes).hex());
    let (status, body) = send(
        &rig.council,
        "PUT",
        &format!("/council/v2/objects/{id}"),
        Body::from(bytes),
        "audio/wav",
    );
    assert!(status.is_success(), "put: {status} {body}");
    id
}

fn beats(rig: &Rig, body: Value) -> (StatusCode, Value) {
    send(
        &rig.council,
        "POST",
        "/nagaya/v1/analysis/beats",
        Body::from(body.to_string()),
        "application/json",
    )
}

#[test]
fn the_route_gives_upstreams_beats_for_the_oracle_clip() {
    let Some(g) = golden::golden() else { return };
    let Some(rig) = rig() else { return };
    let (wave, _) = g.f32("click97_34.wave");
    let id = put(&rig, wav(&wave, 22_050, 1));
    let (status, body) = beats(&rig, json!({"model": "beatthis", "audio": id, "logits": true}));
    assert_eq!(status, StatusCode::OK, "{body}");
    let got: Vec<f64> = body["beats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let downs: Vec<f64> = body["downbeats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let (want, _) = g.f64("small0.click97_34.beats_s");
    let (want_d, _) = g.f64("small0.click97_34.downbeats_s");
    assert_eq!(got, want);
    assert_eq!(downs, want_d);
    let frames = body["frames"].as_u64().unwrap() as usize;
    assert_eq!(frames, 501);
    assert_eq!(body["fps"], 50);
    assert_eq!(body["logits"]["beat"].as_array().unwrap().len(), frames);
    assert_eq!(body["usage"]["chunks"], 1);
}

#[test]
fn audio_at_another_rate_is_a_400_that_names_the_rate() {
    let Some(rig) = rig() else { return };
    let id = put(&rig, wav(&vec![0.0; 44_100], 44_100, 2));
    let (status, body) = beats(&rig, json!({"model": "beatthis", "audio": id}));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "audio");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("44100 Hz") && message.contains("2 channels"),
        "{message}"
    );
}

#[test]
fn an_unknown_model_and_a_missing_object_are_404s() {
    let Some(rig) = rig() else { return };
    let (status, _) = beats(&rig, json!({"model": "nobody", "audio": "blake3:00"}));
    assert_eq!(status, StatusCode::NOT_FOUND);
    let missing = format!("blake3:{}", "ab".repeat(32));
    let (status, body) = beats(&rig, json!({"model": "beatthis", "audio": missing}));
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

#[test]
fn an_unknown_member_is_a_400() {
    let Some(rig) = rig() else { return };
    let (status, body) = beats(&rig, json!({"model": "beatthis", "audio": "blake3:00", "tempo": true}));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[test]
fn identity_lists_the_prefix_with_only_a_beats_service() {
    let Some(rig) = rig() else { return };
    let (status, body) = send(
        &rig.council,
        "GET",
        "/council/v2/identity",
        Body::empty(),
        "application/json",
    );
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["extensions"], json!(["/nagaya/v1"]), "{body}");
    assert!(
        body["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["model"] != "beatthis"),
        "the beats model is not a council model"
    );
}

#[test]
fn the_cpu_group_runs_on_a_thread_named_nagaya_cpu() {
    let Some(_rig) = rig() else { return };
    let names: Vec<String> = std::fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|t| std::fs::read_to_string(t.unwrap().path().join("comm")).ok())
        .map(|n| n.trim().to_owned())
        .collect();
    assert!(names.iter().any(|n| n == "nagaya-cpu"), "{names:?}");
    // The open of the engine starts the 4 workers of `[cpu] threads` (spec 3.2c).
    for i in 0..4 {
        assert!(
            names.iter().any(|n| *n == format!("nagaya-cpu-{i}")),
            "no worker {i}: {names:?}"
        );
    }
    assert!(
        !names.iter().any(|n| n.starts_with("nagaya-group-")),
        "the only group is the CPU group: {names:?}"
    );
}

/// The check runs before any thread starts, so it needs no weights.
#[test]
fn more_threads_than_the_pool_set_refuses_the_start() {
    let mut config = toml::Table::new();
    config.insert("weights".into(), toml::Value::String("/nonexistent.safetensors".into()));
    let profile = PlacementProfile {
        engines: vec![EnginePlacement {
            name: "beatthis".to_owned(),
            kind: nagaya::beatthis::KIND.to_owned(),
            devices: vec![],
            config,
            family: toml::Table::new(),
            lab: toml::Table::new(),
            share: 1,
            residency: Residency::Resident,
            pinned: true,
        }],
        budgets: vec![BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Host)],
            bytes: 1 << 30,
        }],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
        cpu: Some(CpuSpec { threads: 3 }),
    };
    let dir = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(nagaya::beatthis::KIND, Box::new(nagaya::beatthis::BeatThisFactory));
    let sets = nagaya::cpu_sets::CpuSets {
        housekeeping: vec![0, 1],
        pool: vec![2, 3],
        devices: vec![],
    };
    let error = match Start::new(
        profile,
        registry,
        Inventory {
            devices: vec![],
            host_bytes: 1 << 30,
        },
    )
    .store(dir.path())
    .cpu_sets(sets)
    .start()
    {
        Ok(_) => panic!("3 threads on a pool of 2 started"),
        Err(error) => error,
    };
    let text = error.to_string();
    assert!(
        text.contains("`[cpu] threads` is 3") && text.contains("2 CPUs"),
        "{text}"
    );
}

fn cpu_profile(kind: &str, weights: &str, threads: u32) -> PlacementProfile {
    let mut config = toml::Table::new();
    if !weights.is_empty() {
        config.insert("weights".into(), toml::Value::String(weights.into()));
    }
    PlacementProfile {
        engines: vec![EnginePlacement {
            name: "beatthis".to_owned(),
            kind: kind.to_owned(),
            devices: vec![],
            config,
            family: toml::Table::new(),
            lab: toml::Table::new(),
            share: 1,
            residency: Residency::Resident,
            pinned: true,
        }],
        budgets: vec![BudgetSpec {
            tiers: vec![TierSpec::from(Tier::Host)],
            bytes: 1 << 30,
        }],
        store: Some(StoreSpec {
            nvme: 1 << 20,
            host: None,
            interval_copies: 0,
            objects: None,
        }),
        cpu: Some(CpuSpec { threads }),
    }
}

/// As many threads as the pool set passes the check, and the start then fails on the missing weights instead.
#[test]
fn threads_equal_to_the_pool_set_pass_the_check() {
    let dir = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(nagaya::beatthis::KIND, Box::new(nagaya::beatthis::BeatThisFactory));
    let sets = nagaya::cpu_sets::CpuSets {
        housekeeping: vec![0, 1],
        pool: vec![2, 3],
        devices: vec![],
    };
    let profile = cpu_profile(nagaya::beatthis::KIND, "/nonexistent.safetensors", 2);
    let error = match Start::new(
        profile,
        registry,
        Inventory {
            devices: vec![],
            host_bytes: 1 << 30,
        },
    )
    .store(dir.path())
    .cpu_sets(sets)
    .start()
    {
        Ok(_) => panic!("a start with no weights file started"),
        Err(error) => error,
    };
    let text = error.to_string();
    assert!(!text.contains("[cpu] threads"), "{text}");
    assert!(text.contains("nonexistent"), "the start fails on the weights: {text}");
}

/// A fake beats engine whose outputs have `extra` values more than the frames ask for.
struct Fake {
    extra: usize,
}

impl nagaya_abi::Beats for Fake {
    fn beats_max_frames(&self) -> u32 {
        1500
    }

    fn beats(&mut self, _rows: &[f32], frames: u32) -> Result<Vec<f32>, nagaya_abi::PassError> {
        Ok(vec![-5.0; frames as usize * 2 + self.extra])
    }
}

impl nagaya_abi::Engine for Fake {
    fn identity(&self) -> nagaya_abi::Identity {
        nagaya_abi::Identity {
            model: "fake-beats".into(),
            family: "beatthis".into(),
            checkpoint_digest: "fake".into(),
            tokenizer_hash: String::new(),
            template: String::new(),
            numbers: Default::default(),
            plan_table: String::new(),
            bit_settings: Vec::new(),
            canary_hash: String::new(),
            build: Default::default(),
        }
    }

    fn limits(&self) -> nagaya_abi::Limits {
        let config = nagaya_family_beatthis::plan::Config {
            stem_dim: 32,
            transformer_dim: 128,
            n_layers: 1,
            head_dim: 32,
            ff_mult: 4,
        };
        nagaya_mk_beatthis_cpu::engine::limits(&config, 1)
    }

    fn pass(
        &mut self,
        _: &[nagaya_abi::Segment],
        _: &[nagaya_abi::Tap],
    ) -> Result<nagaya_abi::PassOut, nagaya_abi::PassError> {
        unreachable!("a beats engine takes no pass")
    }

    fn last_error(&self) -> Option<String> {
        None
    }

    fn beats(&mut self) -> Option<&mut dyn nagaya_abi::Beats> {
        Some(self)
    }
}

struct FakeFactory(usize);

impl nagaya::admission::Factory for FakeFactory {
    fn plan(
        &self,
        _: &toml::Table,
        _: &[nagaya::admission::GroupDevice],
    ) -> Result<nagaya_abi::Limits, nagaya::admission::PlanError> {
        Ok(nagaya_abi::Engine::limits(&Fake { extra: 0 }))
    }

    fn open(
        &self,
        _: &toml::Table,
        _: &[nagaya::admission::GroupDevice],
    ) -> Result<Box<dyn nagaya_abi::Engine>, nagaya::admission::OpenError> {
        unreachable!("a CPU engine opens with open_cpu")
    }

    fn plan_cpu(&self, _: &toml::Table, _threads: u32) -> Result<nagaya_abi::Limits, nagaya::admission::PlanError> {
        Ok(nagaya_abi::Engine::limits(&Fake { extra: 0 }))
    }

    fn open_cpu(
        &self,
        _: &toml::Table,
        _threads: u32,
    ) -> Result<Box<dyn nagaya_abi::Engine>, nagaya::admission::OpenError> {
        Ok(Box::new(Fake { extra: self.0 }))
    }
}

/// A rig with the fake engine: it needs no weights, so it runs on every machine.
fn fake_rig(extra: usize) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(nagaya::beatthis::KIND, Box::new(FakeFactory(extra)));
    let profile = cpu_profile(nagaya::beatthis::KIND, "", 1);
    let mut process = Start::new(
        profile,
        registry,
        Inventory {
            devices: vec![],
            host_bytes: 8 << 30,
        },
    )
    .store(dir.path())
    .heartbeat(Duration::from_millis(50))
    .timeout(Duration::from_secs(30))
    .start()
    .unwrap();
    process.wait_opens(Duration::from_secs(30)).unwrap();
    let services = nagaya::beats::services(&profile_of(&process), &process.ports()).unwrap();
    let objects = ObjectStore::open(&dir.path().join("objects"), 1 << 26).unwrap();
    let council = Arc::new(
        Council::new(Vec::new(), None)
            .unwrap()
            .with_beats(services)
            .with_object_store(Arc::new(Mutex::new(objects)))
            .with_loads(process.ports()),
    );
    Rig {
        _process: process,
        council,
        _dir: dir,
    }
}

#[test]
fn the_route_runs_on_a_fake_engine_with_no_weights() {
    let rig = fake_rig(0);
    let id = put(&rig, wav(&vec![0.1; 22_050 * 40], 22_050, 1));
    let (status, body) = beats(&rig, json!({"model": "beatthis", "audio": id}));
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["beats"], json!([]), "logits of -5 have no peak");
    assert_eq!(body["usage"]["chunks"], 2, "40 s is two chunks");
    assert!(body.get("logits").is_none(), "logits only on request");
}

#[test]
fn outputs_of_the_wrong_size_are_a_500() {
    let rig = fake_rig(1);
    let id = put(&rig, wav(&vec![0.1; 22_050], 22_050, 1));
    let (status, body) = beats(&rig, json!({"model": "beatthis", "audio": id}));
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(
        body["error"]["message"].as_str().unwrap().contains("finite values"),
        "{body}"
    );
}

/// The plan of an engine runs on a control thread and its open on the group thread, and the process pins the two to
/// other CPU sets (spec 3.2a). The memory that the engine declares must not depend on the CPUs of the calling thread.
/// On 2026-11-26 a declaration that counted `available_parallelism` gave other bytes at the plan and after the open,
/// and the service did not start; the tests had run with no CPU sets, so every thread saw every CPU.
#[test]
fn the_engine_opens_when_the_process_pins_its_threads() {
    let Some(_) = golden::weights("small0") else { return };
    let dir_of_models = std::path::PathBuf::from(std::env::var_os(golden::MODEL_VARIABLE).unwrap());
    let weights = dir_of_models.join("small0.safetensors").display().to_string();
    let online: Vec<u32> = (0..std::thread::available_parallelism().unwrap().get() as u32).collect();
    if online.len() < 4 {
        eprintln!("skipped: the machine has fewer than 4 CPUs to split");
        return;
    }
    let sets = nagaya::cpu_sets::CpuSets {
        housekeeping: online[..2].to_vec(),
        pool: online[2..].to_vec(),
        devices: vec![],
    };
    let dir = tempfile::tempdir().unwrap();
    let mut registry = Registry::new();
    registry.register(nagaya::beatthis::KIND, Box::new(nagaya::beatthis::BeatThisFactory));
    let profile = cpu_profile(nagaya::beatthis::KIND, &weights, 2);
    let mut process = Start::new(
        profile,
        registry,
        Inventory {
            devices: vec![],
            host_bytes: 8 << 30,
        },
    )
    .store(dir.path())
    .cpu_sets(sets)
    .heartbeat(Duration::from_millis(50))
    .timeout(Duration::from_secs(30))
    .start()
    .unwrap();
    process
        .wait_opens(Duration::from_secs(60))
        .expect("the engine opens with pinned threads");
}
