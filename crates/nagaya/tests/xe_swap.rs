//! The swap in place on the two B70 cards (spec 3.3a, milestone 16 step 7): Gemma 4 is resident on both cards, and an
//! on-demand fake engine with `close` shares card 1. A load of the fake unloads Gemma 4 and frees its device memory.
//! An unload of the fake loads Gemma 4 again, and the reply is the same as before the swap (spec 3.3b).
//!
//! It needs both cards and the checkpoint in `GEMMA4_MODEL_DIR`. Stop `nagaya.service` first. Run it alone:
//! `cargo test --release -p nagaya --features xe --test xe_swap -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::cpu_sets::for_machine_with_devices_ignoring;
use nagaya::http::{Council, router};
use nagaya::inventory::Probe;
use nagaya::jobs::{JobBody, JobSpec, Ran};
use nagaya::lifecycle::{self, Options};
use nagaya::probe::FileProbe;
use nagaya::process::Event;
use nagaya::profile::PlacementProfile;
use nagaya::serving::{FamilyRegistry, council};
use nagaya::signal::SignalFlag;
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Capability, Engine, Kind, Limits};
use nagaya_sched::Class;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "gemma-4-26b-a4b";

/// The iGPU of b70host drives the display. The b70host config ignores it (spec 2.1, `ignore_devices`).
const IGNORE: &str = "0000:80:00.0";

fn checkpoint() -> Option<PathBuf> {
    match std::env::var_os("GEMMA4_MODEL_DIR") {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("GEMMA4_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set; name the Gemma 4 checkpoint")
        }
        None => {
            eprintln!("skipped: GEMMA4_MODEL_DIR is not set; set it to the Gemma 4 checkpoint directory");
            None
        }
    }
}

/// The fake engine with `close`. It takes no device memory.
struct Closing;

fn closing_limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities.push(Capability::Close);
    limits.memory.mandatory.clear();
    limits.memory.elastic.clear();
    limits
}

struct Fake(FakeAlways);

impl Engine for Fake {
    fn identity(&self) -> nagaya_abi::Identity {
        self.0.identity()
    }

    fn limits(&self) -> Limits {
        closing_limits()
    }

    fn pass(
        &mut self,
        segments: &[nagaya_abi::Segment],
        taps: &[nagaya_abi::Tap],
    ) -> Result<nagaya_abi::PassOut, nagaya_abi::PassError> {
        self.0.pass(segments, taps)
    }

    fn last_error(&self) -> Option<String> {
        self.0.last_error()
    }

    fn parts(&mut self) -> Option<&mut dyn nagaya_abi::Parts> {
        self.0.parts()
    }

    fn slots(&mut self) -> Option<&mut dyn nagaya_abi::Slots> {
        self.0.slots()
    }

    fn coverage(&mut self) -> Option<&mut dyn nagaya_abi::Coverage> {
        self.0.coverage()
    }
}

impl Factory for Closing {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(closing_limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Fake(FakeAlways::new())))
    }
}

/// The profile of b70host, and the fake on demand on machine device 1.
fn profile(dir: &std::path::Path) -> PlacementProfile {
    let dir = dir.display();
    PlacementProfile::parse(&format!(
        "[[engines]]\nname = \"gemma\"\nkind = \"mk-gemma4-bmg\"\ndevices = [0, 1]\nshare = 1\n\
         [engines.config]\nmodel_dir = \"{dir}\"\nslots = [4096, 16384]\n\
         [engines.family]\nmodel_dir = \"{dir}\"\n\
         [[engines]]\nname = \"voice\"\nkind = \"closing\"\ndevices = [1]\nshare = 1\nresidency = \"on_demand\"\n\
         [[budgets]]\ntiers = [\"device(0)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"device(1)\"]\nbytes = 34359738368\n\
         [[budgets]]\ntiers = [\"host\"]\nbytes = 34359738368\n\
         [store]\nnvme = 17179869184\nhost = 8589934592\n"
    ))
    .expect("the profile parses")
}

fn send(council: &Arc<Council>, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder().method(method).uri(path);
        let body = body.map_or_else(Body::empty, |b| Body::from(b.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

fn generate(council: &Arc<Council>) -> (String, u128) {
    let body = json!({"model": MODEL, "max_tokens": 32, "sample": {"temperature": 0},
        "context": {"messages": [{"role": "user", "content": "Name three colors of the rainbow."}]}});
    let start = Instant::now();
    let (status, generation) = send(council, "POST", "/council/v2/generations", Some(body));
    assert_eq!(status, StatusCode::OK, "{generation}");
    // Gemma 4 can think for all its tokens, so the reply is the reasoning and the content.
    let reply = format!(
        "{}|{}",
        generation["reasoning"].as_str().expect("a reply has reasoning"),
        generation["content"].as_str().expect("a reply has content")
    );
    assert!(reply.len() > 1, "{generation}");
    (reply, start.elapsed().as_millis())
}

/// The device bytes of this process on every `xe` card: the sum of `drm-total-vram0` over its DRM fds.
fn process_vram() -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir("/proc/self/fdinfo").expect("the fdinfo of this process") {
        let entry = entry.expect("an fdinfo entry");
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        if !text
            .lines()
            .any(|line| line.starts_with("drm-driver:") && line.ends_with("xe"))
        {
            continue;
        }
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("drm-total-vram0:") {
                let kib = value.trim().strip_suffix(" KiB").expect("a count of KiB");
                total += kib.trim().parse::<u64>().expect("a count of KiB") * 1024;
            }
        }
    }
    total
}

struct Done(mpsc::Sender<()>);

impl JobBody for Done {
    fn unavailable(&mut self, engine: &str, state: nagaya::notify::EngineState) {
        panic!("engine `{engine}` is {}, and this test keeps it ready", state.as_str());
    }

    fn run(&mut self, _kind: Kind, engines: &mut [Box<dyn Engine>]) -> Ran {
        let _ = engines[1].identity();
        let _ = self.0.send(());
        Ran::Done
    }
}

/// The next swap of the process, with its two lists of engine names.
fn swapped(ready: &lifecycle::Ready) -> (Vec<String>, Vec<String>) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Some(Event::Swapped { unloaded, loaded, .. }) = ready.process.recv(Duration::from_millis(50)) {
            return (unloaded, loaded.into_iter().map(|engine| engine.name).collect());
        }
    }
    panic!("no swap came in 120 s");
}

#[test]
fn gemma4_unloads_for_an_engine_on_demand_and_loads_again_with_the_same_reply() {
    let Some(dir) = checkpoint() else {
        return;
    };
    let probe = FileProbe::new("/");
    let mut inventory = Probe::probe(&probe).expect("the inventory of this machine");
    let ignore = vec![IGNORE.to_owned()];
    inventory.ignore(&ignore).expect("the machine has the iGPU");
    let store = tempfile::tempdir().unwrap();
    let profile = profile(&dir);
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).expect("the CPU sets of this machine");
    let mut registry = Registry::compiled_in();
    registry.register("closing", Box::new(Closing));
    let start = Instant::now();
    let mut ready = lifecycle::start_with(
        profile.clone(),
        registry,
        inventory,
        store.path(),
        None,
        Options::new(Duration::from_secs(600)).cpu_sets(sets),
    )
    .expect("the process opens Gemma 4");
    println!("opened in {} ms", start.elapsed().as_millis());
    let council = Arc::new(
        council(
            &FamilyRegistry::compiled_in(),
            &profile,
            &ready.opened,
            &ready.process.ports(),
            store.path(),
            &[],
            None,
        )
        .expect("the process serves Gemma 4"),
    );
    let (before, ms) = generate(&council);
    println!("generation before the swap in {ms} ms: {before}");
    let loaded = process_vram();

    // A request never starts a load, so the test loads the fake (spec 3.3b).
    let started = Instant::now();
    ready.process.load("voice").expect("a load of `voice`");
    assert_eq!(swapped(&ready), (vec!["gemma".to_owned()], vec!["voice".to_owned()]));
    println!("swap to the fake in {} ms", started.elapsed().as_millis());
    let (done, finished) = mpsc::channel();
    let (group, index) = ready.process.ports().locate("voice").expect("the profile has `voice`");
    ready
        .process
        .submit_job(
            group,
            JobSpec {
                class: Class::Interactive,
                priority: 0,
                deadline: None,
                first: Kind::Read,
                engine: index,
                body: Box::new(Done(done)),
            },
        )
        .unwrap();
    finished
        .recv_timeout(Duration::from_secs(120))
        .expect("the job on the fake ran");
    let unloaded = process_vram();
    println!("device bytes: {loaded} with Gemma 4, {unloaded} after its unload");
    assert!(
        unloaded < 1 << 30,
        "Gemma 4 freed its device memory: {unloaded} bytes stay, and Gemma 4 held {loaded}"
    );

    let started = Instant::now();
    ready.process.unload("voice").expect("an unload of `voice`");
    assert_eq!(swapped(&ready), (vec!["voice".to_owned()], vec!["gemma".to_owned()]));
    println!("swap back to Gemma 4 in {} ms", started.elapsed().as_millis());
    let (after, ms) = generate(&council);
    println!("generation after the swap back in {ms} ms: {after}");
    assert_eq!(after, before, "Gemma 4 gives the same reply after the swap back");

    council.drain();
    let stopped = lifecycle::stop(&mut ready, None, Duration::from_secs(30), &SignalFlag::new());
    println!("stop: {}", stopped.log_line());
}
