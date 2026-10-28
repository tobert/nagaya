//! Generations of a served image model in the process (spec 11.9a), with a fake image engine.
//!
//! The fake's previews hold the step and the image index in each byte. Its images hold the low byte of their seed, so
//! a test sees which seed made which image.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::http::{Council, router};
use nagaya::image::{ImageConfig, ImageFamily, ImageService};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, Engine, Identity, Image, ImageChunk, ImageJob, ImageSampler, ImageStepOut, Lifetime, Limits,
    MandatoryTerm, PassError, PassErrorKind, PassOut, RgbImage, Segment, SlotId, SlotLimit, Tap, Tier,
};
use nagaya_store::objects::ObjectStore;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "painter";

type Jobs = Arc<Mutex<Vec<ImageJob>>>;

/// The `image_clear` calls of the fake.
type Clears = Arc<std::sync::atomic::AtomicUsize>;

/// An image engine with one slot that paints each image in the low byte of its seed.
struct Painter {
    jobs: Jobs,
    clears: Clears,
    job: Option<ImageJob>,
    step: u32,
}

impl Image for Painter {
    fn image_max_pixels(&self) -> u64 {
        4 * 1_572_864
    }

    fn image_begin(&mut self, slot: SlotId, job: &ImageJob) -> Result<(), PassError> {
        assert_eq!(slot, 0, "the fake has one slot");
        self.jobs.lock().unwrap().push(job.clone());
        self.job = Some(job.clone());
        self.step = 0;
        Ok(())
    }

    fn image_step(&mut self, _slot: SlotId) -> Result<ImageStepOut, PassError> {
        let job = self.job.as_ref().expect("a step after a begin");
        self.step += 1;
        let previews = if job.previews {
            (0..job.count)
                .map(|i| {
                    let (width, height) = (job.width / 8, job.height / 8);
                    RgbImage {
                        width,
                        height,
                        rgb: vec![(16 * self.step + i) as u8; (width * height * 3) as usize],
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(ImageStepOut {
            step: self.step,
            steps: job.steps,
            previews,
        })
    }

    fn image_decode(&mut self, _slot: SlotId) -> Result<Vec<RgbImage>, PassError> {
        let job = self.job.as_ref().expect("a decode after a begin");
        assert_eq!(self.step, job.steps, "the decode comes after the last step");
        Ok((0..job.count)
            .map(|i| RgbImage {
                width: job.width,
                height: job.height,
                rgb: vec![job.seed.wrapping_add(u64::from(i)) as u8; (job.width * job.height * 3) as usize],
            })
            .collect())
    }

    fn image_clear(&mut self, _slot: SlotId) {
        self.clears.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.job = None;
    }
}

fn limits() -> Limits {
    let mut limits = FakeAlways::new().limits();
    limits.capabilities = vec![Capability::Image, Capability::Close];
    // An image engine takes no pass, so it declares no segment, as the SDXL engine does.
    limits.max_segments = 0;
    limits.parts = Vec::new();
    limits.slots = vec![SlotLimit {
        bytes: 0,
        count: 1,
        tokens: 0,
    }];
    limits.memory.mandatory = vec![MandatoryTerm {
        name: "weights".to_owned(),
        tier: Tier::Device(0),
        bytes: 100,
        lifetime: Lifetime::Engine,
        kinds: Vec::new(),
    }];
    limits.memory.elastic = Vec::new();
    limits
}

impl Engine for Painter {
    fn identity(&self) -> Identity {
        let mut identity = FakeAlways::new().identity();
        identity.model = "fake-painter".to_owned();
        identity.family = "fake-image".to_owned();
        identity
    }

    fn limits(&self) -> Limits {
        limits()
    }

    fn pass(&mut self, _segments: &[Segment], _taps: &[Tap]) -> Result<PassOut, PassError> {
        Err(PassError {
            kind: PassErrorKind::Refused,
            message: "an image engine takes no pass".to_owned(),
        })
    }

    fn last_error(&self) -> Option<String> {
        None
    }

    fn image(&mut self) -> Option<&mut dyn Image> {
        Some(self)
    }
}

struct Painters(Jobs, Clears);

impl Factory for Painters {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Painter {
            jobs: Arc::clone(&self.0),
            clears: Arc::clone(&self.1),
            job: None,
            step: 0,
        }))
    }
}

/// A family with one id for each character. A prompt of more than 75 characters is too long.
struct Characters;

impl ImageFamily for Characters {
    fn prompt(&self, text: &str) -> Result<(Vec<ImageChunk>, u64), String> {
        if text.chars().count() > 75 {
            return Err("is more than 75 characters; give a shorter prompt".to_owned());
        }
        let ids: Vec<u32> = text.chars().map(u32::from).collect();
        let chunk = ImageChunk {
            ids1: ids.clone(),
            ids2: ids,
            weights: vec![1.0; text.chars().count()],
        };
        Ok((vec![chunk], text.chars().count() as u64))
    }
}

struct Fixture {
    process: Process,
    jobs: Jobs,
    clears: Clears,
    directory: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self::with(Residency::Resident)
    }

    fn with(residency: Residency) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let jobs: Jobs = Arc::default();
        let clears: Clears = Arc::default();
        let mut registry = Registry::new();
        registry.register("painters", Box::new(Painters(Arc::clone(&jobs), Arc::clone(&clears))));
        let profile = PlacementProfile {
            engines: vec![EnginePlacement {
                name: MODEL.to_owned(),
                kind: "painters".to_owned(),
                devices: vec![0],
                config: toml::Table::new(),
                family: toml::Table::new(),
                share: 1,
                pin_identity: false,
                residency,
                idle_unload_s: None,
            }],
            budgets: vec![BudgetSpec {
                tiers: vec![TierSpec::from(Tier::Device(0))],
                bytes: 1000,
            }],
            store: Some(StoreSpec {
                nvme: 1 << 20,
                host: None,
                interval_copies: 0,
                objects: None,
            }),
        };
        let inventory = Inventory {
            devices: vec![Device {
                id: 0,
                name: "device-0".to_owned(),
                pci_address: "0000:00:00.0".to_owned(),
                memory_bytes: 4096,
                unified: false,
                ignored: false,
            }],
            host_bytes: 1 << 20,
        };
        let mut process = Start::new(profile, registry, inventory)
            .store(directory.path())
            .heartbeat(Duration::from_millis(10))
            .timeout(Duration::from_secs(5))
            .start()
            .unwrap();
        process.wait_opens(Duration::from_secs(5)).unwrap();
        Self {
            process,
            jobs,
            clears,
            directory,
        }
    }

    /// A council that serves the image model with an object store, and no text model. Its `clip_skip` default is 2,
    /// so a test sees the default of the config.
    fn council(&self) -> Arc<Council> {
        let mut config = ImageConfig::new(MODEL, MODEL);
        config.clip_skip = 2;
        let service = ImageService::new(config, Arc::new(Characters), self.process.ports()).unwrap();
        let objects = ObjectStore::open(&self.directory.path().join("objects"), 1 << 26).unwrap();
        Arc::new(
            Council::new(Vec::new(), None)
                .unwrap()
                .with_images(vec![service])
                .with_objects(objects),
        )
    }

    fn take_jobs(&self) -> Vec<ImageJob> {
        std::mem::take(&mut *self.jobs.lock().unwrap())
    }
}

/// Send a request through the router. Give the status, the content type, the location and the whole body.
fn send(
    council: &Arc<Council>,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (StatusCode, String, String, Vec<u8>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut request = Request::builder().method(method).uri(path);
        if body.is_some() {
            request = request.header("content-type", "application/json");
        }
        let body = body.map_or_else(Body::empty, |value| Body::from(value.to_string()));
        let response = router(Arc::clone(council))
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        let header = |name| {
            response
                .headers()
                .get(name)
                .map(|value: &axum::http::HeaderValue| value.to_str().unwrap().to_owned())
                .unwrap_or_default()
        };
        let (kind, location) = (header("content-type"), header("location"));
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 26)
            .await
            .unwrap()
            .to_vec();
        (status, kind, location, bytes)
    })
}

fn json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(bytes)))
}

/// A generation of `prompt` with these output options and this `sample`.
fn request(prompt: &str, options: Value, sample: Value) -> Value {
    json!({"model": MODEL, "context": {"messages": [{"role": "user", "content": prompt}]}, "max_tokens": 1,
           "sample": sample, "output": {"mime": "image/png", "options": options}})
}

fn generate(council: &Arc<Council>, body: Value) -> (StatusCode, Value) {
    let (status, _, _, bytes) = send(council, "POST", "/council/v2/generations", Some(body));
    (status, json(&bytes))
}

/// The status and `error.param` of a refused generation.
fn refused(council: &Arc<Council>, body: Value) -> (StatusCode, String) {
    let (status, v) = generate(council, body);
    (status, v["error"]["param"].as_str().unwrap_or("").to_owned())
}

/// The size, the first pixel byte, and the `parameters` text of a PNG.
fn read_png(bytes: &[u8]) -> (u32, u32, u8, Option<String>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(frame.color_type, png::ColorType::Rgb);
    let info = reader.info();
    let text = info
        .uncompressed_latin1_text
        .iter()
        .find(|chunk| chunk.keyword == "parameters")
        .map(|chunk| chunk.text.clone());
    (info.width, info.height, pixels[0], text)
}

/// The bytes of an object.
fn object(council: &Arc<Council>, id: &str) -> Vec<u8> {
    let (status, kind, _, bytes) = send(council, "GET", &format!("/council/v2/objects/{id}"), None);
    assert_eq!(status, StatusCode::OK, "{id}");
    assert_eq!(kind, "image/png", "{id}");
    bytes
}

#[test]
fn identity_lists_the_image_model_as_a_writer_of_png() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, _, _, bytes) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK);
    let v = json(&bytes);
    let model = v["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model"] == MODEL)
        .expect("the image model")
        .clone();
    assert_eq!(model["media"]["outputs"], json!(["image/png"]), "{model}");
    assert_eq!(model["media"]["text_replies"], json!(false), "{model}");
    assert_eq!(model["capabilities"], json!(["generate"]), "{model}");
    assert_eq!(model["limits"]["context_tokens"], json!(225), "{model}");
    assert_eq!(model["limits"]["snapshot_bytes"], json!(0), "{model}");
    assert_eq!(model["status"], json!("ready"), "{model}");
}

#[test]
fn a_generation_writes_one_png_for_each_image_with_its_seed_in_the_a1111_chunk() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let options = json!({"negative": "blurry", "width": 512, "height": 768, "steps": 3, "cfg": 5.5,
                         "sampler": "euler a", "count": 2});
    let (status, done) = generate(&council, request("a red fox", options, json!({"seed": 9})));
    assert_eq!(status, StatusCode::OK, "{done}");
    let jobs = fixture.take_jobs();
    assert_eq!(jobs.len(), 1);
    let clears = fixture.clears.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(clears, 1, "the job clears its slot after the decode");
    let job = &jobs[0];
    assert_eq!(
        (
            job.width,
            job.height,
            job.steps,
            job.cfg,
            job.sampler,
            job.count,
            job.seed,
            job.clip_skip,
            job.previews
        ),
        (512, 768, 3, 5.5, ImageSampler::EulerAncestral, 2, 9, 2, false)
    );
    assert_eq!(
        job.prompt[0].ids1,
        "a red fox".chars().map(u32::from).collect::<Vec<_>>()
    );
    assert_eq!(
        job.negative[0].ids1,
        "blurry".chars().map(u32::from).collect::<Vec<_>>()
    );
    assert_eq!(done["content"], json!(""), "{done}");
    assert_eq!(done["finish"], json!("end_turn"), "{done}");
    assert_eq!(done["usage"]["completion"], json!(0), "{done}");
    assert_eq!(done["usage"]["prompt"], json!(9), "{done}");
    assert_eq!(done["seed"], json!(9), "{done}");
    let outputs = done["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 2, "{done}");
    let mut written = 0;
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(output["index"], json!(i), "{done}");
        assert_eq!(output["mime"], json!("image/png"), "{done}");
        let bytes = object(&council, output["id"].as_str().unwrap());
        assert_eq!(output["bytes"], json!(bytes.len()), "{done}");
        written += bytes.len();
        let (width, height, first, text) = read_png(&bytes);
        assert_eq!((width, height, first), (512, 768, 9 + i as u8), "image {i}");
        let want = format!(
            "a red fox\nNegative prompt: blurry\nSteps: 3, Sampler: Euler a, CFG scale: 5.5, Seed: {}, Size: 512x768, \
             Model: painter, Clip skip: 2",
            9 + i
        );
        assert_eq!(text.as_deref(), Some(want.as_str()), "image {i}");
    }
    let media = &done["usage"]["media"]["written"];
    assert_eq!(media["objects"], json!(2), "{done}");
    assert_eq!(media["bytes"], json!(written), "{done}");
    assert_eq!(media["pixels"], json!(2 * 512 * 768), "{done}");
}

#[test]
fn a_request_with_no_options_gets_the_defaults_and_a_chosen_seed() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let body = json!({"model": MODEL, "context": {"messages": [{"role": "user", "content": "a fox"}]},
                      "max_tokens": 1, "output": {"mime": "image/png"}});
    let (status, done) = generate(&council, body);
    assert_eq!(status, StatusCode::OK, "{done}");
    let job = fixture.take_jobs().remove(0);
    assert_eq!(
        (
            job.width,
            job.height,
            job.steps,
            job.cfg,
            job.sampler,
            job.count,
            job.clip_skip,
            job.previews
        ),
        (1024, 1024, 30, 7.0, ImageSampler::DpmPp2mKarras, 1, 2, false)
    );
    assert!(job.negative.is_empty(), "an empty negative prompt has no chunk");
    assert_eq!(done["seed"], json!(job.seed), "{done}");
    let bytes = object(&council, done["outputs"][0]["id"].as_str().unwrap());
    let text = read_png(&bytes).3.unwrap();
    assert_eq!(
        text,
        format!(
            "a fox\nSteps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: {}, Size: 1024x1024, Model: painter, \
             Clip skip: 2",
            job.seed
        )
    );
}

#[test]
fn a_generation_names_each_member_out_of_range() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let cases = [
        (json!({"width": 500}), "output.options.width"),
        (json!({"width": 448}), "output.options.width"),
        (json!({"width": 520}), "output.options.width"),
        (json!({"height": 2112}), "output.options.height"),
        (json!({"width": 2048, "height": 1024}), "output.options.width"),
        (json!({"steps": 0}), "output.options.steps"),
        (json!({"steps": 151}), "output.options.steps"),
        (json!({"cfg": 0.5}), "output.options.cfg"),
        (json!({"cfg": 30.5}), "output.options.cfg"),
        (json!({"sampler": "ddim"}), "output.options.sampler"),
        (json!({"count": 0}), "output.options.count"),
        (json!({"count": 5}), "output.options.count"),
        (json!({"clip_skip": 3}), "output.options.clip_skip"),
        (json!({"previews": "yes"}), "output.options.previews"),
        (json!({"negative": 3}), "output.options.negative"),
        (json!({"negative": "n".repeat(76)}), "output.options.negative"),
        (json!({"strength": 0.5}), "output.options.strength"),
    ];
    for (options, param) in cases {
        let (status, got) = refused(&council, request("a fox", options.clone(), json!({})));
        assert_eq!((status, got.as_str()), (StatusCode::BAD_REQUEST, param), "{options}");
    }
    for (sample, param) in [
        (json!({"top_p": 0.5}), "sample.top_p"),
        (json!({"top_k": 4}), "sample.top_k"),
        (json!({"min_p": 0.1}), "sample.min_p"),
    ] {
        let (status, got) = refused(&council, request("a fox", json!({}), sample.clone()));
        assert_eq!((status, got.as_str()), (StatusCode::BAD_REQUEST, param), "{sample}");
    }
    let (status, got) = refused(&council, request(&"a fox ".repeat(16), json!({}), json!({})));
    assert_eq!(
        (status, got.as_str()),
        (StatusCode::BAD_REQUEST, "messages"),
        "a prompt too long"
    );
    let none =
        json!({"model": MODEL, "max_tokens": 1, "context": {"messages": [{"role": "user", "content": "a fox"}]}});
    assert_eq!(refused(&council, none), (StatusCode::BAD_REQUEST, "output".to_owned()));
    assert!(fixture.take_jobs().is_empty(), "no refused request ran");
}

/// The server renders no message before the last `user` message, also a `system` message, and a temperature has no
/// effect (spec 11.9a, Proposed 2026-10-28).
#[test]
fn the_last_user_message_is_the_prompt() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let body = json!({"model": MODEL, "max_tokens": 1, "output": {"mime": "image/png", "options": {"steps": 1}},
                      "sample": {"temperature": 0, "seed": 3},
                      "context": {"messages": [{"role": "system", "content": "be nice"},
                                               {"role": "user", "content": "a cat"},
                                               {"role": "assistant", "content": "ok"},
                                               {"role": "user", "content": "a dog"}]}});
    let (status, done) = generate(&council, body);
    assert_eq!(status, StatusCode::OK, "{done}");
    let job = fixture.take_jobs().remove(0);
    assert_eq!(job.prompt[0].ids1, "a dog".chars().map(u32::from).collect::<Vec<_>>());
}

/// The events of a stream body: their names and data.
fn events(body: &[u8]) -> Vec<(String, Value)> {
    let text = String::from_utf8(body.to_vec()).unwrap();
    text.split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .map(|block| {
            let mut name = String::new();
            let mut data = String::new();
            for line in block.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    name = rest.trim().to_owned();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data.push_str(rest.trim());
                }
            }
            (name, json(data.as_bytes()))
        })
        .collect()
}

#[test]
fn a_stream_sends_a_preview_of_each_image_after_each_step_then_the_outputs_and_done() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let mut body = request(
        "a fox",
        json!({"width": 512, "height": 512, "steps": 2, "count": 2, "previews": true}),
        json!({"seed": 40}),
    );
    body["stream"] = json!(true);
    let (status, kind, _, bytes) = send(&council, "POST", "/council/v2/generations", Some(body));
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
    assert!(kind.starts_with("text/event-stream"), "{kind}");
    let events = events(&bytes);
    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        ["preview", "preview", "preview", "preview", "output", "output", "done"]
    );
    for (k, (_, data)) in events[..4].iter().enumerate() {
        let (step, index) = (k / 2 + 1, k % 2);
        assert_eq!(
            (data["step"].clone(), data["index"].clone()),
            (json!(step), json!(index)),
            "{data}"
        );
        let (width, height, first, text) = read_png(&object(&council, data["id"].as_str().unwrap()));
        assert_eq!((width, height, first), (64, 64, (16 * step + index) as u8), "{data}");
        assert!(text.is_none(), "a preview has no parameters chunk");
    }
    let done = &events[6].1;
    for i in 0..2 {
        let output = &events[4 + i].1;
        assert_eq!(
            (output["index"].clone(), output["mime"].clone()),
            (json!(i), json!("image/png"))
        );
        let pending = output["pending"].as_str().unwrap();
        let (status, _, location, _) = send(&council, "GET", &format!("/council/v2/objects/{pending}"), None);
        assert_eq!(
            status,
            StatusCode::PERMANENT_REDIRECT,
            "a sealed pending id moves to its object"
        );
        assert_eq!(
            location,
            format!("/council/v2/objects/{}", done["outputs"][i]["id"].as_str().unwrap())
        );
    }
}

#[test]
fn an_unloaded_image_model_is_a_503_and_a_build_that_names_it_is_a_400() {
    let fixture = Fixture::with(Residency::OnDemand);
    let council = fixture.council();
    let (code, body) = generate(&council, request("a fox", json!({}), json!({})));
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["type"], json!("model_unavailable"), "{body}");
    assert_eq!(body["error"]["status"], json!("unloaded"), "{body}");
    assert!(fixture.take_jobs().is_empty(), "nothing ran");
    // Any route that names the model gives the same 503 first.
    let build = json!({"model": MODEL, "context": {"messages": [{"role": "user", "content": "hello"}]}});
    let (code, _, _, bytes) = send(&council, "POST", "/council/v2/contexts", Some(build));
    let body = json(&bytes);
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["status"], json!("unloaded"), "{body}");

    fixture.process.load(MODEL).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while fixture.process.ports().engine_state(MODEL) != Some(nagaya::notify::EngineState::Ready) {
        assert!(
            std::time::Instant::now() < deadline,
            "the image engine did not load in 5 s"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let (code, body) = generate(&council, request("a fox", json!({"steps": 1}), json!({})));
    assert_eq!(code, StatusCode::OK, "{body}");
    let build = json!({"model": MODEL, "context": {"messages": [{"role": "user", "content": "hello"}]}});
    let (code, _, _, bytes) = send(&council, "POST", "/council/v2/contexts", Some(build));
    let body = json(&bytes);
    assert_eq!(code, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], json!("model"), "{body}");
}

/// The gogi suite against the image model, served in process over HTTP: a model with no text replies (gogi
/// `media.text_replies`).
#[test]
fn the_gogi_suite_passes_for_the_image_model() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server = runtime.spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(listener, router(council)).await.unwrap();
    });
    let url = format!("http://{address}");
    let report = std::thread::spawn(move || {
        gogi_conformance::run(
            gogi_conformance::Opts {
                url,
                restart: None,
                model: Some(MODEL.to_owned()),
            },
            true,
        )
    })
    .join()
    .unwrap();
    server.abort();
    assert_eq!(report.failures(), 0, "{}", report.summary());
}
