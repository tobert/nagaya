//! The route of a served image model of the tokens form, Qwen-Image (spec 11.9c), and its conditioning cache in the
//! store (spec 6.16), with a fake engine and a fake family. These tests need no GPU.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use nagaya::admission::{Factory, GroupDevice, OpenError, PlanError, Registry};
use nagaya::http::{Council, router};
use nagaya::image::{ImageConfig, ImageFamily, ImageForm, ImageService};
use nagaya::inventory::{Device, Inventory};
use nagaya::process::{Process, Start};
use nagaya::profile::{BudgetSpec, EnginePlacement, PlacementProfile, Residency, StoreSpec, TierSpec};
use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, DecodedImage, Engine, Identity, Image, ImageChunk, ImageConditioning, ImageJob, ImagePrompt,
    ImageSampler, ImageStepOut, Lifetime, Limits, MandatoryTerm, PassError, PassErrorKind, PassOut, Segment, SlotId,
    SlotLimit, Tap, Tier,
};
use nagaya_store::objects::ObjectStore;
use serde_json::{Value, json};
use tower::ServiceExt;

const MODEL: &str = "qwen";
/// The cut of the fake family: 2 system ids before the prompt.
const CUT: u32 = 2;
/// The bytes of one conditioning row of the fake engine.
const ROW: usize = 4;

/// What the fake engine saw: each job, as `image_begin` got it.
type Jobs = Arc<Mutex<Vec<ImageJob>>>;

/// An image engine of the tokens form that caches conditionings. Its rows are a function of the ids, and its image
/// paints the first byte of the rows, so a hit and a miss give the same image.
struct Conditioner {
    jobs: Jobs,
    job: Option<(ImageJob, u8)>,
    step: u32,
}

fn rows_of(ids: &[u32], cut: u32) -> Vec<u8> {
    ids[cut as usize..]
        .iter()
        .flat_map(|id| std::iter::repeat_n(*id as u8, ROW))
        .collect()
}

impl Image for Conditioner {
    fn image_max_pixels(&self) -> u64 {
        4 * 2048 * 2048
    }

    fn image_conditioning(&self) -> Option<ImageConditioning> {
        Some(ImageConditioning {
            row_bytes: ROW as u32,
            max_rows: 100,
            encoder_digest: format!("blake3:{}", "5a".repeat(32)),
        })
    }

    fn image_begin(&mut self, _slot: SlotId, job: &ImageJob) -> Result<Option<Vec<u8>>, PassError> {
        self.jobs.lock().unwrap().push(job.clone());
        let ImagePrompt::Tokens { ids, cut } = &job.prompt else {
            return Err(PassError {
                kind: PassErrorKind::Refused,
                message: "the fake reads tokens".to_owned(),
            });
        };
        let (rows, made) = match &job.conditioning {
            Some(rows) => (rows.as_ref().clone(), None),
            None => {
                let rows = rows_of(ids, *cut);
                (rows.clone(), Some(rows))
            }
        };
        self.job = Some((job.clone(), rows[0]));
        self.step = 0;
        Ok(made)
    }

    fn image_step(&mut self, _slot: SlotId) -> Result<ImageStepOut, PassError> {
        self.step += 1;
        let steps = self.job.as_ref().unwrap().0.steps;
        Ok(ImageStepOut {
            step: self.step,
            steps,
            previews: Vec::new(),
        })
    }

    fn image_decode(&mut self, _slot: SlotId) -> Result<Vec<DecodedImage>, PassError> {
        let (job, paint) = self.job.as_ref().unwrap();
        Ok((0..job.count)
            .map(|_| DecodedImage {
                width: job.width,
                height: job.height,
                channels: 4,
                pixels: vec![*paint; (job.width * job.height * 4) as usize],
            })
            .collect())
    }

    fn image_clear(&mut self, _slot: SlotId) {
        self.job = None;
    }
}

impl Engine for Conditioner {
    fn identity(&self) -> Identity {
        let mut identity = FakeAlways::new().identity();
        identity.family = "fake-qwen-image".to_owned();
        identity
    }

    fn limits(&self) -> Limits {
        let mut limits = FakeAlways::new().limits();
        limits.capabilities = vec![Capability::Image];
        limits.parts = Vec::new();
        limits.slots = vec![SlotLimit {
            bytes: 0,
            count: 1,
            tokens: 1024,
        }];
        limits.memory.mandatory = vec![MandatoryTerm {
            name: "weights".to_owned(),
            tier: Tier::Device(0),
            bytes: 100,
            lifetime: Lifetime::Engine,
            kinds: Vec::new(),
        }];
        limits
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

struct Conditioners(Jobs);

impl Factory for Conditioners {
    fn plan(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Limits, PlanError> {
        Ok(Conditioner {
            jobs: Jobs::default(),
            job: None,
            step: 0,
        }
        .limits())
    }

    fn open(&self, _config: &toml::Table, _devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError> {
        Ok(Box::new(Conditioner {
            jobs: Arc::clone(&self.0),
            job: None,
            step: 0,
        }))
    }
}

/// A family of the tokens form: 2 system ids, then one id for each character. `!` is a special token.
struct Characters;

impl ImageFamily for Characters {
    fn form(&self) -> ImageForm {
        ImageForm::Tokens
    }

    fn prompt(&self, _text: &str) -> Result<(Vec<ImageChunk>, u64), String> {
        Err("is not a prompt of chunks".to_owned())
    }

    fn tokens(&self, text: &str) -> Result<(Vec<u32>, u32), String> {
        if text.contains('!') {
            return Err("holds the special token `!`".to_owned());
        }
        let mut ids = vec![900, 901];
        ids.extend(text.chars().map(u32::from));
        Ok((ids, CUT))
    }
}

struct Fixture {
    process: Process,
    jobs: Jobs,
    directory: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let jobs: Jobs = Arc::default();
        let mut registry = Registry::new();
        registry.register("conditioners", Box::new(Conditioners(Arc::clone(&jobs))));
        let profile = PlacementProfile {
            engines: vec![EnginePlacement {
                name: MODEL.to_owned(),
                kind: "conditioners".to_owned(),
                devices: vec![0],
                config: toml::Table::new(),
                family: toml::Table::new(),
                lab: toml::Table::new(),
                share: 1,
                residency: Residency::Resident,
                pinned: false,
            }],
            budgets: vec![
                BudgetSpec {
                    tiers: vec![TierSpec::from(Tier::Device(0))],
                    bytes: 1000,
                },
                BudgetSpec {
                    tiers: vec![TierSpec::from(Tier::Host)],
                    bytes: 1 << 20,
                },
            ],
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
            host_bytes: 1 << 21,
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
            directory,
        }
    }

    fn council(&self) -> Arc<Council> {
        let service = ImageService::new(
            ImageConfig::new(MODEL, MODEL),
            Arc::new(Characters),
            self.process.ports(),
        )
        .unwrap();
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

fn generate(council: &Arc<Council>, body: Value) -> (StatusCode, Value) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri("/council/v2/generations")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

/// Send a request and give its status and its body as text, for a stream.
fn generate_text(council: &Arc<Council>, body: Value) -> (StatusCode, String) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let request = Request::builder()
            .method("POST")
            .uri("/council/v2/generations")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(Arc::clone(council)).oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    })
}

fn request(prompt: &str, options: Value) -> Value {
    json!({"model": MODEL,
           "context": {"thinking": false, "messages": [{"role": "user", "content": prompt}]},
           "max_tokens": 1,
           "output": {"mime": "image/png", "options": options}})
}

#[test]
fn a_request_with_no_options_takes_the_defaults_of_the_tokens_form() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, done) = generate(&council, request("a fox", json!({})));
    assert_eq!(status, StatusCode::OK, "{done}");
    let job = fixture.take_jobs().remove(0);
    assert_eq!(
        (job.width, job.height, job.steps, job.cfg, job.sampler, job.count),
        (1024, 1024, 40, 1.0, ImageSampler::FlowEuler, 1)
    );
    let ImagePrompt::Tokens { ids, cut } = &job.prompt else {
        panic!("the job has the tokens form");
    };
    assert_eq!((ids.len(), *cut), (2 + 5, CUT));
    assert!(job.loras.is_empty() && !job.previews);
}

#[test]
fn a_member_that_the_engine_cannot_run_is_a_400_that_names_it() {
    let fixture = Fixture::new();
    let council = fixture.council();
    for (options, param) in [
        (json!({"negative": "blurry"}), "output.options.negative"),
        (json!({"cfg": 4.5}), "output.options.cfg"),
        (json!({"sampler": "dpm++ 2m karras"}), "output.options.sampler"),
        (json!({"clip_skip": 2}), "output.options.clip_skip"),
        (json!({"spacing": "trailing"}), "output.options.spacing"),
        (json!({"loras": []}), "output.options.loras"),
        (json!({"width": 520}), "output.options.width"),
        (json!({"width": 2048, "height": 2048}), "output.options.width"),
        (json!({"count": 5}), "output.options.count"),
        (json!({"steps": 151}), "output.options.steps"),
    ] {
        let (status, error) = generate(&council, request("a fox", options.clone()));
        assert_eq!(status, StatusCode::BAD_REQUEST, "{options}: {error}");
        assert_eq!(error["error"]["param"], json!(param), "{options}: {error}");
    }
    let (status, error) = generate(&council, request("a fox!", json!({})));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["error"]["param"], json!("messages"), "{error}");
    // A size that is a multiple of 32 and not of 64 is a size of Qwen-Image, not of SDXL.
    let (status, done) = generate(
        &council,
        request("a fox", json!({"width": 544, "height": 800, "negative": ""})),
    );
    assert_eq!(status, StatusCode::OK, "{done}");
    assert!(fixture.take_jobs().len() == 1, "only the good request ran");
}

#[test]
fn the_second_request_of_a_prompt_takes_its_conditioning_from_the_store() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let (status, first) = generate(&council, request("a fox", json!({"width": 512, "height": 512})));
    assert_eq!(status, StatusCode::OK, "{first}");
    let (status, second) = generate(
        &council,
        request("a fox", json!({"width": 512, "height": 512, "count": 2})),
    );
    assert_eq!(status, StatusCode::OK, "{second}");
    let (status, other) = generate(&council, request("a cat", json!({"width": 512, "height": 512})));
    assert_eq!(status, StatusCode::OK, "{other}");
    let jobs = fixture.take_jobs();
    assert_eq!(jobs.len(), 3);
    assert!(jobs[0].conditioning.is_none(), "the first request is a miss");
    let ImagePrompt::Tokens { ids, cut } = &jobs[1].prompt else {
        panic!("the tokens form");
    };
    assert_eq!(
        jobs[1].conditioning.as_deref(),
        Some(&rows_of(ids, *cut)),
        "the second request of the same prompt is a hit with the rows of the first"
    );
    assert!(jobs[2].conditioning.is_none(), "another prompt is a miss");
    let store = fixture.process.ports().store().unwrap().clone();
    let entries = store.lock().unwrap().entries().len();
    assert_eq!(entries, 2, "one entry for each prompt");
    // The image of a hit is the image of the miss: the fake paints the first byte of the rows.
    assert_eq!(
        first["outputs"][0]["bytes"], second["outputs"][0]["bytes"],
        "{first} {second}"
    );
}

/// gogi's contract, "Steps and previews": a served model that makes no previews sends none, and that is not an error.
#[test]
fn a_stream_that_asks_for_previews_gets_its_steps_and_no_preview() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let mut body = request("a fox", json!({"width": 512, "height": 512, "steps": 3}));
    body["stream"] = json!(true);
    body["previews"] = json!(true);
    let (status, text) = generate_text(&council, body);
    assert_eq!(status, StatusCode::OK, "{text}");
    assert_eq!(text.matches("event: progress").count(), 3, "{text}");
    assert!(!text.contains("event: preview"), "{text}");
    assert!(text.contains("event: done"), "{text}");
    let jobs = fixture.jobs.lock().unwrap();
    assert!(!jobs.last().unwrap().previews, "the job asks the engine for no preview");
}

/// Qwen-Image runs text to image only (spec 11.9c): identity lists no inputs, and an image part is a 400 that names
/// the part, also with `strength`.
#[test]
fn qwen_image_reads_no_image() {
    let fixture = Fixture::new();
    let council = fixture.council();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let identity: Value = runtime.block_on(async {
        let request = Request::builder()
            .uri("/council/v2/identity")
            .body(Body::empty())
            .unwrap();
        let response = router(Arc::clone(&council)).oneshot(request).await.unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    });
    let model = identity["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model"] == MODEL)
        .unwrap()
        .clone();
    assert_eq!(model["media"]["inputs"], json!([]), "{model}");
    let id = format!("blake3:{}", "ab".repeat(32));
    let mut body = request("a fox", json!({"strength": 0.5}));
    body["context"]["messages"][0]["content"] = json!([{"type": "text", "text": "a fox"},
                                                       {"type": "object", "id": id, "mime": "image/png"}]);
    let (status, error) = generate(&council, body);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["error"]["param"], json!("messages[0].content[1]"), "{error}");
    assert!(fixture.take_jobs().is_empty());
}
