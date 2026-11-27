//! The media parts of a context through the council service (spec 11.11, "Media through the council"): the object
//! checks, the passes of a build with media spans, and the recipe of a media part. The family is the LFM3 family of
//! Omni, and the engine is the fake engine with the slots of the toy tests. It records each segment.
//!
//! A test needs `LFM3_OMNI_MODEL_DIR`, and `LFM3_MODEL_DIR` for the text model. It skips when either is not set, and it
//! fails then with `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").

mod council_support;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use council_support::{ENGINE, Fixture, Seen, TermFactory};
use gogi_proto::context::Context;
use gogi_proto::generation::GenerationRequest;
use nagaya::council::error::ServiceError;
use nagaya::council::{Built, Schedule, Service, ServiceConfig};
use nagaya::http::{Council, Served, router};
use nagaya::lfm3::StoreObjects;
use nagaya_abi::SegmentInputs;
use nagaya_council::family::Family;
use nagaya_family_lfm3::Lfm3Family;
use nagaya_mk_lfm3_bmg::media::{LatentRule, MediaRules, Span, SpanKind, check_spans, decode};
use nagaya_store::objects::ObjectStore;
use serde_json::{Value, json};
use tower::ServiceExt;

fn folder(variable: &str) -> Option<PathBuf> {
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the LFM3 model folder")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the LFM3 model folder");
            None
        }
    }
}

/// An RGB PNG of `width` by `height` pixels. `seed` changes the pixels and keeps the size.
fn png(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect();
        writer.write_image_data(&pixels).unwrap();
    }
    out
}

/// The printed content id of `bytes`.
fn id_of(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

/// A service of Omni on the fake engine, whose family and service read one object store. The engine records each
/// segment in `seen`.
struct Omni {
    fixture: Fixture,
    store: Arc<Mutex<ObjectStore>>,
    family: Arc<Lfm3Family>,
    seen: Seen,
    _objects: tempfile::TempDir,
}

impl Omni {
    fn start(factory: TermFactory) -> Option<Self> {
        let dir = folder("LFM3_OMNI_MODEL_DIR")?;
        let objects = tempfile::tempdir().unwrap();
        let store = Arc::new(Mutex::new(
            ObjectStore::open(&objects.path().join("objects"), 1 << 26).unwrap(),
        ));
        let family = Arc::new(
            Lfm3Family::from_dir(&dir)
                .unwrap()
                .with_objects(Arc::new(StoreObjects::new(Arc::clone(&store)))),
        );
        let seen = Seen::default();
        let factory = TermFactory {
            seen: Some(seen.clone()),
            ..factory
        };
        let mut fixture = Fixture::with_family(
            factory,
            Arc::clone(&family) as Arc<dyn Family>,
            1 << 26,
            None,
            ServiceConfig::new("lfm3-omni", ENGINE),
        );
        fixture.service = replace_service(&mut fixture).with_objects(Arc::clone(&store));
        Some(Self {
            fixture,
            store,
            family,
            seen,
            _objects: objects,
        })
    }

    /// Put an object and give its id.
    fn put(&self, bytes: &[u8], mime: &str) -> String {
        let id = id_of(bytes);
        let content = nagaya_store::ContentId::from_hex(id.trim_start_matches("blake3:")).unwrap();
        let mut store = self.store.lock().unwrap();
        let staged = store.staging_path();
        std::fs::create_dir_all(staged.parent().unwrap()).unwrap();
        std::fs::write(&staged, bytes).unwrap();
        store.commit(&content, mime, &staged, bytes.len() as u64, 0).unwrap();
        id
    }

    fn build(&self, context: &Context) -> Result<Built, ServiceError> {
        self.fixture.service.build(context, false, Schedule::default())
    }

    /// The placeholders of an image, a clip and a video, and the rules of the engine for them.
    fn rules(&self) -> MediaRules {
        let token = |text: &str| self.family.tokenizer().token_id(text).unwrap();
        MediaRules {
            image_token: Some(token("<|image|>")),
            audio_token: Some(token("<|audio|>")),
            video_token: Some(token("<|video|>")),
            audio_out_token: Some(token("<|audio_out|>")),
            audio_encoder: true,
            latent: Some(
                LatentRule::of(
                    &std::fs::read_to_string(folder("LFM3_OMNI_MODEL_DIR").unwrap().join("config.json")).unwrap(),
                    2048,
                )
                .unwrap(),
            ),
        }
    }
}

/// The service of a fixture, moved out so that a test can give it the object store. The fixture gets a service on the
/// same engine in its place, which the test drops.
fn replace_service(fixture: &mut Fixture) -> Service {
    let other = fixture.attach_again();
    std::mem::replace(&mut fixture.service, other)
}

/// A context of a system message, then a user message with text and an image, marked as a boundary.
fn with_image(id: &str) -> Context {
    serde_json::from_value(json!({"messages": [
        {"role": "system", "content": "Respond with text."},
        {"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
                                     {"type": "object", "id": id, "mime": "image/png"}], "snap": true}]}))
    .unwrap()
}

/// A build sends `TokensAndRows` to each pass that holds a position of the image. The image of 256 by 256 pixels
/// takes 64 positions, and a pass of the fake takes 64 tokens, so the span crosses a block boundary. Each pass that
/// holds a part of it gets the span whole, and the checks of the engine accept each pass (engine-abi "The LFM3 vision
/// binding").
#[test]
fn a_build_sends_the_record_and_the_rows_of_an_image_to_each_pass_that_holds_it() {
    let Some(omni) = Omni::start(TermFactory::toy()) else {
        return;
    };
    let image = png(256, 256, 0);
    let id = omni.put(&image, "image/png");
    let context = with_image(&id);
    let rendering = omni.family.render(&context, false).unwrap();
    let [span] = rendering.media.as_slice() else {
        panic!("one media span: {:?}", rendering.media);
    };
    assert_eq!(
        (span.count, span.mime.as_str(), span.id.as_str()),
        (64, "image/png", id.as_str())
    );
    let want = omni.family.media_rows(&rendering).unwrap();
    let built = omni.build(&context).unwrap();
    assert_eq!(built.usage.fed as usize, rendering.end);
    let media = built.usage.media.expect("a build that reads an object counts it");
    assert_eq!((media.read.objects, media.read.bytes), (1, image.len() as u64));
    let segments = omni.seen.take();
    let mut with_rows = 0;
    let mut covered = 0;
    for segment in &segments {
        let from = segment.positions[0] as usize;
        let to = from + segment.positions.len();
        let holds = from < span.first + span.count && span.first < to;
        match &segment.inputs {
            SegmentInputs::TokensAndRows { tokens, rows } => {
                assert!(holds, "the pass {from}..{to} holds no position of the span {span:?}");
                with_rows += 1;
                assert_eq!(segment.meta, want[0].meta);
                assert_eq!(&rows[..], &want[0].rows[..], "the rows of the span whole");
                let spans = decode(&segment.meta).unwrap();
                assert_eq!(
                    spans,
                    [Span {
                        kind: SpanKind::Image { grid: [1, 16, 16] },
                        first: span.first as u64,
                        count: 64,
                    }]
                );
                let uses = check_spans(&omni.rules(), &spans, tokens, from as u64, rows.len()).unwrap();
                covered += uses[0].take.len();
            }
            SegmentInputs::Tokens(_) => {
                assert!(
                    !holds,
                    "the pass {from}..{to} holds a part of the span {span:?} and has no rows"
                );
                assert!(segment.meta.is_empty());
            }
            SegmentInputs::Rows(_) => panic!("a build sends no rows alone"),
        }
    }
    assert!(
        with_rows >= 2,
        "the span crosses a pass boundary: {with_rows} passes with rows"
    );
    assert_eq!(covered, 64, "the passes fill each position of the span once");
}

/// The same object gives the same recipe, so a second build feeds nothing. Another image of the same size is another
/// context (spec 11.11, tag 9).
/// The family's record of a latent row is the record that the engine takes, at a position of the engine's placeholder
/// (engine-abi "Latent rows of LFM3").
#[test]
fn the_latent_record_of_the_family_is_the_span_of_the_engine() {
    let Some(omni) = Omni::start(TermFactory::toy()) else {
        return;
    };
    let (token, meta) = omni.family.latent_record(30, 1, 2048).unwrap();
    let rules = omni.rules();
    assert_eq!(Some(token), rules.latent.map(|rule| rule.token));
    let spans = decode(&meta).unwrap();
    let want = Span {
        kind: SpanKind::Latent { width: 2048 },
        first: 30,
        count: 1,
    };
    assert_eq!(spans, [want]);
    check_spans(&rules, &spans, &[token], 30, want.bytes()).unwrap();
}

#[test]
fn the_same_object_reuses_the_snapshot_and_another_object_of_one_size_does_not() {
    let Some(omni) = Omni::start(TermFactory::toy()) else {
        return;
    };
    let first = omni.put(&png(256, 256, 0), "image/png");
    let other = omni.put(&png(256, 256, 1), "image/png");
    let built = omni.build(&with_image(&first)).unwrap();
    let again = omni.build(&with_image(&first)).unwrap();
    assert_eq!(again.usage.fed, 0, "{:?}", again.usage);
    assert_eq!(again.head, built.head);
    assert!(again.usage.media.is_some(), "a kept object is a read object too");
    let changed = omni.build(&with_image(&other)).unwrap();
    assert_ne!(changed.head, built.head);
    assert_eq!(changed.usage.kept, 0, "{:?}", changed.usage);
}

/// An interval copy inside the span of an image names the image, so a context with another image of the same size
/// never restores it (spec 6.13, 11.11).
#[test]
fn an_interval_copy_inside_a_span_names_its_object() {
    let factory = TermFactory {
        grain: Some(16),
        interval_copies: 4,
        ..TermFactory::toy()
    };
    let Some(omni) = Omni::start(factory) else {
        return;
    };
    let first = omni.put(&png(256, 256, 0), "image/png");
    let other = omni.put(&png(256, 256, 1), "image/png");
    let context = with_image(&first);
    let rendering = omni.family.render(&context, false).unwrap();
    let span = rendering.media[0].clone();
    assert!(
        span.first < 64 && 64 < span.first + span.count && 64 < rendering.end,
        "the copy after the first pass is inside the span: {span:?}"
    );
    omni.build(&context).unwrap();
    let changed = omni.build(&with_image(&other)).unwrap();
    assert_eq!(changed.usage.kept, 0, "{:?}", changed.usage);
}

/// A generation builds its context with the rows of each span, and counts the object it read.
#[test]
fn a_generation_sends_the_rows_of_its_image() {
    let factory = TermFactory {
        vocab: Some(125_033),
        ..TermFactory::toy()
    };
    let Some(omni) = Omni::start(factory) else {
        return;
    };
    let id = omni.put(&png(256, 256, 0), "image/png");
    let request: GenerationRequest = serde_json::from_value(json!({
        "model": "lfm3-omni", "context": serde_json::to_value(with_image(&id)).unwrap(),
        "max_tokens": 1, "sample": {"temperature": 0}}))
    .unwrap();
    let done = omni.fixture.service.generate(&request, None).unwrap();
    assert_eq!(done.usage.media.map(|m| m.read.objects), Some(1));
    let segments = omni.seen.take();
    assert!(
        segments
            .iter()
            .any(|segment| matches!(segment.inputs, SegmentInputs::TokensAndRows { .. })),
        "a pass of the generation reads the image"
    );
}

/// The head after a reply chains from a recipe that names the image. A later context that holds the reply as an
/// unmarked message starts from that head, so the search of held cuts names the image too (spec 11.3c, 11.11). The
/// image's message is marked in the later context or not: the search reaches the seal at its end as a boundary of the
/// plan, or as a cut from the root.
#[test]
fn a_context_after_a_reply_starts_from_the_head_of_the_reply() {
    for marked in [true, false] {
        let script = council_support::Script::default();
        let factory = TermFactory {
            vocab: Some(125_033),
            script: script.clone(),
            ..TermFactory::toy()
        };
        let Some(omni) = Omni::start(factory) else {
            return;
        };
        let id = omni.put(&png(256, 256, 0), "image/png");
        let context = with_image(&id);
        let prompt = omni.family.render(&context, true).unwrap().tokens.len();
        let mut reply = omni.family.tokenize("Yes.").unwrap();
        reply.push(omni.family.stop_tokens()[0]);
        script.set(prompt, reply);
        let request: GenerationRequest = serde_json::from_value(json!({
            "model": "lfm3-omni", "context": serde_json::to_value(&context).unwrap(),
            "max_tokens": 8, "sample": {"temperature": 0}}))
        .unwrap();
        let done = omni.fixture.service.generate(&request, None).unwrap();
        assert_eq!(done.content, "Yes.");
        let head = done.head.expect("the generation holds the head after its reply");
        let mut later = serde_json::to_value(&context).unwrap();
        later["messages"][1]["snap"] = json!(marked);
        later["messages"].as_array_mut().unwrap().extend([
            json!({"role": "assistant", "content": "Yes."}),
            json!({"role": "user", "content": "More."}),
        ]);
        let later: Context = serde_json::from_value(later).unwrap();
        let built = omni.build(&later).unwrap();
        assert_eq!(built.from, Some(head), "marked {marked}: {:?}", built.usage);
        assert!(built.usage.kept as usize > prompt, "marked {marked}: {:?}", built.usage);
    }
}

/// A raw read builds its context with the rows of each span, and counts the object it read.
#[test]
fn a_raw_read_sends_the_rows_of_its_image() {
    let factory = TermFactory {
        vocab: Some(125_033),
        ..TermFactory::toy()
    };
    let Some(omni) = Omni::start(factory) else {
        return;
    };
    let id = omni.put(&png(256, 256, 0), "image/png");
    let request: gogi_proto::reads::ReadsRequest = serde_json::from_value(json!({
        "model": "lfm3-omni", "contexts": [serde_json::to_value(with_image(&id)).unwrap()],
        "suffixes": ["\nAnswer: "], "letters": ["A", "B"]}))
    .unwrap();
    let read = omni.fixture.service.read(&request).unwrap();
    assert_eq!(read.usage.media.map(|m| m.read.objects), Some(1));
    assert!(
        omni.seen
            .take()
            .iter()
            .any(|segment| matches!(segment.inputs, SegmentInputs::TokensAndRows { .. })),
        "a pass of the read reads the image"
    );
}

/// The read plan of a family renders a case as text, so a decision with `case_media` is a 400 that names it, after
/// the rules of gogi "Contexts" pass. This build reads no case media yet.
#[test]
fn a_decision_with_case_media_is_a_400_that_names_it() {
    let Some(omni) = Omni::start(TermFactory::toy()) else {
        return;
    };
    let id = omni.put(&png(256, 256, 0), "image/png");
    let decide = |media: Value| {
        let request: gogi_proto::decision::DecisionRequest = serde_json::from_value(json!({
            "model": "lfm3-omni", "spec_id": id_of(b"no spec"), "case": "this picture",
            "case_media": media, "contexts": []}))
        .unwrap();
        omni.fixture.service.decide(&request)
    };
    match decide(json!([{"id": id, "mime": "image/png"}])) {
        Err(ServiceError::Invalid(invalid)) => {
            assert_eq!(invalid.param.as_deref(), Some("case_media[0]"));
            assert!(invalid.message.contains("case_media"), "{}", invalid.message);
        }
        other => panic!("a 400 that names case_media[0]: {other:?}"),
    }
    // An object that the server does not hold is the 404 of rule 4 first.
    let absent = id_of(b"not held");
    assert!(matches!(
        decide(json!([{"id": absent, "mime": "image/png"}])),
        Err(ServiceError::NotHeld { param, .. }) if param == "case_media[0]"
    ));
}

/// Each object part follows the rules of gogi "Contexts" in order: a whole id, a media type that the model reads, the
/// media type of the put, and an object that the server holds. The first that fails names the part.
#[test]
fn an_object_part_follows_the_rules_of_gogi_in_order() {
    let Some(omni) = Omni::start(TermFactory::toy()) else {
        return;
    };
    let id = omni.put(&png(256, 256, 0), "image/png");
    let absent = id_of(b"not held");
    let part = |id: &str, mime: &str| -> Context {
        serde_json::from_value(json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Look."},
                                         {"type": "object", "id": id, "mime": mime}]}]}))
        .unwrap()
    };
    let at = "messages[0].content[1]";
    let invalid = |context: Context, words: &str| match omni.build(&context) {
        Err(ServiceError::Invalid(invalid)) => {
            assert_eq!(invalid.param.as_deref(), Some(at));
            assert!(invalid.message.contains(words), "{}", invalid.message);
        }
        other => panic!("a 400 that names {at}: {other:?}"),
    };
    invalid(part("blake3:abc", "image/png"), "blake3:abc");
    invalid(
        part(&absent, "video/mp4"),
        "reads text and image/png, image/jpeg, audio/wav",
    );
    invalid(part(&id, "image/jpeg"), "it was put as image/png");
    match omni.build(&part(&absent, "image/png")) {
        Err(ServiceError::NotHeld { param, message }) => {
            assert_eq!(param, at);
            assert!(message.contains(&absent), "{message}");
        }
        other => panic!("a 404 that names {at}: {other:?}"),
    }
    assert!(omni.build(&part(&id, "image/png")).is_ok());
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
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 24).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    })
}

/// The identity of `lfm3-omni` lists its media inputs, and `lfm3` lists none (spec 11.11). The route takes an object
/// part for Omni and refuses it for `lfm3`, and an object that the server does not hold is a 404 that names the part.
#[test]
fn the_route_takes_object_parts_by_the_media_of_the_identity() {
    let Some(text_dir) = folder("LFM3_MODEL_DIR") else {
        return;
    };
    let Some(mut omni) = Omni::start(TermFactory::toy()) else {
        return;
    };
    let id = omni.put(&png(256, 256, 0), "image/png");
    let text = Service::attach_with_clock(
        omni.fixture.process.ports(),
        Arc::new(Lfm3Family::from_dir(&text_dir).unwrap()),
        omni.fixture.directory.path(),
        ServiceConfig::new("lfm3", ENGINE),
        omni.fixture.reader(),
    )
    .unwrap()
    .with_objects(Arc::clone(&omni.store));
    let service = replace_service(&mut omni.fixture);
    let served = |service| Served {
        service,
        aliases: Vec::new(),
        device: None,
    };
    let council = Arc::new(
        Council::new(vec![served(service), served(text)], None)
            .unwrap()
            .with_object_store(Arc::clone(&omni.store)),
    );
    let (status, identity) = send(&council, "GET", "/council/v2/identity", None);
    assert_eq!(status, StatusCode::OK, "{identity}");
    let media = |name: &str| {
        identity["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["model"] == name)
            .unwrap()["media"]
            .clone()
    };
    assert_eq!(
        media("lfm3-omni"),
        json!({"inputs": ["image/png", "image/jpeg", "audio/wav"], "outputs": []})
    );
    assert_eq!(media("lfm3"), Value::Null);
    let body = |model: &str, id: &str| {
        let mut context = serde_json::to_value(with_image(id)).unwrap();
        context["messages"][1]["snap"] = json!(false);
        json!({"model": model, "context": context})
    };
    let (status, built) = send(&council, "POST", "/council/v2/contexts", Some(body("lfm3-omni", &id)));
    assert_eq!(status, StatusCode::OK, "{built}");
    assert_eq!(built["usage"]["media"]["read"]["objects"], json!(1), "{built}");
    let (status, refused) = send(&council, "POST", "/council/v2/contexts", Some(body("lfm3", &id)));
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!(refused["error"]["param"], json!("messages[1].content[1]"), "{refused}");
    assert!(
        refused["error"]["message"]
            .as_str()
            .unwrap()
            .contains("lfm3 reads text only"),
        "{refused}"
    );
    let absent = id_of(b"not held");
    let (status, missing) = send(
        &council,
        "POST",
        "/council/v2/contexts",
        Some(body("lfm3-omni", &absent)),
    );
    assert_eq!(status, StatusCode::NOT_FOUND, "{missing}");
    assert_eq!(missing["error"]["param"], json!("messages[1].content[1]"), "{missing}");
    assert!(
        missing["error"]["message"].as_str().unwrap().contains(&absent),
        "{missing}"
    );
}
