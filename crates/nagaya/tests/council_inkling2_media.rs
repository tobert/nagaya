//! The image parts of an Inkling 2 context through the council service (spec 11.18, "Images"; engine-abi "The images
//! of Inkling 2"): a chat with an image part becomes the placeholders, the records and the rows that the engine takes.
//! The family is the Inkling 2 family with the object reader of the process, and the engine is the fake engine with
//! the slots of the toy tests. It records each segment, and the checks of the Inkling 2 engine read each one.
//!
//! A test needs `INKLING2_MODEL_DIR`. It skips when the variable is not set, and it fails then with
//! `NAGAYA_REQUIRE_MODELS=1` (AGENTS "Tests that need a model file").

mod council_support;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use council_support::{ENGINE, Fixture, Seen, TermFactory};
use gogi_proto::context::Context;
use nagaya::council::{Schedule, ServiceConfig};
use nagaya::lfm3::StoreObjects;
use nagaya_abi::SegmentInputs;
use nagaya_council::family::Family;
use nagaya_mk_inkling2_bmg::media::{IMAGE_TOKEN, Span, check_spans, decode};
use nagaya_store::objects::ObjectStore;
use nagaya_text_inkling2::Inkling2Family;
use nagaya_text_inkling2::media::Grid;
use serde_json::json;

fn folder() -> Option<PathBuf> {
    let variable = "INKLING2_MODEL_DIR";
    match std::env::var_os(variable) {
        Some(dir) => Some(PathBuf::from(dir)),
        None if std::env::var("NAGAYA_REQUIRE_MODELS").is_ok_and(|v| v == "1") => {
            panic!("{variable} is not set and NAGAYA_REQUIRE_MODELS is set; name the Inkling 2 model folder")
        }
        None => {
            eprintln!("skipped: {variable} is not set; set it to the Inkling 2 model folder");
            None
        }
    }
}

/// An RGB PNG of `width` by `height` pixels.
fn png(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i as u8).wrapping_mul(29)).collect();
        writer.write_image_data(&pixels).unwrap();
    }
    out
}

/// A chat of Inkling 2 on the fake engine, whose family and service read one object store.
struct Chat {
    fixture: Fixture,
    store: Arc<Mutex<ObjectStore>>,
    family: Arc<Inkling2Family>,
    seen: Seen,
    _objects: tempfile::TempDir,
}

impl Chat {
    fn start() -> Option<Self> {
        let dir = folder()?;
        let objects = tempfile::tempdir().unwrap();
        let store = Arc::new(Mutex::new(
            ObjectStore::open(&objects.path().join("objects"), 1 << 26).unwrap(),
        ));
        let family = Arc::new(
            Inkling2Family::from_dir(&dir)
                .unwrap()
                .with_objects(Arc::new(StoreObjects::new(Arc::clone(&store)))),
        );
        let seen = Seen::default();
        let factory = TermFactory {
            seen: Some(seen.clone()),
            ..TermFactory::toy()
        };
        let mut fixture = Fixture::with_family(
            factory,
            Arc::clone(&family) as Arc<dyn Family>,
            1 << 26,
            None,
            ServiceConfig::new("inkling2-mini", ENGINE),
        );
        let other = fixture.attach_again();
        let service = std::mem::replace(&mut fixture.service, other);
        fixture.service = service.with_objects(Arc::clone(&store));
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
        let id = format!("blake3:{}", blake3::hash(bytes).to_hex());
        let content = nagaya_store::ContentId::from_hex(id.trim_start_matches("blake3:")).unwrap();
        let mut store = self.store.lock().unwrap();
        let staged = store.staging_path();
        std::fs::create_dir_all(staged.parent().unwrap()).unwrap();
        std::fs::write(&staged, bytes).unwrap();
        store.commit(&content, mime, &staged, bytes.len() as u64, 0).unwrap();
        id
    }
}

/// A chat with an image part of 400 by 360 px becomes 10 by 9 = 90 placeholders. A pass of the fake takes 64 tokens,
/// so the span crosses a block boundary. Each pass that holds a part of it gets the record and the rows of the image
/// whole, and the checks of the Inkling 2 engine accept each pass. The rows are the rows of the family's processor.
#[test]
fn a_chat_with_an_image_part_becomes_the_placeholders_the_record_and_the_rows_of_the_engine() {
    let Some(chat) = Chat::start() else {
        return;
    };
    let image = png(400, 360);
    let id = chat.put(&image, "image/png");
    let context: Context = serde_json::from_value(json!({"thinking": false, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
                                     {"type": "object", "id": id, "mime": "image/png"}], "snap": true}]}))
    .unwrap();
    let rendering = chat.family.render(&context, false).unwrap();
    let [span] = rendering.media.as_slice() else {
        panic!("one media span: {:?}", rendering.media);
    };
    assert_eq!((span.count, span.mime.as_str()), (90, "image/png"));
    let ids = &rendering.tokens;
    assert!(
        ids[span.first..span.first + span.count]
            .iter()
            .all(|t| *t == IMAGE_TOKEN),
        "each position of the span holds the placeholder"
    );
    assert_eq!(
        ids.iter().filter(|t| **t == IMAGE_TOKEN).count(),
        90,
        "no placeholder outside the span"
    );
    let want = nagaya_text_inkling2::media::prepare(&image, "image/png").unwrap();
    assert_eq!(want.grid, Grid { rows: 9, columns: 10 });

    let built = chat
        .fixture
        .service
        .build(&context, false, Schedule::default())
        .unwrap();
    assert_eq!(built.usage.fed as usize, rendering.end);
    let segments = chat.seen.take();
    let (mut with_rows, mut covered) = (0, 0);
    for segment in &segments {
        let from = segment.positions[0] as usize;
        let to = from + segment.positions.len();
        let holds = from < span.first + span.count && span.first < to;
        match &segment.inputs {
            SegmentInputs::TokensAndRows { tokens, rows } => {
                assert!(holds, "the pass {from}..{to} holds no position of the span");
                with_rows += 1;
                assert_eq!(segment.meta, want.record(span.first), "the record of the family");
                assert_eq!(rows, &want.bytes(), "the rows of the image whole");
                let spans = decode(&segment.meta).unwrap();
                assert_eq!(spans, [Span::image(want.grid, span.first as u64)]);
                let uses = check_spans(&spans, tokens, from as u64, rows.len()).unwrap();
                covered += uses[0].take.len();
            }
            SegmentInputs::Tokens(tokens) => {
                assert!(!holds, "the pass {from}..{to} holds a part of the span and has no rows");
                assert!(segment.meta.is_empty());
                check_spans(&[], tokens, from as u64, 0).unwrap();
            }
            SegmentInputs::Rows(_) => panic!("a build sends no rows alone"),
        }
    }
    assert!(
        with_rows >= 2,
        "the span crosses a pass boundary: {with_rows} passes with rows"
    );
    assert_eq!(covered, 90, "the passes fill each position of the span once");
}

/// The opener of the process gives the family the objects of the process, so a render reads an image part. With no
/// objects, the family refuses the part by name (spec 11.18, "Images").
#[test]
fn the_opener_gives_the_family_the_objects_of_the_process() {
    let Some(dir) = folder() else {
        return;
    };
    let objects = tempfile::tempdir().unwrap();
    let store: nagaya::serving::ObjectReader = Arc::new(Mutex::new(
        ObjectStore::open(&objects.path().join("objects"), 1 << 26).unwrap(),
    ));
    let image = png(80, 40);
    let id = format!("blake3:{}", blake3::hash(&image).to_hex());
    {
        let content = nagaya_store::ContentId::from_hex(id.trim_start_matches("blake3:")).unwrap();
        let mut held = store.lock().unwrap();
        let staged = held.staging_path();
        std::fs::create_dir_all(staged.parent().unwrap()).unwrap();
        std::fs::write(&staged, &image).unwrap();
        held.commit(&content, "image/png", &staged, image.len() as u64, 0)
            .unwrap();
    }
    let context: Context = serde_json::from_value(json!({"thinking": false, "messages": [
        {"role": "user", "content": [{"type": "object", "id": id, "mime": "image/png"}]}]}))
    .unwrap();
    let identity = nagaya_abi::Engine::identity(&nagaya_abi::fake::FakeAlways::new().with_family("inkling2"));
    let table: toml::Table = format!("model_dir = \"{}\"", dir.display()).parse().unwrap();
    let with = nagaya::inkling2::open_family_with(&identity, &table, Some(&store)).unwrap();
    let rendering = with.family.render(&context, true).unwrap();
    assert_eq!(rendering.media.len(), 1);
    assert_eq!(rendering.media[0].count, 2, "an image of 80 by 40 px is 2 tokens");
    let without = nagaya::inkling2::open_family(&identity, &table).unwrap();
    let error = without.family.render(&context, true).unwrap_err();
    let text = format!("{error:?}");
    assert!(text.contains("gives the Inkling 2 family no objects"), "{text}");
}
