//! The media parts of an Omni context through the family interface (spec 11.11): the object reader, the counts of
//! positions from the media, and the rows that the family hands the engine.

mod common;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use common::ids;
use common::model::{
    OMNI_VARIABLE, TEXT_VARIABLE, absent_any, absent_omni, dir, family, fixture_path, golden, golden_path,
};
use gogi_proto::context::Context;
use nagaya_council::family::{Family, FamilyError};
use nagaya_text_lfm3::media::{MediaRows, audio, image};
use nagaya_text_lfm3::{Lfm3Family, MediaKind, Objects};
use serde_json::{Value, json};

const PHOTO: &str = "blake3:1111111111111111111111111111111111111111111111111111111111111111";
const CLIP: &str = "blake3:2222222222222222222222222222222222222222222222222222222222222222";
const STEREO: &str = "blake3:3333333333333333333333333333333333333333333333333333333333333333";
const SMALL: &str = "blake3:4444444444444444444444444444444444444444444444444444444444444444";
const MID: &str = "blake3:5555555555555555555555555555555555555555555555555555555555555555";
const CHIRP: &str = "blake3:6666666666666666666666666666666666666666666666666666666666666666";
/// The two videos of the oracle (`tools/lfm3_golden.sh --stages video`), and the placeholder tokens of a video and a
/// clip in Omni's tokenizer.
#[cfg(feature = "video")]
const VIDEO_AV: &str = "blake3:9999999999999999999999999999999999999999999999999999999999999999";
#[cfg(feature = "video")]
const VIDEO_V: &str = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
#[cfg(feature = "video")]
const VIDEO_TOKEN: u32 = 125_029;
#[cfg(feature = "video")]
const AUDIO_TOKEN: u32 = 125_031;

/// The bytes of PHOTO under another id.
const TWIN: &str = "blake3:8888888888888888888888888888888888888888888888888888888888888888";
const ABSENT: &str = "blake3:7777777777777777777777777777777777777777777777777777777777777777";

/// The objects of a test: a map of ids to bytes.
struct Map(HashMap<String, Vec<u8>>);

impl Objects for Map {
    fn bytes(&self, id: &str) -> Option<Vec<u8>> {
        self.0.get(id).cloned()
    }
}

/// A WAV of PCM in 16 bits at 16 kHz with `channels` channels.
fn wav(channels: u16, samples: usize) -> Vec<u8> {
    let data: Vec<u8> = (0..samples * usize::from(channels))
        .flat_map(|i| (((i * 7919) % 65536) as i32 - 32768).to_le_bytes()[..2].to_vec())
        .collect();
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&(32_000 * u32::from(channels)).to_le_bytes());
    out.extend_from_slice(&(2 * channels).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

fn objects() -> Map {
    let mut map = HashMap::new();
    map.insert(PHOTO.to_string(), std::fs::read(fixture_path("photo.jpg")).unwrap());
    map.insert(TWIN.to_string(), std::fs::read(fixture_path("photo.jpg")).unwrap());
    map.insert(CLIP.to_string(), wav(1, 24_000));
    map.insert(STEREO.to_string(), wav(2, 1_000));
    for (id, file) in [
        (SMALL, "image_small.png"),
        (MID, "image_mid.png"),
        (CHIRP, "chirp.wav"),
        #[cfg(feature = "video")]
        (VIDEO_AV, "video_av.mp4"),
        #[cfg(feature = "video")]
        (VIDEO_V, "video_v.mp4"),
    ] {
        let path = common::model::golden_dir().join(file);
        if path.is_file() {
            map.insert(id.to_string(), std::fs::read(path).unwrap());
        }
    }
    Map(map)
}

/// The Omni family with the objects of the tests.
fn omni() -> &'static Lfm3Family {
    static FAMILY: OnceLock<Lfm3Family> = OnceLock::new();
    FAMILY.get_or_init(|| {
        Lfm3Family::from_dir(&dir(OMNI_VARIABLE))
            .unwrap()
            .with_objects(Arc::new(objects()))
    })
}

fn context(v: Value) -> Context {
    serde_json::from_value(v).expect("the test context is valid")
}

fn user(parts: Value) -> Context {
    context(
        json!({"thinking": false, "messages": [{"role": "system", "content": "Respond with text."},
                                {"role": "user", "content": parts, "snap": true},
                                {"role": "assistant", "content": "Yes."},
                                {"role": "user", "content": "more"}]}),
    )
}

fn param(e: FamilyError) -> (Option<String>, String) {
    match e {
        FamilyError::Client(i) => (i.param, i.message),
        other => panic!("a client error, not {other:?}"),
    }
}

#[test]
fn render_puts_the_positions_of_each_object() {
    if absent_omni() {
        return;
    }
    let c = user(json!([{"type": "object", "id": PHOTO, "mime": "image/jpeg"},
                        {"type": "text", "text": "and"},
                        {"type": "object", "id": CLIP, "mime": "audio/wav"}]));
    // 48 by 64 pixels resize to 224 by 320: 70 positions. 24,000 samples are 149 frames: 19 positions.
    let want = omni().render_media(&c, true, &[70, 19]).unwrap();
    let got = omni().render(&c, true).unwrap();
    assert_eq!(got, want.rendering);
    let prepared = omni().prepare(&c, true).unwrap();
    assert_eq!(prepared.rendering, want.rendering);
    assert_eq!(prepared.inputs.len(), 2);
    let (photo, clip) = (&prepared.inputs[0], &prepared.inputs[1]);
    assert_eq!(photo.span, want.spans[0]);
    assert_eq!(clip.span, want.spans[1]);
    assert_eq!((photo.span.kind, photo.span.count), (MediaKind::Image, 70));
    assert_eq!((clip.span.kind, clip.span.count), (MediaKind::Audio, 19));
    let MediaRows::Image(rows) = &photo.rows else {
        panic!("the photo gives image rows");
    };
    assert_eq!(rows.grid, [1, 14, 20]);
    assert_eq!(rows.values.len(), 280 * image::ROW);
    let MediaRows::Audio(rows) = &clip.rows else {
        panic!("the clip gives audio rows");
    };
    assert_eq!(rows.frames, 149);
    assert_eq!(rows.values.len(), 149 * audio::BINS);
    // A context with no media prepares with no inputs, and its rendering is the render.
    let text = context(json!({"thinking": false, "messages": [{"role": "user", "content": "hello"}]}));
    let plain = omni().prepare(&text, true).unwrap();
    assert!(plain.inputs.is_empty());
    assert_eq!(plain.rendering, omni().render(&text, true).unwrap());
}

#[test]
fn the_media_prompts_of_the_oracle_render_from_their_objects() {
    if absent_omni() {
        return;
    }
    let Some(media) = golden("lfm3-omni.media.json") else {
        return;
    };
    let Some(vision) = golden_path("vision.safetensors") else {
        return;
    };
    let vision = std::fs::read(vision).unwrap();
    let vision = safetensors::SafeTensors::deserialize(&vision).unwrap();
    for prompt in media["prompts"].as_array().unwrap() {
        let name = prompt["name"].as_str().unwrap();
        let (id, mime, question) = match name {
            "image_small" => (SMALL, "image/png", "Describe the image."),
            "image_mid" => (MID, "image/png", "Describe the image."),
            "audio_chirp" => (CHIRP, "audio/wav", "What do you hear?"),
            other => panic!("the oracle has the media prompt {other}"),
        };
        let c = context(json!({"thinking": false, "messages": [{"role": "user", "content": [
            {"type": "object", "id": id, "mime": mime}, {"type": "text", "text": question}]}]}));
        let got = omni().render(&c, true).unwrap();
        assert_eq!(got.tokens, ids(&prompt["ids"]), "{name}");
        if let Some(size) = name.strip_prefix("image_") {
            let prepared = omni().prepare(&c, true).unwrap();
            let want = vision.tensor(&format!("{size}.pixel_values")).unwrap();
            let bytes: Vec<u8> = prepared.inputs[0]
                .rows
                .values()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                bytes,
                want.data(),
                "{name}: the rows are the pixel values of the reference"
            );
        }
    }
}

#[test]
fn a_part_that_the_family_cannot_read_is_a_client_error_that_names_it() {
    if absent_omni() {
        return;
    }
    let at = |parts: Value| user(parts);
    let cases = [
        (
            json!([{"type": "object", "id": ABSENT, "mime": "image/png"}]),
            "is not on this server",
        ),
        (
            json!([{"type": "object", "id": STEREO, "mime": "audio/wav"}]),
            "2 channels",
        ),
        (
            json!([{"type": "object", "id": CLIP, "mime": "image/png"}]),
            "Send a PNG or a JPEG",
        ),
        (
            json!([{"type": "object", "id": PHOTO, "mime": "audio/wav"}]),
            "Send a WAV of PCM",
        ),
    ];
    for (parts, text) in cases {
        let c = at(parts);
        for e in [
            omni().render(&c, true).unwrap_err(),
            omni().prepare(&c, true).unwrap_err(),
        ] {
            let (p, message) = param(e);
            assert_eq!(p.as_deref(), Some("messages[1].content[0]"));
            assert!(message.contains(text), "{message}");
        }
    }
    // An assistant audio part needs the codes of the speech output, which a later step keeps.
    let spoken = context(json!({"thinking": false, "messages": [
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": [{"type": "object", "id": CLIP, "mime": "audio/wav"}]},
        {"role": "user", "content": "again"}]}));
    let (p, message) = param(omni().render(&spoken, true).unwrap_err());
    assert_eq!(p.as_deref(), Some("messages[1].content[0]"));
    assert!(message.contains("assistant audio part"), "{message}");
}

#[test]
fn a_family_without_objects_and_the_text_model_refuse_media() {
    if absent_any() {
        return;
    }
    let c = user(json!([{"type": "object", "id": PHOTO, "mime": "image/jpeg"}]));
    let (p, message) = param(family("omni").render(&c, true).unwrap_err());
    assert_eq!(p.as_deref(), Some("messages[1].content[0]"));
    assert!(message.contains("no objects"), "{message}");
    let text = Lfm3Family::from_dir(&dir(TEXT_VARIABLE))
        .unwrap()
        .with_objects(Arc::new(objects()));
    for e in [text.render(&c, true).unwrap_err(), text.prepare(&c, true).unwrap_err()] {
        let (p, message) = param(e);
        assert_eq!(p.as_deref(), Some("messages[1].content[0]"));
        assert!(message.contains("lfm3 reads text only"), "{message}");
    }
}

/// Omni reads PNG, JPEG and WAV, and `lfm3` reads text only (spec 11.11, "The served LFM3 models").
#[test]
fn omni_lists_its_media_inputs_and_the_text_model_lists_none() {
    if absent_any() {
        return;
    }
    // A build with the feature `video` reads video too (spec 11.11, "The video of a request").
    let mut want = vec!["image/png", "image/jpeg", "audio/wav"];
    if cfg!(feature = "video") {
        want.extend(["video/mp4", "video/webm"]);
    }
    assert_eq!(omni().media_inputs(), want);
    assert!(family("text").media_inputs().is_empty());
}

/// The little-endian words of a record of `meta`, as `nagaya-mk-lfm3-bmg::media` reads them: the kind, the count,
/// the first position as two words, and four words of the shape.
fn words(record: &[u8]) -> Vec<u32> {
    record
        .chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

/// The family gives the engine the record and the rows of each span of a rendering (engine-abi "The LFM3 vision
/// binding"). The rows are the prepared rows in f32, little-endian.
#[test]
fn the_media_rows_of_a_rendering_are_the_record_and_the_rows_of_each_span() {
    if absent_omni() {
        return;
    }
    let c = user(json!([{"type": "object", "id": PHOTO, "mime": "image/jpeg"},
                        {"type": "text", "text": "and"},
                        {"type": "object", "id": CLIP, "mime": "audio/wav"}]));
    let rendering = omni().render(&c, true).unwrap();
    let prepared = omni().prepare(&c, true).unwrap();
    let inputs = omni().media_rows(&rendering).unwrap();
    assert_eq!(inputs.len(), 2);
    let spans: Vec<_> = inputs.iter().map(|input| input.span.clone()).collect();
    assert_eq!(spans, rendering.media);
    assert_eq!((spans[0].mime.as_str(), spans[0].id.as_str()), ("image/jpeg", PHOTO));
    assert_eq!((spans[1].mime.as_str(), spans[1].id.as_str()), ("audio/wav", CLIP));
    let first = prepared.inputs[0].span.first;
    assert_eq!((spans[0].first, spans[0].count), (first, 70));
    assert_eq!(
        words(&inputs[0].meta),
        [1, 70, first as u32, 0, 1, 14, 20, 0],
        "an image: kind 1, its count, its first position, its grid"
    );
    let first = prepared.inputs[1].span.first;
    assert_eq!(
        words(&inputs[1].meta),
        [2, 19, first as u32, 0, 149, 0, 0, 0],
        "a clip: kind 2, its count, its first position, its frames"
    );
    for (input, want) in inputs.iter().zip(&prepared.inputs) {
        let rows: Vec<u8> = want.rows.values().iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(&input.rows[..], &rows[..]);
    }
    // A rendering with no media has no inputs.
    let text = context(json!({"thinking": false, "messages": [{"role": "user", "content": "hello"}]}));
    assert!(
        omni()
            .media_rows(&omni().render(&text, true).unwrap())
            .unwrap()
            .is_empty()
    );
    // A span whose count does not match its object is a defect of the caller.
    let mut wrong = rendering.clone();
    wrong.media[0].count = 71;
    assert!(matches!(omni().media_rows(&wrong), Err(FamilyError::Internal(_))));
}

fn spec() -> gogi_proto::spec::Spec {
    gogi_proto::spec::parse_spec(
        &json!({"name": "g", "instructions": "i", "input_label": "x", "questions": [
            {"id": "verdict", "type": "choice", "instructions": "What. Pick one.", "criteria": [
                {"option": "allow", "means": "a"}, {"option": "no", "means": "c"}]}]}),
        &gogi_proto::spec::SpecRules::default(),
    )
    .unwrap()
}

fn object(id: &str, mime: &str) -> gogi_proto::objects::ObjectRef {
    serde_json::from_value(json!({"id": id, "mime": mime})).unwrap()
}

/// A plain read of Omni takes the case media after the case text, in order, and gives the rows of each at its
/// position in the whole read (spec 11.11, "Media in a decision").
#[test]
fn a_plain_read_puts_the_case_media_after_the_case_and_gives_their_rows() {
    use nagaya_council::family::{Advance, Asked, DecisionInput};
    if absent_omni() {
        return;
    }
    let spec = spec();
    let asked = [Asked {
        question: 0,
        options: vec![0, 1],
    }];
    let c = context(json!({"thinking": false, "messages": [{"role": "user", "content": "hi"}]}));
    let rendering = omni().render(&c, false).unwrap();
    let input = DecisionInput {
        context: Some((&c, &rendering)),
        spec: &spec,
        case: "a case",
        asked: &asked,
    };
    let plain = omni().decision_read(&input).unwrap();
    let media = [object(PHOTO, "image/jpeg"), object(CLIP, "audio/wav")];
    let mut plan = omni().decision_read_with_media(&input, &media).unwrap();
    let inputs = plan.media();
    assert_eq!(inputs.len(), 2);
    let end = rendering.end;
    let (photo, clip) = (&inputs[0].span, &inputs[1].span);
    assert_eq!(
        (photo.id.as_str(), photo.mime.as_str(), photo.count),
        (PHOTO, "image/jpeg", 70)
    );
    assert_eq!(
        (clip.id.as_str(), clip.mime.as_str(), clip.count),
        (CLIP, "audio/wav", 19)
    );
    // The spans are in the prefix, after the case text, the photo first.
    let prefix = plan.prefix().to_vec();
    let at = |p: usize| prefix[p - end];
    assert!(photo.first > end && photo.first + photo.count <= clip.first);
    assert!(clip.first + clip.count <= end + prefix.len());
    let image = omni().tokenize_special("<|image|>").unwrap()[0];
    let audio = omni().tokenize_special("<|audio|>").unwrap()[0];
    assert!(photo.positions().all(|p| at(p) == image));
    assert!(clip.positions().all(|p| at(p) == audio));
    // The template opens an image with its marker, so the case text comes just before the marker.
    let before = omni().detokenize(&prefix[..photo.first - end]).unwrap();
    assert!(before.ends_with("x:\na case<|vision_start|>"), "{before:?}");
    // The rows are the rows that a rendering of the same objects gives.
    let prepared = omni()
        .prepare(
            &user(json!([{"type": "object", "id": PHOTO, "mime": "image/jpeg"}])),
            true,
        )
        .unwrap();
    let rows: Vec<u8> = prepared.inputs[0]
        .rows
        .values()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(&inputs[0].rows[..], &rows[..]);
    // The steps of the questions follow the prefix, and the hash covers the media.
    let Advance::Step(step) = plan.advance(None).unwrap() else {
        panic!("a step first");
    };
    assert!(!step.feed.is_empty());
    let Advance::Done(done) = plan.advance(Some(&[-1.0, -2.0])).unwrap() else {
        panic!("one question");
    };
    let mut plain = plain;
    plain.advance(None).unwrap();
    let Advance::Done(plain_done) = plain.advance(Some(&[-1.0, -2.0])).unwrap() else {
        panic!("one question");
    };
    assert_ne!(done.rendered_hash, plain_done.rendered_hash);
    // The same pixels under another id give the same tokens and another hash: the hash covers the ids.
    let mut twin = omni()
        .decision_read_with_media(&input, &[object(TWIN, "image/jpeg"), object(CLIP, "audio/wav")])
        .unwrap();
    assert_eq!(twin.prefix(), &prefix[..]);
    twin.advance(None).unwrap();
    let Advance::Done(twin_done) = twin.advance(Some(&[-1.0, -2.0])).unwrap() else {
        panic!("one question");
    };
    assert_ne!(twin_done.rendered_hash, done.rendered_hash);
    let again = omni().decision_read_with_media(&input, &media).unwrap();
    assert_eq!(again.prefix(), &prefix[..]);
    // With no case media, the read is the plain read.
    let none = omni().decision_read_with_media(&input, &[]).unwrap();
    assert_eq!(none.prefix(), omni().decision_read(&input).unwrap().prefix());
    assert!(none.media().is_empty());
}

/// A decision context may hold media: the read goes on from its rendering, and the build reads its rows.
#[test]
fn a_plain_read_of_a_context_with_media_goes_on_from_its_rendering() {
    use nagaya_council::family::{Asked, DecisionInput};
    if absent_omni() {
        return;
    }
    let spec = spec();
    let asked = [Asked {
        question: 0,
        options: vec![0, 1],
    }];
    let c = user(json!([{"type": "object", "id": PHOTO, "mime": "image/jpeg"}, {"type": "text", "text": "this"}]));
    let rendering = omni().render(&c, false).unwrap();
    let input = DecisionInput {
        context: Some((&c, &rendering)),
        spec: &spec,
        case: "a case",
        asked: &asked,
    };
    let plan = omni().decision_read(&input).unwrap();
    assert!(plan.media().is_empty(), "the context's media are the build's");
    let text = context(json!({"thinking": false, "messages": [{"role": "user", "content": "this"}]}));
    let text_rendering = omni().render(&text, false).unwrap();
    let text_plan = omni()
        .decision_read(&DecisionInput {
            context: Some((&text, &text_rendering)),
            ..input
        })
        .unwrap();
    // The read after the context is the same text in both: the spec turn and the case turn.
    let tail = |p: &[u32]| omni().detokenize(p).unwrap();
    assert_eq!(tail(plan.prefix()), tail(text_plan.prefix()));
}

/// A queue read holds no media: its case is the result of a call (spec 11.11, "Media in a decision").
#[test]
fn a_queue_read_refuses_case_media_with_a_400_that_names_it() {
    use nagaya_council::family::{Asked, DecisionInput};
    if absent_omni() {
        return;
    }
    let spec = spec();
    let asked = [Asked {
        question: 0,
        options: vec![0, 1],
    }];
    let tool = json!({"name": "next", "input_schema": {"type": "object", "properties": {"answers": {"type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["A", "B"]}}}}}});
    let c = context(
        json!({"thinking": false, "tools": [tool], "queue": "next", "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "c0", "name": "next", "input": {}}]}]}),
    );
    // Without media, the read is a queue read.
    assert!(
        omni()
            .decision_read(&DecisionInput {
                context: Some((&c, &omni().render(&c, false).unwrap())),
                spec: &spec,
                case: "a result",
                asked: &asked
            })
            .is_ok()
    );
    let rendering = omni().render(&c, false).unwrap();
    let input = DecisionInput {
        context: Some((&c, &rendering)),
        spec: &spec,
        case: "a result",
        asked: &asked,
    };
    let error = omni()
        .decision_read_with_media(&input, &[object(PHOTO, "image/jpeg")])
        .err()
        .expect("a queue read with case media is refused");
    assert_eq!(param(error).0.as_deref(), Some("case_media[0]"));
}

/// The ids and the spans of a video part, against the reference with `use_audio_in_video` (spec 11.11, "The video of
/// a request"): each pair writes its time and its video tokens, and the sound's rows follow each pair.
#[cfg(feature = "video")]
#[test]
fn a_video_part_renders_as_the_reference_and_gives_the_rows_of_each_pair_and_each_run() {
    use safetensors::SafeTensors;
    if absent_omni() {
        return;
    }
    let Some(path) = golden_path("video.safetensors") else {
        return;
    };
    let oracle = std::fs::read(path).unwrap();
    let st = SafeTensors::deserialize(&oracle).unwrap();
    let i64s = |name: &str| -> Vec<u32> {
        st.tensor(name)
            .unwrap()
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as u32)
            .collect()
    };
    let f32s = |name: &str| -> Vec<f32> {
        st.tensor(name)
            .unwrap()
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    for (name, id) in [("av", VIDEO_AV), ("v", VIDEO_V)] {
        let c = context(json!({"thinking": false, "messages": [{"role": "user", "content": [
            {"type": "object", "id": id, "mime": "video/mp4"}, {"type": "text", "text": "Describe the video."}]}]}));
        let rendering = omni().render(&c, true).unwrap();
        assert_eq!(
            rendering.tokens,
            i64s(&format!("{name}.ids")),
            "{name}: the ids of the reference"
        );
        let inputs = omni().media_rows(&rendering).unwrap();
        let grid = i64s(&format!("{name}.grid_thw"));
        let (pairs, per) = (grid[0] as usize, (grid[1] * grid[2]) as usize);
        let images: Vec<_> = inputs.iter().filter(|i| words(&i.meta)[0] == 1).collect();
        assert_eq!(images.len(), pairs, "{name}: a span for each pair");
        let patches = f32s(&format!("{name}.pixel_values"));
        for (k, input) in images.iter().enumerate() {
            assert_eq!(
                words(&input.meta)[4..7],
                [1, grid[1], grid[2]],
                "{name}: pair {k} is an image of one frame"
            );
            let want: Vec<u8> = patches[k * per * 1536..(k + 1) * per * 1536]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(&input.rows[..], &want[..], "{name}: the rows of pair {k}");
            assert!(
                input.span.mime.starts_with("video/mp4;avformat="),
                "{}",
                input.span.mime
            );
            assert!(
                rendering.tokens[input.span.positions()]
                    .iter()
                    .all(|&t| t == VIDEO_TOKEN)
            );
        }
        let runs: Vec<_> = inputs.iter().filter(|i| words(&i.meta)[0] == 3).collect();
        if name == "av" {
            // The runs of the sound: 15, 15 and 2 rows of 32, each after the tokens of its pair.
            let counts: Vec<u32> = runs.iter().map(|i| words(&i.meta)[1]).collect();
            assert_eq!(counts, [15, 15, 2]);
            let starts: Vec<u32> = runs.iter().map(|i| words(&i.meta)[5]).collect();
            assert_eq!(starts, [0, 15, 30]);
            let mel = f32s("av.mel");
            for run in &runs {
                let frames = words(&run.meta)[4] as usize;
                assert_eq!(frames * 128, mel.len());
                // The rows are the clip's mel rows, frames first: the transpose of the reference's [128, T].
                let got: Vec<f32> = run
                    .rows
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                // The family computes the mel in f64 and the reference in f32, so the gate has the bounds of the audio
                // gate for each level of the reference value (`tests/media.rs`, ORACLE_MEL_GAPS).
                for (level, bound) in [(-2.0f32, 2.5e-2f32), (-1.0, 6e-4), (0.0, 7e-6)] {
                    let gap = (0..frames)
                        .flat_map(|f| (0..128).map(move |b| (f, b)))
                        .filter(|&(f, b)| mel[b * frames + f] >= level)
                        .map(|(f, b)| (got[f * 128 + b] - mel[b * frames + f]).abs())
                        .fold(0.0f32, f32::max);
                    assert!(gap <= bound, "the mel of the sound from {level} differs by {gap:e}");
                }
                assert!(
                    rendering.tokens[run.span.positions()]
                        .iter()
                        .all(|&tk| tk == AUDIO_TOKEN)
                );
            }
        } else {
            assert!(runs.is_empty(), "a video with no sound has no runs");
        }
    }
}
