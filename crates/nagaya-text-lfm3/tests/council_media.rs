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
    map.insert(CLIP.to_string(), wav(1, 24_000));
    map.insert(STEREO.to_string(), wav(2, 1_000));
    for (id, file) in [(SMALL, "image_small.png"), (MID, "image_mid.png"), (CHIRP, "chirp.wav")] {
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
