//! The images of a context through the Inkling 2 family (engine-abi "The images of Inkling 2"): the decode, the rows
//! of the processor, the placeholders, the `meta` record, the size rule and the refusals.
//!
//! The decode tests read the small files of `tests/fixtures` (`tools/inkling2_media_fixture.py`) and need no model.
//! The oracle tests read `target/inkling2-golden` (`tools/inkling2_golden.sh --stages vision`), and the council tests
//! need the Mini folder. Each skips when its file is absent.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use common::model::{absent, golden_path, mini_dir};
use gogi_proto::context::Context;
use nagaya_council::family::{DecisionInput, Family, FamilyError};
use nagaya_text_inkling2::media::{self, IMAGE_TOKEN, MediaError, PATCH_VALUES, vision};
use nagaya_text_inkling2::{Inkling2Family, Objects};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const IMAGE0: &str = "blake3:1111111111111111111111111111111111111111111111111111111111111111";
const IMAGE1: &str = "blake3:2222222222222222222222222222222222222222222222222222222222222222";
const SMALL: &str = "blake3:3333333333333333333333333333333333333333333333333333333333333333";
const WIDE: &str = "blake3:4444444444444444444444444444444444444444444444444444444444444444";
const JUNK: &str = "blake3:5555555555555555555555555555555555555555555555555555555555555555";
const ABSENT: &str = "blake3:7777777777777777777777777777777777777777777777777777777777777777";

/// The objects of a test: a map of ids to bytes.
struct Map(HashMap<String, Vec<u8>>);

impl Objects for Map {
    fn bytes(&self, id: &str) -> Option<Vec<u8>> {
        self.0.get(id).cloned()
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn fixture_json() -> Value {
    let text = std::fs::read_to_string(common::fixture("media.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// A PNG of RGB with `width x height` pixels of one gray value.
fn png_of(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(&vec![99u8; (width * height * 3) as usize])
            .unwrap();
    }
    out
}

/// The pixel values of the processor for decoded RGB, as the reference returns them.
fn pixel_values(rgb: &media::Rgb) -> Vec<f32> {
    vision::pixel_values(&rgb.bytes, rgb.width, rgb.height).unwrap().1
}

#[test]
fn each_png_of_the_fixture_gives_the_rgb_and_the_pixel_values_of_the_processor() {
    let fixture = fixture_json();
    let images = fixture["images"].as_object().unwrap();
    let mut checked = 0;
    for (name, want) in images {
        if name.ends_with(".jpg") {
            continue;
        }
        let bytes = std::fs::read(common::fixture(name)).unwrap();
        let rgb = media::decode(&bytes, "image/png").unwrap();
        assert_eq!(
            (rgb.width as u64, rgb.height as u64),
            (want["width"].as_u64().unwrap(), want["height"].as_u64().unwrap()),
            "{name}: the size"
        );
        assert_eq!(
            sha(&rgb.bytes),
            want["rgb_sha256"],
            "{name}: the RGB after the convert step"
        );
        let px = pixel_values(&rgb);
        assert_eq!(
            sha(&f32_bytes(&px)),
            want["pixel_values_sha256"],
            "{name}: the pixel values"
        );
        let rows = media::prepare(&bytes, "image/png").unwrap();
        let grid: Vec<u64> = want["grid"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(
            grid,
            [1, rows.grid.rows as u64, rows.grid.columns as u64],
            "{name}: the grid"
        );
        assert_eq!(
            rows.values,
            vision::patch_rows(&px, rows.grid),
            "{name}: the patch rows"
        );
        assert_eq!(rows.values.len(), rows.grid.patches() * PATCH_VALUES);
        checked += 1;
    }
    // gray, gray_alpha, palette, palette_trns, rgb, rgba and photo_pil.
    assert_eq!(checked, 7);
}

#[test]
fn a_jpeg_decodes_within_2_of_pillow_and_its_grid_is_the_grid_of_the_processor() {
    let jpeg = std::fs::read(common::fixture("photo.jpg")).unwrap();
    let pil = media::decode(&std::fs::read(common::fixture("photo_pil.png")).unwrap(), "image/png").unwrap();
    let ours = media::decode(&jpeg, "image/jpeg").unwrap();
    assert_eq!((ours.width, ours.height), (pil.width, pil.height));
    let gaps: Vec<u8> = ours.bytes.iter().zip(&pil.bytes).map(|(a, b)| a.abs_diff(*b)).collect();
    let differ = gaps.iter().filter(|g| **g != 0).count();
    let worst = gaps.iter().copied().max().unwrap();
    eprintln!(
        "photo.jpg: {differ} of {} bytes differ from Pillow, by {worst} at most",
        gaps.len()
    );
    // spec 11.11, "JPEG": the decoder is not libjpeg-turbo, so a JPEG has no gate of bits.
    assert!(worst <= 2, "a byte differs from Pillow by {worst}");
    let want = &fixture_json()["images"]["photo.jpg"]["grid"];
    let rows = media::prepare(&jpeg, "image/jpeg").unwrap();
    assert_eq!(json!([1, rows.grid.rows, rows.grid.columns]), *want);
}

#[test]
fn a_side_outside_40_to_4096_px_is_refused_from_the_header() {
    for (w, h) in [(39, 40), (40, 39), (4097, 40), (40, 4097)] {
        let e = media::grid(&png_of(w, h), "image/png").unwrap_err();
        let MediaError::Size(why) = &e else {
            panic!("{w} x {h}: {e:?}");
        };
        assert!(why.contains("40 to 4096 px"), "{why}");
        assert!(media::prepare(&png_of(w, h), "image/png").is_err());
    }
    for (w, h, tokens) in [(40, 40, 1), (4096, 40, 103), (41, 81, 6)] {
        assert_eq!(media::grid(&png_of(w, h), "image/png").unwrap().tokens(), tokens);
    }
    assert!(matches!(
        media::grid(b"not an image", "image/png"),
        Err(MediaError::Image(_))
    ));
    assert!(matches!(media::grid(b"GIF89a", "image/gif"), Err(MediaError::Mime(_))));
}

#[test]
fn the_record_of_a_span_holds_the_kind_the_count_the_first_position_and_the_grid_in_patches() {
    let rows = media::prepare(&png_of(213, 118), "image/png").unwrap();
    let r = rows.record(18);
    assert_eq!(r.len(), media::RECORD_BYTES);
    let word = |i: usize| u32::from_le_bytes(r[i..i + 4].try_into().unwrap());
    assert_eq!((word(0), word(4)), (1, 18));
    assert_eq!(u64::from_le_bytes(r[8..16].try_into().unwrap()), 18);
    assert_eq!([word(16), word(20), word(24), word(28)], [1, 6, 12, 0]);
    // The engine reads `grid_t * grid_h * grid_w / 4` positions, the count of the record.
    assert_eq!((word(16) * word(20) * word(24) / 4) as usize, rows.positions());
}

/// The tensors of the oracle by name: the shape and the bytes of each.
type Tensors = HashMap<String, (Vec<usize>, Vec<u8>)>;

/// The oracle tensors of the image path, or `None` when the file is absent.
fn oracle() -> Option<Tensors> {
    let path = golden_path("inkling2-mini.vision.safetensors")?;
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensors = st
        .tensors()
        .into_iter()
        .map(|(name, t)| (name, (t.shape().to_vec(), t.data().to_vec())))
        .collect();
    Some(tensors)
}

fn i64s(data: &[u8]) -> Vec<i64> {
    data.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b)).collect()
}

fn png_file(i: usize) -> Option<Vec<u8>> {
    golden_path(&format!("inkling2-vision-image{i}.png")).map(|p| std::fs::read(p).unwrap())
}

#[test]
fn the_pngs_of_the_oracle_give_its_rgb_and_its_pixel_values_in_bits() {
    let Some(t) = oracle() else { return };
    for i in 0..2 {
        let Some(bytes) = png_file(i) else { return };
        let rgb = media::decode(&bytes, "image/png").unwrap();
        let (shape, data) = &t[&format!("image{i}.rgb")];
        assert_eq!(shape, &[rgb.height, rgb.width, 3]);
        let want: Vec<u8> = i64s(data).iter().map(|v| *v as u8).collect();
        assert_eq!(rgb.bytes, want, "image {i}: the RGB of the oracle");
        let px = pixel_values(&rgb);
        let (_, want) = &t[&format!("image{i}.pixel_values")];
        let differ = f32_bytes(&px)
            .as_chunks::<4>()
            .0
            .iter()
            .zip(want.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "image {i}: {} x {} px, {differ} pixel values differ in bits",
            rgb.width, rgb.height
        );
        assert_eq!(differ, 0, "image {i}: the pixel values differ from the processor");
        let rows = media::prepare(&bytes, "image/png").unwrap();
        let (_, grid) = &t[&format!("image{i}.grid")];
        assert_eq!(i64s(grid), [1, rows.grid.rows as i64, rows.grid.columns as i64]);
    }
}

fn family_with(objects: &[(&str, Vec<u8>)]) -> Inkling2Family {
    let map = objects.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect();
    Inkling2Family::from_dir(&mini_dir())
        .unwrap()
        .with_objects(Arc::new(Map(map)))
}

fn image(id: &str, mime: &str) -> Value {
    json!({"type": "object", "id": id, "mime": mime})
}

fn context(v: Value) -> Context {
    serde_json::from_value(v).unwrap()
}

/// The context of the oracle prompt: two images, then the question, with thinking off.
fn oracle_context() -> Context {
    context(json!({
        "thinking": false,
        "messages": [{"role": "user", "content": [
            image(IMAGE0, "image/png"),
            image(IMAGE1, "image/png"),
            {"type": "text", "text": "Describe the two images."}
        ]}]
    }))
}

#[test]
fn a_context_with_the_two_images_of_the_oracle_gives_its_ids_its_spans_and_its_rows() {
    if absent() {
        return;
    }
    let Some(t) = oracle() else { return };
    let (Some(png0), Some(png1)) = (png_file(0), png_file(1)) else {
        return;
    };
    let fam = family_with(&[(IMAGE0, png0.clone()), (IMAGE1, png1.clone())]);
    assert_eq!(fam.media_inputs(), ["image/png", "image/jpeg"]);
    let r = fam.render(&oracle_context(), true).unwrap();
    let want: Vec<u32> = i64s(&t["prompt.ids"].1).iter().map(|v| *v as u32).collect();
    assert_eq!(r.tokens, want, "the ids of the oracle prompt");
    assert_eq!(r.end, want.len());
    assert_eq!(r.boundaries.last().unwrap().tokens, want.len());
    let counts: Vec<usize> = r.media.iter().map(|s| s.count).collect();
    assert_eq!(counts, [18, 192]);
    for span in &r.media {
        assert!(r.tokens[span.positions()].iter().all(|t| *t == IMAGE_TOKEN));
    }
    let placeholders = r.tokens.iter().filter(|t| **t == IMAGE_TOKEN).count();
    assert_eq!(placeholders, 18 + 192, "each placeholder is in a span");
    assert_eq!((r.media[0].id.as_str(), r.media[1].id.as_str()), (IMAGE0, IMAGE1));
    let inputs = fam.media_rows(&r).unwrap();
    assert_eq!(inputs.len(), 2);
    for (i, (input, png)) in inputs.iter().zip([&png0, &png1]).enumerate() {
        let rows = media::prepare(png, "image/png").unwrap();
        assert_eq!(input.meta, rows.record(r.media[i].first), "image {i}: the record");
        // The rows are the patch rows of the oracle's pixel values.
        let px: Vec<f32> = t[&format!("image{i}.pixel_values")]
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        assert_eq!(
            &*input.rows,
            f32_bytes(&vision::patch_rows(&px, rows.grid)),
            "image {i}: the rows"
        );
    }
}

#[test]
fn a_boundary_before_an_image_keeps_its_place_and_one_after_it_moves_by_the_image_tokens() {
    if absent() {
        return;
    }
    let fam = family_with(&[(SMALL, png_of(213, 118))]);
    let ctx = |with_image: bool| {
        let content = if with_image {
            json!([image(SMALL, "image/png"), {"type": "text", "text": "What is it?"}])
        } else {
            json!([{"type": "text", "text": "x"}, {"type": "text", "text": "What is it?"}])
        };
        context(json!({
            "thinking": false,
            "messages": [
                {"role": "system", "content": "Be brief.", "snap": true},
                {"role": "user", "content": content, "snap": true},
                {"role": "assistant", "content": "A test card."},
                {"role": "user", "content": "And now?"}
            ]
        }))
    };
    let r = fam.render(&ctx(true), true).unwrap();
    assert_eq!(r.media.len(), 1);
    assert_eq!(r.media[0].count, 18);
    let span = r.media[0].positions();
    let b: Vec<usize> = r.boundaries.iter().map(|b| b.tokens).collect();
    assert!(
        b[0] <= span.start,
        "the boundary of the system message is before the image"
    );
    assert!(b[1] >= span.end, "the boundary of the image message is after the image");
    // The template writes one placeholder, and the family writes 18: the tokens after the image move by 17.
    let template = fam.render(&ctx(false), true).unwrap();
    assert_eq!(template.boundaries[0].tokens, b[0]);
    assert!(r.media.iter().all(|s| s.mime == "image/png"));
    assert_eq!(fam.media_rows(&r).unwrap()[0].span, r.media[0]);
}

fn refusal(fam: &Inkling2Family, ctx: &Context) -> (Option<String>, String) {
    match fam.render(ctx, true) {
        Err(FamilyError::Client(e)) => (e.param, e.message),
        other => panic!("expected a client error, got {other:?}"),
    }
}

#[test]
fn an_image_that_the_family_cannot_read_is_a_400_that_names_its_part() {
    if absent() {
        return;
    }
    let fam = family_with(&[
        (SMALL, png_of(213, 118)),
        (WIDE, png_of(4097, 40)),
        (JUNK, b"not a png".to_vec()),
    ]);
    let user = |content: Value| context(json!({"thinking": false, "messages": [{"role": "user", "content": content}]}));
    let cases: Vec<(&str, Context, &str)> = vec![
        (
            "a side of 4,097 px",
            user(json!([{"type": "text", "text": "hi"}, image(WIDE, "image/png")])),
            "40 to 4096 px",
        ),
        (
            "bytes that are no PNG",
            user(json!([image(JUNK, "image/png")])),
            "Send a PNG or a JPEG",
        ),
        (
            "a GIF",
            user(json!([image(SMALL, "image/gif")])),
            "image/png and image/jpeg",
        ),
        (
            "an absent object",
            user(json!([image(ABSENT, "image/png")])),
            "is not on this server",
        ),
        (
            "an image of a system message",
            context(json!({"thinking": false, "messages": [
                {"role": "system", "content": [image(SMALL, "image/png")]},
                {"role": "user", "content": "hi"}
            ]})),
            "user or a tool message",
        ),
        (
            "an image of an assistant message",
            context(json!({"thinking": false, "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [image(SMALL, "image/png")]},
                {"role": "user", "content": "next"}
            ]})),
            "user or a tool message",
        ),
    ];
    for (what, ctx, says) in cases {
        let (param, message) = refusal(&fam, &ctx);
        let i = ctx
            .messages
            .iter()
            .position(|m| {
                m.content
                    .as_ref()
                    .is_some_and(|c| matches!(c, gogi_proto::message::Content::Parts(_)))
            })
            .unwrap();
        let parts = match ctx.messages[i].content.as_ref().unwrap() {
            gogi_proto::message::Content::Parts(p) => p,
            _ => unreachable!(),
        };
        let j = parts
            .iter()
            .position(|p| matches!(p, gogi_proto::message::Part::Object { .. }))
            .unwrap();
        assert_eq!(
            param.as_deref(),
            Some(format!("messages[{i}].content[{j}]").as_str()),
            "{what}"
        );
        assert!(message.contains(says), "{what}: {message}");
        assert!(message.starts_with("nothing ran"), "{what}: {message}");
    }
    // A family with no object reader refuses the part too.
    let bare = Inkling2Family::from_dir(&mini_dir()).unwrap();
    let (param, message) = refusal(&bare, &user(json!([image(SMALL, "image/png")])));
    assert_eq!(param.as_deref(), Some("messages[0].content[0]"));
    assert!(message.contains("no objects"), "{message}");
}

#[test]
fn a_decision_with_an_image_is_a_400_until_the_reads_take_media() {
    if absent() {
        return;
    }
    let fam = family_with(&[(SMALL, png_of(213, 118))]);
    let ctx = context(
        json!({"thinking": false, "messages": [{"role": "user", "content": [image(SMALL, "image/png"), {"type": "text", "text": "What is it?"}]}]}),
    );
    let rendering = fam.render(&ctx, false).unwrap();
    let spec = gogi_proto::spec::parse_spec(
        &json!({
            "name": "g",
            "instructions": "Pick one.",
            "input_label": "photo",
            "questions": [{
                "id": "v",
                "type": "choice",
                "instructions": "Is it a test card?",
                "criteria": [{"option": "a", "means": "yes"}, {"option": "b", "means": "no"}]
            }]
        }),
        &gogi_proto::spec::SpecRules::default(),
    )
    .unwrap();
    let asked = [nagaya_council::family::Asked {
        question: 0,
        options: vec![0, 1],
    }];
    let input = DecisionInput {
        context: Some((&ctx, &rendering)),
        spec: &spec,
        case: "the case",
        asked: &asked,
    };
    match fam.decision_read(&input) {
        Err(FamilyError::Client(e)) => assert_eq!(e.param.as_deref(), Some("messages[0].content[0]")),
        other => panic!("expected a client error, got {:?}", other.err()),
    }
    let object: gogi_proto::objects::ObjectRef =
        serde_json::from_value(json!({"id": SMALL, "mime": "image/png"})).unwrap();
    let text_only = DecisionInput { context: None, ..input };
    match fam.decision_read_with_media(&text_only, &[object]) {
        Err(FamilyError::Client(e)) => assert_eq!(e.param.as_deref(), Some("case_media[0]")),
        other => panic!("expected a client error, got {:?}", other.err()),
    }
}
