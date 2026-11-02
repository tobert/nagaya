//! The inputs of an image job (spec 11.9b): the input PNG, the mask, its blur, and the paste after the decode.
//!
//! `tests/fixtures/inputs.json` holds what Pillow 13.0.0, numpy and torch make of small inputs. `tools/sdxl_inputs_fixture.py`
//! writes it once, with no GPU.

use nagaya_text_sdxl::inputs::{self, InputError, Png};
use serde_json::Value;

fn fixture() -> Value {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/inputs.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn hex(value: &Value) -> Vec<u8> {
    let text = value.as_str().expect("a hex string");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn size(case: &Value) -> (u32, u32) {
    (
        case["width"].as_u64().unwrap() as u32,
        case["height"].as_u64().unwrap() as u32,
    )
}

/// Each PNG kind that the nagaya reads gives the pixels of Pillow's `convert("RGB")` and `convert("L")`.
#[test]
fn each_png_kind_gives_the_pixels_of_pillow() {
    let fixture = fixture();
    let cases = fixture["pngs"].as_array().unwrap();
    assert_eq!(cases.len(), 8, "the fixture holds each kind");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let bytes = hex(&case["png"]);
        let (width, height) = size(case);
        let png = Png::read(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((png.width(), png.height()), (width, height), "{name}");
        let rgb = Png::read(&bytes)
            .unwrap()
            .rgb()
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(rgb, hex(&case["rgb"]), "{name}: the RGB pixels");
        let gray = png.gray().unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(gray, hex(&case["gray"]), "{name}: the gray pixels");
    }
}

/// A PNG with 16 bits, or with fewer than 8 bits, in a sample is refused, and the error gives the depth.
#[test]
fn a_png_of_another_depth_is_refused() {
    let fixture = fixture();
    for case in fixture["refused"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let depth = case["depth"].as_u64().unwrap() as u8;
        let bytes = hex(&case["png"]);
        let error = Png::read(&bytes).err().unwrap_or_else(|| panic!("{name} is refused"));
        assert_eq!(error, InputError::Depth(depth), "{name}");
        assert!(error.to_string().contains(&format!("{depth} bits")), "{name}: {error}");
    }
}

/// An RGB PNG with 16 bits in a channel is refused, as a gray one is.
#[test]
fn an_rgb_png_of_16_bits_is_refused() {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, 2, 2);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&[7; 2 * 2 * 6]).unwrap();
    writer.finish().unwrap();
    assert_eq!(Png::read(&bytes).err(), Some(InputError::Depth(16)));
}

/// Bytes that are not a PNG, and a PNG cut short, are refused as not a PNG.
#[test]
fn bytes_that_are_not_a_png_are_refused() {
    let fixture = fixture();
    let whole = hex(&fixture["pngs"][2]["png"]);
    let cut = &whole[..whole.len() - 30];
    for (name, bytes) in [
        ("text", b"GIF89a, not a png".as_slice()),
        ("empty", b"".as_slice()),
        ("cut", cut),
    ] {
        let read = Png::read(bytes).and_then(|png| png.rgb());
        assert!(matches!(read, Err(InputError::NotPng(_))), "{name}: {read:?}");
    }
}

/// A palette index past the end of the palette is not a PNG: the PNG standard calls it an error.
#[test]
fn a_palette_index_past_the_palette_is_refused() {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_palette(vec![1, 2, 3, 4, 5, 6]);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&[1, 2]).unwrap();
    writer.finish().unwrap();
    let read = Png::read(&bytes).unwrap().rgb();
    assert!(matches!(read, Err(InputError::NotPng(_))), "{read:?}");
}

/// The blur of each mask and each radius gives the bytes of Pillow's `ImageFilter.GaussianBlur`.
#[test]
fn the_blur_gives_the_bytes_of_pillow() {
    let fixture = fixture();
    let masks = fixture["blurs"].as_array().unwrap();
    assert_eq!(masks.len(), 5);
    for mask in masks {
        let name = mask["name"].as_str().unwrap();
        let (width, height) = size(mask);
        let gray = hex(&mask["gray"]);
        let blurred = mask["blurred"].as_object().unwrap();
        // The mask `sweep` has each `mask_blur` of a request, 0 to 64; the others have 10 radii.
        let radii = if name == "sweep" { 65 } else { 10 };
        assert_eq!(blurred.len(), radii, "{name}: each radius");
        for (radius, want) in blurred {
            let radius: u32 = radius.parse().unwrap();
            let got = inputs::blur(&gray, width, height, radius);
            assert_eq!(got, hex(want), "{name}: radius {radius}");
        }
    }
}

/// Each byte value maps to f32 as diffusers' `pil_to_numpy` and `normalize` map it: `2 * v / 255 - 1`.
#[test]
fn each_byte_maps_to_the_f32_of_diffusers() {
    let fixture = fixture();
    let want: Vec<u32> = fixture["unit"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
        .collect();
    let bytes: Vec<u8> = (0..=255).collect();
    let got: Vec<u32> = inputs::unit(&bytes).iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want);
}

/// A blurred value of 128 or more repaints, as diffusers' `binarize` at 0.5 of 255 does.
#[test]
fn the_binary_mask_repaints_from_128() {
    let fixture = fixture();
    let want: Vec<u8> = fixture["binary"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let bytes: Vec<u8> = (0..=255).collect();
    assert_eq!(inputs::binary(&bytes), want);
}

/// The latent mask takes the binary mask at pixel `(8x, 8y)`, as torch's `interpolate` in the mode `nearest` does.
#[test]
fn the_latent_mask_takes_each_eighth_pixel() {
    let fixture = fixture();
    let case = &fixture["latent"];
    let (width, height) = size(case);
    let list = |key: &str| -> Vec<u8> {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect()
    };
    assert_eq!(inputs::latent_mask(&list("binary"), width, height), list("latent"));
}

/// The paste keeps the old pixel where the blurred mask is 0, takes the new one where it is 255, and rounds between.
#[test]
fn the_paste_mixes_the_pixels_by_the_blurred_mask() {
    // Two pixels of RGB: a mask of 0 keeps the old pixel; 255 takes the new one.
    assert_eq!(
        inputs::paste(&[200, 100, 0, 200, 100, 0], &[10, 20, 30, 10, 20, 30], &[0, 255]),
        vec![10, 20, 30, 200, 100, 0]
    );
    // (a * new + (255 - a) * old + 127) / 255, in integers.
    let (a, new, old) = (100u32, 200u32, 10u32);
    let want = ((a * new + (255 - a) * old + 127) / 255) as u8;
    assert_eq!(want, 85);
    assert_eq!(
        inputs::paste(&[200, 0, 255], &[10, 255, 0], &[100]),
        vec![want, 155, 100]
    );
    // 1 of 255 rounds to the nearest: (1 * 255 + 254 * 0 + 127) / 255 is 1.
    assert_eq!(inputs::paste(&[255, 255, 255], &[0, 0, 0], &[1]), vec![1, 1, 1]);
}

/// The masked image of a 9-channel UNet is the map of `unit`, with 0 at each pixel that the binary mask repaints.
#[test]
fn the_masked_image_is_0_where_the_mask_repaints() {
    let pixels = [0, 128, 255, 255, 0, 51];
    let unit = inputs::unit(&pixels);
    assert_eq!(
        inputs::masked(&pixels, &[0, 1]),
        [unit[0], unit[1], unit[2], 0.0, 0.0, 0.0]
    );
    assert_eq!(
        inputs::masked(&pixels, &[1, 0]),
        [0.0, 0.0, 0.0, unit[3], unit[4], unit[5]]
    );
}
