//! The media preprocessing of `LFM3-Omni-2B` (spec 11.11, "The media of a request"): the decode, the resize and
//! the patches of an image, and the WAV and the mel rows of a clip, against the reference processor.
//!
//! `tests/fixtures/media.json` comes from `tools/lfm3_media_fixture.py`, so these tests run on each machine. The
//! oracle files of `tools/lfm3_reference.py` (`LFM3_GOLDEN_DIR`) add the images and the clip of the oracle. An
//! absent oracle file skips its test.

mod common;

use common::model::{fixture, fixture_path, golden, golden_path};
use nagaya_text_lfm3::media::{self, MediaError, MediaRows, audio, image};
use safetensors::SafeTensors;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn pair(v: &Value) -> (usize, usize) {
    (v[0].as_u64().unwrap() as usize, v[1].as_u64().unwrap() as usize)
}

/// The pattern image of `tools/lfm3_media_fixture.py`.
fn pattern(height: usize, width: usize) -> image::Rgb {
    let mut bytes = Vec::with_capacity(height * width * 3);
    for y in 0..height as u64 {
        for x in 0..width as u64 {
            for c in 0..3u64 {
                let noise = ((x * 2_654_435_761 + y * 40_503 + c * 97 + x * y * 7) % (1 << 32)) >> 24;
                let smooth = (x + y + 85 * c) % 256;
                let v = if (x / 16 + y / 16) % 2 == 0 { noise } else { smooth };
                bytes.push(v as u8);
            }
        }
    }
    image::Rgb { height, width, bytes }
}

/// The pattern clip of `tools/lfm3_media_fixture.py`.
fn pattern_clip(n: usize) -> Vec<i16> {
    (0..n as i64)
        .map(|i| (((i * 7919 + 12345) % 65536) - 32768) as i16)
        .collect()
}

/// A WAV file of PCM samples with the given header fields.
fn wav(tag: u16, channels: u16, rate: u32, bits: u16, pcm: &[i16]) -> Vec<u8> {
    let data: Vec<u8> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    let block = channels * bits / 8;
    out.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn f32s(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    view.data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

#[test]
fn smart_resize_gives_the_sizes_of_the_fixture() {
    let f = fixture("media.json");
    for case in f["images"].as_array().unwrap() {
        let (h, w) = pair(&case["source"]);
        assert_eq!(image::smart_resize(h, w).unwrap(), pair(&case["resized"]), "{h}x{w}");
        let positions = case["positions"].as_u64().unwrap() as usize;
        assert_eq!(image::positions(h, w).unwrap(), positions, "{h}x{w}");
    }
}

#[test]
fn smart_resize_gives_the_table_of_the_oracle() {
    let Some(facts) = golden("facts.json") else {
        return;
    };
    let table = &facts["vision"]["smart_resize"];
    assert_eq!(table["factor"], image::FACTOR);
    assert_eq!(table["min_pixels"], image::MIN_PIXELS);
    assert_eq!(table["max_pixels"], image::MAX_PIXELS);
    let rows = table["table"].as_array().unwrap();
    assert_eq!(rows.len(), 9);
    for row in rows {
        let (h, w) = pair(&row["source"]);
        assert_eq!(image::smart_resize(h, w).unwrap(), pair(&row["resized"]), "{h}x{w}");
        assert_eq!(
            image::positions(h, w).unwrap() as u64,
            row["positions"].as_u64().unwrap()
        );
    }
}

#[test]
fn smart_resize_rounds_a_half_to_even_as_python_does() {
    // 272 / 32 = 8.5 rounds to 8, 400 / 32 = 12.5 rounds to 12, and 304 / 32 = 9.5 rounds to 10.
    assert_eq!(image::smart_resize(272, 400).unwrap(), (256, 384));
    assert_eq!(image::smart_resize(304, 400).unwrap(), (320, 384));
}

#[test]
fn an_image_over_the_ratio_is_refused() {
    assert_eq!(
        image::smart_resize(10, 2001),
        Err(MediaError::Ratio {
            height: 10,
            width: 2001
        })
    );
    assert!(image::smart_resize(10, 2000).is_ok());
}

#[test]
fn the_resize_and_the_patches_have_the_bits_of_the_reference() {
    let f = fixture("media.json");
    for case in f["images"].as_array().unwrap() {
        let (h, w) = pair(&case["source"]);
        let (rh, rw) = pair(&case["resized"]);
        let resized = image::resize(&pattern(h, w), rh, rw);
        assert_eq!(
            hex(&resized.bytes),
            case["resized_sha256"],
            "resize of {h}x{w} to {rh}x{rw}"
        );
        let values = image::patches(&resized);
        assert_eq!(values.len(), (rh / 16) * (rw / 16) * image::ROW);
        assert_eq!(
            hex(&f32_bytes(&values)),
            case["pixel_values_sha256"],
            "patches of {h}x{w}"
        );
    }
}

#[test]
fn a_png_of_each_kind_decodes_as_pillow_converts_it() {
    let f = fixture("media.json");
    for case in f["decodes"].as_array().unwrap() {
        let file = case["file"].as_str().unwrap();
        if !file.ends_with(".png") {
            continue;
        }
        let bytes = std::fs::read(fixture_path(file)).unwrap();
        let rgb = image::decode(&bytes, "image/png").unwrap();
        assert_eq!((rgb.height, rgb.width), (24, 40), "{file}");
        assert_eq!(hex(&rgb.bytes), case["rgb_sha256"], "{file} ({})", case["mode"]);
    }
}

/// zune-jpeg is not libjpeg-turbo, the decoder of Pillow. This test measures the gap and bounds it.
#[test]
fn a_jpeg_decodes_near_pillow() {
    let jpeg = std::fs::read(fixture_path("photo.jpg")).unwrap();
    let pillow = std::fs::read(fixture_path("photo_pil.png")).unwrap();
    let ours = image::decode(&jpeg, "image/jpeg").unwrap();
    let theirs = image::decode(&pillow, "image/png").unwrap();
    assert_eq!((ours.height, ours.width), (48, 64));
    assert_eq!((theirs.height, theirs.width), (48, 64));
    let gap = ours
        .bytes
        .iter()
        .zip(&theirs.bytes)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    let differ = ours.bytes.iter().zip(&theirs.bytes).filter(|(a, b)| a != b).count();
    eprintln!(
        "jpeg: {differ} of {} bytes differ from Pillow, by at most {gap}",
        ours.bytes.len()
    );
    assert!(gap <= JPEG_GAP, "the JPEG decode differs from Pillow by {gap}");
    // 48 by 64 pixels resize to 224 by 320, which is 70 merge blocks.
    assert_eq!(media::positions(&jpeg, "image/jpeg").unwrap(), 70);
}

/// The largest gap of a byte of the JPEG decode from Pillow's, as measured on `photo.jpg`.
const JPEG_GAP: u8 = 2;

#[test]
fn bad_images_are_refused() {
    assert!(matches!(
        image::decode(b"not an image", "image/png"),
        Err(MediaError::Image(_))
    ));
    assert!(matches!(
        image::decode(b"not an image", "image/jpeg"),
        Err(MediaError::Image(_))
    ));
    assert!(matches!(
        media::positions(b"\x89PNG", "image/png"),
        Err(MediaError::Image(_))
    ));
    // A PNG of 16 bits in a sample.
    let mut deep = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut deep, 2, 2);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Sixteen);
        encoder.write_header().unwrap().write_image_data(&[0; 24]).unwrap();
    }
    assert_eq!(image::decode(&deep, "image/png"), Err(MediaError::Depth(16)));
    assert_eq!(media::positions(&deep, "image/png"), Err(MediaError::Depth(16)));
    assert!(matches!(media::prepare(&deep, "image/gif"), Err(MediaError::Mime(_))));
}

#[test]
fn the_oracle_pngs_give_the_pixel_values_of_the_reference() {
    let Some(path) = golden_path("vision.safetensors") else {
        return;
    };
    let file = std::fs::read(path).unwrap();
    let tensors = SafeTensors::deserialize(&file).unwrap();
    for name in ["small", "mid"] {
        let Some(png) = golden_path(&format!("image_{name}.png")) else {
            return;
        };
        let bytes = std::fs::read(png).unwrap();
        let rgb = image::decode(&bytes, "image/png").unwrap();
        assert_eq!(
            rgb.bytes,
            tensors.tensor(&format!("{name}.png_rgb")).unwrap().data(),
            "{name}"
        );
        let (h, w) = image::smart_resize(rgb.height, rgb.width).unwrap();
        let resized = image::resize(&rgb, h, w);
        let want = tensors.tensor(&format!("{name}.resized_rgb")).unwrap();
        assert_eq!(want.shape(), [h, w, 3]);
        assert_eq!(resized.bytes, want.data(), "{name}: the resize");
        let MediaRows::Image(rows) = media::prepare(&bytes, "image/png").unwrap() else {
            panic!("an image gives image rows");
        };
        let grid = tensors.tensor(&format!("{name}.grid_thw")).unwrap();
        let grid: Vec<usize> = grid
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| i64::from_le_bytes(b) as usize)
            .collect();
        assert_eq!(rows.grid.as_slice(), grid.as_slice(), "{name}");
        let want = f32s(&tensors.tensor(&format!("{name}.pixel_values")).unwrap());
        assert_eq!(f32_bytes(&rows.values), f32_bytes(&want), "{name}: the pixel values");
        assert_eq!(media::positions(&bytes, "image/png").unwrap(), rows.positions());
    }
}

#[test]
fn the_frames_and_the_positions_of_a_clip_follow_the_padding() {
    assert_eq!((audio::padded(0), audio::frames(0)), (400, 1));
    assert_eq!((audio::padded(400), audio::frames(400)), (400, 1));
    assert_eq!((audio::padded(401), audio::frames(401)), (560, 2));
    assert_eq!((audio::padded(560), audio::frames(560)), (560, 2));
    assert_eq!((audio::padded(24_000), audio::frames(24_000)), (24_080, 149));
    assert_eq!(audio::positions(149), 19);
    assert_eq!(audio::positions(8), 1);
    assert_eq!(audio::positions(9), 2);
}

/// The largest gap of a mel value from the reference. The reference computes in f32 with an FFT and a matrix
/// product that are not exact, and this crate computes in f64.
const MEL_GAP: f32 = 2e-6;

/// The bounds of the gap of the oracle clip, for the reference values at each level and above. A value of -1 is a
/// mel sum of 1e-8, and a value of 0 is a mel sum of 1e-4. The bounds are about 1.5 times the gaps of numpy's f64
/// FFT from the reference: 1.5e-2, 4.0e-4 and 4.5e-6.
const ORACLE_MEL_GAPS: [(f32, f32); 3] = [(-2.0, 2.5e-2), (-1.0, 6e-4), (0.0, 7e-6)];

#[test]
fn the_mel_rows_of_short_clips_match_the_reference() {
    let f = fixture("media.json");
    let mut worst = 0.0f32;
    for case in f["clips"].as_array().unwrap() {
        let n = case["samples"].as_u64().unwrap() as usize;
        let rows = audio::mel(&audio::samples(&pattern_clip(n)));
        assert_eq!(rows.frames as u64, case["frames"].as_u64().unwrap(), "{n} samples");
        assert_eq!(
            rows.positions() as u64,
            case["positions"].as_u64().unwrap(),
            "{n} samples"
        );
        // The fixture holds the bins first; the rows hold the frames first.
        let mel = case["mel"].as_array().unwrap();
        assert_eq!(mel.len(), audio::BINS);
        let mut want = vec![0.0f32; rows.frames * audio::BINS];
        for (m, bin) in mel.iter().enumerate() {
            for (t, v) in bin.as_array().unwrap().iter().enumerate() {
                want[t * audio::BINS + m] = v.as_f64().unwrap() as f32;
            }
        }
        worst = worst.max(max_abs(&rows.values, &want));
    }
    eprintln!("mel of the short clips: the largest gap is {worst:e}");
    assert!(worst <= MEL_GAP, "the mel rows differ from the reference by {worst:e}");
}

#[test]
fn the_oracle_clip_gives_the_mel_rows_of_the_reference() {
    let Some(path) = golden_path("audio_in.safetensors") else {
        return;
    };
    let Some(wav_path) = golden_path("chirp.wav") else {
        return;
    };
    let file = std::fs::read(path).unwrap();
    let tensors = SafeTensors::deserialize(&file).unwrap();
    let bytes = std::fs::read(wav_path).unwrap();
    let pcm = audio::read_wav(&bytes).unwrap();
    let want: Vec<u8> = tensors.tensor("pcm").unwrap().data().to_vec();
    assert_eq!(pcm.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), want);
    let samples = audio::samples(&pcm);
    assert_eq!(f32_bytes(&samples), tensors.tensor("samples").unwrap().data());
    // The filters are the bits of the reference.
    let filters = tensors.tensor("mel_filters").unwrap();
    assert_eq!(filters.shape(), [audio::FREQUENCIES, audio::BINS]);
    let want = f32s(&filters);
    let differ = audio::mel_filters()
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differ,
        0,
        "{differ} mel filter values differ, by at most {:e}",
        max_abs(audio::mel_filters(), &want)
    );
    let MediaRows::Audio(rows) = media::prepare(&bytes, "audio/wav").unwrap() else {
        panic!("a clip gives audio rows");
    };
    let mel = tensors.tensor("mel").unwrap();
    assert_eq!(mel.shape(), [audio::BINS, rows.frames]);
    let bins_first = f32s(&mel);
    let mut want = vec![0.0f32; rows.values.len()];
    for m in 0..audio::BINS {
        for t in 0..rows.frames {
            want[t * audio::BINS + m] = bins_first[m * rows.frames + t];
        }
    }
    // The reference rounds its FFT to f32, so a bin of low power holds the noise of that rounding. The chirp has
    // many such bins, so the gate has a bound for each level of the reference value.
    for (level, bound) in ORACLE_MEL_GAPS {
        let gap = rows
            .values
            .iter()
            .zip(&want)
            .filter(|&(_, w)| *w >= level)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("mel of the oracle clip, reference values from {level}: the largest gap is {gap:e}");
        assert!(
            gap <= bound,
            "the mel rows from {level} differ from the reference by {gap:e}"
        );
    }
    assert_eq!(rows.positions(), 19);
    assert_eq!(media::positions(&bytes, "audio/wav").unwrap(), 19);
}

#[test]
fn only_a_wav_of_pcm_in_16_bits_mono_at_16_khz_is_read() {
    let pcm = pattern_clip(500);
    assert_eq!(audio::read_wav(&wav(1, 1, 16_000, 16, &pcm)).unwrap(), pcm);
    assert_eq!(media::positions(&wav(1, 1, 16_000, 16, &pcm), "audio/wav").unwrap(), 1);
    for (what, bytes) in [
        ("stereo", wav(1, 2, 16_000, 16, &pcm)),
        ("44.1 kHz", wav(1, 1, 44_100, 16, &pcm)),
        ("8 bits", wav(1, 1, 16_000, 8, &pcm)),
        ("float", wav(3, 1, 16_000, 16, &pcm)),
        ("not RIFF", b"OggS and more bytes".to_vec()),
        ("no data", wav(1, 1, 16_000, 16, &pcm)[..36].to_vec()),
    ] {
        let e = audio::read_wav(&bytes).expect_err(what);
        assert!(matches!(e, MediaError::Wav(_)), "{what}: {e:?}");
        assert!(
            e.to_string().contains("Send a WAV of PCM in 16 bits, mono, at 16 kHz"),
            "{e}"
        );
    }
}
