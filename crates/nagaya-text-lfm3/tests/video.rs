//! The frames and the sound of a video, from the frames that the reference decoded (spec 11.11, "The video of a
//! request"). The oracle is `video.safetensors` of `tools/lfm3_golden.sh --stages video`. No decoder runs here, so
//! each stage after the decode has a gate of bits.

mod common;

use common::model::golden_path;
use nagaya_text_lfm3::media::image::Rgb;
use nagaya_text_lfm3::media::video;
use safetensors::SafeTensors;

struct Oracle {
    bytes: Vec<u8>,
    meta: serde_json::Value,
}

fn oracle() -> Option<Oracle> {
    let path = golden_path("video.safetensors")?;
    let bytes = std::fs::read(&path).unwrap();
    let (_, meta) = SafeTensors::read_metadata(&bytes).unwrap();
    // The reference keeps its facts as JSON under the key `meta` (`tools/lfm3_reference.py`, `save`).
    let meta: serde_json::Value = serde_json::from_str(&meta.metadata().as_ref().unwrap()["meta"]).unwrap();
    Some(Oracle {
        bytes,
        meta: meta["videos"].clone(),
    })
}

impl Oracle {
    fn tensor(&self, name: &str) -> (Vec<usize>, Vec<u8>) {
        let st = SafeTensors::deserialize(&self.bytes).unwrap();
        let t = st.tensor(name).unwrap_or_else(|_| panic!("the oracle holds {name}"));
        (t.shape().to_vec(), t.data().to_vec())
    }

    fn i64s(&self, name: &str) -> Vec<i64> {
        let (_, data) = self.tensor(name);
        data.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b)).collect()
    }

    fn f32s(&self, name: &str) -> Vec<f32> {
        let (_, data) = self.tensor(name);
        data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }

    /// The frames of a u8 tensor `[n, h, w, 3]`.
    fn frames(&self, name: &str) -> Vec<Rgb> {
        let (shape, data) = self.tensor(name);
        let [n, h, w, 3] = shape[..] else {
            panic!("{name} has the shape {shape:?}")
        };
        (0..n)
            .map(|i| Rgb {
                height: h,
                width: w,
                bytes: data[i * h * w * 3..(i + 1) * h * w * 3].to_vec(),
            })
            .collect()
    }

    fn video(&self, name: &str) -> &serde_json::Value {
        &self.meta[name]
    }
}

const NAMES: [&str; 2] = ["av", "v"];

#[test]
fn the_sampler_takes_2_frames_a_second_evenly_spaced_and_rounded_half_to_even() {
    let Some(o) = oracle() else { return };
    for name in NAMES {
        let m = o.video(name);
        let (total, fps) = (
            m["stream_frames"].as_u64().unwrap() as usize,
            m["fps"].as_f64().unwrap(),
        );
        let got: Vec<i64> = video::sample(total, fps)
            .unwrap()
            .into_iter()
            .map(|i| i as i64)
            .collect();
        assert_eq!(got, o.i64s(&format!("{name}.indices")), "{name}");
    }
    // Fewer than 2 frames cannot make a pair, and the bounds hold.
    assert!(video::sample(1, 30.0).is_err());
    assert_eq!(video::sample(3, 30.0).unwrap(), [0, 1, 2]);
    assert_eq!(video::sample(10_000, 1.0).unwrap().len(), 384);
    assert_eq!(video::sample(300, 30.0).unwrap().len(), 20);
    // linspace(0, 9, 5) holds 4.5, which rounds to the even 4.
    assert_eq!(video::sample(10, 4.0).unwrap(), [0, 2, 4, 7, 9]);
}

#[test]
fn the_frames_resize_to_one_size_by_the_pixel_budget_and_pairs_give_the_patches() {
    let Some(o) = oracle() else { return };
    for name in NAMES {
        let decoded = o.frames(&format!("{name}.decoded_rgb"));
        let grid = o.i64s(&format!("{name}.grid_thw"));
        let (h, w) = video::smart_resize(decoded.len(), decoded[0].height, decoded[0].width).unwrap();
        assert_eq!((h, w), (grid[1] as usize * 16, grid[2] as usize * 16), "{name}");
        let resized = video::resize_frames(&decoded, h, w);
        assert_eq!(
            resized,
            o.frames(&format!("{name}.resized_rgb")),
            "{name}: an odd count repeats the last frame"
        );
        let pairs = video::pair_rows(&resized);
        assert_eq!(pairs.len(), grid[0] as usize, "{name}");
        let values: Vec<f32> = pairs.iter().flat_map(|p| p.values.iter().copied()).collect();
        let want = o.f32s(&format!("{name}.pixel_values"));
        assert_eq!(values.len(), want.len(), "{name}");
        assert!(
            values.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: the patch bits"
        );
        assert!(
            pairs.iter().all(|p| p.grid == [1, grid[1] as usize, grid[2] as usize]),
            "{name}"
        );
    }
    // The bounds of the budget, from `smart_resize` of the checkpoint's video processor on 2026-11-08.
    for (t, h, w, want) in [
        (384, 1080, 1920, (192, 320)),
        (100, 1080, 1920, (352, 640)),
        (4, 20, 20, (32, 32)),
        (5, 30, 1000, (32, 1056)),
        (7, 480, 640, (480, 640)),
        (384, 240, 320, (192, 288)),
    ] {
        assert_eq!(video::smart_resize(t, h, w).unwrap(), want, "{t} frames of {h} by {w}");
    }
}

#[test]
fn each_pair_writes_its_mean_time_and_the_sound_splits_at_the_first_frame_of_each_pair() {
    let Some(o) = oracle() else { return };
    let m = o.video("av");
    let fps = m["fps"].as_f64().unwrap();
    let indices: Vec<usize> = o.i64s("av.indices").into_iter().map(|i| i as usize).collect();
    let times: Vec<String> = video::timestamps(&indices, fps)
        .iter()
        .map(|t| video::time_text(*t))
        .collect();
    assert_eq!(times, ["<0.3 seconds>", "<1.5 seconds>", "<2.4 seconds>"]);
    let audio = m["audio_positions"].as_u64().unwrap() as usize;
    let bounds = video::sound_bounds(&indices, fps, audio);
    assert_eq!(bounds, [0, 15, 30, 32]);
    // The bounds never go down, and never pass the rows of the clip.
    assert_eq!(video::sound_bounds(&[0, 0, 50, 50], 1.0, 10), [0, 10, 10]);
}

/// The decode with ffmpeg's libraries against the frames and the sound that torchcodec gave the reference. With the
/// same ffmpeg build, both give the same bits: on 2026-11-08 with ffmpeg 9.0.2, no byte and no sample differed.
#[cfg(feature = "video")]
#[test]
fn the_decode_gives_the_bits_of_the_frames_and_the_sound_of_the_reference() {
    use nagaya_text_lfm3::media::decode;
    let Some(o) = oracle() else { return };
    for name in NAMES {
        let bytes = std::fs::read(golden_path(&format!("video_{name}.mp4")).unwrap()).unwrap();
        let d = decode::decode(&bytes).unwrap();
        let m = o.video(name);
        assert_eq!(d.total as u64, m["stream_frames"].as_u64().unwrap(), "{name}");
        assert_eq!(d.fps, m["fps"].as_f64().unwrap(), "{name}");
        let want: Vec<usize> = o
            .i64s(&format!("{name}.indices"))
            .into_iter()
            .map(|i| i as usize)
            .collect();
        assert_eq!(d.indices, want, "{name}");
        let frames = o.frames(&format!("{name}.decoded_rgb"));
        let (mut differ, mut most, mut all) = (0usize, 0u8, 0usize);
        for (got, want) in d.frames.iter().zip(&frames) {
            assert_eq!((got.height, got.width), (want.height, want.width), "{name}");
            for (a, b) in got.bytes.iter().zip(&want.bytes) {
                all += 1;
                if a != b {
                    differ += 1;
                    most = most.max(a.abs_diff(*b));
                }
            }
        }
        println!("{name}: {differ} of {all} bytes differ, the most by {most}");
        assert_eq!(d.frames.len(), frames.len(), "{name}");
        assert_eq!(differ, 0, "{name}: the frames differ, the most by {most}");
        match (d.sound.as_ref(), m["sound"].as_bool().unwrap()) {
            (Some(samples), true) => {
                let want = o.f32s(&format!("{name}.samples"));
                assert_eq!(samples.len(), want.len(), "{name}");
                assert!(
                    samples.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{name}: the samples"
                );
            }
            (None, false) => {}
            (got, sound) => panic!(
                "{name}: the sound is {:?} samples, and the video has sound {sound}",
                got.map(Vec::len)
            ),
        }
    }
    println!("{}", decode::versions());
}

/// The limits of the engine come before the work: a pair of more than 2,048 positions, and a sound of more than 4,096
/// mel frames, are refused with the reason (spec 11.11, "The video of a request").
#[test]
fn a_video_past_the_limits_of_the_engine_is_refused_before_the_work() {
    assert!(
        video::limits(1440, 1408, None).is_ok(),
        "45 by 44 blocks are 1,980 positions"
    );
    let e = video::limits(1440, 2560, None).unwrap_err().to_string();
    assert!(e.contains("3600 positions") && e.contains("at most 2048"), "{e}");
    // 4,096 mel frames are 400 + 4,095 * 160 samples; one more frame is too long.
    assert!(video::limits(320, 320, Some(400 + 4_095 * 160)).is_ok());
    let e = video::limits(320, 320, Some(400 + 4_096 * 160))
        .unwrap_err()
        .to_string();
    assert!(e.contains("at most 4096 mel frames"), "{e}");
}
