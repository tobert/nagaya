//! The frames and the sound of a video (spec 11.11, "The video of a request"). The rules follow the checkpoint's video
//! processor (`video_processing_lfm3_vl.py`) and Omni's processor (`processing_lfm3_omni.py`). The decode of the
//! bytes is behind the feature `video`. The steps here take decoded frames, so their tests need no decoder.

use super::MediaError;
use super::image::{self, ImageRows, MERGE, PATCH, ROW, Rgb, TEMPORAL};

/// The frames that the sampler takes for each second of the stream (`fps` of the video processor).
pub const FPS: f64 = 2.0;
/// The fewest and the most frames of a video.
pub const MIN_FRAMES: usize = 4;
pub const MAX_FRAMES: usize = 384;
/// The pixel budget of all the frames of a video together: `size.shortest_edge` and `size.longest_edge`.
pub const MIN_PIXELS: usize = 4_096;
pub const MAX_PIXELS: usize = 25_165_824;
/// The seconds of sound in one row of the audio encoder: the hop of 160 samples, the stride of 8, at 16 kHz.
pub const SECONDS_PER_ROW: f64 = 160.0 * 8.0 / 16_000.0;

/// The indices of the frames that the sampler takes from a stream of `total` frames at `fps` frames each second. The
/// count is the floor of `total / fps * 2`, then at least 4, at most 384, and at most `total`. The indices are evenly
/// spaced from the first frame to the last, as `numpy.linspace`, and rounded half to even.
pub fn sample(total: usize, fps: f64) -> Result<Vec<usize>, MediaError> {
    if total < TEMPORAL {
        return Err(MediaError::Video(format!(
            "it has {total} frames, and a video needs {TEMPORAL} or more"
        )));
    }
    if !(fps.is_finite() && fps > 0.0) {
        return Err(MediaError::Video(format!(
            "its frame rate is {fps}, not a positive number"
        )));
    }
    let count = ((total as f64 / fps * FPS) as usize)
        .clamp(MIN_FRAMES, MAX_FRAMES)
        .min(total);
    if count == 1 {
        return Ok(vec![0]);
    }
    // numpy.linspace: start + i * step, and the last value is the stop.
    let stop = (total - 1) as f64;
    let step = stop / (count - 1) as f64;
    Ok((0..count)
        .map(|i| if i == count - 1 { stop } else { i as f64 * step })
        .map(|x| x.round_ties_even() as usize)
        .collect())
}

/// The one size of every frame of a video of `frames` sampled frames, as `smart_resize` of the video processor with
/// the factor 32, the time factor 2 and the pixel budget. The float steps are Python's.
pub fn smart_resize(frames: usize, height: usize, width: usize) -> Result<(usize, usize), MediaError> {
    if frames < TEMPORAL {
        return Err(MediaError::Video(format!(
            "it has {frames} frames, and a video needs {TEMPORAL} or more"
        )));
    }
    if height == 0 || width == 0 {
        return Err(MediaError::Video(format!("its frames have {height} by {width} pixels")));
    }
    let f = image::FACTOR as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < f || w < f {
        let scale = (f / h).max(f / w);
        h = (h * scale).trunc();
        w = (w * scale).trunc();
    }
    if h.max(w) / h.min(w) > 200.0 {
        return Err(MediaError::Ratio { height, width });
    }
    let t = frames as f64;
    let mut h_bar = (h / f).round_ties_even() * f;
    let mut w_bar = (w / f).round_ties_even() * f;
    let t_bar = (t / TEMPORAL as f64).round_ties_even() * TEMPORAL as f64;
    if t_bar * h_bar * w_bar > MAX_PIXELS as f64 {
        let beta = ((t * h * w) / MAX_PIXELS as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if t_bar * h_bar * w_bar < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (t * h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Resize each frame to `height` by `width` with the resize of an image, and repeat the last frame when the count is
/// odd, so that the frames make pairs.
pub fn resize_frames(frames: &[Rgb], height: usize, width: usize) -> Vec<Rgb> {
    let mut out: Vec<Rgb> = frames.iter().map(|f| image::resize(f, height, width)).collect();
    if !out.len().is_multiple_of(TEMPORAL)
        && let Some(last) = out.last().cloned()
    {
        out.push(last);
    }
    out
}

/// The rows of each pair of resized frames, each as the rows of an image of one frame in time: a patch is the 3
/// channels, then the 2 frames, then 16 rows and 16 columns, in the order of the merge blocks. The frames must be of
/// one size, a multiple of 32, and of an even count.
pub fn pair_rows(frames: &[Rgb]) -> Vec<ImageRows> {
    assert!(
        frames.len().is_multiple_of(TEMPORAL),
        "the frames make pairs: {} frames",
        frames.len()
    );
    let (height, width) = (frames[0].height, frames[0].width);
    assert!(
        frames.iter().all(|f| f.height == height && f.width == width),
        "the frames of a video have one size"
    );
    let (gh, gw) = (height / PATCH, width / PATCH);
    frames
        .as_chunks::<TEMPORAL>()
        .0
        .iter()
        .map(|pair| {
            let mut values = Vec::with_capacity(gh * gw * ROW);
            for by in 0..gh / MERGE {
                for bx in 0..gw / MERGE {
                    for my in 0..MERGE {
                        for mx in 0..MERGE {
                            let (py, px) = ((by * MERGE + my) * PATCH, (bx * MERGE + mx) * PATCH);
                            for c in 0..3 {
                                for frame in pair {
                                    for iy in 0..PATCH {
                                        for ix in 0..PATCH {
                                            let v = frame.bytes[((py + iy) * width + px + ix) * 3 + c];
                                            values.push((f32::from(v) - 127.5) / 127.5);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            ImageRows {
                grid: [1, gh, gw],
                values,
            }
        })
        .collect()
}

/// The time of each pair of the sampled frames `indices`, in seconds: the mean of the indices of its two frames, over
/// the stream's rate. An odd count repeats the last index.
pub fn timestamps(indices: &[usize], fps: f64) -> Vec<f64> {
    let mut padded = indices.to_vec();
    if !padded.len().is_multiple_of(TEMPORAL)
        && let Some(&last) = padded.last()
    {
        padded.push(last);
    }
    padded
        .as_chunks::<TEMPORAL>()
        .0
        .iter()
        .map(|pair| (pair[0] + pair[TEMPORAL - 1]) as f64 / (2.0 * fps))
        .collect()
}

/// The text of the time of a pair, as Omni's processor writes it: `<t seconds>` with one decimal.
pub fn time_text(seconds: f64) -> String {
    format!("<{seconds:.1} seconds>")
}

/// Where the encoder rows of the sound track split, one run for each pair: `rows` rows, cut at the time of the first
/// frame of each pair. The bounds start at 0, end at `rows`, and never go down (`processing_lfm3_omni.py`,
/// `_video_expansion`).
pub fn sound_bounds(indices: &[usize], fps: f64, rows: usize) -> Vec<usize> {
    let starts: Vec<f64> = indices.iter().step_by(TEMPORAL).map(|&i| i as f64 / fps).collect();
    let mut bounds = vec![0];
    for start in &starts[1.min(starts.len())..] {
        bounds.push(rows.min((start / SECONDS_PER_ROW).round_ties_even() as usize));
    }
    bounds.push(rows);
    for i in 1..bounds.len() {
        bounds[i] = bounds[i].max(bounds[i - 1]);
    }
    bounds
}
