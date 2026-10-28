//! The work of the engine on the host: the features of the timestep and of the time ids, the token weights of A1111,
//! the chunks of the batch, the bytes of an image, and the preview. None of it needs a card.

use nagaya_abi::ImageChunk;

use crate::spec::*;

/// diffusers' `get_timestep_embedding` with `flip_sin_to_cos` and no shift, in f32: the cosines, then the sines.
pub fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10000f32.ln()) * i as f32 / half as f32).exp())
        .collect();
    let args: Vec<f32> = freqs.iter().map(|f| t * f).collect();
    args.iter()
        .map(|a| a.cos())
        .chain(args.iter().map(|a| a.sin()))
        .collect()
}

/// The time ids of SDXL for an image of `width` by `height`: the original size, the crop origin and the target size,
/// each as height, then width.
pub fn time_ids(width: usize, height: usize) -> [f32; 6] {
    [height as f32, width as f32, 0.0, 0.0, height as f32, width as f32]
}

/// The input of the add embedding of one row of the batch: the pooled text, then the features of each time id.
pub fn add_features(pooled: &[f32], ids: &[f32; 6]) -> Vec<f32> {
    assert_eq!(pooled.len(), POOLED);
    let mut out = pooled.to_vec();
    for id in ids {
        out.extend(timestep_features(*id, TIME_ID_PROJ));
    }
    out
}

/// The chunk of an empty prompt: the start id, the end id, then the pad of each encoder, all of weight 1.
pub fn empty_chunk() -> ImageChunk {
    let mut ids1 = vec![49407u32; CHUNK_IDS];
    let mut ids2 = vec![0u32; CHUNK_IDS];
    ids1[0] = 49406;
    ids2[0] = 49406;
    ids2[1] = 49407;
    ImageChunk {
        ids1,
        ids2,
        weights: vec![1.0; CHUNK_IDS],
    }
}

/// The token weights of A1111 on the hidden rows of one encoder for one chunk: each row times its weight, then the
/// whole chunk times the ratio of its mean before to its mean after. A chunk with all weights 1 stays as it is.
pub fn weight_chunk(rows: &mut [f32], weights: &[f32]) {
    if weights.iter().all(|w| *w == 1.0) {
        return;
    }
    let width = rows.len() / weights.len();
    let before = rows.iter().map(|v| *v as f64).sum::<f64>() / rows.len() as f64;
    for (t, w) in weights.iter().enumerate() {
        rows[t * width..(t + 1) * width].iter_mut().for_each(|v| *v *= w);
    }
    let after = rows.iter().map(|v| *v as f64).sum::<f64>() / rows.len() as f64;
    let ratio = (before / after) as f32;
    rows.iter_mut().for_each(|v| *v *= ratio);
}

/// The position of the end id in the ids of a chunk: the first 49407.
pub fn eos_position(ids: &[u32]) -> usize {
    ids.iter()
        .position(|id| *id == 49407)
        .expect("a chunk holds its end id")
}

/// The 8-bit RGB of a decoded image: round(clamp(v / 2 + 0.5, 0, 1) * 255), from pixel-major values.
pub fn rgb_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// The linear preview of a latent (spec 11.9): RGB in [0, 1] = `PREVIEW` times the 4 channels, plus `PREVIEW_BIAS`.
/// `tools/sdxl_preview_fit.py` fitted it by least squares on 4 images of SDXL base 1.0 at 512 by 512, against the mean
/// of each 8 by 8 block of the decoded image. Its mean absolute error was 0.072 (2026-10-28).
pub const PREVIEW: [[f32; 4]; 3] = [
    [0.2024, -0.0984, 0.0543, -0.1658],
    [0.2015, 0.0140, 0.0831, -0.1392],
    [0.2203, 0.0379, -0.0187, -0.1275],
];
pub const PREVIEW_BIAS: [f32; 3] = [0.5404, 0.5110, 0.4772];

/// The preview of a pixel-major latent of `w` by `h` rows, at one eighth of the size of the image.
pub fn preview(latent: &[f32], w: usize, h: usize) -> nagaya_abi::RgbImage {
    let rgb = latent
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| {
            (0..3).map(move |c| {
                let v = PREVIEW_BIAS[c] + (0..4).map(|k| PREVIEW[c][k] * px[k]).sum::<f32>();
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            })
        })
        .collect();
    nagaya_abi::RgbImage {
        width: w as u32,
        height: h as u32,
        rgb,
    }
}
