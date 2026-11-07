//! The sampler of the audio codes of Omni (spec 11.11, "The sampler of the audio codes"; engine-abi "The speech
//! calls of LFM3"). It runs on the host, and it needs no card.
//!
//! The depth decoder reveals the 12 codebooks of a frame in three passes of 4 (`maskgit_schedule`). The order of the
//! codebooks and each draw come from the seed and the frame, so the sampler keeps no state (spec 11.2). A code takes
//! the repetition penalty of its codebook over the last 64 frames, then the temperature, then the draw
//! (modeling_lfm3_omni.py:1037-1100, generation_config.json).

use std::cmp::Ordering;

/// The codebooks of a frame (`num_codebooks`).
pub const CODEBOOKS: usize = 12;
/// The codes of each codebook (`codebook_size`), with the end code last.
pub const CODES: usize = 2017;
/// The end code (`audio_eos_code`). A frame in which any codebook gives it ends the audio.
pub const END_CODE: u32 = 2016;
/// The codebooks that each pass of the depth decoder reveals (`maskgit_schedule`).
pub const SCHEDULE: [usize; 3] = [4, 4, 4];
/// The temperature of the audio codes (`audio_temperature`).
pub const TEMPERATURE: f32 = 0.4;
/// The repetition penalty of the audio codes (`audio_repetition_penalty`).
pub const PENALTY: f32 = 1.2;
/// The frames that the penalty covers (`audio_repetition_window`).
pub const WINDOW: usize = 64;

const DRAW_DOMAIN: &[u8] = b"nagaya audio draw v1";

/// The uniform value in [0, 1) of the draw `index` of a frame: the BLAKE3 hash of the domain, the seed, the frame and
/// the index, as the draw of spec 11.2 makes it.
pub fn uniform(seed: u64, frame: u64, index: u8) -> f64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DRAW_DOMAIN);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&frame.to_le_bytes());
    hasher.update(&[index]);
    let hash = hasher.finalize();
    let first = u64::from_le_bytes(hash.as_bytes()[..8].try_into().expect("a hash has 32 bytes"));
    (first >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// The order in which the passes reveal the codebooks: a Fisher-Yates shuffle of 0 to 11. The swap at `i` takes the
/// draw `i`.
pub fn order(seed: u64, frame: u64) -> [usize; CODEBOOKS] {
    let mut out: [usize; CODEBOOKS] = std::array::from_fn(|g| g);
    for i in (1..CODEBOOKS).rev() {
        let j = (uniform(seed, frame, i as u8) * (i + 1) as f64) as usize;
        out.swap(i, j.min(i));
    }
    out
}

/// The draw of the code of codebook `g`.
pub fn code_draw(g: usize) -> u8 {
    (CODEBOOKS + g) as u8
}

/// Apply the penalty to the logits of codebook `g` for each code that it gave in the last `WINDOW` frames of
/// `history`. A code that repeats takes the penalty once, as `RepetitionPenaltyLogitsProcessor` does.
pub fn penalize(logits: &mut [f32], history: &[u32], g: usize) {
    let frames = history.len() / CODEBOOKS;
    let mut seen = vec![false; logits.len()];
    for f in frames.saturating_sub(WINDOW)..frames {
        let code = history[f * CODEBOOKS + g] as usize;
        if code < logits.len() && !seen[code] {
            seen[code] = true;
            let v = logits[code];
            logits[code] = if v < 0.0 { v * PENALTY } else { v / PENALTY };
        }
    }
}

/// The code of the uniform value `u` from penalized logits: the temperature, the softmax in f64, and the inverse CDF
/// over the codes in order of logit down and code up (spec 11.2).
pub fn pick(logits: &[f32], u: f64) -> u32 {
    assert!(!logits.is_empty(), "a codebook has codes");
    assert!((0.0..1.0).contains(&u), "the uniform value {u} is not in [0, 1)");
    let z: Vec<f64> = logits.iter().map(|v| *v as f64 / TEMPERATURE as f64).collect();
    if let Some((code, v)) = z.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        panic!("the logit of code {code} is {v}: the depth decoder gave a value that is not finite");
    }
    let mut ids: Vec<u32> = (0..z.len() as u32).collect();
    ids.sort_unstable_by(|a, b| {
        z[*b as usize]
            .partial_cmp(&z[*a as usize])
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(b))
    });
    let most = z[ids[0] as usize];
    let mut run = 0.0f64;
    let cdf: Vec<f64> = ids
        .iter()
        .map(|id| {
            run += (z[*id as usize] - most).exp();
            run
        })
        .collect();
    let target = u * run;
    ids[cdf.partition_point(|c| *c <= target).min(ids.len() - 1)]
}

/// The code of codebook `g` in frame `frame`, from its logits and the codes of the frames before it.
pub fn sample(logits: &[f32], history: &[u32], g: usize, seed: u64, frame: u64) -> u32 {
    let mut penalized = logits.to_vec();
    penalize(&mut penalized, history, g);
    pick(&penalized, uniform(seed, frame, code_draw(g)))
}

/// True when a frame ends the audio: a codebook gave the end code. The frame that the engine feeds is then the end
/// code in each codebook (modeling_lfm3_omni.py:1734-1737).
pub fn ends(codes: &[u32]) -> bool {
    codes.contains(&END_CODE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(seed: u64) -> Vec<f32> {
        (0..CODES)
            .map(|i| {
                let x = (i as u64 ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
                (x as f32 / 16_777_216.0 - 0.5) * 8.0
            })
            .collect()
    }

    #[test]
    fn the_order_is_a_permutation_that_the_seed_and_the_frame_fix() {
        let a = order(7, 3);
        let mut sorted = a;
        sorted.sort_unstable();
        assert_eq!(sorted, std::array::from_fn(|g| g));
        assert_eq!(a, order(7, 3), "the same seed and frame give the same order");
        assert_ne!(a, order(7, 4), "another frame gives another order");
        assert_ne!(a, order(8, 3), "another seed gives another order");
        // The order of seed 1234 and frame 0, pinned from the first run: a change of the draw changes it.
        assert_eq!(order(1234, 0), [9, 4, 1, 0, 7, 8, 10, 5, 11, 2, 6, 3]);
    }

    #[test]
    fn the_draw_hashes_the_seed_the_frame_and_the_index() {
        let u = uniform(1234, 0, 12);
        assert!((0.0..1.0).contains(&u));
        assert_eq!(u, uniform(1234, 0, 12));
        assert_ne!(u, uniform(1234, 0, 13));
        assert_ne!(u, uniform(1234, 1, 12));
        assert_ne!(u, uniform(1235, 0, 12));
        assert_eq!(code_draw(0), 12);
        assert_eq!(code_draw(11), 23);
    }

    #[test]
    fn the_penalty_divides_a_positive_logit_and_multiplies_a_negative_one_once() {
        let mut row = vec![2.4f32, -1.0, 0.5, 3.0];
        let history = [vec![0u32; CODEBOOKS], vec![0u32; CODEBOOKS], vec![1u32; CODEBOOKS]].concat();
        penalize(&mut row, &history, 5);
        assert_eq!(row, vec![2.4 / 1.2, -1.2, 0.5, 3.0]);
    }

    #[test]
    fn the_penalty_covers_only_the_last_64_frames_of_its_codebook() {
        let mut history = Vec::new();
        for f in 0..65u32 {
            let mut frame = vec![9u32; CODEBOOKS];
            frame[3] = if f == 0 { 1 } else { 2 };
            history.extend(frame);
        }
        let mut row = vec![1.0f32; 10];
        penalize(&mut row, &history, 3);
        assert_eq!(row[1], 1.0, "frame 0 is 65 frames back");
        assert_eq!(row[2], 1.0 / 1.2);
        assert_eq!(row[9], 1.0, "code 9 is in another codebook");
    }

    #[test]
    fn the_pick_follows_the_inverse_cdf_of_the_tempered_softmax() {
        // Two codes with logits 0 and 0.4 ln 3: the tempered weights are 1 and 3, so code 1 takes [0, 0.75).
        let row = vec![0.0f32, (0.4f64 * 3.0f64.ln()) as f32];
        assert_eq!(pick(&row, 0.0), 1);
        assert_eq!(pick(&row, 0.7499), 1);
        assert_eq!(pick(&row, 0.7501), 0);
        assert_eq!(pick(&row, 0.9999), 0);
        // A tie keeps the lower code first.
        assert_eq!(pick(&[1.0, 1.0], 0.25), 0);
        assert_eq!(pick(&[1.0, 1.0], 0.75), 1);
    }

    #[test]
    fn the_same_seed_and_frame_give_the_same_codes_on_each_run() {
        let history: Vec<u32> = (0..10 * CODEBOOKS as u32).map(|i| i % 50).collect();
        let row = logits(3);
        let first: Vec<u32> = (0..CODEBOOKS).map(|g| sample(&row, &history, g, 99, 10)).collect();
        for _ in 0..3 {
            let again: Vec<u32> = (0..CODEBOOKS).map(|g| sample(&row, &history, g, 99, 10)).collect();
            assert_eq!(again, first);
        }
        let other: Vec<u32> = (0..CODEBOOKS).map(|g| sample(&row, &history, g, 99, 11)).collect();
        assert_ne!(other, first, "another frame draws other values");
        assert!(first.iter().all(|c| (*c as usize) < CODES));
    }

    #[test]
    fn a_frame_with_one_end_code_ends_the_audio() {
        let mut codes = vec![5u32; CODEBOOKS];
        assert!(!ends(&codes));
        codes[7] = END_CODE;
        assert!(ends(&codes));
    }
}
