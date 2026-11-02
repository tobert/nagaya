//! The random stream of the engine (spec 11.9, engine-abi "Image"). Each value is a function of the seed, the image,
//! the step, the purpose and its index, so the image of seed `s` does not depend on the other images of its slot. A
//! seed from another tool does not give that tool's image (Decided 2026-10-28).

/// What a draw is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    /// The start latent.
    Start,
    /// The noise of a step of an ancestral or SDE sampler.
    Step,
    /// The draw of the image latent from the VAE encoder (spec 11.9b, "The image latent").
    Encode,
    /// The draw of the masked image latent of a 9-channel UNet (spec 11.9b, "The 9-channel UNet").
    EncodeMasked,
}

/// SplitMix64's finalizer.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform value in (0, 1] from the key of one draw.
fn uniform(key: [u64; 5], half: u64) -> f64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for part in key.iter().chain(std::iter::once(&half)) {
        h = mix(h ^ part);
    }
    ((h >> 11) as f64 + 1.0) / (1u64 << 53) as f64
}

/// `count` values of a standard normal distribution for one image, step and purpose, by Box-Muller in f64.
pub fn normal(seed: u64, image: u32, step: u32, purpose: Purpose, count: usize) -> Vec<f32> {
    let purpose = match purpose {
        Purpose::Start => 1u64,
        Purpose::Step => 2,
        Purpose::Encode => 3,
        Purpose::EncodeMasked => 4,
    };
    (0..count)
        .map(|i| {
            let key = [seed, image as u64, step as u64, purpose, i as u64];
            let (u1, u2) = (uniform(key, 0), uniform(key, 1));
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}
