//! The random stream of the engine is the SDXL engine's, value for value (spec 11.9c, "The steps").

use nagaya_mk_qwenimage21_gfx1151::noise::{self, Purpose};

#[test]
fn the_stream_is_the_stream_of_the_sdxl_engine() {
    for (seed, image, step) in [(0u64, 0u32, 0u32), (1234, 0, 0), (u64::MAX, 3, 7)] {
        for (ours, theirs) in [
            (Purpose::Start, nagaya_mk_sdxl_bmg::noise::Purpose::Start),
            (Purpose::Step, nagaya_mk_sdxl_bmg::noise::Purpose::Step),
        ] {
            let a = noise::normal(seed, image, step, ours, 4096);
            let b = nagaya_mk_sdxl_bmg::noise::normal(seed, image, step, theirs, 4096);
            assert_eq!(a, b, "seed {seed} image {image} step {step}");
        }
    }
    let draws = noise::normal(1234, 0, 0, Purpose::Start, 100_000);
    let mean = draws.iter().map(|&x| f64::from(x)).sum::<f64>() / draws.len() as f64;
    let var = draws.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / draws.len() as f64;
    assert!(
        mean.abs() < 0.02 && (var - 1.0).abs() < 0.02,
        "a standard normal: mean {mean}, variance {var}"
    );
}
