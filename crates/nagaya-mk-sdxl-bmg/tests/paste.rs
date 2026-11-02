//! The paste after the decode against the oracle (`tools/sdxl_inpaint_golden.sh`, spec 11.9b, "The paste after the
//! decode"). The route pastes with `nagaya_text_sdxl::inputs::paste`. This test gives it the decode of each inpainting
//! run of the reference and the input image, and it compares the bytes with the reference's paste.

mod common;

use common::golden;
use nagaya_text_sdxl::inputs::paste;

/// The runs of `inpaint.safetensors`: the sampler and the strength in percent.
const RUNS: [&str; 6] = [
    "euler.s99",
    "euler_a.s99",
    "dpmpp_2m_karras.s99",
    "dpmpp_2m_sde_karras.s99",
    "euler.s100",
    "dpmpp_2m_karras.s100",
];

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().map(|v| *v as u8).collect()
}

#[test]
fn the_paste_gives_the_bytes_of_the_reference() {
    let (Some(inputs), Some(t)) = (golden("image_inputs.safetensors"), golden("inpaint.safetensors")) else {
        return;
    };
    let old = bytes(inputs.values("init.rgb"));
    let blurred = bytes(inputs.values("mask.blurred"));
    assert!(
        blurred.iter().any(|a| *a != 0 && *a != 255),
        "the blurred mask of the oracle has a soft edge"
    );
    for run in RUNS {
        let new = bytes(t.values(&format!("{run}.rgb")));
        let want = bytes(t.values(&format!("{run}.paste")));
        assert!(
            paste(&new, &old, &blurred) == want,
            "{run}: the paste differs from the reference"
        );
        assert!(new != want, "{run}: the paste changes the decode");
    }
}
