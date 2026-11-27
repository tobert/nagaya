//! The paste after the decode against the oracle (`tools/sdxl_inpaint_golden.sh`, spec 11.9b, "The paste after the
//! decode"). The route pastes with `nagaya_family_sdxl::inputs::paste`. This test gives it the decode of each inpainting
//! run of the reference, with the blend and with the 9-channel UNet, and the input image. It compares the bytes with
//! the reference's paste.

mod common;

use common::golden;
use nagaya_family_sdxl::inputs::paste;

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
    let (Some(inputs), Some(four), Some(nine)) = (
        golden("image_inputs.safetensors"),
        golden("inpaint.safetensors"),
        golden("inpaint9.safetensors"),
    ) else {
        return;
    };
    let old = bytes(inputs.values("init.rgb"));
    let blurred = bytes(inputs.values("mask.blurred"));
    assert!(
        blurred.iter().any(|a| *a != 0 && *a != 255),
        "the blurred mask of the oracle has a soft edge"
    );
    // The paste applies to a 9-channel UNet as to the blend (spec 11.9b), and inpaint9 has the runs of inpaint.
    for (t, run) in RUNS.iter().flat_map(|run| [(&four, *run), (&nine, *run)]) {
        let new = bytes(t.values(&format!("{run}.rgb")));
        let want = bytes(t.values(&format!("{run}.paste")));
        assert!(
            paste(&new, &old, &blurred) == want,
            "{run}: the paste differs from the reference"
        );
        assert!(new != want, "{run}: the paste changes the decode");
    }
}
