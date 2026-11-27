//! The CPU form of the tasks of the audio encoder against transformers in f32 on the CPU (spec 11.11, "The LFM3
//! oracle"): the rows after the downsampler, after the layers and after the projector, for the oracle clip of 149
//! frames. The form runs the tasks of `audio::encode` on host memory (`common/cpu.rs`), so it checks the graph, the
//! addresses, the plane and the order of `lin_out` with no card.
//!
//! The form sums in f64 and the reference in f32, so the gaps are the noise of f32. The test prints each gap. Its
//! bound only catches a wrong op. A conv state that is not zeroed moved the encoder rows by 1.3e-2, and the other
//! wrong ops that were tried moved a stage by 9e-2 or more.
//!
//! The oracle clip has 19 positions and the window of the encoder is 32 keys, so this oracle does not test the
//! window. The card test of a longer clip compares the card with this form.

mod common;

use common::Which;
use common::cpu::{CpuEncoder, STAGES, mel_rows};

/// The bound of a defect of the CPU form: a relative error of a stage past this is a wrong op.
const DEFECT_BOUND: f64 = 1e-3;

fn setup() -> Option<(CpuEncoder, std::collections::HashMap<String, common::Tensor>)> {
    if common::absent(Which::Omni) || common::golden_absent("audio_in.safetensors") {
        return None;
    }
    let encoder = CpuEncoder::load(&common::model_dir(Which::Omni));
    Some((encoder, common::golden_tensors("audio_in.safetensors")))
}

#[test]
fn each_stage_of_the_cpu_form_follows_transformers() {
    let Some((mut encoder, golden)) = setup() else { return };
    let mel = mel_rows(&golden["mel"]);
    assert_eq!(mel.len(), 149 * 128);
    let got = encoder.stages(&mel);
    for (name, got) in STAGES.iter().zip(&got) {
        let want = golden[*name].f32();
        let gap = common::rel(got, &want);
        println!(
            "CPU form, {name}: relative error {gap:.3e}, largest difference {:.3e}",
            common::max_abs(got, &want)
        );
        assert!(gap < DEFECT_BOUND, "{name}: relative error {gap}");
    }
}

/// The family computes the mel rows from the WAV in f64 (spec 11.11, "Mel rows"), so its quiet bins differ from the
/// reference. This test prints how far that moves each stage.
#[test]
fn the_mel_rows_of_the_family_give_the_rows_of_transformers() {
    let Some((mut encoder, golden)) = setup() else { return };
    let rows = nagaya_family_lfm3::media::audio::mel(&golden["samples"].f32());
    let reference = mel_rows(&golden["mel"]);
    assert_eq!(rows.values.len(), reference.len());
    println!(
        "family mel: relative error {:.3e}, largest difference {:.3e}",
        common::rel(&rows.values, &reference),
        common::max_abs(&rows.values, &reference)
    );
    let got = encoder.stages(&rows.values);
    for (name, got) in STAGES.iter().zip(&got) {
        let want = golden[*name].f32();
        let gap = common::rel(got, &want);
        println!("family mel, {name}: relative error {gap:.3e}");
        assert!(
            gap < DEFECT_BOUND,
            "{name} from the mel rows of the family: relative error {gap}"
        );
    }
}
