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
use common::cpu::CpuCard;
use nagaya_mk_bmg::safetensors::MappedSafeTensors;
use nagaya_mk_lfm3_bmg::audio::{self, AudioBufs, Clip, Encoder, MAX_CLIP_POSITIONS, Stage};
use nagaya_mk_lfm3_bmg::weights::Upload;

/// The bound of a defect of the CPU form: a relative error of a stage past this is a wrong op.
const DEFECT_BOUND: f64 = 1e-3;

/// The oracle clip, the encoder and its weights in host memory.
struct Setup {
    cpu: CpuCard,
    e: Encoder,
    w: audio::AudioWeights,
    bufs: AudioBufs,
    golden: std::collections::HashMap<String, common::Tensor>,
}

fn setup() -> Option<Setup> {
    if common::absent(Which::Omni) || common::golden_absent("audio_in.safetensors") {
        return None;
    }
    let dir = common::model_dir(Which::Omni);
    let e = Encoder::read(&dir).unwrap().expect("Omni has an audio encoder");
    let map = MappedSafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let mut cpu = CpuCard::new();
    let w = audio::load(&mut cpu, &e, &map.view().unwrap()).unwrap();
    let bufs = AudioBufs::new(&mut |bytes| cpu.alloc(bytes), &e, MAX_CLIP_POSITIONS);
    let golden = common::golden_tensors("audio_in.safetensors");
    Some(Setup {
        cpu,
        e,
        w,
        bufs,
        golden,
    })
}

/// The mel rows `[frames][bins]` of the oracle, which holds the bins first.
fn mel_rows(golden: &common::Tensor) -> (Vec<f32>, usize) {
    let (bins, frames) = (golden.shape[0], golden.shape[1]);
    let v = golden.f32();
    let rows = (0..frames * bins).map(|i| v[(i % bins) * frames + i / bins]).collect();
    (rows, frames)
}

/// Run the tasks of `mel` in three parts, and give the rows after each stage.
fn stages(s: &mut Setup, mel: &[f32], frames: usize) -> [Vec<f32>; 3] {
    let p = audio::positions(frames);
    let out = s.cpu.alloc(p * s.e.out * 4);
    s.cpu.put(s.bufs.mel, mel);
    let clip = Clip {
        mel: s.bufs.mel,
        frames,
        out,
    };
    let all = audio::encode(&s.e, &s.w, &s.bufs, clip, Stage::Rows);
    let down = audio::encode(&s.e, &s.w, &s.bufs, clip, Stage::Downsample).len();
    let enc = audio::encode(&s.e, &s.w, &s.bufs, clip, Stage::Encoder).len();
    s.cpu.run(&all[..down]);
    let a = s.cpu.f32s(s.bufs.x, p * s.e.hidden);
    s.cpu.run(&all[down..enc]);
    let b = s.cpu.f32s(s.bufs.h, p * s.e.hidden);
    s.cpu.run(&all[enc..]);
    let c = s.cpu.f32s(out, p * s.e.out);
    [a, b, c]
}

const NAMES: [&str; 3] = ["downsample", "encoder", "rows"];

#[test]
fn each_stage_of_the_cpu_form_follows_transformers() {
    let Some(mut s) = setup() else { return };
    let (mel, frames) = mel_rows(&s.golden["mel"]);
    assert_eq!(frames, 149);
    let got = stages(&mut s, &mel, frames);
    for (name, got) in NAMES.iter().zip(&got) {
        let want = s.golden[*name].f32();
        let gap = common::rel(got, &want);
        println!(
            "CPU form, {name}: relative error {gap:.3e}, largest difference {:.3e}",
            common::max_abs(got, &want)
        );
        assert!(gap < DEFECT_BOUND, "{name}: relative error {gap}");
    }
}

/// The family computes the mel rows from the WAV in f64 (spec 11.11, "Mel rows"), so its quiet bins differ from the
/// reference. This test prints how far that moves the rows of the projector.
#[test]
fn the_mel_rows_of_the_family_give_the_rows_of_transformers() {
    let Some(mut s) = setup() else { return };
    let samples = s.golden["samples"].f32();
    let rows = nagaya_text_lfm3::media::audio::mel(&samples);
    let (reference, frames) = mel_rows(&s.golden["mel"]);
    assert_eq!(rows.frames, frames);
    println!(
        "family mel: relative error {:.3e}, largest difference {:.3e}",
        common::rel(&rows.values, &reference),
        common::max_abs(&rows.values, &reference)
    );
    let got = stages(&mut s, &rows.values, rows.frames);
    for (name, got) in NAMES.iter().zip(&got) {
        let want = s.golden[*name].f32();
        let gap = common::rel(got, &want);
        println!("family mel, {name}: relative error {gap:.3e}");
        assert!(
            gap < DEFECT_BOUND,
            "{name} from the mel rows of the family: relative error {gap}"
        );
    }
}
