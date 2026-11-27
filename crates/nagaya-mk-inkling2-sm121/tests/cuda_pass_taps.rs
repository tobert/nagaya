//! GPU gate of the taps of the draft head (`docs/gb10-research.md`, D1; `Pass::open_with_taps`): the inputs of the layer
//! visits 5, 23 and 41 and the output of the final norm, for each row of a block.
//!
//! - `taps_move_no_bit_and_do_not_depend_on_the_split`: a pass with taps gives the logits and the slot of a pass without
//!   them, bit for bit, and the tap rows of a position are the same bits in a block of 17, 16 + 1, 9 + 8 and row by row
//!   (design rule 4). The chat prompt of the golden file is the input.
//! - `taps_follow_the_cpu_reference` (needs `MTP_TEXTS` and `MTP_CPU_FEATS`): the cosine of each tap against the features
//!   of the CPU reference, for the text `MTP_TEXT_NAME`. A report with a loose bound: the quantizer flips of the engine
//!   move a late row.
//! - `export_taps` (ignored; `MTP_TEXTS`, `MTP_NAMES`, `MTP_FEATS_OUT`): teacher-force the ids of texts through the pass in
//!   blocks of 512 rows and write `NAME.safetensors` with `feat [T, 4, 2304]` bf16, the input of
//!   `tools/inkling2_dspark_run.py`.
//!
//! `INKLING2_MODEL_DIR=<folder> INKLING2_GOLDEN_DIR=<folder> NAGAYA_TEST_CARDS=<pci address> cargo test --release -p
//!  nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_taps -- --include-ignored --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod common;

use std::path::PathBuf;

use nagaya_mk_inkling2_sm121::pass::step::{Pass, PassConfig, TapRows};
use nagaya_mk_inkling2_sm121::pass::weights::DeviceWeights;
use nagaya_mk_inkling2_sm121::spec::{self, HIDDEN};

const SEQ: usize = 2048;

fn model_dir() -> Option<PathBuf> {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    match std::env::var_os("INKLING2_MODEL_DIR") {
        Some(d) if PathBuf::from(&d).join(spec::WEIGHTS_FILE).exists() => Some(PathBuf::from(d)),
        _ => {
            assert!(
                !require,
                "INKLING2_MODEL_DIR has no checkpoint and NAGAYA_REQUIRE_MODELS is set"
            );
            eprintln!("cuda_pass_taps: skipped, INKLING2_MODEL_DIR has no checkpoint");
            None
        }
    }
}

fn chat_ids() -> Vec<u32> {
    let dir = std::env::var_os("INKLING2_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/inkling2-golden")));
    let path = dir.join("inkling2-mini.quant.oracle.json");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
    json["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "chat")
        .expect("the golden has the prompt chat")["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn open_taps<'d>(dev: &'d nagaya_mk_cuda::driver::CudaDevice, weights: &'d DeviceWeights<'d>, seq: usize) -> Pass<'d> {
    let config = PassConfig {
        seqs: vec![seq],
        max_rows: 512,
        max_cycles: 3,
        profile: false,
        format: nagaya_mk_cuda::attn::CacheFormat::F32,
    };
    Pass::open_with_taps(dev, weights, &config).unwrap_or_else(|e| panic!("{e}"))
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
        .collect()
}

/// `f32` to `bf16`, rounded to nearest even (the finite values of a trunk row).
fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn final_norm(weights: &DeviceWeights<'_>) -> Vec<f32> {
    let w = bf16_to_f32(&weights.norm_bf16.read(0, weights.norm_bf16.bytes()).unwrap());
    assert_eq!(w.len(), HIDDEN, "the final norm weight has {HIDDEN} values");
    w
}

/// The features of every row of `ids`, run in the blocks of `split`: `[T][4][HIDDEN]` as the bits of `f32`.
fn features(pass: &mut Pass<'_>, norm: &[f32], ids: &[u32], split: &[usize]) -> (Vec<u32>, Vec<u32>) {
    pass.slots[0].zero().unwrap();
    let mut at = 0;
    let mut feat = Vec::new();
    let mut logits = Vec::new();
    for &n in split {
        let last = at + n == ids.len();
        // Only the last call asks for a logit row. Each call gives its taps: the pass runs the output mix for them.
        let wanted: Vec<usize> = if last { vec![n - 1] } else { vec![] };
        let out = pass
            .run_slot(0, 3, &ids[at..at + n], &wanted)
            .unwrap_or_else(|e| panic!("{e}"));
        logits = out.iter().map(|v| v.to_bits()).collect();
        let taps: TapRows = pass.tap_rows(n).unwrap_or_else(|e| panic!("{e}"));
        for r in 0..n {
            for plane in taps.features(r, norm) {
                feat.extend(plane.iter().map(|v| v.to_bits()));
            }
        }
        at += n;
    }
    assert_eq!(at, ids.len());
    (feat, logits)
}

#[test]
fn taps_move_no_bit_and_do_not_depend_on_the_split() {
    let Some(dir) = model_dir() else { return };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let norm = final_norm(&weights);
    let mut pass = open_taps(&dev, &weights, 1024);
    pass.use_taps(false).unwrap();
    let ids = chat_ids();
    assert_eq!(ids.len(), 17);
    // A pass with no taps: the logits and the slot hashes that the taps must not move.
    pass.slots[0].zero().unwrap();
    let plain: Vec<u32> = pass.run(&ids, 1).unwrap().iter().map(|v| v.to_bits()).collect();
    let plain_slot = slot_hashes(&pass);
    pass.use_taps(true).unwrap();
    let (reference, logits) = features(&mut pass, &norm, &ids, &[17]);
    assert_eq!(
        logits, plain,
        "the logits of a pass with taps differ from a pass without"
    );
    assert_eq!(
        slot_hashes(&pass),
        plain_slot,
        "the slot after a pass with taps differs"
    );
    let floats = |bits: &[u32]| bits.iter().map(|b| f32::from_bits(*b)).collect::<Vec<f32>>();
    let all = floats(&reference);
    assert_eq!(all.len(), 17 * 4 * HIDDEN);
    assert!(all.iter().all(|v| v.is_finite()), "a tap is not finite");
    for tap in 0..4 {
        let rms: f64 = (0..17)
            .flat_map(|r| all[(r * 4 + tap) * HIDDEN..(r * 4 + tap + 1) * HIDDEN].iter())
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            / (17 * HIDDEN) as f64;
        eprintln!("tap {tap}: rms {:.3}", rms.sqrt());
        assert!(rms > 1e-6, "tap {tap} is zero: the copy did not run");
    }
    for split in [vec![16, 1], vec![9, 8], vec![1; 17], vec![5, 5, 4, 3]] {
        let (feat, logits) = features(&mut pass, &norm, &ids, &split);
        assert_eq!(
            feat, reference,
            "the taps of the split {split:?} differ from the block of 17 rows"
        );
        assert_eq!(logits, plain, "the logits of the split {split:?} differ");
        eprintln!("split {split:?}: taps and logits equal");
    }
}

fn slot_hashes(pass: &Pass<'_>) -> Vec<[u8; 32]> {
    let mut out = Vec::new();
    for run in &pass.slots[0].runs {
        for buf in [&run.k_plane, &run.v_plane, &run.conv] {
            out.push(*blake3::hash(&buf.read(0, buf.bytes()).unwrap()).as_bytes());
        }
    }
    out
}

fn texts() -> Vec<serde_json::Value> {
    let path = std::env::var("MTP_TEXTS").expect("MTP_TEXTS names the texts JSON of prompt_lookup_sim.py");
    serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap()
}

fn ids_of(text: &serde_json::Value) -> Vec<u32> {
    text["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

/// Teacher-force `ids` in blocks of 512 rows and give `[T][4][HIDDEN]` of `f32`.
fn run_text(pass: &mut Pass<'_>, norm: &[f32], ids: &[u32]) -> Vec<f32> {
    pass.slots[0].zero().unwrap();
    let mut feat = Vec::with_capacity(ids.len() * 4 * HIDDEN);
    for block in ids.chunks(512) {
        pass.run_slot(0, 3, block, &[]).unwrap_or_else(|e| panic!("{e}"));
        let taps = pass.tap_rows(block.len()).unwrap();
        for r in 0..block.len() {
            for plane in taps.features(r, norm) {
                feat.extend(plane);
            }
        }
    }
    feat
}

#[test]
fn taps_follow_the_cpu_reference() {
    let (Some(dir), Ok(feats), Ok(name)) = (
        model_dir(),
        std::env::var("MTP_CPU_FEATS"),
        std::env::var("MTP_TEXT_NAME"),
    ) else {
        eprintln!("taps_follow_the_cpu_reference: skipped, MTP_CPU_FEATS or MTP_TEXT_NAME is not set");
        return;
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let norm = final_norm(&weights);
    let mut pass = open_taps(&dev, &weights, SEQ);
    let text = texts()
        .into_iter()
        .find(|t| t["name"] == name.as_str())
        .expect("the text is in MTP_TEXTS");
    let ids = ids_of(&text);
    let ours = run_text(&mut pass, &norm, &ids);
    let bytes = std::fs::read(format!("{feats}/{name}.safetensors")).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let [lo, hi] = [0, 1].map(|i| header["feat"]["data_offsets"][i].as_u64().unwrap() as usize);
    let theirs = bf16_to_f32(&bytes[8 + header_len + lo..8 + header_len + hi]);
    assert_eq!(
        theirs.len(),
        ours.len(),
        "the CPU features have another shape than the text"
    );
    for tap in 0..4 {
        let mut cos_sum = 0.0;
        let mut cos_min: f64 = 1.0;
        let mut rel_sum = 0.0;
        for r in 0..ids.len() {
            let at = (r * 4 + tap) * HIDDEN;
            let (a, b) = (&ours[at..at + HIDDEN], &theirs[at..at + HIDDEN]);
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let diff: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = dot / (na * nb);
            cos_sum += cos;
            cos_min = cos_min.min(cos);
            rel_sum += diff / nb;
        }
        let n = ids.len() as f64;
        eprintln!(
            "{name} tap {tap}: mean cosine {:.5}, min cosine {:.5}, mean relative error {:.4}",
            cos_sum / n,
            cos_min,
            rel_sum / n
        );
        assert!(
            cos_sum / n > 0.98,
            "tap {tap} is far from the CPU reference: mean cosine {}",
            cos_sum / n
        );
    }
}

#[test]
#[ignore = "writes the features of the texts of MTP_NAMES to MTP_FEATS_OUT"]
fn export_taps() {
    let (Some(dir), Ok(out), Ok(names)) = (model_dir(), std::env::var("MTP_FEATS_OUT"), std::env::var("MTP_NAMES"))
    else {
        eprintln!("export_taps: skipped, MTP_FEATS_OUT or MTP_NAMES is not set");
        return;
    };
    let dev = common::gpu::open();
    let weights = DeviceWeights::load(&dev, &dir, |_| {}).unwrap_or_else(|e| panic!("{e}"));
    let norm = final_norm(&weights);
    let mut pass = open_taps(&dev, &weights, SEQ);
    let all = texts();
    for name in names.split(',') {
        let text = all
            .iter()
            .find(|t| t["name"] == name)
            .unwrap_or_else(|| panic!("no text {name} in MTP_TEXTS"));
        let ids = ids_of(text);
        assert!(
            ids.len() <= SEQ,
            "{name} has {} ids, and the slot holds {SEQ}",
            ids.len()
        );
        let started = std::time::Instant::now();
        let feat = run_text(&mut pass, &norm, &ids);
        let data: Vec<u8> = feat.iter().flat_map(|v| to_bf16(*v).to_le_bytes()).collect();
        let header = serde_json::json!({
            "feat": {"dtype": "BF16", "shape": [ids.len(), 4, HIDDEN], "data_offsets": [0, data.len()]}
        })
        .to_string();
        let pad = (8 - header.len() % 8) % 8;
        let mut file = Vec::new();
        file.extend_from_slice(&((header.len() + pad) as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend(std::iter::repeat_n(b' ', pad));
        file.extend_from_slice(&data);
        std::fs::write(format!("{out}/{name}.safetensors"), file).unwrap();
        eprintln!(
            "{name}: {} positions in {:.1} s",
            ids.len(),
            started.elapsed().as_secs_f64()
        );
    }
}
