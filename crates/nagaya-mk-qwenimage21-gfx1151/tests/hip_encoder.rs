//! The conditioning gate of Qwen-Image 2.1 (spec 8.3, "The Qwen-Image oracle"): the encoder on the ids of the oracle
//! against `prompt_embeds` of diffusers in f32. The gap of a row is 1 minus its cosine. Each row's gap must be at most
//! 4 times the gap of the same encoder in bf16 under torch (`prompt_embeds_bf16`), with a floor of 1e-5. Stop
//! `nagaya.service` first (spec 3.1).
//!
//! The test reads FUSAL's directory from `QWENIMAGE21_FUSAL_DIR`, and the oracle from `QWENIMAGE21_GOLDEN_DIR` or
//! `target/qwenimage21-golden` (`tools/qwenimage21_golden.sh`). An absent file skips the test, and
//! `NAGAYA_REQUIRE_MODELS=1` makes it fail. Run:
//! `cargo test -p nagaya-mk-qwenimage21-gfx1151 --features hip --test hip_encoder -- --nocapture`.
#![cfg(feature = "hip")]

mod common;

/// The floor of a bound (spec 8.3, "The bf16 baseline of each gate").
const FLOOR: f64 = 1e-5;

use nagaya_mk_hip::attention::Attention;
use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{Gemms, lineage};
use nagaya_mk_hip::hip::Stream;
use nagaya_mk_hip::plan::PlanTable;
use nagaya_mk_qwenimage21_gfx1151::encoder::Encoder;
use nagaya_mk_qwenimage21_gfx1151::names::HIDDEN;
use nagaya_mk_qwenimage21_gfx1151::safetensors::{Dtype, Safetensors};

fn floats(st: &Safetensors, name: &str, shape: &[usize]) -> Vec<f64> {
    st.read(name, Dtype::F32, shape)
        .unwrap()
        .chunks(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (a.iter().map(|x| x * x).sum::<f64>().sqrt() * b.iter().map(|x| x * x).sum::<f64>().sqrt())
}

#[test]
fn the_encoder_gives_the_conditioning_of_the_oracle() {
    let Some(oracle) = common::oracle() else { return };
    let Some(fusal) = common::fusal_dir() else { return };
    let encode = Safetensors::open(&oracle.join("encode.safetensors")).unwrap();
    let ids_shape = encode.info("ids").unwrap().shape.clone();
    let ids: Vec<u32> = encode
        .read("ids", Dtype::I64, &ids_shape)
        .unwrap()
        .chunks(8)
        .map(|b| u32::try_from(i64::from_le_bytes(b.try_into().unwrap())).unwrap())
        .collect();
    let drop = encode.read("drop", Dtype::I64, &[1]).unwrap();
    let cut = i64::from_le_bytes(drop[..8].try_into().unwrap()) as usize;
    let rows = ids.len() - cut;
    let want = floats(&encode, "prompt_embeds", &[1, rows, HIDDEN]);
    let baseline = floats(&encode, "prompt_embeds_bf16", &[1, rows, HIDDEN]);
    let request: serde_json::Value = serde_json::from_str(&encode.metadata()["request"]).unwrap();
    let path = fusal.join(request["encoder"].as_str().unwrap());
    let started = std::time::Instant::now();
    let file = Safetensors::open(&path).unwrap();
    let mut encoder = Encoder::load(&file).unwrap();
    eprintln!(
        "loaded {:.2} GiB in {:.1} s",
        encoder.bytes() as f64 / (1u64 << 30) as f64,
        started.elapsed().as_secs_f64()
    );
    let stream = Stream::new().unwrap();
    let mut gemms = Gemms::new(PlanTable::in_memory(&lineage().unwrap()), 32 << 20).unwrap();
    let attention = Attention::new(Encoder::attention_capacity(ids.len())).unwrap();
    let mut worst_ratio = 0.0f64;
    let mut failed = Vec::new();
    // Both paths of the attention: the GEMMs with the softmax kernel, and the fused flash kernel.
    for flash in [false, true] {
        encoder.flash = flash;
        eprintln!("the attention of the flash kernel: {flash}");
        let started = std::time::Instant::now();
        let out = encoder.encode(&mut gemms, &attention, &ids, &stream).unwrap();
        eprintln!(
            "first encode of {} ids in {:.0} ms, with the picks",
            ids.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
        let started = std::time::Instant::now();
        let again = encoder.encode(&mut gemms, &attention, &ids, &stream).unwrap();
        eprintln!("second encode in {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
        let got: Vec<u16> = out.to_vec::<u16>().unwrap();
        assert_eq!(got, again.to_vec::<u16>().unwrap(), "two encodes give the same bits");
        for row in 0..rows {
            let g: Vec<f64> = got[(cut + row) * HIDDEN..(cut + row + 1) * HIDDEN]
                .iter()
                .map(|&h| f64::from(bf16::to_f32(h)))
                .collect();
            let w = &want[row * HIDDEN..(row + 1) * HIDDEN];
            let b = &baseline[row * HIDDEN..(row + 1) * HIDDEN];
            let (gap, torch_gap) = (1.0 - cosine(&g, w), 1.0 - cosine(b, w));
            let bound = (4.0 * torch_gap).max(FLOOR);
            eprintln!("row {row:>2}: gap {gap:.2e}, torch bf16 gap {torch_gap:.2e}, bound {bound:.2e}");
            worst_ratio = worst_ratio.max(gap / torch_gap.max(1e-12));
            if gap > bound {
                failed.push((flash, row));
            }
        }
    }
    eprintln!("conditioning gate: the worst row has {worst_ratio:.2} times the gap of torch in bf16");
    assert!(failed.is_empty(), "the conditioning gate failed at the rows {failed:?}");
}
