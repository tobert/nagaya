//! The weights of the vision tower in an upload sink, with no card: each tensor of `model.visual.` once, in the
//! dtype that its op reads. The test skips when the model is absent.

mod common;

use common::Which;
use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors};
use nagaya_mk_inkling2_bmg::vision;
use nagaya_mk_inkling2_bmg::weights::{Count, Upload};

/// A sink that keeps the bytes of each allocation, so the test can read back what the load wrote.
#[derive(Default)]
struct Keep(Vec<Vec<u8>>);

impl Upload for Keep {
    fn alloc(&mut self, bytes: usize) -> u64 {
        self.0.push(vec![0; bytes]);
        self.0.len() as u64 - 1
    }

    fn write(&mut self, at: u64, data: &[u8]) {
        self.0[at as usize].copy_from_slice(data);
    }

    fn writes(&self) -> bool {
        true
    }
}

#[test]
fn the_vision_weights_hold_each_tensor_once_and_widen_the_norms_and_the_scales() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let map = MappedSafeTensors::open(&common::model_weights(Which::Mini)).unwrap();
    let view = map.view().unwrap();
    let names: Vec<&String> = view.tensors.keys().filter(|n| n.starts_with("model.visual.")).collect();
    assert_eq!(names.len(), 422);
    // The list of the plan is the list of the checkpoint, so the load reads only tensors that the plan checked.
    let mut listed: Vec<String> = vision::tensors().into_iter().map(|(name, _, _)| name).collect();
    let mut held: Vec<String> = names.iter().map(|n| n.to_string()).collect();
    listed.sort();
    held.sort();
    assert_eq!(listed, held);
    assert_eq!(vision::check_tensors(&view.tensors), Ok(()));
    // The vectors that the LayerNorms and the layer scales read go up as f32; the rest stays bf16.
    let widened = |n: &str| n.contains(".ln_") || n.contains(".ls_");
    let mut want = 0usize;
    for n in &names {
        let meta = &view.tensors[*n];
        assert_eq!(meta.dtype, Dtype::BF16, "{n}");
        let bytes = meta.data_end - meta.data_start;
        // The two class vectors go up as one f32 row.
        if n.ends_with("class_embedding") || n.ends_with("cls_position_embedding") {
            continue;
        }
        want += if widened(n) { 2 * bytes } else { bytes }.max(64).next_multiple_of(64);
    }
    want += 1024 * 4 + 2 * 207 * 16 * 4;
    let mut count = Count::default();
    let w = vision::load(&mut count, &view).unwrap();
    eprintln!("the vision weights take {} bytes on the card", count.bytes);
    assert_eq!(count.bytes, want);
    assert_eq!(count.bytes, vision::WEIGHTS_BYTES);
    assert_eq!(w.layers.len(), 23);

    let mut keep = Keep::default();
    let w = vision::load(&mut keep, &view).unwrap();
    let f32s = |at: u64| -> Vec<f32> {
        keep.0[at as usize]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    let bf16s = |name: &str| -> Vec<f32> {
        view.tensor_bytes(name)
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect()
    };
    let e = "model.visual.backbone.embeddings";
    let class: Vec<f32> = bf16s(&format!("{e}.class_embedding"))
        .iter()
        .zip(bf16s(&format!("{e}.cls_position_embedding")))
        .map(|(a, b)| a + b)
        .collect();
    assert_eq!(f32s(w.class_row), class);
    assert_eq!(
        f32s(w.layers[7].ls_2),
        bf16s("model.visual.backbone.layers.7.ls_2.gamma")
    );
    assert_eq!(
        f32s(w.layers[22].ln_1.bias),
        bf16s("model.visual.backbone.layers.22.ln_1.bias")
    );
    assert_eq!(
        keep.0[w.layers[3].fc.weight as usize],
        view.tensor_bytes("model.visual.backbone.layers.3.mlp.c_fc.weight")
            .unwrap()
    );
    assert_eq!(f32s(w.rope), vision::rope_table());
}
