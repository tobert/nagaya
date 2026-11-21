//! The plan of the engine from the real model folders: `config.json` and the header of the checkpoint, with no
//! weight bytes (spec 2.4).

mod common;

use common::Which;
use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors};
use nagaya_mk_inkling2_bmg::spec::{self, Config, Model};
use std::path::PathBuf;

fn folder(which: Which) -> PathBuf {
    common::model_weights(which).parent().unwrap().to_path_buf()
}

/// Each model reads as itself, and the weights of the two take the same bytes: they share every tensor.
#[test]
fn the_folders_give_their_model_and_the_bytes_of_the_weights() {
    for (which, model) in [(Which::Mini, Model::Mini), (Which::Varloop, Model::Research)] {
        if common::model_absent(which) {
            continue;
        }
        let dir = folder(which);
        assert_eq!(Model::read(&dir), Ok(model));
        // 15,670,464,512 bytes of expert slabs and 3,210,903,552 bytes of the other tensors, with the rows of zeros of
        // the head and of the sink projections.
        assert_eq!(spec::weights_bytes(&dir), Ok(18_881_368_064));
        let config = Config {
            model_dir: dir,
            slots: vec![32_768; 3],
            max_block: spec::DEFAULT_MAX_BLOCK,
            extra_runs: spec::DEFAULT_EXTRA_RUNS,
        };
        let limits = spec::plan(&config).unwrap();
        limits.check().unwrap();
        let weights = limits.memory.mandatory.iter().find(|t| t.name == "weights").unwrap();
        assert_eq!(weights.bytes, 18_881_368_064);
        let bad = Config {
            extra_runs: 12,
            ..config
        };
        assert!(spec::plan(&bad).unwrap_err().contains("extra_runs is 12"));
    }
}

/// A missing tensor, a tensor of another dtype and a tensor that the engine does not know are refused by name.
#[test]
fn a_header_with_another_tensor_is_refused() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let map = MappedSafeTensors::open(&common::model_weights(Which::Mini)).unwrap();
    let tensors = map.view().unwrap().tensors;
    assert_eq!(spec::weights_bytes_of(&tensors), Ok(18_881_368_064));
    let amax = "model.llm.layers.3.mlp.experts.w2_weight_amax";
    let mut missing = tensors.clone();
    missing.remove(amax).unwrap();
    let e = spec::weights_bytes_of(&missing).unwrap_err();
    assert!(e.contains(&format!("holds no tensor {amax}")), "{e}");
    let scale = "model.llm.layers.21.mlp.experts.w13_weight_scale";
    let mut other = tensors.clone();
    other.get_mut(scale).unwrap().dtype = Dtype::F8E4M3;
    let e = spec::weights_bytes_of(&other).unwrap_err();
    assert!(e.contains(&format!("the tensor {scale} is F8E4M3")), "{e}");
    for name in ["model.mtp.norm.weight", "model.llm.layers.0.mlp.experts.w2_weight_bias"] {
        let mut extra = tensors.clone();
        let mut meta = tensors["model.llm.norm.weight"].clone();
        meta.name = name.to_owned();
        extra.insert(name.to_owned(), meta);
        let e = spec::weights_bytes_of(&extra).unwrap_err();
        assert!(
            e.contains(&format!("the tensor {name}, which the Inkling 2 engine does not know")),
            "{e}"
        );
    }
    // The tower of images is not loaded yet, so its tensors add no bytes.
    let mut visual = tensors.clone();
    visual.retain(|name, _| !name.starts_with("model.visual."));
    assert_eq!(spec::weights_bytes_of(&visual), Ok(18_881_368_064));
}

/// The load of the weights into a count gives the bytes of the plan, so the open of a card finds the plan's bytes.
#[test]
fn the_load_counts_the_bytes_of_the_plan() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let map = MappedSafeTensors::open(&common::model_weights(Which::Mini)).unwrap();
    let view = map.view().unwrap();
    let mut count = nagaya_mk_inkling2_bmg::weights::Count::default();
    let w = nagaya_mk_inkling2_bmg::weights::load(&mut count, &view).unwrap();
    assert_eq!(count.bytes as u64, spec::weights_bytes_of(&view.tensors).unwrap());
    assert_eq!(count.bytes, 18_881_368_064);
    assert_eq!(w.layers.len(), 24);
    assert!(w.layers[20].mxfp8 && !w.layers[19].mxfp8);
    assert!(w.layers[6].gates.iter().all(|g| *g == 0), "a count gives no addresses");
}
