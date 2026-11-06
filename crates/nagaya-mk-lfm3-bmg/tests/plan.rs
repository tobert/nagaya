//! The plan of the engine with no card: the tensors of both checkpoints, and the refusal of another checkpoint.

mod common;

use common::Which;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView};
use nagaya_mk_lfm3_bmg::spec::{self, Backbone, Config, Model};
use nagaya_mk_lfm3_bmg::weights::{self, Count};

/// The device bytes of the backbone: 1,728,801,536 BF16 parameters (`AGENTS.md` milestone 24), each tensor in
/// steps of 64 bytes, which every tensor of the backbone fills.
const WEIGHTS_BYTES: u64 = 1_728_801_536 * 2;

#[test]
fn both_checkpoints_hold_every_tensor_of_the_backbone_and_plan_alike() {
    let mut backbones = Vec::new();
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        let dir = common::model_dir(which);
        let b = Backbone::read(&dir).unwrap();
        assert_eq!(b.model, if which == Which::Text { Model::Text } else { Model::Omni });
        assert_eq!(spec::weights_bytes(&b, &dir).unwrap(), WEIGHTS_BYTES, "{which:?}");
        let config = Config {
            model_dir: dir,
            slots: vec![2048, 32_768],
            max_block: 512,
        };
        let limits = spec::plan(&config).unwrap();
        limits.check().unwrap();
        let weights = limits.memory.mandatory.iter().find(|t| t.name == "weights").unwrap();
        assert_eq!(weights.bytes, WEIGHTS_BYTES);
        backbones.push(b);
    }
    if let [text, omni] = &backbones[..] {
        assert_eq!(
            &Backbone {
                model: Model::Text,
                ..omni.clone()
            },
            text
        );
    }
}

/// A small backbone of two layers, a conv layer and a full layer, with a vocabulary and an MLP of 16.
fn small() -> Backbone {
    let mut config: serde_json::Value = serde_json::from_str(include_str!("fixtures/lfm3.config.json")).unwrap();
    config["layer_types"] = serde_json::json!(["conv", "full_attention"]);
    config["num_hidden_layers"] = 2.into();
    config["intermediate_size"] = 16.into();
    config["vocab_size"] = 16.into();
    Backbone::parse(&config.to_string()).unwrap()
}

/// A checkpoint of the tensors of `b` with zero bytes, after `edit` changed the list.
fn checkpoint(b: &Backbone, edit: impl FnOnce(&mut Vec<(String, Vec<usize>)>)) -> Vec<u8> {
    let mut list: Vec<(String, Vec<usize>)> = weights::tensors(b)
        .into_iter()
        .map(|(name, shape)| (format!("{}{name}", b.model.prefix()), shape))
        .collect();
    edit(&mut list);
    let most = list.iter().map(|(_, s)| s.iter().product::<usize>() * 2).max().unwrap();
    let zeros = vec![0u8; most];
    let entries: Vec<(&str, &str, &[usize], &[u8])> = list
        .iter()
        .map(|(name, shape)| {
            let bytes = shape.iter().product::<usize>() * 2;
            (name.as_str(), "BF16", shape.as_slice(), &zeros[..bytes])
        })
        .collect();
    SafeTensorsView::build_synthetic_safetensors(&entries)
}

fn checked(b: &Backbone, bytes: &[u8]) -> Result<(), String> {
    weights::check(b, &SafeTensorsView::parse(bytes).unwrap())
}

#[test]
fn a_checkpoint_with_a_lost_or_other_tensor_is_refused_before_any_upload() {
    let b = small();
    let whole = checkpoint(&b, |_| {});
    checked(&b, &whole).unwrap();
    let mut count = Count::default();
    weights::load(&mut count, &b, &SafeTensorsView::parse(&whole).unwrap()).unwrap();
    let want: usize = weights::tensors(&b)
        .iter()
        .map(|(_, s)| s.iter().product::<usize>() * 2)
        .sum();
    assert_eq!(count.bytes, want);

    let lost = checkpoint(&b, |list| {
        list.retain(|(n, _)| !n.ends_with("self_attn.gate_proj.weight"))
    });
    let error = checked(&b, &lost).unwrap_err();
    assert!(
        error.contains("no tensor model.layers.1.self_attn.gate_proj.weight"),
        "{error}"
    );

    let shape = checkpoint(&b, |list| {
        let conv = list.iter_mut().find(|(n, _)| n.ends_with("conv.conv.weight")).unwrap();
        conv.1 = vec![2048, 1, 4];
    });
    let error = checked(&b, &shape).unwrap_err();
    assert!(
        error.contains("model.layers.0.conv.conv.weight") && error.contains("[2048, 1, 7]"),
        "{error}"
    );

    let bias = checkpoint(&b, |list| {
        list.push(("model.layers.0.conv.in_proj.bias".to_owned(), vec![6144]))
    });
    let error = checked(&b, &bias).unwrap_err();
    assert!(error.contains("model.layers.0.conv.in_proj.bias"), "{error}");
    let mut count = Count::default();
    assert!(weights::load(&mut count, &b, &SafeTensorsView::parse(&bias).unwrap()).is_err());
    assert_eq!(count.bytes, 0, "a refused checkpoint uploads nothing");

    // A head of its own is a tensor outside the prefix, which a file of LFM3-1.7B must not hold.
    let head = checkpoint(&b, |list| list.push(("lm_head.weight".to_owned(), vec![16, 2048])));
    assert!(checked(&b, &head).unwrap_err().contains("lm_head.weight"));
}

#[test]
fn omni_keeps_its_other_towers_beside_the_backbone() {
    let mut b = small();
    b.model = Model::Omni;
    let towers = checkpoint(&b, |list| {
        list.push(("model.vision_tower.patch.weight".to_owned(), vec![16]));
        list.push(("model.depth_decoder.norm.weight".to_owned(), vec![16]));
    });
    checked(&b, &towers).unwrap();
    let extra = checkpoint(&b, |list| {
        list.push(("model.language_model.extra.weight".to_owned(), vec![16]))
    });
    assert!(
        checked(&b, &extra)
            .unwrap_err()
            .contains("model.language_model.extra.weight")
    );
}

#[test]
fn the_real_headers_open_as_a_mapped_file() {
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        let dir = common::model_dir(which);
        let map = MappedSafeTensors::open(&dir.join(spec::WEIGHTS_FILE)).unwrap();
        let b = Backbone::read(&dir).unwrap();
        weights::check(&b, &map.view().unwrap()).unwrap();
    }
}
