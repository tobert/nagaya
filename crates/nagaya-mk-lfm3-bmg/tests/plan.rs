//! The plan of the engine with no card: the tensors of both checkpoints, and the refusal of another checkpoint.

mod common;

use common::Which;
use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView};
use nagaya_mk_lfm3_bmg::spec::{self, Backbone, Config, Model};
use nagaya_mk_lfm3_bmg::weights::{self, Count};

/// The device bytes of the backbone: 1,728,801,536 BF16 parameters (`AGENTS.md` milestone 24), each tensor in
/// steps of 64 bytes, which every tensor of the backbone fills.
const WEIGHTS_BYTES: u64 = 1_728_801_536 * 2;

/// The device bytes of the audio encoder and its projector of Omni: 115,278,144 BF16 parameters (the checkpoint
/// header, read on 2026-11-06), and 2 bytes more for each of the 2 × 768 values of the LayerNorm, which is F32.
const AUDIO_BYTES: u64 = 115_278_144 * 2 + 2 * 768 * 2;

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
        // Omni also loads its audio encoder (engine-abi "The audio encoder of LFM3").
        let want = match which {
            Which::Text => WEIGHTS_BYTES,
            Which::Omni => WEIGHTS_BYTES + AUDIO_BYTES,
        };
        assert_eq!(spec::weights_bytes(&b, &dir).unwrap(), want, "{which:?}");
        let config = Config {
            model_dir: dir,
            slots: 2,
            max_block: 512,
        };
        let limits = spec::plan(&config).unwrap();
        limits.check().unwrap();
        let weights = limits.memory.mandatory.iter().find(|t| t.name == "weights").unwrap();
        assert_eq!(weights.bytes, want);
        backbones.push(b);
    }
    if let [text, omni] = &backbones[..] {
        assert_eq!(
            &Backbone {
                model: Model::Text,
                audio: None,
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

/// The bytes of the tower and of the projector of Omni in its header: each BF16 tensor, with the weights and the bias
/// of the projector's LayerNorm in F32, so twice their BF16 bytes. Each one takes whole steps of 64 bytes.
fn tower_bytes_of_the_header(view: &SafeTensorsView<'_>) -> u64 {
    view.tensors
        .iter()
        .filter(|(n, _)| n.starts_with("model.vision_tower.") || n.starts_with("model.multi_modal_projector."))
        .map(|(n, t)| {
            let bytes = t.shape.iter().product::<usize>() as u64 * 2;
            let bytes = if n.contains("layer_norm") { 2 * bytes } else { bytes };
            weights::rounded(bytes as usize) as u64
        })
        .sum()
}

#[test]
fn omni_plans_its_tower_and_the_text_model_has_none() {
    for which in Which::BOTH {
        if common::absent(which) {
            continue;
        }
        let dir = common::model_dir(which);
        let config = Config {
            model_dir: dir.clone(),
            slots: 1,
            max_block: 512,
        };
        let limits = spec::plan(&config).unwrap();
        let vision = limits.memory.mandatory.iter().find(|t| t.name == "vision");
        let b = Backbone::read(&dir).unwrap();
        match which {
            Which::Text => {
                assert!(spec::read_vision(&b, &dir).unwrap().is_none());
                assert!(vision.is_none(), "LFM3-1.7B has no vision tower");
                assert_eq!(limits.inputs, nagaya_abi::Inputs::Tokens);
            }
            Which::Omni => {
                let map = MappedSafeTensors::open(&dir.join(spec::WEIGHTS_FILE)).unwrap();
                let want = tower_bytes_of_the_header(&map.view().unwrap());
                // 14 layers and the projector: 189 M parameters in BF16.
                assert!(want > 370_000_000 && want < 390_000_000, "{want}");
                assert_eq!(vision.unwrap().bytes, want);
                assert_eq!(limits.inputs, nagaya_abi::Inputs::TokensAndRows);
                let v = spec::read_vision(&b, &dir).unwrap().unwrap();
                let bytes = spec::vision_bytes(&v, &dir, 512).unwrap();
                println!(
                    "Omni: the tower takes {} weight bytes and {} buffer bytes",
                    bytes.weights, bytes.work
                );
            }
        }
    }
}

/// The names and the shapes of the tensors of a checkpoint.
type TensorList = Vec<(String, Vec<usize>)>;

/// A small tower of one conv layer and one attention layer, with widths of 16s.
fn small_vision() -> nagaya_mk_lfm3_bmg::vision::Vision {
    let mut config: serde_json::Value = serde_json::from_str(include_str!("fixtures/lfm3-omni.config.json")).unwrap();
    let v = &mut config["vision_config"];
    v["hidden_size"] = 64.into();
    v["num_attention_heads"] = 4.into();
    v["num_key_value_heads"] = 2.into();
    v["intermediate_size"] = 32.into();
    v["num_hidden_layers"] = 2.into();
    v["layer_types"] = serde_json::json!(["conv", "full_attention"]);
    config["projector_hidden_size"] = 32.into();
    nagaya_mk_lfm3_bmg::vision::Vision::parse(&config.to_string(), 32).unwrap()
}

#[test]
fn a_tower_with_a_lost_or_other_tensor_is_refused_before_any_upload() {
    use nagaya_mk_lfm3_bmg::vision;
    let v = small_vision();
    let build = |edit: &dyn Fn(&mut TensorList)| -> Vec<u8> {
        let mut list = vision::tensors(&v);
        edit(&mut list);
        let most = list.iter().map(|(_, s)| s.iter().product::<usize>() * 2).max().unwrap();
        let zeros = vec![0u8; most];
        let entries: Vec<(&str, &str, &[usize], &[u8])> = list
            .iter()
            .map(|(n, s)| {
                (
                    n.as_str(),
                    "BF16",
                    s.as_slice(),
                    &zeros[..s.iter().product::<usize>() * 2],
                )
            })
            .collect();
        SafeTensorsView::build_synthetic_safetensors(&entries)
    };
    let whole = build(&|_| {});
    let mut count = Count::default();
    vision::load(&mut count, &v, &SafeTensorsView::parse(&whole).unwrap()).unwrap();
    assert_eq!(
        count.bytes as u64,
        tower_bytes_of_the_header(&SafeTensorsView::parse(&whole).unwrap())
    );

    let lost = build(&|l| l.retain(|(n, _)| !n.ends_with("layers.1.self_attn.k_layernorm.weight")));
    let e = vision::check(&v, &SafeTensorsView::parse(&lost).unwrap()).unwrap_err();
    assert!(
        e.contains("no tensor model.vision_tower.layers.1.self_attn.k_layernorm.weight"),
        "{e}"
    );
    let shape = build(&|l| {
        let conv = l.iter_mut().find(|(n, _)| n.ends_with("spatial_conv.weight")).unwrap();
        conv.1 = vec![64, 1, 5, 5];
    });
    let e = vision::check(&v, &SafeTensorsView::parse(&shape).unwrap()).unwrap_err();
    assert!(e.contains("spatial_conv.weight") && e.contains("[64, 1, 3, 3]"), "{e}");
    let other = build(&|l| l.push(("model.vision_tower.post_layernorm.weight".to_owned(), vec![64])));
    let mut count = Count::default();
    let e = vision::load(&mut count, &v, &SafeTensorsView::parse(&other).unwrap()).unwrap_err();
    assert!(e.contains("model.vision_tower.post_layernorm.weight"), "{e}");
    assert_eq!(count.bytes, 0, "a refused tower uploads nothing");
}

/// Both models have the control table of 8 vectors (spec 11.12). Omni's vector acts on its text passes: a speech frame
/// is a `frame` call, which takes no control list (AGENTS.md milestone 25 step 9).
#[test]
fn both_models_have_a_control_table_of_their_layers_and_width() {
    for text in [
        include_str!("fixtures/lfm3.config.json"),
        include_str!("fixtures/lfm3-omni.config.json"),
    ] {
        let b = Backbone::parse(text).unwrap();
        let limit = spec::control_limit(&b).unwrap_or_else(|| panic!("{:?} has a control table", b.model));
        assert_eq!(
            (limit.tables, limit.layers, limit.width),
            (8, 24, 2048),
            "{:?}",
            b.model
        );
    }
}
