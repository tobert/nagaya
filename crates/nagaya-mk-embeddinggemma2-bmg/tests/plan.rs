//! The plan of the engine from the checkpoint's config and tensor headers, with no card (spec 2.4).

mod common;

use nagaya_abi::{Capability, Kind, Tier};
use nagaya_mk_embeddinggemma2_bmg::spec::{Config, check_config};
use nagaya_mk_embeddinggemma2_bmg::{model, plan};

/// The bytes of the text tower in BF16, from the header of `model.safetensors`, less the 24 layer scalars of 2 bytes,
/// which stay on the host.
const TOWER_BYTES: u64 = 542_005_296 - 24 * 2;

#[test]
fn the_plan_counts_the_text_tower_and_declares_the_embedding_capability() {
    if common::absent() {
        return;
    }
    let limits = plan(&Config {
        model_dir: common::model_dir(),
    })
    .expect("plan EmbeddingGemma 2");
    assert_eq!(limits.capabilities, vec![Capability::Embedding, Capability::Close]);
    assert!(limits.slots.is_empty(), "the engine holds no state");
    assert!(limits.parts.is_empty());
    assert_eq!(limits.max_segments, 0);
    let term = |name: &str| {
        limits
            .memory
            .mandatory
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no term {name}"))
    };
    let weights = term("weights");
    assert_eq!(weights.tier, Tier::Device(0));
    assert_eq!(weights.kinds, vec![Kind::Embed]);
    // Each tensor rounds up to 64 bytes; the gate and up rows are one allocation.
    let tensors = 5 + 24 * 15;
    assert!(
        (TOWER_BYTES..TOWER_BYTES + 64 * tensors).contains(&weights.bytes),
        "the weights term is {} bytes",
        weights.bytes
    );
    let activations = term("activations");
    assert!(activations.bytes >= model::bufs_bytes(2048) as u64);
    println!(
        "weights {} bytes, activations {} bytes, pinned {} bytes",
        weights.bytes,
        activations.bytes,
        term("pinned").bytes
    );
}

/// A copy of the checkpoint's config with one change, in a fresh temporary directory.
fn changed_config(change: impl FnOnce(&mut serde_json::Value)) -> std::path::PathBuf {
    let text = std::fs::read_to_string(common::model_dir().join("config.json")).unwrap();
    let mut config: serde_json::Value = serde_json::from_str(&text).unwrap();
    change(&mut config);
    let dir = std::env::temp_dir().join(format!(
        "nagaya-eg2-plan-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("t").replace("::", "-")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    dir
}

#[test]
fn a_config_with_other_numbers_is_refused() {
    if common::absent() {
        return;
    }
    check_config(&common::model_dir()).expect("the checkpoint's own config passes");
    type Change = Box<dyn FnOnce(&mut serde_json::Value)>;
    let changes: [(&str, Change); 4] = [
        (
            "hidden_size",
            Box::new(|c| c["text_config"]["hidden_size"] = 640.into()),
        ),
        (
            "rope",
            Box::new(|c| c["text_config"]["rope_parameters"]["full_attention"]["rope_theta"] = 10000.0.into()),
        ),
        (
            "layer types",
            Box::new(|c| c["text_config"]["layer_types"][4] = "full_attention".into()),
        ),
        (
            "per-layer config",
            Box::new(|c| c["text_config"]["per_layer_config"]["11"]["head_dim"] = 256.into()),
        ),
    ];
    for (what, change) in changes {
        let dir = changed_config(change);
        let result = check_config(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(result.is_err(), "a config with another {what} passed");
    }
}
