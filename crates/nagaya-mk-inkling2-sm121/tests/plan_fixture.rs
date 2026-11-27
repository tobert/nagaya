//! `plan` against a small checkpoint that the test writes: a real config, a generation config, and a safetensors file
//! with tiny tensors. It needs no model folder and no GPU.

use std::path::{Path, PathBuf};

use nagaya_mk_inkling2_sm121::spec::{self, Config, DEFAULT_CACHE, DEFAULT_EXTRA_RUNS, DEFAULT_MAX_BLOCK};

const CONFIG: &str = include_str!("fixtures/config.json");

/// One tensor of the test file: its name, dtype, shape and byte count.
type Tensor = (String, &'static str, Vec<usize>, usize);

/// The tensors of a valid checkpoint: the expert tensors of 24 layers, one more weight of the LLM, and one of the
/// vision tower.
fn tensors() -> Vec<Tensor> {
    let mut out = Vec::new();
    for layer in 0..24 {
        let dtype = if layer < 20 { "U8" } else { "F8_E4M3" };
        for suffix in ["w13_weight", "w2_weight"] {
            out.push((
                format!("model.llm.layers.{layer}.mlp.experts.{suffix}"),
                dtype,
                vec![1],
                1,
            ));
        }
    }
    out.push(("model.llm.embed_tokens.weight".to_owned(), "BF16", vec![10, 5], 100));
    out.push(("model.visual.patch.weight".to_owned(), "F32", vec![3], 12));
    out
}

/// Write a model folder in a new temporary directory. `generation` is the text of `generation_config.json`.
fn write_model(name: &str, tensors: &[Tensor], generation: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nagaya-sm121-plan-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), CONFIG).unwrap();
    std::fs::write(dir.join("generation_config.json"), generation).unwrap();
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (tensor, dtype, shape, bytes) in tensors {
        header.insert(
            tensor.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let header = serde_json::Value::Object(header).to_string();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + offset, 0);
    std::fs::write(dir.join(spec::WEIGHTS_FILE), file).unwrap();
    dir
}

fn config(dir: &Path, slots: Vec<usize>) -> Config {
    Config {
        model_dir: dir.to_path_buf(),
        slots,
        max_block: DEFAULT_MAX_BLOCK,
        extra_runs: DEFAULT_EXTRA_RUNS,
        taps: false,
        cache: DEFAULT_CACHE,
    }
}

const GENERATION: &str = r#"{"pad_token_id": 199999}"#;

#[test]
fn plan_sums_the_header_and_sizes_the_slots_with_no_model() {
    let dir = write_model("ok", &tensors(), GENERATION);
    assert_eq!(spec::Model::read(&dir).unwrap(), spec::Model::Mini);
    let lim = spec::plan(&config(&dir, vec![524_288, 32_768, 32_768])).unwrap();
    let bytes = |name: &str| lim.memory.mandatory.iter().find(|t| t.name == name).unwrap().bytes;
    // Each of the 48 expert tensors takes 64 bytes, and the embedding takes 100 bytes rounded to 128.
    assert_eq!(bytes("weights"), 48 * 64 + 128);
    assert!(
        lim.memory.mandatory.iter().all(|t| t.name != "vision"),
        "the tower of images is not served yet"
    );
    assert_eq!(bytes("kv"), 34_866_888_704 + 2 * 2_654_633_984);
    // The classes of slots are sorted by length, and equal lengths make one class.
    let classes: Vec<(u32, u64, u64)> = lim.slots.iter().map(|s| (s.count, s.tokens, s.bytes)).collect();
    assert_eq!(classes, vec![(2, 32_768, 2_654_633_984), (1, 524_288, 34_866_888_704)]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn plan_refuses_a_missing_expert_tensor_a_wrong_dtype_and_a_wrong_pad_token() {
    let mut missing = tensors();
    missing.retain(|(name, ..)| name != "model.llm.layers.7.mlp.experts.w2_weight");
    let dir = write_model("missing", &missing, GENERATION);
    let error = spec::plan(&config(&dir, vec![32_768])).unwrap_err();
    assert!(error.contains("model.llm.layers.7.mlp.experts.w2_weight"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);

    let mut wrong = tensors();
    wrong[40].1 = "U8"; // Layer 20 must hold MXFP8 (F8_E4M3).
    let dir = write_model("dtype", &wrong, GENERATION);
    let error = spec::plan(&config(&dir, vec![32_768])).unwrap_err();
    assert!(
        error.contains("model.llm.layers.20.") && error.contains("F8_E4M3"),
        "{error}"
    );
    let _ = std::fs::remove_dir_all(&dir);

    let dir = write_model("pad", &tensors(), r#"{"pad_token_id": 5}"#);
    let error = spec::plan(&config(&dir, vec![32_768])).unwrap_err();
    assert!(error.contains("pad_token_id") && error.contains("199999"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}
