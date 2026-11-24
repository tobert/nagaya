//! The digest of the checkpoint and the text facts, with a small checkpoint that the test writes. The digest must equal
//! the digest of the B70 engine, which reads the same checkpoint from memory.

use std::path::{Path, PathBuf};

use nagaya_mk_bmg::safetensors::SafeTensorsView;
use nagaya_mk_inkling2_sm121::identity::{checkpoint_digest, checkpoint_digest_with, text_facts};

const CONFIG: &str = r#"{"b": 1, "a": {"y": [1, 2.5, "x"], "x": null}}"#;

/// A tensor of the test file: name, dtype, shape and bytes.
type Tensor = (&'static str, &'static str, Vec<usize>, Vec<u8>);

/// Bytes that do not repeat inside a chunk, so that a wrong offset changes the hash.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as u8
        })
        .collect()
}

/// Tensors in an order that is not the bytewise order of the names, with one tensor longer than a chunk of 16 MiB.
fn tensors() -> Vec<Tensor> {
    vec![
        ("model.llm.z.weight", "BF16", vec![4, 8], noise(64, 1)),
        ("model.llm.a.scale", "F8_E4M3", vec![1000], noise(1000, 2)),
        ("model.visual.patch", "F32", vec![3], noise(12, 3)),
        (
            "model.llm.big",
            "U8",
            vec![17 << 20],
            noise((17 << 20) + 3, 4)[..17 << 20].to_vec(),
        ),
        ("model.llm.empty", "BF16", vec![0], Vec::new()),
        ("model.llm.b.ids", "I64", vec![2], noise(16, 5)),
    ]
}

fn write(name: &str, tensors: &[Tensor]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nagaya-sm121-digest-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let entries: Vec<(&str, &str, &[usize], &[u8])> = tensors
        .iter()
        .map(|(n, d, s, b)| (*n, *d, s.as_slice(), b.as_slice()))
        .collect();
    let file = dir.join("model.safetensors");
    std::fs::write(&file, SafeTensorsView::build_synthetic_safetensors(&entries)).unwrap();
    file
}

/// The digest of the B70 engine (`nagaya_mk_inkling2_bmg::engine::checkpoint_digest`), over the file in memory. The
/// function is behind the feature `xe` there, so the test holds its text.
fn oracle(config_text: &str, file: &Path) -> String {
    let raw = std::fs::read(file).unwrap();
    let view = SafeTensorsView::parse(&raw).unwrap();
    let config: serde_json::Value = serde_json::from_str(config_text).unwrap();
    let mut names: Vec<&String> = view.tensors.keys().collect();
    names.sort();
    let mut hasher = blake3::Hasher::new_derive_key("nagaya inkling2 checkpoint v1");
    fn canonical(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let items: Vec<String> = keys
                    .into_iter()
                    .map(|k| format!("{}:{}", serde_json::Value::String(k.clone()), canonical(&map[k])))
                    .collect();
                format!("{{{}}}", items.join(","))
            }
            serde_json::Value::Array(items) => {
                let items: Vec<String> = items.iter().map(canonical).collect();
                format!("[{}]", items.join(","))
            }
            other => other.to_string(),
        }
    }
    let text = canonical(&config);
    hasher.update(&(text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
    for name in names {
        let meta = &view.tensors[name];
        let dtype = format!("{:?}", meta.dtype);
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(dtype.len() as u64).to_le_bytes());
        hasher.update(dtype.as_bytes());
        hasher.update(&(meta.shape.len() as u64).to_le_bytes());
        for dim in &meta.shape {
            hasher.update(&(*dim as u64).to_le_bytes());
        }
        hasher.update(blake3::hash(view.tensor_bytes(name).unwrap()).as_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[test]
fn the_digest_equals_the_digest_of_the_b70_engine_for_any_count_of_workers() {
    let file = write("equal", &tensors());
    let want = oracle(CONFIG, &file);
    assert!(want.starts_with("blake3:") && want.len() == 7 + 64);
    for workers in [1, 3, 16] {
        assert_eq!(
            checkpoint_digest_with(CONFIG, &file, workers).unwrap(),
            want,
            "{workers} workers"
        );
    }
    assert_eq!(checkpoint_digest(CONFIG, &file).unwrap(), want);
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

#[test]
fn the_digest_follows_the_config_and_each_part_of_each_tensor() {
    let base = tensors();
    let file = write("base", &base);
    let digest = |config: &str, file: &Path| checkpoint_digest_with(config, file, 4).unwrap();
    let want = digest(CONFIG, &file);
    // The order of the keys and the white space of config.json do not matter. A value does.
    let reordered = r#"{ "a": {"x": null, "y": [1, 2.5, "x"]},
        "b": 1 }"#;
    assert_eq!(digest(reordered, &file), want);
    assert_ne!(digest(&CONFIG.replace("\"b\": 1", "\"b\": 2"), &file), want);
    // One byte in the last chunk of the long tensor, the name, the dtype and the shape each change the digest.
    let mut bytes = base.clone();
    bytes[3].3[(17 << 20) - 1] ^= 1;
    assert_ne!(digest(CONFIG, &write("byte", &bytes)), want);
    let mut name = base.clone();
    name[1].0 = "model.llm.a.scalf";
    assert_ne!(digest(CONFIG, &write("name", &name)), want);
    let mut dtype = base.clone();
    dtype[0].1 = "F16";
    assert_ne!(digest(CONFIG, &write("dtype", &dtype)), want);
    let mut shape = base.clone();
    shape[0].2 = vec![8, 4];
    assert_ne!(digest(CONFIG, &write("shape", &shape)), want);
    for case in ["byte", "name", "dtype", "shape", "base"] {
        let _ = std::fs::remove_dir_all(
            std::env::temp_dir().join(format!("nagaya-sm121-digest-{}-{case}", std::process::id())),
        );
    }
}

#[test]
fn a_bad_file_gives_an_error_that_names_the_cause() {
    let mut wrong = tensors();
    wrong.push(("model.llm.f64", "F64", vec![1], noise(8, 9)));
    // The in-memory writer takes any dtype string, and the digest refuses the dtype that the B70 reader refuses.
    let file = write("f64", &wrong);
    let error = checkpoint_digest(CONFIG, &file).unwrap_err();
    assert!(error.contains("model.llm.f64") && error.contains("F64"), "{error}");
    assert!(
        checkpoint_digest("{", &file)
            .unwrap_err()
            .contains("config.json is not JSON")
    );
    let missing = file.with_file_name("absent.safetensors");
    assert!(checkpoint_digest(CONFIG, &missing).is_err());
    let _ = std::fs::remove_dir_all(file.parent().unwrap());
}

/// The text facts of the folder equal the facts of the B70 engine: the two engines serve one family.
#[test]
fn the_text_facts_equal_the_facts_of_the_b70_engine() {
    let require = std::env::var_os("NAGAYA_REQUIRE_MODELS").is_some();
    let Some(path) = std::env::var_os("INKLING2_MODEL_DIR") else {
        assert!(
            !require,
            "INKLING2_MODEL_DIR is not set and NAGAYA_REQUIRE_MODELS is set"
        );
        eprintln!("skipping the_text_facts_equal_the_facts_of_the_b70_engine: INKLING2_MODEL_DIR is not set");
        return;
    };
    let dir = PathBuf::from(path);
    if !dir.exists() {
        assert!(!require, "INKLING2_MODEL_DIR points to absent path `{}`", dir.display());
        eprintln!("skipping: `{}` is absent", dir.display());
        return;
    }
    let ours = text_facts(&dir).unwrap();
    let theirs = nagaya_mk_inkling2_bmg::spec::text_facts(&dir).unwrap();
    assert_eq!(
        (ours.tokenizer_hash.as_str(), ours.template.as_str()),
        (theirs.tokenizer_hash.as_str(), theirs.template.as_str())
    );
    assert!(!ours.tokenizer_hash.is_empty() && !ours.template.is_empty());
    assert!(
        text_facts(&dir.join("absent"))
            .unwrap_err()
            .contains("cannot build the Inkling 2 family")
    );
}
