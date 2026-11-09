//! The unembedding of the lab route `project` on the CPU (`docs/lab.md`): small checkpoints with known values, and the
//! oracle stage `project` of transformers for LFM3-1.7B. No card runs.

mod common;

use std::path::Path;

use common::Which;
use nagaya_mk_bmg::safetensors::SafeTensorsView;
use nagaya_mk_lfm3_bmg::lab::{Entry, Facts, JLens, LensError, Unembedding, ends_of, rank};
use nagaya_mk_lfm3_bmg::spec::Model;

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// Write a checkpoint of `model` into `dir`: a table of `table` rows of `width` values, the norm weight `norm`, and the
/// end tokens `ends`. Each value must be exact in BF16.
fn checkpoint(dir: &Path, model: Model, table: &[Vec<f32>], norm: &[f32], ends: &str) {
    let (architecture, config) = match model {
        Model::Text => (
            "Lfm3ForCausalLM",
            serde_json::json!({"num_hidden_layers": 2, "hidden_size": norm.len(), "norm_eps": 0.25}),
        ),
        Model::Omni => (
            "Lfm3OmniForConditionalGeneration",
            serde_json::json!({"text_config": {"num_hidden_layers": 2, "hidden_size": norm.len(), "norm_eps": 0.25}}),
        ),
    };
    let mut config = config;
    config["architectures"] = serde_json::json!([architecture]);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        format!("{{\"eos_token_id\": {ends}}}"),
    )
    .unwrap();
    let flat: Vec<f32> = table.iter().flatten().copied().collect();
    let (embed, embed_bytes) = (format!("{}embed_tokens.weight", model.prefix()), bf16(&flat));
    let (norm_name, norm_bytes) = (format!("{}embedding_norm.weight", model.prefix()), bf16(norm));
    let shape = [table.len(), norm.len()];
    let bytes = SafeTensorsView::build_synthetic_safetensors(&[
        (&embed, "BF16", &shape, &embed_bytes),
        (&norm_name, "BF16", &[norm.len()], &norm_bytes),
    ]);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

fn small(model: Model) -> (tempfile::TempDir, Unembedding) {
    let dir = tempfile::tempdir().unwrap();
    let table = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![-1.0, 0.5, 2.0],
    ];
    checkpoint(dir.path(), model, &table, &[2.0, 0.5, -1.0], "[3, 4]");
    let lab = Unembedding::open(dir.path()).unwrap();
    (dir, lab)
}

#[test]
fn a_score_is_the_row_times_the_norm_weight_times_the_direction() {
    let (_dir, lab) = small(Model::Text);
    assert_eq!((lab.rows(), lab.width(), lab.layers()), (5, 3, 2));
    assert_eq!(lab.ends(), &[3, 4]);
    let scores = lab.scores(&[&[1.0, 2.0, 3.0]]);
    // u = g * v = [2, 1, -3].
    assert_eq!(scores, vec![vec![2.0, 1.0, -3.0, 0.0, -7.5]]);
}

#[test]
fn a_score_has_no_division_by_the_root_mean_square() {
    let (_dir, lab) = small(Model::Text);
    let scores = lab.scores(&[&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]]);
    let tenfold: Vec<f32> = scores[0].iter().map(|s| s * 10.0).collect();
    assert_eq!(
        scores[1], tenfold,
        "a ten times longer direction gives ten times the scores"
    );
}

#[test]
fn omni_nests_the_tensors_of_its_language_model() {
    let (_dir, lab) = small(Model::Omni);
    assert_eq!(lab.model(), Model::Omni);
    assert_eq!(lab.scores(&[&[1.0, 2.0, 3.0]]), vec![vec![2.0, 1.0, -3.0, 0.0, -7.5]]);
}

#[test]
fn a_folder_without_the_tensors_or_the_end_tokens_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    // The tensor names of Omni under the architecture of the text model.
    checkpoint(dir.path(), Model::Omni, &[vec![1.0]], &[1.0], "1");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": 2, "hidden_size": 1}"#,
    )
    .unwrap();
    let error = Unembedding::open(dir.path()).err().unwrap();
    assert!(error.contains("no tensor model.embed_tokens.weight"), "{error}");

    checkpoint(dir.path(), Model::Text, &[vec![1.0]], &[1.0], "null");
    let error = Unembedding::open(dir.path()).err().unwrap();
    assert!(error.contains("eos_token_id"), "{error}");

    checkpoint(dir.path(), Model::Text, &[vec![1.0]], &[1.0], "[1]");
    let error = Unembedding::open(dir.path()).err().unwrap();
    assert!(error.contains("past the 1 rows"), "{error}");
}

/// The facts of a model folder for `trace` read no weights (`docs/lab.md`, "The rules of `trace`").
#[test]
fn the_facts_of_a_folder_are_its_layers_its_width_and_its_end_tokens() {
    for model in [Model::Text, Model::Omni] {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(dir.path(), model, &[vec![1.0, 2.0, 3.0]], &[1.0, 1.0, 1.0], "[3, 4]");
        std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
        let facts = Facts::open(dir.path()).unwrap();
        assert_eq!(
            facts,
            Facts {
                model,
                layers: 2,
                width: 3,
                ends: vec![3, 4]
            }
        );
    }
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path(), Model::Text, &[vec![1.0, 2.0]], &[1.0, 1.0], "[0]");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": 2}"#,
    )
    .unwrap();
    let error = Facts::open(dir.path()).err().unwrap();
    assert!(error.contains("hidden_size"), "{error}");
    // The width of the table and of config.json agree, or the folder is an error.
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": 2, "hidden_size": 3}"#,
    )
    .unwrap();
    let error = Unembedding::open(dir.path()).err().unwrap();
    assert!(error.contains("hidden_size 3"), "{error}");
}

#[test]
fn the_ends_take_the_smaller_id_in_a_tie() {
    let scores = [1.0, 3.0, 3.0, -2.0, 0.0, -2.0];
    let (up, down) = ends_of(&scores, 3);
    let ids = |list: &[Entry]| list.iter().map(|e| e.id).collect::<Vec<_>>();
    assert_eq!(ids(&up), [1, 2, 0]);
    assert_eq!(ids(&down), [3, 5, 4]);
    assert_eq!(up[0].score, 3.0);
    assert_eq!((rank(&scores, 1), rank(&scores, 2), rank(&scores, 5)), (0, 1, 5));
    let (all, _) = ends_of(&scores, 10);
    assert_eq!(ids(&all), [1, 2, 0, 4, 3, 5]);
}

/// The worst gap of a score of transformers from the exact sum, in f32 (Higham, "Accuracy and Stability of Numerical
/// Algorithms", 3.1): the width times the unit roundoff, times the sum of the sizes of the products. Our score is the
/// exact sum, rounded once.
fn bound(row: &[f32], u: &[f32], score: f32) -> f64 {
    let size: f64 = row.iter().zip(u).map(|(r, x)| (*r as f64 * *x as f64).abs()).sum();
    let unit = f32::EPSILON as f64 / 2.0;
    row.len() as f64 * unit * size + unit * score.abs() as f64
}

#[test]
fn the_scores_of_lfm3_match_the_oracle_of_transformers() {
    if common::absent(Which::Text) || common::golden_absent("lfm3.project.safetensors") {
        return;
    }
    let dir = common::model_dir(Which::Text);
    let lab = Unembedding::open(&dir).unwrap();
    assert_eq!((lab.rows(), lab.width(), lab.layers()), (128_000, 2048, 24));
    assert_eq!(lab.ends(), &[124_900]);
    let golden = common::golden_tensors("lfm3.project.safetensors");
    let vector = golden["vector"].f32();
    let layers = golden["layers"].ids();
    assert_eq!(layers, [0, 12, 23]);
    let directions: Vec<&[f32]> = layers
        .iter()
        .map(|l| &vector[*l as usize * 2048..(*l as usize + 1) * 2048])
        .collect();
    let started = std::time::Instant::now();
    let scores = lab.scores(&directions);
    eprintln!("project: {} layers in {:?}", layers.len(), started.elapsed());
    // The time of 1 layer and of all 24 layers, for the record of docs/lab.md. A build with --release measures it.
    let one = std::time::Instant::now();
    lab.scores(&directions[..1]);
    let one = one.elapsed();
    let every: Vec<&[f32]> = vector.chunks(2048).collect();
    let all = std::time::Instant::now();
    lab.scores(&every);
    eprintln!(
        "project: 1 layer in {one:?}, {} layers in {:?}",
        every.len(),
        all.elapsed()
    );
    assert_eq!(lab.scores(&directions), scores, "a second product gives the same bits");

    // The table and the norm weight, read here on their own, for the bound of each score.
    let file = nagaya_mk_bmg::safetensors::MappedSafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let view = file.view().unwrap();
    let read = |name: &str| -> Vec<f32> {
        view.tensor_bytes(name)
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| nagaya_mk_bmg::safetensors::bf16_to_f32(u16::from_le_bytes(*b)))
            .collect()
    };
    let (table, norm) = (read("model.embed_tokens.weight"), read("model.embedding_norm.weight"));
    for (k, layer) in layers.iter().enumerate() {
        let want = golden[&format!("layer{layer}.scores")].f32();
        let u: Vec<f32> = directions[k].iter().zip(&norm).map(|(v, g)| v * g).collect();
        let (mut worst, mut worst_ratio) = (0f64, 0f64);
        for (r, (got, want)) in scores[k].iter().zip(&want).enumerate() {
            let gap = (*got as f64 - *want as f64).abs();
            let limit = bound(&table[r * 2048..(r + 1) * 2048], &u, *want);
            assert!(
                gap <= limit,
                "layer {layer} row {r}: {got} and the oracle {want}, past {limit:e}"
            );
            worst = worst.max(gap);
            worst_ratio = worst_ratio.max(gap / limit);
        }
        eprintln!("project layer {layer}: worst gap {worst:e}, {worst_ratio:.3} of the bound");
        // The ends: an id differs from the oracle's only where the two scores of the oracle are within their bounds.
        let (up, down) = ends_of(&scores[k], 64);
        for (name, ours) in [("up", up), ("down", down)] {
            let theirs = golden[&format!("layer{layer}.{name}_ids")].ids();
            for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
                if a.id != *b {
                    let tie = (want[a.id as usize] as f64 - want[*b as usize] as f64).abs();
                    let row = |id: u32| &table[id as usize * 2048..(id as usize + 1) * 2048];
                    let room = bound(row(a.id), &u, want[a.id as usize]) + bound(row(*b), &u, want[*b as usize]);
                    assert!(
                        tie <= room,
                        "layer {layer} {name}[{i}]: id {} and the oracle's {b}, whose scores differ by {tie:e}",
                        a.id
                    );
                    eprintln!("project layer {layer} {name}[{i}]: a tie of {} and {b}", a.id);
                }
            }
        }
    }
    let at = |id: u32| rank(&scores[2], id);
    eprintln!(
        "project layer 23: rank of \" seawater\" {}, of <|im_end|> {}",
        at(62_590),
        at(124_900)
    );
    assert!(
        at(62_590) < 8,
        "the sea vector puts \" seawater\" near the top at layer 23"
    );
}

/// The lens that `tools/lfm3_jlens_fit.py` writes with its own writer: `layers.0` of [3, 3] in F16 for a model of 2
/// layers, with the weight hash `blake3:abab…`.
const LENS_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jlens_small.safetensors");

fn fixture_hash() -> String {
    format!("blake3:{}", "ab".repeat(32))
}

/// A lens file with these tensors and metadata, in the layout of the fit tool: sorted keys, a compact header padded
/// with spaces to 4 KiB, and the tensors in the order of their names.
fn lens_file(tensors: &[(&str, &str, &[usize], Vec<u8>)], metadata: &[(&str, &str)]) -> Vec<u8> {
    let mut header = std::collections::BTreeMap::new();
    let mut offset = 0;
    let mut sorted: Vec<_> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    for (name, dtype, shape, data) in &sorted {
        header.insert(
            name.to_string(),
            serde_json::json!({"data_offsets": [offset, offset + data.len()], "dtype": dtype, "shape": shape}),
        );
        offset += data.len();
    }
    let meta: serde_json::Map<String, serde_json::Value> = metadata
        .iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
        .collect();
    header.insert("__metadata__".to_owned(), serde_json::Value::Object(meta));
    let text = serde_json::to_vec(&header).unwrap();
    let padding = (4096 - (8 + text.len()) % 4096) % 4096;
    let mut out = ((text.len() + padding) as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&text);
    out.resize(8 + text.len() + padding, b' ');
    for (_, _, _, data) in &sorted {
        out.extend_from_slice(data);
    }
    out
}

fn f16s(values: &[f32]) -> Vec<u8> {
    // Each value in these tests is exact in F16: a small integer or a half.
    values
        .iter()
        .flat_map(|v| {
            let bits: u16 = match *v {
                0.0 => 0x0000,
                1.0 => 0x3c00,
                2.0 => 0x4000,
                0.5 => 0x3800,
                -1.0 => 0xbc00,
                other => panic!("{other} is not a value of these tests"),
            };
            bits.to_le_bytes()
        })
        .collect()
}

/// The F16 bytes of `values` with a NaN in place of entry 4.
fn nan_at_4(values: &[f32]) -> Vec<u8> {
    let mut bytes = f16s(values);
    bytes[8..10].copy_from_slice(&0x7e00u16.to_le_bytes());
    bytes
}

const MATRIX: [f32; 9] = [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, -1.0];

fn good_metadata(hash: &str) -> Vec<(&'static str, String)> {
    vec![
        ("model", "lfm3".to_owned()),
        ("nagaya.lens", "jlens".to_owned()),
        ("nagaya.lens.corpus_sha256", "0".repeat(64)),
        ("nagaya.lens.estimator", "fixture".to_owned()),
        ("nagaya.lens.max_seq_len", "128".to_owned()),
        ("nagaya.lens.prompts", "1".to_owned()),
        ("nagaya.lens.target_layer", "1".to_owned()),
        ("weight_hash", hash.to_owned()),
    ]
}

#[test]
fn the_lens_of_the_fit_tool_moves_a_row_by_its_matrix_and_the_last_layer_is_the_identity() {
    let lens = JLens::read(Path::new(LENS_FIXTURE), &fixture_hash(), 2, 3).unwrap();
    assert_eq!(lens.transport(0, &[1.0, 1.0, 1.0]), [3.0, 1.0, -0.5]);
    assert_eq!(lens.transport(0, &[2.0, -1.0, 4.0]), [0.0, -1.0, -3.0]);
    assert_eq!(lens.transport(1, &[2.0, -1.0, 4.0]), [2.0, -1.0, 4.0]);
    let bytes = std::fs::read(LENS_FIXTURE).unwrap();
    assert_eq!(lens.id(), format!("blake3:{}", blake3::hash(&bytes).to_hex()));
}

#[test]
fn the_tests_write_a_lens_with_the_bytes_of_the_fit_tool() {
    let metadata = good_metadata(&fixture_hash());
    let metadata: Vec<(&str, &str)> = metadata.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let ours = lens_file(&[("layers.0", "F16", &[3, 3], f16s(&MATRIX))], &metadata);
    assert_eq!(ours, std::fs::read(LENS_FIXTURE).unwrap());
}

#[test]
fn a_lens_of_other_weights_is_refused_and_the_error_names_both_hashes() {
    let other = format!("blake3:{}", "cd".repeat(32));
    let error = JLens::read(Path::new(LENS_FIXTURE), &other, 2, 3).err().unwrap();
    assert!(matches!(error, LensError::OtherWeights(_)), "{error:?}");
    let error = error.to_string();
    assert!(error.contains(&fixture_hash()) && error.contains(&other), "{error}");
}

/// A case of a lens that does not fit: what it shows, its tensors, its metadata, the layers and the width of the model,
/// and a word of the error.
type Case<'a> = (
    &'a str,
    Vec<(&'a str, &'a str, Vec<usize>, Vec<u8>)>,
    Vec<(String, String)>,
    u32,
    usize,
    &'a str,
);

#[test]
fn a_lens_that_does_not_fit_the_model_is_refused_with_the_reason() {
    let hash = fixture_hash();
    let good = good_metadata(&hash);
    let meta = |edit: Option<(&str, &str)>| -> Vec<(String, String)> {
        good.iter()
            .map(|(k, v)| match edit {
                Some((key, value)) if key == *k => (k.to_string(), value.to_owned()),
                _ => (k.to_string(), v.clone()),
            })
            .collect()
    };
    let m = MATRIX.to_vec();
    let cases: Vec<Case> = vec![
        ("no tensor of a layer", vec![], meta(None), 2, 3, "layers.0"),
        (
            "another shape",
            vec![("layers.0", "F16", vec![3, 2], f16s(&m[..6]))],
            meta(None),
            2,
            3,
            "[3, 2]",
        ),
        (
            "another dtype",
            vec![("layers.0", "BF16", vec![3, 3], f16s(&m))],
            meta(None),
            2,
            3,
            "F16",
        ),
        (
            "the last layer",
            vec![
                ("layers.0", "F16", vec![3, 3], f16s(&m)),
                ("layers.1", "F16", vec![3, 3], f16s(&m)),
            ],
            meta(None),
            2,
            3,
            "layers.1",
        ),
        (
            "another lens",
            vec![("layers.0", "F16", vec![3, 3], f16s(&m))],
            meta(Some(("nagaya.lens", "tuned"))),
            2,
            3,
            "tuned",
        ),
        (
            "another width",
            vec![("layers.0", "F16", vec![3, 3], f16s(&m))],
            meta(None),
            2,
            4,
            "width",
        ),
        (
            "more layers",
            vec![("layers.0", "F16", vec![3, 3], f16s(&m))],
            meta(None),
            3,
            3,
            "layers.1",
        ),
        (
            "a value that is not finite",
            vec![("layers.0", "F16", vec![3, 3], nan_at_4(&m))],
            meta(None),
            2,
            3,
            "finite",
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    for (what, tensors, metadata, layers, width, needle) in cases {
        let tensors: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(n, d, s, b)| (*n, *d, s.as_slice(), b.clone()))
            .collect();
        let metadata: Vec<(&str, &str)> = metadata.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let path = dir.path().join("lens.safetensors");
        std::fs::write(&path, lens_file(&tensors, &metadata)).unwrap();
        let error = JLens::read(&path, &hash, layers, width)
            .err()
            .unwrap_or_else(|| panic!("{what}: no error"));
        assert!(matches!(error, LensError::Malformed(_)), "{what}: {error:?}");
        assert!(error.to_string().contains(needle), "{what}: {error}");
    }
    let missing = dir.path().join("absent.safetensors");
    let error = JLens::read(&missing, &hash, 2, 3).err().unwrap();
    assert!(matches!(error, LensError::Unread(_)), "{error:?}");
    assert!(error.to_string().contains("absent.safetensors"), "{error}");
}

#[test]
fn a_readout_divides_a_residual_row_by_its_root_mean_square_with_the_epsilon() {
    let (_dir, lab) = small(Model::Text);
    let h = [3.0f32, -1.0, 2.0];
    // The mean of the squares is 14 / 3. With the epsilon 0.25 of the config, the row divides by its root.
    let inverse = 1.0 / ((14.0f64 / 3.0) + 0.25).sqrt();
    let scaled: Vec<f32> = h.iter().map(|x| (*x as f64 * inverse) as f32).collect();
    assert_eq!(lab.readout(&[&h]), lab.scores(&[&scaled]));
    assert_ne!(lab.readout(&[&h]), lab.scores(&[&h]));
}

/// The backward lens is the transpose: `u · (J h) = (Jᵀ u) · h` for each pair, and the last layer is the identity
/// (docs/lab.md, "compose").
#[test]
fn the_backward_lens_is_the_transpose_of_the_lens() {
    let lens = JLens::read(Path::new(LENS_FIXTURE), &fixture_hash(), 2, 3).unwrap();
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>();
    for u in [[1.0f32, 2.0, -1.0], [0.5, 0.0, 3.0]] {
        for h in [[1.0f32, 1.0, 1.0], [2.0, -1.0, 4.0], [0.0, 1.0, 0.0]] {
            let forward = dot(&u, &lens.transport(0, &h));
            let backward = dot(&lens.transport_back(0, &u), &h);
            assert!(
                (forward - backward).abs() < 1e-6,
                "u {u:?}, h {h:?}: {forward} and {backward}"
            );
        }
    }
    assert_eq!(lens.transport_back(1, &[2.0, -1.0, 4.0]), [2.0, -1.0, 4.0]);
}

#[test]
fn a_head_row_is_the_row_of_the_table_and_the_norm_is_its_weight() {
    let (_dir, lab) = small(Model::Text);
    assert_eq!(lab.head_row(4), [-1.0, 0.5, 2.0]);
    assert_eq!(lab.head_row(1), [0.0, 1.0, 0.0]);
    assert_eq!(lab.norm(), [2.0, 0.5, -1.0]);
}
