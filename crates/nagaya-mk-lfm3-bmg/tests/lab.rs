//! The unembedding of the lab route `project` on the CPU (`docs/lab.md`): small checkpoints with known values, and the
//! oracle stage `project` of transformers for LFM3-1.7B. No card runs.

mod common;

use std::path::Path;

use common::Which;
use nagaya_mk_bmg::safetensors::SafeTensorsView;
use nagaya_mk_lfm3_bmg::lab::{Entry, Unembedding, ends_of, rank};
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
        Model::Text => ("Lfm3ForCausalLM", serde_json::json!({"num_hidden_layers": 2})),
        Model::Omni => (
            "Lfm3OmniForConditionalGeneration",
            serde_json::json!({"text_config": {"num_hidden_layers": 2}}),
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
        r#"{"architectures": ["Lfm3ForCausalLM"], "num_hidden_layers": 2}"#,
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
