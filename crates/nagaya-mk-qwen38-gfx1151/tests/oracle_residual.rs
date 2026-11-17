//! The reader of the oracle of the residual, with no GPU (milestone 34).
//!
//! The tests build a small fixture in a temporary directory, as the tool `tools/oracle_fixture.py --residual` writes
//! it. They check that the reader accepts a good one and refuses each fault. A test that reads the real fixture skips
//! when its file is absent.

#[path = "support/oracle.rs"]
mod oracle;

use std::path::Path;

use oracle::{FOLD_WIDTH, RESIDUAL_LAYERS, RESIDUAL_WIDTH, ResidualFixture, SAMPLE};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// A value of the made-up rows: not a constant, so that a swapped layer or index shows.
fn value(layer: usize, j: usize) -> f32 {
    (((j * 7 + layer * 13) % 101) as f32 - 50.0) / 8.0 + 0.5
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn indices(width: usize) -> Vec<usize> {
    (0..SAMPLE).map(|i| i * (width / SAMPLE) + 3).collect()
}

fn norm(values: &[f32]) -> f64 {
    values.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt()
}

/// The rows of one made-up run, as the tool writes them.
struct Made {
    residual: Vec<u8>,
    fold: Vec<u8>,
    json: Value,
}

fn made_run(name: &str, add: bool) -> Made {
    let mut residual = Vec::new();
    let mut rows = Vec::new();
    for layer in 0..RESIDUAL_LAYERS {
        let values: Vec<f32> = (0..RESIDUAL_WIDTH).map(|j| value(layer, j)).collect();
        let raw = bytes_of(&values);
        rows.push(json!({
            "layer": layer,
            "stream_norms": (0..4).map(|s| norm(&values[s * FOLD_WIDTH..(s + 1) * FOLD_WIDTH])).collect::<Vec<_>>(),
            "sample": indices(RESIDUAL_WIDTH).iter().map(|i| f64::from(values[*i])).collect::<Vec<_>>(),
            "sha256": hex(&raw),
        }));
        residual.extend(raw);
    }
    let fold_values: Vec<f32> = (0..FOLD_WIDTH).map(|j| value(99, j)).collect();
    let fold = bytes_of(&fold_values);
    let json = json!({
        "name": name,
        "prompt": "plain",
        "text": "The capital of France is",
        "ids": [760, 6511, 314, 9338, 369],
        "add": add,
        "residual_file": format!("residual.{name}.f32"),
        "fold_file": format!("fold.{name}.f32"),
        "rows": rows,
        "fold": {
            "norm": norm(&fold_values),
            "sample": indices(FOLD_WIDTH).iter().map(|i| f64::from(fold_values[*i])).collect::<Vec<_>>(),
            "sha256": hex(&fold),
        },
        "top_ids": (0..64).collect::<Vec<_>>(),
        "top_logits": (0..64).map(|i| 20.0 - f64::from(i)).collect::<Vec<_>>(),
        "logsumexp": 21.0,
    });
    Made { residual, fold, json }
}

/// A whole made-up fixture: the JSON and the bytes of the rows. The three runs hold the same rows under three names.
struct Fixture {
    json: Value,
    residual: Vec<u8>,
    fold: Vec<u8>,
}

fn made() -> Fixture {
    let plain = made_run("plain", false);
    let chat = made_run("chat", false);
    let added = made_run("plain_add", true);
    let json = json!({
        "format": 1,
        "kind": "residual",
        "dtype": "float32",
        "layers": 48,
        "width": 10240,
        "streams": 4,
        "fold_width": 2560,
        "reference_sha": "0123abc",
        "vocab": 248320,
        "sample_indices": indices(RESIDUAL_WIDTH),
        "fold_sample_indices": indices(FOLD_WIDTH),
        "add": {
            "formula": "v[l][j] = (((31*j + 17*l) mod 257) - 128) / 256",
            "strength": 0.5,
            "strength_f32": 0.5,
            "layers": [8, 24, 40],
            "run": "plain_add",
            "where": "every position of the run, after the layer and before the record",
        },
        "runs": [plain.json, chat.json, added.json],
    });
    Fixture {
        json,
        residual: plain.residual,
        fold: plain.fold,
    }
}

fn parse(json: &Value) -> Result<ResidualFixture, String> {
    oracle::parse_residual(&json.to_string())
}

#[test]
fn a_good_fixture_is_accepted() {
    let fixture = made();
    let parsed = parse(&fixture.json).expect("a good fixture parses");
    assert_eq!(parsed.runs.len(), 3);
    assert_eq!(parsed.reference_sha, "0123abc");
    assert_eq!(parsed.add.layers, [8, 24, 40]);
    assert_eq!(parsed.add.strength_f32, 0.5);
    assert_eq!(parsed.run("plain_add").rows.len(), 48);
    assert_eq!(parsed.run("plain").rows[7].layer, 7);
    assert_eq!(
        oracle::rows_problem(&parsed, parsed.run("plain"), &fixture.residual, &fixture.fold),
        None
    );
}

#[test]
fn a_missing_layer_is_refused() {
    let mut fixture = made();
    // A row is gone, so the list holds 47 layers.
    fixture.json["runs"][0]["rows"].as_array_mut().expect("rows").remove(7);
    let error = parse(&fixture.json).expect_err("47 layers");
    assert!(error.contains("run plain") && error.contains("47 layers"), "{error}");

    // The count is right, but layer 7 holds the number 9: a layer is missing and another comes twice.
    let mut fixture = made();
    fixture.json["runs"][1]["rows"][7]["layer"] = json!(9);
    let error = parse(&fixture.json).expect_err("layer 7 is missing");
    assert!(
        error.contains("run chat") && error.contains("layer 7 is missing"),
        "{error}"
    );
}

#[test]
fn a_wrong_count_is_refused() {
    let mut fixture = made();
    fixture.json["layers"] = json!(47);
    let error = parse(&fixture.json).expect_err("layers");
    assert!(
        error.contains("layers is 47") && error.contains("the engine has 48"),
        "{error}"
    );

    let mut fixture = made();
    fixture.json["width"] = json!(2560);
    assert!(parse(&fixture.json).expect_err("width").contains("width is 2560"));

    let mut fixture = made();
    fixture.json["sample_indices"].as_array_mut().expect("indices").pop();
    assert!(parse(&fixture.json).expect_err("indices").contains("63 indices"));

    let mut fixture = made();
    fixture.json["runs"][2]["rows"][3]["sample"]
        .as_array_mut()
        .expect("sample")
        .pop();
    assert!(parse(&fixture.json).expect_err("sample").contains("63 values"));

    let mut fixture = made();
    fixture.json["runs"].as_array_mut().expect("runs").pop();
    assert!(parse(&fixture.json).expect_err("runs").contains("plain_add is missing"));

    let mut fixture = made();
    fixture.json["runs"][1]["name"] = json!("plain");
    let error = parse(&fixture.json).expect_err("a name twice");
    assert!(
        error.contains("comes twice") || error.contains("chat is missing"),
        "{error}"
    );
}

#[test]
fn a_wrong_dtype_is_refused() {
    let mut fixture = made();
    fixture.json["dtype"] = json!("bfloat16");
    let error = parse(&fixture.json).expect_err("bfloat16");
    assert!(error.contains("dtype is bfloat16, not float32"), "{error}");
}

#[test]
fn a_missing_field_is_refused() {
    for key in ["dtype", "reference_sha", "add", "sample_indices", "runs"] {
        let mut fixture = made();
        fixture.json.as_object_mut().expect("object").remove(key);
        let error = parse(&fixture.json).expect_err(key);
        assert!(error.contains(key), "{key}: {error}");
    }
}

#[test]
fn a_bad_hash_in_the_json_is_refused() {
    let mut fixture = made();
    fixture.json["runs"][0]["rows"][2]["sha256"] = json!("abc");
    let error = parse(&fixture.json).expect_err("a short hash");
    assert!(
        error.contains("layer 2") && error.contains("64 lowercase hex digits"),
        "{error}"
    );
}

#[test]
fn only_the_named_run_has_the_add() {
    let mut fixture = made();
    fixture.json["runs"][0]["add"] = json!(true);
    assert!(
        parse(&fixture.json)
            .expect_err("two adds")
            .contains("only the run plain_add has the add")
    );
}

#[test]
fn a_file_name_must_be_plain() {
    let mut fixture = made();
    fixture.json["runs"][0]["residual_file"] = json!("../residual.plain.f32");
    assert!(
        parse(&fixture.json)
            .expect_err("a path")
            .contains("not a plain file name")
    );
    for name in [
        "sub/residual.plain.f32",
        "sub\\residual.plain.f32",
        ".residual.plain.f32",
    ] {
        let mut fixture = made();
        fixture.json["runs"][0]["fold_file"] = json!(name);
        let error = parse(&fixture.json).expect_err(name);
        assert!(error.contains("not a plain file name"), "{name}: {error}");
    }
}

#[test]
fn a_row_file_of_the_wrong_size_is_refused() {
    let fixture = made();
    let parsed = parse(&fixture.json).expect("parse");
    let run = parsed.run("plain");
    let short = &fixture.residual[..fixture.residual.len() - 4];
    let error = oracle::rows_problem(&parsed, run, short, &fixture.fold).expect("a short file");
    assert!(
        error.contains("residual.plain.f32") && error.contains("1966076 bytes"),
        "{error}"
    );
    let short = &fixture.fold[..fixture.fold.len() - 4];
    let error = oracle::rows_problem(&parsed, run, &fixture.residual, short).expect("a short fold");
    assert!(
        error.contains("fold.plain.f32") && error.contains("10236 bytes"),
        "{error}"
    );
    let mut long = fixture.residual.clone();
    long.extend([0u8; 4]);
    assert!(oracle::rows_problem(&parsed, run, &long, &fixture.fold).is_some());
}

#[test]
fn a_hash_that_does_not_match_its_row_is_refused() {
    let fixture = made();
    let parsed = parse(&fixture.json).expect("parse");
    let run = parsed.run("plain");
    let mut residual = fixture.residual.clone();
    // One bit of layer 5, at an index that the sample does not hold.
    residual[5 * RESIDUAL_WIDTH * 4 + 100 * 4] ^= 1;
    let error = oracle::rows_problem(&parsed, run, &residual, &fixture.fold).expect("a changed row");
    assert!(error.contains("layer 5") && error.contains("SHA-256"), "{error}");
    let mut fold = fixture.fold.clone();
    fold[100 * 4] ^= 1;
    let error = oracle::rows_problem(&parsed, run, &fixture.residual, &fold).expect("a changed fold");
    assert!(error.contains("fold.plain.f32") && error.contains("SHA-256"), "{error}");
}

#[test]
fn a_sample_that_the_file_does_not_hold_is_refused() {
    let mut fixture = made();
    // The hash is right, but the JSON sample of layer 4 differs from the file at its first index.
    fixture.json["runs"][0]["rows"][4]["sample"][0] = json!(123.0);
    let parsed = parse(&fixture.json).expect("parse");
    let error =
        oracle::rows_problem(&parsed, parsed.run("plain"), &fixture.residual, &fixture.fold).expect("a wrong sample");
    assert!(error.contains("layer 4") && error.contains("index 3"), "{error}");
}

#[test]
fn a_norm_that_the_file_does_not_hold_is_refused() {
    let mut fixture = made();
    fixture.json["runs"][0]["rows"][4]["stream_norms"][2] = json!(1.0);
    let parsed = parse(&fixture.json).expect("parse");
    let error =
        oracle::rows_problem(&parsed, parsed.run("plain"), &fixture.residual, &fixture.fold).expect("a wrong norm");
    assert!(error.contains("layer 4") && error.contains("stream 2"), "{error}");
}

#[test]
fn an_absent_file_skips_and_another_fault_fails() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    assert_eq!(oracle::read_optional(&dir.path().join("absent.f32")), Ok(None));
    let file = dir.path().join("here.f32");
    std::fs::write(&file, [1u8, 2, 3]).expect("write");
    assert_eq!(oracle::read_optional(&file), Ok(Some(vec![1, 2, 3])));
    // A directory exists and cannot be read as a file: that is a fault, not an absence.
    let error = oracle::read_optional(dir.path()).expect_err("a directory is not a file");
    assert!(error.contains("cannot read"), "{error}");
}

#[test]
fn the_rows_of_a_run_load_from_a_folder_and_an_absent_file_skips() {
    let fixture = made();
    let parsed = parse(&fixture.json).expect("parse");
    let run = parsed.run("plain");
    let dir = tempfile::tempdir().expect("a temporary directory");
    // Nothing is there: the test skips.
    assert!(oracle::read_rows("a test", dir.path(), &parsed, run, false).is_none());
    std::fs::write(dir.path().join("residual.plain.f32"), &fixture.residual).expect("write");
    // One file is there and the other is not: the test skips.
    assert!(oracle::read_rows("a test", dir.path(), &parsed, run, false).is_none());
    std::fs::write(dir.path().join("fold.plain.f32"), &fixture.fold).expect("write");
    let rows = oracle::read_rows("a test", dir.path(), &parsed, run, false).expect("both files are there");
    assert_eq!(rows.residual.len(), 48 * 10240);
    assert_eq!(rows.fold.len(), 2560);
    assert_eq!(rows.residual[10240 * 3 + 5], value(3, 5));
    assert_eq!(rows.fold[7], value(99, 7));
}

#[test]
#[should_panic(expected = "NAGAYA_REQUIRE_MODELS is set")]
fn an_absent_file_fails_when_the_files_are_required() {
    let fixture = made();
    let parsed = parse(&fixture.json).expect("parse");
    let dir = tempfile::tempdir().expect("a temporary directory");
    oracle::read_rows("a test", dir.path(), &parsed, parsed.run("plain"), true);
}

#[test]
#[should_panic(expected = "SHA-256")]
fn a_bad_file_fails_the_test_and_does_not_skip() {
    let fixture = made();
    let parsed = parse(&fixture.json).expect("parse");
    let dir = tempfile::tempdir().expect("a temporary directory");
    let mut residual = fixture.residual.clone();
    residual[1000] ^= 1;
    std::fs::write(dir.path().join("residual.plain.f32"), residual).expect("write");
    std::fs::write(dir.path().join("fold.plain.f32"), &fixture.fold).expect("write");
    oracle::read_rows("a test", dir.path(), &parsed, parsed.run("plain"), false);
}

#[test]
fn the_real_fixture_parses_when_it_is_there() {
    let Some(fixture) = oracle::residual_fixture("the_real_fixture_parses_when_it_is_there") else {
        return;
    };
    assert_eq!(fixture.runs.len(), 3);
    assert!(!fixture.reference_sha.is_empty());
    if let Some(rows) = oracle::read_rows(
        "the_real_fixture_parses_when_it_is_there",
        &oracle::golden_dir(),
        &fixture,
        fixture.run("plain"),
        oracle::require_files(),
    ) {
        assert_eq!(rows.residual.len(), RESIDUAL_LAYERS * RESIDUAL_WIDTH);
    }
}

#[test]
fn the_default_folder_of_the_rows_is_under_target() {
    // The test does not set the variable. When the variable is set in the environment, it is the folder.
    let dir = oracle::golden_dir();
    match std::env::var_os(oracle::GOLDEN_VARIABLE) {
        Some(set) if !set.is_empty() => assert_eq!(dir, Path::new(&set)),
        _ => assert!(dir.ends_with("target/qwen38-golden"), "{}", dir.display()),
    }
}

// ---- the gate of a layer --------------------------------------------------------------------------------------------

#[test]
fn equal_rows_have_no_gap() {
    let row: Vec<f32> = (0..64).map(|j| value(1, j)).collect();
    assert_eq!(oracle::relative_gap(&row, &row), Ok(0.0));
}

#[test]
fn the_gap_is_the_largest_difference_over_the_rms_of_the_reference() {
    // The reference is 2.0 everywhere, so its rms is 2.0. One value is 0.5 away, so the gap is 0.25.
    let reference = vec![2.0f32; 16];
    let mut engine = reference.clone();
    engine[5] = 2.5;
    engine[9] = 1.75;
    assert_eq!(oracle::relative_gap(&engine, &reference), Ok(0.25));
    // The scale is the reference's: the same difference over a reference of rms 4.0 is 0.125.
    let reference = vec![4.0f32; 16];
    let mut engine = reference.clone();
    engine[0] = 4.5;
    assert_eq!(oracle::relative_gap(&engine, &reference), Ok(0.125));
}

#[test]
fn the_rms_is_the_root_of_the_mean_and_not_the_norm() {
    // Reference [3, 4]: the rms is sqrt(12.5), and the norm is 5.
    let gap = oracle::relative_gap(&[3.0, 5.0], &[3.0, 4.0]).expect("a gap");
    assert!((gap - 1.0 / 12.5f64.sqrt()).abs() < 1e-15, "{gap}");
}

#[test]
fn a_row_that_a_gate_cannot_judge_is_an_error() {
    let ok = vec![1.0f32; 4];
    let nan = vec![1.0, f32::NAN, 1.0, 1.0];
    assert!(oracle::relative_gap(&nan, &ok).expect_err("NaN").contains("index 1"));
    let inf = vec![1.0, 1.0, f32::INFINITY, 1.0];
    assert!(oracle::relative_gap(&inf, &ok).expect_err("Inf").contains("not finite"));
    assert!(
        oracle::relative_gap(&ok, &nan)
            .expect_err("reference NaN")
            .contains("reference")
    );
    assert!(
        oracle::relative_gap(&ok, &[0.0; 4])
            .expect_err("zero")
            .contains("all zero")
    );
    assert!(
        oracle::relative_gap(&ok, &ok[..3])
            .expect_err("length")
            .contains("4 values")
    );
    assert!(oracle::relative_gap(&[], &[]).is_err());
}

#[test]
fn the_gaps_of_the_layers_name_the_layer_that_differs() {
    let reference: Vec<f32> = (0..RESIDUAL_LAYERS * RESIDUAL_WIDTH)
        .map(|i| value(i % 7, i) + 8.0)
        .collect();
    let mut engine = reference.clone();
    engine[13 * RESIDUAL_WIDTH + 200] += 1.0;
    let gaps = oracle::layer_gaps(&engine, &reference).expect("gaps");
    assert_eq!(gaps.len(), 48);
    for (layer, gap) in gaps.iter().enumerate() {
        assert_eq!(*gap > 0.0, layer == 13, "layer {layer}: {gap}");
    }
    assert!(oracle::layer_gaps(&engine[1..], &reference).is_err());
    let mut broken = reference.clone();
    broken[5 * RESIDUAL_WIDTH] = f32::NAN;
    assert!(
        oracle::layer_gaps(&broken, &reference)
            .expect_err("NaN")
            .starts_with("layer 5:")
    );
}

#[test]
fn a_layer_past_the_gate_is_a_breach_and_a_layer_on_the_gate_is_not() {
    let gaps = [0.0, 1e-4, 2e-4, f64::NAN, 3e-4];
    let breaches = oracle::gap_breaches(&gaps, 2e-4);
    assert_eq!(breaches.len(), 2, "{breaches:?}");
    assert!(breaches[0].starts_with("layer 3:"), "{breaches:?}");
    assert!(breaches[1].starts_with("layer 4:"), "{breaches:?}");
    assert!(oracle::gap_breaches(&[0.0, 1e-4], 1e-3).is_empty());
}
