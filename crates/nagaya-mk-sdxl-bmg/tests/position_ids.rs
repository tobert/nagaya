//! The text encoders of the inpainting checkpoint (spec 11.9b) hold `text_model.embeddings.position_ids`, I64 of shape
//! [1, 77], as older exports of transformers do. The engine gives each position its index, so the loader drops the
//! tensor when it holds 0 to 76. Other values would change the encoder, so they are an error. No card or model needed.

use nagaya_mk_bmg::safetensors::{MappedSafeTensors, SafeTensorsView};
use nagaya_mk_sdxl_bmg::weights::items;

const NAME: &str = "text_model.embeddings.position_ids";

/// A text encoder file with `position_ids` of these values and one F16 weight, written to a temporary directory.
fn encoder(ids: &[i64], dir: &std::path::Path) -> MappedSafeTensors {
    let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let weight = [0u8; 2 * 768];
    let file = SafeTensorsView::build_synthetic_safetensors(&[
        (NAME, "I64", &[1, ids.len()], &bytes),
        ("text_model.final_layer_norm.weight", "F16", &[768], &weight),
    ]);
    let path = dir.join("model.safetensors");
    std::fs::write(&path, file).unwrap();
    MappedSafeTensors::open(&path).unwrap()
}

#[test]
fn the_loader_drops_position_ids_of_0_to_76() {
    let dir = tempfile::tempdir().unwrap();
    let ids: Vec<i64> = (0..77).collect();
    let maps = [("te1.", encoder(&ids, dir.path()))];
    let all = items(&maps, false).unwrap();
    let names: Vec<&str> = all.iter().map(|item| item.name.as_str()).collect();
    assert_eq!(names, ["te1.text_model.final_layer_norm.weight"]);
}

#[test]
fn position_ids_of_other_values_are_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut ids: Vec<i64> = (0..77).collect();
    ids.swap(3, 4);
    let maps = [("te2.", encoder(&ids, dir.path()))];
    let error = items(&maps, false).err().expect("the loader refuses other positions");
    assert!(error.contains("te2.text_model.embeddings.position_ids"), "{error}");
    let short: Vec<i64> = (0..76).collect();
    let maps = [("te1.", encoder(&short, dir.path()))];
    assert!(items(&maps, false).is_err(), "76 positions are an error");
}
