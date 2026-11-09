//! Control vectors with no I/O (spec 11.12): the recipe bytes of a control list, the root recipe that takes them, and
//! the vector of the means of two prompt sets.

use nagaya_abi::ControlOp;
use nagaya_council::build::{root_recipe, root_recipe_with_control};
use nagaya_council::control::{ControlEntry, recipe_bytes, vector_of_means};
use nagaya_store::{ContentId, recipe_root};

fn id(byte: u8) -> ContentId {
    ContentId::from_hex(&format!("{byte:02x}").repeat(32)).unwrap()
}

fn entry(byte: u8, op: ControlOp, strength: f64) -> ControlEntry {
    ControlEntry {
        vector: id(byte),
        op,
        strength,
        layers: None,
    }
}

#[test]
fn an_empty_list_has_no_bytes_and_keeps_the_root_of_a_plain_context() {
    assert!(recipe_bytes(&[]).is_empty());
    let digest = [7u8; 32];
    assert_eq!(root_recipe_with_control(&digest, b"g", &[]), root_recipe(&digest, b"g"));
    assert_eq!(root_recipe(&digest, b"g"), recipe_root(&digest, b"g", &[], &[]));
}

#[test]
fn the_bytes_are_the_count_then_the_id_the_op_byte_and_the_f64_strength_of_each_item() {
    let list = [entry(0xab, ControlOp::Add, 4.0), entry(0x01, ControlOp::Clamp, -0.5)];
    let bytes = recipe_bytes(&list);
    let mut want = 2u64.to_le_bytes().to_vec();
    want.extend([0xab; 32]);
    want.push(1);
    want.extend(4.0f64.to_le_bytes());
    want.extend([0x01; 32]);
    want.push(3);
    want.extend((-0.5f64).to_le_bytes());
    assert_eq!(bytes, want);
    assert_eq!(bytes.len(), 8 + 2 * (32 + 1 + 8));
    assert_eq!(recipe_bytes(&[entry(0, ControlOp::Ablate, 1.0)])[8 + 32], 2);
}

/// An item with layers sets the bit 0x80 of its op byte, and its layers follow its strength (spec 11.12, "The
/// strength of each layer"). An item without layers keeps its bytes, and other layers give another root.
#[test]
fn an_item_with_layers_marks_its_op_byte_and_writes_its_layers_after_its_strength() {
    let mut layered = entry(0xab, ControlOp::Ablate, 2.0);
    layered.layers = Some(vec![(3, 1.5), (10, 0.0)]);
    let list = [layered.clone(), entry(0x01, ControlOp::Add, 1.0)];
    let bytes = recipe_bytes(&list);
    let mut want = 2u64.to_le_bytes().to_vec();
    want.extend([0xab; 32]);
    want.push(0x82);
    want.extend(2.0f64.to_le_bytes());
    want.extend(2u32.to_le_bytes());
    want.extend(3u32.to_le_bytes());
    want.extend(1.5f64.to_le_bytes());
    want.extend(10u32.to_le_bytes());
    want.extend(0.0f64.to_le_bytes());
    want.extend([0x01; 32]);
    want.push(1);
    want.extend(1.0f64.to_le_bytes());
    assert_eq!(bytes, want);
    let digest = [3u8; 32];
    let root = |list: &[ControlEntry]| root_recipe_with_control(&digest, b"", list);
    let mut other = layered.clone();
    other.layers = Some(vec![(3, 1.5), (11, 0.0)]);
    assert_ne!(root(&[layered.clone()]), root(&[other]));
    assert_ne!(root(&[layered]), root(&[entry(0xab, ControlOp::Ablate, 2.0)]));
}

#[test]
fn each_part_of_a_list_changes_the_root_and_a_list_of_strength_0_is_not_plain() {
    let digest = [3u8; 32];
    let root = |list: &[ControlEntry]| root_recipe_with_control(&digest, b"", list);
    let base = root(&[entry(1, ControlOp::Add, 4.0), entry(2, ControlOp::Ablate, 1.0)]);
    assert_eq!(
        base,
        root(&[entry(1, ControlOp::Add, 4.0), entry(2, ControlOp::Ablate, 1.0)])
    );
    for other in [
        root(&[entry(2, ControlOp::Ablate, 1.0), entry(1, ControlOp::Add, 4.0)]),
        root(&[entry(1, ControlOp::Add, 4.0), entry(3, ControlOp::Ablate, 1.0)]),
        root(&[entry(1, ControlOp::Clamp, 4.0), entry(2, ControlOp::Ablate, 1.0)]),
        root(&[entry(1, ControlOp::Add, 4.5), entry(2, ControlOp::Ablate, 1.0)]),
        root(&[entry(1, ControlOp::Add, 4.0)]),
    ] {
        assert_ne!(other, base);
    }
    // The bits of the model do not change, but the list has its own recipe (spec 11.12).
    assert_ne!(root(&[entry(1, ControlOp::Add, 0.0)]), root_recipe(&digest, b""));
}

#[test]
fn the_vector_is_the_mean_of_the_positive_rows_less_the_mean_of_the_negative_rows() {
    let positive: [&[f32]; 2] = [&[1.0, 2.0], &[3.0, 6.0]];
    let negative: [&[f32]; 1] = [&[0.5, -1.0]];
    assert_eq!(vector_of_means(&positive, &negative).unwrap(), vec![1.5, 5.0]);
}

#[test]
fn the_sums_are_f64_so_a_small_row_after_a_large_one_is_kept() {
    // In f32, 16,777,216 + 1 + 1 is 16,777,216. In f64 it is 16,777,218.
    let positive: [&[f32]; 3] = [&[16_777_216.0], &[1.0], &[1.0]];
    let negative: [&[f32]; 1] = [&[0.0]];
    let got = vector_of_means(&positive, &negative).unwrap();
    assert_eq!(got, vec![(16_777_218.0f64 / 3.0) as f32]);
    assert_ne!(got, vec![16_777_216.0f32 / 3.0]);
}

#[test]
fn an_empty_set_or_rows_of_other_widths_are_refused() {
    let row: &[f32] = &[1.0, 2.0];
    let short: &[f32] = &[1.0];
    assert!(vector_of_means(&[], &[row]).unwrap_err().contains("positive"));
    assert!(vector_of_means(&[row], &[]).unwrap_err().contains("negative"));
    let error = vector_of_means(&[row, short], &[row]).unwrap_err();
    assert!(error.contains("positive row 1"), "{error}");
    let error = vector_of_means(&[row], &[short]).unwrap_err();
    assert!(error.contains("negative row 0"), "{error}");
    let long: &[f32] = &[1.0, 2.0, 3.0];
    let error = vector_of_means(&[row], &[long]).unwrap_err();
    assert!(error.contains("negative row 0 has 3 values"), "{error}");
}

#[test]
fn a_row_that_is_not_finite_is_refused() {
    let row: &[f32] = &[1.0, 2.0];
    let bad: &[f32] = &[1.0, f32::NAN];
    let error = vector_of_means(&[row], &[bad]).unwrap_err();
    assert!(error.contains("not finite"), "{error}");
}
