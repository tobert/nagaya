//! Tests for the exact comparator (spec 8.3, ds4-recipe 01 "What exact means").

use nagaya_conformance::{Difference, compare_bytes, compare_f32_rows};

#[test]
fn equal_bit_patterns_pass() {
    let left = [1.0f32, -2.5, 0.0];
    let right = [1.0f32, -2.5, 0.0];
    assert!(compare_f32_rows("compare", &left, &right, 3).is_ok());
}

#[test]
fn signed_zeros_are_different() {
    let error = compare_f32_rows("compare", &[0.0], &[-0.0], 1).unwrap_err();
    assert_eq!(error.difference.unwrap().left, 0);
    assert_eq!(error.difference.unwrap().right, 0x8000_0000);
}

#[test]
fn a_nan_fails_even_when_both_bit_patterns_are_equal() {
    let nan = f32::from_bits(0x7fc0_0001);
    let error = compare_f32_rows("compare", &[nan], &[nan], 1).unwrap_err();
    assert!(error.message.contains("not finite"), "{error}");
}

#[test]
fn an_infinity_fails() {
    let error = compare_f32_rows("compare", &[f32::INFINITY], &[f32::INFINITY], 1).unwrap_err();
    assert!(error.message.contains("not finite"), "{error}");
    let error = compare_f32_rows("compare", &[f32::NEG_INFINITY], &[f32::NEG_INFINITY], 1).unwrap_err();
    assert!(error.message.contains("not finite"), "{error}");
}

#[test]
fn row_counts_must_match() {
    let error = compare_f32_rows("compare", &[1.0, 2.0], &[1.0], 1).unwrap_err();
    assert!(error.message.contains("counts differ"), "{error}");
}

#[test]
fn an_empty_dump_fails() {
    let error = compare_f32_rows("compare", &[], &[], 1).unwrap_err();
    assert!(error.message.contains("empty"), "{error}");
}

#[test]
fn a_partial_row_fails() {
    let error = compare_f32_rows("compare", &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 2).unwrap_err();
    assert!(error.message.contains("partial"), "{error}");
}

#[test]
fn the_first_difference_names_the_row_the_index_and_the_count() {
    let left = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = [1.0f32, 2.0, 9.0, 4.0, 8.0, 7.0];
    let error = compare_f32_rows("compare", &left, &right, 3).unwrap_err();
    assert_eq!(
        error.difference,
        Some(Difference {
            row: 0,
            index: 2,
            left: 3.0f32.to_bits(),
            right: 9.0f32.to_bits(),
            differing: 1,
        })
    );
}

#[test]
fn a_later_row_reports_its_own_index() {
    let left = [1.0f32, 2.0, 3.0, 4.0];
    let right = [1.0f32, 2.0, 3.0, 4.5];
    let error = compare_f32_rows("compare", &left, &right, 2).unwrap_err();
    assert_eq!(error.difference.unwrap().row, 1);
    assert_eq!(error.difference.unwrap().index, 1);
}

#[test]
fn bytes_compare_by_value_and_count() {
    assert!(compare_bytes("compare", &[1, 2, 3], &[1, 2, 3]).is_ok());
    let error = compare_bytes("compare", &[1, 2, 3], &[1, 2, 4]).unwrap_err();
    assert_eq!(
        error.difference,
        Some(Difference {
            row: 0,
            index: 2,
            left: 3,
            right: 4,
            differing: 1,
        })
    );
    let error = compare_bytes("compare", &[1, 2], &[1, 2, 3]).unwrap_err();
    assert!(error.message.contains("counts differ"), "{error}");
    let error = compare_bytes("compare", &[], &[]).unwrap_err();
    assert!(error.message.contains("empty"), "{error}");
}
