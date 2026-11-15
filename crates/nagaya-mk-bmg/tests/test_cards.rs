//! The cards of a card test come from `NAGAYA_TEST_CARDS`. The reader of the variable needs no card.

use nagaya_mk_bmg::test_cards::{VARIABLE, parse};

/// A test takes the first addresses of the list, in the order of the list.
#[test]
fn a_test_takes_the_first_addresses_of_the_list() {
    let two = Some("0000:03:00.0,0000:08:00.0");
    assert_eq!(parse(two, 1), Ok(vec!["0000:03:00.0".to_owned()]));
    assert_eq!(
        parse(two, 2),
        Ok(vec!["0000:03:00.0".to_owned(), "0000:08:00.0".to_owned()])
    );
    assert_eq!(parse(Some(" 0000:08:00.0 "), 1), Ok(vec!["0000:08:00.0".to_owned()]));
}

/// A card test runs on no card that the variable does not name: an unset variable is an error, with what to do.
#[test]
fn an_unset_variable_is_an_error_that_names_the_variable() {
    for value in [None, Some(""), Some(" ")] {
        let error = parse(value, 1).unwrap_err();
        assert!(error.contains(VARIABLE), "{error}");
        assert!(error.contains("is not set") && error.contains("PCI address"), "{error}");
    }
}

/// A test of two cards does not run on one card, and it does not take one card two times.
#[test]
fn too_few_addresses_and_a_repeated_address_are_errors() {
    let error = parse(Some("0000:03:00.0"), 2).unwrap_err();
    assert!(error.contains("2 cards") && error.contains("holds 1"), "{error}");
    let error = parse(Some("0000:03:00.0,0000:03:00.0"), 2).unwrap_err();
    assert!(error.contains("two times"), "{error}");
}

/// An address has the form of the bootstrap config: `dddd:bb:dd.f` in hex.
#[test]
fn a_value_that_is_not_a_pci_address_is_an_error() {
    for value in ["0", "card0", "0000:03:00", "0000:0g:00.0", "03:00.0"] {
        let error = parse(Some(value), 1).unwrap_err();
        assert!(error.contains(value) && error.contains("dddd:bb:dd.f"), "{error}");
    }
}
