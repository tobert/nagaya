//! The errors of the family interface.

use gogi_proto::Invalid;
use nagaya_council::family::{Family, FamilyError, ReadPlan, ReplyParser};

#[test]
fn a_client_error_and_an_internal_error_are_told_apart() {
    assert!(FamilyError::client("bad").is_client());
    assert!(FamilyError::client_param("letters", "bad").is_client());
    assert!(!FamilyError::internal("defect").is_client());
}

#[test]
fn an_invalid_from_gogi_is_a_client_error_with_its_param() {
    let e: FamilyError = Invalid::param("queue", "the queue tool does not match").into();
    assert_eq!(
        e,
        FamilyError::Client(Invalid::param("queue", "the queue tool does not match"))
    );
    assert_eq!(e.to_string(), "the queue tool does not match");
}

#[test]
fn an_internal_error_says_that_the_library_failed() {
    assert!(FamilyError::internal("x").to_string().contains("internal error"));
}

/// The traits are object safe: this function compiles only if they are.
#[allow(dead_code)]
fn object_safe(_: &dyn Family, _: &mut dyn ReadPlan, _: Box<dyn ReplyParser>) {}
