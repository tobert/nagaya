//! The reply parser of the Qwen3.8 family.

use gogi_proto::context::Tool;
use nagaya_council::family::{FamilyError, ReplyParser};

use super::Qwen38Family;

pub(crate) fn reply_parser(_: &Qwen38Family, _: bool, _: &[Tool]) -> Result<Box<dyn ReplyParser>, FamilyError> {
    Err(FamilyError::internal("the Qwen3.8 reply parser is not built yet"))
}
