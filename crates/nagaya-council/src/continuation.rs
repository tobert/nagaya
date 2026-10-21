//! Continuations (gogi "Continuations", spec 11.3c): the build plan of a context whose last message is open.
//!
//! A generation builds the context before the open message, as it builds a context before its generation prompt.
//! Then it feeds the tokens of the open message, as it feeds a generation prompt, and the last of them gives the first
//! logit row. The build seals only at boundaries before the open message. A seal changes bits, so a seal inside the
//! open text would make a continuation differ from the reply that it continues.

use crate::family::{FamilyError, Rendering};

/// The build plan of an open context: the rendering `before` of the context without its open message, when its
/// tokens begin the tokens of `open`. Its end is a boundary, as the end of a context always is.
///
/// When `before` is not a prefix of `open`, the plan is the boundaries of `open` before the open message, and it may
/// be empty. `open` must be the rendering of an open context: no generation prompt after it, and its last boundary at
/// its end. Any other rendering is a defect of the family library: the result is [`FamilyError::Internal`].
pub fn open_plan(open: &Rendering, before: &Rendering) -> Result<Rendering, FamilyError> {
    let len = open.tokens.len();
    let Some((last, marked)) = open
        .boundaries
        .split_last()
        .filter(|(last, _)| open.end == len && last.tokens == len)
    else {
        return Err(FamilyError::internal(format!(
            "the rendering of an open context ends at token {} of {len}, with its last boundary at {:?}; an open \
             message renders in place of the generation prompt, and the end of the context is its last boundary",
            open.end,
            open.boundaries.last().map(|b| b.tokens)
        )));
    };
    let start = marked.last().map_or(0, |b| b.tokens);
    if start >= last.tokens {
        return Err(FamilyError::internal(format!(
            "the open message ends at token {} but starts at token {start}",
            last.tokens
        )));
    }
    let prefix = &before.tokens[..before.end];
    if before.end >= start && before.end < len && open.tokens.starts_with(prefix) && !before.boundaries.is_empty() {
        return Ok(Rendering {
            tokens: prefix.to_vec(),
            end: before.end,
            boundaries: before.boundaries.clone(),
            control_text: Vec::new(),
        });
    }
    Ok(Rendering {
        tokens: open.tokens[..start].to_vec(),
        end: start,
        boundaries: marked.to_vec(),
        control_text: Vec::new(),
    })
}
