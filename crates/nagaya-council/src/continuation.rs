//! Continuations (gogi "Continuations", spec 11.3c): the build plan of a context whose last message is open.
//!
//! The engine gives a logit row only for a token that it feeds (engine-abi, "Logit outputs"). A snapshot holds no
//! logit row. Thus a generation after an open message builds all the tokens but the last, and then feeds the last
//! token for the first logit row. The seam boundary marks the end of that build. Its snapshot holds the model state
//! before the last token of the open text.

use crate::family::{Boundary, FamilyError, Rendering};

/// The rendering that a build of an open context makes: the tokens of `open` but the last, with the seam boundary in
/// place of the end boundary. The seam boundary has the index of the open message.
///
/// `open` must be the rendering of an open context: no generation prompt, and a last boundary at its end that is
/// more than one token after the boundary before it. Any other rendering is a defect of the family library: the
/// result is [`FamilyError::Internal`].
pub fn seam_rendering(open: &Rendering) -> Result<Rendering, FamilyError> {
    let len = open.tokens.len();
    let (last, before) = match open.boundaries.split_last() {
        Some((last, before)) if open.end == len && last.tokens == len => (last, before),
        _ => {
            return Err(FamilyError::internal(format!(
                "the rendering of an open context ends at token {} of {len}, with its last boundary at {:?}; an open \
                 message renders in place of the generation prompt, and the end of the context is its last boundary",
                open.end,
                open.boundaries.last().map(|b| b.tokens)
            )));
        }
    };
    let seam = len.saturating_sub(1);
    if seam == 0 || before.last().is_some_and(|b| b.tokens >= seam) {
        return Err(FamilyError::internal(format!(
            "the open message of {len} tokens starts at token {}; an open message renders its start of turn and \
             its open text, so it has at least two tokens",
            before.last().map_or(0, |b| b.tokens)
        )));
    }
    let mut boundaries = before.to_vec();
    boundaries.push(Boundary {
        message: last.message,
        tokens: seam,
    });
    Ok(Rendering {
        tokens: open.tokens[..seam].to_vec(),
        end: seam,
        boundaries,
        control_text: open.control_text.clone(),
    })
}
