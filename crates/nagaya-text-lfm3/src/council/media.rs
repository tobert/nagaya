//! The positions of the media parts of a rendering (spec 11.11).
//!
//! The template writes one placeholder token for each media part, between `<|vision_start|>` and `<|vision_end|>`
//! for an image or a clip that the model reads. Liquid's processor puts the placeholder there as many times as the
//! encoder gives rows, or as the clip has frames (`processing_lfm3_omni.py`). This module does the same with the
//! counts that the caller gives. The counts come from the media preprocessing (`crate::media`).

use nagaya_council::family::{Boundary, FamilyError, Rendering};

use super::Lfm3Family;
use super::messages::{MediaKind, MediaPart};

/// The positions of one media part in a rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaSpan {
    /// The index of the message.
    pub message: usize,
    /// The index of the part in the message's content.
    pub part: usize,
    pub kind: MediaKind,
    /// The placeholder token at each position.
    pub token: u32,
    /// The first position.
    pub first: usize,
    /// The number of positions.
    pub count: usize,
}

/// A rendering with its media parts at their full size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaRendering {
    /// The tokens, the end and the boundaries, with each part at its full size.
    pub rendering: Rendering,
    /// The positions of each media part, in order.
    pub spans: Vec<MediaSpan>,
}

/// Put each placeholder of `rendering` `counts[i]` times. A count of zero, a count list of the wrong length, and a
/// placeholder that does not match its part are defects of the caller or of the template.
pub(crate) fn expand(
    family: &Lfm3Family,
    rendering: Rendering,
    parts: &[MediaPart],
    counts: &[usize],
) -> Result<MediaRendering, FamilyError> {
    if counts.len() != parts.len() {
        return Err(FamilyError::internal(format!(
            "the context has {} media parts, and the caller gave {} counts of positions",
            parts.len(),
            counts.len()
        )));
    }
    if let Some(i) = counts.iter().position(|&c| c == 0) {
        return Err(FamilyError::internal(format!(
            "media part {i} has a count of 0 positions. A part takes one position or more."
        )));
    }
    let mut tokens = Vec::with_capacity(rendering.tokens.len() + counts.iter().sum::<usize>());
    // For each position of the template's tokens, the position in the expanded tokens.
    let mut moved = Vec::with_capacity(rendering.tokens.len() + 1);
    let mut spans = Vec::with_capacity(parts.len());
    for &token in &rendering.tokens {
        moved.push(tokens.len());
        let Some(kind) = family.placeholder_kind(token) else {
            tokens.push(token);
            continue;
        };
        let n = spans.len();
        let Some(part) = parts.get(n) else {
            return Err(FamilyError::internal(
                "the rendering holds more placeholder tokens than the context has media parts. The chat template changed shape.",
            ));
        };
        if part.kind != kind {
            return Err(FamilyError::internal(format!(
                "media part {n} is {:?}, and the template wrote the placeholder of {kind:?}",
                part.kind
            )));
        }
        spans.push(MediaSpan {
            message: part.message,
            part: part.part,
            kind,
            token,
            first: tokens.len(),
            count: counts[n],
        });
        tokens.extend(std::iter::repeat_n(token, counts[n]));
    }
    moved.push(tokens.len());
    if spans.len() != parts.len() {
        return Err(FamilyError::internal(format!(
            "the context has {} media parts, and the template wrote {} placeholder tokens",
            parts.len(),
            spans.len()
        )));
    }
    let at = |p: usize| moved[p];
    let rendering = Rendering {
        end: at(rendering.end),
        boundaries: rendering
            .boundaries
            .iter()
            .map(|b| Boundary {
                message: b.message,
                tokens: at(b.tokens),
            })
            .collect(),
        control_text: rendering.control_text,
        tokens,
    };
    Ok(MediaRendering { rendering, spans })
}
