//! The positions of the media parts of a rendering (spec 11.11).
//!
//! The template writes one placeholder token for each media part, between `<|vision_start|>` and `<|vision_end|>`
//! for an image or a clip that the model reads. Liquid's processor puts the placeholder there as many times as the
//! encoder gives rows, or as the clip has frames (`processing_lfm3_omni.py`). This module does the same with the
//! counts that the caller gives. The counts come from the media preprocessing (`crate::media`).

use nagaya_council::family::{Boundary, FamilyError, Rendering};

use super::Lfm3Family;
use super::messages::{MediaKind, MediaPart};

/// One piece of the expansion of a media part. An image or a clip is one span. A video is the text of each pair's
/// time, its markers, its span of video tokens and its span of sound (spec 11.11, "The video of a request").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    /// Tokens of text or markers, which are not a span.
    Tokens(Vec<u32>),
    /// `count` positions of the placeholder `token`: one span.
    Span { token: u32, count: usize },
}

/// The positions of one span of a media part in a rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaSpan {
    /// The index of the message.
    pub message: usize,
    /// The index of the part in the message's content.
    pub part: usize,
    pub kind: MediaKind,
    /// The index of the span among the spans of its part: 0 for an image or a clip.
    pub piece: usize,
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
    let layouts: Vec<Vec<Piece>> = parts
        .iter()
        .zip(counts)
        .map(|(part, &count)| {
            vec![Piece::Span {
                token: family.placeholder_token(part.kind),
                count,
            }]
        })
        .collect();
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
    expand_layouts(family, rendering, parts, &layouts)
}

/// Put each placeholder of `rendering` as the pieces of `layouts[i]`: its tokens, and a span for each span piece.
/// A layout list of the wrong length, a span of no positions, and a placeholder that does not match its part are
/// defects of the caller or of the template.
pub(crate) fn expand_layouts(
    family: &Lfm3Family,
    rendering: Rendering,
    parts: &[MediaPart],
    layouts: &[Vec<Piece>],
) -> Result<MediaRendering, FamilyError> {
    if layouts.len() != parts.len() {
        return Err(FamilyError::internal(format!(
            "the context has {} media parts, and the caller gave {} layouts",
            parts.len(),
            layouts.len()
        )));
    }
    let mut tokens = Vec::with_capacity(rendering.tokens.len());
    // For each position of the template's tokens, the position in the expanded tokens.
    let mut moved = Vec::with_capacity(rendering.tokens.len() + 1);
    let mut spans = Vec::with_capacity(parts.len());
    let mut n = 0usize;
    for &token in &rendering.tokens {
        moved.push(tokens.len());
        let Some(kind) = family.placeholder_kind(token) else {
            tokens.push(token);
            continue;
        };
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
        let mut piece = 0usize;
        for p in &layouts[n] {
            match p {
                Piece::Tokens(t) => tokens.extend_from_slice(t),
                Piece::Span { token, count } => {
                    if *count == 0 {
                        return Err(FamilyError::internal(format!(
                            "media part {n} has a span of 0 positions. A span takes one position or more."
                        )));
                    }
                    spans.push((
                        n,
                        MediaSpan {
                            message: part.message,
                            part: part.part,
                            kind,
                            piece,
                            token: *token,
                            first: tokens.len(),
                            count: *count,
                        },
                    ));
                    tokens.extend(std::iter::repeat_n(*token, *count));
                    piece += 1;
                }
            }
        }
        n += 1;
    }
    moved.push(tokens.len());
    if n != parts.len() {
        return Err(FamilyError::internal(format!(
            "the context has {} media parts, and the template wrote {n} placeholder tokens",
            parts.len()
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
        media: spans
            .iter()
            .map(|(n, span)| nagaya_council::family::MediaSpan {
                first: span.first,
                count: span.count,
                mime: family.recipe_mime(&parts[*n]),
                id: parts[*n].id.clone(),
            })
            .collect(),
    };
    Ok(MediaRendering {
        rendering,
        spans: spans.into_iter().map(|(_, span)| span).collect(),
    })
}
