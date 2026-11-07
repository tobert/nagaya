//! The pure parts of a build (spec 6.2, 11.3c): the recipe of each boundary, and the steps that feed the rest.

use nagaya_abi::Kind;
use nagaya_store::{ContentId, RecipeId, RecipeMedia, recipe_boundary_media, recipe_root};

use crate::family::{Boundary, FamilyError, MediaSpan, Rendering};

/// A boundary and the recipe of the snapshot there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryRecipe {
    /// The boundary.
    pub boundary: Boundary,
    /// The recipe of the context up to the boundary, with the boundaries before it.
    pub recipe: RecipeId,
}

/// The root recipe of a served model: the identity digest and the geometry key, with no approximation and no lab
/// state (spec 6.2).
pub fn root_recipe(identity_digest: &[u8; 32], geometry_key: &[u8]) -> RecipeId {
    recipe_root(identity_digest, geometry_key, &[], &[])
}

/// The recipe at each boundary of a rendering, in order.
///
/// Each recipe covers the recipe before it, the tokens since that boundary, and the position (spec 6.2). Thus a
/// lookup hashes only the tokens since the last boundary, and the boundaries are part of the content.
///
/// The boundaries must increase, and the last one must be at the end of the rendering. Any other rendering is a
/// defect of the family library: the result is [`FamilyError::Internal`].
pub fn boundary_recipes(root: RecipeId, rendering: &Rendering) -> Result<Vec<BoundaryRecipe>, FamilyError> {
    let Some(last) = rendering.boundaries.last() else {
        return Err(FamilyError::internal(
            "a rendering has no boundary: the end of a context is a boundary",
        ));
    };
    if last.tokens != rendering.end || rendering.end > rendering.tokens.len() {
        return Err(FamilyError::internal(format!(
            "the last boundary is at token {}, but the context ends at token {} of {}",
            last.tokens,
            rendering.end,
            rendering.tokens.len()
        )));
    }
    check_media(&rendering.media, rendering.tokens.len())?;
    let mut recipes = Vec::with_capacity(rendering.boundaries.len());
    let mut previous = root;
    let mut position = 0usize;
    for boundary in &rendering.boundaries {
        if boundary.tokens <= position {
            return Err(FamilyError::internal(format!(
                "the boundary of message {} is at token {}, which is not after token {position}",
                boundary.message, boundary.tokens
            )));
        }
        let recipe = range_recipe(
            &previous,
            &rendering.tokens,
            &rendering.media,
            position,
            boundary.tokens,
        )?;
        recipes.push(BoundaryRecipe {
            boundary: *boundary,
            recipe,
        });
        previous = recipe;
        position = boundary.tokens;
    }
    Ok(recipes)
}

/// Check the media spans of `len` tokens: each holds a position, they lie inside the tokens in the order of their
/// positions with no overlap, and each names a whole content id. Any other span is a defect of the family library.
pub fn check_media(media: &[MediaSpan], len: usize) -> Result<(), FamilyError> {
    let mut after = 0usize;
    for span in media {
        let end = span.first.checked_add(span.count);
        if span.count == 0 || span.first < after || end.is_none_or(|end| end > len) {
            return Err(FamilyError::internal(format!(
                "the media span of {} holds {} positions from position {}; a span holds a position, inside the {len} \
                 tokens, after the span before it",
                span.id, span.count, span.first
            )));
        }
        content_id(span)?;
        after = span.first + span.count;
    }
    Ok(())
}

/// The content id of a span.
fn content_id(span: &MediaSpan) -> Result<ContentId, FamilyError> {
    span.id
        .strip_prefix("blake3:")
        .and_then(|hex| ContentId::from_hex(hex).ok())
        .ok_or_else(|| {
            FamilyError::internal(format!(
                "the media span at position {} names `{}`, which is not a whole content id",
                span.first, span.id
            ))
        })
}

/// The recipe of the tokens `tokens[from..to]` after the recipe `previous`, at the position `to` (spec 6.2). Each media
/// span whose first position is in that range enters it as tag 9, in order (spec 11.11, "Media through the council").
/// A span with a bad content id is [`FamilyError::Internal`].
pub fn range_recipe(
    previous: &RecipeId,
    tokens: &[u32],
    media: &[MediaSpan],
    from: usize,
    to: usize,
) -> Result<RecipeId, FamilyError> {
    let spans: Vec<(&MediaSpan, ContentId)> = media
        .iter()
        .filter(|span| (from..to).contains(&span.first))
        .map(|span| content_id(span).map(|id| (span, id)))
        .collect::<Result<_, _>>()?;
    let items: Vec<RecipeMedia> = spans
        .iter()
        .map(|(span, id)| RecipeMedia {
            mime: &span.mime,
            id,
            positions: span.count as u64,
        })
        .collect();
    Ok(recipe_boundary_media(previous, &tokens[from..to], to as u64, &items))
}

/// The recipes of a rendering with one more boundary, at `cut` (spec 11.3c).
///
/// A cut is the end of a message that the client did not mark. A build that starts from a snapshot at a cut goes on
/// with the boundaries that the snapshot was made with, so its later recipes have the cut too. The cut must lie
/// before the end, and on no boundary. Any other cut is a defect of the caller: the result is
/// [`FamilyError::Internal`].
pub fn recipes_with_cut(
    root: RecipeId,
    rendering: &Rendering,
    cut: Boundary,
) -> Result<Vec<BoundaryRecipe>, FamilyError> {
    recipes_with_cuts(root, rendering, &[cut])
}

/// The recipes of a rendering with more boundaries, at `cuts`, in any order (spec 11.3c).
///
/// A state keeps each seal that its build made, so its recipe names each place where it was sealed. A build from a
/// cut sealed at the cut. A continuation from the head of a cut reply keeps the seal of that head. Each cut must lie
/// before the end, and on no boundary. Any other cut is a defect of the caller: the result is
/// [`FamilyError::Internal`].
pub fn recipes_with_cuts(
    root: RecipeId,
    rendering: &Rendering,
    cuts: &[Boundary],
) -> Result<Vec<BoundaryRecipe>, FamilyError> {
    let mut with_cuts = rendering.clone();
    for cut in cuts {
        if cut.tokens == 0 || cut.tokens >= rendering.end || with_cuts.boundaries.iter().any(|b| b.tokens == cut.tokens)
        {
            return Err(FamilyError::internal(format!(
                "a cut at token {} is not inside the context of {} tokens, between its boundaries",
                cut.tokens, rendering.end
            )));
        }
        let at = with_cuts.boundaries.partition_point(|b| b.tokens < cut.tokens);
        with_cuts.boundaries.insert(at, *cut);
    }
    boundary_recipes(root, &with_cuts)
}

/// One step of a build after its restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Feed the tokens from `from` to `to`.
    Feed {
        /// The first token.
        from: usize,
        /// The token after the last.
        to: usize,
    },
    /// Seal the context and hold a snapshot at a boundary. The number is the index in the boundaries.
    Snapshot {
        /// The index of the boundary.
        boundary: usize,
    },
}

impl Step {
    /// The kind of the quantum that runs the step (spec 11.3c).
    pub fn kind(&self) -> Kind {
        match self {
            Step::Feed { .. } => Kind::Prefill,
            Step::Snapshot { .. } => Kind::Copy,
        }
    }
}

/// The steps that bring a context from `kept` tokens to its end.
///
/// A feed has at most `chunk` tokens and never crosses a boundary. Each span starts its feeds at its own start, so an
/// engine with a feed rule gets its split when `chunk` is the block (spec 6.14). A snapshot follows the feed that reaches each
/// boundary after `kept`. A boundary at or before `kept` is held already and has no step.
pub fn plan_steps(kept: usize, boundaries: &[Boundary], chunk: usize) -> Vec<Step> {
    assert!(chunk > 0, "a chunk has at least one token");
    let mut steps = Vec::new();
    let mut position = kept;
    for (index, boundary) in boundaries.iter().enumerate() {
        if boundary.tokens <= kept {
            continue;
        }
        while position < boundary.tokens {
            let to = boundary.tokens.min(position + chunk);
            steps.push(Step::Feed { from: position, to });
            position = to;
        }
        steps.push(Step::Snapshot { boundary: index });
    }
    steps
}

/// The ranges of tokens that a read feeds, one pass each (spec 11.3c).
///
/// A range has at most `chunk` tokens and at most `max_rows` of the rows. The rows are the token indices whose logits
/// the caller wants. They must increase and lie inside the tokens, or the caller has a defect.
///
/// The plan is lazy. Each range holds at least one token, so a plan has at most `len` ranges.
pub fn plan_chunks(len: usize, rows: &[usize], chunk: usize, max_rows: usize) -> Chunks<'_> {
    assert!(chunk > 0 && max_rows > 0, "a pass holds at least one token and one row");
    assert!(
        rows.windows(2).all(|pair| pair[0] < pair[1]) && rows.last().is_none_or(|last| *last < len),
        "the rows must increase and lie inside the tokens"
    );
    Chunks {
        len,
        rows,
        chunk,
        max_rows,
        start: 0,
    }
}

/// The ranges of [`plan_chunks`], in order.
#[derive(Clone, Debug)]
pub struct Chunks<'a> {
    len: usize,
    rows: &'a [usize],
    chunk: usize,
    max_rows: usize,
    start: usize,
}

impl Iterator for Chunks<'_> {
    type Item = std::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.len {
            return None;
        }
        let start = self.start;
        let mut end = self.len.min(start + self.chunk);
        // The rows of this range start at the first row at or after its start. The row after the last one that
        // fits starts the next range. It is after `start`, because the first row of the range fits.
        let first = self.rows.partition_point(|row| *row < start);
        if let Some(next) = self.rows.get(first + self.max_rows)
            && *next < end
        {
            end = *next;
        }
        assert!(end > start, "a range of a plan holds at least one token");
        self.start = end;
        Some(start..end)
    }
}
