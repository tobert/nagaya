//! The pure parts of a build (spec 6.2, 11.3c): the recipe of each boundary, and the steps that feed the rest.

use nagaya_abi::Kind;
use nagaya_store::{RecipeId, recipe_boundary, recipe_root};

use crate::family::{Boundary, FamilyError, Rendering};

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
        let recipe = recipe_boundary(
            &previous,
            &rendering.tokens[position..boundary.tokens],
            boundary.tokens as u64,
        );
        recipes.push(BoundaryRecipe {
            boundary: *boundary,
            recipe,
        });
        previous = recipe;
        position = boundary.tokens;
    }
    Ok(recipes)
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
/// A feed has at most `chunk` tokens and never crosses a boundary. A snapshot follows the feed that reaches each
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
