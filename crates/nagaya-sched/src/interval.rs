//! The scheduler's part of interval copies (spec 6.13).

use std::collections::BTreeMap;

use nagaya_abi::Limits;

/// One context.
pub type ContextId = u64;

/// An error of the interval logic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntervalError {
    /// The engine declares a grain of 0.
    ZeroGrain,
    /// The engine declares a grain and not `segments_exact`.
    NeedsSegmentsExact,
    /// The context is not tracked.
    UnknownContext(ContextId),
    /// The context is tracked already.
    DuplicateContext(ContextId),
    /// A position is lower than the last one.
    PositionWentBack {
        /// The context.
        context: ContextId,
        /// The last position.
        from: u64,
        /// The new position.
        to: u64,
    },
}

/// The rule for interval copies of one engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntervalPolicy {
    grain: u64,
}

impl IntervalPolicy {
    /// Read the rule from the limits of an engine and the cap of the profile (spec 6.13).
    ///
    /// It gives `None` when the engine declares no grain, or when the cap is 0 (the store refuses every copy then).
    /// A grain of 0 is an error. A grain without `segments_exact` is an error, because a replay needs any split to
    /// give the same bits.
    pub fn from_limits(limits: &Limits, cap: u32) -> Result<Option<Self>, IntervalError> {
        let Some(grain) = limits.checkpoint_grain else {
            return Ok(None);
        };
        if grain == 0 {
            return Err(IntervalError::ZeroGrain);
        }
        if !limits.segments_exact {
            return Err(IntervalError::NeedsSegmentsExact);
        }
        if cap == 0 {
            return Ok(None);
        }
        Ok(Some(Self { grain }))
    }

    /// The grain in tokens.
    pub fn grain(&self) -> u64 {
        self.grain
    }
}

/// A copy that the caller must make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyDue {
    /// The context.
    pub context: ContextId,
    /// The position of the context now, in tokens.
    pub position: u64,
}

/// What the tracker knows about one context.
#[derive(Clone, Debug)]
struct Track<I> {
    /// The position after the last quantum.
    position: u64,
    /// The number of the last multiple of the grain that the tracker offered a copy for, or passed at the start.
    tried: u64,
    /// The copy that the store holds for the context.
    copy: Option<I>,
}

/// The interval copies of the contexts of one engine. `I` names a copy in the store.
///
/// A copy is due after a quantum that moves a context across a multiple of the grain. The tracker offers one copy for
/// each multiple. If the put fails, the caller does nothing, and the tracker waits for the next multiple.
#[derive(Clone, Debug)]
pub struct IntervalCopies<I> {
    policy: IntervalPolicy,
    contexts: BTreeMap<ContextId, Track<I>>,
}

impl<I> IntervalCopies<I> {
    /// Make a tracker with no context.
    pub fn new(policy: IntervalPolicy) -> Self {
        Self {
            policy,
            contexts: BTreeMap::new(),
        }
    }

    /// Start to track a context at a position. The first copy waits for the next multiple.
    pub fn track(&mut self, context: ContextId, position: u64) -> Result<(), IntervalError> {
        if self.contexts.contains_key(&context) {
            return Err(IntervalError::DuplicateContext(context));
        }
        self.contexts.insert(
            context,
            Track {
                position,
                tried: position / self.policy.grain,
                copy: None,
            },
        );
        Ok(())
    }

    /// Stop to track a context. It gives the context's copy. The store keeps that copy until eviction removes it.
    pub fn untrack(&mut self, context: ContextId) -> Result<Option<I>, IntervalError> {
        self.contexts
            .remove(&context)
            .map(|track| track.copy)
            .ok_or(IntervalError::UnknownContext(context))
    }

    /// Tell the tracker that a quantum moved a context to a position. It says when a copy is due.
    pub fn advance(&mut self, context: ContextId, position: u64) -> Result<Option<CopyDue>, IntervalError> {
        let grain = self.policy.grain;
        let track = self
            .contexts
            .get_mut(&context)
            .ok_or(IntervalError::UnknownContext(context))?;
        if position < track.position {
            return Err(IntervalError::PositionWentBack {
                context,
                from: track.position,
                to: position,
            });
        }
        track.position = position;
        let multiple = position / grain;
        if multiple > track.tried {
            track.tried = multiple;
            return Ok(Some(CopyDue { context, position }));
        }
        Ok(None)
    }

    /// Tell the tracker that the store holds a new copy. It gives the previous copy, which the caller removes.
    pub fn stored(&mut self, context: ContextId, copy: I) -> Result<Option<I>, IntervalError> {
        let track = self
            .contexts
            .get_mut(&context)
            .ok_or(IntervalError::UnknownContext(context))?;
        Ok(track.copy.replace(copy))
    }

    /// The copy of a context.
    pub fn copy(&self, context: ContextId) -> Result<Option<&I>, IntervalError> {
        self.contexts
            .get(&context)
            .map(|track| track.copy.as_ref())
            .ok_or(IntervalError::UnknownContext(context))
    }
}
