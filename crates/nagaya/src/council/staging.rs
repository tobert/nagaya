//! Host memory for the copies of engine state (spec 4.2).
//!
//! A mark and the buffer of a snapshot hold a whole state in host memory. The size is the state of one slot: up to
//! about 1.93 GB for a 32,768-token mk context. The ledger charges these bytes before the copy runs, and the charge
//! ends when the copy drops. A [`Staging`] is the charge.
//!
//! The store keeps its own bytes in its own budget. A snapshot buffer is the transient copy before the store takes
//! it, so the charge covers the buffer only.

use nagaya_abi::{Engine, Limits, PartKind, PartLimits, State, Tier};

use super::error::ServiceError;
use crate::process::{ContextChargeError, Ports};

/// A charge of staging bytes on the host tier of the ledger. The charge ends when the value drops.
pub struct Staging {
    ports: Ports,
    engine: String,
    bytes: u64,
}

impl Staging {
    /// Charge `bytes` for a copy of a state of `engine`. A refusal changes nothing and is an error that the client
    /// sees: the ledger could not hold the copy (spec 4.3).
    pub fn take(ports: &Ports, engine: &str, bytes: u64) -> Result<Self, ServiceError> {
        ports
            .admit_staging(engine, Tier::Host, bytes)
            .map_err(|error| match error {
                ContextChargeError::Stopping => ServiceError::Stopping,
                ContextChargeError::Ledger(error @ nagaya_ledger::LedgerError::StagingRefused(_)) => {
                    ServiceError::ContextRefused(error)
                }
                ContextChargeError::Ledger(error) => ServiceError::internal(error.to_string()),
                // The engine leaves the profile and drains, so it is not ready (spec 3.3a).
                ContextChargeError::Leaving(engine) => ServiceError::Unavailable {
                    engine,
                    state: nagaya_telemetry::EngineState::Draining,
                },
            })?;
        Ok(Self {
            ports: ports.clone(),
            engine: engine.to_owned(),
            bytes,
        })
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // The copy is gone, so the bytes cannot stay charged. A release that fails leaves the ledger wrong, and the
        // process stops (spec 4.2, 8.5). A stopped ledger has no charge left to release.
        if let Err(error) = self.ports.release_staging(Tier::Host, self.bytes) {
            self.ports.release_failed(
                &format!("{} staging bytes for engine `{}`", self.bytes, self.engine),
                &error,
            );
        }
    }
}

/// The bytes that a copy of the state of a slot holds: the bytes of the tensors that `park_snapshot` reads.
///
/// A fixed part counts its whole shape. A cells part and a ring part count their rows, from the base of the part to
/// the highest row of the slot.
pub(crate) fn state_bytes(engine: &mut dyn Engine, limits: &Limits, state: State) -> Result<u64, String> {
    let State::Slot(slot) = state;
    let mut high = None;
    let mut total = 0u64;
    for (id, part) in limits.parts.iter().enumerate().filter(|(_, part)| part.is_stored()) {
        let bytes = match part.kind {
            PartKind::Fixed => shape_bytes(part)?,
            PartKind::Cells | PartKind::Ring => {
                let high = match high {
                    Some(high) => high,
                    None => {
                        let rows = engine
                            .slots()
                            .ok_or_else(|| {
                                format!(
                                    "part `{}` has no row count: the engine has no slots capability",
                                    part.name
                                )
                            })?
                            .slot_high(slot);
                        high = Some(rows);
                        rows
                    }
                };
                let start = if part.kind == PartKind::Ring {
                    engine
                        .coverage()
                        .ok_or_else(|| {
                            format!(
                                "part `{}` has no base: the engine has no coverage capability",
                                part.name
                            )
                        })?
                        .coverage(state, id, &[])
                        .start
                } else {
                    0
                };
                high.saturating_sub(start)
                    .checked_mul(shape_bytes(part)?)
                    .ok_or_else(|| format!("part `{}` is too large", part.name))?
            }
            PartKind::Derived => unreachable!("a derived part is not stored"),
        };
        total = total
            .checked_add(bytes)
            .ok_or_else(|| "the state is too large".to_owned())?;
    }
    Ok(total)
}

/// The bytes of the shape of a part: one row for a cells part or a ring part.
fn shape_bytes(part: &PartLimits) -> Result<u64, String> {
    part.shape
        .iter()
        .try_fold(part.dtype.element_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| format!("part `{}` is too large", part.name))
}

#[cfg(test)]
mod tests {
    use nagaya_abi::fake::FakeAlways;
    use nagaya_abi::{OutputsWanted, Segment, SegmentInputs, SlotLimit};

    use super::*;
    use crate::park::park_snapshot;

    /// Feed `rows` tokens to slot 0 of a fake engine.
    fn fed(rows: u64) -> FakeAlways {
        let mut engine = FakeAlways::with_slots(vec![SlotLimit {
            bytes: 16384,
            count: 1,
            tokens: 4096,
        }]);
        // The fake takes 64 tokens in a pass.
        for from in (0..rows).step_by(64) {
            let to = rows.min(from + 64);
            engine
                .pass(
                    &[Segment {
                        state: State::Slot(0),
                        inputs: SegmentInputs::Tokens((from as u32..to as u32).collect()),
                        positions: (from..to).collect(),
                        meta: Vec::new(),
                        outputs: OutputsWanted::default(),
                    }],
                    &[],
                )
                .unwrap();
        }
        engine
    }

    /// The charge is the size of the copy that `park_snapshot` makes, for every row count.
    #[test]
    fn the_charge_is_the_size_of_the_copy() {
        for rows in [0, 1, 7, 100, 1000] {
            let mut engine = fed(rows);
            let limits = engine.limits();
            let state = State::Slot(0);
            let copy = park_snapshot(&mut engine, state).unwrap();
            let copied: u64 = copy.tensors.iter().map(|tensor| tensor.data.len() as u64).sum();
            let charged = state_bytes(&mut engine, &limits, state).unwrap();
            assert_eq!(charged, copied, "{rows} rows");
            if rows > 0 {
                assert!(charged > 0, "{rows} rows");
            }
        }
    }

    /// More rows cost more bytes, so a long context is charged more than a short one.
    #[test]
    fn a_longer_state_costs_more() {
        let limits = fed(0).limits();
        let short = state_bytes(&mut fed(10), &limits, State::Slot(0)).unwrap();
        let long = state_bytes(&mut fed(1000), &limits, State::Slot(0)).unwrap();
        assert!(long > short, "{long} > {short}");
    }
}
