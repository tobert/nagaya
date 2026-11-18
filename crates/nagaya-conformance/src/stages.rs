//! The gates of the stage path and the feed rule (engine-abi D1, D11, spec 8.3).

use nagaya_abi::{
    Capability, Engine, Limits, Output, OutputsWanted, PartId, PartKind, Parts, Residual, Segment, SegmentInputs,
    StageOut, State,
};

use crate::gate::{Outcome, compare_outputs, non_finite, restore_through_store};
use crate::{Failure, compare_bytes};

const STAGES: &str = "stage_path_bits";
const FEED: &str = "feed_rule_bits";

/// The tokens of the gates. They stay below 97, so any vocabulary of the engines holds them.
fn tokens(count: usize, seed: u32) -> Vec<u32> {
    (0..count as u32).map(|index| (index * 7 + seed) % 97 + 1).collect()
}

fn segment(slot: u32, from: u64, tokens: &[u32], logits: bool) -> Segment {
    let to = from + tokens.len() as u64;
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..to).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: if logits { vec![to - 1] } else { Vec::new() },
        },
        layer_loop: None,
    }
}

/// The stored parts of a slot that holds `rows` rows: each fixed part whole, and rows 0 to `rows` of each row part.
fn read_parts(
    gate: &'static str,
    parts: &mut dyn Parts,
    limits: &Limits,
    only: impl Fn(PartId) -> bool,
    slot: u32,
    rows: u64,
) -> Result<Vec<(PartId, Vec<u8>)>, Failure> {
    let mut read = Vec::new();
    for (id, part) in limits
        .parts
        .iter()
        .enumerate()
        .filter(|(id, part)| part.is_stored() && only(*id))
    {
        let size = part.shape.iter().product::<u64>() * part.dtype.element_bytes();
        let result = match part.kind {
            PartKind::Fixed => {
                let mut bytes = vec![0u8; size as usize];
                parts.part_read(State::Slot(slot), id, &mut bytes).map(|()| bytes)
            }
            _ => {
                let mut bytes = vec![0u8; (size * rows) as usize];
                parts
                    .cells_read(State::Slot(slot), id, 0, rows, &mut bytes)
                    .map(|()| bytes)
            }
        };
        let bytes = result.map_err(|error| {
            Failure::new(
                gate,
                format!("the read of part `{}` of slot {slot} failed: {error:?}", part.name),
            )
        })?;
        read.push((id, bytes));
    }
    Ok(read)
}

fn compare_parts(
    gate: &'static str,
    limits: &Limits,
    left: &[(PartId, Vec<u8>)],
    right: &[(PartId, Vec<u8>)],
) -> Result<(), Failure> {
    for ((id, left), (_, right)) in left.iter().zip(right) {
        let part = &limits.parts[*id];
        for (side, bytes) in [("left", left), ("right", right)] {
            if let Some(index) = non_finite(part.dtype, bytes) {
                return Err(Failure::new(
                    gate,
                    format!("part `{}` {side} index {index} is not finite", part.name),
                ));
            }
        }
        compare_bytes(gate, left, right).map_err(|failure| Failure {
            message: format!("part `{}`: {}", part.name, failure.message),
            ..failure
        })?;
    }
    Ok(())
}

/// The passes of the stage gate: slot 0 in three passes, and slot 1 beside slot 0 in the second pass when the engine
/// takes two segments. Each pass holds at most one residual buffer of rows.
fn stage_plan(limits: &Limits) -> Vec<Vec<Segment>> {
    let rows = (limits.residual_rows() as usize)
        .min(limits.max_pass_inputs as usize)
        .max(1);
    let half = (rows / 2).max(1);
    let first = tokens(rows, 3);
    let second = tokens(half, 5);
    let beside = tokens(rows - half, 11);
    let third = tokens(half, 13);
    let mut second_pass = vec![segment(0, rows as u64, &second, false)];
    if limits.max_segments >= 2 && !beside.is_empty() {
        second_pass.push(segment(1, 0, &beside, limits.max_logit_rows >= 2));
    }
    vec![
        vec![segment(0, 0, &first, false)],
        second_pass,
        vec![segment(0, (rows + half) as u64, &third, limits.max_logit_rows >= 1)],
    ]
}

/// For an engine with stages: `Engine::pass` and the stage path give the same outputs and the same parts.
///
/// One engine runs each pass of the plan through `Engine::pass`. A second engine takes its handles and runs each pass
/// stage by stage with one residual buffer. The gate compares the outputs of each pass and every stored part.
pub fn stage_path_bits<E: Engine>(make: &mut dyn FnMut() -> E) -> Outcome {
    let mut whole = make();
    let limits = whole.limits();
    if !limits.capabilities.contains(&Capability::Stages) {
        return Outcome::NotApplicable("the engine declares no stages".to_owned());
    }
    match stage_path_inner(&mut whole, make, &limits) {
        Ok(()) => Outcome::Pass,
        Err(failure) => Outcome::Fail(failure),
    }
}

fn stage_path_inner<E: Engine>(whole: &mut E, make: &mut dyn FnMut() -> E, limits: &Limits) -> Result<(), Failure> {
    let plan = stage_plan(limits);
    let mut expected: Vec<Vec<Output>> = Vec::new();
    for (index, pass) in plan.iter().enumerate() {
        let out = whole.pass(pass, &[]).map_err(|error| {
            Failure::new(
                STAGES,
                format!("pass {index} through Engine::pass failed: {}", error.message),
            )
        })?;
        expected.push(out.outputs);
    }
    let held = |slot: u32| -> u64 {
        plan.iter()
            .flatten()
            .filter(|segment| segment.state == State::Slot(slot))
            .map(|segment| segment.positions.len() as u64)
            .sum()
    };
    let parts = whole
        .parts()
        .ok_or_else(|| Failure::new(STAGES, "the engine has no parts before take_handles"))?;
    let slots: Vec<u32> = (0..2).filter(|slot| held(*slot) > 0).collect();
    let mut want = Vec::new();
    for slot in &slots {
        want.push(read_parts(STAGES, parts, limits, |_| true, *slot, held(*slot))?);
    }

    let mut staged = make();
    let mut handles = staged
        .stages()
        .ok_or_else(|| Failure::new(STAGES, "the engine declares stages and gives no stage accessor"))?
        .take_handles();
    if handles.len() != limits.stages.len() {
        return Err(Failure::new(
            STAGES,
            format!(
                "the engine declares {} stages and gave {} handles",
                limits.stages.len(),
                handles.len()
            ),
        ));
    }
    if staged.pass(&plan[0], &[]).is_ok() {
        return Err(Failure::new(
            STAGES,
            "Engine::pass ran after take_handles; it must refuse",
        ));
    }
    let mut residual = Residual::new(limits.residual_row_bytes as usize, limits.residual_rows() as usize);
    for (index, pass) in plan.iter().enumerate() {
        let mut outputs = None;
        for (stage, handle) in handles.iter_mut().enumerate() {
            let out = handle.stage_pass(pass, &mut residual).map_err(|error| {
                Failure::new(
                    STAGES,
                    format!("stage {stage} of pass {index} failed: {}", error.message),
                )
            })?;
            match (out, stage + 1 == limits.stages.len()) {
                (StageOut::Residual { .. }, false) => {}
                (StageOut::Last(out), true) => outputs = Some(out.outputs),
                (_, last) => {
                    return Err(Failure::new(
                        STAGES,
                        format!("stage {stage} of pass {index} gave the wrong kind of result (last stage: {last})"),
                    ));
                }
            }
        }
        compare_outputs(STAGES, &expected[index], &outputs.unwrap_or_default()).map_err(|failure| Failure {
            message: format!("pass {index}: {}", failure.message),
            ..failure
        })?;
    }
    for (slot, want) in slots.iter().zip(&want) {
        let mut got = Vec::new();
        for (stage, handle) in handles.iter_mut().enumerate() {
            let parts = handle
                .parts()
                .ok_or_else(|| Failure::new(STAGES, format!("stage {stage} has no parts capability")))?;
            got.extend(read_parts(
                STAGES,
                parts,
                limits,
                |id| limits.parts[id].stage as usize == stage,
                *slot,
                held(*slot),
            )?);
        }
        got.sort_by_key(|(id, _)| *id);
        compare_parts(STAGES, limits, want, &got)?;
    }
    Ok(())
}

/// Feed `tokens` into slot 0 from `from` in passes of `block`, and give the outputs of the last pass.
fn feed<E: Engine + ?Sized>(engine: &mut E, from: u64, tokens: &[u32], block: usize) -> Result<Vec<Output>, Failure> {
    let chunks: Vec<&[u32]> = tokens.chunks(block).collect();
    let mut position = from;
    let mut outputs = Vec::new();
    for (index, chunk) in chunks.iter().enumerate() {
        let last = index + 1 == chunks.len();
        let out = engine
            .pass(&[segment(0, position, chunk, last)], &[])
            .map_err(|error| Failure::new(FEED, format!("a pass at position {position} failed: {}", error.message)))?;
        position += chunk.len() as u64;
        if last {
            outputs = out.outputs;
        }
    }
    Ok(outputs)
}

/// For an engine with a feed rule: two spans fed by the rule in one job give the bits of the second span fed after a
/// restore of the first (engine-abi D11). Thus the bits depend only on the tokens and the boundaries.
pub fn feed_rule_bits<E: Engine>(make: &mut dyn FnMut() -> E) -> Outcome {
    let mut once = make();
    let limits = once.limits();
    let Some(rule) = limits.feed else {
        return Outcome::NotApplicable("the engine declares no feed rule".to_owned());
    };
    match feed_rule_inner(&mut once, make, &limits, rule.block as usize) {
        Ok(()) => Outcome::Pass,
        Err(failure) => Outcome::Fail(failure),
    }
}

fn feed_rule_inner<E: Engine>(
    once: &mut E,
    make: &mut dyn FnMut() -> E,
    limits: &Limits,
    block: usize,
) -> Result<(), Failure> {
    // Each span ends with a short pass, so a split that does not start at the span would differ.
    let span = block + block.div_ceil(2);
    let first = tokens(span, 3);
    let second = tokens(span, 17);
    let held = (2 * span) as u64;
    if limits.slots.first().is_none_or(|class| class.tokens < held) {
        return Err(Failure::new(
            FEED,
            format!("slot 0 does not hold the {held} tokens of the gate"),
        ));
    }
    feed(once, 0, &first, block)?;
    let expected = feed(once, span as u64, &second, block)?;
    let parts = once
        .parts()
        .ok_or_else(|| Failure::new(FEED, "the engine has no parts capability"))?;
    let want = read_parts(FEED, parts, limits, |_| true, 0, held)?;

    let mut start = make();
    feed(&mut start, 0, &first, block)?;
    let parts = start
        .parts()
        .ok_or_else(|| Failure::new(FEED, "the engine has no parts capability"))?;
    let kept = read_parts(FEED, parts, limits, |_| true, 0, span as u64)?;

    let mut restored = make();
    restore_through_store(&mut restored, State::Slot(0), limits, &kept, None, FEED)?;
    let got = feed(&mut restored, span as u64, &second, block)?;
    compare_outputs(FEED, &expected, &got)?;
    let parts = restored
        .parts()
        .ok_or_else(|| Failure::new(FEED, "the engine has no parts capability"))?;
    let after = read_parts(FEED, parts, limits, |_| true, 0, held)?;
    compare_parts(FEED, limits, &want, &after)
}
