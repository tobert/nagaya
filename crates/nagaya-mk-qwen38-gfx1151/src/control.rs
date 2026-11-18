//! The control op of the Qwen3.8 engine on the host, the plan of the launch cuts of a pass, and the layout of its
//! taps (engine-abi "The residual of the Qwen3.8 engine").
//!
//! The residual of a token is the 4 streams, 10,240 `f32` values. A pass with a tap or a control item runs as several
//! launches, cut at layer ends. At a cut, the host reads the rows of a tap, and the device op of
//! `hip/src/engine/control.hip` changes the rows. The next launch goes on from them.
//!
//! This code is pure: it needs no GPU and no library. The pass host launches the ranges of a `CutPlan`, the taps use
//! `tap_layout`, and the control table uses `write_sums`. The host op (`apply_row`, `apply_actions`) is the oracle of
//! the device op: the fake device runs it, and the card gate `control_card.rs` compares the bits.

#![allow(dead_code)]

use std::ops::Range;

use nagaya_abi::{ControlItem, ControlLimit, ControlOp, Tap};

use crate::spec;

/// The layers of the model, and so the cuts and the rows of the tap `residual`.
pub(crate) const LAYERS: usize = spec::LAYERS as usize;

/// The streams of the state between two layers.
pub(crate) const STREAMS: usize = 4;

/// The values of one stream.
pub(crate) const STREAM_WIDTH: usize = 2_560;

/// The values of the residual of one token: the 4 streams in stream order.
pub(crate) const WIDTH: usize = STREAMS * STREAM_WIDTH;

/// The values of the fold: the row that the head reads.
pub(crate) const FOLD_WIDTH: usize = STREAM_WIDTH;

/// The entries of the control table (engine-abi "The residual of the Qwen3.8 engine").
pub(crate) const TABLES: u32 = 8;

/// The control limit that the engine declares.
pub(crate) const LIMIT: ControlLimit = ControlLimit {
    tables: TABLES,
    layers: spec::LAYERS,
    width: WIDTH as u32,
};

/// The name of the tap of the residual after each layer.
pub(crate) const RESIDUAL_TAP: &str = "residual";

/// The name of the tap of the row that the head reads.
pub(crate) const FOLD_TAP: &str = "fold";

/// The bytes of the `residual` output of one logit row: 48 rows of 10,240 `f32` values.
pub(crate) const RESIDUAL_BYTES: usize = LAYERS * WIDTH * 4;

/// The bytes of the `fold` output of one logit row.
pub(crate) const FOLD_BYTES: usize = FOLD_WIDTH * 4;

// ---- the arithmetic ---------------------------------------------------------------------------------------------------

/// The lanes of a sum. Lane `k` adds the terms whose index modulo 8 is `k`, in index order.
const LANES: usize = 8;

/// Join 8 lanes as `((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))`.
fn join(s: [f32; LANES]) -> f32 {
    ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))
}

/// The dot product of two rows in a fixed order: each product is rounded, then added, in 8 lanes by index modulo 8,
/// and the lanes join as in `join`. The code never fuses a product with a sum (`mul_add`), so the bits do not depend on
/// the compiler.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "a dot product needs rows of one length");
    let mut lanes = [0.0f32; LANES];
    for (j, (x, y)) in a.iter().zip(b).enumerate() {
        let product = x * y;
        lanes[j % LANES] += product;
    }
    join(lanes)
}

/// The sum of squares of a row: `v . v`. The table computes it once for each layer of a vector, when the vector is
/// written.
pub(crate) fn sum_squares(v: &[f32]) -> f32 {
    dot(v, v)
}

/// The code of an op in the call of the device op (`hip/src/engine/control.hip`).
pub(crate) fn op_code(op: ControlOp) -> u32 {
    match op {
        ControlOp::Add => 0,
        ControlOp::Ablate => 1,
        ControlOp::Clamp => 2,
    }
}

/// The op of a code of the device op, or `None` for a code that names no op.
pub(crate) fn op_of_code(code: u32) -> Option<ControlOp> {
    match code {
        0 => Some(ControlOp::Add),
        1 => Some(ControlOp::Ablate),
        2 => Some(ControlOp::Clamp),
        _ => None,
    }
}

/// Change one row `h` with the vector `v` of its layer, whose sum of squares is `vv` (engine-abi "The residual of the
/// Qwen3.8 engine"). Return false, and change nothing, when `vv` is 0.
///
/// With `n` the square root of `vv` and `along` the dot product of `h` and `v` over `n`, the coefficient `c` is `a` for
/// `add`, `-a * along / n` for `ablate`, and `(a - along) / n` for `clamp`. Then `h[j]` is `h[j] + c * v[j]`.
pub(crate) fn apply_row(op: ControlOp, strength: f32, h: &mut [f32], v: &[f32], vv: f32) -> bool {
    assert_eq!(h.len(), WIDTH, "a residual row has {WIDTH} values");
    assert_eq!(v.len(), WIDTH, "a vector of one layer has {WIDTH} values");
    if vv == 0.0 {
        return false;
    }
    let norm = vv.sqrt();
    let along = dot(h, v) / norm;
    let coefficient = match op {
        ControlOp::Add => strength,
        ControlOp::Ablate => (-strength * along) / norm,
        ControlOp::Clamp => (strength - along) / norm,
    };
    for (x, y) in h.iter_mut().zip(v) {
        let product = coefficient * y;
        *x += product;
    }
    true
}

/// The vector of one layer, as the op reads it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerVector<'a> {
    pub values: &'a [f32],
    pub sum_squares: f32,
}

/// Apply the actions of one cut to `h`, the rows of the pass, in the order of the actions. `vector` gives the vector
/// of an entry at the layer of the cut. Return the count of rows that changed: a zero vector changes none.
pub(crate) fn apply_actions<'a>(h: &mut [f32], actions: &[Action], vector: impl Fn(u32) -> LayerVector<'a>) -> usize {
    assert_eq!(h.len() % WIDTH, 0, "the rows of a pass hold whole residuals");
    let mut changed = 0;
    for action in actions {
        assert!(
            action.rows.end * WIDTH <= h.len(),
            "the action names rows {:?}, but the pass holds {} rows",
            action.rows,
            h.len() / WIDTH
        );
        let layer = vector(action.entry);
        for row in action.rows.clone() {
            let row = &mut h[row * WIDTH..(row + 1) * WIDTH];
            if apply_row(action.op, action.strength, row, layer.values, layer.sum_squares) {
                changed += 1;
            }
        }
    }
    changed
}

// ---- the checks -------------------------------------------------------------------------------------------------------

/// The fault of the control list of segment `segment`, or `None`. The list must fit the limit of the table, and each
/// layer that it names must be a layer of the model.
pub(crate) fn list_problem(segment: usize, items: &[ControlItem]) -> Option<String> {
    if let Err(message) = LIMIT.check(items) {
        return Some(format!("segment {segment}: {message}"));
    }
    for (i, item) in items.iter().enumerate() {
        if let Some((layer, _)) = item
            .layers
            .iter()
            .flatten()
            .find(|(layer, _)| *layer as usize >= LAYERS)
        {
            return Some(format!(
                "segment {segment}: control item {i} names layer {layer}, but the engine has {LAYERS} layers, 0 to {}; name a layer below {LAYERS}",
                LAYERS - 1
            ));
        }
    }
    None
}

/// The fault of the bytes of a `control_write`, or `None`: the entry must exist, the size must be 48 layers of 10,240
/// `f32` values, and each value must be finite. A write that breaks a rule changes nothing.
pub(crate) fn write_problem(index: u32, bytes: &[u8]) -> Option<String> {
    if index >= LIMIT.tables {
        return Some(format!(
            "the control table has {} entries, not entry {index}; write an entry below {}",
            LIMIT.tables, LIMIT.tables
        ));
    }
    if bytes.len() as u64 != LIMIT.entry_bytes() {
        return Some(format!(
            "a vector is {} layers of {} f32 values, {} bytes, not {} bytes; send the whole vector",
            LIMIT.layers,
            LIMIT.width,
            LIMIT.entry_bytes(),
            bytes.len()
        ));
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .position(|b| !f32::from_le_bytes(*b).is_finite())
        .map(|at| {
            format!("value {at} of the vector is not finite; the op needs finite values, so replace it with a number")
        })
}

/// Check the bytes of a `control_write`, and give the sum of squares of each of the 48 layers. The table keeps these
/// sums, so the op does not compute them at each pass.
pub(crate) fn write_sums(index: u32, bytes: &[u8]) -> Result<Vec<f32>, String> {
    if let Some(problem) = write_problem(index, bytes) {
        return Err(problem);
    }
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(values
        .as_chunks::<WIDTH>()
        .0
        .iter()
        .map(|row| sum_squares(row))
        .collect())
}

// ---- the taps ---------------------------------------------------------------------------------------------------------

/// The taps that a pass names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TapSet {
    pub residual: bool,
    pub fold: bool,
}

impl TapSet {
    pub(crate) fn any(self) -> bool {
        self.residual || self.fold
    }
}

/// Read the names of the taps of a pass. An unknown name and a name that comes twice are refused.
pub(crate) fn tap_set(taps: &[Tap]) -> Result<TapSet, String> {
    let mut set = TapSet::default();
    for tap in taps {
        let seen = match tap.name.as_str() {
            RESIDUAL_TAP => &mut set.residual,
            FOLD_TAP => &mut set.fold,
            other => {
                return Err(format!(
                    "the pass names the tap `{other}`, but the engine gives the taps `{RESIDUAL_TAP}` and `{FOLD_TAP}`; name one of them"
                ));
            }
        };
        if *seen {
            return Err(format!(
                "the pass names the tap `{}` twice; name each tap once",
                tap.name
            ));
        }
        *seen = true;
    }
    Ok(set)
}

/// One output of a tap: its name, the segment whose logit row it belongs to, and its size in bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TapOutput {
    pub name: &'static str,
    pub segment: usize,
    pub bytes: usize,
}

/// The tap outputs of a pass, after its `logits` outputs. `wants` says which segments ask for a logit row. Each such
/// row gets one `residual` output and one `fold` output. All the `residual` outputs come first, then all the `fold`
/// outputs, and each group follows the order of the segments. A tap on a pass with no logit row is refused, because it
/// would give nothing.
pub(crate) fn tap_layout(wants: &[bool], taps: TapSet) -> Result<Vec<TapOutput>, String> {
    if !taps.any() {
        return Ok(Vec::new());
    }
    if !wants.iter().any(|want| *want) {
        return Err(
            "the pass names a tap, but no segment asks for a logit row; a tap reads the row of the last token, so ask for that row"
                .to_owned(),
        );
    }
    let rows = || {
        wants
            .iter()
            .enumerate()
            .filter(|(_, want)| **want)
            .map(|(segment, _)| segment)
    };
    let mut outputs = Vec::new();
    if taps.residual {
        outputs.extend(rows().map(|segment| TapOutput {
            name: RESIDUAL_TAP,
            segment,
            bytes: RESIDUAL_BYTES,
        }));
    }
    if taps.fold {
        outputs.extend(rows().map(|segment| TapOutput {
            name: FOLD_TAP,
            segment,
            bytes: FOLD_BYTES,
        }));
    }
    Ok(outputs)
}

// ---- the cuts ---------------------------------------------------------------------------------------------------------

/// What a pass needs to know of one segment to plan its cuts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SegmentShape<'a> {
    /// The tokens of the segment: its rows in the pass. At least 1.
    pub rows: usize,
    /// True when the segment asks for its logit row, the row of its last token.
    pub logit_row: bool,
    pub control: &'a [ControlItem],
}

/// One row of `h` that a cut reads for a tap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadRow {
    pub segment: usize,
    /// The row in the pass: the last row of the segment.
    pub row: usize,
}

/// One change of rows that a cut makes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Action {
    pub segment: usize,
    /// The rows of the segment in the pass.
    pub rows: Range<usize>,
    /// The entry of the control table.
    pub entry: u32,
    pub op: ControlOp,
    pub strength: f32,
}

/// A layer end at which the pass stops its launch. The launch before it ends with the layer `layer`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Cut {
    pub layer: usize,
    /// The rows to read, in the order of the segments. Empty without the tap `residual`.
    pub reads: Vec<ReadRow>,
    /// The changes, in the order of the segments and then of each list.
    pub actions: Vec<Action>,
}

/// The cuts of a pass, in layer order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CutPlan {
    pub cuts: Vec<Cut>,
    /// True when the pass must build with the fused combine of the next layer's norm off, because a cut changes `h`
    /// (engine-abi "The residual of the Qwen3.8 engine").
    pub fuse_hc_off: bool,
}

/// The strength that an item has at layer `layer`, or `None` when the item makes no action there. An item whose layers
/// leave the layer out makes none. An `add` or an `ablate` of strength 0 makes none. A `clamp` of 0 does, because it
/// removes the concept (engine-abi "Control vectors").
fn action_strength(item: &ControlItem, layer: usize) -> Option<f32> {
    let strength = item.strength_at(layer as u32)?;
    if item.op != ControlOp::Clamp && strength == 0.0 {
        return None;
    }
    Some(strength)
}

/// Plan the cuts of a pass. A plain pass has none, a pass with the tap `residual` has one at each layer, and a pass
/// with control actions has one at each layer that has an action. The tap `fold` has none: its row is read after the
/// last launch. A list that breaks a rule of `list_problem` is refused.
pub(crate) fn plan_cuts(segments: &[SegmentShape], taps: TapSet) -> Result<CutPlan, String> {
    let mut starts = Vec::with_capacity(segments.len());
    let mut total = 0;
    for (index, segment) in segments.iter().enumerate() {
        assert!(
            segment.rows > 0,
            "segment {index} has no rows; the caller checks that first"
        );
        if let Some(problem) = list_problem(index, segment.control) {
            return Err(problem);
        }
        starts.push(total);
        total += segment.rows;
    }
    let reads: Vec<ReadRow> = if taps.residual {
        segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| segment.logit_row)
            .map(|(index, segment)| ReadRow {
                segment: index,
                row: starts[index] + segment.rows - 1,
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut cuts = Vec::new();
    for layer in 0..LAYERS {
        let mut actions = Vec::new();
        for (index, segment) in segments.iter().enumerate() {
            for item in segment.control {
                if let Some(strength) = action_strength(item, layer) {
                    actions.push(Action {
                        segment: index,
                        rows: starts[index]..starts[index] + segment.rows,
                        entry: item.index,
                        op: item.op,
                        strength,
                    });
                }
            }
        }
        if !reads.is_empty() || !actions.is_empty() {
            cuts.push(Cut {
                layer,
                reads: reads.clone(),
                actions,
            });
        }
    }
    let fuse_hc_off = cuts.iter().any(|cut| !cut.actions.is_empty());
    Ok(CutPlan { cuts, fuse_hc_off })
}

#[cfg(test)]
mod tests {
    use nagaya_abi::{ControlItem, ControlOp, Tap};

    use super::*;

    // ---- the data of the tests ----

    /// A row of `n` values from a linear congruential generator, in `[-scale / 2, scale / 2)`. The pinned bits below
    /// come from a numpy emulation of the same generator and the same f32 steps, so the arithmetic has a second source.
    fn lcg(seed: u32, n: usize, scale: f64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((f64::from(state >> 8) / 16_777_216.0 - 0.5) * scale) as f32
            })
            .collect()
    }

    /// A row `h` and a vector `v` whose dot product depends on the order of the sum: every other order and the fused
    /// product give other bits (`other_orders_give_other_bits`).
    fn pair() -> (Vec<f32>, Vec<f32>) {
        (lcg(2, WIDTH, 4.0), lcg(102, WIDTH, 2.0))
    }

    const DOT_BITS: u32 = 0x4125_5686;
    const SUM_SQUARES_BITS: u32 = 0x4555_0bb7;

    fn item(index: u32, op: ControlOp, strength: f32, layers: Option<Vec<(u32, f32)>>) -> ControlItem {
        ControlItem {
            index,
            op,
            strength,
            layers,
        }
    }

    fn shape(rows: usize, logit_row: bool, control: &[ControlItem]) -> SegmentShape<'_> {
        SegmentShape {
            rows,
            logit_row,
            control,
        }
    }

    // ---- the sizes ----

    #[test]
    fn the_sizes_are_the_ones_that_the_notes_give() {
        assert_eq!(LAYERS, 48);
        assert_eq!(WIDTH, 10_240);
        assert_eq!(FOLD_WIDTH, 2_560);
        assert_eq!(RESIDUAL_BYTES, 1_966_080);
        assert_eq!(FOLD_BYTES, 10_240);
        assert_eq!(LIMIT.tables, 8);
        assert_eq!(LIMIT.table_bytes(), 15_728_640);
        assert_eq!(LIMIT.entry_bytes(), 1_966_080);
    }

    // ---- the arithmetic ----

    #[test]
    fn the_dot_product_has_the_bits_of_the_8_lane_sum() {
        let (h, v) = pair();
        assert_eq!(dot(&h, &v).to_bits(), DOT_BITS);
        assert_eq!(sum_squares(&v).to_bits(), SUM_SQUARES_BITS);
        assert_eq!(dot(&v, &v).to_bits(), sum_squares(&v).to_bits());
    }

    /// The sums that the code must not be: each one is another order, or a fused product.
    #[test]
    fn other_orders_give_other_bits() {
        let (h, v) = pair();
        let want = f32::from_bits(DOT_BITS);
        // One lane, in index order.
        let sequential = h.iter().zip(&v).fold(0.0f32, |s, (x, y)| s + x * y);
        // The 8 lanes, added one after the other.
        let mut lanes = [0.0f32; 8];
        for (j, (x, y)) in h.iter().zip(&v).enumerate() {
            lanes[j % 8] += x * y;
        }
        let linear = lanes.iter().fold(0.0f32, |s, x| s + x);
        // The 8 lanes, joined by distance 4 and not by neighbours.
        let far = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
        // 4 lanes.
        let mut four = [0.0f32; 4];
        for (j, (x, y)) in h.iter().zip(&v).enumerate() {
            four[j % 4] += x * y;
        }
        let four = (four[0] + four[1]) + (four[2] + four[3]);
        // The product fused with the sum.
        let mut fused = [0.0f32; 8];
        for (j, (x, y)) in h.iter().zip(&v).enumerate() {
            fused[j % 8] = x.mul_add(*y, fused[j % 8]);
        }
        let fused = join(fused);
        for (name, other) in [
            ("sequential", sequential),
            ("linear join", linear),
            ("far join", far),
            ("4 lanes", four),
            ("fused product", fused),
        ] {
            assert_ne!(
                other.to_bits(),
                want.to_bits(),
                "{name} gives the same bits, so the vector is too weak"
            );
        }
    }

    #[test]
    fn an_add_is_h_plus_the_rounded_product_in_bits() {
        let (h, v) = pair();
        let a = 0.37f32;
        let mut row = h.clone();
        assert!(apply_row(ControlOp::Add, a, &mut row, &v, sum_squares(&v)));
        let mut fused_differs = false;
        for j in 0..WIDTH {
            let product = a * v[j];
            assert_eq!(row[j].to_bits(), (h[j] + product).to_bits(), "index {j}");
            fused_differs |= a.mul_add(v[j], h[j]).to_bits() != row[j].to_bits();
        }
        assert!(
            fused_differs,
            "a fused product gives other bits for this pair, so the test sees it"
        );
    }

    /// The dot product of `h` and `v`, over the norm of `v`, in `f64`.
    fn along(h: &[f32], v: &[f32]) -> f64 {
        let dot: f64 = h.iter().zip(v).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let vv: f64 = v.iter().map(|y| f64::from(*y) * f64::from(*y)).sum();
        dot / vv.sqrt()
    }

    #[test]
    fn an_ablate_of_1_leaves_nothing_along_the_vector() {
        let (h, v) = pair();
        // Put a part of 3 vectors into the row, so that the part along the vector is large.
        let h: Vec<f32> = h.iter().zip(&v).map(|(x, y)| x + 3.0 * y).collect();
        let before = along(&h, &v);
        assert!(before.abs() > 1.0, "the row has a part along the vector: {before}");
        let mut row = h.clone();
        assert!(apply_row(ControlOp::Ablate, 1.0, &mut row, &v, sum_squares(&v)));
        let after = along(&row, &v);
        assert!(after.abs() < before.abs() * 1e-4, "{after} is left of {before}");
        // Half of the part goes with a strength of 0.5.
        let mut row = h.clone();
        apply_row(ControlOp::Ablate, 0.5, &mut row, &v, sum_squares(&v));
        assert!((along(&row, &v) - before * 0.5).abs() < before.abs() * 1e-4);
    }

    #[test]
    fn a_clamp_of_a_puts_the_part_along_the_vector_at_a() {
        let (h, v) = pair();
        let h: Vec<f32> = h.iter().zip(&v).map(|(x, y)| x + 3.0 * y).collect();
        for a in [3.5f32, -2.0, 0.0] {
            let mut row = h.clone();
            assert!(apply_row(ControlOp::Clamp, a, &mut row, &v, sum_squares(&v)));
            let got = along(&row, &v);
            assert!((got - f64::from(a)).abs() < 1e-3, "clamp {a}: {got}");
        }
    }

    #[test]
    fn a_clamp_changes_only_the_part_along_the_vector() {
        // Two vectors that are not parallel: the part of `h` along `w` stays.
        let (h, v) = pair();
        let mut w = vec![0.0f32; WIDTH];
        w[0] = 1.0;
        let mut row = h.clone();
        apply_row(ControlOp::Clamp, 1.0, &mut row, &v, sum_squares(&v));
        // The difference is a multiple of `v`: its part along any perpendicular row is 0. Test with the difference.
        let diff: Vec<f32> = row.iter().zip(&h).map(|(a, b)| a - b).collect();
        let ratio = f64::from(diff[7]) / f64::from(v[7]);
        for j in [0usize, 100, 5000, 10_239] {
            assert!((f64::from(diff[j]) - ratio * f64::from(v[j])).abs() < 1e-3, "index {j}");
        }
    }

    #[test]
    fn a_zero_vector_changes_nothing() {
        let zero = vec![0.0f32; WIDTH];
        let mut row = vec![-0.0f32; WIDTH];
        for op in [ControlOp::Add, ControlOp::Ablate, ControlOp::Clamp] {
            assert!(!apply_row(op, 1.0, &mut row, &zero, 0.0), "{op:?}");
            assert!(
                row.iter().all(|x| x.to_bits() == (-0.0f32).to_bits()),
                "{op:?} wrote a row"
            );
        }
    }

    #[test]
    fn the_actions_run_in_the_order_of_the_list() {
        let (h, v) = pair();
        let w = lcg(7, WIDTH, 2.0);
        let vectors = [(v.clone(), sum_squares(&v)), (w.clone(), sum_squares(&w))];
        let lookup = |entry: u32| LayerVector {
            values: &vectors[entry as usize].0,
            sum_squares: vectors[entry as usize].1,
        };
        let action = |entry: u32, op: ControlOp, strength: f32| Action {
            segment: 0,
            rows: 0..1,
            entry,
            op,
            strength,
        };
        let forward = [action(0, ControlOp::Add, 2.0), action(1, ControlOp::Clamp, 1.0)];
        let backward = [action(1, ControlOp::Clamp, 1.0), action(0, ControlOp::Add, 2.0)];
        let mut a = h.clone();
        let mut b = h.clone();
        assert_eq!(apply_actions(&mut a, &forward, lookup), 2);
        assert_eq!(apply_actions(&mut b, &backward, lookup), 2);
        // The result of the list is the result of its steps, one after the other.
        let mut by_hand = h.clone();
        apply_row(ControlOp::Add, 2.0, &mut by_hand, &v, sum_squares(&v));
        apply_row(ControlOp::Clamp, 1.0, &mut by_hand, &w, sum_squares(&w));
        assert_eq!(a, by_hand);
        assert_ne!(a, b, "the order changes the bits");
    }

    #[test]
    fn an_action_changes_the_rows_of_its_segment_only() {
        let (h, v) = pair();
        let vv = sum_squares(&v);
        let mut rows: Vec<f32> = Vec::new();
        for _ in 0..4 {
            rows.extend(&h);
        }
        let action = Action {
            segment: 1,
            rows: 1..3,
            entry: 0,
            op: ControlOp::Add,
            strength: 1.0,
        };
        let changed = apply_actions(&mut rows, &[action], |_| LayerVector {
            values: &v,
            sum_squares: vv,
        });
        assert_eq!(changed, 2);
        assert_eq!(&rows[..WIDTH], &h[..], "row 0 is not in the segment");
        assert_ne!(&rows[WIDTH..2 * WIDTH], &h[..]);
        assert_ne!(&rows[2 * WIDTH..3 * WIDTH], &h[..]);
        assert_eq!(&rows[3 * WIDTH..], &h[..], "row 3 is not in the segment");
    }

    #[test]
    fn a_zero_vector_does_not_count_as_a_change() {
        let (h, _) = pair();
        let zero = vec![0.0f32; WIDTH];
        let mut rows = h.clone();
        let action = Action {
            segment: 0,
            rows: 0..1,
            entry: 0,
            op: ControlOp::Clamp,
            strength: 1.0,
        };
        let changed = apply_actions(&mut rows, &[action], |_| LayerVector {
            values: &zero,
            sum_squares: 0.0,
        });
        assert_eq!(changed, 0);
        assert_eq!(rows, h);
    }

    #[test]
    #[should_panic(expected = "the action names rows")]
    fn an_action_past_the_rows_is_a_defect() {
        let zero = vec![0.0f32; WIDTH];
        let mut rows = vec![0.0f32; WIDTH];
        let action = Action {
            segment: 0,
            rows: 0..2,
            entry: 0,
            op: ControlOp::Add,
            strength: 1.0,
        };
        apply_actions(&mut rows, &[action], |_| LayerVector {
            values: &zero,
            sum_squares: 1.0,
        });
    }

    // ---- the vector of the oracle: the twin of `tools/oracle_fixture.py` ----

    /// `v[l][j] = (((31*j + 17*l) mod 257) - 128) / 256`. Exact in f32.
    fn oracle_vector(layer: usize, j: usize) -> f32 {
        ((((31 * j + 17 * layer) % 257) as i32 - 128) as f32) / 256.0
    }

    /// (layer, j, the numerator of `v * 256`). `tools/test_oracle_fixture.py` has the same table.
    const VECTOR_CASES: [(usize, usize, i32); 6] = [
        (0, 0, -128),
        (0, 1, -97),
        (1, 0, -111),
        (24, 8, 14),
        (40, 10239, 52),
        (47, 5000, -71),
    ];

    #[test]
    fn the_vector_of_the_oracle_matches_the_python_tool() {
        for (layer, j, numerator) in VECTOR_CASES {
            assert_eq!(
                oracle_vector(layer, j),
                numerator as f32 / 256.0,
                "layer {layer}, index {j}"
            );
        }
        assert_eq!(oracle_vector(24, 8), 0.0546875);
        assert_eq!(oracle_vector(40, 10239), 0.203125);
    }

    #[test]
    fn the_table_of_the_python_test_is_the_table_here() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/test_oracle_fixture.py");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let table: Vec<(usize, usize, i32)> = text
            .lines()
            .skip_while(|line| !line.starts_with("VECTOR_CASES = ["))
            .skip(1)
            .take_while(|line| !line.starts_with(']'))
            .map(|line| {
                let numbers: Vec<i64> = line
                    .trim()
                    .trim_matches(|c| c == '(' || c == ')' || c == ',')
                    .split(',')
                    .map(|n| n.trim().parse().unwrap_or_else(|e| panic!("{line}: {e}")))
                    .collect();
                assert_eq!(numbers.len(), 3, "{line}");
                (numbers[0] as usize, numbers[1] as usize, numbers[2] as i32)
            })
            .collect();
        assert_eq!(
            table, VECTOR_CASES,
            "the two tables must agree: change both files together"
        );
    }

    #[test]
    fn the_oracle_vector_works_with_the_op() {
        // An add of the vector of layer 8 at strength 0.5 adds 0.5 * v to each value of the row.
        let v: Vec<f32> = (0..WIDTH).map(|j| oracle_vector(8, j)).collect();
        let mut row = vec![1.0f32; WIDTH];
        assert!(apply_row(ControlOp::Add, 0.5, &mut row, &v, sum_squares(&v)));
        for j in [0usize, 1, 999, 10_239] {
            assert_eq!(row[j], 1.0 + 0.5 * oracle_vector(8, j), "index {j}");
        }
    }

    // ---- the checks of a list ----

    #[test]
    fn a_list_that_fits_the_table_is_accepted() {
        let items = [
            item(0, ControlOp::Add, 1.0, None),
            item(7, ControlOp::Clamp, 0.0, Some(vec![(0, 1.0), (47, 2.0)])),
        ];
        assert_eq!(list_problem(0, &items), None);
        assert_eq!(list_problem(3, &[]), None);
    }

    #[test]
    fn a_list_that_breaks_a_rule_is_refused_and_the_message_names_the_segment() {
        let bad_index = [item(8, ControlOp::Add, 1.0, None)];
        let message = list_problem(2, &bad_index).expect("entry 8");
        assert!(
            message.starts_with("segment 2:") && message.contains("entry 8"),
            "{message}"
        );
        let nan = [item(0, ControlOp::Add, f32::NAN, None)];
        assert!(list_problem(0, &nan).expect("NaN").contains("finite"));
        let inf_layer = [item(0, ControlOp::Add, 1.0, Some(vec![(3, f32::INFINITY)]))];
        assert!(list_problem(0, &inf_layer).expect("Inf").contains("layer 3"));
        let unordered = [item(0, ControlOp::Add, 1.0, Some(vec![(5, 1.0), (4, 1.0)]))];
        assert!(list_problem(0, &unordered).expect("unordered").contains("rising order"));
        let twice = [item(0, ControlOp::Add, 1.0, Some(vec![(5, 1.0), (5, 1.0)]))];
        assert!(list_problem(0, &twice).is_some());
        let none = [item(0, ControlOp::Add, 1.0, Some(Vec::new()))];
        assert!(list_problem(0, &none).expect("no layer").contains("names no layer"));
        let past = [item(0, ControlOp::Add, 1.0, Some(vec![(3, 1.0), (48, 1.0)]))];
        let message = list_problem(1, &past).expect("layer 48");
        assert!(
            message.contains("item 0 names layer 48") && message.contains("0 to 47"),
            "{message}"
        );
    }

    // ---- the bytes of a write ----

    fn good_bytes() -> Vec<u8> {
        (0..LAYERS * WIDTH)
            .flat_map(|i| (((i % 17) as f32) - 8.0).to_le_bytes())
            .collect()
    }

    #[test]
    fn a_good_write_is_accepted() {
        assert_eq!(write_problem(0, &good_bytes()), None);
        assert_eq!(write_problem(7, &good_bytes()), None);
    }

    #[test]
    fn a_bad_write_is_refused_for_its_index_its_size_or_a_value() {
        let bytes = good_bytes();
        let message = write_problem(8, &bytes).expect("entry 8");
        assert!(message.contains("8 entries, not entry 8"), "{message}");
        let message = write_problem(0, &bytes[..bytes.len() - 4]).expect("short");
        assert!(
            message.contains("1966076 bytes") && message.contains("1966080 bytes"),
            "{message}"
        );
        let mut long = bytes.clone();
        long.extend([0u8; 4]);
        assert!(write_problem(0, &long).is_some());
        for (name, bad) in [("NaN", f32::NAN), ("Inf", f32::INFINITY), ("-Inf", f32::NEG_INFINITY)] {
            let mut broken = bytes.clone();
            broken[4 * 1234..4 * 1234 + 4].copy_from_slice(&bad.to_le_bytes());
            let message = write_problem(0, &broken).expect(name);
            assert!(
                message.contains("value 1234") && message.contains("not finite"),
                "{name}: {message}"
            );
        }
    }

    #[test]
    fn the_sums_of_a_write_are_the_sums_of_squares_of_the_layers() {
        let bytes = good_bytes();
        let sums = write_sums(3, &bytes).expect("a good write");
        assert_eq!(sums.len(), 48);
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        for layer in [0usize, 1, 20, 47] {
            let row = &values[layer * WIDTH..(layer + 1) * WIDTH];
            assert_eq!(sums[layer].to_bits(), sum_squares(row).to_bits(), "layer {layer}");
        }
        assert!(write_sums(8, &bytes).is_err());
        assert!(write_sums(0, &bytes[4..]).is_err());
    }

    #[test]
    fn the_sum_of_a_layer_has_the_bits_of_the_8_lane_sum() {
        // Layer 2 holds the vector of `pair`, whose sum of squares depends on the order of the sum.
        let (_, v) = pair();
        let mut bytes = good_bytes();
        let at = 2 * WIDTH * 4;
        for (j, x) in v.iter().enumerate() {
            bytes[at + 4 * j..at + 4 * j + 4].copy_from_slice(&x.to_le_bytes());
        }
        let sums = write_sums(0, &bytes).expect("a good write");
        assert_eq!(sums[2].to_bits(), SUM_SQUARES_BITS);
    }

    #[test]
    fn a_layer_of_zeros_has_the_sum_zero() {
        let mut bytes = good_bytes();
        bytes[WIDTH * 4 * 5..WIDTH * 4 * 6].fill(0);
        let sums = write_sums(0, &bytes).expect("a good write");
        assert_eq!(sums[5], 0.0);
        assert!(sums[4] > 0.0 && sums[6] > 0.0);
    }

    // ---- the names of the taps ----

    fn tap(name: &str) -> Tap {
        Tap { name: name.to_owned() }
    }

    #[test]
    fn the_taps_of_a_pass_are_read_by_name() {
        assert_eq!(tap_set(&[]), Ok(TapSet::default()));
        assert_eq!(
            tap_set(&[tap("residual")]),
            Ok(TapSet {
                residual: true,
                fold: false
            })
        );
        assert_eq!(
            tap_set(&[tap("fold"), tap("residual")]),
            Ok(TapSet {
                residual: true,
                fold: true
            })
        );
    }

    #[test]
    fn an_unknown_tap_and_a_tap_that_comes_twice_are_refused() {
        for name in ["lens", "Residual", ""] {
            let message = tap_set(&[tap(name)]).expect_err(name);
            assert!(
                message.contains(&format!("`{name}`")) && message.contains("`residual` and `fold`"),
                "{message}"
            );
        }
        let message = tap_set(&[tap("fold"), tap("fold")]).expect_err("twice");
        assert!(message.contains("`fold` twice"), "{message}");
    }

    // ---- the layout of the tap outputs ----

    #[test]
    fn two_logit_rows_give_two_residual_outputs_and_then_two_fold_outputs() {
        let both = TapSet {
            residual: true,
            fold: true,
        };
        let outputs = tap_layout(&[true, false, true], both).expect("a layout");
        let named: Vec<(&str, usize, usize)> = outputs.iter().map(|o| (o.name, o.segment, o.bytes)).collect();
        assert_eq!(
            named,
            [
                ("residual", 0, 1_966_080),
                ("residual", 2, 1_966_080),
                ("fold", 0, 10_240),
                ("fold", 2, 10_240),
            ]
        );
    }

    #[test]
    fn one_tap_gives_its_outputs_alone_and_no_tap_gives_none() {
        let residual = TapSet {
            residual: true,
            fold: false,
        };
        let fold = TapSet {
            residual: false,
            fold: true,
        };
        let outputs = tap_layout(&[false, true], residual).expect("a layout");
        assert_eq!(outputs.len(), 1);
        assert_eq!((outputs[0].name, outputs[0].segment), ("residual", 1));
        let outputs = tap_layout(&[true, true, true], fold).expect("a layout");
        assert_eq!(outputs.iter().map(|o| o.segment).collect::<Vec<_>>(), [0, 1, 2]);
        assert!(outputs.iter().all(|o| o.name == "fold" && o.bytes == FOLD_BYTES));
        assert_eq!(tap_layout(&[true], TapSet::default()), Ok(Vec::new()));
        assert_eq!(tap_layout(&[false, false], TapSet::default()), Ok(Vec::new()));
    }

    #[test]
    fn a_tap_with_no_logit_row_is_refused() {
        let residual = TapSet {
            residual: true,
            fold: false,
        };
        let message = tap_layout(&[false, false], residual).expect_err("no logit row");
        assert!(message.contains("no segment asks for a logit row"), "{message}");
        assert!(tap_layout(&[], residual).is_err());
    }

    // ---- the cut plan ----

    fn layers_of(plan: &CutPlan) -> Vec<usize> {
        plan.cuts.iter().map(|cut| cut.layer).collect()
    }

    #[test]
    fn a_plain_pass_has_no_cut_and_keeps_the_fused_combine() {
        let segments = [shape(5, true, &[]), shape(1, true, &[])];
        let plan = plan_cuts(&segments, TapSet::default()).expect("a plan");
        assert_eq!(plan.cuts, []);
        assert!(!plan.fuse_hc_off);
    }

    #[test]
    fn the_tap_residual_cuts_at_each_layer_and_reads_the_last_row_of_each_logit_segment() {
        // Segment 0 has rows 0 to 2, segment 1 has rows 3 and 4 and no logit row, segment 2 has row 5.
        let segments = [shape(3, true, &[]), shape(2, false, &[]), shape(1, true, &[])];
        let taps = TapSet {
            residual: true,
            fold: false,
        };
        let plan = plan_cuts(&segments, taps).expect("a plan");
        assert_eq!(layers_of(&plan), (0..48).collect::<Vec<_>>());
        let reads = [ReadRow { segment: 0, row: 2 }, ReadRow { segment: 2, row: 5 }];
        for cut in &plan.cuts {
            assert_eq!(cut.reads, reads, "layer {}", cut.layer);
            assert_eq!(cut.actions, []);
        }
        assert!(!plan.fuse_hc_off, "a read changes no row, so the fused combine stays");
    }

    #[test]
    fn the_tap_fold_alone_has_no_cut() {
        let fold = TapSet {
            residual: false,
            fold: true,
        };
        let plan = plan_cuts(&[shape(4, true, &[])], fold).expect("a plan");
        assert_eq!(plan.cuts, []);
        assert!(!plan.fuse_hc_off);
    }

    #[test]
    fn both_taps_cut_like_the_tap_residual() {
        let both = TapSet {
            residual: true,
            fold: true,
        };
        let plan = plan_cuts(&[shape(4, true, &[])], both).expect("a plan");
        assert_eq!(plan.cuts.len(), 48);
    }

    #[test]
    fn a_list_on_two_layers_cuts_twice_with_the_rows_of_its_own_segment() {
        let list = [item(2, ControlOp::Add, 1.0, Some(vec![(3, 0.5), (20, 2.0)]))];
        // The list is on segment 1, which has rows 3 and 4. Segment 0 has rows 0 to 2.
        let segments = [shape(3, true, &[]), shape(2, true, &list), shape(1, true, &[])];
        let plan = plan_cuts(&segments, TapSet::default()).expect("a plan");
        assert_eq!(layers_of(&plan), [3, 20]);
        assert!(plan.fuse_hc_off, "a cut that changes rows needs the fused combine off");
        let expected = |strength: f32| Action {
            segment: 1,
            rows: 3..5,
            entry: 2,
            op: ControlOp::Add,
            strength,
        };
        assert_eq!(plan.cuts[0].actions, [expected(0.5)]);
        assert_eq!(plan.cuts[1].actions, [expected(2.0)]);
        assert!(plan.cuts.iter().all(|cut| cut.reads.is_empty()));
    }

    #[test]
    fn a_list_with_no_layers_acts_on_each_layer() {
        let list = [item(0, ControlOp::Ablate, 1.0, None)];
        let plan = plan_cuts(&[shape(1, false, &list)], TapSet::default()).expect("a plan");
        assert_eq!(plan.cuts.len(), 48);
        assert!(
            plan.cuts
                .iter()
                .all(|cut| cut.actions.len() == 1 && cut.actions[0].strength == 1.0)
        );
    }

    #[test]
    fn a_strength_of_0_makes_no_action_for_an_add_or_an_ablate_and_one_for_a_clamp() {
        for op in [ControlOp::Add, ControlOp::Ablate] {
            let whole = [item(0, op, 0.0, None)];
            let plan = plan_cuts(&[shape(2, true, &whole)], TapSet::default()).expect("a plan");
            assert_eq!(plan.cuts, [], "{op:?} of 0 on each layer");
            assert!(!plan.fuse_hc_off);
            let by_layer = [item(0, op, 1.0, Some(vec![(8, 0.0), (9, 1.0)]))];
            let plan = plan_cuts(&[shape(2, true, &by_layer)], TapSet::default()).expect("a plan");
            assert_eq!(layers_of(&plan), [9], "{op:?} of 0 at layer 8");
        }
        let whole = [item(0, ControlOp::Clamp, 0.0, None)];
        let plan = plan_cuts(&[shape(2, true, &whole)], TapSet::default()).expect("a plan");
        assert_eq!(plan.cuts.len(), 48, "a clamp of 0 removes the concept");
        assert!(plan.fuse_hc_off);
        let by_layer = [item(0, ControlOp::Clamp, 1.0, Some(vec![(8, 0.0)]))];
        let plan = plan_cuts(&[shape(2, true, &by_layer)], TapSet::default()).expect("a plan");
        assert_eq!(layers_of(&plan), [8]);
        assert_eq!(plan.cuts[0].actions[0].strength, 0.0);
    }

    #[test]
    fn a_layer_that_the_item_leaves_out_has_no_action() {
        let list = [item(0, ControlOp::Clamp, 1.0, Some(vec![(5, 1.0)]))];
        let plan = plan_cuts(&[shape(1, true, &list)], TapSet::default()).expect("a plan");
        assert_eq!(layers_of(&plan), [5]);
    }

    #[test]
    fn two_items_on_one_layer_keep_the_order_of_the_list() {
        let list = [
            item(4, ControlOp::Add, 1.0, Some(vec![(10, 1.0)])),
            item(1, ControlOp::Clamp, 2.0, Some(vec![(10, 2.0)])),
        ];
        let plan = plan_cuts(&[shape(1, false, &list)], TapSet::default()).expect("a plan");
        assert_eq!(layers_of(&plan), [10]);
        let entries: Vec<u32> = plan.cuts[0].actions.iter().map(|a| a.entry).collect();
        assert_eq!(entries, [4, 1], "the list order, not the entry order");
    }

    #[test]
    fn two_lists_on_two_segments_each_keep_their_own_rows() {
        let first = [item(0, ControlOp::Add, 1.0, Some(vec![(7, 1.0)]))];
        let second = [item(1, ControlOp::Ablate, 1.0, Some(vec![(7, 1.0)]))];
        let segments = [shape(2, true, &first), shape(3, true, &[]), shape(1, true, &second)];
        let plan = plan_cuts(&segments, TapSet::default()).expect("a plan");
        assert_eq!(layers_of(&plan), [7]);
        let actions = &plan.cuts[0].actions;
        assert_eq!(actions.len(), 2);
        assert_eq!(
            (actions[0].segment, actions[0].rows.clone(), actions[0].entry),
            (0, 0..2, 0)
        );
        assert_eq!(
            (actions[1].segment, actions[1].rows.clone(), actions[1].entry),
            (2, 5..6, 1)
        );
    }

    #[test]
    fn a_tap_and_a_list_share_the_cuts() {
        let list = [item(0, ControlOp::Add, 1.0, Some(vec![(3, 1.0)]))];
        let segments = [shape(2, true, &list)];
        let taps = TapSet {
            residual: true,
            fold: false,
        };
        let plan = plan_cuts(&segments, taps).expect("a plan");
        assert_eq!(plan.cuts.len(), 48);
        for cut in &plan.cuts {
            assert_eq!(cut.reads, [ReadRow { segment: 0, row: 1 }]);
            assert_eq!(cut.actions.len(), usize::from(cut.layer == 3), "layer {}", cut.layer);
        }
        assert!(plan.fuse_hc_off);
    }

    #[test]
    fn a_bad_list_makes_the_plan_fail_and_names_its_segment() {
        let good = [item(0, ControlOp::Add, 1.0, None)];
        let bad = [item(9, ControlOp::Add, 1.0, None)];
        let message =
            plan_cuts(&[shape(1, true, &good), shape(1, true, &bad)], TapSet::default()).expect_err("entry 9");
        assert!(message.starts_with("segment 1:"), "{message}");
        let past = [item(0, ControlOp::Add, 1.0, Some(vec![(48, 1.0)]))];
        let message = plan_cuts(&[shape(1, true, &past)], TapSet::default()).expect_err("layer 48");
        assert!(message.contains("layer 48"), "{message}");
    }

    #[test]
    #[should_panic(expected = "has no rows")]
    fn a_segment_with_no_rows_is_a_defect() {
        let _ = plan_cuts(&[shape(0, true, &[])], TapSet::default());
    }
}
