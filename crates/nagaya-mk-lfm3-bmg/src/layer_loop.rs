//! The layer loops of LFM3 (engine-abi "Layer loops of LFM3"): the bounds of a span, the reserve of extra runs in a
//! slot, and the order of the runs of a pass.
//!
//! The layers of LFM3 repeat a group of four from the first full attention layer. A span starts and ends at the edge
//! of a group, so extra run `i` of any loop is a layer of the kind of layer `first + i mod period`. Thus entry `i` of
//! the reserve of a slot has one kind, and the parts of a state do not change with the loop.

use crate::spec::{Backbone, LayerKind};

/// The most runs of a span (engine-abi "Layer loops of LFM3").
pub const MOST_COUNT: usize = 4;

/// A span of layers that runs more than once: layers `start` to `end - 1` run `count` times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Loop {
    pub start: usize,
    pub end: usize,
    pub count: usize,
}

/// One run of a pass: the layer whose weights and kind it takes, and the state that it reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RunAt {
    pub layer: usize,
    /// The index of the state in the slot: the layer itself for its first run, or `layers + i` for entry `i` of the
    /// reserve.
    pub state: usize,
}

/// The repeating group of the layers: its first layer and its period. The group starts at the first full attention
/// layer, and the period is the distance to the next one. `None` when the layers after the first do not repeat it.
pub fn group(b: &Backbone) -> Option<(usize, usize)> {
    let full: Vec<usize> = b.layers_of(LayerKind::Full).collect();
    let (&first, &second) = (full.first()?, full.get(1)?);
    let period = second - first;
    let repeats = (first..b.layers.len()).all(|l| b.layers[l] == b.layers[first + (l - first) % period]);
    repeats.then_some((first, period))
}

/// The layers where a span may start or end: the edges of the groups. A span of whole groups keeps the order of kinds
/// on each run, so the last edge is the last that leaves a whole group before it.
pub fn bounds(b: &Backbone) -> Vec<usize> {
    let Some((first, period)) = group(b) else {
        return Vec::new();
    };
    (first..=b.layers.len()).step_by(period).collect()
}

/// The kind of entry `i` of the reserve of a slot: the kind of layer `first + i mod period`.
pub fn reserve_kind(b: &Backbone, i: usize) -> LayerKind {
    let (first, period) = group(b).expect("a backbone with layer loops repeats its group");
    b.layers[first + i % period]
}

impl Loop {
    /// The runs that the loop adds to each position.
    pub fn extra_runs(&self) -> usize {
        (self.end - self.start) * (self.count - 1)
    }
}

/// The runs of a pass in order, under `layer_loop` with a reserve of `extra` entries. Without a loop, each layer runs
/// once on its own state. The message of an error says which rule the loop breaks.
pub fn runs(b: &Backbone, layer_loop: Option<Loop>, extra: usize) -> Result<Vec<RunAt>, String> {
    let layers = b.layers.len();
    let mut out: Vec<RunAt> = (0..layers).map(|l| RunAt { layer: l, state: l }).collect();
    let Some(lp) = layer_loop else { return Ok(out) };
    let edges = bounds(b);
    if !edges.contains(&lp.start) || !edges.contains(&lp.end) || lp.end <= lp.start {
        return Err(format!(
            "the loop runs layers {} to {}; a span starts and ends at one of {edges:?}, with its end above its start",
            lp.start, lp.end
        ));
    }
    if lp.count == 0 || lp.count > MOST_COUNT {
        return Err(format!(
            "the loop runs its span {} times; the engine runs a span 1 to {MOST_COUNT} times",
            lp.count
        ));
    }
    if lp.extra_runs() > extra {
        return Err(format!(
            "the loop adds {} runs of a layer to each position; each slot holds {extra}",
            lp.extra_runs()
        ));
    }
    let mut entry = 0;
    let mut extra_runs = Vec::with_capacity(lp.extra_runs());
    for _ in 1..lp.count {
        for l in lp.start..lp.end {
            extra_runs.push(RunAt {
                layer: l,
                state: layers + entry,
            });
            entry += 1;
        }
    }
    out.splice(lp.end..lp.end, extra_runs);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::tests::{OMNI_CONFIG, backbone};

    #[test]
    fn the_groups_of_lfm3_start_at_layer_2_every_4_layers() {
        let b = backbone();
        assert_eq!(group(&b), Some((2, 4)));
        assert_eq!(bounds(&b), [2, 6, 10, 14, 18, 22]);
        let omni = Backbone::parse(OMNI_CONFIG).unwrap();
        assert_eq!(bounds(&omni), [2, 6, 10, 14, 18, 22]);
    }

    #[test]
    fn a_backbone_whose_layers_do_not_repeat_has_no_bounds() {
        let mut b = backbone();
        b.layers[13] = LayerKind::Sliding;
        assert_eq!(group(&b), None);
        assert!(bounds(&b).is_empty());
    }

    #[test]
    fn each_entry_of_the_reserve_has_the_kind_of_its_place_in_the_group() {
        let b = backbone();
        let kinds: Vec<LayerKind> = (0..8).map(|i| reserve_kind(&b, i)).collect();
        use LayerKind::{Conv, Full, Sliding};
        assert_eq!(kinds, [Full, Conv, Sliding, Conv, Full, Conv, Sliding, Conv]);
    }

    #[test]
    fn no_loop_runs_each_layer_once_on_its_own_state() {
        let b = backbone();
        let plain = runs(&b, None, 8).unwrap();
        assert_eq!(plain.len(), 24);
        assert!(plain.iter().enumerate().all(|(i, r)| r.layer == i && r.state == i));
        let once = runs(
            &b,
            Some(Loop {
                start: 10,
                end: 14,
                count: 1,
            }),
            0,
        )
        .unwrap();
        assert_eq!(once, plain, "count 1 is the plain pass");
    }

    #[test]
    fn a_loop_runs_its_span_again_on_the_entries_of_the_reserve_in_order() {
        let b = backbone();
        let got = runs(
            &b,
            Some(Loop {
                start: 10,
                end: 14,
                count: 3,
            }),
            8,
        )
        .unwrap();
        let layers: Vec<usize> = got.iter().map(|r| r.layer).collect();
        let mut want: Vec<usize> = (0..14).collect();
        want.extend([10, 11, 12, 13, 10, 11, 12, 13]);
        want.extend(14..24);
        assert_eq!(layers, want);
        let states: Vec<usize> = got.iter().map(|r| r.state).collect();
        assert_eq!(&states[..14], (0..14).collect::<Vec<_>>());
        assert_eq!(
            &states[14..22],
            (24..32).collect::<Vec<_>>(),
            "extra run i takes entry i"
        );
        assert_eq!(&states[22..], (14..24).collect::<Vec<_>>());
        for r in &got[14..22] {
            assert_eq!(
                b.layers[r.layer],
                reserve_kind(&b, r.state - 24),
                "an entry holds a state of its kind"
            );
        }
    }

    #[test]
    fn a_loop_that_breaks_a_rule_is_refused_with_the_rule() {
        let b = backbone();
        let at = |start, end, count, extra| runs(&b, Some(Loop { start, end, count }), extra).unwrap_err();
        assert!(at(11, 14, 2, 8).contains("starts and ends at one of"));
        assert!(at(10, 13, 2, 8).contains("starts and ends at one of"));
        assert!(at(14, 10, 2, 8).contains("end above its start"));
        assert!(at(10, 10, 2, 8).contains("end above its start"));
        assert!(at(10, 14, 0, 8).contains("1 to 4 times"));
        assert!(at(10, 14, 5, 32).contains("1 to 4 times"));
        assert!(at(6, 14, 2, 4).contains("each slot holds 4"));
        assert!(at(10, 14, 2, 0).contains("each slot holds 0"));
    }
}
