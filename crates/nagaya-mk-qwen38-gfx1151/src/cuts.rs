//! The host steps of a pass that is cut at layer ends (milestone 34, engine-abi "The residual of the Qwen3.8 engine").
//!
//! `host::run_pass` launches the ranges of a cut pass. After each launch that ends a layer, it calls `Cuts::at_cut`.
//! After the last launch, it calls `Cuts::finish`. This commit holds the plan and the order of the steps. The steps
//! read and write no row yet: the tap `residual` and the control items come with the next two commits, so no pass
//! reaches `Cuts` with a tap or a list until then.

use crate::control::{self, CutPlan, SegmentShape, TapSet};

/// The host steps of one cut pass.
pub(crate) struct Cuts {
    plan: CutPlan,
    /// For the next cut of the plan, its index in `plan.cuts`.
    next: usize,
}

impl Cuts {
    /// Plan the cuts of a pass of `shapes` with the taps `taps`, or say why the pass is refused. The worker calls it
    /// from the next commit, so only the tests call it now.
    #[allow(dead_code)]
    pub(crate) fn new(shapes: &[SegmentShape], taps: TapSet) -> Result<Cuts, String> {
        Ok(Cuts {
            plan: control::plan_cuts(shapes, taps)?,
            next: 0,
        })
    }

    /// The layers whose ends cut the pass, in rising order.
    pub(crate) fn layers(&self) -> Vec<usize> {
        self.plan.cuts.iter().map(|cut| cut.layer).collect()
    }

    /// True when the pass builds with the fused combine off.
    pub(crate) fn fuse_hc_off(&self) -> bool {
        self.plan.fuse_hc_off
    }

    /// The host step after the launch that ends layer `layer`. The plan names the layers in order.
    pub(crate) fn at_cut(&mut self, layer: usize) -> Result<(), String> {
        let Some(cut) = self.plan.cuts.get(self.next) else {
            return Err(format!("the plan has no cut after layer {layer}"));
        };
        if cut.layer != layer {
            return Err(format!(
                "the plan cuts after layer {}, but the pass stopped after layer {layer}",
                cut.layer
            ));
        }
        self.next += 1;
        Ok(())
    }

    /// The host step after the last launch. Each cut of the plan must have run.
    pub(crate) fn finish(&mut self) -> Result<(), String> {
        if self.next != self.plan.cuts.len() {
            return Err(format!(
                "the pass ran {} of its {} cuts",
                self.next,
                self.plan.cuts.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use nagaya_abi::{ControlItem, ControlOp};

    use super::*;
    use crate::binding::{PassStop, Qwen38Launch};
    use crate::control::WIDTH;
    use crate::fake::{self, FAST, Launch, Rig, StageKey};
    use crate::host::LaunchKnobs;
    use crate::stage::{ST_HC_COMBINE_RMS, Stage};

    const SPIN_TIMEOUT: u32 = 2;

    /// A made-up value of the residual after the launch `k`: finite, and different for each row and index.
    fn pattern(k: usize, row: usize, j: usize) -> f32 {
        ((k * 131 + row * 17 + j * 7) % 211) as f32 / 16.0 - 6.0
    }

    fn shape(rows: usize, logit_row: bool, control: &[ControlItem]) -> SegmentShape<'_> {
        SegmentShape {
            rows,
            logit_row,
            control,
        }
    }

    const RESIDUAL: TapSet = TapSet {
        residual: true,
        fold: false,
    };
    const FOLD: TapSet = TapSet {
        residual: false,
        fold: true,
    };
    /// What `h` held before each launch, as the fake kernel saw it.
    type Seen = Rc<RefCell<Vec<Vec<f32>>>>;

    /// The stage keys of each launch, recorded at the launch (the next upload overwrites the list).
    fn record_stages(rig: &mut Rig, rows: usize) -> (Seen, Rc<RefCell<Vec<Vec<StageKey>>>>) {
        let h = rig.h() as usize;
        let seen: Seen = Rc::default();
        let lists: Rc<RefCell<Vec<Vec<StageKey>>>> = Rc::default();
        let (log, keys) = (Rc::clone(&seen), Rc::clone(&lists));
        fake::on_launch(move |k, launch: &Launch| {
            // SAFETY: as in `fill_h`.
            let h = unsafe { std::slice::from_raw_parts_mut(h as *mut f32, rows * WIDTH) };
            log.borrow_mut().push(h.to_vec());
            for (row, values) in h.chunks_mut(WIDTH).enumerate() {
                for (j, value) in values.iter_mut().enumerate() {
                    *value = pattern(k, row, j);
                }
            }
            keys.borrow_mut().push(fake::stage_keys(launch));
            Qwen38Launch::default()
        });
        (seen, lists)
    }

    fn rig(lens: &[u32]) -> Rig {
        fake::reset();
        fake::clear_hook();
        Rig::new(lens)
    }

    fn plain_keys(lens: &[u32]) -> Vec<StageKey> {
        let mut rig = rig(lens);
        let (_, lists) = record_stages(&mut rig, 1);
        rig.run(FAST, fake::no_knobs(), None).expect("a plain pass runs");
        let lists = lists.borrow();
        assert_eq!(lists.len(), 1, "a plain pass is one launch");
        lists[0].clone()
    }

    // ---- piece 6: the launch cuts -------------------------------------------------------------------------------------

    #[test]
    fn a_plain_pass_makes_one_launch_of_the_whole_list_with_the_arguments_of_today() {
        let mut rig = rig(&[3]);
        rig.run(FAST, fake::no_knobs(), None).expect("a plain pass runs");
        let launches = fake::launches();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].grid, 80, "the occupancy of the fake device, capped at 80");
        assert_eq!(launches[0].skip, u32::MAX, "no mutation knob");
        // The list is the whole list: it ends with the head's matmul.
        let stages = fake::stages_of(&launches[0]);
        assert_eq!(stages.last().expect("stages").qmm().n, crate::pass::VOCAB);
        // The mutation knob reaches the one launch, as it did.
        fake::reset();
        let knobs = LaunchKnobs::from(|name| (name == "QWEN38_MUTATE_SKIP_BARRIER").then(|| "9".to_owned())).unwrap();
        rig.rewind();
        rig.run(FAST, knobs, None).expect("a plain pass with a knob runs");
        assert_eq!(fake::launches()[0].skip, 9);
    }

    #[test]
    fn a_pass_with_a_tap_runs_the_stages_of_a_plain_pass_in_ranges_with_no_gap_and_no_overlap() {
        let plain = plain_keys(&[3]);
        let mut rig = rig(&[3]);
        let (_, lists) = record_stages(&mut rig, 3);
        let shapes = [shape(3, true, &[])];
        let mut cuts = Cuts::new(&shapes, RESIDUAL).expect("a tap and a logit row");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a tapped pass runs");
        let lists = lists.borrow();
        assert_eq!(lists.len(), 49, "48 cuts make 49 launches");
        assert!(lists.iter().all(|list| !list.is_empty()), "no launch is empty");
        let joined: Vec<StageKey> = lists.iter().flatten().copied().collect();
        assert_eq!(
            joined, plain,
            "the ranges are the stages of the plain pass, in order, each once"
        );
        // The tap keeps the fused combine: only a control action turns it off.
        assert!(joined.iter().any(|key| key.0 == ST_HC_COMBINE_RMS));
        // The ranges follow each other in the uploaded list.
        let launches = fake::launches();
        let size = std::mem::size_of::<Stage>();
        for pair in launches.windows(2) {
            assert_eq!(pair[1].stages, pair[0].stages + pair[0].n as usize * size);
        }
    }

    #[test]
    fn a_pass_with_cuts_at_layers_3_and_20_has_3_ranges_that_cover_the_list() {
        let plain = plain_keys(&[2]);
        let mut rig = rig(&[2]);
        let (_, lists) = record_stages(&mut rig, 2);
        let list = [ControlItem {
            index: 0,
            op: ControlOp::Add,
            strength: 1.0,
            layers: Some(vec![(3, 1.0), (20, 1.0)]),
        }];
        let shapes = [shape(2, false, &list)];
        let mut cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a steered pass runs");
        let lists = lists.borrow();
        assert_eq!(lists.len(), 3);
        // A pass with an action builds with the fused combine off, so its list is not the plain list. The fused stages
        // are not in any range.
        assert!(lists.iter().flatten().all(|key| key.0 != ST_HC_COMBINE_RMS));
        assert!(plain.iter().any(|key| key.0 == ST_HC_COMBINE_RMS));
        // The ranges cover the list: layer 3 ends the first, layer 20 ends the second.
        let ends: Vec<u32> = lists.iter().map(|list| list.last().expect("stages").1).collect();
        assert_eq!(
            ends,
            [3, 20, 48],
            "the last stage of each launch: layer 3, layer 20, then the head"
        );
        let launches = fake::launches();
        let size = std::mem::size_of::<Stage>();
        assert_eq!(launches[1].stages, launches[0].stages + launches[0].n as usize * size);
        assert_eq!(launches[2].stages, launches[1].stages + launches[1].n as usize * size);
    }

    #[test]
    fn a_stop_inside_a_cut_names_the_stage_in_the_whole_list_and_poisons_the_contexts() {
        let mut rig = rig(&[2]);
        fake::on_launch(|k, _launch| {
            if k == 5 {
                // The launch that follows 5 cuts stops at its stage 7, on a spin-timeout that the watchdog did not ask for.
                Qwen38Launch {
                    status: SPIN_TIMEOUT,
                    site: 7,
                    wg: 3,
                    watchdog_stopped: 0,
                }
            } else {
                Qwen38Launch::default()
            }
        });
        let shapes = [shape(2, true, &[])];
        let mut cuts = Cuts::new(&shapes, RESIDUAL).expect("a tap");
        let error = rig
            .run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect_err("launch 5 stops");
        let PassStop::Stopped(message) = error else {
            panic!("a spin-timeout stops the pass: {error:?}");
        };
        let launches = fake::launches();
        assert_eq!(launches.len(), 6, "no launch follows the stop");
        let start = (launches[5].stages - launches[0].stages) / std::mem::size_of::<Stage>();
        assert!(start > 7, "the fifth range does not start at 0");
        assert!(message.contains(&format!("at stage {}", start + 7)), "{message}");
        assert!(message.contains("1 context(s) are poisoned"), "{message}");
        assert!(rig.contexts[0].poisoned, "the context is poisoned");
        assert_eq!(rig.contexts[0].n, 5, "a stopped pass advances nothing");
    }

    #[test]
    fn the_mutation_knobs_are_refused_for_a_pass_with_cuts_and_the_message_says_why() {
        let shapes = [shape(2, true, &[])];
        for name in ["QWEN38_MUTATE_SKIP_BARRIER", "QWEN38_MUTATE_STALL_BARRIER"] {
            let mut rig = rig(&[2]);
            let knobs = LaunchKnobs::from(|var| (var == name).then(|| "4".to_owned())).unwrap();
            let mut cuts = Cuts::new(&shapes, RESIDUAL).expect("a tap");
            let error = rig.run(FAST, knobs, Some(&mut cuts)).expect_err("a knob and cuts");
            let PassStop::Refused(message) = error else {
                panic!("the pass is refused before it runs: {error:?}");
            };
            assert!(message.contains(name) || message.contains("MUTATE"), "{message}");
            assert!(message.contains("49 launches"), "{message}");
            assert!(message.contains("unset the knob"), "{message}");
            assert!(fake::launches().is_empty(), "a refused pass launches nothing");
        }
        // A tap that makes no cut (`fold`) is one launch, so the knob reaches it.
        let mut rig = rig(&[2]);
        let knobs = LaunchKnobs::from(|var| (var == "QWEN38_MUTATE_SKIP_BARRIER").then(|| "4".to_owned())).unwrap();
        let mut cuts = Cuts::new(&shapes, FOLD).expect("the fold");
        rig.run(FAST, knobs, Some(&mut cuts))
            .expect("one launch takes the knob");
        assert_eq!(fake::launches().len(), 1);
        assert_eq!(fake::launches()[0].skip, 4);
    }
}
