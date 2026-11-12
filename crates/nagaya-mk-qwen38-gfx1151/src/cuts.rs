//! The host steps of a pass that is cut at layer ends (milestone 34, engine-abi "The residual of the Qwen3.8 engine").
//!
//! `host::run_pass` launches the ranges of a cut pass. After each launch that ends a layer, it calls `Cuts::at_cut`.
//! The step reads the rows of the tap `residual`. After the last launch, `Cuts::finish` reads the rows of the tap
//! `fold`. Every access to the device goes through `hip_rt_memcpy` of the binding: the host never addresses device
//! memory. The control items come with the next commit, and until then the engine refuses a control list, so no
//! pass reaches an action.

use crate::binding::Binding;
use crate::context::Copy as CopyKind;
use crate::control::{self, CutPlan, FOLD_WIDTH, LAYERS, SegmentShape, TapOutput, TapSet, WIDTH};

/// The device buffers that a host step reads and writes.
pub(crate) struct CutDevice<'a> {
    pub binding: &'a Binding,
    /// The scratch `h`: the 4 streams of each row of the pass, `WIDTH` values for each row.
    pub h: *mut f32,
    /// The scratch `mixed`: the row that the head reads, `FOLD_WIDTH` values for each row.
    pub mixed: *const f32,
}

impl CutDevice<'_> {
    /// Copy `out.len()` values from the device at `from` into `out`.
    ///
    /// # Safety
    ///
    /// `from` names that many readable values on the device.
    unsafe fn read(&self, from: *const f32, out: &mut [f32]) {
        unsafe {
            (self.binding.rt_memcpy)(
                out.as_mut_ptr().cast(),
                from.cast(),
                std::mem::size_of_val(out) as u64,
                CopyKind::DeviceToHost as u32,
            )
        };
    }
}

/// The host steps of one cut pass, and the rows that its taps read.
pub(crate) struct Cuts {
    plan: CutPlan,
    taps: TapSet,
    /// The layout of the tap outputs, in the order of the outputs.
    layout: Vec<TapOutput>,
    /// The row of the head of each segment that asks for its logit row, in segment order.
    fold_rows: Vec<usize>,
    /// For the next cut of the plan, its index in `plan.cuts`.
    next: usize,
    /// The `residual` of each logit row: 48 rows of `WIDTH` values.
    residual: Vec<Vec<f32>>,
    /// The `fold` of each logit row.
    fold: Vec<Vec<f32>>,
}

impl Cuts {
    /// Plan the cuts of a pass of `shapes` with the taps `taps`, or say why the pass is refused.
    pub(crate) fn new(shapes: &[SegmentShape], taps: TapSet) -> Result<Cuts, String> {
        let wants: Vec<bool> = shapes.iter().map(|shape| shape.logit_row).collect();
        let layout = control::tap_layout(&wants, taps)?;
        let plan = control::plan_cuts(shapes, taps)?;
        let mut start = 0;
        let mut fold_rows = Vec::new();
        for shape in shapes {
            if shape.logit_row {
                fold_rows.push(start + shape.rows - 1);
            }
            start += shape.rows;
        }
        let reads = if taps.residual { fold_rows.len() } else { 0 };
        Ok(Cuts {
            plan,
            taps,
            layout,
            residual: vec![vec![0.0; LAYERS * WIDTH]; reads],
            fold: vec![vec![0.0; FOLD_WIDTH]; if taps.fold { fold_rows.len() } else { 0 }],
            fold_rows,
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

    /// The host step after the launch that ends layer `layer`: the reads of the tap.
    ///
    /// # Safety
    ///
    /// `device` names the scratch of the pass, and the launch has ended.
    pub(crate) unsafe fn at_cut(&mut self, device: &CutDevice, layer: usize) -> Result<(), String> {
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
        for (k, read) in cut.reads.iter().enumerate() {
            let row = &mut self.residual[k][layer * WIDTH..(layer + 1) * WIDTH];
            unsafe { device.read(device.h.wrapping_add(read.row * WIDTH), row) };
        }
        Ok(())
    }

    /// The host step after the last launch: the rows of the tap `fold`.
    ///
    /// # Safety
    ///
    /// As for `at_cut`.
    pub(crate) unsafe fn finish(&mut self, device: &CutDevice) -> Result<(), String> {
        if self.next != self.plan.cuts.len() {
            return Err(format!(
                "the pass ran {} of its {} cuts",
                self.next,
                self.plan.cuts.len()
            ));
        }
        if self.taps.fold {
            for (k, row) in self.fold_rows.iter().enumerate() {
                unsafe { device.read(device.mixed.wrapping_add(row * FOLD_WIDTH), &mut self.fold[k]) };
            }
        }
        Ok(())
    }

    /// The bytes of the tap outputs, in the order of the layout: each `residual`, then each `fold`. A size that differs
    /// from the layout is a fault.
    pub(crate) fn into_outputs(self) -> Result<Vec<(&'static str, Vec<u8>)>, String> {
        let rows = self.residual.into_iter().chain(self.fold);
        let mut outputs = Vec::with_capacity(self.layout.len());
        for (want, values) in self.layout.iter().zip(rows) {
            let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
            if bytes.len() != want.bytes {
                return Err(format!(
                    "the tap `{}` of segment {} holds {} bytes, but the layout says {}",
                    want.name,
                    want.segment,
                    bytes.len(),
                    want.bytes
                ));
            }
            outputs.push((want.name, bytes));
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use nagaya_abi::{ControlItem, ControlOp};

    use super::*;
    use crate::binding::{PassStop, Qwen38Launch};
    use crate::control::{FOLD_BYTES, RESIDUAL_BYTES};
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
    const BOTH: TapSet = TapSet {
        residual: true,
        fold: true,
    };

    /// What the fake kernel did before each launch: the first `rows` rows of `h`, as the launch saw them.
    type Seen = Rc<RefCell<Vec<Vec<f32>>>>;

    /// Make the fake kernel fill the first `rows` rows of `h` with `pattern(k, ..)` at each launch `k`. It records what
    /// `h` held before the fill: the rows that the host step left after the launch before.
    fn fill_h(rig: &mut Rig, rows: usize) -> Seen {
        let h = rig.h() as usize;
        let seen: Seen = Rc::default();
        let log = Rc::clone(&seen);
        fake::on_launch(move |k, _launch: &Launch| {
            // SAFETY: `h` is a buffer of the fake device that holds 256 rows.
            let h = unsafe { std::slice::from_raw_parts_mut(h as *mut f32, rows * WIDTH) };
            log.borrow_mut().push(h.to_vec());
            for (row, values) in h.chunks_mut(WIDTH).enumerate() {
                for (j, value) in values.iter_mut().enumerate() {
                    *value = pattern(k, row, j);
                }
            }
            Qwen38Launch::default()
        });
        seen
    }

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

    /// The copies from the device whose source is inside `[from, from + bytes)`.
    fn reads_of(from: usize, bytes: usize) -> Vec<fake::CopyRecord> {
        fake::copies()
            .into_iter()
            .filter(|c| c.kind == 2 && c.src >= from && c.src < from + bytes)
            .collect()
    }

    fn row_bytes() -> usize {
        WIDTH * 4
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

    // ---- piece 7: the taps ----------------------------------------------------------------------------------------------

    #[test]
    fn the_tap_residual_reads_the_last_row_of_each_layer_at_its_offset() {
        let mut rig = rig(&[3]);
        let h = rig.h() as usize;
        let _seen = fill_h(&mut rig, 3);
        let shapes = [shape(3, true, &[])];
        let mut cuts = Cuts::new(&shapes, RESIDUAL).expect("a tap");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a tapped pass runs");
        assert_eq!(fake::launches().len(), 49);
        // The row of h is the 4 streams of the pass's repeat stage: its stride is 4 x d values.
        let first = fake::launches()[0].clone();
        let repeat = fake::stages_of(&first)
            .iter()
            .find(|stage| stage.op == crate::stage::ST_REPEAT4)
            .expect("the repeat stage");
        assert_eq!(repeat.rep().h as usize, h, "the repeat writes the scratch h");
        let stride = 4 * repeat.rep().d as usize * 4;
        assert_eq!(stride, row_bytes());
        let reads = reads_of(h, 3 * stride);
        assert_eq!(reads.len(), 48, "one read for each layer: 48 x 1 logit row");
        for (layer, read) in reads.iter().enumerate() {
            assert_eq!(
                read.src,
                h + 2 * stride,
                "layer {layer}: the last row of the segment, row 2"
            );
            assert_eq!(read.bytes as usize, stride, "layer {layer}: one row");
        }
        // The rows came from the device: the row after launch k is pattern(k, 2, ..).
        let outputs = cuts.into_outputs().expect("the outputs");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "residual");
        let bytes = &outputs[0].1;
        assert_eq!(bytes.len(), RESIDUAL_BYTES);
        for layer in [0, 1, 17, 47] {
            for j in [0, 1, 4_000, WIDTH - 1] {
                let at = (layer * WIDTH + j) * 4;
                let value = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                assert_eq!(
                    value.to_bits(),
                    pattern(layer, 2, j).to_bits(),
                    "layer {layer}, index {j}"
                );
            }
        }
    }

    #[test]
    fn the_tap_fold_alone_makes_one_launch_and_one_read_of_the_head_row() {
        let mut rig = rig(&[4, 2]);
        let mixed = rig.mixed() as usize;
        // The kernel leaves a row in `mixed` for each row of the pass.
        let rows = rig.rows();
        // SAFETY: `mixed` holds 256 rows of 2,560 values.
        let device = unsafe { std::slice::from_raw_parts_mut(mixed as *mut f32, rows * FOLD_WIDTH) };
        for (row, values) in device.chunks_mut(FOLD_WIDTH).enumerate() {
            for (j, value) in values.iter_mut().enumerate() {
                *value = pattern(9, row, j);
            }
        }
        let shapes = [shape(4, true, &[]), shape(2, true, &[])];
        let mut cuts = Cuts::new(&shapes, FOLD).expect("the fold");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a pass with the fold runs");
        assert_eq!(fake::launches().len(), 1, "the fold needs no cut");
        let reads = reads_of(mixed, rows * FOLD_WIDTH * 4);
        assert_eq!(reads.len(), 2, "one read for each logit row");
        assert_eq!(reads[0].src, mixed + 3 * FOLD_WIDTH * 4, "segment 0 ends at row 3");
        assert_eq!(reads[1].src, mixed + 5 * FOLD_WIDTH * 4, "segment 1 ends at row 5");
        // The head of the pass reads the same rows: x_row names the last row of each segment.
        let head = fake::stages_of(&fake::launches()[0]).last().expect("the head").qmm().x;
        assert_eq!(
            head as usize, mixed,
            "the head of two segments reads mixed by its row map"
        );
        let outputs = cuts.into_outputs().expect("the outputs");
        let names: Vec<&str> = outputs.iter().map(|o| o.0).collect();
        assert_eq!(names, ["fold", "fold"]);
        for (k, (_, bytes)) in outputs.iter().enumerate() {
            assert_eq!(bytes.len(), FOLD_BYTES);
            let row = [3, 5][k];
            let at = 77 * 4;
            let value = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            assert_eq!(value.to_bits(), pattern(9, row, 77).to_bits(), "logit row {k}");
        }
    }

    #[test]
    fn the_fold_of_one_segment_is_the_row_that_the_head_reads_at_its_last_row() {
        // The head of a pass of one segment reads `x = mixed + (t - 1) * 2560`. The fold reads the same row.
        let mut rig = rig(&[5]);
        let mixed = rig.mixed() as usize;
        let shapes = [shape(5, true, &[])];
        let mut cuts = Cuts::new(&shapes, FOLD).expect("the fold");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a pass with the fold runs");
        let head = fake::stages_of(&fake::launches()[0]).last().expect("the head").qmm().x as usize;
        let reads = reads_of(mixed, 5 * FOLD_WIDTH * 4);
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].src, head, "the fold is the row that the head reads");
    }

    #[test]
    fn two_taps_on_three_segments_give_the_outputs_of_the_layout() {
        let mut rig = rig(&[2, 1, 3]);
        let shapes = [shape(2, true, &[]), shape(1, false, &[]), shape(3, true, &[])];
        let mut cuts = Cuts::new(&shapes, BOTH).expect("both taps");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a tapped pass runs");
        let outputs = cuts.into_outputs().expect("the outputs");
        let described: Vec<(&str, usize)> = outputs.iter().map(|(name, bytes)| (*name, bytes.len())).collect();
        // Segment 1 asks for no logit row, so it has no tap output: 2 rows, `residual` first.
        assert_eq!(
            described,
            [
                ("residual", RESIDUAL_BYTES),
                ("residual", RESIDUAL_BYTES),
                ("fold", FOLD_BYTES),
                ("fold", FOLD_BYTES)
            ]
        );
        // Only the segments with a logit row were read: rows 1 and 5 of h, never row 2 (segment 1).
        let h = rig.h() as usize;
        let reads = reads_of(h, 6 * row_bytes());
        assert_eq!(reads.len(), 96, "48 layers x 2 logit rows");
        assert!(reads.iter().all(|r| {
            let row = (r.src - h) / row_bytes();
            row == 1 || row == 5
        }));
        assert!(reads.iter().step_by(2).all(|r| (r.src - h) / row_bytes() == 1));
    }

    #[test]
    fn a_tap_with_no_logit_row_and_a_tap_twice_are_refused_before_the_pass() {
        let shapes = [shape(2, false, &[])];
        let error = Cuts::new(&shapes, RESIDUAL).err().expect("no logit row");
        assert!(error.contains("no segment asks for a logit row"), "{error}");
        // Nothing planned: a pass with no tap and no list has no cut, and the worker makes no `Cuts` for it.
        let none = Cuts::new(&shapes, TapSet::default()).expect("no tap");
        assert!(none.layers().is_empty() && !none.fuse_hc_off());
    }

    #[test]
    fn a_tap_changes_no_byte_of_the_logits_or_of_the_stage_list() {
        // The host step reads and never writes h when no list is named. The logit rows come from the device after the
        // last launch, as in a plain pass: the logits copy has the same source and size.
        let mut rig = rig(&[3]);
        rig.run(FAST, fake::no_knobs(), None).expect("plain");
        let plain: Vec<_> = fake::copies().into_iter().filter(|c| c.dst != 0).collect();
        let plain_logits: Vec<_> = plain
            .iter()
            .filter(|c| c.kind == 2 && c.bytes == crate::pass::VOCAB as u64 * 4)
            .copied()
            .collect();
        assert_eq!(plain_logits.len(), 1);
        fake::reset();
        rig.rewind();
        let shapes = [shape(3, true, &[])];
        let mut cuts = Cuts::new(&shapes, BOTH).expect("both");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts)).expect("tapped");
        let tapped_logits: Vec<_> = fake::copies()
            .into_iter()
            .filter(|c| c.kind == 2 && c.bytes == crate::pass::VOCAB as u64 * 4)
            .collect();
        assert_eq!(tapped_logits.len(), 1);
        assert_eq!(tapped_logits[0].src, plain_logits[0].src);
        // No byte goes to the device for a tap: the only host-to-device copies are the pass's own (tokens, rows, PLE).
        let writes_to_h = fake::copies()
            .into_iter()
            .filter(|c| c.kind == 1 && c.dst == rig.h() as usize)
            .count();
        assert_eq!(writes_to_h, 0, "a tap writes no row of h");
    }
}
