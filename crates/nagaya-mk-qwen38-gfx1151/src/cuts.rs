//! The host steps of a pass that is cut at layer ends (engine-abi "The residual of the Qwen3.8 engine").
//!
//! `host::run_pass` launches the ranges of a cut pass. After each launch that ends a layer, it calls `Cuts::at_cut`.
//! The step applies the control items of the layer to the rows of `h` with the device op, and it reads the rows of
//! the tap `residual`.
//! After the last launch, `Cuts::finish` reads the rows of the tap `fold`. Every access to the device goes through
//! `hip_rt_memcpy` of the binding: the host never addresses device memory.
//!
//! The control table holds the vectors on the device. The host keeps only the sum of squares of each layer of each
//! entry. The control op runs on the device (`hip/src/engine/control.hip`), so the rows and the vectors stay there.

use std::ffi::c_void;

use nagaya_abi::ControlItem;

use crate::binding::Binding;
use crate::context::Copy as CopyKind;
use crate::control::{
    self, Action, CutPlan, FOLD_WIDTH, LAYERS, LIMIT, SegmentShape, TABLES, TapOutput, TapSet, WIDTH,
};

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

/// The table of control vectors: one device allocation of `LIMIT.table_bytes()`, and the sum of squares of each layer of
/// each entry.
pub(crate) struct ControlTable {
    /// The entries on the device, `LAYERS * WIDTH` values each, in layer order.
    device: *mut f32,
    /// The sum of squares of each layer of each entry. A layer that was never written has the sum 0.
    sums: Vec<Vec<f32>>,
    /// True for an entry that `write` filled.
    written: Vec<bool>,
}

impl ControlTable {
    /// Allocate the table on the device and fill it with zeros. This is the one allocation of the table, made at open.
    ///
    /// # Safety
    ///
    /// `binding` is the linked engine's, on the worker thread.
    pub(crate) unsafe fn new(binding: &Binding) -> ControlTable {
        let bytes = LIMIT.table_bytes();
        let device = unsafe { (binding.rt_malloc)(bytes) } as *mut f32;
        assert!(!device.is_null(), "hip_rt_malloc({bytes}) gave no control table");
        unsafe { (binding.rt_memset)(device.cast(), 0, bytes) };
        ControlTable {
            device,
            sums: vec![vec![0.0; LAYERS]; TABLES as usize],
            written: vec![false; TABLES as usize],
        }
    }

    /// The device address of one entry.
    fn entry(&self, index: u32) -> *mut f32 {
        assert!(
            index < TABLES,
            "the control table has {TABLES} entries, not entry {index}"
        );
        self.device.wrapping_add(index as usize * LAYERS * WIDTH)
    }

    /// Write one vector into an entry. A bad index, a bad size or a value that is not finite is an error, and nothing
    /// changes: the checks of `control::write_sums` run before the copy.
    ///
    /// # Safety
    ///
    /// `binding` is the linked engine's, on the worker thread.
    pub(crate) unsafe fn write(&mut self, binding: &Binding, index: u32, bytes: &[u8]) -> Result<(), String> {
        let sums = control::write_sums(index, bytes)?;
        unsafe {
            (binding.rt_memcpy)(
                self.entry(index).cast::<c_void>(),
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                CopyKind::HostToDevice as u32,
            )
        };
        self.sums[index as usize] = sums;
        self.written[index as usize] = true;
        Ok(())
    }

    /// The device address of the vector of one layer of one entry: `WIDTH` values.
    fn layer(&self, entry: u32, layer: usize) -> *const f32 {
        assert!(layer < LAYERS, "the model has {LAYERS} layers, not layer {layer}");
        self.entry(entry).wrapping_add(layer * WIDTH)
    }

    /// The first control item that names an entry that `write` never filled, or `None`. A pass over such an entry
    /// would read zeros and act on no row, and the caller meant a vector.
    pub(crate) fn unwritten(&self, lists: &[&[ControlItem]]) -> Option<String> {
        for (segment, items) in lists.iter().enumerate() {
            for (i, item) in items.iter().enumerate() {
                if item.index >= TABLES {
                    // The limit check of the plan reports this.
                    continue;
                }
                if !self.written[item.index as usize] {
                    return Some(format!(
                        "segment {segment}: control item {i} names table entry {}, but no control_write filled it; write the vector first",
                        item.index
                    ));
                }
            }
        }
        None
    }

    /// The sum of squares of one layer of one entry.
    pub(crate) fn sum_squares(&self, entry: u32, layer: usize) -> f32 {
        self.sums[entry as usize][layer]
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

    /// The host step after the launch that ends layer `layer`: the control items of the layer, then the reads of the tap.
    ///
    /// # Safety
    ///
    /// `device` names the scratch of the pass, and the launch has ended.
    pub(crate) unsafe fn at_cut(
        &mut self,
        device: &CutDevice,
        table: Option<&ControlTable>,
        layer: usize,
    ) -> Result<(), String> {
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
        // The actions of one segment share its rows, and they come in list order. One read and one write of the
        // rows serve all of them.
        let mut done = 0;
        while done < cut.actions.len() {
            let segment = cut.actions[done].segment;
            let mut end = done;
            while end < cut.actions.len() && cut.actions[end].segment == segment {
                end += 1;
            }
            let Some(table) = table else {
                return Err("the pass has control items, but the engine has no control table".to_owned());
            };
            unsafe { apply_segment(device, table, layer, &cut.actions[done..end]) };
            done = end;
        }
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

/// Apply the actions of one segment at one cut, in list order, with the device op (`hip/src/engine/control.hip`). The
/// rows stay on the device: the host reads and writes none of them. An action whose vector has the sum 0 changes
/// nothing, so it makes no call.
///
/// # Safety
///
/// As for `Cuts::at_cut`.
unsafe fn apply_segment(device: &CutDevice, table: &ControlTable, layer: usize, actions: &[Action]) {
    let rows = actions[0].rows.clone();
    assert!(
        actions.iter().all(|action| action.rows == rows),
        "the actions of one segment share its rows"
    );
    for action in actions {
        let sum_squares = table.sum_squares(action.entry, layer);
        if sum_squares == 0.0 {
            continue;
        }
        unsafe {
            (device.binding.control_apply)(
                device.h.wrapping_add(rows.start * WIDTH),
                rows.len() as u32,
                table.layer(action.entry, layer),
                sum_squares,
                control::op_code(action.op),
                action.strength,
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use nagaya_abi::{ControlItem, ControlOp};

    use super::*;
    use crate::binding::{PassStop, Qwen38Launch};
    use crate::control::{FOLD_BYTES, RESIDUAL_BYTES, apply_row};
    use crate::fake::{self, FAST, Launch, Rig, StageKey};
    use crate::host::LaunchKnobs;
    use crate::stage::{ST_HC_COMBINE_RMS, Stage};

    const SPIN_TIMEOUT: u32 = 2;

    /// A made-up value of the residual after the launch `k`: finite, and different for each row and index.
    fn pattern(k: usize, row: usize, j: usize) -> f32 {
        ((k * 131 + row * 17 + j * 7) % 211) as f32 / 16.0 - 6.0
    }

    /// The made-up vector of an entry: the value at layer `layer`, index `j`.
    fn vector(layer: usize, j: usize) -> f32 {
        ((layer * 29 + j * 13) % 97) as f32 / 32.0 - 1.5
    }

    fn vector_bytes() -> Vec<u8> {
        (0..LAYERS)
            .flat_map(|layer| (0..WIDTH).map(move |j| vector(layer, j)))
            .flat_map(f32::to_le_bytes)
            .collect()
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
        // SAFETY: the fake binding.
        rig.host.control = Some(unsafe { ControlTable::new(&rig.binding) });
        unsafe {
            rig.host
                .control
                .as_mut()
                .unwrap()
                .write(&rig.binding, 0, &vector_bytes())
                .expect("a good vector")
        };
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

    // ---- piece 8: the control table and the host op ------------------------------------------------------------

    fn table(rig: &mut Rig) {
        // SAFETY: the fake binding.
        rig.host.control = Some(unsafe { ControlTable::new(&rig.binding) });
        unsafe {
            rig.host
                .control
                .as_mut()
                .unwrap()
                .write(&rig.binding, 0, &vector_bytes())
                .expect("a good vector")
        };
    }

    fn add(strength: f32, layers: &[(u32, f32)]) -> ControlItem {
        ControlItem {
            index: 0,
            op: ControlOp::Add,
            strength,
            layers: Some(layers.to_vec()),
        }
    }

    #[test]
    fn a_list_on_layers_3_and_20_changes_exactly_the_rows_of_its_segment_at_those_cuts_on_the_device() {
        // Segment 0 has 2 rows and no list. Segment 1 has 3 rows (rows 2, 3 and 4 of the pass) and a list.
        let mut rig = rig(&[2, 3]);
        table(&mut rig);
        let h = rig.h() as usize;
        let seen = fill_h(&mut rig, 5);
        let list = [add(2.0, &[(3, 2.0), (20, 2.0)])];
        let shapes = [shape(2, false, &[]), shape(3, false, &list)];
        let mut cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
        fake::reset();
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a steered pass runs");
        assert_eq!(fake::launches().len(), 3, "2 cuts make 3 launches");
        // The op runs on the device: no copy reads or writes a row of h.
        let region = |from: usize| (from >= h && from < h + 256 * row_bytes()).then(|| (from - h) / row_bytes());
        for copy in fake::copies() {
            for address in [copy.src, copy.dst] {
                assert_eq!(region(address), None, "a copy touches a row of h: {copy:?}");
            }
        }
        // One call of the device op at each of the 2 cuts: the rows 2 to 4 of segment 1, and the vector of the layer.
        let table = rig.host.control.as_ref().unwrap();
        let calls = fake::controls();
        assert_eq!(calls.len(), 2, "one call at each cut");
        for (call, layer) in calls.iter().zip([3, 20]) {
            assert_eq!(call.h, h + 2 * row_bytes(), "the call at layer {layer} starts at row 2");
            assert_eq!(call.rows, 3, "the call at layer {layer} names the 3 rows of segment 1");
            assert_eq!(
                call.v,
                table.layer(0, layer) as usize,
                "the call at layer {layer} names its layer"
            );
            assert_eq!(call.sum_squares, table.sum_squares(0, layer));
            assert_eq!((call.op, call.strength), (control::op_code(ControlOp::Add), 2.0));
        }
        // The bytes written are the op applied to the bytes read. At the start of launch k, `h` holds what the host step
        // left after launch k - 1, and the fake kernel filled `h` with `pattern(k - 1, ..)` in that launch.
        let seen = seen.borrow();
        for (launch, layer) in [(1, 3), (2, 20)] {
            let after = &seen[launch];
            let v: Vec<f32> = (0..WIDTH).map(|j| vector(layer, j)).collect();
            let vv = control::sum_squares(&v);
            let bits = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            for row in 0..5 {
                let kernel: Vec<f32> = (0..WIDTH).map(|j| pattern(launch - 1, row, j)).collect();
                let got = &after[row * WIDTH..(row + 1) * WIDTH];
                if row < 2 {
                    assert_eq!(
                        bits(got),
                        bits(&kernel),
                        "launch {launch}: row {row} of segment 0 is untouched"
                    );
                } else {
                    let mut want = kernel.clone();
                    assert!(apply_row(ControlOp::Add, 2.0, &mut want, &v, vv));
                    assert_eq!(
                        bits(got),
                        bits(&want),
                        "launch {launch}: row {row} is the add of the row that the kernel left"
                    );
                    assert_ne!(bits(got), bits(&kernel), "the add changed the row");
                }
            }
        }
    }

    #[test]
    fn the_list_of_the_first_segment_leaves_the_rows_of_the_second_alone() {
        let mut rig = rig(&[2, 3]);
        table(&mut rig);
        let h = rig.h() as usize;
        let _seen = fill_h(&mut rig, 5);
        let list = [add(1.0, &[(3, 1.0)])];
        let shapes = [shape(2, false, &list), shape(3, false, &[])];
        let mut cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
        fake::reset();
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a steered pass runs");
        for copy in fake::copies() {
            for address in [copy.src, copy.dst] {
                if address >= h && address < h + 256 * row_bytes() {
                    let row = (address - h) / row_bytes();
                    assert!(row < 2, "a copy touches row {row} of the second segment: {copy:?}");
                }
            }
        }
        assert_eq!(fake::launches().len(), 2);
    }

    #[test]
    fn an_add_or_an_ablate_of_strength_0_makes_one_launch_and_no_read() {
        for op in [ControlOp::Add, ControlOp::Ablate] {
            let mut rig = rig(&[2]);
            table(&mut rig);
            let h = rig.h() as usize;
            let list = [ControlItem {
                index: 0,
                op,
                strength: 0.0,
                layers: None,
            }];
            let shapes = [shape(2, false, &list)];
            let mut cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
            assert!(
                !cuts.fuse_hc_off(),
                "{op:?} of 0 acts on no layer, so the pass keeps the fused combine"
            );
            fake::reset();
            let (_, lists) = record_stages(&mut rig, 2);
            rig.run(FAST, fake::no_knobs(), Some(&mut cuts)).expect("a pass runs");
            assert_eq!(fake::launches().len(), 1, "{op:?}");
            assert!(reads_of(h, 2 * row_bytes()).is_empty(), "{op:?}: no read of h");
            assert!(
                lists.borrow()[0].iter().any(|key| key.0 == ST_HC_COMBINE_RMS),
                "{op:?}: the list is the list of a plain pass"
            );
        }
        // A clamp of 0 removes the concept, so it acts on each layer.
        let list = [ControlItem {
            index: 0,
            op: ControlOp::Clamp,
            strength: 0.0,
            layers: None,
        }];
        let shapes = [shape(2, false, &list)];
        let cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
        assert_eq!(cuts.layers().len(), 48);
        assert!(cuts.fuse_hc_off());
    }

    #[test]
    fn fuse_hc_is_off_for_a_pass_with_an_action_and_on_for_a_pass_with_a_tap_or_none() {
        let on = plain_keys(&[2]);
        assert!(on.iter().any(|key| key.0 == ST_HC_COMBINE_RMS));
        // A tap alone keeps the list of the plain pass (checked in full above); a list on one layer builds without it.
        let mut rig = rig(&[2]);
        table(&mut rig);
        let (_, lists) = record_stages(&mut rig, 2);
        let list = [add(1.0, &[(10, 1.0)])];
        let shapes = [shape(2, false, &list)];
        let mut cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a steered pass runs");
        let joined: Vec<StageKey> = lists.borrow().iter().flatten().copied().collect();
        assert!(joined.iter().all(|key| key.0 != ST_HC_COMBINE_RMS));
        assert!(joined.len() > on.len(), "the unfused list has more stages");
    }

    #[test]
    fn a_taps_and_a_list_in_one_pass_read_the_row_after_the_items_of_the_layer() {
        let mut rig = rig(&[2]);
        table(&mut rig);
        let _seen = fill_h(&mut rig, 2);
        let list = [add(3.0, &[(5, 3.0)])];
        let shapes = [shape(2, true, &list)];
        let mut cuts = Cuts::new(&shapes, RESIDUAL).expect("a list and a tap");
        rig.run(FAST, fake::no_knobs(), Some(&mut cuts))
            .expect("a steered, tapped pass runs");
        assert_eq!(fake::launches().len(), 49, "the tap cuts at each layer");
        let outputs = cuts.into_outputs().expect("the outputs");
        let bytes = &outputs[0].1;
        let read = |layer: usize| -> Vec<f32> {
            bytes[layer * WIDTH * 4..(layer + 1) * WIDTH * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        let v: Vec<f32> = (0..WIDTH).map(|j| vector(5, j)).collect();
        let vv = control::sum_squares(&v);
        let mut want: Vec<f32> = (0..WIDTH).map(|j| pattern(5, 1, j)).collect();
        apply_row(ControlOp::Add, 3.0, &mut want, &v, vv);
        assert_eq!(
            read(5).iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "layer 5 is read after its item"
        );
        // The layers with no item are as the kernel left them.
        assert_eq!(read(6)[10].to_bits(), pattern(6, 1, 10).to_bits());
    }

    #[test]
    fn the_table_is_one_allocation_of_the_declared_bytes_and_a_write_allocates_nothing() {
        fake::reset();
        let rig = Rig::new(&[1]);
        let before = fake::mallocs();
        // SAFETY: the fake binding.
        let mut table = unsafe { ControlTable::new(&rig.binding) };
        assert_eq!(fake::mallocs(), before + 1, "the table is one allocation");
        assert_eq!(
            fake::malloc_sizes().last().copied(),
            Some(15_728_640),
            "its size is the declared bytes of the term"
        );
        assert_eq!(LIMIT.table_bytes(), 15_728_640);
        let bytes = vector_bytes();
        // SAFETY: the fake binding.
        unsafe { table.write(&rig.binding, 3, &bytes).expect("a good vector") };
        unsafe { table.write(&rig.binding, 3, &bytes).expect("a second write") };
        assert_eq!(fake::mallocs(), before + 1, "a write makes no allocation");
        // The write is one copy of the entry's bytes to the entry's address.
        let writes: Vec<_> = fake::copies()
            .into_iter()
            .filter(|c| c.kind == 1 && c.bytes == bytes.len() as u64)
            .collect();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].dst, table.entry(3) as usize);
        assert_eq!(writes[0].dst - table.entry(0) as usize, 3 * LAYERS * WIDTH * 4);
        // The table was zero at open: an entry that was never written reads as zeros.
        // SAFETY: the table holds 8 entries of 48 layers.
        let zero = unsafe { std::slice::from_raw_parts(table.entry(5), LAYERS * WIDTH) };
        assert!(zero.iter().all(|x| x.to_bits() == 0));
    }

    #[test]
    fn a_write_refuses_a_bad_index_a_bad_size_and_a_value_that_is_not_finite_and_changes_nothing() {
        let rig = Rig::new(&[1]);
        fake::reset();
        // SAFETY: the fake binding.
        let mut table = unsafe { ControlTable::new(&rig.binding) };
        let good = vector_bytes();
        let mut nan = good.clone();
        nan[4 * 12_345..4 * 12_345 + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut infinite = good.clone();
        infinite[0..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        let copies = fake::copies().len();
        for (index, bytes, why) in [
            (8, good.clone(), "entries"),
            (u32::MAX, good.clone(), "entries"),
            (0, good[..good.len() - 4].to_vec(), "send the whole vector"),
            (0, Vec::new(), "send the whole vector"),
            (0, nan, "value 12345"),
            (0, infinite, "value 0"),
        ] {
            let error = unsafe { table.write(&rig.binding, index, &bytes) }.expect_err("a bad write");
            assert!(error.contains(why), "{error}");
        }
        assert_eq!(fake::copies().len(), copies, "a refused write copies nothing");
        assert!(!table.written.iter().any(|w| *w), "a refused write marks no entry");
        assert!(
            table.sums.iter().all(|sums| sums.iter().all(|s| *s == 0.0)),
            "and keeps no sum"
        );
    }

    #[test]
    fn a_list_that_names_an_entry_that_was_never_written_is_refused() {
        let rig = Rig::new(&[1]);
        // SAFETY: the fake binding.
        let mut table = unsafe { ControlTable::new(&rig.binding) };
        let item = |index| ControlItem {
            index,
            op: ControlOp::Add,
            strength: 1.0,
            layers: None,
        };
        let list = [item(0), item(2)];
        let lists: [&[ControlItem]; 2] = [&[], &list];
        let problem = table.unwritten(&lists).expect("no entry was written");
        assert!(
            problem.contains("segment 1: control item 0 names table entry 0"),
            "{problem}"
        );
        assert!(problem.contains("no control_write filled it"), "{problem}");
        unsafe { table.write(&rig.binding, 0, &vector_bytes()).expect("write 0") };
        let problem = table.unwritten(&lists).expect("entry 2 is still empty");
        assert!(problem.contains("control item 1 names table entry 2"), "{problem}");
        unsafe { table.write(&rig.binding, 2, &vector_bytes()).expect("write 2") };
        assert_eq!(table.unwritten(&lists), None);
    }

    #[test]
    fn a_cut_after_a_layer_that_the_plan_does_not_name_is_an_error_and_so_is_a_missing_table() {
        let mut rig = rig(&[2]);
        let list = [add(1.0, &[(3, 1.0)])];
        let shapes = [shape(2, false, &list)];
        let mut cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
        // No table: the step refuses to act on rows it cannot compute.
        let (h, mixed) = (rig.h(), rig.mixed());
        let device = CutDevice {
            binding: &rig.binding,
            h,
            mixed,
        };
        let error = unsafe { cuts.at_cut(&device, None, 3) }.expect_err("no table");
        assert!(error.contains("no control table"), "{error}");
        let mut cuts = Cuts::new(&shapes, TapSet::default()).expect("a list");
        let error = unsafe { cuts.at_cut(&device, None, 4) }.expect_err("layer 4 is not a cut");
        assert!(error.contains("cuts after layer 3"), "{error}");
        let error = unsafe { cuts.finish(&device) }.expect_err("no cut ran");
        assert!(error.contains("ran 0 of its 1 cuts"), "{error}");
    }
}
