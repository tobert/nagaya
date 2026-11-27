//! The plan of the runs of a pass and the byte layout of its buffers. Nothing here needs a GPU.
//!
//! A run is one layer of the model for one block of rows. The runs go in the order of `run_layers` of the host form:
//! layers 0 to 5, then each cycle of layers 6 to 17, then layers 18 to 23. A buffer is named so that no copy is
//! needed between runs: the output of a run is the input of the next run, or a plane that a later run reads.

use nagaya_mk_cuda::attn::MAX_ROWS;
use nagaya_mk_cuda::experts::{
    ExpertFormat, QuantRowsLayout, route_pair_list_bytes, route_picks_bytes, route_scratch_words,
};
use nagaya_mk_cuda::proj::ProjLayout;
use nagaya_mk_cuda::repack::MatmulShape;

use crate::spec::{
    CYCLE_END, CYCLE_LAYERS, CYCLE_START, HIDDEN, INTERMEDIATE, LAYERS, MAX_EXTRA_RUNS, attn_scratch_bytes, is_full,
};

/// The most cycles of a pass: the reserve of states holds 4 extra cycles.
pub const MAX_CYCLES: usize = 1 + MAX_EXTRA_RUNS / CYCLE_LAYERS;

/// The width of a row of the logits: the vocabulary, padded to the tiles of the head.
pub const LOGIT_COLUMNS: usize = 200_064;

/// The alignment of each buffer and each region of scratch, in bytes.
pub const ALIGN: usize = 256;

/// A buffer of the activations of a step. `B2` has two copies, because a cycle reads the block output of the cycle
/// before while it writes its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Buf {
    /// The output of the embedding, and the first block output.
    B0,
    /// The row after layer 5: the second block output and the start of each cycle.
    B1,
    /// The row after layer 11 of a cycle, with the parity of the cycle.
    B2(u8),
    /// The row after layer 17 of the last cycle: the fourth block output.
    B3,
    /// The row after layer 17 of a cycle that is not the last: the message `y` of the next cycle.
    Y,
    /// The working row of the runs.
    Xw,
    /// The row after the attention of a run: `r`.
    R,
    /// The outputs of the attention mix: 3 rows for each position.
    A,
    /// The outputs of the message mix: 4 rows for each position.
    M,
    /// The outputs of the message gates 0 to 2.
    U,
    /// The outputs of the down matmul of the 4 message gates.
    G,
    /// The output of the joined projection: q, rel, sink and gate.
    ProjOut,
    /// The output of the attention.
    AttnOut,
    /// The output of `OP_OUT_MIX` (`outmix::out_floats`): `e`, then `h_e`, then the row factor of `e`.
    EOut,
    /// The 8 picks and weights of each row.
    Picks,
    /// The list of the pairs of an expert op.
    Pairs,
    /// The quantized `h_e` rows.
    Xq,
    /// The activations of the up matmul of the experts.
    Act,
    /// The quantized activations.
    Actq,
    /// The 8 slots of each row, before the sum.
    Slots,
    /// The input of the head: the output residual of the last run.
    F,
    /// The logits.
    Logits,
}

/// The kinds of buffer that [`StepLayout`] places, in the order of their offsets.
pub const BUFFERS: [Buf; 23] = [
    Buf::B0,
    Buf::B1,
    Buf::B2(0),
    Buf::B2(1),
    Buf::B3,
    Buf::Y,
    Buf::Xw,
    Buf::R,
    Buf::A,
    Buf::M,
    Buf::U,
    Buf::G,
    Buf::ProjOut,
    Buf::AttnOut,
    Buf::EOut,
    Buf::Picks,
    Buf::Pairs,
    Buf::Xq,
    Buf::Act,
    Buf::Actq,
    Buf::Slots,
    Buf::F,
    Buf::Logits,
];

impl Buf {
    /// The buffers that hold a row of the stream and its row factor.
    pub fn has_row_factor(self) -> bool {
        matches!(
            self,
            Buf::B0 | Buf::B1 | Buf::B2(_) | Buf::B3 | Buf::Y | Buf::Xw | Buf::R | Buf::A | Buf::M | Buf::U | Buf::F
        )
    }

    /// The rows of the stream that one position holds in the buffer (1 for a plane, 3 for `A`, 4 for `M`).
    fn rows_per_position(self) -> usize {
        match self {
            Buf::A | Buf::U => 3,
            Buf::M => 4,
            _ => 1,
        }
    }
}

/// The message queries of the cycle layer `l` (0 to 11): query, key, value, expert input. Row `2l` of `queries` is the
/// query and row `2l + 1` the expert input. Row `2l` of `kv_queries` is the key and row `2l + 1` the value.
pub fn message_queries_of(queries: &[f32], kv_queries: &[f32], l: usize) -> Vec<f32> {
    let row = |t: &[f32], r: usize| t[r * HIDDEN..(r + 1) * HIDDEN].to_vec();
    [
        row(queries, 2 * l),
        row(kv_queries, 2 * l),
        row(kv_queries, 2 * l + 1),
        row(queries, 2 * l + 1),
    ]
    .concat()
}

/// The task of a run, in the order that the builder lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskKind {
    Mix,
    GateDown3,
    GateUp3,
    Proj,
    Attn,
    /// The block-form matmul of `wo` before `OutMix`, in a block of `outmix::SPLIT_MIN_ROWS` rows or more only (`run.rs`).
    OutMatmul,
    OutMix,
    GateDownE,
    GateUpE,
    Quant,
    Route,
    Up,
    Down,
}

/// One run of the pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunPlan {
    /// The index of the run in the pass.
    pub run: usize,
    pub layer: usize,
    /// The state (the slot entry) of the run: the layer, or an entry of the reserve for a cycle after the first.
    pub state: usize,
    /// The cycle of the run, from 0. The runs of layers 0 to 5 are in cycle 0, and the runs of layers 18 to 23 are in
    /// the last cycle.
    pub cycle: usize,
    pub full: bool,
    /// The layer starts a block: the residual of its attention restarts from the output of the attention.
    pub first_of_block: bool,
    /// The run reads the messages of the cycle before.
    pub messages: bool,
    /// The rows of the stack of the attention mix: the block outputs and the input row. Empty for layer 0.
    pub stack_attn: Vec<Buf>,
    /// The rows of the stack of the mlp mix: the block outputs, and `R`.
    pub stack_mlp: Vec<Buf>,
    pub x_in: Buf,
    pub x_out: Buf,
    /// The block output of the cycle before, for a run with messages.
    pub p: Option<Buf>,
    /// The row after layer 17 of the cycle before, for a run with messages.
    pub y: Option<Buf>,
}

impl RunPlan {
    /// The depth of the stack of the attention mix.
    pub fn depth_attn(&self) -> usize {
        self.stack_attn.len()
    }

    /// The depth of the stack of the mlp mix.
    pub fn depth_mlp(&self) -> usize {
        self.stack_mlp.len()
    }

    /// The tasks of the run, in order. Layer 0 has no attention mix, and a run with messages has the gates.
    pub fn task_kinds(&self) -> Vec<TaskKind> {
        use TaskKind::*;
        let mut kinds = Vec::new();
        if self.layer > 0 {
            kinds.push(Mix);
        }
        if self.messages {
            kinds.extend([GateDown3, GateUp3]);
        }
        kinds.extend([Proj, Attn, OutMix]);
        if self.messages {
            kinds.extend([GateDownE, GateUpE]);
        }
        kinds.extend([Quant, Route, Up, Down]);
        kinds
    }

    /// The tasks of the run for a block of `rows` rows: `task_kinds` with the matmul task of `OP_OUT_MIX` before each
    /// `OutMix` when the block splits it (`outmix::SPLIT_MIN_ROWS`). The order is what makes the matmul run first: the
    /// graph orders a task after the tasks before it, so `run_tasks` must build exactly this list.
    pub fn task_kinds_for(&self, rows: usize) -> Vec<TaskKind> {
        let mut kinds = Vec::new();
        for kind in self.task_kinds() {
            if kind == TaskKind::OutMix && rows >= nagaya_mk_cuda::outmix::SPLIT_MIN_ROWS {
                kinds.push(TaskKind::OutMatmul);
            }
            kinds.push(kind);
        }
        kinds
    }

    /// The buffers that the run reads, apart from the scratch of its ops.
    pub fn reads(&self) -> Vec<Buf> {
        // The stack of the attention mix ends with the input row. Layer 0 has no mix, and it reads the row alone.
        let mut out = self.stack_attn.clone();
        if out.is_empty() {
            out.push(self.x_in);
        }
        out.extend(self.p);
        out.extend(self.y);
        out
    }
}

/// The block outputs that exist before the run of `layer` in cycle `cycle` of a pass of `cycles` cycles. The run of a
/// layer that starts a block adds its input row to them after it.
fn blocks_before(layer: usize, cycle: usize) -> Vec<Buf> {
    let mut out = Vec::new();
    if layer >= 1 {
        out.push(Buf::B0);
    }
    if layer > CYCLE_START {
        out.push(Buf::B1);
    }
    if layer > CYCLE_START + CYCLE_LAYERS / 2 {
        out.push(Buf::B2((cycle % 2) as u8));
    }
    if layer > CYCLE_END {
        out.push(Buf::B3);
    }
    out
}

/// The runs of a pass of `cycles` cycles, in the order of `run_layers` of the host form.
///
/// # Panics
/// `cycles` is not from 1 to [`MAX_CYCLES`].
pub fn runs(cycles: usize) -> Vec<RunPlan> {
    assert!(
        (1..=MAX_CYCLES).contains(&cycles),
        "a pass has from 1 to {MAX_CYCLES} cycles, not {cycles}. Ask for a count in that range"
    );
    let last = cycles - 1;
    let mut order: Vec<(usize, usize)> = (0..CYCLE_START).map(|l| (l, 0)).collect();
    for c in 0..cycles {
        order.extend((CYCLE_START..CYCLE_END).map(|l| (l, c)));
    }
    order.extend((CYCLE_END..LAYERS).map(|l| (l, last)));
    order
        .into_iter()
        .enumerate()
        .map(|(run, (layer, cycle))| {
            let boundary = layer % 6 == 0;
            let parity = (cycle % 2) as u8;
            let x_in = match layer {
                0 => Buf::B0,
                CYCLE_START => Buf::B1,
                12 => Buf::B2(parity),
                CYCLE_END => Buf::B3,
                _ => Buf::Xw,
            };
            let x_out = match layer {
                5 => Buf::B1,
                11 => Buf::B2(parity),
                17 if cycle == last => Buf::B3,
                17 => Buf::Y,
                _ => Buf::Xw,
            };
            let mut stack_attn = if layer == 0 {
                Vec::new()
            } else {
                blocks_before(layer, cycle)
            };
            if layer > 0 {
                stack_attn.push(x_in);
            }
            let mut stack_mlp = blocks_before(layer, cycle);
            if boundary {
                stack_mlp.push(x_in);
            }
            stack_mlp.push(Buf::R);
            let messages = cycle >= 1 && (CYCLE_START..CYCLE_END).contains(&layer);
            RunPlan {
                run,
                layer,
                state: if cycle >= 1 && (CYCLE_START..CYCLE_END).contains(&layer) {
                    LAYERS + CYCLE_LAYERS * (cycle - 1) + (layer - CYCLE_START)
                } else {
                    layer
                },
                cycle,
                full: is_full(layer),
                first_of_block: boundary,
                messages,
                stack_attn,
                stack_mlp,
                x_in,
                x_out,
                p: messages.then(|| Buf::B2(((cycle - 1) % 2) as u8)),
                y: messages.then_some(Buf::Y),
            }
        })
        .collect()
}

/// The count of tasks of a pass: the embedding, the runs, the output residual and the head.
pub fn task_count(cycles: usize) -> usize {
    2 + 1 + runs(cycles).iter().map(|r| r.task_kinds().len()).sum::<usize>()
}

fn align(bytes: usize) -> usize {
    bytes.next_multiple_of(ALIGN)
}

/// The place of a buffer in the arena of a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    pub offset: usize,
    pub bytes: usize,
    /// The offset of the row factors, for a buffer that has them.
    pub rf_offset: Option<usize>,
}

/// The byte offsets of the buffers of a step of `rows` rows, `logit_rows` of which give logits. Each buffer and each
/// row-factor array starts at a multiple of [`ALIGN`]. The offsets depend on `rows`, because a buffer holds `rows`
/// rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepLayout {
    pub rows: usize,
    pub logit_rows: usize,
    places: Vec<(Buf, Place)>,
    total: usize,
}

impl StepLayout {
    /// The layout of a step.
    ///
    /// # Panics
    /// `rows` is not from 1 to 512, or `logit_rows` is more than `rows`.
    pub fn new(rows: usize, logit_rows: usize) -> Self {
        assert!(
            (1..=MAX_ROWS).contains(&rows),
            "a step has from 1 to {MAX_ROWS} rows, not {rows}. Split the block"
        );
        assert!(
            logit_rows <= rows,
            "a step of {rows} rows cannot give logits for {logit_rows} rows. Ask for at most {rows}"
        );
        let mut at = 0usize;
        let mut places = Vec::with_capacity(BUFFERS.len());
        for buf in BUFFERS {
            let bytes = Self::bytes_of(buf, rows, logit_rows);
            let offset = at;
            at = align(at + bytes);
            let rf_offset = buf.has_row_factor().then(|| {
                let rf = at;
                at = align(at + buf.rows_per_position() * rows * 4);
                rf
            });
            places.push((
                buf,
                Place {
                    offset,
                    bytes,
                    rf_offset,
                },
            ));
        }
        Self {
            rows,
            logit_rows,
            places,
            total: at,
        }
    }

    fn bytes_of(buf: Buf, rows: usize, logit_rows: usize) -> usize {
        let row = HIDDEN * 4;
        match buf {
            Buf::B0 | Buf::B1 | Buf::B2(_) | Buf::B3 | Buf::Y | Buf::Xw | Buf::R | Buf::F => rows * row,
            Buf::A | Buf::U => 3 * rows * row,
            Buf::M => 4 * rows * row,
            // The gates 0 to 2 and the gate 3 of the mlp mix: 64 values for each row.
            Buf::G => 4 * rows * 64 * 4,
            Buf::ProjOut => ProjLayout::output_floats(rows) * 4,
            Buf::AttnOut => rows * 3072 * 4,
            // `e`, then `h_e`, each of `rows` rows. The row factor of `e` follows them.
            Buf::EOut => nagaya_mk_cuda::outmix::out_floats(rows) * 4,
            Buf::Picks => route_picks_bytes(rows),
            Buf::Pairs => route_pair_list_bytes(rows),
            // The larger of the two formats, so that one place serves every layer.
            Buf::Xq => Self::quant_bytes(rows, HIDDEN),
            Buf::Act => 8 * rows * INTERMEDIATE * 4,
            Buf::Actq => Self::quant_bytes(8 * rows, INTERMEDIATE),
            Buf::Slots => 8 * rows * row,
            Buf::Logits => logit_rows * LOGIT_COLUMNS * 4,
        }
    }

    fn quant_bytes(rows: usize, k: usize) -> usize {
        [ExpertFormat::Nvfp4, ExpertFormat::Mxfp8]
            .into_iter()
            .map(|f| {
                QuantRowsLayout::new(f, rows, k)
                    .expect("a quantized buffer of valid rows")
                    .bytes()
            })
            .max()
            .expect("two formats")
    }

    /// The byte offsets in `EOut` of `e`, `h_e` and the row factors of `e` (`outmix::out_offsets`).
    pub fn eout_parts(&self) -> (usize, usize, usize) {
        let (e, h, rf) = nagaya_mk_cuda::outmix::out_offsets(self.rows);
        (e * 4, h * 4, rf * 4)
    }

    /// The place of `buf`.
    pub fn place(&self, buf: Buf) -> Place {
        self.places
            .iter()
            .find(|(b, _)| *b == buf)
            .unwrap_or_else(|| panic!("the layout has no buffer {buf:?}"))
            .1
    }

    /// The offset of `buf` in the arena.
    pub fn offset(&self, buf: Buf) -> usize {
        self.place(buf).offset
    }

    /// The offset of the row factors of `buf`.
    ///
    /// # Panics
    /// The buffer has no row factors.
    pub fn rf(&self, buf: Buf) -> usize {
        self.place(buf)
            .rf_offset
            .unwrap_or_else(|| panic!("the buffer {buf:?} has no row factors. Ask for the row factors of a row buffer"))
    }

    /// The bytes of the arena.
    pub fn total_bytes(&self) -> usize {
        self.total
    }

    /// Every place, with its buffer.
    pub fn places(&self) -> &[(Buf, Place)] {
        &self.places
    }
}

/// The ops that own a region of the scratch of the ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScratchKind {
    Proj,
    Attn,
    OutMix,
    GateDown3,
    GateUp3,
    GateDownE,
    GateUpE,
    Route,
    Up,
    Down,
    Head,
}

/// The regions of the scratch of the ring. Each op has its own region, so that no two tasks of a run share a merge
/// counter. The counters of a region are its first words, and their place does not depend on the rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    regions: Vec<(ScratchKind, usize, usize)>,
    total: usize,
}

impl ScratchLayout {
    /// The regions for blocks of up to `max_rows` rows in slots of up to `max_seq` positions.
    pub fn new(max_rows: usize, max_seq: usize) -> Self {
        assert!(
            (1..=MAX_ROWS).contains(&max_rows),
            "a block has from 1 to {MAX_ROWS} rows, not {max_rows}"
        );
        let proj = ProjLayout { full: true }.scratch_floats(max_rows);
        let head = MatmulShape::new(LOGIT_COLUMNS, HIDDEN)
            .expect("the shape of the head")
            .scratch_floats(max_rows.min(crate::spec::MAX_LOGIT_ROWS));
        let sizes = [
            (ScratchKind::Proj, proj),
            (
                ScratchKind::Attn,
                usize::try_from(attn_scratch_bytes(max_seq, max_rows) / 4).expect("fits"),
            ),
            (ScratchKind::OutMix, nagaya_mk_cuda::outmix::scratch_floats(max_rows)),
            (ScratchKind::GateDown3, 16 + 3 * 24 * max_rows * 16),
            (ScratchKind::GateUp3, 4 + 3 * 36 * max_rows),
            (ScratchKind::GateDownE, 16 + 24 * max_rows * 16),
            (ScratchKind::GateUpE, 4 + 36 * max_rows),
            (ScratchKind::Route, route_scratch_words(max_rows)),
            (ScratchKind::Up, nagaya_mk_cuda::protocol::expert_pair_bound(max_rows)),
            // The counters of the column tiles and of the row finisher, then the sums of squares of the row factors.
            (
                ScratchKind::Down,
                nagaya_mk_cuda::experts::down_scratch_words(HIDDEN, max_rows),
            ),
            (ScratchKind::Head, head),
        ];
        let mut at = 0usize;
        let mut regions = Vec::with_capacity(sizes.len());
        for (kind, floats) in sizes {
            regions.push((kind, at, floats * 4));
            at = align(at + floats * 4);
        }
        Self { regions, total: at }
    }

    /// The byte offset of the region of `kind` in the scratch.
    pub fn offset(&self, kind: ScratchKind) -> usize {
        self.region(kind).0
    }

    /// The bytes of the region of `kind`.
    pub fn bytes(&self, kind: ScratchKind) -> usize {
        self.region(kind).1
    }

    fn region(&self, kind: ScratchKind) -> (usize, usize) {
        let (_, offset, bytes) = *self
            .regions
            .iter()
            .find(|(k, _, _)| *k == kind)
            .unwrap_or_else(|| panic!("the layout has no region {kind:?}"));
        (offset, bytes)
    }

    /// The floats of the whole scratch.
    pub fn total_floats(&self) -> usize {
        self.total / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Who wrote a buffer, as a run index. `None` is the embedding.
    type Writer = Option<usize>;

    /// The writer that the host form needs for each read of run `r` at layer `l` of cycle `c`, found from the layers
    /// alone, with no use of the plan under test.
    fn wanted(cycles: usize, run: usize) -> Vec<(Buf, Writer)> {
        let last = cycles - 1;
        let cycle_run = |c: usize, layer: usize| 6 + CYCLE_LAYERS * c + (layer - CYCLE_START);
        let plan = &runs(cycles)[run];
        let (l, c) = (plan.layer, plan.cycle);
        let mut out = Vec::new();
        let b2 = |c: usize| (Buf::B2((c % 2) as u8), Some(cycle_run(c, 11)));
        // The block outputs before the run.
        if l >= 1 {
            out.push((Buf::B0, None));
        }
        if l > 6 {
            out.push((Buf::B1, Some(5)));
        }
        if l > 12 {
            out.push(b2(c));
        }
        if l > 18 {
            out.push((Buf::B3, Some(cycle_run(last, 17))));
        }
        // The input row: the row after the layer before, but the start of a cycle is the row after layer 5.
        let x_writer = match l {
            0 => None,
            6 => Some(5),
            18 => Some(cycle_run(last, 17)),
            _ => Some(run - 1),
        };
        let x_buf = match l {
            0 => Buf::B0,
            6 => Buf::B1,
            12 => b2(c).0,
            18 => Buf::B3,
            _ => Buf::Xw,
        };
        out.push((x_buf, x_writer));
        if plan.messages {
            out.push(b2(c - 1));
            out.push((Buf::Y, Some(cycle_run(c - 1, 17))));
        }
        out
    }

    #[test]
    fn the_message_queries_have_the_order_of_the_host_form() {
        use nagaya_mk_inkling2_bmg::host::weights::merge_message_queries;
        let queries: Vec<f32> = (0..24 * HIDDEN).map(|i| i as f32).collect();
        let kv: Vec<f32> = (0..24 * HIDDEN).map(|i| -(i as f32) - 1.0).collect();
        let host = merge_message_queries(&queries, &kv);
        assert_eq!(host.len(), 12);
        for (l, four) in host.iter().enumerate() {
            assert_eq!(message_queries_of(&queries, &kv, l), four.concat(), "layer {l}");
        }
    }

    #[test]
    fn the_runs_go_in_the_order_of_the_host_form() {
        let layers: Vec<usize> = runs(3).iter().map(|r| r.layer).collect();
        let mut want: Vec<usize> = (0..6).collect();
        for _ in 0..3 {
            want.extend(6..18);
        }
        want.extend(18..24);
        assert_eq!(layers, want);
        assert_eq!(runs(1).len(), 24);
        assert_eq!(runs(5).len(), 24 + 48);
        assert!(runs(3).iter().enumerate().all(|(i, r)| r.run == i));
    }

    #[test]
    fn the_states_are_the_layers_then_the_reserve_in_order() {
        let plans = runs(3);
        for r in &plans {
            if r.cycle == 0 || r.layer >= CYCLE_END || r.layer < CYCLE_START {
                assert_eq!(r.state, r.layer);
            } else {
                assert_eq!(crate::spec::state_layer(r.state), r.layer, "run {}", r.run);
            }
        }
        let reserve: Vec<usize> = plans.iter().filter(|r| r.state >= LAYERS).map(|r| r.state).collect();
        assert_eq!(reserve, (24..48).collect::<Vec<_>>());
    }

    #[test]
    fn the_task_counts_are_those_of_the_design() {
        let plans = runs(1);
        assert_eq!(plans[0].task_kinds().len(), 7, "layer 0");
        assert!(plans[1..].iter().all(|r| r.task_kinds().len() == 8));
        assert_eq!(task_count(1), 194);
        assert_eq!(task_count(2), 338);
        assert_eq!(task_count(3), 482);
        assert_eq!(task_count(5), 770);
        let message_run = runs(2).into_iter().find(|r| r.messages).unwrap();
        assert_eq!(message_run.task_kinds().len(), 12);
    }

    #[test]
    fn a_block_that_splits_out_mix_lists_its_matmul_task_just_before_it() {
        use nagaya_mk_cuda::outmix::SPLIT_MIN_ROWS;
        for plan in runs(2) {
            let small = plan.task_kinds_for(SPLIT_MIN_ROWS - 1);
            assert_eq!(small, plan.task_kinds());
            let big = plan.task_kinds_for(SPLIT_MIN_ROWS);
            assert_eq!(big.len(), small.len() + 1);
            let at = big.iter().position(|k| *k == TaskKind::OutMix).unwrap();
            assert_eq!(
                big[at - 1],
                TaskKind::OutMatmul,
                "the matmul task is the one before OUT_MIX"
            );
            assert_eq!(big.iter().filter(|k| **k == TaskKind::OutMatmul).count(), 1);
        }
    }

    #[test]
    fn the_depths_of_the_stacks_follow_the_blocks() {
        let plans = runs(3);
        let attn: Vec<usize> = plans[..24].iter().map(|r| r.depth_attn()).collect();
        let mlp: Vec<usize> = plans[..24].iter().map(|r| r.depth_mlp()).collect();
        // Rows 0 to 23 are layers 0 to 5, then the cycle 0 (6 to 17), then 18 to 23.
        let layers: Vec<usize> = plans[..24].iter().map(|r| r.layer).collect();
        assert_eq!(layers[..6], [0, 1, 2, 3, 4, 5]);
        assert_eq!(attn[..6], [0, 2, 2, 2, 2, 2]);
        assert_eq!(attn[6..12], [2, 3, 3, 3, 3, 3]);
        assert_eq!(attn[12..18], [3, 4, 4, 4, 4, 4]);
        assert_eq!(mlp[..6], [2, 2, 2, 2, 2, 2]);
        assert_eq!(mlp[6..12], [3, 3, 3, 3, 3, 3]);
        assert_eq!(mlp[12..18], [4, 4, 4, 4, 4, 4]);
        let tail = &plans[plans.len() - 6..];
        assert_eq!(
            tail.iter().map(|r| r.depth_attn()).collect::<Vec<_>>(),
            [4, 5, 5, 5, 5, 5]
        );
        assert_eq!(tail.iter().map(|r| r.depth_mlp()).collect::<Vec<_>>(), [5; 6]);
    }

    #[test]
    fn every_read_finds_the_row_that_the_right_run_wrote() {
        for cycles in 1..=MAX_CYCLES {
            let plans = runs(cycles);
            // The version of each buffer: the writer of its latest write at the start of each run.
            let mut version: Vec<(Buf, Writer)> = Vec::new();
            let set = |version: &mut Vec<(Buf, Writer)>, buf: Buf, w: Writer| {
                version.retain(|(b, _)| *b != buf);
                version.push((buf, w));
            };
            set(&mut version, Buf::B0, None);
            for plan in &plans {
                let want = wanted(cycles, plan.run);
                let got: Vec<Buf> = plan.reads();
                // The reads of the plan are the stack, the input row, and the messages, in that order.
                let mut expect: Vec<Buf> = want.iter().map(|(b, _)| *b).collect();
                let mut sorted_got = got.clone();
                sorted_got.sort_by_key(|b| format!("{b:?}"));
                expect.sort_by_key(|b| format!("{b:?}"));
                assert_eq!(
                    sorted_got, expect,
                    "cycles {cycles} run {}: the buffers that it reads",
                    plan.run
                );
                for (buf, writer) in &want {
                    let seen = version.iter().find(|(b, _)| b == buf).map(|(_, w)| *w);
                    assert_eq!(
                        seen,
                        Some(*writer),
                        "cycles {cycles} run {} (layer {}, cycle {}) reads {buf:?}",
                        plan.run,
                        plan.layer,
                        plan.cycle
                    );
                }
                set(&mut version, plan.x_out, Some(plan.run));
            }
        }
    }

    #[test]
    fn a_boundary_run_keeps_its_input_as_a_block_output() {
        let plans = runs(3);
        for r in plans.iter().filter(|r| r.first_of_block) {
            assert!(r.layer % 6 == 0);
            assert_eq!(r.stack_mlp[r.stack_mlp.len() - 2], r.x_in, "run {}", r.run);
            assert_ne!(r.x_out, r.x_in, "run {} writes the working row, not its plane", r.run);
        }
        // Layer 0, then layers 6 and 12 in each of 3 cycles, then layer 18.
        assert_eq!(plans.iter().filter(|r| r.first_of_block).count(), 1 + 2 * 3 + 1);
    }

    #[test]
    #[should_panic(expected = "from 1 to 5 cycles, not 6")]
    fn six_cycles_is_a_panic_that_says_the_range() {
        runs(6);
    }

    #[test]
    fn the_step_layout_has_no_overlap_and_aligned_offsets() {
        for (rows, logits) in [(1, 1), (7, 1), (16, 16), (512, 1)] {
            let layout = StepLayout::new(rows, logits);
            let mut spans: Vec<(usize, usize, String)> = Vec::new();
            for (buf, place) in layout.places() {
                assert_eq!(place.offset % ALIGN, 0, "{buf:?}");
                spans.push((place.offset, place.bytes, format!("{buf:?}")));
                if let Some(rf) = place.rf_offset {
                    assert_eq!(rf % ALIGN, 0);
                    spans.push((rf, buf.rows_per_position() * rows * 4, format!("{buf:?} rf")));
                }
            }
            spans.sort();
            for pair in spans.windows(2) {
                assert!(
                    pair[0].0 + pair[0].1 <= pair[1].0,
                    "rows {rows}: {} overlaps {}",
                    pair[0].2,
                    pair[1].2
                );
            }
            let end = spans.last().map(|s| s.0 + s.1).unwrap();
            assert!(end <= layout.total_bytes());
        }
    }

    #[test]
    fn the_sizes_are_those_of_the_plan() {
        let one = StepLayout::new(1, 1);
        assert_eq!(one.place(Buf::Logits).bytes, 800_256);
        assert_eq!(one.place(Buf::Xw).bytes, 9216);
        let sixteen = StepLayout::new(16, 16);
        assert_eq!(sixteen.place(Buf::Logits).bytes, 16 * 800_256);
        // About 0.36 MB at 1 row and 5.7 MB at 16, without the logits.
        let without = |l: &StepLayout| l.total_bytes() - l.place(Buf::Logits).bytes;
        assert!((300_000..450_000).contains(&without(&one)), "{}", without(&one));
        assert!(
            (4_500_000..6_500_000).contains(&without(&sixteen)),
            "{}",
            without(&sixteen)
        );
        let big = StepLayout::new(512, 1);
        assert!(big.total_bytes() < 220_000_000, "{}", big.total_bytes());
        assert_eq!(big.place(Buf::A).bytes, 3 * 512 * 9216);
        // `EOut` is the output of OP_OUT_MIX: e, h_e and the row factors of e.
        assert_eq!(big.place(Buf::EOut).bytes, (2 * 512 * 2304 + 512) * 4);
        assert_eq!(big.eout_parts(), (0, 512 * 2304 * 4, 2 * 512 * 2304 * 4));
        // The gates 0 to 2 and the gate of the mlp mix: 4 gates of 64 values for each row.
        assert_eq!(big.place(Buf::G).bytes, 4 * 512 * 256);
    }

    #[test]
    #[should_panic(expected = "has no row factors")]
    fn the_row_factors_of_a_buffer_without_them_are_a_panic() {
        StepLayout::new(1, 1).rf(Buf::Logits);
    }

    #[test]
    fn the_scratch_regions_are_disjoint_and_the_counters_come_first() {
        let s = ScratchLayout::new(16, 4096);
        let kinds = [
            ScratchKind::Proj,
            ScratchKind::Attn,
            ScratchKind::OutMix,
            ScratchKind::GateDown3,
            ScratchKind::GateUp3,
            ScratchKind::GateDownE,
            ScratchKind::GateUpE,
            ScratchKind::Route,
            ScratchKind::Up,
            ScratchKind::Down,
            ScratchKind::Head,
        ];
        for pair in kinds.windows(2) {
            assert!(s.offset(pair[0]) + s.bytes(pair[0]) <= s.offset(pair[1]), "{pair:?}");
        }
        assert!(s.offset(ScratchKind::Head) + s.bytes(ScratchKind::Head) <= s.total_floats() * 4);
        assert_eq!(s.bytes(ScratchKind::GateDown3), (16 + 1152 * 16) * 4);
        assert_eq!(s.bytes(ScratchKind::GateUp3), (4 + 108 * 16) * 4);
        // The place of the counters does not depend on the rows: the offsets of the first regions do not move.
        let small = ScratchLayout::new(1, 4096);
        assert_eq!(small.offset(ScratchKind::Proj), 0);
        assert!(small.total_floats() < s.total_floats());
    }
}
