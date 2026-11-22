//! A long context of the Inkling 2 engine on a B70 card (milestone 35 step 11): a slot of 131,072 positions, fed
//! with seeded ids in blocks of 512, against the host form (`src/host/forward.rs`).
//!
//! - The host form runs the first 1,536 positions from position 0. At position 1,527 a route of layer 17 in cycle 2
//!   is at a near tie: its eighth and ninth keys differ by 6.0e-8 in the host form, and the card picks the other
//!   expert. After that position the two states differ, so the gate compares no row from position 0 after it.
//! - At each resume point the test reads the state of the slot through `parts`, and the host form runs the next
//!   block from that state. Thus each such block checks the ops at its positions, and not the drift of the state
//!   before it. The blocks at 8,192 and later check the log scale of the full layers, and each block checks the
//!   window of the sliding layers past 1,024.
//! - After the last block, 8 decode steps of the argmax tokens, on the card and on the host.
//!
//! The ids are seeded, and not repeated prose: repeated text puts the routes of the experts near a tie (engine-abi
//! "A route near a tie"). The test also measures the speed: the time of each block and of 8 decode steps at several
//! positions. A second test measures the tower of the largest image, 4,096 by 4,096 px.
//!
//! A resume block can hold a flip too. The resume points of this gate hold none in the card run of 2026-11-21.
//!
//! The test needs the Mini folder, and it skips when the folder is absent. It takes about 20 minutes. Release the
//! card first, and run alone: `NAGAYA_TEST_CARDS=<pci> INKLING2_MODEL_DIR=<folder> cargo test -p
//! nagaya-mk-inkling2-bmg --features xe --release --test xe_long -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::collections::BTreeMap;

use common::Which;
use common::card;
use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::host::forward::{Context, DEFAULT_CYCLES, KV_WIDTH, Model};
use nagaya_mk_inkling2_bmg::host::ops::HIDDEN;
use nagaya_mk_inkling2_bmg::media::IMAGE_TOKEN;
use nagaya_mk_inkling2_bmg::spec::{
    self, CONV_ROW, CONV_ROWS, DEFAULT_EXTRA_RUNS, DEFAULT_MAX_BLOCK, MAX_QUANTUM_MS, Part,
};
use nagaya_text_inkling2::media::{self, Rgb};

/// The positions of the slot. A slot of 164,352 positions is the largest whose buffers all stay in the memory of
/// the card at 3 cycles (engine-abi "The cost of a slot"). This one leaves a margin.
const LONG: usize = 131_072;
/// The host form runs the positions below this one from position 0. The block before it holds the flip at 1,527.
const HOST_END: usize = 1_536;
/// The last block ends at this position. Then 8 decode steps follow.
const END: usize = 130_560;
const STEPS: usize = 8;
/// At each of these positions, 8 decode steps of the seeded ids take the place of the start of a block. The block of
/// the other 504 ids follows, so each later block starts on the grid of 512.
const PROBES: [usize; 4] = [1_024, 8_704, 32_768, 65_536];
/// Before the block at each of these positions, the test reads the state of the slot for the host form. The block at
/// 512 is also inside the span from position 0, so it checks the read of the state.
const RESUMES: [usize; 8] = [512, 4_096, 8_192, 16_384, 32_256, 65_024, 98_304, 130_048];
/// The rows of a resume block whose logits the gate compares.
const RESUME_ROWS: [usize; 5] = [0, 1, 255, 256, 511];
/// The positions below `HOST_END` whose logits the gate compares: none after the flip at 1,527.
const EARLY_ROWS: [usize; 4] = [511, 1_024, 1_031, 1_526];
/// The largest gap of a logit from the host form. The card run of 2026-11-21 measured 1.5e-5 at 2,203 positions
/// (`tests/xe_feed.rs`), and 2.0e-5 at 130,567 (this test).
const HOST_TOL: f64 = 1e-4;

/// `LONG` token ids from a seeded generator, below 150,000, as in `tests/xe_feed.rs`.
fn seeded_ids() -> Vec<u32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..LONG)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 150_000) as u32
        })
        .collect()
}

/// The passes of the feed as `(from, length)`: blocks of 512, and 8 decode steps and a block of 504 at each probe.
fn schedule() -> Vec<(usize, usize)> {
    let block = DEFAULT_MAX_BLOCK;
    let mut out = Vec::new();
    let mut at = 0;
    while at < END {
        if PROBES.contains(&at) {
            out.extend((0..STEPS).map(|k| (at + k, 1)));
            out.push((at + STEPS, block - STEPS));
        } else {
            out.push((at, block));
        }
        at += block;
    }
    out
}

/// The positions whose logits the card gives.
fn wanted() -> Vec<usize> {
    let mut rows: Vec<usize> = EARLY_ROWS.to_vec();
    for r in RESUMES {
        rows.extend(RESUME_ROWS.iter().map(|k| r + k));
    }
    rows.sort();
    rows.dedup();
    rows
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap() as u32
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The id of a part of state `s` in the part list of the engine.
fn part_id(part: Part) -> usize {
    spec::part_list(DEFAULT_EXTRA_RUNS)
        .iter()
        .position(|p| *p == part)
        .expect("the part list holds each part of a state")
}

/// The rows of positions `from..to` of a row part of slot 0, in reads of at most 8,192 rows.
fn read_rows(engine: &mut Inkling2Engine, part: Part, from: usize, to: usize) -> Vec<f32> {
    let id = part_id(part);
    let parts = engine.parts().unwrap();
    let mut out = Vec::with_capacity((to - from) * KV_WIDTH);
    let mut buf = Vec::new();
    let mut at = from;
    while at < to {
        let n = (to - at).min(8_192);
        buf.resize(n * spec::ROW_BYTES as usize, 0u8);
        parts
            .cells_read(State::Slot(0), id, at as u64, n as u64, &mut buf)
            .unwrap_or_else(|e| panic!("a read of rows {at} to {} of {part:?}: {e:?}", at + n));
        out.extend(floats(&buf));
        at += n;
    }
    out
}

/// The host state of slot 0 at position `at`, from the parts of the slot. A full run holds every row, and a sliding
/// run the last 1,024. The conv part holds 3 rows of the four inputs, oldest first, and the host holds each input
/// as 3 rows.
fn host_context(engine: &mut Inkling2Engine, at: usize) -> Context {
    let mut ctx = Context::new(DEFAULT_CYCLES);
    ctx.position = at;
    let runs = spec::runs(DEFAULT_CYCLES);
    assert_eq!(runs.len(), ctx.runs.len(), "the card and the host run the same runs");
    for (run, state) in runs.iter().zip(ctx.runs.iter_mut()) {
        assert_eq!(
            run.layer, state.layer,
            "the card and the host run the layers in one order"
        );
        let first = state.keys.window.map_or(0, |w| at.saturating_sub(w));
        state.keys.first = first;
        state.keys.rows = read_rows(engine, Part::K(run.state), first, at);
        state.values.first = first;
        state.values.rows = read_rows(engine, Part::V(run.state), first, at);
        let mut conv = vec![0u8; spec::CONV_BYTES as usize];
        engine
            .parts()
            .unwrap()
            .part_read(State::Slot(0), part_id(Part::Conv(run.state)), &mut conv)
            .unwrap_or_else(|e| panic!("a read of the conv part of state {}: {e:?}", run.state));
        let conv = floats(&conv);
        let widths = [KV_WIDTH, KV_WIDTH, HIDDEN, HIDDEN];
        let mut offset = 0;
        for (input, width) in state.conv.iter_mut().zip(widths) {
            *input = (0..CONV_ROWS)
                .flat_map(|r| {
                    conv[r * CONV_ROW + offset..r * CONV_ROW + offset + width]
                        .iter()
                        .copied()
                })
                .collect();
            offset += width;
        }
        assert_eq!(offset, CONV_ROW, "the four inputs fill a conv row");
    }
    ctx
}

/// One gap of a logit row from the host form.
struct Gap {
    what: String,
    position: usize,
    gap: f64,
    argmax: bool,
}

fn gap(what: &str, position: usize, card: &[f32], host: &[f32]) -> Gap {
    let gap = common::max_abs(card, host);
    let argmax = argmax(card) == argmax(host);
    eprintln!(
        "{what}: position {position}: largest gap {gap:.3e}, argmax {}",
        if argmax { "agrees" } else { "DIFFERS" }
    );
    Gap {
        what: what.to_owned(),
        position,
        gap,
        argmax,
    }
}

/// The median of `v`, in place.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A slot of 131,072 positions follows the host form: from position 0 up to 1,526, from the state of the slot at
/// eight resume points, and in 8 decode steps after 130,560 positions.
#[test]
fn a_long_context_follows_the_host_form() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let ids = seeded_ids();
    let passes = schedule();
    let wanted = wanted();
    let mut engine = card::open(Which::Mini, vec![LONG], DEFAULT_MAX_BLOCK);

    // The card: the feed, the reads of the state at each resume point, and the times.
    let mut got: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut resumed: Vec<(usize, Context)> = Vec::new();
    let mut times: Vec<(usize, usize, f64)> = Vec::new();
    let started = std::time::Instant::now();
    for (from, n) in &passes {
        let (from, n) = (*from, *n);
        if RESUMES.contains(&from) {
            let read = std::time::Instant::now();
            resumed.push((from, host_context(&mut engine, from)));
            eprintln!("the state at {from}: read in {:.1} s", read.elapsed().as_secs_f64());
        }
        let rows: Vec<u64> = wanted
            .iter()
            .filter(|p| (from..from + n).contains(*p))
            .map(|p| *p as u64)
            .collect();
        let pass = std::time::Instant::now();
        let out = card::pass(&mut engine, 0, from as u64, &ids[from..from + n], &rows);
        times.push((from, n, pass.elapsed().as_secs_f64() * 1e3));
        got.extend(rows.iter().map(|p| *p as usize).zip(out));
        if (from + n).is_multiple_of(16_384) {
            eprintln!(
                "the card fed {} positions in {:.0} s; the last block took {:.0} ms",
                from + n,
                started.elapsed().as_secs_f64(),
                times.last().unwrap().2
            );
        }
    }
    // The last block gives the logits of its last position, and the decode steps follow its argmax.
    let mut tokens = Vec::new();
    let mut decoded = Vec::new();
    let mut step_ms = Vec::new();
    let mut next = argmax(&got[&(END - 1)]);
    for k in 0..STEPS {
        let pos = END + k;
        tokens.push(next);
        let step = std::time::Instant::now();
        let row = card::pass(&mut engine, 0, pos as u64, &[next], &[pos as u64]).remove(0);
        step_ms.push(step.elapsed().as_secs_f64() * 1e3);
        next = argmax(&row);
        decoded.push(row);
    }
    eprintln!(
        "the card fed {} positions in {:.0} s",
        END + STEPS,
        started.elapsed().as_secs_f64()
    );
    drop(engine);

    // The speed: the time of the blocks of 512 near each position, and of the decode steps of each probe.
    for at in [0, 1_536, 8_192, 16_384, 33_280, 66_048, 98_304, 130_048] {
        let (_, n, ms) = times.iter().find(|(f, _, _)| *f == at).unwrap();
        eprintln!(
            "speed: the block at {at} of {n} tokens took {ms:.0} ms, {:.0} tokens each second ({})",
            *n as f64 * 1e3 / ms,
            if *ms > MAX_QUANTUM_MS as f64 {
                "over max_quantum_ms"
            } else {
                "within max_quantum_ms"
            }
        );
    }
    let slowest = times.iter().map(|t| t.2).fold(0.0, f64::max);
    eprintln!("speed: the slowest pass took {slowest:.0} ms, and max_quantum_ms is {MAX_QUANTUM_MS}");
    for p in PROBES {
        let mut ms: Vec<f64> = times
            .iter()
            .filter(|(f, n, _)| *n == 1 && (p..p + STEPS).contains(f))
            .map(|t| t.2)
            .collect();
        eprintln!(
            "speed: a decode step at {p} takes {:.1} ms (median of {STEPS})",
            median(&mut ms)
        );
    }
    eprintln!(
        "speed: a decode step at {END} takes {:.1} ms (median of {STEPS})",
        median(&mut step_ms)
    );

    // The host form from position 0, in the passes of the card, up to HOST_END.
    let model = Model::open(&common::model_weights(Which::Mini)).unwrap();
    let mut gaps = Vec::new();
    let mut ctx = Context::new(DEFAULT_CYCLES);
    let host = std::time::Instant::now();
    let mut early: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    for (from, n) in passes.iter().copied().filter(|(f, _)| *f < HOST_END) {
        let rows: Vec<usize> = wanted
            .iter()
            .filter(|p| (from..from + n).contains(*p))
            .map(|p| p - from)
            .collect();
        let out = model.forward(&mut ctx, &ids[from..from + n], &rows, None);
        early.extend(rows.iter().map(|r| from + r).zip(out));
    }
    drop(ctx);
    eprintln!(
        "the host form of {HOST_END} positions took {:.0} s",
        host.elapsed().as_secs_f64()
    );
    for (p, row) in &early {
        gaps.push(gap("from position 0", *p, &got[p], row));
    }

    // The host form from the state of the card at each resume point: the block of 512, and the decode steps after
    // the last block.
    for (at, mut ctx) in resumed {
        let host = std::time::Instant::now();
        let rows = RESUME_ROWS;
        let out = model.forward(&mut ctx, &ids[at..at + DEFAULT_MAX_BLOCK], &rows, None);
        eprintln!(
            "the host block at {at} from the state of the card took {:.0} s",
            host.elapsed().as_secs_f64()
        );
        for (r, row) in rows.iter().zip(&out) {
            let p = at + r;
            gaps.push(gap(&format!("from the state at {at}"), p, &got[&p], row));
            if let Some(scratch) = early.get(&p) {
                let g = common::max_abs(row, scratch);
                eprintln!(
                    "from the state at {at}: position {p}: the host from the state of the card is within {g:.3e} \
                     of the host from position 0"
                );
                assert!(g < HOST_TOL, "the state read from the card gives other logits: {g:.3e}");
            }
        }
        if at + DEFAULT_MAX_BLOCK == END {
            for (k, (token, row)) in tokens.iter().zip(&decoded).enumerate() {
                let want = model.forward(&mut ctx, &[*token], &[0], None).remove(0);
                gaps.push(gap("a decode step", END + k, row, &want));
            }
        }
    }

    let worst = gaps.iter().map(|g| g.gap).fold(0.0, f64::max);
    eprintln!(
        "the long context: worst gap {worst:.3e} from the host form over {} rows",
        gaps.len()
    );
    for g in &gaps {
        assert!(
            g.argmax,
            "{}: position {}: the argmax differs from the host form",
            g.what, g.position
        );
        assert!(
            g.gap < HOST_TOL,
            "{}: position {}: a logit differs from the host form by {:.3e}",
            g.what,
            g.position,
            g.gap
        );
    }
}

/// The time of the tower of the largest image: 4,096 by 4,096 px, 10,609 positions, in blocks of 512 from position
/// 0. The tower runs once, in the pass of the first block. The test prints that time against `max_quantum_ms`, and
/// it checks that the logits after the image are finite. A second feed of the same image reads the held projector
/// rows (`media::HeldRows`), so its tower does not run, and it gives the same bits.
#[test]
fn the_tower_of_the_largest_image_runs_once() {
    if common::model_absent(Which::Mini) {
        return;
    }
    let side = 4_096;
    // A synthetic image of seeded pixels: the time of the tower does not depend on the pixels.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let bytes: Vec<u8> = (0..side * side * 3)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        })
        .collect();
    let image = media::rows_of(&Rgb {
        width: side,
        height: side,
        bytes,
    })
    .unwrap();
    let n = image.positions();
    eprintln!("the image: {n} positions, a grid of {:?}", image.grid);
    assert_eq!(n, 10_609, "an image of 4,096 by 4,096 px has 10,609 tokens");
    let rows = image.bytes();
    let mut engine = card::open(Which::Mini, vec![16_384], DEFAULT_MAX_BLOCK);
    let segment = |from: usize, to: usize, logits: Vec<u64>| Segment {
        control: Vec::new(),
        state: State::Slot(0),
        inputs: SegmentInputs::TokensAndRows {
            tokens: vec![IMAGE_TOKEN; to - from],
            rows: rows.clone(),
        },
        positions: (from as u64..to as u64).collect(),
        meta: image.record(0),
        outputs: OutputsWanted { logit_rows: logits },
        layer_loop: None,
        cycles: None,
    };
    // A plain block of 512 tokens, for the time of a block with no tower.
    let plain: Vec<u32> = (0..512u32).map(|i| (i * 7_919) % 150_000).collect();
    let block = std::time::Instant::now();
    card::pass(&mut engine, 0, 0, &plain, &[]);
    let plain_ms = block.elapsed().as_secs_f64() * 1e3;
    card::clear(&mut engine, 0);
    let mut first: Vec<f32> = Vec::new();
    for round in 0..2 {
        let runs = engine.tower_runs();
        let mut ms = Vec::new();
        let mut last = Vec::new();
        for from in (0..n).step_by(DEFAULT_MAX_BLOCK) {
            let to = (from + DEFAULT_MAX_BLOCK).min(n);
            let logits = if to == n { vec![n as u64 - 1] } else { Vec::new() };
            let started = std::time::Instant::now();
            let out = engine
                .pass(&[segment(from, to, logits)], &[])
                .unwrap_or_else(|e| panic!("the block {from}..{to}: {}", e.message));
            ms.push(started.elapsed().as_secs_f64() * 1e3);
            if let Some(row) = out.outputs.first() {
                last = floats(&row.bytes);
            }
        }
        let rest: f64 = ms[1..].iter().sum::<f64>() / (ms.len() - 1) as f64;
        eprintln!(
            "round {round}: the first block took {:.0} ms, a later block {rest:.0} ms on average, a plain block \
             {plain_ms:.0} ms; max_quantum_ms is {MAX_QUANTUM_MS}",
            ms[0]
        );
        if round == 0 {
            eprintln!("the tower of the image takes about {:.0} ms", ms[0] - rest);
        }
        let want = if round == 0 { 1 } else { 0 };
        assert_eq!(
            engine.tower_runs() - runs,
            want,
            "round {round}: the tower runs once for a new image, and not for the held one"
        );
        assert!(
            last.iter().all(|v| v.is_finite()),
            "a logit after the image is not finite"
        );
        if round == 0 {
            first = last;
        } else {
            assert!(
                card::other_bits(&first, &last).0 == 0,
                "the held projector rows give other bits"
            );
        }
        card::clear(&mut engine, 0);
    }
}
