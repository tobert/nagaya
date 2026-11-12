//! The taps `residual` and `fold` of the Qwen3.8 engine on gfx1151 (milestone 34 step 5).
//!
//! A pass with a tap runs as 49 launches, cut at the layer ends, and the host reads rows of `h` between two launches.
//! The gates: a tap changes no bit of the logits or of the stored parts; the rows match the oracle on the exact path;
//! a tap in a pass of 16 segments gives the rows of each pass alone.
//!
//! Run on gfx1151host with `nagaya.service` stopped, one thread, and with `QWEN38_SHARDS` set. Run it twice: with
//! `QWEN38_FAST_QMM` unset, and with `QWEN38_FAST_QMM=0`. Only the exact path gates the oracle rows. The fixture and the
//! golden files come from `tools/oracle_fixture.py --residual`. A test that needs them skips with one line when they are
//! absent, or fails with `NAGAYA_REQUIRE_MODELS=1`.

#![cfg(feature = "hip")]

#[path = "support/card.rs"]
mod card;
#[path = "support/oracle.rs"]
mod oracle;

use card::*;
use nagaya_abi::{Engine, State};
use oracle::{Prompt, QmmPath};

/// The gate of the gap of a tapped row from the reference: `max|e - r| / rms(r)`.
///
/// measured on (not yet): fill after the first card run. Until then the value is loose, and each test prints the gap of
/// each layer, so the first run gives the numbers.
const RESIDUAL_GATE: f64 = 0.5;

/// The prompt of a run of the fixture, as `oracle::measure` takes it.
fn prompt_of(run: &oracle::ResidualRun) -> Prompt {
    Prompt {
        name: run.name.clone(),
        source: String::new(),
        text: String::new(),
        ids: run.ids.clone(),
        top_ids: run.top_ids.clone(),
        top_logits: run.top_logits.clone(),
        logsumexp: run.logsumexp,
    }
}

#[test]
fn a_tap_changes_no_bit_of_the_logits_or_of_the_stored_parts() {
    let mut engine = open().expect("open the engine");
    // 1 token takes the fused combine, 17 and 256 do not (`kFuseMaxT` is 16).
    for (case, count) in [1usize, 17, 256].into_iter().enumerate() {
        let (plain_slot, tapped_slot) = (2 * case as u32, 2 * case as u32 + 1);
        let prompt = ids(count, 5);
        let plain = feed(&mut engine, plain_slot, &prompt, &[], &[]);
        let tapped = feed(&mut engine, tapped_slot, &prompt, &["residual", "fold"], &[]);
        assert_eq!(plain.logits.len(), 1);
        assert_eq!(
            bits(&plain.logits[0]),
            bits(&tapped.logits[0]),
            "{count} tokens: a tap changed the logits"
        );
        assert_parts_equal(
            &format!("{count} tokens"),
            &parts(&mut engine, plain_slot),
            &parts(&mut engine, tapped_slot),
        );
        // One row of each tap, of the sizes of the layout, and finite.
        assert_eq!((tapped.residual.len(), tapped.fold.len()), (1, 1));
        assert!(tapped.residual[0].iter().all(|x| x.is_finite()));
        assert!(tapped.fold[0].iter().all(|x| x.is_finite()));
        assert!(plain.residual.is_empty() && plain.fold.is_empty());
    }
}

#[test]
fn the_fold_has_the_bits_of_a_pass_with_no_cut() {
    // `fold` alone makes one launch, and `residual` makes 49. The fold row is the same either way.
    let mut engine = open().expect("open the engine");
    let prompt = ids(9, 11);
    let one = feed(&mut engine, 0, &prompt, &["fold"], &[]);
    let cut = feed(&mut engine, 1, &prompt, &["residual", "fold"], &[]);
    assert_eq!(
        bits(&one.fold[0]),
        bits(&cut.fold[0]),
        "49 launches changed the fold row"
    );
    assert_eq!(bits(&one.logits[0]), bits(&cut.logits[0]));
}

/// Compare the rows of a run of the fixture with the rows of the engine, and print the gap of each layer. Only the exact
/// path gates.
fn check_run(test: &str, run_name: &str) {
    let Some(fixture) = oracle::residual_fixture(test) else {
        return;
    };
    let run = fixture.run(run_name);
    let Some(rows) = oracle::read_rows(test, &oracle::golden_dir(), &fixture, run, oracle::require_files()) else {
        return;
    };
    let path = QmmPath::from_env();
    let mut engine = open().expect("open the engine");
    let got = feed(&mut engine, 0, &run.ids, &["residual", "fold"], &[]);
    assert_eq!((got.residual.len(), got.fold.len()), (1, 1));
    let gaps = oracle::layer_gaps(&got.residual[0], &rows.residual).unwrap_or_else(|error| panic!("{error}"));
    for (layer, gap) in gaps.iter().enumerate() {
        println!(
            "{test}: {run_name}: {} layer {layer}: max|e - r| / rms(r) = {gap:e}",
            path.name()
        );
    }
    let fold_gap = oracle::relative_gap(&got.fold[0], &rows.fold).unwrap_or_else(|error| panic!("{error}"));
    println!(
        "{test}: {run_name}: {} fold: max|e - r| / rms(r) = {fold_gap:e}",
        path.name()
    );
    // The logits of the same pass against the top-64 of the reference.
    let gap = oracle::measure(&prompt_of(run), &got.logits[0]);
    println!("{test}: {run_name}: logits: {gap:?}");
    if path == QmmPath::Exact {
        let breaches = oracle::gap_breaches(&gaps, RESIDUAL_GATE);
        assert!(breaches.is_empty(), "{run_name}: {breaches:?}");
        assert!(
            fold_gap <= RESIDUAL_GATE,
            "{run_name}: the fold gap {fold_gap:e} is past the gate {RESIDUAL_GATE:e}"
        );
        assert_eq!(
            oracle::breaches(&gap, path),
            Vec::<String>::new(),
            "{run_name}: {gap:?}"
        );
    }
}

#[test]
fn the_rows_of_the_plain_run_match_the_oracle_on_the_exact_path() {
    check_run("the_rows_of_the_plain_run_match_the_oracle_on_the_exact_path", "plain");
}

#[test]
fn the_rows_of_the_chat_run_match_the_oracle_on_the_exact_path() {
    check_run("the_rows_of_the_chat_run_match_the_oracle_on_the_exact_path", "chat");
}

#[test]
fn the_fold_is_the_row_that_the_head_reads() {
    // The head reads the fold. The logits of the pass match the reference, and the fold matches the reference's fold,
    // within the same gate. Both come from one pass.
    let test = "the_fold_is_the_row_that_the_head_reads";
    let Some(fixture) = oracle::residual_fixture(test) else {
        return;
    };
    let run = fixture.run("plain");
    let Some(rows) = oracle::read_rows(test, &oracle::golden_dir(), &fixture, run, oracle::require_files()) else {
        return;
    };
    let path = QmmPath::from_env();
    let mut engine = open().expect("open the engine");
    let got = feed(&mut engine, 0, &run.ids, &["fold"], &[]);
    let gap = oracle::measure(&prompt_of(run), &got.logits[0]);
    let fold_gap = oracle::relative_gap(&got.fold[0], &rows.fold).unwrap_or_else(|error| panic!("{error}"));
    println!("{test}: {}: logits {gap:?}, fold gap {fold_gap:e}", path.name());
    if path == QmmPath::Exact {
        assert_eq!(oracle::breaches(&gap, path), Vec::<String>::new(), "{gap:?}");
        assert!(
            fold_gap <= RESIDUAL_GATE,
            "the fold gap {fold_gap:e} is past the gate {RESIDUAL_GATE:e}"
        );
    }
}

#[test]
fn a_tap_in_a_pass_of_16_segments_gives_the_rows_of_each_pass_alone() {
    let mut engine = open().expect("open the engine");
    // 16 segments of 1 to 16 tokens: 136 tokens, the first 12 on small contexts and the last 4 on large ones.
    let lens: Vec<usize> = (1..=16).collect();
    let prompts: Vec<Vec<u32>> = lens.iter().map(|len| ids(*len, 3 * len)).collect();
    let segments: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(slot, prompt)| segment(State::Slot(slot as u32), prompt, 0, true, Vec::new()))
        .collect();
    let joined = run(&mut engine, &segments, &["residual", "fold"]);
    assert_eq!(
        (joined.logits.len(), joined.residual.len(), joined.fold.len()),
        (16, 16, 16)
    );
    let joined_parts: Vec<_> = (0..16).map(|slot| parts(&mut engine, slot)).collect();
    for slot in 0..16u32 {
        clear(&mut engine, slot);
    }
    for (slot, prompt) in prompts.iter().enumerate() {
        let alone = feed(&mut engine, slot as u32, prompt, &["residual", "fold"], &[]);
        assert_eq!(
            bits(&joined.logits[slot]),
            bits(&alone.logits[0]),
            "segment {slot}: the logits of the joined pass differ"
        );
        assert_eq!(
            bits(&joined.residual[slot]),
            bits(&alone.residual[0]),
            "segment {slot}: the residual rows of the joined pass differ"
        );
        assert_eq!(
            bits(&joined.fold[slot]),
            bits(&alone.fold[0]),
            "segment {slot}: the fold row of the joined pass differs"
        );
        assert_parts_equal(
            &format!("segment {slot}"),
            &joined_parts[slot],
            &parts(&mut engine, slot as u32),
        );
    }
}

#[test]
fn a_segment_with_no_logit_row_gives_no_tap_output() {
    let mut engine = open().expect("open the engine");
    let (a, b) = (ids(3, 1), ids(4, 2));
    let wants = [
        segment(State::Slot(0), &a, 0, false, Vec::new()),
        segment(State::Slot(1), &b, 0, true, Vec::new()),
    ];
    let got = run(&mut engine, &wants, &["residual", "fold"]);
    assert_eq!((got.logits.len(), got.residual.len(), got.fold.len()), (1, 1, 1));
    // The unknown tap and a tap with no logit row are refused, and nothing changes.
    let none = [segment(State::Slot(2), &a, 0, false, Vec::new())];
    let error = engine.pass(&none, &taps(&["residual"])).expect_err("no logit row");
    assert!(
        error.message.contains("no segment asks for a logit row"),
        "{}",
        error.message
    );
    let error = engine
        .pass(&[segment(State::Slot(2), &a, 0, true, Vec::new())], &taps(&["lens"]))
        .expect_err("the engine has no lens");
    assert!(error.message.contains("`residual` and `fold`"), "{}", error.message);
    assert_eq!(
        engine.n(State::Slot(2)).unwrap(),
        0,
        "a refused pass changed the context"
    );
}

#[test]
fn the_time_of_a_tap_is_one_line_on_stderr() {
    let mut engine = open().expect("open the engine");
    // A decode step: 8 tokens in the context, then 1. A prefill: 256 tokens.
    for (slot, (tokens, repeats)) in [(1usize, 5usize), (256, 1)].into_iter().enumerate() {
        let slot = slot as u32;
        let mut times = Vec::new();
        for names in [&[][..], &["residual"][..], &["fold"][..]] {
            let mut total = 0.0;
            for _ in 0..repeats {
                clear(&mut engine, slot);
                let state = State::Slot(slot);
                if tokens == 1 {
                    let warm = segment(state, &ids(8, 1), 0, false, Vec::new());
                    run(&mut engine, std::slice::from_ref(&warm), &[]);
                }
                let at = engine.n(state).unwrap();
                let wanted = segment(state, &ids(tokens, 7), at, true, Vec::new());
                total += run(&mut engine, std::slice::from_ref(&wanted), names).millis;
            }
            times.push(total / repeats as f64);
        }
        eprintln!(
            "tap: plain_ms={:.1} residual_ms={:.1} fold_ms={:.1} tokens={tokens}",
            times[0], times[1], times[2]
        );
    }
}
