//! The control vectors of the Qwen3.8 engine on gfx1151.
//!
//! A pass with a control list runs as launches cut at the layer that the list acts on. After each launch, the host
//! reads the rows of the segment from `h`, applies the items in list order, and writes the rows back. The gates: a
//! strength of 0 gives the bits of no item; on the exact path an `add` matches the `plain_add` run of the oracle;
//! segments with their own lists give the bits of each pass alone; a decode step gives the bits of the same token in a
//! longer pass.
//!
//! Run on the Strix Halo machine with `nagaya.service` stopped, one thread, and with `QWEN38_SHARDS` set. Run it twice:
//! with `QWEN38_FAST_QMM` unset, and with `QWEN38_FAST_QMM=0`. Only the exact path gates the oracle. A test that needs
//! the fixture or the golden files skips with one line when they are absent, or fails with `NAGAYA_REQUIRE_MODELS=1`.

#![cfg(feature = "hip")]

#[path = "support/card.rs"]
mod card;
#[path = "support/oracle.rs"]
mod oracle;

use card::*;
use nagaya_abi::{ControlOp, Engine, State};
use oracle::{Prompt, QmmPath};

/// The gate of the gap of a tapped row from the reference: `max|e - r| / rms(r)`.
///
/// Measured on 2026-11-12, on the exact path, against the reference on the CPU: the largest gap of a layer was 1.8e-5
/// (the run `plain`, layer 46), and the largest gap of a fold row was 2.7e-5 (the run with the `add`). The gate is
/// about 10 times that. A tap of the wrong layer or of the wrong buffer gives a gap near 1.
const RESIDUAL_GATE: f64 = 3.0e-4;

/// The formula of the vector of the oracle, as the tool writes it in the fixture.
const FORMULA: &str = "v[l][j] = (((31*j + 17*l) mod 257) - 128) / 256";

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
fn the_engine_declares_the_table_and_writes_and_refuses_as_the_abi_says() {
    let mut engine = open().expect("open the engine");
    let limits = engine.limits();
    limits.check().expect("the limits pass the check of the ABI");
    let control = limits.control.expect("the engine declares a control table");
    assert_eq!((control.tables, control.layers, control.width), (8, 48, 10_240));
    let term = limits
        .memory
        .mandatory
        .iter()
        .find(|term| term.name == "control")
        .expect("the term control");
    assert_eq!(term.bytes, 15_728_640);
    let table = engine.control().expect("the engine gives the control table");
    assert!(
        table.control_write(8, &vector_bytes(0)).is_err(),
        "entry 8 is not in the table"
    );
    assert!(
        table.control_write(0, &vector_bytes(0)[..100]).is_err(),
        "a short vector"
    );
    let mut bad = vector_bytes(0);
    bad[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(table.control_write(0, &bad).is_err(), "a value that is not finite");
    // An entry that no write filled is refused by a pass, and nothing changes.
    let wanted = segment(
        State::Slot(0),
        &ids(2, 1),
        0,
        true,
        vec![item(7, ControlOp::Add, 1.0, None)],
    );
    let error = engine
        .pass(std::slice::from_ref(&wanted), &[])
        .expect_err("entry 7 was never written");
    assert!(
        error.message.contains("no control_write filled it"),
        "{}",
        error.message
    );
    assert_eq!(engine.n(State::Slot(0)).unwrap(), 0);
    // A list that names a layer past the model is refused.
    write_vector(&mut engine, 0, 0);
    let past = segment(
        State::Slot(0),
        &ids(2, 1),
        0,
        true,
        vec![item(0, ControlOp::Add, 1.0, Some(vec![(48, 1.0)]))],
    );
    assert!(engine.pass(std::slice::from_ref(&past), &[]).is_err());
}

#[test]
fn a_strength_of_0_gives_the_bits_of_no_item() {
    let mut engine = open().expect("open the engine");
    write_vector(&mut engine, 0, 0);
    let mut slot = 0u32;
    for count in [1usize, 17, 256] {
        let prompt = ids(count, 9);
        let plain = feed(&mut engine, slot, &prompt, &[], &[]);
        let plain_parts = parts(&mut engine, slot);
        slot += 1;
        for op in [ControlOp::Add, ControlOp::Ablate] {
            let list = [item(0, op, 0.0, None)];
            let steered = feed(&mut engine, slot, &prompt, &[], &list);
            assert_eq!(
                bits(&plain.logits[0]),
                bits(&steered.logits[0]),
                "{count} tokens, {op:?} of 0: the logits differ"
            );
            assert_parts_equal(
                &format!("{count} tokens, {op:?} of 0"),
                &plain_parts,
                &parts(&mut engine, slot),
            );
            slot += 1;
        }
    }
}

#[test]
fn an_add_of_the_vector_of_the_oracle_matches_the_plain_add_run_on_the_exact_path() {
    let test = "an_add_of_the_vector_of_the_oracle_matches_the_plain_add_run_on_the_exact_path";
    let Some(fixture) = oracle::residual_fixture(test) else {
        return;
    };
    assert_eq!(
        fixture.add.formula, FORMULA,
        "the fixture holds another vector than this test writes"
    );
    let run = fixture.run(&fixture.add.run);
    let Some(rows) = oracle::read_rows(test, &oracle::golden_dir(), &fixture, run, oracle::require_files()) else {
        return;
    };
    let path = QmmPath::from_env();
    let mut engine = open().expect("open the engine");
    write_vector(&mut engine, 0, 0);
    let strength = fixture.add.strength_f32;
    let layers: Vec<(u32, f32)> = fixture
        .add
        .layers
        .iter()
        .map(|layer| (*layer as u32, strength))
        .collect();
    let list = [item(0, ControlOp::Add, strength, Some(layers))];
    let got = feed(&mut engine, 0, &run.ids, &["residual", "fold"], &list);
    let gaps = oracle::layer_gaps(&got.residual[0], &rows.residual).unwrap_or_else(|error| panic!("{error}"));
    for (layer, gap) in gaps.iter().enumerate() {
        println!("{test}: {} layer {layer}: max|e - r| / rms(r) = {gap:e}", path.name());
    }
    let fold_gap = oracle::relative_gap(&got.fold[0], &rows.fold).unwrap_or_else(|error| panic!("{error}"));
    println!("{test}: {} fold: max|e - r| / rms(r) = {fold_gap:e}", path.name());
    let gap = oracle::measure(&prompt_of(run), &got.logits[0]);
    println!("{test}: logits of the add run: {gap:?}");
    // The add changes the rows: the run with no add is another run, and the gap between the two is the signal.
    let plain = fixture.run("plain");
    if plain.ids == run.ids
        && let Some(plain_rows) =
            oracle::read_rows(test, &oracle::golden_dir(), &fixture, plain, oracle::require_files())
    {
        let signal = oracle::layer_gaps(&plain_rows.residual, &rows.residual).expect("gaps");
        println!(
            "{test}: the add moves the oracle rows by up to {:e}",
            signal.iter().cloned().fold(0.0, f64::max)
        );
    }
    if path == QmmPath::Exact {
        let breaches = oracle::gap_breaches(&gaps, RESIDUAL_GATE);
        assert!(breaches.is_empty(), "{breaches:?}");
        assert!(
            fold_gap <= RESIDUAL_GATE,
            "the fold gap {fold_gap:e} is past the gate {RESIDUAL_GATE:e}"
        );
        assert_eq!(oracle::breaches(&gap, path), Vec::<String>::new(), "{gap:?}");
    }
}

#[test]
fn segments_with_their_own_lists_and_a_plain_segment_each_give_the_bits_of_their_pass_alone() {
    let mut engine = open().expect("open the engine");
    write_vector(&mut engine, 0, 0);
    write_vector(&mut engine, 1, 5);
    let lists = [
        vec![item(0, ControlOp::Add, 1.5, Some(vec![(5, 1.5), (30, 0.5)]))],
        vec![
            item(1, ControlOp::Ablate, 1.0, None),
            item(0, ControlOp::Clamp, 0.0, Some(vec![(12, 2.0)])),
        ],
        Vec::new(),
    ];
    let lens = [3usize, 5, 4];
    let prompts: Vec<Vec<u32>> = lens.iter().enumerate().map(|(i, len)| ids(*len, 20 + i)).collect();
    let segments: Vec<_> = (0..3)
        .map(|i| segment(State::Slot(i as u32), &prompts[i], 0, true, lists[i].clone()))
        .collect();
    let joined = run(&mut engine, &segments, &["residual", "fold"]);
    assert_eq!(
        (joined.logits.len(), joined.residual.len(), joined.fold.len()),
        (3, 3, 3)
    );
    let joined_parts: Vec<_> = (0..3).map(|slot| parts(&mut engine, slot)).collect();
    for slot in 0..3u32 {
        clear(&mut engine, slot);
    }
    for i in 0..3 {
        let alone = feed(&mut engine, i as u32, &prompts[i], &["residual", "fold"], &lists[i]);
        assert_eq!(
            bits(&joined.logits[i]),
            bits(&alone.logits[0]),
            "segment {i}: the logits differ"
        );
        assert_eq!(
            bits(&joined.residual[i]),
            bits(&alone.residual[0]),
            "segment {i}: the residual rows differ"
        );
        assert_eq!(
            bits(&joined.fold[i]),
            bits(&alone.fold[0]),
            "segment {i}: the fold row differs"
        );
        assert_parts_equal(&format!("segment {i}"), &joined_parts[i], &parts(&mut engine, i as u32));
    }
    // The plain segment gives the bits of a plain pass with no table and no tap. This also shows that the fused combine
    // of the plain pass and the unfused build of the steered pass give equal bits.
    clear(&mut engine, 0);
    let plain = feed(&mut engine, 0, &prompts[2], &[], &[]);
    assert_eq!(
        bits(&joined.logits[2]),
        bits(&plain.logits[0]),
        "the plain segment differs from a plain pass"
    );
}

#[test]
fn a_steered_decode_step_gives_the_bits_of_the_same_token_in_a_longer_steered_pass() {
    let mut engine = open().expect("open the engine");
    write_vector(&mut engine, 0, 0);
    let list = [item(0, ControlOp::Add, 0.75, None)];
    let prompt = ids(5, 40);
    // A pass of 5 tokens on a fresh context.
    let whole = feed(&mut engine, 0, &prompt, &["residual", "fold"], &list);
    let whole_parts = parts(&mut engine, 0);
    // 4 tokens, then a decode step of 1 token.
    let state = State::Slot(1);
    let head = segment(state, &prompt[..4], 0, false, list.to_vec());
    run(&mut engine, std::slice::from_ref(&head), &[]);
    let step = segment(state, &prompt[4..], 4, true, list.to_vec());
    let decoded = run(&mut engine, std::slice::from_ref(&step), &["residual", "fold"]);
    assert_eq!(
        bits(&whole.logits[0]),
        bits(&decoded.logits[0]),
        "the logits of the decode step differ"
    );
    assert_eq!(
        bits(&whole.residual[0]),
        bits(&decoded.residual[0]),
        "the residual rows differ"
    );
    assert_eq!(bits(&whole.fold[0]), bits(&decoded.fold[0]), "the fold row differs");
    assert_parts_equal("a decode step", &whole_parts, &parts(&mut engine, 1));
}

#[test]
fn a_control_write_over_a_used_entry_changes_the_next_pass() {
    let mut engine = open().expect("open the engine");
    let list = [item(0, ControlOp::Add, 1.0, Some(vec![(10, 1.0), (30, 1.0)]))];
    let prompt = ids(6, 60);
    write_vector(&mut engine, 0, 0);
    let first = feed(&mut engine, 0, &prompt, &[], &list);
    write_vector(&mut engine, 0, 9);
    let second = feed(&mut engine, 1, &prompt, &[], &list);
    assert_ne!(
        bits(&first.logits[0]),
        bits(&second.logits[0]),
        "the new vector did not change the pass"
    );
    write_vector(&mut engine, 0, 0);
    let third = feed(&mut engine, 2, &prompt, &[], &list);
    assert_eq!(
        bits(&first.logits[0]),
        bits(&third.logits[0]),
        "the same vector gives the same bits"
    );
}

#[test]
fn an_ablate_of_1_leaves_nothing_along_the_vector_and_a_clamp_puts_the_part_at_a() {
    let mut engine = open().expect("open the engine");
    write_vector(&mut engine, 0, 0);
    let (layer, shift) = (14usize, 0usize);
    let v = vector_layer(shift, layer);
    let prompt = ids(4, 80);
    let plain = feed(&mut engine, 0, &prompt, &["residual"], &[]);
    let along = |row: &[f32]| dot(row, &v) / norm(&v);
    let before = along(layer_row(&plain.residual[0], layer));
    let layers = Some(vec![(layer as u32, 1.0)]);
    let ablated = feed(
        &mut engine,
        1,
        &prompt,
        &["residual"],
        &[item(0, ControlOp::Ablate, 1.0, layers)],
    );
    let row = layer_row(&ablated.residual[0], layer);
    let left = along(row);
    println!(
        "ablate: along the vector before {before:e}, after {left:e}, |h| {:e}",
        norm(row)
    );
    // The rounding of a row of f32 values bounds the part along the vector: about 6e-8 of |h|.
    let tolerance = 1e-6 * norm(row) + 1e-4;
    assert!(
        left.abs() <= tolerance,
        "the part along the vector is {left:e} after an ablate of 1; the bound is {tolerance:e}"
    );
    assert!(
        before.abs() > 100.0 * tolerance,
        "the plain row has no part along the vector ({before:e}), so the test shows nothing"
    );
    // A clamp of a puts the part along the vector at a.
    let a = 3.0f32;
    let layers = Some(vec![(layer as u32, a)]);
    let clamped = feed(
        &mut engine,
        2,
        &prompt,
        &["residual"],
        &[item(0, ControlOp::Clamp, a, layers)],
    );
    let row = layer_row(&clamped.residual[0], layer);
    let at = along(row);
    println!("clamp: along the vector {at:e}, wanted {a}");
    let tolerance = 1e-6 * norm(row) + 1e-4;
    assert!(
        (at - f64::from(a)).abs() <= tolerance,
        "the part along the vector is {at:e}, not {a}; the bound is {tolerance:e}"
    );
}

#[test]
fn the_time_of_a_list_is_one_line_on_stderr() {
    let mut engine = open().expect("open the engine");
    write_vector(&mut engine, 0, 0);
    let three = vec![(10u32, 1.0f32), (20, 1.0), (30, 1.0)];
    let cases: [(&str, Vec<nagaya_abi::ControlItem>); 4] = [
        ("plain", Vec::new()),
        ("add_3_layers", vec![item(0, ControlOp::Add, 1.0, Some(three))]),
        ("add_48_layers", vec![item(0, ControlOp::Add, 1.0, None)]),
        ("ablate_48_layers", vec![item(0, ControlOp::Ablate, 1.0, None)]),
    ];
    for (slot, (tokens, repeats)) in [(1usize, 5usize), (256, 1)].into_iter().enumerate() {
        let slot = slot as u32;
        let mut times = Vec::new();
        for (_, list) in &cases {
            let mut total = 0.0;
            for _ in 0..repeats {
                clear(&mut engine, slot);
                let state = State::Slot(slot);
                if tokens == 1 {
                    let warm = segment(state, &ids(8, 1), 0, false, Vec::new());
                    run(&mut engine, std::slice::from_ref(&warm), &[]);
                }
                let at = engine.n(state).unwrap();
                let wanted = segment(state, &ids(tokens, 7), at, true, list.clone());
                total += run(&mut engine, std::slice::from_ref(&wanted), &[]).millis;
            }
            times.push(total / repeats as f64);
        }
        eprintln!(
            "control: plain_ms={:.1} add_3_layers_ms={:.1} add_48_layers_ms={:.1} ablate_48_layers_ms={:.1} tokens={tokens}",
            times[0], times[1], times[2], times[3]
        );
    }
}
