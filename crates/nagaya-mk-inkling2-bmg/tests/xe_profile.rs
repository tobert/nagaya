//! The profile of the Inkling 2 engine on a B70 card (milestone 36 item 0): where the time of a decode step and of
//! a prefill block goes, for each op kind. The profile binary gives the bits of the engine binary, and its clocks
//! cover each launch.
//!
//! Each table prints. A test skips when the model or the oracle file is absent. Release the card first, and run
//! alone: `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --release --features xe --test xe_profile --
//! --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_abi::Engine;
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::profile::{self, Launch, TaskTime};
use nagaya_mk_inkling2_bmg::spec::{Config, DEFAULT_MAX_BLOCK};

/// A reserve of 4 cycles past the first, so a pass can run 1 to 5 cycles.
const EXTRA_RUNS: usize = 48;
/// The positions of the prefix before the decode steps.
const PREFIX: usize = 1_024;
/// The least share of the host time of a launch call that the tasks of a profile cover. The rest is the fill of the
/// scoreboard, the copy of the ring, the start of the kernel and the wait for its end.
const COVER: f64 = 0.9;
/// The longest boundary of a task, in us. A boundary is the signal, the read of the next task and the wait, about
/// 0.6 us in the run of 2026-11-22. A clock that the compiler moves into the op makes it hundreds of us.
const BOUNDARY_US: f64 = 20.0;

fn absent() -> bool {
    common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.oracle.json")
}

/// The prompt "longer" of the oracle, 637 tokens of prose, repeated to `n` ids.
fn prose(n: usize) -> Vec<u32> {
    let oracle = common::golden_json("inkling2-mini.oracle.json");
    let prompt = oracle["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "longer")
        .expect("the oracle holds the prompt longer")
        .clone();
    let ids: Vec<u32> = prompt["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    ids.iter().copied().cycle().take(n).collect()
}

fn config() -> Config {
    Config {
        model_dir: card::model_dir(Which::Mini),
        slots: vec![4_096],
        max_block: DEFAULT_MAX_BLOCK,
        extra_runs: EXTRA_RUNS,
    }
}

fn open(profiled: bool) -> Inkling2Engine {
    let cards = nagaya_mk_bmg::test_cards::card();
    let opened = if profiled {
        Inkling2Engine::open_profiled(config(), &cards)
    } else {
        Inkling2Engine::open(config(), &cards)
    };
    opened.unwrap_or_else(|e| panic!("the open failed: {e}"))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap() as u32
}

/// The logits of a pass of `tokens` from `from` at `cycles` cycles, for its last position.
fn pass(engine: &mut Inkling2Engine, from: usize, tokens: &[u32], cycles: u8) -> Vec<f32> {
    let last = (from + tokens.len() - 1) as u64;
    let mut segment = card::segment(0, from as u64, tokens, &[last]);
    segment.cycles = Some(cycles);
    let out = engine
        .pass(&[segment], &[])
        .unwrap_or_else(|e| panic!("a pass at {from} of {} tokens: {}", tokens.len(), e.message));
    card::floats(&out.outputs[0].bytes)
}

/// One decode step: its logits, the wall time of its pass in ms, and its launches (none for the engine binary).
struct Step {
    logits: Vec<f32>,
    wall: f64,
    launches: Vec<Launch>,
}

/// Feed the prefix in blocks, then `steps` greedy decode steps.
fn decode(engine: &mut Inkling2Engine, cycles: u8, steps: usize) -> Vec<Step> {
    card::clear(engine, 0);
    let ids = prose(PREFIX);
    let mut logits = Vec::new();
    for at in (0..PREFIX).step_by(DEFAULT_MAX_BLOCK) {
        logits = pass(engine, at, &ids[at..at + DEFAULT_MAX_BLOCK], cycles);
    }
    engine.take_profile();
    let mut out = Vec::new();
    for k in 0..steps {
        let next = argmax(&logits);
        let started = std::time::Instant::now();
        logits = pass(engine, PREFIX + k, &[next], cycles);
        out.push(Step {
            logits: logits.clone(),
            wall: started.elapsed().as_secs_f64() * 1e3,
            launches: engine.take_profile(),
        });
    }
    out
}

/// The task times of each launch. A launch whose clocks fail a check fails the test.
fn times_of(launches: &[Launch]) -> Vec<Vec<TaskTime>> {
    launches
        .iter()
        .enumerate()
        .map(|(i, l)| profile::task_times(l).unwrap_or_else(|e| panic!("launch {i}: {e}")))
        .collect()
}

/// The ticks that the tasks of a launch cover.
fn covered(times: &[TaskTime]) -> u64 {
    times.iter().map(|t| t.boundary + t.work).sum()
}

fn spread(label: &str, values: &mut [f64]) -> String {
    values.sort_by(f64::total_cmp);
    format!(
        "{label}: median {:.2} ms, {:.2} to {:.2} over {}",
        values[values.len() / 2],
        values[0],
        values[values.len() - 1],
        values.len()
    )
}

/// The profile binary gives the bits of the engine binary, and the tasks of each launch cover most of the card clock
/// of the launch call.
#[test]
fn a_profiled_pass_gives_the_bits_of_the_engine_and_covers_its_launches() {
    if absent() {
        return;
    }
    let steps = 8;
    let mut engine = open(false);
    let want = decode(&mut engine, 3, steps);
    let mut walls: Vec<f64> = want.iter().map(|s| s.wall).collect();
    eprintln!("{}", spread("the engine binary, a decode step at 3 cycles", &mut walls));
    assert!(
        want.iter().all(|s| s.launches.is_empty()),
        "the engine binary records no clocks"
    );
    drop(engine);

    let mut engine = open(true);
    let got = decode(&mut engine, 3, steps);
    let mut walls: Vec<f64> = got.iter().map(|s| s.wall).collect();
    eprintln!(
        "{}",
        spread("the profile binary, a decode step at 3 cycles", &mut walls)
    );
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            bits(&g.logits) == bits(&w.logits),
            "decode step {k} of the profile binary gave other bits"
        );
    }
    let launches: Vec<Launch> = got.into_iter().flat_map(|s| s.launches).collect();
    assert!(launches.len() >= steps, "a pass records each of its launches");
    let rate = profile::ticks_per_ns(&launches);
    eprintln!("the clock: {rate:.3} counts each ns");
    for (i, (launch, times)) in launches.iter().zip(times_of(&launches)).enumerate() {
        let share = covered(&times) as f64 / rate / launch.host_ns as f64;
        let longest = times.iter().map(|t| t.boundary).max().unwrap() as f64 / rate / 1e3;
        eprintln!(
            "launch {i}: the tasks cover {:.2}% of the launch call; the longest boundary is {longest:.2} us",
            share * 100.0
        );
        assert!(
            longest < BOUNDARY_US,
            "launch {i}: a boundary of {longest:.1} us is longer than {BOUNDARY_US} us"
        );
        assert!(
            (COVER..=1.0).contains(&share),
            "the tasks of launch {i} cover {:.1}% of the launch call, outside {:.0}% to 100%",
            share * 100.0,
            COVER * 100.0
        );
    }
}

/// The profile of a decode step at position 1,024 at 3, 1 and 5 cycles, and of a prefill block of 512 tokens from
/// position 0 at 3 cycles. It prints a table of each, and the spread of the steps.
#[test]
fn the_profile_of_a_decode_step_and_a_prefill_block() {
    if absent() {
        return;
    }
    let steps = 16;
    let mut engine = open(true);
    for cycles in [3u8, 1, 5] {
        engine.take_profile();
        let ids = prose(PREFIX);
        card::clear(&mut engine, 0);
        pass(&mut engine, 0, &ids[..DEFAULT_MAX_BLOCK], cycles);
        let block = engine.take_profile();
        // The first step, which warms the card after the blocks, is not in the table.
        let mut decoded = decode(&mut engine, cycles, steps + 1);
        decoded.remove(0);
        let step_launches = decoded[0].launches.len();
        let mut walls: Vec<f64> = decoded.iter().map(|s| s.wall).collect();
        let launches: Vec<Launch> = decoded.into_iter().flat_map(|s| s.launches).collect();
        let times = times_of(&launches);
        let rate = profile::ticks_per_ns(&launches);
        let mut kernel: Vec<f64> = times
            .chunks(step_launches)
            .map(|step| step.iter().map(|t| covered(t)).sum::<u64>() as f64 / rate / 1e6)
            .collect();
        let mut call: Vec<f64> = launches
            .chunks(step_launches)
            .map(|step| step.iter().map(|l| l.host_ns as f64 / 1e6).sum())
            .collect();
        let all: Vec<TaskTime> = times.concat();
        eprintln!(
            "\n## A decode step at position {PREFIX} at {cycles} cycles: {} tasks in {step_launches} launches\n",
            all.len() / steps
        );
        eprintln!("{}", profile::table(&profile::rows(&all, profile::by_op), steps, rate));
        eprintln!("{}", spread("the tasks of a step", &mut kernel));
        eprintln!("{}", spread("the launch calls of a step", &mut call));
        eprintln!("{}", spread("the pass of a step", &mut walls));
        if cycles == 3 {
            eprintln!("\n### By variant\n");
            eprintln!(
                "{}",
                profile::table(&profile::rows(&all, profile::by_variant), steps, rate)
            );
            let times = times_of(&block);
            let rate = profile::ticks_per_ns(&block);
            eprintln!("\n## The first block of 512 tokens at {cycles} cycles (one run)\n");
            eprintln!(
                "{}",
                profile::table(&profile::rows(&times.concat(), profile::by_op), 1, rate)
            );
        }
    }
    // The spread of a prefill block of 512 tokens from position 0 at 3 cycles.
    let ids = prose(DEFAULT_MAX_BLOCK);
    let mut blocks = Vec::new();
    let mut walls = Vec::new();
    let mut kernel = Vec::new();
    for _ in 0..5 {
        card::clear(&mut engine, 0);
        engine.take_profile();
        let started = std::time::Instant::now();
        pass(&mut engine, 0, &ids, 3);
        walls.push(started.elapsed().as_secs_f64() * 1e3);
        let launches = engine.take_profile();
        let rate = profile::ticks_per_ns(&launches);
        kernel.push(times_of(&launches).iter().map(|t| covered(t)).sum::<u64>() as f64 / rate / 1e6);
        blocks.extend(launches);
    }
    let rate = profile::ticks_per_ns(&blocks);
    let times = times_of(&blocks);
    eprintln!("\n## A prefill block of 512 tokens from position 0 at 3 cycles, mean of 5\n");
    eprintln!(
        "{}",
        profile::table(&profile::rows(&times.concat(), profile::by_op), 5, rate)
    );
    eprintln!("\n### By variant\n");
    eprintln!(
        "{}",
        profile::table(&profile::rows(&times.concat(), profile::by_variant), 5, rate)
    );
    eprintln!("{}", spread("the tasks of a block", &mut kernel));
    eprintln!("{}", spread("the pass of a block", &mut walls));
}

/// The tokens each second of a batch of 1, 2, 4 and 8 decode rows at 3 cycles, each row at about position 1,024 of
/// its own slot (milestone 36 item 12). Then the profile of a step of 8 rows. The rows of a pass share the fixed costs
/// of a step: the weights of the dense ops, the head and the task boundaries.
#[test]
fn the_speed_of_a_batch_of_decode_rows() {
    if absent() {
        return;
    }
    const ROWS: [usize; 4] = [1, 2, 4, 8];
    let steps = 16;
    // Eight slots with the reserve of 48 do not fit the card, so this test runs at most 3 cycles.
    let config = || Config {
        slots: vec![2_048; 8],
        extra_runs: 24,
        ..config()
    };
    let cards = nagaya_mk_bmg::test_cards::card();
    // Feed each slot a prefix of 1,024 positions, then run `steps` steps of `rows` rows after one step that warms the
    // card. Give the wall time of each step in ms, and the launches of each.
    let measure = |engine: &mut Inkling2Engine, rows: usize, at: &mut [usize; 8], next: &mut [u32; 8]| {
        let mut out = Vec::new();
        for k in 0..=steps {
            let segments: Vec<nagaya_abi::Segment> = (0..rows)
                .map(|s| {
                    let mut seg = card::segment(s as u32, at[s] as u64, &[next[s]], &[at[s] as u64]);
                    seg.cycles = Some(3);
                    seg
                })
                .collect();
            let started = std::time::Instant::now();
            let got = engine
                .pass(&segments, &[])
                .unwrap_or_else(|e| panic!("a batch of {rows}: {}", e.message));
            let wall = started.elapsed().as_secs_f64() * 1e3;
            for s in 0..rows {
                next[s] = argmax(&card::floats(&got.outputs[s].bytes));
                at[s] += 1;
            }
            let launches = engine.take_profile();
            if k > 0 {
                out.push((wall, launches));
            }
        }
        out
    };
    let prefixed = |engine: &mut Inkling2Engine| -> ([usize; 8], [u32; 8]) {
        let mut next = [0u32; 8];
        for (s, first) in next.iter_mut().enumerate() {
            card::clear(engine, s as u32);
            let ids: Vec<u32> = prose(PREFIX + 7 * s).split_off(7 * s);
            let mut logits = Vec::new();
            for at in (0..PREFIX).step_by(DEFAULT_MAX_BLOCK) {
                let last = (at + DEFAULT_MAX_BLOCK - 1) as u64;
                let mut seg = card::segment(s as u32, at as u64, &ids[at..at + DEFAULT_MAX_BLOCK], &[last]);
                seg.cycles = Some(3);
                logits = card::floats(&engine.pass(&[seg], &[]).unwrap().outputs[0].bytes);
            }
            *first = argmax(&logits);
        }
        engine.take_profile();
        ([PREFIX; 8], next)
    };

    let mut engine = Inkling2Engine::open(config(), &cards).unwrap_or_else(|e| panic!("the open failed: {e}"));
    let (mut at, mut next) = prefixed(&mut engine);
    eprintln!(
        "\n## A batch of decode rows at 3 cycles, each row near position {PREFIX}, the median of {steps} steps\n"
    );
    eprintln!("| rows | step ms | tokens/s of a row | tokens/s of the batch |");
    eprintln!("|---:|---:|---:|---:|");
    for rows in ROWS {
        let mut walls: Vec<f64> = measure(&mut engine, rows, &mut at, &mut next)
            .into_iter()
            .map(|(w, _)| w)
            .collect();
        walls.sort_by(f64::total_cmp);
        let ms = walls[walls.len() / 2];
        eprintln!("| {rows} | {ms:.1} | {:.1} | {:.1} |", 1e3 / ms, rows as f64 * 1e3 / ms);
    }
    drop(engine);

    let mut engine = Inkling2Engine::open_profiled(config(), &cards).unwrap_or_else(|e| panic!("the open failed: {e}"));
    let (mut at, mut next) = prefixed(&mut engine);
    for rows in [1usize, 8] {
        let measured = measure(&mut engine, rows, &mut at, &mut next);
        let per_step = measured[0].1.len();
        let launches: Vec<Launch> = measured.into_iter().flat_map(|(_, l)| l).collect();
        let times = times_of(&launches);
        let rate = profile::ticks_per_ns(&launches);
        let all: Vec<TaskTime> = times.concat();
        eprintln!(
            "\n## A step of {rows} decode rows near position {PREFIX} at 3 cycles: {} tasks in {per_step} launches\n",
            all.len() / steps
        );
        eprintln!("{}", profile::table(&profile::rows(&all, profile::by_op), steps, rate));
        let mut kernel: Vec<f64> = times
            .chunks(per_step)
            .map(|step| step.iter().map(|t| covered(t)).sum::<u64>() as f64 / rate / 1e6)
            .collect();
        eprintln!("{}", spread("the tasks of a step", &mut kernel));
    }
}
