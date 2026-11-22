//! The feed of the Inkling 2 engine on a B70 card (milestone 35 step 5, engine-abi "The first Inkling 2 binding",
//! "Bits and splits"): the same split gives the same bits, another split and a decode step stay near the block, the
//! cuts between launches change no bit, and a slot of 32,768 positions follows the host form where its rings wrap.
//! It also measures the time of a prefill of 512 tokens and of a decode step.
//!
//! Each gap prints. A test skips when the model or the oracle file is absent. Release the card first, and run alone:
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --features xe --test xe_feed -- --test-threads=1
//! --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card;
use nagaya_mk_inkling2_bmg::host::forward::{Context, Model};
use nagaya_mk_inkling2_bmg::model::LAUNCH_TASKS;
use nagaya_mk_inkling2_bmg::spec::{DEFAULT_MAX_BLOCK, SLOT_TOKENS};

/// The largest gap of a logit between two splits of the same tokens, or a decode step and a block. The card run of
/// 2026-11-21 measured 9.1e-6 for a decode step, and the splits gave the same bits.
const SPLIT_TOL: f64 = 1e-4;
/// The largest gap of a logit from the host form, past the end of the rings. The card run of 2026-11-21 measured
/// 1.5e-5 at most.
const HOST_TOL: f64 = 1e-4;
/// The host form runs the positions after this one from the state of the card at it, and the positions before it
/// from position 0. The spill split ends a block here.
const RESUME: usize = 1_536;

/// The ids of the prompt "longer" of the oracle, 637 tokens of prose.
fn longer() -> Vec<u32> {
    let oracle = common::golden_json("inkling2-mini.oracle.json");
    let prompt = oracle["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "longer")
        .expect("the oracle holds the prompt longer")
        .clone();
    prompt["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

fn absent() -> bool {
    common::model_absent(Which::Mini) || common::golden_absent("inkling2-mini.oracle.json")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn argmax(v: &[f32]) -> u32 {
    (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap() as u32
}

/// The same split gives the same bits, another split stays near it, a decode step stays near the block, and the
/// cuts between launches change no bit. Then the times of a prefill of 512 tokens and of a decode step.
#[test]
fn the_same_split_gives_the_same_bits_and_a_cut_changes_none() {
    if absent() {
        return;
    }
    let ids = longer();
    let n = ids.len();
    let mut engine = card::open(Which::Mini, vec![SLOT_TOKENS], DEFAULT_MAX_BLOCK);
    let feed = |engine: &mut nagaya_mk_inkling2_bmg::Inkling2Engine, split: &[usize]| {
        let logits = card::feed_split(engine, 0, 0, &ids, split);
        card::clear(engine, 0);
        logits
    };
    let whole = feed(&mut engine, &[512, n - 512]);
    let again = feed(&mut engine, &[512, n - 512]);
    assert!(
        bits(&whole) == bits(&again),
        "the same split gave other bits on a second run"
    );
    for split in [vec![300, n - 300], vec![100, 412, n - 512]] {
        let other = feed(&mut engine, &split);
        let (count, gap) = card::other_bits(&whole, &other);
        eprintln!(
            "the split {split:?} against [512, {}]: {count} logits of other bits, largest gap {gap:.3e}",
            n - 512
        );
        assert!(gap < SPLIT_TOL, "the split {split:?}: a logit differs by {gap:.3e}");
        assert_eq!(
            argmax(&other),
            argmax(&whole),
            "the split {split:?}: the argmax differs"
        );
    }
    // The last position as a decode step after a block of the others.
    let step = feed(&mut engine, &[512, n - 513, 1]);
    let (count, gap) = card::other_bits(&whole, &step);
    eprintln!("a decode step against its position in a block: {count} logits of other bits, largest gap {gap:.3e}");
    assert!(gap < SPLIT_TOL, "a decode step differs from the block by {gap:.3e}");
    // Launches of one or two runs give the bits of one launch for the pass.
    card::feed_split(&mut engine, 0, 0, &ids[..1], &[1]);
    assert_eq!(engine.launches(), 1, "a pass of 3 cycles fits one launch");
    card::clear(&mut engine, 0);
    engine.set_launch_tasks(60);
    let cut = feed(&mut engine, &[512, n - 512]);
    let launches = engine.launches();
    engine.set_launch_tasks(LAUNCH_TASKS);
    eprintln!(
        "a pass cut into {launches} launches gives the bits of one launch: {}",
        bits(&cut) == bits(&whole)
    );
    assert!(
        launches >= 24,
        "a launch of 60 tasks holds one or two runs, and the pass took {launches} launches"
    );
    assert!(bits(&cut) == bits(&whole), "the cuts between launches changed bits");

    // The time of a prefill of 512 tokens from position 0, and of a decode step after 637 positions.
    let mut prefill = Vec::new();
    for _ in 0..5 {
        let started = std::time::Instant::now();
        card::pass(&mut engine, 0, 0, &ids[..512], &[511]);
        prefill.push(started.elapsed().as_secs_f64() * 1e3);
        card::clear(&mut engine, 0);
    }
    prefill.sort_by(f64::total_cmp);
    let mut next = argmax(&card::feed_split(&mut engine, 0, 0, &ids, &[512, n - 512]));
    let steps = 32;
    let started = std::time::Instant::now();
    for k in 0..steps {
        let pos = (n + k) as u64;
        next = argmax(&card::pass(&mut engine, 0, pos, &[next], &[pos]).remove(0));
    }
    let decode = started.elapsed().as_secs_f64() * 1e3 / steps as f64;
    card::clear(&mut engine, 0);
    eprintln!(
        "times: a prefill of 512 tokens takes {:.0} ms (median of 5, {:.0} to {:.0}); a decode step at positions {n} \
         to {} takes {decode:.1} ms ({:.1} tokens each second)",
        prefill[2],
        prefill[0],
        prefill[4],
        n + steps - 1,
        1e3 / decode
    );
}

/// 2,200 token ids from a seeded generator, below 150,000. The prose of "longer", repeated, makes a route of the
/// experts near a tie: at position 1,295 the eighth and the ninth key of the router of layer 11 differ by 4.4e-6 in
/// the host form, the card picks the other expert, and the logits after it differ by up to 1.3. Ids with no repeat
/// had no such flip in the card run of 2026-11-21 (engine-abi "Bits and splits").
fn random_ids() -> Vec<u32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..2200)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % 150_000) as u32
        })
        .collect()
}

/// The logits of the positions `wanted` when `ids` go into `slot` in passes of the lengths of `split`.
fn rows_of(
    engine: &mut nagaya_mk_inkling2_bmg::Inkling2Engine,
    slot: u32,
    ids: &[u32],
    split: &[usize],
    wanted: &[usize],
) -> Vec<Vec<f32>> {
    let mut got = Vec::new();
    let mut at = 0;
    for n in split {
        let rows: Vec<u64> = wanted
            .iter()
            .filter(|p| (at..at + n).contains(*p))
            .map(|p| *p as u64)
            .collect();
        got.extend(card::pass(engine, slot, at as u64, &ids[at..at + n], &rows));
        at += n;
    }
    got
}

/// A slot of 32,768 positions past the end of its rings: 2,200 positions, then 3 decode steps, against the host form.
/// A block across the end of a ring gives the bits of a split with no spill. A slot of 2,048 positions gives the bits
/// of the long slot.
#[test]
fn a_slot_of_32768_follows_the_host_form_where_its_rings_wrap() {
    if absent() {
        return;
    }
    let ids = random_ids();
    // Positions 1,936 to 2,199 cross the end of the ring at 2,048, so 152 rows spill. The other split ends a block at
    // 2,048, so nothing spills.
    let spill = [512, 512, 512, 400, 264];
    let no_spill = [512, 512, 512, 512, 152];
    let wanted = [
        300, 1000, 1023, 1024, 1600, 1935, 2000, 2047, 2048, 2049, 2060, 2100, 2150, 2190, 2199,
    ];
    let mut engine = card::open(Which::Mini, vec![2048, SLOT_TOKENS], DEFAULT_MAX_BLOCK);
    let short = card::feed_split(&mut engine, 0, 0, &ids[..1000], &[512, 488]);
    let long = card::feed_split(&mut engine, 1, 0, &ids[..1000], &[512, 488]);
    assert!(
        bits(&short) == bits(&long),
        "a slot of 2,048 positions and a slot of 32,768 give other bits"
    );
    card::clear(&mut engine, 0);
    card::clear(&mut engine, 1);
    let plain = rows_of(&mut engine, 1, &ids, &no_spill, &wanted);
    card::clear(&mut engine, 1);
    let got = rows_of(&mut engine, 1, &ids, &spill, &wanted);
    for (p, (a, b)) in wanted.iter().zip(got.iter().zip(&plain)) {
        assert!(
            bits(a) == bits(b),
            "position {p}: the split with a spill gives other bits"
        );
    }
    let mut got: Vec<(usize, Vec<f32>)> = wanted.iter().copied().zip(got).collect();
    let mut tokens = Vec::new();
    for k in 0..3 {
        let token = argmax(&got.last().unwrap().1);
        tokens.push(token);
        let pos = ids.len() + k;
        got.push((
            pos,
            card::pass(&mut engine, 1, pos as u64, &[token], &[pos as u64]).remove(0),
        ));
    }
    // At position 1,527 a route is at a near tie, and the card picks the other expert (`tests/xe_long.rs`, engine-abi
    // "A route near a tie"). So the host form runs the blocks after 1,536 from the state of the card at 1,536.
    card::clear(&mut engine, 1);
    card::feed_split(&mut engine, 1, 0, &ids[..RESUME], &[512, 512, 512]);
    let mut resumed = Some(card::host_context(&mut engine, 1, RESUME));
    drop(engine);
    let model = Model::open(&common::model_weights(Which::Mini)).unwrap();
    let mut ctx = Context::new(3);
    let started = std::time::Instant::now();
    let mut at = 0;
    let mut want = Vec::new();
    for n in spill {
        if at == RESUME {
            ctx = resumed.take().expect("one block starts at the resume point");
        }
        let rows: Vec<usize> = wanted
            .iter()
            .filter(|p| (at..at + n).contains(*p))
            .map(|p| p - at)
            .collect();
        want.extend(model.forward(&mut ctx, &ids[at..at + n], &rows, None));
        at += n;
    }
    for token in &tokens {
        want.push(model.forward(&mut ctx, &[*token], &[0], None).remove(0));
    }
    eprintln!(
        "the host form of 2,203 positions took {:.0} s",
        started.elapsed().as_secs_f64()
    );
    let mut worst = 0f64;
    for ((p, g), w) in got.iter().zip(&want) {
        let gap = common::max_abs(g, w);
        eprintln!("position {p}: largest gap {gap:.3e} from the host form");
        assert_eq!(argmax(g), argmax(w), "position {p}: the argmax differs");
        worst = worst.max(gap);
    }
    eprintln!(
        "the ring: worst gap {worst:.3e} from the host form over {} positions",
        got.len()
    );
    assert!(worst < HOST_TOL, "a logit differs from the host form by {worst:.3e}");
}
