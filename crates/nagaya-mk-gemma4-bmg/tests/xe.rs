//! The `Engine` trait of the Gemma 4 engine on the two B70 cards: the bits of the copy, the conformance suite, the
//! ring parts of a long slot, and the refusals.
//!
//! The tests share one `Loaded`: the weights of one engine fill most of both cards. Run with `--test-threads=1`.
#![cfg(feature = "xe")]

mod common;

use std::sync::mpsc;
use std::sync::{Arc, OnceLock};

use nagaya_abi::{
    Engine, OutputsWanted, PartKind, PassErrorKind, Residual, Segment, SegmentInputs, StageHandle, StageOut, State,
    WriteResult,
};
use nagaya_conformance::run_all;
use nagaya_mk_gemma4_bmg::engine::{Gemma4, Loaded, open};
use nagaya_mk_gemma4_bmg::gemma4::{HIDDEN, KV_RING_ROWS};
use nagaya_mk_gemma4_bmg::spec::{self, Config};
use nagaya_store::{EngineFacts, Put, Reason, Restored, Store, Tier, recipe_root, snapshot_from_engine};
use sha2::{Digest, Sha256};

/// The hashes of the logits for the plan of `copy_plan`. The copy of mkviolet gave the first hashes, before the trait
/// (`1d9e1b6`). The split attention of 2026-11-06 changed them, because its count of key segments reads no slot size
/// now (engine-abi "The bits do not depend on the slot size"). The golden tests against transformers passed after it.
const COPY_PREFILL: &str = "7e59fa12bec4367c40e949607ce847beffc697173bf3219953301573a0f22fb8";
const COPY_DECODE: &str = "a57cd4bcb32ccfafb02488aaa697dffeaf6250314ee53acf37eae7440364cee3";
const COPY_BRANCHES: &str = "aa890327c6262a0aa41604c8e9f03119b3f05beb0f36bd5cdb70a573f78bb4d3";

/// The PCI addresses of the `xe` cards, in PCI order: the order of the machine device ids (spec 2.1).
fn cards() -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir("/sys/bus/pci/drivers/xe")
        .expect("the xe driver directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.len() == 12 && name.as_bytes()[4] == b':')
        .collect();
    found.sort();
    found
}

static LOADED: OnceLock<Arc<Loaded>> = OnceLock::new();

fn loaded() -> Arc<Loaded> {
    LOADED
        .get_or_init(|| Loaded::load(&common::model_dir(), &cards()).expect("load the weights on both cards"))
        .clone()
}

fn config(slots: Vec<usize>) -> Config {
    Config {
        model_dir: common::model_dir(),
        slots,
        max_block: 512,
    }
}

fn engine(slots: Vec<usize>) -> Gemma4 {
    open(&loaded(), &config(slots)).expect("open an engine on the loaded weights")
}

fn plain(slot: u32, from: u64, tokens: &[u32], logits: bool) -> Segment {
    let to = from + tokens.len() as u64;
    Segment {
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..to).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: if logits { vec![to - 1] } else { Vec::new() },
        },
    }
}

/// The branch block of `Gemma4Engine::branch_block` as one segment with `meta`.
fn branch_segment(slot: u32, pos0: u64, shared: &[u32], branches: &[Vec<u32>]) -> Segment {
    let shared_end = pos0 as u32 + shared.len() as u32;
    let mut tokens = shared.to_vec();
    let mut meta: Vec<[u32; 4]> = (0..shared.len()).map(|i| [pos0 as u32 + i as u32, 0, 0, 0]).collect();
    let mut rows = Vec::new();
    for branch in branches {
        let first_row = pos0 as u32 + tokens.len() as u32;
        for (j, token) in branch.iter().enumerate() {
            tokens.push(*token);
            meta.push([shared_end + j as u32, shared_end, first_row, 0]);
        }
        rows.push(pos0 + tokens.len() as u64 - 1);
    }
    Segment {
        state: State::Slot(slot),
        positions: meta.iter().map(|m| u64::from(m[0])).collect(),
        meta: meta.iter().flatten().flat_map(|v| v.to_le_bytes()).collect(),
        inputs: SegmentInputs::Tokens(tokens),
        outputs: OutputsWanted { logit_rows: rows },
    }
}

fn hash(rows: &[Vec<u8>]) -> String {
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update(row);
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn prompt() -> Vec<u32> {
    (0..1300u32).map(|i| (1000 + i * 7919) % 200_000).collect()
}

/// The passes of the bits plan, each with the index of its group: 0 prefill, 1 decode, 2 branches. The prefill goes
/// in blocks of 512, as `forward_blocks` fed it.
fn copy_plan() -> Vec<(usize, Segment)> {
    let prompt = prompt();
    let mut plan = Vec::new();
    let chunks: Vec<&[u32]> = prompt.chunks(512).collect();
    for (index, chunk) in chunks.iter().enumerate() {
        plan.push((0, plain(0, index as u64 * 512, chunk, index + 1 == chunks.len())));
    }
    for i in 0..4u32 {
        plan.push((1, plain(0, 1300 + u64::from(i), &[5000 + i], true)));
    }
    let shared: Vec<u32> = (0..200u32).map(|i| (77 + i * 104_729) % 200_000).collect();
    plan.push((2, branch_segment(0, 0, &shared, &[vec![11, 12, 13], vec![21, 22]])));
    plan
}

fn check_copy_hashes(groups: &[Vec<Vec<u8>>; 3], path: &str) {
    println!(
        "{path}: prefill {}, decode {}, branches {}",
        hash(&groups[0]),
        hash(&groups[1]),
        hash(&groups[2])
    );
    assert_eq!(
        hash(&groups[0]),
        COPY_PREFILL,
        "{path}: the prefill logits differ from the copy"
    );
    assert_eq!(
        hash(&groups[1]),
        COPY_DECODE,
        "{path}: the decode logits differ from the copy"
    );
    assert_eq!(
        hash(&groups[2]),
        COPY_BRANCHES,
        "{path}: the branch logits differ from the copy"
    );
}

/// The two stage threads of a group, as the nagaya runs them: stage 0 runs a block while stage 1 runs the block before
/// it, with two residual buffers (engine-abi D1).
struct Pipeline {
    to_first: mpsc::Sender<Arc<Vec<Segment>>>,
    outputs: mpsc::Receiver<Result<StageOut, String>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Pipeline {
    fn start(mut handles: Vec<Box<dyn StageHandle + Send>>, rows: usize) -> Self {
        assert_eq!(handles.len(), 2);
        let mut last = handles.pop().expect("stage 1");
        let mut first = handles.pop().expect("stage 0");
        let (to_first, first_rx) = mpsc::channel::<Arc<Vec<Segment>>>();
        let (to_last, last_rx) = mpsc::channel::<(Arc<Vec<Segment>>, Residual)>();
        let (free_tx, free_rx) = mpsc::channel::<Residual>();
        for _ in 0..2 {
            free_tx.send(Residual::new(HIDDEN * 4, rows)).expect("a free buffer");
        }
        let (out_tx, outputs) = mpsc::channel();
        let out_first = out_tx.clone();
        let free_back = free_tx.clone();
        let t0 = std::thread::spawn(move || {
            for segments in first_rx {
                let mut residual = free_rx.recv().expect("a free residual buffer");
                match first.stage_pass(&segments, &mut residual) {
                    Ok(StageOut::Residual { .. }) => to_last.send((segments, residual)).expect("stage 1 runs"),
                    Ok(other) => panic!("stage 0 gave {other:?}"),
                    Err(error) => {
                        out_first.send(Err(error.message)).expect("the test waits");
                        free_back.send(residual).expect("the buffer goes back");
                    }
                }
            }
        });
        let t1 = std::thread::spawn(move || {
            for (segments, mut residual) in last_rx {
                let out = last.stage_pass(&segments, &mut residual).map_err(|error| error.message);
                free_tx.send(residual).expect("stage 0 takes the buffer back");
                out_tx.send(out).expect("the test waits");
            }
        });
        Self {
            to_first,
            outputs,
            threads: vec![t0, t1],
        }
    }

    fn submit(&self, segments: Vec<Segment>) {
        self.to_first.send(Arc::new(segments)).expect("stage 0 runs");
    }

    fn next(&self) -> Result<StageOut, String> {
        self.outputs.recv().expect("a stage answers")
    }

    fn stop(self) {
        drop(self.to_first);
        for thread in self.threads {
            thread.join().expect("a stage thread panicked");
        }
    }
}

fn logits_of(out: StageOut) -> Vec<Vec<u8>> {
    match out {
        StageOut::Last(out) => out.outputs.into_iter().map(|output| output.bytes).collect(),
        other => panic!("the last stage gave {other:?}"),
    }
}

/// `Engine::pass` and the stage threads give the logits of the copy, bit for bit, for one plan of prefill blocks,
/// decode steps and a branch block (milestone 15s step 5: same block sizes give the same bits).
///
/// Falsified by a stage 0 that copies one row less of the residual, and by a branch block without its meta.
#[test]
fn the_trait_gives_the_bits_of_the_copy() {
    if common::absent(&[]) {
        return;
    }
    let plan = copy_plan();
    let mut whole = engine(vec![2048, 2048]);
    let mut groups: [Vec<Vec<u8>>; 3] = Default::default();
    for (group, segment) in &plan {
        let out = whole
            .pass(std::slice::from_ref(segment), &[])
            .expect("a pass of the plan");
        groups[*group].extend(out.outputs.into_iter().map(|output| output.bytes));
    }
    check_copy_hashes(&groups, "Engine::pass");
    assert_eq!(whole.slot_n(0, 0), Some(0), "a branch block holds no row");

    let mut staged = engine(vec![2048, 2048]);
    let handles = staged.stages().expect("stages").take_handles();
    assert!(
        staged.pass(std::slice::from_ref(&plan[0].1), &[]).is_err(),
        "a pass after take_handles"
    );
    let pipeline = Pipeline::start(handles, 512);
    // A block of the prefill does not wait for the one before it: the two cards overlap.
    for (_, segment) in plan.iter().filter(|(group, _)| *group == 0) {
        pipeline.submit(vec![segment.clone()]);
    }
    let mut groups: [Vec<Vec<u8>>; 3] = Default::default();
    for _ in plan.iter().filter(|(group, _)| *group == 0) {
        groups[0].extend(logits_of(pipeline.next().expect("a prefill block")));
    }
    for (group, segment) in plan.iter().filter(|(group, _)| *group != 0) {
        pipeline.submit(vec![segment.clone()]);
        groups[*group].extend(logits_of(pipeline.next().expect("a pass")));
    }
    pipeline.stop();
    check_copy_hashes(&groups, "the stage threads");
}

/// The conformance suite of spec 8.3 on the two cards, with the gates of the stage path and the feed rule.
#[test]
fn the_conformance_suite_passes_on_the_cards() {
    if common::absent(&[]) {
        return;
    }
    let suite = run_all(&mut || engine(vec![2048, 2048]));
    println!("{suite}");
    assert!(suite.passed(), "{suite}");
    let names: Vec<&str> = suite.not_applicable().map(|gate| gate.name).collect();
    // The engine has no seal and declares no segments_exact (engine-abi D11 rule 4).
    assert_eq!(
        names,
        vec![
            "seal_only_at_boundaries",
            "quantum_same_boundaries",
            "segments_exact_bits"
        ],
        "{suite}"
    );
}

/// In a long slot, a ring part holds the last 4,096 positions and a global part holds all (per-part coverage). A park
/// through the store and a restore into a fresh engine give the bits of the slot that went on.
///
/// Falsified by a coverage that gives every ring part a base of 0, and by a restore that writes a ring from row 0.
#[test]
fn a_long_slot_parks_and_restores_through_the_store() {
    if common::absent(&[]) {
        return;
    }
    let tokens: Vec<u32> = (0..6000u32).map(|i| (300 + i * 7919) % 200_000).collect();
    let mut source = engine(vec![8192]);
    for (index, chunk) in tokens.chunks(512).enumerate() {
        source
            .pass(&[plain(0, index as u64 * 512, chunk, false)], &[])
            .expect("a block");
    }
    let limits = source.limits();
    let mut rows = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        let start = source
            .coverage()
            .expect("coverage")
            .coverage(State::Slot(0), id, &[])
            .start;
        let want = if part.kind == PartKind::Ring {
            6000 - KV_RING_ROWS as u64
        } else {
            0
        };
        assert_eq!(start, want, "part {}", part.name);
        rows.push((id, start, 6000 - start));
    }
    let snapshot = snapshot_from_engine(&mut source, State::Slot(0), &rows).expect("a snapshot");
    let dir = tempfile::tempdir().expect("a temporary directory");
    let mut store = Store::open_with_limits(dir.path(), u64::MAX, u64::MAX, u64::MAX, 0).expect("a store");
    let identity = [7u8; 32];
    let id = store
        .put(Put {
            snapshot: &snapshot,
            recipe: recipe_root(&identity, b"", &[], b""),
            previous_recipe: None,
            engine: EngineFacts::of("gemma4", &limits),
            identity,
            owner: 0,
            tier: Tier::Host,
            reason: Reason::Anchor,
            sealed_recipe: None,
            chains_from: None,
            provenance: None,
            now_ms: 0,
        })
        .expect("the store takes the snapshot");
    let next = plain(0, 6000, &[4242, 4243], true);
    let want = source
        .pass(std::slice::from_ref(&next), &[])
        .expect("the source goes on");
    drop(source);
    let mut target = engine(vec![8192]);
    let restored = store
        .restore(&id, &mut target, State::Slot(0), &identity, 0)
        .expect("the restore runs");
    assert_eq!(restored, Restored::Done);
    assert_eq!(target.slot_n(0, 0), Some(6000));
    let got = target
        .pass(std::slice::from_ref(&next), &[])
        .expect("the target goes on");
    assert_eq!(got.outputs, want.outputs, "the restored slot gives other logits");
}

/// One recipe gives one content id (spec 6.7). Thus the same feed gives the same bits in a slot of 4,096 positions and
/// in a slot of 65,536, the two slots of the b70host profile. A block of 64 after 256 keys is the case where
/// `attn_pv_segments` (`nagaya-mk-bmg/kernels/megakernel.cl:486`) can give each slot its own count of segments.
#[test]
fn the_same_feed_gives_the_same_bits_in_each_slot_size() {
    if common::absent(&[]) {
        return;
    }
    let mut engine = engine(vec![4096, 65_536]);
    let prefix: Vec<u32> = (0..256u32).map(|i| (500 + i * 7919) % 200_000).collect();
    let block: Vec<u32> = (0..64u32).map(|i| (900 + i * 104_729) % 200_000).collect();
    let mut rows: [Vec<Vec<u8>>; 2] = Default::default();
    for (slot, rows) in rows.iter_mut().enumerate() {
        let slot = slot as u32;
        let mut segments = vec![plain(slot, 0, &prefix, true), plain(slot, 256, &block, true)];
        // A block gives at most 16 logit rows: the last 16 of the 64.
        segments[1].outputs.logit_rows = (304..320).collect();
        segments.push(plain(slot, 320, &[4242], true));
        for segment in &segments {
            let out = engine.pass(std::slice::from_ref(segment), &[]).expect("a pass");
            rows.extend(out.outputs.into_iter().map(|output| output.bytes));
        }
    }
    let differ: Vec<usize> = (0..rows[0].len()).filter(|&i| rows[0][i] != rows[1][i]).collect();
    assert!(
        differ.is_empty(),
        "the slots of 4,096 and 65,536 positions give other bits in {} of {} logit rows (first {:?})",
        differ.len(),
        rows[0].len(),
        differ.first()
    );
}

/// A write that mkviolet asserts on comes back refused, and a refused pass or write changes nothing.
///
/// Falsified by a cells write that takes a ring's rows at a later position in a short slot.
#[test]
fn a_bad_write_or_pass_is_refused_and_changes_nothing() {
    if common::absent(&[]) {
        return;
    }
    let mut engine = engine(vec![2048, 2048]);
    let out = engine.pass(&[plain(0, 0, &[1, 2, 3], true)], &[]).expect("a pass");
    let parts = engine.parts().expect("parts");
    let row = 8 * 256 * 4;
    // A ring of a short slot holds every position from 0 (`load_kv_layer` asserted on this).
    assert_eq!(
        parts.cells_write(State::Slot(1), 0, 5, 1, &vec![0u8; row]),
        WriteResult::Refused
    );
    assert_eq!(
        parts.cells_write(State::Slot(1), 0, 2048, 1, &vec![0u8; row]),
        WriteResult::Refused
    );
    assert_eq!(
        parts.cells_write(State::Slot(9), 0, 0, 1, &vec![0u8; row]),
        WriteResult::Refused
    );
    assert_eq!(parts.part_write(State::Slot(0), 0, &[]), WriteResult::Refused);
    assert_eq!(parts.set_n(State::Slot(0), 4), WriteResult::Refused, "no rows past 3");
    let gap = engine.pass(&[plain(0, 4, &[1], true)], &[]).unwrap_err();
    assert_eq!(gap.kind, PassErrorKind::Refused);
    let two = engine
        .pass(&[plain(0, 3, &[1], true), plain(1, 0, &[1], true)], &[])
        .unwrap_err();
    assert_eq!(two.kind, PassErrorKind::Refused);
    // The slot is as it was: the same three tokens again give the same logits.
    let again = engine.pass(&[plain(0, 0, &[1, 2, 3], true)], &[]).expect("a pass");
    assert_eq!(again.outputs, out.outputs);
}

/// A PCI address that no card has refuses the load, before any upload.
#[test]
fn a_card_address_that_no_card_has_is_refused() {
    if common::absent(&[]) {
        return;
    }
    let mut wrong = cards();
    wrong[1] = "0000:80:00.0".to_owned();
    let Err(error) = Loaded::load(&common::model_dir(), &wrong) else {
        panic!("the load took a card that is not there");
    };
    assert!(error.contains("0000:80:00.0"), "{error}");
}

/// The identity names the family, the template and the tokenizer of the checkpoint, and the kernel binary.
#[test]
fn the_identity_names_the_family_and_the_binary() {
    if common::absent(&[]) {
        return;
    }
    let engine = engine(vec![2048]);
    let identity = engine.identity();
    assert_eq!(identity.family, "gemma4");
    assert_eq!(identity.model, spec::MODEL);
    assert_eq!(identity.template, nagaya_text_gemma4::CHAT_TEMPLATE);
    let family = nagaya_text_gemma4::Gemma4Family::from_dir(&common::model_dir()).expect("the family library");
    assert_eq!(identity.tokenizer_hash, family.tokenizer_hash());
    assert!(
        identity.checkpoint_digest.starts_with("blake3:"),
        "{}",
        identity.checkpoint_digest
    );
    assert_eq!(identity.checkpoint_digest.len(), "blake3:".len() + 64);
    let kernel = identity
        .numbers
        .get("kernel.megakernel_bmg.bin")
        .expect("the kernel hash");
    assert!(kernel.starts_with("sha256:"), "{kernel}");
    assert!(identity.plan_table.contains("ocloc"), "{}", identity.plan_table);
    assert!(identity.plan_table.contains("8086:"), "{}", identity.plan_table);
    println!("numbers {:?}; plan table {}", identity.numbers, identity.plan_table);
}

fn timed_pass(pipeline: &Pipeline, segment: Segment) -> (Vec<Vec<u8>>, u128) {
    let start = std::time::Instant::now();
    pipeline.submit(vec![segment]);
    let logits = logits_of(pipeline.next().expect("a pass"));
    (logits, start.elapsed().as_micros())
}

fn argmax(bytes: &[u8]) -> u32 {
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    (0..values.len()).fold(0, |best, i| if values[i] > values[best] { i } else { best }) as u32
}

/// Not a check: decode and a prefill of 4096 tokens in blocks of 512 through the two stage threads, as the nagaya runs
/// them. mkviolet's numbers: 46.9 tokens/s for decode, and 1376 ms for the prefill.
/// `cargo test --release -p nagaya-mk-gemma4-bmg --features xe --test xe -- --ignored --nocapture profile_stage_threads`
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_stage_threads() {
    if common::absent(&[]) {
        return;
    }
    let mut engine = engine(vec![4096, 16_384]);
    let pipeline = Pipeline::start(engine.stages().expect("stages").take_handles(), 512);
    let prompt: Vec<u32> = (0..4096u32).map(|i| (1000 + i * 7919) % 200_000).collect();
    for run in 0..3 {
        let start = std::time::Instant::now();
        let chunks: Vec<&[u32]> = prompt.chunks(512).collect();
        for (index, chunk) in chunks.iter().enumerate() {
            pipeline.submit(vec![plain(0, index as u64 * 512, chunk, index + 1 == chunks.len())]);
        }
        for _ in &chunks {
            pipeline.next().expect("a block");
        }
        println!(
            "run {run}: prefill 4096 in blocks of 512: {} ms",
            start.elapsed().as_millis()
        );
    }
    for context in [20usize, 300, 2000, 8000] {
        let prompt: Vec<u32> = (0..context as u32).map(|i| 1000 + i % 5000).collect();
        let chunks: Vec<&[u32]> = prompt.chunks(512).collect();
        for (index, chunk) in chunks.iter().enumerate() {
            pipeline.submit(vec![plain(1, index as u64 * 512, chunk, index + 1 == chunks.len())]);
        }
        let mut logits = Vec::new();
        for _ in &chunks {
            logits = logits_of(pipeline.next().expect("a block"));
        }
        let mut walls = Vec::new();
        for i in 0..32 {
            let next = argmax(&logits[0]);
            let (out, us) = timed_pass(&pipeline, plain(1, (context + i) as u64, &[next], true));
            logits = out;
            walls.push(us);
        }
        walls.sort();
        println!(
            "decode after {context} tokens, median of 32: {} us ({:.1} tokens/s)",
            walls[16],
            1e6 / walls[16] as f64
        );
    }
    pipeline.stop();
}

/// Not a check: the longest stage quantum. Each stage runs blocks of 512 up to the end of a slot of 65,536 positions,
/// and the test prints the slowest block of each stage. `max_quantum_ms` must hold it (D8).
#[test]
#[ignore = "profile: not a check; run it alone"]
fn profile_longest_stage_quantum() {
    if common::absent(&[]) {
        return;
    }
    let mut engine = engine(vec![65_536]);
    let mut handles = engine.stages().expect("stages").take_handles();
    let mut residual = Residual::new(HIDDEN * 4, 512);
    let tokens: Vec<u32> = (0..65_536u32).map(|i| (1000 + i * 7919) % 200_000).collect();
    let mut slowest = [(0u128, 0usize); 2];
    for (index, chunk) in tokens.chunks(512).enumerate() {
        let segment = plain(0, index as u64 * 512, chunk, true);
        for (stage, handle) in handles.iter_mut().enumerate() {
            let start = std::time::Instant::now();
            handle
                .stage_pass(std::slice::from_ref(&segment), &mut residual)
                .expect("a block");
            let us = start.elapsed().as_micros();
            if us > slowest[stage].0 {
                slowest[stage] = (us, index * 512);
            }
        }
        if index % 16 == 15 {
            println!("block at {}: slowest so far {slowest:?}", index * 512);
        }
    }
    println!("slowest stage quanta (us, first position): {slowest:?}");
}
