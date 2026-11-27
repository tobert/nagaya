//! Images in the pass of the Inkling 2 engine on a B70 card (engine-abi "The images of Inkling 2", "The choices of the
//! pass"): the oracle prompt with two images, fed with the rows and the records of the family, against the `f32`
//! oracle of transformers. The gates:
//!
//! - one block of the prompt gives the top-64 logits and the logsumexp of the last position within the tolerance of
//!   the text gates;
//! - the same split twice gives the same bits;
//! - a split inside an image span gives the logits within the same tolerance, and the gap to one block prints;
//! - the tower runs once for each image of a feed, also for an image in several blocks.
//!
//! The test needs the Mini folder and the oracle of the stage `vision` (`tools/inkling2_golden.sh --stages vision`),
//! and it skips when a file is absent. Run it alone, on a card that no engine of the active profile names
//! (`AGENTS.md`, "The engines on the B70 cards"): `NAGAYA_TEST_CARDS=<pci> INKLING2_MODEL_DIR=<folder> cargo test -p
//! nagaya-mk-inkling2-bmg --features xe --release --test xe_images -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use common::Which;
use common::card::{self, Gaps};
use nagaya_abi::{Engine, OutputsWanted, Segment, SegmentInputs, State};
use nagaya_family_inkling2::media::{self, ImageRows};
use nagaya_mk_inkling2_bmg::Inkling2Engine;
use nagaya_mk_inkling2_bmg::media::IMAGE_TOKEN;
use nagaya_mk_inkling2_bmg::spec::{DEFAULT_MAX_BLOCK, SLOT_TOKENS};

const ORACLE: &str = "inkling2-mini.vision.json";
const TENSORS: &str = "inkling2-mini.vision.safetensors";

/// The tolerance of a logit and of a logsumexp: the tolerance of the text gates (`tests/xe_oracle.rs`).
const LOGIT_TOL: f64 = 2e-4;

fn absent() -> bool {
    common::model_absent(Which::Mini) || common::golden_absent(ORACLE) || common::golden_absent(TENSORS)
}

/// The prompt of the oracle: its ids, and the rows and the first position of each image.
struct Prompt {
    ids: Vec<u32>,
    images: Vec<(ImageRows, usize)>,
    oracle: serde_json::Value,
}

impl Prompt {
    /// The ids of the oracle, and the rows of each image from its PNG through the family, as a request gives them.
    fn read() -> Self {
        let tensors = common::golden_tensors(TENSORS);
        let ids = tensors["prompt.ids"].ids();
        let json = common::golden_json(ORACLE);
        let prompt = json["prompts"][0].clone();
        // The first position of each run of placeholders.
        let firsts: Vec<usize> = (0..ids.len())
            .filter(|t| ids[*t] == IMAGE_TOKEN && (*t == 0 || ids[*t - 1] != IMAGE_TOKEN))
            .collect();
        let images: Vec<(ImageRows, usize)> = prompt["images"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&firsts)
            .map(|(image, first)| {
                let png = std::fs::read(common::golden_dir().join(image["png"].as_str().unwrap())).unwrap();
                let rows = media::prepare(&png, "image/png").unwrap();
                let grid: Vec<u64> = image["grid"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap())
                    .collect();
                assert_eq!(
                    [rows.grid.rows, rows.grid.columns],
                    [grid[1] as usize, grid[2] as usize],
                    "the grid of the family is the grid of the oracle"
                );
                (rows, *first)
            })
            .collect();
        assert_eq!(images.len(), 2, "the prompt holds two images");
        let placeholders = ids.iter().filter(|id| **id == IMAGE_TOKEN).count();
        assert_eq!(placeholders, images.iter().map(|(r, _)| r.positions()).sum::<usize>());
        Self {
            ids,
            images,
            oracle: prompt,
        }
    }

    /// The segment of positions `from..to` of slot 0: the tokens, and the record and the rows of each image that the
    /// block holds a position of, whole. The last block gives the logits of the last position.
    fn segment(&self, from: usize, to: usize) -> Segment {
        let tokens = self.ids[from..to].to_vec();
        let (mut meta, mut rows) = (Vec::new(), Vec::new());
        for (image, first) in &self.images {
            if *first < to && first + image.positions() > from {
                meta.extend(image.record(*first));
                rows.extend(image.bytes());
            }
        }
        let inputs = if meta.is_empty() {
            SegmentInputs::Tokens(tokens)
        } else {
            SegmentInputs::TokensAndRows { tokens, rows }
        };
        let last = (to == self.ids.len()).then_some(to as u64 - 1);
        Segment {
            control: Vec::new(),
            state: State::Slot(0),
            inputs,
            positions: (from as u64..to as u64).collect(),
            meta,
            outputs: OutputsWanted {
                logit_rows: last.into_iter().collect(),
            },
            layer_loop: None,
            cycles: None,
        }
    }

    /// Feed the prompt into an empty slot 0 in blocks of the lengths of `split`, and give the logits of the last
    /// position.
    fn feed(&self, engine: &mut Inkling2Engine, split: &[usize]) -> Vec<f32> {
        assert_eq!(
            split.iter().sum::<usize>(),
            self.ids.len(),
            "the split covers the prompt"
        );
        card::clear(engine, 0);
        let mut at = 0;
        let mut last = Vec::new();
        for n in split {
            let segment = self.segment(at, at + n);
            let out = engine
                .pass(&[segment], &[])
                .unwrap_or_else(|e| panic!("the block {at}..{}: {}", at + n, e.message));
            if let Some(row) = out.outputs.first() {
                last = row
                    .bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
            }
            at += n;
        }
        last
    }
}

fn report(name: &str, g: &Gaps) {
    eprintln!(
        "{name}: top-64 logits within {:.3e}, logsumexp within {:.3e}, argmax {}, {} of the top 64 outside",
        g.logit,
        g.logsumexp,
        if g.top1 { "agrees" } else { "DIFFERS" },
        g.outside
    );
}

fn check(name: &str, g: &Gaps) {
    report(name, g);
    assert!(g.top1, "{name}: the argmax differs from the oracle");
    assert!(g.logit < LOGIT_TOL, "{name}: a top-64 logit differs by {:.3e}", g.logit);
    assert!(
        g.logsumexp < LOGIT_TOL,
        "{name}: the logsumexp differs by {:.3e}",
        g.logsumexp
    );
}

#[test]
fn the_prompt_with_two_images_matches_the_oracle_in_each_split() {
    if absent() {
        return;
    }
    let p = Prompt::read();
    let n = p.ids.len();
    let mut engine = card::open(Which::Mini, vec![SLOT_TOKENS], DEFAULT_MAX_BLOCK);
    let (first1, count1) = (p.images[1].1, p.images[1].0.positions());
    eprintln!(
        "the prompt: {n} positions, image 0 at {} of {}, image 1 at {first1} of {count1}",
        p.images[0].1,
        p.images[0].0.positions()
    );

    // One block.
    let runs = engine.tower_runs();
    let started = std::time::Instant::now();
    let one = p.feed(&mut engine, &[n]);
    eprintln!("one block: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    assert_eq!(engine.tower_runs() - runs, 2, "the tower runs once for each image");
    check("one block", &card::gaps(&one, &p.oracle));

    // The same split again gives the same bits.
    let again = p.feed(&mut engine, &[n]);
    let (differ, gap) = card::other_bits(&one, &again);
    eprintln!("one block twice: {differ} values differ, by {gap:.3e} at most");
    assert_eq!(differ, 0, "the same split gives other bits");

    // A cut inside image 1, and blocks of 64, which cut both images.
    let middle = first1 + count1 / 2;
    for split in [vec![middle, n - middle], card::blocks(n, 64)] {
        let runs = engine.tower_runs();
        let got = p.feed(&mut engine, &split);
        let name = format!("the split {split:?}");
        let (differ, gap) = card::other_bits(&one, &got);
        eprintln!("{name}: {differ} values differ from one block, by {gap:.3e} at most");
        check(&name, &card::gaps(&got, &p.oracle));
        // The engine holds the rows of one image: image 0 runs, then image 1 runs once for all its blocks.
        assert_eq!(
            engine.tower_runs() - runs,
            2,
            "{name}: the tower runs once for each image"
        );
        let twice = p.feed(&mut engine, &split);
        assert_eq!(
            card::other_bits(&got, &twice).0,
            0,
            "{name}: the same split gives other bits"
        );
    }

    // A block with a placeholder that no record names changes nothing.
    card::clear(&mut engine, 0);
    let mut stray = p.segment(0, p.images[0].1 + 1);
    stray.meta.clear();
    stray.inputs = SegmentInputs::Tokens(p.ids[..p.images[0].1 + 1].to_vec());
    let e = engine.pass(&[stray], &[]).unwrap_err();
    assert!(e.message.contains("no media record names it"), "{}", e.message);
    assert_eq!(
        engine.slots().unwrap().slot_high(0),
        0,
        "a refused pass changes nothing"
    );
}
