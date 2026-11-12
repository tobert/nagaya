//! The canary of the Qwen3.8 engine as pure host logic: its inputs, its passes and its hash (spec 8.1).
//!
//! The canary runs fixed token ids through the engine and hashes every logits row that comes back. This module holds
//! the ids, the plan of the passes, the driver, the argmax and the hashes. It runs no pass itself: the caller gives the
//! driver a closure that runs one pass. The worker thread gives a closure over its own pass path, so the canary takes
//! the same path as a client. A host test gives a fake.
//!
//! The ids are literal data. The engine crate has no tokenizer, and the fixture is a test file. A host test checks
//! that the ids equal the oracle fixture.
//!
//! # The byte layout of the canary hash
//!
//! The canary hash is the SHA-256 of these bytes, in this order. Each count is a `u32` and each id is an `i32`, both
//! little-endian.
//!
//! 1. The tag `nagaya qwen38 canary v1`: 23 bytes of ASCII, with no length and no terminator.
//! 2. The count of the short ids (5), then the short ids: 4 + 5 * 4 bytes.
//! 3. The count of the long ids (300), then the long ids: 4 + 300 * 4 bytes.
//! 4. The count of the rows (19), then the 19 row hashes in the order of the passes: 4 + 19 * 32 bytes.
//!
//! A row hash is the SHA-256 of the row's `VOCAB` f32 values as little-endian bytes, as the engine gives them. There
//! is no change to a value: `-0.0` and `0.0` give two hashes. The rows come in the order of the plan: the row of
//! context A's pass, the two rows of context B's two passes, then for each of the 8 decode steps the row of A and the
//! row of B.

use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::spec;

/// The text tag that opens the canary hash. A change of the content of the canary changes the version in the tag.
pub const TAG: &[u8] = b"nagaya qwen38 canary v1";

/// The count of greedy decode steps after the prefills.
pub const DECODE_STEPS: usize = 8;

/// The count of logits rows: one for context A, two for context B, and two for each decode step.
pub const ROWS: usize = 1 + 2 + 2 * DECODE_STEPS;

/// The tokens that context B takes in its first pass. It equals the pass limit of the engine.
const LONG_FIRST_PASS: usize = spec::MAX_PASS_TOKENS as usize;

/// The ids of the prompt `plain` of the oracle fixture. Context A takes them in one pass.
pub const SHORT_IDS: [i32; 5] = [760, 6511, 314, 9338, 369];

/// The first 300 ids of the prompt `longer` of the oracle fixture. Context B takes them in two passes.
pub const LONG_IDS: [i32; 300] = [
    760, 37714, 24279, 12965, 506, 279, 809, 314, 279, 1500, 11, 1332, 279, 13518, 8145, 680, 1083, 13681, 793, 321,
    279, 15879, 26817, 1881, 279, 5894, 382, 13, 12851, 37119, 279, 47788, 513, 12647, 303, 279, 1248, 10679, 23586,
    11, 321, 279, 1118, 6206, 314, 281, 27155, 21780, 303, 680, 314, 279, 8981, 440, 836, 24007, 14273, 1172, 836,
    11378, 13, 2302, 14255, 836, 53768, 383, 279, 28707, 28428, 279, 5224, 63646, 11, 321, 1754, 855, 11988, 421, 780,
    1765, 9111, 4811, 513, 1332, 551, 2047, 1070, 13, 357, 60692, 23082, 45369, 383, 279, 19233, 5459, 440, 1141, 3807,
    11878, 321, 1141, 26729, 1990, 7948, 11, 321, 264, 7914, 73797, 3055, 1141, 13684, 440, 264, 628, 314, 5415, 11,
    24199, 1396, 9978, 303, 2404, 13, 271, 760, 2169, 1461, 13205, 264, 4331, 539, 279, 5817, 440, 40953, 14973, 364,
    1396, 4560, 303, 279, 18921, 13, 357, 12307, 3254, 421, 279, 4560, 369, 8318, 364, 264, 919, 13, 357, 5144, 3254,
    421, 424, 369, 5354, 364, 279, 5459, 13, 357, 1500, 3254, 421, 279, 2859, 513, 3133, 383, 424, 1381, 13, 1216,
    9967, 82, 680, 279, 14973, 440, 780, 34958, 321, 13514, 1070, 1495, 430, 279, 1834, 5601, 383, 11, 321, 279, 2859,
    32315, 506, 279, 4331, 948, 781, 1440, 11, 430, 66509, 32315, 506, 279, 12515, 13, 35254, 40951, 852, 1118, 22819,
    279, 1785, 11, 321, 18022, 682, 3397, 10660, 310, 2222, 424, 13, 271, 623, 279, 5484, 22002, 279, 324, 572, 287,
    2404, 660, 1834, 11, 321, 279, 6220, 59939, 264, 2545, 1172, 279, 8545, 14663, 13, 357, 2404, 261, 6725, 496, 4405,
    1891, 369, 3154, 264, 491, 69650, 3618, 494, 264, 2424, 314, 38486, 13, 1216, 10633, 424, 440, 1564, 56845, 1396,
    2342, 4276, 321, 20541, 310, 18022, 13, 17539, 562, 1384, 449, 79364, 16724, 704, 264, 13964,
];

/// One of the two contexts of the canary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    /// Context A: the short prompt.
    Short,
    /// Context B: the long prompt.
    Long,
}

/// The part of the canary that a pass belongs to. The report times each part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The pass of context A.
    ShortPrefill,
    /// The two passes of context B.
    LongPrefill,
    /// The 8 joined decode passes.
    Decode,
}

/// One pass of the plan: its part and its segments, each as a context and a token count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedPass {
    /// The part of the canary.
    pub phase: Phase,
    /// The segments, in the order of the pass.
    pub segments: Vec<(Which, usize)>,
}

/// The passes of the canary, in order: 5 tokens of A, then 256 and 44 tokens of B, then 8 passes of 2 segments.
pub fn plan() -> Vec<PlannedPass> {
    let mut passes = vec![
        PlannedPass {
            phase: Phase::ShortPrefill,
            segments: vec![(Which::Short, SHORT_IDS.len())],
        },
        PlannedPass {
            phase: Phase::LongPrefill,
            segments: vec![(Which::Long, LONG_FIRST_PASS)],
        },
        PlannedPass {
            phase: Phase::LongPrefill,
            segments: vec![(Which::Long, LONG_IDS.len() - LONG_FIRST_PASS)],
        },
    ];
    for _ in 0..DECODE_STEPS {
        passes.push(PlannedPass {
            phase: Phase::Decode,
            segments: vec![(Which::Short, 1), (Which::Long, 1)],
        });
    }
    passes
}

/// One pass for the closure of the caller to run. Each pass asks for one logits row for each segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassCall {
    /// The part of the canary.
    pub phase: Phase,
    /// The context of each segment.
    pub contexts: Vec<Which>,
    /// The token count of each segment.
    pub lens: Vec<u32>,
    /// The tokens of all segments, one after the other.
    pub tokens: Vec<i32>,
}

/// What a canary run gives.
#[derive(Clone, Debug)]
pub struct Report {
    /// The hash of each of the 19 rows, in the order of the passes.
    pub row_hashes: Vec<[u8; 32]>,
    /// The canary hash (the byte layout is in the module comment).
    pub canary_hash: [u8; 32],
    /// The argmax of each row of context A, in order: 1 from its prefill and 8 from the decode.
    pub sampled_short: Vec<i32>,
    /// The argmax of each row of context B, in order: 2 from its prefill and 8 from the decode.
    pub sampled_long: Vec<i32>,
    /// The milliseconds of the pass of context A.
    pub short_ms: f64,
    /// The milliseconds of the two passes of context B.
    pub long_ms: f64,
    /// The milliseconds of the 8 decode passes.
    pub decode_ms: f64,
    /// The milliseconds of the whole run.
    pub total_ms: f64,
}

/// The index of the largest value of a logits row. A tie gives the lowest index.
///
/// A value that is not finite is an error, because a NaN or an infinity means that the pass went wrong. `row` is the
/// number of the row in the run, for the message.
pub fn argmax(row: usize, values: &[f32]) -> Result<i32, String> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "row {row} of the canary has {value} at index {index}, which is not finite; the pass went wrong, so \
                 free the engine and restart the process"
            ));
        }
        match best {
            Some((_, top)) if value <= top => {}
            _ => best = Some((index, value)),
        }
    }
    match best {
        Some((index, _)) => i32::try_from(index).map_err(|_| format!("row {row} of the canary is too long")),
        None => Err(format!("row {row} of the canary is empty; the engine gave no logits")),
    }
}

/// The SHA-256 of a row's f32 values as little-endian bytes. No value changes: `-0.0` and `0.0` differ.
pub fn row_hash(values: &[f32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Sha256::digest(&bytes).into()
}

/// The canary hash of the ids and the row hashes. The byte layout is in the module comment.
pub fn canary_hash(short: &[i32], long: &[i32], rows: &[[u8; 32]]) -> [u8; 32] {
    let mut sha = Sha256::new();
    sha.update(TAG);
    for ids in [short, long] {
        sha.update((ids.len() as u32).to_le_bytes());
        for id in ids {
            sha.update(id.to_le_bytes());
        }
    }
    sha.update((rows.len() as u32).to_le_bytes());
    for row in rows {
        sha.update(row);
    }
    sha.finalize().into()
}

/// The 64 lowercase hex digits of a hash.
pub fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Run the canary with `pass`, which runs one pass and returns one logits row of `VOCAB` values for each segment.
///
/// The driver takes the argmax of each row. The next token of a context is the argmax of its last row, and the two
/// contexts never share a token. An error from `pass`, a wrong row count, a wrong row length or a value that is not
/// finite ends the run.
pub fn run(mut pass: impl FnMut(&PassCall) -> Result<Vec<Vec<f32>>, String>) -> Result<Report, String> {
    let started = Instant::now();
    let mut fed = [0usize; 2];
    let mut last: [Option<i32>; 2] = [None, None];
    let mut sampled: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
    let mut row_hashes = Vec::with_capacity(ROWS);
    let mut ms = [0f64; 3];
    for planned in plan() {
        let mut call = PassCall {
            phase: planned.phase,
            contexts: Vec::new(),
            lens: Vec::new(),
            tokens: Vec::new(),
        };
        for &(which, count) in &planned.segments {
            let slot = which as usize;
            let ids: &[i32] = if which == Which::Short { &SHORT_IDS } else { &LONG_IDS };
            call.contexts.push(which);
            call.lens.push(count as u32);
            if planned.phase == Phase::Decode {
                let token = last[slot].ok_or_else(|| "a decode step has no token before its prefill".to_owned())?;
                call.tokens.push(token);
            } else {
                call.tokens.extend_from_slice(&ids[fed[slot]..fed[slot] + count]);
                fed[slot] += count;
            }
        }
        let timer = Instant::now();
        let rows = pass(&call)?;
        let elapsed = timer.elapsed().as_secs_f64() * 1000.0;
        ms[planned.phase as usize] += elapsed;
        if rows.len() != call.contexts.len() {
            return Err(format!(
                "a canary pass of {} segments gave {} logits rows; the engine must give one row for each segment",
                call.contexts.len(),
                rows.len()
            ));
        }
        for (which, row) in call.contexts.iter().zip(&rows) {
            if row.len() != spec::VOCAB as usize {
                return Err(format!(
                    "row {} of the canary has {} values, but the vocabulary has {}",
                    row_hashes.len(),
                    row.len(),
                    spec::VOCAB
                ));
            }
            let token = argmax(row_hashes.len(), row)?;
            row_hashes.push(row_hash(row));
            last[*which as usize] = Some(token);
            sampled[*which as usize].push(token);
        }
    }
    if row_hashes.len() != ROWS {
        return Err(format!(
            "the canary made {} rows, but the plan has {ROWS}",
            row_hashes.len()
        ));
    }
    let [sampled_short, sampled_long] = sampled;
    Ok(Report {
        canary_hash: canary_hash(&SHORT_IDS, &LONG_IDS, &row_hashes),
        row_hashes,
        sampled_short,
        sampled_long,
        short_ms: ms[Phase::ShortPrefill as usize],
        long_ms: ms[Phase::LongPrefill as usize],
        decode_ms: ms[Phase::Decode as usize],
        total_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row of `VOCAB` values with one peak at `at`.
    fn peak(at: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; spec::VOCAB as usize];
        row[at] = 1.0;
        row
    }

    fn unhex(text: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).unwrap();
        }
        out
    }

    // Python, with the standard library only:
    //   import hashlib, struct
    //   hashlib.sha256(struct.pack('<4f', 1.0, -2.5, 0.0, 0.00325)).hexdigest()
    //   hashlib.sha256(struct.pack('<4f', 0.5, 0.25, -0.0, 7.0)).hexdigest()
    #[test]
    fn a_row_hash_equals_python_sha256_of_the_little_endian_values() {
        assert_eq!(
            hex(&row_hash(&[1.0, -2.5, 0.0, 0.00325])),
            "38e918286b2706cbcd8ef6d4e0a3e2154507f98dea529e6c3d3af7017617d935"
        );
        assert_eq!(
            hex(&row_hash(&[0.5, 0.25, -0.0, 7.0])),
            "20663a0979e87482ebee4c60ea005a450f2e5a9b42558be6d34ec90d71d97280"
        );
    }

    // Python:
    //   import hashlib, struct
    //   h = [bytes.fromhex(x) for x in ("38e918...d935", "20663a...7280")]   # the two hashes of the test above
    //   m = hashlib.sha256(b"nagaya qwen38 canary v1")
    //   for ids in ([1, 2, 3], [7, 8]): m.update(struct.pack('<I', len(ids)) + struct.pack('<%di' % len(ids), *ids))
    //   m.update(struct.pack('<I', 2) + h[0] + h[1]); m.hexdigest()
    #[test]
    fn the_canary_hash_follows_the_byte_layout() {
        let rows = [
            unhex("38e918286b2706cbcd8ef6d4e0a3e2154507f98dea529e6c3d3af7017617d935"),
            unhex("20663a0979e87482ebee4c60ea005a450f2e5a9b42558be6d34ec90d71d97280"),
        ];
        assert_eq!(
            hex(&canary_hash(&[1, 2, 3], &[7, 8], &rows)),
            "2ffc6b8f1e475b5d6b4713173844f299b50fe4511deb3cf94c9f9223833164b4"
        );
        // A missing last row changes the hash.
        assert_ne!(
            canary_hash(&[1, 2, 3], &[7, 8], &rows[..1]),
            canary_hash(&[1, 2, 3], &[7, 8], &rows)
        );
    }

    #[test]
    fn the_tag_has_23_bytes() {
        assert_eq!(TAG.len(), 23);
    }

    // Python: hashlib.sha256(struct.pack('<f', -0.0)).hexdigest() is 6d58...7bd6, and with 0.0 it is df3f...1119.
    #[test]
    fn negative_zero_and_zero_give_two_hashes() {
        assert_ne!(row_hash(&[-0.0]), row_hash(&[0.0]));
        assert_eq!(
            hex(&row_hash(&[-0.0])),
            "6d58692645c9d1cfaf13541cbd258f86193ef63c2f1d38f6bbca9617372d7bd6"
        );
        assert_eq!(
            hex(&row_hash(&[0.0])),
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
        );
    }

    #[test]
    fn the_argmax_of_a_tie_is_the_lowest_index() {
        assert_eq!(argmax(0, &[1.0, 3.0, 3.0, 2.0]), Ok(1));
        assert_eq!(argmax(0, &[5.0, 5.0, 5.0]), Ok(0));
        assert_eq!(argmax(0, &[-1.0, -2.0]), Ok(0));
    }

    #[test]
    fn a_value_that_is_not_finite_is_an_error_that_names_the_row_and_the_index() {
        let error = argmax(7, &[1.0, f32::NAN, 2.0]).unwrap_err();
        assert!(
            error.contains("row 7") && error.contains("index 1") && error.contains("NaN"),
            "{error}"
        );
        assert!(argmax(0, &[1.0, f32::INFINITY]).is_err());
        assert!(argmax(0, &[f32::NEG_INFINITY]).is_err());
        assert!(argmax(0, &[]).is_err());
    }

    #[test]
    fn the_plan_has_passes_of_5_256_44_and_eight_joined_passes_of_two_segments() {
        let passes = plan();
        let totals: Vec<usize> = passes
            .iter()
            .map(|pass| pass.segments.iter().map(|(_, count)| count).sum())
            .collect();
        assert_eq!(&totals[..3], &[5, 256, 44]);
        assert_eq!(passes.len(), 3 + DECODE_STEPS);
        for pass in &passes[3..] {
            assert_eq!(pass.phase, Phase::Decode);
            assert_eq!(pass.segments, vec![(Which::Short, 1), (Which::Long, 1)]);
        }
        assert!(totals.iter().all(|total| *total <= spec::MAX_PASS_TOKENS as usize));
        assert_eq!(passes.iter().map(|pass| pass.segments.len()).sum::<usize>(), ROWS);
    }

    #[test]
    fn the_ids_equal_the_oracle_fixture_and_fit_the_vocabulary() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/oracle/qwen38.json");
        let fixture: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let ids = |name: &str| -> Vec<i32> {
            let prompt = fixture["prompts"]
                .as_array()
                .unwrap()
                .iter()
                .find(|prompt| prompt["name"] == name)
                .unwrap_or_else(|| panic!("the fixture has no prompt {name}"));
            prompt["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_i64().unwrap() as i32)
                .collect()
        };
        assert_eq!(SHORT_IDS.to_vec(), ids("plain"));
        assert_eq!(LONG_IDS.to_vec(), ids("longer")[..300].to_vec());
        assert!(
            SHORT_IDS
                .iter()
                .chain(&LONG_IDS)
                .all(|id| (0..spec::VOCAB as i32).contains(id))
        );
        assert_eq!(fixture["vocab"].as_u64(), Some(u64::from(spec::VOCAB)));
    }

    /// A fake pass. Context A's argmax at call `k` is `100 + k`, and context B's is `200 + k`.
    fn fake(calls: &mut Vec<PassCall>) -> impl FnMut(&PassCall) -> Result<Vec<Vec<f32>>, String> + '_ {
        move |call| {
            let k = calls.len();
            calls.push(call.clone());
            Ok(call
                .contexts
                .iter()
                .map(|which| peak(if *which == Which::Short { 100 + k } else { 200 + k }))
                .collect())
        }
    }

    #[test]
    fn the_driver_feeds_the_argmax_of_each_context_to_its_next_step() {
        let mut calls = Vec::new();
        let report = run(fake(&mut calls)).unwrap();
        assert_eq!(calls.len(), 3 + DECODE_STEPS);
        // The prefill passes carry the fixed ids, in order.
        assert_eq!(calls[0].tokens, SHORT_IDS.to_vec());
        assert_eq!(calls[1].tokens, LONG_IDS[..256].to_vec());
        assert_eq!(calls[2].tokens, LONG_IDS[256..].to_vec());
        // The first decode step takes A's row of pass 0 and B's row of pass 2.
        assert_eq!(calls[3].tokens, vec![100, 202]);
        // Each later step takes the argmax of the step before, for each context apart.
        for (k, call) in calls.iter().enumerate().skip(4) {
            assert_eq!(
                call.tokens,
                vec![100 + (k as i32 - 1), 200 + (k as i32 - 1)],
                "step {k}"
            );
            assert_eq!(call.lens, vec![1, 1]);
            assert_eq!(call.contexts, vec![Which::Short, Which::Long]);
        }
        assert_eq!(report.row_hashes.len(), ROWS);
        assert_eq!(report.sampled_short.len(), 1 + DECODE_STEPS);
        assert_eq!(report.sampled_long.len(), 2 + DECODE_STEPS);
        assert_eq!(report.sampled_short[0], 100);
        assert_eq!(report.sampled_long[..2], [201, 202]);
        assert_eq!(
            report.canary_hash,
            canary_hash(&SHORT_IDS, &LONG_IDS, &report.row_hashes)
        );
    }

    #[test]
    fn the_driver_stops_on_a_pass_error_a_short_reply_and_a_row_that_is_not_finite() {
        let error = run(|_| Err("the pass stopped".to_owned())).unwrap_err();
        assert_eq!(error, "the pass stopped");
        let error = run(|_| Ok(Vec::new())).unwrap_err();
        assert!(error.contains("gave 0 logits rows"), "{error}");
        let error = run(|call| Ok(vec![vec![0.0; 10]; call.contexts.len()])).unwrap_err();
        assert!(error.contains("10 values"), "{error}");
        let error = run(|call| {
            Ok(call
                .contexts
                .iter()
                .map(|_| {
                    let mut row = peak(0);
                    row[5] = f32::NAN;
                    row
                })
                .collect())
        })
        .unwrap_err();
        assert!(error.contains("row 0") && error.contains("index 5"), "{error}");
    }
}
