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
use crate::worker::Qwen38Error;

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

/// How the open treats the canary (spec 8.1). The canary runs at each load in both forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanaryCheck {
    /// Compare the canary hash with the record. A mismatch or a missing record is an error. The process uses this form.
    Record,
    /// Run the canary and report it, but compare nothing. A person measures a new checkpoint with this form.
    Measure,
}

/// The key of a record: the checkpoint and the two settings that change bits by choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key {
    /// The checkpoint digest as the identity writes it, with the `spot-sha256:` prefix.
    pub checkpoint: String,
    /// The value of `QWEN38_FAST_QMM`: true when unset or 1.
    pub fast_qmm: bool,
    /// The value of `QWEN38_TILES`: true when unset or 1.
    pub tiles: bool,
}

impl Key {
    /// The three key values as text, for a message.
    pub fn describe(&self) -> String {
        format!(
            "checkpoint {}, QWEN38_FAST_QMM={}, QWEN38_TILES={}",
            self.checkpoint,
            u8::from(self.fast_qmm),
            u8::from(self.tiles)
        )
    }
}

/// One line of the table of records.
struct Record {
    checkpoint: &'static str,
    fast_qmm: bool,
    tiles: bool,
    canary_hash: &'static str,
}

/// The table of records (spec 8.1). A build that gives other bits does not open until a commit changes its line.
///
/// To record a new checkpoint, measure it with the test `hip_canary` and add a line. The commit says that it records
/// the bits of that checkpoint. To change a line, the commit says that the bits moved. The hashes were measured on the
/// card on 2026-11-12, in 5 separate processes for each line, and all five were equal.
const RECORDS: &[Record] = &[
    Record {
        checkpoint: "spot-sha256:0e25ef0d3240134fc2ba763e69a4d5e273bbd5bdf7a8d3386834f90bf2f59c37",
        fast_qmm: true,
        tiles: true,
        canary_hash: "8e05c9ab8c16459c412c0164ec5d3e86abf1d78e6e9ec93e79330552d1881a1c",
    },
    Record {
        checkpoint: "spot-sha256:0e25ef0d3240134fc2ba763e69a4d5e273bbd5bdf7a8d3386834f90bf2f59c37",
        fast_qmm: false,
        tiles: true,
        canary_hash: "9017e7aa37fb9b132dd27226b41bc695e50eab149d5fb1f2dbd15ee97e829602",
    },
];

/// The recorded canary hash of `key`, or `None` when the table has no line for all three key values.
pub fn record(key: &Key) -> Option<&'static str> {
    lookup(RECORDS, key)
}

fn lookup(table: &[Record], key: &Key) -> Option<&'static str> {
    table
        .iter()
        .find(|line| line.checkpoint == key.checkpoint && line.fast_qmm == key.fast_qmm && line.tiles == key.tiles)
        .map(|line| line.canary_hash)
}

/// The command that measures a canary hash on the card.
const MEASURE_COMMAND: &str = "cargo test -p nagaya-mk-qwen38-gfx1151 --features hip --test hip_canary -- --nocapture";

/// The error for a key with no record. `measured` is the hash of this load, when a load ran.
fn no_record_error(key: &Key, measured: Option<&str>) -> Qwen38Error {
    let found = match measured {
        Some(hash) => format!(" This load measured the canary hash {hash}."),
        None => String::new(),
    };
    Qwen38Error::canary(format!(
        "this checkpoint and these settings have no canary record ({}).{found} Measure it with `{MEASURE_COMMAND}`,          with QWEN38_SHARDS set and no other GPU client. Then add the line to the table in `src/canary.rs`. The commit          must say that it records the bits of this checkpoint.",
        key.describe()
    ))
}

/// The canary hash for the identity. It comes from the table, so a description needs no GPU.
///
/// With `Record`, a key with no line is an error. With `Measure`, it gives an empty hash, and the identity of that
/// engine cannot match a description that a process serves.
pub fn identity_hash(check: CanaryCheck, key: &Key) -> Result<String, Qwen38Error> {
    match (record(key), check) {
        (Some(hash), _) => Ok(hash.to_owned()),
        (None, CanaryCheck::Measure) => Ok(String::new()),
        (None, CanaryCheck::Record) => Err(no_record_error(key, None)),
    }
}

/// The error for a canary hash that differs from its record.
fn mismatch_error(key: &Key, recorded: &str, report: &Report) -> Qwen38Error {
    let rows: Vec<String> = report.row_hashes.iter().map(hex).collect();
    Qwen38Error::canary(format!(
        "the canary of this build does not match its record ({}). The record is {recorded}. This load measured {}.          The 19 row hashes are: {}. The bits of this build differ from the record. If the change of bits is meant,          replace the value in the table in `src/canary.rs` in a commit that says that the bits moved. If it is not          meant, the build or the driver changed the result, so do not serve it.",
        key.describe(),
        hex(&report.canary_hash),
        rows.join(" ")
    ))
}

/// Decide the open from the canary of a load. `run` is what the load gave: a report, or why the canary did not run.
///
/// A canary that did not run is an ordinary error with its cause, with either form. With `Record`, the report must
/// equal the record.
pub fn verify(check: CanaryCheck, key: &Key, run: &Result<Report, String>) -> Result<(), Qwen38Error> {
    let report = run
        .as_ref()
        .map_err(|cause| Qwen38Error::new(format!("the canary did not finish: {cause}")))?;
    if check == CanaryCheck::Measure {
        return Ok(());
    }
    match record(key) {
        None => Err(no_record_error(key, Some(&hex(&report.canary_hash)))),
        Some(recorded) if recorded == hex(&report.canary_hash) => Ok(()),
        Some(recorded) => Err(mismatch_error(key, recorded, report)),
    }
}

/// The line to add to the table for a measured report, for a person to paste.
pub fn table_line(key: &Key, hash: &[u8; 32]) -> String {
    format!(
        "Record {{ checkpoint: \"{}\", fast_qmm: {}, tiles: {}, canary_hash: \"{}\" }}",
        key.checkpoint,
        key.fast_qmm,
        key.tiles,
        hex(hash)
    )
}

/// The count of canaries that loads ran in this process. A test reads it to see that a load ran one.
static LOAD_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count one canary run of a load.
pub(crate) fn count_load_run() {
    LOAD_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// The count of canaries that loads ran in this process.
pub fn load_runs() -> u64 {
    LOAD_RUNS.load(std::sync::atomic::Ordering::SeqCst)
}

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

impl Which {
    /// The tokens that the context holds at the end of the run: its prompt and the 8 decode steps.
    pub fn tokens_needed(self) -> usize {
        let prompt = match self {
            Which::Short => SHORT_IDS.len(),
            Which::Long => LONG_IDS.len(),
        };
        prompt + DECODE_STEPS
    }
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

    fn key(checkpoint: &str, fast_qmm: bool, tiles: bool) -> Key {
        Key {
            checkpoint: checkpoint.to_owned(),
            fast_qmm,
            tiles,
        }
    }

    const CHECKPOINT: &str = "spot-sha256:0e25ef0d3240134fc2ba763e69a4d5e273bbd5bdf7a8d3386834f90bf2f59c37";

    /// A report with made-up row hashes and the given canary hash.
    fn report(canary_hash: [u8; 32]) -> Report {
        Report {
            row_hashes: (0..ROWS).map(|row| [row as u8; 32]).collect(),
            canary_hash,
            sampled_short: Vec::new(),
            sampled_long: Vec::new(),
            short_ms: 0.0,
            long_ms: 0.0,
            decode_ms: 0.0,
            total_ms: 0.0,
        }
    }

    #[test]
    fn the_table_holds_the_two_measured_records_and_no_other() {
        assert_eq!(
            record(&key(CHECKPOINT, true, true)),
            Some("8e05c9ab8c16459c412c0164ec5d3e86abf1d78e6e9ec93e79330552d1881a1c")
        );
        assert_eq!(
            record(&key(CHECKPOINT, false, true)),
            Some("9017e7aa37fb9b132dd27226b41bc695e50eab149d5fb1f2dbd15ee97e829602")
        );
        // No value is measured for tiles = 0, so the table has no line for it.
        assert_eq!(record(&key(CHECKPOINT, true, false)), None);
        assert_eq!(record(&key(CHECKPOINT, false, false)), None);
        assert_eq!(record(&key("spot-sha256:other", true, true)), None);
    }

    #[test]
    fn a_lookup_with_another_value_of_each_key_part_finds_no_record() {
        for line in RECORDS {
            let found =
                |checkpoint: &str, fast_qmm: bool, tiles: bool| lookup(RECORDS, &key(checkpoint, fast_qmm, tiles));
            assert_eq!(
                found(line.checkpoint, line.fast_qmm, line.tiles),
                Some(line.canary_hash)
            );
            assert_eq!(found(line.checkpoint, !line.fast_qmm, !line.tiles), None);
            assert_ne!(
                found(line.checkpoint, !line.fast_qmm, line.tiles),
                Some(line.canary_hash)
            );
            assert_ne!(
                found(line.checkpoint, line.fast_qmm, !line.tiles),
                Some(line.canary_hash)
            );
            assert_eq!(found("spot-sha256:other", line.fast_qmm, line.tiles), None);
        }
    }

    #[test]
    fn the_checked_description_of_a_key_with_no_record_names_the_key_and_the_command() {
        let error = identity_hash(CanaryCheck::Record, &key("spot-sha256:abc", true, false)).unwrap_err();
        assert_eq!(error.kind(), crate::worker::Qwen38ErrorKind::Canary);
        let message = error.message();
        for needle in [
            "spot-sha256:abc",
            "QWEN38_FAST_QMM=1",
            "QWEN38_TILES=0",
            "cargo test -p nagaya-mk-qwen38-gfx1151 --features hip --test hip_canary -- --nocapture",
            "QWEN38_SHARDS",
            "src/canary.rs",
            "records the bits of this checkpoint",
        ] {
            assert!(message.contains(needle), "`{needle}` is not in `{message}`");
        }
    }

    #[test]
    fn the_measured_description_of_a_key_with_no_record_has_an_empty_hash() {
        assert_eq!(
            identity_hash(CanaryCheck::Measure, &key("spot-sha256:abc", true, false)),
            Ok(String::new())
        );
        // A key with a record gives the record's value in both forms.
        for check in [CanaryCheck::Record, CanaryCheck::Measure] {
            assert_eq!(
                identity_hash(check, &key(CHECKPOINT, true, true)),
                Ok("8e05c9ab8c16459c412c0164ec5d3e86abf1d78e6e9ec93e79330552d1881a1c".to_owned())
            );
        }
    }

    #[test]
    fn the_check_passes_for_the_recorded_hash_and_fails_with_the_mismatch_for_another() {
        let key = key(CHECKPOINT, true, true);
        let recorded = unhex(record(&key).unwrap());
        assert_eq!(verify(CanaryCheck::Record, &key, &Ok(report(recorded))), Ok(()));
        let mut other = recorded;
        other[31] ^= 1;
        let error = verify(CanaryCheck::Record, &key, &Ok(report(other))).unwrap_err();
        assert_eq!(error.kind(), crate::worker::Qwen38ErrorKind::Canary);
        let message = error.message();
        assert!(message.contains("does not match its record"), "{message}");
        assert!(message.contains(&hex(&recorded)), "the record is missing: {message}");
        assert!(
            message.contains(&hex(&other)),
            "the measured hash is missing: {message}"
        );
        assert!(message.contains("QWEN38_FAST_QMM=1"), "{message}");
        assert!(message.contains("that says that the bits moved"), "{message}");
        for row in 0..ROWS {
            assert!(
                message.contains(&hex(&[row as u8; 32])),
                "row {row} is missing: {message}"
            );
        }
    }

    #[test]
    fn the_measure_form_compares_nothing() {
        let key = key(CHECKPOINT, true, true);
        assert_eq!(verify(CanaryCheck::Measure, &key, &Ok(report([7; 32]))), Ok(()));
    }

    #[test]
    fn a_check_of_a_key_with_no_record_gives_the_measured_hash() {
        let key = key("spot-sha256:abc", true, true);
        let error = verify(CanaryCheck::Record, &key, &Ok(report([7; 32]))).unwrap_err();
        assert_eq!(error.kind(), crate::worker::Qwen38ErrorKind::Canary);
        assert!(error.message().contains(&hex(&[7; 32])), "{}", error.message());
    }

    #[test]
    fn a_canary_that_did_not_finish_is_an_ordinary_error_in_both_forms() {
        let key = key(CHECKPOINT, true, true);
        for check in [CanaryCheck::Record, CanaryCheck::Measure] {
            let error = verify(check, &key, &Err("a canary pass stopped on the spin bound".to_owned())).unwrap_err();
            assert_eq!(error.kind(), crate::worker::Qwen38ErrorKind::Other);
            assert!(error.message().contains("spin bound"), "{}", error.message());
        }
    }

    #[test]
    fn the_table_line_is_the_text_of_a_record() {
        let line = table_line(&key("spot-sha256:abc", false, true), &[0xab; 32]);
        assert_eq!(
            line,
            format!(
                "Record {{ checkpoint: \"spot-sha256:abc\", fast_qmm: false, tiles: true, canary_hash: \"{}\" }}",
                "ab".repeat(32)
            )
        );
    }

    #[test]
    fn each_context_needs_its_prompt_and_the_decode_steps() {
        assert_eq!(Which::Short.tokens_needed(), 13);
        assert_eq!(Which::Long.tokens_needed(), 308);
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
