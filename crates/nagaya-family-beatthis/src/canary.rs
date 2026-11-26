//! The canary of the Beat This! engines (spec 8.1): fixed log-mel chunks, and the hash of the head outputs that an
//! engine gives for them. Each engine crate holds the record table of its target and runs the canary at each open.
//!
//! The content: a chunk of 64 frames (shorter than a tile of the GPU kernels and than one part of the split attention)
//! and a whole chunk of 1,500 frames. The values come from a linear congruential generator in integers, in the range of
//! a log-mel row (0 to 8), so the input needs no math function of any machine.

use sha2::{Digest, Sha256};

use crate::mel::MELS;

/// The tag at the start of the hashed bytes. A change of the content or of the layout changes the tag.
pub const TAG: &[u8] = b"nagaya beatthis canary v1";

/// The frames of each chunk of the canary.
pub const FRAMES: [usize; 2] = [64, 1500];

/// The log-mel rows of chunk `i` of the canary: `FRAMES[i]` rows of 128 values in [0, 8).
pub fn input(i: usize) -> Vec<f32> {
    let n = FRAMES[i] * MELS;
    let mut state: u32 = 0x9e37_79b9 ^ (i as u32).wrapping_mul(0x85eb_ca6b);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // 24 bits of the state, as an exact f32 in [0, 8).
            (state >> 8) as f32 / (1u32 << 21) as f32
        })
        .collect()
}

/// The canary hash of the head outputs of each chunk, in order: SHA-256 of the tag, then for each chunk its frames as
/// u32 little-endian and each output value as f32 little-endian bits, with no change to a value.
pub fn hash(outputs: &[Vec<f32>]) -> String {
    assert_eq!(outputs.len(), FRAMES.len(), "the canary has {} chunks", FRAMES.len());
    let mut h = Sha256::new();
    h.update(TAG);
    for (out, &frames) in outputs.iter().zip(&FRAMES) {
        assert_eq!(
            out.len(),
            frames * 2,
            "a chunk of {frames} frames gives {} values",
            frames * 2
        );
        h.update((frames as u32).to_le_bytes());
        for v in out {
            h.update(v.to_le_bytes());
        }
    }
    format!("sha256:{:x}", h.finalize())
}

/// Why a canary refuses an open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The table has no record for the checkpoint on this target.
    NoRecord {
        target: &'static str,
        checkpoint: String,
        measured: String,
    },
    /// The engine gave other bits than its record.
    Mismatch {
        target: &'static str,
        checkpoint: String,
        record: String,
        measured: String,
    },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::NoRecord {
                target,
                checkpoint,
                measured,
            } => write!(
                f,
                "the Beat This! canary has no record for the checkpoint {checkpoint} on {target}; it measured \
                 {measured}. If these bits are intended, add the line (\"{checkpoint}\", \"{measured}\") to the table \
                 in src/canary.rs of the engine crate, in a commit that records the bits (spec 8.1)"
            ),
            Refusal::Mismatch {
                target,
                checkpoint,
                record,
                measured,
            } => write!(
                f,
                "the Beat This! canary of the checkpoint {checkpoint} on {target} gave {measured}, and its record is \
                 {record}: the build or the machine changed the bits. Do not serve it. If the change is intended, \
                 replace the record in a commit that says the bits moved (spec 8.1)"
            ),
        }
    }
}

impl std::error::Error for Refusal {}

/// Why an engine did not open: the canary refused it (spec 8.1, exit 3 at start), or another failure.
#[derive(Debug)]
pub enum OpenFailure {
    Canary(Refusal),
    Other(String),
}

impl std::fmt::Display for OpenFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenFailure::Canary(r) => r.fmt(f),
            OpenFailure::Other(m) => f.write_str(m),
        }
    }
}

/// Checks a measured hash against the record table of a target: `(checkpoint digest, canary hash)` pairs.
pub fn verify(table: &[(&str, &str)], target: &'static str, checkpoint: &str, measured: &str) -> Result<(), Refusal> {
    match table.iter().find(|(c, _)| *c == checkpoint) {
        None => Err(Refusal::NoRecord {
            target,
            checkpoint: checkpoint.to_owned(),
            measured: measured.to_owned(),
        }),
        Some((_, record)) if *record == measured => Ok(()),
        Some((_, record)) => Err(Refusal::Mismatch {
            target,
            checkpoint: checkpoint.to_owned(),
            record: (*record).to_owned(),
            measured: measured.to_owned(),
        }),
    }
}

/// The record of a checkpoint in a table, if any.
pub fn record<'a>(table: &[(&str, &'a str)], checkpoint: &str) -> Option<&'a str> {
    table.iter().find(|(c, _)| *c == checkpoint).map(|(_, r)| *r)
}
