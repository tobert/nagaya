//! The family library of Beat This! (CPJKU), a beat and downbeat tracker: all its logic that needs no device.
//!
//! The model reads a log-mel spectrum of 22,050 Hz mono audio at 50 frames each second. It gives two logits for each
//! frame: beat and downbeat. This library holds the WAV reader, the log-mel spectrum, the chunks of a long input with
//! the join of their logits, and the peak picking of the beat times. The reference is the package `beat_this`
//! (`tools/beatthis_golden.sh`).

pub mod audio;
pub mod chunks;
pub mod error;
pub mod mel;
pub mod peaks;
pub mod plan;
pub mod weights;

pub use error::Error;
