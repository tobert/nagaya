//! The canary records of the CPU engine (spec 8.1): the canary hash (`nagaya_family_beatthis::canary`) of each
//! checkpoint, by the SHA-256 of its weights file. The bits are the same on each machine and each instruction set
//! (spec 3.2c), so the key needs no machine. A checkpoint with no line here does not open.
//!
//! Add a line only in a commit that records the bits. `tests/canary.rs` prints the line of each checkpoint in
//! `BEATTHIS_MODEL_DIR` when it is missing or differs.

/// The target of the records.
pub const TARGET: &str = "cpu";

/// (checkpoint digest, canary hash). Recorded 2026-11-26 on the Strix Halo machine; the test of each instruction set
/// gives the same hashes at SSE, AVX2 and AVX-512.
pub const TABLE: &[(&str, &str)] = &[
    // final0
    (
        "sha256:0a3207a97c0db80d3ab35fdb80a569a52288f586447a18830929ea54141bad2a",
        "sha256:53857bb8463e3e50727669fa29927ade88be4fb5a86ed4f740b73c2edcf8be75",
    ),
    // small0
    (
        "sha256:05eae168833008f4886d0ff31b7ff4b7d3610aa1264bb39fa04ebc598c971258",
        "sha256:085ce95abf894ae51faa1682e41e292ffa8893df8cca37789c72b9e084d2e71e",
    ),
];
