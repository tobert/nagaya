//! The canary records of the gfx1151 engine (spec 8.1): the canary hash (`nagaya_family_beatthis::canary`) of each
//! checkpoint on gfx1151, by the SHA-256 of its weights file. The bits are those of this target and one ROCm build of
//! the kernels. A checkpoint with no line here does not open.
//!
//! Add a line only in a commit that records the bits. `tests/hip_canary.rs` (a card test) prints the line of each
//! checkpoint in `BEATTHIS_MODEL_DIR` when it is missing or differs.

/// The target of the records.
pub const TARGET: &str = "gfx1151";

/// (checkpoint digest, canary hash). Recorded 2026-11-26 on the Strix Halo machine, ROCm 7.2.
pub const TABLE: &[(&str, &str)] = &[
    // final0
    (
        "sha256:0a3207a97c0db80d3ab35fdb80a569a52288f586447a18830929ea54141bad2a",
        "sha256:01631d92aca4ee46ea1edbff89c4f11044e402d3d283fb9b3a05e209c64b238e",
    ),
    // small0
    (
        "sha256:05eae168833008f4886d0ff31b7ff4b7d3610aa1264bb39fa04ebc598c971258",
        "sha256:f3aa5b2eca97a2074428eee4c24678be0e0685e3264631ba718819fd051ffc22",
    ),
];
