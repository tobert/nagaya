//! Runs a test body at each level of the kernels (`isa::Level`) that the CPU has. A level that the CPU does not have
//! is skipped by name on stderr, past the capture of the test harness, so the skip shows in each run. With
//! `NAGAYA_REQUIRE_LEVELS=1` such a level fails the test.

use std::io::Write;

use nagaya_mk_beatthis_cpu::isa::{self, Level};

pub fn each_level(test: &str, mut f: impl FnMut(Level)) {
    for level in Level::ALL {
        if level > isa::detected() {
            let why = format!("the CPU does not have {}", level.name());
            if std::env::var("NAGAYA_REQUIRE_LEVELS").is_ok_and(|v| v == "1") {
                panic!("{test}: {why}, and NAGAYA_REQUIRE_LEVELS is set");
            }
            // Not eprintln!: the harness captures it, and the skip must show.
            let _ = writeln!(std::io::stderr(), "skipped: {test} at {}: {why}", level.name());
            continue;
        }
        isa::with_level(level, || f(level));
    }
}
