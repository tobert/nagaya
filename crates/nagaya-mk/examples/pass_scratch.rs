//! Measure mk's pass scratch once on gfx1151 (milestone 9c).
//!
//! `open` measures the pool at its first asks and refuses a measurement that is past the declared constant. Run this
//! example in two processes, so that each one opens a fresh engine. A refusal is a measurement too: the message holds
//! both numbers. The declared constant comes from `docs/engine-abi.md` and `spec::PASS_SCRATCH_BYTES`.

use nagaya_abi::Engine;

fn main() {
    let engine = nagaya_mk::open().expect("open mk");
    let declared = engine
        .limits()
        .memory
        .mandatory
        .iter()
        .find(|term| term.name == "pass_scratch")
        .map(|term| term.bytes)
        .expect("the limits declare a pass_scratch term");
    println!("measured pass scratch: {} bytes", engine.pass_scratch());
    println!("declared pass scratch: {declared} bytes");
}
