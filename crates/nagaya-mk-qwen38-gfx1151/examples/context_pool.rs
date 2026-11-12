//! Measure what the engine's context pool takes on gfx1151 (spec 4.2, 4.7).
//!
//! `open` makes the declared pool and reads, for each class, the device free bytes and the host's MemAvailable
//! before and after. This example prints them with the declared bytes. Run it in a fresh process.

use nagaya_abi::Engine;

fn main() {
    let engine = nagaya_mk_qwen38_gfx1151::open().expect("open the engine");
    for reading in engine.pool_readings() {
        let class = reading.class;
        println!(
            "class: {} contexts of {} tokens; declared {} bytes ({} each)",
            class.count,
            class.capacity,
            reading.declared,
            reading.declared / u64::from(class.count)
        );
        println!("  device free fell by {} bytes", reading.device);
        println!(
            "  MemAvailable {:?} -> {:?} bytes",
            reading.available_before, reading.available_after
        );
    }
    let declared = engine
        .limits()
        .memory
        .mandatory
        .iter()
        .find(|term| term.name == "context")
        .map(|term| term.bytes)
        .expect("the limits declare a context term");
    println!("declared context term: {declared} bytes");
}
