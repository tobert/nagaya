//! The bound of a pass that runs a long block at depth (engine-abi "The GB10 Inkling 2 binding", "The bound of a
//! block at depth"; D8, "A long pass with a declared bound"). It needs no GPU, so the host tests cover it.
//!
//! The model gives the time of a pass from its first position, its rows and its cycles. It was fitted to the latency
//! gate (`tests/cuda_latency.rs`) with a margin, so that it is above each measure.

use std::num::NonZeroU32;

use crate::spec::MAX_QUANTUM_MS;

/// The ceiling of a long pass in milliseconds. The largest block that a slot holds takes about 18,200 ms by the
/// model, and its bound is 27,300 ms.
pub const MAX_LONG_QUANTUM_MS: u32 = 60_000;

/// The fixed time of a pass at 3 cycles, in microseconds: the head and the launch of a call.
pub const BASE_US: u64 = 100_000;

/// The time of each row at 3 cycles, in microseconds: the matmuls, the experts and the convolutions.
pub const ROW_US: u64 = 1_350;

/// The cost of one position in one run with a full cache, for a block of 16 rows or fewer, in nanoseconds. One tile
/// reads the keys of the run once, and that is bound by memory.
pub const SINGLE_TILE_NS: u64 = 90;

/// The cost of one position in one run with a full cache for each tile of 16 rows of a larger block, in nanoseconds.
/// A block of many tiles is bound by the MMA, and the formats take the same cost.
pub const TILE_NS: u64 = 42;

/// The rows of one tile of the attention.
const TILE_ROWS: u64 = 16;

/// The runs of a pass at `cycles` cycles: the 24 layers and 12 more for each cycle past the first.
fn runs(cycles: u64) -> u64 {
    24 + 12 * (cycles - 1)
}

/// The runs with a full cache: 4 layers and 2 more for each cycle past the first.
fn full_runs(cycles: u64) -> u64 {
    4 + 2 * (cycles - 1)
}

/// The time of a pass of `rows` rows from position `pos0` at `cycles` cycles, in milliseconds, rounded up. The model
/// is above the time of each measured call. `cycles` is 1 to 5 and `rows` is 1 to 512.
pub fn estimate_ms(pos0: usize, rows: usize, cycles: usize) -> u64 {
    assert!(
        (1..=5).contains(&cycles) && rows >= 1,
        "the model of time takes 1 to 5 cycles and 1 or more rows, and it got {cycles} cycles and {rows} rows. This is a defect of the engine: report it"
    );
    let (cycles, rows, end) = (cycles as u64, rows as u64, pos0 as u64 + rows as u64);
    let tiles = rows.div_ceil(TILE_ROWS);
    let rate = SINGLE_TILE_NS.max(TILE_NS * tiles);
    let base_us = (BASE_US + ROW_US * rows) * runs(cycles) / runs(3);
    let depth_us = full_runs(cycles) * end * rate / 1_000;
    (base_us + depth_us).div_ceil(1_000)
}

/// The bound of a pass of `rows` rows from position `pos0` at `cycles` cycles: 1.5 times the estimate. It is `None`
/// when that fits `MAX_QUANTUM_MS`: the pass is not long.
pub fn pass_bound_ms(pos0: usize, rows: usize, cycles: usize) -> Option<NonZeroU32> {
    let bound = (estimate_ms(pos0, rows, cycles) * 3).div_ceil(2);
    if bound <= u64::from(MAX_QUANTUM_MS) {
        return None;
    }
    // The ceiling is above the bound of the largest block (a host test), so the bound is never clamped below 1.5 times
    // the estimate.
    NonZeroU32::new(u32::try_from(bound.min(u64::from(MAX_LONG_QUANTUM_MS))).expect("the bound fits 32 bits"))
}

const _: () = assert!(MAX_LONG_QUANTUM_MS <= nagaya_abi::LONG_QUANTUM_CEILING_MS);
const _: () = assert!(MAX_LONG_QUANTUM_MS > MAX_QUANTUM_MS);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{MAX_BLOCK, MAX_SLOT_TOKENS};

    fn ms(bound: Option<NonZeroU32>) -> Option<u32> {
        bound.map(NonZeroU32::get)
    }

    #[test]
    fn the_estimate_is_the_sum_of_the_base_and_the_depth() {
        // 512 rows from 0 at 3 cycles: (100,000 + 1,350 * 512) us, and 8 runs of 512 positions at 42 * 32 ns.
        assert_eq!(estimate_ms(0, 512, 3), 791 + 6);
        // One row at 524,288 positions, 5 cycles: 12 full runs at 90 ns.
        assert_eq!(
            estimate_ms(524_288, 1, 5),
            (100_000 + 1_350) * 72 / 48 / 1_000 + 566 + 1
        );
        // 512 rows at 524,288 positions, 5 cycles, which took 8,391 ms on `f32`.
        assert_eq!(estimate_ms(524_288, 512, 5), 9_651);
    }

    #[test]
    fn a_pass_that_fits_the_quantum_declares_no_bound() {
        assert_eq!(pass_bound_ms(0, 512, 3), None);
        assert_eq!(pass_bound_ms(1_048_575, 1, 5), None);
        assert_eq!(pass_bound_ms(131_072, 512, 5), None);
        // The estimate is 3,333 ms at most: 1.5 times it is 5,000 ms, which is not above the quantum.
        let edge = (0..524_288)
            .find(|pos0| estimate_ms(*pos0, 512, 3) * 3 / 2 > 5_000)
            .unwrap();
        assert!(edge > 131_072 && edge < 409_600, "the edge is at {edge}");
        assert_eq!(pass_bound_ms(edge - 1, 512, 3), None);
        assert!(pass_bound_ms(edge, 512, 3).is_some());
    }

    #[test]
    fn a_long_block_declares_one_and_a_half_times_the_estimate() {
        assert_eq!(ms(pass_bound_ms(409_600, 512, 3)), Some(7_802));
        assert_eq!(ms(pass_bound_ms(524_288, 512, 5)), Some(14_477));
    }

    #[test]
    fn the_estimate_never_falls_when_a_term_grows() {
        for cycles in 1..=5 {
            for rows in [1, 16, 17, 128, 512] {
                let mut last = 0;
                for pos0 in (0..MAX_SLOT_TOKENS).step_by(8_192) {
                    let now = estimate_ms(pos0, rows, cycles);
                    assert!(now >= last, "{cycles} cycles, {rows} rows, position {pos0}");
                    last = now;
                }
            }
            for pos0 in [0, 4_096, 524_288] {
                let mut last = 0;
                for rows in 1..=MAX_BLOCK {
                    let now = estimate_ms(pos0, rows, cycles);
                    assert!(now >= last, "{cycles} cycles, position {pos0}, {rows} rows");
                    last = now;
                }
            }
        }
        for pos0 in [0, 524_288] {
            for rows in [1, 512] {
                assert!(estimate_ms(pos0, rows, 5) > estimate_ms(pos0, rows, 3));
                assert!(estimate_ms(pos0, rows, 3) > estimate_ms(pos0, rows, 1));
            }
        }
    }

    #[test]
    fn the_ceiling_is_above_the_bound_of_the_largest_block() {
        // The largest block that a slot holds starts 512 rows before the end of a slot of the most positions.
        let worst = (estimate_ms(MAX_SLOT_TOKENS - 512, 512, 5) * 3).div_ceil(2);
        assert!(
            worst < u64::from(MAX_LONG_QUANTUM_MS),
            "the bound of the largest block is {worst} ms"
        );
        assert_eq!(ms(pass_bound_ms(MAX_SLOT_TOKENS - 512, 512, 5)), Some(worst as u32));
        assert!(
            worst * 2 > u64::from(MAX_LONG_QUANTUM_MS) / 2,
            "the ceiling is within a factor of 4"
        );
    }

    #[test]
    #[should_panic(expected = "1 to 5 cycles")]
    fn a_count_of_cycles_outside_the_limit_is_a_defect() {
        estimate_ms(0, 1, 6);
    }
}
