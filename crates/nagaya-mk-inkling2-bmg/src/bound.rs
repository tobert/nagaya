//! The bound of a pass of Inkling 2 that runs the tower of a large image (engine-abi D8, "A long pass with a declared
//! bound"; "The time of a large image"). It needs no card, so the host tests cover it.

use std::num::NonZeroU32;

use crate::spec::MAX_QUANTUM_MS;

/// The ceiling of a long pass in milliseconds: about twice the bound of the largest image (engine-abi D8).
pub const MAX_LONG_QUANTUM_MS: u32 = 180_000;

/// The tower of the largest image, 4,096 by 4,096 px, took about 56,780 ms in one pass on 2026-11-21
/// (`tests/xe_long.rs`).
pub const LARGEST_TOWER_MS: u64 = 56_780;

/// The image tokens of the largest image.
pub const LARGEST_TOKENS: u64 = 10_609;

/// The slowest text block measured, a block of 512 at 164,352 positions (the bound `MAX_QUANTUM_MS`).
pub const SLOWEST_BLOCK_MS: u64 = 2_020;

/// The time of the tower of an image of `tokens` tokens, in milliseconds. The attention of the tower reads each pair
/// of patches, so the time grows with the square of the count (`docs/perf.md`, "The vision tower on a large image").
pub fn tower_ms(tokens: usize) -> u64 {
    let tokens = tokens as u64;
    (LARGEST_TOWER_MS * tokens * tokens).div_ceil(LARGEST_TOKENS * LARGEST_TOKENS)
}

/// The bound of a pass that runs the tower of each image of `towers`, given as its count of tokens. It is 1.5 times
/// the towers plus the slowest block, up to the ceiling. It is `None` when that fits `MAX_QUANTUM_MS`: the pass is
/// not long.
pub fn pass_bound_ms(towers: &[usize]) -> Option<NonZeroU32> {
    if towers.is_empty() {
        return None;
    }
    let estimate: u64 = towers.iter().map(|tokens| tower_ms(*tokens)).sum::<u64>() + SLOWEST_BLOCK_MS;
    let bound = (estimate * 3).div_ceil(2);
    if bound <= u64::from(MAX_QUANTUM_MS) {
        return None;
    }
    NonZeroU32::new(bound.min(u64::from(MAX_LONG_QUANTUM_MS)) as u32)
}

const _: () = assert!(MAX_LONG_QUANTUM_MS <= nagaya_abi::LONG_QUANTUM_CEILING_MS);

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(bound: Option<NonZeroU32>) -> Option<u32> {
        bound.map(NonZeroU32::get)
    }

    #[test]
    fn the_tower_of_the_largest_image_is_the_measured_time() {
        assert_eq!(tower_ms(10_609), 56_780);
        assert_eq!(tower_ms(0), 0);
        // Half the tokens take a quarter of the time.
        assert_eq!(tower_ms(5_305), 14_198);
    }

    /// The largest image gives 1.5 times (56,780 + 2,020) ms. Falsified by a bound with no margin.
    #[test]
    fn the_largest_image_gives_a_bound_with_a_margin() {
        assert_eq!(ms(pass_bound_ms(&[10_609])), Some(88_200));
    }

    /// A pass with no tower, and a small image, fit `MAX_QUANTUM_MS`, so they declare no bound. An image of 1,614
    /// tokens is the first that does.
    #[test]
    fn a_pass_that_fits_the_quantum_bound_is_not_long() {
        assert_eq!(pass_bound_ms(&[]), None);
        assert_eq!(pass_bound_ms(&[90]), None);
        assert_eq!(pass_bound_ms(&[1_613]), None);
        assert_eq!(ms(pass_bound_ms(&[1_614])), Some(5_003));
    }

    /// Two towers add, and the bound stops at the ceiling.
    #[test]
    fn the_towers_of_a_pass_add_up_to_the_ceiling() {
        assert_eq!(ms(pass_bound_ms(&[5_305, 5_305])), Some(45_624));
        assert_eq!(ms(pass_bound_ms(&[10_609, 10_609])), Some(173_370));
        assert_eq!(ms(pass_bound_ms(&[10_609; 3])), Some(MAX_LONG_QUANTUM_MS));
    }
}
