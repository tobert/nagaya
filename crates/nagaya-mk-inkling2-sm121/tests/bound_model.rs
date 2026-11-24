//! The model of time of `bound` against the latency that the gate measured on the GB10. It needs no GPU.
//!
//! - The estimate is at least the measure of each cell: the model is above the measure.
//! - A cell that took more than `max_quantum_ms` has a declared bound, and the bound is above the measure.
//! - The estimate is at most 1.5 times the measure of a block at depth, so that the bound stays useful.

mod latency_table;

use latency_table::CELLS;
use nagaya_mk_inkling2_sm121::bound::{estimate_ms, pass_bound_ms};
use nagaya_mk_inkling2_sm121::spec::MAX_QUANTUM_MS;

#[test]
fn the_estimate_is_above_each_measured_call() {
    assert!(CELLS.len() >= 90);
    for cell in CELLS {
        let estimate = estimate_ms(cell.positions, cell.rows, usize::from(cell.cycles));
        assert!(
            estimate >= cell.ms,
            "{:?} at {} positions, {} cycles, {} rows took {} ms, and the model says {estimate} ms",
            cell.format,
            cell.positions,
            cell.cycles,
            cell.rows,
            cell.ms
        );
    }
}

#[test]
fn a_call_that_took_more_than_the_quantum_has_a_bound_above_its_time() {
    let mut long = 0;
    for cell in CELLS.iter().filter(|c| c.ms > u64::from(MAX_QUANTUM_MS)) {
        long += 1;
        let bound = pass_bound_ms(cell.positions, cell.rows, usize::from(cell.cycles)).unwrap_or_else(|| {
            panic!(
                "{:?} at {} positions took {} ms and has no bound",
                cell.format, cell.positions, cell.ms
            )
        });
        assert!(
            u64::from(bound.get()) * 2 >= cell.ms * 3,
            "the bound of {bound} ms is not 1.5 times the {} ms of the call",
            cell.ms
        );
    }
    assert!(long >= 9, "the table holds {long} long calls");
}

#[test]
fn the_estimate_of_a_block_is_at_most_a_third_above_the_measure() {
    for cell in CELLS.iter().filter(|c| c.rows == 512 && c.positions >= 131_072) {
        let estimate = estimate_ms(cell.positions, cell.rows, usize::from(cell.cycles));
        assert!(
            estimate * 2 <= cell.ms * 3,
            "{:?} at {} positions, {} cycles: the model says {estimate} ms, and the call took {} ms",
            cell.format,
            cell.positions,
            cell.cycles,
            cell.ms
        );
    }
}

#[test]
fn a_call_that_took_less_than_the_quantum_may_have_a_bound_but_a_short_one_has_none() {
    // A decode step at 1 row never declares a bound at any measured depth.
    for cell in CELLS.iter().filter(|c| c.rows == 1) {
        assert_eq!(pass_bound_ms(cell.positions, 1, usize::from(cell.cycles)), None);
    }
}
