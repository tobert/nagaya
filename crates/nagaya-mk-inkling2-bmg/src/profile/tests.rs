use super::*;
use nagaya_mk_bmg::protocol::{OP_EXPERT_UP, OP_MATMUL};

fn task(opcode: u16, n: u32) -> MkTaskDescriptor {
    MkTaskDescriptor {
        opcode,
        n,
        ..Default::default()
    }
}

/// A launch of two tasks on two work-groups, from the clocks `[task][group] = (start, run, end)`.
fn launch(clocks: [[(u64, u64, u64); 2]; 2]) -> Launch {
    let mut words = Vec::new();
    for task in clocks {
        for (start, run, end) in task {
            words.extend([start, run, end]);
        }
    }
    Launch {
        tasks: vec![task(OP_MATMUL, 1), task(OP_EXPERT_UP, 1)],
        clocks: words,
        groups: 2,
        ticks_per_ns: 2.0,
        host_ns: 500,
    }
}

/// Task 0 starts at 10 in group 0 and 12 in group 1, so its boundary runs from 10. Group 1 ends it last at 40, and
/// group 0 passes the wait of task 1 first at 45. Thus task 1 has a boundary of 5 and work until 90.
#[test]
fn the_times_of_the_tasks_add_up_to_the_launch() {
    let times = task_times(&launch([[(10, 11, 30), (12, 14, 40)], [(30, 45, 90), (40, 47, 70)]])).unwrap();
    assert_eq!(
        times.iter().map(|t| (t.boundary, t.work, t.spread)).collect::<Vec<_>>(),
        vec![(1, 29, 10), (5, 45, 20)]
    );
    let sum: u64 = times.iter().map(|t| t.boundary + t.work).sum();
    assert_eq!(
        sum,
        90 - 10,
        "the times cover the launch from the first start to the last end"
    );
}

#[test]
fn a_clock_that_runs_back_fails() {
    let error = task_times(&launch([[(10, 11, 30), (12, 14, 40)], [(30, 45, 44), (40, 47, 70)]])).unwrap_err();
    assert!(error.contains("task 1 in work-group 0"), "{error}");
}

/// Group 0 passes the wait of task 1 at 3,000, and group 1 ended task 0 at 6,000: more than SKEW before.
#[test]
fn a_wait_passed_before_the_last_end_fails() {
    let clocks = [
        [(10, 11, 30), (12, 14, 6_000)],
        [(30, 3_000, 9_000), (6_000, 6_010, 7_000)],
    ];
    let error = task_times(&launch(clocks)).unwrap_err();
    assert!(error.contains("do not agree"), "{error}");
}

/// Group 0 passes the wait of task 1 at 3,500, within SKEW of the end of task 0 at 6,000: a boundary of zero.
#[test]
fn a_wait_within_the_skew_counts_as_no_boundary() {
    let clocks = [
        [(10, 11, 30), (12, 14, 6_000)],
        [(30, 3_500, 9_000), (6_000, 6_010, 7_000)],
    ];
    let times = task_times(&launch(clocks)).unwrap();
    assert_eq!((times[1].boundary, times[1].work), (0, 3_000));
}

#[test]
fn a_launch_with_too_few_clocks_fails() {
    let mut l = launch([[(10, 11, 30), (12, 14, 40)], [(30, 45, 90), (40, 47, 70)]]);
    l.clocks.pop();
    assert!(task_times(&l).unwrap_err().contains("needs 12 clocks"));
}

/// MATMUL takes more ticks than EXPERT_UP, so it comes first, though its key comes later.
#[test]
fn the_rows_sum_each_key_and_put_the_largest_first() {
    let time = |opcode, boundary, work, spread| TaskTime {
        task: task(opcode, 1),
        boundary,
        work,
        spread,
    };
    let times = [
        time(OP_EXPERT_UP, 1, 4, 2),
        time(OP_MATMUL, 2, 20, 3),
        time(OP_EXPERT_UP, 1, 4, 1),
        time(OP_MATMUL, 3, 25, 4),
    ];
    assert_eq!(
        rows(&times, by_op),
        vec![
            Row {
                key: "MATMUL".to_owned(),
                count: 2,
                boundary: 5,
                work: 45,
                spread: 7,
            },
            Row {
                key: "EXPERT_UP".to_owned(),
                count: 2,
                boundary: 2,
                work: 8,
                spread: 3,
            },
        ]
    );
}

#[test]
#[should_panic(expected = "other clocks")]
fn launches_of_two_clocks_have_no_rate() {
    let l = launch([[(10, 11, 30), (12, 14, 40)], [(30, 45, 90), (40, 47, 70)]]);
    let mut other = l.clone();
    other.ticks_per_ns = 2.4;
    ticks_per_ns(&[l, other]);
}
