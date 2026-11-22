//! The host builder of the dependency graph of a task ring. It has no GPU code.
//!
//! Each task declares the byte ranges that it reads and writes. The builder finds the hazards between tasks, and it
//! turns each one into a wait on the earlier task. Then it drops each wait that other waits imply. A task has room
//! for two waits, so the builder puts a JOIN task before a task that keeps more. A wait always points backward in
//! the ring: a grid that walks the ring in order cannot deadlock then, because every block is resident.
//!
//! A task signals its own event, the one that has the index of the task in the ring. A block adds 1 to it after each
//! work item. A wait names the event of the producer and a target: the number of items of the producer.

/// A range of device bytes, from `start` up to but not including `end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u64,
    pub end: u64,
}

impl Span {
    /// The range of `bytes` bytes that starts at the device address `start`.
    pub fn new(start: u64, bytes: u64) -> Self {
        let end = start
            .checked_add(bytes)
            .unwrap_or_else(|| panic!("the range at {start:#x} of {bytes} bytes passes the end of the address space"));
        Self { start, end }
    }

    fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end && self.start < self.end && other.start < other.end
    }
}

/// What the builder knows of one task: its count of items, the ranges that it touches, and the tasks that it must
/// follow whatever the ranges say.
#[derive(Clone, Debug, Default)]
pub struct Node {
    pub items: u32,
    pub reads: Vec<Span>,
    pub writes: Vec<Span>,
    /// Indices of earlier tasks. Use this for an order that no range shows, such as a timing probe.
    pub after: Vec<usize>,
}

/// One wait of a task: the event of the producer, and the count that the event must reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    pub event: u32,
    pub target: u32,
}

/// One task of the planned ring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planned {
    /// The index of the input node, or `None` for a JOIN that the builder inserted.
    pub origin: Option<usize>,
    /// The count of items of the task. A JOIN has 0.
    pub items: u32,
    /// At most two waits. Each one points to an earlier task.
    pub waits: Vec<Wait>,
}

/// The count that the event of a task reaches when the task is done. A task with no items, a JOIN, counts 1.
pub fn signals(items: u32) -> u32 {
    items.max(1)
}

fn conflicts(earlier: &Node, later: &Node) -> bool {
    let any = |xs: &[Span], ys: &[Span]| xs.iter().any(|x| ys.iter().any(|y| x.overlaps(y)));
    // Read after write, write after read, and write after write.
    any(&earlier.writes, &later.reads) || any(&earlier.reads, &later.writes) || any(&earlier.writes, &later.writes)
}

/// Plan the ring: the waits of each task, and the JOIN tasks that fit them in two.
///
/// # Panics
/// A node that waits on itself or on a later node panics, and so does a node whose count of items comes from the
/// plan (`u32::MAX`), because the host cannot give a target for it.
pub fn plan(nodes: &[Node]) -> Vec<Planned> {
    let mut out: Vec<Planned> = Vec::with_capacity(nodes.len());
    // For each planned task, the planned tasks that it waits on, directly or not.
    let mut reach: Vec<Vec<bool>> = Vec::with_capacity(nodes.len());
    let mut position: Vec<usize> = Vec::with_capacity(nodes.len());

    for (j, node) in nodes.iter().enumerate() {
        assert!(
            node.items != u32::MAX,
            "task {j} takes its count of items from the plan on the device. The host cannot give a target for a wait on it. \
             Give the task a count of items, or do not wait on it"
        );
        for &i in &node.after {
            assert!(
                i < j,
                "task {j} waits on task {i}, which is not before it in the ring. A wait must point backward. \
                 Move task {i} before task {j}"
            );
        }
        let mut deps: Vec<usize> = (0..j)
            .filter(|&i| node.after.contains(&i) || conflicts(&nodes[i], node))
            .map(|i| position[i])
            .collect();
        // The nearest producers first. A producer implies only the ones before it, so one pass drops them all.
        deps.sort_unstable_by(|a, b| b.cmp(a));
        let mut kept: Vec<usize> = Vec::new();
        let mut implied = vec![false; out.len()];
        for dep in deps {
            if implied[dep] {
                continue;
            }
            kept.push(dep);
            implied[dep] = true;
            for (flag, reached) in implied.iter_mut().zip(&reach[dep]) {
                *flag |= *reached;
            }
        }
        // Two waits fit. Each more needs a JOIN, which waits on the two oldest and stands in for both.
        while kept.len() > 2 {
            let (x, y) = (
                kept.pop().expect("more than two waits"),
                kept.pop().expect("more than two waits"),
            );
            let join = Planned {
                origin: None,
                items: 0,
                waits: [x, y].iter().map(|&p| wait_on(&out, p)).collect(),
            };
            let mut reached = vec![false; out.len()];
            for p in [x, y] {
                reached[p] = true;
                for (flag, r) in reached.iter_mut().zip(&reach[p]) {
                    *flag |= *r;
                }
            }
            out.push(join);
            reach.push(reached);
            kept.push(out.len() - 1);
        }
        // `kept` has the nearest first. Order the waits by event.
        kept.sort_unstable();
        let mut reached = vec![false; out.len()];
        for &p in &kept {
            reached[p] = true;
            for (flag, r) in reached.iter_mut().zip(&reach[p]) {
                *flag |= *r;
            }
        }
        position.push(out.len());
        out.push(Planned {
            origin: Some(j),
            items: node.items,
            waits: kept.iter().map(|&p| wait_on(&out, p)).collect(),
        });
        reach.push(reached);
    }
    assert_waits_backward(&out);
    out
}

fn wait_on(planned: &[Planned], producer: usize) -> Wait {
    Wait {
        event: producer as u32,
        target: signals(planned[producer].items),
    }
}

/// Panic if a wait of the plan does not point to an earlier task, or if a task has more than two waits.
pub fn assert_waits_backward(planned: &[Planned]) {
    for (index, task) in planned.iter().enumerate() {
        assert!(
            task.waits.len() <= 2,
            "planned task {index} has {} waits, and a descriptor has room for 2. Insert a JOIN before it",
            task.waits.len()
        );
        for wait in &task.waits {
            assert!(
                (wait.event as usize) < index,
                "planned task {index} waits on event {}, which is not before it in the ring. A wait must point backward",
                wait.event
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u64 = 0x1000;
    const B: u64 = 0x2000;
    const C: u64 = 0x3000;
    const D: u64 = 0x4000;
    const LEN: u64 = 0x100;

    fn span(at: u64) -> Span {
        Span::new(at, LEN)
    }

    fn node(items: u32, reads: &[u64], writes: &[u64]) -> Node {
        Node {
            items,
            reads: reads.iter().map(|&a| span(a)).collect(),
            writes: writes.iter().map(|&a| span(a)).collect(),
            after: Vec::new(),
        }
    }

    fn waits(plan: &[Planned], task: usize) -> Vec<(u32, u32)> {
        plan[task].waits.iter().map(|w| (w.event, w.target)).collect()
    }

    #[test]
    fn a_diamond_gets_exactly_its_waits_and_the_far_producer_is_implied() {
        // t0 writes A. t1 and t2 read A and write B and C. t3 reads A, B and C. t3 needs no wait on t0.
        let nodes = [
            node(4, &[], &[A]),
            node(6, &[A], &[B]),
            node(8, &[A], &[C]),
            node(2, &[A, B, C], &[D]),
        ];
        let plan = plan(&nodes);
        assert_eq!(plan.len(), 4);
        assert_eq!(waits(&plan, 0), vec![]);
        assert_eq!(waits(&plan, 1), vec![(0, 4)]);
        assert_eq!(waits(&plan, 2), vec![(0, 4)]);
        assert_eq!(waits(&plan, 3), vec![(1, 6), (2, 8)]);
    }

    #[test]
    fn a_declared_read_makes_a_wait_and_no_read_makes_none() {
        let with = [node(3, &[], &[A]), node(5, &[A], &[B])];
        assert_eq!(waits(&plan(&with), 1), vec![(0, 3)]);
        let without = [node(3, &[], &[A]), node(5, &[C], &[B])];
        assert_eq!(waits(&plan(&without), 1), vec![]);
    }

    #[test]
    fn each_kind_of_hazard_makes_a_wait() {
        // Read after write.
        assert_eq!(waits(&plan(&[node(2, &[], &[A]), node(2, &[A], &[])]), 1), vec![(0, 2)]);
        // Write after read: the later task must not overwrite what the earlier task still reads.
        assert_eq!(waits(&plan(&[node(2, &[A], &[]), node(2, &[], &[A])]), 1), vec![(0, 2)]);
        // Write after write.
        assert_eq!(waits(&plan(&[node(2, &[], &[A]), node(2, &[], &[A])]), 1), vec![(0, 2)]);
        // Two reads of one range are no hazard.
        assert_eq!(waits(&plan(&[node(2, &[A], &[]), node(2, &[A], &[])]), 1), vec![]);
    }

    #[test]
    fn ranges_that_only_touch_or_differ_make_no_wait() {
        let first = Node {
            items: 1,
            writes: vec![Span::new(A, 0x100)],
            ..Node::default()
        };
        let touching = Node {
            items: 1,
            reads: vec![Span::new(A + 0x100, 0x100)],
            ..Node::default()
        };
        let inside = Node {
            items: 1,
            reads: vec![Span::new(A + 0xff, 1)],
            ..Node::default()
        };
        assert_eq!(waits(&plan(&[first.clone(), touching]), 1), vec![]);
        assert_eq!(waits(&plan(&[first, inside]), 1), vec![(0, 1)]);
    }

    #[test]
    fn a_producer_that_an_earlier_wait_implies_is_dropped() {
        // t2 reads what t0 and t1 write, but t1 already waits on t0.
        let nodes = [node(2, &[], &[A]), node(3, &[A], &[B]), node(4, &[A, B], &[C])];
        assert_eq!(waits(&plan(&nodes), 2), vec![(1, 3)]);
    }

    #[test]
    fn three_producers_get_a_join_that_has_no_items() {
        let nodes = [
            node(4, &[], &[A]),
            node(5, &[], &[B]),
            node(6, &[], &[C]),
            node(7, &[A, B, C], &[D]),
        ];
        let plan = plan(&nodes);
        assert_eq!(plan.len(), 5);
        let join = &plan[3];
        assert_eq!((join.origin, join.items), (None, 0));
        assert_eq!(waits(&plan, 3), vec![(0, 4), (1, 5)]);
        assert_eq!(plan[4].origin, Some(3));
        // The consumer waits on the JOIN, which counts 1, and on the producer that is left.
        assert_eq!(waits(&plan, 4), vec![(2, 6), (3, 1)]);
    }

    #[test]
    fn five_producers_get_three_joins_in_a_chain() {
        let mut nodes: Vec<Node> = (0..5).map(|i| node(2, &[], &[A + 0x1000 * i])).collect();
        let reads: Vec<u64> = (0..5).map(|i| A + 0x1000 * i).collect();
        nodes.push(node(1, &reads, &[]));
        let plan = plan(&nodes);
        let joins = plan.iter().filter(|t| t.origin.is_none()).count();
        assert_eq!(joins, 3);
        assert!(plan.iter().all(|t| t.waits.len() <= 2));
    }

    #[test]
    #[should_panic(expected = "A wait must point backward")]
    fn a_forward_wait_panics() {
        let mut late = node(1, &[], &[]);
        late.after = vec![2];
        let _ = plan(&[node(1, &[], &[]), late, node(1, &[], &[])]);
    }

    #[test]
    #[should_panic(expected = "A wait must point backward")]
    fn a_wait_on_itself_panics() {
        let mut me = node(1, &[], &[]);
        me.after = vec![0];
        let _ = plan(&[me]);
    }

    #[test]
    #[should_panic(expected = "A wait must point backward")]
    fn a_planned_forward_wait_panics() {
        let planned = [
            Planned {
                origin: Some(0),
                items: 1,
                waits: vec![Wait { event: 1, target: 1 }],
            },
            Planned {
                origin: Some(1),
                items: 1,
                waits: vec![],
            },
        ];
        assert_waits_backward(&planned);
    }

    #[test]
    #[should_panic(expected = "has room for 2")]
    fn three_waits_in_one_descriptor_panic() {
        let waits = (0..3).map(|event| Wait { event, target: 1 }).collect();
        let planned: Vec<Planned> = (0..3)
            .map(|i| Planned {
                origin: Some(i),
                items: 1,
                waits: vec![],
            })
            .chain([Planned {
                origin: Some(3),
                items: 1,
                waits,
            }])
            .collect();
        assert_waits_backward(&planned);
    }

    #[test]
    #[should_panic(expected = "count of items from the plan")]
    fn a_count_from_the_plan_has_no_target() {
        let _ = plan(&[node(u32::MAX, &[], &[A]), node(1, &[A], &[])]);
    }

    #[test]
    fn explicit_order_makes_a_chain_without_ranges() {
        let mut nodes: Vec<Node> = (0..4).map(|_| node(3, &[], &[])).collect();
        for (i, n) in nodes.iter_mut().enumerate().skip(1) {
            n.after = vec![i - 1];
        }
        let plan = plan(&nodes);
        for i in 1..4 {
            assert_eq!(waits(&plan, i), vec![(i as u32 - 1, 3)]);
        }
    }

    /// A small deterministic generator, so that the random graphs are the same on each run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % bound
        }
    }

    #[test]
    fn random_graphs_keep_every_hazard_ordered_with_two_waits() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for round in 0..300 {
            let count = 2 + rng.next(14) as usize;
            let nodes: Vec<Node> = (0..count)
                .map(|_| {
                    let pick = |rng: &mut Rng| -> Vec<Span> {
                        (0..rng.next(3)).map(|_| span(A + 0x80 * rng.next(8))).collect()
                    };
                    Node {
                        items: 1 + rng.next(5) as u32,
                        reads: pick(&mut rng),
                        writes: pick(&mut rng),
                        after: Vec::new(),
                    }
                })
                .collect();
            let planned = plan(&nodes);
            // Close the waits over the plan: ordered[t][p] is true when t is after p, directly or not.
            let mut ordered = vec![vec![false; planned.len()]; planned.len()];
            for t in 0..planned.len() {
                assert!(planned[t].waits.len() <= 2, "round {round}");
                for w in &planned[t].waits {
                    let p = w.event as usize;
                    assert!(p < t, "round {round}");
                    assert_eq!(w.target, signals(planned[p].items), "round {round}");
                    ordered[t][p] = true;
                    for q in 0..p {
                        ordered[t][q] |= ordered[p][q];
                    }
                }
            }
            let at: Vec<usize> = (0..count)
                .map(|j| planned.iter().position(|t| t.origin == Some(j)).unwrap())
                .collect();
            for j in 0..count {
                for i in 0..j {
                    if conflicts(&nodes[i], &nodes[j]) {
                        assert!(
                            ordered[at[j]][at[i]],
                            "round {round}: task {j} has a hazard with task {i} and no wait that orders them"
                        );
                    }
                }
            }
        }
    }
}
