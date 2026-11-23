use nagaya_abi::{ControlItem, ControlOp, LayerLoop, OutputsWanted, PartKind};

use super::*;
use crate::spec::{self, DEFAULT_CYCLES, DEFAULT_EXTRA_RUNS, MAX_EXTRA_RUNS};

const F32: CacheFormat = CacheFormat::F32;

fn segment(slot: u32, from: u64, tokens: &[u32], logits: &[u64]) -> Segment {
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (from..from + tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted {
            logit_rows: logits.to_vec(),
        },
        layer_loop: None,
        cycles: None,
    }
}

/// The part ids of `k.04` (a ring), `k.05` (cells), `conv.00`, and `k.x00`, the ring of layer 6 in cycle 2.
const RING: PartId = 12;
const CELLS: PartId = 15;
const CONV: PartId = 2;
const RESERVE_RING: PartId = 72;
/// The part `cycles`: the last part of a state, `u32 [1]`.
const CYCLES_24: PartId = 3 * 48;

#[test]
fn the_parts_have_the_order_and_the_names_that_the_spec_declares() {
    let list = part_list(DEFAULT_EXTRA_RUNS);
    assert_eq!(
        (list[RING], list[CELLS], list[CONV], list[RESERVE_RING]),
        (Part::K(4), Part::K(5), Part::Conv(0), Part::K(24))
    );
    assert_eq!((list.len(), list[CYCLES_24]), (CYCLES_24 + 1, Part::Cycles));
    for format in [CacheFormat::F32, CacheFormat::Bf16, CacheFormat::E4m3] {
        let declared = spec::parts(DEFAULT_EXTRA_RUNS, format);
        assert_eq!(declared.len(), list.len());
        for (id, (part, limit)) in list.iter().zip(&declared).enumerate() {
            let name = |kind: &str, s: usize| match s.checked_sub(LAYERS) {
                None => format!("{kind}.{s:02}"),
                Some(i) => format!("{kind}.x{i:02}"),
            };
            let want = match part {
                Part::K(s) => name("k", *s),
                Part::V(s) => name("v", *s),
                Part::Conv(s) => name("conv", *s),
                Part::Cycles => "cycles".to_owned(),
            };
            assert_eq!(limit.name, want, "part {id}");
            if let Part::K(s) | Part::V(s) = part {
                let kind = if is_full(state_layer(*s)) {
                    PartKind::Cells
                } else {
                    PartKind::Ring
                };
                assert_eq!(limit.kind, kind, "part {id}");
                // A row of the part has the bytes that the dtype and the shape give.
                let bytes = limit.shape.iter().product::<u64>() * limit.dtype.element_bytes();
                assert_eq!(bytes, row_bytes(format), "part {id} in {format:?}");
            }
        }
    }
    assert_eq!(
        [F32, CacheFormat::Bf16, CacheFormat::E4m3].map(row_bytes),
        [4096, 2048, 1024]
    );
}

#[test]
fn a_pass_runs_on_from_the_token_count_and_never_goes_back() {
    let mut held = Held::new(&[1024, 4096], DEFAULT_EXTRA_RUNS, F32);
    let block = held.check_pass(&[segment(0, 0, &[1, 2, 3], &[2])], 512).unwrap();
    assert_eq!((block.slot, block.pos0, block.logit_rows.clone()), (0, 0, vec![2]));
    assert!(
        block.zero_conv(),
        "a pass from position 0 starts from conv states of zeros"
    );
    held.ran(&block);
    assert_eq!(held.n[0], 3);
    let next = held.check_pass(&[segment(0, 3, &[4], &[3])], 512).unwrap();
    assert!(!next.zero_conv());
    let back = held.check_pass(&[segment(0, 2, &[4], &[])], 512).unwrap_err();
    assert!(back.contains("does not go back"), "{back}");
    let gap = held.check_pass(&[segment(0, 4, &[4], &[])], 512).unwrap_err();
    assert!(gap.contains("run from 3"), "{gap}");
    let mut control = segment(0, 3, &[1], &[]);
    control.control = vec![ControlItem {
        index: 0,
        op: ControlOp::Add,
        strength: 1.0,
        layers: None,
    }];
    let mut looped = segment(0, 3, &[1], &[]);
    looped.layer_loop = Some(LayerLoop {
        start: 6,
        end: 18,
        count: 2,
    });
    let mut cycled = segment(0, 3, &[1], &[]);
    cycled.cycles = Some(3);
    let mut many = segment(0, 3, &[1], &[]);
    many.cycles = Some(4);
    let mut none = segment(0, 3, &[1], &[]);
    none.cycles = Some(0);
    let mut meta = segment(0, 3, &[1], &[]);
    meta.meta = vec![0; 16];
    let rows = Segment {
        inputs: SegmentInputs::TokensAndRows {
            tokens: vec![1],
            rows: vec![0; 16],
        },
        ..segment(0, 3, &[1], &[])
    };
    let only_rows = Segment {
        inputs: SegmentInputs::Rows(vec![0; 16]),
        ..segment(0, 3, &[1], &[])
    };
    let mut positions = segment(0, 3, &[1, 2], &[]);
    positions.positions.pop();
    for (segments, why) in [
        (vec![], "0 segments"),
        (vec![segment(0, 3, &[1], &[]), segment(1, 0, &[1], &[])], "2 segments"),
        (vec![segment(2, 0, &[1], &[])], "slots"),
        (vec![segment(0, 3, &[], &[])], "0 tokens"),
        (vec![segment(0, 3, &[1; 513], &[])], "513 tokens"),
        (vec![segment(0, 3, &[200_058], &[])], "200058"),
        (vec![segment(0, 3, &[200_057], &[])], ""),
        (vec![segment(0, 3, &[1], &[4])], "logit row at 4"),
        (
            vec![segment(0, 3, &[1; 17], &(3..20).collect::<Vec<_>>())],
            "17 logit rows",
        ),
        (vec![segment(0, 3, &[1; 16], &(3..19).collect::<Vec<_>>())], ""),
        (vec![segment(0, 3, &[1], &[])], ""),
        (vec![control], "control items"),
        (vec![looped], "layer_loops"),
        (vec![cycled], ""),
        (vec![many], "runs 4 cycles; the engine runs 1 to 3"),
        (vec![none], "runs 0 cycles"),
        (vec![meta], "meta"),
        (vec![rows], "tokens only"),
        (vec![only_rows], "tokens only"),
        (vec![positions], "2 tokens and 1 positions"),
        (vec![segment(1, 0, &[1; 512], &[])], ""),
    ] {
        let got = held.check_pass(&segments, 512);
        if why.is_empty() {
            assert!(got.is_ok(), "{got:?}");
        } else {
            let error = got.unwrap_err();
            assert!(error.contains(why), "{why}: {error}");
        }
    }
    let mut full = Held::new(&[8], DEFAULT_EXTRA_RUNS, F32);
    full.n[0] = 6;
    let e = full.check_pass(&[segment(0, 6, &[1, 2, 3], &[])], 512).unwrap_err();
    assert!(e.contains("do not fit"), "{e}");
    // A check changes nothing.
    assert_eq!((held.n.clone(), held.cycles.clone()), (vec![3, 0], vec![Some(3), None]));
}

#[test]
fn a_ring_holds_its_last_rows_and_a_short_slot_holds_every_position() {
    // A slot of 1,536 positions or fewer holds every position in its sliding states; a longer slot wraps its ring.
    assert!(!ringed(1_536, 4) && ringed(1_537, 4));
    assert!(!ringed(409_600, 5), "a full state is never a ring");
    let mut held = Held::new(&[1_536, 4096], DEFAULT_EXTRA_RUNS, F32);
    held.n = vec![1_000, 3_000];
    assert_eq!(held.base(0, 4), 0);
    assert_eq!(held.base(1, 4), 3_000 - 1_536, "the ring is 1,536 rows, not 2,048");
    assert_eq!(
        held.base(1, 24),
        3_000 - 1_536,
        "the reserve entry of layer 6 is a ring"
    );
    assert_eq!(held.base(1, 5), 0);
    assert_eq!(held.base(1, 24 + 5), 0, "the reserve entry of layer 11 holds cells");
    let row = vec![0u8; 4096];
    let span = held.coverage(State::Slot(1), RING, &row.repeat(1_536));
    assert_eq!((span.start, span.end), (1_464, 3_000));
    assert_eq!(held.coverage(State::Slot(1), CONV, &[0; 64]).end, 0);
    let fit = |slot, part, start, end| held.fits(slot, &CoverageSpan { part, start, end });
    // A ring needs the window of 1,024 positions before its end.
    assert!(fit(1, RING, 1976, 3000).is_ok());
    assert!(fit(1, RESERVE_RING, 1976, 3000).is_ok());
    assert!(fit(1, RING, 0, 3000).is_ok());
    let e = fit(1, RING, 1977, 3000).unwrap_err();
    assert!(e.contains("window of 1024"), "{e}");
    assert!(fit(0, RING, 1, 1000).unwrap_err().contains("from 0"));
    assert!(fit(1, CELLS, 1976, 3000).unwrap_err().contains("from 0"));
    assert!(fit(0, CELLS, 0, 1_537).unwrap_err().contains("do not fit"));
    assert!(fit(0, CONV, 0, 0).unwrap_err().contains("conv state"));
    assert!(fit(0, CYCLES_24, 0, 0).unwrap_err().contains("count of cycles"));
    assert!(fit(5, CELLS, 0, 0).unwrap_err().contains("no slot 5"));
}

#[test]
fn a_restore_writes_every_part_and_set_n_takes_the_count_only_then() {
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    held.n[0] = 50;
    let state = State::Slot(0);
    let row = 4096;
    // With no write, a count stays or goes to 0, and never goes back to another count.
    assert!(held.check_set_n(state, 50).is_ok());
    assert!(held.check_set_n(state, 0).is_ok());
    assert!(held.check_set_n(state, 49).unwrap_err().contains("does not go back"));
    assert!(held.check_set_n(state, 4097).unwrap_err().contains("4096 positions"));
    // A cells part starts at 0; a ring part of a long slot keeps the last 1,536 rows of a write.
    assert!(
        held.check_cells_write(state, CELLS, 1, 1, row)
            .unwrap_err()
            .contains("from 0")
    );
    let (_, _, kept) = held.check_cells_write(state, RING, 0, 3000, 3000 * row).unwrap();
    assert_eq!(kept, 1_464..3000);
    assert!(held.check_cells_write(state, RING, 0, 2, row).is_err());
    assert!(held.check_cells_write(state, RING, 4000, 97, 97 * row).is_err());
    assert!(
        held.check_cells_write(state, CONV, 0, 1, row)
            .unwrap_err()
            .contains("fixed part")
    );
    assert!(
        held.check_part_write(state, CELLS, &[0; 79_872])
            .unwrap_err()
            .contains("row part")
    );
    assert!(held.check_part_write(state, CONV, &[0; 79_871]).is_err());
    let list = part_list(DEFAULT_EXTRA_RUNS);
    for (id, part) in list.iter().enumerate() {
        if *part == Part::Cycles {
            held.wrote_count(0, 3);
        } else if matches!(part, Part::Conv(_)) {
            held.check_part_write(state, id, &[0; 79_872]).unwrap();
            held.wrote(0, id, None, false);
        } else if id != RING {
            let start = if is_full(state_layer(part.state())) {
                0
            } else {
                1500 - 1024
            };
            held.wrote(0, id, Some(start..1500), false);
        }
    }
    assert_eq!(held.n[0], 0, "a write empties the slot until set_n");
    let e = held.check_set_n(state, 1500).unwrap_err();
    assert!(e.contains(&format!("part {RING} got no bytes")), "{e}");
    held.wrote(0, RING, Some(477..1500), false);
    assert!(held.check_set_n(state, 1500).unwrap_err().contains("window"));
    held.wrote(0, RING, Some(0..1499), false);
    assert!(held.check_set_n(state, 1500).unwrap_err().contains("up to 1499"));
    held.wrote(0, RING, None, false);
    assert!(held.check_set_n(state, 1500).unwrap_err().contains("got no rows"));
    held.wrote(0, RING, Some(476..1500), false);
    assert_eq!(held.check_set_n(state, 1500), Ok(0));
    held.set_n(0, 1500);
    assert_eq!(held.n[0], 1500);
    assert!(held.check_set_n(state, 1499).is_err(), "set_n took the writes");
    // The rings hold no position before the restored rows, and a later pass moves the base on.
    assert_eq!((held.base(0, 4), held.base(0, 5), held.base(0, 30)), (476, 0, 476));
    assert_eq!(held.ring_from(0), 476);
    held.n[0] = 3000;
    assert_eq!(held.base(0, 4), 1_464);
    held.clear(0);
    held.n[0] = 100;
    assert_eq!(held.base(0, 4), 0, "a clear forgets the restored start");
    assert_eq!(held.ring_from(0), 0);
}

#[test]
fn a_read_stays_inside_the_rows_that_a_part_holds() {
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    held.n[0] = 3000;
    let state = State::Slot(0);
    let row = 4096;
    assert!(held.check_read(state, RING, 1976, 1024, 1024 * row).is_ok());
    assert!(held.check_read(state, RING, 1_464, 1_536, 1_536 * row).is_ok());
    assert!(
        held.check_read(state, RING, 1_463, 1, row)
            .unwrap_err()
            .contains("1464")
    );
    assert!(held.check_read(state, CELLS, 0, 3001, 3001 * row).is_err());
    assert!(held.check_read(state, CELLS, 0, 2, row).is_err());
    assert!(
        held.check_read(state, CONV, 0, 1, row)
            .unwrap_err()
            .contains("fixed part")
    );
    assert_eq!(held.check_part_read(state, RING, 1024 * row).unwrap().2, 1_464);
    assert!(
        held.check_part_read(state, RING, 1000)
            .unwrap_err()
            .contains("whole rows")
    );
    assert!(held.check_part_read(state, CONV, 79_872).is_ok());
    assert!(held.check_part_read(state, CONV, row).is_err());
    assert!(held.check_part_read(State::Slot(1), CONV, 79_872).is_err());
    assert!(
        held.check_part_read(state, 3 * 48 + 1, row)
            .unwrap_err()
            .contains("no part 145")
    );
}

/// The bytes of a row follow the format of the cache, so a read of the same rows takes fewer bytes in a small format.
#[test]
fn a_row_has_the_bytes_of_the_cache_format() {
    for (format, row) in [
        (CacheFormat::F32, 4096),
        (CacheFormat::Bf16, 2048),
        (CacheFormat::E4m3, 1024),
    ] {
        let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, format);
        held.n[0] = 100;
        let state = State::Slot(0);
        assert_eq!(held.row_bytes(), row as u64);
        assert!(held.check_read(state, CELLS, 0, 100, 100 * row).is_ok(), "{format:?}");
        assert!(held.check_read(state, CELLS, 0, 100, 100 * 4096).is_err() || format == CacheFormat::F32);
        assert_eq!(held.check_part_read(state, CELLS, 100 * row).unwrap().2, 0);
        assert!(held.check_cells_write(state, CELLS, 0, 100, 100 * row).is_ok());
        assert!(held.check_cells_write(state, CELLS, 0, 100, 100 * 4096).is_err() || format == CacheFormat::F32);
        let span = held.coverage(state, CELLS, &vec![0; 100 * row]);
        assert_eq!((span.start, span.end), (0, 100));
    }
}

#[test]
fn a_cache_holds_only_finite_values_in_its_format() {
    let f32_bytes = |v: f32| v.to_le_bytes().to_vec();
    assert!(check_values(CacheFormat::F32, &[f32_bytes(1.5), f32_bytes(-0.0)].concat()).is_ok());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let e = check_values(CacheFormat::F32, &[f32_bytes(1.0), f32_bytes(bad)].concat()).unwrap_err();
        assert!(e.contains("value 1 is a NaN or an infinity"), "{e}");
    }
    let bf16 = |bits: u16| bits.to_le_bytes().to_vec();
    assert!(check_values(CacheFormat::Bf16, &[bf16(0x3f80), bf16(0x7f7f)].concat()).is_ok());
    for bad in [0x7f80u16, 0xff80, 0x7fc0] {
        assert!(
            check_values(CacheFormat::Bf16, &[bf16(0), bf16(bad)].concat()).is_err(),
            "{bad:#x}"
        );
    }
    // The largest finite e4m3 value is 0x7e (448). The code 0x7f is the NaN, and the format has no infinity.
    assert!(check_values(CacheFormat::E4m3, &[0x00, 0x7e, 0xfe, 0x80]).is_ok());
    for bad in [0x7fu8, 0xff] {
        assert!(check_values(CacheFormat::E4m3, &[0, bad]).is_err(), "{bad:#x}");
    }
    // A length that is not a whole count of values is an error for each format.
    assert!(check_values(CacheFormat::F32, &[0; 6]).is_err());
    assert!(check_values(CacheFormat::Bf16, &[0; 3]).is_err());
    assert!(check_values(CacheFormat::E4m3, &[]).is_ok());
    // A conv write is `f32`, so a NaN in it is refused with the size right.
    let held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, CacheFormat::E4m3);
    let mut conv = vec![0u8; CONV_BYTES as usize];
    assert!(held.check_part_write(State::Slot(0), CONV, &conv).is_ok());
    conv[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
    let e = held.check_part_write(State::Slot(0), CONV, &conv).unwrap_err();
    assert!(e.contains("NaN"), "{e}");
}

/// A pass runs the count of its segment, or the default. A slot keeps the count that fed its positions, because each
/// run of a cycle reads the keys of its own run at the earlier positions (engine-abi "Inkling 2").
#[test]
fn a_slot_runs_the_count_of_cycles_that_fed_it() {
    let mut held = Held::new(&[1024, 4096], MAX_EXTRA_RUNS, F32);
    let counted = |slot, from, n: usize, cycles| Segment {
        cycles,
        ..segment(slot, from, &vec![1; n], &[])
    };
    let block = held.check_pass(&[counted(0, 0, 3, None)], 512).unwrap();
    assert_eq!(block.cycles, DEFAULT_CYCLES);
    let block = held.check_pass(&[counted(0, 0, 3, Some(5))], 512).unwrap();
    assert_eq!(block.cycles, 5);
    held.ran(&block);
    assert_eq!(held.cycles[0], Some(5));
    let e = held.check_pass(&[counted(0, 3, 1, None)], 512).unwrap_err();
    assert!(
        e.contains("slot 0 holds 3 positions of 5 cycles, and the pass runs 3"),
        "{e}"
    );
    let e = held.check_pass(&[counted(0, 3, 1, Some(2))], 512).unwrap_err();
    assert!(e.contains("feed the slot at 5 cycles"), "{e}");
    assert_eq!(held.check_pass(&[counted(0, 3, 1, Some(5))], 512).unwrap().cycles, 5);
    let e = held.check_pass(&[counted(0, 3, 1, Some(6))], 512).unwrap_err();
    assert!(e.contains("runs 6 cycles; the engine runs 1 to 5"), "{e}");
    // The other slot is empty, so it takes any count.
    assert_eq!(held.check_pass(&[counted(1, 0, 2, Some(1))], 512).unwrap().cycles, 1);
    // A clear empties the slot, so the next pass takes another count.
    held.clear(0);
    assert_eq!(held.cycles[0], None);
    let block = held.check_pass(&[counted(0, 0, 2, Some(2))], 512).unwrap();
    held.ran(&block);
    assert_eq!(held.cycles[0], Some(2));
    // A restore writes parts and a count. The bytes do not say their count, so the next pass sets it; the council
    // keys a state by its count (spec 11.17).
    held.wrote(0, CONV, None, false);
    assert_eq!(held.cycles[0], None);
    held.set_n(0, 0);
    assert_eq!(held.cycles[0], None);
}

/// Feed `n` positions of `cycles` cycles into `slot` of `held`, in blocks of 512.
fn feed(held: &mut Held, slot: u32, n: usize, cycles: u8) {
    let mut at = 0;
    while at < n {
        let take = (n - at).min(512);
        let s = Segment {
            cycles: Some(cycles),
            ..segment(slot, at as u64, &vec![1; take], &[])
        };
        let block = held.check_pass(&[s], 512).unwrap();
        held.ran(&block);
        at += take;
    }
}

/// The parts of a run that the count of the slot does not reach read as zeros, and the part `cycles` reads the count
/// (engine-abi "Inkling 2", "The parts that a count does not reach"). Thus one recipe gives one content id.
#[test]
fn an_inactive_run_reads_as_zeros_and_the_count_is_a_part() {
    let list = part_list(DEFAULT_EXTRA_RUNS);
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    let state = State::Slot(0);
    // An empty slot reads the count 0.
    assert_eq!(held.check_part_read(state, CYCLES_24, 4).unwrap().3, Source::Count(0));
    assert!(held.check_part_read(state, CYCLES_24, 8).is_err());
    feed(&mut held, 0, 3000, 1);
    assert_eq!(held.check_part_read(state, CYCLES_24, 4).unwrap().3, Source::Count(1));
    assert!(held.active(0, 23) && !held.active(0, 24) && !held.active(0, 47));
    let row = held.row_bytes() as usize;
    // A run of layer 4 is active; the ring of layer 6 in cycle 2 is not.
    assert_eq!(
        held.check_read(state, RING, 1_464, 1_536, 1_536 * row).unwrap().2,
        Source::Device
    );
    assert_eq!(
        held.check_read(state, RESERVE_RING, 1_464, 1_536, 1_536 * row)
            .unwrap()
            .2,
        Source::Zeros
    );
    assert_eq!(
        held.check_part_read(state, CONV, CONV_BYTES as usize).unwrap().3,
        Source::Device
    );
    let reserve_conv = RESERVE_RING + 2;
    assert_eq!(list[reserve_conv], Part::Conv(24));
    assert_eq!(
        held.check_part_read(state, reserve_conv, CONV_BYTES as usize)
            .unwrap()
            .3,
        Source::Zeros
    );
    // The cells of layer 11 in cycle 2 read as zeros from position 0.
    let reserve_cells = 3 * (24 + 5);
    assert_eq!(
        held.check_part_read(state, reserve_cells, 3000 * row).unwrap(),
        (0, Part::K(29), 0, Source::Zeros)
    );
    // At 2 cycles, the first 12 entries of the reserve are active.
    let mut two = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    feed(&mut two, 0, 10, 2);
    assert!(two.active(0, 35) && !two.active(0, 36));
    assert_eq!(two.check_part_read(state, CYCLES_24, 4).unwrap().3, Source::Count(2));
    assert_eq!(
        two.check_read(state, RESERVE_RING, 0, 10, 10 * row).unwrap().2,
        Source::Device
    );
}

/// The parts of a restore of `n` positions at `cycles` cycles: each part with its rows, zeros for an inactive part.
fn restore(held: &mut Held, n: u64, cycles: u32) {
    let list = held.list.clone();
    for (id, part) in list.iter().enumerate() {
        match part {
            Part::Cycles => held.wrote_count(0, cycles),
            Part::Conv(s) => held.wrote(0, id, None, !held.runs_at(cycles as usize, *s)),
            Part::K(s) | Part::V(s) => {
                let start = if is_full(state_layer(*s)) {
                    0
                } else {
                    n.saturating_sub(RING_ROWS as u64)
                };
                held.wrote(0, id, Some(start..n), !held.runs_at(cycles as usize, *s));
            }
        }
    }
}

/// The active rings of a restore start at one position, so the base of the slot is the start of each of them. A ring
/// that starts later would hide rows of the others from the reads after the restore (engine-abi "Inkling 2", "A
/// restore").
#[test]
fn the_active_rings_of_a_restore_start_at_one_position() {
    let state = State::Slot(0);
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut held, 3000, 1);
    // Each ring alone fits: 1976 is the start of the window of 1,024 positions before 3000.
    held.wrote(0, RING, Some(1976..3000), false);
    let e = held.check_set_n(state, 3000).unwrap_err();
    assert!(
        e.contains("the active rings of a restore must start at one position")
            && e.contains("1976")
            && e.contains("1464"),
        "{e}"
    );
    // An inactive ring holds zeros, and its start does not count.
    let mut inactive = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut inactive, 3000, 1);
    inactive.wrote(0, RESERVE_RING, Some(1976..3000), true);
    assert_eq!(inactive.check_set_n(state, 3000), Ok(0));
    inactive.set_n(0, 3000);
    assert_eq!(inactive.base(0, 4), 1_464);
}

/// A restore writes the count, and `set_n` takes it with the positions. An inactive part must hold zeros (engine-abi
/// "Inkling 2", "A restore").
#[test]
fn a_restore_keeps_the_count_and_needs_zeros_in_the_inactive_parts() {
    let state = State::Slot(0);
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut held, 3000, 1);
    assert_eq!(held.check_set_n(state, 3000), Ok(0));
    held.set_n(0, 3000);
    assert_eq!((held.n[0], held.cycles[0]), (3000, Some(1)));
    // The next pass runs at the count of the snapshot, not at the default.
    let e = held.check_pass(&[segment(0, 3000, &[1], &[])], 512).unwrap_err();
    assert!(e.contains("3000 positions of 1 cycles, and the pass runs 3"), "{e}");
    // A restore with no count is refused.
    let mut none = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut none, 3000, 1);
    none.writes[0].remove(&CYCLES_24);
    let e = none.check_set_n(state, 3000).unwrap_err();
    assert!(e.contains(&format!("part {CYCLES_24} got no bytes")), "{e}");
    // A count of 0 restores only an empty slot.
    let mut zero = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut zero, 3000, 1);
    zero.wrote_count(0, 0);
    let e = zero.check_set_n(state, 3000).unwrap_err();
    assert!(e.contains("count of cycles 0"), "{e}");
    // An inactive part that does not hold zeros is refused, and the error names it.
    let mut dirty = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut dirty, 3000, 1);
    dirty.wrote(0, RESERVE_RING, Some(1_464..3000), false);
    let e = dirty.check_set_n(state, 3000).unwrap_err();
    assert!(
        e.contains(&format!("part {RESERVE_RING} is a run that 1 cycles do not reach")),
        "{e}"
    );
    // The same bytes at 2 cycles are an active part.
    let mut two = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut two, 3000, 2);
    two.wrote(0, RESERVE_RING, Some(1_464..3000), false);
    assert_eq!(two.check_set_n(state, 3000), Ok(0));
    // The write of a count checks its value, and a second count must be the same.
    let mut write = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    let count = |v: u32| v.to_le_bytes().to_vec();
    assert!(write.check_part_write(state, CYCLES_24, &count(3)).is_ok());
    let e = write.check_part_write(state, CYCLES_24, &count(4)).unwrap_err();
    assert!(e.contains("count of cycles 4; the engine runs 1 to 3"), "{e}");
    assert!(write.check_part_write(state, CYCLES_24, &[0; 8]).is_err());
    write.wrote_count(0, 2);
    assert!(write.check_part_write(state, CYCLES_24, &count(2)).is_ok());
    let e = write.check_part_write(state, CYCLES_24, &count(3)).unwrap_err();
    assert!(e.contains("got the count 2"), "{e}");
    // The rings start at the first restored row of an active ring. An inactive ring from 0 does not move it.
    let mut rings = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    restore(&mut rings, 1500, 1);
    for (id, part) in rings.list.clone().iter().enumerate() {
        if let Part::K(s) | Part::V(s) = part
            && ringed(4096, *s)
            && rings.runs_at(1, *s)
        {
            rings.wrote(0, id, Some(476..1500), false);
        }
    }
    rings.wrote(0, RESERVE_RING, Some(0..1500), true);
    assert_eq!(rings.check_set_n(state, 1500), Ok(0));
    rings.set_n(0, 1500);
    assert_eq!((rings.base(0, 4), rings.base(0, 24)), (476, 0));
}

/// A clear forgets the count, the writes and the start of the rings, and it keeps the positions of the other slots.
#[test]
fn a_clear_empties_one_slot_and_keeps_the_others() {
    let mut held = Held::new(&[4096, 4096], DEFAULT_EXTRA_RUNS, F32);
    feed(&mut held, 0, 600, 2);
    feed(&mut held, 1, 300, 1);
    held.wrote(0, CONV, None, false);
    held.clear(0);
    assert_eq!((held.n[0], held.cycles[0], held.count(0)), (0, None, 0));
    assert_eq!((held.n[1], held.cycles[1], held.count(1)), (300, Some(1), 1));
    assert_eq!(held.check_set_n(State::Slot(0), 0), Ok(0));
    assert!(
        held.check_set_n(State::Slot(0), 5).is_err(),
        "the write is gone, and 5 is a count that goes back"
    );
}

/// A probe of the oracle runs from position 0 of a slot that it empties. Its check comes first, so a probe that the
/// check refuses changes nothing (spec 8.3).
#[test]
fn a_probe_is_checked_before_it_empties_its_slot() {
    let mut held = Held::new(&[4096], DEFAULT_EXTRA_RUNS, F32);
    feed(&mut held, 0, 600, 1);
    let block = held.check_probe(0, &[1, 2, 3], 512).unwrap();
    assert_eq!((block.pos0, block.cycles), (0, DEFAULT_CYCLES));
    assert_eq!((held.n[0], held.cycles[0]), (600, Some(1)), "a check changes nothing");
    let e = held.check_probe(0, &[LOGITS as u32], 512).unwrap_err();
    assert!(e.contains("outside the vocabulary"), "{e}");
    let e = held.check_probe(1, &[1], 512).unwrap_err();
    assert!(e.contains("the engine has 1 slots"), "{e}");
    assert!(held.check_probe(0, &vec![1; 513], 512).is_err());
    // A slot past `u32` is not slot 0 of the probe segment.
    let e = held.check_probe(1 << 32, &[1], 512).unwrap_err();
    assert!(e.contains("the engine has 1 slots, not slot 4294967296"), "{e}");
}
