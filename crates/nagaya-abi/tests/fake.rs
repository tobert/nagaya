#![cfg(feature = "fake")]

use nagaya_abi::fake::{FakeAlways, FakeBroken};
use nagaya_abi::{
    Capability, Determinism, Engine, Lifetime, OutputsWanted, PartError, PartKind, PassStatus, Segment, SegmentInputs,
    State, Tier, WriteResult,
};

#[test]
fn fake_limits_has_four_part_kinds() {
    let limits = FakeAlways::new().limits();
    let names: Vec<&str> = limits.parts.iter().map(|part| part.name.as_str()).collect();
    assert_eq!(names, ["state", "cells", "ring", "pooled"]);
    let kinds: Vec<PartKind> = limits.parts.iter().map(|part| part.kind).collect();
    assert_eq!(
        kinds,
        [PartKind::Fixed, PartKind::Cells, PartKind::Ring, PartKind::Derived,]
    );
    assert_eq!(limits.parts[0].shape, vec![16]);
    assert_eq!(limits.parts[1].shape, vec![8]);
    assert_eq!(limits.parts[2].shape, vec![8]);
    assert_eq!(limits.parts[3].shape, vec![8]);
}

#[test]
fn fake_declares_two_slot_sizes() {
    let limits = FakeAlways::new().limits();
    let sizes: Vec<u64> = limits.slots.iter().map(|slot| slot.bytes).collect();
    assert_eq!(sizes, [64, 256]);
    assert_eq!(limits.slots.len(), 2);
    assert!(limits.slots.iter().all(|slot| slot.count == 1));
}

#[test]
fn fake_declares_one_mandatory_and_one_elastic_term() {
    let limits = FakeAlways::new().limits();
    assert_eq!(limits.memory.mandatory.len(), 1);
    let mandatory = &limits.memory.mandatory[0];
    assert_eq!(mandatory.name, "weights");
    assert_eq!(mandatory.tier, Tier::Device(0));
    assert_eq!(mandatory.bytes, 4096);
    assert_eq!(mandatory.lifetime, Lifetime::Engine);
    assert_eq!(limits.memory.elastic.len(), 1);
    let elastic = &limits.memory.elastic[0];
    assert_eq!(elastic.name, "workspace");
    assert_eq!(elastic.tier, Tier::Host);
    assert_eq!(elastic.unit_bytes, 256);
    assert_eq!(elastic.minimum, 1);
    assert_eq!(elastic.maximum, 8);
}

#[test]
fn fake_is_always_deterministic() {
    assert_eq!(FakeAlways::new().limits().determinism, Determinism::Always);
}

#[test]
fn fake_capabilities_match_accessors() {
    let mut engine = FakeAlways::new();
    assert_eq!(
        engine.limits().capabilities,
        vec![Capability::Parts, Capability::Coverage, Capability::Seal,]
    );
    assert!(engine.parts().is_some());
    assert!(engine.coverage().is_some());
    assert!(engine.seal().is_some());
    assert!(engine.slots().is_none());
}

#[test]
fn fixed_part_round_trips() {
    let mut engine = FakeAlways::new();
    let state = State::Slot(0);
    let bytes: Vec<u8> = (0u8..16).collect();
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_write(state, 0, &bytes), WriteResult::Ok);
    }
    let mut out = [0u8; 16];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_read(state, 0, &mut out), Ok(()));
    }
    assert_eq!(&out[..], &bytes[..]);
}

#[test]
fn cells_append_and_read_a_range() {
    let mut engine = FakeAlways::new();
    let state = State::Slot(1);
    let rows: Vec<u8> = (0u8..4).flat_map(|i| [i; 8]).collect();
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.cells_write(state, 1, 4, &rows), WriteResult::Ok);
    }
    let mut out = [0u8; 16];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.cells_read(state, 1, 1, 2, &mut out), Ok(()));
    }
    assert_eq!(&out[..], &rows[8..24]);
}

#[test]
fn set_n_limits_the_readable_rows() {
    let mut engine = FakeAlways::new();
    let state = State::Slot(1);
    let rows: Vec<u8> = (0u8..2).flat_map(|i| [i; 8]).collect();
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.cells_write(state, 1, 2, &rows), WriteResult::Ok);
        assert_eq!(parts.set_n(state, 1), WriteResult::Ok);
    }
    let mut out = [0u8; 8];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.cells_read(state, 1, 0, 1, &mut out), Ok(()));
        assert_eq!(parts.cells_read(state, 1, 1, 1, &mut out), Err(PartError::Refused));
    }
    assert_eq!(out, [0u8; 8]);
}

#[test]
fn ring_keeps_the_last_window() {
    let mut engine = FakeAlways::new();
    let state = State::Slot(1);
    {
        let parts = engine.parts().unwrap();
        for i in 0u8..6 {
            assert_eq!(parts.cells_write(state, 2, 1, &[i; 8]), WriteResult::Ok);
        }
    }
    let mut out = [0u8; 32];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.cells_read(state, 2, 2, 4, &mut out), Ok(()));
        assert_eq!(
            parts.cells_read(state, 2, 0, 4, &mut [0u8; 32]),
            Err(PartError::Refused)
        );
    }
    let expected: Vec<u8> = (2u8..6).flat_map(|i| [i; 8]).collect();
    assert_eq!(&out[..], &expected[..]);
}

#[test]
fn derived_part_is_not_stored() {
    let mut engine = FakeAlways::new();
    let state = State::Slot(0);
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_write(state, 3, &[0u8; 8]), WriteResult::Refused);
    }
    let mut out = [0u8; 8];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_read(state, 3, &mut out), Err(PartError::Refused));
    }
    assert!(engine.last_error().is_some());
}

#[test]
fn seal_flips_one_bit_of_the_fixed_part() {
    let mut engine = FakeAlways::new();
    let state = State::Slot(0);
    let before = [0u8; 16];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_write(state, 0, &before), WriteResult::Ok);
    }
    engine.seal().unwrap().seal(state).unwrap();
    let mut after = [0u8; 16];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_read(state, 0, &mut after), Ok(()));
    }
    assert_eq!(after[0], 1);
    assert_eq!(&before[1..], &after[1..]);
}

#[test]
fn coverage_fits_both_slot_sizes() {
    let mut engine = FakeAlways::new();
    let coverage = engine.coverage().unwrap();
    let small = coverage.coverage(State::Slot(0), 1, &[0u8; 8 * 8]);
    assert_eq!((small.start, small.end), (0, 8));
    assert!(coverage.fits(&small, 0).is_ok());
    let large = coverage.coverage(State::Slot(0), 1, &[0u8; 8 * 16]);
    assert_eq!((large.start, large.end), (0, 16));
    assert!(coverage.fits(&large, 0).is_err());
    assert!(coverage.fits(&large, 1).is_ok());
}

#[test]
fn broken_fake_loses_a_bit_on_restore() {
    let mut engine = FakeBroken::new();
    let state = State::Slot(0);
    let written = [0xAAu8; 16];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_write(state, 0, &written), WriteResult::Ok);
    }
    let mut read = [0u8; 16];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_read(state, 0, &mut read), Ok(()));
    }
    assert_eq!(read[0] ^ written[0], 1);
    assert_eq!(&read[1..], &written[1..]);
}

#[test]
fn broken_fake_has_the_same_limits() {
    assert_eq!(FakeBroken::new().limits(), FakeAlways::new().limits());
}

#[test]
fn an_empty_pass_is_ok_with_no_outputs() {
    let mut engine = FakeAlways::new();
    let out = engine.pass(&[], &[]).unwrap();
    assert_eq!(out.status, PassStatus::Ok);
    assert_eq!(out.ms, 0);
    assert!(out.outputs.is_empty());
    assert_eq!(engine.last_error(), None);
}

#[test]
fn fake_identity_names_the_model() {
    assert_eq!(FakeAlways::new().identity().model, "fake-always");
    assert_eq!(FakeBroken::new().identity().model, "fake-broken");
}

fn token_segment(tokens: &[u32]) -> Segment {
    Segment {
        state: State::Slot(0),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (0..tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
    }
}

fn read_state(engine: &mut dyn Engine) -> Vec<u8> {
    let mut out = [0u8; 16];
    engine.parts().unwrap().part_read(State::Slot(0), 0, &mut out).unwrap();
    out.to_vec()
}

#[test]
fn different_tokens_give_different_state() {
    let mut first = FakeAlways::new();
    first.pass(&[token_segment(&[1, 2, 3])], &[]).unwrap();
    let mut second = FakeAlways::new();
    second.pass(&[token_segment(&[4, 5, 6])], &[]).unwrap();
    assert_ne!(read_state(&mut first), read_state(&mut second));
}

#[test]
fn same_tokens_give_the_same_state() {
    let mut first = FakeAlways::new();
    let out_first = first.pass(&[token_segment(&[7, 8])], &[]).unwrap();
    let mut second = FakeAlways::new();
    let out_second = second.pass(&[token_segment(&[7, 8])], &[]).unwrap();
    assert_eq!(read_state(&mut first), read_state(&mut second));
    assert_eq!(out_first.outputs, out_second.outputs);
}

#[test]
fn a_pass_appends_cells_and_advances_the_ring() {
    let mut engine = FakeAlways::new();
    engine.pass(&[token_segment(&[1, 2, 3, 4, 5])], &[]).unwrap();
    let mut cells = [0u8; 5 * 8];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_read(State::Slot(0), 1, &mut cells), Ok(()));
    }
    let mut ring_tail = [0u8; 4 * 8];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.cells_read(State::Slot(0), 2, 1, 4, &mut ring_tail), Ok(()));
        assert_eq!(
            parts.cells_read(State::Slot(0), 2, 0, 1, &mut [0u8; 8]),
            Err(PartError::Refused)
        );
    }
}

#[test]
fn broken_corruption_is_visible_after_passes() {
    let mut good = FakeAlways::new();
    let mut broken = FakeBroken::new();
    let segment = token_segment(&[9, 10]);
    let good_out = good.pass(std::slice::from_ref(&segment), &[]).unwrap();
    let broken_out = broken.pass(&[segment], &[]).unwrap();
    assert_eq!(good_out.outputs, broken_out.outputs);
    let expected = good_out.outputs.first().unwrap().bytes.clone();
    assert_ne!(expected, vec![0u8; 16]);
    let good_state = read_state(&mut good);
    let broken_state = read_state(&mut broken);
    assert_eq!(good_state, expected);
    assert_eq!(broken_state[0] ^ good_state[0], 1);
    assert_eq!(&broken_state[1..], &good_state[1..]);
}

#[test]
fn ring_coverage_keeps_the_window_base() {
    let mut engine = FakeAlways::new();
    engine.pass(&[token_segment(&[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    let mut ring = [0u8; 4 * 8];
    {
        let parts = engine.parts().unwrap();
        assert_eq!(parts.part_read(State::Slot(0), 2, &mut ring), Ok(()));
    }
    let coverage = engine.coverage().unwrap();
    let span = coverage.coverage(State::Slot(0), 2, &ring);
    assert_eq!((span.start, span.end), (2, 6));
}
