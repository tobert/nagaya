//! GPU gate of the parts of a slot (milestone 37 step 11): the key and value rows of a state as position-major
//! `[n][8][128]` in the stored format, and the conv state as `[3][6656]` `f32`. The planes are head-major and the conv
//! buffer is stream-major, so each test reads the raw planes and checks the transpose without the code under test.
//! No model is needed.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_pass_parts --
//!  --test-threads=1`

#![cfg(feature = "cuda")]

mod common;

use nagaya_mk_cuda::attn::{CacheFormat, KV_HEADS, row_offset};
use nagaya_mk_inkling2_sm121::pass::slot::{CONV_BYTES, CONV_STREAM_WIDTHS, Slot, conv_offset};
use nagaya_mk_inkling2_sm121::spec::{self, CONV_ROW, CONV_ROWS, RING_ROWS};

const FORMATS: [CacheFormat; 3] = [CacheFormat::F32, CacheFormat::Bf16, CacheFormat::E4m3];

/// Bytes with no NaN in any format: every byte is below 0x7F, and the high byte of a bf16 or f32 value stays small.
fn pattern(len: usize, seed: u32) -> Vec<u8> {
    (0..len)
        .map(|i| {
            ((i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503))
                >> 13) as u8
                % 0x70
        })
        .collect()
}

#[test]
fn cells_are_position_major_on_the_host_and_head_major_on_the_device() {
    let dev = common::gpu::open();
    for format in FORMATS {
        let slot = Slot::new(&dev, 512, 0, format).unwrap();
        let row = format.row_bytes();
        let (state, n) = (5usize, 40usize);
        let bytes = pattern(n * KV_HEADS * row, 1);
        slot.write_cells(state, false, 3, n, &bytes).unwrap();
        let run = &slot.runs[state];
        // The raw plane: head `h`, row `3 + p` holds host row `[p][h]`.
        for p in [0usize, 1, 17, n - 1] {
            for h in 0..KV_HEADS {
                let raw = run
                    .k_plane
                    .read(row_offset(h, 3 + p, run.plane_rows, format), row)
                    .unwrap();
                assert_eq!(
                    raw,
                    bytes[(p * KV_HEADS + h) * row..(p * KV_HEADS + h + 1) * row],
                    "{format:?} p {p} h {h}"
                );
            }
        }
        // The value plane is untouched.
        assert!(
            run.v_plane
                .read(0, run.v_plane.bytes())
                .unwrap()
                .iter()
                .all(|b| *b == 0)
        );
        // A slot reads only below its position.
        let mut slot = slot;
        assert!(
            slot.read_cells(state, false, 3, n).is_err(),
            "a read above the position"
        );
        slot.set_position(3 + n).unwrap();
        assert_eq!(slot.read_cells(state, false, 3, n).unwrap(), bytes, "{format:?}");
        assert_eq!(
            slot.read_cells(state, false, 10, 5).unwrap(),
            bytes[7 * KV_HEADS * row..12 * KV_HEADS * row]
        );
    }
}

#[test]
fn a_sliding_state_wraps_in_its_ring() {
    let dev = common::gpu::open();
    let format = CacheFormat::Bf16;
    let seq = 3 * RING_ROWS;
    let mut slot = Slot::new(&dev, seq, 0, format).unwrap();
    let row = format.row_bytes();
    // State 1 is a sliding layer. The range of 100 positions crosses the end of the ring.
    let (state, from, n) = (1usize, RING_ROWS - 40, 100usize);
    assert_ne!(slot.runs[state].window, 0);
    let bytes = pattern(n * KV_HEADS * row, 2);
    slot.write_cells(state, true, from, n, &bytes).unwrap();
    slot.set_position(from + n).unwrap();
    assert_eq!(slot.read_cells(state, true, from, n).unwrap(), bytes);
    // Position `RING_ROWS + 5` is in row 5, and position `RING_ROWS - 40 + 45` too is in row 5.
    let plane = &slot.runs[state].v_plane;
    for h in [0usize, 7] {
        let raw = plane.read(row_offset(h, 5, RING_ROWS, format), row).unwrap();
        assert_eq!(raw, bytes[(45 * KV_HEADS + h) * row..(45 * KV_HEADS + h + 1) * row]);
    }
    // More positions than the ring holds, and a read of positions that the ring lost, are refused.
    let too_many = pattern((RING_ROWS + 1) * KV_HEADS * row, 3);
    let e = slot.write_cells(state, true, 0, RING_ROWS + 1, &too_many).unwrap_err();
    assert!(e.contains("holds the last"), "{e}");
    slot.set_position(from + n + RING_ROWS).unwrap();
    assert!(slot.read_cells(state, true, from, 10).is_err(), "a read of a lost row");
    // A full state past the capacity of the slot is refused.
    let e = slot
        .write_cells(5, false, seq - 1, 2, &pattern(2 * KV_HEADS * row, 4))
        .unwrap_err();
    assert!(e.contains("capacity of the slot"), "{e}");
}

#[test]
fn the_conv_state_is_three_rows_of_the_four_streams() {
    let dev = common::gpu::open();
    let slot = Slot::new(&dev, 256, 0, CacheFormat::F32).unwrap();
    // A row holds the streams in order; the value is a function of (row, column) that is easy to find again.
    let value = |r: usize, c: usize| -> f32 { (r * 10_000 + c) as f32 };
    let mut host = Vec::with_capacity(CONV_BYTES);
    for r in 0..CONV_ROWS {
        for c in 0..CONV_ROW {
            host.extend_from_slice(&value(r, c).to_le_bytes());
        }
    }
    slot.write_conv(2, &host).unwrap();
    let run = &slot.runs[2];
    let raw = run.conv.read(0, CONV_BYTES).unwrap();
    let mut column = 0;
    for (stream, width) in CONV_STREAM_WIDTHS.iter().enumerate() {
        for r in 0..CONV_ROWS {
            for c in [0usize, width - 1] {
                let at = conv_offset(stream) + (r * width + c) * 4;
                let got = f32::from_le_bytes(raw[at..at + 4].try_into().unwrap());
                assert_eq!(got, value(r, column + c), "stream {stream} row {r} column {c}");
            }
        }
        column += width;
    }
    assert_eq!(slot.read_conv(2).unwrap(), host);
    // The other states stay zero.
    assert!(slot.read_conv(3).unwrap().iter().all(|b| *b == 0));
}

#[test]
fn a_part_that_the_cache_cannot_hold_is_refused_and_writes_nothing() {
    let dev = common::gpu::open();
    for (format, bad) in [
        (CacheFormat::F32, f32::NAN.to_le_bytes().to_vec()),
        (CacheFormat::F32, f32::INFINITY.to_le_bytes().to_vec()),
        (CacheFormat::Bf16, 0x7F80u16.to_le_bytes().to_vec()),
        (CacheFormat::E4m3, vec![0xFFu8]),
        (CacheFormat::E4m3, vec![0x7Fu8]),
    ] {
        let slot = Slot::new(&dev, 256, 0, format).unwrap();
        let row = format.row_bytes();
        let mut bytes = pattern(KV_HEADS * row, 5);
        bytes[row..row + bad.len()].copy_from_slice(&bad);
        let e = slot.write_cells(5, false, 0, 1, &bytes).unwrap_err();
        assert!(e.contains("not finite"), "{format:?}: {e}");
        let run = &slot.runs[5];
        assert!(
            run.k_plane
                .read(0, run.k_plane.bytes())
                .unwrap()
                .iter()
                .all(|b| *b == 0)
        );
        // A short buffer is refused with its size.
        assert!(slot.write_cells(5, false, 0, 1, &bytes[1..]).is_err());
    }
    let slot = Slot::new(&dev, 256, 0, CacheFormat::F32).unwrap();
    let mut conv = vec![0u8; CONV_BYTES];
    conv[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(slot.write_conv(0, &conv).is_err());
    assert!(slot.write_conv(0, &conv[1..]).is_err());
    assert!(slot.read_conv(spec::LAYERS + 1).is_err(), "a state outside the slot");
}

#[test]
fn a_clear_zeroes_the_conv_and_the_position_and_keeps_the_planes() {
    let dev = common::gpu::open();
    let mut slot = Slot::new(&dev, 256, 0, CacheFormat::F32).unwrap();
    let row = CacheFormat::F32.row_bytes();
    let cells = pattern(KV_HEADS * row, 6);
    slot.write_cells(5, false, 0, 1, &cells).unwrap();
    slot.write_conv(5, &pattern(CONV_BYTES, 7)).unwrap();
    slot.set_position(1).unwrap();
    slot.poisoned = true;
    slot.clear().unwrap();
    assert_eq!((slot.position, slot.poisoned), (0, false));
    assert!(slot.read_conv(5).unwrap().iter().all(|b| *b == 0));
    // The plane keeps its bytes: a clear writes no plane.
    slot.set_position(1).unwrap();
    assert_eq!(slot.read_cells(5, false, 0, 1).unwrap(), cells);
}
