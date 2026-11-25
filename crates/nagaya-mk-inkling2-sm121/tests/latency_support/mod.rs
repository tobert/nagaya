//! The helpers of the latency gate (milestone 37 step 11): fill a slot to a position with seeded rows, and time the
//! calls of the engine. The rows are finite values in the format of the cache. The planes are never read above the
//! position, so the gate needs no prefill.

#![allow(dead_code)]

use std::time::Instant;

use nagaya_abi::{Engine, State, WriteResult};
use nagaya_mk_cuda::attn::CacheFormat;
use nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine;
use nagaya_mk_inkling2_sm121::held::{self, Part, part_list, runs_at};
use nagaya_mk_inkling2_sm121::spec::{self, LOGITS, RING_ROWS};

/// The rows that a call after the fill may add: a warm-up step, a step, 16 rows and blocks of 128 and 512.
pub const AFTER_FILL: usize = 1 + 1 + 16 + 128 + 512;

/// The position that the gate names: the count of positions in the slot when the call starts.
pub const POSITIONS: [usize; 5] = [1024, 32_768, 131_072, 409_600, 524_288];

/// The slot length that holds `positions` and the calls after the fill.
pub fn slot_len(positions: usize) -> usize {
    positions + AFTER_FILL + 64
}

/// A xorshift64* stream. The seed fixes the bytes.
pub struct Seeded(u64);

impl Seeded {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A finite value in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

/// `rows` rows of seeded finite values in the cache `format`: 8 heads of 128 values in each row.
pub fn seeded_rows(format: CacheFormat, rows: usize, seed: u64) -> Vec<u8> {
    let values = rows * 8 * 128;
    let mut rng = Seeded::new(seed);
    let mut out = Vec::with_capacity(values * format.row_bytes() / 128);
    for _ in 0..values {
        let x = rng.unit();
        match format {
            CacheFormat::F32 => out.extend_from_slice(&x.to_le_bytes()),
            CacheFormat::Bf16 => out.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes()),
            // The sign, an exponent of 0 to 7 and a mantissa: a finite value of magnitude below 2.
            CacheFormat::E4m3 => out.push((rng.next_u64() as u8) & 0xbf),
        }
    }
    out
}

/// A seeded conv state: 3 rows of finite `f32` values.
pub fn seeded_conv(seed: u64) -> Vec<u8> {
    let mut rng = Seeded::new(seed);
    (0..held::CONV_BYTES as usize / 4)
        .flat_map(|_| (rng.unit() * 0.5).to_le_bytes())
        .collect()
}

/// The rows of a restore: seeded keys and values, and zeros for a run that the count of cycles does not reach.
pub struct Fill {
    pub keys: Vec<u8>,
    pub values: Vec<u8>,
    pub zeros: Vec<u8>,
    pub conv: Vec<u8>,
}

impl Fill {
    pub fn new(format: CacheFormat, max_positions: usize) -> Self {
        let row = held::row_bytes(format) as usize;
        Self {
            keys: seeded_rows(format, max_positions, 11),
            values: seeded_rows(format, max_positions, 23),
            zeros: vec![0; max_positions * row],
            conv: seeded_conv(31),
        }
    }
}

/// Empty slot 0, write each part for `positions` positions at `cycles` cycles, and set the count. The call gives the
/// time of the fill in milliseconds.
pub fn fill_slot(
    engine: &mut Inkling2Sm121Engine,
    fill: &Fill,
    format: CacheFormat,
    seq: usize,
    extra: usize,
    positions: usize,
    cycles: usize,
) -> u128 {
    let start = Instant::now();
    assert_eq!(
        engine.slots().unwrap().slot_clear(0),
        WriteResult::Ok,
        "{:?}",
        engine.last_error()
    );
    let row = held::row_bytes(format) as usize;
    let state = State::Slot(0);
    for (id, part) in part_list(extra).into_iter().enumerate() {
        let result = match part {
            Part::Cycles => engine
                .parts()
                .unwrap()
                .part_write(state, id, &(cycles as u32).to_le_bytes()),
            Part::Conv(s) => {
                let zero = vec![0u8; fill.conv.len()];
                let bytes = if runs_at(cycles, s) { &fill.conv } else { &zero };
                engine.parts().unwrap().part_write(state, id, bytes)
            }
            Part::K(s) | Part::V(s) => {
                // A ring keeps the last rows. A full state keeps every row from 0.
                let from = if held::ringed(seq, s) {
                    positions.saturating_sub(RING_ROWS)
                } else {
                    0
                };
                let n = positions - from;
                let source = if !runs_at(cycles, s) {
                    &fill.zeros
                } else if matches!(part, Part::K(_)) {
                    &fill.keys
                } else {
                    &fill.values
                };
                engine.parts().unwrap().cells_write(
                    state,
                    id,
                    from as u64,
                    n as u64,
                    &source[from * row..positions * row],
                )
            }
        };
        assert_eq!(result, WriteResult::Ok, "part {id}: {:?}", engine.last_error());
    }
    assert_eq!(
        engine.parts().unwrap().set_n(state, positions as u64),
        WriteResult::Ok,
        "{:?}",
        engine.last_error()
    );
    start.elapsed().as_millis()
}

/// Token ids from the seed, all below the vocabulary.
pub fn ids(seed: usize, at: usize, rows: usize) -> Vec<u32> {
    (at..at + rows)
        .map(|i| ((i * 7919 + seed * 104_729 + 13) % LOGITS) as u32)
        .collect()
}

/// One timed call: the wall time in microseconds, and the bound that the engine declared before the call.
pub struct Timed {
    pub us: u64,
    pub bound_ms: Option<u32>,
}

/// Time one call of `rows` rows at position `at` of slot 0, with logits for the last `logit_rows` rows. The engine
/// declares its bound before the call, as the process does.
pub fn timed(engine: &mut Inkling2Sm121Engine, at: usize, rows: usize, logit_rows: usize, cycles: u8) -> Timed {
    let wanted = ((at + rows - logit_rows) as u64..(at + rows) as u64).collect();
    let segment = nagaya_abi::Segment {
        control: Vec::new(),
        state: State::Slot(0),
        inputs: nagaya_abi::SegmentInputs::Tokens(ids(5, at, rows)),
        positions: (at as u64..(at + rows) as u64).collect(),
        meta: Vec::new(),
        outputs: nagaya_abi::OutputsWanted { logit_rows: wanted },
        layer_loop: None,
        cycles: Some(cycles),
    };
    let bound_ms = engine
        .pass_bound_ms(std::slice::from_ref(&segment), &[])
        .unwrap_or_else(|e| panic!("{e}"))
        .map(|b| b.get());
    let start = Instant::now();
    let out = engine.pass(&[segment], &[]).unwrap_or_else(|e| panic!("{}", e.message));
    let us = start.elapsed().as_micros() as u64;
    assert_eq!(out.outputs.len(), logit_rows);
    Timed { us, bound_ms }
}

/// The extra runs of a config that serves 5 cycles.
pub const EXTRA: usize = spec::MAX_EXTRA_RUNS;
