//! The calls of the engine for the retrieval test: the prefill, the greedy decode, and the snapshot and the restore of
//! a slot through the part calls of the `Engine` trait.

use std::time::{Duration, Instant};

use nagaya_abi::{Engine, OutputsWanted, PassOut, Segment, SegmentInputs, State, WriteResult};
use nagaya_mk_inkling2_sm121::engine::Inkling2Sm121Engine;

use super::snapshot::{Reader, Writer};

/// The rows of a prefill block. It is the `max_block` of the engine by default.
pub const BLOCK: usize = 512;

/// A change that a falsification run makes to the restore. The honest restore is `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// Write zeros in place of each conv state.
    ZeroConv,
    /// Flip the sign of each value of the last row of the first key part of a full layer.
    FlipRow,
}

fn segment(slot: u32, from: u64, ids: &[u32], logit_rows: Vec<u64>) -> Segment {
    Segment {
        control: Vec::new(),
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(ids.to_vec()),
        positions: (from..from + ids.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted { logit_rows },
        layer_loop: None,
        cycles: None,
    }
}

/// The f32 values of the first output of a pass.
pub fn logits_of(out: &PassOut) -> Vec<f32> {
    out.outputs[0]
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The BLAKE3 of the bits of a logit row.
pub fn row_hash(row: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in row {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

pub fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .fold((0, f32::MIN), |b, (i, x)| if *x > b.1 { (i, *x) } else { b })
        .0 as u32
}

/// Feed `ids` to `slot` from position `from` in blocks of `BLOCK`. Give the logits of the last token when `last` is
/// true.
pub fn feed(engine: &mut Inkling2Sm121Engine, slot: u32, from: u64, ids: &[u32], last: bool) -> Option<Vec<f32>> {
    let mut at = from;
    let mut logits = None;
    let blocks = ids.chunks(BLOCK).count();
    for (i, block) in ids.chunks(BLOCK).enumerate() {
        let want = if last && i + 1 == blocks {
            vec![at + block.len() as u64 - 1]
        } else {
            Vec::new()
        };
        let out = engine.pass(&[segment(slot, at, block, want)], &[]).unwrap_or_else(|e| {
            panic!(
                "a pass of {} tokens at position {at} failed: {}",
                block.len(),
                e.message
            )
        });
        if last && i + 1 == blocks {
            logits = Some(logits_of(&out));
        }
        at += block.len() as u64;
    }
    logits
}

/// Prefill `ids` into `slot` from position 0. The prefill stops with a panic when it passes `budget`. It prints the
/// progress at each 16,384 positions. Give the time of the prefill.
pub fn prefill(engine: &mut Inkling2Sm121Engine, slot: u32, ids: &[u32], label: &str, budget: Duration) -> Duration {
    let started = Instant::now();
    let mut at = 0usize;
    let mut next_report = 16_384;
    while at < ids.len() {
        assert!(
            started.elapsed() < budget,
            "{label}: the prefill passed its budget of {budget:?} at position {at} of {}; raise NAGAYA_RET_PREFILL_BUDGET_S \
             or shorten the context",
            ids.len()
        );
        let n = (ids.len() - at).min(BLOCK);
        feed(engine, slot, at as u64, &ids[at..at + n], false);
        at += n;
        if at >= next_report {
            eprintln!(
                "RET {label}: prefill {at} of {} positions, {:.0?}",
                ids.len(),
                started.elapsed()
            );
            next_report += 16_384;
        }
    }
    started.elapsed()
}

/// Read every part of `slot` into `writer`, in the order of the part ids. Give the bytes that it read.
pub fn snapshot(engine: &mut Inkling2Sm121Engine, slot: u32, writer: &mut Writer) -> Result<u64, String> {
    let limits = engine.limits();
    let high = engine.slots().unwrap().slot_high(slot);
    assert_eq!(
        writer.n, high,
        "the writer names {} positions, and the slot holds {high}",
        writer.n
    );
    let mut total = 0;
    for (id, part) in limits.parts.iter().enumerate() {
        let row = (part.shape.iter().product::<u64>() * part.dtype.element_bytes()) as usize;
        let (start, rows, len) = if part.is_row() {
            let start = engine.coverage().unwrap().coverage(State::Slot(slot), id, &[]).start;
            (start, high - start, (high - start) as usize * row)
        } else {
            (0, 0, row)
        };
        let mut bytes = vec![0u8; len];
        engine
            .parts()
            .unwrap()
            .part_read(State::Slot(slot), id, &mut bytes)
            .map_err(|e| {
                format!(
                    "cannot read part {} ({}): {e:?}; {:?}",
                    id,
                    part.name,
                    engine.last_error()
                )
            })?;
        writer.add(id, &part.name, start, rows, &bytes)?;
        total += len as u64;
    }
    Ok(total)
}

/// Write the parts of a snapshot into `slot`, then the count. The BLAKE3 is checked before the count, so a damaged file
/// leaves the slot with no positions. Give the time that it took.
pub fn restore(
    engine: &mut Inkling2Sm121Engine,
    slot: u32,
    mut reader: Reader,
    mutation: Mutation,
) -> Result<Duration, String> {
    let started = Instant::now();
    let limits = engine.limits();
    if reader.parts.len() != limits.parts.len() {
        return Err(format!(
            "the snapshot has {} parts, and the engine has {}; it is a snapshot of another engine layout",
            reader.parts.len(),
            limits.parts.len()
        ));
    }
    let mut buf = Vec::new();
    let mut flipped = false;
    for (id, part) in limits.parts.iter().enumerate() {
        let row = (part.shape.iter().product::<u64>() * part.dtype.element_bytes()) as usize;
        let meta = reader.read_next(&mut buf)?;
        if meta.id != id || meta.name != part.name {
            return Err(format!(
                "part {id} of the snapshot is {meta:?}, and the engine names it {}",
                part.name
            ));
        }
        let want = if part.is_row() { meta.rows as usize * row } else { row };
        if buf.len() != want {
            return Err(format!(
                "part {id} has {} bytes, and its rows need {want} bytes of {row} per row",
                buf.len()
            ));
        }
        if mutation == Mutation::ZeroConv && part.name.starts_with("conv.") {
            buf.fill(0);
        }
        if mutation == Mutation::FlipRow && !flipped && part.name.starts_with("k.") && part.is_row() && meta.start == 0
        {
            let width = part.dtype.element_bytes() as usize;
            let last = buf.len() - row;
            for value in 0..row / width {
                buf[last + (value + 1) * width - 1] ^= 0x80;
            }
            flipped = true;
        }
        let result = if part.is_row() {
            engine
                .parts()
                .unwrap()
                .cells_write(State::Slot(slot), id, meta.start, meta.rows, &buf)
        } else {
            engine.parts().unwrap().part_write(State::Slot(slot), id, &buf)
        };
        if result != WriteResult::Ok {
            return Err(format!(
                "the write of part {id} ({}) gave {result:?}: {:?}",
                part.name,
                engine.last_error()
            ));
        }
    }
    let n = reader.n;
    reader.finish()?;
    let result = engine.parts().unwrap().set_n(State::Slot(slot), n);
    if result != WriteResult::Ok {
        return Err(format!("set_n({n}) gave {result:?}: {:?}", engine.last_error()));
    }
    Ok(started.elapsed())
}

/// Decode greedily from `logits`, the logits of the token before position `from`. `stop` sees each token and says
/// whether the reply ended. Give the tokens, the time of each decode step and the hashes of the first `KEEP_ROWS` logit
/// rows, which pin the bits of the start of the reply.
pub const KEEP_ROWS: usize = 4;

pub fn greedy(
    engine: &mut Inkling2Sm121Engine,
    slot: u32,
    mut from: u64,
    mut logits: Vec<f32>,
    max: usize,
    mut stop: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Vec<Duration>, Vec<String>) {
    let mut tokens = Vec::new();
    let mut steps = Vec::new();
    let mut hashes = Vec::new();
    loop {
        if hashes.len() < KEEP_ROWS {
            hashes.push(row_hash(&logits));
        }
        let token = argmax(&logits);
        tokens.push(token);
        if stop(token) || tokens.len() >= max {
            return (tokens, steps, hashes);
        }
        let started = Instant::now();
        logits = feed(engine, slot, from, &[token], true).expect("a pass with a logit row gives logits");
        steps.push(started.elapsed());
        from += 1;
    }
}

/// The digest of an engine identity, as `nagaya::park::identity_digest` computes it (spec 6.2, 12.1). The test crate
/// cannot depend on the process crate, so this is a copy of the same inputs. Proposed (milestone 37 step 11, Sonnet
/// 5.5): the digest may move into `nagaya-abi` later.
pub fn identity_digest(identity: &nagaya_abi::Identity) -> [u8; 32] {
    use nagaya_abi::BitClass;
    fn add_to(writer: &mut impl std::io::Write, bytes: &[u8]) {
        writer.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        writer.write_all(bytes).unwrap();
    }
    fn add(hasher: &mut blake3::Hasher, tag: u16, bytes: &[u8]) {
        hasher.update(&tag.to_le_bytes());
        add_to(hasher, bytes);
    }
    let mut hasher = blake3::Hasher::new_derive_key("nagaya recipe v1");
    add(&mut hasher, 1, identity.model.as_bytes());
    add(&mut hasher, 2, identity.checkpoint_digest.as_bytes());
    add(&mut hasher, 3, identity.tokenizer_hash.as_bytes());
    add(&mut hasher, 4, identity.template.as_bytes());
    let mut numbers = Vec::new();
    numbers.extend_from_slice(&(identity.numbers.len() as u64).to_le_bytes());
    for (name, value) in &identity.numbers {
        add_to(&mut numbers, name.as_bytes());
        add_to(&mut numbers, value.as_bytes());
    }
    add(&mut hasher, 5, &numbers);
    add(&mut hasher, 6, identity.plan_table.as_bytes());
    let mut settings = Vec::new();
    settings.extend_from_slice(&(identity.bit_settings.len() as u64).to_le_bytes());
    for setting in &identity.bit_settings {
        add_to(&mut settings, setting.name.as_bytes());
        add_to(&mut settings, setting.value.as_bytes());
        settings.push(match setting.class {
            BitClass::ReferenceDefining => 1,
            BitClass::Approx => 2,
            BitClass::Policy => 3,
        });
    }
    add(&mut hasher, 7, &settings);
    add(&mut hasher, 8, identity.canary_hash.as_bytes());
    *hasher.finalize().as_bytes()
}
