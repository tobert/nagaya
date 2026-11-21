//! The expansion of the expert planes on a B70 card against the host ops (engine-abi "The first Inkling 2 binding",
//! "The experts on the card"). The load repacks the planes of the checkpoint (`repack`), and `OP_EXPAND_NVFP4` and
//! `OP_EXPAND_MXFP8` read them as the expert ops do. Each `bf16` value must have the bits of the host op.
//!
//! Run it alone, on a card that no engine of the active profile names (`AGENTS.md`, "The engines on the B70 cards"):
//! `NAGAYA_TEST_CARDS=<pci> cargo test -p nagaya-mk-inkling2-bmg --features xe --test xe_expand -- --test-threads=1`.
#![cfg(feature = "xe")]

mod common;

use std::ffi::c_void;
use std::io::{Read, Seek, SeekFrom};

use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::{INKLING2_KERNEL, KERNEL_ENTRY, tasks};
use nagaya_mk_inkling2_bmg::formats::{expand_mxfp8_row, expand_nvfp4_row, reference_expert_factor};
use nagaya_mk_inkling2_bmg::repack::{gate_up_tile_order, repack_codes, repack_scales};
use serde_json::Value;

/// The bytes of slices of the checkpoint, read by offset from its header.
struct Checkpoint {
    file: std::fs::File,
    header: Value,
    data_start: u64,
}

impl Checkpoint {
    fn open() -> Self {
        let path = common::weights_path();
        let mut file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        let mut len = [0u8; 8];
        file.read_exact(&mut len).unwrap();
        let len = u64::from_le_bytes(len);
        let mut json = vec![0u8; len as usize];
        file.read_exact(&mut json).unwrap();
        Self {
            file,
            header: serde_json::from_slice(&json).unwrap(),
            data_start: 8 + len,
        }
    }

    /// `len` bytes of tensor `name`, from byte `start` of the tensor.
    fn bytes(&mut self, name: &str, start: u64, len: usize) -> Vec<u8> {
        let tensor = self
            .header
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint holds no {name}"));
        let begin = tensor["data_offsets"][0].as_u64().unwrap();
        self.file
            .seek(SeekFrom::Start(self.data_start + begin + start))
            .unwrap();
        let mut out = vec![0u8; len];
        self.file.read_exact(&mut out).unwrap();
        out
    }

    fn f32_at(&mut self, name: &str, index: u64) -> f32 {
        let b = self.bytes(name, index * 4, 4);
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

/// A buffer in device memory.
struct Buf(*mut u8, usize);

impl Buf {
    fn of(card: &B70Card, data: &[u8]) -> Self {
        let ptr = card.alloc_device_usm::<u8>(data.len().max(64));
        let stage = card.alloc_host_usm::<u8>(data.len().max(64));
        // SAFETY: `stage` holds `data.len()` bytes of host memory.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), stage, data.len()) };
        card.memcpy_sync(ptr as *mut c_void, stage as *const c_void, data.len());
        card.free(stage as *mut c_void);
        Self(ptr, data.len())
    }

    fn zeros(card: &B70Card, bytes: usize) -> Self {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        Self(ptr, bytes)
    }

    fn read_u16(&self, card: &B70Card) -> Vec<u16> {
        let stage = card.alloc_host_usm::<u8>(self.1.max(64));
        card.memcpy_sync(stage as *mut c_void, self.0 as *const c_void, self.1);
        let mut out = vec![0u16; self.1 / 2];
        // SAFETY: `stage` holds `self.1` bytes, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, self.1) };
        card.free(stage as *mut c_void);
        out
    }

    fn free(self, card: &B70Card) {
        card.free(self.0 as *mut c_void);
    }
}

fn words_as_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The rows of one expert plane in the order of the card: `order[r]` is the checkpoint row of card row `r`.
fn reorder(src: &[u8], row_bytes: usize, order: &[usize]) -> Vec<u8> {
    order
        .iter()
        .flat_map(|&r| src[r * row_bytes..(r + 1) * row_bytes].iter().copied())
        .collect()
}

/// The first card row whose bits differ, with the count of values that differ.
fn first_other_row(card_bits: &[u16], host_bits: &[u16], cols: usize) -> Option<(usize, usize)> {
    let bad = card_bits.iter().zip(host_bits).filter(|(a, b)| a != b).count();
    card_bits
        .iter()
        .zip(host_bits)
        .position(|(a, b)| a != b)
        .map(|i| (i / cols, bad))
}

/// One expert matrix of one layer: expand it on the card from its repacked planes, and on the host from the rows of
/// the checkpoint.
fn check_expert(cluster: &L0Cluster, ring: &mut Ring, ckpt: &mut Checkpoint, layer: usize, expert: usize, w13: bool) {
    let nvfp4 = layer < 20;
    let name = format!(
        "model.llm.layers.{layer}.mlp.experts.{}",
        if w13 { "w13_weight" } else { "w2_weight" }
    );
    let (rows, cols) = if w13 { (2304usize, 2304usize) } else { (2304, 1152) };
    let row_bytes = if nvfp4 { cols / 2 } else { cols };
    let row_scales = if nvfp4 { cols / 16 } else { cols / 32 };
    let codes = ckpt.bytes(&name, (expert * rows * row_bytes) as u64, rows * row_bytes);
    let scales = ckpt.bytes(
        &format!("{name}_scale"),
        (expert * rows * row_scales) as u64,
        rows * row_scales,
    );
    let factor = if nvfp4 {
        reference_expert_factor(
            ckpt.f32_at(&format!("{name}_amax"), expert as u64),
            ckpt.f32_at(&format!("{name}_outer_scale_denominator"), 0),
        )
    } else {
        1.0
    };
    let order: Vec<usize> = if w13 {
        gate_up_tile_order(rows)
    } else {
        (0..rows).collect()
    };
    let codes_card = reorder(&codes, row_bytes, &order);
    let scales_card = reorder(&scales, row_scales, &order);
    let step = if nvfp4 { 8 } else { 16 };
    let planes = repack_codes(&codes_card, rows, row_bytes, step).unwrap();
    let scale_planes = repack_scales(&scales_card, rows, row_scales).unwrap();

    let card = &cluster.cards[0];
    let (a, b) = (Buf::of(card, &words_as_bytes(&planes)), Buf::of(card, &scale_planes));
    let y = Buf::zeros(card, rows * cols * 2);
    let task = if nvfp4 {
        tasks::expand_nvfp4(a.0 as u64, b.0 as u64, y.0 as u64, rows as u32, cols as u32, factor)
    } else {
        tasks::expand_mxfp8(a.0 as u64, b.0 as u64, y.0 as u64, rows as u32, cols as u32)
    };
    ring.set(&[task]);
    // SAFETY: the buffers hold the planes and the rows of the task.
    unsafe { ring.run(card) };
    let card_bits = y.read_u16(card);
    for buf in [a, b, y] {
        buf.free(card);
    }

    let mut host_bits = vec![0u16; rows * cols];
    for (r, &src) in order.iter().enumerate() {
        let out = &mut host_bits[r * cols..(r + 1) * cols];
        let row_codes = &codes[src * row_bytes..(src + 1) * row_bytes];
        let row_scale = &scales[src * row_scales..(src + 1) * row_scales];
        if nvfp4 {
            expand_nvfp4_row(row_codes, row_scale, factor, out).unwrap();
        } else {
            expand_mxfp8_row(row_codes, row_scale, out).unwrap();
        }
    }
    if let Some((row, _)) = first_other_row(&card_bits, &host_bits, cols) {
        eprintln!(
            "row {row}: card {:04x?}, host {:04x?}",
            &card_bits[row * cols..row * cols + 8],
            &host_bits[row * cols..row * cols + 8]
        );
    }
    assert_eq!(
        first_other_row(&card_bits, &host_bits, cols),
        None,
        "layer {layer} expert {expert} {name}: (first card row that differs, values that differ)"
    );
}

#[test]
fn the_card_expands_the_repacked_experts_to_the_bits_of_the_host() {
    if common::absent() {
        return;
    }
    let cluster = L0Cluster::init_test(INKLING2_KERNEL, KERNEL_ENTRY);
    let mut ring = Ring::new(&cluster.cards[0], 1 << 16);
    let mut ckpt = Checkpoint::open();
    for (layer, expert) in [(0, 0), (0, 127), (19, 64), (20, 0), (23, 127)] {
        for w13 in [true, false] {
            check_expert(&cluster, &mut ring, &mut ckpt, layer, expert, w13);
        }
    }
}
