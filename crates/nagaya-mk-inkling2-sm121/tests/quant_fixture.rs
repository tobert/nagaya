//! Compares the host quantizers with the bytes of the vLLM kernels of Inkling 2, bit for bit.
//!
//! The fixture `tests/fixtures/quant_kernels.safetensors` holds the inputs (bf16) and the outputs of the kernels that
//! `tools/inkling2_quant_fixture.py` ran on a GB10. A mismatch names the set, the row, the row class and the block.

mod common;

use common::Fixture;
use nagaya_mk_inkling2_sm121::quant::{MXFP8_BLOCK, NVFP4_BLOCK, quantize_mxfp8_ue8m0, quantize_nvfp4_fouroversix_mae};

#[track_caller]
fn compare_bytes(f: &Fixture, set: &str, what: &str, row: usize, per_block: usize, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{set} row {row}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{set} row {row} (class {}): {what} differ at block {} (byte {i}): got 0x{:02x}, kernel 0x{:02x}",
            f.class(set, row),
            i / per_block,
            got[i],
            want[i]
        );
    }
}

fn check_nvfp4(f: &Fixture, set: &str) {
    let xs = f.rows_f32(&format!("{set}.x"));
    let (values, vshape) = f.raw(&format!("{set}.values"));
    let (scales, sshape) = f.raw(&format!("{set}.scales"));
    let (amax, _) = f.raw(&format!("{set}.amax"));
    assert_eq!(xs.len(), vshape[0]);
    for (r, x) in xs.iter().enumerate() {
        let q = quantize_nvfp4_fouroversix_mae(x).unwrap_or_else(|e| panic!("{set} row {r}: {e}"));
        compare_bytes(
            f,
            set,
            "values",
            r,
            NVFP4_BLOCK / 2,
            &q.codes,
            &values[r * vshape[1]..(r + 1) * vshape[1]],
        );
        compare_bytes(
            f,
            set,
            "scales",
            r,
            1,
            &q.scales,
            &scales[r * sshape[1]..(r + 1) * sshape[1]],
        );
        let want = u32::from_le_bytes(amax[4 * r..4 * r + 4].try_into().unwrap());
        assert_eq!(
            q.amax.to_bits(),
            want,
            "{set} row {r} (class {}): amax bits differ",
            f.class(set, r)
        );
    }
}

#[test]
fn nvfp4_k2304_matches_the_kernel() {
    check_nvfp4(&Fixture::load(), "nvfp4_k2304");
}

#[test]
fn nvfp4_k1152_matches_the_kernel() {
    check_nvfp4(&Fixture::load(), "nvfp4_k1152");
}

#[test]
fn mxfp8_k2304_matches_the_kernel() {
    let f = Fixture::load();
    let set = "mxfp8_k2304";
    let xs = f.rows_f32(&format!("{set}.x"));
    let (values, vshape) = f.raw(&format!("{set}.values"));
    let (scales, sshape) = f.raw(&format!("{set}.scales"));
    for (r, x) in xs.iter().enumerate() {
        let q = quantize_mxfp8_ue8m0(x).unwrap_or_else(|e| panic!("{set} row {r}: {e}"));
        compare_bytes(
            &f,
            set,
            "values",
            r,
            MXFP8_BLOCK,
            &q.codes,
            &values[r * vshape[1]..(r + 1) * vshape[1]],
        );
        compare_bytes(
            &f,
            set,
            "scales",
            r,
            1,
            &q.scales,
            &scales[r * sshape[1]..(r + 1) * sshape[1]],
        );
    }
}
