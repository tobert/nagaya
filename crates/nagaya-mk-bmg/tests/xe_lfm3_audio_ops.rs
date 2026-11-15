//! The ops of the audio encoder of LFM3-Omni-2B on a B70 card (milestone 24 step 7). Each test compares the kernel
//! with the CPU forms of `common/lfm3_audio.rs`, which `lfm3_audio_cpu_forms.rs` checks on the host.
//!
//! Stop `nagaya.service` first: the nagaya must be the only client of the cards (spec 3.1). Run alone:
//! `cargo test -p nagaya-mk-bmg --features xe --test xe_lfm3_audio_ops -- --test-threads=1 --nocapture`.
#![cfg(feature = "xe")]

mod common;

use std::ffi::c_void;

use common::lfm3::{first_other_bits, rel};
use common::lfm3_audio::*;
use nagaya_mk_bmg::level_zero::{B70Card, L0Cluster};
use nagaya_mk_bmg::protocol::MkTaskDescriptor;
use nagaya_mk_bmg::ring::Ring;
use nagaya_mk_bmg::tasks;
use nagaya_mk_bmg::{KERNEL_ENTRY, LFM3_KERNEL};

/// A buffer in device memory, which the test fills and reads through host memory.
struct Buf {
    ptr: *mut u8,
    bytes: usize,
}

impl Buf {
    fn zeros(card: &B70Card, bytes: usize) -> Self {
        let ptr = card.alloc_device_usm::<u8>(bytes.max(64));
        card.fill_zero_sync(ptr as *mut c_void, bytes.max(64));
        Self { ptr, bytes }
    }

    fn of<T: Copy>(card: &B70Card, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::zeros(card, bytes);
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        // SAFETY: `stage` holds `bytes` bytes of host USM, and `data` is `bytes` bytes long.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, stage, bytes) };
        card.memcpy_sync(buf.ptr as *mut c_void, stage as *const c_void, bytes);
        card.free(stage as *mut c_void);
        buf
    }

    fn read(&self, card: &B70Card, n: usize) -> Vec<f32> {
        let bytes = n * 4;
        assert!(bytes <= self.bytes.max(64));
        let stage = card.alloc_host_usm::<u8>(bytes.max(64));
        card.memcpy_sync(stage as *mut c_void, self.ptr as *const c_void, bytes);
        let mut out = vec![0.0f32; n];
        // SAFETY: `stage` holds `bytes` bytes, and `out` has room for them.
        unsafe { std::ptr::copy_nonoverlapping(stage, out.as_mut_ptr() as *mut u8, bytes) };
        card.free(stage as *mut c_void);
        out
    }

    fn at(&self) -> u64 {
        self.ptr as u64
    }

    fn free(self, card: &B70Card) {
        card.free(self.ptr as *mut c_void);
    }
}

struct Card {
    cluster: L0Cluster,
    ring: Ring,
}

impl Card {
    fn open() -> Self {
        let cluster = L0Cluster::init_test(LFM3_KERNEL, KERNEL_ENTRY);
        let ring = Ring::new(&cluster.cards[0], 1 << 20);
        Self { cluster, ring }
    }

    fn card(&self) -> &B70Card {
        &self.cluster.cards[0]
    }

    fn run(&mut self, tasks: &[MkTaskDescriptor]) {
        self.ring.set(tasks);
        // SAFETY: each test sizes its buffers for its tasks.
        unsafe { self.ring.run(&self.cluster.cards[0]) };
    }
}

/// With no silu, the kernel sums the same fused multiply-adds as the CPU form, so the bits agree. With silu, `exp` of
/// the card is not the `exp` of the host, so the test prints the gap and bounds it loosely.
#[test]
fn subsample_matches_its_cpu_form() {
    let mut card = Card::open();
    for (name, n, k, cin, m, silu, seed) in SUBSAMPLE_CASES {
        let (x, w, b) = subsample_inputs(n, k, cin, m, seed);
        let want = subsample(&x, &w, &b, n, k, cin, m, silu);
        let (xb, wb, bb) = (
            Buf::of(card.card(), &x),
            Buf::of(card.card(), &w),
            Buf::of(card.card(), &b),
        );
        let yb = Buf::zeros(card.card(), want.len() * 4);
        card.run(&[tasks::subsample(
            xb.at(),
            wb.at(),
            bb.at(),
            yb.at(),
            n as u32,
            k as u32,
            cin as u32,
            m as u32,
            silu,
        )]);
        let got = yb.read(card.card(), want.len());
        let err = rel(&got, &want);
        println!("{name}: relative error {err:.3e} against the CPU form");
        if silu {
            assert!(err < 1e-5, "{name}: relative error {err}");
        } else {
            assert_eq!(
                first_other_bits(&got, &want),
                None,
                "{name}: the bits differ from the CPU form"
            );
        }
        for buf in [xb, wb, bb, yb] {
            buf.free(card.card());
        }
    }
}
