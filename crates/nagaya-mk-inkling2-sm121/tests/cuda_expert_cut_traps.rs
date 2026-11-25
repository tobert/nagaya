//! GPU gate of the cut of an item in the up and down ops of the experts: a task whose
//! tiles for an item do not divide the 36 tiles of a pair traps, and a task with a divisor runs. The task has real
//! buffers of zeros and an empty pair list, so a check that is missing lets the launch succeed, and the test shows it.
//! The model is not needed.
//!
//! A trap breaks the CUDA context of its process. The test runs each task in a child process: it starts its own
//! executable again, with the variant in `NAGAYA_CUT_VARIANT`. The child leaves at once, with no teardown.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test
//!  cuda_expert_cut_traps -- --test-threads=1`

#![cfg(feature = "cuda")]

mod common;

use std::process::Command;
use std::time::{Duration, Instant};

use common::gpu::open;
use nagaya_mk_cuda::experts::{
    EXPERT_TABLE_CONV_STATE, EXPERT_TABLE_CONV_WEIGHT, EXPERT_TABLE_OUT, EXPERT_TABLE_OUT_RF, EXPERT_TABLE_QACT,
    EXPERT_TABLE_RESIDUAL,
};
use nagaya_mk_cuda::protocol::{FLAG_DOWN, FLAG_PAIRS, FLAG_UP, MkTaskDescriptor, MkTaskTable, OP_EXPERT_NVFP4};
use nagaya_mk_cuda::ring::{Ring, Task};

const VARIABLE: &str = "NAGAYA_CUT_VARIANT";
const TEST: &str = "a_cut_that_does_not_divide_the_tiles_traps_and_a_divisor_runs";

/// The variants: the op, the tiles for an item, and whether the launch must run.
const VARIANTS: [(&str, u32, bool); 12] = [
    ("up", 1, true),
    ("up", 9, true),
    ("up", 36, true),
    ("up", 5, false),
    ("up", 7, false),
    ("up", 72, false),
    ("down", 1, true),
    ("down", 3, true),
    ("down", 36, true),
    ("down", 5, false),
    ("down", 8, false),
    ("down", 72, false),
];

/// The child: launch one task, and exit with 0 when the launch did what the variant expects.
fn child(variant: &str) {
    let (op, tpi) = {
        let mut parts = variant.split(':');
        (
            parts.next().unwrap().to_string(),
            parts.next().unwrap().parse::<u32>().unwrap(),
        )
    };
    let expect_ok = VARIANTS
        .iter()
        .find(|(o, t, _)| *o == op && *t == tpi)
        .expect("a known variant")
        .2;
    let dev = open();
    let zeros = |bytes: usize| {
        let p = dev.alloc_device(bytes).unwrap();
        dev.memset_zero(p, bytes).unwrap();
        p
    };
    let bound = 8usize;
    let tiles = 36u32;
    // A pair list of zeros has no live pair, so no item reads the weights or the rows.
    let pairs = zeros(4096);
    let mut table = MkTaskTable::default();
    // The up op needs the buffer of the quantized rows, and the down op needs its five buffers. Each is a zero buffer.
    for slot in [
        EXPERT_TABLE_QACT,
        EXPERT_TABLE_RESIDUAL,
        EXPERT_TABLE_CONV_WEIGHT,
        EXPERT_TABLE_CONV_STATE,
        EXPERT_TABLE_OUT,
        EXPERT_TABLE_OUT_RF,
    ] {
        table.ptr[slot] = zeros(4096);
    }
    let table_ptr = dev.alloc_device(std::mem::size_of::<MkTaskTable>()).unwrap();
    unsafe { dev.copy_htod(table_ptr, (&raw const table).cast(), std::mem::size_of::<MkTaskTable>()) }.unwrap();
    let mut ring = Ring::new(&dev, 4096, false).unwrap();
    ring.zero_device().unwrap();
    let up = op == "up";
    // The count of items is the bound times the items of a pair, as the pass gives it, and at least 1.
    let items = (bound as u32 * tiles / tpi).max(1);
    let desc = MkTaskDescriptor {
        opcode: OP_EXPERT_NVFP4,
        flags: FLAG_PAIRS | if up { FLAG_UP } else { FLAG_DOWN },
        items,
        rows: if up { 16 } else { 8 },
        a: zeros(4096),
        b: zeros(4096),
        c: zeros(4096),
        d: table_ptr,
        e: pairs,
        f: ring.scratch(),
        n: 2304,
        k: if up { 2304 } else { 1152 },
        kt: 1,
        group_items: if up { tiles / tpi } else { 0 },
        pos0: tpi,
        p0: 1.0,
        p1: 1.0,
        ..Default::default()
    };
    ring.set(&[Task::with_items(desc)]);
    let result = unsafe { ring.run() };
    match (result, expect_ok) {
        (Ok(()), true) => eprintln!("cuda_expert_cut_traps: variant {variant} ran with no error"),
        (Err(error), false) => eprintln!("cuda_expert_cut_traps: variant {variant} trapped: {error}"),
        (Ok(()), false) => {
            eprintln!("cuda_expert_cut_traps: variant {variant} ran with no error, but it must trap");
            std::process::exit(3);
        }
        (Err(error), true) => {
            eprintln!("cuda_expert_cut_traps: variant {variant} must run, and it failed: {error}");
            std::process::exit(4);
        }
    }
    std::process::exit(0);
}

#[test]
fn a_cut_that_does_not_divide_the_tiles_traps_and_a_divisor_runs() {
    if let Ok(value) = std::env::var(VARIABLE) {
        child(&value);
    }
    let me = std::env::current_exe().expect("the path of the test executable");
    for (op, tpi, _) in VARIANTS {
        let name = format!("{op}:{tpi}");
        let mut process = Command::new(&me)
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env(VARIABLE, &name)
            .spawn()
            .expect("start the child");
        let start = Instant::now();
        let status = loop {
            if let Some(status) = process.try_wait().expect("wait for the child") {
                break status;
            }
            if start.elapsed() > Duration::from_secs(60) {
                let _ = process.kill();
                panic!("variant {name}: the child did not end in 60 s; the launch hangs");
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        assert!(status.success(), "variant {name}: the child ended with {status}");
    }
}
