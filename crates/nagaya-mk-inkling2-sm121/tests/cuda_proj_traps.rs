//! GPU gate of the checks of `OP_PROJ` (milestone 37 step 9, wave B lane D): a task that the op cannot run traps, and a
//! good task runs. The task has real buffers of zeros, so a check that is missing lets the launch succeed, and the test
//! shows it. The model is not needed.
//!
//! A trap breaks the CUDA context of its process. The test runs each task in a child process: it starts its own
//! executable again, with the variant in `NAGAYA_PROJ_VARIANT`. The child leaves at once, with no teardown.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_proj_traps
//!  -- --test-threads=1`

#![cfg(feature = "cuda")]

mod common;

use std::process::Command;
use std::time::{Duration, Instant};

use common::gpu::open;
use nagaya_mk_cuda::proj::ProjLayout;
use nagaya_mk_cuda::protocol::{
    FLAG_ACCUMULATE, FLAG_FULL_LAYER, FLAG_NORM, MkTaskTable, PROJ_TABLE_K_CONV, PROJ_TABLE_K_OUT, PROJ_TABLE_K_STATE,
    PROJ_TABLE_NORM_K, PROJ_TABLE_NORM_Q, PROJ_TABLE_NORM_V, PROJ_TABLE_Q_SCALE, PROJ_TABLE_REL_SCALE, PROJ_TABLE_RF_K,
    PROJ_TABLE_RF_V, PROJ_TABLE_V_CONV, PROJ_TABLE_V_OUT, PROJ_TABLE_V_STATE, PROJ_TABLE_X_K, PROJ_TABLE_X_V,
};
use nagaya_mk_cuda::ring::{Ring, Task};

const VARIABLE: &str = "NAGAYA_PROJ_VARIANT";
const TEST: &str = "a_task_that_the_op_cannot_run_traps_and_a_good_task_runs";

/// The variants: a name, and what it is expected to do.
const VARIANTS: [(&str, bool); 9] = [
    ("none", true),
    ("scratch", false),
    ("rows", false),
    ("slot", false),
    ("align", false),
    ("flag", false),
    ("full", false),
    ("no_norm", false),
    ("items", false),
];

/// The child: launch one task, and exit with 0 when the launch did what the variant expects.
fn child(variant: &str) {
    let expect_ok = VARIANTS.iter().find(|(n, _)| *n == variant).expect("a known variant").1;
    let dev = open();
    let layout = ProjLayout { full: false };
    let rows = 16;
    let zeros = |bytes: usize| {
        let p = dev.alloc_device(bytes).unwrap();
        dev.memset_zero(p, bytes).unwrap();
        p
    };
    let packed = zeros(layout.shape().packed_bytes());
    let x = zeros(rows * 2304 * 4);
    let rf = zeros(rows * 4);
    let out = zeros(ProjLayout::output_floats(rows) * 4);
    let ring = Ring::new(&dev, layout.scratch_floats(rows), false).unwrap();
    ring.zero_device().unwrap();
    let norm = zeros(2304 * 2);
    let scale = zeros(24 * 128 * 2);
    let conv = zeros(1024 * 4 * 2);
    let state = zeros(3 * 1024 * 4);
    let slot = zeros(rows * 1024 * 4);
    let mut table = MkTaskTable::default();
    for (slot_index, p) in [
        (PROJ_TABLE_NORM_Q, norm),
        (PROJ_TABLE_NORM_K, norm),
        (PROJ_TABLE_NORM_V, norm),
        (PROJ_TABLE_X_K, x),
        (PROJ_TABLE_X_V, x),
        (PROJ_TABLE_RF_K, rf),
        (PROJ_TABLE_RF_V, rf),
        (PROJ_TABLE_Q_SCALE, scale),
        (PROJ_TABLE_REL_SCALE, scale),
        (PROJ_TABLE_K_CONV, conv),
        (PROJ_TABLE_V_CONV, conv),
        (PROJ_TABLE_K_STATE, state),
        (PROJ_TABLE_V_STATE, state),
        (PROJ_TABLE_K_OUT, slot),
        (PROJ_TABLE_V_OUT, slot),
    ] {
        table.ptr[slot_index] = p;
    }
    match variant {
        "slot" => table.ptr[PROJ_TABLE_V_STATE] = 0,
        "align" => table.ptr[PROJ_TABLE_X_K] += 4,
        _ => {}
    }
    let table_ptr = dev.alloc_device(std::mem::size_of::<MkTaskTable>()).unwrap();
    unsafe { dev.copy_htod(table_ptr, (&raw const table).cast(), std::mem::size_of::<MkTaskTable>()) }.unwrap();
    let mut desc = layout
        .descriptor(rows, 0, 0, x, packed, out, table_ptr, rf, ring.scratch())
        .unwrap();
    match variant {
        "scratch" => desc.scale_bytes -= 1,
        "rows" => desc.rows = 513,
        "flag" => desc.flags |= FLAG_ACCUMULATE,
        "full" => desc.flags |= FLAG_FULL_LAYER,
        "no_norm" => desc.flags &= !FLAG_NORM,
        "items" => desc.items -= 3,
        _ => {}
    }
    let mut ring = ring;
    ring.set(&[Task::with_items(desc)]);
    let result = unsafe { ring.run() };
    match (result, expect_ok) {
        (Ok(()), true) => eprintln!("cuda_proj_traps: variant {variant} ran with no error"),
        (Err(error), false) => eprintln!("cuda_proj_traps: variant {variant} trapped: {error}"),
        (Ok(()), false) => {
            eprintln!("cuda_proj_traps: variant {variant} ran with no error, but it must trap");
            std::process::exit(3);
        }
        (Err(error), true) => {
            eprintln!("cuda_proj_traps: variant {variant} must run, and it failed: {error}");
            std::process::exit(4);
        }
    }
    std::process::exit(0);
}

#[test]
fn a_task_that_the_op_cannot_run_traps_and_a_good_task_runs() {
    if let Ok(value) = std::env::var(VARIABLE) {
        child(&value);
    }
    let me = std::env::current_exe().expect("the path of the test executable");
    for (name, _) in VARIANTS {
        let mut process = Command::new(&me)
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env(VARIABLE, name)
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
