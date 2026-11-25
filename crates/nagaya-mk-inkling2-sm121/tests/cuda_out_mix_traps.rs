//! GPU gate of the checks of `OP_OUT_MIX`: a task that the op cannot run traps, and a
//! good task runs. The task has real buffers of zeros, so a check that is missing lets the launch succeed, and the test
//! shows it. The model is not needed.
//!
//! A trap breaks the CUDA context of its process. The test runs each task in a child process: it starts its own
//! executable again, with the variant in `NAGAYA_OUT_MIX_VARIANT`. The child leaves at once, with no teardown.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_out_mix_traps
//!  -- --test-threads=1`

#![cfg(feature = "cuda")]

mod common;

use std::process::Command;
use std::time::{Duration, Instant};

use common::gpu::open;
use nagaya_mk_cuda::outmix::{self, OutMixBuffers, OutMixShape, OutMixTable, out_floats, scratch_floats};
use nagaya_mk_cuda::protocol::{FLAG_ACCUMULATE, FLAG_SOFTCAP_30, MkTaskTable};
use nagaya_mk_cuda::ring::{Ring, Task};

const VARIABLE: &str = "NAGAYA_OUT_MIX_VARIANT";
const TEST: &str = "a_task_that_the_op_cannot_run_traps_and_a_good_task_runs";

/// The variants: a name, and what it is expected to do.
const VARIANTS: [(&str, bool); 12] = [
    ("none", true),
    ("scratch", false),
    ("rows", false),
    ("depth_low", false),
    ("depth_high", false),
    ("accumulate_no_x", false),
    ("x_no_accumulate", false),
    ("flag", false),
    ("r_slot", false),
    ("no_query", false),
    ("no_norm_weight", false),
    ("align", false),
];

/// The child: launch one task, and exit with 0 when the launch did what the variant expects.
fn child(variant: &str) {
    let expect_ok = VARIANTS.iter().find(|(n, _)| *n == variant).expect("a known variant").1;
    let dev = open();
    let rows = 16;
    let zeros = |bytes: usize| {
        let p = dev.alloc_device(bytes).unwrap();
        dev.memset_zero(p, bytes).unwrap();
        p
    };
    let packed = zeros(outmix::shape().packed_bytes());
    let attn = zeros(rows * 3072 * 4);
    let r = zeros(rows * 2304 * 4);
    let x = zeros(rows * 2304 * 4);
    let ring = Ring::new(&dev, scratch_floats(rows), false).unwrap();
    ring.zero_device().unwrap();
    let plane = zeros(rows * 2304 * 4);
    let plane_rf = zeros(rows * 4);
    let r_rf = zeros(rows * 4);
    let mut table = OutMixTable {
        planes: vec![(plane, plane_rf)],
        r,
        r_rf,
        conv_weight: zeros(2304 * 4 * 2),
        conv_state: zeros(3 * 2304 * 4),
        norm_weight: zeros(2304 * 4),
        out: zeros(out_floats(rows) * 4),
        query: zeros(2304 * 4),
    };
    let mut shape = OutMixShape::new(rows, 2, false, true).unwrap();
    let mut x_ptr = 0;
    match variant {
        "accumulate_no_x" => shape.accumulate = true,
        "x_no_accumulate" => x_ptr = x,
        "no_query" => table.query = 0,
        "no_norm_weight" => table.norm_weight = 0,
        _ => {}
    }
    let attn_ptr = attn;
    let mut built: MkTaskTable = table.table().unwrap();
    if variant == "r_slot" {
        built.ptr[1] = plane;
    }
    let table_ptr = dev.alloc_device(std::mem::size_of::<MkTaskTable>()).unwrap();
    unsafe { dev.copy_htod(table_ptr, (&raw const built).cast(), std::mem::size_of::<MkTaskTable>()) }.unwrap();
    // `descriptor` refuses a bad pair of `accumulate` and `x`, so the variants that need one build it by hand.
    let buffers = OutMixBuffers {
        attn: attn_ptr,
        packed,
        r,
        x: if shape.accumulate { x } else { x_ptr },
        scratch: ring.scratch(),
        table: table_ptr,
    };
    let mut desc = if variant == "accumulate_no_x" || variant == "x_no_accumulate" {
        let mut good = OutMixShape::new(rows, 2, false, true)
            .unwrap()
            .descriptor(&OutMixBuffers { x: 0, ..buffers })
            .unwrap();
        if variant == "accumulate_no_x" {
            good.flags |= FLAG_ACCUMULATE;
        } else {
            good.e = x;
        }
        good
    } else {
        shape.descriptor(&buffers).unwrap()
    };
    match variant {
        "scratch" => desc.scale_bytes -= 1,
        "rows" => desc.rows = 513,
        "depth_low" => desc.layer = 1,
        "depth_high" => desc.layer = 6,
        "flag" => desc.flags |= FLAG_SOFTCAP_30,
        // `descriptor` has no check of the address, so the kernel check is the one that the variant tests.
        "align" => desc.a += 4,
        _ => {}
    }
    let mut ring = ring;
    ring.set(&[Task::with_items(desc)]);
    let result = unsafe { ring.run() };
    match (result, expect_ok) {
        (Ok(()), true) => eprintln!("cuda_out_mix_traps: variant {variant} ran with no error"),
        (Err(error), false) => eprintln!("cuda_out_mix_traps: variant {variant} trapped: {error}"),
        (Ok(()), false) => {
            eprintln!("cuda_out_mix_traps: variant {variant} ran with no error, but it must trap");
            std::process::exit(3);
        }
        (Err(error), true) => {
            eprintln!("cuda_out_mix_traps: variant {variant} must run, and it failed: {error}");
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
