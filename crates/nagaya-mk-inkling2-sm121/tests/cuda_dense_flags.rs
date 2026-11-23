//! GPU gate: the bit 4 of the flags was `FLAG_NORM_UNWEIGHTED`, a norm of the rows with no weight. No engine set it, and
//! the matmul no longer has it. A task with the bit traps, in the decode form and in the block form.
//!
//! The task is a valid matmul of zeros: the weights, the rows and the output are real buffers. The parent runs the
//! task with no extra flag in both forms, so that a trap cannot come from another field. A trap breaks the CUDA context
//! of its process, so the task with the bit runs in a child process, one for each form: the test starts its own
//! executable again, with `NAGAYA_DENSE_FLAG_CHILD` set to the form.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_dense_flags
//!  -- --test-threads=1`

#![cfg(feature = "cuda")]

mod common;

use std::process::Command;
use std::time::{Duration, Instant};

use common::gpu::{open, upload};
use nagaya_mk_cuda::repack::{MatmulShape, repack_bf16};
use nagaya_mk_cuda::ring::{Ring, Task};

const VARIABLE: &str = "NAGAYA_DENSE_FLAG_CHILD";
const TEST: &str = "a_matmul_task_with_the_old_unweighted_norm_bit_traps_in_each_form";
const HIDDEN: usize = 2304;
/// The bit 4 of the flags, which `FLAG_NORM_UNWEIGHTED` held.
const OLD_UNWEIGHTED_BIT: u16 = 1 << 4;

/// Run one matmul of 16 columns over one row of zeros, with the bits `extra` added to the flags. `block` picks the form.
fn run_matmul(block: bool, extra: u16) -> Result<(), String> {
    let dev = open();
    let shape = MatmulShape::new(16, HIDDEN).unwrap();
    let packed = upload(&dev, &repack_bf16(&vec![0u8; 16 * HIDDEN * 2], &shape).unwrap());
    let x = upload(&dev, &vec![0f32; HIDDEN]);
    let y = upload(&dev, &[0f32; 16]);
    let mut ring = Ring::new(&dev, shape.scratch_floats(1), false).unwrap();
    let mut desc = if block {
        shape.descriptor_block(1, x, packed, y).unwrap()
    } else {
        shape.descriptor(1, x, packed, y, ring.scratch()).unwrap()
    };
    desc.flags |= extra;
    ring.set(&[Task::with_items(desc)]);
    ring.zero_device().unwrap();
    unsafe { ring.run_shaped(288, dev.max_dynamic_shared_bytes().unwrap()) }
}

/// The child: run the task with the old bit, and expect an error of the launch.
fn child(form: &str) {
    match run_matmul(form == "block", OLD_UNWEIGHTED_BIT) {
        Err(error) => eprintln!("cuda_dense_flags: the {form} form trapped on the old bit: {error}"),
        Ok(()) => {
            eprintln!("cuda_dense_flags: the {form} form ran with the old bit, but it must trap");
            std::process::exit(3);
        }
    }
    // The context is broken. Leave without a teardown.
    std::process::exit(0);
}

#[test]
fn a_matmul_task_with_the_old_unweighted_norm_bit_traps_in_each_form() {
    if let Ok(form) = std::env::var(VARIABLE) {
        child(&form);
    }
    run_matmul(false, 0).expect("the decode form runs with no extra flag");
    run_matmul(true, 0).expect("the block form runs with no extra flag");
    let me = std::env::current_exe().expect("the path of the test executable");
    for form in ["decode", "block"] {
        let mut process = Command::new(&me)
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env(VARIABLE, form)
            .spawn()
            .expect("start the child");
        let start = Instant::now();
        let status = loop {
            if let Some(status) = process.try_wait().expect("wait for the child") {
                break status;
            }
            if start.elapsed() > Duration::from_secs(60) {
                let _ = process.kill();
                panic!("{form} form: the child did not end in 60 s; the launch hangs");
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        assert!(
            status.success(),
            "{form} form: the child ended with {status}; the matmul did not trap on the old bit"
        );
    }
}
