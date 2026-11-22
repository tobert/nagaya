//! GPU gate: an opcode that is reserved for a lane of wave B, and an opcode that no one owns, trap in `run_op`.
//!
//! A trap breaks the CUDA context of its process. The test runs each launch in a child process: it starts its own
//! executable again, with the opcode in `NAGAYA_RESERVED_OPCODE`. The child exits at once, with no teardown.
//!
//! Run with one thread and a free GPU PCI address:
//! `NAGAYA_TEST_CARDS=0000:0a:00.0 cargo test -p nagaya-mk-cuda --features cuda --test cuda_reserved -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::process::Command;
use std::time::{Duration, Instant};

use nagaya_mk_cuda::INKLING2_CUBIN;
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_OUT_MIX, OP_SELFTEST_TABLE};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::test_cards;

const VARIABLE: &str = "NAGAYA_RESERVED_OPCODE";
const TEST: &str = "each_reserved_opcode_traps_until_its_lane_builds_it";

/// The opcodes that wave B still reserves (the lanes built the others), then two numbers in the range of a lane that no constant names, one number
/// that the table of the B70 used and the GB10 does not, and a test op that needs a task table and has none.
const TRAPPING: [(&str, u16); 5] = [
    ("OP_OUT_MIX", OP_OUT_MIX),
    ("a free number of lane D", 46),
    ("a free number of lane M", 60),
    ("an old number of the B70 table", 142),
    ("OP_SELFTEST_TABLE with no task table", OP_SELFTEST_TABLE),
];

/// The child: launch one task with `opcode`, and expect an error of the launch.
fn child(opcode: u16) {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), INKLING2_CUBIN).expect("open the GPU");
    let mut ring = Ring::new(&dev, 16, false).unwrap();
    ring.zero_device().unwrap();
    ring.set(&[Task::with_items(MkTaskDescriptor {
        opcode,
        items: 8,
        ..Default::default()
    })]);
    match unsafe { ring.run() } {
        Err(error) => eprintln!("cuda_reserved: opcode {opcode} trapped: {error}"),
        Ok(()) => {
            eprintln!("cuda_reserved: opcode {opcode} ran with no error, but it must trap");
            std::process::exit(3);
        }
    }
    // The context is broken. Leave without a teardown.
    std::process::exit(0);
}

#[test]
fn each_reserved_opcode_traps_until_its_lane_builds_it() {
    if let Ok(value) = std::env::var(VARIABLE) {
        child(value.parse().expect("the child gets an opcode number"));
    }
    let me = std::env::current_exe().expect("the path of the test executable");
    for (name, opcode) in TRAPPING {
        let mut process = Command::new(&me)
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env(VARIABLE, opcode.to_string())
            .spawn()
            .expect("start the child");
        let start = Instant::now();
        let status = loop {
            if let Some(status) = process.try_wait().expect("wait for the child") {
                break status;
            }
            if start.elapsed() > Duration::from_secs(60) {
                let _ = process.kill();
                panic!("{name} ({opcode}): the child did not end in 60 s; the launch hangs");
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        assert!(
            status.success(),
            "{name} ({opcode}): the child ended with {status}; the opcode did not trap"
        );
    }
}
