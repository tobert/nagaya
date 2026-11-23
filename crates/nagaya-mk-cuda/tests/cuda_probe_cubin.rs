//! GPU gate: only the probe cubin holds the probe ops. The engine cubin traps on a probe opcode, and the probe cubin
//! runs the same task.
//!
//! The task is a valid `OP_PROBE_TIMER`: a task with empty fields would trap in the probe cubin too, and the test could
//! not tell the two cubins apart. The parent runs the task in the probe cubin, and it checks the result. A trap breaks
//! the CUDA context of its process, so the engine cubin runs the task in a child process: the test starts its own
//! executable again, with `NAGAYA_PROBE_CUBIN_CHILD` set. The child exits at once, with no teardown.
//!
//! Run with one thread and a free GPU PCI address:
//! `NAGAYA_TEST_CARDS=0000:0a:00.0 cargo test -p nagaya-mk-cuda --features cuda --test cuda_probe_cubin -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::process::Command;
use std::time::{Duration, Instant};

use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, OP_PROBE_TIMER};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, INKLING2_PROBE_CUBIN, test_cards};

const VARIABLE: &str = "NAGAYA_PROBE_CUBIN_CHILD";
const TEST: &str = "the_engine_cubin_traps_on_a_probe_opcode_and_the_probe_cubin_runs_it";
const READS: u32 = 64;

/// Run one `OP_PROBE_TIMER` of `READS` reads in `cubin`. The result is the word 3 of the output: the count of reads.
fn run_timer(cubin: &[u8]) -> Result<u64, String> {
    let dev = CudaDevice::open_by_pci(&test_cards::card(), cubin).expect("open the GPU");
    let out = dev.alloc_device(32).expect("allocate");
    dev.memset_zero(out, 32).expect("clear the output");
    let mut ring = Ring::new(&dev, 16, false).unwrap();
    ring.zero_device().unwrap();
    ring.set(&[Task::per_block(MkTaskDescriptor {
        opcode: OP_PROBE_TIMER,
        c: out,
        n: READS,
        ..Default::default()
    })]);
    unsafe { ring.run() }?;
    let mut words = [0u64; 4];
    unsafe { dev.copy_dtoh(words.as_mut_ptr().cast(), out, 32) }?;
    Ok(words[3])
}

/// The child: run the task in the engine cubin, and expect an error of the launch.
fn child() {
    match run_timer(INKLING2_CUBIN) {
        Err(error) => eprintln!("cuda_probe_cubin: the engine cubin trapped on a probe opcode: {error}"),
        Ok(reads) => {
            eprintln!("cuda_probe_cubin: the engine cubin ran a probe opcode ({reads} reads), but it must trap");
            std::process::exit(3);
        }
    }
    // The context is broken. Leave without a teardown.
    std::process::exit(0);
}

#[test]
fn the_engine_cubin_traps_on_a_probe_opcode_and_the_probe_cubin_runs_it() {
    if std::env::var(VARIABLE).is_ok() {
        child();
    }
    let reads = run_timer(INKLING2_PROBE_CUBIN).expect("the probe cubin runs the probe task");
    assert_eq!(
        reads,
        u64::from(READS),
        "the probe cubin did not run the task of the probe"
    );

    let me = std::env::current_exe().expect("the path of the test executable");
    let mut process = Command::new(&me)
        .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
        .env(VARIABLE, "1")
        .spawn()
        .expect("start the child");
    let start = Instant::now();
    let status = loop {
        if let Some(status) = process.try_wait().expect("wait for the child") {
            break status;
        }
        if start.elapsed() > Duration::from_secs(60) {
            let _ = process.kill();
            panic!("the child did not end in 60 s; the launch hangs");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(
        status.success(),
        "the child ended with {status}; the engine cubin did not trap on the probe opcode"
    );
}
