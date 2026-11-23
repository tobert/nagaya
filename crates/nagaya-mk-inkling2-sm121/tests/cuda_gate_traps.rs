//! GPU gate of the row checks of `OP_GATE_DOWN` and `OP_GATE_UP` (milestone 37 step 9, lane M2): a task with more rows
//! than the op computes traps, and a task with the rows that it computes runs. The task has real buffers of zeros, so
//! a check that is missing lets the launch succeed, and the test shows it. The model is not needed.
//!
//! A trap breaks the CUDA context of its process. The test runs each task in a child process: it starts its own
//! executable again, with the variant in `NAGAYA_GATE_VARIANT`. The child leaves at once, with no teardown.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_gate_traps
//!  -- --test-threads=1`

#![cfg(feature = "cuda")]

mod common;

use std::process::Command;
use std::time::{Duration, Instant};

use common::gpu::open;
use nagaya_mk_cuda::protocol::{MkTaskDescriptor, MkTaskTable, OP_GATE_DOWN, OP_GATE_UP};
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::rows::{GATE_UP_ITEM_BYTES, GATE_UP_TILES, gate_down_shape};

const VARIABLE: &str = "NAGAYA_GATE_VARIANT";
const TEST: &str = "a_gate_task_with_too_many_rows_traps_and_a_good_task_runs";

/// The variants: the op, the rows, and whether the launch must run.
const VARIANTS: [(&str, u32, bool); 4] = [
    ("down", 16, true),
    ("down", 17, false),
    ("up", 16, true),
    ("up", 17, false),
];

/// The child: launch one task, and exit with 0 when the launch did what the variant expects.
fn child(variant: &str) {
    let (op, rows, expect_ok) = {
        let mut parts = variant.split(':');
        let op = parts.next().unwrap().to_string();
        let rows: u32 = parts.next().unwrap().parse().unwrap();
        let ok = VARIANTS
            .iter()
            .find(|(o, r, _)| *o == op && *r == rows)
            .expect("a known variant")
            .2;
        (op, rows, ok)
    };
    let dev = open();
    let zeros = |bytes: usize| {
        let p = dev.alloc_device(bytes).unwrap();
        dev.memset_zero(p, bytes).unwrap();
        p
    };
    let n = rows as usize;
    let shape = gate_down_shape();
    let (a, m) = (zeros(n * 2304 * 4), zeros(n * 2304 * 4));
    let (rf, u, u_rf) = (zeros(n * 4), zeros(n * 2304 * 4), zeros(n * 4));
    let g = zeros(n * 64 * 4);
    let down = zeros(shape.packed_bytes());
    let up = zeros(GATE_UP_TILES * GATE_UP_ITEM_BYTES);
    let mut table = MkTaskTable::default();
    let desc = if op == "down" {
        table.ptr[..12].copy_from_slice(&[a, 0, 0, m, 0, 0, rf, 0, 0, rf, 0, 0]);
        MkTaskDescriptor {
            opcode: OP_GATE_DOWN,
            items: 24,
            rows,
            b: down,
            c: g,
            n: 1,
            k: shape.k as u32,
            kt: shape.kt as u32,
            group_items: shape.slices() as u32,
            item_bytes: shape.item_bytes() as u32,
            ..Default::default()
        }
    } else {
        table.ptr[..12].copy_from_slice(&[a, 0, 0, m, 0, 0, u, 0, 0, u_rf, 0, 0]);
        MkTaskDescriptor {
            opcode: OP_GATE_UP,
            items: 36,
            rows,
            a: g,
            b: up,
            n: 1,
            k: 64,
            item_bytes: GATE_UP_ITEM_BYTES as u32,
            ..Default::default()
        }
    };
    let table_ptr = dev.alloc_device(std::mem::size_of::<MkTaskTable>()).unwrap();
    unsafe { dev.copy_htod(table_ptr, (&raw const table).cast(), std::mem::size_of::<MkTaskTable>()) }.unwrap();
    // Room for 512 rows of either op, so a missing check cannot fault on the scratch.
    let mut ring = Ring::new(&dev, 16 + 24 * 512 * 16 + 36 * 512 + 8, false).unwrap();
    ring.zero_device().unwrap();
    let mut desc = desc;
    desc.d = table_ptr;
    desc.f = ring.scratch();
    ring.set(&[Task::with_items(desc)]);
    let result = unsafe { ring.run() };
    match (result, expect_ok) {
        (Ok(()), true) => eprintln!("cuda_gate_traps: variant {variant} ran with no error"),
        (Err(error), false) => eprintln!("cuda_gate_traps: variant {variant} trapped: {error}"),
        (Ok(()), false) => {
            eprintln!("cuda_gate_traps: variant {variant} ran with no error, but it must trap");
            std::process::exit(3);
        }
        (Err(error), true) => {
            eprintln!("cuda_gate_traps: variant {variant} must run, and it failed: {error}");
            std::process::exit(4);
        }
    }
    std::process::exit(0);
}

#[test]
fn a_gate_task_with_too_many_rows_traps_and_a_good_task_runs() {
    if let Ok(value) = std::env::var(VARIABLE) {
        child(&value);
    }
    let me = std::env::current_exe().expect("the path of the test executable");
    for (op, rows, _) in VARIANTS {
        let name = format!("{op}:{rows}");
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
