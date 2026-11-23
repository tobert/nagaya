//! GPU gate of the cubins of the cache formats (milestone 37 step 11): a cubin of one format runs a task of that format,
//! and it traps on a task of another format, in `OP_ATTN`. The default cubin holds the f32 and the e4m3 formats, and it
//! traps on a bf16 task. The buffers hold zeros, so a missing check lets the launch succeed, and the test shows it. The
//! model is not needed.
//!
//! A trap breaks the CUDA context of its process. The test runs each variant in a child process: it starts its own
//! executable again, with the variant in `NAGAYA_CACHE_VARIANT`. The child leaves at once, with no teardown.
//!
//! `NAGAYA_TEST_CARDS=<pci address> cargo test -p nagaya-mk-inkling2-sm121 --features cuda --test cuda_cache_traps
//!  -- --test-threads=1`

#![cfg(feature = "cuda")]

use std::process::Command;
use std::time::{Duration, Instant};

use nagaya_mk_cuda::attn::{self, AttnBuffers, AttnShape, CacheFormat, plane_bytes};
use nagaya_mk_cuda::driver::CudaDevice;
use nagaya_mk_cuda::ring::{Ring, Task};
use nagaya_mk_cuda::{INKLING2_CUBIN, test_cards};

const VARIABLE: &str = "NAGAYA_CACHE_VARIANT";
const TEST: &str = "a_cubin_of_one_format_traps_on_a_task_of_another_format";

/// A variant: the cubin (`None` is the default cubin), the format of the task, and whether the launch must run.
const VARIANTS: [(&str, Option<CacheFormat>, CacheFormat, bool); 11] = [
    ("default_f32", None, CacheFormat::F32, true),
    ("default_e4m3", None, CacheFormat::E4m3, true),
    ("default_bf16", None, CacheFormat::Bf16, false),
    ("f32_f32", Some(CacheFormat::F32), CacheFormat::F32, true),
    ("f32_e4m3", Some(CacheFormat::F32), CacheFormat::E4m3, false),
    ("f32_bf16", Some(CacheFormat::F32), CacheFormat::Bf16, false),
    ("e4m3_e4m3", Some(CacheFormat::E4m3), CacheFormat::E4m3, true),
    ("e4m3_f32", Some(CacheFormat::E4m3), CacheFormat::F32, false),
    ("bf16_bf16", Some(CacheFormat::Bf16), CacheFormat::Bf16, true),
    ("bf16_f32", Some(CacheFormat::Bf16), CacheFormat::F32, false),
    ("bf16_e4m3", Some(CacheFormat::Bf16), CacheFormat::E4m3, false),
];

fn child(variant: &str) {
    let (_, cubin, format, expect_ok) = *VARIANTS.iter().find(|v| v.0 == variant).expect("a known variant");
    let image = cubin.map_or(INKLING2_CUBIN, CacheFormat::cubin);
    let dev = CudaDevice::open_by_pci(&test_cards::card(), image).expect("open the GPU");
    let zeros = |dev: &CudaDevice, bytes: usize| {
        let p = dev.alloc_device(bytes).unwrap();
        dev.memset_zero(p, bytes).unwrap();
        p
    };
    let (rows, plane_rows) = (1usize, 64usize);
    let shape = AttnShape::new(rows, 0, 0, plane_rows, format).unwrap();
    let q = zeros(&dev, rows * 24 * 128 * 4);
    let rel = zeros(&dev, rows * 24 * 16 * 4);
    let sinks = zeros(&dev, rows * 24 * 4);
    let gate = zeros(&dev, rows * 24 * 128 * 4);
    let proj = zeros(&dev, 16 * 1024 * 2);
    let planes = plane_bytes(plane_rows, format);
    let (k, v) = (zeros(&dev, planes), zeros(&dev, planes));
    let out = zeros(&dev, rows * 24 * 128 * 4);
    let table_host = [attn::table(rel, sinks, gate, proj)];
    let table = dev.alloc_device(std::mem::size_of_val(&table_host)).unwrap();
    unsafe { dev.copy_htod(table, table_host.as_ptr().cast(), std::mem::size_of_val(&table_host)) }.unwrap();
    let mut ring = Ring::new(&dev, shape.scratch_floats(), false).unwrap();
    ring.zero_device().unwrap();
    let desc = shape.descriptor(&AttnBuffers {
        q,
        k,
        v,
        out,
        scratch: ring.scratch(),
        table,
    });
    ring.set(&[Task::with_items(desc)]);
    let result = unsafe { ring.run() };
    match (result, expect_ok) {
        (Ok(()), true) => eprintln!("cuda_cache_traps: variant {variant} ran with no error"),
        (Err(error), false) => eprintln!("cuda_cache_traps: variant {variant} trapped: {error}"),
        (Ok(()), false) => {
            eprintln!("cuda_cache_traps: variant {variant} ran with no error, but it must trap");
            std::process::exit(3);
        }
        (Err(error), true) => {
            eprintln!("cuda_cache_traps: variant {variant} must run, and it failed: {error}");
            std::process::exit(4);
        }
    }
    std::process::exit(0);
}

#[test]
fn a_cubin_of_one_format_traps_on_a_task_of_another_format() {
    if let Ok(value) = std::env::var(VARIABLE) {
        child(&value);
    }
    let me = std::env::current_exe().expect("the path of the test executable");
    for (name, ..) in VARIANTS {
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
