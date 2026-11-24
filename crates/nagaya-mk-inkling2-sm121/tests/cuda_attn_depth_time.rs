//! The kernel time of `OP_ATTN` at depth, from the profile cubin of each cache format (lane T). It reports and asserts
//! nothing: the wall time of a launch holds about 10 ms of launch cost, and the profile holds the time of the task.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_attn_depth_time -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod long_support;

use long_support::*;
use nagaya_mk_cuda::attn::CacheFormat;

#[test]
fn the_kernel_time_of_a_full_run_by_depth_is_reported() {
    let Some(full) = layer(5) else {
        return;
    };
    let depths: Vec<usize> = std::env::var("DEPTHS").map_or_else(
        |_| vec![8192, 131_072, 409_600, 524_288],
        |s| s.split(',').map(|x| x.parse().unwrap()).collect(),
    );
    for format in [CacheFormat::F32, CacheFormat::Bf16, CacheFormat::E4m3] {
        let pool = make_pool(full, format);
        let dev = open_profile_cache(format);
        for &positions in &depths {
            let last = positions - 1;
            let planes = Planes::new(&dev, full, &pool, 0, plane_rows_for(0, last), last);
            for rows in [512usize, 1] {
                let input = make_rows(full, positions - rows, rows, 53);
                let run = run_attn(&dev, &planes, &input, rows, positions - rows, 48, 2, true);
                eprintln!(
                    "cuda_depth_time: {format:?} full {rows:3} rows at {positions:6} positions: kernel {:9.3} ms",
                    run.kernel_ns as f64 / 1e6
                );
            }
            planes.free(&dev);
        }
    }
}
