//! GPU gates of `OP_ATTN` at long context, with the real projection weights of
//! Inkling 2 Mini and seeded planes.
//!
//! One test runs a depth: 409,600 and 524,288 positions. Each depth runs a full run (layer 5) and a sliding run (layer 0),
//! at 1, 16 and 512 rows, in each cache format on the cubin of that format. The block of `rows` rows ends at the last
//! position, so each row reads the whole cache that is before it. The gates of each case:
//! - the accuracy against the host form `ops::attend` (the bound of the binding, 4e-6 of the row maximum, at every depth). The host form reads every
//!   position of every head, so it checks only a sample of the rows: row 0 at 1 row, rows 0 and 15 at 16 rows, and
//!   rows 0, 255 and 511 at 512 rows. The device runs all the rows;
//! - the bits of each sampled row of a block equal the bits of the same position run alone as a decode step, which
//!   checks the rows that the host form does not;
//! - for a bf16 or an e4m3 cache: the f32 form on the cubin of the f32 format, on the same values, gives the same bits.
//!
//! The speed is reported (wall time of the least launch, bytes of the cache, GB/s against 252 GB/s) and not asserted.
//!
//! `NAGAYA_TEST_CARDS=<pci address> INKLING2_MODEL_DIR=<folder> cargo test -p nagaya-mk-inkling2-sm121 --features cuda
//!  --test cuda_long_attn -- --test-threads=1 --nocapture`

#![cfg(feature = "cuda")]

mod long_support;

use long_support::*;
use nagaya_mk_cuda::attn::{self, CacheFormat, HEAD_DIM, HEADS, KV_HEADS, REL_STATES};

const FORMATS: [CacheFormat; 3] = [CacheFormat::F32, CacheFormat::Bf16, CacheFormat::E4m3];
const ROWS: [usize; 3] = [1, 16, 512];

/// The rows that the host form checks, for a block of `rows` rows.
fn host_sample(rows: usize) -> Vec<usize> {
    match rows {
        1 => vec![0],
        16 => vec![0, 15],
        _ => vec![0, 255, 511],
    }
}

/// A pool of f32 rows that hold the values of `pool` (a bf16 or an e4m3 pool): the expansion is exact.
fn expanded(pool: &Pool) -> Pool {
    let bytes = |rows: &[Vec<f32>]| -> Vec<Vec<u8>> {
        rows.iter()
            .map(|r| r.iter().flat_map(|v| v.to_le_bytes()).collect())
            .collect()
    };
    Pool {
        format: CacheFormat::F32,
        keys: pool.keys.clone(),
        values: pool.values.clone(),
        packed_keys: bytes(&pool.keys),
        packed_values: bytes(&pool.values),
    }
}

/// The row `t` of `input` as a block of one row.
fn one_row(input: &Rows, t: usize) -> Rows {
    let width = HEADS * HEAD_DIM;
    Rows {
        q: input.q[t * width..(t + 1) * width].to_vec(),
        rel: input.rel[t * HEADS * REL_STATES..(t + 1) * HEADS * REL_STATES].to_vec(),
        sinks: input.sinks[t * HEADS..(t + 1) * HEADS].to_vec(),
        gate: input.gate[t * width..(t + 1) * width].to_vec(),
    }
}

/// The cache bytes that a pass reads: both planes over the visible positions of the 8 KV heads.
fn cache_bytes(format: CacheFormat, positions: usize, window: usize) -> usize {
    let visible = positions - attn::lowest_key(positions - 1, window);
    2 * visible * KV_HEADS * format.row_bytes()
}

/// The outputs of the blocks of a case, for each count of rows.
struct Case {
    outs: Vec<Vec<f32>>,
}

fn long_depth(positions: usize) {
    let (Some(full), Some(slide)) = (layer(5), layer(0)) else {
        return;
    };
    let width = HEADS * HEAD_DIM;
    let last = positions - 1;
    let mut worst_all = 0f64;
    eprintln!(
        "cuda_long: depth {positions}: format, run, rows, worst error (host sample), bits (rows alone, f32 form), time, GB/s"
    );
    for (layer, window) in [(full, 0usize), (slide, WINDOW)] {
        let kind = if window == 0 { "full   " } else { "sliding" };
        let plane_rows = plane_rows_for(window, last);
        let inputs: Vec<(usize, Rows)> = ROWS
            .iter()
            .map(|&rows| (rows, make_rows(layer, positions - rows, rows, 53)))
            .collect();
        for format in FORMATS {
            let pool = make_pool(layer, format);
            let case = {
                let dev = open_cache(format);
                let planes = Planes::new(&dev, layer, &pool, window, plane_rows, last);
                let mut outs = Vec::new();
                for (rows, input) in &inputs {
                    let rows = *rows;
                    let pos0 = positions - rows;
                    let launches = if rows == 512 { 0 } else { 2 };
                    let run = run_attn(&dev, &planes, input, rows, pos0, 48, launches, false);
                    // The accuracy: the sample of rows against the host form.
                    let mut worst = 0f64;
                    for t in host_sample(rows) {
                        let want = host_row(layer, &pool, input, t, pos0 + t, window);
                        let (rel, _) = worst_error(&run.out[t * width..(t + 1) * width], &want);
                        let exact = exact_row(layer, &pool, input, t, pos0 + t, window);
                        let (rel_exact, _) = worst_error(&run.out[t * width..(t + 1) * width], &exact);
                        let (host_exact, _) = worst_error(&want, &exact);
                        eprintln!(
                            "cuda_long: diag {format:?} {kind} {rows} row {t}: device-host {rel:.2e}, device-f64 {rel_exact:.2e}, host-f64 {host_exact:.2e}"
                        );
                        worst = worst.max(rel);
                        // The host form follows the `f64` form: it is a sound oracle at this depth.
                        assert!(
                            host_exact <= ACCURACY_BOUND,
                            "the host form is {host_exact:e} from the f64 form at {positions} positions"
                        );
                        assert!(
                            rel <= ACCURACY_BOUND,
                            "{format:?} {kind} {rows} rows at {positions} positions: row {t} error {rel:e}, bound {ACCURACY_BOUND:e}"
                        );
                    }
                    // The rows alone: the same position as a decode step gives the bits of the block.
                    if rows > 1 {
                        for t in host_sample(rows) {
                            let alone = run_attn(&dev, &planes, &one_row(input, t), 1, pos0 + t, 48, 0, false).out;
                            assert_eq!(
                                bits(&alone),
                                bits(&run.out[t * width..(t + 1) * width]),
                                "{format:?} {kind} {rows} rows at {positions} positions: row {t} differs as a decode step"
                            );
                        }
                    }
                    let bytes = cache_bytes(format, positions, window);
                    eprintln!(
                        "cuda_long: {positions} {format:?} {kind} {rows:3} rows: error {worst:.2e}, rows-alone bits equal, {:8.1} ms, cache {:7.1} MB = {:6.1} GB/s",
                        run.wall_ns as f64 / 1e6,
                        bytes as f64 / 1e6,
                        bytes as f64 / run.wall_ns as f64
                    );
                    worst_all = worst_all.max(worst);
                    outs.push(run.out);
                }
                planes.free(&dev);
                Case { outs }
            };
            // The f32 form on the same values gives the bits of the packed form.
            if format != CacheFormat::F32 {
                let dev = open_cache(CacheFormat::F32);
                let same_values = expanded(&pool);
                let planes = Planes::new(&dev, layer, &same_values, window, plane_rows, last);
                for ((rows, input), got) in inputs.iter().zip(&case.outs) {
                    let want = run_attn(&dev, &planes, input, *rows, positions - rows, 48, 0, false).out;
                    assert_eq!(
                        bits(got),
                        bits(&want),
                        "{format:?} {kind} {rows} rows at {positions} positions: differs from the f32 form on the same values"
                    );
                }
                planes.free(&dev);
                eprintln!("cuda_long: {positions} {format:?} {kind}: bits equal to the f32 form at 1, 16 and 512 rows");
            }
        }
    }
    eprintln!(
        "cuda_long: depth {positions}: the worst error of all cases is {worst_all:.2e}, bound {ACCURACY_BOUND:.0e}"
    );
}

#[test]
fn op_attn_at_409600_positions_matches_the_host_form() {
    long_depth(409_600);
}

#[test]
fn op_attn_at_524288_positions_matches_the_host_form() {
    long_depth(524_288);
}

/// The error of one full decode row of an f32 cache by depth, against the host form and against a form with `f64`
/// sums. It is reported, to show how the error of the device grows with the count of positions.
#[test]
fn the_error_of_a_full_decode_row_by_depth_is_reported() {
    let Some(full) = layer(5) else {
        return;
    };
    let dev = open_cache(CacheFormat::F32);
    let pool = make_pool(full, CacheFormat::F32);
    for positions in [4096usize, 16_384, 65_536, 131_072, 262_144, 409_600, 524_288] {
        let last = positions - 1;
        let planes = Planes::new(&dev, full, &pool, 0, plane_rows_for(0, last), last);
        let input = make_rows(full, last, 1, 53);
        let run = run_attn(&dev, &planes, &input, 1, last, 48, 0, false);
        let (rel_host, _) = worst_error(&run.out, &host_row(full, &pool, &input, 0, last, 0));
        let (rel_exact, _) = worst_error(&run.out, &exact_row(full, &pool, &input, 0, last, 0));
        eprintln!("cuda_long: depth {positions:6}: device-host {rel_host:.2e}, device-f64 {rel_exact:.2e}");
        planes.free(&dev);
    }
}
