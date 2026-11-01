//! Time each hipBLASLt candidate that the plan table may pick for the large GEMMs of a Qwen-Image step at 1024 by 1024:
//! the candidates that the name rule allows and that give the same bits in two runs (spec 12.1). It prints the time of
//! each one, the first one (the pick of today's rule), and the fastest one.
//! Stop `nagaya.service` first (spec 3.1). Run:
//! `cargo run --release -p nagaya-mk-hip --features hip --example gemm_timing`.

use std::time::Instant;

use nagaya_mk_hip::gemm::Blaslt;
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::plan::{DataType, GemmShape, refused};

/// Bytes of bf16 values in [-1, 1) from a small generator.
fn values(count: usize, seed: u64) -> Vec<u16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
            (x.to_bits() >> 16) as u16
        })
        .collect()
}

fn main() {
    const MAX_WORKSPACE: u64 = 64 << 20;
    const RUNS: u32 = 10;
    let blaslt = Blaslt::new().unwrap();
    let stream = Stream::new().unwrap();
    let workspace = DeviceBuffer::new(MAX_WORKSPACE as usize).unwrap();
    // (rows, inputs, outputs, count in a step): qkv, out, MLP in, MLP out, and the two other projections of a block,
    // times the 32 blocks.
    let shapes = [
        (4096, 4096, 12288, 32),
        (4096, 4096, 4096, 96),
        (4096, 4096, 24576, 32),
        (4096, 12288, 4096, 32),
    ];
    let (mut first_total, mut fastest_total) = (0.0f64, 0.0f64);
    for (rows, inputs, outputs, count) in shapes {
        let shape = GemmShape::linear(rows, inputs, outputs, DataType::Bf16);
        let (a_count, b_count, c_count) = shape.elements();
        let a = DeviceBuffer::from_slice(&values(a_count, 1)).unwrap();
        let b = DeviceBuffer::from_slice(&values(b_count, 2)).unwrap();
        let c = DeviceBuffer::from_slice(&values(c_count, 3)).unwrap();
        let one = DeviceBuffer::new(c_count * 2).unwrap();
        let two = DeviceBuffer::new(c_count * 2).unwrap();
        println!("== rows {rows} in {inputs} out {outputs}, {count} in a step");
        let mut timed = Vec::new();
        for (index, _) in blaslt.candidates(&shape, MAX_WORKSPACE, 16).unwrap() {
            let name = blaslt.kernel_name(index).unwrap();
            if let Some(why) = refused(&name) {
                println!("  {index:>6} refused: {why}");
                continue;
            }
            let Ok(gemm) = blaslt.prepare(&shape, index, MAX_WORKSPACE) else {
                continue;
            };
            let run = |out: &DeviceBuffer| unsafe {
                blaslt
                    .run(
                        &gemm,
                        1.0,
                        a.ptr(),
                        b.ptr(),
                        0.0,
                        c.ptr(),
                        out.ptr(),
                        std::ptr::null(),
                        &workspace,
                        &stream,
                    )
                    .unwrap()
            };
            one.write(&vec![0u8; c_count * 2]).unwrap();
            two.write(&vec![0xffu8; c_count * 2]).unwrap();
            run(&one);
            run(&two);
            stream.synchronize().unwrap();
            let same = shape.output_elements().all({
                let (x, y) = (one.to_vec::<u16>().unwrap(), two.to_vec::<u16>().unwrap());
                move |e| x[e] == y[e]
            });
            let started = Instant::now();
            for _ in 0..RUNS {
                run(&one);
            }
            stream.synchronize().unwrap();
            let ms = started.elapsed().as_secs_f64() * 1e3 / f64::from(RUNS);
            let tflops = 2.0 * f64::from(rows) * f64::from(inputs) * f64::from(outputs) / (ms * 1e-3) / 1e12;
            println!("  {index:>6} {ms:>8.2} ms {tflops:>5.1} TFLOPS same bits {same} {name}");
            if same {
                timed.push((index, ms));
            }
        }
        let first = timed.first().copied().unwrap();
        let fastest = timed
            .iter()
            .copied()
            .fold(first, |best, x| if x.1 < best.1 { x } else { best });
        println!(
            "  first {} at {:.2} ms, fastest {} at {:.2} ms: {:.0} ms against {:.0} ms in a step",
            first.0,
            first.1,
            fastest.0,
            fastest.1,
            first.1 * f64::from(count),
            fastest.1 * f64::from(count)
        );
        first_total += first.1 * f64::from(count);
        fastest_total += fastest.1 * f64::from(count);
    }
    println!("a step: the first picks {first_total:.0} ms, the fastest picks {fastest_total:.0} ms");
}
