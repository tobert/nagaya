//! The hipBLASLt GEMMs on gfx1151, each against a plain CPU form on random inputs (spec 12.1, "The plan table of a HIP
//! engine"). Stop `nagaya.service` first: the nagaya must be the only GPU client (spec 3.1).
//! Run: `cargo test -p nagaya-mk-hip --features hip --test hip_gemm -- --test-threads=1`.
#![cfg(feature = "hip")]

use nagaya_mk_hip::bf16;
use nagaya_mk_hip::gemm::{Gemms, lineage};
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::plan::{DataType, GemmShape, PlanTable};

/// A small random source: xorshift64*.
struct Rng(u64);

impl Rng {
    /// A value in [-1, 1), rounded to bf16.
    fn bf16(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        bf16::round((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.bf16()).collect()
    }
}

fn bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| bf16::from_f32(v)).collect()
}

fn values(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| bf16::to_f32(b)).collect()
}

/// `y[r, o] = sum_i x[r, i] w[o, i]`, in f64.
fn linear_cpu(x: &[f32], w: &[f32], rows: usize, inputs: usize, outputs: usize) -> Vec<f32> {
    let mut y = vec![0.0; rows * outputs];
    for r in 0..rows {
        for o in 0..outputs {
            let sum: f64 = (0..inputs)
                .map(|i| f64::from(x[r * inputs + i]) * f64::from(w[o * inputs + i]))
                .sum();
            y[r * outputs + o] = sum as f32;
        }
    }
    y
}

/// The largest error over the largest magnitude of the CPU form.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    worst / scale
}

fn gemms() -> Gemms {
    Gemms::new(PlanTable::in_memory(&lineage().unwrap()), 32 << 20).unwrap()
}

#[test]
fn a_linear_layer_matches_the_cpu_form() {
    let (rows, inputs, outputs) = (37, 160, 96);
    let mut rng = Rng(7);
    let x = rng.vec(rows * inputs);
    let w = rng.vec(outputs * inputs);
    let stream = Stream::new().unwrap();
    let mut gemms = gemms();
    let shape = GemmShape::linear(rows as u32, inputs as u32, outputs as u32, DataType::Bf16);
    let xd = DeviceBuffer::from_slice(&bits(&x)).unwrap();
    let wd = DeviceBuffer::from_slice(&bits(&w)).unwrap();
    let yd = DeviceBuffer::new(rows * outputs * 2).unwrap();
    // SAFETY: the buffers hold the elements of the shape.
    unsafe {
        gemms
            .run(&shape, 1.0, wd.ptr(), xd.ptr(), 0.0, yd.ptr(), yd.ptr(), &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let got = values(&yd.to_vec::<u16>().unwrap());
    let want = linear_cpu(&x, &w, rows, inputs, outputs);
    let error = rel(&got, &want);
    assert!(error < 4e-3, "relative error {error}");
}

#[test]
fn beta_one_adds_the_product_to_c_in_place() {
    let (rows, inputs, outputs) = (19, 64, 48);
    let mut rng = Rng(11);
    let x = rng.vec(rows * inputs);
    let w = rng.vec(outputs * inputs);
    let residual = rng.vec(rows * outputs);
    let stream = Stream::new().unwrap();
    let mut gemms = gemms();
    let shape = GemmShape::linear(rows as u32, inputs as u32, outputs as u32, DataType::Bf16);
    let xd = DeviceBuffer::from_slice(&bits(&x)).unwrap();
    let wd = DeviceBuffer::from_slice(&bits(&w)).unwrap();
    let yd = DeviceBuffer::from_slice(&bits(&residual)).unwrap();
    // SAFETY: the buffers hold the elements of the shape, and C and D are the same buffer.
    unsafe {
        gemms
            .run(&shape, 1.0, wd.ptr(), xd.ptr(), 1.0, yd.ptr(), yd.ptr(), &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let got = values(&yd.to_vec::<u16>().unwrap());
    let want: Vec<f32> = linear_cpu(&x, &w, rows, inputs, outputs)
        .iter()
        .zip(&residual)
        .map(|(p, r)| p + r)
        .collect();
    let error = rel(&got, &want);
    assert!(error < 4e-3, "relative error {error}");
}

#[test]
fn an_f32_output_keeps_more_bits_than_bf16() {
    let (rows, inputs, outputs) = (8, 512, 32);
    let mut rng = Rng(13);
    let x = rng.vec(rows * inputs);
    let w = rng.vec(outputs * inputs);
    let stream = Stream::new().unwrap();
    let mut gemms = gemms();
    let shape = GemmShape::linear(rows as u32, inputs as u32, outputs as u32, DataType::F32);
    let xd = DeviceBuffer::from_slice(&bits(&x)).unwrap();
    let wd = DeviceBuffer::from_slice(&bits(&w)).unwrap();
    let yd = DeviceBuffer::new(rows * outputs * 4).unwrap();
    // SAFETY: the buffers hold the elements of the shape.
    unsafe {
        gemms
            .run(&shape, 1.0, wd.ptr(), xd.ptr(), 0.0, yd.ptr(), yd.ptr(), &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let got = yd.to_vec::<f32>().unwrap();
    let want = linear_cpu(&x, &w, rows, inputs, outputs);
    let error = rel(&got, &want);
    assert!(
        error < 1e-5,
        "relative error {error}: an f32 output must not round to bf16"
    );
}

#[test]
fn a_strided_batch_runs_each_matrix_on_its_own_inputs() {
    // Three products of 24 by 40 and 40 by 16, as attention runs one for each head.
    let (batch, m, n, k) = (3usize, 24usize, 16usize, 40usize);
    let mut rng = Rng(17);
    // Column-major: A is m by k with lda m, B is k by n with ldb k, C is m by n with ldc m.
    let a = rng.vec(batch * m * k);
    let b = rng.vec(batch * k * n);
    let shape = GemmShape {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        trans_a: false,
        trans_b: false,
        lda: m as u64,
        ldb: k as u64,
        ldc: m as u64,
        batch: batch as u32,
        stride_a: (m * k) as u64,
        stride_b: (k * n) as u64,
        stride_c: (m * n) as u64,
        type_ab: DataType::Bf16,
        type_cd: DataType::F32,
        bias: false,
    };
    let stream = Stream::new().unwrap();
    let mut gemms = gemms();
    let ad = DeviceBuffer::from_slice(&bits(&a)).unwrap();
    let bd = DeviceBuffer::from_slice(&bits(&b)).unwrap();
    let cd = DeviceBuffer::new(batch * m * n * 4).unwrap();
    // SAFETY: the buffers hold the elements of the batch.
    unsafe {
        gemms
            .run(&shape, 1.0, ad.ptr(), bd.ptr(), 0.0, cd.ptr(), cd.ptr(), &stream)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let got = cd.to_vec::<f32>().unwrap();
    let mut want = vec![0.0f32; batch * m * n];
    for s in 0..batch {
        for j in 0..n {
            for i in 0..m {
                let sum: f64 = (0..k)
                    .map(|p| f64::from(a[s * m * k + p * m + i]) * f64::from(b[s * k * n + j * k + p]))
                    .sum();
                want[s * m * n + j * m + i] = sum as f32;
            }
        }
    }
    let error = rel(&got, &want);
    assert!(error < 1e-5, "relative error {error}");
}

#[test]
fn a_pick_goes_to_the_file_and_the_next_open_uses_it() {
    let dir = tempfile::tempdir().unwrap();
    let lineage = lineage().unwrap();
    let shape = GemmShape::linear(64, 4096, 1024, DataType::Bf16);
    let stream = Stream::new().unwrap();
    let mut first = Gemms::new(PlanTable::open(dir.path(), &lineage).unwrap(), 32 << 20).unwrap();
    let index = first.prepare(&shape, &stream).unwrap();
    assert_eq!(first.table().pick(&shape), Some(index));
    let table = PlanTable::open(dir.path(), &lineage).unwrap();
    assert_eq!(table.pick(&shape), Some(index), "the file holds the pick");
    let mut second = Gemms::new(table, 32 << 20).unwrap();
    assert_eq!(second.prepare(&shape, &stream).unwrap(), index);
    assert_eq!(second.table().len(), 1);
}

#[test]
fn a_pick_comes_from_the_heuristic_candidates_and_the_name_rule_allows_it() {
    let shape = GemmShape::linear(4096, 4096, 4096, DataType::Bf16);
    let stream = Stream::new().unwrap();
    let mut gemms = gemms();
    let index = gemms.prepare(&shape, &stream).unwrap();
    let candidates = gemms.blaslt().candidates(&shape, 32 << 20, 16).unwrap();
    assert!(
        candidates.iter().any(|(candidate, _)| *candidate == index),
        "the pick {index} is a heuristic candidate"
    );
    let name = gemms.blaslt().kernel_name(index).unwrap();
    assert_eq!(nagaya_mk_hip::plan::refused(&name), None, "{name}");
}

#[test]
fn a_bias_adds_one_value_to_each_output_column() {
    let (rows, inputs, outputs) = (21, 96, 40);
    let mut rng = Rng(23);
    let x = rng.vec(rows * inputs);
    let w = rng.vec(outputs * inputs);
    let bias = rng.vec(outputs);
    let residual = rng.vec(rows * outputs);
    let stream = Stream::new().unwrap();
    let mut gemms = gemms();
    let shape = GemmShape::linear(rows as u32, inputs as u32, outputs as u32, DataType::Bf16).with_bias();
    let xd = DeviceBuffer::from_slice(&bits(&x)).unwrap();
    let wd = DeviceBuffer::from_slice(&bits(&w)).unwrap();
    let bd = DeviceBuffer::from_slice(&bits(&bias)).unwrap();
    let yd = DeviceBuffer::from_slice(&bits(&residual)).unwrap();
    // SAFETY: the buffers hold the elements of the shape and the bias, and C and D are the same buffer.
    unsafe {
        gemms
            .run_bias(
                &shape,
                1.0,
                wd.ptr(),
                xd.ptr(),
                1.0,
                yd.ptr(),
                yd.ptr(),
                bd.ptr(),
                &stream,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    let got = values(&yd.to_vec::<u16>().unwrap());
    let want: Vec<f32> = linear_cpu(&x, &w, rows, inputs, outputs)
        .iter()
        .enumerate()
        .map(|(i, p)| p + bias[i % outputs] + residual[i])
        .collect();
    let error = rel(&got, &want);
    assert!(error < 4e-3, "relative error {error}");
}
