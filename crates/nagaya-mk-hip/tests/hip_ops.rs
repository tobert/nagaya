//! The row and element ops of `hip/ops.hip` on gfx1151, each against a CPU form that rounds to bf16 at the same
//! points. A result may differ by one bf16 step, from the order of an f32 sum. Stop `nagaya.service` first (spec 3.1).
//! Run: `cargo test -p nagaya-mk-hip --features hip --test hip_ops -- --test-threads=1`.
#![cfg(feature = "hip")]

use std::ffi::c_void;

use nagaya_mk_hip::bf16;
use nagaya_mk_hip::hip::{DeviceBuffer, Stream};
use nagaya_mk_hip::ops::{self, RopePairs};
use nagaya_mk_hip::rope;

struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    /// `n` bf16 values in [-scale, scale).
    fn vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bf16::from_f32(self.f() * scale)).collect()
    }
}

fn r(x: f32) -> f32 {
    bf16::round(x)
}

fn v(h: u16) -> f32 {
    bf16::to_f32(h)
}

/// The distance in bf16 steps between two finite values of the same sign, or a large number.
fn steps(a: u16, b: u16) -> u32 {
    let key = |h: u16| {
        if h & 0x8000 != 0 {
            -i32::from(h & 0x7fff)
        } else {
            i32::from(h)
        }
    };
    key(a).abs_diff(key(b))
}

/// Check `got` against `want` at the elements `(row, col)` of a strided layout. Each may differ by one step.
fn check(got: &[u16], want: &[f32], rows: usize, dim: usize, stride: usize, what: &str) {
    let mut worst = 0;
    for row in 0..rows {
        for col in 0..dim {
            let g = got[row * stride + col];
            let w = bf16::from_f32(want[row * dim + col]);
            let d = steps(g, w);
            assert!(d <= 1, "{what}: row {row} col {col}: got {} want {}", v(g), v(w));
            worst = worst.max(d);
        }
    }
    eprintln!("{what}: worst {worst} steps over {rows} rows of {dim}");
}

fn upload(data: &[u16]) -> DeviceBuffer {
    DeviceBuffer::from_slice(data).unwrap()
}

fn ptr(b: &DeviceBuffer) -> *mut c_void {
    b.ptr()
}

#[test]
fn rmsnorm_with_a_weight_and_with_a_zero_centred_weight() {
    let (rows, dim, stride) = (7, 4096, 4100);
    let mut rng = Rng(3);
    let x = rng.vec(rows * stride, 4.0);
    let w = rng.vec(dim, 1.0);
    let stream = Stream::new().unwrap();
    for offset in [0.0f32, 1.0] {
        let (xd, wd) = (upload(&x), upload(&w));
        let out = upload(&vec![0u16; rows * stride]);
        // SAFETY: the buffers hold the strided rows.
        unsafe {
            ops::rmsnorm(
                ptr(&xd),
                stride,
                ptr(&wd),
                offset,
                ptr(&out),
                stride,
                rows,
                dim,
                1e-6,
                &stream,
            )
            .unwrap()
        };
        stream.synchronize().unwrap();
        let mut want = vec![0.0; rows * dim];
        for row in 0..rows {
            let xs = &x[row * stride..row * stride + dim];
            let mean = xs.iter().map(|&h| f64::from(v(h)).powi(2)).sum::<f64>() / dim as f64;
            let inv = (1.0 / (mean + 1e-6).sqrt()) as f32;
            for col in 0..dim {
                want[row * dim + col] = v(xs[col]) * inv * (v(w[col]) + offset);
            }
        }
        check(
            &out.to_vec::<u16>().unwrap(),
            &want,
            rows,
            dim,
            stride,
            &format!("rmsnorm offset {offset}"),
        );
    }
}

#[test]
fn layernorm_modulate_scales_the_norm_of_each_row() {
    let (rows, dim, stride) = (5, 4096, 12288);
    let mut rng = Rng(5);
    let x: Vec<u16> = rng
        .vec(rows * stride, 3.0)
        .iter()
        .map(|&h| bf16::from_f32(v(h) + 0.75))
        .collect();
    let scale = rng.vec(dim, 0.5);
    let stream = Stream::new().unwrap();
    let (xd, sd) = (upload(&x), upload(&scale));
    let out = upload(&vec![0u16; rows * dim]);
    // SAFETY: the buffers hold the strided rows.
    unsafe { ops::layernorm_modulate(ptr(&xd), stride, ptr(&sd), ptr(&out), dim, rows, dim, 1e-6, &stream).unwrap() };
    stream.synchronize().unwrap();
    let mut want = vec![0.0; rows * dim];
    for row in 0..rows {
        let xs: Vec<f64> = x[row * stride..row * stride + dim]
            .iter()
            .map(|&h| f64::from(v(h)))
            .collect();
        let mean = xs.iter().sum::<f64>() / dim as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for col in 0..dim {
            want[row * dim + col] = r(((xs[col] - mean) * inv) as f32) * r(1.0 + v(scale[col]));
        }
    }
    check(
        &out.to_vec::<u16>().unwrap(),
        &want,
        rows,
        dim,
        dim,
        "layernorm_modulate",
    );
}

#[test]
fn gate_add_adds_the_tanh_gated_rows() {
    let (rows, dim, stride) = (9, 256, 300);
    let mut rng = Rng(7);
    let h = rng.vec(rows * stride, 8.0);
    let y = rng.vec(rows * dim, 2.0);
    let gate = rng.vec(dim, 3.0);
    let stream = Stream::new().unwrap();
    let (hd, yd, gd) = (upload(&h), upload(&y), upload(&gate));
    // SAFETY: the buffers hold the strided rows.
    unsafe { ops::gate_add(ptr(&hd), stride, ptr(&yd), dim, ptr(&gd), rows, dim, &stream).unwrap() };
    stream.synchronize().unwrap();
    let mut want = vec![0.0; rows * dim];
    for row in 0..rows {
        for col in 0..dim {
            let added = r(r(v(gate[col]).tanh()) * v(y[row * dim + col]));
            want[row * dim + col] = v(h[row * stride + col]) + added;
        }
    }
    let got = hd.to_vec::<u16>().unwrap();
    check(&got, &want, rows, dim, stride, "gate_add");
    for row in 0..rows {
        assert_eq!(
            got[row * stride + dim..(row + 1) * stride],
            h[row * stride + dim..(row + 1) * stride],
            "padding"
        );
    }
}

#[test]
fn silu_mul_reads_the_gate_half_then_the_up_half() {
    let (rows, half) = (6, 384);
    let mut rng = Rng(11);
    let x = rng.vec(rows * 2 * half, 6.0);
    let stream = Stream::new().unwrap();
    let xd = upload(&x);
    let out = upload(&vec![0u16; rows * half]);
    // SAFETY: the buffers hold the rows.
    unsafe { ops::silu_mul(ptr(&xd), 2 * half, half, ptr(&out), half, rows, &stream).unwrap() };
    stream.synchronize().unwrap();
    let mut want = vec![0.0; rows * half];
    for row in 0..rows {
        for j in 0..half {
            let g = v(x[row * 2 * half + j]);
            let u = v(x[row * 2 * half + half + j]);
            want[row * half + j] = r(g / (1.0 + (-g).exp())) * u;
        }
    }
    check(&out.to_vec::<u16>().unwrap(), &want, rows, half, half, "silu_mul");
}

#[test]
fn gelu_tanh_and_silu_in_place() {
    let n = 5000;
    let mut rng = Rng(13);
    let x = rng.vec(n, 6.0);
    let stream = Stream::new().unwrap();
    let gd = upload(&x);
    let sd = upload(&x);
    // SAFETY: the buffers hold `n` values.
    unsafe {
        ops::gelu_tanh(ptr(&gd), 1000, 5, 1000, &stream).unwrap();
        ops::silu(ptr(&sd), n, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    let gelu: Vec<f32> = x
        .iter()
        .map(|&h| {
            let a = v(h);
            0.5 * a * (1.0 + (0.797_884_6 * (a + 0.044715 * a * a * a)).tanh())
        })
        .collect();
    let silu: Vec<f32> = x.iter().map(|&h| v(h) / (1.0 + (-v(h)).exp())).collect();
    check(&gd.to_vec::<u16>().unwrap(), &gelu, 1, n, n, "gelu_tanh");
    check(&sd.to_vec::<u16>().unwrap(), &silu, 1, n, n, "silu");
}

/// The CPU form of `head_norm_rope` for one row and head.
fn norm_rope_cpu(head: &[u16], w: Option<&[u16]>, cs: &[f32], adjacent: bool) -> Vec<f32> {
    let d = head.len();
    let half = d / 2;
    let mean = head.iter().map(|&h| f64::from(v(h)).powi(2)).sum::<f64>() / d as f64;
    let inv = (1.0 / (mean + 1e-6).sqrt()) as f32;
    let norm = |i: usize| {
        let n = r(v(head[i]) * inv);
        match w {
            Some(w) => r(n * v(w[i])),
            None => n,
        }
    };
    let mut out = vec![0.0; d];
    for i in 0..half {
        let (a, b) = if adjacent { (2 * i, 2 * i + 1) } else { (i, i + half) };
        let (na, nb) = (norm(a), norm(b));
        let (c, s) = (cs[2 * i], cs[2 * i + 1]);
        out[a] = na * c - nb * s;
        out[b] = nb * c + na * s;
    }
    out
}

#[test]
fn head_norm_rope_normalizes_and_turns_each_head_for_both_pair_forms() {
    let (rows, heads, head_dim) = (11, 4, 128);
    let stride = 3 * heads * head_dim;
    let mut rng = Rng(17);
    let x = rng.vec(rows * stride, 2.0);
    let w = rng.vec(head_dim, 1.5);
    let stream = Stream::new().unwrap();
    let positions: Vec<i64> = (0..rows as i64).map(|p| p * 37).collect();
    let axes: Vec<[i64; 3]> = rope::step_positions(3, 2, 4);
    assert_eq!(axes.len(), rows);
    for (pairs, table) in [
        (RopePairs::Halves, rope::half_table(&positions, head_dim, 5_000_000.0)),
        (RopePairs::Adjacent, rope::axes_table(&axes, [16, 56, 56], 10000.0)),
    ] {
        for weight in [true, false] {
            let xd = upload(&x);
            let wd = upload(&w);
            let cs = DeviceBuffer::from_slice(&table).unwrap();
            let w_ptr = if weight { wd.ptr() } else { std::ptr::null_mut() };
            // The second third of each row: the keys of a fused q, k and v projection.
            let keys = unsafe { xd.ptr().cast::<u16>().add(heads * head_dim) }.cast();
            // SAFETY: the buffers hold the strided rows and the table.
            unsafe {
                ops::head_norm_rope(
                    keys,
                    stride,
                    rows,
                    heads,
                    head_dim,
                    w_ptr,
                    1e-6,
                    cs.ptr(),
                    pairs,
                    &stream,
                )
                .unwrap()
            };
            stream.synchronize().unwrap();
            let got = xd.to_vec::<u16>().unwrap();
            let mut want = vec![0.0; rows * heads * head_dim];
            for row in 0..rows {
                for h in 0..heads {
                    let start = row * stride + heads * head_dim + h * head_dim;
                    let head = &x[start..start + head_dim];
                    let cs_row = &table[row * head_dim..(row + 1) * head_dim];
                    let out = norm_rope_cpu(head, weight.then_some(&w[..]), cs_row, pairs == RopePairs::Adjacent);
                    want[(row * heads + h) * head_dim..(row * heads + h + 1) * head_dim].copy_from_slice(&out);
                }
            }
            let keys_got: Vec<u16> = (0..rows)
                .flat_map(|row| got[row * stride + heads * head_dim..row * stride + 2 * heads * head_dim].to_vec())
                .collect();
            check(
                &keys_got,
                &want,
                rows,
                heads * head_dim,
                heads * head_dim,
                &format!("{pairs:?} weight {weight}"),
            );
            for row in 0..rows {
                let (q, rest) = (row * stride, row * stride + 2 * heads * head_dim);
                assert_eq!(
                    got[q..q + heads * head_dim],
                    x[q..q + heads * head_dim],
                    "the queries stay"
                );
                assert_eq!(
                    got[rest..(row + 1) * stride],
                    x[rest..(row + 1) * stride],
                    "the values stay"
                );
            }
        }
    }
}

#[test]
fn gather_rows_copies_the_rows_of_the_ids() {
    let (vocab, dim) = (50, 96);
    let mut rng = Rng(19);
    let table = rng.vec(vocab * dim, 1.0);
    let ids: Vec<i32> = vec![7, 0, 49, 7, 13];
    let stream = Stream::new().unwrap();
    let td = upload(&table);
    let id = DeviceBuffer::from_slice(&ids).unwrap();
    let out = upload(&vec![0u16; ids.len() * dim]);
    // SAFETY: the ids are rows of the table.
    unsafe { ops::gather_rows(ptr(&td), id.ptr(), ptr(&out), ids.len(), dim, &stream).unwrap() };
    stream.synchronize().unwrap();
    let got = out.to_vec::<u16>().unwrap();
    for (row, &id) in ids.iter().enumerate() {
        let id = id as usize;
        assert_eq!(
            got[row * dim..(row + 1) * dim],
            table[id * dim..(id + 1) * dim],
            "row {row}"
        );
    }
}
