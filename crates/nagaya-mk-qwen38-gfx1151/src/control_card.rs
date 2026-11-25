//! The card gate of the device control op (`hip/src/engine/control.hip`): it gives the bits of the host op of
//! `control.rs` for each op, each row and each value.
//!
//! The rows and the vector come from a linear congruential generator, as the tests of `control.rs` make them. Their
//! dot products depend on the order of the sum, so a kernel with another order or a fused product fails here. The test
//! opens no engine: it allocates two buffers and calls the op. Run it with `nagaya.service` stopped, as each GPU test.

use nagaya_abi::ControlOp;

use crate::binding::{self, rt};
use crate::context::Copy as CopyKind;
use crate::control::{self, WIDTH};

/// The rows of a call: the most rows of a pass.
const ROWS: usize = 256;

/// A row of `n` values in `[-scale / 2, scale / 2)`, from the generator of the tests of `control.rs`.
fn lcg(seed: u32, n: usize, scale: f64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((f64::from(state >> 8) / 16_777_216.0 - 0.5) * scale) as f32
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Two device buffers: `ROWS` rows and one vector.
struct Buffers {
    h: *mut f32,
    v: *mut f32,
}

impl Buffers {
    fn new(v: &[f32]) -> Buffers {
        let h = unsafe { rt::hip_rt_malloc((ROWS * WIDTH * 4) as u64) } as *mut f32;
        let device_v = unsafe { rt::hip_rt_malloc((WIDTH * 4) as u64) } as *mut f32;
        assert!(!h.is_null() && !device_v.is_null(), "hip_rt_malloc gave no buffer");
        unsafe {
            rt::hip_rt_memcpy(
                device_v.cast(),
                v.as_ptr().cast(),
                (WIDTH * 4) as u64,
                CopyKind::HostToDevice as u32,
            )
        };
        Buffers { h, v: device_v }
    }

    fn put(&self, rows: &[f32]) {
        unsafe {
            rt::hip_rt_memcpy(
                self.h.cast(),
                rows.as_ptr().cast(),
                (rows.len() * 4) as u64,
                CopyKind::HostToDevice as u32,
            )
        };
    }

    fn get(&self) -> Vec<f32> {
        let mut out = vec![0f32; ROWS * WIDTH];
        unsafe {
            rt::hip_rt_memcpy(
                out.as_mut_ptr().cast(),
                self.h.cast(),
                (out.len() * 4) as u64,
                CopyKind::DeviceToHost as u32,
            )
        };
        out
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        unsafe {
            rt::hip_rt_free(self.h.cast());
            rt::hip_rt_free(self.v.cast());
        }
    }
}

/// The first value whose bits differ, as "row r, index j: got x, want y", or `None`.
fn first_difference(got: &[f32], want: &[f32]) -> Option<String> {
    let at = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits())?;
    Some(format!(
        "row {}, index {}: got {:e}, want {:e}",
        at / WIDTH,
        at % WIDTH,
        got[at],
        want[at]
    ))
}

#[test]
fn the_device_op_gives_the_bits_of_the_host_op_for_each_op_and_each_row() {
    let (binding, _) = binding::linked().expect("a hip build links the engine");
    let start: Vec<f32> = (0..ROWS).flat_map(|row| lcg(2 + row as u32, WIDTH, 4.0)).collect();
    let v = lcg(102, WIDTH, 2.0);
    let sum_squares = control::sum_squares(&v);
    let buffers = Buffers::new(&v);
    let cases = [
        (ControlOp::Add, 0.75),
        (ControlOp::Add, -2.5),
        (ControlOp::Ablate, 1.0),
        (ControlOp::Ablate, 0.5),
        (ControlOp::Clamp, 3.0),
        (ControlOp::Clamp, -1.0),
        (ControlOp::Clamp, 0.0),
    ];
    for (op, strength) in cases {
        buffers.put(&start);
        unsafe {
            (binding.control_apply)(
                buffers.h,
                ROWS as u32,
                buffers.v,
                sum_squares,
                control::op_code(op),
                strength,
            )
        };
        let got = buffers.get();
        let mut want = start.clone();
        for row in want.chunks_mut(WIDTH) {
            assert!(control::apply_row(op, strength, row, &v, sum_squares));
        }
        assert_ne!(bits(&want), bits(&start), "{op:?} {strength} changes the rows");
        if let Some(difference) = first_difference(&got, &want) {
            panic!("{op:?} of strength {strength}: the device op differs from the host op at {difference}");
        }
    }
}

#[test]
fn a_call_changes_only_the_rows_that_it_names() {
    let (binding, _) = binding::linked().expect("a hip build links the engine");
    let start: Vec<f32> = (0..ROWS).flat_map(|row| lcg(500 + row as u32, WIDTH, 4.0)).collect();
    let v = lcg(103, WIDTH, 2.0);
    let sum_squares = control::sum_squares(&v);
    let buffers = Buffers::new(&v);
    buffers.put(&start);
    // Rows 7, 8 and 9 of the buffer, as the rows of a segment in the middle of a pass.
    let (from, rows) = (7, 3);
    let op = ControlOp::Ablate;
    unsafe {
        (binding.control_apply)(
            buffers.h.wrapping_add(from * WIDTH),
            rows as u32,
            buffers.v,
            sum_squares,
            control::op_code(op),
            1.0,
        )
    };
    let got = buffers.get();
    let mut want = start.clone();
    for row in want[from * WIDTH..(from + rows) * WIDTH].chunks_mut(WIDTH) {
        assert!(control::apply_row(op, 1.0, row, &v, sum_squares));
    }
    if let Some(difference) = first_difference(&got, &want) {
        panic!("a call on rows {from} to {}: {difference}", from + rows - 1);
    }
}
