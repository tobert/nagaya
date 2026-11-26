//! The CPU interpreter of the Beat This! plan: the first engine of the CPU group (spec 3.2c).
//!
//! It runs a plan of `nagaya_family_beatthis::plan` task by task with the kernels of `kernels`. A GPU port of the
//! model runs the same plan with its own kernels, so this crate is also the reference of each port.

pub mod engine;
pub mod kernels;

use nagaya_family_beatthis::plan::{Mark, Plan, Task};

/// Runs one chunk. `weights` are the prepared weights of the plan (`nagaya_family_beatthis::weights::prepare`), and
/// `input` its log-mel rows. `tap` sees each mark's rows when its task ends. Gives the plan's output buffer.
///
/// A plan that fails its check, weights of another count, or an input of another size panics: they are defects of
/// the caller, and a run on them would read past a buffer.
pub fn run(
    plan: &Plan,
    weights: &[Vec<f32>],
    input: &[f32],
    threads: usize,
    mut tap: impl FnMut(&Mark, &[f32]),
) -> Vec<f32> {
    if let Err(e) = plan.check() {
        panic!("the plan of {} frames fails its check: {e}", plan.frames);
    }
    assert_eq!(
        weights.len(),
        plan.weights.len(),
        "the weights are not those of the plan"
    );
    for (i, (w, spec)) in weights.iter().zip(&plan.weights).enumerate() {
        assert_eq!(
            w.len(),
            spec.prep.len(),
            "weight {i} has {} values, and the plan reads {}",
            w.len(),
            spec.prep.len()
        );
    }
    let mut bufs: Vec<Vec<f32>> = plan.buffers.iter().map(|&n| vec![0.0f32; n]).collect();
    assert_eq!(
        input.len(),
        bufs[plan.input.0].len(),
        "the input is not {} rows of log-mel values",
        plan.frames
    );
    bufs[plan.input.0].copy_from_slice(input);
    for (i, task) in plan.tasks.iter().enumerate() {
        step(task, &mut bufs, weights, threads);
        for m in plan.marks.iter().filter(|m| m.after == i) {
            tap(m, &bufs[m.buf.0][..plan.frames * m.freqs * m.channels]);
        }
    }
    std::mem::take(&mut bufs[plan.output.0])
}

/// Runs one task of a plan on `bufs`, the buffers of the plan. `run` calls it for each task in order.
pub fn step(task: &Task, bufs: &mut [Vec<f32>], weights: &[Vec<f32>], threads: usize) {
    let w = |id: nagaya_family_beatthis::plan::WeightId| weights[id.0].as_slice();
    let (out, _) = task.writes();
    let mut y = std::mem::take(&mut bufs[out.0]);
    match *task {
        Task::Affine {
            x,
            rows,
            cols,
            scale,
            shift,
            ..
        } => kernels::affine(
            &bufs[x.0][..rows * cols],
            &mut y[..rows * cols],
            cols,
            w(scale),
            w(shift),
            threads,
        ),
        Task::Conv {
            x,
            frames,
            f_in,
            cin,
            cout,
            kf,
            kt,
            stride,
            kernel,
            ..
        } => {
            let f_out = (f_in - kf) / stride + 1;
            kernels::conv(
                &bufs[x.0][..frames * f_in * cin],
                &mut y[..frames * f_out * cout],
                frames,
                f_in,
                cin,
                cout,
                kf,
                kt,
                stride,
                w(kernel),
                threads,
            )
        }
        Task::Gelu { n, .. } => kernels::gelu(&mut y[..n], threads),
        Task::RmsNorm {
            x, rows, dim, gamma, ..
        } => kernels::rms_norm(&bufs[x.0][..rows * dim], &mut y[..rows * dim], dim, w(gamma), threads),
        Task::Linear {
            x,
            rows,
            din,
            dout,
            weight,
            bias,
            ..
        } => kernels::linear(
            &bufs[x.0][..rows * din],
            &mut y[..rows * dout],
            din,
            dout,
            w(weight),
            bias.map(w),
            threads,
        ),
        Task::Attention {
            qkv,
            gates,
            a,
            b,
            along,
            heads,
            head_dim,
            freqs,
            ..
        } => {
            let rows = a * b;
            kernels::attention(
                &bufs[qkv.0][..rows * 3 * heads * head_dim],
                &bufs[gates.0][..rows * heads],
                &mut y[..rows * heads * head_dim],
                a,
                b,
                along,
                heads,
                head_dim,
                w(freqs),
                threads,
            )
        }
        Task::AddInto { x, n, .. } => kernels::add_into(&mut y[..n], &bufs[x.0][..n], threads),
    }
    bufs[out.0] = y;
}
