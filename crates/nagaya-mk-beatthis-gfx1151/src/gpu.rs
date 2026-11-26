//! The host loop of the plan on one gfx1151 card.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::{c_int, c_long, c_void};
use std::time::Instant;

use nagaya_family_beatthis::plan::{Axis, Mark, Plan, Task, WeightId};
use nagaya_mk_hip::hip::{DeviceBuffer, HipError, Stream, check};

unsafe extern "C" {
    fn bt_hip_affine(
        x: *const f32,
        y: *mut f32,
        n: c_long,
        cols: c_int,
        scale: *const f32,
        shift: *const f32,
        s: *mut c_void,
    ) -> c_int;
    fn bt_hip_conv(
        x: *const f32,
        y: *mut f32,
        n: c_long,
        frames: c_int,
        f_in: c_int,
        cin: c_int,
        cout: c_int,
        kf: c_int,
        kt: c_int,
        stride: c_int,
        kernel: *const f32,
        s: *mut c_void,
    ) -> c_int;
    fn bt_hip_gelu(x: *mut f32, n: c_long, s: *mut c_void) -> c_int;
    fn bt_hip_rms_norm(
        x: *const f32,
        y: *mut f32,
        rows: c_long,
        dim: c_int,
        gamma: *const f32,
        s: *mut c_void,
    ) -> c_int;
    fn bt_hip_linear(
        x: *const f32,
        y: *mut f32,
        rows: c_long,
        din: c_int,
        dout: c_int,
        weight: *const f32,
        bias: *const f32,
        s: *mut c_void,
    ) -> c_int;
    fn bt_hip_add_into(acc: *mut f32, x: *const f32, n: c_long, s: *mut c_void) -> c_int;
    fn bt_hip_attention(
        qkv: *const f32,
        qk: *mut f32,
        gates: *const f32,
        out: *mut f32,
        a: c_int,
        b: c_int,
        axis: c_int,
        heads: c_int,
        hd: c_int,
        freqs: *const f32,
        s: *mut c_void,
    ) -> c_int;
}

/// The scratch of one chunk length: a device buffer for each buffer of its plan, and the rotated q and k.
struct Scratch {
    bufs: Vec<DeviceBuffer>,
    qk: DeviceBuffer,
}

/// The prepared weights of a checkpoint in device memory, the stream of the card, and the scratch of each chunk
/// length that a run met. A run allocates no device memory after the first run of its length.
pub struct Gpu {
    weights: Vec<DeviceBuffer>,
    stream: Stream,
    scratch: RefCell<HashMap<usize, Scratch>>,
}

fn err(e: HipError) -> String {
    format!("{}: HIP error {} ({})", e.call, e.code, e.text)
}

/// The rotated q and k of the largest attention of a plan, in values.
fn rotated(plan: &Plan) -> usize {
    plan.tasks
        .iter()
        .map(|t| match *t {
            Task::Attention {
                a, b, heads, head_dim, ..
            } => a * b * 2 * heads * head_dim,
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

impl Gpu {
    /// Upload the prepared weights of a plan (`nagaya_family_beatthis::weights::prepare`).
    pub fn new(weights: &[Vec<f32>]) -> Result<Self, String> {
        let weights = weights
            .iter()
            .map(|w| DeviceBuffer::from_slice(w))
            .collect::<Result<Vec<_>, _>>()
            .map_err(err)?;
        Ok(Self {
            weights,
            stream: Stream::new().map_err(err)?,
            scratch: RefCell::default(),
        })
    }

    fn w(&self, id: WeightId) -> *const f32 {
        self.weights[id.0].ptr() as *const f32
    }

    /// Run one chunk of the plan and give its output buffer. `tap` sees each mark's rows after its task; with `taps`
    /// the loop waits for the card at each mark, so a timed run passes `taps = false`.
    ///
    /// A plan that fails its check, or weights and an input of other sizes, panic: a launch on them would read past a
    /// buffer, which on a card is a fault, not a panic.
    pub fn run(
        &self,
        plan: &Plan,
        input: &[f32],
        taps: bool,
        tap: impl FnMut(&Mark, &[f32]),
    ) -> Result<Vec<f32>, String> {
        self.run_timed(plan, input, taps, tap, None)
    }

    /// `run`, and with `times`, the wall milliseconds of each task, each with a wait for the card after the task. The
    /// waits slow the run, so the times rank the tasks; they do not add up to a run without them.
    pub fn run_timed(
        &self,
        plan: &Plan,
        input: &[f32],
        taps: bool,
        mut tap: impl FnMut(&Mark, &[f32]),
        mut times: Option<&mut Vec<f64>>,
    ) -> Result<Vec<f32>, String> {
        if let Err(e) = plan.check() {
            panic!("the plan of {} frames fails its check: {e}", plan.frames);
        }
        assert_eq!(
            self.weights.len(),
            plan.weights.len(),
            "the weights are not those of the plan"
        );
        for (i, (w, spec)) in self.weights.iter().zip(&plan.weights).enumerate() {
            assert_eq!(
                w.bytes(),
                4 * spec.prep.len(),
                "weight {i} has another size than the plan reads"
            );
        }
        assert_eq!(
            input.len(),
            plan.buffers[plan.input.0],
            "the input is not {} rows of log-mel values",
            plan.frames
        );
        let mut cache = self.scratch.borrow_mut();
        let scratch = match cache.entry(plan.frames) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let bufs = plan
                    .buffers
                    .iter()
                    .map(|&n| DeviceBuffer::new(4 * n.max(1)))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(err)?;
                let qk = DeviceBuffer::new(4 * rotated(plan).max(1)).map_err(err)?;
                e.insert(Scratch { bufs, qk })
            }
        };
        let Scratch { bufs, qk } = &*scratch;
        // Two plans of one length and config are one plan (plan::build), so the cached sizes hold. Check them anyway.
        assert!(
            bufs.len() == plan.buffers.len()
                && bufs.iter().zip(&plan.buffers).all(|(b, &n)| b.bytes() == 4 * n.max(1))
                && qk.bytes() == 4 * rotated(plan).max(1),
            "the scratch of {} frames is of another plan",
            plan.frames
        );
        bufs[plan.input.0].write(input).map_err(err)?;
        let p = |b: nagaya_family_beatthis::plan::Buf| bufs[b.0].ptr() as *mut f32;
        let s = self.stream.raw();
        for (i, task) in plan.tasks.iter().enumerate() {
            let started = Instant::now();
            // SAFETY: Plan::check proved that each task reads and writes inside its buffers, and each buffer and weight
            // has the size of the plan (asserted above). The buffers live in the cache of this Gpu.
            let code = unsafe { self.launch(task, &p, qk, s) };
            check("a Beat This! launch", code).map_err(|e| format!("task {i} ({task:?}): {}", err(e)))?;
            if let Some(times) = times.as_deref_mut() {
                self.stream.synchronize().map_err(err)?;
                times.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            if taps {
                let marks: Vec<&Mark> = plan.marks.iter().filter(|m| m.after == i).collect();
                if !marks.is_empty() {
                    self.stream.synchronize().map_err(err)?;
                    for m in marks {
                        let mut rows = vec![0.0f32; plan.frames * m.freqs * m.channels];
                        bufs[m.buf.0].read(&mut rows).map_err(err)?;
                        tap(m, &rows);
                    }
                }
            }
        }
        self.stream.synchronize().map_err(err)?;
        let mut out = vec![0.0f32; plan.buffers[plan.output.0]];
        bufs[plan.output.0].read(&mut out).map_err(err)?;
        Ok(out)
    }

    /// Queue the kernel of one task.
    ///
    /// # Safety
    /// Each buffer of the task must hold what it reads and writes (`Plan::check`), and `qk` the rotated q and k of
    /// its attention.
    unsafe fn launch(
        &self,
        task: &Task,
        p: &impl Fn(nagaya_family_beatthis::plan::Buf) -> *mut f32,
        qk: &DeviceBuffer,
        s: *mut c_void,
    ) -> c_int {
        // SAFETY: the caller's contract.
        unsafe {
            match *task {
                Task::Affine {
                    x,
                    y,
                    rows,
                    cols,
                    scale,
                    shift,
                } => bt_hip_affine(
                    p(x),
                    p(y),
                    (rows * cols) as c_long,
                    cols as c_int,
                    self.w(scale),
                    self.w(shift),
                    s,
                ),
                Task::Conv {
                    x,
                    y,
                    frames,
                    f_in,
                    cin,
                    cout,
                    kf,
                    kt,
                    stride,
                    kernel,
                } => bt_hip_conv(
                    p(x),
                    p(y),
                    (frames * ((f_in - kf) / stride + 1) * cout) as c_long,
                    frames as c_int,
                    f_in as c_int,
                    cin as c_int,
                    cout as c_int,
                    kf as c_int,
                    kt as c_int,
                    stride as c_int,
                    self.w(kernel),
                    s,
                ),
                Task::Gelu { x, n } => bt_hip_gelu(p(x), n as c_long, s),
                Task::RmsNorm { x, y, rows, dim, gamma } => {
                    bt_hip_rms_norm(p(x), p(y), rows as c_long, dim as c_int, self.w(gamma), s)
                }
                Task::Linear {
                    x,
                    y,
                    rows,
                    din,
                    dout,
                    weight,
                    bias,
                } => bt_hip_linear(
                    p(x),
                    p(y),
                    rows as c_long,
                    din as c_int,
                    dout as c_int,
                    self.w(weight),
                    bias.map_or(std::ptr::null(), |b| self.w(b)),
                    s,
                ),
                Task::Attention {
                    qkv,
                    gates,
                    out,
                    a,
                    b,
                    along,
                    heads,
                    head_dim,
                    freqs,
                } => bt_hip_attention(
                    p(qkv),
                    qk.ptr() as *mut f32,
                    p(gates),
                    p(out),
                    a as c_int,
                    b as c_int,
                    match along {
                        Axis::A => 0,
                        Axis::B => 1,
                    },
                    heads as c_int,
                    head_dim as c_int,
                    self.w(freqs),
                    s,
                ),
                Task::AddInto { acc, x, n } => bt_hip_add_into(p(acc), p(x), n as c_long, s),
            }
        }
    }
}
