//! The pass in Rust: the checks of the segments, the scratch pool, the upload, the launch rules and the advance of
//! each context (milestone 15 step 6).
//!
//! mk's `engine.hip` `segments_problem`, `run_pass`, `failed_pass`, `run_stages_now` and the `ScratchPool` held these in
//! C++ (mk 555a2a3). The C++ engine keeps only the launches under the control block and the watchdog
//! (`Engine::launch_fused`, `Engine::launch_each`). Every call here runs on the worker thread.

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::time::Instant;

use crate::binding::{Binding, PassStop, Qwen38Launch, Qwen38Segment};
use crate::context::{Copy, Qwen38Context};
use crate::pass::{self, Builder, Knobs, PassBuffers, Scratch, Seg, Weights};
use crate::ple;
use crate::stage::Stage;

/// `rt.h` `kSpinTimeout`: a bounded spin ran out of time.
const SPIN_TIMEOUT: u32 = 2;
/// The most workgroups of a fused pass: 4 for each WGP. The grid sweep of 2026-09-27 ran 80 cleanly with no scratch.
const MAX_FUSED_GRID: i32 = 80;

/// The named device buffers of the pass: one for each name, made and zeroed at its first ask, and kept for the
/// engine's life (`engine.hip` `ScratchPool`).
#[derive(Default)]
pub(crate) struct Pool {
    bufs: HashMap<String, (*mut f32, usize)>,
}

/// The pool with the binding that allocates its buffers.
struct PoolScratch<'a> {
    binding: &'a Binding,
    pool: &'a mut Pool,
}

impl Scratch for PoolScratch<'_> {
    /// A later ask of a name for more floats than its buffer holds stops the program: the stages built on it would
    /// run past its end.
    fn get(&mut self, name: &str, floats: usize) -> *mut f32 {
        if let Some(&(p, n)) = self.pool.bufs.get(name) {
            assert!(
                floats <= n,
                "scratch {name}: asked for {floats} floats, its buffer holds {n}"
            );
            return p;
        }
        let bytes = 4 * floats as u64;
        let p = unsafe { (self.binding.rt_malloc)(bytes) } as *mut f32;
        assert!(!p.is_null(), "mk_rt_malloc({bytes}) gave no buffer for scratch {name}");
        unsafe { (self.binding.rt_memset)(p.cast(), 0, bytes) };
        self.pool.bufs.insert(name.to_owned(), (p, floats));
        p
    }
}

/// The host side of the engine: the scratch pool and the fixed buffers of a pass.
pub(crate) struct Host {
    pool: Pool,
    /// The token ids of a pass, `MAX_PASS_TOKENS` on the device.
    tokens: *mut i32,
    /// The last row of each segment of a pass of several segments (the head's `x_row`), `MAX_SEGMENTS` on the device.
    head_rows: *mut u32,
    /// The rope table of the attention prep, made at the first build.
    inv_freq: *mut f32,
    /// The stage list of a pass, in pinned host memory that the device reads, and its room in stages.
    stages: *mut Stage,
    stage_cap: usize,
}

impl Host {
    /// The fixed buffers of the pass (`engine.hip` `wire`).
    ///
    /// # Safety
    ///
    /// `binding` is the linked engine's, on the worker thread.
    pub(crate) unsafe fn new(binding: &Binding) -> Host {
        let tokens = unsafe { (binding.rt_malloc)(4 * u64::from(pass::MAX_PASS_TOKENS)) } as *mut i32;
        let head_rows = unsafe { (binding.rt_malloc)(4 * u64::from(pass::MAX_SEGMENTS)) } as *mut u32;
        assert!(
            !tokens.is_null() && !head_rows.is_null(),
            "mk_rt_malloc gave no pass buffer"
        );
        Host {
            pool: Pool::default(),
            tokens,
            head_rows,
            inv_freq: std::ptr::null_mut(),
            stages: std::ptr::null_mut(),
            stage_cap: 0,
        }
    }

    /// The rope table on the device, made at its first use (`blocks.hip` `inv_freq_table`).
    fn inv_freq(&mut self, binding: &Binding) -> *const f32 {
        if self.inv_freq.is_null() {
            let table = pass::inv_freq();
            let bytes = std::mem::size_of_val(&table) as u64;
            let p = unsafe { (binding.rt_malloc)(bytes) } as *mut f32;
            assert!(!p.is_null(), "mk_rt_malloc gave no rope table");
            unsafe { (binding.rt_memcpy)(p.cast(), table.as_ptr().cast(), bytes, Copy::HostToDevice as u32) };
            self.inv_freq = p;
        }
        self.inv_freq
    }

    /// Copy `stages` into the pinned stage list, which grows to hold them, and give its address.
    fn upload(&mut self, binding: &Binding, stages: &[Stage]) -> *const Stage {
        if stages.len() > self.stage_cap {
            if !self.stages.is_null() {
                unsafe { (binding.rt_host_free)(self.stages.cast()) };
            }
            self.stage_cap = stages.len();
            let bytes = (std::mem::size_of::<Stage>() * self.stage_cap) as u64;
            self.stages = unsafe { (binding.rt_host_malloc)(bytes) } as *mut Stage;
            assert!(!self.stages.is_null(), "mk_rt_host_malloc gave no stage list");
        }
        let bytes = std::mem::size_of_val(stages) as u64;
        unsafe {
            (binding.rt_memcpy)(
                self.stages.cast(),
                stages.as_ptr().cast(),
                bytes,
                Copy::HostToDevice as u32,
            )
        };
        self.stages
    }

    /// The scratch buffer `name` of `floats`.
    fn scratch(&mut self, binding: &Binding, name: &str, floats: usize) -> *mut f32 {
        PoolScratch {
            binding,
            pool: &mut self.pool,
        }
        .get(name, floats)
    }
}

/// The first problem of `segments` as one pass, or `None` (`engine.hip` `segments_problem`): 1 to 16 segments, each of
/// one or more tokens, 256 in all, each fitting its context's capacity, and no two on one context.
pub(crate) fn segments_problem(segments: &[Qwen38Segment]) -> Option<String> {
    let n = segments.len();
    if n == 0 || n > pass::MAX_SEGMENTS as usize {
        return Some(format!(
            "{n} segments: a pass takes 1..={} segments",
            pass::MAX_SEGMENTS
        ));
    }
    let mut total = 0u64;
    for (i, s) in segments.iter().enumerate() {
        if s.context.is_null() || s.tokens.is_null() {
            return Some(format!("segment {i}: no context or no tokens pointer"));
        }
        if s.t == 0 {
            return Some(format!("segment {i} has no tokens"));
        }
        total += u64::from(s.t);
    }
    if total > u64::from(pass::MAX_PASS_TOKENS) {
        return Some(format!(
            "segments of {total} tokens in all: a pass takes 1..{}",
            pass::MAX_PASS_TOKENS
        ));
    }
    for (i, s) in segments.iter().enumerate() {
        // SAFETY: the pointer is not null, and the worker gives live contexts of this engine.
        let c = unsafe { &*(s.context as *const Qwen38Context) };
        if u64::from(c.n) + u64::from(s.t) > u64::from(c.max) {
            return Some(format!(
                "segment {i}: context full: {} + {} tokens past its capacity {}",
                c.n, s.t, c.max
            ));
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = unsafe {
                (
                    &*(segments[i].context as *const Qwen38Context),
                    &*(segments[j].context as *const Qwen38Context),
                )
            };
            if a.kv == b.kv || a.idx_keys == b.idx_keys {
                return Some(format!("segments {i} and {j} share a K/V buffer (one context twice)"));
            }
        }
    }
    None
}

/// The knobs of the launch of a pass (`engine.hip` `run_pass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchKnobs {
    /// `QWEN38_MUTATE_SKIP_BARRIER`: the stage whose grid barrier the kernel skips (a test knob), or none.
    pub skip: Option<u32>,
    /// `QWEN38_MUTATE_STALL_BARRIER`: workgroup 0 leaves at the first grid barrier at or after this stage without arriving,
    /// so the pass stops on a real spin-timeout (a test knob), or none.
    pub stall: Option<u32>,
    /// `QWEN38_MEGA_GRID`: the workgroups of the launch for an experiment, or the occupancy's.
    pub grid: Option<i32>,
}

impl LaunchKnobs {
    /// The knobs from `var`. A value that is not a number is an error that refuses the pass; mk's `atoi` read it as 0.
    pub(crate) fn from(var: impl Fn(&str) -> Option<String>) -> Result<LaunchKnobs, String> {
        let number = |name: &str| -> Result<Option<u32>, String> {
            var(name)
                .map(|value| {
                    value
                        .parse::<u32>()
                        .map_err(|_| format!("{name}={value}: want a stage number"))
                })
                .transpose()
        };
        let knobs = LaunchKnobs {
            skip: number("QWEN38_MUTATE_SKIP_BARRIER")?,
            stall: number("QWEN38_MUTATE_STALL_BARRIER")?,
            grid: match var("QWEN38_MEGA_GRID") {
                None => None,
                Some(value) => match value.parse::<i32>() {
                    Ok(grid) if grid > 0 => Some(grid),
                    _ => return Err(format!("QWEN38_MEGA_GRID={value}: want a workgroup count above 0")),
                },
            },
        };
        if knobs.skip.is_some() && knobs.stall.is_some() {
            return Err("QWEN38_MUTATE_STALL_BARRIER: not with QWEN38_MUTATE_SKIP_BARRIER".to_owned());
        }
        Ok(knobs)
    }

    /// The skip word of `mega.hip` for a pass of `n` stages: the stage to skip, the stall stage with the top bit set,
    /// or `u32::MAX`. A stall stage past the pass is an error.
    pub(crate) fn skip_word(&self, n: u32) -> Result<u32, String> {
        match (self.skip, self.stall) {
            (Some(stage), _) => Ok(stage),
            (None, Some(stage)) if stage < n => Ok(0x8000_0000 | stage),
            (None, Some(stage)) => Err(format!(
                "QWEN38_MUTATE_STALL_BARRIER={stage}: want a stage in 0..{}",
                n.saturating_sub(1)
            )),
            (None, None) => Ok(u32::MAX),
        }
    }
}

/// The pass of `segments` and its parts that the worker gives to `run_pass`.
pub(crate) struct PassIn<'a> {
    pub segments: &'a [Qwen38Segment],
    pub ple: &'a [f32],
    pub knobs: Knobs,
    pub launch: LaunchKnobs,
}

/// The name of a status of the control block.
fn status_name(binding: &Binding, status: u32) -> String {
    let name = unsafe { (binding.status_name)(status) };
    if name.is_null() {
        return format!("status {status}");
    }
    unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
}

/// Stop the program: a launch failed in a way that no caller can recover from (`HIP_DIE`).
fn die(why: &str) -> ! {
    eprintln!("engine: {why}");
    std::process::abort();
}

/// One fused pass (`engine.hip` `run_pass`). The worker checked the segments, the poison and the knobs first. `logits`
/// takes one row of `VOCAB` values for each segment.
///
/// A spin-timeout that the watchdog did not ask for stops the pass: each of its contexts is poisoned and not advanced,
/// and the result is `PassStop::Stopped`. Any other stop of the kernel stops the program.
///
/// # Safety
///
/// `binding` and `engine` are the linked engine's, on the worker thread, and each segment names a live context and `t`
/// readable tokens.
pub(crate) unsafe fn run_pass(
    binding: &Binding,
    engine: *mut c_void,
    host: &mut Host,
    weights: &Weights,
    input: PassIn<'_>,
    logits: Option<&mut [f32]>,
) -> Result<f64, PassStop> {
    let segments = input.segments;
    let ns = segments.len();
    let mut tokens: Vec<i32> = Vec::new();
    let mut head_rows = Vec::with_capacity(ns);
    for s in segments {
        tokens.extend_from_slice(unsafe { std::slice::from_raw_parts(s.tokens, s.t as usize) });
        head_rows.push(tokens.len() as u32 - 1);
    }
    let t = tokens.len() as u32;
    assert_eq!(
        input.ple.len(),
        t as usize * ple::EMBD,
        "the PLE rows must hold EMBD values for each token"
    );
    let start = Instant::now();
    unsafe {
        (binding.rt_memcpy)(
            host.tokens.cast(),
            tokens.as_ptr().cast(),
            4 * u64::from(t),
            Copy::HostToDevice as u32,
        );
        if ns > 1 {
            (binding.rt_memcpy)(
                host.head_rows.cast(),
                head_rows.as_ptr().cast(),
                4 * ns as u64,
                Copy::HostToDevice as u32,
            );
        }
    }
    // Each segment's n-gram rows, which the worker gathered from its own context's history and tokens.
    let ple_emb = host.scratch(binding, "ple_emb", pass::MAX_PASS_TOKENS as usize * pass::EMBD as usize);
    unsafe {
        (binding.rt_memcpy)(
            ple_emb.cast(),
            input.ple.as_ptr().cast(),
            std::mem::size_of_val(input.ple) as u64,
            Copy::HostToDevice as u32,
        )
    };
    let segs: Vec<Seg> = segments
        .iter()
        .map(|s| Seg {
            c: s.context as *mut Qwen38Context,
            t: s.t,
        })
        .collect();
    let want_logits = logits.is_some();
    let buffers = PassBuffers {
        tokens: host.tokens,
        head_rows: host.head_rows,
        inv_freq: host.inv_freq(binding),
    };
    let built = Builder {
        w: weights,
        scratch: &mut PoolScratch {
            binding,
            pool: &mut host.pool,
        },
        buffers,
        knobs: input.knobs,
    }
    .build(&segs, t, want_logits);
    let n = built.stages.len() as u32;
    let skip = input.launch.skip_word(n).map_err(PassStop::Refused)?;
    let stages = host.upload(binding, &built.stages);
    let grid = match input.launch.grid {
        Some(grid) => grid,
        None => unsafe { (binding.max_coop_grid)() }.min(MAX_FUSED_GRID),
    };
    let mut result = Qwen38Launch::default();
    unsafe { (binding.launch_fused)(engine, stages, n, grid, skip, &mut result) };
    if result.status != 0 {
        let why = format!(
            "megakernel stopped: {} at stage {} (wg {})",
            status_name(binding, result.status),
            result.site,
            result.wg
        );
        // Recoverable only for a spin-timeout that the watchdog did not ask for (its deadline or the memory floor stay
        // fatal). The control block's barrier count may be left mid-count; the next begin_pass does the full reset.
        if result.status == SPIN_TIMEOUT && result.watchdog_stopped == 0 {
            for s in &segs {
                // SAFETY: a live context of this engine.
                unsafe { (*s.c).poisoned = true };
            }
            let error = format!("{why}; the pass's {ns} context(s) are poisoned");
            eprintln!("engine: {error}");
            return Err(PassStop::Stopped(error));
        }
        die(&why);
    }
    if let Some(logits) = logits {
        assert_eq!(
            logits.len(),
            ns * pass::VOCAB as usize,
            "the logits take one row for each segment"
        );
        let src = host.scratch(binding, "logits", pass::MAX_SEGMENTS as usize * pass::VOCAB as usize);
        unsafe {
            (binding.rt_memcpy)(
                logits.as_mut_ptr().cast(),
                src.cast(),
                std::mem::size_of_val(logits) as u64,
                Copy::DeviceToHost as u32,
            )
        };
    }
    for (s, segment) in segs.iter().zip(segments) {
        // SAFETY: a live context of this engine, and `t` readable tokens.
        let c = unsafe { &mut *s.c };
        let tk = unsafe { std::slice::from_raw_parts(segment.tokens, segment.t as usize) };
        advance(c, tk);
    }
    Ok(start.elapsed().as_secs_f64() * 1e3)
}

/// Advance a context by the tokens of a clean pass: the count, the pooled-key watermark (the pass's pool stages wrote
/// the blocks that it completed), and the PLE history.
fn advance(c: &mut Qwen38Context, tokens: &[i32]) {
    let t = tokens.len();
    c.n += t as u32;
    c.pooled_n = c.n / crate::stage::IDX_RATIO;
    if t >= 2 {
        c.ple_hist = [tokens[t - 2], tokens[t - 1]];
    } else {
        c.ple_hist = [c.ple_hist[1], tokens[0]];
    }
}

/// Pool every complete block of the index keys of a restored context, one launch for each attention layer (the
/// restore of `engine.hip` `restore_n`). The context's `pooled_n` is the caller's to set.
///
/// # Safety
///
/// As for `run_pass`; `c` is a live context of this engine.
pub(crate) unsafe fn pool_context(
    binding: &Binding,
    engine: *mut c_void,
    host: &mut Host,
    weights: &Weights,
    c: &Qwen38Context,
) {
    if c.n / crate::stage::IDX_RATIO == 0 {
        return;
    }
    let buffers = PassBuffers {
        tokens: host.tokens,
        head_rows: host.head_rows,
        inv_freq: host.inv_freq(binding),
    };
    let builder = Builder {
        w: weights,
        scratch: &mut PoolScratch {
            binding,
            pool: &mut host.pool,
        },
        buffers,
        knobs: Knobs {
            fast_qmm: true,
            fuse: true,
            fuse_hc: true,
            fuse_hcmix: true,
            elide: true,
            tail: true,
            debug_layers: 0,
        },
    };
    let pool = builder.pool_stages(c, 0);
    let stages = host.upload(binding, &pool);
    let mut result = Qwen38Launch::default();
    unsafe { (binding.launch_each)(engine, stages, pool.len() as u32, &mut result) };
    if result.status != 0 {
        die(&format!(
            "stage launch outside a pass stopped: {} at stage {} of {} (wg {})",
            status_name(binding, result.status),
            result.site,
            pool.len(),
            result.wg
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(n: u32, max: u32, kv: usize) -> Qwen38Context {
        Qwen38Context {
            n,
            max,
            gdn_state: std::ptr::null_mut(),
            gdn_conv: std::ptr::null_mut(),
            ple_conv: std::ptr::null_mut(),
            kv: kv as *mut f32,
            idx_keys: (kv + 1) as *mut f32,
            idx_pooled: std::ptr::null_mut(),
            pooled_n: 0,
            ple_hist: [0; 2],
            poisoned: false,
        }
    }

    fn segment(c: &mut Qwen38Context, tokens: &[i32]) -> Qwen38Segment {
        Qwen38Segment {
            context: c as *mut Qwen38Context as *mut c_void,
            tokens: tokens.as_ptr(),
            t: tokens.len() as u32,
        }
    }

    #[test]
    fn segments_that_one_pass_cannot_hold_are_refused_with_the_first_problem() {
        let tokens = [1i32; 300];
        let (mut a, mut b) = (context(0, 256, 16), context(250, 256, 32));
        assert_eq!(segments_problem(&[segment(&mut a, &tokens[..3])]), None);
        let problem = segments_problem(&[]).unwrap();
        assert!(problem.contains("1..=16 segments"), "{problem}");
        let problem = segments_problem(&[segment(&mut a, &tokens[..0])]).unwrap();
        assert_eq!(problem, "segment 0 has no tokens");
        let problem = segments_problem(&[segment(&mut a, &tokens[..257])]).unwrap();
        assert!(problem.contains("257 tokens in all"), "{problem}");
        let problem = segments_problem(&[segment(&mut a, &tokens[..1]), segment(&mut b, &tokens[..7])]).unwrap();
        assert_eq!(problem, "segment 1: context full: 250 + 7 tokens past its capacity 256");
        let one = segment(&mut a, &tokens[..1]);
        let problem = segments_problem(&[one, one]).unwrap();
        assert!(problem.contains("share a K/V buffer"), "{problem}");
        let mut null = one;
        null.context = std::ptr::null_mut();
        assert_eq!(
            segments_problem(&[null]).unwrap(),
            "segment 0: no context or no tokens pointer"
        );
    }

    #[test]
    fn a_clean_pass_advances_the_count_the_watermark_and_the_history() {
        let mut c = context(6, 256, 16);
        c.ple_hist = [3, 4];
        advance(&mut c, &[9]);
        assert_eq!((c.n, c.pooled_n, c.ple_hist), (7, 1, [4, 9]));
        advance(&mut c, &[10, 11, 12]);
        assert_eq!((c.n, c.pooled_n, c.ple_hist), (10, 2, [11, 12]));
    }

    #[test]
    fn the_launch_knobs_take_numbers_and_refuse_the_rest() {
        let none = LaunchKnobs::from(|_| None).unwrap();
        assert_eq!(none.skip_word(100), Ok(u32::MAX));
        let stall = LaunchKnobs::from(|name| (name == "QWEN38_MUTATE_STALL_BARRIER").then(|| "5".to_owned())).unwrap();
        assert_eq!(stall.skip_word(100), Ok(0x8000_0005));
        assert!(stall.skip_word(5).unwrap_err().contains("want a stage in 0..4"));
        let both = LaunchKnobs::from(|name| name.starts_with("QWEN38_MUTATE").then(|| "1".to_owned()));
        assert!(both.unwrap_err().contains("not with QWEN38_MUTATE_SKIP_BARRIER"));
        let grid = LaunchKnobs::from(|name| (name == "QWEN38_MEGA_GRID").then(|| "abc".to_owned()));
        assert!(grid.unwrap_err().starts_with("QWEN38_MEGA_GRID=abc"));
        let grid = LaunchKnobs::from(|name| (name == "QWEN38_MEGA_GRID").then(|| "40".to_owned())).unwrap();
        assert_eq!(grid.grid, Some(40));
    }
}
