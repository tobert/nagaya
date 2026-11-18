//! A fake device and fake weights for the host tests of the pass (milestone 34). Nothing here runs on a GPU.
//!
//! `weights` has every tensor that the pass build names, with the quant types and the shapes of the real model, so the
//! build makes the stage list of a real pass. `binding` has fake runtime calls: a "device" buffer is host memory, a
//! copy is a copy of host bytes, and a launch only records its arguments. A test can give the launch a hook that
//! writes into a buffer, as a kernel would. The record is per thread, so tests that run in parallel do not meet.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};

use crate::binding::{
    self, Binding, CtxFree, CtxMax, CtxNew, EngineOpen, PassStop, Qwen38Launch, Relayout, RoundGdn, RtHostRegister,
    RtHostUnregister, StateBytes, StateRead,
};
use crate::context::Qwen38Context;
use crate::control;
use crate::cuts::Cuts;
use crate::host::{self, Host, LaunchKnobs, PassIn};
use crate::pass::{Knobs, Scratch, Tensor, Weights};
use crate::stage::*;

const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q6_K: u32 = 14;
const IQ4_NL: u32 = 20;
const IQ3_S: u32 = 21;

/// The bytes of one row of `k` values in a quant type.
fn row_bytes(kind: u32, k: u64) -> u64 {
    match kind {
        F32 => k * 4,
        Q8_0 => k / 32 * 34,
        Q6_K => k / 256 * 210,
        IQ4_NL => k / 32 * 18,
        IQ3_S => k / 256 * 110,
        other => panic!("the fake weights have no type {other}"),
    }
}

/// The fake weights of the real shapes. A tensor `d` is a fake address that nothing reads.
pub(crate) fn weights() -> Weights {
    let mut next = 0x2000_0000_0000u64;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut put = |name: String, kind: u32, ne: [u64; 3]| {
        next += 1 << 32;
        tensors.insert(
            name,
            Tensor {
                kind,
                ne,
                row_bytes: row_bytes(kind, ne[0]),
                d: next as *const u8,
            },
        );
    };
    let embd = u64::from(EMBD);
    let hc_dim = 4 * embd;
    for il in 0..crate::pass::LAYERS {
        let b = format!("blk.{il}.");
        for side in ["attn", "ffn"] {
            put(format!("{b}hc_{side}_norm.weight"), F32, [embd, 1, 1]);
            put(format!("{b}hc_{side}_down.weight"), Q8_0, [hc_dim, 320, 1]);
            put(format!("{b}hc_{side}_up.weight"), Q8_0, [320, hc_dim, 1]);
            put(format!("{b}hc_{side}_inject.weight"), F32, [hc_dim, 4, 1]);
        }
        if il % 4 == 3 {
            let (heads, kv_heads, head_dim) = (u64::from(HEADS), u64::from(KV_HEADS), u64::from(HEAD_DIM));
            put(format!("{b}attn_q.weight"), Q8_0, [embd, heads * 2 * head_dim, 1]);
            put(format!("{b}attn_k.weight"), Q8_0, [embd, kv_heads * head_dim, 1]);
            put(format!("{b}attn_v.weight"), Q8_0, [embd, kv_heads * head_dim, 1]);
            put(format!("{b}attn_output.weight"), Q8_0, [heads * head_dim, embd, 1]);
            put(format!("{b}attn_q_norm.weight"), F32, [head_dim, 1, 1]);
            put(format!("{b}attn_k_norm.weight"), F32, [head_dim, 1, 1]);
            let (idx_heads, idx_dim) = (u64::from(IDX_HEADS), u64::from(IDX_DIM));
            put(
                format!("{b}indexer.q_proj.weight"),
                Q8_0,
                [embd, idx_heads * idx_dim, 1],
            );
            put(format!("{b}indexer.k_proj.weight"), Q8_0, [embd, idx_dim, 1]);
            put(format!("{b}indexer.q_norm.weight"), F32, [idx_dim, 1, 1]);
            put(format!("{b}indexer.k_norm.weight"), F32, [idx_dim, 1, 1]);
        } else {
            let (conv, val, v_heads) = (u64::from(GDN_CONV_DIM), u64::from(GDN_VAL_DIM), u64::from(GDN_V_HEADS));
            put(format!("{b}attn_qkv.weight"), Q8_0, [embd, conv, 1]);
            put(format!("{b}attn_gate.weight"), Q8_0, [embd, val, 1]);
            put(format!("{b}ssm_alpha.weight"), Q8_0, [embd, v_heads, 1]);
            put(format!("{b}ssm_beta.weight"), Q8_0, [embd, v_heads, 1]);
            put(format!("{b}ssm_a"), F32, [v_heads, 1, 1]);
            put(format!("{b}ssm_dt.bias"), F32, [v_heads, 1, 1]);
            put(format!("{b}ssm_conv1d.weight"), F32, [4, conv, 1]);
            put(format!("{b}ssm_norm.weight"), F32, [128, 1, 1]);
            put(format!("{b}ssm_out.weight"), Q8_0, [val, embd, 1]);
        }
        let (experts, ff) = (u64::from(EXPERTS), 640u64);
        put(format!("{b}ffn_gate_inp.weight"), F32, [embd, experts, 1]);
        put(format!("{b}ffn_gate_inp_shexp.weight"), F32, [embd, 1, 1]);
        put(format!("{b}ffn_gate_shexp.weight"), Q8_0, [embd, ff, 1]);
        put(format!("{b}ffn_up_shexp.weight"), Q8_0, [embd, ff, 1]);
        put(format!("{b}ffn_down_shexp.weight"), Q8_0, [ff, embd, 1]);
        put(format!("{b}ffn_gate_exps.weight"), IQ3_S, [embd, ff, experts]);
        put(format!("{b}ffn_up_exps.weight"), IQ3_S, [embd, ff, experts]);
        put(format!("{b}ffn_down_exps.weight"), IQ4_NL, [ff, embd, experts]);
        if il == 1 {
            put(format!("{b}ple_key.weight"), Q8_0, [embd, hc_dim, 1]);
            put(format!("{b}ple_value.weight"), Q8_0, [embd, embd, 1]);
            put(format!("{b}ple_norm_key.weight"), F32, [embd, 1, 1]);
            put(format!("{b}ple_norm_query.weight"), F32, [embd, 1, 1]);
            put(format!("{b}ple_norm_conv.weight"), F32, [embd, 1, 1]);
            put(format!("{b}ple_conv1d.weight"), F32, [9, hc_dim, 1]);
        }
    }
    put(
        "token_embd.weight".to_owned(),
        Q8_0,
        [embd, u64::from(crate::pass::VOCAB), 1],
    );
    put("output_hc_norm.weight".to_owned(), F32, [embd, 1, 1]);
    put("output_hc_down.weight".to_owned(), Q8_0, [hc_dim, 320, 1]);
    put("output_hc_up.weight".to_owned(), Q8_0, [320, hc_dim, 1]);
    put(
        "output.weight".to_owned(),
        Q6_K,
        [embd, u64::from(crate::pass::VOCAB), 1],
    );
    Weights::new(&tensors).unwrap_or_else(|error| panic!("the fake weights miss a tensor: {error}"))
}

/// A context with fake buffer addresses that nothing reads. `slot` keeps the buffers of two contexts apart.
pub(crate) fn context(n: u32, max: u32, slot: u64) -> Box<Qwen38Context> {
    let base = 0x4000_0000_0000u64 + (slot << 40);
    let pointer = |part: u64| (base + (part << 36)) as *mut f32;
    Box::new(Qwen38Context {
        n,
        max,
        gdn_state: pointer(1),
        gdn_conv: pointer(2),
        ple_conv: pointer(3),
        kv: pointer(4),
        idx_keys: pointer(5),
        idx_pooled: pointer(6),
        pooled_n: n / 4,
        ple_hist: [0; 2],
        poisoned: false,
    })
}

/// A scratch pool of fake addresses: it gives each name its own range, and no memory stands behind it.
#[derive(Default)]
pub(crate) struct BumpScratch {
    next: u64,
    names: HashMap<String, (u64, usize)>,
}

impl Scratch for BumpScratch {
    fn get(&mut self, name: &str, floats: usize) -> *mut f32 {
        if let Some(&(at, held)) = self.names.get(name) {
            assert!(
                floats <= held,
                "scratch {name}: asked for {floats} floats, it holds {held}"
            );
            return at as *mut f32;
        }
        let at = 0x6000_0000_0000 + self.next;
        self.next += (4 * floats as u64).next_multiple_of(256) + 256;
        self.names.insert(name.to_owned(), (at, floats));
        at as *mut f32
    }
}

// ---- the fake runtime ---------------------------------------------------------------------------------------------

/// One launch that the fake device saw.
#[derive(Clone, Debug)]
pub(crate) struct Launch {
    /// The address of the first stage of the launch.
    pub stages: usize,
    pub n: u32,
    pub grid: i32,
    pub skip: u32,
}

/// One copy that the fake device saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyRecord {
    pub dst: usize,
    pub src: usize,
    pub bytes: u64,
    /// 1: host to device. 2: device to host.
    pub kind: u32,
}

/// One call of the control op that the fake device saw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ControlRecord {
    /// The address of the first row.
    pub h: usize,
    pub rows: u32,
    /// The address of the vector of the layer.
    pub v: usize,
    pub sum_squares: f32,
    pub op: u32,
    pub strength: f32,
}

/// What a hook says about a launch: the result that the launch returns.
pub(crate) type LaunchHook = Box<dyn FnMut(usize, &Launch) -> Qwen38Launch>;

#[derive(Default)]
struct Device {
    allocations: Vec<(*mut u8, Layout)>,
    launches: Vec<Launch>,
    copies: Vec<CopyRecord>,
    hook: Option<LaunchHook>,
    controls: Vec<ControlRecord>,
    /// The bytes of each call of `hip_rt_malloc` and `hip_rt_host_malloc`, since the thread began.
    sizes: Vec<u64>,
}

impl Drop for Device {
    fn drop(&mut self) {
        for (pointer, layout) in self.allocations.drain(..) {
            unsafe { dealloc(pointer, layout) };
        }
    }
}

thread_local! {
    static DEVICE: RefCell<Device> = RefCell::new(Device::default());
}

/// The launches since the last `reset`.
pub(crate) fn launches() -> Vec<Launch> {
    DEVICE.with(|d| d.borrow().launches.clone())
}

/// The copies since the last `reset`, in order.
pub(crate) fn copies() -> Vec<CopyRecord> {
    DEVICE.with(|d| d.borrow().copies.clone())
}

/// The calls of the control op since the last `reset`, in order.
pub(crate) fn controls() -> Vec<ControlRecord> {
    DEVICE.with(|d| d.borrow().controls.clone())
}

/// Forget the launches, the copies and the control calls. The buffers stay.
pub(crate) fn reset() {
    DEVICE.with(|d| {
        let mut d = d.borrow_mut();
        d.launches.clear();
        d.copies.clear();
        d.controls.clear();
    });
}

/// Run `hook` at each launch. It gets the index of the launch and the launch, and it gives the result.
pub(crate) fn on_launch(hook: impl FnMut(usize, &Launch) -> Qwen38Launch + 'static) {
    DEVICE.with(|d| d.borrow_mut().hook = Some(Box::new(hook)));
}

/// Forget the hook of the launches.
pub(crate) fn clear_hook() {
    DEVICE.with(|d| d.borrow_mut().hook = None);
}

/// The allocations that the fake device made on this thread.
pub(crate) fn mallocs() -> usize {
    DEVICE.with(|d| d.borrow().sizes.len())
}

/// The bytes of each allocation that the fake device made on this thread, in order.
pub(crate) fn malloc_sizes() -> Vec<u64> {
    DEVICE.with(|d| d.borrow().sizes.clone())
}

fn allocate(bytes: u64) -> *mut c_void {
    let layout = Layout::from_size_align(bytes.max(1) as usize, 256).expect("a fake buffer layout");
    // SAFETY: the layout is not empty.
    let pointer = unsafe { alloc_zeroed(layout) };
    assert!(!pointer.is_null(), "the fake device has no memory for {bytes} bytes");
    DEVICE.with(|d| {
        let mut d = d.borrow_mut();
        d.sizes.push(bytes);
        d.allocations.push((pointer, layout));
    });
    pointer.cast()
}

unsafe extern "C" fn rt_malloc(bytes: u64) -> *mut c_void {
    allocate(bytes)
}

unsafe extern "C" fn rt_host_malloc(bytes: u64) -> *mut c_void {
    allocate(bytes)
}

/// The fake frees nothing before the end of the thread, so a pointer stays valid.
unsafe extern "C" fn rt_host_free(_pointer: *mut c_void) {}

unsafe extern "C" fn rt_memset(pointer: *mut c_void, value: i32, bytes: u64) {
    unsafe { std::ptr::write_bytes(pointer.cast::<u8>(), value as u8, bytes as usize) };
}

unsafe extern "C" fn rt_memcpy(dst: *mut c_void, src: *const c_void, bytes: u64, kind: u32) {
    DEVICE.with(|d| {
        d.borrow_mut().copies.push(CopyRecord {
            dst: dst as usize,
            src: src as usize,
            bytes,
            kind,
        })
    });
    unsafe { std::ptr::copy_nonoverlapping(src.cast::<u8>(), dst.cast::<u8>(), bytes as usize) };
}

unsafe extern "C" fn launch_fused(
    _engine: *mut c_void,
    stages: *const Stage,
    n: u32,
    grid: i32,
    skip: u32,
    out: *mut Qwen38Launch,
) {
    let launch = Launch {
        stages: stages as usize,
        n,
        grid,
        skip,
    };
    // Take the hook out while it runs: it may call `copies()` or `launches()`.
    let (index, hook) = DEVICE.with(|d| {
        let mut d = d.borrow_mut();
        d.launches.push(launch.clone());
        (d.launches.len() - 1, d.hook.take())
    });
    let result = match hook {
        Some(mut hook) => {
            let result = hook(index, &launch);
            DEVICE.with(|d| d.borrow_mut().hook = Some(hook));
            result
        }
        None => Qwen38Launch::default(),
    };
    unsafe { *out = result };
}

/// The control op of the fake device: the host op on the memory of the fake device, with the checks of `control.hip`.
unsafe extern "C" fn control_apply(h: *mut f32, rows: u32, v: *const f32, sum_squares: f32, op: u32, strength: f32) {
    DEVICE.with(|d| {
        d.borrow_mut().controls.push(ControlRecord {
            h: h as usize,
            rows,
            v: v as usize,
            sum_squares,
            op,
            strength,
        })
    });
    assert!((1..=256).contains(&rows), "control_apply: {rows} rows: want 1 to 256");
    assert!(
        sum_squares > 0.0,
        "control_apply: the sum of squares {sum_squares} is not above 0"
    );
    let op = control::op_of_code(op).unwrap_or_else(|| panic!("control_apply: op {op}: want 0 to 2"));
    // SAFETY: the caller names `rows` rows at `h` and one vector at `v` in the memory of the fake device.
    let (h, v) = unsafe {
        (
            std::slice::from_raw_parts_mut(h, rows as usize * control::WIDTH),
            std::slice::from_raw_parts(v, control::WIDTH),
        )
    };
    for row in h.chunks_mut(control::WIDTH) {
        assert!(control::apply_row(op, strength, row, v, sum_squares));
    }
}

unsafe extern "C" fn max_coop_grid() -> i32 {
    80
}

unsafe extern "C" fn status_name(_status: u32) -> *const c_char {
    static NAME: &CStr = c"fake";
    NAME.as_ptr()
}

/// A call that a fake test must not reach.
unsafe extern "C" fn never() {
    std::process::abort();
}

/// A binding of the fake runtime. A call that the pass host never makes aborts the process.
pub(crate) fn binding() -> Binding {
    let never_call: unsafe extern "C" fn() = never;
    // The transmutes only name the call types. The fake never calls them, and `never` takes no argument.
    unsafe {
        use std::mem::transmute as to;
        Binding {
            engine_open: to::<unsafe extern "C" fn(), EngineOpen>(never_call),
            relayout: to::<unsafe extern "C" fn(), Relayout>(never_call),
            rt_host_register: to::<unsafe extern "C" fn(), RtHostRegister>(never_call),
            rt_host_unregister: to::<unsafe extern "C" fn(), RtHostUnregister>(never_call),
            rt_host_malloc,
            rt_host_free,
            rt_malloc,
            rt_memset,
            rt_memcpy,
            ctx_new: to::<unsafe extern "C" fn(), CtxNew>(never_call),
            ctx_free: to::<unsafe extern "C" fn(), CtxFree>(never_call),
            ctx_n: to::<unsafe extern "C" fn(), crate::binding::CtxN>(never_call),
            ctx_max: to::<unsafe extern "C" fn(), CtxMax>(never_call),
            ctx_owns_kv: to::<unsafe extern "C" fn(), crate::binding::CtxOwnsKv>(never_call),
            ctx_poisoned: to::<unsafe extern "C" fn(), crate::binding::CtxPoisoned>(never_call),
            state_bytes: to::<unsafe extern "C" fn(), StateBytes>(never_call),
            part_bytes: to::<unsafe extern "C" fn(), crate::binding::PartBytes>(never_call),
            state_read: to::<unsafe extern "C" fn(), StateRead>(never_call),
            state_write: to::<unsafe extern "C" fn(), crate::binding::StateWrite>(never_call),
            cells_write: to::<unsafe extern "C" fn(), crate::binding::CellsWrite>(never_call),
            set_n: to::<unsafe extern "C" fn(), crate::binding::SetN>(never_call),
            round_gdn: to::<unsafe extern "C" fn(), RoundGdn>(never_call),
            launch_fused,
            launch_each: to::<unsafe extern "C" fn(), crate::binding::LaunchEach>(never_call),
            control_apply,
            max_coop_grid,
            status_name,
            mem_get_info: to::<unsafe extern "C" fn(), crate::binding::MemGetInfo>(never_call),
        }
    }
}

// ---- a pass over the fake device ----------------------------------------------------------------------------------

/// What a stage says of itself in a test: its op, its layer, its segment and its schedule word.
pub(crate) type StageKey = (u32, u32, u32, u32);

/// The stages of a launch, read from the memory that the launch names.
pub(crate) fn stage_keys(launch: &Launch) -> Vec<StageKey> {
    // SAFETY: the host uploaded `n` stages at this address, and the fake memory stays valid until the thread ends.
    let stages = unsafe { std::slice::from_raw_parts(launch.stages as *const Stage, launch.n as usize) };
    stages.iter().map(|s| (s.op, s.layer, s.sub, s.sched)).collect()
}

/// The `Stage` structs of a launch, read from the memory that the launch names.
pub(crate) fn stages_of(launch: &Launch) -> &'static [Stage] {
    // SAFETY: as for `stage_keys`. The caller reads the list before the next upload.
    unsafe { std::slice::from_raw_parts(launch.stages as *const Stage, launch.n as usize) }
}

/// The fast path of a pass, with every fusion on, as the engine runs it.
pub(crate) const FAST: Knobs = Knobs {
    fast_qmm: true,
    fuse: true,
    fuse_hc: true,
    fuse_hcmix: true,
    elide: true,
    tail: true,
    debug_layers: 0,
};

/// A pass host over the fake device: its fake weights, its contexts and its scratch pool.
pub(crate) struct Rig {
    pub binding: Binding,
    pub host: Host,
    weights: Weights,
    /// Boxed: the pass keeps the address of each context, so a context must not move.
    #[allow(clippy::vec_box)]
    pub contexts: Vec<Box<Qwen38Context>>,
    lens: Vec<u32>,
    tokens: Vec<i32>,
}

impl Rig {
    /// A rig with one context for each length in `lens`, each holding 5 tokens.
    pub(crate) fn new(lens: &[u32]) -> Rig {
        let binding = binding();
        // SAFETY: the fake binding makes host memory.
        let host = unsafe { Host::new(&binding) };
        Rig {
            host,
            binding,
            weights: weights(),
            contexts: (0..lens.len()).map(|i| context(5, 4096, i as u64)).collect(),
            lens: lens.to_vec(),
            tokens: lens.iter().flat_map(|len| (0..*len as i32).map(|i| 1000 + i)).collect(),
        }
    }

    /// The device scratch `h`.
    pub(crate) fn h(&mut self) -> *mut f32 {
        self.host.scratch(
            &self.binding,
            "h",
            crate::pass::MAX_PASS_TOKENS as usize * crate::pass::HC_DIM as usize,
        )
    }

    /// The device scratch `mixed`.
    pub(crate) fn mixed(&mut self) -> *mut f32 {
        self.host.scratch(
            &self.binding,
            "mixed",
            crate::pass::MAX_PASS_TOKENS as usize * crate::pass::EMBD as usize,
        )
    }

    /// The tokens of the pass.
    pub(crate) fn rows(&self) -> usize {
        self.lens.iter().map(|len| *len as usize).sum()
    }

    /// Put each context back to 5 tokens, so that the next pass builds the same stage list.
    pub(crate) fn rewind(&mut self) {
        for (i, context) in self.contexts.iter_mut().enumerate() {
            **context = *context_at(i as u64);
        }
    }

    /// One pass with the logits, over the fake device.
    pub(crate) fn run(&mut self, knobs: Knobs, launch: LaunchKnobs, cuts: Option<&mut Cuts>) -> Result<f64, PassStop> {
        let pointers: Vec<*mut c_void> = self
            .contexts
            .iter_mut()
            .map(|context| &mut **context as *mut Qwen38Context as *mut c_void)
            .collect();
        let segments = binding::segments(&pointers, &self.lens, &self.tokens);
        let ple = vec![0f32; self.rows() * crate::ple::EMBD];
        let mut logits = vec![0f32; self.lens.len() * crate::pass::VOCAB as usize];
        let input = PassIn {
            segments: &segments,
            ple: &ple,
            knobs,
            launch,
            cuts,
        };
        // SAFETY: the fake binding, a null engine that the fake launch ignores, and live fake contexts.
        unsafe {
            host::run_pass(
                &self.binding,
                std::ptr::null_mut(),
                &mut self.host,
                &self.weights,
                input,
                Some(&mut logits),
            )
        }
    }
}

/// A fresh context of the rig: 5 tokens.
fn context_at(slot: u64) -> Box<Qwen38Context> {
    context(5, 4096, slot)
}

/// The launch knobs with none set.
pub(crate) fn no_knobs() -> LaunchKnobs {
    LaunchKnobs::from(|_| None).expect("no knob is set")
}
