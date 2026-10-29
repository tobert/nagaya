//! The graphs of the engine as lists of tasks for the shared interpreter: the two text encoders, the UNet and the VAE
//! decoder of SDXL base 1.0. The reference is diffusers 0.42 with transformers 5.21.
//!
//! Activations are f32 and pixel-major (`kernels/sdxl_ops.cl`). The batch of a UNet step holds each image twice: the
//! rows of the negative prompt for every image first, then the rows of the prompt, as diffusers joins them. The
//! builders here need no card. Each one gives runs: a run is one ring of tasks.

use nagaya_mk_bmg::protocol::{MAX_TASKS_PER_CARD, MkTaskDescriptor};
use nagaya_mk_bmg::tasks as t;

use crate::spec::*;
use crate::weights::{Tensor, Weights};

/// Bytes of an allocation, rounded up to 256.
pub fn rounded(bytes: usize) -> usize {
    bytes.div_ceil(256) * 256
}

/// A list of tasks that fits in one ring.
#[derive(Clone, Debug, Default)]
pub struct Run {
    pub tasks: Vec<MkTaskDescriptor>,
}

impl Run {
    fn push(&mut self, task: MkTaskDescriptor) {
        assert!(
            self.tasks.len() + 1 < MAX_TASKS_PER_CARD,
            "a run of the image engine holds at most {} tasks",
            MAX_TASKS_PER_CARD - 1
        );
        self.tasks.push(task);
    }
}

/// A region of device memory that the graphs carve into buffers. A count gives addresses from 0.
#[derive(Clone, Copy, Debug)]
pub struct Carve {
    next: u64,
    end: u64,
}

impl Carve {
    pub fn new(at: u64, bytes: u64) -> Self {
        Self {
            next: at,
            end: at + bytes,
        }
    }

    /// A buffer of `floats` f32 values.
    pub fn floats(&mut self, floats: usize) -> u64 {
        let at = self.next;
        self.next += rounded(floats * 4) as u64;
        assert!(
            self.next <= self.end,
            "the activation arena is too small; this is a defect of the plan"
        );
        at
    }

    pub fn used(&self, from: u64) -> u64 {
        self.next - from
    }
}

fn w(weights: &Weights, name: &str, shape: &[usize]) -> Tensor {
    weights.get(name, shape).unwrap_or_else(|e| panic!("{e}"))
}

fn w32(weights: &Weights, name: &str, shape: &[usize]) -> u64 {
    weights.f32(name, shape).unwrap_or_else(|e| panic!("{e}"))
}

/// The planes of a weight for the matrix units, or `None` when it runs as a plain op.
fn planes(weights: &Weights, name: &str, shape: &[usize]) -> Option<crate::weights::Planes> {
    weights.planes(name, shape).unwrap_or_else(|e| panic!("{e}"))
}

/// Check that the checkpoint has each tensor of each graph, with its shape: build the graphs of both encoders with each
/// `clip_skip`, of the UNet at the largest image, and of the VAE, and run none of them.
pub fn check(weights: &Weights) -> Result<(), String> {
    weights.probe(|| {
        let mut carve = Carve::new(0, u64::MAX / 2);
        let text = TextBufs::carve(&mut carve, 2 * MAX_CHUNKS);
        for second in [false, true] {
            for skip in [2, 3] {
                text_encoder(weights, second, &text, 2 * MAX_CHUNKS, skip, [1, 1]);
            }
        }
        let rows = MAX_IMAGES * MAX_IMAGE_PIXELS / (VAE_SCALE * VAE_SCALE);
        let bufs = UnetBufs::carve(&mut carve, MAX_IMAGES, rows, MAX_CHUNKS * CHUNK_IDS);
        let size = UnetSize {
            images: 1,
            h: 128,
            w: 128,
            ctx: MAX_CHUNKS * CHUNK_IDS,
        };
        unet(weights, &bufs, &size);
        let vae_bufs = VaeBufs::carve(&mut carve, 128 * 128 * VAE_SCALE * VAE_SCALE);
        vae(weights, &vae_bufs, 128, 128);
    })
}

/// A linear layer of `name` (`.weight` [cout][cin], or [cout][cin][1][1] with `conv`, and an optional `.bias`). It
/// runs on the matrix units when the weight has planes.
#[allow(clippy::too_many_arguments)]
fn linear_any(
    run: &mut Run,
    wt: &Weights,
    name: &str,
    conv: bool,
    bias: bool,
    x: u64,
    y: u64,
    rows: usize,
    cin: usize,
    cout: usize,
    acc: bool,
) {
    let wname = format!("{name}.weight");
    let shape: &[usize] = if conv { &[cout, cin, 1, 1] } else { &[cout, cin] };
    let b = if bias {
        Some(w(wt, &format!("{name}.bias"), &[cout]))
    } else {
        None
    };
    match planes(wt, &wname, shape) {
        Some(p) => run.push(t::gemm(
            x,
            p.at,
            b.map_or(0, |b| b.at),
            y,
            rows as u32,
            1,
            1,
            cin as u32,
            cout as u32,
            (1, 1, 0),
            p.planes,
            b.is_some_and(|b| !b.f32),
            acc,
        )),
        None => {
            let weight = w(wt, &wname, shape);
            let b = b.map_or(0, |b| b.at);
            run.push(t::linear(
                x,
                weight.at,
                b,
                y,
                rows as u32,
                cin as u32,
                cout as u32,
                weight.f32,
                acc,
            ));
        }
    }
}

/// A linear layer of `name` (`.weight` [cout][cin] and an optional `.bias`).
#[allow(clippy::too_many_arguments)]
fn linear(
    run: &mut Run,
    wt: &Weights,
    name: &str,
    bias: bool,
    x: u64,
    y: u64,
    rows: usize,
    cin: usize,
    cout: usize,
    acc: bool,
) {
    linear_any(run, wt, name, false, bias, x, y, rows, cin, cout, acc);
}

/// A 1 x 1 convolution of `name` (`.weight` [cout][cin][1][1] and `.bias`) as a linear layer over rows.
#[allow(clippy::too_many_arguments)]
fn conv1x1(run: &mut Run, wt: &Weights, name: &str, x: u64, y: u64, rows: usize, cin: usize, cout: usize, acc: bool) {
    linear_any(run, wt, name, true, true, x, y, rows, cin, cout, acc);
}

/// A convolution of `name` with a k x k kernel and its bias.
#[allow(clippy::too_many_arguments)]
fn conv(
    run: &mut Run,
    wt: &Weights,
    name: &str,
    x: u64,
    y: u64,
    images: usize,
    (h, wd): (usize, usize),
    (cin, cout): (usize, usize),
    (kernel, stride): (usize, usize),
    acc: bool,
) {
    let wname = format!("{name}.weight");
    let b = w(wt, &format!("{name}.bias"), &[cout]);
    if let Some(p) = planes(wt, &wname, &[cout, cin, kernel, kernel]) {
        run.push(t::gemm(
            x,
            p.at,
            b.at,
            y,
            images as u32,
            h as u32,
            wd as u32,
            cin as u32,
            cout as u32,
            (kernel as u32, stride as u32, (kernel / 2) as u32),
            p.planes,
            !b.f32,
            acc,
        ));
        return;
    }
    let weight = w(wt, &wname, &[cout, cin, kernel, kernel]);
    run.push(t::conv2d(
        x,
        weight.at,
        b.at,
        y,
        images as u32,
        h as u32,
        wd as u32,
        cin as u32,
        cout as u32,
        kernel as u32,
        stride as u32,
        (kernel / 2) as u32,
        weight.f32,
        acc,
    ));
}

/// A GroupNorm of `name` (`.weight` and `.bias` of `k` channels) in 32 groups.
#[allow(clippy::too_many_arguments)]
fn groupnorm(
    run: &mut Run,
    wt: &Weights,
    name: &str,
    x: u64,
    y: u64,
    images: usize,
    pixels: usize,
    k: usize,
    eps: f32,
    silu: bool,
) {
    let weight = w(wt, &format!("{name}.weight"), &[k]);
    let b = w(wt, &format!("{name}.bias"), &[k]);
    run.push(t::groupnorm(
        x,
        weight.at,
        b.at,
        y,
        images as u32,
        pixels as u32,
        k as u32,
        GROUPS,
        eps,
        weight.f32,
        silu,
    ));
}

/// A LayerNorm of `name` with eps 1e-5, from the F32 copies of its weights.
fn layernorm(run: &mut Run, wt: &Weights, name: &str, x: u64, y: u64, rows: usize, k: usize) {
    let weight = w32(wt, &format!("{name}.weight"), &[k]);
    let b = w32(wt, &format!("{name}.bias"), &[k]);
    run.push(t::layernorm(x, weight, b, y, rows as u32, k as u32, 1e-5));
}

/// The buffers of the text encoders, for `seqs` sequences of 77 rows.
#[derive(Clone, Copy, Debug)]
pub struct TextBufs {
    /// The ids of encoder 1 and of encoder 2, then the position of each row (u32).
    pub ids1: u64,
    pub ids2: u64,
    pub pos: u64,
    pub x: u64,
    pub n: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub att: u64,
    pub mlp: u64,
    /// The hidden state that the job takes, of each encoder.
    pub hidden1: u64,
    pub hidden2: u64,
    /// The rows of the end ids of the prompt and of the negative prompt, and the pooled output of encoder 2.
    pub eos: u64,
    pub pooled: u64,
}

impl TextBufs {
    pub fn carve(carve: &mut Carve, seqs: usize) -> Self {
        let rows = seqs * CHUNK_IDS;
        let wide = TEXT2.hidden;
        Self {
            ids1: carve.floats(rows),
            ids2: carve.floats(rows),
            pos: carve.floats(rows),
            x: carve.floats(rows * wide),
            n: carve.floats(rows * wide),
            q: carve.floats(rows * wide),
            k: carve.floats(rows * wide),
            v: carve.floats(rows * wide),
            att: carve.floats(rows * wide),
            mlp: carve.floats(rows * TEXT2.mlp),
            hidden1: carve.floats(rows * TEXT1.hidden),
            hidden2: carve.floats(rows * TEXT2.hidden),
            eos: carve.floats(2 * wide),
            pooled: carve.floats(2 * POOLED),
        }
    }
}

/// The runs of one text encoder over `seqs` sequences: the hidden state `-clip_skip`, in A1111's count, goes to
/// `hidden`. For encoder 2 also the pooled output: the final norm and the projection of the rows `eos_rows` (one of
/// the negative prompt, one of the prompt).
pub fn text_encoder(
    wt: &Weights,
    second: bool,
    b: &TextBufs,
    seqs: usize,
    clip_skip: usize,
    eos_rows: [usize; 2],
) -> Vec<Run> {
    let (shape, p, ids, hidden) = if second {
        (TEXT2, "te2.text_model", b.ids2, b.hidden2)
    } else {
        (TEXT1, "te1.text_model", b.ids1, b.hidden1)
    };
    let (rows, hid) = (seqs * CHUNK_IDS, shape.hidden);
    let mut runs = vec![Run::default()];
    let run = runs.last_mut().expect("a run");
    let tok = w32(wt, &format!("{p}.embeddings.token_embedding.weight"), &[49408, hid]);
    let pos = w32(
        wt,
        &format!("{p}.embeddings.position_embedding.weight"),
        &[CHUNK_IDS, hid],
    );
    run.push(t::embed(ids, tok, b.x, rows as u32, hid as u32, 1, true, false));
    run.push(t::embed(b.pos, pos, b.x, rows as u32, hid as u32, 1, true, true));
    for layer in 0..shape.layers {
        let l = format!("{p}.encoder.layers.{layer}");
        layernorm(run, wt, &format!("{l}.layer_norm1"), b.x, b.n, rows, hid);
        for (proj, out) in [("q_proj", b.q), ("k_proj", b.k), ("v_proj", b.v)] {
            linear(
                run,
                wt,
                &format!("{l}.self_attn.{proj}"),
                true,
                b.n,
                out,
                rows,
                hid,
                hid,
                false,
            );
        }
        run.push(t::sdpa(
            b.q,
            b.k,
            b.v,
            b.att,
            seqs as u32,
            CHUNK_IDS as u32,
            CHUNK_IDS as u32,
            (hid / HEAD) as u32,
            HEAD as u32,
            true,
        ));
        linear(
            run,
            wt,
            &format!("{l}.self_attn.out_proj"),
            true,
            b.att,
            b.x,
            rows,
            hid,
            hid,
            true,
        );
        layernorm(run, wt, &format!("{l}.layer_norm2"), b.x, b.n, rows, hid);
        linear(
            run,
            wt,
            &format!("{l}.mlp.fc1"),
            true,
            b.n,
            b.mlp,
            rows,
            hid,
            shape.mlp,
            false,
        );
        if shape.quick_gelu {
            run.push(t::quick_gelu(b.mlp, b.mlp, rows as u32, shape.mlp as u32));
        } else {
            run.push(t::gelu(b.mlp, b.mlp, rows as u32, shape.mlp as u32));
        }
        linear(
            run,
            wt,
            &format!("{l}.mlp.fc2"),
            true,
            b.mlp,
            b.x,
            rows,
            shape.mlp,
            hid,
            true,
        );
        // hidden_states[-clip_skip] is the output of layer `layers - clip_skip`, with layers counted from 0.
        if layer + clip_skip == shape.layers {
            run.push(t::copy(b.x, hidden, (rows * hid) as u32));
        }
    }
    if second {
        layernorm(run, wt, &format!("{p}.final_layer_norm"), b.x, b.n, rows, hid);
        for (i, row) in eos_rows.iter().enumerate() {
            run.push(t::copy(
                b.n + (row * hid * 4) as u64,
                b.eos + (i * hid * 4) as u64,
                hid as u32,
            ));
        }
        linear(
            run,
            wt,
            "te2.text_projection",
            false,
            b.eos,
            b.pooled,
            2,
            hid,
            POOLED,
            false,
        );
    }
    runs
}

/// The size of a UNet step: images (each run twice), and the latent rows and columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnetSize {
    pub images: usize,
    pub h: usize,
    pub w: usize,
    /// The rows of the text context of each image: 77 times the chunks.
    pub ctx: usize,
}

impl UnetSize {
    /// The batch of a step: each image with the negative prompt, then each image with the prompt.
    pub fn batch(&self) -> usize {
        2 * self.images
    }

    /// The rows and columns of level `l`.
    pub fn dims(&self, l: usize) -> (usize, usize) {
        (self.h >> l, self.w >> l)
    }

    /// The rows of all images of the batch at level `l`.
    pub fn rows(&self, l: usize) -> usize {
        let (h, w) = self.dims(l);
        self.batch() * h * w
    }
}

/// The buffers of a UNet step.
#[derive(Clone, Copy, Debug)]
pub struct UnetBufs {
    /// The inputs: the latents of the batch, the timestep features, the add features, and the context.
    pub x_in: u64,
    pub t_proj: u64,
    pub add_proj: u64,
    pub ctx: u64,
    /// The output: the noise prediction of the batch.
    pub eps: u64,
    pub emb: u64,
    pub emb_a: u64,
    pub semb: u64,
    pub temb: u64,
    pub skips: [u64; 9],
    pub a: u64,
    pub c: u64,
    pub cat: u64,
    pub t1: u64,
    pub t2: u64,
    pub h: u64,
    pub n: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub att: u64,
    pub ck: u64,
    pub cv: u64,
    pub ff: u64,
    pub ffh: u64,
    /// The BF16 planes of the keys and values of one attention (`OP_ATTN_PREP`).
    pub planes: u64,
}

impl UnetBufs {
    /// Carve the buffers for at most `max` latent rows of all images of one prompt, and `ctx` context rows.
    pub fn carve(carve: &mut Carve, images: usize, max_rows: usize, ctx: usize) -> Self {
        let r0 = 2 * max_rows;
        let b2 = 2 * images;
        let skip_floats = [
            320 * r0,
            320 * r0,
            320 * r0,
            320 * r0 / 4,
            640 * r0 / 4,
            640 * r0 / 4,
            640 * r0 / 16,
            1280 * r0 / 16,
            1280 * r0 / 16,
        ];
        Self {
            x_in: carve.floats(LATENT * r0),
            t_proj: carve.floats(b2 * TIME_PROJ),
            add_proj: carve.floats(b2 * ADD_IN),
            ctx: carve.floats(b2 * ctx * CONTEXT),
            eps: carve.floats(LATENT * r0),
            emb: carve.floats(b2 * TIME),
            emb_a: carve.floats(b2 * TIME),
            semb: carve.floats(b2 * TIME),
            temb: carve.floats(b2 * TIME),
            skips: skip_floats.map(|f| carve.floats(f)),
            // The upsampler of up block 1 writes 640 channels at level 0.
            a: carve.floats(640 * r0),
            c: carve.floats(640 * r0),
            cat: carve.floats(960 * r0),
            t1: carve.floats(960 * r0),
            t2: carve.floats(320 * r0),
            h: carve.floats(160 * r0),
            n: carve.floats(160 * r0),
            q: carve.floats(160 * r0),
            k: carve.floats(160 * r0),
            v: carve.floats(160 * r0),
            att: carve.floats(160 * r0),
            ck: carve.floats(b2 * ctx * 1280),
            cv: carve.floats(b2 * ctx * 1280),
            ff: carve.floats(1280 * r0),
            ffh: carve.floats(640 * r0),
            // Six ushorts of each value of the keys, padded to 16 rows: at most the rows of level 1 by 640 channels,
            // or of the context.
            planes: carve.floats(3 * (160 * r0 + b2 * 16 * 1280).max(b2 * (ctx + 16) * 1280)),
        }
    }
}

/// A resnet block: `out = shortcut(x) + conv2(silu(gn2(conv1(silu(gn1(x))) + temb)))`. With `semb`, the time
/// embedding of each image is added after conv1 (UNet); with none, there is no time embedding (VAE).
#[allow(clippy::too_many_arguments)]
fn resnet(
    run: &mut Run,
    wt: &Weights,
    name: &str,
    x: u64,
    out: u64,
    (t1, t2, temb): (u64, u64, u64),
    semb: Option<u64>,
    images: usize,
    (h, wd): (usize, usize),
    (cin, cout): (usize, usize),
    eps: f32,
) {
    let px = h * wd;
    let rows = images * px;
    groupnorm(run, wt, &format!("{name}.norm1"), x, t1, images, px, cin, eps, true);
    conv(
        run,
        wt,
        &format!("{name}.conv1"),
        t1,
        t2,
        images,
        (h, wd),
        (cin, cout),
        (3, 1),
        false,
    );
    if let Some(semb) = semb {
        linear(
            run,
            wt,
            &format!("{name}.time_emb_proj"),
            true,
            semb,
            temb,
            images,
            TIME,
            cout,
            false,
        );
        run.push(t::add_image_row(temb, t2, images as u32, px as u32, cout as u32));
    }
    groupnorm(run, wt, &format!("{name}.norm2"), t2, t1, images, px, cout, eps, true);
    if cin == cout {
        run.push(t::copy(x, out, (rows * cout) as u32));
    } else {
        conv1x1(
            run,
            wt,
            &format!("{name}.conv_shortcut"),
            x,
            out,
            rows,
            cin,
            cout,
            false,
        );
    }
    conv(
        run,
        wt,
        &format!("{name}.conv2"),
        t1,
        out,
        images,
        (h, wd),
        (cout, cout),
        (3, 1),
        true,
    );
}

/// A Transformer2DModel with linear projections, in place on `x`: `x += proj_out(blocks(proj_in(gn(x))))`.
#[allow(clippy::too_many_arguments)]
fn transformer(
    run: &mut Run,
    wt: &Weights,
    name: &str,
    x: u64,
    b: &UnetBufs,
    size: &UnetSize,
    level: usize,
    depth: usize,
) {
    let c = LEVELS[level];
    let (hh, ww) = size.dims(level);
    let px = hh * ww;
    let batch = size.batch();
    let rows = batch * px;
    let heads = (c / HEAD) as u32;
    groupnorm(run, wt, &format!("{name}.norm"), x, b.t1, batch, px, c, 1e-6, false);
    linear(run, wt, &format!("{name}.proj_in"), true, b.t1, b.h, rows, c, c, false);
    for block in 0..depth {
        let bl = format!("{name}.transformer_blocks.{block}");
        layernorm(run, wt, &format!("{bl}.norm1"), b.h, b.n, rows, c);
        for (proj, out) in [("to_q", b.q), ("to_k", b.k), ("to_v", b.v)] {
            linear(
                run,
                wt,
                &format!("{bl}.attn1.{proj}"),
                false,
                b.n,
                out,
                rows,
                c,
                c,
                false,
            );
        }
        run.push(t::attn_prep(
            b.k,
            b.v,
            b.planes,
            batch as u32,
            px as u32,
            heads,
            HEAD as u32,
        ));
        run.push(t::attn_xmx(
            b.q,
            b.planes,
            b.att,
            batch as u32,
            px as u32,
            px as u32,
            heads,
            HEAD as u32,
        ));
        linear(
            run,
            wt,
            &format!("{bl}.attn1.to_out.0"),
            true,
            b.att,
            b.h,
            rows,
            c,
            c,
            true,
        );
        layernorm(run, wt, &format!("{bl}.norm2"), b.h, b.n, rows, c);
        linear(run, wt, &format!("{bl}.attn2.to_q"), false, b.n, b.q, rows, c, c, false);
        let ctx_rows = batch * size.ctx;
        linear(
            run,
            wt,
            &format!("{bl}.attn2.to_k"),
            false,
            b.ctx,
            b.ck,
            ctx_rows,
            CONTEXT,
            c,
            false,
        );
        linear(
            run,
            wt,
            &format!("{bl}.attn2.to_v"),
            false,
            b.ctx,
            b.cv,
            ctx_rows,
            CONTEXT,
            c,
            false,
        );
        run.push(t::attn_prep(
            b.ck,
            b.cv,
            b.planes,
            batch as u32,
            size.ctx as u32,
            heads,
            HEAD as u32,
        ));
        run.push(t::attn_xmx(
            b.q,
            b.planes,
            b.att,
            batch as u32,
            px as u32,
            size.ctx as u32,
            heads,
            HEAD as u32,
        ));
        linear(
            run,
            wt,
            &format!("{bl}.attn2.to_out.0"),
            true,
            b.att,
            b.h,
            rows,
            c,
            c,
            true,
        );
        layernorm(run, wt, &format!("{bl}.norm3"), b.h, b.n, rows, c);
        linear(
            run,
            wt,
            &format!("{bl}.ff.net.0.proj"),
            true,
            b.n,
            b.ff,
            rows,
            c,
            8 * c,
            false,
        );
        run.push(t::geglu_half(b.ff, b.ffh, rows as u32, (4 * c) as u32));
        linear(
            run,
            wt,
            &format!("{bl}.ff.net.2"),
            true,
            b.ffh,
            b.h,
            rows,
            4 * c,
            c,
            true,
        );
    }
    linear(run, wt, &format!("{name}.proj_out"), true, b.h, x, rows, c, c, true);
}

/// The names of the runs of a UNet step, in order. A test reads the hidden state after each run.
pub const UNET_RUNS: [&str; 9] = ["emb", "down0", "down1", "down2", "mid", "up0", "up1", "up2", "out"];

/// The runs of one UNet step, and the buffer that holds the hidden state after each run.
pub fn unet(wt: &Weights, b: &UnetBufs, size: &UnetSize) -> Vec<(Run, u64)> {
    let batch = size.batch();
    let resnet_bufs = (b.t1, b.t2, b.temb);
    let mut runs: Vec<(Run, u64)> = Vec::new();

    // The embeddings: time, and the pooled text with the time ids.
    let mut run = Run::default();
    linear(
        &mut run,
        wt,
        "unet.time_embedding.linear_1",
        true,
        b.t_proj,
        b.emb_a,
        batch,
        TIME_PROJ,
        TIME,
        false,
    );
    run.push(t::silu_mul(b.emb_a, 0, b.emb_a, batch as u32, TIME as u32));
    linear(
        &mut run,
        wt,
        "unet.time_embedding.linear_2",
        true,
        b.emb_a,
        b.emb,
        batch,
        TIME,
        TIME,
        false,
    );
    linear(
        &mut run,
        wt,
        "unet.add_embedding.linear_1",
        true,
        b.add_proj,
        b.emb_a,
        batch,
        ADD_IN,
        TIME,
        false,
    );
    run.push(t::silu_mul(b.emb_a, 0, b.emb_a, batch as u32, TIME as u32));
    linear(
        &mut run,
        wt,
        "unet.add_embedding.linear_2",
        true,
        b.emb_a,
        b.emb,
        batch,
        TIME,
        TIME,
        true,
    );
    run.push(t::silu_mul(b.emb, 0, b.semb, batch as u32, TIME as u32));
    conv(
        &mut run,
        wt,
        "unet.conv_in",
        b.x_in,
        b.skips[0],
        batch,
        size.dims(0),
        (LATENT, LEVELS[0]),
        (3, 1),
        false,
    );
    runs.push((run, b.skips[0]));

    // The down blocks. Each resnet and each downsampler writes a skip.
    let mut skip = 1;
    let mut x = b.skips[0];
    let mut cin = LEVELS[0];
    for level in 0..3 {
        let mut run = Run::default();
        let c = LEVELS[level];
        for j in 0..2 {
            let out = b.skips[skip];
            let name = format!("unet.down_blocks.{level}");
            resnet(
                &mut run,
                wt,
                &format!("{name}.resnets.{j}"),
                x,
                out,
                resnet_bufs,
                Some(b.semb),
                batch,
                size.dims(level),
                (cin, c),
                1e-5,
            );
            if DEPTH[level] > 0 {
                transformer(
                    &mut run,
                    wt,
                    &format!("{name}.attentions.{j}"),
                    out,
                    b,
                    size,
                    level,
                    DEPTH[level],
                );
            }
            x = out;
            cin = c;
            skip += 1;
        }
        if level < 2 {
            let out = b.skips[skip];
            let name = format!("unet.down_blocks.{level}.downsamplers.0.conv");
            conv(
                &mut run,
                wt,
                &name,
                x,
                out,
                batch,
                size.dims(level),
                (c, c),
                (3, 2),
                false,
            );
            x = out;
            skip += 1;
        }
        runs.push((run, x));
    }

    // The middle block.
    let mut run = Run::default();
    resnet(
        &mut run,
        wt,
        "unet.mid_block.resnets.0",
        x,
        b.a,
        resnet_bufs,
        Some(b.semb),
        batch,
        size.dims(2),
        (1280, 1280),
        1e-5,
    );
    transformer(&mut run, wt, "unet.mid_block.attentions.0", b.a, b, size, 2, DEPTH[2]);
    resnet(
        &mut run,
        wt,
        "unet.mid_block.resnets.1",
        b.a,
        b.c,
        resnet_bufs,
        Some(b.semb),
        batch,
        size.dims(2),
        (1280, 1280),
        1e-5,
    );
    runs.push((run, b.c));

    // The up blocks. Each resnet joins the hidden state with the last skip along the channels.
    let mut x = b.c;
    let mut ch = 1280;
    let mut other = b.a;
    for up in 0..3 {
        let level = 2 - up;
        let c = LEVELS[level];
        let (hh, ww) = size.dims(level);
        let rows = batch * hh * ww;
        let mut run = Run::default();
        for j in 0..3 {
            skip -= 1;
            let cs = skip_channels(skip);
            let cin = ch + cs;
            run.push(t::copy_cols(x, b.cat, rows as u32, ch as u32, cin as u32, 0));
            run.push(t::copy_cols(
                b.skips[skip],
                b.cat,
                rows as u32,
                cs as u32,
                cin as u32,
                ch as u32,
            ));
            let name = format!("unet.up_blocks.{up}");
            resnet(
                &mut run,
                wt,
                &format!("{name}.resnets.{j}"),
                b.cat,
                other,
                resnet_bufs,
                Some(b.semb),
                batch,
                (hh, ww),
                (cin, c),
                1e-5,
            );
            if DEPTH[level] > 0 {
                transformer(
                    &mut run,
                    wt,
                    &format!("{name}.attentions.{j}"),
                    other,
                    b,
                    size,
                    level,
                    DEPTH[level],
                );
            }
            std::mem::swap(&mut x, &mut other);
            ch = c;
        }
        if up < 2 {
            run.push(t::upsample2x(x, b.t1, batch as u32, hh as u32, ww as u32, c as u32));
            let name = format!("unet.up_blocks.{up}.upsamplers.0.conv");
            conv(
                &mut run,
                wt,
                &name,
                b.t1,
                other,
                batch,
                (hh * 2, ww * 2),
                (c, c),
                (3, 1),
                false,
            );
            std::mem::swap(&mut x, &mut other);
        }
        runs.push((run, x));
    }

    // The output.
    let mut run = Run::default();
    let px = size.h * size.w;
    groupnorm(
        &mut run,
        wt,
        "unet.conv_norm_out",
        x,
        b.t1,
        batch,
        px,
        LEVELS[0],
        1e-5,
        true,
    );
    conv(
        &mut run,
        wt,
        "unet.conv_out",
        b.t1,
        b.eps,
        batch,
        size.dims(0),
        (LEVELS[0], LATENT),
        (3, 1),
        false,
    );
    runs.push((run, b.eps));
    assert_eq!(skip, 0, "the up blocks take each skip of the down blocks");
    runs
}

/// The channels of skip `i`: conv_in, the two resnets and the downsampler of level 0, then of level 1, then the two
/// resnets of level 2.
fn skip_channels(i: usize) -> usize {
    [320, 320, 320, 320, 640, 640, 640, 1280, 1280][i]
}

/// The buffers of a VAE decode of one image of at most `max_pixels` pixels.
#[derive(Clone, Copy, Debug)]
pub struct VaeBufs {
    pub z: u64,
    pub a: u64,
    pub c: u64,
    pub t1: u64,
    pub t2: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub att: u64,
    /// The output: 3 values of each pixel, in [-1, 1] before the clamp.
    pub image: u64,
}

impl VaeBufs {
    pub fn carve(carve: &mut Carve, max_pixels: usize) -> Self {
        let latent = max_pixels / (VAE_SCALE * VAE_SCALE);
        Self {
            z: carve.floats(LATENT * latent),
            a: carve.floats(256 * max_pixels),
            c: carve.floats(256 * max_pixels),
            t1: carve.floats(256 * max_pixels),
            t2: carve.floats(256 * max_pixels),
            q: carve.floats(512 * latent),
            k: carve.floats(512 * latent),
            v: carve.floats(512 * latent),
            att: carve.floats(512 * latent),
            image: carve.floats(3 * max_pixels),
        }
    }
}

/// The run of the VAE decode of one latent of `h` by `w` rows, already divided by the scaling factor.
pub fn vae(wt: &Weights, b: &VaeBufs, h: usize, w: usize) -> Run {
    let mut run = Run::default();
    let bufs = (b.t1, b.t2, 0);
    let mut px = h * w;
    conv1x1(&mut run, wt, "vae.post_quant_conv", b.z, b.c, px, LATENT, LATENT, false);
    conv(
        &mut run,
        wt,
        "vae.decoder.conv_in",
        b.c,
        b.a,
        1,
        (h, w),
        (LATENT, 512),
        (3, 1),
        false,
    );
    let m = "vae.decoder.mid_block";
    resnet(
        &mut run,
        wt,
        &format!("{m}.resnets.0"),
        b.a,
        b.c,
        bufs,
        None,
        1,
        (h, w),
        (512, 512),
        1e-6,
    );
    let at = format!("{m}.attentions.0");
    groupnorm(
        &mut run,
        wt,
        &format!("{at}.group_norm"),
        b.c,
        b.t1,
        1,
        px,
        512,
        1e-6,
        false,
    );
    for (proj, out) in [("to_q", b.q), ("to_k", b.k), ("to_v", b.v)] {
        linear(
            &mut run,
            wt,
            &format!("{at}.{proj}"),
            true,
            b.t1,
            out,
            px,
            512,
            512,
            false,
        );
    }
    run.push(t::sdpa(b.q, b.k, b.v, b.att, 1, px as u32, px as u32, 1, 512, false));
    linear(
        &mut run,
        wt,
        &format!("{at}.to_out.0"),
        true,
        b.att,
        b.c,
        px,
        512,
        512,
        true,
    );
    resnet(
        &mut run,
        wt,
        &format!("{m}.resnets.1"),
        b.c,
        b.a,
        bufs,
        None,
        1,
        (h, w),
        (512, 512),
        1e-6,
    );
    let (mut x, mut other) = (b.a, b.c);
    let (mut hh, mut ww, mut ch) = (h, w, 512);
    for (up, &c) in VAE_LEVELS.iter().enumerate() {
        for j in 0..3 {
            let name = format!("vae.decoder.up_blocks.{up}.resnets.{j}");
            resnet(&mut run, wt, &name, x, other, bufs, None, 1, (hh, ww), (ch, c), 1e-6);
            std::mem::swap(&mut x, &mut other);
            ch = c;
        }
        if up < 3 {
            run.push(t::upsample2x(x, b.t1, 1, hh as u32, ww as u32, c as u32));
            hh *= 2;
            ww *= 2;
            let name = format!("vae.decoder.up_blocks.{up}.upsamplers.0.conv");
            conv(&mut run, wt, &name, b.t1, other, 1, (hh, ww), (c, c), (3, 1), false);
            std::mem::swap(&mut x, &mut other);
        }
    }
    px = hh * ww;
    groupnorm(
        &mut run,
        wt,
        "vae.decoder.conv_norm_out",
        x,
        b.t1,
        1,
        px,
        128,
        1e-6,
        true,
    );
    conv(
        &mut run,
        wt,
        "vae.decoder.conv_out",
        b.t1,
        b.image,
        1,
        (hh, ww),
        (128, 3),
        (3, 1),
        false,
    );
    run
}
