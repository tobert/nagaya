//! The plan of one chunk of Beat This!: a task list with no arithmetic in it, for every interpreter of the model.
//!
//! A plan holds f32 buffers, the weights that its tasks read, the tasks in order, and marks that name the rows of the
//! oracle (`parts.*` of `tools/beatthis_golden.sh`). The CPU engine runs it first; a GPU port runs the same plan with
//! a kernel for each task kind. Thus a port needs no knowledge of the model, only of the seven task kinds.
//!
//! Layout. A buffer is row-major f32. The frontend keeps channels last: a row is one frame and one frequency, so a
//! buffer of the frontend is `[frames, freqs, channels]` (the oracle's rows are `[channels, freqs, frames]`). The
//! transformer rows are `[frames, dim]`. The model's `b c f t -> b t (c f)` before the projection reads channels
//! first; the plan reads the channels-last rows and permutes the columns of the projection's weight instead.

use crate::Error;
use crate::mel::MELS;

/// The hyper_parameters of a checkpoint that change the plan. The rest are fixed by this family: `spect_dim` 128,
/// a sum head, and partial transformers in the frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub stem_dim: usize,
    pub transformer_dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub ff_mult: usize,
}

impl Config {
    /// Reads the `hyper_parameters` JSON of a converted checkpoint. A model that this family does not build is an
    /// error that names the key.
    pub fn from_json(text: &str) -> Result<Config, Error> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| Error::Config(format!("hyper_parameters is not JSON: {e}")))?;
        let int = |key: &str| -> Result<usize, Error> {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| Error::Config(format!("{key} is not a whole number")))
        };
        // BeatThis() defaults both to true, and the checkpoint of final0 does not name them.
        for key in ["sum_head", "partial_transformers"] {
            if v.get(key).is_some_and(|b| b.as_bool() != Some(true)) {
                return Err(Error::Config(format!(
                    "{key} is not true, and this family builds only that model"
                )));
            }
        }
        if int("spect_dim")? != MELS {
            return Err(Error::Config(format!("spect_dim is not {MELS}")));
        }
        let c = Config {
            stem_dim: int("stem_dim")?,
            transformer_dim: int("transformer_dim")?,
            n_layers: int("n_layers")?,
            head_dim: int("head_dim")?,
            ff_mult: int("ff_mult")?,
        };
        if c.head_dim == 0
            || !c.head_dim.is_multiple_of(2)
            || !c.stem_dim.is_multiple_of(c.head_dim)
            || !c.transformer_dim.is_multiple_of(c.head_dim)
        {
            return Err(Error::Config(format!(
                "head_dim {} must be even and divide stem_dim {} and transformer_dim {}",
                c.head_dim, c.stem_dim, c.transformer_dim
            )));
        }
        Ok(c)
    }
}

/// A buffer of the plan, by index into `Plan::buffers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf(pub usize);

/// A prepared weight of the plan, by index into `Plan::weights`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightId(pub usize);

/// How a weight of the plan comes from the tensors of the checkpoint. Each form is exact but `BnScale` and
/// `BnShift`, which fold a BatchNorm in eval mode (eps 1e-5) into a scale and a shift.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prep {
    /// The tensor as it is, row-major.
    Plain { name: String, shape: Vec<usize> },
    /// `weight / sqrt(running_var + eps)`, for each channel.
    BnScale { prefix: String, channels: usize },
    /// `bias - running_mean * weight / sqrt(running_var + eps)`, for each channel.
    BnShift { prefix: String, channels: usize },
    /// A conv weight `[cout, cin, kf, kt]` reordered to `[cout, kf, kt, cin]`, for channels-last input.
    ConvKernel {
        name: String,
        cout: usize,
        cin: usize,
        kf: usize,
        kt: usize,
    },
    /// A linear weight `[rows, channels * freqs]` whose columns read channel-major, reordered to read
    /// frequency-major: column `f * channels + c` of the result is column `c * freqs + f` of the tensor.
    PermuteColumns {
        name: String,
        rows: usize,
        channels: usize,
        freqs: usize,
    },
}

impl Prep {
    /// The tensors of the checkpoint that this weight reads, with their shapes.
    pub fn sources(&self) -> Vec<(String, Vec<usize>)> {
        match self {
            Prep::Plain { name, shape } => vec![(name.clone(), shape.clone())],
            Prep::BnScale { prefix, channels } | Prep::BnShift { prefix, channels } => {
                ["weight", "bias", "running_mean", "running_var"]
                    .iter()
                    .map(|p| (format!("{prefix}.{p}"), vec![*channels]))
                    .collect()
            }
            Prep::ConvKernel {
                name,
                cout,
                cin,
                kf,
                kt,
            } => vec![(name.clone(), vec![*cout, *cin, *kf, *kt])],
            Prep::PermuteColumns {
                name,
                rows,
                channels,
                freqs,
            } => {
                vec![(name.clone(), vec![*rows, channels * freqs])]
            }
        }
    }

    /// The f32 values of the prepared weight.
    pub fn len(&self) -> usize {
        match self {
            Prep::Plain { shape, .. } => shape.iter().product(),
            Prep::BnScale { channels, .. } | Prep::BnShift { channels, .. } => *channels,
            Prep::ConvKernel { cout, cin, kf, kt, .. } => cout * cin * kf * kt,
            Prep::PermuteColumns {
                rows, channels, freqs, ..
            } => rows * channels * freqs,
        }
    }

    /// True when the weight has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One weight of the plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Weight {
    pub prep: Prep,
}

/// The axis of a grid of rows that an attention runs along. Row `i * b + j` of a grid of `a` by `b` rows is the
/// token at `(i, j)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// The sequences are the `b` columns, each of `a` tokens: attention over frames.
    A,
    /// The sequences are the `a` rows, each of `b` tokens: attention over frequencies.
    B,
}

/// One task. Each reads and writes whole f32 buffers from offset 0. Output buffers never alias input buffers, but in
/// `Gelu`, which works in place, and `AddInto`, which adds into its first buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Task {
    /// `y[r, c] = x[r, c] * scale[c] + shift[c]`.
    Affine {
        x: Buf,
        y: Buf,
        rows: usize,
        cols: usize,
        scale: WeightId,
        shift: WeightId,
    },
    /// A 2-D convolution of channels-last rows `[frames, f_in, cin]` to `[frames, f_out, cout]`, with
    /// `f_out = (f_in - kf) / stride + 1`: no padding over frequency, a stride over frequency, padding `kt / 2` of
    /// zeros over frames, stride 1 over frames, no bias.
    /// `y[t, o, co] = sum over (i < kf, j < kt, ci) of kernel[co, i, j, ci] * x[t + j - kt / 2, o * stride + i, ci]`.
    Conv {
        x: Buf,
        y: Buf,
        frames: usize,
        f_in: usize,
        cin: usize,
        cout: usize,
        kf: usize,
        kt: usize,
        stride: usize,
        kernel: WeightId,
    },
    /// `x = x * (1 + erf(x / sqrt(2))) / 2`, in place: the exact GELU of torch.
    Gelu { x: Buf, n: usize },
    /// `y[r] = x[r] / max(|x[r]|, 1e-12) * sqrt(dim) * gamma`: the RMSNorm of the model, which is `F.normalize`.
    RmsNorm {
        x: Buf,
        y: Buf,
        rows: usize,
        dim: usize,
        gamma: WeightId,
    },
    /// `y[r, o] = sum over i of weight[o, i] * x[r, i] + bias[o]`.
    Linear {
        x: Buf,
        y: Buf,
        rows: usize,
        din: usize,
        dout: usize,
        weight: WeightId,
        bias: Option<WeightId>,
    },
    /// Self-attention along one axis of a grid of `a` by `b` rows. Row `r` of `qkv` holds q, k and v, each of
    /// `heads * head_dim` values, head-major. Each token rotates its q and k by its position `p` along the axis:
    /// pair `(2i, 2i + 1)` turns by `p * freqs[i]`. Then `softmax(q k / sqrt(head_dim)) v` over the tokens of its
    /// sequence, no mask. Each head's result is multiplied by `sigmoid(gates[r, h])`, and written to `out[r]`,
    /// head-major.
    Attention {
        qkv: Buf,
        gates: Buf,
        out: Buf,
        a: usize,
        b: usize,
        along: Axis,
        heads: usize,
        head_dim: usize,
        freqs: WeightId,
    },
    /// `acc[i] += x[i]`.
    AddInto { acc: Buf, x: Buf, n: usize },
}

/// A buffer after a task, named for the row of the oracle that it holds: `frames` rows of `freqs * channels` values,
/// channels last.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mark {
    pub name: String,
    pub after: usize,
    pub buf: Buf,
    pub freqs: usize,
    pub channels: usize,
}

/// The plan of one chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub frames: usize,
    /// The length of each buffer, in f32 values.
    pub buffers: Vec<usize>,
    pub weights: Vec<Weight>,
    pub tasks: Vec<Task>,
    pub marks: Vec<Mark>,
    /// The log-mel rows of the chunk: `[frames, 128]`.
    pub input: Buf,
    /// The two logits of each frame from the head, `[frames, 2]`: the head's beat and downbeat outputs. The sum head
    /// adds them on the host (`logits`).
    pub output: Buf,
}

/// The beat and downbeat logits of the sum head from the plan's output: `beat = out[t, 0] + out[t, 1]` in f32, and
/// `downbeat = out[t, 1]`, as `SumHead` gives them.
pub fn logits(output: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (rows, rest) = output.as_chunks::<2>();
    assert!(rest.is_empty(), "the output of a plan has 2 values for each frame");
    rows.iter().map(|[b, d]| (b + d, *d)).unzip()
}

struct Builder {
    buffers: Vec<usize>,
    weights: Vec<Weight>,
    tasks: Vec<Task>,
    marks: Vec<Mark>,
}

impl Builder {
    fn buffer(&mut self) -> Buf {
        self.buffers.push(0);
        Buf(self.buffers.len() - 1)
    }

    /// Grows a buffer to hold `n` values. The buffers end at the largest use of each, so no value is spare.
    fn uses(&mut self, buf: Buf, n: usize) {
        self.buffers[buf.0] = self.buffers[buf.0].max(n);
    }

    fn weight(&mut self, prep: Prep) -> WeightId {
        self.weights.push(Weight { prep });
        WeightId(self.weights.len() - 1)
    }

    fn plain(&mut self, name: String, shape: Vec<usize>) -> WeightId {
        self.weight(Prep::Plain { name, shape })
    }

    fn task(&mut self, task: Task) {
        for (buf, n) in task.extents() {
            self.uses(buf, n);
        }
        self.tasks.push(task);
    }

    fn mark(&mut self, name: String, buf: Buf, freqs: usize, channels: usize) {
        let after = self.tasks.len() - 1;
        self.marks.push(Mark {
            name,
            after,
            buf,
            freqs,
            channels,
        });
    }

    /// A linear of `rows` rows from `din` to `dout` values, with the weight `name.weight` and maybe `name.bias`.
    fn linear(&mut self, x: Buf, y: Buf, (rows, din, dout): (usize, usize, usize), name: &str, bias: bool) {
        let weight = self.plain(format!("{name}.weight"), vec![dout, din]);
        let bias = bias.then(|| self.plain(format!("{name}.bias"), vec![dout]));
        self.task(Task::Linear {
            x,
            y,
            rows,
            din,
            dout,
            weight,
            bias,
        });
    }

    fn batch_norm(&mut self, x: Buf, y: Buf, rows: usize, channels: usize, prefix: &str) {
        let scale = self.weight(Prep::BnScale {
            prefix: prefix.to_owned(),
            channels,
        });
        let shift = self.weight(Prep::BnShift {
            prefix: prefix.to_owned(),
            channels,
        });
        self.task(Task::Affine {
            x,
            y,
            rows,
            cols: channels,
            scale,
            shift,
        });
    }
}

/// The scratch buffers of one attention and one feed-forward block.
struct Scratch {
    normed: Buf,
    qkv: Buf,
    gates: Buf,
    heads: Buf,
    proj: Buf,
    hidden: Buf,
}

/// The attention of `roformer.Attention` at `prefix`, added into `x`. The result of `to_out` stays in `proj`.
#[allow(clippy::too_many_arguments)]
fn attention(
    b: &mut Builder,
    s: &Scratch,
    x: Buf,
    (ga, gb): (usize, usize),
    along: Axis,
    dim: usize,
    hd: usize,
    prefix: &str,
) {
    let rows = ga * gb;
    let heads = dim / hd;
    let gamma = b.plain(format!("{prefix}.norm.gamma"), vec![dim]);
    b.task(Task::RmsNorm {
        x,
        y: s.normed,
        rows,
        dim,
        gamma,
    });
    b.linear(
        s.normed,
        s.qkv,
        (rows, dim, 3 * dim),
        &format!("{prefix}.to_qkv"),
        false,
    );
    b.linear(
        s.normed,
        s.gates,
        (rows, dim, heads),
        &format!("{prefix}.to_gates"),
        true,
    );
    let freqs = b.plain(format!("{prefix}.rotary_embed.freqs"), vec![hd / 2]);
    b.task(Task::Attention {
        qkv: s.qkv,
        gates: s.gates,
        out: s.heads,
        a: ga,
        b: gb,
        along,
        heads,
        head_dim: hd,
        freqs,
    });
    b.linear(s.heads, s.proj, (rows, dim, dim), &format!("{prefix}.to_out.0"), false);
    b.task(Task::AddInto {
        acc: x,
        x: s.proj,
        n: rows * dim,
    });
}

/// The feed-forward of `roformer.FeedForward` at `prefix`, added into `x`. Its output stays in `proj`.
fn feed_forward(b: &mut Builder, s: &Scratch, x: Buf, rows: usize, dim: usize, mult: usize, prefix: &str) {
    let gamma = b.plain(format!("{prefix}.net.0.gamma"), vec![dim]);
    b.task(Task::RmsNorm {
        x,
        y: s.normed,
        rows,
        dim,
        gamma,
    });
    b.linear(
        s.normed,
        s.hidden,
        (rows, dim, dim * mult),
        &format!("{prefix}.net.1"),
        true,
    );
    b.task(Task::Gelu {
        x: s.hidden,
        n: rows * dim * mult,
    });
    b.linear(
        s.hidden,
        s.proj,
        (rows, dim * mult, dim),
        &format!("{prefix}.net.4"),
        true,
    );
    b.task(Task::AddInto {
        acc: x,
        x: s.proj,
        n: rows * dim,
    });
}

/// The plan of a chunk of `frames` frames.
pub fn build(c: &Config, frames: usize) -> Plan {
    let t = frames;
    let mut b = Builder {
        buffers: Vec::new(),
        weights: Vec::new(),
        tasks: Vec::new(),
        marks: Vec::new(),
    };
    let input = b.buffer();
    let mut cur = b.buffer();
    let mut other = b.buffer();
    let s = Scratch {
        normed: b.buffer(),
        qkv: b.buffer(),
        gates: b.buffer(),
        heads: b.buffer(),
        proj: b.buffer(),
        hidden: b.buffer(),
    };
    let output = b.buffer();
    b.uses(input, t * MELS);

    // The stem: BatchNorm1d over the mel bins, a conv of 4 by 3 with stride 4 over frequency, BatchNorm2d, GELU.
    b.batch_norm(input, other, t, MELS, "frontend.stem.bn1d");
    let (mut freqs, mut dim) = (MELS / 4, c.stem_dim);
    let kernel = b.weight(Prep::ConvKernel {
        name: "frontend.stem.conv2d.weight".into(),
        cout: dim,
        cin: 1,
        kf: 4,
        kt: 3,
    });
    b.task(Task::Conv {
        x: other,
        y: cur,
        frames: t,
        f_in: MELS,
        cin: 1,
        cout: dim,
        kf: 4,
        kt: 3,
        stride: 4,
        kernel,
    });
    b.batch_norm(cur, other, t * freqs, dim, "frontend.stem.bn2d");
    std::mem::swap(&mut cur, &mut other);
    b.task(Task::Gelu {
        x: cur,
        n: t * freqs * dim,
    });
    b.mark("stem".into(), cur, freqs, dim);

    // Three frontend blocks: attention over frequency, then over frames, then a conv of 2 by 3 with stride 2 over
    // frequency that doubles the channels.
    for i in 0..3 {
        let p = format!("frontend.blocks.{i}");
        let rows = t * freqs;
        attention(
            &mut b,
            &s,
            cur,
            (t, freqs),
            Axis::B,
            dim,
            c.head_dim,
            &format!("{p}.partial.attnF"),
        );
        feed_forward(&mut b, &s, cur, rows, dim, 4, &format!("{p}.partial.ffF"));
        attention(
            &mut b,
            &s,
            cur,
            (t, freqs),
            Axis::A,
            dim,
            c.head_dim,
            &format!("{p}.partial.attnT"),
        );
        feed_forward(&mut b, &s, cur, rows, dim, 4, &format!("{p}.partial.ffT"));
        let kernel = b.weight(Prep::ConvKernel {
            name: format!("{p}.conv2d.weight"),
            cout: 2 * dim,
            cin: dim,
            kf: 2,
            kt: 3,
        });
        b.task(Task::Conv {
            x: cur,
            y: other,
            frames: t,
            f_in: freqs,
            cin: dim,
            cout: 2 * dim,
            kf: 2,
            kt: 3,
            stride: 2,
            kernel,
        });
        freqs /= 2;
        dim *= 2;
        b.batch_norm(other, cur, t * freqs, dim, &format!("{p}.norm"));
        b.task(Task::Gelu {
            x: cur,
            n: t * freqs * dim,
        });
        b.mark(format!("block{i}"), cur, freqs, dim);
    }

    // The projection to the transformer: its weight reads the channels-last rows.
    let d = c.transformer_dim;
    let weight = b.weight(Prep::PermuteColumns {
        name: "frontend.linear.weight".into(),
        rows: d,
        channels: dim,
        freqs,
    });
    let bias = Some(b.plain("frontend.linear.bias".into(), vec![d]));
    b.task(Task::Linear {
        x: cur,
        y: other,
        rows: t,
        din: freqs * dim,
        dout: d,
        weight,
        bias,
    });
    std::mem::swap(&mut cur, &mut other);
    b.mark("linear".into(), cur, 1, d);

    for l in 0..c.n_layers {
        let p = format!("transformer_blocks.layers.{l}");
        attention(&mut b, &s, cur, (t, 1), Axis::A, d, c.head_dim, &format!("{p}.0"));
        b.marks.push(Mark {
            name: format!("layer{l}.attn"),
            after: b.tasks.len() - 2,
            buf: s.proj,
            freqs: 1,
            channels: d,
        });
        feed_forward(&mut b, &s, cur, t, d, c.ff_mult, &format!("{p}.1"));
        b.marks.push(Mark {
            name: format!("layer{l}.ff"),
            after: b.tasks.len() - 2,
            buf: s.proj,
            freqs: 1,
            channels: d,
        });
    }
    let gamma = b.plain("transformer_blocks.norm.gamma".into(), vec![d]);
    b.task(Task::RmsNorm {
        x: cur,
        y: s.normed,
        rows: t,
        dim: d,
        gamma,
    });
    b.mark("final_norm".into(), s.normed, 1, d);
    b.linear(s.normed, output, (t, d, 2), "task_heads.beat_downbeat_lin", true);

    Plan {
        frames,
        buffers: b.buffers,
        weights: b.weights,
        tasks: b.tasks,
        marks: b.marks,
        input,
        output,
    }
}

impl Task {
    /// The buffers that the task reads, each with the count of values that it reads from offset 0.
    pub fn reads(&self) -> Vec<(Buf, usize)> {
        match *self {
            Task::Affine { x, rows, cols, .. } => vec![(x, rows * cols)],
            Task::Conv {
                x, frames, f_in, cin, ..
            } => vec![(x, frames * f_in * cin)],
            Task::Gelu { x, n } => vec![(x, n)],
            Task::RmsNorm { x, rows, dim, .. } => vec![(x, rows * dim)],
            Task::Linear { x, rows, din, .. } => vec![(x, rows * din)],
            Task::Attention {
                qkv,
                gates,
                a,
                b,
                heads,
                head_dim,
                ..
            } => {
                vec![(qkv, a * b * 3 * heads * head_dim), (gates, a * b * heads)]
            }
            Task::AddInto { acc, x, n } => vec![(acc, n), (x, n)],
        }
    }

    /// The buffer that the task writes, with the count of values that it writes from offset 0.
    pub fn writes(&self) -> (Buf, usize) {
        match *self {
            Task::Affine { y, rows, cols, .. } => (y, rows * cols),
            Task::Conv {
                y,
                frames,
                f_in,
                cout,
                kf,
                stride,
                ..
            } => (y, frames * ((f_in - kf) / stride + 1) * cout),
            Task::Gelu { x, n } => (x, n),
            Task::RmsNorm { y, rows, dim, .. } => (y, rows * dim),
            Task::Linear { y, rows, dout, .. } => (y, rows * dout),
            Task::Attention {
                out,
                a,
                b,
                heads,
                head_dim,
                ..
            } => (out, a * b * heads * head_dim),
            Task::AddInto { acc, n, .. } => (acc, n),
        }
    }

    /// The buffers that the task reads or writes, each with the count of values that it touches from offset 0.
    pub fn extents(&self) -> Vec<(Buf, usize)> {
        match *self {
            Task::Affine { x, y, rows, cols, .. } => vec![(x, rows * cols), (y, rows * cols)],
            Task::Conv {
                x,
                y,
                frames,
                f_in,
                cin,
                cout,
                kf,
                stride,
                ..
            } => {
                let f_out = (f_in - kf) / stride + 1;
                vec![(x, frames * f_in * cin), (y, frames * f_out * cout)]
            }
            Task::Gelu { x, n } => vec![(x, n)],
            Task::RmsNorm { x, y, rows, dim, .. } => vec![(x, rows * dim), (y, rows * dim)],
            Task::Linear {
                x, y, rows, din, dout, ..
            } => vec![(x, rows * din), (y, rows * dout)],
            Task::Attention {
                qkv,
                gates,
                out,
                a,
                b,
                heads,
                head_dim,
                ..
            } => {
                let rows = a * b;
                vec![
                    (qkv, rows * 3 * heads * head_dim),
                    (gates, rows * heads),
                    (out, rows * heads * head_dim),
                ]
            }
            Task::AddInto { acc, x, n } => vec![(acc, n), (x, n)],
        }
    }

    /// The weights that the task reads, each with the count of values that it reads.
    pub fn weights(&self) -> Vec<(WeightId, usize)> {
        match *self {
            Task::Affine { cols, scale, shift, .. } => vec![(scale, cols), (shift, cols)],
            Task::Conv {
                cin,
                cout,
                kf,
                kt,
                kernel,
                ..
            } => vec![(kernel, cout * kf * kt * cin)],
            Task::Gelu { .. } | Task::AddInto { .. } => vec![],
            Task::RmsNorm { dim, gamma, .. } => vec![(gamma, dim)],
            Task::Linear {
                din,
                dout,
                weight,
                bias,
                ..
            } => {
                let mut v = vec![(weight, dout * din)];
                v.extend(bias.map(|b| (b, dout)));
                v
            }
            Task::Attention { head_dim, freqs, .. } => vec![(freqs, head_dim / 2)],
        }
    }
}

impl Plan {
    /// Checks the plan before any interpreter runs it: each task stays inside its buffers, reads weights of the
    /// right size, and writes no buffer that it reads, but where its kind allows. Each read reads exactly the values
    /// that the last write of that buffer wrote, so no task reads a value that no task wrote, and a shape that does
    /// not join the next task fails here. A GPU interpreter relies on this check, because an index past a buffer
    /// there is a fault, and a read of an old value is silent.
    pub fn check(&self) -> Result<(), String> {
        let buf_len = |b: Buf| {
            self.buffers
                .get(b.0)
                .copied()
                .ok_or_else(|| format!("no buffer {}", b.0))
        };
        let mut written: Vec<Option<usize>> = vec![None; self.buffers.len()];
        if let Some(w) = written.get_mut(self.input.0) {
            *w = Some(self.frames * MELS);
        }
        for (i, task) in self.tasks.iter().enumerate() {
            for (b, n) in task.reads() {
                match written.get(b.0).copied().flatten() {
                    Some(w) if w == n => {}
                    Some(w) => {
                        return Err(format!(
                            "task {i} ({task:?}) reads {n} values of buffer {}, and {w} were written",
                            b.0
                        ));
                    }
                    None => return Err(format!("task {i} ({task:?}) reads buffer {}, which no task wrote", b.0)),
                }
            }
            let (wb, wn) = task.writes();
            if let Some(w) = written.get_mut(wb.0) {
                *w = Some(wn);
            }
            for (b, n) in task.extents() {
                let len = buf_len(b)?;
                if n > len {
                    return Err(format!(
                        "task {i} ({task:?}) touches {n} values of buffer {} of {len}",
                        b.0
                    ));
                }
            }
            for (w, n) in task.weights() {
                let weight = self
                    .weights
                    .get(w.0)
                    .ok_or_else(|| format!("task {i} reads no weight {}", w.0))?;
                if weight.prep.len() != n {
                    return Err(format!(
                        "task {i} reads {n} values of weight {} of {}",
                        w.0,
                        weight.prep.len()
                    ));
                }
            }
            let ext = task.extents();
            let distinct = match task {
                Task::Gelu { .. } => true,
                _ => ext
                    .iter()
                    .enumerate()
                    .all(|(j, (a, _))| ext[..j].iter().all(|(b, _)| a != b)),
            };
            if !distinct {
                return Err(format!("task {i} ({task:?}) reads and writes one buffer"));
            }
        }
        for m in &self.marks {
            if m.after >= self.tasks.len() {
                return Err(format!("mark {} is after task {}, past the last", m.name, m.after));
            }
            let n = self.frames * m.freqs * m.channels;
            if n > buf_len(m.buf)? {
                return Err(format!("mark {} reads {n} values of buffer {}", m.name, m.buf.0));
            }
        }
        if buf_len(self.input)? != self.frames * MELS || buf_len(self.output)? != self.frames * 2 {
            return Err("the input or the output buffer has the wrong size".into());
        }
        Ok(())
    }
}
