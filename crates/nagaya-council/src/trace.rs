//! The numbers of a trace (`docs/lab.md`, "The rules of `trace`"): for each step of a reply, the distribution of its
//! logit row, and for each traced layer, the norm of the residual row, its part along each vector and the logit lens.
//! The object is canonical safetensors (engine-abi D10, rules 3 and 5). No I/O.
//!
//! Each sum runs in `f64` in a fixed order, and the lens of each layer runs on one thread. Thus the same records give
//! the same bytes with any count of threads.

use std::collections::BTreeMap;

/// The top entries of each step when the request names no `top_k`.
pub const TOP_K: usize = 8;

/// The most top entries of each step.
pub const MAX_TOP_K: usize = 64;

/// The most threads of the lens of one step. Each thread takes a run of layers.
pub const MAX_THREADS: usize = 8;

/// The direction of one control vector at each layer of the model: its values over its norm, or `None` for a layer
/// that the vector does not hold or that is all zeros.
#[derive(Clone, Debug, PartialEq)]
pub struct Direction {
    /// The object id of the vector, as the client sent it.
    pub id: String,
    pub units: Vec<Option<Vec<f64>>>,
}

impl Direction {
    /// The direction of a vector with these layers, for a model of `layers` layers of `width` values. A layer of
    /// another width, or past the layer count, is an error: the caller checked the fit first.
    pub fn new(id: &str, vector: &BTreeMap<u32, Vec<f32>>, layers: usize, width: usize) -> Result<Self, String> {
        let mut units = vec![None; layers];
        for (layer, values) in vector {
            let at = *layer as usize;
            if at >= layers || values.len() != width {
                return Err(format!(
                    "the vector {id} has a layer {layer} of {} values; the model has {layers} layers of {width}",
                    values.len()
                ));
            }
            let norm = values.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
            if norm > 0.0 {
                units[at] = Some(values.iter().map(|x| *x as f64 / norm).collect());
            }
        }
        Ok(Self {
            id: id.to_owned(),
            units,
        })
    }
}

/// A lens that reads residual rows on the CPU (spec 13.2), such as a J-Lens.
pub trait Readout: Send + Sync {
    /// The logit row of each residual row, with the layer of the row.
    fn logits(&self, rows: &[(u32, &[f32])]) -> Vec<Vec<f32>>;
}

/// The lens of each traced layer.
#[derive(Clone)]
pub enum Lens {
    None,
    /// The logit lens, from the tap `lens` of the engine.
    Logit,
    /// A lens on the CPU from the tap `residual`, with its name and the content id of its object.
    Cpu {
        name: String,
        id: String,
        readout: std::sync::Arc<dyn Readout>,
    },
}

impl Lens {
    /// The name of the lens in the metadata of the object, or `""` for none.
    pub fn name(&self) -> &str {
        match self {
            Lens::None => "",
            Lens::Logit => "logit",
            Lens::Cpu { name, .. } => name,
        }
    }

    /// True when the trace has a lens.
    pub fn some(&self) -> bool {
        !matches!(self, Lens::None)
    }
}

impl std::fmt::Debug for Lens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lens::Cpu { name, id, .. } => write!(f, "Cpu({name}, {id})"),
            other => write!(f, "{}", if other.some() { "Logit" } else { "None" }),
        }
    }
}

/// Two lenses are equal when they have the same kind, name and object.
impl PartialEq for Lens {
    fn eq(&self, other: &Lens) -> bool {
        match (self, other) {
            (Lens::Cpu { name: a, id: x, .. }, Lens::Cpu { name: b, id: y, .. }) => a == b && x == y,
            (Lens::None, Lens::None) | (Lens::Logit, Lens::Logit) => true,
            _ => false,
        }
    }
}

/// What a trace records, after the route checked the request.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub model: String,
    pub weight_hash: String,
    /// The layers of the trace, in the order of the object.
    pub layers: Vec<u32>,
    /// The layers of the model: the rows of the residual tap.
    pub model_layers: usize,
    /// The width of the residual.
    pub width: usize,
    /// The width of a logit row.
    pub vocab: usize,
    pub top_k: usize,
    /// The end tokens of the model.
    pub ends: Vec<u32>,
    pub vectors: Vec<Direction>,
    /// The lens of each traced layer.
    pub lens: Lens,
    /// True when the object holds the residual rows.
    pub rows: bool,
}

/// The records of a trace, one for each step.
#[derive(Clone, Debug)]
pub struct Trace {
    plan: Plan,
    tokens: Vec<u32>,
    entropy: Vec<f32>,
    top_ids: Vec<u32>,
    top_logprobs: Vec<f32>,
    end_logprob: Vec<f32>,
    norms: Vec<f32>,
    along: Vec<f32>,
    lens_ids: Vec<u32>,
    lens_logprobs: Vec<f32>,
    rows: Vec<f32>,
}

/// The log of the sum of the exponentials of a logit row, and the entropy of its softmax, in nats. The arithmetic is
/// the arithmetic of the sampler's untempered stats, so the entropy has the bits of `h` of a token event.
pub fn distribution(row: &[f32]) -> (f64, f64) {
    let l: Vec<f64> = row.iter().map(|x| *x as f64).collect();
    let max = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: Vec<f64> = l.iter().map(|&x| x - max).collect();
    let e: Vec<f64> = z.iter().map(|&x| x.exp()).collect();
    let sum: f64 = e.iter().sum();
    let weighted: f64 = e.iter().zip(&z).map(|(a, b)| a * b).sum();
    (max + sum.ln(), sum.ln() - weighted / sum)
}

/// The `k` largest entries of a row, the largest first. A tie goes to the smaller id.
pub fn top(row: &[f32], k: usize) -> Vec<u32> {
    let by = |a: &u32, b: &u32| row[*b as usize].total_cmp(&row[*a as usize]).then(a.cmp(b));
    let mut ids: Vec<u32> = (0..row.len() as u32).collect();
    let k = k.min(ids.len());
    if k < ids.len() {
        ids.select_nth_unstable_by(k, by);
    }
    ids.truncate(k);
    ids.sort_by(by);
    ids
}

/// The top entries of a row and their log probabilities.
fn top_logprobs(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let (lse, _) = distribution(row);
    let ids = top(row, k);
    let logprobs = ids.iter().map(|id| (row[*id as usize] as f64 - lse) as f32).collect();
    (ids, logprobs)
}

/// The top entries of each row, on at most `threads` threads. Each row runs on one thread.
pub fn tops_of(rows: &[&[f32]], k: usize, threads: usize) -> Vec<(Vec<u32>, Vec<f32>)> {
    let threads = threads.clamp(1, rows.len().max(1));
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(per)
            .map(|part| scope.spawn(move || part.iter().map(|row| top_logprobs(row, k)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a thread of the lens panicked"))
            .collect()
    })
}

/// The dot product of a row and a unit vector, summed in `f64` in order.
fn along(row: &[f32], unit: &[f64]) -> f32 {
    row.iter().zip(unit).map(|(h, u)| *h as f64 * u).sum::<f64>() as f32
}

/// The Euclidean norm of a row, summed in `f64` in order.
fn norm(row: &[f32]) -> f32 {
    row.iter().map(|h| *h as f64 * *h as f64).sum::<f64>().sqrt() as f32
}

/// One tensor of the object: its name, its dtype, its shape and its bytes. Each dtype of a trace has 4 bytes. The
/// bytes are absent when only the size counts.
struct Tensor {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Option<Vec<u8>>,
}

impl Tensor {
    fn len(&self) -> usize {
        self.shape.iter().product::<usize>() * 4
    }
}

fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The alignment of the data start (engine-abi D10, rule 5).
const ALIGN: usize = 4096;

impl Trace {
    /// An empty trace of a plan.
    pub fn new(plan: Plan) -> Self {
        Self {
            plan,
            tokens: Vec::new(),
            entropy: Vec::new(),
            top_ids: Vec::new(),
            top_logprobs: Vec::new(),
            end_logprob: Vec::new(),
            norms: Vec::new(),
            along: Vec::new(),
            lens_ids: Vec::new(),
            lens_logprobs: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// The plan of the trace.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The steps that the trace holds.
    pub fn steps(&self) -> usize {
        self.tokens.len()
    }

    /// Record one step: the sampled token, its logit row, the residual tap of that row (`model_layers` rows of
    /// `width` values) and, with the lens, the tap `lens` (`model_layers` logit rows). A record of another size is an
    /// error, and the trace does not change.
    pub fn record(&mut self, token: u32, logits: &[f32], residual: &[f32], lens: Option<&[f32]>) -> Result<(), String> {
        let p = &self.plan;
        if logits.len() != p.vocab {
            return Err(format!(
                "a logit row has {} values; the trace reads {}",
                logits.len(),
                p.vocab
            ));
        }
        if residual.len() != p.model_layers * p.width {
            return Err(format!(
                "the residual tap has {} values; {} layers of {} are {}",
                residual.len(),
                p.model_layers,
                p.width,
                p.model_layers * p.width
            ));
        }
        match (&p.lens, lens) {
            (Lens::Logit, Some(lens)) if lens.len() == p.model_layers * p.vocab => {}
            (Lens::Logit, Some(lens)) => {
                return Err(format!(
                    "the tap lens has {} values; {} layers of {} are {}",
                    lens.len(),
                    p.model_layers,
                    p.vocab,
                    p.model_layers * p.vocab
                ));
            }
            (Lens::Logit, None) => return Err("the trace has the logit lens, and the step has no tap lens".to_owned()),
            (Lens::None, Some(_)) => return Err("the trace has no lens, and the step has the tap lens".to_owned()),
            (Lens::Cpu { name, .. }, Some(_)) => {
                return Err(format!(
                    "the trace has the lens {name} on the CPU, and the step has the tap lens"
                ));
            }
            (Lens::None | Lens::Cpu { .. }, None) => {}
        }
        let (lse, entropy) = distribution(logits);
        let ids = top(logits, p.top_k);
        let row = |l: u32| &residual[l as usize * p.width..(l as usize + 1) * p.width];
        self.tokens.push(token);
        self.entropy.push(entropy as f32);
        self.top_logprobs
            .extend(ids.iter().map(|id| (logits[*id as usize] as f64 - lse) as f32));
        self.top_ids.extend(ids);
        self.end_logprob
            .extend(p.ends.iter().map(|id| (logits[*id as usize] as f64 - lse) as f32));
        for l in &p.layers {
            let h = row(*l);
            self.norms.push(norm(h));
            for v in &p.vectors {
                self.along
                    .push(v.units[*l as usize].as_deref().map_or(f32::NAN, |unit| along(h, unit)));
            }
            if p.rows {
                self.rows.extend_from_slice(h);
            }
        }
        let cpu_rows;
        let rows: Option<Vec<&[f32]>> = match (&p.lens, lens) {
            (Lens::Logit, Some(lens)) => Some(
                p.layers
                    .iter()
                    .map(|l| &lens[*l as usize * p.vocab..(*l as usize + 1) * p.vocab])
                    .collect(),
            ),
            (Lens::Cpu { readout, .. }, None) => {
                let input: Vec<(u32, &[f32])> = p.layers.iter().map(|l| (*l, row(*l))).collect();
                cpu_rows = readout.logits(&input);
                if let Some(bad) = cpu_rows.iter().find(|r| r.len() != p.vocab) {
                    return Err(format!(
                        "the readout gave a logit row of {} values; the trace reads {}",
                        bad.len(),
                        p.vocab
                    ));
                }
                Some(cpu_rows.iter().map(Vec::as_slice).collect())
            }
            _ => None,
        };
        if let Some(rows) = rows {
            let threads = std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(MAX_THREADS);
            for (ids, logprobs) in tops_of(&rows, p.top_k, threads) {
                self.lens_ids.extend(ids);
                self.lens_logprobs.extend(logprobs);
            }
        }
        Ok(())
    }

    /// The tensors of a trace of `steps` steps of this plan, with these bytes, in bytewise order of their names.
    fn tensors(plan: &Plan, steps: usize, trace: Option<&Trace>) -> Vec<Tensor> {
        let (s, l, k, e, v, w) = (
            steps,
            plan.layers.len(),
            plan.top_k.min(plan.vocab),
            plan.ends.len(),
            plan.vectors.len(),
            plan.width,
        );
        let u = |values: &dyn Fn(&Trace) -> Vec<u8>| trace.map(values);
        let mut out = vec![
            Tensor {
                name: "end_ids",
                dtype: "U32",
                shape: vec![e],
                bytes: Some(u32s(&plan.ends)),
            },
            Tensor {
                name: "end_logprob",
                dtype: "F32",
                shape: vec![s, e],
                bytes: u(&|t| f32s(&t.end_logprob)),
            },
            Tensor {
                name: "entropy",
                dtype: "F32",
                shape: vec![s],
                bytes: u(&|t| f32s(&t.entropy)),
            },
            Tensor {
                name: "layers",
                dtype: "U32",
                shape: vec![l],
                bytes: Some(u32s(&plan.layers)),
            },
            Tensor {
                name: "norms",
                dtype: "F32",
                shape: vec![s, l],
                bytes: u(&|t| f32s(&t.norms)),
            },
            Tensor {
                name: "tokens",
                dtype: "U32",
                shape: vec![s],
                bytes: u(&|t| u32s(&t.tokens)),
            },
            Tensor {
                name: "top_ids",
                dtype: "U32",
                shape: vec![s, k],
                bytes: u(&|t| u32s(&t.top_ids)),
            },
            Tensor {
                name: "top_logprobs",
                dtype: "F32",
                shape: vec![s, k],
                bytes: u(&|t| f32s(&t.top_logprobs)),
            },
        ];
        if v > 0 {
            out.push(Tensor {
                name: "along",
                dtype: "F32",
                shape: vec![s, l, v],
                bytes: u(&|t| f32s(&t.along)),
            });
        }
        if plan.lens.some() {
            out.push(Tensor {
                name: "lens_ids",
                dtype: "U32",
                shape: vec![s, l, k],
                bytes: u(&|t| u32s(&t.lens_ids)),
            });
            out.push(Tensor {
                name: "lens_logprobs",
                dtype: "F32",
                shape: vec![s, l, k],
                bytes: u(&|t| f32s(&t.lens_logprobs)),
            });
        }
        if plan.rows {
            out.push(Tensor {
                name: "rows",
                dtype: "F32",
                shape: vec![s, l, w],
                bytes: u(&|t| f32s(&t.rows)),
            });
        }
        out.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
        out
    }

    /// The header of the canonical file: the JSON with sorted keys and no whitespace, padded with spaces so that the
    /// data starts on a 4 KiB boundary, after its 8-byte length.
    fn header(plan: &Plan, tensors: &[Tensor]) -> Vec<u8> {
        let quote = |text: &str| serde_json::to_string(text).expect("a string is JSON");
        let vectors: Vec<&str> = plan.vectors.iter().map(|v| v.id.as_str()).collect();
        // The keys are in byte order. Only a lens with an object names its id, so the other traces keep their bytes.
        let mut metadata = vec![("lens", plan.lens.name().to_owned())];
        if let Lens::Cpu { id, .. } = &plan.lens {
            metadata.push(("lens_id", id.clone()));
        }
        metadata.extend([
            ("model", plan.model.clone()),
            ("vectors", vectors.join(",")),
            ("weight_hash", plan.weight_hash.clone()),
        ]);
        let mut header = String::from("{\"__metadata__\":{");
        for (i, (key, value)) in metadata.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!("{}:{}", quote(key), quote(value)));
        }
        header.push('}');
        let mut at = 0usize;
        for t in tensors {
            let end = at + t.len();
            let shape: Vec<String> = t.shape.iter().map(|n| n.to_string()).collect();
            header.push_str(&format!(
                ",{}:{{\"data_offsets\":[{at},{end}],\"dtype\":{},\"shape\":[{}]}}",
                quote(t.name),
                quote(t.dtype),
                shape.join(",")
            ));
            at = end;
        }
        header.push('}');
        let padding = (ALIGN - (8 + header.len()) % ALIGN) % ALIGN;
        let header_len = header.len() + padding;
        let mut out = Vec::with_capacity(8 + header_len);
        out.extend_from_slice(&(header_len as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(8 + header_len, b' ');
        out
    }

    /// The bytes of the object.
    pub fn to_bytes(&self) -> Vec<u8> {
        let tensors = Self::tensors(&self.plan, self.steps(), Some(self));
        let mut out = Self::header(&self.plan, &tensors);
        for t in &tensors {
            let bytes = t.bytes.as_deref().expect("the tensors of a trace have their bytes");
            assert_eq!(bytes.len(), t.len(), "the tensor {} has the bytes of its shape", t.name);
            out.extend_from_slice(bytes);
        }
        out
    }

    /// The size in bytes of the object of a trace of `steps` steps of `plan`. The route checks it for `max_tokens`
    /// steps before the job, and the size grows with the steps.
    pub fn object_bytes(plan: &Plan, steps: usize) -> u64 {
        let tensors = Self::tensors(plan, steps, None);
        (Self::header(plan, &tensors).len() + tensors.iter().map(Tensor::len).sum::<usize>()) as u64
    }
}

#[cfg(test)]
mod tests;
