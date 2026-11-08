//! The unembedding of a direction on the CPU, for the lab route `project` (`docs/lab.md`, "`project`"). No card runs.
//!
//! The head of LFM3 is the embedding table (`tie_word_embeddings`). The unembedding of a direction `v` multiplies `v`
//! by the weight of the final norm, element by element, and then by each row of the table. It does not divide by the
//! root mean square of `v`, because a direction has no scale. `Lfm3RMSNorm` multiplies by its weight, not by 1 plus
//! its weight (`modeling_lfm3.py:66` of LFM3-1.7B); the oracle stage `project` checks it.
//!
//! The file stays mapped, and the process holds no copy of the table. A score sums its products in `f64` in a fixed
//! order, so a request gives the same bits on each run, with any count of threads.

use std::path::Path;

use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors, SafeTensorsView, bf16_to_f32, f16_to_f32};

use crate::spec::{Model, WEIGHTS_FILE};

/// The most threads of one product. Each thread takes a run of rows of the table.
pub const MAX_THREADS: usize = 8;

/// The unembedding of one LFM3 checkpoint: its mapped weights and the facts of the model folder.
pub struct Unembedding {
    file: MappedSafeTensors,
    model: Model,
    /// The rows of the table, which is also the head.
    rows: usize,
    /// The width of the residual.
    width: usize,
    /// The layers of the backbone.
    layers: u32,
    /// The end tokens: `eos_token_id` of `generation_config.json`.
    ends: Vec<u32>,
    /// The weight of the final norm, as `f32`.
    norm: Vec<f32>,
    /// The epsilon of the final norm: `norm_eps` of `config.json`.
    eps: f64,
}

/// One entry of a projection: a row of the head and its score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub id: u32,
    pub score: f32,
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} is not JSON: {e}", path.display()))
}

/// The facts of an LFM3 model folder that need no weights, for the lab route `trace`: the model, the layers, the
/// width of the residual and the end tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Facts {
    pub model: Model,
    pub layers: u32,
    pub width: usize,
    /// The end tokens: `eos_token_id` of `generation_config.json`.
    pub ends: Vec<u32>,
}

impl Facts {
    /// Read `config.json` and `generation_config.json` of the model folder `dir`. A folder that is not an LFM3 model
    /// folder is an error that names the file or the key.
    pub fn open(dir: &Path) -> Result<Facts, String> {
        let config = read_json(&dir.join("config.json"))?;
        let (model, text) = match config["architectures"][0].as_str() {
            Some("Lfm3ForCausalLM") => (Model::Text, &config),
            Some("Lfm3OmniForConditionalGeneration") => (Model::Omni, &config["text_config"]),
            other => {
                return Err(format!(
                    "config.json of {} has architectures {other:?}; project reads Lfm3ForCausalLM and \
                     Lfm3OmniForConditionalGeneration",
                    dir.display()
                ));
            }
        };
        let layers = text["num_hidden_layers"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("config.json of {} has no count num_hidden_layers", dir.display()))?;
        let width = text["hidden_size"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("config.json of {} has no count hidden_size", dir.display()))?;
        let generation = read_json(&dir.join("generation_config.json"))?;
        let ends: Vec<u32> = match &generation["eos_token_id"] {
            serde_json::Value::Number(n) => n.as_u64().into_iter().collect(),
            serde_json::Value::Array(list) => list.iter().filter_map(|n| n.as_u64()).collect(),
            _ => Vec::new(),
        }
        .into_iter()
        .filter_map(|n| u32::try_from(n).ok())
        .collect();
        if ends.is_empty() {
            return Err(format!(
                "generation_config.json of {} has no eos_token_id of ids; the model folder names its end tokens there",
                dir.display()
            ));
        }
        Ok(Facts {
            model,
            layers,
            width,
            ends,
        })
    }
}

impl Unembedding {
    /// Map the weights of the model folder `dir`, and read its `config.json` and `generation_config.json`. A folder
    /// that is not an LFM3 checkpoint is an error that names the file or the tensor.
    pub fn open(dir: &Path) -> Result<Unembedding, String> {
        let Facts {
            model,
            layers,
            width: config_width,
            ends,
        } = Facts::open(dir)?;
        let file = MappedSafeTensors::open(&dir.join(WEIGHTS_FILE))?;
        let view = file.view()?;
        let prefix = model.prefix();
        let tensor = |name: &str| {
            let full = format!("{prefix}{name}");
            let meta = view
                .tensors
                .get(&full)
                .ok_or_else(|| format!("the checkpoint {} has no tensor {full}", dir.display()))?;
            if meta.dtype != Dtype::BF16 {
                return Err(format!("tensor {full} is {:?}; project reads BF16", meta.dtype));
            }
            Ok((full, meta.shape.clone()))
        };
        let (embed, embed_shape) = tensor("embed_tokens.weight")?;
        let (norm_name, norm_shape) = tensor("embedding_norm.weight")?;
        let [rows, width] = embed_shape[..] else {
            return Err(format!(
                "tensor {embed} has the shape {embed_shape:?}; it is [rows, width]"
            ));
        };
        if width != config_width {
            return Err(format!(
                "tensor {embed} has the width {width}, and config.json of {} has hidden_size {config_width}",
                dir.display()
            ));
        }
        if norm_shape != [width] || width == 0 || rows == 0 {
            return Err(format!(
                "tensor {norm_name} has the shape {norm_shape:?}, and {embed} has {embed_shape:?}; the norm has the width \
                 of the table"
            ));
        }
        if let Some(id) = ends.iter().find(|id| **id as usize >= rows) {
            return Err(format!(
                "generation_config.json of {} names the end token {id}, past the {rows} rows of the head",
                dir.display()
            ));
        }
        let norm = view
            .tensor_bytes(&norm_name)
            .expect("the view found the tensor")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
            .collect();
        drop(view);
        let config = read_json(&dir.join("config.json"))?;
        let text = match model {
            Model::Text => &config,
            Model::Omni => &config["text_config"],
        };
        let eps = text["norm_eps"]
            .as_f64()
            .filter(|eps| *eps > 0.0)
            .ok_or_else(|| format!("config.json of {} has no positive number norm_eps", dir.display()))?;
        Ok(Unembedding {
            file,
            model,
            rows,
            width,
            layers,
            ends,
            norm,
            eps,
        })
    }

    /// The model of the checkpoint.
    pub fn model(&self) -> Model {
        self.model
    }

    /// The rows of the head: a projection has this many scores.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The width of the residual.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The layers of the backbone.
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// The end tokens of the model, from `generation_config.json`.
    pub fn ends(&self) -> &[u32] {
        &self.ends
    }

    /// The logit rows of residual rows, as the final norm and the head give them: each row divides by its root mean
    /// square with the epsilon, and then takes the scores of a direction. The lenses of `trace` read through it.
    pub fn readout(&self, rows: &[&[f32]]) -> Vec<Vec<f32>> {
        let scaled: Vec<Vec<f32>> = rows
            .iter()
            .map(|h| {
                let mean = h.iter().map(|x| *x as f64 * *x as f64).sum::<f64>() / h.len() as f64;
                let inverse = 1.0 / (mean + self.eps).sqrt();
                h.iter().map(|x| (*x as f64 * inverse) as f32).collect()
            })
            .collect();
        let refs: Vec<&[f32]> = scaled.iter().map(Vec::as_slice).collect();
        self.scores(&refs)
    }

    /// The scores of each direction in `vectors`: one score for each row of the head. Each direction has `width`
    /// values; another length is a defect of the caller, and it panics.
    pub fn scores(&self, vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        for v in vectors {
            assert_eq!(v.len(), self.width, "a direction has the width of the residual");
        }
        // The weight of the final norm, element by element, with no division (docs/lab.md).
        let units: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| v.iter().zip(&self.norm).map(|(x, g)| x * g).collect())
            .collect();
        let view = self.file.view().expect("open parsed the header");
        let table = view
            .tensor_bytes(&format!("{}embed_tokens.weight", self.model.prefix()))
            .expect("open found the table");
        let mut out = vec![vec![0f32; self.rows]; vectors.len()];
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, MAX_THREADS);
        let per = self.rows.div_ceil(threads);
        let width = self.width;
        let parts: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let range = (t * per).min(self.rows)..((t + 1) * per).min(self.rows);
                    let units = &units;
                    scope.spawn(move || {
                        let mut row = vec![0f32; width];
                        let mut part = vec![Vec::with_capacity(range.len()); units.len()];
                        for r in range {
                            let bytes = &table[r * width * 2..(r + 1) * width * 2];
                            for (x, b) in row.iter_mut().zip(bytes.as_chunks::<2>().0) {
                                *x = bf16_to_f32(u16::from_le_bytes(*b));
                            }
                            for (u, scores) in units.iter().zip(part.iter_mut()) {
                                scores.push(dot(&row, u));
                            }
                        }
                        part
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a thread of the product panicked"))
                .collect()
        });
        for (k, scores) in out.iter_mut().enumerate() {
            let mut at = 0;
            for part in &parts {
                scores[at..at + part[k].len()].copy_from_slice(&part[k]);
                at += part[k].len();
            }
            assert_eq!(at, self.rows, "the parts cover the rows");
        }
        out
    }
}

/// The dot product of a row and a direction. A product of a BF16 value and an `f32` value is exact in `f64`. The sum
/// runs in 8 lanes in `f64`, and the lanes add in a fixed tree, so the bits do not depend on the thread.
fn dot(row: &[f32], u: &[f32]) -> f32 {
    let mut lanes = [0f64; 8];
    let (rows, row_tail) = row.as_chunks::<8>();
    let (us, u_tail) = u.as_chunks::<8>();
    for (r, v) in rows.iter().zip(us) {
        for i in 0..8 {
            lanes[i] += r[i] as f64 * v[i] as f64;
        }
    }
    for (i, (r, v)) in row_tail.iter().zip(u_tail).enumerate() {
        lanes[i] += *r as f64 * *v as f64;
    }
    let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    sum as f32
}

/// The `k` largest and the `k` smallest entries of `scores`, each from its end. A tie goes to the smaller id.
pub fn ends_of(scores: &[f32], k: usize) -> (Vec<Entry>, Vec<Entry>) {
    let mut ids: Vec<u32> = (0..scores.len() as u32).collect();
    let by_score = |a: &u32, b: &u32| scores[*b as usize].total_cmp(&scores[*a as usize]).then(a.cmp(b));
    let k = k.min(ids.len());
    let entry = |id: u32| Entry {
        id,
        score: scores[id as usize],
    };
    let mut up = ids.clone();
    if k < up.len() {
        up.select_nth_unstable_by(k, by_score);
    }
    up.truncate(k);
    up.sort_by(by_score);
    let by_low = |a: &u32, b: &u32| scores[*a as usize].total_cmp(&scores[*b as usize]).then(a.cmp(b));
    if k < ids.len() {
        ids.select_nth_unstable_by(k, by_low);
    }
    ids.truncate(k);
    ids.sort_by(by_low);
    (
        up.into_iter().map(entry).collect(),
        ids.into_iter().map(entry).collect(),
    )
}

/// The 0-based rank of `id` from the top: the count of scores above its score, plus the count of equal scores of
/// smaller ids. It is the place of `id` in the order of [`ends_of`].
pub fn rank(scores: &[f32], id: u32) -> usize {
    let own = scores[id as usize];
    scores
        .iter()
        .enumerate()
        .filter(|(i, s)| s.total_cmp(&own).is_gt() || (s.total_cmp(&own).is_eq() && (*i as u32) < id))
        .count()
}

/// The value of `nagaya.lens` in the metadata of a J-Lens.
pub const JLENS: &str = "jlens";

/// Why a lens file does not read.
#[derive(Clone, Debug, PartialEq)]
pub enum LensError {
    /// The file cannot be read.
    Unread(String),
    /// The lens was fitted for other weights.
    OtherWeights(String),
    /// The file is not a lens of this model.
    Malformed(String),
}

impl std::fmt::Display for LensError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LensError::Unread(why) | LensError::OtherWeights(why) | LensError::Malformed(why) => f.write_str(why),
        }
    }
}

/// A J-Lens of one model (`docs/lab.md`, "The rules of the J-Lens"): the matrix `J_l` of each layer before the last,
/// in `f32`. The lens of the last layer is the identity.
pub struct JLens {
    id: String,
    layers: u32,
    width: usize,
    /// `J_l` for `l` from 0 to `layers - 2`, each `width` rows of `width` values. Row `i` gives entry `i` of `J_l h`.
    matrices: Vec<Vec<f32>>,
}

impl JLens {
    /// Read the lens file `path` of a model with the weights `weight_hash`, `layers` layers and a residual of `width`.
    /// A file that does not fit is an error that names the reason.
    pub fn read(path: &Path, weight_hash: &str, layers: u32, width: usize) -> Result<JLens, LensError> {
        let bytes = std::fs::read(path)
            .map_err(|e| LensError::Unread(format!("cannot read the lens {}: {e}", path.display())))?;
        let at = |why: String| LensError::Malformed(format!("the lens {}: {why}", path.display()));
        let view = SafeTensorsView::parse(&bytes).map_err(at)?;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..view.data_offset])
            .map_err(|e| at(format!("the header is not JSON: {e}")))?;
        let meta = &header["__metadata__"];
        let kind = meta["nagaya.lens"].as_str().unwrap_or("");
        if kind != JLENS {
            return Err(at(format!("its metadata names the lens {kind:?}, not {JLENS:?}")));
        }
        let hash = meta["weight_hash"].as_str().unwrap_or("");
        if hash != weight_hash {
            return Err(LensError::OtherWeights(format!(
                "the lens {}: {}",
                path.display(),
                format_args!(
                    "it was fitted for weight_hash {hash}, and the served model has weight_hash {weight_hash}; fit a \
                     lens for these weights"
                )
            )));
        }
        let held = layers.saturating_sub(1);
        let mut names: Vec<&String> = view.tensors.keys().collect();
        names.sort();
        if let Some(name) = names
            .iter()
            .find(|name| !(0..held).any(|l| **name == &format!("layers.{l}")))
        {
            return Err(at(format!(
                "it holds the tensor {name}; a lens of {layers} layers holds layers.0 to layers.{} only, because the \
                 lens of the last layer is the identity",
                held.saturating_sub(1)
            )));
        }
        let mut matrices = Vec::with_capacity(held as usize);
        for l in 0..held {
            let name = format!("layers.{l}");
            let meta = view
                .tensors
                .get(&name)
                .ok_or_else(|| at(format!("it holds no tensor {name}")))?;
            if meta.dtype != Dtype::F16 {
                return Err(at(format!("tensor {name} is {:?}; a lens holds F16", meta.dtype)));
            }
            if meta.shape != [width, width] {
                return Err(at(format!(
                    "tensor {name} has the shape {:?}, and the residual has the width {width}; it is [{width}, {width}]",
                    meta.shape
                )));
            }
            let data = view.tensor_bytes(&name).expect("the view found the tensor");
            matrices.push(
                data.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                    .collect(),
            );
        }
        Ok(JLens {
            id: format!("blake3:{}", blake3::hash(&bytes).to_hex()),
            layers,
            width,
            matrices,
        })
    }

    /// The content id of the lens file.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// `J_l h` for the residual row `h` of layer `layer`. The sums run in `f64` in a fixed order. A layer past the
    /// model or a row of another width is a defect of the caller, and it panics.
    pub fn transport(&self, layer: u32, h: &[f32]) -> Vec<f32> {
        assert!(
            layer < self.layers,
            "the layer {layer} is in the model of {} layers",
            self.layers
        );
        assert_eq!(h.len(), self.width, "a row has the width of the residual");
        let Some(matrix) = self.matrices.get(layer as usize) else {
            return h.to_vec();
        };
        matrix.chunks_exact(self.width).map(|row| dot(row, h)).collect()
    }
}
