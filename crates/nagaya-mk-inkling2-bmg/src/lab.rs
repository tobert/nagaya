//! The head of a direction on the CPU, for the lab routes `project` and `trace` (`docs/lab.md`, "The rules of
//! `project`", "Inkling 2"). No card runs.
//!
//! The head is the first 200,058 rows of `model.llm.unembed.weight`, and the final norm is `model.llm.norm.weight`. A
//! score of a direction `v` is the row of the head times `g v`, element by element, with no division and no softcap,
//! because a direction has no scale. The file stays mapped, and the process holds no copy of the head. A score sums
//! its products in `f64` in a fixed order, so a request gives the same bits on each run, with any count of threads.

use std::path::Path;

use nagaya_mk_bmg::safetensors::{Dtype, MappedSafeTensors, bf16_to_f32};

use crate::host::ops::HIDDEN;
use crate::host::weights::{LAYERS, LOGITS};
use crate::spec::{Model, WEIGHTS_FILE};

/// The most threads of one product. Each thread takes a run of rows of the head.
pub const MAX_THREADS: usize = 8;

/// The tensor of the head and of the final norm.
pub const HEAD: &str = "model.llm.unembed.weight";
pub const NORM: &str = "model.llm.norm.weight";

/// The facts of a model folder that need no weights: the layers, the width of the residual and the end tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Facts {
    pub model: Model,
    pub layers: u32,
    pub width: usize,
    /// The end tokens: `eos_token_id` of `generation_config.json`.
    pub ends: Vec<u32>,
}

impl Facts {
    /// Read `config.json` and `generation_config.json` of the folder `dir`. A folder that is not Inkling 2 is an error
    /// that names the file or the key.
    pub fn open(dir: &Path) -> Result<Facts, String> {
        let model = Model::read(dir)?;
        let path = dir.join("generation_config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let generation: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{} is not JSON: {e}", path.display()))?;
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
                "{} has no eos_token_id of ids; the model folder names its end tokens there",
                path.display()
            ));
        }
        if let Some(id) = ends.iter().find(|id| **id as usize >= LOGITS) {
            return Err(format!(
                "{} names the end token {id}, past the {LOGITS} rows of the head",
                path.display()
            ));
        }
        Ok(Facts {
            model,
            layers: LAYERS as u32,
            width: HIDDEN,
            ends,
        })
    }
}

/// The head of one Inkling 2 checkpoint: its mapped weights, the weight of the final norm and the facts of the folder.
pub struct Head {
    file: MappedSafeTensors,
    facts: Facts,
    norm: Vec<f32>,
}

impl Head {
    /// Map the weights of the folder `dir`, and read its facts. A folder whose head or norm has another dtype or shape
    /// is an error that names the tensor.
    pub fn open(dir: &Path) -> Result<Head, String> {
        let facts = Facts::open(dir)?;
        let file = MappedSafeTensors::open(&dir.join(WEIGHTS_FILE))?;
        let view = file.view()?;
        for (name, rows) in [(HEAD, Some(LOGITS)), (NORM, None)] {
            let meta = view
                .tensors
                .get(name)
                .ok_or_else(|| format!("the checkpoint {} has no tensor {name}", dir.display()))?;
            let fits = match rows {
                Some(rows) => meta.shape.len() == 2 && meta.shape[0] >= rows && meta.shape[1] == HIDDEN,
                None => meta.shape == [HIDDEN],
            };
            if meta.dtype != Dtype::BF16 || !fits {
                return Err(format!(
                    "tensor {name} is {:?} {:?}; the lab reads BF16 with the width {HIDDEN}",
                    meta.dtype, meta.shape
                ));
            }
        }
        let norm = view
            .tensor_bytes(NORM)
            .expect("the view found the tensor")
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16_to_f32(u16::from_le_bytes(*b)))
            .collect();
        drop(view);
        Ok(Head { file, facts, norm })
    }

    /// The facts of the folder.
    pub fn facts(&self) -> &Facts {
        &self.facts
    }

    /// The rows of the head: a projection has this many scores.
    pub fn rows(&self) -> usize {
        LOGITS
    }

    /// The width of the residual.
    pub fn width(&self) -> usize {
        HIDDEN
    }

    /// The end tokens of the model.
    pub fn ends(&self) -> &[u32] {
        &self.facts.ends
    }

    /// The scores of each direction in `vectors`: one score for each row of the head. A direction has `HIDDEN` values;
    /// another length is a defect of the caller, and it panics.
    pub fn scores(&self, vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        for v in vectors {
            assert_eq!(v.len(), HIDDEN, "a direction has the width of the residual");
        }
        let units: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| v.iter().zip(&self.norm).map(|(x, g)| x * g).collect())
            .collect();
        let view = self.file.view().expect("open parsed the header");
        let table = view.tensor_bytes(HEAD).expect("open found the head");
        let rows = LOGITS;
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, MAX_THREADS);
        let per = rows.div_ceil(threads);
        let parts: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let range = (t * per).min(rows)..((t + 1) * per).min(rows);
                    let units = &units;
                    scope.spawn(move || {
                        let mut row = vec![0f32; HIDDEN];
                        let mut part = vec![Vec::with_capacity(range.len()); units.len()];
                        for r in range {
                            let bytes = &table[r * HIDDEN * 2..(r + 1) * HIDDEN * 2];
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
        (0..vectors.len())
            .map(|k| parts.iter().flat_map(|part| part[k].iter().copied()).collect())
            .collect()
    }
}

/// The dot product of a row and a direction. A product of a BF16 value and an `f32` value is exact in `f64`. The sum
/// runs in 8 lanes in `f64`, and the lanes add in a fixed tree, so the bits do not depend on the thread.
pub fn dot(row: &[f32], u: &[f32]) -> f32 {
    let mut lanes = [0f64; 8];
    let (rows, row_tail) = row.as_chunks::<8>();
    let (us, u_tail) = u.as_chunks::<8>();
    for (r, v) in rows.iter().zip(us) {
        for i in 0..8 {
            lanes[i] += f64::from(r[i]) * f64::from(v[i]);
        }
    }
    for (i, (r, v)) in row_tail.iter().zip(u_tail).enumerate() {
        lanes[i] += f64::from(*r) * f64::from(*v);
    }
    let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    sum as f32
}
