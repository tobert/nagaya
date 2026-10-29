//! The LoRA files of an image request (spec 11.9a). A file is parsed on the host: each module keeps its down and up
//! matrices in f32, and its scale. The engine adds each delta to the output of its module, and the base weights stay as
//! they are.
//!
//! The keys are the kohya names (`lora_unet_`, `lora_te1_`, `lora_te2_`) and the diffusers names (`unet.`,
//! `text_encoder.`, `text_encoder_2.`). Each key must name a module of the engine, or the file is an error.

use std::collections::BTreeMap;

use crate::weights::{f16_to_f32, f32_values};

/// The part of the engine that a module belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Namespace {
    Unet,
    TextEncoder,
    TextEncoder2,
}

/// A module of the engine that a LoRA can change: its diffusers name without `.weight`, and its shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub namespace: Namespace,
    pub name: String,
    /// The inputs and outputs of the module.
    pub cin: usize,
    pub cout: usize,
    /// 1 for a linear layer and a 1 x 1 convolution, 3 for a 3 x 3 convolution.
    pub kernel: usize,
}

/// The LoRA of one module: `scale * up(down(x))` is added to the module's output.
#[derive(Clone, Debug, PartialEq)]
pub struct Module {
    pub target: Target,
    pub rank: usize,
    /// `weight * alpha / rank`, or `weight` when the file has no alpha.
    pub scale: f32,
    /// The down matrix, `rank` rows of `cin * kernel * kernel` values.
    pub down: Vec<f32>,
    /// The up matrix, `cout` rows of `rank` values.
    pub up: Vec<f32>,
}

/// A LoRA file of one request, with its weight. Its modules are in the order of their targets.
#[derive(Clone, Debug, PartialEq)]
pub struct Lora {
    pub modules: Vec<Module>,
}

/// The matrices of one module as the keys give them.
#[derive(Default)]
struct Parts {
    /// The first key of the module, for the messages.
    key: String,
    down: Option<(Vec<usize>, Vec<f32>)>,
    up: Option<(Vec<usize>, Vec<f32>)>,
    alpha: Option<f32>,
}

/// A tensor as f32 values, with its shape.
type Values = (Vec<usize>, Vec<f32>);

/// The tensors of a safetensors file, as f32 values with their shapes. F32, F16 and BF16 are read.
fn tensors(bytes: &[u8]) -> Result<BTreeMap<String, Values>, String> {
    let len = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")) as usize)
        .ok_or_else(|| "the LoRA file is shorter than its header length".to_owned())?;
    let header_end = 8usize
        .checked_add(len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| "the LoRA file ends inside its header".to_owned())?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])
        .map_err(|e| format!("the LoRA file has a header that is not JSON: {e}"))?;
    let data = &bytes[header_end..];
    let entries = header
        .as_object()
        .ok_or_else(|| "the LoRA file has a header that is not an object".to_owned())?;
    let mut out: BTreeMap<String, Values> = BTreeMap::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry["dtype"].as_str().unwrap_or_default();
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or_else(|| format!("the tensor {name} has no shape"))?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("the tensor {name} has a shape that is not a list of sizes"))?;
        let (start, end) = match (entry["data_offsets"][0].as_u64(), entry["data_offsets"][1].as_u64()) {
            (Some(s), Some(e)) if s <= e && (e as usize) <= data.len() => (s as usize, e as usize),
            _ => return Err(format!("the tensor {name} has offsets outside the file")),
        };
        let raw = &data[start..end];
        let values: Vec<f32> = match dtype {
            "F32" => f32_values(raw),
            "F16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                .collect(),
            "BF16" => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
                .collect(),
            other => {
                return Err(format!(
                    "the tensor {name} has the dtype {other}; a LoRA needs F32, F16 or BF16"
                ));
            }
        };
        let count: usize = shape.iter().product();
        if count != values.len() {
            return Err(format!(
                "the tensor {name} has the shape {shape:?} and {} values",
                values.len()
            ));
        }
        out.insert(name.clone(), (shape, values));
    }
    Ok(out)
}

/// The module name of a key, with each dot as an underscore. The kohya names have underscores only, so both forms
/// meet here.
fn joined(name: &str) -> String {
    name.replace('.', "_")
}

/// The namespace and the module part of a key, or an error that names the key.
fn split(key: &str) -> Result<(Namespace, &str), String> {
    let kohya = [
        ("lora_unet_", Namespace::Unet),
        ("lora_te1_", Namespace::TextEncoder),
        ("lora_te2_", Namespace::TextEncoder2),
    ];
    let diffusers = [
        ("unet.", Namespace::Unet),
        ("text_encoder_2.", Namespace::TextEncoder2),
        ("text_encoder.", Namespace::TextEncoder),
    ];
    for (prefix, namespace) in kohya.iter().chain(&diffusers) {
        if let Some(rest) = key.strip_prefix(prefix) {
            return Ok((*namespace, rest));
        }
    }
    Err(format!(
        "the LoRA key {key} names no module of SDXL base; it starts with `lora_unet_`, `lora_te1_`, `lora_te2_`, \
         `unet.`, `text_encoder.` or `text_encoder_2.`"
    ))
}

/// The role of a key's tensor, from its suffix: down, up or alpha. The suffix gives the module part too.
fn role(key: &str) -> Option<(&str, Role)> {
    const SUFFIXES: [(&str, Role); 8] = [
        (".lora_down.weight", Role::Down),
        (".lora_up.weight", Role::Up),
        (".lora_A.weight", Role::Down),
        (".lora_B.weight", Role::Up),
        (".lora_linear_layer.down.weight", Role::Down),
        (".lora_linear_layer.up.weight", Role::Up),
        (".alpha", Role::Alpha),
        (".lora_alpha", Role::Alpha),
    ];
    SUFFIXES
        .iter()
        .find_map(|(suffix, role)| key.strip_suffix(suffix).map(|module| (module, *role)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Down,
    Up,
    Alpha,
}

/// The LoRA of a file, with its weight, for the modules of `targets`. A key that names no target, or a matrix that
/// does not fit its module, is an error that names the key.
pub fn parse(bytes: &[u8], weight: f32, targets: &[Target]) -> Result<Lora, String> {
    let mut groups: BTreeMap<(Namespace, String), Parts> = BTreeMap::new();
    for (key, (shape, values)) in tensors(bytes)? {
        let (namespace, rest) = split(&key)?;
        let (module, role) =
            role(rest).ok_or_else(|| format!("the LoRA key {key} is not a down, an up or an alpha of a module"))?;
        let parts = groups.entry((namespace, joined(module))).or_default();
        if parts.key.is_empty() {
            parts.key = key.clone();
        }
        match role {
            Role::Down => parts.down = Some((shape, values)),
            Role::Up => parts.up = Some((shape, values)),
            Role::Alpha => {
                if values.len() != 1 {
                    return Err(format!(
                        "the LoRA key {key} is an alpha with {} values; it takes one",
                        values.len()
                    ));
                }
                parts.alpha = Some(values[0]);
            }
        }
    }
    let mut modules = Vec::new();
    for ((namespace, module), parts) in groups {
        let target = targets
            .iter()
            .find(|t| t.namespace == namespace && joined(&t.name) == module)
            .ok_or_else(|| format!("the LoRA key {} names no module of SDXL base", parts.key))?;
        let (down_shape, down) = parts
            .down
            .ok_or_else(|| format!("the LoRA key {} has no down matrix", parts.key))?;
        let (up_shape, up) = parts
            .up
            .ok_or_else(|| format!("the LoRA key {} has no up matrix", parts.key))?;
        let rank = *down_shape.first().unwrap_or(&0);
        let fits_down = rank > 0
            && down_shape.get(1) == Some(&target.cin)
            && down.len() == rank * target.cin * target.kernel * target.kernel;
        let fits_up =
            up_shape.first() == Some(&target.cout) && up_shape.get(1) == Some(&rank) && up.len() == target.cout * rank;
        if !fits_down || !fits_up {
            return Err(format!(
                "the LoRA key {} has the down shape {down_shape:?} and the up shape {up_shape:?}; the module takes \
                 {} inputs and gives {} outputs",
                parts.key, target.cin, target.cout
            ));
        }
        let alpha = parts.alpha.unwrap_or(rank as f32);
        modules.push(Module {
            target: target.clone(),
            rank,
            scale: weight * alpha / rank as f32,
            down,
            up,
        });
    }
    Ok(Lora { modules })
}

/// The delta of one module for `rows` rows of `x`, of `cin` values each: `scale * up(down(x))`, one row of `cout` values
/// for each row of `x`. The engine runs the same sums on the card; this is the reference, for the host tests.
pub fn delta(module: &Module, x: &[f32], rows: usize) -> Vec<f32> {
    assert_eq!(module.target.kernel, 1, "the reference delta is for a linear layer");
    let (cin, cout, rank) = (module.target.cin, module.target.cout, module.rank);
    assert_eq!(x.len(), rows * cin, "x has {} values for {rows} rows of {cin}", x.len());
    let mut out = Vec::with_capacity(rows * cout);
    for row in x.chunks_exact(cin) {
        let hidden: Vec<f32> = (0..rank)
            .map(|k| {
                let down = &module.down[k * cin..(k + 1) * cin];
                row.iter().zip(down).map(|(x, d)| x * d).sum()
            })
            .collect();
        for o in 0..cout {
            let up = &module.up[o * rank..(o + 1) * rank];
            let y: f32 = hidden.iter().zip(up).map(|(h, u)| h * u).sum();
            out.push(module.scale * y);
        }
    }
    out
}
