//! Control vectors with no I/O (spec 11.12): the recipe bytes of a control list, and the vector of the means of two
//! prompt sets.

use std::collections::BTreeMap;

use nagaya_abi::ControlOp;
use nagaya_store::ContentId;

/// One item of the control list of a context: the vector object, the operation and the strength that the client sent.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlEntry {
    /// The content id of the vector object.
    pub vector: ContentId,
    pub op: ControlOp,
    pub strength: f64,
    /// The layers that the item names, each with the number that the client sent, in rising layer order. `None`
    /// when the item names none (spec 11.12, "The strength of each layer").
    pub layers: Option<Vec<(u32, f64)>>,
}

/// The byte of an operation in the recipe: 1 for `add`, 2 for `ablate` and 3 for `clamp` (spec 11.12).
pub fn op_byte(op: ControlOp) -> u8 {
    match op {
        ControlOp::Add => 1,
        ControlOp::Ablate => 2,
        ControlOp::Clamp => 3,
    }
}

/// The lab state of a control list in the root recipe (spec 11.12, tag 4), little-endian: a `u64` count, then for each
/// item the 32 bytes of the vector's content id, the operation byte and the `f64` strength. An empty list has no bytes,
/// so the root of a plain context does not change. An item with layers sets the bit `0x80` of its operation byte, and
/// a `u32` count and each `u32` layer with its `f64` number follow its strength. An item without layers keeps its
/// bytes.
pub fn recipe_bytes(list: &[ControlEntry]) -> Vec<u8> {
    if list.is_empty() {
        return Vec::new();
    }
    let mut bytes = Vec::with_capacity(8 + list.len() * (32 + 1 + 8));
    bytes.extend_from_slice(&(list.len() as u64).to_le_bytes());
    for item in list {
        bytes.extend_from_slice(item.vector.as_bytes());
        let layered = if item.layers.is_some() { 0x80 } else { 0 };
        bytes.push(op_byte(item.op) | layered);
        bytes.extend_from_slice(&item.strength.to_le_bytes());
        if let Some(layers) = &item.layers {
            bytes.extend_from_slice(&(layers.len() as u32).to_le_bytes());
            for (layer, number) in layers {
                bytes.extend_from_slice(&layer.to_le_bytes());
                bytes.extend_from_slice(&number.to_le_bytes());
            }
        }
    }
    bytes
}

/// The layers of a composed control vector (`docs/lab.md`, "compose"). Each `add` is a direction for each of its
/// layers with a weight; each `remove` is a direction for each of its layers. At each layer, the sum of the weighted
/// adds comes first, in list order. Then each remove takes the part of the sum along its unit direction out, in list
/// order. The sums run in `f64`. A layer whose result has norm 0 is left out, and so is a remove of a zero direction.
pub fn compose(
    adds: &[(&BTreeMap<u32, Vec<f32>>, f64)],
    removes: &[&BTreeMap<u32, Vec<f32>>],
) -> BTreeMap<u32, Vec<f32>> {
    let mut sums: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (layers, weight) in adds {
        for (layer, row) in layers.iter() {
            let sum = sums.entry(*layer).or_insert_with(|| vec![0.0; row.len()]);
            assert_eq!(sum.len(), row.len(), "the rows of a composed vector have one width");
            for (s, x) in sum.iter_mut().zip(row) {
                *s += weight * f64::from(*x);
            }
        }
    }
    for layers in removes {
        for (layer, row) in layers.iter() {
            let Some(sum) = sums.get_mut(layer) else { continue };
            // The part along x is ((s · x) / (x · x)) x, with no square root, so an exact case stays exact.
            let square = row.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>();
            if square == 0.0 {
                continue;
            }
            let along = sum.iter().zip(row).map(|(s, x)| s * f64::from(*x)).sum::<f64>() / square;
            for (s, x) in sum.iter_mut().zip(row) {
                *s -= along * f64::from(*x);
            }
        }
    }
    sums.into_iter()
        .map(|(layer, sum)| (layer, sum.into_iter().map(|s| s as f32).collect::<Vec<f32>>()))
        .filter(|(_, row)| row.iter().any(|x| *x != 0.0))
        .collect()
}

/// The control vector of two prompt sets (spec 11.12): the mean of the `positive` rows less the mean of the `negative`
/// rows, as `f32`. Each row is the residual after each layer at the last prompt position, `[layers][width]`. The sums
/// are `f64` in the order of the rows, so the same rows give the same bits on each run (engine-abi D3).
pub fn vector_of_means(positive: &[&[f32]], negative: &[&[f32]]) -> Result<Vec<f32>, String> {
    let width = positive.first().map(|row| row.len()).unwrap_or(0);
    let mean = |set: &[&[f32]], name: &str| -> Result<Vec<f64>, String> {
        if set.is_empty() {
            return Err(format!("the {name} set has no row; each set needs one prompt or more"));
        }
        let mut total = vec![0f64; width];
        for (i, row) in set.iter().enumerate() {
            if row.len() != width {
                return Err(format!(
                    "{name} row {i} has {} values, and the first positive row has {width}",
                    row.len()
                ));
            }
            if let Some(at) = row.iter().position(|v| !v.is_finite()) {
                return Err(format!("value {at} of {name} row {i} is not finite"));
            }
            for (t, v) in total.iter_mut().zip(row.iter()) {
                *t += f64::from(*v);
            }
        }
        let count = set.len() as f64;
        Ok(total.into_iter().map(|t| t / count).collect())
    };
    let positive = mean(positive, "positive")?;
    let negative = mean(negative, "negative")?;
    Ok(positive.iter().zip(&negative).map(|(p, n)| (p - n) as f32).collect())
}
