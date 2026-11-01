//! The conditionings of the image engines in the store (spec 6.16): the encoder output of a prompt, as an entry with
//! a recipe, so that a hit skips the encoder.
//!
//! The recipe is a root recipe with the encoder digest and the fixed key `nagaya conditioning v1` as tag 2, then one
//! boundary at the end of the template ids. The snapshot is one cells part `conditioning` of `[rows, width]` in bf16,
//! whose coverage is the positions from the cut to the last id.

use nagaya_abi::{Dtype, PartKind};

use crate::{RecipeId, Snapshot, StoreError, Tensor, recipe_boundary, recipe_root};

/// The fixed key of tag 2 of the root recipe.
pub const KEY: &[u8] = b"nagaya conditioning v1";

/// The name of the one part of a conditioning snapshot.
pub const PART: &str = "conditioning";

/// The 32 bytes of an encoder digest: `blake3:` and 64 lowercase hex digits.
pub fn digest_bytes(text: &str) -> Result<[u8; 32], StoreError> {
    let bad = || {
        StoreError::Invalid(format!(
            "the encoder digest `{text}` is not `blake3:` and 64 hex digits"
        ))
    };
    let hex = text.strip_prefix("blake3:").ok_or_else(bad)?;
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(bad());
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| bad())?;
    }
    Ok(out)
}

/// The recipe of the conditioning of `ids` under the encoder `digest`.
pub fn recipe(digest: &[u8; 32], ids: &[u32]) -> RecipeId {
    let root = recipe_root(digest, KEY, &[], b"");
    recipe_boundary(&root, ids, ids.len() as u64)
}

/// The canonical snapshot of a conditioning: `rows` bytes of rows of `row_bytes`, for the positions `cut..ids`.
pub fn snapshot(rows: &[u8], row_bytes: usize, cut: u64, ids: u64) -> Result<Snapshot, StoreError> {
    let count = ids.saturating_sub(cut);
    if row_bytes == 0 || !row_bytes.is_multiple_of(2) || count == 0 || rows.len() as u64 != count * row_bytes as u64 {
        return Err(StoreError::Invalid(format!(
            "{} bytes are not {count} rows of {row_bytes} bytes for the positions {cut}..{ids}",
            rows.len()
        )));
    }
    Ok(Snapshot {
        tensors: vec![Tensor {
            name: PART.to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::BF16,
            shape: vec![count, (row_bytes / 2) as u64],
            data: rows.to_vec(),
            coverage: Some((cut, ids)),
            block_format: None,
        }],
        parent: None,
    })
}

/// The rows of a conditioning snapshot, after a check of its form: one cells part `conditioning` in bf16 of `rows`
/// rows of `row_bytes`, and no parent.
pub fn rows(snapshot: &Snapshot, row_bytes: usize, rows: usize) -> Result<Vec<u8>, StoreError> {
    let [tensor] = snapshot.tensors.as_slice() else {
        return Err(StoreError::Invalid(format!(
            "a conditioning has one part, and this snapshot has {}",
            snapshot.tensors.len()
        )));
    };
    let shape = [rows as u64, (row_bytes / 2) as u64];
    if snapshot.parent.is_some()
        || tensor.name != PART
        || tensor.kind != PartKind::Cells
        || tensor.dtype != Dtype::BF16
        || tensor.shape != shape
        || tensor.data.len() != rows * row_bytes
    {
        return Err(StoreError::Invalid(format!(
            "the snapshot is not a conditioning of {rows} rows of {row_bytes} bytes: part `{}`, shape {:?}",
            tensor.name, tensor.shape
        )));
    }
    Ok(tensor.data.clone())
}
