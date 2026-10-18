//! Canonical safetensors, content ids, recipes, entries, lookup, and eviction.

use std::collections::BTreeMap;

use nagaya_abi::{CoverageSpan, Dtype, Engine, PartKind, State};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde_json::{Map, Value};

/// The alignment of the data section.
const ALIGN: usize = 4096;

/// One stored part as a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// The part name.
    pub name: String,
    /// The part kind.
    pub kind: PartKind,
    /// The element type.
    pub dtype: Dtype,
    /// The tensor shape.
    pub shape: Vec<u64>,
    /// The tensor bytes.
    pub data: Vec<u8>,
    /// The positions that a cells or ring part holds.
    pub coverage: Option<(u64, u64)>,
    /// The block format, when the dtype is U8 for a block format.
    pub block_format: Option<String>,
}

/// One snapshot as a set of tensors and a parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// The stored tensors.
    pub tensors: Vec<Tensor>,
    /// The parent content id for a delta.
    pub parent: Option<ContentId>,
}

/// The BLAKE3 hash of a canonical file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContentId([u8; 32]);

/// The BLAKE3 recipe hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecipeId([u8; 32]);

/// A store error.
#[derive(Debug)]
pub enum StoreError {
    /// The input is not valid.
    Invalid(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Invalid(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl ContentId {
    /// Read a content id from 64 lowercase hex characters.
    pub fn from_hex(text: &str) -> Result<Self, StoreError> {
        if text.len() != 64
            || !text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(StoreError::Invalid("content id hex".to_owned()));
        }
        let mut bytes = [0u8; 32];
        for (index, pair) in text.as_bytes().chunks(2).enumerate() {
            let pair = std::str::from_utf8(pair).map_err(|_| StoreError::Invalid("content id hex".to_owned()))?;
            bytes[index] =
                u8::from_str_radix(pair, 16).map_err(|_| StoreError::Invalid("content id hex".to_owned()))?;
        }
        Ok(ContentId(bytes))
    }

    /// The content id in hex.
    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl RecipeId {
    /// The recipe id in hex.
    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl Snapshot {
    /// Write the canonical file.
    pub fn encode(&self) -> Result<Vec<u8>, StoreError> {
        let mut tensors: Vec<&Tensor> = self.tensors.iter().collect();
        tensors.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
        for pair in tensors.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(StoreError::Invalid("duplicate part name".to_owned()));
            }
        }

        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        if let Some(parent) = &self.parent {
            metadata.insert("nagaya.parent".to_owned(), parent.hex());
        }

        let mut header: BTreeMap<String, Value> = BTreeMap::new();
        let mut offset = 0u64;
        for tensor in &tensors {
            check_tensor(tensor)?;
            metadata.insert(
                format!("nagaya.part.{}.kind", tensor.name),
                kind_name(tensor.kind)?.to_owned(),
            );
            if let Some((start, end)) = tensor.coverage {
                metadata.insert(
                    format!("nagaya.part.{}.coverage", tensor.name),
                    format!("{start}..{end}"),
                );
            }
            if let Some(block) = &tensor.block_format {
                metadata.insert(format!("nagaya.part.{}.block", tensor.name), block.clone());
            }

            let end = offset + tensor.data.len() as u64;
            let mut entry: BTreeMap<String, Value> = BTreeMap::new();
            entry.insert("dtype".to_owned(), Value::String(dtype_name(tensor.dtype).to_owned()));
            entry.insert(
                "shape".to_owned(),
                Value::Array(tensor.shape.iter().map(|&value| Value::from(value)).collect()),
            );
            entry.insert(
                "data_offsets".to_owned(),
                Value::Array(vec![Value::from(offset), Value::from(end)]),
            );
            header.insert(
                tensor.name.clone(),
                Value::Object(entry.into_iter().collect::<Map<String, Value>>()),
            );
            offset = end;
        }
        header.insert(
            "__metadata__".to_owned(),
            Value::Object(
                metadata
                    .into_iter()
                    .map(|(key, value)| (key, Value::String(value)))
                    .collect::<Map<String, Value>>(),
            ),
        );

        let json = serde_json::to_vec(&header).map_err(|error| StoreError::Invalid(error.to_string()))?;
        let padding = (ALIGN - ((8 + json.len()) % ALIGN)) % ALIGN;
        let header_len = json.len() + padding;
        let mut out = Vec::with_capacity(8 + header_len + offset as usize);
        out.extend_from_slice(&(header_len as u64).to_le_bytes());
        out.extend_from_slice(&json);
        out.resize(8 + header_len, b' ');
        for tensor in &tensors {
            out.extend_from_slice(&tensor.data);
        }
        Ok(out)
    }

    /// Read a canonical file.
    pub fn decode(bytes: &[u8]) -> Result<Self, StoreError> {
        let (header_len, head) =
            SafeTensors::read_metadata(bytes).map_err(|error| StoreError::Invalid(error.to_string()))?;
        let data = bytes
            .get(8 + header_len..)
            .ok_or_else(|| StoreError::Invalid("short canonical file".to_owned()))?;
        let metadata = head.metadata().clone().unwrap_or_default();
        let parent = match metadata.get("nagaya.parent") {
            Some(text) => Some(ContentId::from_hex(text)?),
            None => None,
        };

        let mut tensors = Vec::new();
        for name in head.offset_keys() {
            let info = head
                .info(&name)
                .ok_or_else(|| StoreError::Invalid("missing tensor info".to_owned()))?;
            let kind_text = metadata
                .get(&format!("nagaya.part.{name}.kind"))
                .ok_or_else(|| StoreError::Invalid("missing part kind".to_owned()))?;
            let kind = kind_from_name(kind_text)?;
            let coverage = match metadata.get(&format!("nagaya.part.{name}.coverage")) {
                Some(text) => Some(parse_span(text)?),
                None => None,
            };
            let block_format = metadata.get(&format!("nagaya.part.{name}.block")).cloned();
            let dtype = dtype_from_safetensors(info.dtype)?;
            let shape: Vec<u64> = info.shape.iter().map(|&value| value as u64).collect();
            let tensor_data = data
                .get(info.data_offsets.0..info.data_offsets.1)
                .ok_or_else(|| StoreError::Invalid("tensor out of range".to_owned()))?
                .to_vec();
            let tensor = Tensor {
                name,
                kind,
                dtype,
                shape,
                data: tensor_data,
                coverage,
                block_format,
            };
            check_tensor(&tensor)?;
            tensors.push(tensor);
        }
        tensors.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
        let snapshot = Snapshot { tensors, parent };
        if snapshot.encode()? != bytes {
            return Err(StoreError::Invalid("non-canonical file".to_owned()));
        }
        Ok(snapshot)
    }

    /// The content id of this snapshot.
    pub fn content_id(&self) -> Result<ContentId, StoreError> {
        Ok(content_id(&self.encode()?))
    }
}

/// The content id of canonical file bytes.
pub fn content_id(bytes: &[u8]) -> ContentId {
    ContentId(*blake3::hash(bytes).as_bytes())
}

/// Make a snapshot from one engine state. `rows` gives the start and count of each row part.
pub fn snapshot_from_engine(
    engine: &mut dyn Engine,
    state: State,
    rows: &[(usize, u64, u64)],
) -> Result<Snapshot, StoreError> {
    let limits = engine.limits();
    let State::Slot(slot_id) = state;
    let slot_bytes = limits
        .slots
        .get(slot_id as usize)
        .ok_or_else(|| StoreError::Invalid("unknown slot".to_owned()))?
        .bytes;
    if slot_bytes == 0 {
        return Err(StoreError::Invalid("empty slot".to_owned()));
    }

    let mut tensors = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        match part.kind {
            PartKind::Derived => continue,
            PartKind::Fixed => {
                let length = part.shape.iter().product::<u64>() as usize * dtype_bytes(part.dtype);
                let mut data = vec![0u8; length];
                engine
                    .parts()
                    .ok_or_else(|| StoreError::Invalid("no parts capability".to_owned()))?
                    .part_read(state, id, &mut data)
                    .map_err(|error| StoreError::Invalid(format!("part read: {error:?}")))?;
                tensors.push(Tensor {
                    name: part.name.clone(),
                    kind: part.kind,
                    dtype: part.dtype,
                    shape: part.shape.clone(),
                    data,
                    coverage: None,
                    block_format: None,
                });
            }
            PartKind::Cells | PartKind::Ring => {
                let (_, start, count) = *rows
                    .iter()
                    .find(|(part_id, _, _)| *part_id == id)
                    .ok_or_else(|| StoreError::Invalid("missing rows".to_owned()))?;
                let row_shape = part.shape.clone();
                let row_bytes = row_shape.iter().product::<u64>() as usize * dtype_bytes(part.dtype);
                let mut data = vec![0u8; count as usize * row_bytes];
                engine
                    .parts()
                    .ok_or_else(|| StoreError::Invalid("no parts capability".to_owned()))?
                    .cells_read(state, id, start, count, &mut data)
                    .map_err(|error| StoreError::Invalid(format!("cells read: {error:?}")))?;
                let mut shape = vec![count];
                shape.extend(row_shape);
                let span = CoverageSpan {
                    part: id,
                    start,
                    end: start + count,
                };
                engine
                    .coverage()
                    .ok_or_else(|| StoreError::Invalid("no coverage capability".to_owned()))?
                    .fits(&span, slot_id)
                    .map_err(StoreError::Invalid)?;
                tensors.push(Tensor {
                    name: part.name.clone(),
                    kind: part.kind,
                    dtype: part.dtype,
                    shape,
                    data,
                    coverage: Some((start, start + count)),
                    block_format: None,
                });
            }
        }
    }
    Ok(Snapshot { tensors, parent: None })
}

/// The recipe of a root boundary.
pub fn recipe_root(
    identity_digest: &[u8; 32],
    geometry_key: &[u8],
    approximations: &[Vec<u8>],
    lab_state: &[u8],
) -> RecipeId {
    let mut hasher = blake3::Hasher::new_derive_key("nagaya recipe v1");
    add_item(&mut hasher, 1, identity_digest);
    add_item(&mut hasher, 2, geometry_key);
    let mut tags = Vec::new();
    tags.extend_from_slice(&(approximations.len() as u64).to_le_bytes());
    for tag in approximations {
        tags.extend_from_slice(&(tag.len() as u64).to_le_bytes());
        tags.extend_from_slice(tag);
    }
    add_item(&mut hasher, 3, &tags);
    add_item(&mut hasher, 4, lab_state);
    RecipeId(*hasher.finalize().as_bytes())
}

/// The recipe at a later boundary.
pub fn recipe_boundary(previous: &RecipeId, tokens: &[u32], position: u64) -> RecipeId {
    let mut hasher = blake3::Hasher::new_derive_key("nagaya recipe v1");
    add_item(&mut hasher, 5, &previous.0);
    let mut token_bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        token_bytes.extend_from_slice(&token.to_le_bytes());
    }
    add_item(&mut hasher, 6, &token_bytes);
    add_item(&mut hasher, 7, &position.to_le_bytes());
    RecipeId(*hasher.finalize().as_bytes())
}

fn add_item(hasher: &mut blake3::Hasher, tag: u16, bytes: &[u8]) {
    hasher.update(&tag.to_le_bytes());
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn check_tensor(tensor: &Tensor) -> Result<(), StoreError> {
    if tensor.name == "__metadata__" {
        return Err(StoreError::Invalid("reserved tensor name".to_owned()));
    }
    if tensor.kind == PartKind::Derived {
        return Err(StoreError::Invalid("derived part is never stored".to_owned()));
    }
    let elements: u64 = tensor.shape.iter().product();
    let expected = elements as usize * dtype_bytes(tensor.dtype);
    if tensor.data.len() != expected {
        return Err(StoreError::Invalid(format!("wrong data length for {}", tensor.name)));
    }
    match tensor.kind {
        PartKind::Fixed if tensor.coverage.is_some() => Err(StoreError::Invalid("fixed part has coverage".to_owned())),
        PartKind::Cells | PartKind::Ring if tensor.coverage.is_none() => {
            Err(StoreError::Invalid("row part has no coverage".to_owned()))
        }
        _ => Ok(()),
    }
}

fn kind_name(kind: PartKind) -> Result<&'static str, StoreError> {
    match kind {
        PartKind::Fixed => Ok("fixed"),
        PartKind::Cells => Ok("cells"),
        PartKind::Ring => Ok("ring"),
        PartKind::Derived => Err(StoreError::Invalid("derived part is never stored".to_owned())),
    }
}

fn kind_from_name(name: &str) -> Result<PartKind, StoreError> {
    match name {
        "fixed" => Ok(PartKind::Fixed),
        "cells" => Ok(PartKind::Cells),
        "ring" => Ok(PartKind::Ring),
        _ => Err(StoreError::Invalid(format!("unknown part kind {name}"))),
    }
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::U8 => "U8",
        Dtype::U32 => "U32",
    }
}

fn dtype_bytes(dtype: Dtype) -> usize {
    match dtype {
        Dtype::U8 => 1,
        Dtype::F16 | Dtype::BF16 => 2,
        Dtype::F32 | Dtype::U32 => 4,
    }
}

fn dtype_from_safetensors(dtype: SafeDtype) -> Result<Dtype, StoreError> {
    match dtype {
        SafeDtype::F32 => Ok(Dtype::F32),
        SafeDtype::F16 => Ok(Dtype::F16),
        SafeDtype::BF16 => Ok(Dtype::BF16),
        SafeDtype::U8 => Ok(Dtype::U8),
        SafeDtype::U32 => Ok(Dtype::U32),
        other => Err(StoreError::Invalid(format!("unsupported dtype {other:?}"))),
    }
}

fn parse_span(text: &str) -> Result<(u64, u64), StoreError> {
    let (start, end) = text
        .split_once("..")
        .ok_or_else(|| StoreError::Invalid("bad coverage string".to_owned()))?;
    let start = parse_canonical_u64(start, "coverage start")?;
    let end = parse_canonical_u64(end, "coverage end")?;
    if start > end {
        return Err(StoreError::Invalid("coverage start after end".to_owned()));
    }
    Ok((start, end))
}

fn parse_canonical_u64(text: &str, what: &str) -> Result<u64, StoreError> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(StoreError::Invalid(format!("bad {what}")));
    }
    if text.len() > 1 && text.starts_with('0') {
        return Err(StoreError::Invalid(format!("bad {what}")));
    }
    text.parse::<u64>()
        .map_err(|_| StoreError::Invalid(format!("bad {what}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_id_hex_is_canonical() {
        let lower = "ab".repeat(32);
        assert!(ContentId::from_hex(&lower).is_ok());
        assert!(ContentId::from_hex(&lower.to_uppercase()).is_err());
        let plus = format!("+{}", &lower[1..]);
        assert!(ContentId::from_hex(&plus).is_err());
    }

    #[test]
    fn coverage_span_is_canonical() {
        assert_eq!(parse_span("0..2").unwrap(), (0, 2));
        assert!(parse_span("+0..02").is_err());
        assert!(parse_span("00..2").is_err());
        assert!(parse_span("0..02").is_err());
        assert!(parse_span("2..0").is_err());
    }
}
