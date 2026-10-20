//! Canonical safetensors, content ids, recipes, entries, lookup, and eviction.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nagaya_abi::{
    BitDependence, ChainLimits, CoverageSpan, Determinism, Dtype, Engine, PartKind, SlotLimit, State, WriteResult,
};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The alignment of the data section.
const ALIGN: usize = 4096;

/// The longest pin lease, one day in milliseconds.
const MAX_PIN_MS: u64 = 86_400_000;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentId([u8; 32]);

/// The BLAKE3 recipe hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RecipeId([u8; 32]);

/// A store error.
#[derive(Debug)]
pub enum StoreError {
    /// The input is not valid.
    Invalid(String),
    /// The file system failed.
    Io(String),
    /// Two content ids under one recipe have the same coverage.
    Tripwire {
        /// The refused recipe.
        recipe: RecipeId,
        /// The quarantined content ids.
        ids: Vec<ContentId>,
    },
    /// The entry is quarantined.
    Quarantined(ContentId),
    /// The entry does not fit the slot.
    Fit(String),
    /// Two chains are incomparable under the deepest rule. The fields are the tips of the two chains.
    Ambiguous {
        /// The tip of the first chain.
        left: ContentId,
        /// The tip of the second chain.
        right: ContentId,
    },
    /// The budget cannot hold the new entry.
    NoSpace,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Invalid(message) => write!(formatter, "{message}"),
            StoreError::Io(message) => write!(formatter, "{message}"),
            StoreError::Tripwire { recipe, ids } => write!(
                formatter,
                "the recipe {} has two content ids with the same coverage: {}",
                recipe.hex(),
                ids.iter().map(ContentId::hex).collect::<Vec<String>>().join(", ")
            ),
            StoreError::Quarantined(id) => write!(formatter, "the entry {} is quarantined", id.hex()),
            StoreError::Fit(message) => write!(formatter, "{message}"),
            StoreError::Ambiguous { left, right } => write!(
                formatter,
                "the chains with the tips {} and {} are incomparable: neither covers more in every part",
                left.hex(),
                right.hex()
            ),
            StoreError::NoSpace => write!(formatter, "the budget cannot hold the entry"),
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

impl Serialize for ContentId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.hex())
    }
}

impl<'de> Deserialize<'de> for ContentId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        ContentId::from_hex(&text).map_err(serde::de::Error::custom)
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

            let end = offset
                .checked_add(tensor.data.len() as u64)
                .ok_or_else(|| StoreError::Invalid("data offset overflow".to_owned()))?;
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
        let data_len = usize::try_from(offset).map_err(|_| StoreError::Invalid("data too large".to_owned()))?;
        let capacity = 8usize
            .checked_add(header_len)
            .and_then(|value| value.checked_add(data_len))
            .ok_or_else(|| StoreError::Invalid("file too large".to_owned()))?;
        let mut out = Vec::with_capacity(capacity);
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
    let slot_bytes =
        slot_capacity(&limits.slots, slot_id).ok_or_else(|| StoreError::Invalid("unknown slot".to_owned()))?;
    if slot_bytes == 0 {
        return Err(StoreError::Invalid("empty slot".to_owned()));
    }

    let mut tensors = Vec::new();
    for (id, part) in limits.parts.iter().enumerate() {
        match part.kind {
            PartKind::Derived => continue,
            PartKind::Fixed => {
                let length = checked_len(&part.shape, part.dtype)?;
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
                    block_format: part.block_format.clone(),
                });
            }
            PartKind::Cells | PartKind::Ring => {
                let (_, start, count) = *rows
                    .iter()
                    .find(|(part_id, _, _)| *part_id == id)
                    .ok_or_else(|| StoreError::Invalid("missing rows".to_owned()))?;
                let row_shape = part.shape.clone();
                let row_bytes = checked_len(&row_shape, part.dtype)?;
                let count =
                    usize::try_from(count).map_err(|_| StoreError::Invalid("row count too large".to_owned()))?;
                let length = count
                    .checked_mul(row_bytes)
                    .ok_or_else(|| StoreError::Invalid("row bytes overflow".to_owned()))?;
                let mut data = vec![0u8; length];
                engine
                    .parts()
                    .ok_or_else(|| StoreError::Invalid("no parts capability".to_owned()))?
                    .cells_read(state, id, start, count as u64, &mut data)
                    .map_err(|error| StoreError::Invalid(format!("cells read: {error:?}")))?;
                let span = match engine.coverage() {
                    Some(coverage) => {
                        let span = coverage.coverage(state, id, &data);
                        if span.start > span.end {
                            return Err(StoreError::Invalid("coverage start after end".to_owned()));
                        }
                        coverage.fits(&span, slot_id).map_err(StoreError::Invalid)?;
                        span
                    }
                    None => CoverageSpan {
                        part: id,
                        start: 0,
                        end: count as u64,
                    },
                };
                let mut shape = vec![span.end - span.start];
                shape.extend(row_shape);
                tensors.push(Tensor {
                    name: part.name.clone(),
                    kind: part.kind,
                    dtype: part.dtype,
                    shape,
                    data,
                    coverage: Some((span.start, span.end)),
                    block_format: part.block_format.clone(),
                });
            }
        }
    }
    for tensor in &tensors {
        check_tensor(tensor)?;
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

/// The recipe of an interval copy (spec 6.13). It adds the item "not sealed" (tag 8, no bytes) to the recipe of
/// the sealed snapshot at the same position. Thus a lookup for a sealed snapshot never finds a copy.
pub fn recipe_not_sealed(sealed: &RecipeId) -> RecipeId {
    let mut hasher = blake3::Hasher::new_derive_key("nagaya recipe v1");
    add_item(&mut hasher, 5, &sealed.0);
    add_item(&mut hasher, 8, &[]);
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
    if tensor.block_format.is_some() && tensor.dtype != Dtype::U8 {
        return Err(StoreError::Invalid("block format needs U8".to_owned()));
    }
    let expected = checked_len(&tensor.shape, tensor.dtype)?;
    if tensor.data.len() != expected {
        return Err(StoreError::Invalid(format!("wrong data length for {}", tensor.name)));
    }
    match tensor.kind {
        PartKind::Fixed if tensor.coverage.is_some() => Err(StoreError::Invalid("fixed part has coverage".to_owned())),
        PartKind::Cells | PartKind::Ring => {
            let Some((start, end)) = tensor.coverage else {
                return Err(StoreError::Invalid("row part has no coverage".to_owned()));
            };
            if start > end {
                return Err(StoreError::Invalid("coverage start after end".to_owned()));
            }
            let Some(&rows) = tensor.shape.first() else {
                return Err(StoreError::Invalid("row part has no rows dimension".to_owned()));
            };
            if rows != end - start {
                return Err(StoreError::Invalid("shape rows do not match coverage".to_owned()));
            }
            Ok(())
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
        Dtype::U16 => "U16",
        Dtype::U32 => "U32",
        Dtype::U64 => "U64",
        Dtype::I8 => "I8",
        Dtype::I16 => "I16",
        Dtype::I32 => "I32",
        Dtype::I64 => "I64",
        Dtype::F8E4M3 => "F8_E4M3",
        Dtype::F8E5M2 => "F8_E5M2",
    }
}

fn slot_capacity(slots: &[SlotLimit], slot: u32) -> Option<u64> {
    let mut remaining = slot as usize;
    for class in slots {
        if remaining < class.count as usize {
            return Some(class.bytes);
        }
        remaining -= class.count as usize;
    }
    None
}

fn checked_len(shape: &[u64], dtype: Dtype) -> Result<usize, StoreError> {
    let elements = shape
        .iter()
        .try_fold(1u64, |total, &dim| total.checked_mul(dim))
        .ok_or_else(|| StoreError::Invalid("shape overflow".to_owned()))?;
    let bytes = elements
        .checked_mul(dtype.element_bytes())
        .ok_or_else(|| StoreError::Invalid("shape overflow".to_owned()))?;
    usize::try_from(bytes).map_err(|_| StoreError::Invalid("shape too large".to_owned()))
}

fn dtype_from_safetensors(dtype: SafeDtype) -> Result<Dtype, StoreError> {
    match dtype {
        SafeDtype::F32 => Ok(Dtype::F32),
        SafeDtype::F16 => Ok(Dtype::F16),
        SafeDtype::BF16 => Ok(Dtype::BF16),
        SafeDtype::U8 => Ok(Dtype::U8),
        SafeDtype::U16 => Ok(Dtype::U16),
        SafeDtype::U32 => Ok(Dtype::U32),
        SafeDtype::U64 => Ok(Dtype::U64),
        SafeDtype::I8 => Ok(Dtype::I8),
        SafeDtype::I16 => Ok(Dtype::I16),
        SafeDtype::I32 => Ok(Dtype::I32),
        SafeDtype::I64 => Ok(Dtype::I64),
        SafeDtype::F8_E4M3 => Ok(Dtype::F8E4M3),
        SafeDtype::F8_E5M2 => Ok(Dtype::F8E5M2),
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

/// A place where the store keeps an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tier {
    /// Host memory.
    Host,
    /// NVMe.
    Nvme,
}

/// Why the store holds an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reason {
    /// An anchor.
    Anchor,
    /// An interval.
    Interval,
    /// A prefill that finished.
    PrefillDone,
    /// Eviction.
    Evict,
    /// Reclaim.
    Reclaim,
    /// Shutdown.
    Shutdown,
}

/// The strength of the last check of an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Check {
    /// Every byte was hashed.
    Full,
    /// Some bytes were hashed.
    Spot,
    /// The entry is not checked yet.
    None,
}

/// The state of an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryState {
    /// The entry is served.
    Live,
    /// The entry is quarantined.
    Quarantined,
}

/// Why the store quarantined an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quarantine {
    /// Two content ids under one recipe have the same coverage.
    Tripwire,
    /// The blob does not match its content id.
    ContentId,
    /// The blob fails the engine's identity.
    Identity,
    /// The index disagrees with the blob.
    Index,
}

/// The determinism class of the engine that made an entry. It survives a restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeterminismClass {
    /// The same identity and inputs give the same bits in any process.
    Always,
    /// The same, because the plan table fixes every free choice.
    Pinned,
    /// The same bits within one process only.
    PerProcess,
}

impl From<Determinism> for DeterminismClass {
    fn from(determinism: Determinism) -> Self {
        match determinism {
            Determinism::Always => DeterminismClass::Always,
            Determinism::Pinned => DeterminismClass::Pinned,
            Determinism::PerProcess => DeterminismClass::PerProcess,
        }
    }
}

/// A history that changes the bits of the engine that made an entry. It survives a restart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitDependenceClass {
    /// The bits depend on which state is resident.
    Residency,
}

impl From<BitDependence> for BitDependenceClass {
    fn from(dependence: BitDependence) -> Self {
        match dependence {
            BitDependence::Residency => BitDependenceClass::Residency,
        }
    }
}

/// The chain limits of the engine that made an entry. They survive a restart.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntryChain {
    /// The deepest chain.
    pub max_depth: u32,
    /// The largest delta as a fraction of the full base snapshot.
    pub delta_fraction: f32,
}

/// What a put says about its engine (spec 6.1). One store serves every engine, so each put carries these facts
/// and the entry records them. The tripwire (6.7) and the chain checks (6.5) read the recorded values.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineFacts {
    /// The engine name.
    pub name: String,
    /// The determinism class of the engine.
    pub determinism: Determinism,
    /// The history that changes the bits of the engine.
    pub bit_dependence: Vec<BitDependence>,
    /// The chain limits of the engine.
    pub chain: ChainLimits,
}

impl EngineFacts {
    /// The facts of an engine, from its name and its limits.
    pub fn of(name: &str, limits: &nagaya_abi::Limits) -> Self {
        EngineFacts {
            name: name.to_owned(),
            determinism: limits.determinism,
            bit_dependence: limits.bit_dependence.clone(),
            chain: limits.chain,
        }
    }
}

/// The positions that one part holds.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PartCoverage {
    /// The part name.
    pub part: String,
    /// The first position.
    pub start: u64,
    /// The position after the last.
    pub end: u64,
}

/// One entry in the store index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// The content id.
    pub content_id: ContentId,
    /// The recipe.
    pub recipe: RecipeId,
    /// The parent content id for a delta.
    pub parent: Option<ContentId>,
    /// The coverage per part.
    pub coverage: Vec<PartCoverage>,
    /// The tier.
    pub tier: Tier,
    /// The size in bytes.
    pub size: u64,
    /// The owner.
    pub owner: u64,
    /// The creation time in milliseconds.
    pub created_ms: u64,
    /// The last use in milliseconds.
    pub last_used_ms: u64,
    /// The number of hits.
    pub hits: u64,
    /// The reason.
    pub reason: Reason,
    /// The provenance of an approximate entry.
    pub provenance: Option<String>,
    /// The strength of the last check.
    pub check: Check,
    /// The engine identity digest.
    pub identity: [u8; 32],
    /// The entry state.
    pub state: EntryState,
    /// The name of the engine that made the snapshot.
    pub engine: String,
    /// The determinism class of the engine that made the snapshot.
    pub determinism: DeterminismClass,
    /// The history that changes the bits of that engine.
    pub bit_dependence: Vec<BitDependenceClass>,
    /// The chain limits of that engine.
    pub chain: EntryChain,
    /// The store epoch that made the entry.
    pub epoch: u64,
    /// Why the store quarantined the entry.
    pub quarantine: Option<Quarantine>,
}

/// A pin on an entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pin {
    /// The owner.
    pub owner: u64,
    /// The expiry time in milliseconds.
    pub expires_ms: u64,
}

/// The lookup tolerance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tolerance {
    /// Exact bits only.
    Exact,
    /// The approved approximations, by method name. Exact entries are also accepted.
    Approx(Vec<String>),
}

/// The facts for one put.
pub struct Put<'a> {
    /// The snapshot.
    pub snapshot: &'a Snapshot,
    /// The recipe.
    pub recipe: RecipeId,
    /// The previous recipe, for a delta. It must equal the parent entry's recipe.
    pub previous_recipe: Option<RecipeId>,
    /// The engine that made the snapshot, with its class and its chain limits.
    pub engine: EngineFacts,
    /// The engine identity digest.
    pub identity: [u8; 32],
    /// The owner.
    pub owner: u64,
    /// The tier.
    pub tier: Tier,
    /// The reason.
    pub reason: Reason,
    /// The recipe of the sealed snapshot at the same position. Set for an interval copy and for no other put.
    /// The recipe of the copy must be `recipe_not_sealed` of this one (spec 6.2 tag 8, 6.13).
    pub sealed_recipe: Option<RecipeId>,
    /// The provenance of an approximate entry.
    pub provenance: Option<String>,
    /// The current time in milliseconds.
    pub now_ms: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Index {
    entries: BTreeMap<ContentId, Entry>,
    /// The live pins of each entry: one per owner (spec 6.10).
    #[serde(deserialize_with = "read_pins")]
    pins: BTreeMap<ContentId, Vec<Pin>>,
    refused: BTreeSet<RecipeId>,
    epoch: u64,
}

/// A place where a test injects a write failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailPoint {
    /// Fail before the blob write.
    BlobWrite,
    /// Fail before the blob rename.
    BlobRename,
    /// Fail before the index write.
    IndexWrite,
    /// Fail before the index rename.
    IndexRename,
}

/// The eviction rank of an interval copy (spec 6.9 step 1). Its key is the time of last use.
const RANK_INTERVAL: u8 = 0;
/// The eviction rank of a chain interior (spec 6.9 step 3).
const RANK_INTERIOR: u8 = 1;
/// The eviction rank of every other entry: tips, roots and standalone entries (spec 6.9 step 4).
const RANK_BY_VALUE: u8 = 2;

/// The directory under the store that holds the blobs of cleared tripwires. No budget counts it.
const QUARANTINE: &str = "quarantine";

/// The content-addressed store.
#[derive(Debug)]
pub struct Store {
    dir: PathBuf,
    index: Index,
    checked: BTreeSet<ContentId>,
    /// The canonical bytes of the host-tier entries. A restart empties it (spec 6.11).
    /// `Arc<[u8]>` is the simplest holder that is sound: the bytes are immutable, a reader shares them with no copy,
    /// and the map key is the content id, so equal bytes cannot sit under two keys.
    host_blobs: BTreeMap<ContentId, Arc<[u8]>>,
    budgets: BTreeMap<Tier, u64>,
    owner_quota: u64,
    /// The most interval copies that the store keeps (spec 6.13). Zero means the store refuses them.
    interval_copies: u32,
    /// The largest serial in a quarantine name. `open` sets the maximum; a unit test sets a smaller one.
    quarantine_serials: u32,
    fail_point: Cell<Option<FailPoint>>,
}

/// The result of a restore that reached the engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Restored {
    /// Every part was written.
    Done,
    /// The engine refused the first write, so the state is as it was. `part` is the part that it refused.
    Refused {
        /// The refused part.
        part: String,
    },
    /// A write failed, or the engine refused a write after an earlier write changed the state.
    /// The state is poisoned (D6). `part` is the part that stopped the restore.
    Failed {
        /// The part that stopped the restore.
        part: String,
    },
}

impl Store {
    /// Open the store in `dir`, or make an empty one, with no budget in any tier. For tests only (spec 4.1a).
    #[cfg(any(test, feature = "testing"))]
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let mut budgets = BTreeMap::new();
        budgets.insert(Tier::Host, u64::MAX);
        budgets.insert(Tier::Nvme, u64::MAX);
        Self::open_inner(dir.into(), budgets, u64::MAX, 0)
    }

    /// Open the store with a byte budget for NVMe and none for host memory. For tests only (spec 4.1a).
    #[cfg(any(test, feature = "testing"))]
    pub fn open_with_nvme_budget(dir: impl Into<PathBuf>, budget: u64) -> Result<Self, StoreError> {
        Self::open_with_limits(dir, u64::MAX, budget, u64::MAX, 0)
    }

    /// Open the store with a budget for each tier, an owner quota and a cap on interval copies (spec 6.13).
    /// Each put carries its engine's chain limits.
    pub fn open_with_limits(
        dir: impl Into<PathBuf>,
        host_budget: u64,
        nvme_budget: u64,
        owner_quota: u64,
        interval_copies: u32,
    ) -> Result<Self, StoreError> {
        let mut budgets = BTreeMap::new();
        budgets.insert(Tier::Host, host_budget);
        budgets.insert(Tier::Nvme, nvme_budget);
        Self::open_inner(dir.into(), budgets, owner_quota, interval_copies)
    }

    fn open_inner(
        dir: PathBuf,
        budgets: BTreeMap<Tier, u64>,
        owner_quota: u64,
        interval_copies: u32,
    ) -> Result<Self, StoreError> {
        std::fs::create_dir_all(dir.join("blobs")).map_err(io_error)?;
        // 6.7: a cleared tripwire keeps its blobs here until a person removes them.
        std::fs::create_dir_all(dir.join(QUARANTINE)).map_err(io_error)?;
        let path = dir.join("index.json");
        let mut index: Index = if path.is_file() {
            let bytes = std::fs::read(&path).map_err(io_error)?;
            serde_json::from_slice(&bytes).map_err(json_error)?
        } else {
            Index::default()
        };
        // 6.11: the host tier does not survive a restart, so its entries and their pins leave the index.
        // An index from before the host tier held its bytes in memory also left a file for each such entry.
        let lost: Vec<ContentId> = index
            .entries
            .values()
            .filter(|entry| entry.tier == Tier::Host)
            .map(|entry| entry.content_id)
            .collect();
        for id in &lost {
            index.entries.remove(id);
            index.pins.remove(id);
            match std::fs::remove_file(dir.join("blobs").join(id.hex())) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_error(error)),
            }
        }
        // 6.11: each entry is checked again at its first use after a restart.
        index.epoch = index.epoch.saturating_add(1);
        for entry in index.entries.values_mut() {
            entry.check = Check::None;
        }
        let store = Store {
            dir,
            index,
            checked: BTreeSet::new(),
            host_blobs: BTreeMap::new(),
            budgets,
            owner_quota,
            interval_copies,
            quarantine_serials: u32::MAX,
            fail_point: Cell::new(None),
        };
        store.persist()?;
        Ok(store)
    }

    /// Set a tier's budget. A profile change can do this.
    pub fn set_budget(&mut self, tier: Tier, budget: u64) {
        self.budgets.insert(tier, budget);
    }

    /// Set the cap on interval copies. A lower cap takes effect at the next interval put.
    pub fn set_interval_copies(&mut self, copies: u32) {
        self.interval_copies = copies;
    }

    /// Remove one interval copy, for example the previous copy of a context (spec 6.13). It returns false when
    /// the store has no such entry, because the cap can have removed the copy already.
    pub fn remove(&mut self, id: &ContentId, now_ms: u64) -> Result<bool, StoreError> {
        let Some(entry) = self.index.entries.get(id) else {
            return Ok(false);
        };
        if entry.reason != Reason::Interval {
            return Err(StoreError::Invalid(
                "remove drops an interval copy only; evict frees other entries".to_owned(),
            ));
        }
        // 6.8: a quarantined entry stays, because a person needs it to find the cause.
        if entry.state != EntryState::Live {
            return Err(StoreError::Quarantined(*id));
        }
        if self.has_live_pin(id, now_ms) {
            return Err(StoreError::Invalid(
                "a pin holds the copy; remove it after the pin ends".to_owned(),
            ));
        }
        let mut index = self.index.clone();
        prune_pins(&mut index, now_ms);
        index.entries.remove(id);
        let previous = std::mem::replace(&mut self.index, index);
        if let Err(error) = self.persist() {
            self.index = previous;
            return Err(error);
        }
        // The index no longer names the copy, so its bytes can go.
        self.remove_blobs(&[*id])?;
        Ok(true)
    }

    /// Set the byte quota of one owner in one tier.
    pub fn set_owner_quota(&mut self, quota: u64) {
        self.owner_quota = quota;
    }

    /// Make the next write fail at one point.
    pub fn fail_next(&mut self, point: FailPoint) {
        self.fail_point.set(Some(point));
    }

    /// The bytes that the host tier holds in memory, for a test or a report.
    pub fn host_resident_bytes(&self) -> u64 {
        self.host_blobs.values().map(|bytes| bytes.len() as u64).sum()
    }

    /// The entries, for a test or a report.
    pub fn entries(&self) -> &BTreeMap<ContentId, Entry> {
        &self.index.entries
    }

    /// The pins, for a test or a report.
    pub fn pins(&self) -> &BTreeMap<ContentId, Vec<Pin>> {
        &self.index.pins
    }

    /// The refused recipes, for a test or a report.
    pub fn refused(&self) -> &BTreeSet<RecipeId> {
        &self.index.refused
    }

    /// Put a snapshot in the store.
    pub fn put(&mut self, put: Put<'_>) -> Result<ContentId, StoreError> {
        self.check_reason(&put)?;
        let bytes = put.snapshot.encode()?;
        let id = content_id(&bytes);
        if let Some(existing) = self.index.entries.get(&id) {
            let state = existing.state;
            let quarantine = existing.quarantine;
            let recipe = existing.recipe;
            let parent = existing.parent;
            let tier = existing.tier;
            if state == EntryState::Quarantined {
                if quarantine == Some(Quarantine::Tripwire) {
                    return Err(self.tripwire_error(put.recipe));
                }
                return Err(StoreError::Quarantined(id));
            }
            if recipe != put.recipe || parent != put.snapshot.parent {
                return Err(StoreError::Invalid(
                    "the content id already has another recipe".to_owned(),
                ));
            }
            if tier != put.tier {
                return Err(StoreError::Invalid(format!(
                    "the entry is in tier {tier:?}; the put asks for tier {:?}. Evict the entry first",
                    put.tier
                )));
            }
            return Ok(id);
        }
        if self.index.refused.contains(&put.recipe) {
            return Err(self.tripwire_error(put.recipe));
        }

        let coverage = coverage_of(put.snapshot);
        let size = bytes.len() as u64;
        if put.snapshot.parent.is_some() {
            self.check_delta(&put, &coverage, size)?;
        }

        // 6.7: two content ids under one recipe with the same coverage are a defect.
        // Each side counts only when its own recorded class allows the tripwire.
        let conflict = tripwire_on(put.engine.determinism, &put.engine.bit_dependence)
            && self.index.entries.values().any(|entry| {
                entry.recipe == put.recipe
                    && entry.coverage == coverage
                    && entry.state == EntryState::Live
                    && entry.tripwire_on()
            });
        if conflict {
            return self.put_tripwire(put, &bytes, id, coverage);
        }

        // 6.13: above the cap, the least recently used copies go at once.
        let capped = if put.reason == Reason::Interval {
            self.cap_victims(put.now_ms)?
        } else {
            Vec::new()
        };
        let capped_set: BTreeSet<ContentId> = capped.iter().copied().collect();

        // 6.9: choose the victims before the store writes the new entry.
        let need = self.need_bytes(&put, size, &capped_set);
        // 6.5: the new entry needs its whole chain. A victim is never an ancestor of the new entry.
        let mut spared = capped_set.clone();
        spared.extend(self.ancestors(put.snapshot.parent));
        let mut victims = if need == 0 {
            Vec::new()
        } else {
            self.choose_victims(put.owner, put.tier, need, put.now_ms, &spared)
        };
        if need > 0 {
            let freed: u64 = victims
                .iter()
                .filter_map(|victim| self.index.entries.get(victim))
                .map(|entry| entry.size)
                .sum();
            if freed < need {
                return Err(StoreError::NoSpace);
            }
        }
        victims.extend(capped);

        self.store_bytes(put.tier, &bytes, &id)?;

        let mut index = self.index.clone();
        prune_pins(&mut index, put.now_ms);
        for victim in &victims {
            index.entries.remove(victim);
            index.pins.remove(victim);
        }
        index.entries.insert(
            id,
            Entry {
                content_id: id,
                recipe: put.recipe,
                parent: put.snapshot.parent,
                coverage,
                tier: put.tier,
                size,
                owner: put.owner,
                created_ms: put.now_ms,
                last_used_ms: put.now_ms,
                hits: 0,
                reason: put.reason,
                provenance: put.provenance,
                check: Check::Full,
                identity: put.identity,
                state: EntryState::Live,
                engine: put.engine.name.clone(),
                determinism: put.engine.determinism.into(),
                bit_dependence: put.engine.bit_dependence.iter().map(|&item| item.into()).collect(),
                chain: EntryChain {
                    max_depth: put.engine.chain.max_depth,
                    delta_fraction: put.engine.chain.delta_fraction,
                },
                epoch: self.index.epoch,
                quarantine: None,
            },
        );
        let previous = std::mem::replace(&mut self.index, index);
        if let Err(error) = self.persist() {
            self.index = previous;
            return Err(self.rollback_blob(&id, put.tier, error));
        }
        // The index no longer names the victims, so their bytes can go.
        self.remove_blobs(&victims)?;
        Ok(id)
    }

    /// The tripwire path: store the new entry and quarantine both content ids.
    fn put_tripwire(
        &mut self,
        put: Put<'_>,
        bytes: &[u8],
        id: ContentId,
        coverage: Vec<PartCoverage>,
    ) -> Result<ContentId, StoreError> {
        self.store_bytes(put.tier, bytes, &id)?;
        let mut index = self.index.clone();
        let epoch = self.index.epoch;
        prune_pins(&mut index, put.now_ms);
        let mut ids = vec![id];
        for entry in index.entries.values_mut() {
            if entry.recipe == put.recipe
                && entry.coverage == coverage
                && entry.state == EntryState::Live
                && entry.tripwire_on()
            {
                entry.state = EntryState::Quarantined;
                entry.quarantine = Some(Quarantine::Tripwire);
                ids.push(entry.content_id);
            }
        }
        let size = bytes.len() as u64;
        index.entries.insert(
            id,
            Entry {
                content_id: id,
                recipe: put.recipe,
                parent: put.snapshot.parent,
                coverage,
                tier: put.tier,
                size,
                owner: put.owner,
                created_ms: put.now_ms,
                last_used_ms: put.now_ms,
                hits: 0,
                reason: put.reason,
                provenance: put.provenance,
                check: Check::Full,
                identity: put.identity,
                state: EntryState::Quarantined,
                engine: put.engine.name.clone(),
                determinism: put.engine.determinism.into(),
                bit_dependence: put.engine.bit_dependence.iter().map(|&item| item.into()).collect(),
                chain: EntryChain {
                    max_depth: put.engine.chain.max_depth,
                    delta_fraction: put.engine.chain.delta_fraction,
                },
                epoch,
                quarantine: Some(Quarantine::Tripwire),
            },
        );
        index.refused.insert(put.recipe);
        let previous = std::mem::replace(&mut self.index, index);
        if let Err(error) = self.persist() {
            self.index = previous;
            return Err(self.rollback_blob(&id, put.tier, error));
        }
        ids.sort();
        Err(StoreError::Tripwire {
            recipe: put.recipe,
            ids,
        })
    }

    /// Read an entry, and check it on the first use after a restart.
    pub fn get(&mut self, id: &ContentId, now_ms: u64) -> Result<Option<&Entry>, StoreError> {
        if !self.index.entries.contains_key(id) {
            return Ok(None);
        }
        self.check(id)?;
        self.touch(id, now_ms);
        Ok(self.index.entries.get(id))
    }

    /// Look up a chain by recipe. A miss is an empty result, not an error.
    pub fn lookup(
        &mut self,
        recipe: RecipeId,
        tolerance: &Tolerance,
        engine: &mut dyn Engine,
        slot: u32,
        now_ms: u64,
    ) -> Result<Vec<ContentId>, StoreError> {
        // 6.7: only a tripwire refuses the whole recipe.
        if self.index.refused.contains(&recipe) {
            return Err(self.tripwire_error(recipe));
        }
        let ids: Vec<ContentId> = self
            .index
            .entries
            .values()
            .filter(|entry| entry.recipe == recipe && entry.state == EntryState::Live)
            .map(|entry| entry.content_id)
            .collect();
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let limits = engine.limits();
        let mut coverage = engine.coverage();
        let mut chains: Vec<(Vec<ContentId>, Vec<PartCoverage>)> = Vec::new();
        for id in ids {
            // 6.8: a quarantine of one entry does not refuse the recipe. Skip its chain.
            // 6.5: a missing parent stays an error.
            let chain = match self.chain(id) {
                Ok(chain) => chain,
                Err(StoreError::Quarantined(_)) => continue,
                Err(error) => return Err(error),
            };
            if chain.iter().any(|member| self.check(member).is_err()) {
                continue;
            }
            let entries: Vec<Entry> = chain
                .iter()
                .filter_map(|member| self.index.entries.get(member).cloned())
                .collect();
            if !chain_accepts(&entries, tolerance, self.index.epoch) {
                continue;
            }
            let union = chain_coverage(&entries);
            if let Some(fits) = coverage.as_mut() {
                let mut ok = true;
                for span in &union {
                    let Some(part) = limits.parts.iter().position(|candidate| candidate.name == span.part) else {
                        return Err(StoreError::Invalid(format!("unknown part {}", span.part)));
                    };
                    let span = CoverageSpan {
                        part,
                        start: span.start,
                        end: span.end,
                    };
                    if fits.fits(&span, slot).is_err() {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    continue;
                }
            }
            chains.push((chain, union));
        }

        if chains.is_empty() {
            return Ok(Vec::new());
        }
        let mut best: Vec<&(Vec<ContentId>, Vec<PartCoverage>)> = chains
            .iter()
            .filter(|(_, coverage)| chains.iter().all(|(_, other)| dominates(coverage, other)))
            .collect();
        // Zero maxima means the chains are incomparable.
        if best.is_empty() {
            // The coverage order is partial and has no greatest chain, so some two chains are incomparable.
            let (left, right) = chains
                .iter()
                .enumerate()
                .flat_map(|(index, left)| chains[index + 1..].iter().map(move |right| (left, right)))
                .find(|((_, left), (_, right))| !dominates(left, right) && !dominates(right, left))
                .map(|((left, _), (right, _))| (left, right))
                .expect("a set with no greatest chain holds two incomparable chains");
            let tip = |chain: &Vec<ContentId>| *chain.last().expect("a chain is not empty");
            return Err(StoreError::Ambiguous {
                left: tip(left),
                right: tip(right),
            });
        }
        // 6.6: a tie is served by the newest created_ms, then the smaller content id.
        best.sort_by(|left, right| {
            let left_tip = left.0.last().and_then(|id| self.index.entries.get(id));
            let right_tip = right.0.last().and_then(|id| self.index.entries.get(id));
            let left_created = left_tip.map_or(0, |entry| entry.created_ms);
            let right_created = right_tip.map_or(0, |entry| entry.created_ms);
            right_created
                .cmp(&left_created)
                .then_with(|| left.0.last().cmp(&right.0.last()))
        });
        let chain = best[0].0.clone();
        if let Some(tip) = chain.last() {
            self.touch(tip, now_ms);
        }
        Ok(chain)
    }

    /// Look up a chain and pin it until the load ends.
    pub fn lookup_and_pin(
        &mut self,
        recipe: RecipeId,
        tolerance: &Tolerance,
        engine: &mut dyn Engine,
        slot: u32,
        pin: Pin,
        now_ms: u64,
    ) -> Result<Vec<ContentId>, StoreError> {
        check_pin_lease(pin.expires_ms, now_ms)?;
        let chain = self.lookup(recipe, tolerance, engine, slot, now_ms)?;
        for id in &chain {
            set_pin(&mut self.index.pins, *id, pin.clone());
        }
        self.persist()?;
        Ok(chain)
    }

    /// Restore an entry and its chain into an engine state.
    pub fn restore(
        &mut self,
        id: &ContentId,
        engine: &mut dyn Engine,
        state: State,
        identity: &[u8; 32],
        now_ms: u64,
    ) -> Result<Restored, StoreError> {
        self.check(id)?;
        let entry = self
            .index
            .entries
            .get(id)
            .ok_or_else(|| StoreError::Invalid("missing entry".to_owned()))?
            .clone();
        if entry.state != EntryState::Live {
            return Err(StoreError::Quarantined(*id));
        }
        if entry.identity != *identity {
            self.quarantine(id, Quarantine::Identity)?;
            return Err(StoreError::Quarantined(*id));
        }

        // 6.5: a chain with a missing parent is an error, never a short chain.
        let chain = self.chain(*id)?;
        let mut snapshots = Vec::new();
        for member in &chain {
            self.check(member)?;
            let snapshot = self.read_snapshot(member)?;
            // 6.11: every entry's index facts must match its blob.
            let record = self
                .index
                .entries
                .get(member)
                .ok_or_else(|| StoreError::Invalid("missing entry".to_owned()))?;
            if snapshot.parent != record.parent || coverage_of(&snapshot) != record.coverage {
                self.quarantine(member, Quarantine::Index)?;
                return Err(StoreError::Invalid("the index disagrees with the blob".to_owned()));
            }
            snapshots.push(snapshot);
        }

        let limits = engine.limits();
        let State::Slot(slot) = state;
        let entries: Vec<Entry> = chain
            .iter()
            .filter_map(|member| self.index.entries.get(member).cloned())
            .collect();
        let union = chain_coverage(&entries);
        if let Some(coverage) = engine.coverage() {
            for span in &union {
                let part = limits
                    .parts
                    .iter()
                    .position(|candidate| candidate.name == span.part)
                    .ok_or_else(|| StoreError::Invalid(format!("unknown part {}", span.part)))?;
                let span = CoverageSpan {
                    part,
                    start: span.start,
                    end: span.end,
                };
                coverage.fits(&span, slot).map_err(StoreError::Fit)?;
            }
        }

        let mut written = 0usize;
        // The rows that the snapshots hold: the end of the cells coverage. `set_n` takes them after the last part.
        let mut cells_end: Option<(String, u64)> = None;
        for snapshot in &snapshots {
            for tensor in &snapshot.tensors {
                let index = limits
                    .parts
                    .iter()
                    .position(|candidate| candidate.name == tensor.name)
                    .ok_or_else(|| StoreError::Invalid(format!("unknown part {}", tensor.name)))?;
                let result = match tensor.kind {
                    PartKind::Fixed => engine
                        .parts()
                        .ok_or_else(|| StoreError::Invalid("no parts".to_owned()))?
                        .part_write(state, index, &tensor.data),
                    PartKind::Cells | PartKind::Ring => {
                        let (from, rows) = tensor.coverage.map_or((0, 0), |(start, end)| (start, end - start));
                        if let (PartKind::Cells, Some((_, end))) = (tensor.kind, tensor.coverage)
                            && cells_end.as_ref().is_none_or(|(_, seen)| end > *seen)
                        {
                            cells_end = Some((tensor.name.clone(), end));
                        }
                        engine
                            .parts()
                            .ok_or_else(|| StoreError::Invalid("no parts".to_owned()))?
                            .cells_write(state, index, from, rows, &tensor.data)
                    }
                    PartKind::Derived => continue,
                };
                // 6.8: stop at the first write that is not ok. A later part must not land on a bad base.
                let part = tensor.name.clone();
                match result {
                    WriteResult::Ok => written += 1,
                    // A refusal changes nothing, but only when no earlier write changed the state.
                    WriteResult::Refused if written == 0 => return Ok(Restored::Refused { part }),
                    WriteResult::Refused | WriteResult::Failed => return Ok(Restored::Failed { part }),
                }
            }
        }
        // The ABI: the caller calls `set_n` after it writes every part. The writes changed the state, so a refusal
        // here is a failure (6.8).
        if let Some((part, end)) = cells_end {
            let result = engine
                .parts()
                .ok_or_else(|| StoreError::Invalid("no parts".to_owned()))?
                .set_n(state, end);
            if result != WriteResult::Ok {
                return Ok(Restored::Failed { part });
            }
        }
        // 6.8: a refusal or a failure leaves the entry as it is.
        self.touch(id, now_ms);
        Ok(Restored::Done)
    }

    fn read_snapshot(&mut self, id: &ContentId) -> Result<Snapshot, StoreError> {
        let bytes = self.read_bytes(id)?;
        match Snapshot::decode(&bytes) {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => {
                self.quarantine(id, Quarantine::ContentId)?;
                Err(error)
            }
        }
    }

    /// Pin an entry for an owner until an expiry. A lease is at most one day.
    pub fn pin(&mut self, id: ContentId, owner: u64, expires_ms: u64, now_ms: u64) -> Result<(), StoreError> {
        check_pin_lease(expires_ms, now_ms)?;
        let Some(entry) = self.index.entries.get(&id) else {
            return Err(StoreError::Invalid("missing entry".to_owned()));
        };
        if entry.state != EntryState::Live {
            return Err(StoreError::Quarantined(id));
        }
        set_pin(&mut self.index.pins, id, Pin { owner, expires_ms });
        self.persist()
    }

    /// Remove the pin of one owner. The pins of other owners stay (spec 6.10).
    pub fn unpin(&mut self, id: &ContentId, owner: u64) -> Result<(), StoreError> {
        if let Some(pins) = self.index.pins.get_mut(id) {
            pins.retain(|pin| pin.owner != owner);
            if pins.is_empty() {
                self.index.pins.remove(id);
            }
        }
        self.persist()
    }

    /// Clear a tripwire refusal after a person knows the cause.
    /// It removes the recipe's quarantined entries from the index and moves their blobs to `quarantine/`.
    pub fn clear_refusal(&mut self, recipe: RecipeId) -> Result<(), StoreError> {
        if !self.index.refused.contains(&recipe) {
            return Ok(());
        }
        let bad: Vec<ContentId> = self
            .index
            .entries
            .values()
            .filter(|entry| entry.recipe == recipe && entry.quarantine == Some(Quarantine::Tripwire))
            .map(|entry| entry.content_id)
            .collect();
        // Find every destination name first. A name that is not free must leave the index and the blobs as they are.
        // A host entry has no file, so its bytes are written to `quarantine/` instead of moved (spec 6.8: a failure
        // never deletes content).
        let mut moves: Vec<(PathBuf, PathBuf)> = Vec::new();
        let mut saves: Vec<(Arc<[u8]>, PathBuf)> = Vec::new();
        for id in &bad {
            let to = self.quarantine_path(id)?;
            if self.index.entries.get(id).is_some_and(|entry| entry.tier == Tier::Host) {
                saves.push((self.read_bytes(id)?, to));
            } else {
                moves.push((self.blob_path(id), to));
            }
        }
        let mut index = self.index.clone();
        index.refused.remove(&recipe);
        for id in &bad {
            index.entries.remove(id);
            index.pins.remove(id);
        }
        let previous = std::mem::replace(&mut self.index, index);
        if let Err(error) = self.persist() {
            self.index = previous;
            return Err(error);
        }
        for id in &bad {
            self.checked.remove(id);
            self.host_blobs.remove(id);
        }
        for (bytes, to) in &saves {
            write_synced(to, bytes).map_err(io_error)?;
        }
        if !saves.is_empty() {
            sync_dir(&self.dir.join(QUARANTINE)).map_err(io_error)?;
        }
        self.move_blobs_to_quarantine(&moves)?;
        Ok(())
    }

    /// Clear the quarantine of one entry after a person repairs its blob.
    pub fn clear_entry(&mut self, id: &ContentId) -> Result<(), StoreError> {
        let Some(entry) = self.index.entries.get(id).cloned() else {
            return Err(StoreError::Invalid("missing entry".to_owned()));
        };
        if entry.state != EntryState::Quarantined {
            return Ok(());
        }
        if entry.quarantine == Some(Quarantine::Tripwire) {
            return Err(StoreError::Invalid(
                "a tripwire quarantine clears with its recipe".to_owned(),
            ));
        }
        let bytes = self.read_bytes(id)?;
        if content_id(&bytes) != *id {
            return Err(StoreError::Quarantined(*id));
        }
        let snapshot = Snapshot::decode(&bytes)?;
        if snapshot.parent != entry.parent || coverage_of(&snapshot) != entry.coverage {
            return Err(StoreError::Invalid("the index disagrees with the blob".to_owned()));
        }
        if let Some(entry) = self.index.entries.get_mut(id) {
            entry.state = EntryState::Live;
            entry.quarantine = None;
            entry.check = Check::Full;
        }
        self.checked.insert(*id);
        self.persist()
    }

    /// Choose and remove victims for an owner in one tier. A byte in one tier does not make room in the other (6.9).
    pub fn evict(
        &mut self,
        owner: u64,
        tier: Tier,
        need_bytes: u64,
        now_ms: u64,
    ) -> Result<Vec<ContentId>, StoreError> {
        if need_bytes == 0 {
            return Ok(Vec::new());
        }
        let victims = self.choose_victims(owner, tier, need_bytes, now_ms, &BTreeSet::new());
        let freed: u64 = victims
            .iter()
            .filter_map(|victim| self.index.entries.get(victim))
            .map(|entry| entry.size)
            .sum();
        if freed < need_bytes {
            return Err(StoreError::NoSpace);
        }
        let mut index = self.index.clone();
        prune_pins(&mut index, now_ms);
        for id in &victims {
            index.entries.remove(id);
            index.pins.remove(id);
        }
        let previous = std::mem::replace(&mut self.index, index);
        if let Err(error) = self.persist() {
            self.index = previous;
            return Err(error);
        }
        // 6.9 and the banto: persist the index first, then delete the blobs.
        self.remove_blobs(&victims)?;
        Ok(victims)
    }

    fn choose_victims(
        &self,
        owner: u64,
        tier: Tier,
        need_bytes: u64,
        now_ms: u64,
        excluding: &BTreeSet<ContentId>,
    ) -> Vec<ContentId> {
        if need_bytes == 0 {
            return Vec::new();
        }
        let parents: BTreeSet<ContentId> = self
            .index
            .entries
            .values()
            .filter(|entry| entry.state == EntryState::Live)
            .filter_map(|entry| entry.parent)
            .collect();
        // 6.9 step 1: interval copies go first, the least recently used first. Their rank is below `RANK_INTERIOR`.
        // Step 2: a pinned entry is exempt, so the loop below skips it.
        let mut groups: Vec<(u8, u128, Vec<ContentId>)> = Vec::new();
        for entry in self.index.entries.values() {
            if entry.owner != owner || entry.state != EntryState::Live {
                continue;
            }
            if entry.tier != tier || excluding.contains(&entry.content_id) {
                continue;
            }
            if self.pin_protects(entry, now_ms) {
                continue;
            }
            // A group can go only when every live descendant has the same owner and no pin.
            let mut group = vec![entry.content_id];
            let mut evictable = true;
            for child in self.descendants(entry.content_id) {
                let Some(child_entry) = self.index.entries.get(&child) else {
                    continue;
                };
                if child_entry.owner != owner || self.pin_protects(child_entry, now_ms) || child_entry.tier != tier {
                    evictable = false;
                    break;
                }
                group.push(child);
            }
            if !evictable {
                continue;
            }
            // 6.9 step 3: a chain interior goes before a tip or a root. Step 4 ranks all the others together.
            let interior = entry.parent.is_some() && parents.contains(&entry.content_id);
            if entry.reason == Reason::Interval {
                groups.push((RANK_INTERVAL, entry.last_used_ms as u128, group));
                continue;
            }
            let rank = if interior { RANK_INTERIOR } else { RANK_BY_VALUE };
            groups.push((rank, value_per_byte(entry, now_ms), group));
        }
        groups.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

        let mut victims = Vec::new();
        let mut chosen = BTreeSet::new();
        let mut bytes = 0u64;
        for (_, _, group) in groups {
            if group.iter().any(|id| chosen.contains(id)) {
                continue;
            }
            let group_bytes: u64 = group
                .iter()
                .filter_map(|id| self.index.entries.get(id))
                .map(|entry| entry.size)
                .sum();
            for id in &group {
                chosen.insert(*id);
            }
            victims.extend(group);
            bytes = bytes.saturating_add(group_bytes);
            if bytes >= need_bytes {
                break;
            }
        }
        victims
    }

    /// True when a pin keeps the entry from eviction.
    fn pin_protects(&self, entry: &Entry, now_ms: u64) -> bool {
        // An entry is pinned while any owner's pin is live.
        match self.index.pins.get(&entry.content_id) {
            Some(pins) => pins.iter().any(|pin| pin.expires_ms > now_ms) && entry.size <= self.budget_of(entry.tier),
            None => false,
        }
    }

    fn budget_of(&self, tier: Tier) -> u64 {
        self.budgets.get(&tier).copied().unwrap_or(u64::MAX)
    }

    /// The bytes that the put must free. Entries in `leaving` are already chosen to go, so they do not count.
    fn need_bytes(&self, put: &Put<'_>, size: u64, leaving: &BTreeSet<ContentId>) -> u64 {
        let tier_total: u64 = self
            .index
            .entries
            .values()
            .filter(|entry| !leaving.contains(&entry.content_id))
            .filter(|entry| entry.state == EntryState::Live && entry.tier == put.tier)
            .map(|entry| entry.size)
            .sum();
        let owner_total: u64 = self
            .index
            .entries
            .values()
            .filter(|entry| !leaving.contains(&entry.content_id))
            .filter(|entry| entry.state == EntryState::Live && entry.owner == put.owner && entry.tier == put.tier)
            .map(|entry| entry.size)
            .sum();
        let by_budget = tier_total.saturating_add(size).saturating_sub(self.budget_of(put.tier));
        let by_quota = owner_total.saturating_add(size).saturating_sub(self.owner_quota);
        by_budget.max(by_quota)
    }

    /// Check that the reason, the tier and the recipe of a put agree (spec 6.13). An interval copy goes to the host
    /// tier, has no parent, and has the recipe with the item "not sealed". No other put has that item.
    fn check_reason(&self, put: &Put<'_>) -> Result<(), StoreError> {
        if put.reason != Reason::Interval {
            if put.sealed_recipe.is_some() {
                return Err(StoreError::Invalid(
                    "only an interval copy has a recipe with the item not sealed; remove sealed_recipe".to_owned(),
                ));
            }
            return Ok(());
        }
        if self.interval_copies == 0 {
            return Err(StoreError::Invalid(
                "the store keeps no interval copies: interval_copies in the [store] table of the profile is 0. \
                 Set it above 0 or do not put an interval copy"
                    .to_owned(),
            ));
        }
        if put.tier != Tier::Host {
            return Err(StoreError::Invalid(
                "an interval copy goes to the host tier, never to NVMe (spec 6.13)".to_owned(),
            ));
        }
        if put.snapshot.parent.is_some() {
            return Err(StoreError::Invalid(
                "an interval copy holds whole parts and has no parent (spec 6.13)".to_owned(),
            ));
        }
        match put.sealed_recipe {
            Some(sealed) if put.recipe == recipe_not_sealed(&sealed) => Ok(()),
            _ => Err(StoreError::Invalid(
                "the recipe of an interval copy must be recipe_not_sealed of the sealed recipe at its position: \
                 the recipe needs the item not sealed (spec 6.2 tag 8)"
                    .to_owned(),
            )),
        }
    }

    /// The copies to remove so that one more copy fits the cap: the least recently used first (spec 6.13).
    /// A pinned copy stays. When pins stop the store from getting under the cap, the put is refused.
    fn cap_victims(&self, now_ms: u64) -> Result<Vec<ContentId>, StoreError> {
        let mut copies: Vec<&Entry> = self
            .index
            .entries
            .values()
            .filter(|entry| entry.reason == Reason::Interval && entry.state == EntryState::Live)
            .collect();
        let remove = (copies.len() + 1).saturating_sub(self.interval_copies as usize);
        copies.sort_by_key(|entry| (entry.last_used_ms, entry.created_ms, entry.content_id));
        let victims: Vec<ContentId> = copies
            .into_iter()
            .filter(|entry| !self.has_live_pin(&entry.content_id, now_ms))
            .take(remove)
            .map(|entry| entry.content_id)
            .collect();
        if victims.len() < remove {
            return Err(StoreError::NoSpace);
        }
        Ok(victims)
    }

    fn has_live_pin(&self, id: &ContentId, now_ms: u64) -> bool {
        self.index
            .pins
            .get(id)
            .is_some_and(|pins| pins.iter().any(|pin| pin.expires_ms > now_ms))
    }

    fn check_delta(&self, put: &Put<'_>, coverage: &[PartCoverage], size: u64) -> Result<(), StoreError> {
        let parent_id = put
            .snapshot
            .parent
            .ok_or_else(|| StoreError::Invalid("a delta needs a parent".to_owned()))?;
        let parent = self
            .index
            .entries
            .get(&parent_id)
            .ok_or_else(|| StoreError::Invalid("the delta's parent is missing".to_owned()))?;
        if parent.state != EntryState::Live {
            return Err(StoreError::Quarantined(parent_id));
        }
        if parent.reason == Reason::Interval {
            return Err(StoreError::Invalid(
                "a delta cannot have an interval copy as its parent: the copy is not sealed (spec 6.13)".to_owned(),
            ));
        }
        // 6.11: a restart loses the host tier, so an NVMe delta must not depend on a host parent.
        if put.tier == Tier::Nvme && parent.tier == Tier::Host {
            return Err(StoreError::Invalid(
                "an NVMe delta cannot have a host parent: a restart loses the host tier. Put the delta in the host tier"
                    .to_owned(),
            ));
        }
        let previous = put
            .previous_recipe
            .ok_or_else(|| StoreError::Invalid("a delta needs its previous recipe".to_owned()))?;
        if previous != parent.recipe {
            return Err(StoreError::Invalid(
                "the previous recipe is not the parent's recipe".to_owned(),
            ));
        }
        if put.recipe == parent.recipe {
            return Err(StoreError::Invalid("the delta's recipe is not a later one".to_owned()));
        }
        let parent_chain = self.chain(parent_id)?;
        let depth = parent_chain.len();
        if depth > put.engine.chain.max_depth as usize {
            return Err(StoreError::Invalid(
                "the chain is deeper than the engine allows".to_owned(),
            ));
        }
        // 6.5 check 3: the limit is a fraction of the full base snapshot. The parent can be a delta.
        let base_size = parent_chain
            .first()
            .and_then(|base| self.index.entries.get(base))
            .map(|base| base.size)
            .ok_or_else(|| StoreError::Invalid("the chain has no base".to_owned()))?;
        if (size as f64) > (put.engine.chain.delta_fraction as f64) * (base_size as f64) {
            return Err(StoreError::Invalid(
                "the delta is larger than the engine allows".to_owned(),
            ));
        }
        for parent_span in &parent.coverage {
            let span = coverage
                .iter()
                .find(|span| span.part == parent_span.part)
                .ok_or_else(|| StoreError::Invalid(format!("the delta has no {} part", parent_span.part)))?;
            if span.start != parent_span.end {
                return Err(StoreError::Invalid(format!(
                    "{} does not start at the parent's end",
                    span.part
                )));
            }
        }
        Ok(())
    }

    /// The ids from `parent` up to the root of its chain, in the entries that the index holds.
    fn ancestors(&self, parent: Option<ContentId>) -> Vec<ContentId> {
        let mut found = Vec::new();
        let mut next = parent;
        while let Some(id) = next {
            if found.contains(&id) {
                break;
            }
            found.push(id);
            next = self.index.entries.get(&id).and_then(|entry| entry.parent);
        }
        found
    }

    fn descendants(&self, base: ContentId) -> Vec<ContentId> {
        let mut found = Vec::new();
        let mut visited = BTreeSet::new();
        visited.insert(base);
        let mut queue = vec![base];
        while let Some(current) = queue.pop() {
            for entry in self.index.entries.values() {
                if entry.parent == Some(current) && entry.state == EntryState::Live && visited.insert(entry.content_id)
                {
                    found.push(entry.content_id);
                    queue.push(entry.content_id);
                }
            }
        }
        found
    }

    fn chain(&self, tip: ContentId) -> Result<Vec<ContentId>, StoreError> {
        let mut chain = vec![tip];
        let mut visited = BTreeSet::new();
        visited.insert(tip);
        let mut current = tip;
        loop {
            let entry = self
                .index
                .entries
                .get(&current)
                .ok_or_else(|| StoreError::Invalid(format!("the chain has a missing entry {}", current.hex())))?;
            if entry.state != EntryState::Live {
                return Err(StoreError::Quarantined(current));
            }
            let Some(parent) = entry.parent else {
                break;
            };
            let parent_entry = self.index.entries.get(&parent).ok_or_else(|| {
                StoreError::Invalid(format!(
                    "the chain of {} has a missing parent {}",
                    tip.hex(),
                    parent.hex()
                ))
            })?;
            if parent_entry.state != EntryState::Live {
                return Err(StoreError::Quarantined(parent));
            }
            if !visited.insert(parent) {
                return Err(StoreError::Invalid("the chain has a cycle".to_owned()));
            }
            chain.push(parent);
            current = parent;
        }
        chain.reverse();
        Ok(chain)
    }

    fn tripwire_error(&self, recipe: RecipeId) -> StoreError {
        let ids = self
            .index
            .entries
            .values()
            .filter(|entry| entry.recipe == recipe && entry.quarantine == Some(Quarantine::Tripwire))
            .map(|entry| entry.content_id)
            .collect();
        StoreError::Tripwire { recipe, ids }
    }

    fn check(&mut self, id: &ContentId) -> Result<(), StoreError> {
        if self.checked.contains(id) {
            return Ok(());
        }
        let Some(entry) = self.index.entries.get(id) else {
            return Err(StoreError::Invalid("missing entry".to_owned()));
        };
        if entry.state != EntryState::Live {
            return Err(StoreError::Quarantined(*id));
        }
        let bytes = self.read_bytes(id)?;
        if content_id(&bytes) != *id {
            self.quarantine(id, Quarantine::ContentId)?;
            return Err(StoreError::Quarantined(*id));
        }
        if let Some(entry) = self.index.entries.get_mut(id) {
            entry.check = Check::Full;
        }
        self.checked.insert(*id);
        Ok(())
    }

    fn quarantine(&mut self, id: &ContentId, reason: Quarantine) -> Result<(), StoreError> {
        if let Some(entry) = self.index.entries.get_mut(id) {
            entry.state = EntryState::Quarantined;
            entry.quarantine = Some(reason);
        }
        self.checked.remove(id);
        self.persist()
    }

    fn touch(&mut self, id: &ContentId, now_ms: u64) {
        if let Some(entry) = self.index.entries.get_mut(id) {
            entry.hits = entry.hits.saturating_add(1);
            entry.last_used_ms = now_ms;
        }
    }

    /// Keep the canonical bytes of an entry in its tier: in memory for host, in a file for NVMe.
    fn store_bytes(&mut self, tier: Tier, bytes: &[u8], id: &ContentId) -> Result<(), StoreError> {
        match tier {
            Tier::Host => {
                self.host_blobs.insert(*id, Arc::from(bytes));
                Ok(())
            }
            Tier::Nvme => self.write_blob(bytes, id),
        }
    }

    /// Read the canonical bytes of an entry from its tier. A missing host entry is an error, never an empty read.
    fn read_bytes(&self, id: &ContentId) -> Result<Arc<[u8]>, StoreError> {
        let entry = self
            .index
            .entries
            .get(id)
            .ok_or_else(|| StoreError::Invalid("missing entry".to_owned()))?;
        match entry.tier {
            Tier::Host => self.host_blobs.get(id).cloned().ok_or_else(|| {
                StoreError::Invalid(format!(
                    "the host tier holds no bytes for the entry {}; the index and the memory disagree",
                    id.hex()
                ))
            }),
            Tier::Nvme => match std::fs::read(self.blob_path(id)) {
                Ok(bytes) => Ok(Arc::from(bytes)),
                Err(error) => Err(read_failed(id, error)),
            },
        }
    }

    fn write_blob(&self, bytes: &[u8], id: &ContentId) -> Result<(), StoreError> {
        let path = self.blob_path(id);
        let temp = self.dir.join("blobs").join(format!("{}.tmp", id.hex()));
        if self.fail_point.get() == Some(FailPoint::BlobWrite) {
            self.fail_point.set(None);
            return Err(StoreError::Io("injected blob write".to_owned()));
        }
        write_synced(&temp, bytes).map_err(io_error)?;
        if self.fail_point.get() == Some(FailPoint::BlobRename) {
            self.fail_point.set(None);
            let _ = std::fs::remove_file(&temp);
            return Err(StoreError::Io("injected blob rename".to_owned()));
        }
        std::fs::rename(&temp, &path).map_err(io_error)?;
        sync_dir(&self.dir.join("blobs")).map_err(io_error)?;
        Ok(())
    }

    fn rollback_blob(&mut self, id: &ContentId, tier: Tier, error: StoreError) -> StoreError {
        if tier == Tier::Host {
            self.host_blobs.remove(id);
            return error;
        }
        match std::fs::remove_file(self.blob_path(id)) {
            Ok(()) => error,
            Err(remove_error) if remove_error.kind() == std::io::ErrorKind::NotFound => error,
            Err(remove_error) => StoreError::Io(format!("{error}; the new blob also failed to remove: {remove_error}")),
        }
    }

    fn remove_blobs(&mut self, ids: &[ContentId]) -> Result<(), StoreError> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut files = false;
        for id in ids {
            self.checked.remove(id);
            // A host entry has no file.
            if self.host_blobs.remove(id).is_some() {
                continue;
            }
            files = true;
            match std::fs::remove_file(self.blob_path(id)) {
                Ok(()) => {}
                // The index no longer names the entry. An absent blob is the wanted state.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_error(error)),
            }
        }
        if files {
            sync_dir(&self.dir.join("blobs")).map_err(io_error)?;
        }
        Ok(())
    }

    /// Move cleared blobs to `quarantine/`, and never delete them. A failed move is an error.
    fn move_blobs_to_quarantine(&self, moves: &[(PathBuf, PathBuf)]) -> Result<(), StoreError> {
        if moves.is_empty() {
            return Ok(());
        }
        for (from, to) in moves {
            match std::fs::rename(from, to) {
                Ok(()) => {}
                // The index no longer names the entry. An absent blob is the state that a crash between
                // the index write and the move leaves. There is nothing to keep.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(StoreError::Io(format!(
                        "the blob {} did not move to quarantine: {error}",
                        from.display()
                    )));
                }
            }
        }
        sync_dir(&self.dir.join(QUARANTINE)).map_err(io_error)?;
        sync_dir(&self.dir.join("blobs")).map_err(io_error)
    }

    /// The name that no earlier quarantined blob holds. The content id comes first; a serial follows it.
    fn quarantine_path(&self, id: &ContentId) -> Result<PathBuf, StoreError> {
        next_quarantine_path(&self.dir.join(QUARANTINE), &id.hex(), self.quarantine_serials)
    }

    fn persist(&self) -> Result<(), StoreError> {
        let path = self.dir.join("index.json");
        let temp = self.dir.join("index.json.tmp");
        let bytes = serde_json::to_vec(&self.index).map_err(json_error)?;
        if self.fail_point.get() == Some(FailPoint::IndexWrite) {
            self.fail_point.set(None);
            return Err(StoreError::Io("injected index write".to_owned()));
        }
        write_synced(&temp, &bytes).map_err(io_error)?;
        if self.fail_point.get() == Some(FailPoint::IndexRename) {
            self.fail_point.set(None);
            let _ = std::fs::remove_file(&temp);
            return Err(StoreError::Io("injected index rename".to_owned()));
        }
        std::fs::rename(&temp, &path).map_err(io_error)?;
        sync_dir(&self.dir).map_err(io_error)?;
        Ok(())
    }

    fn blob_path(&self, id: &ContentId) -> PathBuf {
        self.dir.join("blobs").join(id.hex())
    }
}

/// The name that no earlier quarantined blob holds. The content id comes first; a serial follows it.
/// The search stops at `max_serial`. An error names the blob when every name is taken.
fn next_quarantine_path(dir: &Path, name: &str, max_serial: u32) -> Result<PathBuf, StoreError> {
    let mut serial = 0u32;
    loop {
        let path = if serial == 0 {
            dir.join(name)
        } else {
            dir.join(format!("{name}.{serial}"))
        };
        if !path.exists() {
            return Ok(path);
        }
        match serial.checked_add(1) {
            Some(next) if next <= max_serial => serial = next,
            _ => {
                return Err(StoreError::Io(format!(
                    "no quarantine name is free for the blob {name}; remove an old quarantined blob"
                )));
            }
        }
    }
}

/// True when the tripwire (spec 6.7) applies to an engine of this class: its bits are the same in any process
/// and do not depend on residency.
fn tripwire_on(determinism: Determinism, bit_dependence: &[BitDependence]) -> bool {
    determinism != Determinism::PerProcess && !bit_dependence.contains(&BitDependence::Residency)
}

impl Entry {
    /// True when the tripwire applies to this entry, by the class that the entry recorded.
    fn tripwire_on(&self) -> bool {
        self.determinism != DeterminismClass::PerProcess
            && !self.bit_dependence.contains(&BitDependenceClass::Residency)
    }
}

/// A blob read error is not a verification failure (spec 6.8). The entry stays in the index and the next use
/// reads the blob again. The caller sees an error, and a lookup treats it as a miss.
fn read_failed(id: &ContentId, error: std::io::Error) -> StoreError {
    eprintln!(
        "nagaya-store: cannot read the blob of {}: {error}; the entry stays and the next use checks it again",
        id.hex()
    );
    io_error(error)
}

fn prune_pins(index: &mut Index, now_ms: u64) {
    for pins in index.pins.values_mut() {
        pins.retain(|pin| pin.expires_ms > now_ms);
    }
    index.pins.retain(|_, pins| !pins.is_empty());
}

/// Set the pin of one owner on an entry. It replaces that owner's earlier pin and no other.
fn set_pin(pins: &mut BTreeMap<ContentId, Vec<Pin>>, id: ContentId, pin: Pin) {
    let list = pins.entry(id).or_default();
    list.retain(|held| held.owner != pin.owner);
    list.push(pin);
}

/// Read the pins of an index. The old format held one pin object for each entry. The new format holds a list.
fn read_pins<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<BTreeMap<ContentId, Vec<Pin>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stored {
        One(Pin),
        Many(Vec<Pin>),
    }
    let stored = BTreeMap::<ContentId, Stored>::deserialize(deserializer)?;
    Ok(stored
        .into_iter()
        .map(|(id, stored)| {
            let pins = match stored {
                Stored::One(pin) => vec![pin],
                Stored::Many(pins) => pins,
            };
            (id, pins)
        })
        .collect())
}

fn check_pin_lease(expires_ms: u64, now_ms: u64) -> Result<(), StoreError> {
    if expires_ms.saturating_sub(now_ms) > MAX_PIN_MS {
        return Err(StoreError::Invalid("a pin cannot last longer than one day".to_owned()));
    }
    Ok(())
}

fn coverage_of(snapshot: &Snapshot) -> Vec<PartCoverage> {
    let mut coverage: Vec<PartCoverage> = snapshot
        .tensors
        .iter()
        .filter_map(|tensor| {
            tensor.coverage.map(|(start, end)| PartCoverage {
                part: tensor.name.clone(),
                start,
                end,
            })
        })
        .collect();
    coverage.sort();
    coverage
}

fn chain_coverage(entries: &[Entry]) -> Vec<PartCoverage> {
    let mut spans: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for entry in entries {
        for span in &entry.coverage {
            let slot = spans.entry(span.part.as_str()).or_insert((span.start, span.end));
            slot.0 = slot.0.min(span.start);
            slot.1 = slot.1.max(span.end);
        }
    }
    spans
        .into_iter()
        .map(|(part, (start, end))| PartCoverage {
            part: part.to_owned(),
            start,
            end,
        })
        .collect()
}

fn chain_accepts(entries: &[Entry], tolerance: &Tolerance, epoch: u64) -> bool {
    match tolerance {
        Tolerance::Exact => entries.iter().all(|entry| {
            entry.provenance.is_none() && (entry.determinism != DeterminismClass::PerProcess || entry.epoch == epoch)
        }),
        Tolerance::Approx(methods) => entries
            .iter()
            .all(|entry| entry.provenance.as_ref().is_none_or(|method| methods.contains(method))),
    }
}

fn dominates(left: &[PartCoverage], right: &[PartCoverage]) -> bool {
    let parts: BTreeSet<&str> = left.iter().chain(right.iter()).map(|part| part.part.as_str()).collect();
    parts.iter().all(|part| {
        let left_end = left
            .iter()
            .find(|candidate| candidate.part == *part)
            .map_or(0, |candidate| candidate.end);
        let right_end = right
            .iter()
            .find(|candidate| candidate.part == *part)
            .map_or(0, |candidate| candidate.end);
        left_end >= right_end
    })
}

fn value_per_byte(entry: &Entry, now_ms: u64) -> u128 {
    let age = now_ms.saturating_sub(entry.last_used_ms) as u128;
    let base = (entry.hits as u128 + 1) * 1_000_000;
    base.saturating_sub(age.min(base)) / entry.size.max(1) as u128
}

fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_dir(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

fn io_error(error: std::io::Error) -> StoreError {
    StoreError::Io(error.to_string())
}

fn json_error(error: serde_json::Error) -> StoreError {
    StoreError::Invalid(error.to_string())
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

    #[test]
    fn check_tensor_enforces_row_rules() {
        let fixed = Tensor {
            name: "f".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::U8,
            shape: vec![1],
            data: vec![0],
            coverage: None,
            block_format: None,
        };
        assert!(check_tensor(&fixed).is_ok());

        let shape_mismatch = Tensor {
            name: "c".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::U8,
            shape: vec![5, 8],
            data: vec![0; 40],
            coverage: Some((0, 2)),
            block_format: None,
        };
        assert!(check_tensor(&shape_mismatch).is_err());

        let start_after_end = Tensor {
            name: "r".to_owned(),
            kind: PartKind::Ring,
            dtype: Dtype::U8,
            shape: vec![0, 8],
            data: Vec::new(),
            coverage: Some((5, 3)),
            block_format: None,
        };
        assert!(check_tensor(&start_after_end).is_err());

        let no_rows = Tensor {
            name: "n".to_owned(),
            kind: PartKind::Cells,
            dtype: Dtype::U8,
            shape: Vec::new(),
            data: vec![0],
            coverage: Some((0, 0)),
            block_format: None,
        };
        assert!(check_tensor(&no_rows).is_err());
    }

    #[test]
    fn every_dtype_round_trips_through_the_canonical_file() {
        for dtype in Dtype::ALL {
            let width = dtype.element_bytes() as usize;
            let tensor = Tensor {
                name: "t".to_owned(),
                kind: PartKind::Fixed,
                dtype,
                shape: vec![3],
                data: (0..3 * width).map(|byte| byte as u8 + 1).collect(),
                coverage: None,
                block_format: None,
            };
            let snapshot = Snapshot {
                tensors: vec![tensor.clone()],
                parent: None,
            };
            let bytes = snapshot.encode().unwrap();
            let back = Snapshot::decode(&bytes).unwrap();
            assert_eq!(back.tensors, vec![tensor], "{dtype:?}");
        }
    }

    #[test]
    fn dtype_names_are_the_safetensors_names() {
        let names: Vec<_> = Dtype::ALL.iter().map(|&dtype| dtype_name(dtype)).collect();
        assert_eq!(
            names,
            [
                "F32", "F16", "BF16", "U8", "U16", "U32", "U64", "I8", "I16", "I32", "I64", "F8_E4M3", "F8_E5M2"
            ]
        );
    }

    #[test]
    fn block_format_requires_u8() {
        let float = Tensor {
            name: "f".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::F32,
            shape: vec![1],
            data: vec![0; 4],
            coverage: None,
            block_format: Some("IQ4_NL".to_owned()),
        };
        assert!(check_tensor(&float).is_err());

        let byte = Tensor {
            name: "b".to_owned(),
            kind: PartKind::Fixed,
            dtype: Dtype::U8,
            shape: vec![1],
            data: vec![0],
            coverage: None,
            block_format: Some("IQ4_NL".to_owned()),
        };
        assert!(check_tensor(&byte).is_ok());
    }

    #[test]
    fn a_full_quarantine_reports_an_error() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["abc", "abc.1", "abc.2"] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        let result = next_quarantine_path(dir.path(), "abc", 2);
        assert!(matches!(result, Err(StoreError::Io(_))), "{result:?}");
    }

    #[test]
    fn a_full_quarantine_leaves_the_index_and_the_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = Store::open(dir.path()).unwrap();
        store.quarantine_serials = 1;
        let recipe = recipe_root(&[9u8; 32], b"", &[], b"");
        let snapshot = |seed: u8| Snapshot {
            tensors: vec![Tensor {
                name: "state".to_owned(),
                kind: PartKind::Fixed,
                dtype: Dtype::U8,
                shape: vec![4],
                data: vec![seed; 4],
                coverage: None,
                block_format: None,
            }],
            parent: None,
        };
        fn put<'a>(snapshot: &'a Snapshot, recipe: RecipeId) -> Put<'a> {
            Put {
                snapshot,
                recipe,
                previous_recipe: None,
                engine: EngineFacts {
                    name: "test".to_owned(),
                    determinism: Determinism::Always,
                    bit_dependence: Vec::new(),
                    chain: ChainLimits {
                        max_depth: u32::MAX,
                        delta_fraction: 1.0,
                    },
                },
                identity: [1u8; 32],
                owner: 1,
                tier: Tier::Nvme,
                reason: Reason::Anchor,
                sealed_recipe: None,
                provenance: None,
                now_ms: 10,
            }
        }
        let first = store.put(put(&snapshot(1), recipe)).unwrap();
        let error = store.put(put(&snapshot(2), recipe)).unwrap_err();
        let second = match error {
            StoreError::Tripwire { ids, .. } => ids.into_iter().find(|id| *id != first).unwrap(),
            other => panic!("expected the tripwire, got {other:?}"),
        };
        for id in [first, second] {
            for name in [id.hex(), format!("{}.1", id.hex())] {
                std::fs::write(dir.path().join(QUARANTINE).join(name), b"").unwrap();
            }
        }

        assert!(store.clear_refusal(recipe).is_err());
        assert_eq!(store.entries().len(), 2);
        assert!(store.refused().contains(&recipe));
        assert!(dir.path().join("blobs").join(first.hex()).exists());
        assert!(dir.path().join("blobs").join(second.hex()).exists());
        assert_eq!(std::fs::read_dir(dir.path().join(QUARANTINE)).unwrap().count(), 4);
    }
}
