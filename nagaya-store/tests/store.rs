use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, ChainLimits, Coverage, CoverageSpan, Determinism, Dtype, Engine, Identity, Inputs, Limits,
    MemoryLimits, OutputsWanted, PartError, PartKind, PartLimits, Parts, PassError, PassOut, Segment, SegmentInputs,
    SlotLimit, State, WriteResult,
};
use nagaya_store::{ContentId, Snapshot, Tensor, content_id, recipe_boundary, recipe_root, snapshot_from_engine};
use serde_json::Value;

fn segment(slot: u32, tokens: &[u32]) -> Segment {
    Segment {
        state: State::Slot(slot),
        inputs: SegmentInputs::Tokens(tokens.to_vec()),
        positions: (0..tokens.len() as u64).collect(),
        meta: Vec::new(),
        outputs: OutputsWanted::default(),
    }
}

fn tensor(name: &str, kind: PartKind, shape: Vec<u64>, data: Vec<u8>, coverage: Option<(u64, u64)>) -> Tensor {
    Tensor {
        name: name.to_owned(),
        kind,
        dtype: nagaya_abi::Dtype::U8,
        shape,
        data,
        coverage,
        block_format: None,
    }
}

#[test]
fn equal_coverage_gives_equal_bytes() {
    let mut small = FakeAlways::new();
    let mut large = FakeAlways::new();
    small.pass(&[segment(0, &[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    large.pass(&[segment(1, &[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    let rows = [(1, 0, 6), (2, 2, 4)];
    let small = snapshot_from_engine(&mut small, State::Slot(0), &rows).unwrap();
    let large = snapshot_from_engine(&mut large, State::Slot(1), &rows).unwrap();
    assert_eq!(small.encode().unwrap(), large.encode().unwrap());
    assert_eq!(small.content_id().unwrap(), large.content_id().unwrap());
}

#[test]
fn a_delta_names_its_parent() {
    let base = Snapshot {
        tensors: vec![tensor("state", PartKind::Fixed, vec![16], vec![7; 16], None)],
        parent: None,
    };
    let base_id = base.content_id().unwrap();
    let delta = Snapshot {
        tensors: vec![
            tensor("state", PartKind::Fixed, vec![16], vec![8; 16], None),
            tensor("cells", PartKind::Cells, vec![1, 8], vec![9; 8], Some((6, 7))),
        ],
        parent: Some(base_id),
    };
    let bytes = delta.encode().unwrap();
    let decoded = Snapshot::decode(&bytes).unwrap();
    assert_eq!(decoded.parent, Some(base_id));
    let other = Snapshot {
        parent: Some(content_id(b"another parent")),
        ..delta.clone()
    };
    assert_ne!(delta.content_id().unwrap(), other.content_id().unwrap());
}

#[test]
fn the_fixture_keeps_the_canonical_bytes() {
    let fixture = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot.safetensors"
    ));
    let decoded = Snapshot::decode(fixture).unwrap();
    assert_eq!(decoded.encode().unwrap(), fixture);
    let parent = ContentId::from_hex("1111111111111111111111111111111111111111111111111111111111111111").unwrap();
    assert_eq!(decoded.parent, Some(parent));
    let state = decoded.tensors.iter().find(|tensor| tensor.name == "state").unwrap();
    assert_eq!(state.data, (0u8..16).collect::<Vec<_>>());
    let cells = decoded.tensors.iter().find(|tensor| tensor.name == "cells").unwrap();
    assert_eq!(cells.data, (16u8..32).collect::<Vec<_>>());
    assert_eq!(cells.coverage, Some((0, 2)));
    assert_eq!(cells.block_format.as_deref(), Some("IQ4_NL"));
    let expected = ContentId::from_hex("b1012c1fd9439ba81489195014e1fd31b2227d019dfedd52a8ba72f09657a4ae").unwrap();
    assert_eq!(content_id(fixture), expected);
    assert_eq!(decoded.content_id().unwrap(), expected);
}

#[test]
fn the_recipe_is_deterministic_and_chains() {
    let identity = [1u8; 32];
    let root = recipe_root(&identity, b"geometry", &[b"approx".to_vec()], b"lab");
    assert_eq!(root, recipe_root(&identity, b"geometry", &[b"approx".to_vec()], b"lab"));
    assert_eq!(
        root.hex(),
        "668bd9fc91f16b4f7fd5bdaaffa1b4ce53e32537111956f26bb85f608cb8170a"
    );
    assert_ne!(
        root,
        recipe_root(&[2u8; 32], b"geometry", &[b"approx".to_vec()], b"lab")
    );
    let first = recipe_boundary(&root, &[1, 2, 3], 3);
    assert_eq!(first, recipe_boundary(&root, &[1, 2, 3], 3));
    assert_eq!(
        first.hex(),
        "93cf3f0db6369a63ad2c6541071ee33a0937bfcaefdeb1739c68a946cbbb00e7"
    );
    assert_ne!(first, recipe_boundary(&root, &[1, 2, 4], 3));
    let other_root = recipe_root(&[2u8; 32], b"geometry", &[b"approx".to_vec()], b"lab");
    assert_ne!(first, recipe_boundary(&other_root, &[1, 2, 3], 3));
}

fn parse_header(bytes: &[u8]) -> (usize, serde_json::Value) {
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    let value = serde_json::from_str(text.trim_end()).unwrap();
    (header_len, value)
}

fn rebuild(original: &[u8], header_len: usize, value: &serde_json::Value) -> Vec<u8> {
    let original_len = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let mut json = serde_json::to_vec(value).unwrap();
    assert!(json.len() <= header_len);
    json.resize(header_len, b' ');
    let mut out = Vec::new();
    out.extend_from_slice(&(header_len as u64).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&original[8 + original_len..]);
    out
}

#[test]
fn decode_refuses_noncanonical_files() {
    let fixture = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/snapshot.safetensors"
    ));
    let (header_len, value) = parse_header(fixture);

    // An 8-byte-aligned data start is not canonical.
    let mut json = serde_json::to_vec(&value).unwrap();
    let aligned = (8 + json.len()).next_multiple_of(8) - 8;
    json.resize(aligned, b' ');
    let mut eight = Vec::new();
    eight.extend_from_slice(&(aligned as u64).to_le_bytes());
    eight.extend_from_slice(&json);
    eight.extend_from_slice(&fixture[8 + header_len..]);
    assert!(Snapshot::decode(&eight).is_err());

    // The tensors must be in bytewise name order in the header.
    let meta = value.get("__metadata__").unwrap();
    let cells = value.get("cells").unwrap();
    let state = value.get("state").unwrap();
    let swapped = format!(
        "{{\"__metadata__\":{},\"state\":{},\"cells\":{}}}",
        serde_json::to_string(meta).unwrap(),
        serde_json::to_string(state).unwrap(),
        serde_json::to_string(cells).unwrap()
    );
    let mut header = swapped.into_bytes();
    header.resize(header_len, b' ');
    let mut unsorted = Vec::new();
    unsorted.extend_from_slice(&(header_len as u64).to_le_bytes());
    unsorted.extend_from_slice(&header);
    unsorted.extend_from_slice(&fixture[8 + header_len..]);
    assert!(Snapshot::decode(&unsorted).is_err());

    // An extra metadata key is not allowed in sealed bytes.
    let mut extra_value = value.clone();
    extra_value["__metadata__"]
        .as_object_mut()
        .unwrap()
        .insert("nagaya.owner".to_owned(), Value::String("owner".to_owned()));
    assert!(Snapshot::decode(&rebuild(fixture, header_len, &extra_value)).is_err());

    // Trailing bytes are not allowed.
    let mut trailing = fixture.to_vec();
    trailing.push(0);
    assert!(Snapshot::decode(&trailing).is_err());

    // The parent hex must be lowercase.
    let base = Snapshot {
        tensors: vec![tensor("state", PartKind::Fixed, vec![16], vec![1; 16], None)],
        parent: None,
    };
    let delta = Snapshot {
        tensors: vec![tensor("state", PartKind::Fixed, vec![16], vec![2; 16], None)],
        parent: Some(base.content_id().unwrap()),
    };
    let delta_bytes = delta.encode().unwrap();
    let (delta_len, mut delta_value) = parse_header(&delta_bytes);
    let parent = delta_value["__metadata__"]
        .as_object_mut()
        .unwrap()
        .get("nagaya.parent")
        .unwrap()
        .as_str()
        .unwrap()
        .to_uppercase();
    delta_value["__metadata__"]
        .as_object_mut()
        .unwrap()
        .insert("nagaya.parent".to_owned(), Value::String(parent));
    assert!(Snapshot::decode(&rebuild(&delta_bytes, delta_len, &delta_value)).is_err());

    // The coverage text must be canonical.
    let cells_snapshot = Snapshot {
        tensors: vec![tensor("cells", PartKind::Cells, vec![2, 8], vec![3; 16], Some((0, 2)))],
        parent: None,
    };
    let cells_bytes = cells_snapshot.encode().unwrap();
    let (cells_len, mut cells_value) = parse_header(&cells_bytes);
    cells_value["__metadata__"].as_object_mut().unwrap().insert(
        "nagaya.part.cells.coverage".to_owned(),
        Value::String("+0..02".to_owned()),
    );
    assert!(Snapshot::decode(&rebuild(&cells_bytes, cells_len, &cells_value)).is_err());
}

#[test]
fn encode_refuses_metadata_tensor() {
    let snapshot = Snapshot {
        tensors: vec![tensor("__metadata__", PartKind::Fixed, vec![1], vec![0], None)],
        parent: None,
    };
    assert!(snapshot.encode().is_err());
}

#[test]
fn a_huge_shape_is_an_error() {
    let snapshot = Snapshot {
        tensors: vec![tensor("state", PartKind::Fixed, vec![u64::MAX, 2], Vec::new(), None)],
        parent: None,
    };
    assert!(snapshot.encode().is_err());
}

#[test]
fn a_slot_id_maps_through_the_size_classes() {
    let mut engine = FakeAlways::with_slots(vec![
        SlotLimit { bytes: 64, count: 4 },
        SlotLimit { bytes: 256, count: 2 },
    ]);
    engine.pass(&[segment(3, &[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    engine.pass(&[segment(4, &[1, 2, 3, 4, 5, 6])], &[]).unwrap();
    let rows = [(1, 0, 6), (2, 2, 4)];
    let small = snapshot_from_engine(&mut engine, State::Slot(3), &rows).unwrap();
    let large = snapshot_from_engine(&mut engine, State::Slot(4), &rows).unwrap();
    assert_eq!(small.tensors.len(), 3);
    assert_eq!(large.tensors.len(), 3);
}

struct TestEngine {
    with_coverage: bool,
}

impl Engine for TestEngine {
    fn identity(&self) -> Identity {
        Identity {
            model: "test".to_owned(),
            checkpoint_digest: String::new(),
            tokenizer_hash: String::new(),
            template: String::new(),
            numbers: Default::default(),
            plan_table: String::new(),
            bit_settings: Vec::new(),
            canary_hash: String::new(),
        }
    }

    fn limits(&self) -> Limits {
        Limits {
            capabilities: if self.with_coverage {
                vec![Capability::Parts, Capability::Coverage]
            } else {
                vec![Capability::Parts]
            },
            parts: vec![PartLimits {
                name: "cells".to_owned(),
                kind: PartKind::Cells,
                dtype: Dtype::U8,
                shape: vec![8],
            }],
            inputs: Inputs::Tokens,
            devices: vec![0],
            stages: Vec::new(),
            max_segments: 1,
            max_pass_inputs: 1,
            max_logit_rows: 1,
            slots: vec![SlotLimit { bytes: 64, count: 1 }],
            max_quantum_ms: 10,
            memory: MemoryLimits::default(),
            chain: ChainLimits {
                max_depth: 0,
                delta_fraction: 0.0,
            },
            segments_exact: true,
            determinism: Determinism::Always,
            bit_dependence: Vec::new(),
            taps: Vec::new(),
        }
    }

    fn pass(&mut self, _segments: &[Segment], _taps: &[nagaya_abi::Tap]) -> Result<PassOut, PassError> {
        Ok(PassOut::default())
    }

    fn last_error(&self) -> Option<String> {
        None
    }

    fn parts(&mut self) -> Option<&mut dyn Parts> {
        Some(self)
    }

    fn coverage(&mut self) -> Option<&mut dyn Coverage> {
        if self.with_coverage { Some(self) } else { None }
    }
}

impl Parts for TestEngine {
    fn part_read(&mut self, _state: State, _part: usize, _out: &mut [u8]) -> Result<(), PartError> {
        Err(PartError::Refused)
    }

    fn part_write(&mut self, _state: State, _part: usize, _bytes: &[u8]) -> WriteResult {
        WriteResult::Refused
    }

    fn cells_read(
        &mut self,
        _state: State,
        _part: usize,
        _from: u64,
        _n: u64,
        out: &mut [u8],
    ) -> Result<(), PartError> {
        for (index, byte) in out.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        Ok(())
    }

    fn cells_write(&mut self, _state: State, _part: usize, _n: u64, _bytes: &[u8]) -> WriteResult {
        WriteResult::Refused
    }

    fn set_n(&mut self, _state: State, _n: u64) -> WriteResult {
        WriteResult::Refused
    }
}

impl Coverage for TestEngine {
    fn coverage(&self, _state: State, part: usize, _bytes: &[u8]) -> CoverageSpan {
        CoverageSpan { part, start: 3, end: 5 }
    }

    fn fits(&self, _coverage: &CoverageSpan, _slot: u32) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn coverage_capability_sets_the_span() {
    let mut engine = TestEngine { with_coverage: true };
    let rows = [(0, 0, 2)];
    let snapshot = snapshot_from_engine(&mut engine, State::Slot(0), &rows).unwrap();
    assert_eq!(snapshot.tensors[0].coverage, Some((3, 5)));
    assert_eq!(snapshot.tensors[0].shape, vec![2, 8]);
}

#[test]
fn no_coverage_records_zero_to_n() {
    let mut engine = TestEngine { with_coverage: false };
    let rows = [(0, 0, 2)];
    let snapshot = snapshot_from_engine(&mut engine, State::Slot(0), &rows).unwrap();
    assert_eq!(snapshot.tensors[0].coverage, Some((0, 2)));
}
