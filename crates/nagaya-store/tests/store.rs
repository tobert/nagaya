use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{
    Capability, ChainLimits, Coverage, CoverageSpan, Determinism, Dtype, Engine, Identity, Inputs, Limits,
    MemoryLimits, OutputsWanted, PartError, PartKind, PartLimits, Parts, PassError, PassOut, Segment, SegmentInputs,
    SlotLimit, State, WriteResult,
};
use nagaya_store::{
    ContentId, RecipeMedia, Snapshot, Tensor, content_id, recipe_boundary, recipe_boundary_media, recipe_decoded,
    recipe_not_sealed, recipe_root, snapshot_from_engine,
};
use serde_json::Value;

fn segment(slot: u32, tokens: &[u32]) -> Segment {
    Segment {
        control: Vec::new(),
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

    let names: Vec<&str> = small.tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    assert_eq!(names, ["state", "cells", "ring"]);
    let kinds: Vec<PartKind> = small.tensors.iter().map(|tensor| tensor.kind).collect();
    assert_eq!(kinds, [PartKind::Fixed, PartKind::Cells, PartKind::Ring]);
    assert_eq!(small.tensors[0].shape, vec![16]);
    assert_eq!(small.tensors[1].shape, vec![6, 8]);
    assert_eq!(small.tensors[2].shape, vec![4, 8]);
    assert_eq!(small.tensors[0].coverage, None);
    assert_eq!(small.tensors[1].coverage, Some((0, 6)));
    assert_eq!(small.tensors[2].coverage, Some((2, 6)));
    assert_eq!(small.tensors[0].data.len(), 16);
    assert_eq!(small.tensors[1].data.len(), 48);
    let expected_ring: Vec<u8> = [3u8, 4, 5, 6]
        .into_iter()
        .flat_map(|token| [token, 0, 0, 0, token, 0, 0, 0])
        .collect();
    assert_eq!(small.tensors[2].data, expected_ring);

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
    assert_eq!(
        root.hex(),
        "668bd9fc91f16b4f7fd5bdaaffa1b4ce53e32537111956f26bb85f608cb8170a"
    );
    assert_ne!(
        root,
        recipe_root(&[2u8; 32], b"geometry", &[b"approx".to_vec()], b"lab")
    );
    assert_ne!(
        root,
        recipe_root(&identity, b"other geometry", &[b"approx".to_vec()], b"lab")
    );
    assert_ne!(root, recipe_root(&identity, b"geometry", &[b"other".to_vec()], b"lab"));
    assert_ne!(
        root,
        recipe_root(&identity, b"geometry", &[b"approx".to_vec()], b"other lab")
    );
    let first = recipe_boundary(&root, &[1, 2, 3], 3);
    assert_eq!(
        first.hex(),
        "93cf3f0db6369a63ad2c6541071ee33a0937bfcaefdeb1739c68a946cbbb00e7"
    );
    assert_ne!(first, recipe_boundary(&root, &[1, 2, 4], 3));
    assert_ne!(first, recipe_boundary(&root, &[1, 2, 3], 4));
    let other_root = recipe_root(&[2u8; 32], b"geometry", &[b"approx".to_vec()], b"lab");
    assert_ne!(first, recipe_boundary(&other_root, &[1, 2, 3], 3));
}

#[test]
fn the_not_sealed_item_changes_the_recipe() {
    let root = recipe_root(&[1u8; 32], b"geometry", &[b"approx".to_vec()], b"lab");
    let sealed = recipe_boundary(&root, &[1, 2, 3], 3);
    let copy = recipe_not_sealed(&sealed);
    // Fixed vector: BLAKE3 derive-key "nagaya recipe v1" over the item 5 (the sealed recipe) and the item 8
    // (no bytes), made apart from this crate.
    assert_eq!(
        copy.hex(),
        "3cd303f01cc295d55ef3f02d148f59f78cda1e0c8393374e4868bc648e847de0"
    );
    assert_ne!(
        copy, sealed,
        "a copy never has the recipe of the sealed snapshot at its position"
    );
    assert_ne!(copy, recipe_not_sealed(&recipe_boundary(&root, &[1, 2, 4], 3)));
    assert_ne!(copy, recipe_not_sealed(&recipe_boundary(&root, &[1, 2, 3], 4)));
    // The copy of the root differs from the root.
    assert_ne!(recipe_not_sealed(&root), root);
}

/// The recipe of a decoded head has the items 5, 6, 7 and 10 (spec 6.2, 6.14). It is never the recipe of the same
/// tokens that the feed rule fed.
#[test]
fn the_decoded_item_changes_the_recipe() {
    let root = recipe_root(&[1u8; 32], b"", &[], &[]);
    let prompt = recipe_boundary(&root, &[1, 2, 3], 3);
    let tokens = [7u32, 8, 9];
    let head = recipe_decoded(&prompt, &tokens, 6);
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let expected = items_hash(&[
        (5, unhex(&prompt.hex())),
        (6, token_bytes),
        (7, 6u64.to_le_bytes().to_vec()),
        (10, Vec::new()),
    ]);
    assert_eq!(head.hex(), expected);
    assert_ne!(
        head,
        recipe_boundary(&prompt, &tokens, 6),
        "a decoded head never has the recipe of the same tokens fed by the rule"
    );
    assert_ne!(head, recipe_decoded(&prompt, &[7, 8, 10], 6));
    assert_ne!(head, recipe_decoded(&prompt, &tokens, 7));
    assert_ne!(head, recipe_decoded(&root, &tokens, 6));
    // A head with no decoded token differs from the recipe of its prompt.
    assert_ne!(recipe_decoded(&prompt, &[], 3), prompt);
}

/// The hash of an item list, written apart from the crate: each item is a `u16` tag, a `u64` length and its bytes.
fn items_hash(items: &[(u16, Vec<u8>)]) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("nagaya recipe v1");
    for (tag, bytes) in items {
        hasher.update(&tag.to_le_bytes());
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    hasher.finalize().to_hex().to_string()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// A media part enters the recipe of its range as tag 9, after tag 7 (spec 6.2, 11.11 "Media through the council").
#[test]
fn a_media_part_enters_the_recipe_as_tag_9() {
    let root = recipe_root(&[1u8; 32], b"", &[], &[]);
    let tokens = [5u32, 9, 9, 9, 6];
    let image = ContentId::from_hex(&"07".repeat(32)).unwrap();
    let media = [RecipeMedia {
        mime: "image/png",
        id: &image,
        positions: 3,
    }];
    let recipe = recipe_boundary_media(&root, &tokens, 5, &media);
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let mut item = (9u64).to_le_bytes().to_vec();
    item.extend(b"image/png");
    item.extend([7u8; 32]);
    item.extend(3u64.to_le_bytes());
    let expected = items_hash(&[
        (5, unhex(&root.hex())),
        (6, token_bytes),
        (7, 5u64.to_le_bytes().to_vec()),
        (9, item),
    ]);
    assert_eq!(recipe.hex(), expected);
    // With no media part, the recipe is the recipe of the tokens alone, so a text model keeps its recipes.
    assert_eq!(
        recipe_boundary_media(&root, &tokens, 5, &[]),
        recipe_boundary(&root, &tokens, 5)
    );
    assert_ne!(recipe, recipe_boundary(&root, &tokens, 5));
    // Another object, another media type, or another count gives another recipe.
    let other = ContentId::from_hex(&"08".repeat(32)).unwrap();
    for changed in [
        RecipeMedia { id: &other, ..media[0] },
        RecipeMedia {
            mime: "image/jpeg",
            ..media[0]
        },
        RecipeMedia {
            positions: 4,
            ..media[0]
        },
    ] {
        assert_ne!(
            recipe_boundary_media(&root, &tokens, 5, &[changed]),
            recipe,
            "{changed:?}"
        );
    }
    // Two parts in the other order give another recipe.
    let two = [media[0], RecipeMedia { id: &other, ..media[0] }];
    let swapped = [two[1], two[0]];
    assert_ne!(
        recipe_boundary_media(&root, &tokens, 5, &two),
        recipe_boundary_media(&root, &tokens, 5, &two[..1]),
        "each part is an item"
    );
    assert_ne!(
        recipe_boundary_media(&root, &tokens, 5, &two),
        recipe_boundary_media(&root, &tokens, 5, &swapped)
    );
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
        SlotLimit {
            bytes: 64,
            count: 4,
            tokens: 4096,
        },
        SlotLimit {
            bytes: 256,
            count: 2,
            tokens: 4096,
        },
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
    dtype: Dtype,
    block_format: Option<String>,
}

impl TestEngine {
    fn new(with_coverage: bool) -> Self {
        TestEngine {
            with_coverage,
            dtype: Dtype::U8,
            block_format: None,
        }
    }
}

impl Engine for TestEngine {
    fn identity(&self) -> Identity {
        Identity {
            model: "test".to_owned(),
            checkpoint_digest: String::new(),
            family: "fake".to_owned(),
            tokenizer_hash: String::new(),
            template: String::new(),
            numbers: Default::default(),
            plan_table: String::new(),
            bit_settings: Vec::new(),
            canary_hash: String::new(),
            build: std::collections::BTreeMap::new(),
        }
    }

    fn limits(&self) -> Limits {
        Limits {
            capabilities: if self.with_coverage {
                vec![Capability::Parts, Capability::Coverage]
            } else {
                vec![Capability::Parts]
            },
            parts: vec![
                PartLimits {
                    name: "cells".to_owned(),
                    kind: PartKind::Cells,
                    dtype: self.dtype,
                    shape: vec![8],
                    block_format: self.block_format.clone(),
                    stage: 0,
                },
                PartLimits {
                    name: "fixed".to_owned(),
                    kind: PartKind::Fixed,
                    dtype: self.dtype,
                    shape: vec![4],
                    block_format: self.block_format.clone(),
                    stage: 0,
                },
            ],
            inputs: Inputs::Tokens,
            devices: vec![0],
            stages: Vec::new(),
            max_segments: 1,
            max_pass_inputs: 1,
            max_logit_rows: 1,
            slots: vec![SlotLimit {
                bytes: 64,
                count: 1,
                tokens: 4096,
            }],
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
            checkpoint_grain: None,
            feed: None,
            residual_row_bytes: 0,
            control: None,
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
    fn part_read(&mut self, _state: State, _part: usize, out: &mut [u8]) -> Result<(), PartError> {
        out.fill(7);
        Ok(())
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

    fn cells_write(&mut self, _state: State, _part: usize, _from: u64, _n: u64, _bytes: &[u8]) -> WriteResult {
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
    let mut engine = TestEngine::new(true);
    let rows = [(0, 0, 2)];
    let snapshot = snapshot_from_engine(&mut engine, State::Slot(0), &rows).unwrap();
    assert_eq!(snapshot.tensors[0].coverage, Some((3, 5)));
    assert_eq!(snapshot.tensors[0].shape, vec![2, 8]);
}

#[test]
fn no_coverage_records_zero_to_n() {
    let mut engine = TestEngine::new(false);
    let rows = [(0, 0, 2)];
    let snapshot = snapshot_from_engine(&mut engine, State::Slot(0), &rows).unwrap();
    assert_eq!(snapshot.tensors[0].coverage, Some((0, 2)));
}

#[test]
fn a_block_format_on_a_u8_part_round_trips() {
    let mut engine = TestEngine::new(false);
    engine.block_format = Some("IQ4_NL".to_owned());
    let snapshot = snapshot_from_engine(&mut engine, State::Slot(0), &[(0, 0, 2)]).unwrap();
    let back = Snapshot::decode(&snapshot.encode().unwrap()).unwrap();
    for tensors in [&snapshot.tensors, &back.tensors] {
        assert_eq!(tensors.len(), 2);
        for tensor in tensors {
            assert_eq!(tensor.block_format.as_deref(), Some("IQ4_NL"), "{}", tensor.name);
        }
    }
}

#[test]
fn a_part_with_no_block_format_stays_without_one() {
    let mut engine = TestEngine::new(false);
    let snapshot = snapshot_from_engine(&mut engine, State::Slot(0), &[(0, 0, 2)]).unwrap();
    assert!(snapshot.tensors.iter().all(|tensor| tensor.block_format.is_none()));
}

#[test]
fn a_block_format_on_another_dtype_is_refused() {
    let mut engine = TestEngine::new(false);
    engine.dtype = Dtype::F32;
    engine.block_format = Some("IQ4_NL".to_owned());
    assert!(snapshot_from_engine(&mut engine, State::Slot(0), &[(0, 0, 2)]).is_err());
}

/// One pass over `tokens` at `from`, with the logit row of the last one.
fn feed_at(engine: &mut FakeAlways, slot: u32, from: u64, tokens: &[u32]) -> Vec<u8> {
    let last = from + tokens.len() as u64 - 1;
    let out = engine
        .pass(
            &[Segment {
                control: Vec::new(),
                state: State::Slot(slot),
                inputs: SegmentInputs::Tokens(tokens.to_vec()),
                positions: (from..=last).collect(),
                meta: Vec::new(),
                outputs: OutputsWanted { logit_rows: vec![last] },
            }],
            &[],
        )
        .unwrap();
    out.outputs
        .into_iter()
        .find(|output| output.name == "logits")
        .unwrap()
        .bytes
}

#[test]
fn written_snapshots_put_a_state_back_in_a_slot() {
    let tokens: Vec<u32> = (1..=20).collect();
    let mut engine = FakeAlways::with_slots(vec![SlotLimit {
        bytes: 16384,
        count: 1,
        tokens: 4096,
    }])
    .with_vocab(16);
    feed_at(&mut engine, 0, 0, &tokens);
    let snapshot = snapshot_from_engine(&mut engine, State::Slot(0), &[(1, 0, 20), (2, 16, 4)]).unwrap();
    // The slot moves on, and then it is cleared.
    feed_at(&mut engine, 0, 20, &[7, 7, 7]);
    assert_eq!(engine.slots().unwrap().slot_clear(0), WriteResult::Ok);
    let written = nagaya_store::write_snapshots(&mut engine, State::Slot(0), std::slice::from_ref(&snapshot)).unwrap();
    assert_eq!(written, nagaya_store::Restored::Done);
    assert_eq!(engine.slots().unwrap().slot_high(0), 20);
    let mut straight = FakeAlways::with_slots(vec![SlotLimit {
        bytes: 16384,
        count: 1,
        tokens: 4096,
    }])
    .with_vocab(16);
    feed_at(&mut straight, 0, 0, &tokens);
    assert_eq!(feed_at(&mut engine, 0, 20, &[9]), feed_at(&mut straight, 0, 20, &[9]));
}
