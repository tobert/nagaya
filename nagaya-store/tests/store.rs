use nagaya_abi::fake::FakeAlways;
use nagaya_abi::{Engine, OutputsWanted, PartKind, Segment, SegmentInputs, State};
use nagaya_store::{ContentId, Snapshot, Tensor, content_id, recipe_boundary, recipe_root, snapshot_from_engine};

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
    let expected = ContentId::from_hex("f92f9e355291b7a21db577ed995e2a84d5ab1fee8142bb91ae58e38ce7e67be5").unwrap();
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
