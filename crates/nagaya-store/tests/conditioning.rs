//! The conditionings of the image engines in the store (spec 6.16). These tests need no GPU.

use nagaya_abi::{ChainLimits, Determinism};
use nagaya_store::conditioning;
use nagaya_store::{EngineFacts, Put, Reason, Store, Tier};

const ROW: usize = 8;

fn digest(seed: u8) -> [u8; 32] {
    [seed; 32]
}

fn facts() -> EngineFacts {
    EngineFacts {
        name: "qwen-image".to_owned(),
        determinism: Determinism::Pinned,
        bit_dependence: Vec::new(),
        chain: ChainLimits {
            max_depth: 0,
            delta_fraction: 0.0,
        },
    }
}

fn store(dir: &std::path::Path) -> Store {
    Store::open_with_limits(dir, 1 << 20, 1 << 20, u64::MAX, 0).unwrap()
}

/// Put the conditioning of `ids` (cut 2) under `digest`, as the job of an image engine puts it.
fn put(store: &mut Store, digest: &[u8; 32], ids: &[u32], rows: &[u8]) {
    let snapshot = conditioning::snapshot(rows, ROW, 2, ids.len() as u64).unwrap();
    store
        .put(Put {
            snapshot: &snapshot,
            recipe: conditioning::recipe(digest, ids),
            previous_recipe: None,
            chains_from: None,
            engine: facts(),
            identity: [9; 32],
            owner: 1,
            tier: Tier::Host,
            reason: Reason::PrefillDone,
            sealed_recipe: None,
            provenance: None,
            now_ms: 10,
        })
        .unwrap();
}

#[test]
fn the_recipe_names_the_encoder_and_each_id() {
    let ids = [1u32, 2, 3, 4];
    let r = conditioning::recipe(&digest(1), &ids);
    assert_eq!(r, conditioning::recipe(&digest(1), &ids), "the same encoder and ids");
    assert_ne!(r, conditioning::recipe(&digest(2), &ids), "another encoder");
    assert_ne!(r, conditioning::recipe(&digest(1), &[1, 2, 3, 5]), "another id");
    assert_ne!(r, conditioning::recipe(&digest(1), &[1, 2, 3]), "fewer ids");
}

#[test]
fn a_put_conditioning_comes_back_by_its_recipe() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = store(dir.path());
    let ids = [7u32, 8, 9, 10, 11];
    let rows: Vec<u8> = (0..3 * ROW as u8).collect();
    put(&mut store, &digest(1), &ids, &rows);
    let (_, snapshot) = store
        .find(conditioning::recipe(&digest(1), &ids), 20)
        .unwrap()
        .expect("a hit");
    assert_eq!(conditioning::rows(&snapshot, ROW, 3).unwrap(), rows);
    assert!(
        store
            .find(conditioning::recipe(&digest(1), &[7, 8, 9, 10]), 20)
            .unwrap()
            .is_none(),
        "other ids"
    );
    assert!(
        store
            .find(conditioning::recipe(&digest(2), &ids), 20)
            .unwrap()
            .is_none(),
        "another encoder"
    );
    let entry = store.entries().values().next().unwrap();
    assert_eq!(
        (entry.hits, entry.last_used_ms, entry.tier),
        (1, 20, Tier::Host),
        "the hit counts"
    );
}

#[test]
fn a_snapshot_that_is_not_a_conditioning_is_refused() {
    assert!(
        conditioning::snapshot(&[0; 24], ROW, 2, 4).is_err(),
        "3 rows for 2 positions"
    );
    assert!(conditioning::snapshot(&[0; 16], 7, 2, 4).is_err(), "an odd row");
    let snapshot = conditioning::snapshot(&[0; 16], ROW, 2, 4).unwrap();
    assert!(conditioning::rows(&snapshot, ROW, 3).is_err(), "the wrong row count");
    assert!(conditioning::rows(&snapshot, 4, 4).is_err(), "the wrong width");
    assert_eq!(conditioning::rows(&snapshot, ROW, 2).unwrap(), vec![0; 16]);
}

#[test]
fn an_encoder_digest_is_blake3_and_64_hex_digits() {
    let text = format!("blake3:{}", "ab".repeat(32));
    assert_eq!(conditioning::digest_bytes(&text).unwrap(), [0xab; 32]);
    for bad in [
        "ab".repeat(32),
        format!("sha256:{}", "ab".repeat(32)),
        format!("blake3:{}", "ab".repeat(31)),
        format!("blake3:{}", "AB".repeat(32)),
        format!("blake3:{}g", "a".repeat(63)),
    ] {
        assert!(conditioning::digest_bytes(&bad).is_err(), "{bad}");
    }
}
