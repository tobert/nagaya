//! The hash of a weight file: the hash of the hashes of its chunks, in order. These tests need no GPU.

use nagaya_mk_qwenimage21_gfx1151::digest::file_hash_in;

/// The plain form: each chunk in order, on one thread.
fn reference(bytes: &[u8], chunk: usize) -> blake3::Hash {
    let mut all = blake3::Hasher::new();
    if bytes.is_empty() {
        all.update(blake3::hash(&[]).as_bytes());
    }
    for part in bytes.chunks(chunk) {
        all.update(blake3::hash(part).as_bytes());
    }
    all.finalize()
}

#[test]
fn the_hash_of_four_workers_is_the_hash_of_the_chunks_in_order() {
    let dir = tempfile::tempdir().unwrap();
    // No chunk, part of one, exactly two, more chunks than workers with a short last one.
    for size in [0usize, 700, 2000, 9_999, 23_001] {
        let bytes: Vec<u8> = (0..size).map(|i| (i * 31 % 251) as u8).collect();
        let path = dir.path().join(format!("f{size}"));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(
            file_hash_in(&path, 1000).unwrap(),
            reference(&bytes, 1000),
            "{size} bytes"
        );
    }
}

#[test]
fn a_file_that_cannot_be_read_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    assert!(file_hash_in(&dir.path().join("absent"), 1000).is_err());
}
