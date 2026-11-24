//! The memo and the cache file of the digest of the checkpoint, on a small checkpoint that each test writes in a
//! temporary folder. No test reads the real cache folder.

use std::cell::Cell;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use nagaya_mk_bmg::safetensors::SafeTensorsView;
use nagaya_mk_inkling2_sm121::digest_cache::{Memo, cache_dir, cached_digest};
use nagaya_mk_inkling2_sm121::identity::checkpoint_digest_with;

const CONFIG: &str = r#"{"a": 1}"#;

fn bytes(seed: u8) -> Vec<u8> {
    (0..4096u32)
        .map(|i| (i as u8).wrapping_mul(seed).wrapping_add(seed))
        .collect()
}

fn write_checkpoint(dir: &Path, seed: u8) -> PathBuf {
    let data = bytes(seed);
    let entries: Vec<(&str, &str, &[usize], &[u8])> = vec![("model.llm.a", "U8", &[4096], &data)];
    let file = dir.join("model.safetensors");
    std::fs::write(&file, SafeTensorsView::build_synthetic_safetensors(&entries)).unwrap();
    file
}

/// A test folder: the checkpoint, and a folder for the cache files.
struct Setup {
    _root: tempfile::TempDir,
    weights: PathBuf,
    cache: PathBuf,
}

fn setup() -> Setup {
    let root = tempfile::tempdir().unwrap();
    let weights = write_checkpoint(root.path(), 3);
    let cache = root.path().join("cache").join("nagaya").join("digests");
    Setup {
        _root: root,
        weights,
        cache,
    }
}

/// The digest through the cache with a fresh memo, and the count of reads of the whole file.
fn digest(setup: &Setup, config: &str, reads: &Cell<u32>) -> String {
    cached_digest(&Memo::default(), Some(&setup.cache), config, &setup.weights, || {
        reads.set(reads.get() + 1);
        checkpoint_digest_with(config, &setup.weights, 2)
    })
    .unwrap()
}

/// Let the clock of the file system move, so that a later write has another `ctime`.
fn tick() {
    std::thread::sleep(Duration::from_millis(30));
}

#[test]
fn a_hit_gives_the_digest_of_a_full_read_and_reads_nothing() {
    let setup = setup();
    let reads = Cell::new(0);
    let first = digest(&setup, CONFIG, &reads);
    assert_eq!(first, checkpoint_digest_with(CONFIG, &setup.weights, 1).unwrap());
    assert_eq!(reads.get(), 1);
    let second = digest(&setup, CONFIG, &reads);
    assert_eq!(
        (second, reads.get()),
        (first, 1),
        "a fresh memo reads the file of the cache"
    );
}

#[test]
fn the_memo_serves_a_second_call_with_no_cache_folder() {
    let setup = setup();
    let memo = Memo::default();
    let reads = Cell::new(0);
    let call = || {
        cached_digest(&memo, None, CONFIG, &setup.weights, || {
            reads.set(reads.get() + 1);
            checkpoint_digest_with(CONFIG, &setup.weights, 1)
        })
        .unwrap()
    };
    assert_eq!(call(), call());
    assert_eq!(reads.get(), 1);
}

#[test]
fn a_touch_of_the_file_misses() {
    let setup = setup();
    let reads = Cell::new(0);
    let first = digest(&setup, CONFIG, &reads);
    tick();
    let file = std::fs::File::options().write(true).open(&setup.weights).unwrap();
    file.set_modified(SystemTime::now() + Duration::from_secs(10)).unwrap();
    assert_eq!(digest(&setup, CONFIG, &reads), first, "the bytes did not change");
    assert_eq!(reads.get(), 2, "a new mtime and a new ctime make a miss");
}

#[test]
fn a_change_of_the_config_text_misses() {
    let setup = setup();
    let reads = Cell::new(0);
    let first = digest(&setup, CONFIG, &reads);
    let other = r#"{"a": 2}"#;
    let second = digest(&setup, other, &reads);
    assert_eq!(reads.get(), 2);
    assert_ne!(first, second, "the config is part of the digest");
    assert_eq!(second, checkpoint_digest_with(other, &setup.weights, 1).unwrap());
}

#[test]
fn a_rewrite_of_the_same_size_with_the_old_mtime_misses() {
    let setup = setup();
    let reads = Cell::new(0);
    let first = digest(&setup, CONFIG, &reads);
    let old = std::fs::metadata(&setup.weights).unwrap().modified().unwrap();
    tick();
    let before = std::fs::metadata(&setup.weights).unwrap().len();
    write_checkpoint(setup.weights.parent().unwrap(), 5);
    assert_eq!(
        std::fs::metadata(&setup.weights).unwrap().len(),
        before,
        "the size must not change"
    );
    std::fs::File::options()
        .write(true)
        .open(&setup.weights)
        .unwrap()
        .set_modified(old)
        .unwrap();
    let second = digest(&setup, CONFIG, &reads);
    assert_ne!(first, second);
    assert_eq!(second, checkpoint_digest_with(CONFIG, &setup.weights, 1).unwrap());
    assert_eq!(reads.get(), 2, "only the ctime tells this rewrite");
}

#[test]
fn a_cache_file_that_is_wrong_gives_a_recompute_and_a_new_file() {
    let setup = setup();
    let reads = Cell::new(0);
    let first = digest(&setup, CONFIG, &reads);
    let file = std::fs::read_dir(&setup.cache)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().is_some_and(|ext| ext == "json"))
        .expect("the cache file");
    let good = std::fs::read_to_string(&file).unwrap();
    let wrong = [
        "not json at all".to_owned(),
        String::new(),
        good.replace("blake3:", "blake3-"),
        good.replace("nagaya inkling2 checkpoint v1", "nagaya inkling2 checkpoint v0"),
        good.replace(&first[7..], &"0".repeat(63)),
    ];
    for (index, text) in wrong.iter().enumerate() {
        std::fs::write(&file, text).unwrap();
        let before = reads.get();
        assert_eq!(digest(&setup, CONFIG, &reads), first, "wrong file {index}");
        assert_eq!(reads.get(), before + 1, "wrong file {index} must miss");
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            good,
            "the miss writes the file again"
        );
    }
}

#[test]
fn a_cache_folder_that_cannot_be_written_still_gives_the_digest() {
    let setup = setup();
    // A folder below a plain file cannot be made.
    let blocker = setup.weights.parent().unwrap().join("blocker");
    std::fs::write(&blocker, b"x").unwrap();
    let bad = blocker.join("digests");
    let reads = Cell::new(0);
    let got = cached_digest(&Memo::default(), Some(&bad), CONFIG, &setup.weights, || {
        reads.set(reads.get() + 1);
        checkpoint_digest_with(CONFIG, &setup.weights, 1)
    })
    .unwrap();
    assert_eq!(got, checkpoint_digest_with(CONFIG, &setup.weights, 1).unwrap());
    assert_eq!(reads.get(), 1);
}

#[test]
fn a_file_that_changes_during_the_read_is_not_kept() {
    let setup = setup();
    let reads = Cell::new(0);
    tick();
    let during = cached_digest(&Memo::default(), Some(&setup.cache), CONFIG, &setup.weights, || {
        reads.set(reads.get() + 1);
        let digest = checkpoint_digest_with(CONFIG, &setup.weights, 1);
        tick();
        write_checkpoint(setup.weights.parent().unwrap(), 9);
        digest
    })
    .unwrap();
    let after = digest(&setup, CONFIG, &reads);
    assert_eq!(reads.get(), 2, "the digest of a changing file is not cached");
    assert_ne!(during, after);
    assert_eq!(after, checkpoint_digest_with(CONFIG, &setup.weights, 1).unwrap());
}

#[test]
fn a_missing_weights_file_is_an_error_and_not_a_cached_digest() {
    let setup = setup();
    let missing = setup.weights.with_file_name("absent.safetensors");
    let error = cached_digest(&Memo::default(), Some(&setup.cache), CONFIG, &missing, || {
        Ok("blake3:x".to_owned())
    })
    .unwrap_err();
    assert!(error.contains("absent.safetensors"), "{error}");
}

#[test]
fn the_cache_folder_follows_xdg_then_home_and_ignores_an_empty_variable() {
    let os = |text: &str| Some(OsString::from(text));
    assert_eq!(cache_dir(os("/x"), os("/h")), Some(PathBuf::from("/x/nagaya/digests")));
    assert_eq!(
        cache_dir(None, os("/h")),
        Some(PathBuf::from("/h/.cache/nagaya/digests"))
    );
    assert_eq!(
        cache_dir(os(""), os("/h")),
        Some(PathBuf::from("/h/.cache/nagaya/digests"))
    );
    assert_eq!(cache_dir(None, None), None);
    assert_eq!(cache_dir(os(""), os("")), None);
}
