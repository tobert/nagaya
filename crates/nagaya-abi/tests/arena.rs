//! The arena capability of the fake engine (engine-abi "Arena calls in v0", spec 10.2).

use std::fs::File;
use std::os::unix::fs::{FileExt, MetadataExt};

use nagaya_abi::fake::{FAKE_ARENA_PART, FakeAlways};
use nagaya_abi::{ArenaFd, ArenaPart, Capability, CheckStrength, Engine};

fn bytes_of(fd: &ArenaFd) -> Vec<u8> {
    let file = File::from(fd.fd.try_clone().unwrap());
    let size = file.metadata().unwrap().len() as usize;
    let mut out = vec![0u8; size];
    file.read_exact_at(&mut out, 0).unwrap();
    out
}

fn inode(fd: &ArenaFd) -> u64 {
    File::from(fd.fd.try_clone().unwrap()).metadata().unwrap().ino()
}

#[test]
fn a_fake_with_an_arena_exports_one_fd_that_holds_its_bytes() {
    let mut engine = FakeAlways::with_arena(b"weights one");
    assert!(engine.limits().capabilities.contains(&Capability::Arena));
    let fds = engine
        .arena()
        .expect("the fake has the arena capability")
        .arena_export()
        .unwrap();
    assert_eq!(fds.len(), 1);
    assert_eq!(fds[0].device, 0);
    assert_eq!(fds[0].part, FAKE_ARENA_PART);
    assert_eq!(bytes_of(&fds[0]), b"weights one");
    assert_eq!(
        FakeAlways::arena_parts(),
        vec![ArenaPart {
            device: 0,
            part: FAKE_ARENA_PART.to_owned()
        }]
    );
}

#[test]
fn an_attach_with_the_same_bytes_has_a_full_check_and_keeps_the_stored_object() {
    let mut first = FakeAlways::with_arena(b"weights one");
    let stored = first.arena().unwrap().arena_export().unwrap();
    let stored_inode = inode(&stored[0]);
    let (mut second, check) = FakeAlways::attach_arena(stored, b"weights one").unwrap();
    assert_eq!(check, CheckStrength::Full);
    let again = second.arena().unwrap().arena_export().unwrap();
    assert_eq!(
        inode(&again[0]),
        stored_inode,
        "the attach uses the stored memory, not a copy"
    );
}

#[test]
fn an_attach_with_other_bytes_is_a_mismatch() {
    let mut first = FakeAlways::with_arena(b"weights one");
    let stored = first.arena().unwrap().arena_export().unwrap();
    let error = FakeAlways::attach_arena(stored, b"weights two").err().unwrap();
    assert!(error.contains("differs"), "{error}");
}

#[test]
fn an_attach_with_no_fd_or_another_part_is_a_mismatch() {
    assert!(FakeAlways::attach_arena(Vec::new(), b"w").is_err());
    let mut first = FakeAlways::with_arena(b"w");
    let mut stored = first.arena().unwrap().arena_export().unwrap();
    stored[0].part = "head".to_owned();
    let error = FakeAlways::attach_arena(stored, b"w").err().unwrap();
    assert!(error.contains("head"), "{error}");
}

#[test]
fn a_fake_with_no_arena_has_no_arena_capability() {
    let mut engine = FakeAlways::new();
    assert!(!engine.limits().capabilities.contains(&Capability::Arena));
    assert!(engine.arena().is_none());
}
