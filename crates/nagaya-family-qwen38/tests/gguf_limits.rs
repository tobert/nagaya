//! The GGUF reader never allocates more than the bytes that it has read can fill, and it checks keys in linear time.
//!
//! A damaged file can claim any length. These tests give the reader a small stream and a large declared length, and
//! they watch the largest single allocation that the reading thread asks for.

use nagaya_family_qwen38::Error;
use nagaya_family_qwen38::gguf::Metadata;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

thread_local! {
    static WATCHING: Cell<bool> = const { Cell::new(false) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

/// Passes each request to the system allocator. On a watched thread, it records the largest single request.
struct Watcher;

unsafe impl GlobalAlloc for Watcher {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn record(size: usize) {
    if WATCHING.with(Cell::get) {
        LARGEST.with(|l| l.set(l.get().max(size)));
    }
}

#[global_allocator]
static ALLOCATOR: Watcher = Watcher;

/// Run `f` and give its result with the largest single allocation that it asked for.
fn watched<T>(f: impl FnOnce() -> T) -> (T, usize) {
    LARGEST.with(|l| l.set(0));
    WATCHING.with(|w| w.set(true));
    let result = f();
    WATCHING.with(|w| w.set(false));
    (result, LARGEST.with(Cell::get))
}

const GIB: u64 = 1 << 30;
/// No single allocation of a read of a small stream may be larger than this.
const SMALL: usize = 1 << 20;

fn string(s: &str) -> Vec<u8> {
    let mut out = (s.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(s.as_bytes());
    out
}

fn header(kv_count: u64) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&kv_count.to_le_bytes());
    out
}

/// A file with one array. `count` is what the array claims. The bytes end after the claim.
fn array_file(element_type: u32, count: u64) -> Vec<u8> {
    let mut f = header(1);
    f.extend(string("a"));
    f.extend_from_slice(&9u32.to_le_bytes());
    f.extend_from_slice(&element_type.to_le_bytes());
    f.extend_from_slice(&count.to_le_bytes());
    f
}

/// A file with one string. `claim` is the length that it declares. The bytes end after a few characters.
fn string_file(claim: u64) -> Vec<u8> {
    let mut f = header(1);
    f.extend(string("s"));
    f.extend_from_slice(&8u32.to_le_bytes());
    f.extend_from_slice(&claim.to_le_bytes());
    f.extend_from_slice(b"short");
    f
}

fn gguf_message(result: Result<Metadata, Error>) -> String {
    match result {
        Err(Error::Gguf(m)) => m,
        other => panic!("expected a Gguf error, got {other:?}"),
    }
}

#[test]
fn an_array_count_that_the_stream_cannot_fill_is_an_error_with_no_big_allocation() {
    // 2^35 values of one byte fit a declared file of 2^40 bytes. Each value takes about 32 bytes in memory, so a vector of that size is 1 TiB.
    let file = array_file(0, 1 << 35);
    let (result, largest) = watched(|| Metadata::read(file.as_slice(), 1 << 40));
    let message = gguf_message(result);
    assert!(
        largest < SMALL,
        "the reader asked for one allocation of {largest} bytes"
    );
    assert!(message.contains("read failed"), "{message}");
}

#[test]
fn an_array_that_the_file_fills_reads_and_takes_memory_in_proportion() {
    // 100000 bytes of values take 3.2 MB in memory. The vector may double, so 8 MB is the most that one request takes.
    let mut f = array_file(0, 100_000);
    f.resize(f.len() + 100_000, 7);
    let (result, largest) = watched(|| Metadata::read(f.as_slice(), f.len() as u64));
    let m = result.unwrap();
    assert!(matches!(m.get("a"), Some(nagaya_family_qwen38::gguf::Value::Array(a)) if a.len() == 100_000));
    assert!(
        largest <= 8 << 20,
        "the reader asked for one allocation of {largest} bytes"
    );
}

#[test]
fn a_large_string_length_is_an_error_before_the_allocation() {
    // 8 MiB is below the cap, and the declared file is 1 TiB. The stream holds five bytes.
    let file = string_file(8 << 20);
    let (result, largest) = watched(|| Metadata::read(file.as_slice(), 1 << 40));
    gguf_message(result);
    assert!(
        largest < SMALL,
        "the reader asked for one allocation of {largest} bytes"
    );
}

#[test]
fn a_string_length_above_the_cap_names_the_cap() {
    let file = string_file(GIB);
    let (result, largest) = watched(|| Metadata::read(file.as_slice(), 1 << 40));
    let message = gguf_message(result);
    assert!(message.contains("16 MiB"), "{message}");
    assert!(
        largest < SMALL,
        "the reader asked for one allocation of {largest} bytes"
    );
}

#[test]
fn a_string_below_the_cap_still_reads() {
    let text = "x".repeat(3 << 20);
    let mut f = header(1);
    f.extend(string("s"));
    f.extend_from_slice(&8u32.to_le_bytes());
    f.extend(string(&text));
    let m = Metadata::read(f.as_slice(), f.len() as u64).unwrap();
    assert_eq!(m.get("s"), Some(&nagaya_family_qwen38::gguf::Value::Str(text)));
}

#[test]
fn many_keys_are_checked_in_linear_time() {
    let n = 100_000u64;
    let mut f = header(n);
    for i in 0..n {
        f.extend(string(&format!("key.{i:07}")));
        f.extend_from_slice(&0u32.to_le_bytes());
        f.push(1);
    }
    let start = Instant::now();
    let m = Metadata::read(f.as_slice(), f.len() as u64).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(m.len(), n as usize);
    assert!(m.get("key.0099999").is_some());
    assert!(seconds < 5.0, "100000 keys took {seconds} seconds");
}

#[test]
fn a_repeated_key_at_the_end_of_many_keys_is_found() {
    let n = 1000u64;
    let mut f = header(n + 1);
    for i in 0..n {
        f.extend(string(&format!("key.{i}")));
        f.extend_from_slice(&0u32.to_le_bytes());
        f.push(1);
    }
    f.extend(string("key.7"));
    f.extend_from_slice(&0u32.to_le_bytes());
    f.push(1);
    let message = gguf_message(Metadata::read(f.as_slice(), f.len() as u64));
    assert!(message.contains("key.7") && message.contains("twice"), "{message}");
}
