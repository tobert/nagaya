//! Print the SHA-256 of the engine's weight arena, so that two builds can show the same arena bytes.
//!
//! The example opens the engine and finds the arena by its memfd name, `qwen38-weights` (`engine.hip` `create_arena`).
//! It reads the arena through `/proc/self/fd`, because the seals forbid a writable map. The example never closes the
//! engine (spec 3.3).

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use sha2::{Digest, Sha256};

/// The link target of the engine's arena fd: the memfd name, as the kernel gives it.
const ARENA_LINK: &str = "/memfd:qwen38-weights (deleted)";

/// The fd of this process that is the engine's arena. Stop when there is not exactly one.
fn arena_fd() -> PathBuf {
    let mut found = Vec::new();
    for entry in std::fs::read_dir("/proc/self/fd").expect("read /proc/self/fd") {
        let path = entry.expect("an fd entry").path();
        if let Ok(target) = std::fs::read_link(&path)
            && target.as_os_str() == ARENA_LINK
        {
            found.push(path);
        }
    }
    assert_eq!(found.len(), 1, "want one fd named {ARENA_LINK}, found {found:?}");
    found.pop().expect("one fd")
}

fn main() {
    let start = Instant::now();
    let _engine = nagaya_mk_qwen38_gfx1151::open().expect("open the engine");
    let opened = start.elapsed().as_secs_f64();
    let path = arena_fd();
    let mut file = std::fs::File::open(&path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 64 << 20];
    let mut bytes = 0u64;
    loop {
        let count = file
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("read {} at byte {bytes}: {error}", path.display()));
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        bytes += count as u64;
    }
    let digest: String = hash.finalize().iter().map(|byte| format!("{byte:02x}")).collect();
    println!("arena bytes: {bytes}");
    println!("arena sha256: {digest}");
    println!(
        "open {opened:.1} s, digest {:.1} s",
        start.elapsed().as_secs_f64() - opened
    );
}
