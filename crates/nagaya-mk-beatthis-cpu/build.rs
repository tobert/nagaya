//! The fingerprint of the sources whose bits the engine gives: this crate's `src` and the family's `src`. The engine
//! has no canary yet (spec 11.19), so its identity holds this fingerprint in `numbers`, and a change of a kernel or of
//! the plan changes the identity digest (spec 8.5, 12.1).

use std::path::Path;

use sha2::{Digest, Sha256};

fn add(dir: &Path, hasher: &mut Sha256) {
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    for f in files {
        if f.is_dir() {
            add(&f, hasher);
        } else if f.extension().is_some_and(|e| e == "rs") {
            println!("cargo:rerun-if-changed={}", f.display());
            hasher.update(f.file_name().unwrap().to_string_lossy().as_bytes());
            hasher.update(std::fs::read(&f).unwrap());
        }
    }
}

fn main() {
    let mut h = Sha256::new();
    add(Path::new("src"), &mut h);
    add(Path::new("../nagaya-family-beatthis/src"), &mut h);
    println!("cargo:rustc-env=NAGAYA_BEATTHIS_SOURCE_SHA256={:x}", h.finalize());
}
