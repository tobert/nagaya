//! Compiles `hip/beatthis.hip` with ROCm's clang++ into a static library, in a build with the feature `hip`.
//! `-ffp-contract=off` keeps each multiply and add of `kernels/beatthis_ops.h` apart, so the bits are those of the
//! source on gfx1151 (spec 8.5).

use std::path::{Path, PathBuf};
use std::process::Command;

const ROCM: &str = "/opt/rocm";

const FLAGS: &[&str] = &[
    "-O3",
    "-std=c++20",
    "-fPIC",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-unused-parameter",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-D__HIP_PLATFORM_AMD__=1",
    "-x",
    "hip",
    "--offload-arch=gfx1151",
];

fn run(program: &Path, arguments: &[String], directory: &Path, what: &str) -> Vec<u8> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
        .unwrap_or_else(|error| panic!("{what}: cannot run {}: {error}", program.display()));
    if !output.status.success() {
        panic!(
            "{what} failed with {}:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.stdout
}

/// Add each source file under `dir` to the fingerprint, in name order.
fn add(dir: &Path, hasher: &mut sha2::Sha256) {
    use sha2::Digest;
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    for f in files {
        if f.is_dir() {
            add(&f, hasher);
        } else if f
            .extension()
            .is_some_and(|e| matches!(e.to_str(), Some("rs" | "h" | "hip")))
        {
            hasher.update(f.file_name().unwrap().to_string_lossy().as_bytes());
            hasher.update(std::fs::read(&f).unwrap());
        }
    }
}

fn main() {
    // The engine has no canary yet (spec 11.19), so its identity holds a fingerprint of the sources whose bits it gives.
    {
        use sha2::Digest;
        let mut h = sha2::Sha256::new();
        for dir in ["src", "hip", "kernels", "../nagaya-family-beatthis/src"] {
            add(Path::new(dir), &mut h);
        }
        println!(
            "cargo:rustc-env=NAGAYA_BEATTHIS_GFX1151_SOURCE_SHA256={:x}",
            h.finalize()
        );
        println!("cargo:rerun-if-changed=src");
        println!("cargo:rerun-if-changed=../nagaya-family-beatthis/src");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=hip");
    println!("cargo:rerun-if-changed=kernels");
    if std::env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let rocm = PathBuf::from(ROCM);
    let mut arguments: Vec<String> = FLAGS.iter().map(|f| (*f).to_owned()).collect();
    arguments.extend([
        "-isystem".to_owned(),
        rocm.join("include").display().to_string(),
        "-Ikernels".to_owned(),
        "-c".to_owned(),
        "hip/beatthis.hip".to_owned(),
        "-o".to_owned(),
        "-".to_owned(),
    ]);
    let object = run(
        &rocm.join("llvm/bin/clang++"),
        &arguments,
        &manifest,
        "the compile of beatthis.hip",
    );
    let path = out.join("beatthis.o");
    std::fs::write(&path, object).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    let library = out.join("libnagaya_mk_beatthis_gfx1151.a");
    let _ = std::fs::remove_file(&library);
    run(
        &rocm.join("llvm/bin/llvm-ar"),
        &[
            "crs".to_owned(),
            library.display().to_string(),
            path.display().to_string(),
        ],
        &out,
        "the archive",
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=nagaya_mk_beatthis_gfx1151");
    println!("cargo:rustc-link-search=native={}", rocm.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}
