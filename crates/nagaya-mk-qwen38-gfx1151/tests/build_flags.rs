//! The compiles of `build.rs` are the reference's compiles, flag for flag.
//!
//! The fixture holds the nine entries of `build/compile_commands.json` for the units of `libqwen38engine.so`, from
//! the reference CMake build at commit `555a2a3`. The float flags are part of the model's definition, and the conformance
//! gates compare bits. Thus a flag that differs fails here, before a GPU run.
//!
//! The build compiles six of the nine. The Rust reader took the place of two GGUF units of the reference in milestone 15 step 3,
//! and the Rust pass build took the place of `blocks.hip` in step 6.
//! The thin extern layer `extern.hip` took the place of the C ABI `capi.hip` of the reference in step 4, with the same compile.

#[allow(dead_code)]
#[path = "../build.rs"]
mod build;

use std::path::Path;

/// The source root of the reference's build, as the fixture names it.
const REFERENCE_SRC: &str = "/home/atobey/src/megakernel-qwen38-flashnext-strixhalo/src";

/// The name of a unit here for its name in the reference: `extern.hip` for `capi.hip` of the reference, and the same name for the others.
fn rename(word: &str) -> String {
    word.replace("/engine/capi.hip", "/engine/extern.hip")
}

#[test]
fn each_compile_is_the_compile_of_the_reference_cmake_build() {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/build/compile_commands.json"),
    )
    .expect("read the fixture");
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text).expect("the fixture is a JSON list");
    let rust = [
        format!("{REFERENCE_SRC}/engine/blocks.hip"),
        format!("{REFERENCE_SRC}/gguf/gguf.cpp"),
        format!("{REFERENCE_SRC}/gguf/dequant.cpp"),
    ];
    let (gone, entries): (Vec<_>, Vec<_>) = entries.into_iter().partition(|entry| {
        let file = entry["file"].as_str().expect("an entry names its file");
        file.starts_with(&format!("{REFERENCE_SRC}/gguf/")) || file == rust[0]
    });
    let mut gone: Vec<&str> = gone.iter().map(|entry| entry["file"].as_str().unwrap()).collect();
    gone.sort();
    let mut want: Vec<&str> = rust.iter().map(String::as_str).collect();
    want.sort();
    assert_eq!(gone, want, "only the units that Rust took the place of are left out");
    let src = Path::new("/nagaya/hip/src");
    let out = Path::new("/out");
    let units = build::units(src, out);
    assert_eq!(
        units.len(),
        entries.len(),
        "one compile for each unit of the reference library"
    );
    for (unit, entry) in units.iter().zip(&entries) {
        let command = entry["command"].as_str().expect("an entry has a command");
        let mut words = command.split_whitespace();
        assert_eq!(
            Path::new(words.next().expect("a command names its compiler")),
            build::compiler()
        );
        let mut expected = Vec::new();
        let mut after_o = false;
        for word in words {
            // CMake's module map of a C++ unit is an empty file. It gives no flag.
            if word.starts_with('@') && word.ends_with(".modmap") {
                continue;
            }
            // The compiler writes the object to stdout, so no path of the build directory is in the command.
            if after_o {
                expected.push("-".to_owned());
                after_o = false;
                continue;
            }
            after_o = word == "-o";
            expected.push(rename(&word.replace(REFERENCE_SRC, &src.display().to_string())));
        }
        assert_eq!(unit.arguments, expected, "the compile of {}", unit.source.display());
        assert_eq!(
            unit.source.display().to_string(),
            rename(
                &entry["file"]
                    .as_str()
                    .expect("an entry names its file")
                    .replace(REFERENCE_SRC, &src.display().to_string())
            )
        );
    }
}

#[test]
fn every_unit_keeps_the_float_flags() {
    for unit in build::units(Path::new("/s"), Path::new("/o")) {
        for flag in ["-ffp-contract=off", "-fno-fast-math"] {
            assert!(
                unit.arguments.iter().any(|argument| argument == flag),
                "{} has no {flag}",
                unit.source.display()
            );
        }
    }
}
