//! The compiles and links of the op libraries in `ops_build.rs` are mk's compiles and links, flag for flag.
//!
//! The fixture holds the compile and the link of each op of mk's CMake build at mk commit `555a2a3`
//! (`tools/op_build_fixture.py` writes it). The float flags are part of the model's definition, and the op tests compare
//! bits. Thus a flag that differs fails here, before a GPU run. The runtime `rt.cpp` has its own entry in the fixture of
//! `build_flags.rs`.

#[allow(dead_code)]
#[path = "../build.rs"]
mod build;

use serde_json::Value;
use std::path::Path;

/// The source root of mk's build, as the fixture names it.
fn fixture() -> Value {
    let text =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/build/ops_commands.json"))
            .expect("read the fixture");
    serde_json::from_str(&text).expect("the fixture is JSON")
}

fn hip_src() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("hip").join("src")
}

#[test]
fn the_ops_are_the_targets_of_mk() {
    let fixture = fixture();
    let expected: Vec<String> = fixture["links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            let library = link["library"].as_str().unwrap();
            library
                .strip_prefix("libmk_")
                .unwrap()
                .strip_suffix(".so")
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(build::ops_build::op_names(&hip_src()), expected);
}

#[test]
fn each_op_compile_is_the_compile_of_mk_cmake_build() {
    let fixture = fixture();
    let reference_src = fixture["reference_src"].as_str().unwrap().to_owned();
    let src = Path::new("/nagaya/hip/src");
    let entries = fixture["compiles"].as_array().unwrap();
    assert_eq!(entries.len(), 11);
    for entry in entries {
        let file = entry["file"].as_str().unwrap();
        let name = Path::new(file).file_stem().unwrap().to_str().unwrap();
        let mut words = entry["command"].as_str().unwrap().split_whitespace();
        assert_eq!(Path::new(words.next().unwrap()), build::compiler());
        let expected: Vec<String> = words
            .map(|word| word.replace(&reference_src, &src.display().to_string()))
            .collect();
        // mk writes the object to its build directory. The path is the same text here.
        let object = format!("src/ops/CMakeFiles/mk_{name}.dir/{name}.hip.o");
        let got = build::ops_build::op_compile(src, name, Path::new(&object));
        assert_eq!(got, expected, "the compile of {name}.hip");
    }
}

#[test]
fn each_op_link_has_the_flags_of_mk_cmake_build() {
    let fixture = fixture();
    // The words before the soname are the same in every link rule of mk. The rule is the same text for each library.
    let rule = fixture["link_rule"].as_str().unwrap();
    assert!(
        rule.contains("-fPIC $LANGUAGE_COMPILE_FLAGS $ARCH_FLAGS $LINK_FLAGS $SONAME_FLAG$SONAME -o $TARGET_FILE $in"),
        "the link rule of mk changed: {rule}"
    );
    for link in fixture["links"].as_array().unwrap() {
        let library = link["library"].as_str().unwrap();
        let name = library.strip_prefix("libmk_").unwrap().strip_suffix(".so").unwrap();
        let mut expected = vec!["-fPIC".to_owned()];
        expected.extend(
            link["language_flags"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .map(str::to_owned),
        );
        expected.extend(
            link["link_flags"]
                .as_str()
                .unwrap()
                .split_whitespace()
                .map(str::to_owned),
        );
        expected.push(format!("-Wl,-soname,{}", link["soname"].as_str().unwrap()));
        expected.push("-o".to_owned());
        let got = build::ops_build::op_link(
            name,
            Path::new("/out").join(library).as_path(),
            Path::new("/out/op.o"),
            Path::new("/out/rt.o"),
            Path::new("/opt/rocm/lib/libamdhip64.so"),
            "/builtins.a",
        );
        assert_eq!(&got[..expected.len()], expected.as_slice(), "the link of {library}");
        assert_eq!(got[expected.len()], format!("/out/{library}"));
    }
}
