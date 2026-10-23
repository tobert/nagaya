//! The host checks of the op tests: no GPU and no feature. They hold the generator and the fixture files.
//!
//! The generator in `tests/ops/inputs.rs` must give what `tools/op_fixtures.py` gives, and the fixtures must come from one
//! run of one revision of mk and stay small.

#[path = "ops/fixture.rs"]
mod fixture;
#[path = "ops/inputs.rs"]
mod inputs;

use fixture::Fixture;
use inputs::Qtype;

/// The groups on disk, in name order.
fn groups() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(fixture::directory())
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".json").map(str::to_owned)
        })
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no fixture group on disk");
    names
}

#[test]
fn the_generator_matches_the_python_generator() {
    let f = Fixture::load("gen");
    let (head, _) = f.u32s("u64_head");
    let want: Vec<u32> = (0..16u64)
        .flat_map(|i| {
            let v = inputs::u64_at(7, i);
            [v as u32, (v >> 32) as u32]
        })
        .collect();
    assert_eq!(head, want, "u64_at");
    let rust = inputs::normals(9, 64);
    assert_eq!(
        rust.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        f.f32_flat("normals").iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "normals"
    );
    let rust = inputs::uniforms(11, 64);
    assert_eq!(
        rust.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        f.f32_flat("uniforms").iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "uniforms"
    );
    assert_eq!(inputs::integers(13, 64, 300), f.u32s("integers").0, "integers");
    assert_eq!(inputs::permutation(15, 100), f.u32s("permutation").0, "permutation");
    assert_eq!(inputs::bytes(17, 100), f.bytes("bytes"), "bytes");
}

#[test]
fn the_raw_rows_match_the_python_generator() {
    let f = Fixture::load("gen");
    let cases = f.meta("raw").as_array().expect("the raw cases").clone();
    assert_eq!(cases.len(), 7, "one case for each type");
    for case in cases {
        let name = case["type"].as_str().unwrap();
        let qtype = [
            Qtype::F32,
            Qtype::Q8_0,
            Qtype::Q6K,
            Qtype::Iq4Nl,
            Qtype::Iq3S,
            Qtype::Iq4Xs,
            Qtype::Bf16,
        ]
        .into_iter()
        .find(|q| q.name() == name)
        .expect("a known type");
        let raw = inputs::raw_rows(
            qtype,
            case["seed"].as_u64().unwrap(),
            case["rows"].as_u64().unwrap() as usize,
            case["k"].as_u64().unwrap() as usize,
        );
        let len = case["len"].as_u64().unwrap() as usize;
        assert_eq!(
            &raw[..len],
            f.bytes(&format!("raw_{name}")).as_slice(),
            "raw rows of {name}"
        );
    }
}

#[test]
fn all_fixtures_come_from_one_run_of_one_revision_of_mk() {
    let names = groups();
    let first = Fixture::load(&names[0]);
    let header = &first.json["header"];
    let rev = header["mk_rev"].as_str().expect("mk_rev");
    assert!(
        rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
        "mk_rev {rev:?} is not a git revision"
    );
    for name in &names {
        assert_eq!(&Fixture::load(name).json["header"], header, "the header of {name}");
    }
}

/// Each fixture file is read by a test, and each test reads a file that exists. The test finds the reads in the sources.
#[test]
fn the_groups_on_disk_are_the_groups_that_the_tests_load() {
    let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut loaded = std::collections::BTreeSet::new();
    let mut paths: Vec<_> = std::fs::read_dir(tests.join("ops"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.push(tests.join("ops_host.rs"));
    for path in paths {
        if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).unwrap();
            for part in text.split("Fixture::load(\"").skip(1) {
                loaded.insert(part.split('"').next().unwrap().to_owned());
            }
        }
    }
    let on_disk: std::collections::BTreeSet<String> = groups().into_iter().collect();
    assert_eq!(
        on_disk, loaded,
        "a fixture file that no test loads, or a group with no file"
    );
}

/// A site that traps without recording would be a silent trap. `MK_TRAP` is the only way to trap (`hip/src/rt/trap.h`).
#[test]
fn no_bare_builtin_trap_outside_the_trap_header() {
    fn walk(dir: &std::path::Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, found);
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ["h", "hip", "cpp"].contains(&ext) && path.file_name().is_some_and(|n| n != "trap.h") {
                let text = std::fs::read_to_string(&path).unwrap();
                for (number, line) in text.lines().enumerate() {
                    if line.contains("__builtin_trap") {
                        found.push(format!("{}:{}", path.display(), number + 1));
                    }
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("hip/src"),
        &mut found,
    );
    assert!(found.is_empty(), "a bare __builtin_trap: {found:?}");
}

/// Each group but `gen` names the files of mk that it ports, as paths in mk's `tests/ops`.
#[test]
fn each_group_names_the_mk_files_it_reads() {
    for name in groups() {
        let files = Fixture::load(&name)
            .meta("mk_files")
            .as_array()
            .expect("mk_files")
            .clone();
        if name == "gen" {
            assert!(files.is_empty(), "gen reads no file of mk");
            continue;
        }
        assert!(!files.is_empty(), "the group {name} names no file of mk");
        for file in files {
            let file = file.as_str().unwrap();
            assert!(
                file.starts_with("tests/ops/") && file.ends_with(".py"),
                "{name}: {file} is not an op test of mk"
            );
        }
    }
}

#[test]
fn the_fixtures_stay_small() {
    let total: u64 = std::fs::read_dir(fixture::directory())
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(
        total < 1_000_000,
        "the op fixtures take {total} bytes, which is over 1 MB"
    );
}
