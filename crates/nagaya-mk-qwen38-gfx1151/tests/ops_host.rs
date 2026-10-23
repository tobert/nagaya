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

const GROUPS: [&str; 2] = ["gen", "qmm"];

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
    let first = Fixture::load(GROUPS[0]);
    let header = &first.json["header"];
    let rev = header["mk_rev"].as_str().expect("mk_rev");
    assert!(
        rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
        "mk_rev {rev:?} is not a git revision"
    );
    for name in GROUPS {
        assert_eq!(&Fixture::load(name).json["header"], header, "the header of {name}");
    }
}

#[test]
fn the_group_list_is_the_list_of_files() {
    let mut on_disk: Vec<String> = std::fs::read_dir(fixture::directory())
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".json").map(str::to_owned)
        })
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = GROUPS.iter().map(|s| (*s).to_owned()).collect();
    listed.sort();
    assert_eq!(
        on_disk, listed,
        "a fixture file that no test reads, or a group with no file"
    );
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
