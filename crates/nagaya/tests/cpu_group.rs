//! The CPU group of a profile (spec 3.2c): an engine with `devices = []`, the `[cpu]` table, and one group for all
//! CPU engines of a machine.

use nagaya::placement::device_groups;
use nagaya::profile::{CpuSpec, PlacementProfile};

const PROFILE: &str = r#"
[[engines]]
name = "beats"
kind = "fake"
devices = []
share = 1

[[engines]]
name = "qwen"
kind = "fake"
devices = [0]
share = 1

[[engines]]
name = "onsets"
kind = "fake"
devices = []
share = 1

[[budgets]]
tiers = ["device(0)", "host"]
bytes = 1000

[store]
nvme = 4096

[cpu]
threads = 8
"#;

#[test]
fn a_profile_with_a_cpu_engine_and_a_cpu_table_parses_and_writes_back() {
    let p = PlacementProfile::parse(PROFILE).unwrap();
    assert_eq!(p.cpu, Some(CpuSpec { threads: 8 }));
    let again = PlacementProfile::parse(&p.to_toml().unwrap()).unwrap();
    assert_eq!(again, p);
}

#[test]
fn a_cpu_engine_without_a_cpu_table_is_refused_by_name() {
    let text = PROFILE.replace("\n[cpu]\nthreads = 8\n", "");
    let e = PlacementProfile::parse(&text).unwrap_err().to_string();
    assert!(e.contains("engine `beats`") && e.contains("[cpu]"), "{e}");
}

#[test]
fn a_cpu_table_without_a_cpu_engine_is_refused() {
    let text = PROFILE.replace("devices = []", "devices = [0]");
    let e = PlacementProfile::parse(&text).unwrap_err().to_string();
    assert!(e.contains("no engine with `devices = []`"), "{e}");
}

#[test]
fn a_cpu_table_of_no_threads_is_refused() {
    let e = PlacementProfile::parse(&PROFILE.replace("threads = 8", "threads = 0"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("threads` is 0"), "{e}");
}

#[test]
fn a_cpu_table_with_an_unknown_key_is_refused() {
    let e = PlacementProfile::parse(&PROFILE.replace("threads = 8", "threads = 8\ncores = 4")).unwrap_err();
    assert!(e.to_string().contains("cores"), "{e}");
}

/// The profile `PUT` reads JSON, so the rules must hold there too.
#[test]
fn the_json_form_follows_the_same_rules() {
    let p = PlacementProfile::parse(PROFILE).unwrap();
    let mut json = serde_json::to_value(&p).unwrap();
    assert_eq!(serde_json::from_value::<PlacementProfile>(json.clone()).unwrap(), p);
    json.as_object_mut().unwrap().remove("cpu");
    let e = serde_json::from_value::<PlacementProfile>(json)
        .unwrap_err()
        .to_string();
    assert!(e.contains("[cpu]"), "{e}");
}

#[test]
fn every_cpu_engine_is_in_one_group() {
    let p = PlacementProfile::parse(PROFILE).unwrap();
    let groups = device_groups(&p);
    assert_eq!(groups.len(), 2, "{groups:?}");
    let cpu: Vec<_> = groups.iter().filter(|g| g.devices.is_empty()).collect();
    assert_eq!(cpu.len(), 1, "one CPU group: {groups:?}");
    assert_eq!(cpu[0].engines, [0, 2]);
    let gpu: Vec<_> = groups.iter().filter(|g| !g.devices.is_empty()).collect();
    assert_eq!(gpu[0].engines, [1]);
}
