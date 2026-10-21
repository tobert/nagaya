//! The CPU sets of spec 3.2a (milestone 14). Each rule has one test.

use nagaya::cpu_sets::{CpuSetError, CpuSets, CpuSetupError, for_machine, split};
use nagaya::exit::ExitCode;
use nagaya::probe::FileProbe;

/// The online CPUs of gfx1151host: 16 cores, two threads each (`lscpu`).
fn gfx1151host_online() -> Vec<u32> {
    (0..32).collect()
}

fn range(from: u32, to: u32) -> Vec<u32> {
    (from..to).collect()
}

fn sets(online: &[u32]) -> CpuSets {
    split(online).unwrap_or_else(|error| panic!("the split of {online:?} failed: {error}"))
}

#[test]
fn the_housekeeping_set_takes_the_two_lowest_cpus() {
    assert_eq!(sets(&gfx1151host_online()).housekeeping, vec![0, 1]);
}

#[test]
fn the_pool_set_takes_every_other_cpu() {
    assert_eq!(sets(&gfx1151host_online()).pool, range(2, 32));
}

#[test]
fn the_pool_size_is_the_number_of_pool_cpus() {
    assert_eq!(sets(&gfx1151host_online()).pool_size(), 30);
    assert_eq!(sets(&[0, 1, 2, 3]).pool_size(), 2);
}

#[test]
fn four_cpus_split_two_and_two() {
    let split = sets(&[0, 1, 2, 3]);
    assert_eq!(split.housekeeping, vec![0, 1]);
    assert_eq!(split.pool, vec![2, 3]);
}

#[test]
fn the_two_sets_do_not_overlap_and_hold_every_online_cpu() {
    let holes = vec![0, 1, 2, 5, 6, 7];
    for online in [gfx1151host_online(), vec![0, 1, 2, 3], holes] {
        let split = sets(&online);
        assert!(split.housekeeping.iter().all(|cpu| !split.pool.contains(cpu)));
        let mut all: Vec<u32> = split.housekeeping.iter().chain(split.pool.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, online);
    }
}

#[test]
fn offline_cpus_are_in_neither_set() {
    let split = sets(&[0, 1, 2, 5, 6, 7]);
    assert_eq!(split.housekeeping, vec![0, 1]);
    assert_eq!(split.pool, vec![2, 5, 6, 7]);
}

#[test]
fn the_housekeeping_set_takes_the_two_lowest_online_cpus() {
    let split = sets(&[4, 5, 6, 7]);
    assert_eq!(split.housekeeping, vec![4, 5]);
    assert_eq!(split.pool, vec![6, 7]);
}

#[test]
fn one_cpu_is_refused() {
    assert_eq!(split(&[0]), Err(CpuSetError::TooFewCpus { online: 1, needed: 3 }));
}

#[test]
fn two_cpus_are_refused() {
    assert_eq!(split(&[0, 1]), Err(CpuSetError::TooFewCpus { online: 2, needed: 3 }));
}

#[test]
fn three_cpus_are_the_smallest_split() {
    let split = sets(&[0, 1, 2]);
    assert_eq!(split.housekeeping, vec![0, 1]);
    assert_eq!(split.pool, vec![2]);
}

#[test]
fn an_unsorted_list_is_an_error() {
    assert_eq!(split(&[2, 0, 1]), Err(CpuSetError::NotIncreasing { index: 1 }));
}

#[test]
fn a_repeated_cpu_is_an_error() {
    assert_eq!(split(&[0, 1, 1, 2]), Err(CpuSetError::NotIncreasing { index: 2 }));
}

/// A machine root that has only the list of online CPUs.
fn root_with_online(list: &str) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("sys/devices/system/cpu");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("online"), format!("{list}\n")).unwrap();
    root
}

fn for_root(root: &tempfile::TempDir) -> Result<CpuSets, CpuSetupError> {
    for_machine(&FileProbe::new(root.path()))
}

#[test]
fn the_sets_come_from_the_online_list_of_the_file_probe() {
    let root = root_with_online("0-2,5-7");
    let split = for_root(&root).unwrap();
    assert_eq!(split.housekeeping, vec![0, 1]);
    assert_eq!(split.pool, vec![2, 5, 6, 7]);
}

#[test]
fn a_machine_with_two_online_cpus_refuses_the_start() {
    let error = for_root(&root_with_online("0-1")).unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
    assert!(error.to_string().contains("2 online"), "{error}");
}

#[test]
fn an_online_list_out_of_order_refuses_the_start() {
    let error = for_root(&root_with_online("3,1,2")).unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
}

#[test]
fn a_machine_with_no_online_list_refuses_the_start() {
    let root = tempfile::tempdir().unwrap();
    let error = for_root(&root).unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
    assert!(error.to_string().contains("sys/devices/system/cpu/online"), "{error}");
}
