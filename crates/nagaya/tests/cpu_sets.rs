//! The CPU sets of spec 3.2a (milestone 14). Each rule has one test.

use nagaya::cpu_sets::{CpuSetError, CpuSets, CpuSetupError, for_machine, split};
use nagaya::exit::ExitCode;
use nagaya::probe::FileProbe;

/// The online CPUs of the Strix Halo machine: 16 cores, two threads each (`lscpu`).
fn strix_halo_online() -> Vec<u32> {
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
    assert_eq!(sets(&strix_halo_online()).housekeeping, vec![0, 1]);
}

#[test]
fn the_pool_set_takes_every_other_cpu() {
    assert_eq!(sets(&strix_halo_online()).pool, range(2, 32));
}

#[test]
fn the_pool_size_is_the_number_of_pool_cpus() {
    assert_eq!(sets(&strix_halo_online()).pool_size(), 30);
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
    for online in [strix_halo_online(), vec![0, 1, 2, 3], holes] {
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

// Spec 3.2b: the CPU set of each device.

use nagaya::cpu_sets::{device_sets, for_machine_with_devices, for_machine_with_devices_ignoring};

fn fixture(machine: &str) -> FileProbe {
    FileProbe::new(format!("{}/tests/fixtures/probe/{machine}", env!("CARGO_MANIFEST_DIR")))
}

/// The local lists of devices that are not ignored.
fn some(local: Vec<Vec<u32>>) -> Vec<Option<Vec<u32>>> {
    local.into_iter().map(Some).collect()
}

/// The two L3 domains of the 9950X of the B70 machine.
fn ccds() -> Vec<Vec<u32>> {
    vec![
        [0..8, 16..24].into_iter().flatten().collect(),
        [8..16, 24..32].into_iter().flatten().collect(),
    ]
}

#[test]
fn two_cards_with_no_locality_take_one_l3_domain_each_less_the_housekeeping_cpus() {
    let online: Vec<u32> = (0..32).collect();
    let pool: Vec<u32> = (2..32).collect();
    let sets = device_sets(&pool, &online, &ccds(), &some(vec![online.clone(), online.clone()])).unwrap();
    assert_eq!(sets[0], [2..8, 16..24].into_iter().flatten().collect::<Vec<u32>>());
    assert_eq!(sets[1], [8..16, 24..32].into_iter().flatten().collect::<Vec<u32>>());
}

#[test]
fn more_cards_than_domains_take_the_domains_in_turn() {
    let online: Vec<u32> = (0..32).collect();
    let pool: Vec<u32> = (2..32).collect();
    let all = vec![online.clone(); 3];
    let sets = device_sets(&pool, &online, &ccds(), &some(all)).unwrap();
    assert_eq!(sets[2], sets[0]);
    assert_ne!(sets[1], sets[0]);
}

#[test]
fn a_card_with_a_local_list_takes_the_local_cpus_in_the_pool() {
    let online: Vec<u32> = (0..32).collect();
    let pool: Vec<u32> = (2..32).collect();
    let local = vec![(0..8).collect::<Vec<u32>>(), (8..16).collect()];
    let sets = device_sets(&pool, &online, &ccds(), &some(local)).unwrap();
    assert_eq!(sets[0], (2..8).collect::<Vec<u32>>());
    assert_eq!(sets[1], (8..16).collect::<Vec<u32>>());
}

#[test]
fn one_card_keeps_the_pool_set() {
    let online: Vec<u32> = (0..32).collect();
    let pool: Vec<u32> = (2..32).collect();
    let sets = device_sets(&pool, &online, &ccds(), &some(vec![online.clone()])).unwrap();
    assert_eq!(sets, vec![pool]);
}

#[test]
fn a_local_list_with_no_pool_cpu_is_an_error() {
    let online: Vec<u32> = (0..32).collect();
    let pool: Vec<u32> = (2..32).collect();
    let local = vec![vec![0, 1], vec![8, 9]];
    let error = device_sets(&pool, &online, &ccds(), &some(local)).unwrap_err();
    assert_eq!(error, CpuSetError::NoPoolCpu { device: 0 });
}

#[test]
fn a_domain_with_no_pool_cpu_is_an_error() {
    let online: Vec<u32> = (0..4).collect();
    let pool = vec![2, 3];
    let domains = vec![vec![0, 1], vec![2, 3]];
    let error = device_sets(&pool, &online, &domains, &some(vec![online.clone(), online.clone()])).unwrap_err();
    assert_eq!(error, CpuSetError::NoPoolCpu { device: 0 });
}

#[test]
fn the_b70_machine_gives_card_0_and_card_1_one_ccd_each() {
    let sets = for_machine_with_devices(&fixture("b70-pair")).unwrap();
    assert_eq!(sets.housekeeping, vec![0, 1]);
    assert_eq!(
        sets.device(0),
        [2..8, 16..24].into_iter().flatten().collect::<Vec<u32>>()
    );
    assert_eq!(
        sets.device(1),
        [8..16, 24..32].into_iter().flatten().collect::<Vec<u32>>()
    );
}

#[test]
fn the_strix_halo_machine_has_one_device_and_it_keeps_the_pool_set() {
    let sets = for_machine_with_devices(&fixture("strix-halo")).unwrap();
    let plain = for_machine(&fixture("strix-halo")).unwrap();
    assert_eq!((&sets.housekeeping, &sets.pool), (&plain.housekeeping, &plain.pool));
    assert_eq!(sets.devices, vec![sets.pool.clone()]);
}

#[test]
fn the_gb10_machine_has_one_device_and_it_keeps_the_pool_set() {
    let sets = for_machine_with_devices(&fixture("dgx-spark")).unwrap();
    assert_eq!(sets.device(0), sets.pool);
}

#[test]
fn a_set_made_by_the_plain_split_gives_each_device_the_pool() {
    let sets = split(&[0, 1, 2, 3]).unwrap();
    assert_eq!(sets.device(7), vec![2, 3]);
}

#[test]
fn an_ignored_device_gets_no_set_and_takes_no_l3_domain() {
    let online: Vec<u32> = (0..32).collect();
    let pool: Vec<u32> = (2..32).collect();
    // Device 0 is ignored. Devices 1 and 2 take the first and the second domain, as devices 0 and 1 would.
    let local = vec![None, Some(online.clone()), Some(online.clone())];
    let sets = device_sets(&pool, &online, &ccds(), &local).unwrap();
    assert_eq!(sets[0], Vec::<u32>::new());
    assert_eq!(sets[1], [2..8, 16..24].into_iter().flatten().collect::<Vec<u32>>());
    assert_eq!(sets[2], [8..16, 24..32].into_iter().flatten().collect::<Vec<u32>>());
}

#[test]
fn one_device_that_is_not_ignored_keeps_the_pool_set() {
    let online: Vec<u32> = (0..32).collect();
    let pool: Vec<u32> = (2..32).collect();
    let local = vec![Some(online.clone()), None];
    let sets = device_sets(&pool, &online, &ccds(), &local).unwrap();
    assert_eq!(sets, vec![pool, Vec::new()]);
}

#[test]
fn the_b70_machine_with_the_igpu_ignored_keeps_the_sets_of_the_cards() {
    let sets = for_machine_with_devices_ignoring(&fixture("b70-pair"), &["0000:80:00.0".to_owned()]).unwrap();
    assert_eq!(
        sets.device(0),
        [2..8, 16..24].into_iter().flatten().collect::<Vec<u32>>()
    );
    assert_eq!(
        sets.device(1),
        [8..16, 24..32].into_iter().flatten().collect::<Vec<u32>>()
    );
    assert_eq!(sets.devices[2], Vec::<u32>::new());
}

#[test]
#[should_panic(expected = "device 2 is ignored")]
fn the_set_of_an_ignored_device_is_a_defect_to_ask_for() {
    let sets = for_machine_with_devices_ignoring(&fixture("b70-pair"), &["0000:80:00.0".to_owned()]).unwrap();
    sets.device(2);
}

#[test]
fn an_ignored_address_that_the_machine_lacks_is_a_probe_error() {
    let error = for_machine_with_devices_ignoring(&fixture("b70-pair"), &["0000:99:00.0".to_owned()]).unwrap_err();
    assert!(error.to_string().contains("0000:99:00.0"), "{error}");
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
}

#[test]
fn the_b70_machine_with_one_card_left_gives_it_the_pool_set() {
    let ignore = ["0000:80:00.0".to_owned(), "0000:08:00.0".to_owned()];
    let probe = fixture("b70-pair");
    let machine = probe.machine().unwrap();
    assert!(
        machine
            .devices
            .iter()
            .any(|facts| facts.device.pci_address == ignore[1])
    );
    let sets = for_machine_with_devices_ignoring(&probe, &ignore).unwrap();
    assert_eq!(sets.devices.iter().filter(|set| !set.is_empty()).count(), 1);
    assert_eq!(sets.devices[0], sets.pool);
}
