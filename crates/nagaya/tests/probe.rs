use std::fs;
use std::path::{Path, PathBuf};

use nagaya::inventory::{Device, Inventory, Probe};
use nagaya::probe::{DeviceFacts, FileProbe, Machine, Nic, Nvme, Vendor};

fn fixture(machine: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/probe")
        .join(machine)
}

fn machine(name: &str) -> Machine {
    FileProbe::new(fixture(name)).machine().unwrap()
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::write(target, fs::read(entry.path()).unwrap()).unwrap();
        }
    }
}

/// Copy a fixture to a temporary directory, so that a test can break it.
fn broken_copy(name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    copy_tree(&fixture(name), &root);
    (dir, root)
}

fn probe_error(root: &Path) -> String {
    FileProbe::new(root).machine().unwrap_err().to_string()
}

fn nvme(name: &str, model: &str, sectors: u64) -> Nvme {
    Nvme {
        name: name.to_owned(),
        model: model.to_owned(),
        bytes: sectors * 512,
    }
}

fn nic(name: &str, driver: &str, pci_id: &str, speed_mbps: Option<u32>) -> Nic {
    Nic {
        name: name.to_owned(),
        driver: driver.to_owned(),
        pci_id: pci_id.to_owned(),
        speed_mbps,
    }
}

fn facts(
    id: u32,
    name: &str,
    memory_bytes: u64,
    unified: bool,
    vendor: Vendor,
    address: &str,
    pci_id: &str,
) -> DeviceFacts {
    DeviceFacts {
        device: Device {
            id,
            name: name.to_owned(),
            memory_bytes,
            unified,
        },
        vendor,
        pci_address: address.to_owned(),
        pci_id: pci_id.to_owned(),
    }
}

#[test]
fn gfx1151host_has_one_unified_gfx1151() {
    let machine = machine("gfx1151host");
    assert_eq!(
        machine.devices,
        vec![facts(
            0,
            "gfx1151",
            133_143_986_176,
            true,
            Vendor::Amd,
            "0000:c2:00.0",
            "1002:1586"
        )]
    );
    assert_eq!(machine.host_bytes, 131_156_408 * 1024);
    assert_eq!(machine.cpus, (0..32).collect::<Vec<u32>>());
    assert_eq!(
        machine.nvme,
        vec![
            nvme("nvme0n1", "Samsung SSD 9100 PRO 8TB", 15_628_053_168),
            nvme("nvme1n1", "INTEL SSDPEDMD016T4", 3_125_627_568),
        ]
    );
    assert_eq!(machine.nics, vec![nic("enp191s0", "r8169", "10ec:8126", Some(1000))]);
}

#[test]
fn b70host_has_two_xe_cards_with_32_gib_each() {
    let machine = machine("b70host");
    assert_eq!(
        machine.devices,
        vec![
            facts(
                0,
                "xe-e223",
                32 << 30,
                false,
                Vendor::Intel,
                "0000:03:00.0",
                "8086:e223"
            ),
            facts(
                1,
                "xe-e223",
                32 << 30,
                false,
                Vendor::Intel,
                "0000:84:00.0",
                "8086:e223"
            ),
        ]
    );
    assert_eq!(machine.host_bytes, 131_912_348 * 1024);
    assert_eq!(machine.cpus, (0..24).collect::<Vec<u32>>());
    assert_eq!(
        machine.nvme,
        vec![nvme("nvme0n1", "Samsung SSD 990 PRO 4TB", 7_814_037_168)]
    );
    assert_eq!(
        machine.nics,
        vec![
            nic("enp1s0f0np0", "mlx5_core", "15b3:a2d6", Some(100_000)),
            nic("enp1s0f1np1", "mlx5_core", "15b3:a2d6", None),
        ]
    );
}

#[test]
fn sm121host_has_one_unified_gb10_that_uses_the_host_memory() {
    let machine = machine("sm121host");
    let host_bytes = 124_538_176 * 1024;
    assert_eq!(
        machine.devices,
        vec![facts(
            0,
            "NVIDIA GB10",
            host_bytes,
            true,
            Vendor::Nvidia,
            "000f:01:00.0",
            "10de:2e12"
        )]
    );
    assert_eq!(machine.host_bytes, host_bytes);
    assert_eq!(machine.cpus, (0..20).collect::<Vec<u32>>());
    assert_eq!(
        machine.nvme,
        vec![nvme("nvme0n1", "Samsung SSD 990 PRO 4TB", 7_814_037_168)]
    );
    assert_eq!(
        machine.nics,
        vec![
            nic("enP7s7", "r8127", "10ec:8127", Some(10_000)),
            nic("enp1s0f0np0", "mlx5_core", "15b3:1021", Some(200_000)),
        ]
    );
}

#[test]
fn the_probe_trait_gives_the_inventory_of_the_machine() {
    let probe = FileProbe::new(fixture("gfx1151host"));
    let inventory = probe.probe().unwrap();
    assert_eq!(inventory, probe.machine().unwrap().inventory());
    assert_eq!(
        inventory,
        Inventory {
            devices: vec![Device {
                id: 0,
                name: "gfx1151".to_owned(),
                memory_bytes: 133_143_986_176,
                unified: true,
            }],
            host_bytes: 131_156_408 * 1024,
        }
    );
}

#[test]
fn the_fixtures_stay_small() {
    fn size(path: &Path) -> u64 {
        fs::read_dir(path)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                match entry.file_type().unwrap().is_dir() {
                    true => size(&entry.path()),
                    false => entry.metadata().unwrap().len(),
                }
            })
            .sum()
    }
    let total = size(&fixture(""));
    assert!(total < 200 * 1024, "the fixtures hold {total} bytes");
}

#[test]
fn a_missing_meminfo_is_an_error_that_names_the_file() {
    let (_dir, root) = broken_copy("gfx1151host");
    fs::remove_file(root.join("proc/meminfo")).unwrap();
    let message = probe_error(&root);
    assert!(message.contains("proc/meminfo"), "{message}");
}

#[test]
fn an_empty_root_is_an_error_and_not_an_empty_machine() {
    let dir = tempfile::tempdir().unwrap();
    assert!(FileProbe::new(dir.path()).machine().is_err());
}

#[test]
fn a_malformed_memtotal_is_an_error() {
    let (_dir, root) = broken_copy("gfx1151host");
    fs::write(root.join("proc/meminfo"), "MemTotal:       lots kB\n").unwrap();
    let message = probe_error(&root);
    assert!(message.contains("MemTotal"), "{message}");
}

#[test]
fn a_meminfo_without_memtotal_is_an_error() {
    let (_dir, root) = broken_copy("gfx1151host");
    fs::write(root.join("proc/meminfo"), "MemFree:       100 kB\n").unwrap();
    let message = probe_error(&root);
    assert!(message.contains("MemTotal"), "{message}");
}

#[test]
fn a_gpu_node_without_memory_banks_is_an_error() {
    let (_dir, root) = broken_copy("gfx1151host");
    fs::remove_dir_all(root.join("sys/class/kfd/kfd/topology/nodes/1/mem_banks")).unwrap();
    let message = probe_error(&root);
    assert!(message.contains("mem_banks"), "{message}");
}

#[test]
fn a_gpu_memory_that_matches_neither_gtt_nor_vram_is_an_error() {
    let (_dir, root) = broken_copy("gfx1151host");
    let path = root.join("sys/bus/pci/devices/0000:c2:00.0/mem_info_gtt_total");
    fs::write(path, "1073741824\n").unwrap();
    let message = probe_error(&root);
    assert!(message.contains("gfx1151"), "{message}");
    assert!(message.contains("mem_info_vram_total"), "{message}");
}

#[test]
fn a_machine_with_no_kfd_has_no_amd_device() {
    let (_dir, root) = broken_copy("gfx1151host");
    fs::remove_dir_all(root.join("sys/class/kfd")).unwrap();
    let machine = FileProbe::new(&root).machine().unwrap();
    assert!(machine.devices.is_empty());
    assert_eq!(machine.cpus.len(), 32);
}

#[test]
fn an_xe_card_with_a_small_bar_is_an_error() {
    let (_dir, root) = broken_copy("b70host");
    let path = root.join("sys/class/drm/card0/device/resource");
    let text = fs::read_to_string(&path).unwrap().replace(
        "0x0000004000000000 0x00000047ffffffff",
        "0x0000004000000000 0x000000400fffffff",
    );
    fs::write(path, text).unwrap();
    let message = probe_error(&root);
    assert!(message.contains("0000:03:00.0"), "{message}");
}

#[test]
fn an_nvidia_model_that_the_probe_does_not_know_is_an_error() {
    let (_dir, root) = broken_copy("sm121host");
    let path = root.join("proc/driver/nvidia/gpus/000f:01:00.0/information");
    let text = fs::read_to_string(&path)
        .unwrap()
        .replace("NVIDIA GB10", "NVIDIA RTX 9000");
    fs::write(path, text).unwrap();
    let message = probe_error(&root);
    assert!(message.contains("NVIDIA RTX 9000"), "{message}");
}

#[test]
fn a_malformed_cpu_list_is_an_error() {
    let (_dir, root) = broken_copy("gfx1151host");
    fs::write(root.join("sys/devices/system/cpu/online"), "0-x\n").unwrap();
    let message = probe_error(&root);
    assert!(message.contains("sys/devices/system/cpu/online"), "{message}");
}

#[test]
fn a_malformed_nvme_size_is_an_error() {
    let (_dir, root) = broken_copy("gfx1151host");
    fs::write(root.join("sys/block/nvme0n1/size"), "big\n").unwrap();
    let message = probe_error(&root);
    assert!(message.contains("nvme0n1/size"), "{message}");
}
