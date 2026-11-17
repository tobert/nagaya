//! The inventory that the start path reads (spec 2.1, 10.4). A build without the fake engine probes the files of the
//! machine. These tests use the fixed tree of the Strix Halo machine and a broken tree.

use std::path::{Path, PathBuf};

use nagaya::exit::ExitCode;
use nagaya::inventory::{Device, FakeProbe, Inventory, Probe};
use nagaya::probe::FileProbe;

fn strix_halo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/probe/strix-halo")
}

#[test]
fn the_start_path_reads_the_inventory_of_the_strix_halo_machine_through_the_probe_trait() {
    // The start path holds the probe as `&dyn Probe`, so a later probe for another vendor needs no change there.
    let probe: &dyn Probe = &FileProbe::new(strix_halo());
    let inventory = probe.probe().expect("the strix-halo tree probes");
    assert_eq!(
        inventory,
        Inventory {
            devices: vec![Device {
                id: 0,
                name: "gfx1151".to_owned(),
                pci_address: "0000:c2:00.0".to_owned(),
                memory_bytes: 133_143_986_176,
                unified: true,
                ignored: false,
            }],
            host_bytes: 131_156_408 * 1024,
        }
    );
}

#[test]
fn a_probe_error_at_start_refuses_the_start_with_exit_3() {
    // Spec 10.4: the machine is wrong, so a restart cannot help. The unit has `RestartPreventExitStatus=3`.
    let error = FakeProbe::failing("no GPU node in the topology").probe().unwrap_err();
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
    assert_eq!(error.exit_code().code(), 3);
}

#[test]
fn a_tree_with_no_meminfo_is_a_probe_error_that_names_the_file() {
    let directory = tempfile::tempdir().unwrap();
    let error = FileProbe::new(directory.path()).probe().unwrap_err();
    assert!(error.to_string().contains("proc/meminfo"), "{error}");
    assert_eq!(error.exit_code(), ExitCode::RefusedStart);
}
