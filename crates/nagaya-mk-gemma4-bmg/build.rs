//! Compiles the kernels for the B70 cards with `ocloc`, ahead of time, when the feature `xe` is on.
//!
//! The flags are mkviolet's (`build.rs` in mkviolet). The output goes to `OUT_DIR`, and the crate embeds it.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/megakernel.cl");
    if std::env::var_os("CARGO_FEATURE_XE").is_none() {
        return;
    }
    println!("cargo:rustc-link-lib=ze_loader");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let source = manifest.join("kernels/megakernel.cl");
    let status = Command::new("ocloc")
        .arg("compile")
        .arg("-file")
        .arg(&source)
        .args(["-device", "bmg-g31", "-options", "-cl-std=CL3.0", "-out_dir"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("cannot run ocloc: {e}. Install ocloc, or build without the feature xe."));
    assert!(
        status.success(),
        "ocloc did not compile {} for bmg-g31: {status}",
        source.display()
    );
    let binary = out.join("megakernel_bmg.bin");
    assert!(
        binary.is_file(),
        "ocloc wrote no {}. Read the ocloc output above.",
        binary.display()
    );
}
