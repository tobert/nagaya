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
    // The compiler is in the plan table lineage of the identity (spec 12.1).
    let version = Command::new("ocloc")
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("cannot run ocloc --version: {e}"));
    let version = String::from_utf8_lossy(&version.stdout).trim().to_owned();
    assert!(!version.is_empty(), "ocloc --version printed no version");
    println!("cargo:rustc-env=GEMMA4_OCLOC_VERSION={version}");
    // `ocloc query` writes its answer to a file of that name in the working directory.
    let status = Command::new("ocloc")
        .args(["query", "IGC_REVISION"])
        .current_dir(&out)
        .status()
        .unwrap_or_else(|e| panic!("cannot run ocloc query: {e}"));
    assert!(status.success(), "ocloc query IGC_REVISION failed: {status}");
    let igc = std::fs::read_to_string(out.join("IGC_REVISION"))
        .unwrap_or_else(|e| panic!("ocloc query wrote no IGC_REVISION: {e}"));
    let igc = match igc.trim_matches(|c: char| c == '\0' || c.is_whitespace()) {
        "" => "unknown".to_owned(),
        revision => revision.to_owned(),
    };
    println!("cargo:rustc-env=GEMMA4_IGC_REVISION={igc}");
    let binary = out.join("megakernel_bmg.bin");
    assert!(
        binary.is_file(),
        "ocloc wrote no {}. Read the ocloc output above.",
        binary.display()
    );
}
