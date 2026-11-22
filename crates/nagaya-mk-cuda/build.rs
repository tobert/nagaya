//! Compiles the CUDA C++ persistent megakernel interpreter with `nvcc` ahead of time when the feature `cuda` is on.
//!
//! Each engine gets its own `.cubin` in `OUT_DIR`, embedded via `include_bytes!`. The flags fix float rounding
//! (`--fmad=false`) for reproducible bits and target `sm_121a` for GB10 Blackwell block-scaled MMA instructions.

use std::path::{Path, PathBuf};
use std::process::Command;

const ARCH: &str = "sm_121a";

fn find_nvcc() -> PathBuf {
    if let Some(path) = std::env::var_os("CUDA_NVCC") {
        return PathBuf::from(path);
    }
    if let Ok(output) = Command::new("nvcc").arg("--version").output()
        && output.status.success()
    {
        return PathBuf::from("nvcc");
    }
    let default = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if default.is_file() {
        return default;
    }
    panic!(
        "cannot find nvcc. Install the CUDA toolkit, set CUDA_NVCC, add nvcc to PATH, or build without the feature cuda."
    );
}

fn compile_cubin(nvcc: &Path, source: &Path, kernels: &Path, out: &Path, name: &str, defines: &[&str]) {
    let dest = out.join(name);
    let mut cmd = Command::new(nvcc);
    cmd.args(["-cubin", &format!("-arch={ARCH}"), "-std=c++20", "--fmad=false", "-O3"]);
    for def in defines {
        cmd.arg(format!("-D{def}"));
    }
    cmd.arg("-I").arg(kernels).arg("-o").arg(&dest).arg(source);
    let output = cmd.output().unwrap_or_else(|e| {
        panic!(
            "cannot run {}: {e}. Check CUDA_NVCC or build without the feature cuda.",
            nvcc.display()
        )
    });
    assert!(
        output.status.success(),
        "nvcc failed to compile {} ({name}) for {ARCH}: {}\n{}",
        source.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        dest.is_file(),
        "nvcc wrote no {}. Read the nvcc output above.",
        dest.display()
    );
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/megakernel.cu");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = find_nvcc();
    let version_out = Command::new(&nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("cannot run {} --version: {e}", nvcc.display()));
    assert!(
        version_out.status.success(),
        "{} --version failed: {}",
        nvcc.display(),
        version_out.status
    );
    let version_text = String::from_utf8_lossy(&version_out.stdout);
    let version = version_text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(str::trim)
        .unwrap_or("unknown");
    println!("cargo:rustc-env=CUDA_NVCC_VERSION={version}");

    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let kernels = manifest.join("kernels");
    let source = kernels.join("megakernel.cu");

    compile_cubin(
        &nvcc,
        &source,
        &kernels,
        &out,
        "megakernel_inkling2_sm121.cubin",
        &["MK_INKLING2"],
    );
    compile_cubin(
        &nvcc,
        &source,
        &kernels,
        &out,
        "megakernel_inkling2_profile_sm121.cubin",
        &["MK_INKLING2", "MK_PROFILE"],
    );
    println!("cargo:rustc-link-lib=cuda");
}
