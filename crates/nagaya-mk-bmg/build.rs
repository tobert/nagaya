//! Compiles the kernels for the B70 cards with `ocloc`, ahead of time, when the feature `xe` is on.
//!
//! The flags are mkviolet's (`build.rs` in mkviolet). The output goes to `OUT_DIR`, and the crate embeds it. The Gemma 4
//! binary keeps the command of milestone 15s, so its bytes do not change (milestone 17 step 4).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/megakernel.cl");
    println!("cargo:rerun-if-changed=kernels/qwen3tts_ops.cl");
    println!("cargo:rerun-if-changed=kernels/sdxl_ops.cl");
    println!("cargo:rerun-if-changed=kernels/embeddinggemma2_ops.cl");
    println!("cargo:rerun-if-changed=kernels/lfm3_ops.cl");
    println!("cargo:rerun-if-changed=kernels/lfm3_vision_ops.cl");
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
    // The speech engine's binary: the same source with its ops (MK_QWEN3TTS), which the Gemma 4 binary does not hold.
    let kernels = manifest.join("kernels");
    let options = format!("-cl-std=CL3.0 -D MK_QWEN3TTS -I {}", kernels.display());
    let status = Command::new("ocloc")
        .arg("compile")
        .arg("-file")
        .arg(&source)
        .args([
            "-device",
            "bmg-g31",
            "-options",
            &options,
            "-output",
            "megakernel_qwen3tts",
            "-out_dir",
        ])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("cannot run ocloc: {e}. Install ocloc, or build without the feature xe."));
    assert!(
        status.success(),
        "ocloc did not compile {} with MK_QWEN3TTS for bmg-g31: {status}",
        source.display()
    );
    // The image engine's binary: the speech ops and its own (MK_SDXL). Neither other binary holds the image ops.
    let options = format!("-cl-std=CL3.0 -D MK_QWEN3TTS -D MK_SDXL -I {}", kernels.display());
    let status = Command::new("ocloc")
        .arg("compile")
        .arg("-file")
        .arg(&source)
        .args([
            "-device",
            "bmg-g31",
            "-options",
            &options,
            "-output",
            "megakernel_sdxl",
            "-out_dir",
        ])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("cannot run ocloc: {e}. Install ocloc, or build without the feature xe."));
    assert!(
        status.success(),
        "ocloc did not compile {} with MK_SDXL for bmg-g31: {status}",
        source.display()
    );
    // The embedding engine's binary: the speech ops and its own (MK_EMBEDDINGGEMMA2), with GeGLU as its OP_GEGLU.
    let options = format!(
        "-cl-std=CL3.0 -D MK_QWEN3TTS -D MK_EMBEDDINGGEMMA2 -I {}",
        kernels.display()
    );
    let status = Command::new("ocloc")
        .arg("compile")
        .arg("-file")
        .arg(&source)
        .args([
            "-device",
            "bmg-g31",
            "-options",
            &options,
            "-output",
            "megakernel_embeddinggemma2",
            "-out_dir",
        ])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("cannot run ocloc: {e}. Install ocloc, or build without the feature xe."));
    assert!(
        status.success(),
        "ocloc did not compile {} with MK_EMBEDDINGGEMMA2 for bmg-g31: {status}",
        source.display()
    );
    // The LFM3 engine's binary: the speech ops and its own (MK_LFM3), with the rotated pairs of a partial rotation. It
    // also holds the ops of embeddinggemma2_ops.cl for its vision tower, with SwiGLU as its OP_GEGLU.
    let options = format!("-cl-std=CL3.0 -D MK_QWEN3TTS -D MK_LFM3 -I {}", kernels.display());
    let status = Command::new("ocloc")
        .arg("compile")
        .arg("-file")
        .arg(&source)
        .args([
            "-device",
            "bmg-g31",
            "-options",
            &options,
            "-output",
            "megakernel_lfm3",
            "-out_dir",
        ])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("cannot run ocloc: {e}. Install ocloc, or build without the feature xe."));
    assert!(
        status.success(),
        "ocloc did not compile {} with MK_LFM3 for bmg-g31: {status}",
        source.display()
    );
    // The compiler is in the plan table lineage of the identity (spec 12.1).
    let version = Command::new("ocloc")
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("cannot run ocloc --version: {e}"));
    let version = String::from_utf8_lossy(&version.stdout).trim().to_owned();
    assert!(!version.is_empty(), "ocloc --version printed no version");
    println!("cargo:rustc-env=BMG_OCLOC_VERSION={version}");
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
    println!("cargo:rustc-env=BMG_IGC_REVISION={igc}");
    for name in [
        "megakernel_bmg.bin",
        "megakernel_qwen3tts_bmg.bin",
        "megakernel_sdxl_bmg.bin",
        "megakernel_embeddinggemma2_bmg.bin",
        "megakernel_lfm3_bmg.bin",
    ] {
        let binary = out.join(name);
        assert!(
            binary.is_file(),
            "ocloc wrote no {}. Read the ocloc output above.",
            binary.display()
        );
    }
}
