//! Build the shared HIP ops of gfx1151 and the hipBLASLt shim into the static library `libnagaya_mk_hip.a`, under the
//! feature `hip`. A build without `hip` compiles nothing and needs no ROCm.
//!
//! Each `.hip` file in `hip/` is device code, and each `.cpp` file is host code. The compiler is ROCm's `clang++`, as
//! for `nagaya-mk-qwen38-gfx1151`. The script gives the build facts to the crate as environment variables, and the
//! identity holds them (spec 12.1, "The plan table of a HIP engine").

use std::path::{Path, PathBuf};
use std::process::Command;

/// The ROCm install.
const ROCM: &str = "/opt/rocm";

/// The flags of every unit. The ops keep IEEE float math, so a kernel computes the same bits on each run.
const COMMON: &[&str] = &[
    "-O3",
    "-std=c++20",
    "-fPIC",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-unused-parameter",
    "-fno-fast-math",
    "-D__HIP_PLATFORM_AMD__=1",
];

/// The flags of a device unit, after `COMMON`.
const DEVICE: &[&str] = &["-x", "hip", "--offload-arch=gfx1151"];

fn run(program: &Path, arguments: &[String], directory: &Path, what: &str) -> Vec<u8> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
        .unwrap_or_else(|error| panic!("{what}: cannot run {}: {error}", program.display()));
    if !output.status.success() {
        panic!(
            "{what} failed with {}:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.stdout
}

fn first_line(bytes: &[u8], what: &str) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .next()
        .unwrap_or_else(|| panic!("{what} printed nothing"))
        .trim()
        .to_owned()
}

/// The SHA-256 of `bytes` as 64 lowercase hex digits.
fn sha2_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The hipBLASLt version from its header: `major.minor.patch-tweak`.
fn hipblaslt_version(rocm: &Path) -> String {
    let header = rocm.join("include").join("hipblaslt").join("hipblaslt-version.h");
    let text = std::fs::read_to_string(&header).unwrap_or_else(|error| panic!("read {}: {error}", header.display()));
    let value = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("#define {name}")))
            .map(|rest| rest.trim().to_owned())
            .unwrap_or_else(|| panic!("{} has no {name}", header.display()))
    };
    format!(
        "{}.{}.{}-{}",
        value("HIPBLASLT_VERSION_MAJOR"),
        value("HIPBLASLT_VERSION_MINOR"),
        value("HIPBLASLT_VERSION_PATCH"),
        value("HIPBLASLT_VERSION_TWEAK")
    )
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=hip");
    if std::env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let rocm = PathBuf::from(ROCM);
    let compiler = rocm.join("llvm").join("bin").join("clang++");
    let archiver = rocm.join("llvm").join("bin").join("llvm-ar");
    let src = manifest.join("hip");

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&src)
        .unwrap_or_else(|error| panic!("read {}: {error}", src.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| matches!(path.extension().and_then(|e| e.to_str()), Some("hip" | "cpp")))
        .collect();
    sources.sort();
    let mut objects = Vec::new();
    for source in &sources {
        let name = source.file_name().unwrap().to_string_lossy().into_owned();
        let mut arguments: Vec<String> = COMMON.iter().map(|flag| (*flag).to_owned()).collect();
        if source.extension().is_some_and(|e| e == "hip") {
            arguments.extend(DEVICE.iter().map(|flag| (*flag).to_owned()));
        }
        // ROCm is a system include, so its own warnings do not stop the build.
        arguments.push("-isystem".to_owned());
        arguments.push(rocm.join("include").display().to_string());
        arguments.push(format!("-I{}", src.display()));
        // The object goes to stdout, so no path of OUT_DIR enters the command or the debug information.
        arguments.extend(["-c".to_owned(), format!("hip/{name}"), "-o".to_owned(), "-".to_owned()]);
        let object = run(&compiler, &arguments, &manifest, &format!("the compile of {name}"));
        let path = out.join(format!("{name}.o"));
        std::fs::write(&path, object).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
        objects.push(path);
    }
    let library = out.join("libnagaya_mk_hip.a");
    let _ = std::fs::remove_file(&library);
    let mut arguments = vec!["crs".to_owned(), library.display().to_string()];
    arguments.extend(objects.iter().map(|object| object.display().to_string()));
    run(&archiver, &arguments, &out, "the archive of libnagaya_mk_hip.a");

    let bytes = std::fs::read(&library).unwrap_or_else(|error| panic!("read {}: {error}", library.display()));
    let compiler_version = first_line(
        &run(&compiler, &["--version".to_owned()], &out, "clang++ --version"),
        "clang++",
    );
    let hip_version = first_line(
        &run(
            &rocm.join("bin").join("hipconfig"),
            &["--version".to_owned()],
            &out,
            "hipconfig --version",
        ),
        "hipconfig",
    );
    println!("cargo:rustc-env=NAGAYA_MK_HIP_BUILD_SHA256={}", sha2_hex(&bytes));
    println!("cargo:rustc-env=NAGAYA_MK_HIP_BUILD_COMPILER={compiler_version}");
    println!("cargo:rustc-env=NAGAYA_MK_HIP_BUILD_HIP={hip_version}");
    println!(
        "cargo:rustc-env=NAGAYA_MK_HIP_BUILD_HIPBLASLT={}",
        hipblaslt_version(&rocm)
    );

    // Each HIP object registers its kernels from a constructor, so every object goes into the binary.
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=nagaya_mk_hip");
    println!("cargo:rustc-link-search=native={}", rocm.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=hipblaslt");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
