//! Build mk's engine from `hip/` into the static library `libmkengine.a`, under the feature `hip`.
//!
//! The commands are the commands of mk's CMake build at mk commit `555a2a3`, from mk's `build/compile_commands.json`.
//! The flags are part of the model's definition: the conformance gates compare bits. Thus this script calls the
//! compiler with an exact list of flags, and adds no flag of its own. A build without `hip` compiles nothing and
//! needs no ROCm.
//!
//! The script gives the build facts to the crate as environment variables. The identity holds them (spec 12.1).

use std::path::{Path, PathBuf};
use std::process::Command;

/// The ROCm install. mk's CMake build names the same path (mk `CMakeLists.txt:4-5`).
const ROCM: &str = "/opt/rocm";

/// The one device target. mk builds for gfx1151 only.
const OFFLOAD_ARCH: &str = "--offload-arch=gfx1151";

/// The flags that mk's top-level CMake file and its build type `RelWithDebInfo` give to every translation unit.
///
/// `-ffp-contract=off -fno-fast-math` keep the float operations of the model's definition. The order is CMake's order.
const COMMON: &[&str] = &["-O2", "-g", "-DNDEBUG", "-std=gnu++20"];

/// The warning and float flags, after the target flags, in CMake's order.
const WARNINGS_AND_FLOATS: &[&str] = &[
    "-fPIC",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-ffp-contract=off",
    "-fno-fast-math",
];

/// The definitions that CMake's target `hip::host` gives to each HIP translation unit.
const HIP_DEFINES: &[&str] = &["-DUSE_PROF_API=1", "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1"];

/// The translation units of mk's target `mkengine` (mk `src/engine/CMakeLists.txt:1`).
const ENGINE_UNITS: &[&str] = &[
    "engine.hip",
    "mega.hip",
    "blocks.hip",
    "capi.hip",
    "convert.hip",
    "round.hip",
];

/// The plain C++ translation units of mk's target `mkgguf_static` (mk `src/gguf/CMakeLists.txt:5-8`).
const GGUF_UNITS: &[&str] = &["gguf.cpp", "dequant.cpp"];

/// The compiler writes each object to stdout, and the script writes it to `OUT_DIR`.
///
/// clang hashes the command line into the name of a HIP symbol (`__hip_cuid_`), and it writes the working directory
/// into the debug information. Thus a path of `OUT_DIR` in the command, or `OUT_DIR` as the working directory, makes
/// the archive differ for each profile and feature set. Its SHA-256 is in the identity, so it must not differ.
const OBJECT_TO_STDOUT: &[&str] = &["-o", "-"];

/// One compile: the source, the object and the arguments in their order.
pub(crate) struct Unit {
    pub(crate) source: PathBuf,
    pub(crate) object: PathBuf,
    pub(crate) arguments: Vec<String>,
}

/// The compiler of every unit. mk's CMake build calls it directly, not the wrapper hipcc (mk `CMakeLists.txt:4-5`).
pub(crate) fn compiler() -> PathBuf {
    Path::new(ROCM).join("llvm").join("bin").join("clang++")
}

/// Every compile of the engine, in the order of mk's targets: the engine units, `rt.cpp`, then the GGUF units.
///
/// `src` is the directory `hip/src`, and `out` is the directory of the objects.
pub(crate) fn units(src: &Path, out: &Path) -> Vec<Unit> {
    let mut units: Vec<Unit> = ENGINE_UNITS.iter().map(|name| engine_unit(src, out, name)).collect();
    units.push(rt_unit(src, out));
    units.extend(GGUF_UNITS.iter().map(|name| gguf_unit(src, out, name)));
    units
}

fn strings(flags: &[&str]) -> Vec<String> {
    flags.iter().map(|flag| (*flag).to_owned()).collect()
}

fn include(directory: &Path) -> String {
    format!("-I{}", directory.display())
}

/// The compile of one engine unit, as mk's `compile_commands.json` gives it for `src/engine/engine.hip`.
fn engine_unit(src: &Path, out: &Path, name: &str) -> Unit {
    let mut arguments = strings(HIP_DEFINES);
    arguments.push("-Dmkengine_EXPORTS".to_owned());
    for directory in ["engine", "ops", "rt", "gguf"] {
        arguments.push(include(&src.join(directory)));
    }
    arguments.extend(strings(COMMON));
    arguments.push(OFFLOAD_ARCH.to_owned());
    arguments.extend(strings(WARNINGS_AND_FLOATS));
    // The engine target adds -O3 after the build type's -O2 (mk `src/engine/CMakeLists.txt:4`). The last -O wins.
    arguments.push("-O3".to_owned());
    hip_tail(
        src.join("engine").join(name),
        out.join(format!("engine.{name}.o")),
        arguments,
    )
}

/// The compile of mk's runtime `rt.cpp` as HIP (mk `src/rt/CMakeLists.txt:2`).
fn rt_unit(src: &Path, out: &Path) -> Unit {
    let mut arguments = strings(HIP_DEFINES);
    arguments.push(include(&src.join("rt")));
    arguments.extend(strings(COMMON));
    arguments.push(OFFLOAD_ARCH.to_owned());
    arguments.extend(strings(WARNINGS_AND_FLOATS));
    hip_tail(src.join("rt").join("rt.cpp"), out.join("rt.rt.cpp.o"), arguments)
}

/// Add `-o - -x hip -c <source>`, in the order of mk's compile commands.
fn hip_tail(source: PathBuf, object: PathBuf, mut arguments: Vec<String>) -> Unit {
    arguments.extend(strings(OBJECT_TO_STDOUT));
    arguments.extend(strings(&["-x", "hip", "-c"]));
    arguments.push(source.display().to_string());
    Unit {
        source,
        object,
        arguments,
    }
}

/// The compile of one GGUF unit as plain C++, with no HIP flag (mk `src/gguf/CMakeLists.txt:9`).
fn gguf_unit(src: &Path, out: &Path, name: &str) -> Unit {
    let source = src.join("gguf").join(name);
    let object = out.join(format!("gguf.{name}.o"));
    let mut arguments = vec![include(&src.join("gguf")), "-x".to_owned(), "c++".to_owned()];
    arguments.extend(strings(COMMON));
    arguments.extend(strings(WARNINGS_AND_FLOATS));
    arguments.extend(strings(OBJECT_TO_STDOUT));
    arguments.push("-c".to_owned());
    arguments.push(source.display().to_string());
    Unit {
        source,
        object,
        arguments,
    }
}

/// Run a command in `directory`, and give its stdout. Stop the build with its stderr when it fails.
fn run(program: &Path, arguments: &[String], directory: &Path, what: &str) -> Vec<u8> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(directory)
        .output()
        .unwrap_or_else(|error| panic!("{what}: cannot run {}: {error}", program.display()));
    if !output.status.success() {
        panic!(
            "{what} failed with {}:\n{} {}\n{}",
            output.status,
            program.display(),
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.stdout
}

/// The first line of a command's output, trimmed.
fn first_line(bytes: &[u8], what: &str) -> String {
    let text = String::from_utf8_lossy(bytes);
    let line = text.lines().next().unwrap_or_default().trim().to_owned();
    assert!(!line.is_empty(), "{what} printed no version");
    line
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=hip");
    if std::env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let src = manifest.join("hip").join("src");
    let rocm = Path::new(ROCM);
    let compiler = compiler();
    let archiver = rocm.join("llvm").join("bin").join("llvm-ar");
    for tool in [&compiler, &archiver] {
        assert!(
            tool.is_file(),
            "the feature hip needs {}; install ROCm at {ROCM}, or build without the feature hip",
            tool.display()
        );
    }

    let units = units(&src, &out);
    for unit in &units {
        assert!(
            unit.source.is_file(),
            "the mk source {} is missing",
            unit.source.display()
        );
    }
    // Each unit compiles on its own thread. mega.hip takes the most time.
    std::thread::scope(|scope| {
        for unit in &units {
            let compiler = &compiler;
            let manifest = &manifest;
            scope.spawn(move || {
                // The manifest directory is the working directory: it is the same for each profile.
                let what = format!("the compile of {}", unit.source.display());
                let object = run(compiler, &unit.arguments, manifest, &what);
                assert!(!object.is_empty(), "{what} wrote no object");
                std::fs::write(&unit.object, object)
                    .unwrap_or_else(|error| panic!("write {}: {error}", unit.object.display()));
            });
        }
    });

    let library = out.join("libmkengine.a");
    if library.exists() {
        std::fs::remove_file(&library).unwrap_or_else(|error| panic!("remove {}: {error}", library.display()));
    }
    // llvm-ar writes a deterministic archive by default: no time stamps, owners or modes.
    let mut arguments = vec!["crs".to_owned(), library.display().to_string()];
    arguments.extend(units.iter().map(|unit| unit.object.display().to_string()));
    run(&archiver, &arguments, &out, "the archive of libmkengine.a");

    let bytes = std::fs::read(&library).unwrap_or_else(|error| panic!("read {}: {error}", library.display()));
    let sha256: String = sha2_hex(&bytes);
    let compiler_version = first_line(
        &run(&compiler, &["--version".to_owned()], &out, "clang++ --version"),
        "clang++",
    );
    let hipconfig = rocm.join("bin").join("hipconfig");
    let hip_version = first_line(
        &run(&hipconfig, &["--version".to_owned()], &out, "hipconfig --version"),
        "hipconfig",
    );
    println!("cargo:rustc-env=MK_ENGINE_BUILD_SHA256={sha256}");
    println!("cargo:rustc-env=MK_ENGINE_BUILD_COMPILER={compiler_version}");
    println!("cargo:rustc-env=MK_ENGINE_BUILD_HIP={hip_version}");

    // Every object of the archive goes into the binary. Each HIP object registers its kernels from a constructor, and
    // the linker drops an object that no symbol names.
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=mkengine");
    // The HIP runtime and the C++ library stay shared, as for mk's libmkengine.so.
    println!("cargo:rustc-link-search=native={}", rocm.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// The SHA-256 of `bytes` as 64 lowercase hex digits.
fn sha2_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
