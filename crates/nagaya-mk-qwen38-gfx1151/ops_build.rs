//! Build the op libraries of mk for the op tests: one shared library `libmk_<op>.so` for each `hip/src/ops/<op>.hip`.
//!
//! The commands are the commands of mk's CMake build at mk commit `555a2a3` (`src/ops/CMakeLists.txt`). They use the
//! flags of `build.rs` for the engine, and add `-O3` for the op target. The op tests load these libraries at run time.
//! Each library holds its own copy of the runtime `rt.cpp`, as mk's `libmkrt.a` is static. Thus each library has its own
//! trap record.
//!
//! `build.rs` calls `build_ops` only under the feature `hip`. Cargo has no way to build a script only for tests, so a
//! build of the feature `hip` also builds these libraries.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The ROCm install. mk's CMake build names the same path (mk `CMakeLists.txt:4-5`).
const ROCM: &str = "/opt/rocm";

/// The flags that mk's top-level CMake file and its build type `RelWithDebInfo` give to every translation unit, in
/// CMake's order: the build type, the target and the warnings and float flags.
const FLAGS: &[&str] = &[
    "-O2",
    "-g",
    "-DNDEBUG",
    "-std=gnu++20",
    "--offload-arch=gfx1151",
    "-fPIC",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-ffp-contract=off",
    "-fno-fast-math",
];

/// The definitions that CMake's target `hip::host` gives to each HIP translation unit.
const HIP_DEFINES: &[&str] = &["-DUSE_PROF_API=1", "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1"];

/// The flags of the link of a HIP shared library in mk's CMake build.
const LINK_FLAGS: &[&str] = &[
    "-fPIC",
    "-O2",
    "-g",
    "-DNDEBUG",
    "--offload-arch=gfx1151",
    "-shared",
    "--hip-link",
    "--rtlib=compiler-rt",
    "-unwindlib=libgcc",
];

fn strings(flags: &[&str]) -> Vec<String> {
    flags.iter().map(|flag| (*flag).to_owned()).collect()
}

/// Run a command, and stop the build with its output when it fails.
fn run(program: &Path, arguments: &[String], what: &str) -> String {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("{what}: cannot run {}: {error}", program.display()));
    if !output.status.success() {
        panic!(
            "{what} failed with {}:\n{} {}\n{}{}",
            output.status,
            program.display(),
            arguments.join(" "),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

/// The names of the ops: the stem of each `*.hip` file in `src/ops`, in name order.
fn op_names(src: &Path) -> Vec<String> {
    let directory = src.join("ops");
    let mut names: Vec<String> = std::fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
                .path()
        })
        .filter(|path| path.extension().is_some_and(|extension| extension == "hip"))
        .map(|path| {
            path.file_stem()
                .expect("a file has a stem")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no op source in {}", directory.display());
    names
}

/// Compile one source as HIP into `object`. `define` is the name of the CMake export definition, or empty.
fn compile(compiler: &Path, src: &Path, source: &Path, object: &Path, define: Option<&str>, op_flag: bool) {
    let mut arguments = strings(HIP_DEFINES);
    arguments.extend(define.map(|name| format!("-D{name}")));
    if define.is_some() {
        arguments.push(format!("-I{}", src.join("ops").display()));
    }
    arguments.push(format!("-I{}", src.join("rt").display()));
    arguments.extend(strings(FLAGS));
    // The op target adds -O3 after the build type's -O2 (mk `src/ops/CMakeLists.txt`). The last -O wins.
    if op_flag {
        arguments.push("-O3".to_owned());
    }
    arguments.extend(strings(&["-o"]));
    arguments.push(object.display().to_string());
    arguments.extend(strings(&["-x", "hip", "-c"]));
    arguments.push(source.display().to_string());
    run(compiler, &arguments, &format!("the compile of {}", source.display()));
}

/// Build every op library into `out/ops`, and give its directory to the crate as `NAGAYA_MK_OPS_DIR`.
///
/// `src` is the directory `hip/src` and `out` is the directory `OUT_DIR`.
pub(crate) fn build_ops(src: &Path, out: &Path) {
    let rocm = Path::new(ROCM);
    let compiler = rocm.join("llvm").join("bin").join("clang++");
    assert!(
        compiler.is_file(),
        "the feature hip needs {}; install ROCm at {ROCM}, or build without the feature hip",
        compiler.display()
    );
    let directory = out.join("ops");
    std::fs::create_dir_all(&directory).unwrap_or_else(|error| panic!("make {}: {error}", directory.display()));
    let builtins = run(
        &compiler,
        &strings(&["--print-libgcc-file-name", "--rtlib=compiler-rt"]),
        "the path of the compiler runtime",
    );
    let hip_runtime: PathBuf = rocm.join("lib").join("libamdhip64.so");

    // The runtime of mk compiles once, and every library links its object. All compiles run first, one thread each.
    let rt_object = directory.join("rt.o");
    let names = op_names(src);
    std::thread::scope(|scope| {
        scope.spawn(|| compile(&compiler, src, &src.join("rt").join("rt.cpp"), &rt_object, None, false));
        for name in &names {
            let (compiler, directory) = (&compiler, &directory);
            scope.spawn(move || {
                let source = src.join("ops").join(format!("{name}.hip"));
                let object = directory.join(format!("{name}.o"));
                compile(
                    compiler,
                    src,
                    &source,
                    &object,
                    Some(&format!("mk_{name}_EXPORTS")),
                    true,
                );
            });
        }
    });
    std::thread::scope(|scope| {
        for name in &names {
            let (compiler, directory, builtins, hip_runtime, rt_object) =
                (&compiler, &directory, &builtins, &hip_runtime, &rt_object);
            scope.spawn(move || {
                let library = directory.join(format!("libmk_{name}.so"));
                let mut arguments = strings(LINK_FLAGS);
                arguments.push(format!("-Wl,-soname,libmk_{name}.so"));
                arguments.push("-o".to_owned());
                arguments.push(library.display().to_string());
                arguments.push(directory.join(format!("{name}.o")).display().to_string());
                arguments.push(rt_object.display().to_string());
                arguments.push(hip_runtime.display().to_string());
                arguments.push(builtins.clone());
                run(compiler, &arguments, &format!("the link of libmk_{name}.so"));
            });
        }
    });
    println!("cargo:rustc-env=NAGAYA_MK_OPS_DIR={}", directory.display());
}
