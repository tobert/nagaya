//! Build the op libraries of mk for the op tests: one shared library `libqwen38_<op>.so` for each `hip/src/ops/<op>.hip`.
//!
//! The commands are the commands of mk's CMake build at mk commit `555a2a3` (`src/ops/CMakeLists.txt`). The compiles use
//! the flags of `build.rs` for the engine, and add `-O3` for the op target. The op tests load these libraries at run
//! time. Each library holds its own copy of the runtime `rt.cpp`, as mk's `libmkrt.a` is static. Thus each library has its
//! own trap record.
//!
//! `build.rs` calls `build_ops` under the feature `hip`. Cargo has no way to build a script only for tests, so a build of
//! the feature `hip` also builds these libraries. The libraries do not enter the engine.

use super::{COMMON, HIP_DEFINES, OFFLOAD_ARCH, ROCM, WARNINGS_AND_FLOATS, compiler, include, strings};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The flags of the link of a HIP shared library in mk's CMake build (`LANGUAGE_COMPILE_FLAGS`, then `LINK_FLAGS`).
const LINK_FLAGS: &[&str] = &[
    "-fPIC",
    "-O2",
    "-g",
    "-DNDEBUG",
    OFFLOAD_ARCH,
    "-shared",
    "--hip-link",
    "--rtlib=compiler-rt",
    "-unwindlib=libgcc",
];

/// The names of the ops: the stem of each `*.hip` file in `src/ops`, in name order.
pub(crate) fn op_names(src: &Path) -> Vec<String> {
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

/// The compile of `ops/<name>.hip`, as mk's `compile_commands.json` gives it for the target `mk_<name>`.
pub(crate) fn op_compile(src: &Path, name: &str, object: &Path) -> Vec<String> {
    let mut arguments = strings(HIP_DEFINES);
    arguments.push(format!("-Dqwen38_{name}_EXPORTS"));
    arguments.push(include(&src.join("ops")));
    arguments.push(include(&src.join("rt")));
    arguments.extend(strings(COMMON));
    arguments.push(OFFLOAD_ARCH.to_owned());
    arguments.extend(strings(WARNINGS_AND_FLOATS));
    // The op target adds -O3 after the build type's -O2 (mk `src/ops/CMakeLists.txt`). The last -O wins.
    arguments.push("-O3".to_owned());
    tail(&mut arguments, object, &src.join("ops").join(format!("{name}.hip")));
    arguments
}

/// The compile of the runtime `rt.cpp` for the op libraries, as mk's `compile_commands.json` gives it for the target `mkrt`.
pub(crate) fn rt_compile(src: &Path, object: &Path) -> Vec<String> {
    let mut arguments = strings(HIP_DEFINES);
    arguments.push(include(&src.join("rt")));
    arguments.extend(strings(COMMON));
    arguments.push(OFFLOAD_ARCH.to_owned());
    arguments.extend(strings(WARNINGS_AND_FLOATS));
    tail(&mut arguments, object, &src.join("rt").join("rt.cpp"));
    arguments
}

fn tail(arguments: &mut Vec<String>, object: &Path, source: &Path) {
    arguments.extend(strings(&["-o"]));
    arguments.push(object.display().to_string());
    arguments.extend(strings(&["-x", "hip", "-c"]));
    arguments.push(source.display().to_string());
}

/// The link of `libqwen38_<name>.so`, as mk's `build.ninja` gives it: the object of the op, the object of the runtime, the HIP
/// runtime and the compiler runtime.
pub(crate) fn op_link(
    name: &str,
    library: &Path,
    object: &Path,
    rt_object: &Path,
    hip_runtime: &Path,
    builtins: &str,
) -> Vec<String> {
    let mut arguments = strings(LINK_FLAGS);
    arguments.push(format!("-Wl,-soname,libqwen38_{name}.so"));
    arguments.push("-o".to_owned());
    arguments.push(library.display().to_string());
    arguments.push(object.display().to_string());
    arguments.push(rt_object.display().to_string());
    arguments.push(hip_runtime.display().to_string());
    arguments.push(builtins.to_owned());
    arguments
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

/// Build every op library into `out/ops`, and give its directory to the crate as `NAGAYA_MK_OPS_DIR`.
///
/// `src` is the directory `hip/src` and `out` is the directory `OUT_DIR`.
pub(crate) fn build_ops(src: &Path, out: &Path) {
    let compiler = compiler();
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
    let hip_runtime: PathBuf = Path::new(ROCM).join("lib").join("libamdhip64.so");

    // The runtime of mk compiles once, and every library links its object. All compiles run first, one thread each.
    let rt_object = directory.join("rt.o");
    let names = op_names(src);
    std::thread::scope(|scope| {
        let (compiler, rt_object) = (&compiler, &rt_object);
        scope.spawn(move || run(compiler, &rt_compile(src, rt_object), "the compile of rt.cpp"));
        for name in &names {
            let directory = &directory;
            scope.spawn(move || {
                let object = directory.join(format!("{name}.o"));
                run(
                    compiler,
                    &op_compile(src, name, &object),
                    &format!("the compile of {name}.hip"),
                );
            });
        }
    });
    std::thread::scope(|scope| {
        for name in &names {
            let (compiler, directory, builtins, hip_runtime, rt_object) =
                (&compiler, &directory, &builtins, &hip_runtime, &rt_object);
            scope.spawn(move || {
                let library = directory.join(format!("libqwen38_{name}.so"));
                let object = directory.join(format!("{name}.o"));
                let arguments = op_link(name, &library, &object, rt_object, hip_runtime, builtins);
                run(compiler, &arguments, &format!("the link of libqwen38_{name}.so"));
            });
        }
    });
    println!("cargo:rustc-env=NAGAYA_MK_OPS_DIR={}", directory.display());
}
