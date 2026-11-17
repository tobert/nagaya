#!/usr/bin/env python3
"""Run the op tests against a mutated copy of the op sources, to see that a test fails when the kernel is wrong.

The tool copies `hip/src` of nagaya-mk-qwen38-gfx1151 into a work directory, replaces one exact text in one file,
builds the op libraries from the copy with the flags of `ops_build.rs`, and runs the test target `ops` with
`QWEN38_OPS_LIBDIR` set to the copy. It never changes the sources of the repository, so several runs can go at the same time.

    python3 tools/op_mutate.py LABEL FILE OLD NEW [--tests FILTER] [--work DIR]

FILE is relative to `hip/src` (for example `ops/qmm.h`). OLD must occur once in FILE, or with `--all` at least once, and
then every occurrence changes. The run takes the GPU lock when the
variable `GPU_LOCK` names a lock file, and it refuses to run when `nagaya.service` is active.
Prints the failed tests, or the abort of the test process. It exits with 0 when a test failed or the process aborted,
and with 1 when no test failed.
"""
import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates/nagaya-mk-qwen38-gfx1151"
ROCM = Path("/opt/rocm")
CXX = ROCM / "llvm/bin/clang++"
HIP_DEFINES = ["-DUSE_PROF_API=1", "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1"]
FLAGS = ["-O2", "-g", "-DNDEBUG", "-std=gnu++20", "--offload-arch=gfx1151", "-fPIC", "-Wall", "-Wextra",
         "-Wno-unused-parameter", "-ffp-contract=off", "-fno-fast-math"]
LINK = ["-fPIC", "-O2", "-g", "-DNDEBUG", "--offload-arch=gfx1151", "-shared", "--hip-link", "--rtlib=compiler-rt",
        "-unwindlib=libgcc"]


def run(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{what} failed:\n{' '.join(map(str, cmd))}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def build(src: Path, out: Path):
    builtins = run([CXX, "--print-libgcc-file-name", "--rtlib=compiler-rt"], "compiler runtime")
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (src / "ops").glob("*.hip"))

    def compile_one(name):
        if name == "rt":
            cmd = [CXX, *HIP_DEFINES, f"-I{src / 'rt'}", *FLAGS, "-o", out / "rt.o", "-x", "hip", "-c", src / "rt/rt.cpp"]
        else:
            cmd = [CXX, *HIP_DEFINES, f"-Dqwen38_{name}_EXPORTS", f"-I{src / 'ops'}", f"-I{src / 'rt'}", *FLAGS, "-O3", "-o",
                   out / f"{name}.o", "-x", "hip", "-c", src / "ops" / f"{name}.hip"]
        run(cmd, f"compile {name}")

    with concurrent.futures.ThreadPoolExecutor(len(names) + 1) as pool:
        list(pool.map(compile_one, ["rt", *names]))

    def link_one(name):
        run([CXX, *LINK, f"-Wl,-soname,libqwen38_{name}.so", "-o", out / f"libqwen38_{name}.so", out / f"{name}.o", out / "rt.o",
             ROCM / "lib/libamdhip64.so", builtins], f"link {name}")

    with concurrent.futures.ThreadPoolExecutor(len(names)) as pool:
        list(pool.map(link_one, names))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("file")
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--tests", default="")
    p.add_argument("--all", action="store_true", help="replace every occurrence of OLD")
    p.add_argument("--work", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "op_mutate")
    a = p.parse_args()
    r = subprocess.run(["systemctl", "--user", "is-active", "nagaya.service"], capture_output=True, text=True)
    if r.stdout.strip() == "active":
        sys.exit("nagaya.service is active: stop here and report")
    work = a.work / a.label.replace(" ", "_").replace("/", "_")
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(CRATE / "hip/src", work / "src")
    target = work / "src" / a.file
    text = target.read_text()
    if text.count(a.old) != 1 and not (a.all and text.count(a.old) > 1):
        sys.exit(f"the text occurs {text.count(a.old)} times in {a.file}")
    target.write_text(text.replace(a.old, a.new))
    build(work / "src", work / "lib")
    cmd = ["cargo", "test", "-p", "nagaya-mk-qwen38-gfx1151", "--test", "ops", "--features", "hip", "--", "--test-threads=1"]
    cmd += a.tests.split()
    lock = os.environ.get("GPU_LOCK")
    if lock:
        cmd = ["flock", lock, *cmd]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "QWEN38_OPS_LIBDIR": str(work / "lib")})
    text = r.stdout + r.stderr
    if "could not compile" in text:
        sys.exit("the tests did not compile:\n" + text[-2000:])
    failed = [l for l in text.splitlines() if l.startswith("test ") and l.endswith("FAILED")]
    # A mutation can abort the test process with a device trap: that is a failure too, and no line says which test.
    failed += [l.strip() for l in text.splitlines() if "process didn't exit successfully" in l and "signal" in l]
    summary = [l for l in r.stdout.splitlines() if l.startswith("test result")]
    print(f"=== MUTATION {a.label} ({a.file})")
    print("\n".join(failed) or "NO TEST FAILED")
    print("\n".join(summary[:1]))
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if failed else 1)


main()
