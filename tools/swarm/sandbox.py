"""The sandbox evaluator. It runs a program that a model wrote in bwrap, with no network, a read-only root, a memory
limit and a timeout. The worker is binpack.py of this folder."""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(src, seed, timeout=60):
    """The score of a program on the sets of one seed: {"excess": percent} or {"fail": reason}."""
    cmd = ["prlimit", "--as=4000000000", "--",
           "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--ro-bind", HERE, "/w", "--proc", "/proc", "--dev", "/dev",
           "--tmpfs", "/tmp", "--unshare-all", "--die-with-parent", "--new-session", "--chdir", "/tmp",
           "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
           "python3", "-I", "/w/binpack.py", str(seed)]
    try:
        r = subprocess.run(cmd, input=src, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"fail": "timeout"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"fail": f"worker: rc {r.returncode} {r.stderr[-200:]}"}
