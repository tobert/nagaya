#!/usr/bin/env python3
"""Write the fixture of the op builds: the compiles and the links of mk's op libraries, from mk's CMake build.

The fixture is `tests/fixtures/build/ops_commands.json` of nagaya-mk-qwen38-gfx1151. The test `ops_build_flags` checks that
`ops_build.rs` calls the compiler with the flags of mk's own build. The tool reads `build/compile_commands.json` and
`build/build.ninja` of mk, and it writes nothing into mk. It needs only the standard library.

    python3 tools/op_build_fixture.py [--mk MK] [--out OUT]
"""
import argparse
import json
import re
from pathlib import Path

DEFAULT_MK = Path.home() / "src/megakernel-qwen38-flashnext-strixhalo"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "crates/nagaya-mk-qwen38-gfx1151/tests/fixtures/build/ops_commands.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mk", type=Path, default=DEFAULT_MK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build = args.mk / "build"
    root = str(args.mk / "src")
    commands = json.loads((build / "compile_commands.json").read_text())
    compiles = [{"file": e["file"], "command": e["command"]} for e in commands if "/src/ops/" in e["file"]]
    ninja = (build / "build.ninja").read_text()
    links = []
    for match in re.finditer(r"^build src/ops/(libmk_\w+\.so): HIP_SHARED_LIBRARY_LINKER__\w+ (.*)\n((?:  .*\n)+)", ninja, re.M):
        fields = dict(line.strip().split(" = ", 1) for line in match.group(3).splitlines())
        links.append({"library": match.group(1), "language_flags": fields["LANGUAGE_COMPILE_FLAGS"],
                      "link_flags": fields["LINK_FLAGS"], "soname": fields["SONAME"]})
    rules = (build / "CMakeFiles/rules.ninja").read_text()
    rule = re.search(r"^rule HIP_SHARED_LIBRARY_LINKER__mk_qmm_RelWithDebInfo\n  command = (.*)\n", rules, re.M).group(1)
    data = {"reference_src": root, "compiles": sorted(compiles, key=lambda c: c["file"]), "links": sorted(links, key=lambda l: l["library"]),
            "link_rule": rule}
    assert len(data["compiles"]) == len(data["links"]) and data["compiles"], "one compile and one link for each op"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"{args.out.name}: {args.out.stat().st_size} bytes, {len(compiles)} ops")


main()
