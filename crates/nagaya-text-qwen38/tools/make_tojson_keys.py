#!/usr/bin/env python3
"""Write tests/fixtures/tojson_keys.json: `tojson` of maps with keys that are not strings.

JSON text has only string keys, so these cases are template sources. Each source runs in jinja2 with the setup of
mk's `service/template.py` (`ImmutableSandboxedEnvironment`, `tojson` is `json.dumps` with `ensure_ascii=False`).
A source that Python refuses has `"error": true`. Run it from the crate directory:

    python3 -I tools/make_tojson_keys.py
"""
import json
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

SOURCES = [
    "{1: 2}", "{-3: 'x'}", "{0: 0}", "{true: 1}", "{false: 1}", "{none: 1}", "{1.5: 1}", "{1.0: 1}", "{-0.0: 1}",
    "{1e16: 1}", "{1e-5: 1}", "{0.1: 1}", "{170141183460469231731687303715884105727: 1}",
    "170141183460469231731687303715884105728", "340282366920938463463374607431768211455",
    "{170141183460469231731687303715884105728: 1}", "{340282366920938463463374607431768211455: 1}",
"{'a': 1, 2: 3, none: 4, true: 5, 1.5: 6}",
    "{1: {2: {3: [4, {5: 6}]}}}", "[{1: 2}, {'k': {false: none}}]", "{'é': 1, 7: 'é'}",
    "{(1, 2): 3}", "{'a': {(1,): 2}}", "{[1][0]: 1}",
]


def main() -> None:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    cases = []
    for src in SOURCES:
        template = "{{ (" + src + ")|tojson }}"
        try:
            cases.append({"src": template, "out": env.from_string(template).render()})
        except Exception:
            cases.append({"src": template, "error": True})
    lines = ['{"jinja2": ' + json.dumps(jinja2.__version__) + ',', '"cases": [']
    lines.append(",\n".join(json.dumps(c, separators=(",", ":"), ensure_ascii=True) for c in cases))
    lines.append("]}")
    Path("tests/fixtures/tojson_keys.json").write_text("\n".join(lines) + "\n", encoding="ascii")


main()
