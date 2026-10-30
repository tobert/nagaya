"""Tests of the script of nagaya_web.html in QuickJS, with a stub page and a fake fetch.

Run: python3 -m unittest test_nagaya_web_page (in tools/). The tests need the `quickjs` package, and they skip
without it, so the tools still need only the standard library. Make a venv for them with `uv venv` and
`uv pip install quickjs`.
"""

import json
import pathlib
import re
import unittest

try:
    import quickjs
except ImportError:
    quickjs = None

PAGE = pathlib.Path(__file__).with_name("nagaya_web.html")

# The served models and the profile of b70host, cut to what the page reads.
IDENTITY = {"models": [
    {"model": "gemma-4-26b-a4b", "engine": "nagaya:84828966646e6ce1", "status": "ready", "capabilities": ["generate"]},
    {"model": "sdxl-0", "engine": "sdxl-0", "status": "unloaded", "capabilities": ["generate"],
     "media": {"inputs": [], "outputs": ["image/png"], "text_replies": False}},
    {"model": "sdxl-1", "engine": "sdxl-1", "status": "unloaded", "capabilities": ["generate"],
     "media": {"inputs": [], "outputs": ["image/png"], "text_replies": False}},
    {"model": "embeddinggemma-2", "engine": "embeddinggemma-2", "status": "ready", "capabilities": ["embed", "dry_run"],
     "media": {"inputs": [], "outputs": [], "text_replies": False},
     "embeddings": {"tasks": ["query", "document", "similarity"], "dimensions": [768, 512, 256, 128],
                    "inputs_per_request": 64}},
]}
PROFILE = {"engines": [
    {"name": "gemma", "kind": "mk-gemma4-bmg", "devices": [0, 1]},
    {"name": "sdxl-0", "kind": "mk-sdxl-bmg", "devices": [0], "residency": "on_demand"},
    {"name": "sdxl-1", "kind": "mk-sdxl-bmg", "devices": [1], "residency": "on_demand"},
    {"name": "embeddinggemma-2", "kind": "mk-embeddinggemma2-bmg", "devices": [1], "pinned": True},
]}

# A stub of the DOM that holds each element by id, and a fetch that answers as the nagaya does.
STUB = r"""
var calls = [], confirms = [];
function el(id) { return {id, value: "", textContent: "", innerHTML: "", dataset: {}, style: {}, disabled: false,
  hidden: false, checked: false, classList: {toggle() {}}, addEventListener() {}, querySelector() { return null; },
  appendChild() {}}; }
var els = {};
var document = {getElementById: (id) => els[id] || (els[id] = el(id)), querySelectorAll: () => [],
  querySelector: () => null, addEventListener() {}, createElement: () => el("script"), head: {appendChild() {}}};
var window = {}, localStorage = {getItem: () => null, setItem() {}}, performance = {now: () => 0};
function setTimeout(f) { f(); } function setInterval() {} function alert() {}
function confirm(text) { confirms.push(text); return true; }
function reply(status, v) {
  return Promise.resolve({ok: status < 300, status, text: () => Promise.resolve(JSON.stringify(v)),
                          json: () => Promise.resolve(v)});
}
function fetch(path, o) {
  o = o || {}; calls.push([o.method || "GET", path, o.body || null]);
  if (path === "/council/v2/identity") return reply(200, IDENTITY);
  if (path === "/nagaya/v1/profile") return reply(200, PROFILE);
  const load = path.match(/^\/nagaya\/v1\/engines\/([^/]+)\/load$/);
  if (load) {
    if (!o.body) return reply(409, {error: {message: `nothing changed: engine \`${load[1]}\` does not fit: short`}});
    return reply(202, {engine: load[1], unload: JSON.parse(o.body).unload, load: [load[1]]});
  }
  if (path === "/council/v2/embeddings") {
    // The query is [1, 0], and document i is at the angle i, so the cosine falls with i up to pi.
    const b = JSON.parse(o.body);
    const v = b.inputs.map((t, i) => b.task === "query" ? [1, 0] : [Math.cos(i), Math.sin(i)]);
    return reply(200, {embeddings: v.map((vector, index) => ({index, tokens: 3, vector}))});
  }
  return reply(200, {});
}
"""


@unittest.skipIf(quickjs is None, "the package quickjs is absent")
class Page(unittest.TestCase):
    def run_page(self, test):
        """Run the page script and then `test`, an async body that fills `out`. Give `out`."""
        script = re.search(r"<script>(.*)</script>", PAGE.read_text(), re.S).group(1)
        context = quickjs.Context()
        context.eval(f"var IDENTITY = {json.dumps(IDENTITY)}, PROFILE = {json.dumps(PROFILE)};" + STUB + script +
                     "var out = {}; (async () => { await refresh();" + test +
                     "})().catch((e) => { out.error = String(e); });")
        while context.execute_pending_job():
            pass
        out = json.loads(context.eval("JSON.stringify(out)"))
        self.assertNotIn("error", out)
        return out

    def test_a_load_that_does_not_fit_unloads_the_loaded_engines_on_its_cards_but_no_pinned_engine(self):
        out = self.run_page("""
            out.reply = await loadEngine("sdxl-1");
            out.bodies = calls.filter((c) => c[1].endsWith("/load")).map((c) => c[2]);
            out.confirms = confirms.length;""")
        self.assertEqual(out["bodies"], [None, '{"unload":["gemma"]}'])
        self.assertEqual(out["confirms"], 1)
        self.assertEqual(out["reply"]["unload"], ["gemma"])

    def test_the_documents_rank_by_their_cosine_to_the_query(self):
        out = self.run_page("""
            document.getElementById("em-model").value = "embeddinggemma-2"; fillEmbed();
            document.getElementById("em-query").value = "q";
            document.getElementById("em-docs").value = "zero\\none\\ntwo";
            await document.getElementById("em-go").onclick();
            out.error_text = document.getElementById("em-error").textContent;
            out.ranked = document.getElementById("em-out").innerHTML.match(/<span>[^<]*<\\/span>/g);
            out.bodies = calls.filter((c) => c[1] === "/council/v2/embeddings").map((c) => JSON.parse(c[2]));""")
        self.assertEqual(out["error_text"], "")
        self.assertEqual(out["ranked"], ["<span>zero</span>", "<span>one</span>", "<span>two</span>"])
        self.assertEqual([(b["task"], b["dimensions"]) for b in out["bodies"]], [("document", 768), ("query", 768)])

    def test_the_engine_table_turns_off_the_buttons_that_do_nothing(self):
        out = self.run_page('out.rows = document.getElementById("eng-rows").innerHTML;')
        rows = out["rows"].split("</tr>")
        # gemma is loaded: no load. sdxl-0 is unloaded: no unload. The pinned embedder: neither.
        self.assertRegex(rows[0], r'data-load="gemma" disabled')
        self.assertRegex(rows[1], r'data-unload="sdxl-0"\s+disabled')
        self.assertRegex(rows[3], r'data-load="embeddinggemma-2" disabled')
        self.assertRegex(rows[3], r'data-unload="embeddinggemma-2"\s+disabled')


if __name__ == "__main__":
    unittest.main()
