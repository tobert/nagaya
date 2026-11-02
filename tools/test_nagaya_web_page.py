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
    {"model": "voice", "engine": "voice", "status": "ready", "capabilities": ["generate"],
     "media": {"inputs": ["audio/wav"], "outputs": ["audio/wav"], "text_replies": False}},
    {"model": "pony-0", "engine": "pony-0", "status": "ready", "capabilities": ["generate"],
     "media": {"inputs": ["image/png"], "outputs": ["image/png"], "text_replies": False}},
]}
PROFILE = {"engines": [
    {"name": "gemma", "kind": "mk-gemma4-bmg", "devices": [0, 1]},
    {"name": "sdxl-0", "kind": "mk-sdxl-bmg", "devices": [0], "residency": "on_demand"},
    {"name": "sdxl-1", "kind": "mk-sdxl-bmg", "devices": [1], "residency": "on_demand"},
    {"name": "embeddinggemma-2", "kind": "mk-embeddinggemma2-bmg", "devices": [1], "pinned": True},
    {"name": "pony-0", "kind": "mk-sdxl-bmg", "devices": [0], "residency": "on_demand"},
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
// A stream of server-sent events, read in one piece. QuickJS has no TextDecoder, so the reader gives text.
class TextDecoder { decode(v) { return v || ""; } }
class AbortController { constructor() { this.signal = {}; } abort() {} }
function sse(list) {
  let text = list.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join("");
  const read = () => { const value = text; text = ""; return Promise.resolve(value ? {value, done: false} : {done: true}); };
  return Promise.resolve({ok: true, status: 200, body: {getReader: () => ({read})}});
}
function reply(status, v) {
  return Promise.resolve({ok: status < 300, status, text: () => Promise.resolve(JSON.stringify(v)),
                          json: () => Promise.resolve(v)});
}
function fetch(path, o) {
  // The first nagaya answers its paths under /at/b70host/ as the plain paths, and the calls record them plain.
  path = path.replace(/^\/at\/b70host\//, "/");
  o = o || {}; calls.push([o.method || "GET", path, o.body || null]);
  if (path === "/app/nagayas") return reply(200, {nagayas: [{name: "b70host", url: "http://a"}, {name: "gfx1151host", url: "http://b"}]});
  if (path === "/at/gfx1151host/council/v2/identity") return reply(200, {models: [{model: "qwen-image-fusal",
    engine: "qwen-image-fusal", status: "ready", capabilities: ["generate"],
    media: {inputs: [], outputs: ["image/png"], text_replies: false}}]});
  if (path === "/at/gfx1151host/nagaya/v1/profile") return reply(200, {engines: [{name: "qwen-image-fusal",
    kind: "qwenimage21", devices: [0]}]});
  if (path === "/council/v2/identity") return reply(200, IDENTITY);
  if (path === "/nagaya/v1/profile" && o.method === "PUT") {
    return reply(202, {actions: JSON.parse(o.body).engines.map((e) => ({engine: e.name, action: "keep"})), restart: false});
  }
  if (path === "/nagaya/v1/profile") return reply(200, PROFILE);
  const load = path.match(/^\/nagaya\/v1\/engines\/([^/]+)\/load$/);
  if (load) {
    // The message of a load that does not fit, as the nagaya writes it (process/loads.rs). It names the engines in the
    // way, and the page does not guess them.
    if (!o.body) return reply(409, {error: {type: "invalid_request", param: "name", message: "nothing changed: engine `" +
      load[1] + "` does not fit: device(1) lacks 1024 bytes; the loaded engines in the way are `gemma`, `voice`, and " +
      "a load can name them in `unload`; the pinned engines `embeddinggemma-2` stay loaded"}});
    return reply(202, {engine: load[1], unload: JSON.parse(o.body).unload, load: [load[1]]});
  }
  if (path === "/council/v2/embeddings") {
    // The query is [1, 0], and document i is at the angle i, so the cosine falls with i up to pi.
    const b = JSON.parse(o.body);
    const v = b.inputs.map((t, i) => b.task === "query" ? [1, 0] : [Math.cos(i), Math.sin(i)]);
    return reply(200, {embeddings: v.map((vector, index) => ({index, tokens: 3, vector}))});
  }
  if (path === "/council/v2/generations" && JSON.parse(o.body).output?.options?.strength > 1) {
    return reply(400, {error: {type: "invalid_request", param: "output.options.strength", message: "nothing ran: too much"}});
  }
  if (/\/council\/v2\/generations$/.test(path) && JSON.parse(o.body).stream) {
    return sse([["progress", {index: 0, step: 1, of: 1}],
                   ["done", {outputs: [{index: 0, id: "blake3:" + "d".repeat(64), mime: "image/png"}], seed: 7}]]);
  }
  if (path === "/council/v2/generations") {
    return reply(200, {outputs: [{index: 0, id: "blake3:" + "a".repeat(64), mime: "audio/wav"}],
                       usage: {media: {written: {objects: 1, bytes: 80684, audio_ms: 1680}}}});
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

    def test_a_load_that_does_not_fit_unloads_the_engines_that_the_nagaya_names_in_its_way(self):
        out = self.run_page("""
            out.reply = await loadEngine("sdxl-1");
            out.bodies = calls.filter((c) => c[1].endsWith("/load")).map((c) => c[2]);
            out.confirms = confirms;""")
        self.assertEqual(out["bodies"], [None, '{"unload":["gemma","voice"]}'])
        self.assertEqual(len(out["confirms"]), 1)
        self.assertIn("embeddinggemma-2 stays loaded", out["confirms"][0])
        self.assertEqual(out["reply"]["unload"], ["gemma", "voice"])

    def test_the_choice_of_a_nagaya_sends_each_request_to_it_and_each_image_keeps_its_nagaya(self):
        out = self.run_page("""
            out.hosts = document.getElementById("host").innerHTML;
            document.getElementById("host").value = "gfx1151host";
            await document.getElementById("host").onchange();
            out.models = document.getElementById("img-model").innerHTML;
            out.paths = calls.map((c) => c[1]).filter((p) => p.endsWith("/identity"));
            out.here = tile({id: "blake3:" + "b".repeat(64), host: "gfx1151host", model: "m", prompt: "p", seed: 1,
                             size: "64x64", steps: 1});
            out.there = tile({id: "blake3:" + "c".repeat(64), model: "m", prompt: "p", seed: 1, size: "64x64",
                              steps: 1});""")
        self.assertIn('<option value="gfx1151host" title="http://b">gfx1151host</option>', out["hosts"])
        self.assertEqual(out["models"], '<option value="qwen-image-fusal">qwen-image-fusal</option>')
        self.assertEqual(out["paths"][-1], "/at/gfx1151host/council/v2/identity")
        self.assertIn('src="/at/gfx1151host/council/v2/objects/blake3:bbbb', out["here"])
        # An image from before the choice of nagayas belongs to the first nagaya, which serves the plain paths.
        self.assertIn('src="/council/v2/objects/blake3:cccc', out["there"])

    # The values of the image form, as a person sets them, for both kinds of image model.
    FORM = """
        const set = (id, v) => { document.getElementById(id).value = v; };
        set("img-size", "1024x1024"); set("img-steps", "30"); set("img-count", "1"); set("img-cfg", "7");
        set("img-sampler", "dpm++ 2m karras"); set("img-spacing", "leading"); set("img-clip", "2");
        set("img-prompt", "a cat"); set("img-negative", "blurry"); set("img-seed", "");
        const body = () => JSON.parse(calls.filter((c) => /generations$/.test(c[1])).pop()[2]);"""

    def test_an_sdxl_request_sends_the_options_of_the_form(self):
        out = self.run_page(self.FORM + """
            set("img-model", "pony-0"); drawImageForm();
            await document.getElementById("img-go").onclick();
            out.error_text = document.getElementById("img-error").textContent;
            out.options = body().output.options;""")
        self.assertEqual(out["error_text"], "")
        self.assertEqual(out["options"], {"width": 1024, "height": 1024, "steps": 30, "cfg": 7,
                                          "sampler": "dpm++ 2m karras", "count": 1, "negative": "blurry",
                                          "clip_skip": 2})

    def test_a_qwen_image_request_sends_only_the_options_that_the_model_takes(self):
        out = self.run_page(self.FORM + """
            document.getElementById("host").value = "gfx1151host";
            await document.getElementById("host").onchange();
            set("img-model", "qwen-image-fusal"); drawImageForm();
            out.off = ["img-negative", "img-cfg", "img-sampler", "img-spacing", "img-clip", "img-lora-file"]
              .filter((id) => !document.getElementById(id).disabled);
            set("img-steps", "40");
            await document.getElementById("img-go").onclick();
            out.error_text = document.getElementById("img-error").textContent;
            out.options = body().output.options;""")
        self.assertEqual(out["error_text"], "")
        self.assertEqual(out["off"], [])
        self.assertEqual(out["options"], {"width": 1024, "height": 1024, "steps": 40, "count": 1})

    # A start image and a mask as the page holds them. The test gives their object ids, because QuickJS has no canvas.
    INPUTS = """
        init = {width: 832, height: 1216};
        inputObjects = async () => ({image: "blake3:" + "e".repeat(64), mask: mask});"""

    def test_an_inpainting_request_puts_the_image_and_the_mask_first_and_lets_the_image_set_the_size(self):
        out = self.run_page(self.FORM + "let mask = 'blake3:' + 'f'.repeat(64);" + self.INPUTS + """
            set("img-model", "pony-0"); drawImageForm();
            out.box = document.getElementById("img-init-box").hidden;
            set("img-strength", "0.6"); set("img-mask-blur", "8");
            await document.getElementById("img-go").onclick();
            out.error_text = document.getElementById("img-error").textContent;
            out.body = body();""")
        self.assertEqual(out["error_text"], "")
        self.assertFalse(out["box"])
        self.assertEqual(out["body"]["context"]["messages"][-1]["content"], [
            {"type": "object", "id": "blake3:" + "e" * 64, "mime": "image/png"},
            {"type": "object", "id": "blake3:" + "f" * 64, "mime": "image/png"},
            {"type": "text", "text": "a cat"}])
        options = out["body"]["output"]["options"]
        self.assertNotIn("width", options)
        self.assertNotIn("height", options)
        self.assertEqual((options["strength"], options["mask_blur"]), (0.6, 8))

    def test_an_img2img_request_with_no_mask_sends_no_mask_and_no_blur_and_the_default_strength(self):
        out = self.run_page(self.FORM + "let mask = null;" + self.INPUTS + """
            set("img-model", "pony-0"); drawImageForm();
            set("img-strength", ""); set("img-mask-blur", "8");
            await document.getElementById("img-go").onclick();
            out.body = body();""")
        self.assertEqual([p["type"] for p in out["body"]["context"]["messages"][-1]["content"]], ["object", "text"])
        self.assertNotIn("mask_blur", out["body"]["output"]["options"])
        self.assertNotIn("strength", out["body"]["output"]["options"])

    def test_the_refusal_of_an_image_job_shows_as_the_nagaya_wrote_it(self):
        out = self.run_page(self.FORM + "let mask = null;" + self.INPUTS + """
            set("img-model", "pony-0"); drawImageForm();
            set("img-strength", "2");
            await document.getElementById("img-go").onclick();
            out.error_text = document.getElementById("img-error").textContent;""")
        self.assertEqual(out["error_text"], "output.options.strength: nothing ran: too much")

    def test_a_model_with_no_image_input_hides_the_start_image_and_refuses_to_send_one(self):
        out = self.run_page(self.FORM + "let mask = null;" + self.INPUTS + """
            document.getElementById("host").value = "gfx1151host";
            await document.getElementById("host").onchange();
            set("img-model", "qwen-image-fusal"); drawImageForm();
            out.box = document.getElementById("img-init-box").hidden;
            const before = calls.length;
            await document.getElementById("img-go").onclick();
            out.sent = calls.slice(before).filter((c) => /generations$/.test(c[1])).length;
            out.error_text = document.getElementById("img-error").textContent;""")
        self.assertTrue(out["box"])
        self.assertEqual(out["sent"], 0)
        self.assertIn("takes no start image", out["error_text"])

    def test_a_pin_puts_the_profile_with_the_engine_resident_and_pinned_and_changes_no_other_engine(self):
        out = self.run_page("""
            out.reply = await pinEngine("pony-0", true);
            out.bodies = calls.filter((c) => c[0] === "PUT" && c[1] === "/nagaya/v1/profile").map((c) => JSON.parse(c[2]));""")
        self.assertEqual(len(out["bodies"]), 1)
        want = json.loads(json.dumps(PROFILE))
        want["engines"][4] = {"name": "pony-0", "kind": "mk-sdxl-bmg", "devices": [0], "pinned": True}
        self.assertEqual(out["bodies"][0], want)

    def test_an_unpin_puts_the_profile_with_the_pin_off(self):
        out = self.run_page("""
            await pinEngine("embeddinggemma-2", false);
            out.body = JSON.parse(calls.filter((c) => c[0] === "PUT" && c[1] === "/nagaya/v1/profile")[0][2]);""")
        self.assertEqual(out["body"]["engines"][3]["pinned"], False)
        self.assertEqual(out["body"]["engines"][:3], PROFILE["engines"][:3])

    def test_the_engine_table_offers_a_pin_only_for_a_loaded_engine(self):
        out = self.run_page('out.rows = document.getElementById("eng-rows").innerHTML;')
        rows = out["rows"].split("</tr>")
        self.assertRegex(rows[1], r'data-pin="sdxl-0"[^>]*disabled')
        self.assertRegex(rows[4], r'data-pin="pony-0"\s*>Pin<')
        self.assertRegex(rows[3], r'data-unpin="embeddinggemma-2"\s*>Unpin<')

    def test_a_line_of_speech_sends_the_sampler_that_is_set_and_no_member_that_is_empty(self):
        out = self.run_page("""
            const set = (id, v) => { document.getElementById(id).value = v; };
            const speak = async () => {
              set("sp-input", "hello"); await document.getElementById("sp-go").onclick();
              return JSON.parse(calls.filter((c) => /generations$/.test(c[1])).pop()[2]).sample;
            };
            set("sp-model", "voice"); set("sp-seed", "5"); set("sp-temp", "0.7"); set("sp-topk", "20");
            out.set = await speak();
            set("sp-seed", ""); set("sp-temp", ""); set("sp-topk", "");
            out.empty = await speak() ?? null;""")
        self.assertEqual(out["set"], {"seed": 5, "temperature": 0.7, "top_k": 20})
        self.assertIsNone(out["empty"])

    def test_an_error_names_its_param_when_the_message_does_not(self):
        out = self.run_page("""
            try { await call("POST", "/council/v2/generations", {output: {options: {strength: 2}}}); }
            catch (e) { out.message = e.message; out.param = e.param; }""")
        self.assertEqual(out["message"], "output.options.strength: nothing ran: too much")
        self.assertEqual(out["param"], "output.options.strength")

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

    def test_a_line_of_speech_shows_the_length_of_its_audio_and_a_playback_error(self):
        out = self.run_page("""
            document.getElementById("sp-model").value = "voice";
            document.getElementById("sp-input").value = "hello";
            await document.getElementById("sp-go").onclick();
            out.error_text = document.getElementById("sp-error").textContent;
            out.lines = document.getElementById("sp-lines").innerHTML;""")
        self.assertEqual(out["error_text"], "")
        self.assertIn("1.7 s", out["lines"])
        self.assertIn("onerror=", out["lines"])

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
