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
    {"model": "gemma-4-26b-a4b", "engine": "nagaya:84828966646e6ce1", "status": "ready",
     "capabilities": ["generate", "thinking"]},
    # As on b70host, identity gives an LFM3 model the hash of its engine, not the name of the engine in the profile.
    {"model": "lfm3-omni", "engine": "nagaya:f3f9adb00b28ace0", "status": "ready",
     "capabilities": ["generate", "control_vectors"],
     "media": {"inputs": ["image/png", "image/jpeg", "audio/wav"], "outputs": ["audio/wav"], "text_replies": True}},
    {"model": "lfm3", "engine": "nagaya:e974350551901f8c", "status": "ready", "capabilities": ["generate", "thinking", "control_vectors"],
     "limits": {"control_vector_contexts": 64}},
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
    {"name": "lfm3-omni", "kind": "mk-lfm3-bmg", "devices": [0]},
    {"name": "lfm3", "kind": "mk-lfm3-bmg", "devices": [0]},
]}

# A trace object of lfm3 from b70host (2026-11-08): 4 steps, the layers 8 and 23, top_k 3, the logit lens, and the
# vector LAB_VECTOR, which holds the layers 9 to 23. Thus its `along` at layer 8 is NaN.
TRACE_BYTES = list(pathlib.Path(__file__).with_name("fixtures").joinpath("lab_trace.safetensors").read_bytes())
LAB_VECTOR = "blake3:58c25c11d174204c4e0a0f1f37574c25136e08828fa6167a81a9ad4ccf7760b3"
TRACE_ID = "blake3:a44f076e76b36e55d9e54733d4167b472e14d1ef0515952df120d842be072259"

# A stub of the DOM that holds each element by id, and a fetch that answers as the nagaya does.
STUB = r"""
var calls = [], confirms = [];
function el(id) { return {id, value: "", textContent: "", innerHTML: "", dataset: {}, style: {}, disabled: false,
  hidden: false, checked: false, classList: {toggle() {}}, addEventListener() {}, querySelector() { return null; },
  appendChild() {}}; }
var els = {};
var listeners = [];
var document = {getElementById: (id) => els[id] || (els[id] = el(id)), querySelectorAll: () => [],
  querySelector: () => null, addEventListener(name, f) { if (name === "click") listeners.push(f); },
  createElement: () => el("script"), head: {appendChild() {}}};
var window = {}, localStorage = {getItem: () => null, setItem() {}}, performance = {now: () => 0};
function setTimeout(f) { f(); } function setInterval() {} function alert() {}
function set(id, v) { document.getElementById(id).value = v; }
function confirm(text) { confirms.push(text); return true; }
// A stream of server-sent events, read in one piece. QuickJS has no TextDecoder, so the reader gives text.
class TextDecoder { decode(v) { return typeof v === "string" ? v : v ? String.fromCharCode(...v) : ""; } }
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
  // The cluster route names the served models of each engine of this node (spec 2.3).
  if (path === "/nagaya/v1/cluster") return reply(200, {nodes: [
    {name: "gfx1151host", self: false, engines: [{name: "lfm3", state: "ready", models: ["elsewhere"]}]},
    {name: "b70host", self: true, engines: [{name: "lfm3-omni", state: "ready", models: ["lfm3-omni"]},
                                               {name: "lfm3", state: "ready", models: ["lfm3"]}]}]});
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
  if (path === "/council/v2/objects/blake3:" + "e".repeat(64)) return reply(404, {error: {type: "not_found",
    message: "no object has this id"}});
  if (path === "/nagaya/v1/lab/project") {
    const b = JSON.parse(o.body);
    return reply(200, {model: b.model, through: b.through, layers: [
      {layer: 12, up: [{id: 1, token: " sorry", score: 0.4}], down: [{id: 2, token: " Sure", score: -0.3}],
       end: [{id: 124900, token: "<|im_end|>", score: 0.05, rank: 13901}]},
      {layer: 23, up: [{id: 3, token: "I", score: 2.7}], down: [{id: 4, token: "<b>", score: -1.7}],
       end: [{id: 124900, token: "<|im_end|>", score: -0.16, rank: 82169}]}]});
  }
  if (path === "/nagaya/v1/lab/trace") {
    const b = JSON.parse(o.body);
    return reply(200, {model: b.model, content: b.context.control ? "Hi there! <b>" : "Hi there!", finish: "max_tokens",
                       usage: {prompt: 17, completion: 4}, trace: TRACE_ID});
  }
  if (path === "/council/v2/objects/" + TRACE_ID) {
    return Promise.resolve({ok: true, status: 200, arrayBuffer: () => Promise.resolve(new Uint8Array(TRACE_BYTES).buffer)});
  }
  if (path === "/nagaya/v1/detokenize") {
    const b = JSON.parse(o.body), id = b.tokens[0];
    return reply(200, {model: b.model, text: {41619: "Hi", 912: " there", 9: "!", 53841: "<b>"}[id] ?? `t${id}`});
  }
  if (path === "/council/v2/control-vectors") {
    const b = JSON.parse(o.body);
    return reply(200, {model: b.model, vector: "blake3:" + "c".repeat(64), layers: 24, width: 2048,
                       contexts: {positive: b.positive.length, negative: b.negative.length}, usage: {prompt: 40}});
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
    const b = JSON.parse(o.body);
    // A spoken reply of Omni holds its text and one WAV; a text reply holds no output.
    const outputs = b.model === "lfm3-omni" && !b.output ? []
      : [{index: 0, id: "blake3:" + "a".repeat(64), mime: "audio/wav"}];
    return reply(200, {content: b.model === "lfm3-omni" ? "A <b>cat</b>." : "", finish: "end_turn", outputs,
                       usage: {prompt: 90, completion: 5, media: {written: {objects: 1, bytes: 80684, audio_ms: 1680}}}});
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
        context.eval(f"var IDENTITY = {json.dumps(IDENTITY)}, PROFILE = {json.dumps(PROFILE)}, "
                     f"TRACE_BYTES = {json.dumps(TRACE_BYTES)}, TRACE_ID = {json.dumps(TRACE_ID)}, "
                     f"LAB_VECTOR = {json.dumps(LAB_VECTOR)};" + STUB + script +
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

    def test_each_context_names_thinking_and_only_a_model_that_thinks_gets_true(self):
        out = self.run_page(self.FORM + """
            set("img-model", "pony-0"); drawImageForm();
            await document.getElementById("img-go").onclick();
            out.image = body().context.thinking;
            set("sp-model", "voice"); set("sp-input", "hello"); await document.getElementById("sp-go").onclick();
            out.speech = body().context.thinking;
            document.getElementById("chat-thinking").checked = true;
            for (const model of ["gemma-4-26b-a4b", "lfm3-omni"]) {
              set("chat-model", model); set("chat-input", "hi");
              await document.getElementById("chat-go").onclick();
              out[model] = body().context.thinking;
            }""")
        self.assertEqual((out["image"], out["speech"]), (False, False))
        self.assertEqual((out["gemma-4-26b-a4b"], out["lfm3-omni"]), (True, False))

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

    VECTOR = "blake3:" + "c" * 64

    def test_a_vector_request_sends_one_plain_context_for_each_line_and_keeps_the_vector(self):
        out = self.run_page("""
            set("cv-model", "lfm3"); set("cv-toward", "the sea\\n\\n  waves  "); set("cv-away", "the desert");
            set("cv-name", "sea");
            await document.getElementById("cv-make").onclick();
            const made = calls.filter((c) => c[1] === "/council/v2/control-vectors");
            out.body = JSON.parse(made[0][2]);
            out.keep = JSON.parse(calls.filter((c) => c[1] === "/council/v2/keeps/nagaya-web").pop()[2]).objects;
            out.saved = vectors.map((v) => [v.name, v.id, v.model]);""")
        plain = lambda text: {"thinking": False, "messages": [{"role": "user", "content": text}]}
        self.assertEqual(out["body"], {"model": "lfm3", "positive": [plain("the sea"), plain("waves")],
                                       "negative": [plain("the desert")]})
        self.assertIn(self.VECTOR, out["keep"])
        self.assertEqual(out["saved"], [["sea", self.VECTOR, "lfm3"]])

    def test_a_chat_turn_sends_the_control_item_only_with_a_vector_of_its_model(self):
        out = self.run_page(f"""
            vectors = [{{name: "sea", id: "{self.VECTOR}", model: "lfm3", host}}];
            const body = () => JSON.parse(calls.filter((c) => c[1] === "/council/v2/generations").pop()[2]);
            set("chat-model", "lfm3");
            set("chat-vector", "{self.VECTOR}"); set("chat-op", "ablate"); set("chat-strength", "0.5");
            set("chat-input", "hi"); await document.getElementById("chat-go").onclick();
            out.steered = body().context.control;
            set("chat-vector", ""); set("chat-input", "hi"); await document.getElementById("chat-go").onclick();
            out.plain = body().context.control === undefined;
            set("chat-model", "lfm3-omni"); set("chat-vector", "{self.VECTOR}");
            set("chat-input", "hi"); await document.getElementById("chat-go").onclick();
            out.other = body().context.control === undefined;""")
        self.assertEqual(out["steered"], [{"vector": self.VECTOR, "op": "ablate", "strength": 0.5}])
        self.assertTrue(out["plain"])
        self.assertTrue(out["other"])

    def test_a_try_runs_the_prompt_plain_and_then_with_the_item(self):
        out = self.run_page(f"""
            vectors = [{{name: "sea", id: "{self.VECTOR}", model: "lfm3", host}}];
            set("cv-model", "lfm3"); set("cv-vector", "{self.VECTOR}"); set("cv-op", "add"); set("cv-strength", "4");
            set("cv-prompt", "Tell me about a place.");
            await document.getElementById("cv-try").onclick();
            out.bodies = calls.filter((c) => c[1] === "/council/v2/generations").map((c) => JSON.parse(c[2]).context);""")
        self.assertEqual(len(out["bodies"]), 2)
        self.assertNotIn("control", out["bodies"][0])
        self.assertEqual(out["bodies"][1]["control"], [{"vector": self.VECTOR, "op": "add", "strength": 4}])
        self.assertEqual(out["bodies"][0]["messages"], out["bodies"][1]["messages"])

    def test_a_vector_request_refuses_an_empty_side_and_a_side_past_the_limit(self):
        out = self.run_page("""
            set("cv-model", "lfm3"); set("cv-toward", ""); set("cv-away", "x");
            await document.getElementById("cv-make").onclick();
            out.empty = document.getElementById("cv-error").textContent;
            set("cv-toward", Array.from({length: 65}, (_, i) => `p${i}`).join("\\n"));
            await document.getElementById("cv-make").onclick();
            out.many = document.getElementById("cv-error").textContent;
            out.sent = calls.filter((c) => c[1] === "/council/v2/control-vectors").length;""")
        self.assertIn("toward", out["empty"])
        self.assertIn("64", out["many"])
        self.assertEqual(out["sent"], 0)


    def test_a_vector_added_by_id_is_saved_for_the_model_and_kept_only_when_the_nagaya_holds_it(self):
        out = self.run_page(f"""
            set("cv-model", "lfm3"); set("cv-add-id", "  {self.VECTOR}  "); set("cv-add-name", "late layers");
            await document.getElementById("cv-add").onclick();
            out.saved = vectors.map((v) => [v.name, v.id, v.model]);
            out.keep = JSON.parse(calls.filter((c) => c[1] === "/council/v2/keeps/nagaya-web").pop()[2]).objects;
            set("cv-add-id", "blake3:" + "e".repeat(64)); await document.getElementById("cv-add").onclick();
            out.missing = document.getElementById("cv-error").textContent;
            set("cv-add-id", "not an id"); await document.getElementById("cv-add").onclick();
            out.bad = document.getElementById("cv-error").textContent;
            out.count = vectors.length;""")
        self.assertEqual(out["saved"], [["late layers", self.VECTOR, "lfm3"]])
        self.assertIn(self.VECTOR, out["keep"])
        self.assertIn("does not hold", out["missing"])
        self.assertIn("blake3:", out["bad"])
        self.assertEqual(out["count"], 1)


    def test_the_lab_projects_a_saved_vector_and_shows_each_layer_with_its_end_rank_escaped(self):
        out = self.run_page(f"""
            vectors = [{{name: "refusal", id: "{self.VECTOR}", model: "lfm3", host}}];
            set("lab-model", "lfm3"); fillVectors(); set("lab-vector", "{self.VECTOR}"); set("lab-lens-k", "8");
            await document.getElementById("lab-project").onclick();
            out.body = JSON.parse(calls.filter((c) => c[1] === "/nagaya/v1/lab/project").pop()[2]);
            out.html = document.getElementById("lab-lens").innerHTML;""")
        self.assertEqual(out["body"], {"model": "lfm3", "vector": self.VECTOR, "through": "unembed", "top_k": 8})
        self.assertIn("82169", out["html"])
        self.assertIn(" sorry", out["html"])
        self.assertIn("&lt;b&gt;", out["html"])
        self.assertNotIn("<b>", out["html"])

    def test_the_lab_projects_through_the_jlens_when_the_person_chooses_it(self):
        out = self.run_page(f"""
            vectors = [{{name: "refusal", id: "{self.VECTOR}", model: "lfm3", host}}];
            set("lab-model", "lfm3"); fillVectors(); set("lab-vector", "{self.VECTOR}"); set("lab-lens-k", "8");
            set("lab-through", "jlens");
            await document.getElementById("lab-project").onclick();
            out.body = JSON.parse(calls.filter((c) => c[1] === "/nagaya/v1/lab/project").pop()[2]);
            out.html = document.getElementById("lab-lens").innerHTML;""")
        self.assertEqual(out["body"]["through"], "jlens")
        self.assertIn("J-Lens", out["html"])

    def test_a_model_that_speaks_beside_its_text_is_a_chat_model_and_not_a_speech_model(self):
        out = self.run_page("""
            out.chat = document.getElementById("chat-model").innerHTML;
            out.speech = document.getElementById("sp-model").innerHTML;""")
        self.assertIn('value="lfm3-omni"', out["chat"])
        self.assertNotIn("lfm3-omni", out["speech"])
        self.assertIn('value="voice"', out["speech"])

    def test_the_omni_tab_lists_the_models_that_read_media_and_reply_with_text(self):
        out = self.run_page('out.models = document.getElementById("omni-model").innerHTML;')
        self.assertEqual(out["models"], '<option value="lfm3-omni">lfm3-omni</option>')

    # The values of the Omni form, as a person sets them, with one image attached.
    OMNI = """
        set("omni-model", "lfm3-omni"); set("omni-system", "Respond with interleaved text and audio.");
        set("omni-max", "64"); set("omni-temp", "0.3"); set("omni-output", "interleaved");
        omniFiles = [{id: "blake3:" + "f".repeat(64), mime: "image/png", name: "cat.png", host}];
        set("omni-input", "What is this?");
        const sent = () => calls.filter((c) => c[1] === "/council/v2/generations").map((c) => JSON.parse(c[2]));"""

    def test_an_omni_turn_sends_the_text_and_the_objects_and_plays_the_spoken_reply(self):
        out = self.run_page(self.OMNI + """
            await document.getElementById("omni-go").onclick();
            out.body = sent()[0]; out.log = document.getElementById("omni-log").innerHTML;
            out.err = document.getElementById("omni-error").textContent;
            out.keep = JSON.parse(calls.filter((c) => c[1] === "/council/v2/keeps/nagaya-web").pop()[2]).objects;
            set("omni-input", "And now?"); set("omni-output", "");
            await document.getElementById("omni-go").onclick();
            out.second = sent()[1];""")
        self.assertEqual(out["err"], "")
        body = out["body"]
        self.assertEqual(body["model"], "lfm3-omni")
        self.assertEqual(body["output"], {"mime": "audio/wav", "options": {"mode": "interleaved"}})
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["sample"], {"temperature": 0.3})
        self.assertFalse(body["context"]["thinking"])
        system, user = body["context"]["messages"]
        self.assertEqual(system, {"role": "system", "content": "Respond with interleaved text and audio.", "snap": True})
        self.assertEqual(user["content"], [{"type": "text", "text": "What is this?"},
                                           {"type": "object", "id": "blake3:" + "f" * 64, "mime": "image/png"}])
        # The reply's text shows escaped, and its audio plays from the object.
        self.assertIn("A &lt;b&gt;cat&lt;/b&gt;.", out["log"])
        self.assertNotIn("<b>cat", out["log"])
        self.assertIn("<audio controls", out["log"])
        self.assertIn('/council/v2/objects/blake3:' + "a" * 64 + '"', out["log"])
        self.assertIn("blake3:" + "a" * 64, out["keep"])
        self.assertIn("blake3:" + "f" * 64, out["keep"])
        # The next turn sends the spoken reply back as its text and its audio (spec 11.11, "Read it back"), and no
        # output asks for text only.
        second = out["second"]
        self.assertNotIn("output", second)
        reply = second["context"]["messages"][2]
        self.assertEqual(reply, {"role": "assistant", "content": [{"type": "text", "text": "A <b>cat</b>."},
                                 {"type": "object", "id": "blake3:" + "a" * 64, "mime": "audio/wav"}]})
        self.assertEqual(second["context"]["messages"][3]["content"], [{"type": "text", "text": "And now?"}])

    def test_the_omni_tab_refuses_a_file_of_a_type_that_the_model_does_not_read(self):
        out = self.run_page("""
            set("omni-model", "lfm3-omni");
            await document.getElementById("omni-file").onchange({target: {files: [{name: "x.wav", type: "audio/x-wav"}],
                                                                           value: ""}});
            out.err = document.getElementById("omni-error").textContent;
            out.puts = calls.filter((c) => c[0] === "PUT").length;""")
        self.assertIn("x.wav is audio/x-wav", out["err"])
        self.assertIn("audio/wav", out["err"])
        self.assertEqual(out["puts"], 0)

    def test_a_reply_mode_fills_in_the_system_prompt_of_the_model_card_and_keeps_one_the_person_wrote(self):
        out = self.run_page("""
            const pick = (mode) => { set("omni-output", mode); document.getElementById("omni-output").onchange();
                                     return document.getElementById("omni-system").value; };
            out.speech = pick("speech"); out.mixed = pick("interleaved"); out.text = pick("");
            set("omni-system", "Perform TTS in British English using a male voice.");
            out.kept = pick("speech");""")
        self.assertEqual(out["speech"], "Perform TTS in American English using a female voice.")
        self.assertEqual(out["mixed"], "Respond with interleaved text and audio.")
        self.assertEqual(out["text"], "")
        self.assertEqual(out["kept"], "Perform TTS in British English using a male voice.")

    def test_speech_mode_speaks_a_long_text_in_chunks_of_whole_sentences(self):
        long = " ".join(["This is sentence number %d of a long story that goes on." % i for i in range(1, 9)])
        out = self.run_page(f"""
            set("omni-model", "lfm3-omni"); set("omni-output", "speech");
            document.getElementById("omni-output").onchange();
            set("omni-input", {json.dumps(long)});
            await document.getElementById("omni-go").onclick();
            out.bodies = calls.filter((c) => c[1] === "/council/v2/generations").map((c) => JSON.parse(c[2]));
            out.log = document.getElementById("omni-log").innerHTML;
            out.chunks = speechChunks("one two. " + "w ".repeat(90).trim() + ". three.");""")
        bodies = out["bodies"]
        # 8 sentences of 12 words: 3 a chunk, so 3 chunks; each a TTS request of one user message.
        self.assertEqual(len(bodies), 3)
        for b in bodies:
            system, user = b["context"]["messages"]
            self.assertEqual(system["content"], "Perform TTS in American English using a female voice.")
            self.assertLessEqual(len(user["content"].split()), 40)
            self.assertEqual(b["output"], {"mime": "audio/wav", "options": {"mode": "speech"}})
        self.assertEqual(" ".join(b["context"]["messages"][1]["content"] for b in bodies), long)
        self.assertEqual(out["log"].count("<audio controls"), 3)
        # A sentence longer than a chunk splits at its words.
        self.assertEqual([len(c.split()) for c in out["chunks"]], [2, 40, 40, 10, 1])

    def test_an_omni_turn_takes_a_vector_of_its_model_and_not_one_of_another_model(self):
        omni_vector = "blake3:" + "9" * 64
        out = self.run_page(self.OMNI + f"""
            vectors = [{{name: "omni refusal", id: "{omni_vector}", model: "lfm3-omni", host}},
                       {{name: "text refusal", id: "{LAB_VECTOR}", model: "lfm3", host}}];
            fillVectors();
            out.options = document.getElementById("omni-vector").innerHTML;
            set("omni-vector", "{omni_vector}"); set("omni-op", "ablate"); set("omni-strength", "0.5");
            await document.getElementById("omni-go").onclick();
            out.body = sent()[0];""")
        self.assertIn("omni refusal", out["options"])
        self.assertNotIn("text refusal", out["options"])
        self.assertEqual(out["body"]["context"]["control"], [{"vector": omni_vector, "op": "ablate", "strength": 0.5}])

    def test_the_lab_lists_only_the_local_models_of_the_lfm3_engine_kind(self):
        out = self.run_page('out.models = document.getElementById("lab-model").innerHTML;')
        self.assertEqual(out["models"], '<option value="lfm3-omni">lfm3-omni</option><option value="lfm3">lfm3</option>')

    def test_a_model_with_the_hash_of_its_engine_gets_the_engine_that_the_cluster_route_names(self):
        out = self.run_page('out.engine = engineOf(models.find((m) => m.model === "lfm3"));')
        self.assertEqual(out["engine"], "lfm3")

    def test_a_trace_object_parses_into_its_tensors_with_their_shapes(self):
        out = self.run_page("""
            const t = await traceObject(TRACE_ID);
            const n = (x) => Number.isNaN(x) ? "nan" : +x.toFixed(4);
            out.meta = t.meta; out.layers = Array.from(t.layers.data); out.tokens = Array.from(t.tokens.data);
            out.along_shape = t.along.shape; out.along = Array.from(t.along.data).map(n);
            out.entropy = Array.from(t.entropy.data).map(n); out.lens_shape = t.lens_ids.shape;
            out.lens_last = t.lens_ids.data[t.lens_ids.data.length - 3];""")
        self.assertEqual(out["meta"]["vectors"], LAB_VECTOR)
        self.assertEqual(out["layers"], [8, 23])
        self.assertEqual(out["tokens"], [41619, 912, 9, 53841])
        self.assertEqual(out["along_shape"], [4, 2, 1])
        self.assertEqual(out["along"][0], "nan")
        self.assertEqual(out["along"][1], 0.075)
        self.assertEqual(out["entropy"], [1.087, 1.3431, 0.483, 1.6207])
        self.assertEqual(out["lens_shape"], [4, 2, 3])
        self.assertEqual(out["lens_last"], 53841)

    # The values of the lab form, as a person sets them.
    LAB = f"""
        vectors = [{{name: "refusal", id: "{LAB_VECTOR}", model: "lfm3", host}}];
        set("lab-model", "lfm3"); fillVectors();
        set("lab-vector", LAB_VECTOR); set("lab-op", "ablate"); set("lab-strength", "0.5");
        set("lab-prompt", "Say hi."); set("lab-max", "4"); set("lab-temp", "0.7"); set("lab-k", "3");
        set("lab-use-lens", "logit");
        const traced = () => calls.filter((c) => c[1] === "/nagaya/v1/lab/trace").map((c) => JSON.parse(c[2]));"""

    def test_a_trace_runs_the_prompt_plain_and_then_steered_with_one_seed_and_the_vector_in_along(self):
        out = self.run_page(self.LAB + """
            await document.getElementById("lab-trace").onclick();
            out.bodies = traced(); out.err = document.getElementById("lab-error").textContent;""")
        self.assertEqual(out["err"], "")
        plain, steered = out["bodies"]
        self.assertNotIn("control", plain["context"])
        self.assertEqual(steered["context"]["control"], [{"vector": LAB_VECTOR, "op": "ablate", "strength": 0.5}])
        self.assertEqual(plain["context"]["messages"], steered["context"]["messages"])
        for body in out["bodies"]:
            self.assertEqual(body["trace"], {"vectors": [LAB_VECTOR], "top_k": 3, "lens": "logit"})
            self.assertEqual(body["max_tokens"], 4)
            self.assertNotIn("stream", body)
        self.assertEqual(plain["sample"]["temperature"], 0.7)
        self.assertEqual(plain["sample"], steered["sample"])
        # A seed past 2^53 does not survive a JavaScript number, so the page picks one below it.
        self.assertIsInstance(plain["sample"]["seed"], int)
        self.assertLess(plain["sample"]["seed"], 2 ** 53)

    def test_a_trace_with_no_vector_runs_plain_only_and_sends_no_vectors_and_no_lens_when_it_is_off(self):
        out = self.run_page(self.LAB + """
            set("lab-vector", ""); set("lab-use-lens", "");
            await document.getElementById("lab-trace").onclick();
            out.bodies = traced();""")
        self.assertEqual(len(out["bodies"]), 1)
        self.assertNotIn("control", out["bodies"][0]["context"])
        self.assertEqual(out["bodies"][0]["trace"], {"top_k": 3})

    def test_a_trace_with_the_jlens_sends_it_and_the_page_keeps_each_trace_object(self):
        out = self.run_page(self.LAB + """
            set("lab-use-lens", "jlens");
            await document.getElementById("lab-trace").onclick();
            out.bodies = traced();
            out.keep = JSON.parse(calls.filter((c) => c[1] === "/council/v2/keeps/nagaya-web").pop()[2]).objects;
            out.saved = traces.map((x) => [x.id, x.model]);""")
        self.assertEqual([b["trace"]["lens"] for b in out["bodies"]], ["jlens", "jlens"])
        self.assertIn(TRACE_ID, out["keep"])
        self.assertIn(LAB_VECTOR, out["keep"])
        self.assertEqual(out["saved"], [[TRACE_ID, "lfm3"]])

    def test_the_lab_draws_both_runs_and_the_chosen_step_with_the_token_text_escaped(self):
        out = self.run_page(self.LAB + """
            await document.getElementById("lab-trace").onclick();
            out.charts = document.getElementById("lab-charts").innerHTML;
            out.replies = document.getElementById("lab-replies").innerHTML;
            await labStep(3);
            out.step = document.getElementById("lab-step").innerHTML;
            out.asked = calls.filter((c) => c[1] === "/nagaya/v1/detokenize").map((c) => JSON.parse(c[2]).tokens[0]);""")
        for name in ("entropy", "plain", "steered", "<svg", "<polyline"):
            self.assertIn(name, out["charts"])
        # Both runs, each with its 4 steps, and each token a button for its step.
        self.assertEqual(out["replies"].count("data-lab-step="), 8)
        self.assertIn("&lt;b&gt;", out["replies"])
        self.assertNotIn("<b>", out["replies"] + out["step"])
        self.assertIn("&lt;b&gt;", out["step"])
        self.assertIn("23", out["step"])
        # Layer 8 has no direction in the vector, so its along shows as a dash and not as NaN.
        self.assertNotIn("NaN", out["step"] + out["charts"])
        self.assertEqual(len(out["asked"]), len(set(out["asked"])))

    def test_forget_removes_the_vector_of_its_row_when_another_nagaya_has_vectors_first(self):
        out = self.run_page(f"""
            vectors = [{{name: "far", id: "blake3:" + "f".repeat(64), model: "lfm3", host: "gfx1151host"}},
                       {{name: "near", id: "{self.VECTOR}", model: "lfm3", host}}];
            fillVectors();
            const row = document.getElementById("cv-list").innerHTML.match(/data-drop-vector="(\\d+)"/)[1];
            for (const f of listeners) f({{target: {{dataset: {{dropVector: row}}}}}});
            out.left = vectors.map((v) => v.name);""")
        self.assertEqual(out["left"], ["far"])


if __name__ == "__main__":
    unittest.main()
