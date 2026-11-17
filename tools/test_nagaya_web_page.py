"""Tests of the scripts of web/index.html in QuickJS, with a stub page and a fake fetch.

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

WEB = pathlib.Path(__file__).with_name("web")


def page_script():
    """The scripts that the page names, in the order of the page, as one script. They share one scope, as in a browser."""
    names = re.findall(r'<script src="/app/web/([^"]+)"></script>', WEB.joinpath("index.html").read_text())
    return "\n".join(WEB.joinpath(name).read_text() for name in names)


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
  appendChild() {}, insertAdjacentHTML(where, html) { this.innerHTML += html; }}; }
var els = {};
var listeners = [];
var document = {getElementById: (id) => els[id] || (els[id] = el(id)), querySelectorAll: () => [],
  querySelector: () => null, addEventListener(name, f) { if (name === "click") listeners.push(f); },
  createElement: () => el("script"), head: {appendChild() {}}};
var window = {addEventListener() {}}, location = {hash: ""}, history = {replaceState() {}}, localStorage = {getItem: () => null, setItem() {}}, performance = {now: () => 0};
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


class Markup(unittest.TestCase):
    """Checks of the page text that need no QuickJS: the stub page of `Page` makes each element that a script asks for,
    so it does not see an id that the page lacks."""

    def test_each_id_that_a_script_reads_is_in_the_page(self):
        page = WEB.joinpath("index.html").read_text()
        ids = set(re.findall(r'\bid="([^"]+)"', page))
        # `$("name")` with a plain name. An id that a script builds, such as `cur-bar-${i}`, is in no markup.
        asked = set(re.findall(r'\$\("([a-z0-9-]+)"\)', page_script()))
        self.assertGreater(len(asked), 100)
        self.assertEqual(sorted(asked - ids), [])

    def test_each_model_list_has_the_element_of_its_state(self):
        page = WEB.joinpath("index.html").read_text()
        lists = re.findall(r'modelSelect\("([a-z-]+)"', page_script())
        self.assertGreaterEqual(len(lists), 7)
        ids = set(re.findall(r'\bid="([^"]+)"', page))
        self.assertEqual([name for name in lists if f"{name}-state" not in ids], [])

    def test_the_choice_of_the_nagaya_is_in_the_rail_and_the_bar_holds_the_loaded_models_only(self):
        page = WEB.joinpath("index.html").read_text()
        rail = page[page.index('<aside id="rail">'):page.index("</aside>")]
        bar = page[page.index("<header>"):page.index("</header>")]
        self.assertIn('id="host"', rail)
        self.assertNotIn('id="host"', bar)
        self.assertIn('id="chips"', bar)

    def test_a_chat_thinks_unless_the_person_turns_it_off(self):
        # LFM3 thinks by default, and its card says that a reply with thinking off "can be inconsistent". On
        # 2026-11-11, a second turn with thinking off wrote a second close-think tag in 6 to 23 of 24 replies.
        page = WEB.joinpath("index.html").read_text()
        self.assertIn('<input id="chat-thinking" type="checkbox" checked>', page)

    def test_the_sampler_of_a_chat_starts_from_the_model_card_of_lfm3(self):
        # The card gives temperature 0.1 and top k 50 for general work, and 1.0, top p 0.95 and top k 20 for math.
        # The page starts warmer than 0.1, because a chat at 0.1 repeats itself, and it keeps the top k of 50.
        page = WEB.joinpath("index.html").read_text()
        self.assertIn('<input id="chat-temp" type="number" min="0" step="0.1" value="0.3">', page)
        self.assertIn('<input id="chat-topk" type="number" min="1" value="50">', page)
        self.assertIn('<input id="chat-topp" type="number" min="0.01" max="1" step="0.01" placeholder="off">', page)

    def test_the_instructions_of_a_tab_are_behind_its_help_button_and_long_hints_are_tooltips(self):
        # The instructions of a tab are behind its help button, and a long hint is a tooltip, so the page stays short.
        page = WEB.joinpath("index.html").read_text()
        self.assertNotIn('<p class="note">', page)
        sections = re.findall(r'<section id="(\w+)">(.*?)</section>', page, re.S)
        self.assertEqual(len(sections), 8)
        for name, body in sections:
            head = body[:body.index("</div>\n", body.index('<div class="head">'))]
            self.assertEqual(head.count('<details class="help">'), 1, name)
            self.assertEqual(body.count('<details class="help">'), 1, name)
        # A hint that stays beside its label is short. A longer one is the hover text of a mark.
        for hint in re.findall(r'<span class="note">— (.*?)</span>', page, re.S):
            self.assertLessEqual(len(hint), 20, hint)
        self.assertGreaterEqual(len(re.findall(r'<span class="tip" tabindex="0" title="[^"]+">ⓘ</span>', page)), 10)

    def test_the_tray_of_the_chat_is_beside_its_log_and_the_voice_starts_british(self):
        page = WEB.joinpath("index.html").read_text()
        self.assertIn('<div class="dock"><div id="chat-log"></div><div id="chat-tray" class="tray"></div></div>', page)
        voice = re.search(r'<textarea id="sp-voice" rows="4">(.*?)</textarea>', page, re.S).group(1)
        self.assertIn("British", voice)
        self.assertIn("woman", voice)

    def test_each_prompt_box_of_the_steering_tab_takes_dataset_files_and_hides_its_choices_at_first(self):
        page = WEB.joinpath("index.html").read_text()
        for side in ("toward", "away"):
            self.assertIn(f'<input id="cv-{side}-file" type="file" '
                          'accept=".txt,.parquet,.jsonl,.ndjson,.json,.csv,.tsv,text/plain"', page)
            self.assertIn(f'<div id="cv-{side}-pick" class="fields" hidden>', page)
            for what in ("column", "count", "order", "seed"):
                self.assertIn(f'id="cv-{side}-{what}"', page)
        steer = page[page.index('<section id="steer">'):]
        steer = steer[:steer.index("</section>")]
        help_text = steer[steer.index('<div class="help-text">'):steer.index("</div></details>")]
        for word in ("Parquet", "Snappy", "ZSTD", "JSON Lines", "CSV"):
            self.assertIn(word, help_text)

    def test_each_tab_of_the_rail_has_its_section(self):
        page = WEB.joinpath("index.html").read_text()
        tabs = re.findall(r'<button data-tab="(\w+)">', page)
        self.assertEqual(sorted(tabs), sorted(re.findall(r'<section id="(\w+)">', page)))
        self.assertEqual(len(tabs), len(set(tabs)))


@unittest.skipIf(quickjs is None, "the package quickjs is absent")
class Page(unittest.TestCase):
    def run_page(self, test, before=""):
        """Run the page script and then `test`, an async body that fills `out`. Give `out`. `before` runs after the
        stub and before the page script."""
        script = page_script()
        context = quickjs.Context()
        context.eval(f"var IDENTITY = {json.dumps(IDENTITY)}, PROFILE = {json.dumps(PROFILE)}, "
                     f"TRACE_BYTES = {json.dumps(TRACE_BYTES)}, TRACE_ID = {json.dumps(TRACE_ID)}, "
                     f"LAB_VECTOR = {json.dumps(LAB_VECTOR)};" + STUB + before + script +
                     "var out = {}; (async () => { await refresh();" + test +
                     "})().catch((e) => { out.error = String(e); });")
        while context.execute_pending_job():
            pass
        out = json.loads(context.eval("JSON.stringify(out)"))
        self.assertNotIn("error", out)
        return out

    def test_the_start_of_the_page_puts_the_keep_of_what_this_browser_holds(self):
        # Before the split of the page, the keep of the start read `vectors` before its declaration, and it never ran.
        vector = "blake3:" + "7" * 64
        out = self.run_page("""
            out.keeps = calls.filter((c) => c[0] === "PUT" && c[1] === "/council/v2/keeps/nagaya-web").map((c) => c[2]);
            out.failed = document.getElementById("keep-error").textContent;""",
                            before=f"localStorage.getItem = (key) => key === 'nagaya-vectors' ? "
                                   f"JSON.stringify([{{id: '{vector}', model: 'lfm3', name: 'v'}}]) : null;")
        self.assertEqual(out["failed"], "")
        self.assertEqual(json.loads(out["keeps"][0]), {"objects": [vector], "ttl_s": 30 * 86400})

    def test_the_bar_shows_the_models_that_are_loaded_and_one_that_failed_and_no_unloaded_one(self):
        out = self.run_page("""
            out.before = document.getElementById("chips").innerHTML;
            models[0].status = "failed"; models[1].status = "loading"; models[2].status = "draining";
            IDENTITY.models = models;
            await refresh();
            out.after = document.getElementById("chips").innerHTML;""")
        for name in ("gemma-4-26b-a4b", "lfm3-omni", "lfm3", "embeddinggemma-2", "voice", "pony-0"):
            self.assertIn(f"</i>{name}</span>", out["before"])
        # sdxl-0 and sdxl-1 are unloaded: the Engines tab lists them, and the bar does not.
        self.assertNotIn("sdxl", out["before"])
        self.assertIn("2 not loaded", out["before"])
        self.assertIn('<span class="chip failed" title="failed"><i></i>gemma-4-26b-a4b</span>', out["after"])
        self.assertIn('<span class="chip loading" title="loading"><i></i>lfm3-omni</span>', out["after"])
        self.assertIn('<span class="chip draining" title="draining"><i></i>lfm3</span>', out["after"])

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

    # The end of a chat stream as the nagaya of b70host sent it on 2026-11-11, with a system message at index 0.
    CHAT_DONE = {"model": "lfm3", "content": "Hi there!", "reasoning": "", "finish": "end_turn",
                 "usage": {"prompt": 23, "kept": 18, "fed": 5, "completion": 4, "reasoning": 0},
                 "stats": {"reasoning": None, "content": {"tokens": 4, "mean_logprob": -0.2362, "min_logprob": -0.6274,
                                                          "mean_entropy": 0.4974, "max_entropy": 0.9935}},
                 "snapshots": [{"id": "blake3:8336d95c" + "0" * 56, "message": 0, "tokens": 10, "tier": "host",
                                "bytes": 1097728},
                               {"id": "blake3:390ed0d6" + "1" * 56, "message": 1, "tokens": 18, "tier": "device",
                                "bytes": 1458176}]}

    def chat_turn(self, done):
        return self.run_page(f"const DONE = {json.dumps(done)};" + """
            const real = fetch;
            fetch = (path, o) => /generations$/.test(path) ? (calls.push(["POST", path, o.body]), sse([["done", DONE]]))
                                                           : real(path, o);
            set("chat-model", "lfm3"); set("chat-system", "Be brief."); set("chat-input", "hi <b>");
            await document.getElementById("chat-go").onclick();
            out.failed = document.getElementById("chat-error").textContent;
            out.sent = calls.filter((c) => /generations$/.test(c[1])).pop()[2].length;
            out.log = document.getElementById("chat-log").innerHTML;""")

    def test_a_chat_reply_shows_what_the_page_sent_and_what_the_nagaya_kept_and_ran(self):
        out = self.chat_turn(self.CHAT_DONE)
        self.assertEqual(out["failed"], "")
        user, reply = out["log"].split('<div class="msg assistant">')
        # The message of the person carries the content id of the snapshot at its end, as a link to the snapshot.
        self.assertIn('href="/at/b70host/council/v2/snapshots/blake3:390ed0d6' + "1" * 56 + '"', user)
        self.assertIn("390ed0d6", user)
        self.assertIn("18 tokens", user)
        self.assertIn("device", user)
        self.assertNotIn("8336d95c", user)
        # One status line says that the whole text went, and how few tokens ran.
        self.assertEqual(reply.count('class="stat"'), 1)
        self.assertEqual(user.count('class="stat"'), 1)
        self.assertIn(f"4 tokens · end_turn · sent {out['sent']} B (whole chat) · ran 5 of 23 prompt tokens", reply)
        # The snapshot of the system message has no message in the log, so the hover text of the line names it.
        self.assertIn("message 0: blake3:8336d95c", reply)
        self.assertIn("1.1 MB", reply)
        self.assertIn("kept 18", reply)
        # The hover text also has how sure the model was of its tokens (gogi "Generations", `stats`).
        self.assertIn("mean entropy 0.50 nats, at most 0.99", reply)
        self.assertIn("mean log probability -0.24, at least -0.63", reply)

    # Two chat turns against a nagaya that names a context record in `done` (gogi "Context records"). `RECORDS` gives
    # the `context` of each reply in turn; null is a nagaya from before the records. `refuse` makes the next request
    # that names a parent a 404, as after the nagaya dropped the record.
    TURNS = """
        const sent = [], asked = [];
        let refuse = false, turn = 0;
        const real = fetch;
        fetch = (path, o) => {
          if (!/generations$/.test(path)) return real(path, o);
          const body = JSON.parse(o.body);
          asked.push(body.context);
          if (body.context.parent && refuse) {
            refuse = false;
            return reply(404, {error: {type: "not_found", param: "context.parent", message: "nothing was read: no record"}});
          }
          sent.push(body.context);
          const context = RECORDS[turn++];
          return sse([["done", {content: "reply " + turn, reasoning: "", finish: "end_turn", snapshots: [],
                                usage: {prompt: 10 * turn, kept: 0, fed: 10 * turn, completion: 2}, ...(context ? {context} : {})}]]);
        };
        const say = async (text) => { set("chat-input", text); await document.getElementById("chat-go").onclick(); };
        set("chat-model", "lfm3"); set("chat-system", "Be brief.");"""
    R1, R2 = "blake3:" + "1" * 64, "blake3:" + "2" * 64

    def turns(self, records, test):
        out = self.run_page(f"const RECORDS = {json.dumps(records)};" + self.TURNS + test + """
            out.sent = sent; out.asked = asked;
            out.failed = document.getElementById("chat-error").textContent;
            out.log = document.getElementById("chat-log").innerHTML;""")
        self.assertEqual(out["failed"], "")
        return out

    def test_the_second_chat_turn_names_the_record_of_the_first_and_sends_only_its_new_message(self):
        out = self.turns([self.R1, self.R2], 'await say("one"); await say("two");')
        first, second = out["sent"]
        self.assertEqual(first, {"thinking": False, "messages": [{"role": "system", "content": "Be brief.", "snap": True},
                                                                 {"role": "user", "content": "one", "snap": True}]})
        self.assertEqual(second, {"parent": self.R1, "thinking": False, "control": [],
                                  "messages": [{"role": "user", "content": "two", "snap": True}]})
        # The reply shows that only the new message went, and the record of each reply as a link.
        self.assertEqual(out["log"].count("(new message only)"), 1)
        self.assertIn(f'href="/at/b70host/council/v2/contexts/{self.R2}"', out["log"])
        self.assertEqual(out["log"].count("(whole chat)"), 1)
        # Each of the 4 messages has one status line or none.
        self.assertLessEqual(out["log"].count('class="stat"'), 4)
        self.assertEqual(out["log"].count('<div class="msg assistant">'), 2)

    def test_a_record_that_the_nagaya_dropped_makes_the_page_send_the_whole_chat_once(self):
        out = self.turns([self.R1, self.R2], 'await say("one"); refuse = true; await say("two");')
        self.assertEqual([("parent" in c) for c in out["asked"]], [False, True, False])
        whole = out["sent"][1]
        self.assertEqual([m["role"] for m in whole["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(whole["messages"][2], {"role": "assistant", "content": "reply 1"})
        self.assertEqual(whole["messages"][3]["content"], "two")
        self.assertEqual(out["log"].count("(whole chat)"), 2)

    def test_a_nagaya_with_no_records_gets_the_whole_chat_each_turn(self):
        out = self.turns([None, None], 'await say("one"); await say("two");')
        self.assertEqual([("parent" in c) for c in out["sent"]], [False, False])
        self.assertEqual(len(out["sent"][1]["messages"]), 4)
        self.assertNotIn("/council/v2/contexts/", out["log"])

    def test_a_reply_with_no_record_ends_the_use_of_the_record_before_it(self):
        # The record of turn 1 does not hold turn 2, so turn 3 must not name it.
        out = self.turns([self.R1, None, self.R2], 'await say("one"); await say("two"); await say("three");')
        self.assertEqual([("parent" in c) for c in out["sent"]], [False, True, False])
        self.assertEqual(len(out["sent"][2]["messages"]), 6)

    def test_a_new_system_message_or_a_new_chat_sends_the_whole_chat_and_other_settings_stand_beside_the_parent(self):
        out = self.turns([self.R1, self.R2, self.R1, self.R2], """
            await say("one");
            document.getElementById("chat-thinking").checked = true;
            await say("two");
            set("chat-system", "Be long.");
            await say("three");
            await document.getElementById("chat-clear").onclick();
            await say("four");""")
        one, two, three, four = out["sent"]
        # lfm3 thinks, so the choice goes beside the parent, and the record serves both.
        self.assertEqual((two["parent"], two["thinking"]), (self.R1, True))
        self.assertNotIn("parent", three)
        self.assertEqual(three["messages"][0]["content"], "Be long.")
        self.assertEqual(len(three["messages"]), 6)
        self.assertNotIn("parent", four)
        self.assertEqual([m["content"] for m in four["messages"]], ["Be long.", "four"])

    def test_a_chat_turn_sends_the_sampler_fields_that_are_set_and_no_field_that_is_empty(self):
        out = self.run_page("""
            const sent = [];
            const real = fetch;
            fetch = (path, o) => /generations$/.test(path)
              ? (sent.push(JSON.parse(o.body).sample), sse([["done", {content: "ok", finish: "end_turn", snapshots: [],
                                                                     usage: {prompt: 1, fed: 1, completion: 1}}]]))
              : real(path, o);
            const say = async () => { set("chat-input", "hi"); await document.getElementById("chat-go").onclick(); };
            set("chat-model", "lfm3"); set("chat-temp", "0.3"); set("chat-topk", "50"); set("chat-topp", "");
            await say();
            set("chat-topk", ""); set("chat-topp", "0.95"); set("chat-temp", "1");
            await say();
            out.sent = sent;
            out.failed = document.getElementById("chat-error").textContent;""")
        self.assertEqual(out["failed"], "")
        self.assertEqual(out["sent"], [{"temperature": 0.3, "top_k": 50}, {"temperature": 1, "top_p": 0.95}])

    # A chat with one reply of two sentences, and a way to press a button of a message.
    SEND_TO = """
        chat.push({role: "user", content: "Say hello."});
        chat.push({role: "assistant", content: "Hello there. A red fox sits in the snow."});
        const press = async (data) => { for (const f of listeners) await f({target: {dataset: data}}); };
        const bodies = () => calls.filter((c) => /generations$/.test(c[1])).map((c) => JSON.parse(c[2]));"""

    def test_a_chat_message_has_a_button_for_speech_and_for_an_image_only_when_a_model_is_chosen(self):
        out = self.run_page(self.SEND_TO + """
            out.speak = document.getElementById("chat-speak").innerHTML;
            out.draw = document.getElementById("chat-draw").innerHTML;
            drawChat(); out.off = document.getElementById("chat-log").innerHTML;
            set("chat-speak", "voice"); set("chat-draw", "pony-0");
            drawChat(); out.on = document.getElementById("chat-log").innerHTML;""")
        # The speech list has the speech models and the models that speak beside their text. The first choice is off.
        self.assertTrue(out["speak"].startswith('<option value="">off</option>'))
        for name in ("voice", "lfm3-omni"):
            self.assertIn(f'<option value="{name}">{name}</option>', out["speak"])
        self.assertNotIn("gemma", out["speak"])
        for name in ("sdxl-0", "pony-0"):
            self.assertIn(f'<option value="{name}">{name}</option>', out["draw"])
        self.assertNotIn("data-chat-speak", out["off"])
        self.assertNotIn("data-chat-draw", out["off"])
        self.assertIn('data-chat-speak="1"', out["on"])
        self.assertIn('data-chat-draw="0"', out["on"])

    def test_a_message_goes_to_a_speech_model_in_chunks_and_its_clips_go_to_the_tray(self):
        out = self.run_page(self.SEND_TO + """
            set("chat-speak", "voice"); set("sp-voice", "A calm voice."); set("sp-lang", "english");
            await press({chatSpeak: "1"});
            out.bodies = bodies();
            out.tray = document.getElementById("chat-tray").innerHTML;
            out.clips = document.getElementById("chat-job-0-clips").innerHTML;
            out.status = document.getElementById("chat-job-0-status").textContent;
            out.keep = JSON.parse(calls.filter((c) => c[1] === "/council/v2/keeps/nagaya-web").pop()[2]).objects;""")
        (body,) = out["bodies"]
        self.assertEqual(body["model"], "voice")
        self.assertEqual(body["context"]["messages"], [{"role": "system", "content": "A calm voice.", "snap": True},
                                                       {"role": "user", "content": "Hello there. A red fox sits in the snow."}])
        self.assertEqual(body["output"], {"mime": "audio/wav", "options": {"language": "english"}})
        self.assertIn("message 2", out["tray"])
        self.assertIn("voice", out["tray"])
        self.assertIn('<audio controls src="/at/b70host/council/v2/objects/blake3:' + "a" * 64 + '"', out["clips"])
        self.assertIn("1 clip", out["status"])
        self.assertIn("blake3:" + "a" * 64, out["keep"])

    def test_each_chunk_of_speech_continues_from_the_clips_before_it_so_the_voice_stays(self):
        # A text of 3 sentences of 20 words each is 2 chunks of at most 40 words. The second request holds the first
        # chunk and its clip, as the Speech tab does with "Keep the lines before" (spec 11.8a).
        sentence = " ".join(["word"] * 19) + " end."
        out = self.run_page(self.SEND_TO + f"""
            chat.push({{role: "assistant", content: "{sentence} {sentence} {sentence}"}});
            set("chat-speak", "voice"); set("sp-voice", "A calm voice.");
            await press({{chatSpeak: "2"}});
            out.bodies = bodies();""")
        first, second = out["bodies"]
        clip = {"role": "assistant", "content": [{"type": "object", "id": "blake3:" + "a" * 64, "mime": "audio/wav"}],
                "snap": True}
        self.assertEqual(first["context"]["messages"], [{"role": "system", "content": "A calm voice.", "snap": True},
                                                        {"role": "user", "content": f"{sentence} {sentence}"}])
        self.assertEqual(second["context"]["messages"], [{"role": "system", "content": "A calm voice.", "snap": True},
                                                         {"role": "user", "content": f"{sentence} {sentence}", "snap": True},
                                                         clip, {"role": "user", "content": sentence}])

    def test_a_message_goes_to_a_model_that_speaks_beside_its_text_with_the_prompt_of_its_speech_mode(self):
        out = self.run_page(self.SEND_TO + """
            set("chat-speak", "lfm3-omni");
            await press({chatSpeak: "1"});
            out.bodies = bodies();""")
        (body,) = out["bodies"]
        self.assertEqual(body["model"], "lfm3-omni")
        self.assertEqual(body["output"], {"mime": "audio/wav", "options": {"mode": "speech"}})
        self.assertEqual(body["context"]["messages"][0]["role"], "system")
        self.assertIn("TTS", body["context"]["messages"][0]["content"])
        self.assertEqual(body["context"]["messages"][1]["content"], "Hello there. A red fox sits in the snow.")

    def test_a_message_goes_to_an_image_model_and_its_image_goes_to_the_tray_and_the_gallery(self):
        out = self.run_page(self.SEND_TO + """
            set("chat-draw", "pony-0");
            await press({chatDraw: "1"});
            out.bodies = bodies();
            out.clips = document.getElementById("chat-job-0-clips").innerHTML;
            out.status = document.getElementById("chat-job-0-status").textContent;
            out.gallery = gallery.map((g) => [g.model, g.prompt]);""")
        (body,) = out["bodies"]
        self.assertEqual((body["model"], body["stream"], body["max_tokens"]), ("pony-0", True, 1))
        self.assertEqual(body["context"]["messages"], [{"role": "user", "content": "Hello there. A red fox sits in the snow."}])
        self.assertEqual(body["output"], {"mime": "image/png", "options": {
            "width": 1024, "height": 1024, "steps": 30, "count": 1, "cfg": 7, "sampler": "dpm++ 2m karras", "negative": ""}})
        self.assertIn('/council/v2/objects/blake3:' + "d" * 64, out["clips"])
        self.assertIn("seed 7", out["status"])
        self.assertEqual(out["gallery"], [["pony-0", "Hello there. A red fox sits in the snow."]])

    def test_a_job_that_fails_shows_its_error_in_the_tray_and_the_chat_goes_on(self):
        out = self.run_page(self.SEND_TO + """
            const real = fetch;
            fetch = (path, o) => /generations$/.test(path)
              ? reply(400, {error: {type: "invalid_request", message: "nothing ran: no voice"}}) : real(path, o);
            set("chat-speak", "voice");
            await press({chatSpeak: "1"});
            out.status = document.getElementById("chat-job-0-status").textContent;
            out.chat = chat.length;""")
        self.assertIn("nothing ran: no voice", out["status"])
        self.assertEqual(out["chat"], 2)

    def test_a_chat_reply_with_no_snapshots_shows_its_counts_and_no_id(self):
        done = dict(self.CHAT_DONE, snapshots=[], usage={"prompt": 23, "fed": 23, "completion": 4})
        out = self.chat_turn(done)
        self.assertEqual(out["failed"], "")
        self.assertNotIn("/snapshots/", out["log"])
        self.assertIn("ran 23 of 23 prompt tokens", out["log"])

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

    def test_the_layers_of_a_chat_turn_give_the_strength_of_each_layer(self):
        out = self.run_page(f"""
            vectors = [{{name: "sea", id: "{self.VECTOR}", model: "lfm3", host}}];
            const sent = () => calls.filter((c) => c[1] === "/council/v2/generations").length;
            const body = () => JSON.parse(calls.filter((c) => c[1] === "/council/v2/generations").pop()[2]);
            set("chat-model", "lfm3");
            set("chat-vector", "{self.VECTOR}"); set("chat-op", "ablate"); set("chat-strength", "1");
            set("chat-layers", "9-11, 14:0.5"); document.getElementById("chat-layers").oninput();
            out.note = document.getElementById("chat-layers-note").textContent;
            set("chat-input", "hi"); await document.getElementById("chat-go").onclick();
            out.shaped = body().context.control;
            document.getElementById("chat-layers-early").onclick();
            out.early = document.getElementById("chat-layers").value;
            set("chat-input", "hi"); await document.getElementById("chat-go").onclick();
            out.early_sent = body().context.control[0].layers;
            document.getElementById("chat-layers-all").onclick();
            set("chat-input", "hi"); await document.getElementById("chat-go").onclick();
            out.all = body().context.control;
            const before = sent();
            set("chat-layers", "9-x"); document.getElementById("chat-layers").oninput();
            out.bad_note = document.getElementById("chat-layers-note").textContent;
            set("chat-input", "hi"); await document.getElementById("chat-go").onclick();
            out.bad_sent = sent() - before;
            out.chat_error = document.getElementById("chat-error").textContent;
            out.twice = (() => {{ try {{ parseLayers("3, 2-4"); return "ok"; }} catch (e) {{ return e.message; }} }})();""")
        self.assertEqual(out["shaped"], [{"vector": self.VECTOR, "op": "ablate", "strength": 1,
                                          "layers": {"9": 1, "10": 1, "11": 1, "14": 0.5}}])
        self.assertIn("4 layers", out["note"])
        self.assertEqual(out["early"], "9-14")
        self.assertEqual(out["early_sent"], {str(l): 1 for l in range(9, 15)})
        self.assertEqual(out["all"], [{"vector": self.VECTOR, "op": "ablate", "strength": 1}])
        self.assertEqual(out["bad_sent"], 0)
        self.assertIn("9-x", out["chat_error"])
        self.assertIn("9-x", out["bad_note"])
        self.assertIn("layer 3 twice", out["twice"])

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
        # Each file of a message shows the content id that the request names in place of its bytes.
        self.assertIn('objects/blake3:' + "f" * 64 + '" target="_blank" title="blake3:' + "f" * 64, out["log"])
        self.assertIn(">ffffffff</a>", out["log"])
        self.assertIn(">aaaaaaaa</a>", out["log"])
        # The next turn sends the spoken reply back as its text and its audio (spec 11.11, "Assistant audio"), and no
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
            out.puts = calls.filter((c) => c[0] === "PUT" && c[1].includes("/objects/")).length;""")
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


    # ---- prompt sets from dataset files (the Steering tab)
    # `load` gives a file to the input of a side, as a person does. `pick` changes one control of the side.
    DATA = """
        const fileOf = (name, text) => ({name, text: async () => text, arrayBuffer: async () => new ArrayBuffer(8)});
        const load = (side, name, text) =>
          document.getElementById(`cv-${side}-file`).onchange({target: {files: [fileOf(name, text)]}});
        const pick = (side, what, v) => { set(`cv-${side}-${what}`, v);
                                          return document.getElementById(`cv-${side}-${what}`).onchange(); };
        const box = (side) => document.getElementById(`cv-${side}`).value;
        const note = () => document.getElementById("cv-status").textContent;
        const fault = () => document.getElementById("cv-error").textContent;
        const shown = (side) => !document.getElementById(`cv-${side}-pick`).hidden;
        const sent = async () => { set("cv-model", "lfm3"); await document.getElementById("cv-make").onclick();
          const b = JSON.parse(calls.filter((c) => c[1] === "/council/v2/control-vectors").pop()[2]);
          return [b.positive.map((c) => c.messages[0].content), b.negative.map((c) => c.messages[0].content)]; };
        set("cv-model", "lfm3");"""

    def data_page(self, test, **files):
        """Run `test` after `DATA`, with each value of `files` as a constant of its name."""
        consts = "".join(f"const {name} = {json.dumps(value)};" for name, value in files.items())
        return self.run_page(self.DATA + consts + test)

    def test_a_text_file_loads_into_its_box_as_before_and_shows_no_dataset_controls(self):
        out = self.data_page("""
            await load("toward", "t.txt", "a\\n\\n b \\n");
            out.box = box("toward"); out.note = note(); out.shown = shown("toward");
            set("cv-away", "z"); out.sent = await sent();""")
        self.assertEqual(out["box"], "a\n\n b \n")
        self.assertEqual(out["note"], "t.txt: 2 prompts in toward")
        self.assertFalse(out["shown"])
        self.assertEqual(out["sent"], [["a", "b"], ["z"]])

    def test_a_json_lines_file_fills_the_box_from_its_usual_column_and_counts_the_skipped_rows(self):
        lines = "\n".join(json.dumps(row) for row in [
            {"id": "a1", "goal": "Write a poem", "n": 1}, {"id": "a2", "goal": "", "n": 2},
            {"id": "a3", "goal": 7, "n": 3}, {"id": "a4", "goal": "Tell a story", "n": 4}, {"id": "a5", "n": 5}])
        out = self.data_page("""
            await load("toward", "harm.jsonl", lines + "\\n\\n");
            out.box = box("toward"); out.note = note(); out.fault = fault(); out.shown = shown("toward");
            out.columns = document.getElementById("cv-toward-column").innerHTML;
            out.column = document.getElementById("cv-toward-column").value;
            await load("away", "x.ndjson", '"plain one"\\n"plain two"');
            out.away = box("away"); out.sent = await sent();
            await load("away", "bad.jsonl", '{"a": "x"}\\n{oops'); out.bad = fault(); out.kept = box("away");""",
                             lines=lines)
        self.assertEqual(out["fault"], "")
        self.assertTrue(out["shown"])
        # `id` is text too, and it comes first, but `goal` is a usual name of a prompt column.
        self.assertEqual(out["column"], "goal")
        self.assertEqual(re.findall(r'<option value="(\w+)">', out["columns"]), ["id", "goal", "n"])
        self.assertEqual(out["box"], "Write a poem\nTell a story")
        for part in ("harm.jsonl", "5 rows", "column goal", "2 chosen", "3 skipped"):
            self.assertIn(part, out["note"])
        self.assertNotIn("poem", out["note"])
        self.assertEqual(out["away"], "plain one\nplain two")
        self.assertEqual(out["sent"], [["Write a poem", "Tell a story"], ["plain one", "plain two"]])
        self.assertIn("line 2", out["bad"])
        self.assertEqual(out["kept"], "plain one\nplain two")

    def test_a_json_file_is_an_array_of_objects_or_of_strings_or_an_object_with_one_array(self):
        out = self.data_page("""
            await load("toward", "a.json", JSON.stringify([{text: "one"}, {text: "two"}])); out.objects = box("toward");
            await load("toward", "b.json", JSON.stringify(["three", "four"])); out.strings = box("toward");
            await load("toward", "c.json", JSON.stringify({version: 2, data: [{prompt: "five"}]}));
            out.member = box("toward"); out.fault = fault();
            const refused = [];
            for (const v of [{a: [1], b: [2]}, {a: 1}, 5, [], [1, 2], [{a: "x"}, "y"]]) {
              await load("toward", "bad.json", JSON.stringify(v)); refused.push(fault());
            }
            await load("toward", "broken.json", "[{"); refused.push(fault());
            out.refused = refused; out.after = box("toward");""")
        self.assertEqual(out["objects"], "one\ntwo")
        self.assertEqual(out["strings"], "three\nfour")
        self.assertEqual(out["member"], "five")
        self.assertEqual(out["fault"], "")
        self.assertEqual(len(out["refused"]), 7)
        for text in out["refused"]:
            self.assertRegex(text, r"^(bad|broken)\.json: ")
        self.assertIn("2 array members", out["refused"][0])
        self.assertIn("no array member", out["refused"][1])
        # A refused file changes no box.
        self.assertEqual(out["after"], "five")

    def test_a_csv_file_keeps_quoted_commas_quotes_and_newlines_and_a_tsv_file_splits_at_tabs(self):
        csv = '﻿id,prompt,note\r\n1,"Say ""hi"", then stop",x\r\n2,"line one\nline two",y\r\n3,plain,"a,b"\r\n'
        tsv = "n\tquestion\n1\tWhy, though?\n2\t\"tab\there\"\n"
        out = self.data_page("""
            await load("toward", "p.csv", csv);
            out.fault = fault(); out.box = box("toward"); out.note = note();
            out.columns = document.getElementById("cv-toward-column").innerHTML;
            await pick("toward", "column", "note"); out.notes = box("toward");
            await pick("toward", "column", "prompt");
            await load("away", "q.tsv", tsv); out.away = box("away"); out.fault2 = fault();
            out.sent = await sent();
            await load("away", "short.csv", "a,b\\n1\\n"); out.short = fault();
            await load("away", "open.csv", 'a,b\\n1,"x\\n'); out.open = fault();
            out.kept = box("away");""", csv=csv, tsv=tsv)
        self.assertEqual(out["fault"], "")
        self.assertEqual(re.findall(r'<option value="(\w+)">', out["columns"]), ["id", "prompt", "note"])
        # The box shows the newline inside a prompt as a mark, and the request holds the prompt whole.
        self.assertEqual(out["box"], 'Say "hi", then stop\nline one↵line two\nplain')
        self.assertIn("1 with a newline", out["note"])
        self.assertEqual(out["notes"], "x\ny\na,b")
        self.assertEqual(out["fault2"], "")
        self.assertEqual(out["away"], "Why, though?\ntab\there")
        self.assertEqual(out["sent"], [['Say "hi", then stop', "line one\nline two", "plain"],
                                       ["Why, though?", "tab\there"]])
        self.assertIn("row 2", out["short"])
        self.assertIn("quote", out["open"])
        self.assertEqual(out["kept"], "Why, though?\ntab\there")

    def test_the_default_column_is_the_first_usual_name_with_text_and_else_the_first_text_column(self):
        out = self.data_page("""
            const column = async (rows) => { await load("toward", "d.json", JSON.stringify(rows));
                                             return [document.getElementById("cv-toward-column").value, fault()]; };
            out.usual = await column([{category: "c", Instruction: "do it", text: "t"}]);
            out.empty_usual = await column([{n: 1, prompt: "", body: "b"}, {n: 2, prompt: null, body: "c"}]);
            out.first_text = await column([{n: 1, body: "b", more: "m"}]);
            out.before = box("toward");
            out.none = await column([{n: 1, flag: true}]);
            out.after = box("toward");
            await load("toward", "d.json", JSON.stringify([{category: "c", prompt: "p"}]));
            await pick("toward", "column", "category"); out.changed = [box("toward"), note()];""")
        self.assertEqual(out["usual"], ["Instruction", ""])
        self.assertEqual(out["empty_usual"], ["body", ""])
        self.assertEqual(out["first_text"], ["body", ""])
        self.assertIn("no column with text", out["none"][1])
        self.assertEqual(out["after"], out["before"])
        self.assertEqual(out["changed"][0], "c")
        self.assertIn("column category", out["changed"][1])

    def test_the_count_starts_at_the_limit_of_the_model_and_random_rows_repeat_with_their_seed(self):
        rows = [{"prompt": f"p{i}"} for i in range(100)]
        rows[3]["prompt"] = ""
        out = self.data_page("""
            await load("toward", "many.json", JSON.stringify(rows));
            out.count = document.getElementById("cv-toward-count").value;
            out.first = box("toward").split("\\n"); out.note = note();
            await pick("toward", "count", "5"); out.five = box("toward").split("\\n");
            await pick("toward", "order", "random"); await pick("toward", "seed", "7");
            out.seven = box("toward").split("\\n"); out.random_note = note();
            await pick("toward", "seed", "8"); out.eight = box("toward").split("\\n");
            await pick("toward", "seed", "7"); out.again = box("toward").split("\\n");
            await pick("toward", "count", "500"); out.all = box("toward").split("\\n").length;
            await pick("toward", "count", ""); out.each = box("toward").split("\\n").length;
            await pick("toward", "count", "0"); out.zero = fault(); out.kept = box("toward").split("\\n").length;
            await pick("toward", "count", "5"); await pick("toward", "seed", "-1"); out.seed = fault();
            await pick("toward", "seed", "7"); out.cleared = fault();
            set("cv-model", "lfm3-omni"); await load("away", "many.json", JSON.stringify(rows));
            out.no_limit = [document.getElementById("cv-away-count").value, box("away").split("\\n").length];""",
                             rows=rows)
        usable = [f"p{i}" for i in range(100) if i != 3]
        self.assertEqual(out["count"], "64")
        self.assertEqual(out["first"], usable[:64])
        for part in ("100 rows", "64 chosen", "first", "1 skipped"):
            self.assertIn(part, out["note"])
        self.assertEqual(out["five"], usable[:5])
        self.assertEqual(len(out["seven"]), 5)
        self.assertEqual(len(set(out["seven"])), 5)
        self.assertLessEqual(set(out["seven"]), set(usable))
        self.assertNotEqual(out["seven"], usable[:5])
        self.assertNotEqual(out["seven"], out["eight"])
        self.assertEqual(out["seven"], out["again"])
        self.assertIn("random with seed 7", out["random_note"])
        self.assertEqual(out["all"], 99)
        self.assertEqual(out["each"], 99)
        self.assertIn("count", out["zero"])
        self.assertEqual(out["kept"], 99)
        self.assertIn("seed", out["seed"])
        self.assertEqual(out["cleared"], "")
        # lfm3-omni gives no limit in this identity, so each row with text is chosen.
        self.assertEqual(out["no_limit"], ["", 99])

    def test_a_box_that_a_person_edits_after_a_dataset_load_is_lines_again(self):
        out = self.data_page("""
            await load("toward", "n.json", JSON.stringify(["one\\ntwo", "three"]));
            out.box = box("toward");
            set("cv-toward", "one↵two\\nthree\\nfour"); document.getElementById("cv-toward").oninput();
            out.note = note(); set("cv-away", "z"); out.sent = await sent();
            // A text file after a dataset ends the dataset of its side, also when it holds the text that the box had.
            await load("toward", "n.json", JSON.stringify(["one\\ntwo", "three"]));
            await load("toward", "same.txt", "one↵two\\nthree"); out.text = await sent();""")
        self.assertEqual(out["box"], "one↵two\nthree")
        self.assertIn("each line of toward is one prompt", out["note"])
        self.assertEqual(out["sent"], [["one↵two", "three", "four"], ["z"]])
        self.assertEqual(out["text"], [["one↵two", "three"], ["z"]])

    def test_a_parquet_file_goes_through_the_reader_and_then_the_same_choices(self):
        out = self.data_page("""
            let got = null;
            readParquet = async (buffer) => { got = buffer.byteLength;
              return {columns: ["source", "prompt"], rows: [{source: "s", prompt: "from parquet"}, {source: "s", prompt: 3}]}; };
            await load("toward", "train-00000-of-00001.parquet", "not read as text");
            out.got = got; out.box = box("toward"); out.note = note(); out.fault = fault();
            out.column = document.getElementById("cv-toward-column").value;
            readParquet = async () => { throw new Error("parquet unsupported compression codec: ZSTD"); };
            await load("toward", "z.parquet", ""); out.codec = fault(); out.after = box("toward");""")
        self.assertEqual(out["got"], 8)
        self.assertEqual(out["fault"], "")
        self.assertEqual(out["column"], "prompt")
        self.assertEqual(out["box"], "from parquet")
        for part in ("train-00000-of-00001.parquet", "2 rows", "1 chosen", "1 skipped"):
            self.assertIn(part, out["note"])
        self.assertIn("ZSTD", out["codec"])
        self.assertIn("Snappy", out["codec"])
        self.assertEqual(out["after"], "from parquet")

    def test_the_parquet_reader_names_the_columns_of_the_schema_and_says_when_it_does_not_load(self):
        out = self.data_page("""
            // QuickJS loads no module from the network, as a browser with no route to the CDN.
            await load("toward", "a.parquet", ""); out.offline = fault(); out.box = box("toward");
            const asked = [];
            parquetReader = Promise.resolve({
              parquetMetadataAsync: async (file) => ({of: file.byteLength}),
              parquetSchema: (metadata) => ({children: [{element: {name: "prompt"}}, {element: {name: "label"}}]}),
              parquetReadObjects: async (o) => { asked.push(Object.keys(o).sort().join()); return [{prompt: "p", label: 1}]; }});
            const table = await readParquet(new ArrayBuffer(8));
            out.columns = table.columns; out.rows = table.rows.length; out.asked = asked;""")
        self.assertIn("did not load", out["offline"])
        self.assertIn("cdn.jsdelivr.net", out["offline"])
        self.assertEqual(out["box"], "")
        self.assertEqual(out["columns"], ["prompt", "label"])
        self.assertEqual(out["rows"], 1)
        self.assertEqual(out["asked"], ["file,metadata"])

    def test_the_parquet_reader_is_pinned_to_one_version_and_loads_only_at_a_use(self):
        script = WEB.joinpath("steer.js").read_text()
        self.assertEqual(re.findall(r'"(https://[^"]*hyparquet[^"]*)"', script),
                         ["https://cdn.jsdelivr.net/npm/hyparquet@1.33.0/+esm"])
        self.assertEqual(script.count("import("), 1)
        self.assertIn("import(", script[script.index("async function readParquet"):])


if __name__ == "__main__":
    unittest.main()
