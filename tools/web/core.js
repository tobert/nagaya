"use strict";
// The core of the page: the nagaya on show, the calls, the served models, the objects and the tabs. It loads first.
// Each tab has its own file, which loads after it and shares these names. `README.md` says how to add a tab.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;"}[c]));

// ---- the state that this browser keeps
// A browser can refuse localStorage. Then the page works, and it forgets its state at a reload.
/** The text that this browser keeps under `key`, or null. */
function recall(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}
function remember(key, text) {
  try { localStorage.setItem(key, text); } catch (e) {}
}
/** The JSON value that this browser keeps under `key`, or `otherwise` when it keeps none that parses. */
function recallJson(key, otherwise) {
  try { return JSON.parse(recall(key)) ?? otherwise; } catch (e) { return otherwise; }
}
const rememberJson = (key, value) => remember(key, JSON.stringify(value));

// ---- the served models
let models = [], engines = [];
// The profile engine of each served model of this nagaya, from the cluster route: {model: engine}. Identity gives some
// models the hash of their engine, not its name in the profile.
let engineNames = {};

/** Whether a served model thinks: only then does a context to it take `thinking: true` (gogi "Contexts"). */
const thinks = (name) => models.some((m) => m.model === name && m.capabilities?.includes("thinking"));
const kindOf = (m) => {
  if (m.capabilities?.includes("embed")) return "embed";
  const out = m.media?.outputs || [];
  if (out.some((o) => o.startsWith("image/"))) return "image";
  // A model whose replies hold text, such as Omni, speaks beside its text: it is a chat model (spec 11.11).
  if (out.some((o) => o.startsWith("audio/")) && m.media?.text_replies === false) return "speech";
  return m.media?.text_replies === false ? "other" : "text";
};
/** The profile engine of a served model: the one that the cluster route names, the one that its identity names, or
 *  else the engine that names no model. */
function engineOf(m) {
  if (engineNames[m.model]) return engineNames[m.model];
  if (engines.some((e) => e.name === m.engine)) return m.engine;
  const named = new Set(models.map((x) => x.engine));
  return engines.find((e) => !named.has(e.name))?.name;
}
const modelOf = (engine) => models.find((m) => engineOf(m) === engine);

// ---- the nagayas
// The app passes /at/<name>/ to the nagaya with that name (nagaya_web.py). With no list, the page uses the plain paths,
// which go to the first nagaya. `host` is the name of the nagaya that the page shows, or "" for the plain paths.
let host = "", firstHost = "";
const base = (h = host) => h ? `/at/${encodeURIComponent(h)}` : "";
/** Whether a thing with the nagaya name `h` is on the nagaya that the page shows. No name is the first nagaya. */
const here = (h) => (h || firstHost) === (host || firstHost);
const hostsReady = (async () => {
  try {
    const r = await fetch("/app/nagayas");
    const list = r.ok ? (await r.json()).nagayas || [] : [];
    if (!list.length) return;
    firstHost = list[0].name;
    const saved = recall("nagaya-host");
    host = list.some((n) => n.name === saved) ? saved : firstHost;
    $("host").innerHTML = list.map((n) =>
      `<option value="${esc(n.name)}" title="${esc(n.url)}">${esc(n.name)}</option>`).join("");
    $("host").value = host;
    $("host").hidden = list.length < 2;
  } catch (e) {
    // An app from before the list of nagayas has no /app/nagayas. The plain paths still work.
  }
})();
// A tab gives `onHost` a function that draws what belongs to one nagaya, such as its gallery.
const hostDraws = [];
const onHost = (draw) => { hostDraws.push(draw); };
$("host").onchange = async () => {
  host = $("host").value;
  remember("nagaya-host", host);
  models = []; engines = [];
  await refresh();
  for (const draw of hostDraws) draw();
  keepAll();
};

// ---- calls
async function call(method, path, body) {
  const r = await fetch(base() + path, {method, headers: body ? {"content-type": "application/json"} : {},
                               body: body ? JSON.stringify(body) : undefined});
  const text = await r.text();
  let v = null;
  try { v = text ? JSON.parse(text) : null; } catch (e) { v = {error: {message: text}}; }
  if (!r.ok) throw failure(v, `${method} ${path} returned ${r.status}`, r.status);
  return v;
}

/** The Error of a gogi error body. The message names `param` when the nagaya's message does not. */
function failure(v, otherwise, status) {
  const param = v?.error?.param, text = v?.error?.message || otherwise;
  const error = new Error(param && !text.includes(param) ? `${param}: ${text}` : text);
  error.status = status; error.param = param;
  return error;
}

/** Read a server-sent event stream; call `on(name, data)` for each event. */
async function events(response, on) {
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
      let name = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) on(name, JSON.parse(data));
    }
  }
}

/** POST a streamed generation. Throws the error body of a refused one. */
async function stream(body, signal, on) {
  const r = await fetch(base() + "/council/v2/generations", {method: "POST", signal,
    headers: {"content-type": "application/json"}, body: JSON.stringify({...body, stream: true})});
  if (!r.ok) {
    const v = await r.json().catch(() => ({}));
    throw failure(v, `the generation returned ${r.status}`, r.status);
  }
  let done = null;
  await events(r, (name, data) => {
    if (name === "done") done = data;
    else if (name === "error") throw failure(data, "the generation failed");
    else on(name, data);
  });
  if (!done) throw new Error("the stream ended with no done event");
  return done;
}

// ---- the refresh
// A tab gives `onRefresh` a function that draws what depends on the served models. The functions run in the order
// of the files, after each read of the nagaya.
const refreshDraws = [];
const onRefresh = (draw) => { refreshDraws.push(draw); };

async function refresh() {
  await hostsReady;
  const asked = host;
  let profile;
  try {
    const [id, active, cluster] = await Promise.all([call("GET", "/council/v2/identity"),
                                            call("GET", "/nagaya/v1/profile").catch(() => ({engines: []})),
                                            call("GET", "/nagaya/v1/cluster").catch(() => ({nodes: []}))]);
    // A refresh that started before the choice of another nagaya ends with no effect.
    if (asked !== host) return;
    profile = active;
    models = id.models; engines = profile.engines || [];
    engineNames = {};
    for (const e of (cluster?.nodes || []).find((n) => n.self)?.engines || []) {
      for (const model of e.models || []) engineNames[model] = e.name;
    }
  } catch (e) {
    $("chips").innerHTML = `<span class="chip failed">the nagaya does not answer</span>`;
    $("eng-error").textContent = e.message;
    return;
  }
  $("chips").innerHTML = models.map((m) =>
    `<span class="chip ${esc(m.status)}" title="${esc(m.status)}"><i></i>${esc(m.model)}</span>`).join("");
  for (const draw of refreshDraws) draw(profile);
}

/**
 * Fill the model list `id` with the served models that `accepts` takes, and show the state of the chosen one. The
 * list keeps its choice, and a new page takes the choice that this browser keeps.
 */
function fillSelect(id, accepts) {
  const sel = $(id), keep = sel.value;
  const list = models.filter(accepts);
  const html = list.map((m) => `<option value="${esc(m.model)}">${esc(m.model)}</option>`).join("");
  if (sel.dataset.html !== html) {
    sel.innerHTML = html; sel.dataset.html = html;
    const saved = recall(`nagaya-${id}`);
    if (list.some((m) => m.model === keep)) sel.value = keep;
    else if (list.some((m) => m.model === saved)) sel.value = saved;
  }
  const m = list.find((x) => x.model === sel.value), state = $(`${id}-state`);
  state.innerHTML = m ? `<span class="${esc(m.status)}">${esc(m.status)}</span>` +
    (m.status === "ready" ? "" : ` <button class="plain small" data-load="${esc(engineOf(m) || "")}">Load</button>`)
    : "none served";
}
/** Make `id` the model list of a tab: `accepts` chooses its models, and this browser keeps the choice. */
function modelSelect(id, accepts) {
  $(id).onchange = () => { remember(`nagaya-${id}`, $(id).value); refresh(); };
  onRefresh(() => fillSelect(id, accepts));
}

// ---- objects
/** The path of an object on the nagaya `h`. An image or a LoRA keeps the name of its nagaya. */
const objectUrl = (id, h = host) => `${base(h)}/council/v2/objects/${id}`;

// ---- content ids
// The nagaya names an object and a snapshot by a content id: `blake3:` and 64 hex digits. The page shows the first 8.
const shortId = (id) => String(id).replace(/^blake3:/, "").slice(0, 8);
const sizeOf = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} kB` : `${n} B`;
/** The count of bytes of a text in UTF-8, which is its size in a request. */
const utf8Length = (text) => encodeURIComponent(text).replace(/%[0-9A-F]{2}/g, "x").length;
/** A snapshot of a reply as a link to its record on the nagaya: its id, its tokens, its tier and its size. */
function snapshotChip(s) {
  const facts = [`${s.tokens} tokens`, s.tier].concat(s.bytes == null ? [] : [sizeOf(s.bytes)]).filter(Boolean);
  return `<a class="cid" href="${esc(base())}/council/v2/snapshots/${esc(s.id)}" target="_blank" ` +
    `title="${esc(s.id)}: the model state after this message. A request that starts with the same tokens starts ` +
    `from it.">${esc(shortId(s.id))}</a> <span class="note">${esc(facts.join(" · "))}</span>`;
}
/** A context record as a link to its context on the nagaya: the next request names this id as its parent. */
function recordChip(id) {
  return `<a class="cid" href="${esc(base())}/council/v2/contexts/${esc(id)}" target="_blank" title="${esc(id)}: ` +
    `the record of this chat with this reply. The next turn names it and sends only its new message.">${
    esc(shortId(id))}</a>`;
}
/** An object of a message as a link to its bytes: its id and its type. A request names the id, not the bytes. */
function objectChip(f) {
  return `<a class="cid" href="${esc(objectUrl(f.id, f.host))}" target="_blank" title="${esc(f.id)}: the request ` +
    `names this id, and the nagaya reads the bytes from its store.">${esc(shortId(f.id))}</a> ` +
    `<span class="note">${esc([f.name, f.mime].filter(Boolean).join(" · "))}</span>`;
}

// A tab gives `keeps` a function that returns its objects, each {id, host}. One keep holds them all, so the store
// does not evict them. A keep lasts 30 days (spec 6.15).
const keepLists = [];
const keeps = (list) => { keepLists.push(list); };
async function keepAll() {
  const objects = [...new Set(keepLists.flatMap((list) => list()).filter((x) => here(x.host)).map((x) => x.id))]
    .filter((id) => /^blake3:[0-9a-f]{64}$/.test(id));
  try {
    await call("PUT", "/council/v2/keeps/nagaya-web", {objects, ttl_s: 30 * 86400});
  } catch (e) {
    $("keep-error").textContent = `the keep failed: ${e.message}`;
  }
}

let hashWasm = null;
/** The BLAKE3 id of a file, from hash-wasm, which loads on the first use. */
async function blake3Id(file, status) {
  if (!hashWasm) {
    hashWasm = new Promise((ok, fail) => {
      const tag = document.createElement("script");
      tag.src = "https://cdn.jsdelivr.net/npm/hash-wasm@4.11.0/dist/index.umd.min.js";
      tag.onload = () => ok(window.hashwasm);
      tag.onerror = () => { hashWasm = null; fail(new Error("hash-wasm did not load from cdn.jsdelivr.net")); };
      document.head.appendChild(tag);
    });
  }
  const hasher = await (await hashWasm).createBLAKE3();
  hasher.init();
  const reader = file.stream().getReader();
  let done = 0;
  for (;;) {
    const {value, done: end} = await reader.read();
    if (end) break;
    hasher.update(value); done += value.length;
    status(`hashing ${(100 * done / file.size).toFixed(0)}%`);
  }
  return `blake3:${hasher.digest("hex")}`;
}

/** Put a blob as an object of the nagaya on show, and give its id. A held object is not sent again (spec 6.15). */
async function putObject(blob, mime, status) {
  const id = await blake3Id(blob, status);
  const head = await fetch(objectUrl(id), {method: "HEAD"});
  if (head.status === 404) {
    status(`sending ${(blob.size / 1e6).toFixed(1)} MB…`);
    const r = await fetch(objectUrl(id), {method: "PUT", body: blob, headers: {"content-type": mime}});
    if (!r.ok) {
      const v = await r.json().catch(() => ({}));
      throw failure(v, `the put returned ${r.status}`, r.status);
    }
  } else if (!head.ok) {
    throw new Error(`HEAD of the object returned ${head.status}`);
  }
  return id;
}

// ---- tabs
// A tab is a button of the rail with `data-tab`, and the section with that id. The address keeps the tab after `#`,
// so a link opens a tab, and this browser keeps the last one.
const tabButtons = () => [...document.querySelectorAll("nav button[data-tab]")];
function showTab(name) {
  const buttons = tabButtons();
  if (!buttons.some((b) => b.dataset.tab === name)) return false;
  for (const b of buttons) b.classList.toggle("on", b.dataset.tab === name);
  for (const s of document.querySelectorAll("main > section")) s.classList.toggle("on", s.id === name);
  remember("nagaya-tab", name);
  return true;
}
for (const b of tabButtons()) {
  b.onclick = () => {
    showTab(b.dataset.tab);
    history.replaceState(null, "", `#${b.dataset.tab}`);
    $("rail").classList.remove("open");
  };
}
$("rail-toggle").onclick = () => $("rail").classList.toggle("open");
