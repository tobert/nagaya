"use strict";
// The Steering tab: make a control vector from two prompt sets, and try it against the plain model (spec 11.12).
// This file loads after the other tabs, because `fillVectors` fills the vector list of each of them.

// The saved control vectors of this browser, newest first: {name, id, model, host, toward, away}.
let vectors = recallJson("nagaya-vectors", []);
const saveVectors = () => rememberJson("nagaya-vectors", vectors);
/** Whether a served model takes a control list in its contexts (gogi "Contexts", spec 11.12). */
const steers = (name) => models.some((m) => m.model === name && m.capabilities?.includes("control_vectors"));
/** The control list of a context: one item when a saved vector of the model is chosen, and else none. */
function controlOf(model, id, op, strength, layers) {
  const v = vectors.find((x) => x.id === id && x.model === model && here(x.host));
  if (!steers(model) || !v) return undefined;
  const item = {vector: v.id, op, strength: +strength};
  if (layers) item.layers = layers;
  return [item];
}
/** The `layers` of a control item from text such as "9-14" or "9-14:0.3, 15-22:1": each entry is a layer or a range,
 *  with an optional number after ":" (1 when it has none). Empty text is every layer: `null` (spec 11.12, "The
 *  strength of each layer"). A mistake throws an Error that quotes the entry. */
function parseLayers(text) {
  const entries = text.split(/[,\s]+/).map((e) => e.trim()).filter(Boolean);
  if (!entries.length) return null;
  const out = {};
  for (const entry of entries) {
    const m = entry.match(/^(\d+)(?:-(\d+))?(?::(-?\d*\.?\d+(?:e-?\d+)?))?$/i);
    if (!m) throw new Error(`"${entry}" is not a layer, a range like 9-14, or either with ":number"`);
    const [from, to] = [+m[1], m[2] === undefined ? +m[1] : +m[2]];
    if (to < from) throw new Error(`"${entry}" runs backwards`);
    const number = m[3] === undefined ? 1 : +m[3];
    for (let l = from; l <= to; l++) {
      if (String(l) in out) throw new Error(`"${entry}" names layer ${l} twice`);
      out[String(l)] = number;
    }
  }
  return out;
}
const prompts = (text) => text.split("\n").map((t) => t.trim()).filter(Boolean);
const plainContext = (text) => ({thinking: false, messages: [{role: "user", content: text}]});
function fillVectors() {
  for (const [id, model] of [["cv-vector", $("cv-model").value], ["chat-vector", $("chat-model").value],
                            ["lab-vector", $("lab-model").value], ["omni-vector", $("omni-model").value]]) {
    const list = vectors.filter((v) => v.model === model && here(v.host));
    const none = {"chat-vector": "none", "lab-vector": "none: trace plain only", "omni-vector": "none"}[id]
      || "choose a vector";
    const html = `<option value="">${none}</option>` +
      list.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}</option>`).join("");
    const sel = $(id), keep = sel.value;
    if (sel.dataset.html !== html) { sel.innerHTML = html; sel.dataset.html = html; sel.value = keep; }
  }
  // Each button names the index of its vector in `vectors`, the list that Forget and Lens change and read.
  $("cv-list").innerHTML = vectors.map((v, i) => [v, i]).filter(([v]) => here(v.host)).map(([v, i]) =>
    `<div class="lora"><span class="name" title="${esc(v.id)}">${esc(v.name)} <span class="note">${esc(v.model)}, ${
      v.toward} toward, ${v.away} away</span></span><span></span>
     <button class="plain small" data-drop-vector="${i}">Forget</button></div>`).join("") || `<p class="note">None yet.</p>`;
}
document.addEventListener("click", (ev) => {
  const {dropVector} = ev.target.dataset || {};
  if (dropVector === undefined) return;
  vectors.splice(+dropVector, 1); saveVectors(); fillVectors(); keepAll();
});
for (const side of ["toward", "away"]) {
  $(`cv-${side}-file`).onchange = async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    $(`cv-${side}`).value = await file.text();
    $("cv-status").textContent = `${file.name}: ${prompts($(`cv-${side}`).value).length} prompts in ${side}`;
  };
}
$("cv-make").onclick = async () => {
  const model = $("cv-model").value, positive = prompts($("cv-toward").value), negative = prompts($("cv-away").value);
  const most = models.find((m) => m.model === model)?.limits?.control_vector_contexts;
  const status = (t) => { $("cv-status").textContent = t; };
  $("cv-error").textContent = "";
  if (!model) { $("cv-error").textContent = "the nagaya serves no model with control_vectors"; return; }
  for (const [side, list] of [["toward", positive], ["away", negative]]) {
    if (!list.length) { $("cv-error").textContent = `write one prompt or more in ${side}`; return; }
    if (most && list.length > most) {
      $("cv-error").textContent = `${side} has ${list.length} prompts, and ${model} takes at most ${most} on each side`;
      return;
    }
  }
  $("cv-make").disabled = true;
  try {
    await ready(model, status);
    status("making the vector…");
    const started = performance.now();
    const r = await call("POST", "/council/v2/control-vectors",
                         {model, positive: positive.map(plainContext), negative: negative.map(plainContext)});
    const name = $("cv-name").value.trim() || `vector ${vectors.length + 1}`;
    vectors = [{name, id: r.vector, model, host, toward: positive.length, away: negative.length}]
      .concat(vectors.filter((v) => v.id !== r.vector));
    saveVectors(); fillVectors(); await keepAll();
    status(`${name}: ${r.layers} layers of ${r.width}, from ${positive.length + negative.length} prompts, ${
      ((performance.now() - started) / 1000).toFixed(1)} s`);
  } catch (e) {
    $("cv-error").textContent = e.message; status("");
  }
  $("cv-make").disabled = false;
};
$("cv-add").onclick = async () => {
  const id = $("cv-add-id").value.trim(), model = $("cv-model").value;
  $("cv-error").textContent = "";
  if (!/^blake3:[0-9a-f]{64}$/.test(id)) { $("cv-error").textContent = "an id is blake3: and 64 hex digits"; return; }
  if (!model) { $("cv-error").textContent = "the nagaya serves no model with control_vectors"; return; }
  try {
    const r = await fetch(base() + `/council/v2/objects/${id}`, {method: "HEAD"});
    if (!r.ok) { $("cv-error").textContent = `the nagaya does not hold ${id} (HEAD returned ${r.status})`; return; }
  } catch (e) {
    $("cv-error").textContent = e.message; return;
  }
  const name = $("cv-add-name").value.trim() || `vector ${vectors.length + 1}`;
  vectors = [{name, id, model, host, toward: "?", away: "?"}].concat(vectors.filter((v) => v.id !== id));
  saveVectors(); fillVectors(); await keepAll();
  $("cv-status").textContent = `${name} added for ${model}`;
};
$("cv-try").onclick = async () => {
  const model = $("cv-model").value, text = $("cv-prompt").value.trim();
  const control = controlOf(model, $("cv-vector").value, $("cv-op").value, $("cv-strength").value);
  const status = (t) => { $("cv-try-status").textContent = t; };
  $("cv-try-error").textContent = "";
  if (!text) { $("cv-try-error").textContent = "write a prompt"; return; }
  if (!control) { $("cv-try-error").textContent = "choose a saved vector of this model"; return; }
  $("cv-try").disabled = true;
  try {
    await ready(model, status);
    for (const [id, list] of [["cv-plain", undefined], ["cv-steered", control]]) {
      const context = plainContext(text);
      if (list) context.control = list;
      $(id).textContent = "";
      status(id === "cv-plain" ? "plain…" : "steered…");
      const done = await stream({model, max_tokens: +$("cv-max").value, sample: {temperature: +$("cv-temp").value},
                                 context}, undefined, (name, d) => {
        if (name === "token" && d.phase === "content") $(id).textContent += d.piece;
      });
      if (typeof done.content === "string") $(id).textContent = done.content;
    }
    status("");
  } catch (e) {
    $("cv-try-error").textContent = e.message; status("");
  }
  $("cv-try").disabled = false;
};
modelSelect("cv-model", (m) => steers(m.model));
onRefresh(fillVectors);
keeps(() => vectors);
