"use strict";
// The Steering tab: make a control vector from two prompt sets, and try it against the plain model (spec 11.12).
// A prompt set is the lines of its box, or the rows of a dataset file: Parquet, JSON Lines, JSON, CSV or TSV.
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
// ---- a prompt set from a dataset file
// A table is {columns, rows}: the column names in the order of the file, and each row as an object from a column to
// its value. Each reader below gives a table, and it throws an Error that says what the file holds.

/** The table of a list of rows. Each row is an object, or each row is a string, which gives the one column `text`. */
function tableOf(list, what) {
  if (!list.length) throw new Error(`${what} holds no row`);
  const plain = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
  if (list.every((v) => typeof v === "string")) return {columns: ["text"], rows: list.map((text) => ({text}))};
  const odd = list.findIndex((v) => !plain(v));
  if (odd >= 0) {
    throw new Error(`row ${odd + 1} of ${what} is ${JSON.stringify(list[odd]).slice(0, 40)}; ` +
                    "each row is an object, or each row is a string");
  }
  const columns = [];
  for (const row of list) for (const name of Object.keys(row)) if (!columns.includes(name)) columns.push(name);
  return {columns, rows: list};
}
/** The table of JSON Lines: one JSON value on each line that is not empty. */
function parseJsonLines(text) {
  const list = [];
  text.split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    try { list.push(JSON.parse(line)); } catch (e) { throw new Error(`line ${i + 1} is not JSON: ${e.message}`); }
  });
  return tableOf(list, "the file");
}
/** The table of a JSON file: an array of rows, or an object with one member that is an array of rows. */
function parseJson(text) {
  let v;
  try { v = JSON.parse(text); } catch (e) { throw new Error(`the file is not JSON: ${e.message}`); }
  if (Array.isArray(v)) return tableOf(v, "the array");
  if (v === null || typeof v !== "object") throw new Error("the file holds one value; the page reads an array of rows");
  const members = Object.keys(v).filter((k) => Array.isArray(v[k]));
  if (members.length !== 1) {
    throw new Error(members.length ? `the object has ${members.length} array members (${members.join(", ")}); the ` +
      "page reads an object with one" : "the object has no array member; the page reads an array of rows");
  }
  return tableOf(v[members[0]], `the member ${members[0]}`);
}
/** The table of a CSV or TSV file. The first row is the header. A field in quotes can hold the separator, a newline,
 *  and a quote written twice. */
function parseSeparated(text, separator) {
  if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
  const records = [];
  let record = [], field = "", quoted = false, wasQuoted = false;
  const endField = () => { record.push(field); field = ""; wasQuoted = false; };
  // A line with no character is no record. A line of one empty quoted field is a record.
  const endRecord = (blank) => { if (!blank) { endField(); records.push(record); } record = []; };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c !== "\"") field += c;
      else if (text[i + 1] === "\"") { field += c; i++; }
      else quoted = false;
    } else if (c === "\"" && field === "" && !wasQuoted) { quoted = true; wasQuoted = true; }
    else if (c === separator) endField();
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      endRecord(!record.length && field === "" && !wasQuoted);
    } else field += c;
  }
  if (quoted) throw new Error(`a quote of record ${records.length + 1} does not close`);
  endRecord(!record.length && field === "" && !wasQuoted);
  if (!records.length) throw new Error("the file holds no header row");
  const [columns, ...rest] = records;
  if (new Set(columns).size !== columns.length) throw new Error(`the header names a column twice: ${columns.join(", ")}`);
  if (!rest.length) throw new Error("the file holds a header and no row");
  return {columns, rows: rest.map((fields, i) => {
    if (fields.length !== columns.length) {
      throw new Error(`row ${i + 2} has ${fields.length} fields, and the header has ${columns.length}`);
    }
    return Object.fromEntries(columns.map((name, k) => [name, fields[k]]));
  })};
}

// hyparquet reads Parquet with no compression and with Snappy. Each other codec, such as ZSTD or GZip, needs a second
// package, which the page does not load. The module loads at the first Parquet file.
let parquetReader = null;
/** The table of a Parquet file. */
async function readParquet(buffer) {
  if (!parquetReader) {
    parquetReader = import("https://cdn.jsdelivr.net/npm/hyparquet@1.33.0/+esm").catch((e) => {
      parquetReader = null;
      throw new Error(`the Parquet reader hyparquet did not load from cdn.jsdelivr.net: ${e.message}`);
    });
  }
  const reader = await parquetReader;
  const metadata = await reader.parquetMetadataAsync(buffer);
  const columns = reader.parquetSchema(metadata).children.map((c) => c.element.name);
  return {columns, rows: await reader.parquetReadObjects({file: buffer, metadata})};
}
/** The table of a dataset file, by the end of its name, or null for a file of prompt lines. */
async function readDataset(file) {
  const kind = file.name.toLowerCase().match(/\.([a-z0-9]+)$/)?.[1];
  if (kind === "parquet") {
    try {
      return await readParquet(await file.arrayBuffer());
    } catch (e) {
      throw new Error(/compression codec/.test(e.message) ? `${e.message}; the page reads Parquet with Snappy or with ` +
        "no compression" : e.message);
    }
  }
  if (kind === "jsonl" || kind === "ndjson") return parseJsonLines(await file.text());
  if (kind === "json") return parseJson(await file.text());
  if (kind === "csv") return parseSeparated(await file.text(), ",");
  if (kind === "tsv") return parseSeparated(await file.text(), "\t");
  return null;
}

/** Whether a value of a row is a prompt: text with a character that is not a space. */
const isText = (v) => typeof v === "string" && v.trim() !== "";
const USUAL_COLUMNS = ["prompt", "text", "instruction", "goal", "question", "input", "behavior", "request"];
/** The column that a table starts with: the first with text and a usual name, or else the first with text. */
function defaultColumn(table) {
  const texts = table.columns.filter((name) => table.rows.some((row) => isText(row[name])));
  if (!texts.length) throw new Error(`no column with text among ${table.columns.join(", ")}`);
  return texts.find((name) => USUAL_COLUMNS.includes(name.toLowerCase())) || texts[0];
}
/** A generator of numbers from 0 to 1 that repeats for its seed (mulberry32). */
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** The prompts of a column: `count` rows with text, the first ones or random ones of `seed`, in the order of the
 *  file. `count` null is each row with text. Gives {prompts, skipped}. */
function choosePrompts(table, column, count, random, seed) {
  const texts = table.rows.map((row) => row[column]).filter(isText);
  const skipped = table.rows.length - texts.length;
  if (count === null || count >= texts.length) return {prompts: texts, skipped};
  if (!random) return {prompts: texts.slice(0, count), skipped};
  // A partial Fisher-Yates shuffle of the row numbers: the first `count` places hold the choice.
  const order = texts.map((_, i) => i), next = seeded(seed);
  for (let i = 0; i < count; i++) {
    const j = i + Math.floor(next() * (order.length - i));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return {prompts: order.slice(0, count).sort((a, b) => a - b).map((i) => texts[i]), skipped};
}

// The dataset of each side, {name, table}, and its chosen prompts, {shown, prompts}. A prompt of a dataset can hold
// a newline, so the request takes the prompts from here, and the box shows each on one line with ↵ for a newline.
const datasets = {}, chosen = {};
/** The prompts of a side for a request: the chosen rows of its dataset while the box shows them, and else the lines
 *  of the box. */
const sidePrompts = (side) => chosen[side] && $(`cv-${side}`).value === chosen[side].shown
  ? chosen[side].prompts : prompts($(`cv-${side}`).value);
/** Fill the box of a side from its dataset and the choices of the person. */
function drawChoice(side) {
  const {name, table} = datasets[side], field = (what) => $(`cv-${side}-${what}`).value;
  const column = field("column"), random = field("order") === "random";
  $("cv-error").textContent = "";
  try {
    const count = field("count").trim() === "" ? null : +field("count"), seed = +(field("seed").trim() || 0);
    if (count !== null && !(Number.isInteger(count) && count >= 1)) {
      throw new Error(`the count of ${side} is a whole number from 1, or empty for each row`);
    }
    if (!(Number.isInteger(seed) && seed >= 0 && seed <= 0xFFFFFFFF)) {
      throw new Error(`the seed of ${side} is a whole number from 0 to 4294967295`);
    }
    const {prompts: list, skipped} = choosePrompts(table, column, count, random, seed);
    if (!list.length) throw new Error(`the column ${column} of ${name} holds no text`);
    const shown = list.map((text) => text.replace(/\r\n|\r|\n/g, "↵")).join("\n");
    chosen[side] = {shown, prompts: list};
    $(`cv-${side}`).value = shown;
    const lines = list.filter((text) => /[\r\n]/.test(text)).length;
    $("cv-status").textContent = `${name}: ${table.rows.length} rows, column ${column}, ${list.length} chosen (${
      random ? `random with seed ${seed}` : "first"}), ${skipped} skipped in ${side}` +
      (lines ? `; ${lines} with a newline, shown as ↵ and sent whole` : "");
  } catch (e) {
    $("cv-error").textContent = e.message;
  }
}
for (const side of ["toward", "away"]) {
  $(`cv-${side}-file`).onchange = async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    $("cv-error").textContent = "";
    let table, column;
    try {
      table = await readDataset(file);
      if (table) column = defaultColumn(table);
    } catch (e) {
      // A file that the page cannot read changes no box and no choice.
      $("cv-error").textContent = `${file.name}: ${e.message}`;
      return;
    }
    if (!table) {
      delete datasets[side]; delete chosen[side];
      $(`cv-${side}-pick`).hidden = true;
      $(`cv-${side}`).value = await file.text();
      $("cv-status").textContent = `${file.name}: ${prompts($(`cv-${side}`).value).length} prompts in ${side}`;
      return;
    }
    datasets[side] = {name: file.name, table};
    $(`cv-${side}-column`).innerHTML = table.columns.map((name) =>
      `<option value="${esc(name)}">${esc(name)}</option>`).join("");
    $(`cv-${side}-column`).value = column;
    // The count starts at the most contexts that the model takes on a side. With no limit, it is each row.
    const most = models.find((m) => m.model === $("cv-model").value)?.limits?.control_vector_contexts;
    $(`cv-${side}-count`).value = most ? String(most) : "";
    $(`cv-${side}-order`).value = "first";
    $(`cv-${side}-pick`).hidden = false;
    drawChoice(side);
  };
  for (const what of ["column", "count", "order", "seed"]) $(`cv-${side}-${what}`).onchange = () => drawChoice(side);
  $(`cv-${side}`).oninput = () => {
    if (!chosen[side] || $(`cv-${side}`).value === chosen[side].shown) return;
    $("cv-status").textContent = `the box was edited: each line of ${side} is one prompt, and a ↵ is sent as it is`;
  };
}
$("cv-make").onclick = async () => {
  const model = $("cv-model").value, positive = sidePrompts("toward"), negative = sidePrompts("away");
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
