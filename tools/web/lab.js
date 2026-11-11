"use strict";
// The Trace tab: `project` and `trace` of the lab routes, with their charts (docs/lab.md).

// ---- lab
/** Whether the lab routes serve a served model: its engine on this nagaya has the kind mk-lfm3-bmg (docs/lab.md). A
 *  model of a peer has no engine in this profile, and the lab routes do not forward. */
const labs = (m) => engines.some((e) => e.name === engineNames[m.model] && e.kind === "mk-lfm3-bmg");
/** The saved vector of the lab form, or undefined. */
const labVector = () => vectors.find((v) => v.id === $("lab-vector").value && v.model === $("lab-model").value &&
                                            here(v.host));
/** The recent trace objects, newest first. The keep holds them, because the route sets no keep (docs/lab.md). */
let traces = recallJson("nagaya-traces", []);
const saveTraces = () => rememberJson("nagaya-traces", traces);

$("lab-project").onclick = async () => {
  const v = labVector();
  $("lab-error").textContent = "";
  if (!v) { $("lab-error").textContent = "choose a saved vector of this model"; return; }
  $("lab-lens").innerHTML = `<p class="note">projecting ${esc(v.name)}…</p>`;
  try {
    const r = await call("POST", "/nagaya/v1/lab/project",
                         {model: v.model, vector: v.id, through: $("lab-through").value || "unembed", top_k: +$("lab-lens-k").value || 8});
    const words = (list) => list.map((t) => `<code>${esc(t.token ?? `#${t.id}`)}</code>`).join(" ");
    const through = r.through === "jlens" ? "the J-Lens" : "the unembedding";
    $("lab-lens").innerHTML = `<h2>${esc(v.name)} through ${through}</h2><table class="lens"><tr><th>layer</th><th>up</th>
      <th>down</th><th>end rank</th></tr>` + r.layers.map((l) => `<tr><td>${l.layer ?? ""}</td><td>${words(l.up)}</td>
      <td>${words(l.down)}</td><td>${l.end.map((e) => `${esc(e.token ?? e.id)} ${e.rank}`).join(", ")}</td></tr>`)
      .join("") + "</table>";
  } catch (e) {
    $("lab-lens").innerHTML = ""; $("lab-error").textContent = e.message;
  }
};

/** The tensors of a canonical safetensors file (D10): {meta, <name>: {shape, data}}. The page reads F32 and U32. */
function safetensors(buffer) {
  const view = new DataView(buffer);
  const n = view.getUint32(0, true) + view.getUint32(4, true) * 2 ** 32;
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, n)));
  const out = {meta: header.__metadata__ || {}};
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const Type = {F32: Float32Array, U32: Uint32Array}[t.dtype];
    if (!Type) throw new Error(`the trace holds ${name} of the type ${t.dtype}; the page reads only F32 and U32`);
    const [a, b] = t.data_offsets;
    out[name] = {shape: t.shape, data: new Type(buffer.slice(8 + n + a, 8 + n + b))};
  }
  return out;
}
/** The trace object with id `id`, from the store of this nagaya. */
async function traceObject(id) {
  const r = await fetch(base() + `/council/v2/objects/${id}`);
  if (!r.ok) throw new Error(`the read of the trace ${id} returned ${r.status}; the store may have evicted it`);
  return safetensors(await r.arrayBuffer());
}

// The text of each token id, from the detokenize route. A key holds a promise, so two asks for one id send one request.
const pieces = new Map();
function pieceOf(model, id) {
  const key = `${host}\n${model}\n${id}`;
  if (!pieces.has(key)) {
    pieces.set(key, call("POST", "/nagaya/v1/detokenize", {model, tokens: [id]})
      .then((r) => r.text, () => { pieces.delete(key); return `#${id}`; }));
  }
  return pieces.get(key);
}
const piecesOf = (model, ids) => Promise.all(Array.from(ids, (id) => pieceOf(model, id)));
/** A piece as the page shows it: a newline as ↵, so that each token stays on its line. */
const shown = (text) => esc(text.replace(/\n/g, "↵"));

// Each run of the last trace: {name, cls, model, done, t, pieces}. `labAt` is the chosen step.
let labRuns = [], labAt = 0;

$("lab-trace").onclick = async () => {
  const model = $("lab-model").value, text = $("lab-prompt").value.trim(), v = labVector();
  const status = (t) => { $("lab-status").textContent = t; };
  $("lab-error").textContent = "";
  if (!model) { $("lab-error").textContent = "this nagaya serves no model of the lab routes"; return; }
  if (!text) { $("lab-error").textContent = "write a prompt"; return; }
  const trace = {top_k: +$("lab-k").value || 8};
  if (v) trace.vectors = [v.id];
  if ($("lab-use-lens").value) trace.lens = $("lab-use-lens").value;
  // One seed for both runs. A seed past 2^53 does not survive a JavaScript number.
  const sample = {temperature: +$("lab-temp").value, seed: Math.floor(Math.random() * 2 ** 53)};
  const control = v && controlOf(model, v.id, $("lab-op").value, $("lab-strength").value);
  const runs = [["plain", "s1", undefined]].concat(control ? [["steered", "s2", control]] : []);
  $("lab-trace").disabled = true;
  try {
    await ready(model, status);
    labRuns = [];
    for (const [name, cls, list] of runs) {
      const context = plainContext(text);
      if (list) context.control = list;
      status(`tracing ${name}…`);
      const done = await call("POST", "/nagaya/v1/lab/trace", {model, max_tokens: +$("lab-max").value, sample, context,
                                                               trace});
      const t = await traceObject(done.trace);
      labRuns.push({name, cls, model, done, t, pieces: await piecesOf(model, t.tokens.data)});
      traces = [{id: done.trace, host, model, name}].concat(traces.filter((x) => x.id !== done.trace)).slice(0, 40);
    }
    saveTraces(); keepAll();
    labAt = 0;
    status(`seed ${sample.seed}`);
    drawLab();
    await labStep(0);
  } catch (e) {
    $("lab-error").textContent = e.message; status("");
  }
  $("lab-trace").disabled = false;
};

/** The log probability of any end token at each step: the log of the sum over the E end tokens. */
function endOf(t) {
  const [s, e] = t.end_logprob.shape, d = t.end_logprob.data, out = new Float32Array(s);
  for (let i = 0; i < s; i++) {
    let top = -Infinity;
    for (let j = 0; j < e; j++) top = Math.max(top, d[i * e + j]);
    let sum = 0;
    for (let j = 0; j < e; j++) sum += Math.exp(d[i * e + j] - top);
    out[i] = top + Math.log(sum);
  }
  return out;
}
const fmt = (x) => Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(2);

/** An SVG line chart of one measure over the steps, with one line for each run and a hit column for each step. */
function lineChart(title, series) {
  const W = 640, H = 150, L = 40, R = 70, T = 8, B = 20;
  const n = Math.max(...series.map((s) => s.values.length));
  const finite = series.flatMap((s) => Array.from(s.values).filter(Number.isFinite));
  if (!n || !finite.length) return "";
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (hi === lo) { lo -= 1; hi += 1; }
  const x = (i) => L + (n > 1 ? i / (n - 1) : 0.5) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
  for (const v of [lo, (lo + hi) / 2, hi]) {
    svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/>` +
      `<text x="${L - 4}" y="${y(v) + 4}" text-anchor="end">${fmt(v)}</text>`;
  }
  svg += `<text x="${L}" y="${H - 4}">step 0</text><text x="${W - R}" y="${H - 4}" text-anchor="end">${n - 1}</text>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(title)}</text>`;
  if (labAt < n) svg += `<line class="now" x1="${x(labAt)}" x2="${x(labAt)}" y1="${T}" y2="${H - B}"/>`;
  const labels = [];
  for (const s of series) {
    // A value that is not finite breaks the line, so each part is its own polyline.
    let part = [];
    const flush = () => {
      if (part.length) svg += `<polyline class="${s.cls}" points="${part.join(" ")}"/>`;
      part = [];
    };
    s.values.forEach((v, i) => Number.isFinite(v) ? part.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`) : flush());
    flush();
    const last = s.values.length - 1;
    if (Number.isFinite(s.values[last])) labels.push({name: s.name, x: x(last) + 6, y: y(s.values[last]) + 4});
  }
  // The name of each line goes at its end. Two names stay 12 units apart, so that they do not overlap.
  labels.sort((a, b) => a.y - b.y).forEach((label, i) => {
    if (i > 0) label.y = Math.max(label.y, labels[i - 1].y + 12);
    svg += `<text x="${label.x}" y="${label.y}">${esc(label.name)}</text>`;
  });
  const w = (W - L - R) / Math.max(n - 1, 1);
  for (let i = 0; i < n; i++) {
    const values = series.map((s) => `${s.name} ${Number.isFinite(s.values[i]) ? fmt(s.values[i]) : "—"}`);
    svg += `<rect class="hit" data-lab-step="${i}" x="${x(i) - w / 2}" y="${T}" width="${w}" height="${H - T - B}">` +
      `<title>step ${i}: ${esc(values.join(", "))}</title></rect>`;
  }
  return svg + "</svg>";
}

/** The places on the layer axis of a trace where vector 0 has a direction: the layers whose `along` is a number. */
function heldLayers(t) {
  const [n, l, nv] = t.along.shape;
  return Array.from({length: l}, (_, k) => k).filter((k) =>
    Array.from({length: n}, (_, i) => t.along.data[(i * l + k) * nv]).some(Number.isFinite));
}

/** An SVG heat map of `along` [S, L, V] for vector 0 of one run: a row for each layer that the vector holds, the last
 *  layer on top. The color is blue for a negative value and red for a positive one, over a gray middle. `top` is the
 *  largest size that both runs share, so that one color means one value. */
function alongMap(run, top) {
  const a = run.t.along, layers = run.t.layers.data, held = heldLayers(run.t);
  const [n, l, nv] = a.shape, W = 640, L = 40, R = 70, h = 9, rows = held.length, H = rows * h + 6;
  const w = (W - L - R) / n;
  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="along, ${esc(run.name)}">`;
  held.forEach((k, r) => {
    const yk = (rows - 1 - r) * h;
    svg += `<rect x="${L}" y="${yk}" width="${W - L - R}" height="${h - 1}" fill="var(--mid)"/>`;
    if (r % 4 === 0 || r === rows - 1) svg += `<text x="${L - 4}" y="${yk + h - 1}" text-anchor="end">L${layers[k]}</text>`;
    for (let i = 0; i < n; i++) {
      const v = a.data[(i * l + k) * nv];
      if (!Number.isFinite(v) || v === 0) continue;
      svg += `<rect x="${(L + i * w).toFixed(1)}" y="${yk}" width="${(w + 0.3).toFixed(1)}" height="${h - 1}" ` +
        `fill="var(${v > 0 ? "--pos" : "--neg"})" fill-opacity="${Math.min(Math.abs(v) / top, 1).toFixed(3)}"/>`;
    }
  });
  if (labAt < n) svg += `<line class="now" x1="${L + (labAt + 0.5) * w}" x2="${L + (labAt + 0.5) * w}" y1="0" y2="${rows * h}"/>`;
  svg += `<text x="${W - R + 6}" y="${h}">${esc(run.name)}</text>`;
  for (let i = 0; i < n; i++) {
    svg += `<rect class="hit" data-lab-step="${i}" x="${L + i * w}" y="0" width="${w}" height="${rows * h}">` +
      `<title>step ${i}</title></rect>`;
  }
  return svg + "</svg>";
}

function drawLab() {
  $("lab-replies").innerHTML = labRuns.map((run, r) => {
    const tokens = run.pieces.map((p, i) => `<button class="tok${i === labAt ? " on" : ""}" data-lab-step="${i}" ` +
      `title="step ${i}, entropy ${fmt(run.t.entropy.data[i])} nats">${shown(p)}</button>`).join("");
    const id = run.done.trace;
    return `<h2>${esc(run.name)}</h2><div class="msg assistant lab-reply">${tokens}</div>
      <div class="note">${run.pieces.length} steps, ${esc(run.done.finish)} · <a href="${base()}/council/v2/objects/${
        esc(id)}" download="trace-${esc(run.name)}.safetensors" title="${esc(id)}">trace object</a></div>`;
  }).join("");
  if (!labRuns.length) { $("lab-charts").innerHTML = ""; return; }
  const legend = `<div class="legend">` + labRuns.map((run) =>
    `<span><i class="${run.cls}"></i>${esc(run.name)}</span>`).join("") + "</div>";
  let html = `<h2>Each step</h2>` + legend +
    lineChart("entropy, nats", labRuns.map((run) => ({name: run.name, cls: run.cls, values: run.t.entropy.data}))) +
    lineChart("log P(end token)", labRuns.map((run) => ({name: run.name, cls: run.cls, values: endOf(run.t)})));
  const withAlong = labRuns.filter((run) => run.t.along);
  if (withAlong.length) {
    let top = 0;
    for (const run of withAlong) for (const v of run.t.along.data) if (Number.isFinite(v)) top = Math.max(top, Math.abs(v));
    const t = withAlong[0].t, held = new Set(heldLayers(t));
    const none = Array.from(t.layers.data).filter((_, k) => !held.has(k));
    html += `<h2>along the vector, each layer <span class="note">— h · v̂; blue below 0, red above, up to ${
      fmt(top || 1)}${none.length ? `; the vector holds no direction at the layers ${none.join(", ")}` : ""}</span></h2>` +
      withAlong.map((run) => alongMap(run, top || 1)).join("");
  }
  $("lab-charts").innerHTML = html;
}

/** Choose step `i`, and show its distribution and each layer of each run. */
async function labStep(i) {
  labAt = i;
  drawLab();
  const parts = [];
  for (const run of labRuns) {
    const t = run.t, n = t.tokens.shape[0];
    if (i >= n) { parts.push(`<div><h2>${esc(run.name)}</h2><p class="note">no step ${i}</p></div>`); continue; }
    const k = t.top_ids.shape[1], l = t.layers.shape[0];
    const top = Array.from(t.top_ids.data.slice(i * k, (i + 1) * k));
    const lens = t.lens_ids ? Array.from(t.lens_ids.data.slice(i * l * k, (i + 1) * l * k)) : [];
    const text = await piecesOf(run.model, top.concat(lens));
    const word = (j, lp) => `<code>${shown(text[j])}</code> <span class="note">${(Math.exp(lp) * 100).toFixed(1)}%</span>`;
    const tops = top.map((_, j) => word(j, t.top_logprobs.data[i * k + j])).join(" ");
    let rows = "";
    for (let m = l - 1; m >= 0; m--) {
      const along = t.along ? t.along.data[(i * l + m) * t.along.shape[2]] : NaN;
      const lensCells = t.lens_ids ? Array.from({length: k}, (_, j) =>
        word(k + m * k + j, t.lens_logprobs.data[(i * l + m) * k + j])).join(" ") : "";
      rows += `<tr><td>${t.layers.data[m]}</td><td>${fmt(t.norms.data[i * l + m])}</td>
        <td>${Number.isFinite(along) ? fmt(along) : "—"}</td><td>${lensCells}</td></tr>`;
    }
    parts.push(`<div><h2>${esc(run.name)}, step ${i}: <code>${shown(run.pieces[i])}</code></h2>
      <p class="note">entropy ${fmt(t.entropy.data[i])} nats · log P(end token) ${fmt(endOf(t)[i])}</p>
      <div>${tops}</div><table class="lens"><tr><th>layer</th><th>norm</th><th>along</th><th>lens</th></tr>${rows}</table>
      </div>`);
  }
  $("lab-step").innerHTML = `<div class="pair">${parts.join("")}</div>`;
}
document.addEventListener("click", (ev) => {
  const step = ev.target.dataset?.labStep ?? ev.target.closest?.("[data-lab-step]")?.dataset.labStep;
  if (step !== undefined) labStep(+step);
});
modelSelect("lab-model", labs);
keeps(() => traces);
