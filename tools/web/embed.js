"use strict";
// The Embeddings tab: rank documents against a query, or compare them with each other (spec 11.10).

function fillEmbed() {
  const e = models.find((x) => x.model === $("em-model").value)?.embeddings;
  const fill = (id, list, want) => {
    const html = (list || []).map((t) => `<option>${esc(t)}</option>`).join("");
    if ($(id).dataset.html === html) return;
    $(id).innerHTML = html; $(id).dataset.html = html;
    if (list?.includes(want)) $(id).value = want;
  };
  fill("em-qtask", e?.tasks, "query"); fill("em-dtask", e?.tasks, "document");
  fill("em-dims", e?.dimensions, e?.dimensions?.[0]);
}
$("em-model").addEventListener("change", fillEmbed);

/** The vectors of the inputs, in their order, and the count of their tokens. */
async function embed(model, task, inputs) {
  const r = await call("POST", "/council/v2/embeddings", {model, task, inputs, dimensions: +$("em-dims").value});
  const vectors = [];
  for (const x of r.embeddings) vectors[x.index] = x.vector;
  return {vectors, tokens: r.embeddings.reduce((n, x) => n + x.tokens, 0)};
}
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

$("em-go").onclick = async () => {
  const model = $("em-model").value, query = $("em-query").value.trim();
  const docs = $("em-docs").value.split("\n").map((t) => t.trim()).filter(Boolean);
  const status = (t) => { $("em-status").textContent = t; };
  const most = models.find((x) => x.model === model)?.embeddings?.inputs_per_request || 64;
  $("em-error").textContent = "";
  if (!model) { $("em-error").textContent = "the nagaya serves no embedding model"; return; }
  if (!docs.length) { $("em-error").textContent = "write one document or more"; return; }
  if (docs.length > most) { $("em-error").textContent = `a request takes at most ${most} inputs`; return; }
  $("em-go").disabled = true;
  try {
    await ready(model, status);
    status("embedding…");
    const started = performance.now();
    const d = await embed(model, $("em-dtask").value, docs);
    if (query) {
      const q = await embed(model, $("em-qtask").value, [query]);
      const ranked = docs.map((text, i) => ({text, score: dot(q.vectors[0], d.vectors[i])}))
        .sort((a, b) => b.score - a.score);
      $("em-out").innerHTML = `<h2>Documents by cosine to the query</h2>` + ranked.map((r) =>
        `<div class="score"><b>${r.score.toFixed(3)}</b><div class="bar"><div style="width:${
          Math.max(0, 100 * r.score).toFixed(1)}%"></div></div><span>${esc(r.text)}</span></div>`).join("");
      status(`${docs.length + 1} inputs, ${d.tokens + q.tokens} tokens, ${d.vectors[0].length} dimensions, ${
        ((performance.now() - started) / 1000).toFixed(2)} s`);
    } else {
      const head = docs.map((t, i) => `<th title="${esc(t)}">${i + 1}</th>`).join("");
      const rows = docs.map((t, i) => `<tr><th title="${esc(t)}">${i + 1}. ${esc(t)}</th>${docs.map((_, j) => {
        const c = dot(d.vectors[i], d.vectors[j]);
        return `<td style="background:color-mix(in srgb, var(--accent) ${Math.max(0, 100 * c).toFixed(0)}%, transparent)">${
          c.toFixed(2)}</td>`;
      }).join("")}</tr>`).join("");
      $("em-out").innerHTML = `<h2>Cosine between the documents</h2><table class="matrix"><tr><th></th>${head}</tr>${rows}</table>`;
      status(`${docs.length} inputs, ${d.tokens} tokens, ${d.vectors[0].length} dimensions, ${
        ((performance.now() - started) / 1000).toFixed(2)} s`);
    }
  } catch (e) {
    $("em-error").textContent = e.message; status("");
  }
  $("em-go").disabled = false;
};
modelSelect("em-model", (m) => kindOf(m) === "embed");
onRefresh(fillEmbed);
