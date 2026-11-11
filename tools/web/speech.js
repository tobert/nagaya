"use strict";
// The Speech tab: lines in a designed voice (spec 11.8a).

let lines = [];
function drawLines() {
  $("sp-lines").innerHTML = lines.map((l) =>
    `<div class="line"><div><div>${esc(l.text)}</div><audio controls src="${objectUrl(l.id, l.host)}"
       onerror="this.nextElementSibling.textContent = 'the browser cannot play this audio'"></audio>
       <div class="note">${l.ms == null ? "" : `${(l.ms / 1000).toFixed(1)} s`}</div></div>
     <a class="note" href="${objectUrl(l.id, l.host)}" download="line.wav">wav</a></div>`).join("") ||
    `<p class="note">No lines yet.</p>`;
}
drawLines();
$("sp-clear").onclick = () => { lines = []; drawLines(); $("sp-error").textContent = ""; };
$("sp-input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("sp-go").click(); } };
$("sp-go").onclick = async () => {
  const text = $("sp-input").value.trim(), model = $("sp-model").value;
  if (!text || !model) return;
  $("sp-error").textContent = "";
  const messages = [{role: "system", content: $("sp-voice").value, snap: true}];
  if ($("sp-keep").checked) {
    for (const l of lines.filter((x) => here(x.host))) {
      messages.push({role: "user", content: l.text, snap: true});
      messages.push({role: "assistant", content: [{type: "object", id: l.id, mime: "audio/wav"}], snap: true});
    }
  }
  messages.push({role: "user", content: text});
  const body = {model, max_tokens: 1, context: {thinking: false, messages},
                output: {mime: "audio/wav", options: {language: $("sp-lang").value}}};
  // An empty field leaves the member out, so the nagaya uses the sampler of the checkpoint (spec 11.8).
  const sample = {};
  if ($("sp-seed").value !== "") sample.seed = +$("sp-seed").value;
  if ($("sp-temp").value !== "") sample.temperature = +$("sp-temp").value;
  if ($("sp-topk").value !== "") sample.top_k = +$("sp-topk").value;
  if (Object.keys(sample).length) body.sample = sample;
  $("sp-go").disabled = true;
  try {
    await ready(model, (t) => { $("sp-error").textContent = t; });
    $("sp-error").textContent = "speaking…";
    const done = await call("POST", "/council/v2/generations", body);
    lines.push({text, host, id: done.outputs[0].id, ms: done.usage?.media?.written?.audio_ms}); $("sp-input").value = ""; drawLines();
    $("sp-lines").querySelector(".line:last-child audio")?.play().catch(() => {});
    $("sp-error").textContent = "";
  } catch (e) {
    $("sp-error").textContent = e.message;
  }
  $("sp-go").disabled = false;
};
modelSelect("sp-model", (m) => kindOf(m) === "speech");
