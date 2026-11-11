"use strict";
// The Chat tab: a conversation with a text model, with thinking and one control vector.

// ---- chat
let chat = [], chatAbort = null;
/**
 * What one turn cost, under its reply. The page sent the whole context as text (gogi "Design"). The nagaya started
 * from the longest snapshot that it holds, so it ran only the tokens after it. `others` are the snapshots of the
 * turn that belong to no message of the log, such as the one of the system message.
 */
function receiptOf(r) {
  if (!r) return "";
  const u = r.usage, kept = u.kept ?? 0;
  return `<div class="stat">the page sent the whole context, ${sizeOf(r.sent)} · the nagaya kept ${kept} of ${
    u.prompt} prompt tokens and ran ${u.fed}</div>` +
    r.others.map((s) => `<div class="stat">message ${s.message}: ${snapshotChip(s)}</div>`).join("");
}
function drawChat(live) {
  const all = live ? chat.concat([live]) : chat;
  $("chat-log").innerHTML = all.map((m) => {
    const think = m.reasoning ? `<details ${m.open ? "open" : ""}><summary>thinking</summary>${esc(m.reasoning)}</details>` : "";
    const stat = m.stat ? `<div class="stat">${esc(m.stat)}</div>` : "";
    const held = m.snapshot ? `<div class="stat">${snapshotChip(m.snapshot)}</div>` : "";
    return `<div class="msg ${m.role}">${think}${esc(m.content || "")}${stat}${held}${receiptOf(m.receipt)}</div>`;
  }).join("");
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}
$("chat-clear").onclick = () => { chat = []; drawChat(); $("chat-error").textContent = ""; };
$("chat-stop").onclick = () => chatAbort?.abort();
$("chat-input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-go").click(); } };
$("chat-go").onclick = async () => {
  const text = $("chat-input").value.trim(), model = $("chat-model").value;
  if (!text || !model) return;
  $("chat-error").textContent = "";
  let layers;
  try {
    layers = parseLayers($("chat-layers").value);
  } catch (e) {
    $("chat-error").textContent = `Layers: ${e.message}`;
    return;
  }
  chat.push({role: "user", content: text}); $("chat-input").value = ""; drawChat();
  const messages = [];
  if ($("chat-system").value.trim()) messages.push({role: "system", content: $("chat-system").value.trim(), snap: true});
  for (const m of chat) {
    const wire = {role: m.role, content: m.content};
    if (m.reasoning) wire.reasoning = m.reasoning;
    if (m.role === "user") wire.snap = true;
    messages.push(wire);
  }
  const body = {model, max_tokens: +$("chat-max").value, sample: {temperature: +$("chat-temp").value},
                context: {thinking: $("chat-thinking").checked && thinks(model), messages}};
  const control = controlOf(model, $("chat-vector").value, $("chat-op").value, $("chat-strength").value, layers);
  if (control) body.context.control = control;
  const sent = utf8Length(JSON.stringify({...body, stream: true}));
  const live = {role: "assistant", content: "", reasoning: "", open: true};
  $("chat-go").disabled = true; $("chat-stop").hidden = false;
  chatAbort = new AbortController();
  let tps = null;
  try {
    await ready(model, (t) => { live.stat = t; drawChat(live); });
    live.stat = "";
    const done = await stream(body, chatAbort.signal, (name, d) => {
      if (name !== "token") return;
      if (d.phase === "reasoning") live.reasoning += d.piece; else if (d.phase === "content") live.content += d.piece;
      tps = d.tps ?? tps;
      live.stat = `${d.n} tokens${tps ? ` · ${tps.toFixed(1)} tok/s` : ""}`;
      drawChat(live);
    });
    const u = done.usage;
    // A snapshot names the index of its message in the request, where the system message is index 0.
    const first = messages.length - chat.length, others = [];
    for (const s of done.snapshots || []) {
      if (chat[s.message - first]) chat[s.message - first].snapshot = s; else others.push(s);
    }
    chat.push({role: "assistant", content: done.content, reasoning: done.reasoning,
               stat: `${u.completion} tokens${tps ? ` · ${tps.toFixed(1)} tok/s` : ""} · ${done.finish}`,
               receipt: {sent, usage: u, others}});
  } catch (e) {
    // The turn did not finish, so the message goes back to the input, and the chat stays as it was.
    chat.pop(); $("chat-input").value = text;
    $("chat-error").textContent = e.name === "AbortError" ? "stopped; the partial reply is not kept" : e.message;
  }
  drawChat();
  $("chat-go").disabled = false; $("chat-stop").hidden = true;
};
/** The note under the Layers field: what the field means, or its mistake. */
function layersNote() {
  try {
    const layers = parseLayers($("chat-layers").value);
    if (!layers) return "— every layer of the vector";
    const keys = Object.keys(layers);
    const numbers = [...new Set(Object.values(layers))];
    const shape = numbers.length === 1 ? `×${numbers[0]} each` : "with their own numbers";
    return `— ${keys.length} layers (${keys[0]} to ${keys[keys.length - 1]}), ${shape}`;
  } catch (e) {
    return `— ${e.message}`;
  }
}
$("chat-layers").oninput = () => { $("chat-layers-note").textContent = layersNote(); };
for (const [id, text] of [["chat-layers-all", ""], ["chat-layers-early", "9-14"], ["chat-layers-late", "15-22"]]) {
  $(id).onclick = () => { $("chat-layers").value = text; $("chat-layers").oninput(); };
}
modelSelect("chat-model", (m) => kindOf(m) === "text");
