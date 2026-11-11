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
  const sent = r.byParent ? `1 new message, ${sizeOf(r.sent)}, and the id of the rest` : `the whole context, ${sizeOf(r.sent)}`;
  const record = r.record ? `<div class="stat">context ${recordChip(r.record)}</div>` : "";
  return `<div class="stat">the page sent ${sent} · the nagaya kept ${kept} of ${u.prompt} prompt tokens and ran ${
    u.fed}</div>` + record +
    r.others.map((s) => `<div class="stat">message ${s.message}: ${snapshotChip(s)}</div>`).join("");
}
// The record of the chat with its last reply, when the nagaya names one: {id, host, system} (gogi "Context
// records"). The next turn names it as its parent and sends only its new message. A new system message changes the
// first message, so that turn sends the chat whole.
let chatRecord = null;
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
$("chat-clear").onclick = () => { chat = []; chatRecord = null; drawChat(); $("chat-error").textContent = ""; };
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
  const system = $("chat-system").value.trim(), thinking = $("chat-thinking").checked && thinks(model);
  const control = controlOf(model, $("chat-vector").value, $("chat-op").value, $("chat-strength").value, layers);
  /** The chat as one context: each message of the page, with a boundary at each message of the person. */
  const whole = () => {
    const messages = system ? [{role: "system", content: system, snap: true}] : [];
    for (const m of chat) {
      const wire = {role: m.role, content: m.content};
      if (m.reasoning) wire.reasoning = m.reasoning;
      if (m.role === "user") wire.snap = true;
      messages.push(wire);
    }
    return control ? {thinking, control, messages} : {thinking, messages};
  };
  // The settings of this turn stand beside the parent and replace those of the record.
  let byParent = !!chatRecord && here(chatRecord.host) && chatRecord.system === system;
  const body = {model, max_tokens: +$("chat-max").value, sample: {temperature: +$("chat-temp").value},
                context: byParent ? {parent: chatRecord.id, thinking, control: control || [],
                                     messages: [{role: "user", content: text, snap: true}]} : whole()};
  const live = {role: "assistant", content: "", reasoning: "", open: true};
  $("chat-go").disabled = true; $("chat-stop").hidden = false;
  chatAbort = new AbortController();
  let tps = null;
  try {
    await ready(model, (t) => { live.stat = t; drawChat(live); });
    live.stat = "";
    const run = () => stream(body, chatAbort.signal, (name, d) => {
      if (name !== "token") return;
      if (d.phase === "reasoning") live.reasoning += d.piece; else if (d.phase === "content") live.content += d.piece;
      tps = d.tps ?? tps;
      live.stat = `${d.n} tokens${tps ? ` · ${tps.toFixed(1)} tok/s` : ""}`;
      drawChat(live);
    });
    let done;
    try {
      done = await run();
    } catch (e) {
      // The nagaya dropped the record: a 404 that names the parent. The page has the chat, so it sends it whole.
      if (!byParent || e.status !== 404 || e.param !== "context.parent") throw e;
      byParent = false; chatRecord = null; body.context = whole();
      done = await run();
    }
    const sent = utf8Length(JSON.stringify({...body, stream: true}));
    chatRecord = done.context ? {id: done.context, host, system} : null;
    const u = done.usage;
    // A snapshot names the index of its message in the whole context, where the system message is index 0.
    const first = system ? 1 : 0, others = [];
    for (const s of done.snapshots || []) {
      if (chat[s.message - first]) chat[s.message - first].snapshot = s; else others.push(s);
    }
    chat.push({role: "assistant", content: done.content, reasoning: done.reasoning,
               stat: `${u.completion} tokens${tps ? ` · ${tps.toFixed(1)} tok/s` : ""} · ${done.finish}`,
               receipt: {sent, usage: u, others, byParent, record: done.context}});
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
