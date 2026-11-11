"use strict";
// The Chat tab: a conversation with a text model, with thinking and one control vector.

// ---- chat
let chat = [], chatAbort = null;
/**
 * What one turn cost, as the end of the status line of its reply, and as the hover text of that line. The page sent
 * the chat whole, or the id of its record and the new message (gogi "Context records"). The nagaya started from the
 * longest snapshot that it holds, so it ran only the tokens after it. `others` are the snapshots of the turn that
 * belong to no message of the log, such as the one of the system message.
 */
function receiptOf(r) {
  const u = r.usage, kept = u.kept ?? 0;
  const parts = [`sent ${sizeOf(r.sent)} (${r.byParent ? "new message only" : "whole chat"})`,
                 `ran ${u.fed} of ${u.prompt} prompt tokens`];
  const hover = [r.byParent ? "The page sent the id of the record of the chat, and the new message."
                            : "The page sent the chat whole.",
                 `The nagaya kept ${kept} prompt tokens from its snapshots and ran ${u.fed}.`]
    .concat(["content", "reasoning"].filter((part) => r.stats?.[part]).map((part) => {
      const s = r.stats[part];
      return `The ${part}: mean entropy ${s.mean_entropy.toFixed(2)} nats, at most ${s.max_entropy.toFixed(2)}; ` +
        `mean log probability ${s.mean_logprob.toFixed(2)}, at least ${s.min_logprob.toFixed(2)}.`;
    }))
    .concat(r.others.map((s) => `Snapshot of message ${s.message}: ${s.id}, ${s.tokens} tokens, ${s.tier}` +
      (s.bytes == null ? "" : `, ${sizeOf(s.bytes)}`) + "."));
  return {text: parts.join(" · "), hover: hover.join(" "), record: r.record};
}
/** The one status line of a message: the counts of a reply, what its turn cost, and the id of what the nagaya holds. */
function statusOf(m) {
  const parts = m.stat ? [esc(m.stat)] : [];
  let hover = "";
  if (m.receipt) {
    const r = receiptOf(m.receipt);
    parts.push(esc(r.text));
    if (r.record) parts.push(recordChip(r.record));
    hover = r.hover;
  }
  if (m.snapshot) parts.push(snapshotChip(m.snapshot));
  return parts.length ? `<div class="stat" title="${esc(hover)}">${parts.join(" · ")}</div>` : "";
}
// The record of the chat with its last reply, when the nagaya names one: {id, host, system} (gogi "Context
// records"). The next turn names it as its parent and sends only its new message. A new system message changes the
// first message, so that turn sends the chat whole.
let chatRecord = null;
function drawChat(live) {
  const all = live ? chat.concat([live]) : chat;
  $("chat-log").innerHTML = all.map((m, i) => {
    const think = m.reasoning ? `<details ${m.open ? "open" : ""}><summary>thinking</summary>${esc(m.reasoning)}</details>` : "";
    return `<div class="msg ${m.role}">${think}${esc(m.content || "")}${statusOf(m)}${sendButtons(m, i)}</div>`;
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
  // An empty field leaves its member out, so the nagaya does not cut the choice of tokens by it.
  const sample = {temperature: +$("chat-temp").value};
  if ($("chat-topk").value !== "") sample.top_k = +$("chat-topk").value;
  if ($("chat-topp").value !== "") sample.top_p = +$("chat-topp").value;
  const body = {model, max_tokens: +$("chat-max").value, sample,
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
               receipt: {sent, usage: u, others, byParent, record: done.context, stats: done.stats}});
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
// ---- a message of the chat as speech or as an image
// The person chooses a speech model and an image model. Then each message has a button for each. A job runs beside
// the chat: the chat goes on, and the clips or the image go into the tray under it when they are made. The tray
// keeps its elements, so a new message does not stop a clip that plays.

/** A model that speaks: a speech model, or a model that speaks beside its text, such as Omni (spec 11.8a, 11.11). */
const speaks = (m) => kindOf(m) === "speech" ||
  (omnis(m) && (m.media?.outputs || []).some((o) => o.startsWith("audio/")));
function fillSendTo() {
  for (const [id, accepts] of [["chat-speak", speaks], ["chat-draw", (m) => kindOf(m) === "image"]]) {
    const sel = $(id), keep = sel.value || recall(`nagaya-${id}`) || "";
    const list = models.filter(accepts);
    const html = `<option value="">off</option>` +
      list.map((m) => `<option value="${esc(m.model)}">${esc(m.model)}</option>`).join("");
    if (sel.dataset.html === html) continue;
    sel.innerHTML = html; sel.dataset.html = html;
    sel.value = list.some((m) => m.model === keep) ? keep : "";
  }
}
onRefresh(fillSendTo);
for (const id of ["chat-speak", "chat-draw"]) {
  $(id).onchange = () => { remember(`nagaya-${id}`, $(id).value); drawChat(); };
}
/** The buttons of a message with text: one for each kind of model that the person chose. */
function sendButtons(m, i) {
  if (!m.content || m.open) return "";
  const speak = $("chat-speak").value, draw = $("chat-draw").value;
  const buttons = (speak ? `<button class="plain small" data-chat-speak="${i}" title="speak with ${esc(speak)}">speak</button>` : "") +
    (draw ? ` <button class="plain small" data-chat-draw="${i}" title="draw with ${esc(draw)}">draw</button>` : "");
  return buttons ? `<div class="stat">${buttons}</div>` : "";
}
let chatJobs = 0;
/** The objects of the jobs of the tray, for the keep of the page. */
let chatMade = [];
keeps(() => chatMade);
/** Add a job to the tray, and give the way to change its status and to add to its results. */
function trayJob(what, i, model) {
  const n = chatJobs++;
  $("chat-tray").insertAdjacentHTML("beforeend", `<div class="job"><div class="note">${what} of message ${i + 1} · ${
    esc(model)} · <span id="chat-job-${n}-status"></span></div><div id="chat-job-${n}-clips" class="clips"></div></div>`);
  return {status: (text) => { $(`chat-job-${n}-status`).textContent = text; },
          add: (html) => { $(`chat-job-${n}-clips`).insertAdjacentHTML("beforeend", html); },
          clips: () => $(`chat-job-${n}-clips`)};
}
/** Speak message `i` with the chosen model. A long text goes in chunks of whole sentences, and each clip is added
 *  when it is made, so the first one can play while the next ones are made. */
async function chatSpeak(i) {
  const m = chat[i], model = $("chat-speak").value;
  if (!m?.content || !model) return;
  const job = trayJob("speech", i, model);
  try {
    await ready(model, job.status);
    const speech = kindOf(models.find((x) => x.model === model)) === "speech";
    const chunks = speechChunks(m.content);
    let ms = 0, clips = 0;
    for (const [n, chunk] of chunks.entries()) {
      job.status(`speaking part ${n + 1} of ${chunks.length}…`);
      // A speech model takes the voice of the Speech tab. A model that speaks beside its text takes the prompt of its
      // speech mode.
      const body = speech
        ? {model, max_tokens: 1, context: {thinking: false, messages: [{role: "system", content: $("sp-voice").value},
                                                                        {role: "user", content: chunk}]},
           output: {mime: "audio/wav", options: {language: $("sp-lang").value || "auto"}}}
        : {model, max_tokens: 512, sample: {temperature: 0.3},
           context: {thinking: false, messages: [{role: "system", content: OMNI_SYSTEM.speech},
                                                 {role: "user", content: chunk}]},
           output: {mime: "audio/wav", options: {mode: "speech"}}};
      const done = await call("POST", "/council/v2/generations", body);
      for (const o of (done.outputs || []).filter((o) => o.mime === "audio/wav")) {
        chatMade = chatMade.concat([{id: o.id, host}]).slice(-200);
        job.add(`<audio controls src="${esc(objectUrl(o.id))}"></audio>`);
        if (clips++ === 0) job.clips().querySelector?.("audio")?.play?.().catch(() => {});
      }
      ms += done.usage?.media?.written?.audio_ms || 0;
    }
    job.status(`${clips} clip${clips === 1 ? "" : "s"}${ms ? ` · ${(ms / 1000).toFixed(1)} s of speech` : ""}`);
    keepAll();
  } catch (e) {
    job.status(e.message);
  }
}
/** Draw message `i` with the chosen model: the text of the message is the prompt. */
async function chatDraw(i) {
  const m = chat[i], model = $("chat-draw").value;
  if (!m?.content || !model) return;
  const job = trayJob("image", i, model);
  const options = quickImageOptions(model);
  const body = {model, max_tokens: 1, context: {thinking: false, messages: [{role: "user", content: m.content}]},
                output: {mime: "image/png", options}};
  const started = performance.now();
  try {
    await ready(model, job.status);
    job.status("waiting…");
    const done = await stream(body, undefined, (name, d) => {
      if (name === "progress") job.status(`step ${d.step} of ${d.of}`);
    });
    const made = done.outputs.map((o) => ({id: o.id, host, model, prompt: m.content, seed: done.seed + o.index,
                                           size: `${options.width}x${options.height}`, steps: options.steps}));
    job.add(`<div class="tiles">${made.map(tile).join("")}</div>`);
    gallery = made.concat(gallery); saveGallery(); drawGallery(); keepAll();
    job.status(`seed ${done.seed} · ${((performance.now() - started) / 1000).toFixed(1)} s`);
  } catch (e) {
    job.status(e.message);
  }
}
document.addEventListener("click", (ev) => {
  const {chatSpeak: speak, chatDraw: draw} = ev.target.dataset || {};
  if (speak !== undefined) return chatSpeak(+speak);
  if (draw !== undefined) return chatDraw(+draw);
});

modelSelect("chat-model", (m) => kindOf(m) === "text");
