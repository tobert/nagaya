"use strict";
// The Chat tab: a conversation with a model whose replies hold text. The form shows only the controls that the served
// model declares in its identity: thinking, files, a spoken reply, one control vector, a layer loop and cycles.

// ---- the controls of the served model
/** The served model of the tab, or undefined. */
const chatModel = () => models.find((m) => m.model === $("chat-model").value);
const declares = (m, capability) => !!m?.capabilities?.includes(capability);
/** Whether a list of media types, each `type/subtype` or `type/*`, covers `mime` (gogi "Identity"). */
const reads = (inputs, mime) => !!mime && inputs.some((t) => t === mime || (t.endsWith("/*") && mime.startsWith(t.slice(0, -1))));
/** A model that speaks beside its text, such as LFM3-Omni-2B (spec 11.11). */
const talks = (m) => !!m && kindOf(m) === "text" && (m.media?.outputs || []).some((o) => o.startsWith("audio/"));
/** Fill a select with options of `values`, and keep its choice. A choice that is gone becomes the first value. */
function fillOptions(sel, values, labels = values) {
  const html = values.map((v, i) => `<option value="${esc(v)}">${esc(labels[i])}</option>`).join("");
  if (sel.dataset.html === html) return;
  const keep = sel.value;
  sel.innerHTML = html; sel.dataset.html = html;
  sel.value = values.includes(keep) ? keep : values[0];
}
/** The note beside the layer loop: the extra runs of the span, and the most that the model runs. */
function loopNote() {
  const limit = chatModel()?.layer_loops;
  if (!limit || $("chat-loop-start").value === "") return "— off";
  const extra = (+$("chat-loop-end").value - +$("chat-loop-start").value) * (+$("chat-loop-count").value - 1);
  return `— ${extra} of ${limit.extra_runs} extra runs`;
}
for (const id of ["chat-loop-start", "chat-loop-end", "chat-loop-count"]) {
  $(id).oninput = $(id).onchange = () => { $("chat-loop-note").textContent = loopNote(); };
}
/** Show the controls that the chosen model declares, and set their bounds from its identity. */
function drawChatForm() {
  const m = chatModel(), inputs = m?.media?.inputs || [];
  $("chat-thinking-box").hidden = !declares(m, "thinking");
  $("chat-steer-box").hidden = !declares(m, "control_vectors");
  $("chat-attach").hidden = !inputs.length;
  $("chat-file").accept = inputs.join(",");
  $("chat-output-box").hidden = !talks(m);
  const loops = declares(m, "layer_loops") ? m.layer_loops : null;
  $("chat-loop-box").hidden = !loops;
  if (loops) {
    // A span starts at a bound below the last one, and it ends at a bound above the first one.
    const bounds = (loops.bounds || []).map(String);
    fillOptions($("chat-loop-start"), [""].concat(bounds.slice(0, -1)), ["off"].concat(bounds.slice(0, -1)));
    fillOptions($("chat-loop-end"), bounds.slice(1));
    $("chat-loop-count").max = loops.count;
    $("chat-loop-note").textContent = loopNote();
  }
  const cycles = declares(m, "cycles") ? m.cycles : null;
  $("chat-cycles-box").hidden = !cycles;
  if (cycles) {
    $("chat-cycles").min = cycles.min; $("chat-cycles").max = cycles.max;
    $("chat-cycles").placeholder = `${cycles.default}, the count of the model`;
  }
}
/** The `loop` of a context: only for a model with `layer_loops`, and only when the person chose a start. */
function loopOf(m) {
  if (!declares(m, "layer_loops") || $("chat-loop-start").value === "") return undefined;
  return {start: +$("chat-loop-start").value, end: +$("chat-loop-end").value, count: +$("chat-loop-count").value};
}
/** The `cycles` of a context: only for a model with `cycles`, and not at its `default`. A context with the count of
 *  the model leaves the member out, so its records match those of other clients (gogi "Snapshots and pins"). */
function cyclesOf(m) {
  const text = String($("chat-cycles").value).trim();
  if (!declares(m, "cycles") || text === "" || +text === m.cycles?.default) return undefined;
  return +text;
}

// ---- the files of a message
/** The files of the next message, and the objects of the chat that the keep holds. Each file is {id, mime, name, host}. */
let chatFiles = [], chatObjects = [];
keeps(() => chatObjects);
const drawFiles = () => { $("chat-files").textContent = chatFiles.map((f) => f.name).join(", "); };
/** The refusal of the first file that the model does not read, or undefined when it reads each one. */
function unread(m, files) {
  const inputs = m?.media?.inputs || [];
  const f = files.find((x) => !reads(inputs, x.mime));
  if (!f) return undefined;
  return `${f.name} is ${f.mime || "of no known type"}; ${m?.model} reads ${inputs.length ? inputs.join(", ") : "no files"}` +
    ". Choose a model that reads it, or start a new chat.";
}
$("chat-file").onchange = async (ev) => {
  const files = [...ev.target.files], status = (t) => { $("chat-status").textContent = t; };
  ev.target.value = "";
  $("chat-error").textContent = "";
  try {
    for (const file of files) {
      const refused = unread(chatModel(), [{name: file.name, mime: file.type}]);
      if (refused) throw new Error(refused);
      const id = await putObject(file, file.type, status);
      chatFiles.push({id, mime: file.type, name: file.name, host});
    }
  } catch (e) { $("chat-error").textContent = e.message; }
  status(""); drawFiles();
};
/** The system prompt of each reply mode, from the model card of LFM3-Omni-2B. Speech mode is TTS: it speaks the
 *  user's text with the voice of its prompt, and with no prompt it has no voice. */
const REPLY_SYSTEM = {"": "", interleaved: "Respond with interleaved text and audio.",
                      speech: "Perform TTS in American English using a female voice."};
$("chat-output").onchange = () => {
  // A prompt that the person wrote stays; an empty one or the prompt of another mode follows the mode.
  const now = $("chat-system").value.trim();
  if (!now || Object.values(REPLY_SYSTEM).includes(now)) $("chat-system").value = REPLY_SYSTEM[$("chat-output").value];
};
/** The words of a chunk of speech. Speech runs about 2.5 words a second, so a chunk is about 16 s: the model card
 *  generates at most 256 tokens of speech, about 20 s, and longer speech drifts. */
const SPEECH_WORDS = 40;
/** The text in chunks of whole sentences of at most `SPEECH_WORDS` words. A longer sentence splits at its words. */
function speechChunks(text) {
  const sentences = text.split(/(?<=[.!?;:])\s+/).map((s) => s.trim()).filter(Boolean);
  const chunks = [];
  let now = [];
  const flush = () => { if (now.length) chunks.push(now.join(" ")); now = []; };
  for (const sentence of sentences) {
    const words = sentence.split(/\s+/);
    if (now.length + words.length > SPEECH_WORDS) flush();
    for (let i = 0; i < words.length; i += SPEECH_WORDS) {
      if (words.length > SPEECH_WORDS) { flush(); now = words.slice(i, i + SPEECH_WORDS); flush(); }
      else now = now.concat(words);
    }
  }
  flush();
  return chunks;
}
// The clips of one reply play one after the other.
document.addEventListener("ended", (ev) => {
  if (ev.target?.tagName !== "AUDIO") return;
  // The id of each clip stands between the clips, so the next clip is the next audio of the same message or of
  // the same job of the chat tray.
  const clips = [...ev.target.closest(".msg, .job")?.querySelectorAll("audio") || []];
  clips[clips.indexOf(ev.target) + 1]?.play().catch(() => {});
}, true);
/** Remove the audio part that a 404 names, when it is the spoken reply of an assistant message: the nagaya holds no
 *  codes of it, so the reply goes back as its text (spec 11.11, "Assistant audio"). A reply with no text leaves the
 *  context. Gives whether it removed a part. */
function dropSpokenPart(messages, e) {
  const at = /^messages\[(\d+)\]\.content\[(\d+)\]$/.exec(e.param || "");
  const message = at && messages[+at[1]], part = message?.content?.[+at[2]];
  if (e.status !== 404 || message?.role !== "assistant" || part?.mime !== "audio/wav") return false;
  message.content.splice(+at[2], 1);
  if (!message.content.length) messages.splice(+at[1], 1);
  return true;
}

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
/** The files of a message: a player for each audio file, and the content id of each file. */
const filesOf = (m) => (m.files || []).map((f) =>
  (f.mime === "audio/wav" ? `<audio controls src="${esc(objectUrl(f.id, f.host))}"></audio>` : "") +
  `<div class="stat">${objectChip(f)}</div>`).join("");
// The record of the chat with its last reply, when the nagaya names one: {id, host, system, loop, cycles} (gogi
// "Context records"). The next turn names it as its parent and sends only its new message. A new system message
// changes the first message, so that turn sends the chat whole.
let chatRecord = null;
function drawChat(live) {
  const all = live ? chat.concat([live]) : chat;
  $("chat-log").innerHTML = all.map((m, i) => {
    const think = m.reasoning ? `<details ${m.open ? "open" : ""}><summary>thinking</summary>${esc(m.reasoning)}</details>` : "";
    return `<div class="msg ${m.role}">${think}${esc(m.content || "")}${filesOf(m)}${statusOf(m)}${sendButtons(m, i)}</div>`;
  }).join("");
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}
/** A message as the context sends it. A message with files is a list of parts: its text, then an object part for
 *  each file. The person's messages are boundaries. */
function wireOf(m) {
  const files = m.files || [];
  const content = files.length ? (m.content ? [{type: "text", text: m.content}] : [])
    .concat(files.map((f) => ({type: "object", id: f.id, mime: f.mime}))) : m.content;
  const wire = {role: m.role, content};
  if (m.reasoning) wire.reasoning = m.reasoning;
  if (m.role === "user") wire.snap = true;
  return wire;
}
const playLast = () => document.querySelector?.("#chat-log .msg:last-child audio")?.play?.().catch(() => {});
$("chat-clear").onclick = () => {
  chat = []; chatRecord = null; chatFiles = []; drawFiles(); drawChat(); $("chat-error").textContent = "";
};
$("chat-stop").onclick = () => chatAbort?.abort();
$("chat-input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-go").click(); } };
/** Speech mode is TTS: each chunk of the text is its own request, so no clip drifts, and the clips play in order. It
 *  is not a turn of the record, so the next turn sends the chat whole. */
async function speakTurn(model, text, system, control) {
  const status = (t) => { $("chat-status").textContent = t; };
  chat.push({role: "user", content: text}); $("chat-input").value = ""; drawChat();
  $("chat-go").disabled = true;
  try {
    await ready(model, status);
    const clips = [];
    let ms = 0;
    for (const [i, chunk] of speechChunks(text).entries()) {
      status(`speaking part ${i + 1}…`);
      const body = {model, max_tokens: +$("chat-max").value, sample: {temperature: +$("chat-temp").value},
                    context: {thinking: false, messages: (system ? [{role: "system", content: system}] : [])
                      .concat([{role: "user", content: chunk}])},
                    output: {mime: "audio/wav", options: {mode: "speech"}}};
      if (control) body.context.control = control;
      const done = await call("POST", "/council/v2/generations", body);
      for (const o of done.outputs || []) if (o.mime === "audio/wav") clips.push({id: o.id, mime: o.mime, host});
      ms += done.usage?.media?.written?.audio_ms || 0;
    }
    chat.push({role: "assistant", content: "", files: clips,
               stat: `${clips.length} clips · ${(ms / 1000).toFixed(1)} s of speech`});
    chatObjects = chatObjects.concat(clips).slice(-200);
    chatRecord = null;
    keepAll();
  } catch (e) {
    chat.pop(); $("chat-input").value = text;
    $("chat-error").textContent = e.message;
  }
  status("");
  drawChat(); playLast();
  $("chat-go").disabled = false;
}
$("chat-go").onclick = async () => {
  const text = $("chat-input").value.trim(), model = $("chat-model").value, m = chatModel();
  if ((!text && !chatFiles.length) || !model) return;
  $("chat-error").textContent = "";
  let layers;
  try {
    layers = parseLayers($("chat-layers").value);
  } catch (e) {
    $("chat-error").textContent = `Layers: ${e.message}`;
    return;
  }
  const files = chatFiles.filter((f) => here(f.host)), refused = unread(m, files);
  if (refused) { $("chat-error").textContent = refused; return; }
  const system = $("chat-system").value.trim(), thinking = $("chat-thinking").checked && thinks(model);
  const control = controlOf(model, $("chat-vector").value, $("chat-op").value, $("chat-strength").value, layers);
  const output = talks(m) ? $("chat-output").value : "";
  if (output === "speech" && text && !files.length) return speakTurn(model, text, system, control);
  const loop = loopOf(m), cycles = cyclesOf(m);
  chat.push({role: "user", content: text, files}); chatFiles = []; drawFiles(); $("chat-input").value = ""; drawChat();
  /** The chat as one context: each message of the page, with a boundary at each message of the person. */
  const whole = () => {
    const context = {thinking};
    if (control) context.control = control;
    if (loop) context.loop = loop;
    if (cycles !== undefined) context.cycles = cycles;
    context.messages = (system ? [{role: "system", content: system, snap: true}] : []).concat(chat.map(wireOf));
    return context;
  };
  /** The record and the new message. The settings of this turn stand beside the parent and replace those of the
   *  record; null drops a loop or a count of cycles that the record holds (gogi "Context records"). */
  const byRecord = () => {
    const context = {parent: chatRecord.id, thinking, control: control || []};
    if (loop) context.loop = loop; else if (chatRecord.loop) context.loop = null;
    if (cycles !== undefined) context.cycles = cycles; else if (chatRecord.cycles !== undefined) context.cycles = null;
    context.messages = [wireOf(chat[chat.length - 1])];
    return context;
  };
  let byParent = !!chatRecord && here(chatRecord.host) && chatRecord.system === system;
  // An empty field leaves its member out, so the nagaya does not cut the choice of tokens by it.
  const sample = {temperature: +$("chat-temp").value};
  if ($("chat-topk").value !== "") sample.top_k = +$("chat-topk").value;
  if ($("chat-topp").value !== "") sample.top_p = +$("chat-topp").value;
  const body = {model, max_tokens: +$("chat-max").value, sample, context: byParent ? byRecord() : whole()};
  if (output) body.output = {mime: "audio/wav", options: {mode: output}};
  const live = {role: "assistant", content: "", reasoning: "", open: true};
  $("chat-go").disabled = true; $("chat-stop").hidden = false;
  chatAbort = new AbortController();
  let tps = null;
  try {
    await ready(model, (t) => { live.stat = t; drawChat(live); });
    live.stat = "";
    const once = () => stream(body, chatAbort.signal, (name, d) => {
      if (name !== "token") return;
      if (d.phase === "reasoning") live.reasoning += d.piece; else if (d.phase === "content") live.content += d.piece;
      tps = d.tps ?? tps;
      live.stat = `${d.n} tokens${tps ? ` · ${tps.toFixed(1)} tok/s` : ""}`;
      drawChat(live);
    });
    let done;
    for (;;) {
      try {
        done = await once();
        break;
      } catch (e) {
        // The nagaya dropped the record: a 404 that names the parent. The page has the chat, so it sends it whole.
        if (byParent && e.status === 404 && e.param === "context.parent") {
          byParent = false; chatRecord = null; body.context = whole();
        } else if (byParent || !dropSpokenPart(body.context.messages, e)) {
          throw e;
        }
      }
    }
    const sent = utf8Length(JSON.stringify({...body, stream: true}));
    chatRecord = done.context ? {id: done.context, host, system, loop, cycles} : null;
    const u = done.usage;
    // A snapshot names the index of its message in the whole context, where the system message is index 0.
    const first = system ? 1 : 0, others = [];
    for (const s of done.snapshots || []) {
      if (chat[s.message - first]) chat[s.message - first].snapshot = s; else others.push(s);
    }
    const spoken = (done.outputs || []).filter((o) => o.mime === "audio/wav").map((o) => ({id: o.id, mime: o.mime, host}));
    const ms = spoken.length ? u.media?.written?.audio_ms || 0 : 0;
    chat.push({role: "assistant", content: done.content, reasoning: done.reasoning, files: spoken,
               stat: `${u.completion} tokens${tps ? ` · ${tps.toFixed(1)} tok/s` : ""} · ${done.finish}` +
                 (ms ? ` · ${(ms / 1000).toFixed(1)} s of speech` : ""),
               receipt: {sent, usage: u, others, byParent, record: done.context, stats: done.stats}});
    if (files.length || spoken.length) {
      chatObjects = chatObjects.concat(files, spoken).slice(-200);
      keepAll();
    }
  } catch (e) {
    // The turn did not finish, so the message and its files go back to the input, and the chat stays as it was.
    chat.pop(); $("chat-input").value = text; chatFiles = files; drawFiles();
    $("chat-error").textContent = e.name === "AbortError" ? "stopped; the partial reply is not kept" : e.message;
  }
  drawChat(); playLast();
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

/** A model that speaks: a speech model, or a model that speaks beside its text (spec 11.8a, 11.11). */
const speaks = (m) => kindOf(m) === "speech" || talks(m);
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
    // A speech model takes the voice of the Speech tab. Each chunk continues from the chunks and the clips before
    // it, so the voice stays the same through the text (spec 11.8a). The nagaya holds their state, so a chunk feeds
    // only its own text.
    const spoken = [{role: "system", content: $("sp-voice").value, snap: true}];
    for (const [n, chunk] of chunks.entries()) {
      job.status(`speaking part ${n + 1} of ${chunks.length}…`);
      // A model that speaks beside its text takes the prompt of its speech mode, and each chunk alone.
      const body = speech
        ? {model, max_tokens: 1, context: {thinking: false, messages: spoken.concat([{role: "user", content: chunk}])},
           output: {mime: "audio/wav", options: {language: $("sp-lang").value || "auto"}}}
        : {model, max_tokens: 512, sample: {temperature: 0.3},
           context: {thinking: false, messages: [{role: "system", content: REPLY_SYSTEM.speech},
                                                 {role: "user", content: chunk}]},
           output: {mime: "audio/wav", options: {mode: "speech"}}};
      const done = await call("POST", "/council/v2/generations", body);
      const made = (done.outputs || []).filter((o) => o.mime === "audio/wav");
      if (made.length) {
        spoken.push({role: "user", content: chunk, snap: true},
                    {role: "assistant", content: [{type: "object", id: made[0].id, mime: "audio/wav"}], snap: true});
      }
      for (const o of made) {
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
// The form follows the model: this runs after the model list has its choice.
onRefresh(drawChatForm);
