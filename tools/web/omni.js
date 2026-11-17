"use strict";
// The Omni tab: a conversation with a model that reads media and can speak its replies (spec 11.11).

// ---- omni
/** A model of the Omni tab: it reads media and its replies hold text (spec 11.11). */
const omnis = (m) => (m.media?.inputs?.length ?? 0) > 0 && m.media?.text_replies !== false;
/** The conversation of the tab, the files of the next message, and the objects that the keep holds. */
let omni = [], omniFiles = [], omniKept = [];
function omniAccept() {
  const m = models.find((x) => x.model === $("omni-model").value);
  $("omni-file").accept = (m?.media?.inputs || []).join(",");
}
function drawOmni() {
  $("omni-log").innerHTML = omni.map((t) => `<div class="msg ${t.role}"><div>${esc(t.text)}</div>` +
    t.files.map((f) => (f.mime === "audio/wav" ? `<audio controls src="${esc(objectUrl(f.id, f.host))}"></audio>` : "") +
      `<div class="stat">${objectChip(f)}</div>`).join("") +
    (t.stat ? `<div class="note">${esc(t.stat)}</div>` : "") + "</div>").join("");
  $("omni-files").textContent = omniFiles.map((f) => f.name).join(", ");
}
$("omni-file").onchange = async (ev) => {
  const files = [...ev.target.files], status = (t) => { $("omni-status").textContent = t; };
  ev.target.value = "";
  try {
    const reads = models.find((x) => x.model === $("omni-model").value)?.media?.inputs || [];
    for (const file of files) {
      if (!reads.includes(file.type)) {
        throw new Error(`${file.name} is ${file.type || "of no known type"}; this model reads ${reads.join(", ")}`);
      }
      const id = await putObject(file, file.type, status);
      omniFiles.push({id, mime: file.type, name: file.name, host});
    }
    status(""); drawOmni();
  } catch (e) { $("omni-error").textContent = e.message; status(""); }
};
$("omni-clear").onclick = () => { omni = []; omniFiles = []; drawOmni(); };
/** The system prompt of each reply mode, from the model card of LFM3-Omni-2B. Speech mode is TTS: it speaks the
 *  user's text with the voice of its prompt, and with no prompt it has no voice. */
const OMNI_SYSTEM = {"": "", interleaved: "Respond with interleaved text and audio.",
                     speech: "Perform TTS in American English using a female voice."};
$("omni-output").onchange = () => {
  // A prompt that the person wrote stays; an empty one or the prompt of another mode follows the mode.
  const now = $("omni-system").value.trim();
  if (!now || Object.values(OMNI_SYSTEM).includes(now)) $("omni-system").value = OMNI_SYSTEM[$("omni-output").value];
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
/** The control list of an Omni context: the vector of the tab, when one of this model is chosen (spec 11.12). */
const omniControl = (model) => controlOf(model, $("omni-vector").value, $("omni-op").value, $("omni-strength").value);
/** Speech mode is TTS: each chunk of the text is its own request, so no clip drifts, and the clips play in order. */
async function omniSpeak(model, text) {
  const system = $("omni-system").value.trim();
  const clips = [];
  let frames = 0;
  for (const [i, chunk] of speechChunks(text).entries()) {
    $("omni-status").textContent = `speaking part ${i + 1}…`;
    const body = {model, max_tokens: +$("omni-max").value, sample: {temperature: +$("omni-temp").value},
                  context: {thinking: false, messages: (system ? [{role: "system", content: system}] : [])
                    .concat([{role: "user", content: chunk}])},
                  output: {mime: "audio/wav", options: {mode: "speech"}}};
    const control = omniControl(model);
    if (control) body.context.control = control;
    const done = await call("POST", "/council/v2/generations", body);
    for (const o of done.outputs || []) if (o.mime === "audio/wav") clips.push({id: o.id, mime: o.mime, host});
    frames += done.usage?.media?.written?.audio_ms || 0;
  }
  return {clips, seconds: frames / 1000};
}
// The clips of one reply play one after the other.
document.addEventListener("ended", (ev) => {
  if (ev.target?.tagName !== "AUDIO") return;
  // The id of each clip stands between the clips, so the next clip is the next audio of the same message or of
  // the same job of the chat tray.
  const clips = [...ev.target.closest(".msg, .job")?.querySelectorAll("audio") || []];
  clips[clips.indexOf(ev.target) + 1]?.play().catch(() => {});
}, true);

/** Send the request. If the server holds no codes of a spoken reply (404 that names its part), send that reply's
 * text instead and try again (spec 11.11, "Assistant audio"). A reply with no text leaves the context. */
async function generate(body) {
  for (;;) {
    try { return await call("POST", "/council/v2/generations", body); } catch (e) {
      const at = /^messages\[(\d+)\]\.content\[(\d+)\]$/.exec(e.param || ""), messages = body.context.messages;
      const part = at && messages[+at[1]]?.content?.[+at[2]];
      if (e.status !== 404 || messages[+at?.[1]]?.role !== "assistant" || part?.mime !== "audio/wav") throw e;
      messages[+at[1]].content.splice(+at[2], 1);
      if (!messages[+at[1]].content.length) messages.splice(+at[1], 1);
    }
  }
}
$("omni-go").onclick = async () => {
  const text = $("omni-input").value.trim(), model = $("omni-model").value;
  if ((!text && !omniFiles.length) || !model) return;
  $("omni-error").textContent = "";
  if ($("omni-output").value === "speech" && text && !omniFiles.length) {
    omni.push({role: "user", text, files: []}); $("omni-input").value = ""; drawOmni();
    $("omni-go").disabled = true;
    try {
      await ready(model, (t) => { $("omni-status").textContent = t; });
      const {clips, seconds} = await omniSpeak(model, text);
      omni.push({role: "assistant", text: "", files: clips,
                 stat: `${clips.length} clips · ${seconds.toFixed(1)} s of speech`});
      omniKept = omniKept.concat(clips).slice(-200);
      keepAll();
    } catch (e) {
      omni.pop(); $("omni-input").value = text;
      $("omni-error").textContent = e.message;
    }
    $("omni-status").textContent = "";
    drawOmni();
    document.querySelector?.("#omni-log .msg:last-child audio")?.play?.().catch(() => {});
    $("omni-go").disabled = false;
    return;
  }
  const files = omniFiles.filter((f) => here(f.host));
  omni.push({role: "user", text, files}); omniFiles = []; $("omni-input").value = ""; drawOmni();
  const messages = [];
  if ($("omni-system").value.trim()) messages.push({role: "system", content: $("omni-system").value.trim(), snap: true});
  for (const t of omni) {
    const parts = (t.text ? [{type: "text", text: t.text}] : [])
      .concat(t.files.map((f) => ({type: "object", id: f.id, mime: f.mime})));
    // A spoken reply goes back as its text and its audio, whose codes the service keeps (spec 11.11).
    messages.push(t.role === "user" ? {role: "user", content: parts, snap: true} : {role: "assistant", content: parts});
  }
  const body = {model, max_tokens: +$("omni-max").value, sample: {temperature: +$("omni-temp").value},
                context: {thinking: false, messages}};
  if ($("omni-output").value) body.output = {mime: "audio/wav", options: {mode: $("omni-output").value}};
  const control = omniControl(model);
  if (control) body.context.control = control;
  $("omni-go").disabled = true;
  try {
    await ready(model, (t) => { $("omni-status").textContent = t; });
    $("omni-status").textContent = "thinking…";
    const done = await generate(body);
    const spoken = (done.outputs || []).filter((o) => o.mime === "audio/wav").map((o) => ({id: o.id, mime: o.mime, host}));
    const u = done.usage || {};
    omni.push({role: "assistant", text: done.content || "", files: spoken,
               stat: `${u.completion ?? "?"} tokens · ${done.finish ?? ""}` +
                 (u.media?.written?.audio_ms ? ` · ${(u.media.written.audio_ms / 1000).toFixed(1)} s of speech` : "")});
    omniKept = omniKept.concat(files, spoken).slice(-200);
    keepAll();
  } catch (e) {
    omni.pop(); $("omni-input").value = text; omniFiles = files;
    $("omni-error").textContent = e.message;
  }
  $("omni-status").textContent = "";
  drawOmni();
  $("omni-go").disabled = false;
};
modelSelect("omni-model", omnis);
onRefresh(omniAccept);
keeps(() => omniKept);
