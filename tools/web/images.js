"use strict";
// The Images tab: the form of an image job, LoRAs, the start image and its mask, and the gallery of this browser.

// The start image of img2img and inpainting: {blob, width, height}, or null (spec 11.9b).
let init = null;
const takesImages = (model) =>
  !!models.find((x) => x.model === model)?.media?.inputs?.some((t) => t === "image/png" || t === "image/*");

let gallery = recallJson("nagaya-gallery", []);
const saveGallery = () => rememberJson("nagaya-gallery", gallery.slice(0, 120));
function tile(item) {
  const [w, h] = item.size.split("x").map(Number);
  const url = objectUrl(item.id, item.host ?? "");
  return `<div class="tile" style="--ratio:${w}/${h}"><a href="${url}" target="_blank">
    <img src="${url}" loading="lazy" onerror="this.closest('.tile').style.opacity=.35"></a>
    <div class="meta"><a href="#" class="use" data-init="${url}" title="use as the start image">start</a>
    <div class="p" title="${esc(item.prompt)}">${esc(item.prompt)}</div>
    ${item.host && !here(item.host) ? `${esc(item.host)} · ` : ""}${esc(item.model)} · seed ${esc(item.seed)} · ${esc(item.size)} · ${esc(item.steps)} steps</div></div>`;
}
function drawGallery() { $("img-gallery").innerHTML = gallery.map(tile).join("") || `<p class="note">None yet.</p>`; }
drawGallery();
$("img-clear").onclick = () => { gallery = []; saveGallery(); drawGallery(); keepAll(); };

// The spacing applies to the Euler samplers only; with DPM++ it is a 400 (spec 11.9a).
const euler = () => $("img-sampler").value.startsWith("euler");
/** Whether a served model is Qwen-Image. The profile kind of its engine says so (spec 11.9c). */
function qwenImage(model) {
  const m = models.find((x) => x.model === model), name = m && engineOf(m);
  return /qwenimage/.test(engines.find((e) => e.name === name)?.kind || "");
}
// SDXL takes multiples of 64 up to 1,572,864 pixels (spec 11.9a). Qwen-Image takes multiples of 32 up to 2,097,152
// pixels (spec 11.9c), so it adds larger sizes.
const SIZES = ["1024x1024", "896x1152", "1152x896", "832x1216", "1216x832", "768x1344", "1344x768", "512x512"];
const QWEN_SIZES = SIZES.concat(["1440x1440", "1664x928", "928x1664", "1280x1280"]);
/** Turn off each field that the image model does not take, and give the sizes that it takes. */
function drawImageForm() {
  const qwen = qwenImage($("img-model").value);
  for (const id of ["img-negative", "img-cfg", "img-sampler", "img-clip", "img-lora-file"]) $(id).disabled = qwen;
  $("img-spacing").disabled = qwen || !euler();
  $("img-qwen-note").hidden = !qwen;
  $("img-init-box").hidden = !takesImages($("img-model").value);
  $("img-size").disabled = !!init;
  $("img-size-note").textContent = init ? "— the start image sets it" : "";
  $("img-prompt-note").hidden = qwen;
  const sizes = qwen ? QWEN_SIZES : SIZES, keep = $("img-size").value;
  const html = sizes.map((x) => `<option>${x}</option>`).join("");
  if ($("img-size").dataset.html !== html) {
    $("img-size").innerHTML = html; $("img-size").dataset.html = html;
    $("img-size").value = sizes.includes(keep) ? keep : sizes[0];
  }
}
$("img-sampler").addEventListener("change", drawImageForm); drawImageForm();
modelSelect("img-model", (m) => kindOf(m) === "image");
onRefresh(drawImageForm);
$("img-model").addEventListener("change", () => {
  if (qwenImage($("img-model").value)) {
    // The defaults of Qwen-Image: 40 steps, CFG 1 and Euler (spec 11.9c).
    $("img-steps").value = 40; $("img-cfg").value = 1; $("img-sampler").value = "euler";
  } else if (/lightning/i.test($("img-model").value)) {
    // SDXL-Lightning 4-step needs 4 steps, CFG 1, Euler and trailing spacing (its model card).
    $("img-steps").value = 4; $("img-cfg").value = 1; $("img-sampler").value = "euler"; $("img-spacing").value = "trailing";
  }
  drawImageForm();
});

// LoRAs are objects (spec 6.15). The page keeps its list in this browser, and a keep holds the objects in the store.
let loras = recallJson("nagaya-loras", []);
const saveLoras = () => rememberJson("nagaya-loras", loras);
function drawLoras() {
  $("img-loras").innerHTML = loras.map((l, i) => !here(l.host) ? "" : `<div class="lora">
    <label class="name" title="${esc(l.id)}"><input type="checkbox" data-lora-on="${i}" ${l.on ? "checked" : ""}>
      ${esc(l.name)}</label>
    <input type="number" min="-2" max="2" step="0.05" value="${esc(l.weight)}" data-lora-weight="${i}">
    <button class="plain small" data-lora-drop="${i}" title="remove from this list">×</button></div>`).join("") ||
    `<p class="note">None.</p>`;
}
drawLoras();
onHost(drawGallery); onHost(drawLoras);
keeps(() => loras); keeps(() => gallery);
$("img-loras").addEventListener("change", (ev) => {
  const on = ev.target.dataset.loraOn, weight = ev.target.dataset.loraWeight;
  if (on !== undefined) loras[+on].on = ev.target.checked;
  if (weight !== undefined) loras[+weight].weight = +ev.target.value;
  saveLoras();
});
$("img-loras").addEventListener("click", (ev) => {
  const drop = ev.target.dataset.loraDrop;
  if (drop === undefined) return;
  loras.splice(+drop, 1); saveLoras(); drawLoras(); keepAll();
});
$("img-lora-file").onchange = async (ev) => {
  const file = ev.target.files[0], status = (t) => { $("img-lora-status").textContent = t; };
  ev.target.value = "";
  if (!file) return;
  try {
    const id = await putObject(file, "application/octet-stream", status);
    if (!loras.some((l) => l.id === id && here(l.host))) {
      loras.push({name: file.name.replace(/\.safetensors$/, ""), id, weight: 1, on: true, host});
    }
    saveLoras(); drawLoras(); await keepAll();
    status(`${file.name} is held`);
  } catch (e) {
    status(e.message);
  }
};
// ---- image inputs (spec 11.9b)
// The start image and the mask are PNG objects in the last user message, the image first. The mask canvas has the size
// of the image, and it is opaque: white repaints, and black keeps.
const maskCanvas = $("img-mask-canvas");
const canvasBlob = (canvas) => new Promise((ok, fail) =>
  canvas.toBlob((b) => b ? ok(b) : fail(new Error("the browser did not make a PNG")), "image/png"));
function clearMask() {
  const g = maskCanvas.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
}
/** Whether the mask repaints no pixel. Then the job is img2img, and the page sends no mask. */
function maskEmpty() {
  const d = maskCanvas.getContext("2d").getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
  for (let i = 0; i < d.length; i += 4) if (d[i] || d[i + 1] || d[i + 2]) return false;
  return true;
}
/** Show a start image. A PNG goes as it is. Another format becomes a PNG of the same size: the page never resizes. */
async function loadInit(blob, name) {
  const bitmap = await createImageBitmap(blob);
  const {width, height} = bitmap;
  const c = $("img-init-canvas");
  c.width = width; c.height = height; c.getContext("2d").drawImage(bitmap, 0, 0);
  maskCanvas.width = width; maskCanvas.height = height; clearMask();
  init = {blob: blob.type === "image/png" ? blob : await canvasBlob(c), width, height};
  $("img-init").hidden = false; $("img-init-clear").hidden = false;
  $("img-init-status").textContent = `${name}, ${width} × ${height}`;
  drawImageForm();
}
$("img-init-file").onchange = async (ev) => {
  const file = ev.target.files[0];
  ev.target.value = "";
  if (!file) return;
  try {
    await loadInit(file, file.name);
  } catch (e) {
    $("img-init-status").textContent = `the image did not open: ${e.message}`;
  }
};
$("img-init-clear").onclick = () => {
  init = null; $("img-init").hidden = true; $("img-init-clear").hidden = true; $("img-init-status").textContent = "";
  drawImageForm();
};
$("img-mask-clear").onclick = clearMask;
$("img-mask-file").onchange = async (ev) => {
  const file = ev.target.files[0];
  ev.target.value = "";
  if (!file || !init) return;
  try {
    const bitmap = await createImageBitmap(file);
    // The nagaya refuses a mask of another size (spec 11.9b), and the page does not resize it.
    if (bitmap.width !== init.width || bitmap.height !== init.height) {
      throw new Error(`the mask is ${bitmap.width} × ${bitmap.height}, and the image is ${init.width} × ${
        init.height}; choose a mask of the size of the image`);
    }
    clearMask(); maskCanvas.getContext("2d").drawImage(bitmap, 0, 0);
    $("img-init-status").textContent = `mask ${file.name}`;
  } catch (e) {
    $("img-init-status").textContent = e.message;
  }
};
// The brush paints in pixels of the image, so it has the same size on a small and a large screen.
let brushAt = null;
function brush(ev) {
  const r = maskCanvas.getBoundingClientRect();
  const x = (ev.clientX - r.left) * maskCanvas.width / r.width, y = (ev.clientY - r.top) * maskCanvas.height / r.height;
  const g = maskCanvas.getContext("2d"), size = +$("img-brush").value || 48;
  g.strokeStyle = g.fillStyle = $("img-erase").checked ? "#000" : "#fff";
  g.lineWidth = size; g.lineCap = "round"; g.lineJoin = "round";
  g.beginPath();
  if (brushAt) { g.moveTo(brushAt[0], brushAt[1]); g.lineTo(x, y); g.stroke(); }
  else { g.arc(x, y, size / 2, 0, 2 * Math.PI); g.fill(); }
  brushAt = [x, y];
}
maskCanvas.addEventListener("pointerdown", (ev) => { maskCanvas.setPointerCapture(ev.pointerId); brushAt = null; brush(ev); });
maskCanvas.addEventListener("pointermove", (ev) => { if (brushAt) brush(ev); });
for (const name of ["pointerup", "pointercancel"]) maskCanvas.addEventListener(name, () => { brushAt = null; });
/** Put the start image, and the mask when it repaints a pixel. Give their ids; the mask is null for img2img. */
async function inputObjects(status) {
  const image = await putObject(init.blob, "image/png", status);
  const mask = maskEmpty() ? null : await putObject(await canvasBlob(maskCanvas), "image/png", status);
  return {image, mask};
}
for (const id of ["img-gallery", "img-current"]) {
  $(id).addEventListener("click", async (ev) => {
    const url = ev.target.dataset?.init;
    if (!url) return;
    ev.preventDefault();
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`the image returned ${r.status}`);
      await loadInit(await r.blob(), "an image of the gallery");
    } catch (e) {
      $("img-init-status").textContent = e.message;
    }
  });
}

let imgAbort = null;
$("img-stop").onclick = () => imgAbort?.abort();
$("img-go").onclick = async () => {
  const model = $("img-model").value;
  // The image sets the size of a job with an image (spec 11.9b).
  const [width, height] = init ? [init.width, init.height] : $("img-size").value.split("x").map(Number);
  const count = +$("img-count").value, steps = +$("img-steps").value;
  const status = (t) => { $("img-status").textContent = t; };
  $("img-error").textContent = "";
  if (!model) { $("img-error").textContent = "the nagaya serves no image model"; return; }
  if (init && !takesImages(model)) {
    $("img-error").textContent = `${model} takes no start image; remove the start image, or choose a model that takes one`;
    return;
  }
  const options = {width, height, steps, count};
  if (!qwenImage(model)) {
    Object.assign(options, {cfg: +$("img-cfg").value, sampler: $("img-sampler").value, negative: $("img-negative").value});
    if ($("img-clip").value) options.clip_skip = +$("img-clip").value;
    if (euler()) options.spacing = $("img-spacing").value;
    const on = loras.filter((l) => l.on && here(l.host));
    if (on.length > 8) { $("img-error").textContent = "a request takes at most 8 LoRAs"; return; }
    if (on.length) options.loras = on.map((l) => ({id: l.id, weight: l.weight}));
  }
  if (init) {
    delete options.width; delete options.height;
    if ($("img-strength").value !== "") options.strength = +$("img-strength").value;
  }
  const body = {model, max_tokens: 1, context: {thinking: false, messages: [{role: "user", content: $("img-prompt").value}]},
                output: {mime: "image/png", options}, previews: $("img-previews").checked};
  if ($("img-seed").value !== "") body.sample = {seed: +$("img-seed").value};
  $("img-go").disabled = true; $("img-stop").hidden = false;
  imgAbort = new AbortController();
  $("img-current").innerHTML = Array.from({length: count}, (_, i) =>
    `<div class="tile" style="--ratio:${width}/${height}"><img id="cur-img-${i}" alt="">
     <div class="meta"><div class="bar"><div id="cur-bar-${i}"></div></div><span id="cur-text-${i}">waiting</span></div></div>`).join("");
  const started = performance.now();
  try {
    await ready(model, status);
    if (init) {
      const inputs = await inputObjects(status), content = [{type: "object", id: inputs.image, mime: "image/png"}];
      if (inputs.mask) {
        content.push({type: "object", id: inputs.mask, mime: "image/png"});
        if ($("img-mask-blur").value !== "") options.mask_blur = +$("img-mask-blur").value;
      }
      content.push({type: "text", text: $("img-prompt").value});
      body.context.messages[0].content = content;
    }
    status("generating…");
    const done = await stream(body, imgAbort.signal, (name, d) => {
      if (name === "progress") {
        $(`cur-bar-${d.index}`).style.width = `${100 * d.step / d.of}%`;
        $(`cur-text-${d.index}`).textContent = `step ${d.step} of ${d.of}`;
        status(`step ${d.step} of ${d.of} · ${((performance.now() - started) / 1000).toFixed(1)} s`);
      } else if (name === "preview" && d.id) {
        $(`cur-img-${d.index}`).src = objectUrl(d.id);
      } else if (name === "output") {
        $(`cur-text-${d.index}`).textContent = "decoding";
      }
    });
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const made = done.outputs.map((o) => ({id: o.id, host, model, prompt: $("img-prompt").value, seed: done.seed + o.index,
                                           size: `${width}x${height}`, steps}));
    $("img-current").innerHTML = made.map(tile).join("");
    gallery = made.concat(gallery); saveGallery(); drawGallery(); keepAll();
    status(`${made.length} image${made.length > 1 ? "s" : ""} in ${seconds} s, seed ${done.seed}`);
  } catch (e) {
    $("img-error").textContent = e.name === "AbortError" ? "stopped" : e.message; status("");
  }
  $("img-go").disabled = false; $("img-stop").hidden = true;
};
