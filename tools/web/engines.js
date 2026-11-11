"use strict";
// The Engines tab, and the loads that each tab uses: the load of an engine, the pin, and the wait for a model.

document.addEventListener("click", async (ev) => {
  const {load, unload, pin, unpin} = ev.target.dataset || {};
  if (!load && !unload && !pin && !unpin) return;
  const name = load || unload || pin || unpin;
  ev.target.disabled = true;
  try {
    if (pin || unpin) {
      const question = pin ? `Pin ${name}? The profile changes: ${name} becomes resident and pinned, so no load or unload ` +
        "removes it, and the next start loads it." : `Unpin ${name}? It stays loaded, and an unload can then remove it.`;
      if (confirm(question)) {
        const r = await pinEngine(name, !!pin);
        $("eng-note").textContent = `the profile changed${r?.restart ? ", and the nagaya restarts" : ""}`;
      }
    } else {
      const r = load ? await loadEngine(name)
                     : await call("POST", `/nagaya/v1/engines/${encodeURIComponent(name)}/unload`);
      $("eng-note").textContent = r?.unload?.length ? `loading ${name} unloads ${r.unload.join(", ")}` : "";
    }
  } catch (e) {
    $("eng-error").textContent = e.message; alert(e.message);
  }
  ev.target.disabled = false;
  refresh();
});

/**
 * Pin or unpin an engine with a profile change (spec 3.3). The page reads the active profile and puts it back with one
 * change. A pin makes the engine resident too, because the nagaya refuses a pin on an on-demand engine (spec 3.3a).
 */
async function pinEngine(name, on) {
  const profile = await call("GET", "/nagaya/v1/profile");
  const entry = profile.engines.find((e) => e.name === name);
  if (!entry) throw new Error(`the profile has no engine ${name}; refresh the page`);
  entry.pinned = on;
  if (on) delete entry.residency;
  return await call("PUT", "/nagaya/v1/profile", profile);
}

/** The engine names in one clause of a 409 message, such as "the loaded engines in the way are `a`, `b`". */
function namesAfter(message, clause) {
  const at = message.indexOf(clause);
  if (at < 0) return [];
  const rest = message.slice(at + clause.length).split(";")[0];
  return [...rest.matchAll(/`([^`]+)`/g)].map((m) => m[1]).filter((n) => n !== "unload");
}

/**
 * Load an engine. A load that does not fit is a 409 that names the loaded engines in its way and the pins (spec 3.3a).
 * Then ask to unload the engines that the nagaya names, and send the load again with them in `unload`.
 */
async function loadEngine(name) {
  const path = `/nagaya/v1/engines/${encodeURIComponent(name)}/load`;
  try {
    return await call("POST", path);
  } catch (e) {
    if (e.status !== 409 || !/does not fit/.test(e.message)) throw e;
    const way = namesAfter(e.message, "the loaded engines in the way are");
    if (!way.length) throw e;
    const pins = namesAfter(e.message, "the pinned engines");
    const stays = pins.length ? `\n${pins.join(", ")} ${pins.length > 1 ? "stay" : "stays"} loaded: pinned.` : "";
    if (!confirm(`${e.message}\n\nUnload ${way.join(", ")} and load ${name}?${stays}`)) throw e;
    return await call("POST", path, {unload: way});
  }
}

/** Wait until a served model is ready, and load its engine when it is not. */
async function ready(model, status) {
  let m = models.find((x) => x.model === model);
  if (m?.status === "ready") return;
  status(`loading ${model}…`);
  if (m && m.status !== "loading") {
    const r = await loadEngine(engineOf(m));
    if (r?.unload?.length) status(`loading ${model}; it unloads ${r.unload.join(", ")}…`);
  }
  for (let i = 0; i < 600; i++) {
    await new Promise((ok) => setTimeout(ok, 1000));
    await refresh();
    m = models.find((x) => x.model === model);
    if (m?.status === "ready") return;
    if (m?.status === "failed") throw new Error(`${model} failed to load`);
  }
  throw new Error(`${model} did not become ready in 10 min`);
}

onRefresh((profile) => {
  $("eng-error").textContent = "";
  const text = JSON.stringify(profile, null, 2);
  if ($("eng-profile").textContent !== text) $("eng-profile").textContent = text;
  $("eng-rows").innerHTML = engines.map((e) => {
    const m = modelOf(e.name), st = m?.status || "absent";
    const loaded = st === "ready" || st === "loading";
    return `<tr><td><b>${esc(e.name)}</b></td><td><code>${esc(e.kind)}</code></td><td>${esc(e.devices.join(", "))}</td>
      <td>${esc(e.residency || "resident")}${e.pinned ? ", pinned" : ""}</td>
      <td>${esc(m?.model || "")}</td><td class="${esc(st)}">${esc(st)}</td>
      <td class="buttons"><button class="plain small" data-load="${esc(e.name)}" ${loaded ? "disabled" : ""}>Load</button>
          <button class="plain small" data-unload="${esc(e.name)}" ${!loaded || e.pinned ? "disabled" : ""}
            ${e.pinned ? `title="pinned: unpin it to unload it"` : ""}>Unload</button>
          ${e.pinned ? `<button class="plain small" data-unpin="${esc(e.name)}">Unpin</button>`
            : `<button class="plain small" data-pin="${esc(e.name)}" ${st === "ready" ? "" : `disabled title="load it first"`}>Pin</button>`}
          </td></tr>`;
  }).join("");
});
