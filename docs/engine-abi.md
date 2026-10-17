# Engine ABI v0 (draft)

The interface every engine exports to the nagaya. Terms are defined in `docs/ontology.md`.

Drafted 2026-10-17 from mk's C ABI (`src/engine/capi.hip`, `src/engine/engine.h`) and mkviolet's Gemma 4 engine
(`src/gemma4.rs`, `src/held.rs`, `src/protocol.rs`, `src/hot_handoff.rs`). Revised the same day after a kaibo review
(deepseek cast) checked it against copies of that source. The review's headline: the first draft's call list was mk's
`Context` model, and mkviolet, the second engine, already failed half of it. So the ABI is now a small core that
every engine has, plus capabilities an engine declares.

## Shape

The nagaya is Rust. An engine is a Rust type that implements the `Engine` trait. An engine written in another
language (mk: C++ and HIP behind `extern "C"`) exports a C ABI, and a thin Rust wrapper implements the trait on top.
mkviolet's Gemma 4 engine implements the trait directly.

Each device has one nagaya thread. Calls for one device come from that thread only and are synchronous; a pass
blocks it. mk already works this way. mkviolet does not yet: `pipeline_stage` spawns a thread per card per stage,
which moves into the nagaya's device threads.

## Core (every engine)

```
open(config) -> engine           config: devices, weights, engine-specific settings (mkviolet: slot sizes, max_block)
attach(config, arena_set)        the same, on an existing arena set
identity() -> {model, weights_hash, tokenizer_hash, template, numbers, bit settings}
limits() -> {
    capabilities: [...],         which optional calls below this engine has
    parts: [{name, kind: fixed|cells|ring|derived, bytes(n)}],
    inputs: tokens | rows | tokens+rows, row width and dtype,
    stages: [{device, layers, head: bool}],
    max_segments, max_pass_inputs, max_logit_rows, slots per device and their sizes,
    taps: [...], with the conditions each needs (mk: unfused, one segment)
}
pass(segments, taps) -> status, ms, outputs
    segment = {state, inputs, positions, meta, outputs wanted}
last_error() -> why
close()
```

`positions` and `meta` cover mkviolet's `pos0` and per-token `meta` (`PipeJob`, `gemma4.rs:998-1004`); `outputs
wanted` covers `logit_rows`. `state` is a context handle where the engine has contexts, or a slot index where it has
slots.

## Capabilities (declared in `limits()`)

```
arena        arena_export() -> {name: fd}, arena_check(config, arena_set) -> ok | why, check: full | spot
contexts     ctx_new(max) / ctx_fork / ctx_copy / ctx_free / ctx_n
slots        slot_load(slot, snapshot parts) / slot_high(slot)
parts        part_read(state, part) / part_write(state, part, bytes)
             cells_read(state, part, from, n) / cells_write(state, part, n, bytes) / set_n(state, n)
seal         seal(state)                 the engine's boundary rule
geometry     geometry_key(state) -> bytes   slot geometry that changes bits, added to the recipe
coverage     coverage(part bytes) -> positions held; fits(coverage, slot) -> ok | why
             without it, a snapshot holds every position, 0 to n
poison       poisoned(state) -> bool
admission    pass_problem(segments) -> ok | why
stages       stage_pass(stage, segments, residual_in) -> residual_out | outputs
telemetry    op_times(), dispatches(), ...
```

## How mk maps

| ABI | mk (`capi.hip`) | Notes |
|---|---|---|
| open / attach | `mk_engine_open(s1, s2, s3)`, `mk_engine_attach(…, path)` | no device argument: the HIP device is the calling thread's current one, so the wrapper sets it on the device thread; weights are three GGUF shards |
| arena | `mk_engine_arena`, `mk_arena_check` | one fd; full check |
| identity | built in the Python service | moves into the engine; must include `fused` and `MK_FAST_QMM` |
| contexts | `mk_ctx_new`, `mk_ctx_fork`, `mk_ctx_copy`, `mk_ctx_free`, `mk_ctx_n` | a fork shares its parent's K/V and index keys |
| parts | `mk_ctx_state_read`, `mk_ctx_state_write`, `mk_ctx_cells_write`, `mk_ctx_set_n` | parts 0-2 and 5 fixed, 3-4 cells; `idx_pooled` derived. No `cells_read` range yet: snapshots are stored whole (about 316 MB at n = 3584) |
| seal | `mk_ctx_round_gdn_bf16` | changes bits in place |
| poison / last_error | `mk_ctx_poisoned`, `mk_last_error` | |
| admission / pass | `mk_segments_problem`, `mk_pass_segments` | tokens only (PLE needs token ids); outputs are logits and hidden per segment |
| taps | `mk_pass` (layers, routes), `mk_pass_gdn_ab` (gates) | unfused, one segment; logits go straight to the caller's buffer, the other taps are copied |
| telemetry | `mk_stage_times`, `mk_stage_detail`, `mk_dispatches`, `mk_host_stats` | mk's stages are ops (`stage.h`), so these are op times |

## How mkviolet maps

| ABI | mkviolet (`gemma4.rs` unless named) | Notes |
|---|---|---|
| open / attach | `load_slots(cluster, model_dir, slot_seqs, max_block)`, `load_slots_stored` | slot sizes and `max_block` are load-time config |
| arena | 60 DMA-BUF slabs (`slab_fd_name(l, card)`), `verify_stored_slab` | spot check: first and last 64 bytes per tensor |
| slots | `set_slot`, `load_kv`, `load_kv_layer`; `kv_high` | no contexts, no stored token count, no fork |
| parts | `save_kv_layer_from` (a range read), `load_kv_layer` | KV only: cells on global layers, ring on sliding layers |
| geometry / coverage | slot size, ring rows (`kv_ring`, `kv_keep_from`); `KvSnapshot::join` is the fit check | changes rows held, not bits, so no `geometry_key`; today's id is a recipe (`held.rs:312-325`) |
| pass | `run_card0` then `run_card1`; `pipeline_stage` for jobs in flight on both cards | two stages |
| segments | `PipeJob`; `branch_block` with per-token `meta` | a branch block is atomic and is not a snapshot |
| inputs | `write_embeddings` puts rows into `Buf::X` | rows today, made from tokens on the host |
| class | `CardLock`, `GpuLock` | moves out of the engine into the nagaya scheduler |

## Decisions

Decided: D1, D3. Deferred to the first engine port: D2. Proposed, not yet reviewed with Amy: D4-D7.

**D1. Engines that span devices.** (a) The engine schedules its own stages and the nagaya sees one quantum per pass;
or (b) the engine declares stages and exports `stage_pass`, and the nagaya schedules each stage on its device.
(b) lets the scheduler fill an idle card, as mkviolet's low-priority path does today, and the same residual boundary
is what a model split across machines needs later. But (b) needs the full segment shape (positions, meta, logit rows,
a snapshot to restore, the slot) and must stop another job writing the slot between a block's two stages.
Decided 2026-10-17 (Amy: "yeah that sound reasonable"): mkviolet starts with (a); (b) comes once the segment shape
is settled, and first for splits across machines. Amy may stop running one model across both B70 cards and use
them independently, "since they're the fastest memory we have at home". Today's Gemma 4 does not fit on one card,
so per-card work means smaller engines (a smaller Gemma, "or probably a qwen and lfm"), each with one stage.

**D2. Inputs.** An engine declares tokens, rows, or both. mk needs token ids even when given rows, because its
per-layer embeddings (PLE) hash them and its PLE history is a fixed part (`engine.h:75`, `ple_rows`). gmk's audio rows
and diffusion latents have no tokens. Proposed: `inputs = tokens | rows | tokens+rows`, declared per engine.
Deferred 2026-10-17 (Amy: "sort it out with the first engine"): mkviolet's Gemma 4 is the first port, and it takes
rows made from tokens on the host today.

**D3. Snapshot ids are content ids.** Decided 2026-10-17 (Amy: "yeah, 1, cas is awesome"). Two keys:

- The **content id** names a stored snapshot: a BLAKE3 hash of its sealed bytes, and of its parent's content id when
  it is a piece of a chain (mkviolet holds only the rows after its parent). mk stores snapshots whole, so its ids
  have no parent. Dedup, integrity, pins, and moving snapshots between machines all use it.
- The **recipe** is a hash the nagaya computes, with the engine unloaded if need be, from the identity (bit settings
  included), the inputs and the boundaries, plus `geometry_key` for an engine whose bits depend on slot geometry.
  The store maps a recipe to one or more content ids, which answers "do we already have this?" before a build, and
  prefix lookups are hashing.

One recipe can have several content ids because coverage differs. In mkviolet, slot size decides which rows a
snapshot *holds*, not what their bits *are*: `kv_ring` and `kv_keep_from` choose the positions a slot keeps, and a
ring row holds the same K and V for its position as a full plane does (`gemma4.rs:44-63`). `join` fails on missing
rows, not on different bits. Each stored snapshot therefore records its coverage (positions held, per part), and
fit is a separate check (`coverage` capability). This corrects review finding 3, which put geometry in the id.

Two content ids under one recipe that cover the same positions mean the engine did not reproduce its bits, or the
recipe misses an input. The nagaya refuses to serve that recipe and reports it; it never picks one. Hashing an mk
snapshot whole (about 316 MB) costs tens of milliseconds at BLAKE3 speeds, small next to a pass.

**D4. Cost before a pass.** Neither engine estimates; both report time after. Proposed: no estimate call in v0; the
nagaya fits a cost model per engine from pass times and segment sizes.

**D5. Rust trait first, C ABI as a binding.** Handles are opaque pointers, errors are a status plus `last_error`,
buffers are caller-owned.

**D6. Poison stays fatal.** A poisoned context is freed, never read. A pass that returns a poison status poisons
every context in it (mk: "-1, every context poisoned"). The nagaya fails the jobs that held those segments and keeps
the others.

**D7. Seal and park are separate calls.** Seal runs only at boundaries in the chain and may change bits. Park is
`part_read` to another tier and back, bit for bit. The scheduler parks; it never seals to free a slot.

## Not covered yet

- Training: an engine that writes its weights state object (an arena that is not read-only), backward passes,
  optimizer state as fixed parts (full size at every snapshot unless stored another way), micro-batch quanta that
  filler work is preempted between.
- Image engines: a diffusion latent is a fixed part replaced at each step; there is no `n` to extend, and the outputs
  need a latent or image tap. mk's painter is the first case (mk `docs/painter.md`).
- Streams: period, deadline and late policy in `limits()` or per job; a pass is blocking, so a stream's deadline is
  met by scheduling, not by the engine.
- Unloading an engine while its arena stays staged in host memory for a fast swap back.
