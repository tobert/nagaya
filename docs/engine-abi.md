# Engine ABI v0 (draft)

This doc defines the interface that each engine exports to the nagaya. `docs/ontology.md` defines the terms.

We drafted the ABI on 2026-10-17 from two sources:

- mk's C ABI (`src/engine/capi.hip`, `src/engine/engine.h`);
- mkviolet's Gemma 4 engine (`src/gemma4.rs`, `src/held.rs`, `src/protocol.rs`, `src/hot_handoff.rs`).

On the same day, a kaibo review (deepseek cast) checked the draft against copies of that source. The review's
headline: the first draft's call list was mk's `Context` model, and mkviolet, the second engine, already failed half
of it. Thus the ABI is now a small core that every engine has, plus capabilities that an engine declares.

## Shape

The nagaya is Rust. An engine is a Rust type that implements the `Engine` trait. Some engines use another language
(mk: C++ and HIP behind `extern "C"`). Such an engine exports a C ABI, and a thin Rust wrapper implements the trait
on top of it. mkviolet's Gemma 4 engine implements the trait directly.

Each device has one nagaya thread. Calls for one device come only from that thread, and they are synchronous. A
pass blocks the thread. mk already works this way. mkviolet does not yet: `pipeline_stage` spawns a thread per card
per stage. That work moves into the nagaya's device threads.

## Core (every engine)

```
open(config) -> engine           config: devices, weights, engine-specific settings (mkviolet: slot sizes, max_block)
attach(config, arena_set)        the same, on an existing arena set
identity() -> {model, weights_hash, tokenizer_hash, template, numbers, plan table, bit settings}
limits() -> {
    capabilities: [...],         which optional calls below this engine has
    parts: [{name, kind: fixed|cells|ring|derived, bytes(n)}],
    inputs: tokens | rows | tokens+rows, row width and dtype,
    stages: [{device, layers, head: bool}],
    max_segments, max_pass_inputs, max_logit_rows, slots per device and their sizes,
    max_quantum_ms,              the longest a quantum may hold its device (D8)
    memory: {mandatory, elastic},  the engine's ledger terms (spec 4.2)
    batch_exact: bool,           segments that share a pass get the bits of a pass alone (spec 5.7)
    determinism: always | pinned | per-process,   and bit_dependence: [residency, ...] (D3)
    taps: [...], with the conditions each needs (mk: unfused, one segment)
}
pass(segments, taps) -> status, ms, outputs
    segment = {state, inputs, positions, meta, outputs wanted}
last_error() -> why
close()
```

`positions` and `meta` cover mkviolet's `pos0` and per-token `meta` (`PipeJob`, `gemma4.rs:998-1004`). `outputs
wanted` covers `logit_rows`. `state` is a context handle when the engine has contexts. It is a slot index when the
engine has slots.

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
resize       resize(term, units) -> ok | why   an elastic term, only between quanta (spec 4.5)
poison       poisoned(state) -> bool
admission    pass_problem(segments) -> ok | why
stages       stage_pass(stage, segments, residual_in) -> residual_out | outputs
telemetry    op_times(), dispatches(), ...
```

`slot_load` and `part_write` return one of three results (spec 6.8):

- ok;
- refused, and nothing changed;
- failed after a change. The state is poisoned (D6).

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
| determinism | `always` | `stage.h`: "every op's output bits depend only on its work items"; top-k ties break by index; bit settings in the identity |

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
| determinism | not yet known | the first port runs one recipe in two processes and compares content ids |

## Decisions

Decided: D1, D3, D8, D9, D10. Deferred to the first engine port: D2. Proposed, not yet reviewed with Amy: D4-D7.

**D1. Engines that span devices.** There are two options:

- (a) The engine schedules its own stages. The nagaya sees one quantum per pass.
- (b) The engine declares stages and exports `stage_pass`. The nagaya schedules each stage on its device.

Option (b) lets the scheduler fill an idle card, as mkviolet's low-priority path does today. A model split across
machines needs the same residual boundary later. But (b) needs the full segment shape: positions, meta, logit rows,
a snapshot to restore, and the slot. It must also stop another job from writing the slot between a block's two
stages.

Decided 2026-10-17 (Amy: "yeah that sound reasonable"): mkviolet starts with (a). Option (b) comes when the segment
shape is settled, and first for splits across machines.

Amy may stop running one model across both B70 cards and use them independently, "since they're the fastest memory
we have at home". Today's Gemma 4 does not fit on one card. Thus per-card work means smaller engines, each with one
stage: a smaller Gemma, "or probably a qwen and lfm".

**D2. Inputs.** An engine declares tokens, rows, or both. mk needs token ids even when it gets rows. Its per-layer
embeddings (PLE) hash the token ids, and its PLE history is a fixed part (`engine.h:75`, `ple_rows`). gmk's audio rows
and diffusion latents have no tokens. Proposed: `inputs = tokens | rows | tokens+rows`, declared per engine.

Deferred 2026-10-17 (Amy: "sort it out with the first engine"). mkviolet's Gemma 4 is the first port. Today it takes
rows that the host makes from tokens.

**D3. Snapshot ids are content ids.** Decided 2026-10-17 (Amy: "yeah, 1, cas is awesome"). There are two keys:

- The **content id** names a stored snapshot. It is a BLAKE3 hash of the snapshot's sealed bytes. When the snapshot
  is a piece of a chain, the hash also covers its parent's content id. (mkviolet holds only the rows after the
  parent.) mk stores snapshots whole, so its ids have no parent. Dedup, integrity, pins, and moves of snapshots
  between machines all use the content id.
- The **recipe** is a hash that the nagaya computes, also when the engine is not loaded. It covers the identity
  (bit settings included), the inputs and the boundaries. For an engine whose bits depend on slot geometry, it also
  covers `geometry_key`. The store maps a recipe to one or more content ids. Thus the store can answer "do we
  already have this?" before a build, and a prefix lookup is only hashing.

One recipe can have several content ids, because coverage differs. In mkviolet, slot size decides which rows a
snapshot *holds*, not what their bits *are*. `kv_ring` and `kv_keep_from` choose the positions that a slot keeps. A
ring row holds the same K and V for its position as a full plane does (`gemma4.rs:44-63`). `join` fails on missing
rows, not on different bits.

Thus each stored snapshot records its coverage (positions held, per part), and fit is
a separate check (`coverage` capability). This corrects review finding 3, which put geometry in the id.

Two content ids under one recipe that cover the same positions have two possible causes. The engine did not
reproduce its bits, or the recipe misses an input. The nagaya refuses to serve that recipe and reports it. It never
picks one. A BLAKE3 hash of a whole mk snapshot (about 316 MB) takes tens of milliseconds, small next to a pass.

That refusal is a tripwire. It has a meaning only when the engine promises the same bits. Amended 2026-10-17 (Amy:
"having some tolerance discernment makes sense. yeah go ahead"): each engine declares which promise it makes.

| `determinism` | Promise | The refusal | Reuse of a snapshot |
|---|---|---|---|
| `always` | the same identity and inputs give the same bits in any process | on: a mismatch is a bug | exact |
| `pinned` | the same, because every free evaluation choice comes from the engine's plan table (ontology) | on | exact while the plan table's lineage holds |
| `per-process` | the same bits within one process only | within one process only | exact within the process; across processes only for a request that accepts it (tolerance, ontology) |

Some engines have a history that changes their bits (ds4-recipe's routing reads which experts are cached). Such an
engine also declares `bit_dependence: residency`. Then these rules apply:

- the nagaya claims no identical bits under interleaving for it;
- quantum condition 7 does not hold;
- the refusal is off.

Engines we build are `always` or `pinned`, with no bit dependence.

When the refusal is off, an **audit** takes its place. The audit is a filler job. From time to time, it builds a
stored recipe again in a fresh process, under a pinned plan table and exact routing. It compares the content ids and
reports. It uses only idle time. ds4-recipe does this by hand ("pinned plans for every experiment and byte-equality
gate").

**D4. Cost before a pass.** Neither engine estimates a cost. Both report the time after the pass. Proposed: no
estimate call in v0. The nagaya fits a cost model per engine from pass times and segment sizes.

**D5. Rust trait first, C ABI as a binding.** Handles are opaque pointers. Errors are a status plus `last_error`.
The caller owns the buffers.

**D6. Poison stays fatal.** The nagaya frees a poisoned context and never reads it. A pass that returns a poison
status poisons every context in it (mk: "-1, every context poisoned"). The nagaya fails the jobs that held those
segments and keeps the others.

**D7. Seal and park are separate calls.** Seal runs only at boundaries in the chain, and it can change bits. Park is
`part_read` to another tier and back, bit for bit. The scheduler parks. It never seals to free a slot.

**D8. Quanta have a declared length bound.** Decided 2026-10-17 (Amy: "a lil head of line blocking is okay in this
design. max_quantum_ms seems like a good call"). An engine declares `max_quantum_ms` in `limits()`. When a pass
would run longer, the engine splits it into shorter quanta. In v0 there are no yields inside a pass. A quantum
still runs to its end, so the worst wait behind other work is one quantum.

ds4-recipe chose cooperative yields after
a short turn waited 283.95 s behind a long prefill (`docs/prior-art.md` 3). We take the bounded wait instead.

**D9. Spare memory goes by weight.** Decided 2026-10-17 (Amy: "we'll assign some weight per machine"). Each
machine's placement profile gives every engine on it a weight. The ledger first charges all mandatory bytes. It
then divides what remains among the engines' elastic terms by those weights (`docs/prior-art.md` 2).

**D10. Snapshots at rest are canonical safetensors.** Decided 2026-10-17 (Amy: "yeah, seems like a good call"). In
device memory, each engine keeps its own layout. The model and the kernels decide that layout, and it stays private.

In the store, every snapshot is a safetensors file in one canonical form. Thus any tool can read any engine's parts
without that engine's code: numpy, a trace viewer, a lens job, the audit. The HF checkpoints are safetensors too, so
weights and state at rest use one format. These rules apply:

1. **One part, one tensor.** Each part is a tensor with the name, dtype and shape that `limits().parts` declares.
   The data is little-endian and row-major, with no engine padding.
2. **A canonical header.** Keys are sorted, the JSON has no extra whitespace, and the tensors are in a fixed order.
   `__metadata__` holds only facts about the bytes: the part kinds, the coverage, the parent's content id, and each
   block format. It never holds times, hit counts or owners (spec 6.3).
3. **Dtypes.** Use a native dtype where one exists: F32, F16, BF16, the integer types, and F8 (E4M3, E5M2). A block
   format that safetensors does not have (IQ4_NL, MXFP4, FP4 KV) is `U8`, and `__metadata__` names the format.
4. **Alignment.** Safetensors permits no gaps between tensors. Pad the header to 4 KiB. Where a tensor must start
   on a 4 KiB boundary for direct IO, put an explicit `U8` padding tensor before it.
5. **Chains.** A delta is a safetensors file. It holds the new rows of each cells part, and each fixed and ring part
   whole. Its metadata names its parent's content id and the rows that it covers.

The content id is the BLAKE3 hash of this canonical file. Each engine converts between its layout and the canonical
form on the way into and out of the store. For mk, that is the copy it already does: its copy re-strides cells by
capacity (`engine.h:169`). For mkviolet, the conversion reorders the rows of each ring.

A park from device memory to host memory can keep the engine's native bytes, if the conversion is too slow for short
parks. Such a park has no content id. It gets one when it goes into the store. **Open:** measure the conversion on
both engines before we use this exception.

## Not covered yet

- Training. An engine writes its weights state object, so its arena is not read-only. It also needs backward
  passes and optimizer state as fixed parts. That state is full size at every snapshot, unless the engine stores it
  another way. Filler work gives way between micro-batch quanta.
- Image engines. A diffusion latent is a fixed part that each step replaces. There is no `n` to extend, and the
  outputs need a latent or image tap. mk's painter is the first case (mk `docs/painter.md`).
- Streams. They need a period, a deadline and a late policy, in `limits()` or per job. A pass blocks, so the
  scheduler meets a stream's deadline, not the engine.
- Unloading an engine while its arena stays staged in host memory for a fast swap back.
