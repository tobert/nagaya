# Engine ABI v0 (draft)

This doc defines the interface that each engine exports to the nagaya. `docs/ontology.md` defines the terms.

We drafted the ABI on 2026-10-17 from two sources:

- mk's C ABI (`src/engine/capi.hip`, `src/engine/engine.h`);
- mkviolet's Gemma 4 engine (`src/gemma4.rs`, `src/held.rs`, `src/protocol.rs`, `src/hot_handoff.rs`).

On the same day, a kaibo review (deepseek cast) checked the draft against copies of that source. The review's
headline: the first draft's call list was mk's `Context` model, and mkviolet, the second engine, already failed half
of it. Thus the ABI is now a small core that every engine has, plus capabilities that an engine declares.

v0 covers three kinds of engine: inference, image batch, and training (Amy, 2026-10-17).

## Shape

The nagaya is Rust. An engine is a Rust type that implements the `Engine` trait. Some engines use another language
(mk: C++ and HIP behind `extern "C"`). Such an engine exports a C ABI, and a thin Rust wrapper implements the trait
on top of it. mkviolet's Gemma 4 engine implements the trait directly.

Each engine runs on a **device group**: one device, or the set of devices that the engine spans. Each device group
has one nagaya thread. Calls for one engine come only from that thread, and they are synchronous. A pass blocks the
thread.

An engine that spans devices runs its own work for each device inside the pass. mk already works this way,
with a group of one device. mkviolet's group is both B70 cards; its `pipeline_stage` threads stay inside its pass.

### Rust shape

Decided 2026-10-18 (Amy chose it in an interview). The `Engine` trait holds the core calls. Each capability is its
own trait. `Engine` has one accessor for each capability, and the accessor returns `None` when the engine does not
have it. Thus an engine has no stub methods, and `limits().capabilities` comes from the accessors.

```rust
trait Engine {
    fn identity(&self) -> Identity;
    fn limits(&self) -> Limits;
    fn pass(&mut self, segments: &[Segment], taps: &[Tap]) -> Result<PassOut, PassError>;
    fn last_error(&self) -> Option<String>;
    fn parts(&mut self) -> Option<&mut dyn Parts> { None }
    fn slots(&mut self) -> Option<&mut dyn Slots> { None }
    fn seal(&mut self) -> Option<&mut dyn Seal> { None }
    // ... one accessor for each capability below
}

trait EngineFactory {
    fn plan(config: &toml::Table) -> Result<Limits, PlanError>;
    fn open(config: &toml::Table) -> Result<Box<dyn Engine>, OpenError>;
    fn attach(config: &toml::Table, arena_set: ArenaSet) -> Result<Box<dyn Engine>, OpenError>;
}
```

The config is a TOML table from the placement profile. The engine parses it and refuses a field that it does not
know. Thus the nagaya holds every engine as `Box<dyn Engine>`.

## Core (every engine)

```
plan(config) -> limits           reads no weights; the ledger admits open() from this result
open(config) -> engine           config: devices, weights, engine-specific settings (mkviolet: slot sizes, max_block)
attach(config, arena_set)        the same, on an existing arena set
identity() -> {
    model, checkpoint_digest,    the logical digest of the weights (spec 12.2)
    tokenizer_hash, template, numbers, plan table,
    bit settings: [{name, value, class}],   class: reference-defining | approx | policy
    canary_hash,
}
limits() -> {
    capabilities: [...],         which optional calls below this engine has
    parts: [{name, kind: fixed|cells|ring|derived, dtype, shape}],
                                 fixed: the whole shape; cells and ring: row_shape, one row per position
    inputs: tokens | rows | tokens+rows, row width and dtype,
    devices: [...],              the device group
    stages: [{device, layers, head: bool}],
    max_segments, max_pass_inputs, max_logit_rows, slots per device and their sizes,
    max_quantum_ms,              the longest a quantum may hold its device group (D8)
    memory: {                    the engine's ledger terms (spec 4.2), keyed by kind of quantum
        mandatory: [{name, tier, bytes, lifetime: engine | context | loan, kinds}],
        elastic: [{name, tier, unit_bytes, minimum, maximum, kinds}],
    },                           tier: device(i) | host; i is an index into the device group
    chain: {max_depth, delta_fraction},   how the store may chain this engine's snapshots (spec 6.5)
    segments_exact: bool,        segments that share a pass get the bits of a pass alone (spec 5.7)
    determinism: always | pinned | per-process,   and bit_dependence: [residency, ...] (D3)
    taps: [...], with the conditions each needs (mk: unfused, one segment)
}
pass(segments, taps) -> status, ms, outputs
    segment = {state, inputs, positions, meta, outputs wanted}
last_error() -> why
```

`limits()` after `open` must equal the result of `plan` for the same config. A difference is a defect, and the
nagaya refuses the engine.

`positions` and `meta` cover mkviolet's `pos0` and per-token `meta` (`PipeJob`, `gemma4.rs:998-1004`). `outputs
wanted` covers `logit_rows`. `state` is a context handle when the engine has contexts. It is a slot index when the
engine has slots.

`pass` never allocates. Its scratch is a mandatory term, keyed by the kinds of quantum that use it. These calls do
allocate, and the ledger admits each one before it runs: `open`, `attach`, `ctx_new`, `ctx_fork`, `ctx_copy`,
`part_write`, and `resize`. (`set_n` and `cells_write` do not allocate: mk sizes a context at `ctx_new(max)`.)

## Capabilities (declared in `limits()`)

```
arena        arena_export() -> {name: fd, arena_id, check: full | spot}
             arena_check(config, arena_set) -> ok | why, check: full | spot
             check strength is one enum for arenas and store entries: full | spot | none
contexts     ctx_new(max) / ctx_fork / ctx_copy / ctx_free / ctx_n
slots        slot_high(slot) / slot_clear(slot)
parts        part_read(state, part) / part_write(state, part, bytes)
             cells_read(state, part, from, n) / cells_write(state, part, n, bytes) / set_n(state, n)
seal         seal(state)                 the engine's boundary rule
geometry     geometry_key(state) -> bytes   slot geometry that changes bits, added to the recipe
coverage     coverage(part bytes) -> positions held; fits(coverage, slot) -> ok | why
             without it, a snapshot holds every position, 0 to n
resize       resize(term, units) -> ok | why   an elastic term, only between quanta (spec 4.5)
poison       poisoned(state) -> bool
pass_check   pass_problem(segments) -> ok | why   the engine's own check of a pass; not the ledger's admission
stages       stage_pass(stage, segments, residual_in) -> residual_out | outputs
io           io_submit(IO requests [(file, offset, len, dest)], class) -> ticket; io_wait(ticket, prefix | all)
             dest is a device-buffer handle that the engine gives; the nagaya's IO service does the reads
telemetry    op_times(), dispatches(), ...; pass(…, trace_context) so engine spans nest under the quantum span
train        train_pass(segments) -> status, ms, loss     a forward, a backward and an optimizer step
close        close()                     frees the engine in the process
```

An engine without `close` stays open until the process exits. The nagaya changes such an engine only through a
planned restart (spec 3.3, exit code 4). Its `Drop` does not free it, and a comment says why. Decided 2026-10-18.

`parts` works on any `state` (a context handle or a slot index). `part_write` and `cells_write` return one of three
results (spec 6.8):

- ok;
- refused, and nothing changed;
- failed after a change. The state is poisoned (D6).

`arena_id` is the content id of the packed bytes that the engine runs. The identity carries the checkpoint digest
instead, so a new packing does not change the identity.

## Proposed: v0 Rust shapes

**Proposed** by deepseek-flash, 2026-10-18. The ABI text names the calls and the fields, but it does not fix
the Rust types. The first code needs concrete types. This section gives the simplest shapes that fit the
decisions. Amy can change them later.

- `State` is `Slot(u32)` in v0. A context handle arrives with the `contexts` capability.
- `PartLimits` holds one `shape` vector. A fixed or derived part uses the whole shape. A cells or ring part
  uses its row shape.
- `SlotLimit` is `{bytes, count}`. The fake engine declares two sizes: 64 bytes and 256 bytes.
- `Capability` is an enum. `Limits::capabilities` is a `Vec<Capability>`. The fake engine declares `Parts`,
  `Coverage` and `Seal`.
- A write returns `WriteResult`: `Ok`, `Refused` or `Failed`. A read returns `Result<(), PartError>`.
- `Coverage` is one `{start, end}` position range for one part. A ring keeps the last rows that fit.
- `Seal::seal` returns `Result<(), SealError>` and changes bits in place.
- The first code defines the core, `Parts`, `Coverage`, `Seal` and `Slots`. Each other capability trait
  arrives with the milestone that uses it.
- The `EngineFactory` trait arrives with the process crate. The fake engines are constructed directly.
- The `fake` feature is enabled for this crate's tests by a dev-dependency on the crate itself. The plain
  house test runs the fake tests; a normal build leaves them out.

**Corrected** 2026-10-18: a default feature still let the tests be skipped. The dev-dependency ties the
feature to the test build. This is a plain correction to the Proposed section, not a new design choice.

### Fake engines

`FakeAlways` is deterministic (`always`). Its parts are:

- `state`, fixed, `U8`, shape [16];
- `cells`, cells, `U8`, row shape [8];
- `ring`, ring, `U8`, row shape [8], window 4 rows;
- `pooled`, derived, `U8`, row shape [8].

Its slots are 64 bytes and 256 bytes, one each. Its seal flips the low bit of the first `state` byte.
Its `pass` returns `Ok` with no outputs.
`FakeBroken` is the same, but `part_read` flips the low bit of byte 0. Thus a park and restore loses a bit,
and gate 8.3 (park and restore) must fail it.

## How mk maps

| ABI | mk (`capi.hip`) | Notes |
|---|---|---|
| open / attach | `mk_engine_open(s1, s2, s3)`, `mk_engine_attach(…, path)` | no device argument: the HIP device is the calling thread's current one, so the wrapper sets it on the group's thread; weights are three GGUF shards |
| plan | none yet | the wrapper computes the terms from the config and mk's fixed shapes |
| arena | `mk_engine_arena`, `mk_arena_check` | one fd; full check |
| identity | built in the Python service | moves into the engine; must include `fused` and `MK_FAST_QMM` |
| contexts | `mk_ctx_new`, `mk_ctx_fork`, `mk_ctx_copy`, `mk_ctx_free`, `mk_ctx_n` | a fork shares its parent's K/V and index keys |
| parts | `mk_ctx_state_read`, `mk_ctx_state_write`, `mk_ctx_cells_write`, `mk_ctx_set_n` | parts 0-2 and 5 fixed, 3-4 cells; `idx_pooled` derived. No `cells_read` range yet: snapshots are stored whole (about 316 MB at n = 3584, ontology "Part") |
| seal | `mk_ctx_round_gdn_bf16` | changes bits in place |
| poison / last_error | `mk_ctx_poisoned`, `mk_last_error` | |
| part writes | `mk_ctx_state_write`, `mk_ctx_cells_write`, `mk_ctx_set_n` | return nothing and abort on input they refuse; the wrapper checks every precondition first (part index, bytes, an empty context that owns its K/V for parts 3-5, n within `max`, not poisoned) and returns refused; after the call, `mk_ctx_poisoned` gives failed (decided 2026-10-18) |
| close | none | `mk_engine_close` exists, but HIP teardown segfaults in the process; mk does not declare `close` |
| attach path | `mk_engine_attach(…, path)` | the path is `/proc/self/fd/<n>` for an fd from systemd's store |
| pass_check / pass | `mk_segments_problem`, `mk_pass_segments` | tokens only (PLE needs token ids); outputs are logits and hidden per segment; `mk_pass_segments` aborts the process on bad segments (`MK_DIE`), so the wrapper calls `mk_segments_problem` first |
| taps | `mk_pass` (layers, routes), `mk_pass_gdn_ab` (gates) | unfused, one segment; logits go straight to the caller's buffer, the other taps are copied |
| telemetry | `mk_stage_times`, `mk_stage_detail`, `mk_dispatches`, `mk_host_stats` | mk's stages are ops (`stage.h`), so these are op times |
| determinism | `always` | `stage.h`: "every op's output bits depend only on its work items"; top-k ties break by index; bit settings in the identity |

## How mkviolet maps

| ABI | mkviolet (`gemma4.rs` unless named) | Notes |
|---|---|---|
| open / attach | `load_slots(cluster, model_dir, slot_seqs, max_block)`, `load_slots_stored` | slot sizes and `max_block` are load-time config |
| device group | both B70 cards | one nagaya thread; `run_card0`, `run_card1` and `pipeline_stage` stay inside the pass |
| arena | 60 DMA-BUF slabs (`slab_fd_name(l, card)`), `verify_stored_slab` | spot check: first and last 64 bytes per tensor |
| slots | `set_slot`, `kv_high`, `slot_clear` | no contexts, no stored token count, no fork |
| parts | `save_kv_layer_from` (a range read), `load_kv`, `load_kv_layer` | `state` is the slot index; KV only: cells on global layers, ring on sliding layers |
| geometry / coverage | slot size, ring rows (`kv_ring`, `kv_keep_from`); `KvSnapshot::join` is the fit check | changes rows held, not bits, so no `geometry_key`; today's id is a recipe (`held.rs:312-325`) |
| pass | `run_card0` then `run_card1`; `pipeline_stage` for jobs in flight on both cards | two stages |
| segments | `PipeJob`; `branch_block` with per-token `meta` | a branch block is atomic and is not a snapshot |
| inputs | `write_embeddings` puts rows into `Buf::X` | rows today, made from tokens in the nagaya |
| class | `CardLock`, `GpuLock` | moves out of the engine into the nagaya scheduler |
| determinism | not yet known | the first port runs one recipe in two starts and compares content ids |

## Decisions

Decided: D1, D3-D10. D2 is narrowed and waits for the first port.

**D1. Engines that span devices.** There are two options:

- (a) The engine schedules its own stages. The nagaya sees one quantum per pass.
- (b) The engine declares stages and exports `stage_pass`. The nagaya schedules each stage on its device.

Option (b) lets the scheduler fill an idle card, as mkviolet's low-priority path does today. A model split across
machines needs the same residual boundary later. But (b) needs the full segment shape: positions, meta, logit rows,
a snapshot to restore, and the slot. It must also stop another job from writing the slot between a block's two
stages.

Decided 2026-10-17 (Amy: "yeah that sound reasonable"): mkviolet starts with (a). Option (b) comes when the segment
shape is settled, and first for splits across machines.

Under (a), the engine's devices are one device group with one nagaya thread (Shape). Decided 2026-10-17: "one thread
per device group".

Amy may stop running one model across both B70 cards and use them independently, "since they're the fastest memory
we have at home". Today's Gemma 4 does not fit on one card. Thus per-card work means smaller engines, each with a
group of one card: a smaller Gemma, "or probably a qwen and lfm".

**D2. Inputs.** An engine declares tokens, rows, or both. mk needs token ids even when it gets rows. Its per-layer
embeddings (PLE) hash the token ids, and its PLE history is a fixed part (`engine.h:75`, `ple_rows`). gmk's audio rows
and diffusion latents have no tokens. Proposed: `inputs = tokens | rows | tokens+rows`, declared per engine.

Deferred 2026-10-17 (Amy: "sort it out with the first engine"). mkviolet's Gemma 4 is the first port. Today it takes
rows that the nagaya makes from tokens.

Settled the same day: an engine never takes text. The family library in the nagaya turns text into tokens (spec
11.3). Only the choice between tokens, rows, or both waits for the first port.

**D3. Snapshots are named by content ids.** Decided 2026-10-17 (Amy: "yeah, 1, cas is awesome"). There are two keys:

- The **content id** names a stored snapshot: the BLAKE3 hash of its sealed bytes (`blake3:<hex>` in council v2).
  When the snapshot is a piece of a chain, the hash also covers its parent's content id. (mkviolet holds only the
  rows after the parent.) mk stores snapshots whole, so its ids have no parent. Dedup, integrity, pins, and moves of
  snapshots between machines all use the content id.
- The **recipe** is a hash that the nagaya computes, also when the engine is not loaded. It covers the identity
  (bit settings included), the inputs and the boundaries. For an engine whose bits depend on slot geometry, it also
  covers `geometry_key`. Spec 6.2 holds the full list of items and the encoding. The store maps a recipe to one or
  more content ids. Thus the store can answer "do we
  already have this?" before a build, and a prefix lookup is only hashing.

One recipe can have several content ids, because coverage differs. In mkviolet, slot size decides which rows a
snapshot *holds*, not what their bits *are*. `kv_ring` and `kv_keep_from` choose the positions that a slot keeps. A
ring row holds the same K and V for its position as a full plane does (`gemma4.rs:44-63`). `join` fails on missing
rows, not on different bits.

Thus each stored snapshot records its coverage (positions held, per part), and fit is a separate check (`coverage`
capability). This corrects review finding 3, which put geometry in the id.

Two content ids under one recipe that cover the same positions have two possible causes. The engine did not
reproduce its bits, or the recipe misses an input. The nagaya refuses to serve that recipe. It quarantines the
conflicting entries and reports them. It never picks one. A person clears the refusal after the cause is known.

A BLAKE3 hash of a whole mk snapshot (about 316 MB) takes tens of milliseconds, small next to a pass.

That refusal is a tripwire. It has a meaning only when the engine promises the same bits. Amended 2026-10-17 (Amy:
"having some tolerance discernment makes sense. yeah go ahead"): each engine declares which promise it makes.

| `determinism` | Promise | The refusal | Reuse of a snapshot |
|---|---|---|---|
| `always` | the same identity and inputs give the same bits in any process | on: a mismatch is a bug | exact |
| `pinned` | the same, because every free evaluation choice comes from the engine's plan table (ontology) | on | exact while the plan table's lineage holds |
| `per-process` | the same bits within one process only | within one process only | exact within the process; across processes only for a served model whose tolerance accepts it (ontology) |

Some engines have a history that changes their bits (ds4-recipe's routing reads which experts are cached). Such an
engine also declares `bit_dependence: residency`. Then these rules apply:

- the nagaya claims no identical bits under interleaving for it;
- quantum condition 7 does not hold;
- the refusal is off;
- a memory term whose size changes its bits is mandatory, not elastic, so reclaim cannot change it.

Engines we build are `always` or `pinned`, with no bit dependence.

When the refusal is off, an **audit** takes its place. Decided 2026-10-17: the audit runs "across restarts". After
each start, a filler job in the nagaya builds a sample of stored recipes again and compares the content ids. The new
start is the fresh process, so there is no second GPU client. The audit reports each mismatch and uses only idle
time. ds4-recipe does this by hand ("pinned plans for every experiment and byte-equality gate").

**D4. Cost before a pass.** Decided 2026-10-18. Neither engine estimates a cost, and both report the time after the
pass. There is no estimate call in v0. The nagaya fits a cost model per engine from pass times and segment sizes. It
uses the fit for store eviction value-per-byte and deadline checks between quanta. The engine picks its own chunk
sizes for `max_quantum_ms`.

**D5. Rust trait first, C ABI as a binding.** Decided 2026-10-18. Handles are opaque pointers. Errors are a status
plus `last_error`. The caller owns the buffers.

**D6. Poison stays fatal.** Decided 2026-10-18. The nagaya frees a poisoned context and never reads it. For a slot
engine, it marks a poisoned slot dirty until `slot_clear`, `part_write`, or a pass from position 0 resets it. A pass
that returns a poison status poisons every context or slot in it (mk: "-1, every context poisoned"). The nagaya fails
those jobs, keeps the others, and runs the canary before more work runs (spec 8.2).

**D7. Seal and park are separate calls.** Decided 2026-10-18. Seal runs only at boundaries in the chain, and it can
change bits. Park uses the `parts` calls to move state between tiers bit for bit, for both contexts and slots. The
scheduler parks. It never seals to free a slot.

**D8. Quanta have a declared length bound.** Decided 2026-10-17 (Amy: "a lil head of line blocking is okay in this
design. max_quantum_ms seems like a good call"). An engine declares `max_quantum_ms` in `limits()`. The engine
cannot know a pass's time before it runs. Thus it splits long work into chunks whose size it measures to fit the
bound (mk: 256-token prefill chunks). In v0 there are no yields inside a pass. A quantum still runs to its end, so
the worst wait behind other work is one quantum.

A watchdog thread reports each quantum that runs past the bound.

ds4-recipe chose cooperative yields after a short turn waited 283.95 s behind a long prefill (`docs/prior-art.md`
3). We take the bounded wait instead.

**D9. Spare memory goes by share.** Decided 2026-10-17 (Amy: "we'll assign some weight per machine"). Each
machine's placement profile gives every engine on it a share. The ledger first charges all mandatory bytes. It then
divides what remains among the engines' elastic terms by those shares (`docs/prior-art.md` 2). Each elastic term
gets at least its minimum. A profile whose mandatory bytes and minimums do not fit is refused at start.

**D10. Snapshots at rest are canonical safetensors.** Decided 2026-10-17 (Amy: "yeah, seems like a good call"). In
device memory, each engine keeps its own layout. The model and the kernels decide that layout, and it stays private.

In the store, every snapshot is a safetensors file in one canonical form. Thus any tool can read any engine's parts
without that engine's code: numpy, a trace viewer, a lens job, the audit. The HF checkpoints are safetensors too, so
weights and state at rest use one format. These rules apply:

1. **One part, one tensor.** Each part is a tensor with the name and dtype that `limits().parts` declares. The data
   is little-endian and row-major, with no engine padding.
2. **Coverage sets the shape.** A cells or ring part holds the rows of its coverage, in position order. A ring is
   unrolled into position order. Slot capacity never sets a shape. Thus equal coverage gives equal bytes, from any
   slot size.
3. **A canonical header.** Keys are sorted. The JSON has no whitespace except the padding at its end (rule 5).
   The tensors are in bytewise order of their names, in the header and in the data. A derived part is never
   stored. `__metadata__` holds only facts about the bytes: the part kinds, the coverage, the parent's content id,
   and each block format. It never holds times, hit counts or owners (spec 6.3).
4. **Dtypes.** Use a native dtype where one exists: F32, F16, BF16, the integer types, and F8 (E4M3, E5M2). A block
   format that safetensors does not have (IQ4_NL, MXFP4, FP4 KV) is `U8`, and `__metadata__` names the format.
5. **Alignment.** Pad the JSON header with spaces (0x20), so that the 8-byte length and the header end on a 4 KiB
   boundary. Thus the data starts on a 4 KiB boundary. v0 puts no padding between tensors.
6. **Chains.** A delta is a safetensors file. For each cells and ring part, it holds the rows after its parent's
   end. It holds each fixed part whole. Its metadata names its parent's content id and the rows that it covers.
7. **Shapes.** A fixed part has the shape that `limits().parts` declares. A cells or ring part has the shape
   `[rows held, ...row_shape]`.

Coverage is in `__metadata__`, so it is in the content id. The same rows at other positions are a different state,
because a position changes the bits that later passes read (RoPE). Fit to a slot is not in the content id.

Decided 2026-10-18: name order, 4 KiB data start, shapes, and coverage in the id.

The content id is the BLAKE3 hash of this canonical file. Each engine converts between its layout and the canonical
form on the way into and out of the store. For mk, that is close to the copy it already does: its copy re-strides
cells by capacity (`engine.h:169`). For mkviolet, the conversion unrolls each ring into position order.

A park from device memory to host memory can keep the engine's native bytes, if the conversion is too slow for short
parks. Such a park has no content id. It gets one when it goes into the store. **Open:** measure the conversion on
both engines before we use this exception.

## Training

Decided 2026-10-18. A training engine declares the `train` capability:

- The weights are a state object that the engine writes. The engine opens its arena read-write.
- A `train` quantum is one micro-batch (`train_pass`): a forward pass, a backward pass, and an optimizer step.
- The optimizer state and the weights are fixed parts in `limits().parts`. They are the same size at every snapshot,
  unless the engine stores them another way.
- A training checkpoint is a snapshot of the weights and the optimizer parts through `part_read` and `part_write` into
  the store (D10).
- The job can resume at every quantum boundary. Training is filler: it gives way between micro-batch quanta, and it
  declares its checkpoint interval (spec 5.5).
- Atomic reductions in a backward pass make training `per-process` at best. An engine that pins its reductions can
  declare `pinned`.

## Image

Decided 2026-10-18. An image engine uses `pass(segments, taps)` with quantum kind `step`:

- A diffusion latent is a fixed part. Each `step` quantum replaces it. There is no `n` to extend.
- The outputs are a latent tap or an image tap.
- The seed is an input, so it is in the recipe.
- mk's painter is the first case (mk `docs/painter.md`). mk's `YieldHook` showed that a paused run matched the
  unpaused one, max pixel difference 0 (ontology "Quantum").

## Not covered yet

- Streams. They need a period, a deadline and a late policy, in `limits()` or per job. A pass blocks, so the
  scheduler meets a stream's deadline, not the engine.
- Unloading an engine while its arena stays staged in host memory for a fast swap back.
