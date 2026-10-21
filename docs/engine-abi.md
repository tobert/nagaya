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

Each engine runs on a **device group**: the set of devices that one or more engines use. Each device group has one
nagaya thread. Calls for the engines in the group come only from that thread, and they are synchronous. A pass
blocks the thread.

An engine that spans devices declares stages (D1). The group then has one stage thread for each device of its
stages, and each stage quantum runs on the thread of its device. mk has a group of one device and no stages.
mkviolet's group is both B70 cards, with two stages.

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

trait Factory {                  // in the `nagaya` crate, crates/nagaya/src/admission.rs
    fn plan(&self, config: &toml::Table) -> Result<Limits, PlanError>;
    fn open(&self, config: &toml::Table) -> Result<Box<dyn Engine>, OpenError>;
    // attach(config, arena_set) arrives with arenas through a restart (spec 10.2); see the next section
}
```

The config is a TOML table from the placement profile. The engine parses it and refuses a field that it does not
know. Thus the nagaya holds every engine as `Box<dyn Engine>`.

**The factory gets its devices.** **Proposed** (claude-opus-5-5, 2026-10-21). Today `open` takes only the config
(`crates/nagaya/src/admission.rs:73`). mk uses the HIP device that is current on the calling thread.

```rust
struct GroupDevice { id: DeviceId, name: String, pci_address: String, memory_bytes: u64 }

trait Factory {
    fn plan(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Limits, PlanError>;
    fn open(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Box<dyn Engine>, OpenError>;
    fn attach(&self, config: &toml::Table, devices: &[GroupDevice], arenas: ArenaSet)
        -> Result<Box<dyn Engine>, OpenError>;
}
```

- `devices` holds the machine devices of the engine, in the order of the profile's `devices` list. The position in
  the slice is the group index (spec 2.4). `id` is the machine device id.
- The inventory keeps the PCI address of each device for this. Today `Machine::inventory` drops it
  (`crates/nagaya/src/probe.rs:76-81`).
- A Level Zero engine maps each PCI address to its Level Zero device with `zeDevicePciGetPropertiesExt`. It refuses
  an address that it does not find, and an address that two devices give. mkviolet takes the cards in the order of
  `zeDeviceGet` (`L0Cluster::init`, `level_zero.rs:745-790`), and that order can differ from the PCI order.
- A HIP engine selects its device with `hipDeviceGetByPCIBusId` on the thread of its device.
- An engine that gets the wrong count of devices refuses with `PlanError`.

Reason: the profile names the devices, so the engine must run on those devices and no others.

## Core (every engine)

```
plan(config) -> limits           reads no weights; the ledger admits open() from this result
open(config) -> engine           config: devices, weights, engine-specific settings (mkviolet: slot sizes, max_block)
attach(config, arena_set)        the same, on an existing arena set
identity() -> {
    model, checkpoint_digest,    the logical digest of the weights (spec 12.2)
    tokenizer_hash, template, family, numbers, plan table,   family: Proposed (spec 11.3)
    bit settings: [{name, value, class}],   class: reference-defining | approx | policy
    canary_hash,
}
limits() -> {
    capabilities: [...],         which optional calls below this engine has
    parts: [{name, kind: fixed|cells|ring|derived, dtype, shape}],
                                 fixed: the whole shape; cells and ring: row_shape, one row per position
    inputs: tokens | rows | tokens+rows, row width and dtype,
    devices: [...],              the device group
    stages: [{device, layers, head: bool}],   device is a group index (spec 2.4); residual_row_bytes for a boundary
    max_segments, max_pass_inputs, max_logit_rows, slots per device and their sizes,
    max_quantum_ms,              the longest a quantum may hold its device group (D8)
    memory: {                    the engine's ledger terms (spec 4.2), keyed by kind of quantum
        mandatory: [{name, tier, bytes, lifetime: engine | context | loan, kinds, pinned?}],   pinned: spec 2.2
        elastic: [{name, tier, unit_bytes, minimum, maximum, kinds}],
    },                           tier: device(i) | host; i is an index into the device group
    chain: {max_depth, delta_fraction},   how the store may chain this engine's snapshots (spec 6.5)
    segments_exact: bool,        segments that share a pass get the bits of a pass alone; no sum's order
                                 depends on the segment count, the row count or the unit count (spec 5.7)
    checkpoint_grain: tokens?,   the scheduler makes an interval copy at each multiple (spec 6.13)
    feed: {block: tokens}?,      the feed rule (D11): each span between boundaries goes in passes of `block`
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
             cells_read(state, part, from, n) / cells_write(state, part, from, n, bytes) / set_n(state, n)
seal         seal(state)                 the engine's boundary rule
geometry     geometry_key(state) -> bytes   slot geometry that changes bits, added to the recipe
coverage     coverage(part bytes) -> positions held; fits(coverage, slot) -> ok | why
             without it, a snapshot holds every position, 0 to n
resize       resize(term, units) -> ok | why   an elastic term, only between quanta (spec 4.5)
poison       poisoned(state) -> bool
pass_check   pass_problem(segments) -> ok | why   the engine's own check of a pass; not the ledger's admission
stages       stage_pass(stage, segments, residual_in) -> residual_out | outputs   (D1)
io           io_submit(IO requests [(file, offset, len, dest)], class) -> ticket; io_wait(ticket, prefix | all)
             dest is a device-buffer handle that the engine gives; the nagaya's IO service does the reads
telemetry    op_times(), dispatches(), ...; pass(…, trace_context) so engine spans nest under the quantum span
train        train_pass(segments) -> status, ms, loss     a forward, a backward and an optimizer step
close        close()                     frees the engine in the process
```

An engine without `close` stays open until the process exits. The nagaya changes such an engine only through a
planned restart (spec 3.3, exit code 4). Its `Drop` does not free it, and a comment says why. Decided 2026-10-18.

**Per-part coverage.** **Proposed** (claude-opus-5-5, 2026-10-21). This closes the old note: the park used one row
count, `slot_high`, for every row part of a slot.

- The park and the staging charge (spec 4.2) read `coverage(state, part)` for each cells part and each ring part.
  They store and count the rows of that coverage.
- `slot_high` stays the token count of the slot. A count of 0 means that the slot holds no token (spec 10.3).
- An engine whose row parts hold different rows must declare `coverage`. mkviolet's do (below).

Reason: in a long slot, a sliding layer of Gemma 4 holds the last 4,096 positions, and a global layer holds every
position. One count cannot give both.

`parts` works on any `state` (a context handle or a slot index). `part_write` and `cells_write` return one of three
results (spec 6.8):

- ok;
- refused, and nothing changed;
- failed after a change. The state is poisoned (D6).

`arena_id` is the content id of the packed bytes that the engine runs. The identity carries the checkpoint digest
instead, so a new packing does not change the identity.

### The first row of a cells write

Decided 2026-10-20 (reading). `cells_write` takes `from`, the position of its first row. The call
matches `cells_read(state, part, from, n, out)`. A Cells part appends from 0, so its write names the number of rows it
holds; the first write names 0. A Ring part may name a later position, because its window moves.

An engine that cannot place the rows at `from` returns `WriteResult::Refused` and changes nothing. It must not move
the rows to another position.

Reason: without `from`, a ring that held positions 2 to 6 came back from the store at positions 0 to 4. That is a
silent loss of state (spec 8.5).

## v0 Rust shapes

Decided 2026-10-20 (reading). The ABI text names the calls and the fields, but it does not fix the Rust types.
This section gives the simplest shapes that fit the decisions.

- `State` is `Slot(u32)` in v0. A context handle arrives with the `contexts` capability.
- `PartLimits` holds one `shape` vector. A fixed or derived part uses the whole shape. A cells or ring part
  uses its row shape.
- `SlotLimit` is `{bytes, count}`. The fake engine declares two sizes: 64 bytes and 256 bytes.
- `Capability` is an enum. `Limits::capabilities` is a `Vec<Capability>`. The fake engine declares `Parts`,
  `Coverage`, `Seal` and `Slots`.
- A write returns `WriteResult`: `Ok`, `Refused` or `Failed`. A read returns `Result<(), PartError>`.
- `Coverage` is one `{start, end}` position range for one part. The call takes the `State` too, so a ring
  reports the positions that it holds. A ring keeps the last rows that fit.
- `Seal::seal` returns `Result<(), SealError>` and changes bits in place.
- The first code defines the core, `Parts`, `Coverage`, `Seal` and `Slots`. Each other capability trait
  arrives with the milestone that uses it.
- The `Factory` trait is in the process crate. The fake engines are constructed directly.
- `Dtype` has the native dtypes of D10 rule 4: F32, F16, BF16, F8E4M3, F8E5M2, I8, I16, I32, I64, U8, U16, U32 and
  U64. `Dtype::element_bytes` gives the size of one element, and every crate uses it. Amy, 2026-10-20: "add them
  now, they should be designed in early even if used late".
- `PartLimits` has `block_format: Option<String>` ("Canonical file bytes"). An `Output` has a `dtype`, so a
  compare of outputs can find NaN and Inf (spec 8.3).
- `Limits::checkpoint_grain` is `Option<u64>`, in tokens. `None` means that the engine makes no interval copies
  (spec 6.13).
- The `fake` feature is enabled for this crate's tests by a dev-dependency on the crate itself. The plain
  house test runs the fake tests; a normal build leaves them out.

**Corrected** 2026-10-18: a default feature still let the tests be skipped. The dev-dependency ties the
feature to the test build. This is a plain correction, not a new design choice.

**Amended** by deepseek-flash, 2026-10-18: `Coverage::coverage` takes the `State`. The ring's base position
is in the state, not in the part bytes. The old call reported 0..n after a wrap.

### A pass that did not run

**A refused pass returns an error.** A pass that the mk binding refuses before it calls mk returns an error, not
`PassStatus::Ok`. `PassStatus` stays `Ok | Poison | Timeout` for passes that ran. Decided 2026-10-18.

The mk binding refuses a pass before it calls mk when a segment breaks an mk rule. The rule can be a position that
is not the context's n, a token past the vocabulary, or rows instead of tokens. That pass did not run, and no context
changed. A caller tells a refusal from a pass that ran, and it need not read `last_error` to know (review finding 9).

A refused pass returns a `PassError` with kind `PassErrorKind::Refused`. Each engine refuses a pass with more
segments than `max_segments` in this way, before it changes any state.

A fault after the pass is different. The mk binding returns `Poison` when the logits buffer is shorter than the
wanted rows. The pass ran, so the state is unknown (D6, spec 8.2). A unit test pins this decision. Real mk gives the
full buffer, so the test calls the builder and not a pass.

### Canonical file bytes

Decided 2026-10-20 (reading). The `safetensors` crate reads the canonical file, but its writer
aligns the header to 8 bytes and does not order the tensors. Thus this code writes the header and data itself.
The metadata keys are:

- `nagaya.part.<name>.kind`: `fixed`, `cells` or `ring`.
- `nagaya.part.<name>.coverage`: `start..end` for a cells or ring part.
- `nagaya.part.<name>.block`: the block format, when the dtype is `U8` for a block format.
- `nagaya.parent`: the parent content id in hex, for a delta.

Reason: D10 rules 3 and 5 need sorted keys and a 4 KiB data start. The crate cannot write them. The crate
still reads the file.

`PartLimits` has a field `block_format: Option<String>`, so `snapshot_from_engine` can write the block format. The
store refuses a block format on a dtype that is not `U8`. Decided 2026-10-20 (reading).

### Fake engines

`FakeAlways` is deterministic (`always`). Its parts are:

- `state`, fixed, `U8`, shape [16];
- `cells`, cells, `U8`, row shape [8];
- `ring`, ring, `U8`, row shape [8], window 4 rows;
- `pooled`, derived, `U8`, row shape [8].

Its slots are one of 64 bytes and seven of 256 bytes. Its seal flips the low bit of the first `state` byte.

Its `pass` changes state and is deterministic. For each token it appends one `cells` row and one `ring` row.
It sets the 16-byte `state` part from the token bytes, and it returns the state as one output. It reads no
clock, random value or address. The fake pass allocates its test rows; the fake is not a ledger engine.

`FakeBroken` is the same, but `part_read` flips the low bit of byte 0. Thus a park and restore loses a bit,
and gate 8.3 (park and restore) must fail it.

`FakeRingBase` is the same as `FakeAlways`, but it reports the wrong base position for its ring. The park gate must
fail it. `FakeAlways::with_slots` makes a `FakeAlways` with other slot sizes. Each new gate of spec 8.3 adds its own
broken fake.

### Logit outputs

The mk binding gives logits one way, and the fake engine follows it. Decided 2026-10-20 (reading, from a note by
Sonnet 5.5 in milestone 12).

- `logit_rows` holds positions that the segment feeds. A position outside `positions` is a refused pass.
- A pass gives one output for each entry of `logit_rows`, named `logits`.
  The order is segment order, then request order.
- The dtype is F32 and the bytes are little-endian. A row has one element for each token of the vocabulary.
- The vocabulary size is not in `limits()`. The family library knows it (spec 11.3).
- The fake engine gives logits only when it has a vocabulary size. Its outputs for one segment are `state`, then its
  `logits` rows. The mk binding gives `logits` only, and only for the last position.

Reason: the fake engine must serve a family with a vocabulary, and the output rows must compare as F32 in the gates
(spec 8.3). The simplest rule is the one that mk already follows.

## How mk maps

| ABI | mk (`capi.hip`) | Notes |
|---|---|---|
| open / attach | `mk_engine_open(s1, s2, s3)`, `mk_engine_attach(…, path)` | no device argument: the HIP device is the calling thread's current one, so the wrapper sets it on the group's thread; weights are three GGUF shards |
| plan | none yet | the wrapper computes the terms from the config and mk's fixed shapes |
| arena | `mk_engine_arena`, `mk_arena_check` | one fd; full check |
| identity | built in the Python service | moves into the engine; must include `fused` and `MK_FAST_QMM` |
| contexts | `mk_ctx_new`, `mk_ctx_fork`, `mk_ctx_copy`, `mk_ctx_free`, `mk_ctx_n` | a fork shares its parent's K/V and index keys |
| parts | `mk_ctx_state_read`, `mk_ctx_state_write`, `mk_ctx_cells_write`, `mk_ctx_set_n` | parts 0-2 and 5 fixed, 3-4 cells; `idx_pooled` derived. `cells_read` takes only rows 0 to n, the whole part: snapshots are stored whole (about 316 MB at n = 3584, ontology "Part") |
| slots | `mk_ctx_n`, `mk_ctx_free` then `mk_ctx_new` | `slot_high` is the token count; `slot_clear` makes a new context of the same capacity |
| coverage | none | rows 0 to n of each cells part; `fits` compares the rows with the slot's capacity |
| seal | `mk_ctx_round_gdn_bf16` | changes bits in place |
| poison / last_error | `mk_ctx_poisoned`, `mk_last_error` | |
| part writes | `mk_ctx_state_write`, `mk_ctx_cells_write`, `mk_ctx_set_n` | return nothing and abort on input they refuse; the wrapper checks every precondition first (part index, bytes, an empty context that owns its K/V for parts 3-5, n within `max`, not poisoned) and returns refused; a write to a context that is already poisoned is refused, because this call changes nothing; after the call, `mk_ctx_poisoned` gives failed (decided 2026-10-18) |
| close | none | `mk_engine_close` exists, but HIP teardown segfaults in the process; mk does not declare `close` |
| attach path | `mk_engine_attach(…, path)` | the path is `/proc/self/fd/<n>` for an fd from systemd's store |
| pass_check / pass | `mk_segments_problem`, `mk_pass_segments` | tokens only (PLE needs token ids); outputs are logits and hidden per segment; `mk_pass_segments` aborts the process on bad segments (`MK_DIE`), so the wrapper calls `mk_segments_problem` first |
| taps | `mk_pass` (layers, routes), `mk_pass_gdn_ab` (gates) | unfused, one segment; logits go straight to the caller's buffer, the other taps are copied |
| telemetry | `mk_stage_times`, `mk_stage_detail`, `mk_dispatches`, `mk_host_stats` | mk's stages are ops (`stage.h`), so these are op times |
| determinism | `always` | `stage.h`: "every op's output bits depend only on its work items"; top-k ties break by index; bit settings in the identity |

### v0 mk binding choices

Decided 2026-10-20 (reading). The milestone 8 binding needs these choices. Each is the simplest choice that fits
the ABI.

- **A context is a state.** `State` is `Slot(u32)` in v0, so each mk context has a slot index. `limits().slots`
  declares the pool that one handle creates at open, as two classes of slots. The small class comes first, as in
  the fake engine.
  - Slots 0 to 11 hold 4,096 tokens each. Slots 12 to 15 hold 32,768 tokens each.
  - The process opens one handle. The conformance suite opens a handle for each gate.
  - This binding does not bind `mk_ctx_fork`.
- **mk declares `slots` and `coverage`.** Decided by senpai 2026-10-20 night (vetoable). The park on stop (spec 10.3)
  and the council build (spec 11.3c) need both.
  - `slot_high(slot)` is the context's token count (`mk_ctx_n`). A slot with no context gives 0.
  - `slot_clear(slot)` frees the context and makes a new one of the same capacity, under the same slot. mk has no
    call that empties a context. The free comes first, because the context term holds one context for each slot.
  - A clear also empties a poisoned context (spec 8.2). `mk_ctx_free` does not check the poison.
  - When mk makes a context of another capacity, the slot has no context, and the clear gives `Failed`.
  - `coverage` gives rows 0 to n for part 3 (`kv`) and part 4 (`idx_keys`), from the byte count of the rows. It
    gives no position for a fixed part.
  - `fits` accepts a span of a cells part that starts at 0 and ends at or below the slot's capacity in tokens.
  - `cells_read` takes `from` 0 and `n` equal to the token count. Then it is a whole read. It refuses every other
    range, because mk has no range read.

  **Open:** `mk_ctx_new` stops the program when `hipMalloc` fails (`MK_CHECK`, `engine.hip:561-569`). Thus a clear
  that cannot allocate ends the process, and the binding never sees a null context. The free comes first, so the
  allocator holds the bytes. mk needs a reset call that keeps the buffers, so that a clear does not allocate.
  - The factory of the process crate gives `plan(config)` (next rule), and this binding fixes the pool.
- **The factory plans from `weights_bytes`.** Decided by senpai 2026-10-20 night (vetoable).
  - mk knows the size of its weight arena only after the weights load (`mk_engine_arena`). The arena is a repacked
    copy of the tensors, so `plan` cannot compute the size.
  - The config of the engine has one key, `weights_bytes`. The factory refuses any other key.
  - `plan` returns the limits of the pool with this size in the weights term.
  - `open` loads the weights and compares the real arena with the key. A difference refuses the start (exit 3,
    spec 10.4). The message gives the real size for the profile.
  - The factory reads `MK_ENGINE_LIB`, `MK_SHARDS` and the optional pin `MK_ENGINE_LIB_SHA256` from the environment.
  - `plan` refuses a missing variable, or a library or a shard that is not a file. The message names the variable.
  - The kind in the profile is `mk-qwen38-gfx1151`. A build with the feature `hip` registers it.
- **The pool is 12 contexts of 4,096 tokens and 4 of 32,768.** Decided by senpai 2026-10-20 night (vetoable). The
  pool had 16 contexts of 256 tokens (Amy, 2026-10-20: "grow the pool"). A kaijutsu context needs thousands of
  tokens, so 256 cannot serve council v2.
  - The count stays 16, which equals `max_segments`. A pass can hold one segment for each context.
  - mk takes any capacity that is a multiple of 256, up to 32,768 (`check_capacity`).
  - Two sizes keep the long contexts few and the short ones many.
  - The measured bytes are in "The context term holds the pool", below.
  - To change the classes, edit `CONTEXT_CLASSES` in `spec.rs`. The tests derive their numbers from it.
- **A long prompt takes several passes.** A pass holds at most 256 tokens (`kMaxPassTokens`), so `max_pass_inputs`
  stays 256. The caller feeds a prompt in passes of at most 256 tokens. Each pass starts at the context's token
  count. mk gives the same bits for any split (spec 6.13).
- **mk declares a checkpoint grain of 256 tokens.** Decided by senpai 2026-10-20 night (vetoable). The GPU tests
  of spec 6.13 measured a copy round trip and pass splits across 256 tokens. mk also declares `segments_exact`,
  which a grain needs. The grain is `CHECKPOINT_GRAIN` in `spec.rs`.
  - A profile with `interval_copies = 0` makes no copies. The gfx1151host profile has 0.
- **`max_quantum_ms` is 10000.** mk splits a pass at 256 tokens (`kMaxPassTokens`). A prefill chunk of that size takes
  about 2 s. mk's watchdog stops the program at a 10 s deadline (`engine.h:302-306`), so the declared bound is 10 s.
- **`cells_write` writes `n` rows from `from`.** The binding refuses a `from` other than 0 and changes nothing. mk
  writes n cells into an empty context that owns its K/V. The caller calls `set_n` after it writes every part.
- **`positions` are strict and `meta` is empty.** mk appends the tokens at the context's n. The binding refuses a
  segment whose first position is not n, or whose positions are not contiguous. mk reads no per-token metadata, so
  `meta` must be empty.
- **A pass is fused by default.** The binding passes `fused = 1`. A tap needs the unfused path and one segment. This
  binding declares no taps.
- **The checkpoint digest is a spot check.** The binding hashes each shard's size and its GGUF header, as mk's
  `weight_hash` does. The digest starts with `spot-sha256:`.
- **`tokenizer_hash` is a hash of the GGUF tokenizer.** mk loads no tokenizer file. The binding hashes the
  tokenizer keys of the GGUF metadata, and the family library computes the same hash from the same keys (spec 11.3).
  Amy chose this on 2026-10-20, so a change of the tokenizer changes the recipe. The hash is `sha256:` and 64 hex
  digits, made by `nagaya-text-qwen38` (`hash.rs`). It covers `tokenizer.ggml.model`, `.pre`, `.tokens`,
  `.token_type` and `.merges`, and the bos, eos, eot, padding and unknown token ids when they exist. The template is
  apart: it is the identity's `template`.
- **The template is the GGUF's text.** The identity's `template` is the text of `tokenizer.chat_template` in the
  first shard. A missing key, or a value that is not a string, is an error at open that names the key and the file.
- **The context term holds the pool.** The mandatory term `context` has lifetime `engine` and holds all 16
  contexts. A read of a poisoned context is an error. The bytes follow from mk's `new_context`:
  - The fixed state is 4 × (36×48×128×128 + 36×3×10,240 + 9×10,240) = 118,038,528 bytes.
  - Each token adds 4 × (12×2×2×256 + 12×128 + 12×128÷4) = 56,832 bytes.
  - That is 49,152 bytes of K/V, 6,144 of index keys and 1,536 of pooled keys.
  - A context of 4,096 tokens is 118,038,528 + 4,096 × 56,832 = 350,822,400 bytes.
  - A context of 32,768 tokens is 118,038,528 + 32,768 × 56,832 = 1,980,309,504 bytes.
  - The declared term adds 4 MiB (4,194,304 bytes) to each context for the allocator.
  - The term is 12 × 350,822,400 + 4 × 1,980,309,504 + 16 × 4,194,304 = 12,198,215,680 bytes.

  Measured on 2026-10-20 night on gfx1151 by `examples/context_pool.rs`. Three runs gave equal device numbers.
  - `hipMemGetInfo` fell by 4,229,955,584 bytes for the 12 small contexts. That is 1,673,899 bytes above the state
    bytes for each context.
  - It fell by 7,929,331,712 bytes for the 4 large contexts. That is 2,023,424 bytes above for each context.
  - Both are under the declared bytes. `open` reads both numbers and refuses a pool that takes more than it declares.

  The host side (spec 4.7):
  - The weights take 61 GiB. One handle leaves about 51 GB of MemAvailable. Each further handle takes about 12.2 GB.
  - `open` refuses a pool that would leave less than the 12 GiB floor. Thus at most three handles are open at once.
  - A gate of the conformance suite holds at most three handles.
- **`canary_hash` is empty.** This binding does not run mk's canary. The process does not run it yet (spec 8.1).
- **The identity carries `MK_FAST_QMM`, `MK_TILES` and `fused`.** They are `Approx`, `ReferenceDefining` and
  `Approx`. mk reads `MK_FAST_QMM` at each pass build (`engine.h:27-32`). `MK_TILES` chooses the tiled or the GGUF
  layout at load (`engine.h:139-146`). The binding has not measured whether the two give equal bits. A change after
  open refuses the pass, as for `MK_FAST_QMM`.

Reason for all: the code must not guess.

**Corrected** by deepseek-flash, 2026-10-18: `max_quantum_ms` was 30000. mk's watchdog aborts a pass at 10 s
(`engine.h:302-306`), so a 30 s bound promises a quantum that mk would stop. Review finding 13.

**The deploy profile pins the mk library.** The identity records the SHA-256 of `libmkengine.so`. The deploy profile
requires that pin. The binding reads `MK_ENGINE_LIB_SHA256` and refuses a library with another digest before it binds
any symbol. Tests may leave the pin unset. mk has no call for the vocabulary size, so the binding cannot check
`kVocab` against the library either. Decided 2026-10-18.

Reason: without the pin, nothing compares the digest in the identity with an expected one (review finding 2).

**mk's pass scratch is a mandatory term.** The ABI says that `pass` never allocates and that its scratch is a
mandatory term. mk allocates one scratch pool at its first ask and keeps it for the engine's life
(`engine.h:104-112`). The term is `{name: "pass_scratch", tier: device(0), bytes: the measured constant (milestone
9c), lifetime: engine, kinds: [prefill, decode]}`.

mk has no call for the size. The process measures the pool once on gfx1151 and declares that constant. It refuses
the engine when a later measurement is larger than the constant. Decided 2026-10-18. **Open** (nagaya-coder-12,
2026-10-18): the declared bytes are the measured device cost, which is 59 800 564 bytes above the pool's 263 934
988-byte ask. The HIP runtime's first launch is the likely source of 38 522 880 bytes that the pool and the
allocator's rounding do not explain.

The wrapper's host buffers also wait for the process crate (the review's note on finding 9).

### Test limits of the spin-timeout opt-in and the poisoned set_n path

Decided 2026-10-20 (reading): these limits stay, and this section records them.

The binding binds `mk_engine_recover_spin_timeouts` and calls it with `on = 1` at open (`566446c`, in the open path
of `worker.rs`). A misspelled symbol makes open fail with an undefined-symbol error, and `f99f9c0` quotes that
mutation. Thus the suite pins the symbol's name only.

Removing the binding and the `on = 1` call leaves every test green. mk has no query for the flag, so the suite cannot
see the call. The opt-in is not pinned.

Two paths have no test. A forced spin-timeout, and the `-1` to `PassStatus::Poison` path that follows it, need GPU
contention, so no test in this suite reaches them. The poisoned `set_n` path has no test on real mk, because mk's
`restore_n` stops the program on a poisoned context (`engine.hip:775-776`). A stub binding tests the refusal.

Reason: a test that passes when the code is broken is a defect, so the suite reports these limits instead of a false
check. A lab hook for the first case is not in v0.

### Test limit of the open-time part check

Decided 2026-10-20 (reading): this limit stays, and this section records it.

The wrapper calls `parts_problem` at open (in the open path of `worker.rs`), and it
refuses a library with other part sizes before any write. Real mk matches the declared shapes, so the check passes
in every test.

Deleting the call leaves every test green. Only `parts_problem` has a test (in `spec.rs`): it refuses a larger part
with `a larger part must be a mismatch`. The call at open is not pinned. A fake library would pin it, and this suite
has none.

Reason: a test that passes when the code is broken is a defect. A person must see this limit before trusting the
check.

## How mkviolet maps

| ABI | mkviolet (`gemma4.rs` unless named) | Notes |
|---|---|---|
| open / attach | `load_slots(cluster, model_dir, slot_seqs, max_block)`, `load_slots_stored` | slot sizes and `max_block` are load-time config |
| device group | both B70 cards | one group thread and one stage thread for each card (D1) |
| stages | `run_card0`, `run_card1` (`gemma4.rs:1393-1436`) | stage 0: layers 0-14 on card 0; stage 1: layers 15-29 and the head on card 1 (`CARD_LAYERS`, `gemma4.rs:84`); the residual is F32 rows of 2,816 values, through host memory |
| arena | 30 DMA-BUF slabs, one for each layer on its card (`slab_fd_name(l, card)`, `main.rs:76-108`), `verify_stored_slab` | spot check: first and last 64 bytes per tensor; the head has no slab today |
| slots | `set_slot`, `kv_high`, `slot_clear` | no contexts, no stored token count, no fork |
| parts | `save_kv_layer_from` (a range read), `load_kv`, `load_kv_layer` | `state` is the slot index; KV only: cells on global layers, ring on sliding layers |
| part names | the K and V planes of each layer (`KvSnapshot`) | `k.<ll>` and `v.<ll>`, F32, two digits for the layer; see "mkviolet's parts" |
| geometry / coverage | slot size, ring rows (`kv_ring`, `kv_keep_from`); `KvSnapshot::join` is the fit check | changes rows held, not bits, so no `geometry_key`; today's id is a recipe (`snapshot_id`, `held.rs:329-342`) |
| pass | `run_card0` then `run_card1`; `pipeline_stage` for jobs in flight on both cards (`gemma4.rs:1438-1505`) | the nagaya's scheduler takes the place of `pipeline_stage` (D1) |
| segments | `PipeJob`; `branch_block` with per-token `meta` | a branch block is atomic and is not a snapshot |
| inputs | tokens; `write_embeddings` looks up the rows on the host and puts them into `Buf::X` (`gemma4.rs:1376-1384`) | tokens (D2) |
| class | `CardLock`, `GpuLock` | moves out of the engine into the nagaya scheduler, one choice for each device (D1) |
| determinism | `always`, with a feed rule (D11) | bits depend on the token count of a pass; the first port runs one recipe in two starts and compares content ids |

### mkviolet's parts

**Proposed** (claude-opus-5-5, 2026-10-21). mkviolet keeps only KV (`new_slots`, `gemma4.rs:444-475`). The device
holds each layer's K and V as F32 planes. These parts follow from that layout:

- Each layer has two parts, `k.<ll>` and `v.<ll>`, with `ll` the layer in two digits. The dtype is F32, so the store
  converts no value.
- A global layer is a layer `l` with `l % 6 == 5`: 5, 11, 17, 23 and 29 (`layer_geom`, `gemma4.rs:93-99`). Its parts
  are cells, with the row shape [2, 512]. Its coverage is positions 0 to n.
- Each other layer is a sliding layer. Its parts are rings of 4,096 rows (`KV_RING_ROWS`), with the row shape
  [8, 256]. In a slot of 4,096 positions or fewer the ring never wraps, so the coverage is 0 to n. In a longer slot
  it is the last 4,096 positions (`kv_ring`, `kv_keep_from`, `gemma4.rs:30-65`).
- A ring plane has 1,024 more rows of spill (`KV_RING_SPILL`), which `OP_COPY` moves to the start of the ring. The
  spill is a device layout, so the canonical file never holds it.
- The canonical file (D10) unrolls each ring into position order. A token costs 450,560 bytes in all parts
  (`kv_bytes_per_token`).
- Each part names its stage (D1): layers 0-14 are stage 0 on card 0, and layers 15-29 are stage 1 on card 1.
- `fits` follows `KvSnapshot::join` (`gemma4.rs:940-978`). A ring part that starts after position 0 fits only a slot
  with a ring.

## Decisions

Decided: D1-D11.

**D1. Engines that span devices.** There are two options:

- (a) The engine schedules its own stages. The nagaya sees one quantum per pass.
- (b) The engine declares stages and exports `stage_pass`. The nagaya schedules each stage on its device.

Option (b) lets the scheduler fill an idle card, as mkviolet's low-priority path does today. A model split across
machines needs the same residual boundary later. But (b) needs the full segment shape: positions, meta, logit rows,
a snapshot to restore, and the slot. It must also stop another job from writing the slot between a block's two
stages.

Decided 2026-10-17 (Amy: "yeah that sound reasonable"): mkviolet starts with (a). Option (b) comes when the segment
shape is settled, and first for splits across machines.

Decided 2026-10-21 (Amy: "I want the dual card architecture in place sooner than later to make sure it fits in the
overall nagaya architecture well: move stage_pass up"). Option (b) comes in the first port of mkviolet. This replaces
the start with (a). The rules below settle the segment shape and the hold of the slot.

A group can serve more than one engine: two engines that use one device share one group, one thread and one
scheduler. Decided 2026-10-17: "one thread per device group". With stages, that thread runs the scheduler, and it has
stage threads under it.

**The stage segment.** A stage pass takes segments. Each segment has these members:

- `state`: the slot. Every stage of a block names the same slot.
- `positions` and `meta`: the first position and the per-token `meta` (mkviolet `pos0` and `meta`, `PipeJob`,
  `gemma4.rs:998-1004`). A plain block has no `meta`.
- `inputs`: the tokens, on the first stage only (D2). A later stage reads the residual.
- `logit_rows`: the rows of the block whose logits the last stage gives. mkviolet takes at most 16
  (`MAX_LOGIT_ROWS`).
- the residual: a host buffer that the nagaya owns (D5). Stage s writes it, and stage s + 1 reads it. A row is
  `limits().residual_row_bytes`: 2,816 F32 values (11,264 bytes) for mkviolet.

**Proposed** (claude-opus-5-5, 2026-10-21): a restore is not a member of the segment. mkviolet restores a held
snapshot into the slot inside a pipelined pass: each card writes its layers first (`PipeJob.kv`,
`gemma4.rs:1477-1490`, `council_server.rs:859-874`). In the nagaya, a restore is a `copy` quantum on the device of a
stage. It writes only the parts of that stage, through `Store::restore` (spec 8.3). It runs on that device just before
the first stage quantum of the block. Reason: one restore path for every engine, with the same overlap of cards.

**Proposed** (claude-opus-5-5, 2026-10-21): each part names its stage, `limits().parts[].stage`. A stage handle reads
and writes only the parts of its stage. mkviolet's parts are the KV of one layer, so each part has one card.

**Proposed** (claude-opus-5-5, 2026-10-21): the Rust shape of `stages`.

```rust
trait Stages {
    // Once, after open: one handle for each stage, in stage order. Each handle moves to the thread of its device.
    fn take_handles(&mut self) -> Vec<Box<dyn StageHandle + Send>>;
}
trait StageHandle {
    fn stage_pass(&mut self, segments: &[Segment], residual: &mut Residual) -> Result<StageOut, PassError>;
    fn parts(&mut self) -> Option<&mut dyn Parts> { None }   // the parts of this stage only
}
```

- `Engine::pass` runs every stage in order on the calling thread. The conformance gates compare it with the stage
  path.
- After `take_handles`, `Engine::pass` and the `parts` calls of the `Engine` refuse (`PassErrorKind::Refused`). Each
  call that touches a device then goes through the handle of that device.
- `identity`, `limits` and `last_error` stay on the `Engine`, on the group thread.

**Proposed** (claude-opus-5-5, 2026-10-21): the threads of a group with stages.

- The group thread runs the one scheduler of the group (spec 5.5). It sends each stage quantum to the stage thread
  of its device, and it gets the end back.
- A stage thread holds the handle of its device. It runs one quantum at a time, and it makes every call to its
  device.
- The ledger stays on its own thread (spec 3.2). The group thread asks it by message, as now.

Reason: one scheduler sees both cards, so it can fill an idle card. One thread for each device keeps each card busy
while the other runs, as mkviolet's `pipeline_stage` does with a scoped thread for each card.

**The hold of the slot.** A job holds its slot with a lease from its first quantum to its end (spec 5.6). The
scheduler orders the stage quanta of the slot:

1. Stage s + 1 of a block is ready only after stage s of that block ends and writes the residual.
2. Stage s of block k + 1 is ready only after stage s of block k ends.
3. No quantum of another job writes the slot while the lease lives. This is quantum condition 5 (ontology).

Thus two blocks of one job run at the same time on the two cards, as mkviolet's `run_jobs` and `feed` do
(`gemma4.rs:1507-1516`, `mk_api.rs:697-714`). Each stage writes the rows of its own layers only, so the two stages
never write the same bytes.

**Classes for each device.** Between quanta on each device, the scheduler runs the ready stage quantum of the highest
class for that device (spec 5.3, 5.4). A stage quantum charges its run time to its job, as a pass does. Thus a
council read takes card 0 while a generation's stage 1 runs on card 1. mkviolet does this with `CardLock` and
`GpuLock` at low priority (`pipeline_stage` with `slot`).

**Bounds and the watchdog.** A stage quantum is a quantum (spec 5.1). `max_quantum_ms` bounds each stage quantum, and
the engine chooses its `feed.block` to fit it. Each stage thread beats a heartbeat of its own. The watchdog reports a
stage quantum past the bound, and a stall of any stage thread stops the process (spec 7.4).

**The residual term.** **Proposed** (claude-opus-5-5, 2026-10-21). The residual buffers are a mandatory host term of
the engine, `residual`, with lifetime `engine`. It has two buffers for each boundary between stages. One holds a
block for stage s + 1, while stage s writes the next block. Each buffer holds `feed.block` rows. mkviolet's term is 2 ×
512 × 11,264 = 11,534,336 bytes.

Amy may stop running one model across both B70 cards and use them independently, "since they're the fastest memory
we have at home". Today's Gemma 4 does not fit on one card. Thus per-card work means smaller engines, each with a
group of one card: a smaller Gemma, "or probably a qwen and lfm".

**D2. Inputs.** An engine declares tokens, rows, or both. mk needs token ids even when it gets rows. Its per-layer
embeddings (PLE) hash the token ids, and its PLE history is a fixed part (`engine.h:75`, `ple_rows`). gmk's audio rows
and diffusion latents have no tokens. Proposed: `inputs = tokens | rows | tokens+rows`, declared per engine.

Deferred 2026-10-17 (Amy: "sort it out with the first engine"). mkviolet's Gemma 4 is the first port.

Settled the same day: an engine never takes text. The family library in the nagaya turns text into tokens (spec
11.3).

Settled 2026-10-21 by the code of the first port, as Amy asked. mkviolet takes token ids. It looks up each
embedding row on the host, scales it, and writes it into card 0's residual (`write_embeddings`,
`gemma4.rs:1376-1384`). Thus mkviolet declares `inputs = tokens`, as mk does. The embedding stays inside the
engine. The ABI keeps `rows` and `tokens+rows` for gmk's audio rows and diffusion latents.

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
| `per-process` | the same bits within one process only | off in the store; the audit job checks (spec 6.7, 8.4) | exact within the process; across processes only for a served model whose tolerance accepts it (ontology) |

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

**D11. A feed rule.** Decided 2026-10-21 (Amy, interview: "we should do what mkviolet does now? maybe change nagaya
if we have a better approach."). An engine whose bits depend on the token count of a pass declares a feed rule. The
scheduler then feeds every kept span in one fixed split, so the bits depend only on the inputs.

mkviolet chooses the reduction of a matmul by the token count of the pass (`megakernel.cl:54-57`, `:478-760`):

- 1 token: `SPLIT_K` = 4 subgroups share the K of one row;
- 2 to 8 tokens: XMX tiles of 16 rows, with K split over `KSPLIT` = 8 subgroups;
- 9 to 15 tokens: XMX work items of 16 rows by 32 tokens;
- 16 tokens or more: a work-group takes 64 tiles and shares the tokens' x in local memory.

The same split gives the same bits. Five runs of a 600-token prefill in blocks of 256, then 24 one-token steps, gave
equal logits (`golden.rs:431-458`). mkviolet's `snapshot_id` hashes the identity, the tokens and the boundaries
(`held.rs:329-342`). It feeds in `max_block` passes (`Gemma4Engine::block_jobs`, `gemma4.rs:1624-1635`).

The rules:

1. `limits().feed` is `{block}`, in tokens. The scheduler feeds each span in passes of `block` tokens from the start
   of the span. The last pass holds the remainder.
2. A span starts at position 0 or at a boundary. It ends at the next boundary, or at the end of the fed tokens.
3. `block` is a bit setting with the class `reference-defining`. Thus it is in the identity and in every recipe
   (spec 6.2). mkviolet's `block` is its `max_block`, 512 (`MAX_BLOCK`, `main.rs:35`).
4. An engine with a feed rule declares `segments_exact: false` and no checkpoint grain (spec 6.13).
5. Its `determinism` stays `always`. The rule fixes the split, so the same identity and inputs give the same bits in
   any process.
6. A pass that writes rows that the store keeps has one segment. Segments whose rows no one keeps may share a pass,
   such as reads and branch blocks. Their outputs fall under the tolerance of the served model.
7. A slot can hold a span in part, when its last pass held fewer than `block` tokens. A job that continues the span
   feeds that last pass again, with the new tokens after it.

Spec 6.14 gives how decode steps, drafts and branch blocks meet the rule.

Reason: this is what mkviolet does now, and it keeps the recipe as tokens plus boundaries. The other way is a kernel
whose reductions do not depend on the token count (spec 5.7). That costs speed on the one-token path, and it is a
change inside mkviolet's kernels.

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
