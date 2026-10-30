# Engine ABI v0 (draft)

This doc defines the interface that each engine exports to the nagaya. The nagaya is an exokernel for megakernels:
it owns the machine, and this ABI is the one boundary between it and an engine. Below the boundary, an engine keeps
its layouts, its kernels and its caches. Above it, the nagaya keeps the ledger, the scheduler and the store.
`docs/ontology.md` defines the terms, and the four other projects that this doc names.

We drafted the ABI on 2026-10-17 from two sources, our two learning megakernels:

- mk's C ABI (`src/engine/capi.hip`, `src/engine/engine.h`);
- mkviolet's Gemma 4 engine (`src/gemma4.rs`, `src/held.rs`, `src/protocol.rs`, `src/hot_handoff.rs`).

A file path after "mk" or "mkviolet" in this doc names the source of a copy or of a measurement. The copies live in
`crates/nagaya-mk-qwen38-gfx1151` and `crates/nagaya-mk-gemma4-bmg`.

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

**The factory gets its devices.** Decided 2026-10-22 (Amy: "accepted"). Before this rule, `open` took only the
config. mk uses the HIP device that is current on the calling thread.

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
- The inventory keeps the PCI address of each device for this (`crates/nagaya/src/probe.rs`).
- A Level Zero engine maps each PCI address to its Level Zero device with `zeDevicePciGetPropertiesExt`. It refuses
  an address that it does not find, and an address that two devices give. mkviolet takes the cards in the order of
  `zeDeviceGet` (`L0Cluster::init`, `level_zero.rs:745-790`), and that order can differ from the PCI order.
- A HIP engine selects its device with `hipDeviceGetByPCIBusId` on the thread of its device.
- An engine that gets the wrong count of devices refuses with `PlanError`.

Reason: the profile names the devices, so the engine must run on those devices and no others.

**Arena calls in v0.** Decided 2026-10-22 (Amy: "accepted"). These are the Rust shapes of the `arena`
capability and of `attach` (spec 10.2).

```rust
struct ArenaFd { device: usize, part: String, fd: OwnedFd }   // device is a group index
enum CheckStrength { Full, Spot, None }
trait Arena { fn arena_export(&mut self) -> Result<Vec<ArenaFd>, String>; }   // Engine::arena()

trait Factory {
    // ... plan and open as above
    fn arena_parts(&self, config: &toml::Table, devices: &[GroupDevice]) -> Result<Vec<ArenaPart>, PlanError>;
    fn attach(&self, config: &toml::Table, devices: &[GroupDevice], arenas: Vec<ArenaFd>)
        -> Result<Attached, AttachError>;
}
struct ArenaPart { device: usize, part: String }
struct Attached { engine: Box<dyn Engine>, check: CheckStrength }
enum AttachError { Mismatch(String), Open(OpenError) }
```

- `arena_parts` reads no weights, as `plan` does. A kind with no arena gives no part, and that is the default.
- `attach` does `arena_check` first, on the devices of the group. A mismatch returns `AttachError::Mismatch` and
  leaves nothing open. Then the process drops the fds and calls `open`.
- `arena_export` gives a new fd for each part. The engine keeps its own fds. The process sends each new fd to systemd
  and then closes it.
- The process gives `attach` a duplicate of each stored fd (spec 10.2 step 3), and `attach` owns it.
- v0 has no `arena_id`, because nothing reads it. The check compares the arena with the build.

Reason: mkviolet checks its slabs on the open cards (`verify_stored_slab`), so the check and the attach share one
open of the cards. After a mismatch, `open` opens the cards again.

## Core (every engine)

```
plan(config) -> limits           reads no weights; the ledger admits open() from this result
open(config) -> engine           config: devices, weights, engine-specific settings (mkviolet: slot sizes, max_block)
attach(config, arena_set)        the same, on an existing arena set
identity() -> {
    model, checkpoint_digest,    the logical digest of the weights (spec 12.2)
    tokenizer_hash, template, family, numbers, plan table,   family: toy | qwen38 | gemma4 (spec 11.3)
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

**`close` in Rust.** Proposed by senpai (Claude Opus 5.5), 2026-10-24, for the swap in place (spec 3.3a), and
Decided 2026-10-28 (Amy, by interview).

- `close` is the `Drop` of the engine. The trait gets no new method.
- An engine that declares `close` frees each device allocation and each host allocation of its own in `Drop`. After
  the drop, the free memory of each of its devices is what it was before `open`.
- A gate of the conformance suite checks it on the device. It opens the engine, runs a pass and drops it, many times.
  It reads the free device memory after each drop.
- The arena fds that systemd holds are not the engine's own. The process removes them before the drop (spec 3.3a).

Reason: the `Box<dyn Engine>` already drops at the end of the group thread. A drop that frees everything needs no
second path, and a test can count the memory after it.

**Per-part coverage.** Decided 2026-10-22 (Amy: "accepted"). This closes the old note: the park used one row
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
- `SlotLimit` is `{bytes, count, tokens}`. `tokens` is the most tokens that a slot of this size holds. The fake engine
  declares two sizes: 64 bytes and 256 bytes, both with 4096 tokens. The fake does not tie bytes to tokens.
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

`FakeFeed` has a feed rule (D11), in one stage or in two (D1). It has two layers, and each layer has one cells part.
A row is a hash of the row before it, the input of the layer, the position and the token count of the pass. Thus a
wrong split gives other bits, as in mkviolet. The residual is the output of layer 0, so one stage and two stages give
the same bits. `FakeFeed::reverse_residual` breaks the stage path, and `FakeFeed::history` breaks the feed rule.

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

| ABI | mk (`extern.hip`) | Notes |
|---|---|---|
| open / attach | `mk_engine_open(watchdog_deadline_s)`; no attach | no device argument: the HIP device is the calling thread's current one, so the wrapper sets it on the group's thread; the Rust reader parses the three GGUF shards, and the Rust host fills the arena (Proposed notes below); the C++ engine is the runtime layer only: the launches under the control block and the watchdog |
| plan | none yet | the wrapper computes the terms from the config and mk's fixed shapes |
| arena | `arena::plan` and `arena::create` in Rust (`arena.rs`) | one sealed memfd in mk's `arena.h` layout; no check of another process's arena yet: milestone 15 step 6 removed the C++ check, which had no call |
| identity | built in the Python service | moves into the engine; must include `fused` and `MK_FAST_QMM` |
| contexts | `context::ctx_new`, `ctx_free`, `ctx_n` in Rust (`context.rs`) | over the runtime calls `mk_rt_malloc`, `mk_rt_free` and `mk_rt_memset` of `rt.cpp`; no fork and no copy |
| parts | `context::state_read`, `state_write`, `cells_write`, `set_n` in Rust | over `mk_rt_memcpy` and `mk_rt_memcpy_2d`; after `set_n` the worker pools the index keys (`host.rs` `pool_context`, through `mk_launch_each`). Parts 0-2 and 5 fixed, 3-4 cells; `idx_pooled` derived. `cells_read` takes only rows 0 to n, the whole part: snapshots are stored whole (about 316 MB at n = 3584, ontology "Part") |
| slots | `ctx_n`, `ctx_free` then `ctx_new` | `slot_high` is the token count; `slot_clear` makes a new context of the same capacity |
| coverage | none | rows 0 to n of each cells part; `fits` compares the rows with the slot's capacity |
| seal | `mk_ctx_round_gdn_bf16` | changes bits in place |
| poison / last_error | `MkContext::poisoned`, `mk_last_error` | |
| part writes | `context::state_write`, `cells_write`, `set_n` | return nothing and stop the program on input they refuse; the wrapper checks every precondition first (part index, bytes, an empty context for parts 3-5, n within `max`, not poisoned) and returns refused; a write to a context that is already poisoned is refused, because this call changes nothing; after the call, the poison flag gives failed (decided 2026-10-18) |
| close | none | `Engine::~Engine` exists, but HIP teardown segfaults in the process; mk does not declare `close` |
| pass_check / pass | `host::segments_problem`, `host::run_pass` in Rust; `mk_launch_fused` | tokens only (PLE needs token ids); the worker refuses a segment count outside 1 to 16, then a problem of `segments_problem`, then a poisoned context, then a bad knob; the host gathers the PLE rows (`ple.rs`), builds the stages (`pass.rs`), and launches them; outputs are logits per segment |
| taps | none | milestone 15 step 6 removed the taps (layers, routes, gates), which had no call |
| telemetry | none | milestone 15 step 6 removed the op times and the event ring of a pass, which had no call |
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

  **Open:** `context::ctx_new` stops the program when `hipMalloc` fails (`mk_rt_malloc` in `rt.cpp`). Thus a clear
  that cannot allocate ends the process, and the binding never sees a null context. The free comes first, so the
  allocator holds the bytes. mk needs a reset call that keeps the buffers, so that a clear does not allocate.
  - The factory of the process crate gives `plan(config)` (next rule), and this binding fixes the pool.
- **The factory plans from `weights_bytes`.** Decided by senpai 2026-10-20 night (vetoable).
  - The size of the weight arena comes from `arena::plan`, which reads the tensor records of the shards at open. The
    arena is a repacked copy of the tensors, and `plan` reads no shard, so it cannot compute the size.
  - The config of the engine has one key, `weights_bytes`. The factory refuses any other key.
  - `plan` returns the limits of the pool with this size in the weights term.
  - `open` loads the weights and compares the real arena with the key. A difference refuses the start (exit 3,
    spec 10.4). The message gives the real size for the profile.
  - The factory reads `MK_SHARDS` from the environment. The engine is in the binary, so no variable names a library.
  - `plan` refuses a missing `MK_SHARDS`, or a shard that is not a file. The message names the variable.
  - The kind in the profile is `mk-qwen38-gfx1151`. A build with the feature `hip` registers it.
- **The pool is 12 contexts of 4,096 tokens and 4 of 32,768.** Decided by senpai 2026-10-20 night (vetoable). The
  pool had 16 contexts of 256 tokens (Amy, 2026-10-20: "grow the pool"). A kaijutsu context needs thousands of
  tokens, so 256 cannot serve council v2.
  - The count stays 16, which equals `max_segments`. A pass can hold one segment for each context.
  - mk takes any capacity that is a multiple of 256, up to 32,768 (`context.rs` `ctx_new`).
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
  about 2 s. mk's watchdog stops the program at a 10 s deadline (`worker.rs` `watchdog_deadline`), so the declared
  bound is 10 s.
- **`cells_write` writes `n` rows from `from`.** The binding refuses a `from` other than 0 and changes nothing. mk
  writes n cells into an empty context that owns its K/V. The caller calls `set_n` after it writes every part.
- **`positions` are strict and `meta` is empty.** mk appends the tokens at the context's n. The binding refuses a
  segment whose first position is not n, or whose positions are not contiguous. mk reads no per-token metadata, so
  `meta` must be empty.
- **A pass is fused.** Milestone 15 step 6 removed the unfused pass, which no call used. The restore's pool stages
  still run one launch each (`mk_launch_each`). This binding declares no taps.
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
  contexts. A read of a poisoned context is an error. The bytes follow from `context.rs` `ctx_new`:
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
  `Approx`. The worker reads `MK_FAST_QMM` at each pass build (`worker.rs` `fast_qmm`). `MK_TILES` chooses the tiled
  or the GGUF layout at load (`worker.rs` `tiles`, `arena.rs` `plan`). The binding has not measured whether the two
  give equal bits. A change after
  open refuses the pass, as for `MK_FAST_QMM`.

Reason for all: the code must not guess.

**Corrected** by deepseek-flash, 2026-10-18: `max_quantum_ms` was 30000. mk's watchdog aborts a pass at 10 s
(`engine.h:302-306`), so a 30 s bound promises a quantum that mk would stop. Review finding 13.

**The binary links mk's engine statically.** Amy decided this on 2026-10-22, by interview (AGENTS.md milestone 15).
It replaces the rule "The deploy profile pins the mk library" (Decided 2026-10-18).

- The crate holds a copy of mk's sources in `hip/`, from mk commit `555a2a3`. The layout under `src/` is mk's.
- Under the feature `hip`, `build.rs` compiles the engine units into `libmkengine.a`. It uses ROCm's `clang++` and the
  flags of mk's CMake build, flag for flag. A test compares them with mk's `compile_commands.json`
  (`tests/build_flags.rs`). The float flags `-ffp-contract=off -fno-fast-math` stay.
- The crate links the archive whole, so that each HIP object registers its kernels. The HIP runtime and the C++
  library stay shared.
- `MK_ENGINE_LIB` and the library pin `MK_ENGINE_LIB_SHA256` are gone. The build facts take the place of the pin.
  `numbers` holds `engine_build_sha256`, the SHA-256 of `libmkengine.a`. It also holds `engine_compiler` and
  `engine_hip`, the versions of the compiler and of HIP. No path is in `numbers` (spec 12.1).
- The build facts stay in the identity digest until the canary runs at each start (spec 12.1). Thus a new build
  makes the held entries unusable. Amy accepted one refill for this change.
- The compiler writes each object to stdout, and runs in the crate directory. Thus the archive has the same SHA-256
  for each profile and feature set. A path of the build directory in the command would change a HIP symbol name.
- The SHA-256 differs between two checkouts, because `__FILE__` and the debug information hold the source paths.
  Build the deployed binary from the same checkout each time.
- mk has no call for the vocabulary size. `extern.hip` states `kVocab` and the other sizes of the binding's buffers
  in `static_assert` lines. A host test compares them with `spec.rs` (`src/layout.rs`).

Reason: the binary now holds its engine. The digest of the archive is a fact of the build, so a pin would only
compare the build with itself.

**The Rust host and the extern layer.** Milestone 15 moved the host code of mk into Rust. Step 3 moved the GGUF
reader (`gguf.rs`), and step 5 moved the stage schedule (`schedule.rs`). Step 6 moved the arena plan, the PLE gather
and the stage build (`arena.rs`, `ple.rs` and `pass.rs`):

- `mk_engine_open` takes a `mk_weight_table` (`weight_table.h`): each tensor of the arena by name, in its final
  layout. The export of the arena (spec 10.2) builds the attach in Rust, from `arena.rs`.
- Rust mirrors `Stage` and the argument type of each op as `#[repr(C)]` types (`src/stage.rs`). One file holds each
  size, offset and constant of that mirror (`hip/src/engine/stage_layout.def`). Both the C++ build and the Rust build
  check against it, so a change on one side alone stops a build.
- `MK_ELIDE` and the other knobs of the build are read in Rust at each pass, and a bad value refuses the pass.
- The C++ extern layer (`hip/src/engine/extern.hip`) holds only the calls that the binding uses. Each call takes C
  types and calls one `Engine` method or reads one field. The binding declares each call of the layer, and a test
  compares the two lists. Step 4 removed the calls that only mk's Python harness and lab used, and step 6
  removed the C++ methods behind them.

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

The binding bound `mk_engine_recover_spin_timeouts` and called it with `on = 1` at open (`566446c`). Milestone 15
step 6 moved the rule into Rust, and the flag went. A pass can stop on a spin-timeout that the watchdog did not ask
for. Then `host.rs` `run_pass` always poisons the contexts of the pass. Any other stop ends the process. The test knob
`MK_MUTATE_STALL_BARRIER` can force a real spin-timeout, so a test of that path is now possible; none is written yet.

Two paths have no test. A forced spin-timeout, and the `-1` to `PassStatus::Poison` path that follows it, need GPU
contention, so no test in this suite reaches them. The poisoned `set_n` path has no test on real mk, because mk's
`set_n` stops the program on a poisoned context (`context.rs`). A stub binding tests the refusal.

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

This table maps mkviolet's code, the source of `nagaya-mk-gemma4-bmg`, onto the ABI. The file names and the lines
are mkviolet's, as they were when the copy was made.

| ABI | mkviolet (`gemma4.rs` unless named) | Notes |
|---|---|---|
| open / attach | `load_slots(cluster, model_dir, slot_seqs, max_block)`, `load_slots_stored` | slot sizes and `max_block` are load-time config |
| device group | both B70 cards | one group thread and one stage thread for each card (D1) |
| stages | `run_card0`, `run_card1` (`gemma4.rs:1393-1436`) | stage 0: layers 0-14 on card 0; stage 1: layers 15-29 and the head on card 1 (`CARD_LAYERS`, `gemma4.rs:84`); the residual is F32 rows of 2,816 values, through host memory |
| arena | 30 DMA-BUF slabs, one for each layer on its card (`slab_fd_name(l, card)`, `main.rs:76-108`), `verify_stored_slab` | spot check: first and last 64 bytes per tensor; mkviolet gave the head no slab, and the nagaya's copy gives it one, 31 slabs in all (spec 10.2) |
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

Decided 2026-10-22 (Amy: "accepted"). mkviolet keeps only KV (`new_slots`, `gemma4.rs:444-475`). The device
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

### The first Gemma 4 binding

Decided 2026-10-22 (Amy: "accepted"). The trait on the copy (`crates/nagaya-mk-gemma4-bmg/src/engine.rs`) needs
these choices. Each is the simplest choice that fits the rules above.

- **The config.** The kind is `mk-gemma4-bmg`. Its config has three keys. `slots` lists the positions of each slot.
  `max_block` is the feed block, 512 when the key is absent. `model_dir` names the checkpoint, else
  `GEMMA4_MODEL_DIR` names it. A slot holds 1 to 65,536 positions.
- **One segment in a pass.** One task ring runs one block of one slot, so `max_segments` is 1.
- **Positions.** A plain segment starts at the token count of its slot or before it. A start before it goes back,
  and the rings must still hold the window before the start (`kv_can_rewind`).
- **A branch block is a segment with `meta`.** The meta has 16 bytes for each token: the position, the gap start,
  the gap end and 0, as little-endian u32 values. Token `t` writes the row `pos0 + t`, and `pos0` is its first
  position. `logit_rows` names these rows. For a plain segment, a row is its position.
- **A branch block holds no row.** After it, the token count of the slot is `pos0` (spec 6.14).
- **A write empties the slot until `set_n`.** `cells_write` puts the token count of its slot at 0. Then `set_n(n)`
  takes the count when each part of the stage got rows that end at `n` and fit the slot. Without such writes,
  `set_n` only goes back, as a pass does.
- **A whole read reads rows from the base.** `part_read` reads as many rows as its buffer holds, from the first
  position that the part holds. The engine has no fixed part, so `part_write` is refused.
- **The memory terms.** `weights.<i>`, `kv.<i>` and `work.<i>` are on `device(i)`. `pinned.<i>` holds the pinned
  host buffers of card `i`. `embed` holds the embedding table that the host reads, and `residual` holds the two
  residual buffers. Each is the sum of the sizes that the code asks for. No driver measurement checks them yet.
- **`max_quantum_ms` is 2000.** The slowest stage quantum was the last block of 512 tokens in a slot of 65,536
  positions. It took 1,184 ms on card 1 and 920 ms on card 0. Measured on 2026-10-21 by
  `profile_longest_stage_quantum` in `tests/xe.rs`.
- **The checkpoint digest.** It is `blake3:` and a BLAKE3 hash with the key `nagaya gemma4 checkpoint v1`. The hash
  covers the canonical `config.json` (sorted keys, no whitespace), then each tensor in bytewise order of its name.
  A tensor adds its name, its dtype, its shape and the BLAKE3 hash of its bytes. The engine computes it at open.
- **The numbers and the lineage.** `numbers` holds the SHA-256 of the kernel binary, the versions of `ocloc` and IGC,
  the driver version and `max_block`. The plan table names the compiler, the driver and the PCI vendor, device and
  revision of each card (spec 12.1).

Reason: the code must not guess, and each choice keeps the bits of the copy. A test compares the logits of the trait
with the logits of the copy before the trait, bit for bit (`the_trait_gives_the_bits_of_the_copy`).

**Was open:** the stage threads lose about 2% of decode speed. mkviolet's `pipeline_stage` keeps the idle card awake.
A card that idles for 10 ms or more waits 100-550 us at its next command (`B70Card::wake`). The stage threads do not
wake it.

On 2026-10-21, decode after 20 tokens gave 46.1 tokens/s on the stage threads and 47.1 on the copy. A test thread
that woke both cards each 3 ms gave 47.6. A fix needs a call on the stage thread between quanta. That is a change
of the ABI, so Amy decides.

**The wake call.** Decided 2026-10-22 (Amy: "add the call"). A stage thread keeps its card awake between the quanta
of a job.

**The shape of the wake call**, and when the stage thread makes it. Decided by senpai 2026-10-22 (vetoable). Amy
reads it in her review of the docs. Not built yet.

```rust
trait StageHandle {
    // ... stage_pass and parts as above
    fn wake(&mut self) -> Result<(), PassError> { Ok(()) }   // mkviolet: B70Card::wake
}
```

- The stage thread calls `wake` when it has no quantum and 3 ms passed since its last call to the device.
- It calls `wake` only while a job of its group holds a slot lease (5.6). With no job, it makes no call, so the
  card can idle. Reason: a card that the nagaya keeps awake uses more power at idle.
- A wake is not a quantum. It charges no job, and the scheduler does not see it.
- A wake that fails is a pass failure of the stage (spec 8.2).
- An engine whose devices do not sleep keeps the default, which does nothing.

Reason for 3 ms: the test thread of 2026-10-21 woke the cards each 3 ms and gave 47.6 tokens/s.

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

Decided 2026-10-22 (Amy: "seems ok, approved."), updated 2026-10-24 (Amy): a restore is not a member of the segment.
mkviolet restores a held snapshot into the slot inside a pipelined pass: each card writes its layers first
(`PipeJob.kv`, `gemma4.rs:1477-1490`, `council_server.rs:859-874`). In the nagaya, `StageScheduler` has `Item::Restore`
for a per-stage copy quantum. The v0 job path (`StagedJobs`) runs a restore in a whole-group `copy` quantum
(`Item::Whole(Kind::Copy)`). Each stage thread writes its own parts before the first block.

Decided 2026-10-22 (Amy: "accepted"): each part names its stage, `limits().parts[].stage`. A stage handle reads
and writes only the parts of its stage. mkviolet's parts are the KV of one layer, so each part has one card.

Decided 2026-10-22 (Amy: "accepted"): the Rust shape of `stages`.

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

Decided 2026-10-22 (Amy: "accepted"): the threads of a group with stages.

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

**The residual term.** Decided 2026-10-22 (Amy: "accepted"). The residual buffers are a mandatory host term of
the engine, `residual`, with lifetime `engine`. It has two buffers for each boundary between stages. One holds a
block for stage s + 1, while stage s writes the next block. Each buffer holds `feed.block` rows. mkviolet's term is 2 ×
512 × 11,264 = 11,534,336 bytes.

**The stage path in v0.** Decided 2026-10-22 (Amy: "accepted"). The rules above leave these choices open. The
first build of the stage path takes the simplest choice for each.

- **A free residual buffer is a ready condition.** A block holds a residual buffer from the start of stage s to the
  end of stage s + 1. Stage s of a block is ready only when a buffer at its boundary is free. Reason: rules 1 and 2
  alone let three blocks wait at one boundary, and the term holds two.
- **The residual of a pass.** A stage pass with several segments puts their rows in the residual in segment order. A
  pass never holds more rows than one buffer.
- **Copy quanta hold the group.** A whole-group copy quantum (`Item::Whole`) reads or writes the parts of every
  stage, for example a restore, a snapshot, a mark or a lookup. It starts only when every device of the group is free
  and no block of its job waits. No device may have a ready quantum of a higher rank either. A free device does not
  start a quantum of a lower rank before it. Reason: one path into the store for every engine, and a wait of at most one
  quantum (D8).
- **Filler on each device.** A filler stage quantum runs on a device that has no ready quantum of a higher class. This
  is the reading of 5.5 for a group with stages. Reason: D1 fills an idle card.
- **A stage handle has slots and coverage.** `StageHandle` also has `slots()` and `coverage()`, for the parts of its
  stage. A slot clear and the base position of a ring are device state of each stage. After `take_handles`, the
  `Engine` gives `None` for `parts`, `slots`, `coverage` and `seal`.
- **What the first build refuses.** An engine with stages must not declare `seal` or a checkpoint grain. Each stage
  has its own device. The `residual` term holds 2 × (stages − 1) × rows × `residual_row_bytes` bytes. The rows are
  `feed.block`, or `max_pass_inputs` for an engine with no feed rule. The process refuses other limits at open.

Amy may stop running one model across both B70 cards and use them independently, "since they're the fastest memory
we have at home". Today's Gemma 4 does not fit on one card. Thus per-card work means smaller engines, each with a
group of one card: a smaller Gemma, "or probably a qwen and lfm".

**D2. Inputs.** An engine declares tokens, rows, or both. mk needs token ids even when it gets rows. Its per-layer
embeddings (PLE) hash the token ids, and its PLE history is a fixed part (`engine.h:75`, `ple_rows`). gmk's audio rows
and diffusion latents have no tokens. Proposed, and Decided 2026-10-28 (Amy, by interview): `inputs = tokens | rows |
tokens+rows`, declared per engine.

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

A measurement on the B70s on 2026-10-21 showed that the split changes bits. mkviolet fed the same 32 tokens in 1
block, in blocks of 5 and in blocks of 1. The largest logit differences from HF were 5.8e-5, 3.4e-5 and 3.2e-5.

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

Decided 2026-10-18, before any image engine. An image engine used `pass(segments, taps)` with quantum kind `step`. A
diffusion latent was a fixed part, and each `step` quantum replaced it. mk's painter was the first case (mk
`docs/painter.md`). mk's `YieldHook` showed that a paused run matched the unpaused one, max pixel difference 0.

**The image capability.** Proposed by senpai (Claude Opus 5.5), 2026-10-28, for spec 11.9, and Decided 2026-10-28 (Amy,
by interview). It replaces the rule of 2026-10-18 above. Reason: SDXL has two text encoders, token weights, a sampler
with history, a VAE decode and previews. A segment of tokens has no place for them, as it had none for speech
("Speech").

An image engine declares the `image` capability. Its `pass` refuses. It has these calls:

```
image_begin(slot, job) -> ok | why      encodes the prompts, makes the noise, and keeps the job in the slot
image_step(slot) -> {step, previews}    runs one step of each image of the slot
image_decode(slot) -> images            decodes each latent with the VAE into RGB
image_clear(slot)                       frees the state of the slot
```

- **The job.** It holds the ids and the weight of each token of both prompts, in chunks of 75 (spec 11.9a). It
  holds `clip_skip`. It also holds the width, the height, the step count, the CFG scale, the sampler,
  the image count and the seed. The engine never takes text (D2): its inputs are tokens.
- **Quanta.** `image_begin` is one quantum of kind `embed`. `image_step` and `image_decode` are each one quantum of
  kind `step`. Each one runs every image of the slot. The engine declares the most pixels of a slot that keep a
  quantum within `max_quantum_ms` (D8). The nagaya refuses a larger request before it runs.
- **CFG.** One step runs the prompt and the negative prompt of each image in one pass of the UNet.
- **Random stream.** The engine makes each noise value from the seed, the image, the step and the purpose, as the
  draws of spec 11.2. Thus the image of seed `s` does not depend on the other images of its slot.
- **Previews.** With previews on, `image_step` also gives each latent in RGB at one eighth of the size, by a linear
  map. The nagaya makes the PNG.
- **Images.** `image_decode` gives 8-bit RGB. The nagaya makes the PNG and its A1111 chunk.
- **State.** A slot holds the latents, the encoder output, the sampler history and the counters of the random stream.
  It declares no part, because the store keeps no image state (spec 11.9).
- **Determinism.** The same identity, job and seed give the same bits.
- **Later inputs.** img2img, inpainting and a hires pass are optional members of the job (spec 11.9, "The design of
  later image work"). An engine that lacks one refuses a job that names it. A VAE encode is one more quantum of
  kind `step`.
- **Spacing and LoRAs.** Proposed by senpai (Claude Haiku 5.5), 2026-10-29. Decided 2026-10-29 (Amy, by interview):
  the limits are those of spec 11.9a, with a rank of at most 128. `ImageJob` takes `spacing` (`leading` or
  `trailing`) and `loras`, a list of each LoRA's bytes and weight. The engine parses the bytes and adds each delta
  to its module's output (spec 11.9a). The engine refuses `trailing` with a DPM++ sampler. Reason: the spacing of the
  schedule is the engine's, and the spec gives the rule.

## Embedding

**The embedding capability.** Proposed by Claude Opus 5.5 (a senpai fork), 2026-10-30, for spec 11.10. Reason: an
embedding reads tokens and gives one vector. It has no slot state, no logits and no sampler. A segment of a pass has
no place for the pooled vector, as it had none for an image ("Image").

An embedding engine declares the `embedding` capability. Its `pass` refuses. It has one call:

```
embed(ids) -> vector | why      runs one input and gives its pooled vector
```

- **Limits.** The engine declares `embedding.max_tokens`, the most ids of one call, and `embedding.dimension`, the
  length of its vector. EmbeddingGemma 2 gives 2048 and 768.
- **The ids.** They are the family's ids of one input, with `<bos>` and `<eos>` (spec 11.10). The engine never takes
  text (D2).
- **Quanta.** Each call is one quantum of kind `embed`. The engine declares `max_tokens` so that one call stays
  within `max_quantum_ms` (D8). The nagaya refuses a longer input before it runs.
- **The vector.** It is the mean over every position of the model's last output, in f32, before any normalization.
  For EmbeddingGemma 2, the last output is that of `embedding_projection`. The nagaya normalizes (spec 11.10).
- **State.** None. A call holds nothing after it ends. The engine declares no part, no slot and no context.
- **Determinism.** The same identity and ids give the same bits. A call does not depend on the calls before it. Thus
  the vector of an input does not depend on the other inputs of its request, as gogi's rule asks.
- **Batches.** One call runs one input. A later engine may run several inputs in one call only when each vector keeps
  the bits of its call alone. Reason: in the oracle, a padded batch of 4 differed from one input at a time by
  2.4e-7 (`cases.json`, `batch_vs_single_max_abs`). A batch that pads as torch does breaks the rule.
- **Later inputs.** Audio and images are optional members of a later call. An engine that lacks one refuses a call
  that names it.

## Speech

Proposed by senpai (Claude Opus 5.5), 2026-10-24, for milestone 17, and Decided 2026-10-27 (Amy, by interview). The
first speech engine is Qwen3-TTS VoiceDesign on a B70 card (`AGENTS.md` milestone 17).

A speech engine declares the `speech` capability. It takes no pass through `pass`; its `pass` refuses, as the pass of
an engine with stages does. It has these calls:

```
speech_begin(slot, prompt, sampler) -> ok | why   feeds the prompt into the slot, and keeps the sampler there
speech_frames(slot, max) -> {codes, audio, end}   makes at most `max` frames and their audio; stops at end-of-speech
speech_clear(slot)                                frees the state of the slot
```

- **Prompt.** A prompt position has two ids: one from the text vocabulary and one from the codec vocabulary. Either id
  can be absent. Qwen3-TTS sums the projected text embedding and the codec embedding of each position. The family
  library makes the pairs (spec 11.3); the engine never takes text (D2).
- **Sampler.** The engine samples on the card, because one frame of Qwen3-TTS has 16 samples that depend on each
  other. The sampler has `temperature`, `top_k`, `repetition_penalty` and a `seed`. It has no `top_p`, because
  the request sampler fixes `top_p` at 1.0 (spec 11.8), and 1.0 keeps every id. The same identity,
  prompt, sampler and seed give the same codes and the same audio bits.
- **Frames.** A frame is 16 codes, one for each codebook, and 1,920 audio samples at 24 kHz (80 ms). `speech_frames`
  is one quantum of kind `decode`. The engine sizes `max` so that a quantum fits `max_quantum_ms` (D8).
- **Audio.** The codec decoder runs on the card in the same quantum. Its output is signed 16-bit PCM, mono, 24 kHz. The
  decoder keeps the codes of the last frames of the slot as its left context. Thus the audio of each quantum joins the
  audio before it.
- **State.** A slot holds the KV of the talker and the history of codebook 0 for the repetition penalty. It also holds
  the counter of the random stream and the codes of the left context. Until M18, the store kept no speech state:
  a request started with an empty slot, and `speech_clear` ended it.

**Held speech state.** Proposed by senpai (Claude Opus 5.5), 2026-10-25, for spec 11.8a. Decided 2026-10-27 (Amy, by
interview). Amy decided that a speech context holds state as a chat does ("Level 2 sounds awesome"). This replaces the
reason that a speech slot had no part to snapshot. The speech calls become these:

```
speech_feed(slot, positions, sampler?) -> ok | why  feeds positions after the state of the slot
speech_frames(slot, max) -> {codes, audio, end}     as before
speech_clear(slot)                                   as before
```

The state of a slot is read and written by the part capability that text engines declare ("Capabilities"). Thus
the store keeps a speech snapshot as it keeps any snapshot, with no call of its own.

- **Positions.** A position is a text position, a text id and a codec id, either of which can be absent, as before.
  Or it is a frame position: the 16 codes of a frame. The engine feeds it as a frame that it made. That is the sum of
  the 16 code embeddings and the trailing text `tts_pad`.
- **Feed.** `speech_feed` into an empty slot is the old `speech_begin`. Into a slot with state, it adds positions
  after the last one. A slot whose line has not ended refuses a feed.
- **A feed with a sampler starts a line.** The sampler, the history of codebook 0 and the codec state start again.
  The frame counter of the random stream starts at 0. The feed ends with the sample of the line's first codebook 0.
- **A feed with no sampler holds its rows** and starts no line. Reason: a build holds a snapshot at each boundary inside
  a context, and a line starts only after the last one.
- **Parts.** The engine declares one cells part, `talker_kv`: the talker KV of each position, one row each. This
  replaces a second part `speech_tail` for the history of codebook 0 and the codec's left context. Reason: each line
  starts both again, so only the KV carries over. Two slots with the same state give the same bytes.
- **Restore.** The nagaya writes the parts of a snapshot into an empty slot and sets its row count, as for a text
  engine (6.8). A feed then goes on from the last row.
- **Determinism.** A restore and a feed can give the same codes as one feed of all the positions into an empty slot.
  They do when the engine's passes keep the bits of a split (`segments_exact`), which the engine declares.

**Passes on separate cards of one group.** Proposed by senpai (Claude Opus 5.5), 2026-10-24, and Decided 2026-10-27
(Amy, by interview). Amy decided the rule on 2026-10-24 (`AGENTS.md` milestone 17, "Yes, design it now"); this is its
text.

- A pass of an engine with one device can run on its device. A stage of an engine with stages can run at that time
  on another device of the group.
- The group thread sends the pass to the stage thread of its device, so each device still runs one quantum at a time.
- Both engines must be loaded: the ledger must hold both (spec 3.3a).
- The scheduler of the group orders both engines' quanta by class and priority (spec 5.3, 5.4).

Reason: during one-token decode of an engine with two stages, each card waits while the other runs its stage. A pass
of a small engine on the waiting card costs the large engine nothing.

**How a group loads both.** Proposed by senpai (Claude Opus 5.5), 2026-10-24, for milestone 17 step 8. Decided
2026-10-27 (Amy, by interview). The rule above leaves the load open.

- An engine with one device loads beside a loaded engine with stages when the ledger admits both. It opens on the
  stage thread of its device, and its quanta run there. No engine unloads.
- When it does not fit, the load is a 409. The client names the engine with stages in `unload` (spec 3.3a).
- A load of an engine with stages unloads only the engines that the load names. Each other engine stays loaded on
  its device thread, and the engine with stages gives its handles to those threads (spec 3.2, "The device threads
  live as long as the group"). Proposed by Claude Opus 5.5, 2026-10-30. This replaces the moves of 2026-10-30.
- An unload of an engine on a stage thread runs there. The engine with stages stays loaded.
- A quantum of the engine with one device holds only its device. The scheduler ranks it with the stage quanta of
  that device by class and priority (spec 5.3, 5.4).

Reason: an engine opens on one thread and stays on it, because some engines need one thread (mk). A job body moves
to the stage thread for its quantum, and the engine never moves. The first rule of this list keeps the large engine
loaded when the small one fits.

## Not covered yet

- Streams. They need a period, a deadline and a late policy, in `limits()` or per job. A pass blocks, so the
  scheduler meets a stream's deadline, not the engine.
- Unloading an engine while its arena stays staged in host memory for a fast load again.
