# Engine ABI v0 (draft)

This doc defines the interface that each engine exports to the nagaya. The nagaya is an exokernel for megakernels:
it owns the machine, and this ABI is the one boundary between it and an engine. Below the boundary, an engine keeps
its layouts, its kernels and its caches. Above it, the nagaya keeps the ledger, the scheduler and the store.
`docs/ontology.md` defines the terms, and the first engines that this doc names.

The ABI comes from two sources: the C ABI of the first Qwen3.8 engine, and the first Gemma 4 engine. Their copies
live in `crates/nagaya-mk-qwen38-gfx1151` and `crates/nagaya-mk-gemma4-bmg`.

The ABI is a small core that every engine has, plus capabilities that an engine declares. Reason: the first Gemma 4
engine does not have half of the calls of the `Context` model of the first Qwen3.8 engine.

v0 covers three kinds of engine: inference, image batch, and training.

## Shape

The nagaya is Rust. An engine is a Rust type that implements the `Engine` trait. Some engines use another language
(The Qwen3.8 engine: C++ and HIP behind `extern "C"`). Such an engine exports a C ABI, and a thin Rust wrapper
implements the trait on top of it. The Gemma 4 engine implements the trait directly.

Each engine runs on a **device group**: the set of devices that one or more engines use. Each device group has one
nagaya thread. Calls for the engines in the group come only from that thread, and they are synchronous. A pass
blocks the thread.

An engine that spans devices declares stages (D1). Each stage quantum runs on the device thread of its device. The
device threads live as long as the group (spec 3.2). The Qwen3.8 engine has a group of one device and no stages.
The group of the Gemma 4 engine is both B70 cards, with two stages.

### Rust shape

The `Engine` trait holds the core calls. Each capability is its
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

**The factory gets its devices.** The Qwen3.8 engine uses the HIP device that is current on the calling thread.

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
  an address that it does not find, and an address that two devices give. The first Gemma 4 engine takes the cards in
  the order of `zeDeviceGet`, and that order can differ from the PCI order.
- A HIP engine selects its device with `hipDeviceGetByPCIBusId` on the thread of its device.
- An engine that gets the wrong count of devices refuses with `PlanError`.

Reason: the profile names the devices, so the engine must run on those devices and no others.

**Arena calls in v0.** These are the Rust shapes of the `arena`
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

Reason: the Gemma 4 engine checks its slabs on the open cards (`verify_stored_slab`), so the check and the attach
share one open of the cards. After a mismatch, `open` opens the cards again.

## Core (every engine)

```
plan(config) -> limits           reads no weights; the ledger admits open() from this result
open(config) -> engine           config: devices, weights, engine-specific settings (Gemma 4: slot sizes, max_block)
attach(config, arena_set)        the same, on an existing arena set
identity() -> {
    model, checkpoint_digest,    the logical digest of the weights (spec 12.2)
    tokenizer_hash, template, family, numbers, plan table,   family: toy | qwen38 | gemma4 (spec 11.3)
    bit settings: [{name, value, class}],   class: reference-defining | approx | policy
    canary_hash,
    build,                       the facts of the build; the identity digest does not cover them (spec 12.1)
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
    taps: [...], with the conditions each needs
}
pass(segments, taps) -> status, ms, outputs
    segment = {state, inputs, positions, meta, outputs wanted}
last_error() -> why
```

`limits()` after `open` must equal the result of `plan` for the same config. A difference is a defect, and the
nagaya refuses the engine.

`positions` and `meta` cover the `pos0` and the per-token `meta` of the first Gemma 4 engine (`PipeJob`). `outputs
wanted` covers `logit_rows`. `state` is a context handle when the engine has contexts. It is a slot index when the
engine has slots.

`pass` never allocates. Its scratch is a mandatory term, keyed by the kinds of quantum that use it. These calls do
allocate, and the ledger admits each one before it runs: `open`, `attach`, `ctx_new`, `ctx_fork`, `ctx_copy`,
`part_write`, and `resize`. (`set_n` and `cells_write` do not allocate: the Qwen3.8 engine sizes a context at
`ctx_new(max)`.)

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
control      control_write(index, bytes) / segment.control   control vectors in a table on the device (below)
layer_loops  segment.layer_loop                a span of layers that runs more than once (below)
```

An engine without `close` stays open until the process exits. The nagaya changes such an engine only through a
planned restart (spec 3.3, exit code 4). Its `Drop` does not free it, and a comment says why.

**`close` in Rust.**

- `close` is the `Drop` of the engine. The trait gets no new method.
- An engine that declares `close` frees each device allocation and each host allocation of its own in `Drop`. After
  the drop, the free memory of each of its devices is what it was before `open`.
- A gate of the conformance suite checks it on the device. It opens the engine, runs a pass and drops it, many times.
  It reads the free device memory after each drop.
- The arena fds that systemd holds are not the engine's own. The process removes them before the drop (spec 3.3a).

Reason: the `Box<dyn Engine>` already drops at the end of the group thread. A drop that frees everything needs no
second path, and a test can count the memory after it.

**The close of the Qwen3.8 engine.** HIP teardown stops the process (spec 3.3), so the Qwen3.8 engine's close frees
the weights and keeps the runtime.

- The drop of the last handle frees the contexts and the weights. The HIP runtime, the C++ engine and the pass host
  stay. A test binary keeps a `Hold`, so its handles do not load the weights again.
- The residue is not "what it was before `open`". `tests/hip_close.rs` measures it on gfx1151: 0.16 GiB of the
  process and 0.83 GiB of GTT, the same after each of three closes.
- A drop gives back 61 GiB of the process. Each open after a close gives the same logits bits, and the oracle hashes
  do not change.
- A load after a close refuses shards whose spot digest or arena size changed, because the identity names the first
  ones. A restart takes the new shards.

**Per-part coverage.**

- The park and the staging charge (spec 4.2) read `coverage(state, part)` for each cells part and each ring part.
  They store and count the rows of that coverage.
- `slot_high` stays the token count of the slot. A count of 0 means that the slot holds no token (spec 10.3).
- An engine whose row parts hold different rows must declare `coverage`. The Gemma 4 engine's do (below).

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

`cells_write` takes `from`, the position of its first row. The call
matches `cells_read(state, part, from, n, out)`. A Cells part appends from 0, so its write names the number of rows it
holds; the first write names 0. A Ring part may name a later position, because its window moves.

An engine that cannot place the rows at `from` returns `WriteResult::Refused` and changes nothing. It must not move
the rows to another position.

Reason: without `from`, a ring that holds positions 2 to 6 comes back from the store at positions 0 to 4. That is a
silent loss of state (spec 8.5).

## v0 Rust shapes

The ABI text names the calls and the fields, but it does not fix the Rust types.
This section gives the simplest shapes that fit the rules.

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
  arrives with its first use.
- The `Factory` trait is in the process crate. The fake engines are constructed directly.
- `Dtype` has the native dtypes of D10 rule 4: F32, F16, BF16, F8E4M3, F8E5M2, I8, I16, I32, I64, U8, U16, U32 and
  U64. `Dtype::element_bytes` gives the size of one element, and every crate uses it.
- `PartLimits` has `block_format: Option<String>` ("Canonical file bytes"). An `Output` has a `dtype`, so a
  compare of outputs can find NaN and Inf (spec 8.3).
- `Limits::checkpoint_grain` is `Option<u64>`, in tokens. `None` means that the engine makes no interval copies
  (spec 6.13).
- The `fake` feature is enabled for this crate's tests by a dev-dependency on the crate itself. The plain
  house test runs the fake tests; a normal build leaves them out.

Reason: with a default feature, a build can skip the tests. The dev-dependency ties the feature to the test build.

`Coverage::coverage` takes the `State`, because the ring's base position is in the state, not in the part bytes.
Without the state, a ring reports 0..n after a wrap.

### A pass that did not run

**A refused pass returns an error.** A pass that the Qwen3.8 binding refuses before it calls the Qwen3.8 engine returns
an error, not `PassStatus::Ok`. `PassStatus` stays `Ok | Poison | Timeout` for passes that ran.

The Qwen3.8 binding refuses a pass before it calls the Qwen3.8 engine when a segment breaks a rule of the Qwen3.8
engine. The rule can be a position that is not the context's n, a token past the vocabulary, or rows instead of tokens.
That pass did not run, and no context changed. A caller tells a refusal from a pass that ran, and it need not read
`last_error` to know.

A refused pass returns a `PassError` with kind `PassErrorKind::Refused`. Each engine refuses a pass with more
segments than `max_segments` in this way, before it changes any state.

A fault after the pass is different. The Qwen3.8 binding returns `Poison` when the logits buffer is shorter than the
wanted rows. The pass ran, so the state is unknown (D6, spec 8.2). A unit test pins this rule. The real engine gives
the full buffer, so the test calls the builder and not a pass.

### Canonical file bytes

The `safetensors` crate reads the canonical file, but its writer
aligns the header to 8 bytes and does not order the tensors. Thus this code writes the header and data itself.
The metadata keys are:

- `nagaya.part.<name>.kind`: `fixed`, `cells` or `ring`.
- `nagaya.part.<name>.coverage`: `start..end` for a cells or ring part.
- `nagaya.part.<name>.block`: the block format, when the dtype is `U8` for a block format.
- `nagaya.parent`: the parent content id in hex, for a delta.

Reason: D10 rules 3 and 5 need sorted keys and a 4 KiB data start. The crate cannot write them. The crate
still reads the file.

`PartLimits` has a field `block_format: Option<String>`, so `snapshot_from_engine` can write the block format. The
store refuses a block format on a dtype that is not `U8`.

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

`FakeFeed` has a feed rule (D11), in one stage or in two (D1). It has two layers, and each layer has one cells part. A
row is a hash of the row before it, the input of the layer, the position and the token count of the pass. Thus a wrong
split gives other bits, as in the Gemma 4 engine. The residual is the output of layer 0, so one stage and two stages
give the same bits. `FakeFeed::reverse_residual` breaks the stage path, and `FakeFeed::history` breaks the feed rule.

### Logit outputs

The Qwen3.8 binding gives logits one way, and the fake engine follows it.

- `logit_rows` holds positions that the segment feeds. A position outside `positions` is a refused pass.
- A pass gives one output for each entry of `logit_rows`, named `logits`.
  The order is segment order, then request order.
- The dtype is F32 and the bytes are little-endian. A row has one element for each token of the vocabulary.
- The vocabulary size is not in `limits()`. The family library knows it (spec 11.3).
- The fake engine gives logits only when it has a vocabulary size. Its outputs for one segment are `state`, then its
  `logits` rows. A pass of several segments with a vocabulary gives `logits` rows only.
  The Qwen3.8 binding gives `logits` only, and only for the last position.

Reason: the fake engine must serve a family with a vocabulary, and the output rows must compare as F32 in the gates
(spec 8.3). The simplest rule is the one that the Qwen3.8 engine already follows.

## How the Qwen3.8 engine maps

| ABI | The Qwen3.8 engine (`extern.hip`) | Notes |
|---|---|---|
| open / attach | `qwen38_engine_open(watchdog_deadline_s)`; no attach | no device argument: the HIP device is the calling thread's current one, so the wrapper sets it on the group's thread; the Rust reader parses the three GGUF shards, and the Rust host fills the arena; the C++ engine is the runtime layer only: the launches under the control block and the watchdog |
| plan | none yet | the wrapper computes the terms from the config and the Qwen3.8 engine's fixed shapes |
| arena | `arena::plan` and `arena::create` in Rust (`arena.rs`) | one sealed memfd; no check of another process's arena yet |
| identity | `identity` in `spec.rs` | built in Rust; includes `fused` and `QWEN38_FAST_QMM` |
| contexts | `context::ctx_new`, `ctx_free`, `ctx_n` in Rust (`context.rs`) | over the runtime calls `hip_rt_malloc`, `hip_rt_free` and `hip_rt_memset` of `rt.cpp`; no fork and no copy |
| parts | `context::state_read`, `state_write`, `cells_write`, `set_n` in Rust | over `hip_rt_memcpy` and `hip_rt_memcpy_2d`; after `set_n` the worker pools the index keys (`host.rs` `pool_context`, through `qwen38_launch_each`). Parts 0-2 and 5 fixed, 3-4 cells; `idx_pooled` derived. `cells_read` takes only rows 0 to n, the whole part: snapshots are stored whole (about 316 MB at n = 3584, ontology "Part") |
| slots | `ctx_n`, `ctx_free` then `ctx_new` | `slot_high` is the token count; `slot_clear` makes a new context of the same capacity |
| coverage | none | rows 0 to n of each cells part; `fits` compares the rows with the slot's capacity |
| seal | `qwen38_ctx_round_gdn_bf16` | changes bits in place |
| poison / last_error | `Qwen38Context::poisoned` (`context.rs`); `last_error` of the handle (`engine.rs`) | |
| part writes | `context::state_write`, `cells_write`, `set_n` | return nothing and stop the program on input they refuse; the wrapper checks every precondition first (part index, bytes, an empty context for parts 3-5, n within `max`, not poisoned) and returns refused; a write to a context that is already poisoned is refused, because this call changes nothing; after the call, the poison flag gives failed |
| close | the drop of the last handle or `Hold` (`engine.rs`) | frees the contexts, then the weights: the arena, the shard maps, the PLE table, the weight table (`worker.rs` `unload`); the next `open` loads them again from the same shards; the HIP runtime, the C++ engine and the pass host stay (see below) |
| pass_check / pass | `host::segments_problem`, `host::run_pass` in Rust; `qwen38_launch_fused` | tokens only (PLE needs token ids); the worker refuses a segment count outside 1 to 16, then a problem of `segments_problem`, then a poisoned context, then a bad knob; the host gathers the PLE rows (`ple.rs`), builds the stages (`pass.rs`), and launches them; outputs are logits per segment, then the tap outputs; a pass with a tap or a control item runs as several launches (see below) |
| taps | `limits().taps`: `residual` and `fold` | `cuts.rs`; a pass with `residual` runs as 49 launches, cut after each layer, and the host copies the last row of `h` of each logit row with `hip_rt_memcpy`; `fold` is the row of `mixed` that the head reads, copied after the last launch; the rules are in "The residual of the Qwen3.8 engine" |
| control | `control()`, `limits().control`, the term `control` | `cuts.rs` `ControlTable`: one device allocation of 15,728,640 bytes at open, after the scratch probe; `control_write` is one copy to the entry; the host op reads the vector of a layer from the device, applies the items to the rows of its segment at the cuts, and writes the rows back; no device code changes |
| telemetry | none | no op times and no event ring of a pass |
| determinism | `always` | `stage.h`: "every op's output bits depend only on its work items"; top-k ties break by index; bit settings in the identity |

### v0 binding choices

The binding makes these choices. Each is the simplest choice that fits the ABI.

- **A context is a state.** `State` is `Slot(u32)` in v0, so each context of the Qwen3.8 engine has a slot index.
  `limits().slots` declares the pool that one handle creates at open, as two classes of slots. The small class comes
  first, as in the fake engine.
  - Slots 0 to 11 hold 4,096 tokens each. Slots 12 to 15 hold 32,768 tokens each.
  - The process opens one handle. The conformance suite opens a handle for each gate.
  - The engine has no call that forks a context (`context.rs`: "the engine has no fork").
- **The Qwen3.8 engine declares `slots` and `coverage`.** The park on stop (spec 10.3) and the council build (spec
  11.3c) need both.
  - `slot_high(slot)` is the context's token count (`ctx_n` in `context.rs`). A slot with no context gives 0.
  - `slot_clear(slot)` frees the context and makes a new one of the same capacity, under the same slot. The Qwen3.8
    engine has no call that empties a context. The free comes first, because the context term holds one context for each
    slot.
  - A clear also empties a poisoned context (spec 8.2). `ctx_free` (`context.rs`) does not check the poison.
  - When the Qwen3.8 engine makes a context of another capacity, the slot has no context, and the clear gives `Failed`.
  - `coverage` gives rows 0 to n for part 3 (`kv`) and part 4 (`idx_keys`), from the byte count of the rows. It
    gives no position for a fixed part.
  - `fits` accepts a span of a cells part that starts at 0 and ends at or below the slot's capacity in tokens.
  - `cells_read` takes `from` 0 and `n` equal to the token count. Then it is a whole read. It refuses every other
    range, because the Qwen3.8 engine has no range read.

  **Open:** `context::ctx_new` stops the program when `hipMalloc` fails (`hip_rt_malloc` in `rt.cpp`). Thus a clear
  that cannot allocate ends the process, and the binding never sees a null context. The free comes first, so the
  allocator holds the bytes. The Qwen3.8 engine needs a reset call that keeps the buffers, so that a clear does not
  allocate.
  - The factory of the process crate gives `plan(config)` (next rule), and this binding fixes the pool.
- **The factory plans from `weights_bytes`.**
  - The size of the weight arena comes from `arena::plan`, which reads the tensor records of the shards at open. The
    arena is a repacked copy of the tensors, and `plan` reads no shard, so it cannot compute the size.
  - The config of the engine has one key, `weights_bytes`. The factory refuses any other key.
  - `plan` returns the limits of the pool with this size in the weights term.
  - `open` loads the weights and compares the real arena with the key. A difference refuses the start (exit 3,
    spec 10.4). The message gives the real size for the profile.
  - The factory reads `QWEN38_SHARDS` from the environment. The engine is in the binary, so no variable names a library.
  - `plan` refuses a missing `QWEN38_SHARDS`, or a shard that is not a file. The message names the variable.
  - The kind in the profile is `mk-qwen38-gfx1151`. A build with the feature `hip` registers it.
- **The pool is 12 contexts of 4,096 tokens and 4 of 32,768.** Reason: a context of a council v2 client needs
  thousands of tokens.
  - The count stays 16, which equals `max_segments`. A pass can hold one segment for each context.
  - The Qwen3.8 engine takes any capacity that is a multiple of 256, up to 32,768 (`context.rs` `ctx_new`).
  - Two sizes keep the long contexts few and the short ones many.
  - The measured bytes are in "The context term holds the pool", below.
  - To change the classes, edit `CONTEXT_CLASSES` in `spec.rs`. The tests derive their numbers from it.
- **A long prompt takes several passes.** A pass holds at most 256 tokens (`kMaxPassTokens`), so `max_pass_inputs`
  stays 256. The caller feeds a prompt in passes of at most 256 tokens. Each pass starts at the context's token
  count. The Qwen3.8 engine gives the same bits for any split (spec 6.13).
- **The Qwen3.8 engine declares a checkpoint grain of 256 tokens.** The GPU tests of spec 6.13 measured a copy round
  trip and pass splits across 256 tokens. The Qwen3.8 engine also declares `segments_exact`, which a grain needs. The
  grain is `CHECKPOINT_GRAIN` in `spec.rs`.
  - A profile with `interval_copies = 0` makes no copies. The profile of the Strix Halo machine has 0.
- **`max_quantum_ms` is 10000.** The Qwen3.8 engine splits a pass at 256 tokens (`kMaxPassTokens`). A prefill chunk of
  that size takes about 2 s. The Qwen3.8 engine's watchdog stops the program at a 10 s deadline (`worker.rs`
  `watchdog_deadline`), so the declared bound is 10 s.
- **`cells_write` writes `n` rows from `from`.** The binding refuses a `from` other than 0 and changes nothing. The
  Qwen3.8 engine writes n cells into an empty context that owns its K/V. The caller calls `set_n` after it writes every
  part.
- **`positions` are strict and `meta` is empty.** The Qwen3.8 engine appends the tokens at the context's n. The binding
  refuses a segment whose first position is not n, or whose positions are not contiguous. The Qwen3.8 engine reads no
  per-token metadata, so `meta` must be empty.
- **A pass is fused.** The restore's pool stages run one launch each (`qwen38_launch_each`). This binding declares the
  taps `residual` and `fold`, and a table of control vectors ("The residual of the Qwen3.8 engine"). A pass with a tap
  or a control item runs as several launches of the same kernel, cut at layer ends. A pass with neither runs as one
  launch.
- **The checkpoint digest is a spot check.** The binding hashes each shard's size and its GGUF header, as the Qwen3.8
  engine's `weight_hash` does. The digest starts with `spot-sha256:`.
- **`tokenizer_hash` is a hash of the GGUF tokenizer.** The Qwen3.8 engine loads no tokenizer file. The binding hashes
  the tokenizer keys of the GGUF metadata, and the family library computes the same hash from the same keys (spec 11.3).
  Thus a change of the tokenizer changes the recipe. The hash is `sha256:` and 64 hex
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

  Measured on gfx1151 by `examples/context_pool.rs`. Three runs gave equal device numbers.
  - `hipMemGetInfo` fell by 4,229,955,584 bytes for the 12 small contexts. That is 1,673,899 bytes above the state
    bytes for each context.
  - It fell by 7,929,331,712 bytes for the 4 large contexts. That is 2,023,424 bytes above for each context.
  - Both are under the declared bytes. `open` reads both numbers and refuses a pool that takes more than it declares.

  The host side (spec 4.7):
  - The weights take 61 GiB. One handle leaves about 51 GB of MemAvailable. Each further handle takes about 12.2 GB.
  - `open` refuses a pool that would leave less than the 12 GiB floor. Thus at most three handles are open at once.
  - A gate of the conformance suite holds at most three handles.
- **`canary_hash` comes from a table of records.** The key is the checkpoint digest, `fast_qmm` and `tiles`. The engine
  runs its canary at each load, after the scratch probe, and the open fails when the hash differs from the record. A
  key with no record is an error of `describe` and of `open`. A second form of the open, `open_with(Measure)`, compares
  nothing: a person uses it to measure a new checkpoint (spec 8.1).
- **The identity carries `QWEN38_FAST_QMM`, `QWEN38_TILES` and `fused`.** They are `Approx`, `ReferenceDefining` and
  `Approx`. The worker reads `QWEN38_FAST_QMM` at each pass build (`worker.rs` `fast_qmm`). `QWEN38_TILES` chooses the
  tiled or the GGUF layout at load (`worker.rs` `tiles`, `arena.rs` `plan`). The binding has not measured whether
  the two give equal bits. A change after open refuses the pass, as for `QWEN38_FAST_QMM`.

Reason for all: the code must not guess.

**The binary links the Qwen3.8 engine statically.**

- The crate holds a copy of the sources of the first Qwen3.8 engine in `hip/`. The layout under `src/` is the same.
- Under the feature `hip`, `build.rs` compiles the engine units into `libqwen38engine.a`. It uses ROCm's `clang++` and
  the flags of the CMake build of the first Qwen3.8 engine, flag for flag. A test compares them with its
  `compile_commands.json` (`tests/build_flags.rs`). The float flags `-ffp-contract=off -fno-fast-math` stay.
- The crate links the archive whole, so that each HIP object registers its kernels. The HIP runtime and the C++
  library stay shared.
- `build` holds `engine_build_sha256`, the SHA-256 of `libqwen38engine.a`. It also holds `engine_compiler` and
  `engine_hip`, the versions of the compiler and of HIP. No path is in `build` (spec 12.1).
- The build facts are not in the identity digest, because the canary decides (spec 12.1). Thus a new build with the
  same canary hash keeps the held entries.
- The compiler writes each object to stdout, and runs in the crate directory. Thus the archive has the same SHA-256
  for each profile and feature set. A path of the build directory in the command would change a HIP symbol name.
- The SHA-256 differs between two checkouts, because `__FILE__` and the debug information hold the source paths.
  Build the deployed binary from the same checkout each time.
- The Qwen3.8 engine has no call for the vocabulary size. `extern.hip` states `kVocab` and the other sizes of the
  binding's buffers in `static_assert` lines. A host test compares them with `spec.rs` (`src/layout.rs`).

Reason: the binary holds its engine. The digest of the archive is a fact of the build, so a pin would only
compare the build with itself.

**The Rust host and the extern layer.** The host code of the Qwen3.8 engine is Rust. It holds the GGUF reader
(`gguf.rs`) and the stage schedule (`schedule.rs`). It also holds the arena plan, the PLE gather and the stage build
(`arena.rs`, `ple.rs` and `pass.rs`):

- `qwen38_engine_open` takes only the watchdog deadline (`binding.rs`). The arena holds each tensor by name, in its
  final layout (`arena.rs`). The export of the arena (spec 10.2) builds it in Rust, from `arena.rs`.
- Rust mirrors `Stage` and the argument type of each op as `#[repr(C)]` types (`src/stage.rs`). One file holds each
  size, offset and constant of that mirror (`hip/src/engine/stage_layout.def`). Both the C++ build and the Rust build
  check against it, so a change on one side alone stops a build.
- `QWEN38_ELIDE` and the other knobs of the build are read in Rust at each pass, and a bad value refuses the pass.
- The C++ extern layer (`hip/src/engine/extern.hip`) holds only the calls that the binding uses. Each call takes C
  types and calls one `Engine` method or reads one field. The binding declares each call of the layer, and a test
  compares the two lists.

**The Qwen3.8 engine's pass scratch is a mandatory term.** The ABI says that `pass` never allocates and that its scratch
is a mandatory term. The Qwen3.8 engine allocates one scratch pool at its first ask and keeps it for the engine's life.
The term is `{name: "pass_scratch", tier: device(0), bytes: the measured constant, lifetime: engine, kinds:
[prefill, decode]}`.

The Qwen3.8 engine has no call for the size. The process measures the pool once on gfx1151 and declares that constant.
It refuses the engine when a later measurement is larger than the constant. **Open:** the declared bytes are the
measured device cost, which is 59 800 564 bytes above the pool's 263 934 988-byte ask. The HIP runtime's first launch is
the likely source of 38 522 880 bytes that the pool and the allocator's rounding do not explain.

The wrapper's host buffers also wait for the process crate.

### Test limits of the spin-timeout opt-in and the poisoned set_n path

These limits stay, and this section records them.

The rule of a spin-timeout is in Rust. A pass can stop on a spin-timeout that the watchdog did
not ask for. Then `host.rs` `run_pass` always poisons the contexts of the pass. Any other stop ends the process. The
test knob `QWEN38_MUTATE_STALL_BARRIER` can force a real spin-timeout, so a test of it is now possible; none is written.

Two paths have no test. A forced spin-timeout, and the `-1` to `PassStatus::Poison` path that follows it, need GPU
contention, so no test in this suite reaches them. The poisoned `set_n` path has no test on the real engine, because
`set_n` stops the program on a poisoned context (`context.rs`). A stub binding tests the refusal.

Reason: a test that passes when the code is broken is a defect, so the suite reports these limits instead of a false
check. A lab hook for the first case is not in v0.

### Test limit of the open-time part check

This limit stays, and this section records it.

The wrapper calls `parts_problem` at open (in the open path of `worker.rs`), and it
refuses a library with other part sizes before any write. The real engine matches the declared shapes, so the check
passes in every test.

Deleting the call leaves every test green. Only `parts_problem` has a test (in `spec.rs`): it refuses a larger part
with `a larger part must be a mismatch`. The call at open is not pinned. A fake library would pin it, and this suite
has none.

Reason: a test that passes when the code is broken is a defect. A person must see this limit before trusting the
check.

### The residual of the Qwen3.8 engine

The state between two layers of the Qwen3.8 engine is 4 streams of 2,560 values each. The residual
of a token is the 4 streams in stream order: 10,240 `f32` values. The tap `residual` and a control vector have this
width.

- **The tap `residual`.** For each logit row, it gives 48 rows of 10,240 `f32` values: the residual after layers 0 to
  47. Each row is the residual after the control items of its layer. It has no row for the head. The size is 1,966,080
  bytes for each logit row. Reason: spec 13.1 and the process give a tap one width, and this width equals the width
  of `limits().control`.
- **The tap `fold`.** For each logit row, it gives the row that the head reads: 2,560 `f32`
  values. The 4 streams fold into this row after layer 47. It is a tap of its own. Reason: the head's row is not a
  residual of the layers, and a latent step reads it.
- **The order of the outputs.** A pass gives one output for each tap and each logit row. The outputs follow the
  `logits` outputs. All the `residual` outputs come first, then all the `fold` outputs. Each group follows the order
  of the segments.
- **The launch cuts.** A pass with a tap or a control item runs as several fused launches of the same kernel, cut at
  layer ends. Between two launches, the host reads the rows of a tap, and the control op changes rows on the device.
  A pass with neither runs as one launch. The mutation knobs of the stage list refuse a pass
  with cuts, because they name stage indices of one launch.
- **The reason for the cuts.** No device code changes, and a plain pass keeps its stage list and its bits. A small
  change of the device code can fault the GPU, and the cause is not known. A card run measures the cost
  of the launches. The control op is a small kernel of its own that runs at the launch cuts (`control.hip`). An op
  inside the megakernel can take its place later, with equal bits.
- **The table.** `limits().control` is `{tables: 8, layers: 48, width: 10240}`. The term `control` is 15,728,640
  bytes on the device, with the lifetime `engine`. Eight entries are the count that LFM3 has.
- **The op.** The op runs on the device, at a cut, on the rows of the segment. No row goes to the host. The sum of
  `h . v` adds in 8 lanes by index modulo 8, and the lanes join as `((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))`.
  The host adds `v . v` in the same order when the vector is written. Each product and each sum is a rounded
  operation of its own: the code never fuses a product with a sum. An item with `v . v` of 0 acts on no row.
- **The oracle of the op.** The host op of `control.rs` is the oracle of the device op, and the gate is equal bits.
  The card gate `control_card.rs` runs each op on 256 rows whose sums depend on their order.
- **The coefficient.** Let `n` be the square root of `v . v`, and `p` be `h . v` divided by `n`. The coefficient `c` of
  `add` is `a`. For `ablate`, it is `-a * p / n`. For `clamp`, it is `(a - p) / n`. The op sets `h[j]` to
  `h[j] + c * v[j]`.
- **The fused combine.** A pass with one control action or more builds with the fused combine of the next layer's
  norm off. Without this, the norm of layer `l + 1` reads the row before the op changes it. The two forms give equal
  bits, and a card gate checks that.
- **The tap `lens`.** The Qwen3.8 engine does not declare `lens` yet. It needs 48 logit rows for each logit row, and
  the scratch of the logits holds 16. The lab routes that need it come later.
- **The latent rows.** A latent step reads the tap `fold`. The host scales the row to the mean norm of the rows of
  `token_embd`. The pass copies the row over the embedding row of the position. The record is the kind 5 with the
  width 2,560, as for LFM3. The placeholder of a latent position is the padding token that the
  GGUF tokenizer keys name.

## How the Gemma 4 engine maps

This table maps the code of the first Gemma 4 engine, the source of `nagaya-mk-gemma4-bmg`, onto the ABI.

| ABI | the first engine | Notes |
|---|---|---|
| open / attach | `load_slots(cluster, model_dir, slot_seqs, max_block)`, `load_slots_stored` | slot sizes and `max_block` are load-time config |
| device group | both B70 cards | one group thread and one stage thread for each card (D1) |
| stages | `run_card0`, `run_card1` | stage 0: layers 0-14 on card 0; stage 1: layers 15-29 and the head on card 1 (`CARD_LAYERS`); the residual is F32 rows of 2,816 values, through host memory |
| arena | 30 DMA-BUF slabs, one for each layer on its card (`slab_fd_name(l, card)`), `verify_stored_slab` | spot check: first and last 64 bytes per tensor; the first engine gave the head no slab, and the nagaya's copy gives it one, 31 slabs in all (spec 10.2) |
| slots | `set_slot`, `kv_high`, `slot_clear` | no contexts, no stored token count, no fork |
| parts | `save_kv_layer_from` (a range read), `load_kv`, `load_kv_layer` | `state` is the slot index; KV only: cells on global layers, ring on sliding layers |
| part names | the K and V planes of each layer (`KvSnapshot`) | `k.<ll>` and `v.<ll>`, F32, two digits for the layer; see "The parts of the Gemma 4 engine" |
| geometry / coverage | slot size, ring rows (`kv_ring`, `kv_keep_from`); `KvSnapshot::join` is the fit check | changes rows held, not bits, so no `geometry_key`; the id is a recipe (`snapshot_id`) |
| pass | `run_card0` then `run_card1`; `pipeline_stage` for jobs in flight on both cards | the nagaya's scheduler takes the place of `pipeline_stage` (D1) |
| segments | `PipeJob`; `branch_block` with per-token `meta` | a branch block is atomic and is not a snapshot |
| inputs | tokens; `write_embeddings` looks up the rows on the host and puts them into `Buf::X` | tokens (D2) |
| class | `CardLock`, `GpuLock` | moves out of the engine into the nagaya scheduler, one choice for each device (D1) |
| determinism | `always`, with a feed rule (D11) | bits depend on the token count of a pass; the first port runs one recipe in two starts and compares content ids |

### The parts of the Gemma 4 engine

The first engine keeps only KV (`new_slots`). The device holds each layer's K and V as F32 planes. These parts follow
from that layout:

- Each layer has two parts, `k.<ll>` and `v.<ll>`, with `ll` the layer in two digits. The dtype is F32, so the store
  converts no value.
- A global layer is a layer `l` with `l % 6 == 5`: 5, 11, 17, 23 and 29 (`layer_geom`). Its parts
  are cells, with the row shape [2, 512]. Its coverage is positions 0 to n.
- Each other layer is a sliding layer. Its parts are rings of 4,096 rows (`KV_RING_ROWS`), with the row shape
  [8, 256]. In a slot of 4,096 positions or fewer the ring never wraps, so the coverage is 0 to n. In a longer slot
  it is the last 4,096 positions (`kv_ring`, `kv_keep_from`).
- A ring plane has 1,024 more rows of spill (`KV_RING_SPILL`), which `OP_COPY` moves to the start of the ring. The
  spill is a device layout, so the canonical file never holds it.
- The canonical file (D10) unrolls each ring into position order. A token costs 450,560 bytes in all parts
  (`kv_bytes_per_token`).
- Each part names its stage (D1): layers 0-14 are stage 0 on card 0, and layers 15-29 are stage 1 on card 1.
- `fits` follows `KvSnapshot::join`. A ring part that starts after position 0 fits only a slot
  with a ring.

### The first Gemma 4 binding

The trait on the copy (`crates/nagaya-mk-gemma4-bmg/src/engine.rs`) needs
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
  positions. It took 1,184 ms on card 1 and 920 ms on card 0. Measured by
  `profile_longest_stage_quantum` in `tests/xe.rs`.
- **The checkpoint digest.** It is `blake3:` and a BLAKE3 hash with the key `nagaya gemma4 checkpoint v1`. The hash
  covers the canonical `config.json` (sorted keys, no whitespace), then each tensor in bytewise order of its name.
  A tensor adds its name, its dtype, its shape and the BLAKE3 hash of its bytes. The engine computes it at open.
- **The numbers and the lineage.** `numbers` holds the SHA-256 of the kernel binary, the versions of `ocloc` and IGC,
  the driver version and `max_block`. The plan table names the compiler, the driver and the PCI vendor, device and
  revision of each card (spec 12.1).

Reason: the code must not guess, and each choice keeps the bits of the copy. A test compares the logits of the trait
with the logits of the copy before the trait, bit for bit (`the_trait_gives_the_bits_of_the_copy`).

**The wake call.** A stage thread keeps its card awake between the quanta of a job. Not built yet.

Reason: a card that idles for 10 ms or more waits 100-550 us at its next command (`B70Card::wake`). The first Gemma 4
engine keeps the idle card awake in `pipeline_stage`. Without a wake, decode after 20 tokens gives 46.1 tokens/s on the
stage threads and 47.1 on the copy, about 2% less.

**The shape of the wake call**, and when the stage thread makes it.

```rust
trait StageHandle {
    // ... stage_pass and parts as above
    fn wake(&mut self) -> Result<(), PassError> { Ok(()) }   // the first Gemma 4 engine: B70Card::wake
}
```

- The stage thread calls `wake` when it has no quantum and 3 ms passed since its last call to the device.
- It calls `wake` only while a job of its group holds a slot lease (5.6). With no job, it makes no call, so the
  card can idle. Reason: a card that the nagaya keeps awake uses more power at idle.
- A wake is not a quantum. It charges no job, and the scheduler does not see it.
- A wake that fails is a pass failure of the stage (spec 8.2).
- An engine whose devices do not sleep keeps the default, which does nothing.

Reason for 3 ms: a test thread that wakes both cards each 3 ms gives 47.6 tokens/s.

## Decisions

**D1. Engines that span devices.** An engine can declare stages and export `stage_pass`. The nagaya then schedules
each stage on its device. Reason: the scheduler can fill an idle card. A model split across machines needs the same
residual boundary later. The Gemma 4 engine declares stages. The rules below give the segment shape and the hold of
the slot.

A group can serve more than one engine. Two engines that use one device share one group, one thread and one
scheduler. With stages, that thread runs the scheduler, and it has stage threads under it.

**The stage segment.** A stage pass takes segments. Each segment has these members:

- `state`: the slot. Every stage of a block names the same slot.
- `positions` and `meta`: the first position and the per-token `meta`. A plain block has no `meta`.
- `inputs`: the tokens, on the first stage only (D2). A later stage reads the residual.
- `logit_rows`: the rows of the block whose logits the last stage gives. The Gemma 4 engine takes at most 16
  (`MAX_LOGIT_ROWS`).
- the residual: a host buffer that the nagaya owns (D5). Stage s writes it, and stage s + 1 reads it. A row is
  `limits().residual_row_bytes`: 2,816 F32 values (11,264 bytes) for the Gemma 4 engine.

A restore is not a member of the segment. In the nagaya, `StageScheduler` has `Item::Restore` for a per-stage copy
quantum. The v0 job path (`StagedJobs`) runs a restore in a whole-group `copy` quantum (`Item::Whole(Kind::Copy)`).
Each stage thread writes its own parts before the first block.

Each part names its stage, `limits().parts[].stage`. A stage handle reads and writes only the parts of its stage.
The parts of the Gemma 4 engine are the KV of one layer, so each part has one card.

The Rust shape of `stages`.

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

The threads of a group with stages.

- The group thread runs the one scheduler of the group (spec 5.5). It sends each stage quantum to the stage thread
  of its device, and it gets the end back.
- A stage thread holds the handle of its device. It runs one quantum at a time, and it makes every call to its
  device.
- The ledger stays on its own thread (spec 3.2). The group thread asks it by message.

Reason: one scheduler sees both cards, so it can fill an idle card. One thread for each device keeps each card busy
while the other runs.

**The hold of the slot.** A job holds its slot with a lease from its first quantum to its end (spec 5.6). The
scheduler orders the stage quanta of the slot:

1. Stage s + 1 of a block is ready only after stage s of that block ends and writes the residual.
2. Stage s of block k + 1 is ready only after stage s of block k ends.
3. No quantum of another job writes the slot while the lease lives. This is quantum condition 5 (ontology).

Thus two blocks of one job run at the same time on the two cards. Each stage writes the rows of its own layers
only, so the two stages never write the same bytes.

**Classes for each device.** Between quanta on each device, the scheduler runs the ready stage quantum of the highest
class for that device (spec 5.3, 5.4). A stage quantum charges its run time to its job, as a pass does. Thus a
council read takes card 0 while a generation's stage 1 runs on card 1.

**Bounds and the watchdog.** A stage quantum is a quantum (spec 5.1). `max_quantum_ms` bounds each stage quantum, and
the engine chooses its `feed.block` to fit it. Each stage thread beats a heartbeat of its own. The watchdog reports a
stage quantum past the bound, and a stall of any stage thread stops the process (spec 7.4).

**The residual term.** The residual buffers are a mandatory host term of the engine, `residual`, with lifetime
`engine`. It has two buffers for each boundary between stages. One holds a block for stage s + 1, while stage s
writes the next block. Each buffer holds `feed.block` rows. The term of the Gemma 4 engine is 2 × 512 × 11,264 =
11,534,336 bytes.

**The stage path in v0.** The first build of the stage path makes these choices.

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
  applies spec 5.5 to a group with stages. Reason: D1 fills an idle card.
- **A stage handle has slots and coverage.** `StageHandle` also has `slots()` and `coverage()`, for the parts of its
  stage. A slot clear and the base position of a ring are device state of each stage. After `take_handles`, the
  `Engine` gives `None` for `parts`, `slots`, `coverage` and `seal`.
- **What the first build refuses.** An engine with stages must not declare `seal` or a checkpoint grain. Each stage
  has its own device. The `residual` term holds 2 × (stages − 1) × rows × `residual_row_bytes` bytes. The rows are
  `feed.block`, or `max_pass_inputs` for an engine with no feed rule. The process refuses other limits at open.

Gemma 4 does not fit on one B70 card. Each speech, image or embedding engine uses one card. Such an engine is in the
group of Gemma 4, and it runs on the device thread of its card (spec 3.2).

**D2. Inputs.** An engine declares tokens, rows, or both: `inputs = tokens | rows | tokens+rows`. An engine never
takes text. The family library in the nagaya turns text into tokens (spec 11.3).

The Qwen3.8 engine needs token ids even when it gets rows. Its per-layer embeddings (PLE) hash the token ids, and its
PLE history is a fixed part (`engine.h`, `ple_rows`). Audio rows and diffusion latents have no tokens.

The Gemma 4 engine declares `inputs = tokens`, as the Qwen3.8 engine does. It looks up each embedding row on the host,
scales it, and writes it into card 0's residual (`write_embeddings`). Thus the embedding stays inside the engine. The
ABI keeps `rows` and `tokens+rows` for audio rows, diffusion latents and media (see "LFM3").

**D3. Snapshots are named by content ids.** There are two keys:

- The **content id** names a stored snapshot: the BLAKE3 hash of its sealed bytes (`blake3:<hex>` in council v2).
  When the snapshot is a piece of a chain, the hash also covers its parent's content id. The Qwen3.8 engine stores
  snapshots whole, so its ids have no parent. Dedup, integrity, pins, and moves of snapshots between machines all use
  the content id.
- The **recipe** is a hash that the nagaya computes, also when the engine is not loaded. It covers the identity
  (bit settings included), the inputs and the boundaries. For an engine whose bits depend on slot geometry, it also
  covers `geometry_key`. Spec 6.2 holds the full list of items and the encoding. The store maps a recipe to one or
  more content ids. Thus the store can answer "do we already have this?" before a build, and a prefix lookup is only
  hashing.

One recipe can have several content ids, because coverage differs. In the Gemma 4 engine, slot size decides which
rows a snapshot *holds*, not what their bits *are*. `kv_ring` and `kv_keep_from` choose the positions that a slot
keeps. A ring row holds the same K and V for its position as a full plane does. `join` fails on missing rows, not on
different bits.

Thus each stored snapshot records its coverage (positions held, per part), and fit is a separate check (`coverage`
capability). The content id holds no slot geometry.

**The bits do not depend on the slot size.** The split attention of a block (`attn_pv_segments`, `megakernel.cl`)
chooses its count of key segments with no slot size. The host gives the partial sums enough scratch: 2^21 plus the
block's floats (`tasks::attn_pv_scratch_floats`). The card test `the_same_feed_gives_the_same_bits_in_each_slot_size`
checks it. An engine whose bits change with a slot size declares `geometry_key` instead.

The attention of a block of the LFM3 backbone has no key segments. It cuts the keys into chunks
(`attn_stream_rounds`), and the size of a chunk reads no slot size (`tasks::attn_stream_scratch_floats`).

Two content ids under one recipe that cover the same positions have two possible causes. The engine did not
reproduce its bits, or the recipe misses an input. The nagaya refuses to serve that recipe. It quarantines the
conflicting entries and reports them. It never picks one. A person clears the refusal after the cause is known.

A BLAKE3 hash of a whole snapshot of the Qwen3.8 engine (about 316 MB) takes tens of milliseconds, small next to a
pass.

That refusal is a tripwire. It has a meaning only when the engine promises the same bits. Each engine declares which
promise it makes.

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

When the refusal is off, an **audit** takes its place. The audit runs across restarts. After each start, a filler job
in the nagaya builds a sample of stored recipes again and compares the content ids. The new start is the fresh
process, so there is no second GPU client. The audit reports each mismatch and uses only idle time. ds4-recipe does
this by hand ("pinned plans for every experiment and byte-equality gate").

**D4. Cost before a pass.** An engine gives no cost estimate, and it reports the time after the pass. There is no
estimate call in v0. The nagaya fits a cost model per engine from pass times and segment sizes. It uses the fit for
store eviction value-per-byte and deadline checks between quanta. The engine picks its own chunk sizes for
`max_quantum_ms`.

**D5. Rust trait first, C ABI as a binding.** Handles are opaque pointers. Errors are a status plus `last_error`. The
caller owns the buffers.

**D6. Poison stays fatal.** The nagaya frees a poisoned context and never reads it. For a slot engine, it marks a
poisoned slot dirty until `slot_clear`, `part_write`, or a pass from position 0 resets it. A pass that returns a
poison status poisons every context or slot in it (the Qwen3.8 engine: "-1, every context poisoned"). The nagaya
fails those jobs, keeps the others, and runs the canary before more work runs (spec 8.2).

**D7. Seal and park are separate calls.** Seal runs only at boundaries in the chain, and it can change bits. Park
uses the `parts` calls to move state between tiers bit for bit, for both contexts and slots. The scheduler parks. It
never seals to free a slot.

**D8. Quanta have a declared length bound.** An engine declares `max_quantum_ms` in `limits()`. The engine cannot know
a pass's time before it runs. Thus it splits long work into chunks whose size it measures to fit the bound (the
Qwen3.8 engine: 256-token prefill chunks). In v0 there are no yields inside a pass. A quantum still runs to its end,
so the worst wait behind other work is one quantum.

A watchdog thread reports each quantum that runs past the bound.

The size of a quantum is a balance. Amy, 2026-11-22: "I'd say 1-10s chunks are fine in the nagaya, we want there to
be opportunities to pipeline and have hot caches and so on, but also we do some chunky work sharing, and want health
checks". Thus a bound from 1,000 to 10,000 ms is in range. Split expensive work at a rough boundary where a switch
of work is possible, even if the scheduler seldom switches there.

Each engine chooses the bound that suits its model and its device (Amy, 2026-11-22). The guidance is a quantum
below 10 s, and an engine can make its own tradeoff. Amy, 2026-11-22: "if we have a case where we need to allow for
a continuous run for a few minutes let's make it possible."

**A long pass with a declared bound** (accepted 2026-11-22).
Before this rule, the watchdog did not read `max_quantum_ms`. It stopped the process when a group sent no beat for
30 s, and a pass sends no beat (spec 7.4). Thus no pass could run for more than 30 s. The rule is built, and spec 7.4
("A long pass") gives the watchdog and the route.

- `Limits` has `max_long_quantum_ms: Option<NonZeroU32>`. `None` means that the engine has no long pass. When it is
  set, it is more than `max_quantum_ms` and at most `LONG_QUANTUM_CEILING_MS`, 180,000 ms. An engine with stages
  declares none. `Limits::check` refuses each other value.
- `Engine` has `pass_bound_ms(segments, taps) -> Result<Option<NonZeroU32>, String>`, with a default of `Ok(None)`.
  It is a host call that does no device work and changes no state. `Some(b)` declares a long pass. `Err` refuses the
  pass before it runs.
- `Limits::check_pass_bound` holds the rules of `b`. The engine declares a ceiling, and `b` is more than
  `max_quantum_ms` and at most the ceiling. The pass holds one segment. A bound that breaks a rule refuses the pass
  before it runs.
- The process asks for the bound of each pass just before the pass, after the scheduler chose the quantum. A long
  pass keeps the order of classes. The scheduler runs the ready quantum of the highest rank, and it joins only
  plain decode rows of one token.
- The watchdog arms a deadline for that pass alone: its start plus the larger of the timeout and `b` plus the beat
  interval. An ordinary pass keeps the timeout of 30 s. A pass past `b` gives one report. A pass past the deadline
  stops the process, as a stall does. A result that returns after its deadline is never published.
- The read-only route `GET /nagaya/v1/devices/{id}/busy` shows the long pass to other clients: the group, the devices,
  `bound_ms`, `elapsed_ms` and `overdue`.
- The scheduler does not read the bound. It chooses by the order of classes, as for each quantum. The bound comes
  from the segments, and the body of a job builds them in its quantum, after the choice. Decided 2026-11-22. Amy:
  "if I want to prioritize interactive work, the thing should be scheduled as batch in the first place".
- Not in the first version: a long stage quantum, and beats from the device while a pass runs.
- Reason: a group timeout that grows with the largest bound slows the detection of a real hang in every pass. A
  bound for each kind of pass cannot tell a short feed block from a long one. A declared bound for each pass keeps
  the health check of each ordinary pass.
- The Inkling 2 engine is the first user: the tower of a large image, about 57 s, in one pass ("The time of a large
  image"). Tower slices (D8, "The time of a large image") stay the better path where they give useful boundaries.
- The image calls of SDXL and Qwen-Image do not use `pass`, so they get no long pass. They declare a
  `max_quantum_ms` of 60,000, which the watchdog does not read. Each of their quanta is one step or one decode, and
  each such quantum must still end within the timeout of 30 s.

ds4-recipe chose cooperative yields after a short turn waited 283.95 s behind a long prefill. The nagaya uses the
bounded wait instead.

**D9. Spare memory goes by share.** Each machine's placement profile gives every engine on it a share. The ledger
first charges all mandatory bytes. It then divides what remains among the engines' elastic terms by those shares
(after ds4-recipe). Each elastic term gets at least its minimum. A profile whose mandatory bytes and minimums do
not fit is refused at start.

**D10. Snapshots at rest are canonical safetensors.** In device memory, each engine keeps its own layout. The model
and the kernels decide that layout, and it stays private.

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

The content id is the BLAKE3 hash of this canonical file. Each engine converts between its layout and the canonical
form on the way into and out of the store. For the Qwen3.8 engine, that is close to the copy it already does: its copy
re-strides cells by capacity (`engine.h`). For the Gemma 4 engine, the conversion unrolls each ring into position
order.

A park from device memory to host memory can keep the engine's native bytes, if the conversion is too slow for short
parks. Such a park has no content id. It gets one when it goes into the store. **Open:** measure the conversion on
the Qwen3.8 engine and the Gemma 4 engine before we use this exception.

**D11. A feed rule.** An engine whose bits depend on the token count of a pass declares a feed rule. The scheduler
then feeds every kept span in one fixed split, so the bits depend only on the inputs.

The Gemma 4 engine chooses the reduction of a matmul by the token count of the pass (`megakernel.cl`):

- 1 token: `SPLIT_K` = 4 subgroups share the K of one row;
- 2 to 8 tokens: XMX tiles of 16 rows, with K split over `KSPLIT` = 8 subgroups;
- 9 to 15 tokens: XMX work items of 16 rows by 32 tokens;
- 16 tokens or more: a work-group takes 64 tiles and shares the tokens' x in local memory.

Thus the split changes bits. In one measurement on the B70 cards, the engine fed 32 tokens in 1 block, in blocks
of 5 and in blocks of 1. The largest logit differences from HF were 5.8e-5, 3.4e-5 and 3.2e-5.

The same split gives the same bits. Five runs of a 600-token prefill in blocks of 256, then 24 one-token steps, gave
equal logits. The Gemma 4 engine feeds in passes of `max_block` tokens (`Gemma4Engine::block_jobs`).

The rules:

1. `limits().feed` is `{block}`, in tokens. The scheduler feeds each span in passes of `block` tokens from the start
   of the span. The last pass holds the remainder.
2. A span starts at position 0 or at a boundary. It ends at the next boundary, or at the end of the fed tokens.
3. `block` is a bit setting with the class `reference-defining`. Thus it is in the identity and in every recipe
   (spec 6.2). The `block` of the Gemma 4 engine is its `max_block`, 512 (`spec::DEFAULT_MAX_BLOCK`).
4. An engine with a feed rule declares `segments_exact: false` and no checkpoint grain (spec 6.13).
5. Its `determinism` stays `always`. The rule fixes the split, so the same identity and inputs give the same bits in
   any process.
6. A pass that writes rows that the store keeps has one segment. Segments whose rows no one keeps may share a pass,
   such as reads and branch blocks. Their outputs fall under the tolerance of the served model.
7. A slot can hold a span in part, when its last pass held fewer than `block` tokens. A job that continues the span
   feeds that last pass again, with the new tokens after it.

Spec 6.14 gives how decode steps, drafts and branch blocks meet the rule.

The store keeps the decode rows of a reply under the recipe of a decoded head. Thus:

- In rule 2, a span also starts at a decoded head. A decoded head is a seal, as a boundary is.
- Rule 6 stays as it is. A reply whose decode row shared a pass holds no decoded head, so the store keeps no row of
  a pass with two segments.

Reason: the feed rule keeps the recipe as tokens plus boundaries. The other way is a kernel whose reductions do not
depend on the token count (spec 5.7). That costs speed on the one-token path.

## Training

A training engine declares the `train` capability:

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

**The image capability.** An image engine declares the `image` capability. Its `pass` refuses. Reason: SDXL has two
text encoders, token weights, a sampler with history, a VAE decode and previews. A segment of tokens has no place for
them, as it has none for speech ("Speech"). The capability has these calls:

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
  kind `step`. The SDXL engine has img2img and inpainting (spec 11.9b), and no hires pass.
- **Spacing and LoRAs.** The limits are those of spec 11.9a, with a rank of at most 128. `ImageJob` takes `spacing`
  (`leading` or `trailing`) and `loras`, a list of each LoRA's bytes and weight. The engine parses the bytes and adds
  each delta to its module's output (spec 11.9a). The engine refuses `trailing` with a DPM++ sampler. Reason: the
  spacing of the schedule is the engine's, and the spec gives the rule.
- **Image inputs.** Reason: the nagaya decodes and checks each PNG on the route, so the engine takes pixels, as it
  takes tokens and not text (D2).
  - `ImageJob` takes `init`: the RGB bytes of the image, 3 for each pixel, and `strength`. It also takes `mask`: the
    blurred gray mask, 1 byte for each pixel. The engine makes the binary mask from it.
  - `image_begin` cuts the schedule to its last `n` steps (spec 11.9b). The first `image_step` of a job with `init`
    runs the VAE encode as its quantum, and it gives step 0 with no previews. The nagaya sends no `progress` for it.
  - The random stream takes the purposes `encode` and `encode masked` for the draws of the image latents.
  - The engine reads the input channels of its UNet from `conv_in.weight`, 4 or 9. A 9-channel UNet refuses a job
    without `init` and `mask`. A 4-channel UNet blends the latent after each step.
  - `image_decode` gives the decode with no paste. The nagaya pastes the kept pixels back before the PNG encode.

**The declaration of a 9-channel UNet.** Reason: spec 11.9b says that a 9-channel UNet without an image and a mask is a
400 that names `messages[i].content`. A refusal of `image_begin` does not say which part the engine refused.

- The image capability gets the call `image_needs_mask() -> bool`. It is true when the engine refuses a job without
  `init` and `mask`. An SDXL engine with a 9-channel UNet gives true. The default is false.
- When `image_begin` refuses a job without a mask, and `image_needs_mask` is true, the route names the content of the
  last `user` message. Other refusals keep their parts.
- The engine still checks the job itself, and the call adds no quantum. Thus `PassError` keeps its form.

**Qwen-Image 2.1.** Reason: the job above has the shape of SDXL. Qwen-Image has one encoder with no
chunks, a flow sampler, a cached conditioning and an RGBA decode. Each change adds a form or a member, and no call.

- `ImageJob.prompt` takes one of two forms. `chunks` is the form of SDXL above. `tokens` is the form of Qwen-Image:
  the ids of the whole template, and the cut (spec 11.9c). An engine refuses the form that it does not read.
- `clip_skip` is in the `chunks` form, because only a CLIP encoder reads it. The `tokens` form has no negative
  prompt, because Qwen-Image refuses one (spec 11.9c).
- `ImageSampler` gets `FlowEuler`. The request value `euler` gives it on a flow model, and `Euler` on SDXL. Qwen-Image
  refuses every other sampler, `Trailing`, a `cfg` other than 1.0, LoRAs and previews. SDXL refuses `FlowEuler`.
- `ImageJob` gets `conditioning`: the rows of a hit in the store (spec 6.16), or none. With rows, the engine skips the
  encoder. It refuses rows whose count is not the id count less the cut.
- `image_begin` returns the rows that the encoder made on a miss. It returns none on a hit, and none on an engine
  with no `conditioning` capability. The nagaya makes the canonical file of spec 6.16 from them.
- The capability `conditioning: {row_bytes, max_rows, encoder_digest}` names the size of the rows and the encoder.
  Qwen-Image declares 8192 and 1010. An engine without it never gets rows. The call `image_conditioning` of the image
  capability gives it, and gives none for an engine with no such cache, such as SDXL.
- `encoder_digest` is the encoder digest of spec 6.16. It does not cover the transformer or its LoRA. Thus variants
  with the same encoder share their conditionings.
- The engine still declares no part. The nagaya keeps the rows in the store, and the engine never reads the store.
- `image_decode` gives 8-bit images with a channel count: 3 for RGB, or 4 for RGBA. Qwen-Image gives RGBA, because
  its VAE has 4 output channels. SDXL gives RGB. The nagaya writes the PNG with the same channels.
- A slot of Qwen-Image holds the latents, the conditioning rows, and their keys and values in each of the 32 blocks.
  The keys and values of a template of 1024 ids take 505 MiB. The plan counts them for each slot.
- `image_begin` is one quantum of kind `embed`, and each step and the decode are one quantum of kind `step` (spec
  11.9c). The engine declares the most pixels of a slot: 4 images of 2,097,152 pixels.
- The engine declares the determinism `pinned`, with the plan table of spec 12.1.

## Embedding

**The embedding capability.** Reason: an embedding reads tokens and gives one vector. It has no slot state, no logits
and no sampler. A segment of a pass has no place for the pooled vector, as it has none for an image ("Image").

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

The first speech engine is Qwen3-TTS VoiceDesign on a B70 card.

A speech engine declares the `speech` capability. It takes no pass through `pass`; its `pass` refuses, as the pass of
an engine with stages does. It has these calls:

```
speech_feed(slot, positions, sampler?) -> ok | why  feeds positions after the state of the slot
speech_frames(slot, max) -> {codes, audio, end}     makes at most `max` frames and their audio; stops at end-of-speech
speech_clear(slot)                                  frees the state of the slot
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
  the counter of the random stream and the codes of the left context.

**Held speech state.** A speech context holds state as a chat does. The part capability that text engines declare
reads and writes the state of a slot ("Capabilities"). Thus the store keeps a speech snapshot as it keeps any
snapshot, with no call of its own.

- **Positions.** A position is a text position: a text id and a codec id, either of which can be absent, as in the
  prompt. Or it is a frame position: the 16 codes of a frame. The engine feeds it as a frame that it made. That is
  the sum of the 16 code embeddings and the trailing text `tts_pad`.
- **Feed.** `speech_feed` into an empty slot feeds the prompt. Into a slot with state, it adds positions after the
  last one. A slot whose line has not ended refuses a feed.
- **A feed with a sampler starts a line.** The sampler, the history of codebook 0 and the codec state start again.
  The frame counter of the random stream starts at 0. The feed ends with the sample of the line's first codebook 0.
- **A feed with no sampler holds its rows** and starts no line. Reason: a build holds a snapshot at each boundary inside
  a context, and a line starts only after the last one.
- **Parts.** The engine declares one cells part, `talker_kv`: the talker KV of each position, one row each. No part
  holds the history of codebook 0 or the codec's left context. Reason: each line starts both again, so only the KV
  carries over. Two slots with the same state give the same bytes.
- **Restore.** The nagaya writes the parts of a snapshot into an empty slot and sets its row count, as for a text
  engine (6.8). A feed then goes on from the last row.
- **Determinism.** A restore and a feed can give the same codes as one feed of all the positions into an empty slot.
  They do when the engine's passes keep the bits of a split (`segments_exact`), which the engine declares.

**Passes on separate cards of one group.**

- A pass of an engine with one device can run on its device. A stage of an engine with stages can run at that time
  on another device of the group.
- The group thread sends the pass to the stage thread of its device, so each device still runs one quantum at a time.
- Both engines must be loaded: the ledger must hold both (spec 3.3a).
- The scheduler of the group orders both engines' quanta by class and priority (spec 5.3, 5.4).

Reason: during one-token decode of an engine with two stages, each card waits while the other runs its stage. A pass
of a small engine on the waiting card costs the large engine nothing.

**How a group loads both.**

- An engine with one device loads beside a loaded engine with stages when the ledger admits both. It opens on the
  stage thread of its device, and its quanta run there. No engine unloads.
- When it does not fit, the load is a 409. The client names the engine with stages in `unload` (spec 3.3a).
- A load of an engine with stages unloads only the engines that the load names. Each other engine stays loaded on its
  device thread, and the engine with stages gives its handles to those threads (spec 3.2, "The device threads live as
  long as the group").
- An unload of an engine on a stage thread runs there. The engine with stages stays loaded.
- A quantum of the engine with one device holds only its device. The scheduler ranks it with the stage quanta of
  that device by class and priority (spec 5.3, 5.4).

Reason: an engine opens on one thread and stays on it, because some engines need one thread (the Qwen3.8 engine). A job
body moves to the stage thread for its quantum, and the engine never moves. The first rule of this list keeps the large
engine loaded when the small one fits.

## LFM3

The engine kind `mk-lfm3-bmg` serves `LFM3-1.7B` and `LFM3-Omni-2B` on one B70 card (spec 11.11).

**Inputs.** The engine declares `inputs = tokens+rows` (D2). The tokens are the ids of the family, with one
placeholder for each position of a media part. The rows are the prepared media: the patches of an image or the mel
bins of a clip, in f32, from the family. `meta` names each media span: its first position, its count of positions,
and its kind. The encoder of the span runs inside the pass, and its rows replace the placeholders. Reason: the
recipe names the object, not the rows of the encoder (spec 11.11), so the encoder output need not leave the card.

**The rows of a span.** Reason: this is the layout of the reference with the fewest changes.

- An image gives one row of 1,536 values for each patch of 16 by 16 pixels. A row is the 3 channels, then the 2
  copies in time, then the pixels. The rows go in the order of the merge blocks (`image_processing_lfm3_vl.py:186`).
- The `meta` of an image also names its grid: `grid_t`, `grid_h` and `grid_w`. The span has `grid_t * grid_h *
  grid_w / 4` positions.
- A clip gives one row of 128 mel values for each frame. This is the transpose of the reference, which holds the
  bins first. The `meta` of a clip names its count of frames, and the span has `ceil(frames / 8)` positions.

**Speech.** Omni declares the capability `speech_frames`. Its `pass` gives logits as a text engine does. The host
samples each text token, so the text sampler stays in one place (spec 11.2). The calls `frame`, `codec` and
`codec_clear` make the audio ("The speech calls of LFM3").

`frame` runs the depth decoder on the last row and samples 12 codes. It feeds the frame into the slot as its next
position, so the next `pass` or `frame` reads it. `end` means a codebook gave the end code (spec 11.11). The end frame
is fed too, as the reference feeds it.

The codec keeps its window for each slot, and it holds no state in a snapshot. The window is the sliding attention
of the codec decoder: 30 positions at 75 Hz, or 5 frames
(`configuration_liquid_codec.py:57`). In the oracle, a decode frame by frame gives the decode of the whole clip within
1.2e-5.

**The text model.** `LFM3-1.7B` declares no media and no `speech_frames`. Its engine is the same code with other
weights, and the open checks that the config has no vision tower, audio encoder or depth decoder.

### The first LFM3 binding

The text path of the backbone (`crates/nagaya-mk-lfm3-bmg`) follows these rules.

- **The config.** The kind is `mk-lfm3-bmg`. Its config has four keys. `model_dir` names the model folder, and it
  has no default, because one kind serves two models. `slots` is a count of slots of 32,768 positions, or a list of
  the positions of each slot. `max_block` is the feed block: 512 when the key is absent, and 1 to 512.
- **The reserve.** `extra_runs` is the reserve of layer loops in each slot ("Layer loops of LFM3"). It is 0 when the
  key is absent, else a multiple of 4 up to 64.
- **Slot sizes.** A slot holds 1 to 262,144 positions, the
  `max_position_embeddings` of LFM3-1.7B. The list goes from short to long, and the engine refuses another order.
  The host gives a job the lowest free slot that holds it (`slots::take`), so the order makes it the shortest. A
  long job then takes a long slot, and a short job leaves it free while a short slot is free.
  - Each slot charges its own bytes to the term `kv`. The 6 full layers take 24 KiB of each position, and the
    sliding layers take fixed rings. A slot of 32,768 positions takes 868,859,904 bytes, and one of 262,144 takes
    6,505,365,504.
  - The buffers of a block hold no scores of the backbone. The attention of a block keeps a running softmax over
    chunks of the keys (`OP_ATTN_STREAM`).
  - The longest slot sizes the scratch of the ring, and no other part of the term `work`. The scratch is 60 MB for
    32,768 positions and 270 MB for 262,144.
  - The attention of a block reads no slot size ("The bits do not depend on the slot size"). Thus a long slot gives
    the bits of a short slot, and the recipe does not change. The card test `xe_feed` checks the bits and the quantum.
  - The speech codec of Omni keeps its bound of 6 codec positions for each of 32,768 positions. A frame past it is
    refused.
- **The parts.** A conv layer has `conv.<ll>`, a fixed F32 part of 6 rows of 2,048 values, oldest first. An attention
  layer has `k.<ll>` and `v.<ll>`, F32 rows of 4 heads of 128. They are cells on the full layers and rings on the
  sliding layers. `ll` is the layer in two digits.
- **The rings.** A ring has 2,048 rows and a spill of 1,024 rows (`KV_RING_SPILL`). Its coverage is the positions
  that it holds: the last 2,048, and after a restore none before the restored rows. A restore of a ring part must
  hold the last 256 positions.
- **No way back.** A pass starts at the token count of its slot. `set_n` with no write keeps the count or sets it to
  0. A write empties the slot until `set_n`, as for Gemma 4. Reason: the conv state holds only the last 6 inputs, so
  the slot cannot go back. A job that continues a span (D11 rule 7) restores its last boundary first.
- **Position 0.** A pass from position 0 starts from conv states of zeros, as the reference does.
- **Logits.** A logit row has the width of the family's vocabulary, 125,033 values (spec 11.11). A token id at or
  past that width is refused. The head runs once for each logit row, so the bits of a row do not depend on the count
  of rows.
- **Inputs.** The text path takes tokens. "The LFM3 vision binding" gives the rows of the media (D2).
- **The checkpoint digest.** It is `blake3:` and a BLAKE3 hash with the key `nagaya lfm3 checkpoint v1`. It covers the
  canonical `config.json` and each tensor of `model.safetensors`, as for Gemma 4. Omni's other towers are in it, so a
  later step that runs them keeps the digest.
- **The identity.** `model` is the served name: `lfm3` or `lfm3-omni`. `numbers` hold the BLAKE3 of the kernel
  binary, the compiler and driver versions, `max_block` and the ring rows. `feed.block` is a bit setting of the class
  `reference-defining` (D11 rule 3).
- **The memory terms.** `weights`, `kv` and `work` are on `device(0)`, and `pinned` is host memory. `open` refuses a
  device size that is not the sum of the terms. `max_quantum_ms` is 2,000, a budget that a card test measures.
- **Not built.** The engine has no arena, no seal and no stages. A restart uploads the weights again.

### The audio encoder of LFM3

The audio encoder of Omni follows these rules.

- **The entry.** The module `audio` of `nagaya-mk-lfm3-bmg` gives the tasks of one clip. The tasks read the mel rows
  of the clip on the card. They write `ceil(frames / 8)` rows of 2,048 values to an address that the caller names.
  The tasks go into the ring of the pass that reads the span, before the layers.
- **The longest clip.** A clip has at most 4,096 frames, which is 512 positions or about 41 s. A longer clip is a
  400 that names the part. Reason: a span of 512 positions fits the largest feed block (`max_block`).
- **The memory.** The engine of Omni loads the weights of the encoder and the projector at the open. It holds one
  set of buffers for the longest clip. The terms `weights` and `work` hold their bytes.
- **One clip at a time.** The encoder holds no state between clips, as in the reference. Each conv layer starts
  from a state of zeros, and the attention reads only the keys of its clip, at positions from 0.
- **The plane.** The tasks keep the plane of the downsampler with the channels last. The open puts the columns of
  `lin_out` in that order once. Thus each 1 x 1 convolution is a plain matmul.

### The speech calls of LFM3

These are the speech calls of Omni ("Speech" above).

```
frame(slot, draw, logits) -> {codes, end, logits?}   draw = {seed, frame, history}
codec(slot, codes) -> samples                         f32 samples at 24 kHz
codec_clear(slot) -> samples                          the last 480 samples; it frees the window
```

- **The capability.** `speech_frames` is its own capability, the trait `SpeechFrames` of `nagaya-abi`. Omni declares
  it beside the parts, the slots and the coverage, and its `pass` stays a text pass. Reason: the capability `speech`
  of Qwen3-TTS refuses each pass. The accessor of the trait is `Engine::frames`, because `Speech` has a call named
  `speech_frames`.
- **The hidden row.** `frame` reads the final norm of the last position of the slot. Each pass of Omni keeps that
  row. After a restore, a write or a clear, the slot holds no such row, and `frame` refuses until a pass runs.
- **The history.** The caller gives the codes of the frames of the output so far. The engine reads the last 64 of
  them, and it keeps no history (spec 11.2, "The draw keeps no state"). The end frame is not in the history, as in
  the reference (`modeling_lfm3_omni.py:1734-1739`).
- **The logits.** With `logits`, `frame` also gives the logit row of the frame that it fed. Reason: in `interleaved`
  mode a text token can follow a frame, and the host samples it (spec 11.11).
- **The draw.** The uniform value is the BLAKE3 hash of `nagaya audio draw v1`, the seed, the frame and an index.
  The seed and the frame are 8 bytes each, little-endian. The index is 1 byte. The value then follows spec 11.2.
- **The order.** The order of the codebooks is a Fisher-Yates shuffle of 0 to 11. The swap at `i`, from 11 down to
  1, takes `j = floor(u × (i + 1))` with the index `i`. The three passes reveal the order in groups of 4.
- **The pick.** Codebook `g` draws with the index `12 + g`. The engine applies the penalty to the f32 logits and
  divides by the temperature. It then picks as spec 11.2 does, in f64.
- **The audio.** `codec` gives f32 samples, and the host turns them into PCM. The first call after a clear gives
  1,440 samples, and each later call gives 1,920. `codec_clear` gives the last 480. Thus `n` frames give `1,920 × n`
  samples, as the stream of the reference does (`modeling_liquid_codec.py:341-356`).
- **The overlap-add.** The card runs the decoder, the polar form and the inverse FFT with the window. The host adds
  the frames and divides by the envelope. Reason: that is 2,880 additions for each frame, and the tail of 960
  samples then stays in host memory.
- **The ring of the codec.** Each attention layer of the codec keeps its keys and values in a ring of 128 rows for
  each slot. The positions of the codec run from 0 after a clear, 6 for each frame. A slot holds at most 196,608 of
  them, 6 for each position of the slot.
- **The memory.** Omni adds the term `speech` on `device(0)`. It holds the depth decoder, the codebook table, the
  codec, their buffers, and the codec state of each slot. `LFM3-1.7B` has no such term.

### The LFM3 vision binding

The vision tower of Omni follows these rules.

- **The meta of a span.** `meta` holds one record of 32 bytes for each media span, in the order of the positions.
  A record is little-endian. It holds the kind (1 for an image, 2 for a clip), the count of positions, the first
  position as `u64`, and four words. For an image they are `grid_t`, `grid_h`, `grid_w` and 0. For a clip they
  are the count of mel frames and three zeros. Reason: the rules above name these values, and a fixed record needs
  no parser.
- **The rows of a segment.** They are the rows of each span whole, in the order of the records.
- **A span longer than a block.** An image takes up to 2,048 positions, and a feed block holds at most 512. Each
  pass that holds positions of a span gets the rows of the whole span. It runs the encoder and keeps the rows of
  its own positions. Reason: the encoders hold no state between quanta (spec 11.11), and the attention of the tower
  reads every patch. The cost is one run of the tower for each block of the span.
- **Placeholders.** Each position of a span in the block holds the placeholder of its kind. Each placeholder in the
  block is in a span. A pass that breaks either rule is refused, so a placeholder never enters the backbone as its
  embedding row.
- **Inputs.** Omni declares `tokens+rows`, and `lfm3` declares `tokens`. A segment of tokens alone, with no `meta`,
  is a text block for both. Thus the gates of 8.3 that make token segments run on an engine that declares
  `tokens+rows`. Only an engine that declares `rows` skips them.
- **The ops.** The tower runs on the LFM3 binary. The binary also holds the ops of the embedding binary, and its
  `OP_GEGLU` stays SwiGLU. The ops of the tower are `OP_VISION_ROPE` (100), `OP_VISION_CONV` (101) and
  `OP_VISION_ATTN` (102), the attention in both directions. It gives the bits of `OP_BIATTN` (`docs/perf.md`,
  "The vision tower on a large image").
  The LayerNorm of the projector reads F32 weights, so the open uploads its BF16 weights as F32. Each value is exact.
- **The memory terms.** The weights of the tower and of the projector are the term `vision` on `device(0)`. The
  buffers of the tower are in `work`, sized for the largest image of 8,192 patches. `lfm3` has neither.
- **The output.** The rows of the encoder stay on the card. The pass copies them over the embedding rows of the
  placeholders, before the first layer of the backbone.

**Batched passes.** Several contexts share one decode pass of the LFM3 engine:

- A context's bits never depend on the other segments of its pass. Each row of the matrix multiply for 2 to 8 rows
  sums in one order. Thus the decode takes that path, also for one row.
- The engine declares 8 segments. The weight ops run once over the rows of all segments. Each row has its own
  rotation, attention, convolution and K and V rows, on its own slot.
- `segments_exact` stays false, because LFM3 has a feed rule (D11 rule 4). Decode segments share a pass by D11 rule 6,
  because the store keeps no decode row of such a pass (spec 6.14). Other segments run one in a pass.
- A pass joins only segments with equal control lists.
- Omni's speech frames stay one context in a pass.

### Latent rows of LFM3

- **A latent span.** A new kind of media span carries rows of `width` f32 values. The pass copies each row over the
  embedding row of its placeholder, with no encoder. Its `meta` names the first position and the count.
- **The first loop is on the host.** Each step is a pass of one position with a latent span and the tap of the
  residual after the last layer. A client that reads along also wants the logit row. The host applies the final
  norm and the scale, and it gives the row to the next pass. Reason: this needs no new kernel. A trial shows if a
  loop on the card is worth its cost.
- **The rows of the merge.** The host reads the K and V rows of each part with `cells_read`. It turns the keys of the
  full attention layers, and it writes the result with `cells_write`. The rotated half of a key turns by the difference
  of its new and old positions, with the theta of its layer.

The layout of a latent span has these rules. Reason: the engine must check the record and the placeholder.

- **The record.** The kind is 5. The first of the four words is the width of a row, and the other three are 0.
- **The placeholder.** A latent position holds `pad_token_id` of `config.json`: `<|pad|>`, id 124,893 in both models.
  The family names the same token. A latent row of another width is refused.
- **The declaration.** `LFM3-1.7B` keeps `inputs: tokens`. A latent span is a lab input, and no council route sends
  rows to the text model.
- **The copy.** The host writes the rows into the buffer of the normed rows. A copy puts them over the embedding rows
  after the embedding. Thus the text model needs no vision tower for it.

### Layer loops of LFM3

Both LFM3 models have 24 layers. Layers 0 and 1 are convolutions, and a group of four follows from layer 2: full
attention, convolution, sliding attention, convolution. Layers 22 and 23 start a sixth group. Each rule below is
**Proposed** (Claude Opus 5.5, 2026-11-18).

- **The limits.** `bounds` is `[2, 6, 10, 14, 18, 22]`, the edges of the groups. `count` is 4. `extra_runs` comes
  from the profile, and it is a multiple of 4. With 0, the engine declares no `layer_loops`.
- **The reserve.** Entry `i` has the kind of layer `2 + i mod 4`. Each span starts at a group edge and holds whole
  groups, so extra run `i` is a layer of that kind. An entry of a full layer holds cells, and an entry of a sliding
  layer holds a ring, as the layer does.
- **The parts.** The parts of entry `i` follow the parts of the layers, with the names `k.x<ii>`, `v.x<ii>` and
  `conv.x<ii>`.
- **The ring.** The builder walks the runs in order. Each run takes the weights of its layer, and the address of the
  state of the layer or of its entry. No kernel changes: each task of a run is a task of a plain pass at other
  addresses.
- **The cost.** An entry costs what its layer costs in each slot. A slot of 32,768 positions holds the planes of a
  full layer as 2 × 32,768 rows. The ledger charges the reserve in the slot's size.
- **Taps and speech.** The tap `residual` and the lens read the row after the last run of each layer. Omni keeps the
  hidden row of the last run for `frame`.

## Control vectors

This section holds the engine's part of the rules of control vectors.

```
limits().control: {tables, layers, width}?   tables vectors fit on the device at once
control_write(index, bytes) -> ok | why      writes one vector into table entry index; never allocates
segment.control: [{index, op: add | ablate | clamp, strength: f32}]   in order, for every layer the vector holds
taps: residual                               the residual after each layer at each logit row (spec 13.1)
taps: lens                                   the final norm and the head of each layer's row (spec 13.1)
```

- **The table.** `open` allocates the table as a mandatory term with lifetime `engine`. Thus `pass` and
  `control_write` never allocate. The worker keeps the content id of the vector in each entry, and it writes an
  entry again when a job needs another vector.
- **The bytes.** `control_write` takes `layers` times `width` `f32` values, little-endian, in layer order. A layer
  that the vector does not hold is all zeros, and the op skips it. Other sizes get "why", and nothing changes.
- **The op.** After layer `l`, before layer `l + 1` reads it, each item changes each residual row `h`. Let `v` be
  the vector of layer `l`, and `α` the strength. `add` gives `h + α·v`. `ablate` gives `h − α·(h·v̂)·v̂`, and
  `clamp` gives `h + (α − h·v̂)·v̂`.
- **Bits.** The dot product and the norm of `v` sum in `f32` in a fixed order (D3). An `add` or an `ablate` with
  `α` 0 must give the bits of no item, so the op skips such an item. A `clamp` always changes `h`: with `α` 0, it
  removes the concept, as an `ablate` with `α` 1 does.
- **Segments.** The segments of one pass can name other lists. The worker never puts two lists on one state: the
  list is part of the root recipe of the state (spec 6.2).

Reason: a table that `open` sizes keeps the rule that `pass` never allocates. A vector of 24 layers at 2,048 values
is 196,608 bytes. Thus a table of 8 entries is 1.5 MiB on a card of 32 GiB.

## Layer loops

This section holds the engine's part of the rules of layer loops (spec 11.16, gogi "Layer loops"). Each rule is
**Proposed** (Claude Opus 5.5, 2026-11-18).

```
limits().layer_loops: {layers, bounds, count, extra_runs}?   the spans that the engine runs more than once
segment.layer_loop: {start, end, count}?                      layers start to end - 1 run count times
```

- **The order.** A pass runs layers 0 to `start - 1` once, then layers `start` to `end - 1` `count` times, then the
  other layers once. Each run reads the residual that the run before it wrote.
- **The state of a run.** The first run of a layer uses the state of the layer. Extra run `i` uses entry `i` of the
  reserve of the slot, in the order of the runs. An entry holds the state of one layer: keys and values, or a
  recurrent state. Every run reads the weights of its layer.
- **The reserve.** `open` allocates `extra_runs` entries in each slot, so `pass` never allocates. The parts of a
  state list the entries after the layers. A snapshot holds them, as it holds each part.
- **The bounds.** A span starts and ends at a bound. An engine whose layers have several kinds names the edges of
  its repeating group. Thus entry `i` has the same kind in each loop, and the parts of a state do not change.
- **Checks.** A segment with a loop to an engine without `layer_loops` is refused. So is a loop that breaks a limit.
  The segments of one pass name one loop, or none.
- **Bits.** A loop of `count` 1 gives the bits of the same segment with no loop. A pass with no loop keeps its bits.
- **Control.** An item acts after each run of each layer that its vector holds.
- **Taps.** The tap `residual` gives the row of each layer after its last run.

Reason: the reserve keeps the rule that `pass` never allocates, and fixed parts keep the store's rules of a snapshot.
The worker joins only the segments of one loop, so the engine never mixes two loops in one pass.

**The worker of the table.** The worker keeps the content id of each entry (above). It follows these rules.

- A job claims its vectors when it goes into the queue. The claim ends when the job ends.
- The worker binds the vectors of a job in its first quantum, a copy quantum, before its first pass. A held vector
  keeps its entry. A vector that no entry holds goes into an empty entry. Else it goes into the entry of least recent
  use whose vector no claim needs.
- When no entry is free of claims, the job is a 429 `busy`. It changes no entry.
- The worker counts the opens of the engine. A new open has a new table, so the worker forgets each entry.
- An engine with stages and `control` is not built. A pass of another job can be in flight during a copy quantum
  (D1).

Reason: a claim keeps an entry that a queued job needs, and the least recent use keeps the vectors that clients use
most.

## Inkling 2

The engine kind `mk-inkling2-bmg` serves Inkling 2 Mini and the variable-loop research model on one B70 card
(milestone 35). The facts come from the checkpoint and from transformers `inkling2_loop` at `046bdb8cc2`. Each rule
is **Proposed** (Claude Opus 5.5, 2026-11-21).

```
limits().cycles: {min, max, default}?   the counts of cycles that the engine runs
segment.cycles: u8?                      the count of cycles of each position; none gives `default`
taps: residual                           the row after each run, in the order of the runs
```

**Runs and cycles.** A cycle is one walk through layers 6 to 17. A position runs layers 0 to 5, then `count`
cycles, then layers 18 to 23. Cycle 1 reads the residual after layer 5. Each later cycle starts again from that
residual and from the block outputs after layer 5. Each of its layers also reads a message from the cycle before it,
through four learned gates: query, key, value and expert input. At 3 cycles a position has 48 runs.

**Cycles are not layer loops.** A layer loop runs a span again on its own output ("Layer loops"). A cycle starts
again from layer 6 and reads a message. Thus a cycle has a capability, a limit and a segment field of its own. The
engine declares no `layer_loops`.

- `min` is 1, and `default` is 3. Mini is post-trained at 3 cycles, and the research model at 1 to 5.
- `max` is 1 + `extra_runs` / 12. `extra_runs` comes from the profile: 24 by default, and 48 at most.
- A segment with a count outside `min` to `max` is refused. The segments of one pass name one count.
- The count is part of the root recipe of a context. A context keeps one count: each run of an earlier position
  holds keys and values that a later position reads.
- The ABI has `Capability::Cycles`, `CyclesLimit {min, max, default, start, end}` and `Segment.cycles`, a count or
  none (`Limits::check_segment_cycles`). A segment names a loop or cycles, and not both.
- gogi has the context member `cycles` (gogi "Cycles"). The council passes it to `segment.cycles` (spec 11.17).
- **A slot keeps its count.** A slot that holds positions refuses a pass at another count, and the error names
  both counts. A clear drops the count, and the next pass sets it.
- **The part `cycles`.** It is the last part of a state: a fixed part, `u32 [1]`. A read gives the count of the
  slot, or 0 for an empty slot. A restore writes it. Thus a snapshot holds its count, and a restore keeps it.
- **The parts that a count does not reach.** A run is active when it is outside the cycles, or in a cycle up to the
  count. The parts of an inactive run of the reserve read as zeros: rows and conv. The engine does not read the card
  for them. Thus one recipe gives one content id, whatever ran in the slot before.
  - An inactive ring holds the last `RING_ROWS` positions of the slot, or all of them in a short slot. An inactive
    cells part holds every position from 0.
- **A restore.** The writes come in any order. `set_n` with a count of 1 or more needs a write of each part.
  - The count of `cycles` is 1 to `max`. A second write of another count is refused. A count of 0 restores only
    an empty slot.
  - The rows of an active part follow the rules of "The host spec". An inactive part must hold zeros, else
    `set_n` is refused, and the error names the part.
  - `set_n` takes the count with the positions. The rings start at the first restored row of an active ring.
  - The active rings must start at one position, else `set_n` is refused, and the error names two of them.
    Reason: the base of a ring is one position for each slot (`Held::base`). A later start would hide the rows
    of the other rings before it from each read after the restore. Each snapshot that the engine reads has one
    start, so the rule refuses only a snapshot from another source.
- Reason: a fixed part keeps `limits().parts` static and the ABI as it is. Zeros on a read cost no write to the
  card. A snapshot of 1 cycle still holds the reserve rows, as zeros (review by ChatGPT through kaibo `gpt-plan`).
- The identity names `state` 2, the layout of these parts. Thus a snapshot of the parts before `cycles` does not
  share a recipe with one after.
- Each rule of this list from "The part `cycles`" is **Proposed** (Claude Opus 5.5, 2026-11-21).

**The weights on the card.** Expanded to `bf16`, the experts take about 49 GB, which is more than the 32 GiB of the
card. Thus the experts stay in the format of the checkpoint: NVFP4 in layers 0 to 19, and MXFP8 in layers 20 to 23.

- A kernel expands each value as the reference does: the product in `f32`, then a round to `bf16`.
- The factor of an NVFP4 expert is `amax / denominator`. The loader of the reference rounds each of the two to
  `bf16` first, and the engine does the same (`reference_expert_factor`). Reason: then the weights of the engine are
  the weights of the oracle. In 3 of 4 slices, the `f32` factor moves 148 to 1,294 values (Proposed, ChatGPT by
  kaibo `gpt-plan`, 2026-11-21).
- A loader of a later transformers that moves these bits is a new engine identity, and the gap is measured again.
- In the checkpoint, row `2i` of `w13` is the gate row `i`, and row `2i + 1` is the up row `i`. The engine keeps
  that order.
- The other weights keep the dtype of the checkpoint: `bf16`, and `f32` for the queries of the residuals, the
  router bias and the message queries.
- Every sum is `f32`. A run in `bf16` differs from the `f32` reference by about 0.2 in each logit (milestone 35
  step 1).

**The parts.** Each run has three parts, and `ll` is its layer in two digits:

- `k.<ll>` and `v.<ll>`: `f32` rows of 8 heads of 128, after the clip of the key and the norm of the value. They are
  cells on the full layers 5, 11, 17 and 23, and rings of 2,048 rows on the other layers.
- `conv.<ll>`: a fixed `f32` part of 3 rows of 6,656 values, oldest first. A row holds the inputs of four
  convolutions, in this order:
  - the key, 1,024 values;
  - the value, 1,024 values;
  - the output of the attention, 2,304 values;
  - the output of the experts, 2,304 values.
- The runs of cycles 2 and later take the parts of the reserve: `k.x<ii>`, `v.x<ii>` and `conv.x<ii>`. They go in
  the order of the runs, as in "Layer loops".

**The cost of a slot.** At 3 cycles, 8 runs of the full layers hold cells: 65,536 bytes of each position. The 40
runs of the sliding layers hold rings of 2,048 rows and a spill of 1,024 rows, as for LFM3: 960 MiB. The conv parts
take 3.8 MB. Thus a slot of 32,768 positions takes 3,157,950,464 bytes, about 2.9 GiB. Each more cycle adds 2 runs
of cells and 10 rings.
- A ring holds more rows than the window of 1,024. Reason: a block of 512 writes its keys before its attention
  reads them. Its first query still needs the key 1,023 positions before it. The keys of one block can span
  1,535 positions (review by ChatGPT through kaibo, 2026-11-21).
- Each position of the longest slot also adds 3,120 bytes to the term `work`: the scratch of the attention for a
  block of 512. Thus a position costs 68,656 bytes at 3 cycles, and 101,424 bytes at up to 5.
- **The largest slot, as measured.** The card run of 2026-11-21 measured one slot on one B70 card, with a feed
  block of 512 (milestone 35 step 11). Each size is the largest slot whose buffers all stay in the memory of the card:
  - With `extra_runs` 24: 164,352 positions. The term `kv` is 11,781,439,488 bytes, and the process holds
    33,374,960 KiB of the card. At 164,864 positions, 130,732 KiB go to system memory.
  - With `extra_runs` 48: 106,496 positions. The term `kv` is 11,984,683,008 bytes. At 107,008 positions, a part
    goes to system memory.
- **The open refuses a slot past the card.** The driver puts a buffer that does not fit in system memory. Then a
  block of 512 from position 0 takes 4.0 to 5.0 s, where it takes 0.78 s on the card. Thus the open reads the
  memory of the process on its card before and after its allocations (`nagaya_mk_bmg::vram`). These are the keys
  `drm-total-vram0` and `drm-resident-vram0` of each `xe` fd of `/proc/self/fdinfo` (spec 9.6). The open refuses in
  two cases, and it frees what it took:
  - the memory of the card grew by less than the device bytes of the terms and of the ring;
  - a byte of a buffer of the process on the card is not resident there.
- **The open fills the ring before the check.** Under memory pressure, Level Zero creates each buffer with
  `DRM_XE_GEM_CREATE_FLAG_DEFER_BACKING`. Such a buffer gets its memory only at its first use. Until then, fdinfo
  counts it in `drm-total-system`, and it is not resident. The open writes each buffer of the terms, but not the
  scoreboard and the scratch of the ring. Thus the open fills them with zeros (`Ring::zero_device`), and then it reads
  the memory again.
- The service on 2026-11-21 showed this. It refused the open of two slots of 32,768 positions. The card had grown
  by 31,220,137,984 bytes of 31,279,353,476. The scratch of the ring, 105,252 KiB, had no memory yet. After
  the first pass, it was on the card. The card gate `tests/xe_fit_deferred.rs` sets the debug keys of Level Zero
  that defer each buffer, and it opens the same slots.
- The card fills its own two buffers at its open: the device copy of the tasks and the wake buffer (`open_card`).
  The check counts them with the scratch and the scoreboard, as the term `work` does (`spec::ring_device_bytes`).
- The driver also places the state of a launch at the first launch. Thus an open that passes the check runs the
  halt alone, and then it checks again. The first check comes first, because the launch of an open that the
  card does not hold fails with no device memory.
- Before review round 2, the first pass took more memory of the card after the check. With deferred memory it
  took 983,040 bytes: the device copy of the tasks and the state of a launch. The gate now checks that the first
  pass takes no memory. Without deferred memory, the open of 164,352 positions now holds 67,186,688 bytes more,
  and it still passes (`tests/xe_fit.rs`).
- The card gate `tests/xe_fit.rs` ran on 2026-11-21. A slot of 262,144 positions asks for 40,679,296,644 bytes.
  The driver puts 34,031,300,608 of them on the card, and the open is refused. Then a slot of 164,352 opens, and it
  holds 34,016,722,944 bytes of the card.
- Each rule of the two items above is **Proposed** (Claude Opus 5.5, 2026-11-21, by the review of DeepSeek through
  kaibo). Reason: a slot in system memory is a silent fallback, and spec 8.5 asks for a crash first.
- The ledger does not stop it. The budget of a card in the B70 example is 32 GiB. With it, the ledger admits up to
  170,585 positions at 24 and 110,533 at 48. The driver and the kernels take about 0.24 GB more than the terms.
  - Thus a profile names a slot of at most 164,352 positions at 3 cycles.
  - A budget of the usable memory of the card, not of its 32 GiB, lets the ledger refuse such a profile first. The
    owner chooses the budget of the profile.
- **The limits of the check.** The check is a heuristic over the memory of the whole process on the card. It
  cannot tell the bytes of this open from the bytes of another engine of the process.
  - A free on the card during the open can make it refuse an open that fits. An allocation on the card during
    the open can hide a shortfall.
  - Another card adds nothing, because the sum reads only the fds of this card.
  - The engines of one card share one group thread, so the opens and unloads of a swap run one after the other.
    Thus no other engine of the card allocates or frees during an open.
  - It is still useful. It refused the slot of 262,144 positions, and its second case sees a buffer that the
    driver moved to system memory. The ledger sees neither, because it holds targets, not a measurement (4.1).
- **A refused open in a swap stops the process.** The swap does not expect an open to fail, so it panics
  (`crates/nagaya/src/process/staged.rs:1864`). The factory of Inkling 2 keeps the default `open_problem`,
  which answers none (`crates/nagaya/src/admission.rs:150`). Thus the load route cannot refuse such a load first
  (spec 3.3b).
  - Milestone 34 step 1 holds the repair: the load route refuses a load that the memory check of the engine
    refuses. It is built for the Qwen3.8 engine. An `open_problem` of the factory of Inkling 2 is still to do.

**Positions.** The model has no rotary embedding.

- Each layer adds a bias to each score: the layer input of the query mixes 16 learned profiles of the distance. The
  bias is 0 at a distance of 1,024 or more.
- The full layers also scale the query and the bias by `1 + 0.1 * ln(max((p + 1) / 8192, 1))`, where `p` is the
  position. Below position 8,191 the scale is 1.
- Each head adds a sink: one more score of each position, with a value of zero, from the layer input.

**Position 0 and no way back.** A pass from position 0 starts from conv states of zeros, as the reference does. A
pass starts at the token count of its slot, as for LFM3 ("The first LFM3 binding", "No way back").

**Logits.** A logit row has 200,058 values, the `unpadded_vocab_size` of the config. The engine applies the softcap
of 30, as the reference does: the logits are `30 * tanh(z / 30)`.

**Taps.** The tap `residual` gives the row after each run, not after each layer: 48 rows at 3 cycles. Reason: each
run of a cycle has its own row, and the lab compares the cycles. A control item acts after each run of each layer
that its vector holds. "The lab of Inkling 2" gives the other taps: the block outputs, the messages, the gates and
the routes.

**Inputs.** A segment holds tokens, or tokens with the rows of their image and latent spans (`tokens+rows`, "The
images of Inkling 2", "The lab of Inkling 2"). A segment of rows with no tokens is refused.

**Not covered yet.** The drafts of the MTP head, which wait (milestone 35 step 10).

### The images of Inkling 2

Images follow the pattern of LFM3 ("LFM3", "The LFM3 vision binding"). The family prepares the rows of an image.
The engine runs the vision tower inside the pass, and the rows of the projector replace the placeholders. The facts
come from transformers `inkling2` at `046bdb8cc2`. Each rule is **Proposed** (Claude Opus 5.5, 2026-11-21).

- **No resize.** The processor pads each image on the right and at the bottom to a multiple of 40 px. The pad value
  is -1 on the scale of 0 to 255 (`image_processing_inkling2.py:89`). Then each value `v` becomes
  `(v - 127.5) / 127.5` in `f32` (line 91). An image with an alpha channel goes over a checkerboard of 204 and 153
  first (line 67).
- **Tokens and patches.** An image token is a square of 40 px. It holds 2 by 2 patches of 20 px. An image of `H` by
  `W` px has `rows = ceil(H / 40)` and `columns = ceil(W / 40)`, and thus `rows * columns` image tokens.
- **The placeholder.** The template writes `<|content_image|><|unused_200054|><|end_message|>`. The family puts one
  placeholder `<|unused_200054|>` (id 200,054) for each image token in place of the one that the template writes
  (`processing_inkling2.py:37`).
- **The rows of a span.** The family gives one row of 1,200 `f32` values for each patch. A row is the 20 lines of
  the patch, each line is its 20 pixels, and each pixel is its 3 channels (`modeling_inkling2.py:1230`).
- **The order of the rows.** The rows go in the order of the image tokens, line by line. The 4 patches of one token
  follow each other, line by line. This is the order of the merge (`vision_utils.py:119`).
- **The meta of a span.** It is the record of "The LFM3 vision binding": the kind 1, the count of positions, the
  first position, then `grid_t = 1`, `grid_h = 2 * rows`, `grid_w = 2 * columns` and 0. The grid counts patches.
  Reason: one reader and one rule serve both engines, because the span has `grid_t * grid_h * grid_w / 4` positions.
- **Sizes.** The family accepts each side of 1 to 4,096 px. A side past 4,096 px is a 400 that names the part, and
  the council never resizes an image. Thus an image has at most 103 by 103 = 10,609 image tokens. @tobert decided
  the bound on 2026-11-21: "yep, match the model card. use the 4096 rule". The model card says: "For optimal
  performance, each image dimension should be between 40px to 4096px".
- A side below 40 px pads to 40 px, as the reference pads it. Thus a small image has one token. @tobert chose this
  on 2026-11-21. Reason: the 40 px of the model card is advice for quality, and the reference accepts a small image.

**What the engine runs.** The tower runs over each image of the pass alone: its class row and its `4 * T` patches,
for `T` image tokens. These are the steps:

1. The patch embedding: a matrix of 1,024 by 1,200, with no bias.
2. The position embedding. The table holds 32 by 32 rows, and the engine resamples it to the grid of patches with
   bilinear weights. The weights are those of `F.interpolate` with `align_corners=False` and a clamp at the border
   (`vision_utils.py:206`).
3. The class row in front of the patches: `class_embedding + cls_position_embedding`. Then the LayerNorm `ln_pre`.
4. 23 layers. Each layer is `x + ls_1 * attn(ln_1(x))`, then `x + ls_2 * mlp(ln_2(x))`. The attention has 16 heads
   of 64 values, with biases, in both directions. The MLP is `c_proj(gelu(c_fc(x)))`, with the GELU of `erf`.
5. A rotary embedding in two axes, on the query and the key, with a base of 10,000. A patch in line `r` and column
   `c` has the position `(r + 1, c + 1)`, and the class row has `(0, 0)`.
6. The channel pair `j` of a head turns by the column for `j < 16`, and by the line for `j >= 16`. Its frequency is
   `10000^(-(j mod 16) / 16)` (`modeling_inkling2.py:1035`).
7. The class row goes. The 4 rows of each token join into one row of 4,096 values. The projector is
   `proj_out(gelu(proj_in(x)))`, from 4,096 to 4,096 to 2,304, with no bias.

**The output.** The pass copies the rows of the projector over the normed embedding rows of the placeholders. The
engine does not norm them (`modeling_inkling2_loop.py:1694`).

**The weights.** The 422 tensors of `model.visual.` stay `bf16`, and every sum is `f32`, as in the text model. They
take the term `vision` on `device(0)`. The two models hold the same tensors.

**The host form.** `src/host/vision.rs` holds the tower in `f32`, and it uses the rules of the processor of the
family library. It is the oracle of the family and of the device ops.

**The choices of the family.** The rules above leave these choices open. Each one is **Proposed** (Claude Opus 5.5,
2026-11-21).

- **One copy of the rules.** The rules of the processor are in `nagaya-text-inkling2` (`processor`), and the host
  form of the engine uses them. Reason: the engine crate depends on the family crate for its identity, so the rules
  cannot live in the engine crate. The oracle of the host form still checks the one copy.
- **The decode.** The family decodes a PNG or a JPEG as the LFM3 family does (spec 11.11, "The media of a
  request"). A PNG with an alpha channel or a `tRNS` chunk keeps its alpha, and `rgba_to_rgb` puts it over the
  checkerboard. Reason: the processor does this for each image that Pillow says has transparency
  (`image_processing_inkling2.py:50`).
- **The parts of a message.** A message with an image part gives the template its parts as a list, in order. Thus
  each text part is a block of its own, as in the reference. A message of text parts only stays one block.
- **Roles.** An image part of a `system` or an `assistant` message is a 400 that names the part. The template
  refuses such a part too.
- **Decisions.** A plain read holds the images of its context and of its `case_media` (spec 11.18, "Reads"). The
  case turn holds each object of `case_media` in its order, then the text of the case. An error of a case image
  names `case_media[i]`. A queue read refuses each image with a 400 that names it.
- **Objects.** The process gives the family a reader of the objects of this machine, as for LFM3. A family with no
  reader refuses an image part with a 400 that names the part.

**The choices of the tower.** Each one is **Proposed** (Claude Opus 5.5, 2026-11-21).

- **The ops.** The binary adds `OP_INKLING2_VISION_EMBED` (180), `OP_INKLING2_VISION_ROPE` (181) and
  `OP_INKLING2_VISION_ATTN` (182) in `kernels/inkling2_vision_ops.cl`. The tower also uses `OP_MATMUL`, `OP_BIAS`,
  `OP_LAYERNORM`, `OP_GELU`, `OP_SCALE_ADD` and their `bf16` weights. The open uploads the norms and the layer
  scales as `f32`, and each value is exact.
- **The rotary table.** At its open, the engine computes the cosine and the sine of each angle on the host, with the
  rules of the host form. The table holds 207 positions by 16 frequencies. Reason: the op then gives the bits of
  the host form. 207 is the class row and the 206 patch lines of a side of 4,096 px.
- **The position embedding.** For each patch line and each patch column, the host computes two taps and their two
  bilinear weights. It uses the rules of the host form (`vision::axes`) at the plan of a span. The op reads them, so
  it gives the bits of the host form. Reason: the division of the card is not correctly rounded. With weights of its
  own, the op moved 38,990 of 73,728 values of a grid of 3 by 6 tokens (2026-11-21).
  - The taps of the largest image take 6,592 bytes. The class row is an `OP_COPY` before the op.
- **The attention** is `OP_VISION_ATTN` of LFM3 under a new opcode, with 16 heads of 64 values and the scale 1/8.
- **One image at a time.** The tower runs over each image of the pass alone, in the order of the spans, on one set
  of buffers. Thus a call with two images gives each image the bits of a call with that image alone.
- **The buffers.** They hold the largest image of the sizes rule: 42,436 patches and the class row. The tower
  uses its six buffers again between its steps, so they take 1,564,397,568 bytes of `work`. The patch rows and the
  projector rows of that image take 301,465,344 bytes more. Reason: the family refuses no image of the range of the
  model card, so the engine must not refuse one either.
- **The weights** of the tower and the rotary table take 636,825,472 bytes (`vision::WEIGHTS_BYTES`).
- **The merge.** The class row and the patch rows are one buffer. The projector reads the rows after the class row
  as `T` rows of 4,096 values, so the merge needs no op.

**The choices of the pass.** These rules tie the tower to a pass of the engine. Each one is **Proposed** (Claude
Opus 5.5, 2026-11-21). Reason: each is the rule of "The LFM3 vision binding" or of its code, and the docs leave the
choice open.

- **Inputs.** The engine declares `tokens+rows`. Both models hold the tower, so both declare it. A segment of tokens
  alone, with no `meta`, is a text block, as for LFM3.
- **The records.** The engine reads the record of "The meta of a span" with the kind 1 only. `grid_t` is 1, and the
  fourth word is 0. `grid_h` and `grid_w` are even, from 2 to 206 patches, and the count of positions is
  `grid_h * grid_w / 4`. Another record is refused, and the message names the record.
- **The rows of a segment.** They are the rows of each span whole, in the order of the records. A span of `T`
  positions has `4 * T` rows of 1,200 `f32` values.
- **Placeholders.** Each position of a span in the block holds the placeholder 200,054. Each placeholder in the
  block is in a span. Thus the count of placeholders of an image equals its count of projector rows. A pass that
  breaks either rule is refused, so a placeholder never enters the model as its embedding row.
- **A span longer than a block.** Each pass that holds positions of a span gets the rows of the whole span. The
  engine keeps the projector rows of the last image that the tower ran, with its grid and the BLAKE3 of its rows. A
  later block of the same image copies its rows from them, and the tower does not run again. Reason: an image of
  10,609 tokens meets 21 feed blocks of 512. The tower of one image gives the same bits in each run, so the bits do
  not change. The LFM3 engine does the same (`HeldRows`).
- **The place.** The tower of each span runs in launches of its own, before the launches of the pass. A copy then
  puts the projector rows of the block's positions into the media rows of the block. In the first launch of the pass,
  after the norm of the embedding, a copy puts the media rows over the rows of the placeholders.
- **The memory terms.** The term `vision` holds the weights of the tower, 636,825,472 bytes. The term `work` adds
  the buffers of the tower, the patch rows, the taps and the projector rows of the largest image. It also adds the
  media rows of a block, 2,304 `f32` values for each position. For a block of 512 that is 1,870,588,096 bytes more.
- **The time of a large image.** The card gate `tests/xe_long.rs` measured an image of 4,096 by 4,096 px, 10,609
  tokens, on 2026-11-21. Its tower takes about 56,780 ms in the pass of its first block, and a later block takes
  843 ms. The bound `max_quantum_ms` is 5,000 ms, so that pass is a long pass (D8, "A long pass with a declared
  bound"). The ceiling `max_long_quantum_ms` is 180,000 ms.
- **The bound of a pass with a tower** (accepted 2026-11-22). `pass_bound_ms` finds the towers
  that the pass runs, from the held rows, as the pass does. The tower of an image of `n` tokens takes about
  56,780 ms times the square of `n / 10,609`. Reason: its attention reads each pair of patches. The bound is 1.5
  times the sum of the towers plus 2,020 ms, the slowest text block, up to the ceiling (`src/bound.rs`).
  - The largest image gives 88,200 ms. A pass with no tower, or whose bound is 5,000 ms or less, declares none. An
    image of 1,614 tokens or more declares a bound.
  - A square law can be low for a small image. That does no harm: the deadline is never less than the timeout of
    30 s.
  - The bound changes no state, so it hashes the rows of each image, and the pass hashes them again. One hash of
    the largest image, 203,692,800 bytes, took 20.4 ms on the B70 machine (BLAKE3, a release build, 2026-11-22).
    That is 2.4% of the 843 ms of a later block, and about 430 ms over the 21 blocks of the image. A cache keyed by
    the address of the rows was refused: a stale entry could skip a tower and give wrong bits.
  - The square law rests on one measurement, the largest image. A measurement of smaller images can change the
    rule.
  - The card gate `crates/nagaya/tests/xe_inkling2_long_pass.rs` runs a cold image of 4,096 px through the process,
    with the watchdog at its defaults. It passed on card 0 on 2026-11-22, on `7f6880a`. The pass of 10,631 tokens
    declared 88,200 ms, and the route showed 57,309 ms elapsed with `overdue` false. The generation took 75,073 ms.
- **The tower in quanta of its own** (from ChatGPT through kaibo `gpt-plan`, 2026-11-21; accepted 2026-11-22 under
  D8). The family keeps the bound of 4,096 px. The tower runs in quanta before the pass that reads its rows. Each
  quantum fits `max_quantum_ms`.
  - The engine chooses the slices, and it keeps the state of a tower that is not done. A slice can be a few layers,
    or the queries of one layer in parts. A slice must not cut the image, because the attention of the tower reads
    the whole image.
  - The scheduler runs each slice at the class of the job. A pass reads the projector rows only after the tower is
    done. The state of a tower that is not done goes when its job ends. The result of `pass` does not change.
  - Reason: a call that runs the whole tower still holds the card for 57 s. Slices let other ready quanta run
    between them.
  - Not built. The card gate of the change runs the largest image with no held rows. A test of the process checks
    the order of the quanta and the end of a job.

### The first Inkling 2 binding

The text path of `crates/nagaya-mk-inkling2-bmg` follows these rules. It uses the interpreter of `nagaya-mk-bmg`
with the ops of `kernels/inkling2_ops.cl`, in a binary of its own (`MK_INKLING2`). Each rule is **Proposed** (Claude
Opus 5.5, 2026-11-21).

- **The ring of one run.** A run of layer `l` reads `x`, the residual, and `B`, the block outputs of this cycle.
  It writes `x` again. Each name below is one buffer of the block. A run is these tasks, in this order:
  1. `a = mix(B, x)`, the attention residual of each position over its block outputs and `x`. A full layer has 3
     queries, so `a_q`, `a_k` and `a_v` are three mixes. A sliding layer has one, and `a_q = a_k = a_v = a`.
  2. In cycles 2 and later: `m = mix(P, y)`, the 4 messages of the layer. `P` is the block outputs of the cycle
     before, and `y` is its last residual. Then `a_q`, `a_k` and `a_v` each go through their gate with their message.
  3. `h_q`, `h_k`, `h_v`: the norms of `a_q`, `a_k` and `a_v`. A sliding layer has one norm. Its gates act before
     the norm, so in cycles 2 and later its three rows differ.
  4. From `h_q`: the query, the bias states, the sinks and the output gate. From `h_k`: the key and the taps of a
     dynamic key convolution. From `h_v`: the value and the taps of a dynamic value convolution.
  5. The clip of the query, and its log scale on a full layer. The convolutions of the key and the value, the clip
     of the key and the norm of the value. The key and the value go into the rows of the run's state.
  6. The attention, with the bias of the distance and the sink. Then the output gate, the output matmul, and the
     convolution after the attention: `o`.
  7. On the first layer of a block, `x` goes into `B` as a new block output, and `r = o`. On another layer,
     `r = x + o`.
  8. `e = mix(B, r)`, and in cycles 2 and later its gate with the fourth message. Then the norm, the router, the
     plan, the experts, the sum of the 8 picks and the convolution after the experts: `f`. Then `x = r + f`.
- **The new ops.** The binary adds these ops, which the other engines do not have:
  - the attention residual, and the message gate;
  - the clips and the log scale;
  - the convolution with a residual, static and dynamic;
  - the attention with a bias and a sink, and the gate of its output;
  - the router of sigmoid scores, and the experts in NVFP4 and in MXFP8.

  `OP_MOE_PLAN` and `OP_MOE_GATHER` serve as they are: 128 experts and 8 picks are their constants.
- **The experts on the card.** A lane of an expert op expands 16 values of its row, one block of NVFP4, for each
  step of 16 columns. It rounds them to `bf16` as the host op does, and the XMX sums take `x` in three `bf16` parts,
  as `OP_MATMUL` does. The weight of a pick multiplies the output of the down matmul, as in the reference.
  - The load repacks each expert across its output rows: a plane of codes `[expert][16 rows][16 columns][2
    words][16 lanes]` and a plane of scales. Thus the 16 lanes of a subgroup read 64 bytes in one load. The gate
    of the op reads the repacked planes, so the repack is part of what the gate checks.
  - The decode of E4M3 is exact, and it keeps the sign of zero. The order is `(e2m1 * scale) * factor`, with no
    fused multiply-add and no other order.
  - The sum of the 8 picks has a fixed order, with no atomics. The reference sums in the order of the experts, and
    the card test names the gap of the order that the op uses.
  - **The row path** (milestone 36 item 1). A block of 1 to 8 tokens takes a path with no XMX. Each expert then has
    at most 8 entries, and a lane expands each weight of its row once for them. A block of 9 or more tokens takes
    the XMX path.
  - The row path splits the steps of a row into fixed parts: 16 for the gate and up rows, 8 for the down rows.
    The subgroups of one work-group take the parts of one tile. The first part adds the parts in order, so the
    count of work-groups changes no bit.
  - A part sums in `f32` with a fused multiply-add, in 4 sums of columns `i mod 4`. The weights keep the bits of the
    XMX path. A step of NVFP4 writes the 8 magnitudes of its codes to local memory, and a code reads its value
    back. The values of MXFP8 come from the bits of the code and the exponent of the scale.
  - Each path gives a token the same bits in any block of its counts. A token alone and in a block of 9 or more
    differ by the order of the sums, as the matmuls do (D11). The card test `xe_experts` measures both paths
    against the host form.
- **The attention.** The op computes the bias of each score inline. It is the dot product of the 16 bias states of
  the query and the column of the distance. A distance of 1,024 or more has no bias.
  - The distance and the log scale read absolute positions, never a row of a ring.
  - The sink enters once for each query and head. The merge of the partial sums starts from `(sink, 1, 0)`. Thus
    the sink adds to the sum, and not to the output.
  - The chunks of keys sit on a grid of absolute key positions. Each kind of layer has one chunk size: 256 keys on
    a sliding layer, and 2,048 on a full layer. Thus the terms of a position come in the same order in each block.
  - A decode step is a block of one position. It has no op of its own, because another op would sum in another
    order. The card test counts the values of a decode step that differ from the block.
  - Each rule of the two items above is **Proposed** (Claude Opus 5.5, 2026-11-21).
- **The gate of each op.** Each device op has a host form in `src/host/`, and the card test of the op compares
  them. The expansion gives equal bits. An op that sums gives the bits of its host form in the same order, or a
  measured gap that its test names.
- **Bits and splits.** A cut between launches changes no bit. The same block split gives the same bits. Another
  split of the same tokens gives the logits within a tolerance that a card test measures. Reason: a small change
  near the eighth pick of the router can change the set of experts.
- **A pass in several launches.** A ring holds at most 2,047 tasks (`MAX_TASKS_PER_CARD`). A pass whose tasks do not
  fit runs as several launches, cut between two runs. The state of the block stays in the buffers between them, so
  the cuts change no bit.
- **The buffers of a block.** All are `f32`: the residual, the block outputs, the block outputs of the cycle
  before, and the input of the cycles. The block outputs hold at most 4 rows of each position. The engine copies the
  residual and the block outputs after layer 5, and each cycle starts from the copy.

**The host spec.** `src/spec.rs` holds the rules of the engine that need no card. These rules are **Proposed**
(Claude Opus 5.5, 2026-11-21).

- **The config.** The kind is `mk-inkling2-bmg`, and its config has four keys:
  - `model_dir` names the model folder. It has no default, because one kind serves two models.
  - `slots` is a count of slots of 32,768 positions, or a list of the positions of each slot, from short to long.
    A slot holds 1 to 1,048,576 positions, the `model_max_length` of the config.
  - `max_block` is the feed block: 512 when the key is absent, and 1 to 512.
  - `extra_runs` is the reserve of each slot: 24 when the key is absent, and 24, 36 or 48. Reason: a reserve of
    less than 24 cannot run the default of 3 cycles.
- **The model.** The open reads `config.json` and checks each field that the ops read, with its value. Another
  value is refused, and the message names the field and the value. `loop_config.is_varloop` is false for Mini and
  true for the research model. The two configs differ in no other field that the engine reads.
- **The runs.** A pass runs the count of its segment, or 3 with none (`spec::cycles_limit`).
  State `l` of a slot holds the run of layer `l` outside the cycles, or in cycle 1. Cycle `c` of 2 or more uses
  entries `12 * (c - 2)` to `12 * (c - 2) + 11` of the reserve.
- **Slot sizes.** Each sliding run holds a ring of 2,048 rows and a spill of 1,024, in a slot of any size. In a
  slot of 2,048 positions or fewer, the ring does not wrap, so the run still holds every position.
  - **Proposed** (Claude Opus 5.5, 2026-11-21): the ring in a short slot. Reason: `OP_INKLING2_ATTN` applies the
    window only to a ring, and it finds the value plane at 3,072 rows after the key plane. A short slot of cells
    would attend to keys past the window.
  - A slot of 2,048 positions takes 1,144,684,544 bytes with a reserve of 24.
  - A slot of 32,768 positions takes 3,157,950,464 bytes with a reserve of 24, and 4,736,925,696 with 48.
  - A slot of 262,144 positions takes 18,190,336,000 bytes with a reserve of 24.
- **A restore of a ring part** must hold the last 1,024 positions, the window of a sliding layer.
- **The probe of the oracle** (8.3) runs from position 0 of a slot that it empties. Its check comes first, so a probe
  that the check refuses changes nothing (`Held::check_probe`).
- **`last_error`** gives the failure of the last call that returns a status (D5): `pass`, the part calls,
  `set_n`, `slot_clear` and `control_write`. Each of them clears it first, so a call that does not fail leaves
  `None`. A call that returns its own message changes nothing in it: `fits`, which reads only, and the probe
  `residual`.
- **The slot of a probe** is checked before the probe segment names it as `u32`. Thus slot 2^32 is refused, and
  it is not slot 0.
- **Tokens.** A token id of 200,058 or more is refused. Reason: the rows past the unpadded vocabulary are padding
  of the head and the embedding.
- **The term `weights`.** The plan counts the bytes of the header, and it reads no weight bytes (spec 2.4).
  - Each expert matrix is one slab of `repack::expert_slab`: the codes, the scales, and one `f32` factor of each
    expert for NVFP4. The tensors `amax` and `outer_scale_denominator` go into the factors.
  - The embedding holds its first 200,058 rows. The head holds 200,064 rows: the first 200,058 rows of
    `unembed`, then 6 rows of zeros.
  - The sink projection holds 32 rows: its 24 rows, then 8 rows of zeros.
  - Each other tensor of `model.llm.` keeps its bytes. Some tensors share one allocation ("The weights on the
    card").
  - Each allocation rounds up to 64 bytes. Mini gives 18,881,368,064 bytes.
  - The plan checks the name, the dtype and the shape of each tensor that the load reads (`spec::text_tensors`).
    A missing tensor, or one of another dtype or shape, is an error that names it. Thus the load never meets a
    missing tensor. Another tensor of `model.llm.` is refused too.
  - The plan also checks the 422 tensors of `model.visual.` (`vision::tensors`), and `vision::load` runs the same
    check (`vision::check_tensors`). They are the term `vision`, so they add no bytes to `weights`. Another tensor
    of `model.visual.` is refused.
- **The term `work`.** It is the buffers of a block (`spec::block_buffers`), the logits and the ring of the card.
  The list of buffers is the list of `model::Bufs`, and the ring builder names its use. The scratch is the scratch of
  `OP_INKLING2_ATTN` for a block of 512 (`tasks::inkling2_attn_scratch_floats`).
- **The other terms.** `kv` is the sum of the slots, on `device(0)`. `pinned` is host memory: the tasks, the
  statistics and a staging buffer of 64 MiB, as for LFM3.
- **The limits.** One segment in each pass, or a batch of 2 to 8 ("A batch of decode rows"), and 16 logit rows.
  `max_quantum_ms` is 5,000 (D8, 2026-11-22). The slowest text pass of the long gate took 1,768 ms. A block at
  164,352 positions takes about 2,020 ms, from a line through the measured blocks. The tower of a large image runs
  in a long pass ("The time of a large image"). The capabilities are `parts`, `slots`, `coverage`, `close` and
  `cycles`.
- **Not declared.** `layer_loops`. A pass with a layer loop is refused. "The lab of Inkling 2" adds the taps,
  `control` and latent rows.

**The ring builder.** `src/model.rs` builds the tasks of a pass, and `src/weights.rs` puts the weights on the card.
Each rule is **Proposed** (Claude Opus 5.5, 2026-11-21).

- **The weights on the card.** Some tensors go into one allocation, so that one op reads them:
  - On a full layer, `wk_dv` and `k_sconv.weight_proj` are one matrix of 1,056 rows. Thus one matmul writes the
    input row of the dynamic convolution: 1,024 values, then the 32 taps. The value is the same.
  - The three matrices of a message gate are one buffer: `down_weight`, `forget_weight`, then `gain_weight`.
  - The message queries are one `f32` buffer `[12][4][2304]`, in the order query, key, value, expert input.
- **The head and the sinks.** `OP_MATMUL` reads whole tiles of 16 rows, except a matmul of one row with at most
  65,536 rows. Thus the head holds 200,064 rows, and the sink projection holds 32.
  - A logit row is its own matmul of one row, with `FLAG_SOFTCAP_30`. Thus its bits do not depend on the other logit
    rows of the pass. The engine reads the first 200,058 values of each row.
  - `OP_EMBED` with an `f32` table copies the 24 sinks of each position out of its row of 32. A table of ids on
    the card names rows of 8 values. The copy is exact.
- **The input of layer 0.** Layer 0 has no attention residual: its query, key and value read the embedding row.
- **The norms of a sliding layer.** Outside the cycles, a sliding layer takes one norm. In a cycle after the first,
  the three gates give three rows, and each row takes the same norm.
- **The buffers of a cycle.** The block outputs `B` keep their first two rows in each cycle: the embedding and the
  row after layer 5. Each cycle after the first starts with three copies: `B` row 2 into `P`, `x` into `y`, and
  the row after layer 5 into `x`.
- **Launches.** The ring of a pass is a list of runs. A launch holds whole runs, at most 2,046 tasks, and the tasks
  before the first run and after the last. A pass of 3 cycles fits one launch: 1,635 tasks, or 1,715 when its
  block crosses the end of a ring.
- **The bits of a decode step.** Each matmul of a pass carries `FLAG_TILED`, so one token sums in the order of 2 to
  8 tokens (milestone 36 item 10). Thus a block of 1 to 8 tokens gives each token the bits of the same tokens decoded
  one at a time. `OP_MATMUL` sums a block of 9 or more in another order. Thus a decode step does not give the bits of
  its position in such a block. The card test `xe_feed` counts the values that differ.
  - The card run of 2026-11-21 had no flag. Then 181,182 of the 200,058 logits of a decode step differed from the
    block of 512, by 9.1e-6 at most. The splits `[300, 337]` and `[100, 412, 125]` gave the bits of `[512, 125]`.
  - The card run of 2026-11-22, with the flag: 185,519 logits differ, by 1.24e-5 at most. The splits give the same
    bits as before. The card test `xe_batch` gave equal bits for blocks of 2, 5 and 8 tokens and their decode steps.
  - **Proposed** (Claude Opus 5.5, 2026-11-22): the flag on each matmul. Reason: a batch of decode rows needs the
    bits of each row alone.
- **A batch of decode rows** (milestone 36 item 11). A pass of 2 to 8 segments runs one decode token of each, each on
  its own slot at its own position (`model::batch`). Reason: swarms and parallel contexts share the fixed costs of a
  step. Each rule of this item is **Proposed** (Claude Opus 5.5, 2026-11-22).
  - A segment of a batch is one token with no media, and it asks for the logits of its token. The segments are on
    slots of their own, and they have one control list and one count of cycles. A batch names no tap. Any other
    batch is refused before a slot changes (`Held::check_batch`).
  - The ops of the weights run once over the rows. The matmuls take the tiled path, and each other op takes the
    rows one at a time inside its task. The expert ops take the row path for 1 to 8 tokens.
  - The ops of a slot run for each row on its own slot. They are the four convolutions, the clips of the key and the
    value, the attention, and the log clips of a full layer.
  - Thus a row gets the bits of its pass alone (spec 5.7). The card test `xe_batch` runs 8 contexts at other
    positions in batches of 8, 4, 3, 2 and 1, at 3 cycles and at 1. Each row gave the bits of its pass alone.
  - `max_segments` is 8, or `max_block` when it is less. Reason: a ninth row takes another path of `OP_MATMUL` and
    of the expert ops.
  - `segments_exact` stays false, because the engine has a feed rule (D11 rule 4). The process joins decode rows by
    D11 rule 6, and it joins only rows of one count of cycles (`jobs::joins`).
  - The terms do not change: the buffers of a block hold 512 rows, and a batch has at most 8.
- **A route near a tie.** The eighth and the ninth key of the router can be near. There, a small gap of the card from
  the host form can choose another expert, and the rows after that position differ more.
  - The prose of the prompt "longer", repeated to 2,200 positions, flips a route at position 1,295. There the two keys
    of layer 11 differ by 4.4e-6 in the host form. The logits after it differ by up to 1.3.
  - Thus the gate of a ring that wraps feeds ids from a seeded generator, which flip no route in 2,200 positions.
  - Seeded ids also flip a route in a longer context. At position 1,527 of the long gate (`tests/xe_long.rs`), the
    route of layer 17 in cycle 2 is at a near tie. In the host form its eighth key is 0.253102571 (expert 0), and
    its ninth key is 0.253102511 (expert 96): a gap of 6.0e-8. The card picks expert 96. The logits differ by 0.56
    at that position and by 0.055 at the next. Later they differ by about 2e-3 to 9e-3, and the argmax agrees.
  - The host form with sums of `f32` in its attention picked expert 96 there too. Its first flip was at 4,846, in
    layer 7 of cycle 2. There the keys were 0.172177047 (expert 74) and 0.172176898 (expert 17), a gap of 1.5e-7.
    Thus a gap below about 2e-7 can go either way.
  - Thus a long gate compares the host form from position 0 only before the first flip. After it, the host form
    runs each checked block from the state that the card read.
  - The host form sums its attention in `f64`. Sums of `f32` in key order drift with the count of keys. From one
    state at 130,048 positions, they moved the logits by up to 1.6e-2.
- **The identity.** The engine names `inkling2-mini` or `inkling2-varloop`. The tokenizer hash and the template
  come from `nagaya-text-inkling2` (`spec::text_facts`). Thus the hash is `sha256:` of the whole `tokenizer.json`,
  and the check of the registry at the start compares like with like (spec 11.3).
- **The numbers of the config.** The identity names `max_block` and `extra_runs` (`spec::identity_numbers`).
  `max_block` sets the bits, and `extra_runs` sets the parts of a state. Thus two reserves never share a recipe.

### The lab of Inkling 2

These rules add the taps, the control vectors and the latent rows of milestone 35 step 8. Each rule is **Proposed**
(Claude Opus 5.5, 2026-11-21). Reason: each follows the rule of LFM3 ("Control vectors", "Latent rows of LFM3"), and
the docs leave the choice open.

```
limits().taps: residual, fold, blocks, messages, gates, routes
limits().control: {tables: 8, layers: 24, width: 2304}
segment.control: as for LFM3; an item acts after each run of each layer that its vector holds
```

**The taps.** A tap gives its rows for each logit row of the pass, in the order of the logit rows. Let `R` be the
runs of the pass, and `c` its count of cycles. The rows are `f32`, except the picks.

- `residual`: `[R][2304]`, the row after each run, in the order of the runs, after the control items of the run.
  The tap declares the condition `each run` (`TAP_EACH_RUN`), so the council reads the row of each layer after its
  last run (spec 11.17).
- `fold`: `[2304]`, the input row of the head: the output residual over the block outputs and the last row, then the
  final norm. A latent step reads it.
- `blocks`: `[c + 3][2304]`, the block outputs in the order that the model writes them. The first two are the
  normed embedding and the row after layer 5. Then come the row after layer 11 of each cycle, and the row after
  layer 17 of the last cycle.
- `messages`: `[12 (c - 1)][4][2304]`, the four messages of each layer of each cycle after the first. Their order is
  query, key, value and expert input. A pass of 1 cycle gives no bytes.
- `gates`: `[12 (c - 1)][4][2304]`, the output of each message gate, in the order of `messages`.
- `routes`: two outputs. `routes` is `u32 [R][8]`, the picks of each run in rising expert order. `routes.weights` is
  `f32 [R][8]`, the weight of each pick.

The oracle file of "plain" holds `message<ii>` as `[T][4][2304]` and `gate<ii>` in the same order. Thus a tap and
the oracle have one layout.

- **Bits.** A tap only copies rows (`OP_COPY`), so a pass with a tap gives the logits of the same pass with none. A
  pass with no tap has the tasks that it had before.
- **The buffers.** The term `work` holds the buffers of every tap, for 16 logit rows at the most cycles of the
  config. They take 36,372,480 bytes with `extra_runs` 24, and 68,567,040 bytes with 48. Reason: `pass` never
  allocates. `fold` needs no buffer: the engine reads the final norm of the logit row.
- **The cost.** A tap adds copies to each run for each logit row. `residual` and `routes` add 1 each. `messages`
  and `gates` add 4 each in a cycle after the first. The launches cut between runs, so many copies add launches.
- **The rows of `blocks`.** Each row of `blocks` but the first is also a row of `residual`. The tap gives them in the
  shape that the attention residuals read.
- **Taps and logit rows.** A tap needs a logit row. A pass with a tap and no logit row gives outputs of no bytes.

**Control vectors.** They follow "Control vectors". These rules are of this engine:

- **The table** holds 8 vectors of 24 layers of 2,304 values: 221,184 bytes each, 1,769,472 bytes in all. The
  mandatory term `control` holds it.
- **The place.** An item acts after each run of each layer that its vector holds. At 3 cycles, an item of layer 8
  acts three times. It acts before the copy of the row after layer 5 and before the taps. Thus each cycle starts
  from the changed row, and the taps read it.
- **The op** is `OP_INKLING2_CONTROL` (190) in `kernels/inkling2_lab_ops.cl`. It is `OP_CONTROL` of LFM3 under the
  opcode of this binary, because the binary does not hold `lfm3_ops.cl`. Its host form is `host::ops::control`.
- **The bits.** An `add` or an `ablate` of strength 0 gets no task, so it gives the bits of no item. The card gate
  names the gap of each op from its host form.
- **The cost of the op.** The op changes the kernel binary, so the identity digest changes once. Then each held
  context of the engine fills again once.

**Latent rows.** They follow "Latent rows of LFM3". These rules are of this engine:

- **The record.** The kind is 5, and the first of the four words is 2,304, the width of a row. The other three words
  are 0, and the count of positions is 1 or more.
- **The placeholder.** A latent position holds `pad_token_id` of `generation_config.json`: `<|endoftext|>`, id
  199,999 in both models. The open reads it, and an id of 200,058 or more is refused.
- **The rows.** A latent span takes one row of 2,304 `f32` values for each position. The rows of a segment are the
  rows of each span whole, in the order of the records. One segment can hold image spans and latent spans.
- **The copy.** The engine writes the rows of the block's positions into the media rows. The copy after the norm of
  the embedding then puts them over the rows of their placeholders, with no norm. Thus a latent row takes the place
  of a normed embedding row, as a projector row does.
- **A placeholder outside a span.** `<|endoftext|>` is a token of the vocabulary. Thus a placeholder outside a latent
  span enters the model as its embedding row. Only a position in a latent span must hold it.
- **The map of a step.** A latent step reads the tap `fold`, the input of the head. The host scales the row to the
  mean norm of the normed embedding rows. Reason: the row takes the place of a normed embedding row.
- **The family.** The council sends latent rows only when the family gives a latent record (spec 11.11). The family
  of Inkling 2 gives none yet, so the council refuses `latent` and `soft` for Inkling 2.

## Not covered yet

- Streams. They need a period, a deadline and a late policy, in `limits()` or per job. A pass blocks, so the
  scheduler meets a stream's deadline, not the engine.
- Unloading an engine while its arena stays staged in host memory for a fast load again.
