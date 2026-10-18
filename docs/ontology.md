# Ontology

This doc has one term for each thing. Each entry gives the definition, then where the thing already exists.

- mk is `~/src/megakernel-qwen38-flashnext-strixhalo`.
- mkviolet is `~/src/mkviolet`.

The nouns are in four groups:

- places: where things are;
- things: what is held;
- time: what runs;
- holding: who keeps what, and until when.

We revised this doc on 2026-10-17, after a kaibo review (deepseek cast). The review checked the first draft against
copies of the source of both engines. Its main corrections:

- cell storage is different in each engine;
- sliding layers are rings;
- seal and park are different steps;
- slot geometry changes what a snapshot holds. We later resolved this as coverage (`docs/engine-abi.md` D3).

## Places

**Machine.** One computer. It runs exactly one nagaya process, and that process owns all of the machine's GPUs.
The machines today:

- gfx1151host: Strix Halo, one gfx1151 GPU, 128 GB unified memory;
- b70host: two Intel Arc Pro B70, 64 GB GDDR6 total, Ryzen 9 9950X, 128 GB DDR5, 4 TB NVMe;
- sm121host: DGX Spark, one Blackwell GPU, 128 GB unified memory;
- rented machines (`~/src/rented-gpu`).

The B70 cards are the fastest memory we have. The two 128 GB machines hold bigger models, for tasks where capability
is more important than speed. The machines are wired. The links are gigabit today, and 10 GbE needs a little work.
b70host can go faster (it has a BlueField-2), and gfx1151host can go to 5 GbE.

**Device.** One thing that runs quanta. A device is a GPU, or the CPU when an engine runs work there. (An example is
mkviolet's AVX-512 lens, mkviolet `docs/03-distributed-state-architecture.md` 2.5.)

mkviolet keeps one Level Zero context per card (`gemma4.rs:7`). Its `pipeline_stage` starts a thread per card for
each stage. In the nagaya, each device has one long-lived thread, and every call for that device comes from it. That
rule is new, and mkviolet's per-stage threads move into it.

**Tier.** A place where a state object can be: device memory, host memory, NVMe, or another machine. Tiers are a
graph with measured bandwidths per machine, not a fixed list. On gfx1151host and the DGX Spark, device memory and host
memory are the same LPDDR5X. On b70host they are separate: 608 GB/s GDDR6 per card, about 90 GB/s DDR5, and
about 14.8 GB/s NVMe (mkviolet `docs/03`).

## Things

**Engine.** A model's code and its weights, loaded on one or more devices of one machine. It exports the engine ABI
(`docs/engine-abi.md`). It has an identity with these items:

- the model;
- the checkpoint digest of its weights (see "two weight ids" below);
- its tokenizer and template;
- the binary that calculates its numbers (council `identity`; mk's `libmkengine:<sha>+numbers:<sha>`);
- every setting that changes bits (mk: `fused`, and `MK_FAST_QMM`'s f16 WMMA path versus exact f32,
  `engine.h:19-32`).

An engine has two weight ids (`docs/spec.md` 12.2):

- the **checkpoint digest** is logical. It is a hash of the canonical config and each tensor's payload, so a new
  packing does not change it. The identity carries it.
- the **arena id** is the content id of the packed bytes that the engine runs. The arena records it with the
  strength of its check.

Two runs with different identities are not expected to match.

**Plan table.** An engine's record of the free evaluation choices that it made: which GEMM algorithm for each
shape, fold orders, and split counts. Each choice is valid, but different choices give different bits. Thus the
table is part of the identity. It is append-only: the engine times a new shape once and appends its pick, and an
entry never changes. A recipe's bits depend only on the entries for the shapes that it used. Thus snapshots stay
valid as the table grows.

A new library, driver or device starts a new **lineage**, which is a new identity. ds4-recipe times plans per
process. It left a fixed table as "Optional idea, not needed", because "ragged-tail shapes ... a fixed table cannot
cover" (ds4-recipe `07-prefill-pipeline.md`). An append-only table covers new shapes. mk has no library GEMMs and
needs no plan table.

**Build.** The binary only. The history of how a context was made is its **chain**.

**Arena.** An engine's weights as memory that can outlive the process: a set of named file descriptors.

- mk has one sealed arena per engine. Another engine attaches to it read-only after a full check against its own
  build (`capi.hip:12-27`, `mk_arena_check`).
- mkviolet has one DMA-BUF slab per layer per card, 60 in all. systemd's file-descriptor store keeps them through
  `systemctl restart` (`gemma4.rs:143-146`, `slab_fd_name`; `hot_handoff.rs` `store_fd`). Its check,
  `verify_stored_slab`, compares only the first and last 64 bytes of each tensor (`gemma4.rs:284-287`).

An arena records how strong its check is. The arena is how the weights state object is held in memory. An inference
engine opens it read-only, and a training engine writes it. mkviolet's `memfd` host arena (`HotDdr5Arena`,
`hot_handoff.rs:130-201`) holds host-tier state and a header, not weights. Thus it is not an arena in this sense.

**State object.** Everything that a pass reads besides its inputs, made of parts. There are three kinds:

- **weights**: fixed parts only, with no inputs (decided 2026-10-17, Amy: "state object"). Inference only reads
  them, and training writes them.
- a **context**: what passes run on and extend.
- a **snapshot**: a context sealed at a boundary that nothing passes into. Reads run on a copy (mk `fork`), or they
  load it into a slot (mkviolet `load_kv`). A snapshot is stored whole or as pieces of a chain.

**Part.** One named piece of a state object. There are four kinds:

| Kind | What it is | Examples |
|---|---|---|
| fixed | a constant-size block that each pass replaces whole | mk GDN recurrent state `[36][48][128][128]` f32, GDN conv, PLE conv, PLE history (`engine.h:63-75`); a diffusion latent |
| cells | rows indexed by position, only appended | mk KV and index keys; mkviolet KV on global layers |
| ring | rows in a window that wraps: new rows overwrite the oldest | mkviolet sliding layers (`SLIDING_WINDOW` 1024, `KV_RING_ROWS` 4096, `kv_ring`, `gemma4.rs:30-46`) |
| derived | rebuilt from other parts, never stored | mk pooled index keys (`idx_pooled`) |

How a part is stored depends on the engine as well as the kind:

- mk reads a cells part whole, cells 0 to n (`engine.h:173-174`), with no range read. Thus an mk snapshot is stored
  whole. At n = 3584, that is about 316 MB. The parts are KV (176 MB), index keys (22 MB), GDN state (113 MB),
  GDN conv (4.4 MB) and PLE conv (0.4 MB).
- mkviolet stores a chain. A global layer's snapshot holds the rows after its parent. A ring layer's snapshot holds
  a suffix that can overlap its parent's rows. `KvSnapshot::join` merges the chain for a given slot size. It fails
  if the chain does not have the rows that the slot needs (`gemma4.rs:941-979`).

A range read (`docs/engine-abi.md`, `cells_read`) would let mk store chains too.

**Boundary.** A position in a context's chain where the context may be sealed. Council's `snap: true` turns and
spec layers are boundaries. Boundaries are part of a snapshot's recipe.

**Seal.** The engine's rule at a boundary. It may change bits: mk rounds the GDN recurrent state to bf16 in place
(gdn-bf16-v1, `round_gdn_bf16`, `engine.h:189-193`, mk `docs/park.md`). mkviolet's seal changes nothing.

**Park.** A move of a state object to another tier, bit for bit. A park never seals. Seals occur only at boundaries
that the chain declares. Thus when the scheduler parks a context to free its slot, later results do not change.

**Snapshot id.** A content id: a hash of a stored snapshot's sealed bytes, in the canonical safetensors form
(`docs/engine-abi.md` D10). When the snapshot is stored as a piece of
a chain, the hash also covers its parent's content id (`docs/engine-abi.md` D3). Each stored snapshot records its
**coverage**: the positions that it holds per part. The same chain, stored from different slots, holds different
rows. mkviolet's sliding layers keep a ring's worth in a large slot and every row in a small one
(`gemma4.rs:44-63`). Whether a snapshot fits a slot is a check on coverage, not part of the id.

**Recipe.** What a snapshot was made from: a hash of the engine identity, the inputs, and the boundaries along the
chain. For an engine whose bits depend on slot geometry, the recipe also has a geometry key. The store maps a recipe
to the content ids made from it. Thus the nagaya can ask "do we already have this?" before it builds. mkviolet's
current `snapshot_id` is a recipe (`held.rs:312-325`).

**Tolerance.** What a request accepts in place of exact bits: `exact`, or named approximations. Each approximation
has a quality budget that its approver measured once. (ds4-recipe's `[approx — approved]` uses KL, dNLL, and the
agreement of top-1 tokens against a noise floor.) A snapshot made by an approximation has its own recipe, tagged
with the method. A lookup returns only snapshots in the request's tolerance. A snapshot of a `per-process` engine
from another process is one such approximation.

**Slot.** Device memory that an engine reserves for one live context.

- mkviolet fixes its slots and their sizes at load (`new_slots`, `gemma4.rs:445-475`). It has no stored token
  count: the position is the caller's `pos0`, and `kv_high` holds the high-water mark.
- mk allocates contexts on demand (`new_context`) and keeps fork buffers on a free list (`fork_free_`).

Both are "a context resident on a device". An engine's limits say how many fit.

## Time

**Op.** A step inside one pass, private to the engine. The nagaya never sees ops.

- mkviolet: a 64-byte `MkTaskDescriptor` in a per-card task ring, ordered by scoreboard wait and signal events
  (`protocol.rs`).
- mk: one entry in a pass's stage list (`stage.h`, "A stage is one op with its arguments"). mk's word "stage" names
  an op, not a stage in the sense below.

**Pass.** One run of an engine over a set of segments. mk: one cooperative launch over all 48 layers, with an MVP
target of 256 tokens in total (mk `DESIGN.md`, "Workloads and passes"). mkviolet: one block through both cards.

**Stage.** The part of a pass that runs on one device: a range of layers, and maybe the head. mkviolet has two
stages: layers 0-14 on card 0, and layers 15-29 and the head on card 1 (`CARD_LAYERS`, `gemma4.rs:84`). mk has one.

**Residual.** The hidden rows that one stage hands to the next. mkviolet hands them across on the host:
`run_card0` returns them, and `run_card1` writes them to card 1.

**Quantum.** The unit that the nagaya schedules: one pass, or one stage of a pass. It is not longer than the
engine's `max_quantum_ms` (`docs/engine-abi.md` D8). A quantum runs to its end. The nagaya changes what runs only
between quanta. Other work between quanta gives the same bits as a run without it, when all of these conditions
hold:

1. the same engine identity, which includes the arithmetic path (mk `fused`, `MK_FAST_QMM`);
2. the same boundaries, so the same seals;
3. state that leaves a slot is parked bit for bit, not sealed;
4. on resume, a slot that the parked state's coverage fits (mkviolet ring rows);
5. no other job's quantum writes the slot between two stages of one block (mkviolet's stages share a slot);
6. lab-only state that changes bits (mk's attention cell mask) counts as an input;
7. engine history that feeds arithmetic is the same. This fails for an engine that declares `bit_dependence`, such
   as ds4-recipe's routing by which experts are cached.

mk's `YieldHook` (`lab/paint/yieldhook.py`) showed this for diffusion. A paused run matched the run without a pause,
with a maximum pixel difference of 0.

**Kind.** What a quantum does: `read`, `prefill`, `decode`, `embed`, `copy`, `step` (an image step), or `train` (a
micro-batch) (`docs/spec.md` 5.2). Metrics and the cost model use kinds. A kind does not set urgency; a class does.
mk calls it `mk.gpu.kind`.

**Segment.** One job's share of a pass: a context, its new inputs, where they go, and which outputs it wants. A pass
carries segments from several jobs. Each segment gets the bits that a pass of it alone would give (mk `engine.h`
`Segment`, up to 16 per pass, `tests/engine/test_segments.py`). mk's `segments_problem` refuses two segments that
share K/V.

mkviolet's branch block has a different shape. It is one shared prefix plus branches in one block, with per-token
`meta`, all in one slot, when it fits (`branch_block`, `gemma4.rs:1521-1554`). The next forward from that position
overwrites its branch rows. Thus a branch block is atomic and is not a snapshot. Council v2's ride puts a decision
on a generation's context (mkviolet `council_v2.rs`).

**Job.** The quanta of one request, in order, with a class. Examples:

- a council decision: prefill quanta, then read quanta;
- a generation: prefill, then one decode quantum per token;
- an image: encode, steps, decode;
- training: micro-batch quanta until something preempts it.

**Class.** How urgently a job's quanta run. mkviolet has two classes: a council decision goes before a generation's
next stage (`gemma4.rs` `CardLock`, `GpuLock`). The planned set is interactive, batch, and filler (training that
gives way to anything else). The term is "class", not "tier", because tier already names a place.

**Stream.** A job with no end, a period, and a deadline per period. Examples are frames of home video, audio hops
for the cochlea, and beat tracking. A stream says what to do when it is late: drop, catch up, or degrade. It reports
missed deadlines. A class says how urgent work is, and a stream also says how often. Neither engine has a stream
yet.

## Holding

**Lease.** Time on a resource, with an end. A rented machine is a lease on hardware (until Sunday 9 pm). An arena
that the file-descriptor store keeps is a lease on memory that outlives the process. To drain a lease:

1. Accept no new jobs.
2. Let quanta go to their ends.
3. Park pinned and wanted state to a lasting tier.
4. Let the resource go.

**Pin.** A lease on a snapshot and the chain under it. It keeps them out of eviction until it expires. mkviolet
keeps pins in `pins.json` beside the snapshots, so they outlive a restart (`held.rs` `PinSet { snapshots, expires
}`).

**Ledger.** One machine's account of its memory. It has a budget per tier: one shared pool on gfx1151host and sm121host, and
separate device and host budgets on b70host. The ledger admits every engine's mandatory bytes first. It divides
what remains among the engines' elastic terms, by the weights in the machine's placement profile
(`docs/engine-abi.md` D9). Counts come from the ledger's arithmetic, never from probes of free memory. The idea
comes from ds4-recipe's memory ledger (`docs/prior-art.md` 2).

"Tenant" in the README is a figure of speech for an engine, not a term.

## Open questions

None are open. These were decided on 2026-10-17:

- weights are a state object (see Arena);
- an engine that spans devices schedules its own stages for now (D1);
- snapshot ids are content ids, with recipes for lookup and coverage for fit (D3).
