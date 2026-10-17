# Ontology

One term per thing. Each entry gives the definition, then where the thing already exists. mk is
`~/src/megakernel-qwen38-flashnext-strixhalo`; mkviolet is `~/src/mkviolet`.

The nouns fall into four groups: places (where things are), things (what is held), time (what runs), and holding
(who keeps what, and until when).

Revised 2026-10-17 after a kaibo review (deepseek cast) checked the first draft against copies of both engines'
source. Its main corrections: cell storage differs by engine, sliding layers are rings, seal and park are different
steps, and a snapshot id must carry the slot geometry.

## Places

**Machine.** One computer. It runs exactly one nagaya process, which owns all of the machine's GPUs.
Today: gfx1151host (Strix Halo, one gfx1151 GPU, unified memory), the B70 box (two Intel Arc Pro B70, 64 GB GDDR6 total,
Ryzen 9 9950X, 128 GB DDR5, 4 TB NVMe), a third lab machine, and rented machines (`~/src/rented-gpu`).

**Device.** One thing that runs quanta: a GPU, or the CPU when an engine runs work there (mkviolet's AVX-512 lens,
`docs/03-distributed-state-architecture.md` 2.5). mkviolet keeps one Level Zero context per card
(`gemma4.rs:7`); its `pipeline_stage` spawns a thread per card for each stage. In the nagaya, each device has one
long-lived thread and every call for that device comes from it. That rule is new; mkviolet's per-stage threads
move into it.

**Tier.** A place a state object can be: device memory, host memory, NVMe, or another machine. Tiers are a graph with
measured bandwidths per machine, not a fixed list. On gfx1151host, device memory and host memory are the same DDR5; on the
B70 box they are separate (608 GB/s GDDR6 per card, about 90 GB/s DDR5, about 14.8 GB/s NVMe, mkviolet `docs/03`).

## Things

**Engine.** A model's code and weights, loaded on one or more devices of one machine. It exports the engine ABI
(`docs/engine-abi.md`). It has an identity: the model, a hash of its weights, its tokenizer and template, the binary
that computes its numbers (council `identity`; mk's `libmkengine:<sha>+numbers:<sha>`), and every setting that changes
bits (mk: `fused`, `MK_FAST_QMM`'s f16 WMMA path versus exact f32, `engine.h:19-32`). Two runs with different
identities are not expected to match.

**Build.** The binary only. The history of how a context was made is its **chain**.

**Arena.** An engine's weights as memory that can outlive the process: a set of named file descriptors. mk has one
sealed arena per engine; another engine attaches to it read-only after a full check against its own build
(`capi.hip:12-27`, `mk_arena_check`). mkviolet has one DMA-BUF slab per layer per card, 60 in all, kept through
`systemctl restart` by systemd's file-descriptor store (`gemma4.rs:143-146`, `slab_fd_name`; `hot_handoff.rs`
`store_fd`); its check, `verify_stored_slab`, compares only the first and last 64 bytes of each tensor
(`gemma4.rs:284-287`). An arena records how strong its check is. mkviolet's `memfd` host arena (`HotDdr5Arena`,
`hot_handoff.rs:130-201`) holds host-tier state and a header, not weights, so it is not an arena in this sense.

**State object.** Everything a pass continues from, made of parts. Two kinds:

- a **context** is what passes run on;
- a **snapshot** is a context sealed at a boundary that nothing passes into. Reads run on a copy (mk `fork`) or load
  it into a slot (mkviolet `load_kv`).

**Part.** One named piece of a state object. Four kinds:

| Kind | What it is | Examples |
|---|---|---|
| fixed | a constant-size block that each pass replaces whole | mk GDN recurrent state `[36][48][128][128]` f32, GDN conv, PLE conv, PLE history (`engine.h:63-75`); a diffusion latent |
| cells | rows indexed by position, only appended | mk KV and index keys; mkviolet KV on global layers |
| ring | rows in a window that wraps: new rows overwrite the oldest | mkviolet sliding layers (`SLIDING_WINDOW` 1024, `KV_RING_ROWS` 4096, `kv_ring`, `gemma4.rs:30-46`) |
| derived | rebuilt from other parts, never stored | mk pooled index keys (`idx_pooled`) |

How a part is stored depends on the engine as well as the kind:

- mk reads a cells part whole, cells 0 to n (`engine.h:173-174`), with no range read. An mk snapshot is therefore
  stored whole. At n = 3584 that is about 176 MB of KV, 22 MB of index keys, 113 MB of GDN state, 4.4 MB of GDN conv
  and 0.4 MB of PLE conv: about 316 MB.
- mkviolet stores a chain: a global layer's snapshot holds the rows after its parent, and a ring layer's snapshot
  holds a suffix that can overlap its parent's rows. `KvSnapshot::join` merges the chain for a given slot size and
  fails if the chain lacks rows the slot needs (`gemma4.rs:941-979`).

A range read (`docs/engine-abi.md`, `cells_read`) would let mk store chains too.

**Boundary.** A position in a context's chain where it may be sealed. Council's `snap: true` turns and spec layers
are boundaries. Boundaries are part of a snapshot's id.

**Seal.** The engine's rule at a boundary. It may change bits: mk rounds the GDN recurrent state to bf16 in place
(gdn-bf16-v1, `round_gdn_bf16`, `engine.h:189-193`, mk `docs/park.md`). mkviolet's seal changes nothing.

**Park.** Moving a state object to another tier, bit for bit. Parking never seals. Sealing happens only at boundaries
the chain declares, so parking a context because the scheduler needs its slot leaves no trace in later results.

**Snapshot id.** A hash of the engine identity, the inputs, the boundaries along the chain, and the engine's
geometry for that state (`docs/engine-abi.md` D3). Geometry matters: mkviolet's rows depend on the slot size, and the
same chain joins for one slot size and fails for another (`gemma4.rs:1692-1693`). mkviolet's current id hashes the
identity, the tokens and the cuts only (`held.rs:312-325`).

**Slot.** Device memory an engine reserves for one live context. mkviolet fixes its slots and their sizes at load
(`new_slots`, `gemma4.rs:445-475`) and has no stored token count: the position is the caller's `pos0`, and `kv_high`
holds the high-water mark. mk allocates contexts on demand (`new_context`) and keeps fork buffers on a free list
(`fork_free_`). Both are "a context resident on a device", and an engine's limits say how many fit.

## Time

**Op.** A step inside one pass, private to the engine. mkviolet: a 64-byte `MkTaskDescriptor` in a per-card task
ring, ordered by scoreboard wait and signal events (`protocol.rs`). mk: a stage in a stage list (`stage.h`). The
nagaya never sees ops.

**Pass.** One run of an engine over a set of segments. mk: one cooperative launch over all 48 layers, with an MVP
target of 256 tokens in total (mk `DESIGN.md`, "Workloads and passes"). mkviolet: one block through both cards.

**Stage.** The part of a pass that runs on one device: a range of layers, maybe the head. mkviolet has two: layers
0-14 on card 0; layers 15-29 and the head on card 1 (`CARD_LAYERS`, `gemma4.rs:84`). mk has one.

**Residual.** The hidden rows handed from one stage to the next. mkviolet hands them across on the host
(`run_card0` returns them; `run_card1` writes them to card 1).

**Quantum.** The unit the nagaya schedules: one pass, or one stage of a pass. A quantum runs to its end; the nagaya
changes what runs only between quanta. Interleaving other work between quanta gives the same bits as an uninterrupted
run when all of these hold:

1. the same engine identity, including the arithmetic path (mk `fused`, `MK_FAST_QMM`);
2. the same boundaries, so the same seals;
3. state that leaves a slot is parked bit for bit, not sealed;
4. the same slot geometry on resume (mkviolet ring rows);
5. no other job's quantum writes the slot between two stages of one block (mkviolet's stages share a slot);
6. lab-only state that changes bits (mk's attention cell mask) counts as an input.

mk's `YieldHook` (`lab/paint/yieldhook.py`) showed this for diffusion: a paused run matched the unpaused one, max
pixel difference 0.

**Segment.** One job's share of a pass: a context, its new inputs, where they go, and which outputs it wants. A pass
carries segments from several jobs, and each gets the bits a pass of it alone would give (mk `engine.h` `Segment`,
up to 16 per pass, `tests/engine/test_segments.py`; mk's `segments_problem` refuses two segments that share K/V).
mkviolet's branch block is a different shape: one shared prefix plus branches in one block with per-token `meta`,
all in one slot, when it fits (`branch_block`, `gemma4.rs:1521-1554`). Its branch rows are overwritten by the next
forward from that position, so a branch block is atomic and is not a snapshot. Council v2's ride puts a decision on
a generation's context (`council_v2.rs`).

**Job.** The quanta of one request, in order, with a class. A council decision is prefill quanta then read quanta; a
generation is prefill then one decode quantum per token; an image is encode, steps, decode; training is micro-batch
quanta until preempted.

**Class.** How urgently a job's quanta run. mkviolet has two: a council decision goes before a generation's next stage
(`gemma4.rs` `CardLock`, `GpuLock`). The planned set: interactive, batch, filler (training that gives way to anything
else). "Class", not "tier": tier already names a place.

**Stream.** A job with no end, a period, and a deadline per period: frames of home video, audio hops for the cochlea,
beat tracking. It says what to do when late (drop, catch up, or degrade) and reports missed deadlines. A class says
how urgent work is; a stream also says how often. Neither engine has one yet.

## Holding

**Lease.** Time on a resource with an end. A rented machine is a lease on hardware (until Sunday 9 pm). An arena kept
by the file-descriptor store is a lease on memory that outlives the process. Draining a lease: accept no new jobs, let
quanta reach their ends, park pinned and wanted state to a lasting tier, then let go.

**Pin.** A lease on a snapshot and the chain under it: it stays out of eviction until it expires. mkviolet keeps pins
in `pins.json` beside the snapshots so they outlive a restart (`held.rs` `PinSet { snapshots, expires }`).

"Tenant" in the README is a figure of speech for an engine, not a term.

## Open questions

1. Is a weight arena a state object (fixed parts only, no inputs), or its own thing? A training engine writes its
   weights, so for training the answer may be "state object".
2. Does an engine that spans devices expose its stages to the nagaya, or schedule them itself? `docs/engine-abi.md` D1.
3. Who supplies the geometry in a snapshot id: the engine as a key the nagaya hashes, or the whole id? D3.
