# Ontology

This doc has one term for each thing that the nagaya needs. The nagaya is an exokernel for megakernels: one process
owns a machine, and each engine keeps its own abstractions. Each entry gives the definition, then often an example
from the Qwen3.8 engine or the Gemma 4 engine. A term in bold inside an entry is defined there. The spec
(`docs/spec.md`) and the ABI (`docs/engine-abi.md`) hold the rules and the details.

The terms are in four groups: places, things, time and holding. Places are where things are, and things are what is
held. Time is what runs, and holding is who keeps what, and until when.

## The other projects

The docs name these other projects. Each one is defined here and nowhere else.

**The first Qwen3.8 engine.** The practice engine whose code the nagaya copied into `nagaya-mk-qwen38-gfx1151`.

**The first Gemma 4 engine.** The practice engine whose code the nagaya copied into `nagaya-mk-gemma4-bmg`.

**gogi.** The council v2 contract (github.com/tobert/gogi): the text `docs/council-api.md`, the types in `gogi-proto`,
the suite `gogi-conformance`, and the toy family `gogi-toy`. gogi is normative today. When `docs/spec.md` and gogi
differ about the API, gogi is the rule (`docs/spec.md` 11.1).

**kaijutsu.** The agent. It coordinates models through council v2: casts, seats, judges and bumps (`docs/spec.md`
1). It is the first client of the nagaya, and the nagaya follows its telemetry conventions (`docs/spec.md` 9).

## Places

**Machine.** One computer. It runs one nagaya process, and that process owns all of the machine's GPUs. The machines
today are a Strix Halo machine and a B70 machine (`docs/spec.md` 2.1), plus rented machines. The processes gossip their
inventory, active profile, and engine identities through `chitchat` (`docs/spec.md` 2.3). Any machine can act as the
**coordinator** for a client request (`docs/spec.md` 11.1).

**Device.** One thing that runs quanta: a GPU, or the CPU when an engine runs work there.
A **device group** is the set of devices that one or more engines use. Each device group has one long-lived nagaya
thread, and every engine call for the group comes from that thread. Two engines that use one device share one group.
The Gemma 4 engine spans both B70 cards, so its group is the two cards. Each card runs one stage, on a stage thread
of its own (`docs/engine-abi.md` D1).

A device has two names. Its **machine device id** is its id on the machine, as the inventory gives it. The profile,
the budgets, the metrics and the fd names use it. Its **group index** is its position in the profile's `devices`
list of an engine. The engine ABI uses it. The ledger maps one to the other, so an engine never sees a machine
device id (`docs/spec.md` 2.4, "One name for each device").

**Placement profile.** The list of a machine's engines, their devices, their config, the budgets, the shares, the
store's budgets and pinned identities (`docs/spec.md` 2.4). It is one TOML file in the store directory.

**Bootstrap config.** The small TOML file, `nagaya.toml`, that a process reads at start. It holds the bind
addresses, the store directory, the seed peers, the node name and the park budget (`docs/spec.md` 2.1). It holds no
engine.

**Tier.** A place where a state object can be: device memory, host memory, NVMe, or another machine. Tiers are a
graph with measured edges per machine, not a fixed list. On a machine with unified memory, device memory and host
memory are one pool. On the B70 machine they are separate: 608 GB/s GDDR6 per card, about 90 GB/s DDR5, about
14.8 GB/s NVMe.

## Things

**Engine.** A model's code and its weights, loaded on one device group. It exports the engine ABI. Its identity
holds the model, the checkpoint digest of its weights, and its tokenizer and template. It also holds the engine's
**build** (the binary that calculates its numbers), its plan table lineage, and its canary hash. Last, it holds
every setting that changes bits (`docs/spec.md` 12.1). Two runs with different identities are not expected to match.

An engine is **fatal** after a canary mismatch or a fourth timeout in 600 s at runtime. The nagaya refuses its
later jobs, and the other engines continue (`docs/spec.md` 8.2). Its served models report the council status
`unloaded` (`docs/spec.md` 11.1).

**Family.** The text rules that a set of models share. They are a tokenizer, a chat template, the escapes for
special tokens, and the parsers for tool calls and thinking. Qwen, Gemma and LFM are families. The nagaya holds one
library per family and selects it by the engine identity. An engine never sees text (`docs/spec.md` 11.3).

**Plan table.** An engine's record of its free evaluation choices, such as the GEMM algorithm for each shape. Each
choice is valid, but different choices give different bits, so the table is part of the identity. The table is
append-only: the engine times a new shape once and keeps its pick. A new library, driver or device starts a new
**lineage**, which is a new identity (`docs/engine-abi.md` D3). The Qwen3.8 engine has no library GEMMs and needs no
plan table.

**Arena.** The memory that holds an engine's weights and can outlive the process: a set of named file descriptors.
An arena records how strong its check is. The Qwen3.8 engine has no check of a stored arena yet (`docs/engine-abi.md`,
"How the Qwen3.8 engine maps"). The Gemma 4 engine has 31 DMA-BUF slabs, one for each of its 30 layers and one for
the head. It checks the first and last 64 bytes of each tensor (`docs/spec.md` 10.2). An inference engine opens its
arena read-only, and a training engine writes it.

**State object.** Everything that a pass reads besides its inputs, made of parts. There are three kinds:

- **weights**: fixed parts only, with no inputs. Inference reads them, and training writes them.
- a **context**: what passes run on and extend.
- a **snapshot**: a context sealed at a boundary. Reads run on a copy, or load it into a slot (`load_kv` of
  the Gemma 4 engine).

**Part.** One named piece of a state object. There are four kinds:

| Kind | What it is | Examples |
|---|---|---|
| fixed | a constant-size block that each pass replaces whole | Qwen3.8 GDN recurrent state, GDN conv, PLE history (`engine.h`); a diffusion latent |
| cells | rows indexed by position, only appended | Qwen3.8 KV and index keys; Gemma 4 KV on global layers |
| ring | rows in a window that wraps: new rows overwrite the oldest | Gemma 4 sliding layers (`KV_RING_ROWS` 4096) |
| derived | rebuilt from other parts, never stored | Qwen3.8 pooled index keys (`idx_pooled`) |

**Chain.** The history of how a context was made: its inputs and boundaries, in order. The store can keep a
snapshot as pieces of its chain, and each piece names its parent's content id. The Qwen3.8 engine stores its snapshots
whole, about 316 MB at n = 3584. The Gemma 4 engine stores chains and joins them for a slot
(`KvSnapshot::join`).

**Boundary.** A position in a chain where a context may be sealed. Clients mark boundaries: council's `snap: true`
messages and spec layers.

**Seal.** The engine's rule at a boundary. It may change bits: the Qwen3.8 engine rounds the GDN recurrent state to bf16
in place (`round_gdn_bf16`). The Gemma 4 engine declares no seal.

**Park.** A move of a state object to another tier, bit for bit. A park never seals. Thus when the scheduler parks a
context to free its slot, later results do not change (`docs/engine-abi.md` D7).

**Interval copy.** A store entry of a context's parts at a multiple of the engine's checkpoint **grain**, taken
between quanta. It is not sealed, so the context's later bits do not change. Thus it is not a snapshot, which is
sealed. Its recipe has the item "not sealed", so a lookup for a snapshot never finds it. It lives only in host memory
(`docs/spec.md` 6.13).

**Content id.** The name of a stored snapshot: the BLAKE3 hash of its canonical safetensors file. For a piece of a
chain, the hash also covers the parent's content id (`docs/engine-abi.md` D3, D10). Each stored snapshot also
records its **coverage**: the positions that it holds in each part. Coverage is in the canonical file, so it is in
the content id. Whether a snapshot fits a slot is a check on coverage, not part of the id.

An object also has a content id. The printed form of each content id is `blake3:` and 64 hex digits, always whole
(`docs/spec.md` 6.2).

**Object.** Bytes and a media type that a client or an engine gives to the store: an image, a sound, a video, a MIDI
file. It is named by its content id. The store can evict a **cached** object. A **kept** object has a **keep**: a
lease with an owner and an expiry (`docs/spec.md` 6.15). kaijutsu's CAS holds the same kind of thing.

**Control vector.** An object that holds one vector for each layer of one model. A context names it with an
operation and a strength, and the engine changes the residual after each such layer (`docs/spec.md` 11.12). Other
texts call it a steering vector.

**Layer loop.** A span of a model's layers that runs more than once for each position, as a context's `loop` names
it (`docs/spec.md` 11.16). Other texts call it a looped or recurrent block, or layer duplication.

**Run.** One execution of one layer for one position. A layer outside a loop has one run, and a layer inside it has
`count` runs. Each run has its own state: keys and values, or a recurrent state.

**Recipe.** What a snapshot was made from: a hash of the engine identity, the inputs and the boundaries. For an
engine that declares a geometry key, the recipe holds that key too. The store maps a recipe to the content ids made
from it, so the nagaya can ask "do we already have this?" before it builds.

**Tolerance.** What a served model accepts in place of exact bits: `exact`, or named approximations. Each
approximation has a quality budget that its approver measured once. A snapshot made by an approximation has its own
recipe, tagged with the method. A lookup returns only snapshots in the served model's tolerance.

A **served model** is one name that clients send in council v2's `model` field (gogi). It names an engine and a
tolerance. A request never carries a tolerance. An approximation is a served model of its own, with its own name
(gogi `docs/council-api.md`, "Served models").

**Store.** The one content-addressed store on each machine. It holds the snapshots of every engine in host memory
and on NVMe, with budgets from the ledger (`docs/spec.md` section 6).

- The **index** holds what sealed bytes must not hold: times, hit counts, owners and reasons.
- An **entry** is one record in the index: a content id, its recipe, its parent, its coverage, and where it is.
- An **owner** is the client principal that created an entry or a job. Each owner has a **quota**: a byte limit,
  so one owner cannot evict another owner's entries.

**Slot.** Device memory that an engine reserves for one live context. The Gemma 4 engine fixes its slots and their
sizes at load (`new_slots`). The Qwen3.8 engine allocates contexts on demand (`ctx_new`). An engine's limits
say how many fit.

A slot's **geometry** is its size and its ring rows. Geometry changes coverage: a small slot keeps every row, and a
large one keeps a ring's worth. It changes bits only for an engine that declares a geometry key.

## Time

**Op.** A step inside one pass, private to the engine. The nagaya never sees ops. The ops of the B70 engines are 64-byte
task descriptors (`crates/nagaya-mk-bmg/src/protocol.rs`). The ops of the Qwen3.8 engine are the entries of a pass's
stage list: "A stage is one op with its arguments" (`stage.h`). Its word "stage" names an op, not a stage in the sense
below.

**Pass.** One run of an engine over a set of segments. Qwen3.8 engine: one cooperative launch over all 48 layers.
Gemma 4 engine: one block through both cards.

**Stage.** The part of a pass that runs on one device. The Gemma 4 engine has two: layers 0-14 on card 0, and layers
15-29 and the head on card 1 (`CARD_LAYERS`). The rows that one stage hands to the next are the **residual**. The
Qwen3.8 engine has one stage.

**Quantum.** The unit that the nagaya schedules: one pass, or one stage of a pass. It is not longer than the
engine's `max_quantum_ms` (D8), and it runs to its end. The nagaya changes what runs only between quanta. Other work
between quanta gives the same bits as a run without it, when all of these conditions hold:

1. the same engine identity, which includes the arithmetic path (`fused` and `QWEN38_FAST_QMM` of the Qwen3.8 engine);
2. the same boundaries, so the same seals;
3. state that leaves a slot is parked bit for bit, not sealed;
4. on resume, a slot that the parked state's coverage fits;
5. no other job's quantum writes the slot between two stages of one block;
6. lab-only state that changes bits (the attention cell mask of the Qwen3.8 engine) counts as an input;
7. engine history that feeds arithmetic is the same. This fails for an engine that declares `bit_dependence`.

**Kind.** What a quantum does: `read`, `prefill`, `decode`, `embed`, `copy`, `step` (an image step), or `train` (a
micro-batch). Metrics and the cost model use kinds. A kind does not set urgency; a class does.

**Segment.** One job's share of a pass: a context, its new inputs, where they go, and which outputs it wants. A pass
carries segments of one class, from one or more jobs. Each segment gets the bits that a pass of it alone would give. The
Qwen3.8 engine takes up to 16 segments in a pass (`Segment` in `engine.h`). The Gemma 4 engine's branch block is not a
segment. It puts a prefix and its branches in one slot, and the next forward overwrites the branch rows
(`branch_block`).

**Job.** The quanta of one request, in order, with a class. A council decision is prefill quanta, then read quanta.
A generation is prefill, then one decode quantum per token. An image is encode, steps, decode. Training is
micro-batch quanta.

A job is **ready** when no quantum of it runs, the inputs of its next quantum exist, and the ledger admitted that
quantum. The scheduler chooses only among ready jobs. A filler job declares a **checkpoint interval**: the time between
its checkpoints.

**Class.** How urgently a job's quanta run: `interactive`, `batch`, or `filler`. Filler runs only when no other work
is ready on its device. The term is "class", not "tier", because tier names a place.

## Holding

**Lease.** Time on a resource, with an end. A rented machine is a lease on hardware. An arena in systemd's
file-descriptor store is a lease on memory that outlives the process. A job's hold on a live context is a lease,
not a pin. A **loan** is a lease on a mandatory term that one job at a time holds, such as prefill scratch. Each
terminal state of the job releases it: done, cancelled, failed or expired.

To drain a lease:

1. Accept no new jobs.
2. Let each quantum finish.
3. Park pinned and wanted state to a lasting tier.
4. Release the resource.

**Pin.** A lease on a snapshot and the chain under it. It keeps them out of eviction until it expires. Pins outlive
a restart.

**Ledger.** One machine's account of its memory, with a budget per tier. The ledger admits every engine's mandatory
bytes first. It divides what remains among the engines' elastic terms by the shares in the machine's placement
profile (D9). Each elastic term gets at least its minimum. Counts come from the ledger's arithmetic, never from
probes of free memory.

- A **budget** is a byte target for one or more tiers, from the placement profile. It is not a measurement of free
  memory. On a machine with unified memory, one budget holds both `device(0)` and
  `host` (`docs/spec.md` 4.1).
- A **term** is one claim of an engine on a budget, from `limits().memory`. A **mandatory term** has fixed bytes.
  An **elastic term** has a unit size, a minimum and a maximum count of units (`docs/spec.md` 4.2).
- A **share** is an engine's weight in the division of spare bytes (D9).
- **Admission** is only the ledger's check of a call that allocates, or of a profile, before it runs
  (`docs/spec.md` 4.3, 2.4). The client API's limit on requests in flight is the **door limit**, not admission
  (`docs/spec.md` 5.10).
- A **staging charge** is the ledger's charge for a copy of engine state in host memory. It lasts from the start of
  the copy to its drop (`docs/spec.md` 4.2). It is not the staged path of an IO request (`docs/spec.md` 7.2).

The **safety floor** is a different check. It watches the host's free memory for what the ledger cannot see
(`docs/spec.md` 4.7).
