# nagaya spec v0 (draft)

This spec says what the nagaya is and what it does, one rule at a time. It is written in Simplified Technical
English (`AGENTS.md`).

- `docs/ontology.md` defines each term once.
- `docs/engine-abi.md` defines the calls and the decisions D1-D10. The spec refers to them by number.
- `docs/prior-art.md` and `docs/reviews.md` record where the rules came from.

When this spec and another doc do not agree, correct one of them. Neither one wins without a decision.

Status: draft, 2026-10-18. The sources are the design conversations with Amy and inventories of mk
(`~/src/megakernel-qwen38-flashnext-strixhalo`) and mkviolet (`~/src/mkviolet`). The code follows `AGENTS.md`. Each open
question has the mark **Open**, and section 15 lists all of them.

## 1. Scope

The nagaya is one process on each machine. It owns the machine's GPUs, and engines run inside it. It decides three
things: where engines and state live, what runs next on each device, and what the store keeps. It does not decide
what a client asks a model.

| Layer | Owns | Does not own |
|---|---|---|
| kaijutsu (the agent) | coordination: casts, seats, judges, bumps, which model answers what | devices, memory, scheduling |
| nagaya | placement, the ledger, scheduling, the store, IO, machine telemetry, lifecycle, the client API | what a request means; kernels |
| engine | a model's code and weights, its kernels, its private layouts and caches | anything that another engine uses |

Amy, 2026-10-17: "Coordination belongs in kaijutsu, yeah. It's the agent."

Goals for v0:

- Several engines run on one machine at the same time: inference endpoints with different models, image batch, and
  training as filler. All three kinds of engine are in v0 (Amy, 2026-10-17). Thus the ABI must cover image steps
  and training before v0 is done.
- Each engine gives the same bits for the same identity and inputs.
- Nothing is lost or corrupted without a report.

Not in v0:

- One model split across machines. The links are gigabit today (README, "Between machines").
- Engines that load and unload in place while the process runs. In v0, a change of engines restarts the process
  (3.3).
- Cooperative yields inside a pass (D8).
- Tensor transfer between engines (13.3).

## 2. Machines, gossip and placement

**2.1 Inventory.** At start, the nagaya probes its own machine: its devices (amdgpu KFD, Intel Xe, or NVIDIA NVML),
its host memory, and its NVMe store path. It needs only a tiny bootstrap config (`nagaya.toml` or flags) with its
bind addresses, store directory, and seed peer addresses.

| Machine | Devices | Memory |
|---|---|---|
| gfx1151host | Strix Halo, one gfx1151 GPU | 128 GB unified |
| b70host | two Intel Arc Pro B70, a BlueField-2 | 64 GB GDDR6, 128 GB DDR5, 4 TB NVMe |
| sm121host | DGX Spark, one Blackwell GPU | 128 GB unified |

Telemetry takes hardware attributes from the probed inventory, not from constants in code. (mkviolet puts B70 values
in `telemetry.rs` as constants.) A CPU that runs work is a device in the inventory too, with host memory as its tier.
A machine with no GPU engine can also join the cluster to serve as a coordinator, run CPU devices (13.2), or hold a
store tier.

**2.2 Tier edges.** An edge records more than a bandwidth (`docs/prior-art.md` 5):

- the rate for each read size and queue depth;
- whether direct IO works into the destination;
- the alignment that the edge needs.

The nagaya measures edges at install, stores the result in its local store index, and can measure them again. Each
measurement is data with a date.

**2.3 Gossip (`chitchat`).** The nagaya processes form a cluster over UDP with `chitchat` (Scuttlebutt reconciliation
and phi-accrual failure detection). Gossip data stays small and write-light. Each machine writes only its own node
state:

- `inventory`: its probed devices, tiers and measured edges;
- `profile:active`: the placement profile that it currently runs;
- `profile:desired:<host>`: a target profile for `<host>` with a version, set when a client posts a profile to any
  machine;
- `engine:<name>`: the engine's `identity()`, its served models, its state (`loading`, `ready`, `draining`,
  `unloaded`, `fatal`), and its HTTP address.

Pins and snapshot entries never go in gossip.

**2.4 Placement profile.** A placement profile names a machine's engines, their devices, their config, each tier's
budget (4.1), each engine's share (D9), and pinned identities (12.1). A client reads or updates a machine's profile
through `GET` and `PUT /nagaya/v1/profile` on any machine in the cluster. The target machine persists its accepted
profile in its store directory, so it starts with that profile after a restart.

Before the nagaya applies a profile, it calls `plan(config)` for each engine, which gives the engine's limits without
a read of the weights. The nagaya adds all mandatory terms and the minimum of each elastic term. If the sum does not
fit the budgets (section 4), the nagaya refuses the profile before it opens an engine. The refusal gives the
shortfall in bytes for each tier. At cold start, a refused profile exits 3 (10.4). At runtime, a refused `PUT`
returns 507 and keeps the active profile.

**Proposed** by deepseek-flash, 2026-10-18: the refusal reports one entry for each short budget, with the tiers
that share it. A caller that adds the entries does not count a pooled budget twice. On gfx1151host, `device(0)` and
`host` are one pool (4.2).

## 3. Process

**3.1 One process per machine.** The nagaya owns every GPU on the machine (Amy: "we'd do 1 process per host that
owns the gpus"). The nagaya is the only GPU client. A tool that needs a device sends its work through the nagaya, or
stops the nagaya first. (ds4-recipe: a shader fault with a second GPU client made the SMU stop responding.)

**3.2 Threads and CPU isolation.** The nagaya isolates control work from compute work with `sched_setaffinity` and OS
thread priorities:

- **Control runtime (`tokio`).** Runs `chitchat` gossip and the HTTP coordinator on reserved housekeeping cores
  (`sched_setaffinity`) at elevated priority. It never runs tokenization, hashing, safetensors conversion, or engine
  passes directly.
- **One ledger thread.** Runs on the housekeeping cores and owns every budget: device memory, host memory and NVMe.
  Group threads and the store send admission requests to it. No other thread writes the ledger.
- **One watchdog thread.** Runs on the housekeeping cores and watches each group thread from outside, because a pass
  blocks the thread that it watches (7.4).
- **One thread per GPU device group.** A device group is the set of devices that one engine uses. Most groups have
  one device. mkviolet's Gemma 4 uses both B70 cards, so its group has two. Every engine call for a group comes from
  its thread, and the call is synchronous. Inside a pass, the engine runs its own work on each of its devices (D1).
- **A bounded CPU worker pool.** Rendering, tokenization, hashing, safetensors conversion, and CPU device passes
  (13.2) run on a bounded pool pinned to the remaining cores at `SCHED_BATCH`. Long CPU work yields between chunks
  to obey `max_quantum_ms`. On unified-memory machines (`gfx1151host` and `sm121host`), the scheduler pauses background CPU
  memory work during a GPU pass to protect GPU memory bandwidth.

**3.3 Engine lifecycle and profile swaps.** The nagaya opens each engine at start, from the profile. mk cannot close
inside a process today: "the HIP runtime's own teardown segfaults" (mk `service/__main__.py`). Thus in v0, when
`PUT /nagaya/v1/profile` changes the engines, the nagaya persists the profile and stops as in 10.3. It keeps its
arenas in the file-descriptor store (10.2) and exits with code 4, and systemd starts it again. Later, an engine whose
`close()` is safe loads and unloads in place (README door 1).

**Open:** Do the GPU queues of an idle engine slow the other engines on its device? mk has a note that its idle
queues slow another process, but we did not find the source. A measurement answers the question.

## 4. Ledger

The ledger is the machine's account of memory (ontology "Ledger", D9, ds4-recipe `05-memory-ledger.md`). The
ledger thread owns it (3.2).

**4.1 Budgets.** Each tier has one budget. A budget is a byte target in the placement profile (2.4). It is not a
measurement of free memory.

- gfx1151host and sm121host: one shared pool for device and host memory.
- b70host: one budget for each card's device memory, and one for host memory.
- Every machine: one budget for the store's host-memory tier, and one for the store on NVMe.

**4.2 Terms.** Each engine declares its terms in `limits().memory`. `plan(config)` gives the same terms before
`open` (2.4).

- Mandatory terms `{name, tier, bytes, lifetime, kinds}`. The lifetime is per engine, per context, or loaned (one
  holder at a time).
- Elastic terms `{name, tier, unit_bytes, minimum, maximum, kinds}`. The minimum and maximum are counts of units.
- A term's tier is `device(i)` or `host`, where `i` is an index into the engine's device group. The ledger maps each
  tier to a budget (4.1). On gfx1151host and sm121host, `device(0)` and `host` map to one pool. NVMe belongs to the store.
- Each term has a key by kind of quantum, where the bytes change by kind. A prefill's scratch is not a decode's
  scratch.

The nagaya declares its own terms in the ledger: staging, store buffers and IO. The ledger charges every allocation.
Bytes without a charge are a defect. (ds4-recipe: vision weights without a charge "show up as untracked bytes".)

**4.3 Admission.** The ledger thread admits each call that allocates before the call runs. These calls are `open`,
`attach`, `ctx_new`, `ctx_fork`, `ctx_copy`, `part_write`, and each resize. The check includes transient peaks. A
refusal charges nothing and changes nothing.

`pass` never allocates. Its scratch for each kind is a mandatory term, which the ledger charges at `open`.

When the ledger refuses a quantum's call, the quantum waits in the scheduler and keeps its place. A request that can
never fit, even with all reclaim done (4.6), gets 413 `too_large`. Its `param` names the limit in identity.

**4.4 Spare memory.** The ledger first charges all mandatory terms and the minimum of each elastic term. If they do
not fit, the nagaya refuses the profile (2.4). The ledger then divides what remains in each budget like this:

1. Give each engine a part of the spare bytes in proportion to its share (D9).
2. Inside an engine, split its part among its elastic terms in proportion to each term's minimum bytes.
3. Round each term down to whole units, and cap it at its maximum.
4. Leave the bytes that remain unassigned, and report them.

Counts come from this arithmetic, never from free memory. Decided 2026-10-18.

Some engines have bits that depend on the size of a cache. Such an engine declares that cache as a mandatory term,
not an elastic one. Thus reclaim and resize cannot change its bits, and each start gives it the same size.

**4.5 Resize.** An elastic term changes size only between quanta, through `resize(term, units)`. It does not shrink
while a lease refers to what would go. (A pin is a lease too.)

**4.6 Reclaim.** When admission needs bytes, the ledger tries these steps in this order:

1. Park idle contexts that have no lease to a lower tier, the oldest first (D7).
2. Shrink the elastic terms of idle engines to their minimums.
3. Refuse the request.

A park out of a slot uses `part_read`, which changes nothing in the slot. The nagaya frees the device copy only
after the copy in the lower tier is durable. Reclaim never seals. It never discards state that has no other copy.

**4.7 Host safety floor.** mk keeps 12 GiB of MemAvailable free (mk `service/contexts.py`, `FLOOR_BYTES`). In the
nagaya, a safety floor is a check for what the ledger cannot see: other processes, tmpfs and swap. It does not set
any size. When free memory goes below the floor, the ledger refuses new admissions, resizes nothing, and reports
the event.

On machines with unified memory, RSS and the OOM scorer do not show GTT. Thus the nagaya sets its own OOM score.

## 5. Scheduling

**5.1 Units.** A job is the quanta of one request, in order, with a class (ontology). A quantum is one pass or one
stage. It runs to its end, and the nagaya chooses what runs next only between quanta.

An engine meets its `max_quantum_ms` (D8) when it chooses the size of each unit of work. The engine knows its own
speed, so it picks the size from its own measurements. (mk splits a prefill into 256-token chunks in its Python
service, and that split moves into mk's engine wrapper.) The watchdog reports each quantum that runs past the bound
(7.4).

**5.2 Kinds.** Each quantum has a kind: `read`, `prefill`, `decode`, `embed`, `copy`, `step` (an image step), or
`train` (a micro-batch). Metrics and the cost model use kinds. A kind does not set urgency (mk `mk.gpu.kind`).

**5.3 Classes.** The classes are `interactive`, `batch` and `filler`. A client sends `interactive` or `batch` in
council v2's `class` field (gogi). Only jobs that the nagaya or a profile starts, such as training and the audit, are
`filler`. Between quanta, the scheduler always runs a
ready quantum of a higher class first. A quantum is ready when its job has no quantum that runs, its inputs are
present, and the ledger admitted it (4.3).

A council decision is interactive. A job's class starts to apply at its first quantum, not when the job arrives.
mkviolet's `lazy_high` takes the card at the council's first GPU step. That change cut decisions beside a prefill
from 383-702 ms to 300-390 ms.

**5.4 Order in a class.** The classes do not schedule the same way (Amy, 2026-10-18: "batch/filler can be fifo-ish,
while interactive is more cooperative multitasking"):

- In `batch` and `filler`, the scheduler orders jobs by integer priority (0 by default) and then time of arrival.
  One job runs its quanta in flow until it finishes, or until a higher class or higher priority job becomes ready.
  The scheduler does not interleave jobs of equal priority in `batch` or `filler`.
- In `interactive`, the scheduler interleaves ready jobs by three keys: integer priority (0 by default), owed
  seconds, and time of arrival. Owed seconds is the device time that a job is owed under equal sharing among the
  present jobs in `interactive`, clamped to ±30 s. A job that leaves takes only its own balance (ds4-recipe
  `11-multi-session-serving.md` section 6).

**Settled** by gogi (`~/src/gogi/docs/council-api.md:159-160`): "a higher priority runs first within its
class". **Proposed** by deepseek-flash, 2026-10-18: the scheduler picks the higher priority first, then the
higher owed balance, then the earlier arrival. On a quantum of d ms for job j among n present interactive jobs,
j's balance falls by d*(n-1)/n. Each other job's balance rises by d/n. Round down, and clamp the balance to
±30 s. A job that leaves removes its own balance.

The clamp forgives the excess, so the balances sum to zero only while no balance is at a bound.

**Proposed** by deepseek-flash, 2026-10-18: when an interactive pass has several segments, split the pass time
equally among its segments. The rounding remainder goes to the earliest-arrival segment.

**5.5 Filler.** A filler job declares a checkpoint interval: the time between its checkpoints. It must be resumable:
its state parks when it gives way (Ai2's "unallocated" time). It runs only when no interactive or batch quantum is
ready on its device group.

**Proposed** by deepseek-flash, 2026-10-18: one scheduler serves one device group (spec 3.2, D1: one thread per
device group). The filler gate is therefore per group.

**5.6 No preemption.** Nothing interrupts a quantum that runs. Priority orders the work that waits. Priority never
moves a continuation off the context that holds it. The job holds that context with a lease.

**5.7 Segments.** Quanta from several jobs on one engine can share a pass as segments. The scheduler permits this
only when the engine declares `segments_exact` (`docs/prior-art.md` 1). Only segments of one class share a pass, and
the quantum takes that class.

Each class has a join window: the time that the scheduler waits to add more ready segments to a pass. It never
waits for jobs that are not ready. In v0 each window is 0 ms, so a pass takes only the segments that are ready when
the scheduler chooses. The windows are in the bootstrap config (2.1), so a measurement can change them. Decided
2026-10-18.

**5.8 Deadlines.** A job can have a deadline (council v2 `timeout_ms`). The scheduler checks it between quanta. A
job after its deadline goes to the terminal state "expired", and its segments do not run again. On a status code,
an expired job is a 504 `timeout`. On a stream that already sent its first event, the nagaya sends a terminal
`error` event of type `timeout` (gogi).

**5.9 Cancellation.** A client cancels one job. The scheduler drops the job's next segments, and the quantum that
runs goes to its end. Cancellation is not failure: a cancelled job keeps its source snapshots. (ds4-recipe: the
loss of one such snapshot cost a cold prefill of 35 minutes.)

Every terminal state of a job releases its loans and leases. The terminal states are done, cancelled, failed and
expired. A release does not delete a snapshot. Only eviction (6.9) removes a stored entry.

**5.10 Admission at the door.** The client API admits a limited number of requests in flight for each engine (mk:
16). Above that limit it answers 429 with `Retry-After`.

**5.11 Cost model (D4).** The nagaya fits a robust affine cost from production quanta, keyed by engine, kind,
segment count and context depth. Quanta that ran beside other work are not in the fit. The nagaya uses the fit for
store eviction value-per-byte (6.9) and deadline checks between quanta (5.8). Before a fit exists, the nagaya records
"no model yet". An engine can have its own private planner, which stays in the engine (mkviolet `council_server.rs`).

## 6. State and the store

**6.1 One store per machine.** One content-addressed store holds the snapshots of every engine (D3). The store has
tiers (host memory, NVMe) and gets its budgets from the ledger. It replaces these stores:

- mk's parked files (`.mkctx`, a 50 GiB budget);
- mkviolet's `council_server` snapshots, its `held::Store`, and the prefix cache in `mk_api`. These are three
  formats and three SHA-256 id schemes on one host.

**6.2 Keys.** A snapshot's content id is the BLAKE3 hash of its sealed bytes. When the snapshot is a piece of a
chain, the hash includes its parent's content id. A snapshot's recipe is a hash of these items:

- the engine identity, which includes the tokenizer and the template;
- the logical inputs: the rendered text or the bytes;
- the boundaries;
- the engine's `geometry_key`, if the engine declares one (D3);
- the tags of each approximation;
- lab state that changes bits (6.12).

The recipe is BLAKE3 in derive-key mode, with the context string `nagaya recipe v1`. Each item is a `u16` tag, a
`u64` length and its bytes, all little-endian, in a fixed tag order. Tokens are `u32` little-endian. Recipes chain at
boundaries:

- The root recipe covers the identity digest, the `geometry_key`, the approximation tags, and the lab state.
- The recipe at boundary k covers the recipe at boundary k-1, the tokens since that boundary, and the position k.

Thus a prefix lookup hashes only the tokens since the last boundary. The identity digest is the same tagged hash over
the fields of `identity()` (12.1). The inputs are tokens, not text: the tokenizer and template are in the identity.
Decided 2026-10-18.

**Proposed** by deepseek-flash, 2026-10-18. The tags are: 1 identity digest, 2 geometry key, 3 approximation
tags, 4 lab state; 5 previous recipe, 6 tokens, 7 boundary position. The approximation item holds a `u64`
count, then each tag as a `u64` length and its bytes. The caller gives the identity digest; the identity hash
is spec 12.1. Reason: 6.2 fixes the item shape, but it does not give the tag numbers.

mk's `context_id` and mkviolet's `snap:<sha256>` are the input part of a recipe. They do not include the engine
identity. mk's id is a SHA-256 of the tokens and boundaries with `|gdn-bf16-v1|`.

**6.3 Sealed bytes are canonical.** In the store, each snapshot is a canonical safetensors file (D10). Each cells
part and ring part has the shape of its coverage: the rows that it holds, in position order. Thus two slots of
different sizes that hold the same rows make the same file. Padding is zero.

**Proposed** by deepseek-flash, 2026-10-18: spec 6.3 says "Padding is zero". D10 rule 5 says the JSON header
pads with spaces (0x20). The simplest reading is that "zero" names the data section and the header uses spaces.
The code follows that reading. Amy decides which rule changes.

Times, hit counts, owners and reasons go in the store's index, never in the sealed bytes. If they go in the sealed
bytes, two copies of one state get different content ids. Then the store keeps both copies, because it cannot see
that they are the same state.

**6.4 Entries.** Each entry in the index records:

- the content id, the recipe, the parent, and the coverage;
- the tier, the size, and the owner. The owner is the client principal that created the entry;
- the times of creation and last use;
- the reason for the store: anchor, interval, prefill done, evict, reclaim, or shutdown;
- the provenance of an approximate entry;
- the strength of the store's last check of the entry: `full` (every byte hashed) or `none` yet.

**6.5 Chains.** Each engine declares a maximum chain depth and a size fraction. (ds4-recipe: depth 1, and a delta
only when it is at most a quarter of a full snapshot.) A delta holds, for each cells part and ring part, the rows
after its parent's end. It holds each fixed part whole. The store never evicts a base while a delta needs it.

**6.6 Lookup.** A lookup is by recipe. It returns the chain of entries that covers the request, from its base to
its tip. The store keeps only the chains that fit the slot (coverage) and the served model's tolerance (ontology).

The store picks the deepest exact chain. The deepest chain is the one whose furthest covered position is highest in
every part. The store gives an approximate chain only to a served model whose tolerance names that method.

**6.7 The tripwire.** Some engines declare `always` or `pinned` (D3). For those engines, two content ids under one
recipe with the same coverage are a defect. The store quarantines both entries, refuses that recipe, and reports
it. A person clears the refusal after the cause is known.

For other engines, the audit job (8.4) does the check.

**6.8 Restore.** The nagaya checks the content id, the identity and the fit before it calls the engine.
`part_write` and `cells_write` return one of three results:

- ok;
- refused, and nothing changed;
- failed after a change. The state is poisoned (D6).

The store quarantines an entry only when its content id or its identity fails verification. A refusal leaves the
entry as it is. A failure never deletes content.

**6.9 Eviction.** The store chooses victims in this order:

1. Pinned entries are exempt.
2. Thin the chains: interiors go before tips and roots.
3. Remove whole entries by value per byte: recency, hits, and the cost to build them again.

The formula for value per byte is a choice of the implementation. It never changes bits. The store chooses all
victims before it deletes one. It also chooses them before it writes the new entry.

Each owner has a quota, so one long conversation cannot remove the others (a ds4-recipe incident). If a pin alone
is larger than a budget, the pin loses its protection, so eviction cannot deadlock.

**6.10 Pins.** A pin is a lease on a snapshot, with an owner and an expiry (mkviolet v2: one day at most). A request
pin holds a snapshot from its lookup to the end of its load. Pins stay through a restart (mkviolet `pins.json`).

**6.11 Restart.** At start, the store loads its index again. It checks each entry when a lookup first uses it. It
wakes pinned entries into slots when the profile asks for them. (mk wakes its pinned parked contexts before
`READY=1`.)

**6.12 Steering and lab inputs.** Some state changes bits but is not an input, for example mk's attention cell mask
or a steering vector. Such state goes into the recipe. Thus a steered or masked result never has a plain recipe.

## 7. IO

**7.1 One IO service per machine.** Engines (through the `io` capability) and the store send IO requests
`(file, offset, len, dest)` with a class to the IO service. The service owns these jobs:

- the rate on each edge and the queue depth;
- hedged reads across copies on different drives;
- the loss of a drive, and its return.

An IO request holds a lease on its destination until its final wait ends. This is also true after an error.

**7.2 Direct and staged bytes.** The service counts direct bytes and staged bytes separately. Thus a change from
direct IO to staged IO is visible.

**7.3 Replicas.** A store on several NVMe drives names each copy by content id. "Same size is not same bytes". A
copy that has no verification gives no reads.

**7.4 Watchdog.** A drive that hangs can hold a quantum until the kernel times out (about 30 s in ds4-recipe).
The watchdog thread (3.2) reports each quantum that is past `max_quantum_ms` by a margin. It runs on its own thread,
because the group thread is blocked in the pass.

## 8. Correctness at runtime

**8.1 Canary.** The nagaya records a canary for each engine: fixed inputs and the exact outputs that the engine gave
for them. The nagaya runs the canary through the engine, with its own sampler and decision code. mk's canary has
pilot reads and an 8-step greedy decode (mk `service/canary.py`). It keeps letter logits as f32 bit patterns, the
top 5, and a SHA-256 of all logits. The canary's hash is part of the engine identity.

The nagaya runs the canary again before the engine is ready, with no tolerance. At start, a mismatch makes the
nagaya exit with code 3, "do not restart" (10.4). After start, a mismatch marks the engine fatal (8.2), and the
process continues with its other engines.

**8.2 Pass failures.** A pass returns ok, poison (D6), or timeout. The nagaya treats outputs that are not finite as
poison. After a poison or a timeout, the nagaya does these steps:

1. Fail the jobs in that pass. Free their contexts, or mark their slots dirty until a job clears them (`slot_clear`),
   overwrites them from position 0, or writes them through `part_write`.
2. Run the canary again on the group thread, before any other work.
3. If the canary does not match, or if this is the fourth timeout in 600 s, mark the engine fatal. Refuse every
   later job for it.

(mk `service/gpu.py` does this.)

**8.3 Conformance.** Each engine has a conformance suite in the nagaya's tests. It checks these properties:

- park and restore keep every bit;
- seal runs only at boundaries;
- the quantum conditions hold (ontology "Quantum");
- an admission refusal changes nothing.

The suite compares values by integer bit pattern. NaN or Inf fails, even when the bit patterns are equal. Counts
must also match. Each gate keeps a broken variant that it must fail (ds4-recipe `01-correctness-and-measurement.md`).

**8.4 Audit.** For some engines the tripwire is not reliable, because of their determinism class. For them, the
nagaya runs an audit after each start, as a filler job in the same process. The audit builds a sample of stored
recipes again and compares the content ids (D3). A restart is the fresh process that the audit needs, so the audit
needs no second GPU client.

**8.5 Crash over corruption.** An unexpected failure in the nagaya stops the process (exit 1). The process does not
continue in a bad state.

## 9. Observability

The nagaya follows kaijutsu's conventions (`~/src/kaijutsu/docs/telemetry.md`). mkviolet already follows them.

**9.1 Activation.** The nagaya always logs to stderr. It also exports in these cases:

- OTLP over gRPC to the machine's collector, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set;
- OTLP JSON-lines files, when `NAGAYA_OTEL_FILE_DIR` or `KAIJUTSU_OTEL_FILE_DIR` is set.

`OTEL_SDK_DISABLED=true` stops all export. The `service.name` is `nagaya`. The resource attributes carry
`host.name` and the hardware from the inventory.

**9.2 Traces.** When a request has a W3C `traceparent` and `tracestate`, its span is a child of the caller's span.
The nagaya records an invalid header as an attribute, not an error. The spans nest in this order:

1. the HTTP server span, `<METHOD> <route template>`;
2. `nagaya.job`, with engine, class and request kind;
3. `nagaya.quantum` for each quantum, with engine, device group, kind, class, segments, and wait and run times in ms.

Decode quanta make metrics only: "one span per generated token would bury a generation's trace" (mk). The trace
context goes to the group thread with the job. The nagaya gives it to the engine through the telemetry capability,
and the engine adds its own spans under the quantum. Today, neither engine carries it onto GPU work.

**9.3 Sampling.** The nagaya uses kaijutsu's rule. It always keeps errors and spans whose parent is sampled. It
keeps 100% of the spans in some dot-qualified prefixes and 10% of the others. `nagaya.` and `council.` are 100%
prefixes, so `nagaya.job` and `nagaya.quantum` are always kept. Decode quanta make no spans, so the volume stays low.

**9.4 Metrics.** The nagaya keeps each metric attribute to a bounded set of values. Context, trace and job ids are
never attributes.

| Metric | Unit | Attributes |
|---|---|---|
| `nagaya.quantum.wait.duration`, `nagaya.quantum.run.duration` | ms | engine, device group, kind, class |
| `nagaya.quantum.tokens` | {token} | engine, kind |
| `nagaya.ledger.bytes` | By | tier, engine, term |
| `nagaya.ledger.refusals` | {refusal} | tier, engine, gate (budget, floor, resize) |
| `nagaya.store.bytes`, `nagaya.store.entries` | By, {entry} | tier, engine |
| `nagaya.store.lookups` | {lookup} | engine, outcome (exact, approx, miss, refused) |
| `nagaya.store.evictions` | {entry} | tier, reason |
| `nagaya.io.bytes` | By | edge, path (direct, staged), class |
| `nagaya.arena.handoff` | {handoff} | engine, status (hot, cold) |
| `nagaya.engine.state` | 1 | engine, state (loading, ready, draining, unloaded, fatal) |
| `http.server.request.duration` | s | http.route, status |

Each GPU vendor has a sensor provider: amdgpu (the SMU `gpu_metrics` table and KFD counters), Intel Xe, and NVIDIA
NVML. Each provider maps its values onto shared names:

| Metric | Unit | Attributes |
|---|---|---|
| `nagaya.device.clock` | MHz | device, clock |
| `nagaya.device.power` | mW | device, rail |
| `nagaya.device.temperature` | Cel | device, sensor |
| `nagaya.device.activity` | % | device |
| `nagaya.device.memory.used` | By | device, domain |
| `nagaya.device.throttle` | {event} | device, reason |
| `nagaya.device.evicted` | ms | device (amdgpu KFD only) |

mk's `mk.gpu.*` names and mkviolet's `mkviolet.*` names map onto these names. Council decision metrics (count,
spread, confidence) belong to the council layer (11.1).

**9.5 Each instrument has a call site under test.** mkviolet defines `record_gpu_pass` but calls it only from its
smoke test. It calls `record_council_decision` only in tests. Its docs name instruments for routing entropy and
J-Lens drift that do not exist. Thus a test shows that each instrument records on the path that it describes.

**9.6 Evidence on failure.** Before and after a quantum that fails, the nagaya reads the machine's counters. It logs
the change with the failure. mk reads these counters: KFD evicted ms, MemAvailable, GTT, and from `/proc/vmstat`
compaction stalls, migrations, swap-outs and direct scans.

**9.7 Status feed.** Each machine has one SSE status feed (mk `/mk/v1/status`, mkviolet `PassBus`):

- a snapshot when a client connects;
- one `quantum` event for each quantum, with engine, device group, kind, ms, and queue length by class;
- a keepalive every 15 s.

The feed disconnects a slow subscriber. It does not skip events for it. A subscriber that wants less volume asks for
a filter by engine or kind when it connects.

## 10. Lifecycle

**10.1 systemd.** The unit is `Type=notify`. The process sends `READY=1` when every engine in the profile has
opened, passed its canary, and woken its pinned state. `STATUS=` gives the state of each engine.

**10.2 Arenas through a restart.** systemd's file-descriptor store keeps the arenas (`FDSTORE=1`). Each fd has the
name `<engine>/<device>/<part>`. systemd reads `FileDescriptorStoreMax` from the unit file before the process starts.
The unit file sets a generous ceiling (`FileDescriptorStoreMax=512`), and profile admission (2.4) refuses a profile
that needs more arena fds than the ceiling. (mkviolet keeps 61 fds under a limit of 64, `systemd/mkviolet.service`.)

At start, the nagaya does these steps for each engine:

1. Check each stored arena against the engine's build, and record the strength of the check.
2. If an arena does not match, make sure that a cold load fits the ledger first (2.4). Then drop that engine's
   stored fds and load it cold. If a cold load does not fit, keep the fds and exit 3.
3. Duplicate each fd before use. (In mkviolet on 2026-10-16, an import closed the stored fd and caused a crash
   loop.)

`systemctl stop` clears the store. A restart keeps it.

**10.3 Stop.** On SIGTERM, the nagaya does these steps:

1. Accept no new jobs.
2. Let each running quantum go to its end.
3. Park pinned and wanted contexts to NVMe in a time budget (ds4-recipe: 10 s). The budget is in the bootstrap config
   (2.1).
4. Exit.

A second signal makes the process exit at once. Cancellation of running work follows 5.9. **Open:** the conversion
to canonical safetensors (D10) is not yet measured, so we do not know how many contexts fit in the budget.

**10.4 Exit codes.**

| Code | Meaning | systemd |
|---|---|---|
| 0 | a clean stop | no restart |
| 1 | a crash | restart |
| 3 | a refusal to start: a canary mismatch at start, an identity mismatch, or a profile that does not fit | `RestartPreventExitStatus=3` |
| 4 | a planned restart: a profile changed the engines (3.3) | `RestartForceExitStatus=4` |

The restart policy follows mk's unit: `Restart=on-failure`, `RestartSec=30`, and at most 3 starts in 600 s.
systemd counts each start against that limit, also a start after code 4. Thus a fourth profile swap in 600 s fails.

## 11. Client API

**11.1 Council and coordinator routing.** The nagaya serves council v2 for all engines in the cluster. It does not
serve v1: no client speaks v1 (gogi `docs/council-api.md`). The contract is gogi (`~/src/gogi`):
`docs/council-api-v2.md`, the types in `gogi-proto`, and the suite in `gogi-conformance`. mkviolet and kaijutsu move
to the same gogi before the nagaya serves them (Amy, 2026-10-18).

Any machine can receive any request and act as its coordinator (Cassandra style). Clients prefer a coordinator with
separate host memory (`b70host` or a CPU rig) to save unified memory bandwidth on `gfx1151host` and `sm121host` (3.2). The
coordinator finds the target engine from the request's `model` field and the `chitchat` cluster state (2.3). Because
the coordinator has the family libraries (11.3) and the engine's `identity()` from gossip, it can render, tokenize
and compute the recipe locally. It then schedules the job locally or forwards the work to the machine that holds the
engine.

The API has contexts, specs, decisions, snapshots, pins, reads, and generations with riding decisions. In council v2
responses, a snapshot `id` is its BLAKE3 content id (`blake3:<hex>`), and the store looks up prefixes by recipe
(6.6). gogi sets these request rules:

- Each request names a served model (ontology) in `model`. A request without it is a 400.
- `identity.models` lists every served model in the cluster, with its status from gossip (2.3).
- The statuses are `ready`, `loading`, `draining` and `unloaded`. A fatal engine's served models report `unloaded`.
- A served model that is not `ready` gets 503 `model_unavailable`.
- `class` is `interactive` (the default) or `batch` (5.3). `priority` is an `i32`, 0 by default (5.4).
- A request carries no tolerance. Its served model sets the tolerance (6.6).

Pooling and the decision numbers come from `gogi_proto::math`, which all engines share. The council
`identity.engine` covers two things: the engine identity, and the nagaya code that turns logits into decisions.
(mk: `libmkengine:<sha>+numbers:<sha>`.)

**11.2 Generation.** The nagaya runs one sampler for all engines. It applies top_k, top_p, min_p, then temperature.
The draw is seeded, and the response reports the seed. Each engine has its own defaults.

All engines share the SSE framing. Speculative drafts change how the sampler uses the seed (mkviolet's
prompt-lookup drafts). Thus a request says if drafts are on, and the reported seed includes that setting.

**11.3 Text front-end.** Each model family has its own text rules. They are a tokenizer, a template, escapes for
special tokens in client text, and parsers for tool calls and thinking. The nagaya holds one library per family
(ontology "Family"), and the engine identity selects it by its `tokenizer_hash` and `template`. Engines never see
text. They take tokens, rows, or both (D2).

Decided 2026-10-17 (Amy: "oh, yes, b for sure, plus that can support the tokenize apis nicely for when we bring back
kaijutsu's semantic search and stuff"). Thus one renderer serves every engine of a family, on any vendor's card.
The rendering rules in gogi, such as where a boundary exists and where thinking renders, get one implementation and
one set of conformance tests. mk's C ABI does not need a tokenizer.

**11.4 `/mk/v1`.** kaijutsu's mk provider uses `model`, `render` and `generate` (mkviolet `mk_api.rs`).

Decided 2026-10-17: the nagaya does not serve `/mk/v1`. kaijutsu moves to council v2 and uses the gogi reference
(`~/src/gogi`). Amy: "kaijutsu will move to council v2 and use the new gogi (~/src/gogi) reference. it's happening
now, on b70host". In council v2 the server renders messages with its model's own template, and the template is
part of identity (gogi `docs/council-api-v2.md`, "Terms").

**11.4a Extensions.** gogi puts model-specific routes under the server's own prefix, listed in
`identity.extensions` (gogi `docs/council-api-v2.md`, "Extensions"). The nagaya's prefix is `/nagaya/v1`:

- `profile` and `cluster`: dynamic placement profiles and the `chitchat` cluster view (2.3, 2.4);
- `cas/{content_id}`: plain HTTP `GET` and `HEAD` for content-addressed snapshots and tap artifacts across machines;
- `render`, `tokenize` and `detokenize`: served by the family library for bench tools and semantic search;
- `embeddings`: served by the engine through a pass with a hidden-state output;
- `status` (9.7) and `traces` (13.1).

**11.5 Errors.** All routes use gogi's error body, `{"error": {"type", "message", "param"?}, "request_id"}`, and the
`request-id` header. Council routes use only gogi's error table (gogi `docs/council-api-v2.md`, "Errors"). The
`/nagaya/v1` routes add two codes:

| Code | `error.type` | When |
|---|---|---|
| 409 | `conflict` | a profile `PUT` with an old version (2.3) |
| 507 | `no_memory` | a profile that does not fit its budgets (2.4) |

An unexpected failure is a 500 `internal`, and the process stops (8.5).

**11.6 Binding.** The HTTP and gossip listen addresses and seed peers are in the bootstrap config (2.1). (mkviolet
has a tailnet address in its code.)

## 12. Identity

**12.1 Fields.** An engine identity has the fields of `identity()` in the ABI:

- `model`;
- `checkpoint_digest`, the first of the two weight ids (12.2);
- `tokenizer_hash`, the hash of the tokenizer file that the engine loads;
- `template`;
- `numbers`: the engine binary, and the nagaya code that turns its outputs into answers;
- the lineage of the plan table;
- the bit settings, each with its class (`docs/prior-art.md` 1);
- the canary hash.

The nagaya keeps each engine's expected identity in the store's index. At start, it compares the identity from
`identity()` with the expected one. A mismatch makes the old entries of that engine unusable. It is a refusal to
start only when the profile pins the identity.

**12.2 Two weight ids.** The checkpoint digest is logical. It hashes the canonical config and each tensor's payload,
so a new packing does not change it. The arena id is the content id of the packed bytes that the engine runs.
`arena_export` returns it.

The identity carries the checkpoint digest. The arena records its own id and the strength of its check. mk's
`weight_hash` and mkviolet's `header_hash` hash only the shard headers and sizes. They are spot checks, and the
identity labels them as spot checks until we replace them.

**12.3 Identity defects to correct in the port.**

- mkviolet loads EmbeddingGemma 2's `tokenizer.json` for Gemma 4. The file "differs ... by a few hundred bytes;
  which entries differ is not yet checked" (`tokenizer.rs`). Thus its tokenizer hash names the wrong file.
- mkviolet's metrics label card 1 `google/embeddinggemma-2`.

## 13. Taps, lenses and composition

**13.1 Taps.** An engine declares its taps in `limits()`, with the conditions that each tap needs (mk: unfused, one
segment). A tap gives one of these outputs: a residual at a layer (rows), routes, or gates. The store keeps large
tap outputs as content-addressed artifacts, with an owner and a budget. (mk keeps traces as per-layer `.npy` files.)

**13.2 Lenses.** A lens (mkviolet J-Lens) is a state object of its own. It belongs to one model, has a content id,
and runs on a CPU device (2.1). Two models cannot share a lens. Models are compared in the lens's concept
coordinates, not in raw residuals.

**13.3 Composition between models.** kaijutsu composes models through the council API, for example a judge that
follows a coder. The nagaya keeps the engines loaded side by side and schedules their classes. Tensor transfer
between engines is out of v0. (mkviolet `docs/02` designs KV bridges and residual splices.) If it comes, these
rules apply:

- it stays on one machine;
- it uses an `inject` capability;
- each adapter is a state object with its own identity.

## 14. Porting notes

The inventories found these items. We correct each one when its code moves. We do not copy them.

- mk splits a prefill in its Python service (`contexts._feed`). The split moves under `max_quantum_ms` (5.1).
- mk orders jobs first in, first out in its one high tier. The nagaya uses the order in 5.4.
- mk's "park" writes through to disk and seals at each "hold". In nagaya terms, a hold is a declared boundary, and
  a write to disk is a park to NVMe (D7).
- mk's MemAvailable floor becomes the safety floor in 4.7. It does not set any size.
- mkviolet's three caches merge into the store (6.1). Its fixed budgets become ledger terms.
- mkviolet's telemetry setup leaks a runtime (`Box::leak`). Its HTTP server is hand-written, and its bind address is
  in its code.
- mkviolet has instruments without call sites (9.5) and wrong model labels (12.3).
- mkviolet's README gives `GEN_SEQ` 16384 and its `main.rs` gives 65536. Its `docs/07` is out of date.
- mkviolet's determinism class is not known. The first port runs one recipe in two processes and compares the
  content ids (engine-abi, "How mkviolet maps"). The rule that our engines are `always` or `pinned` (README) is a
  requirement on the port. The test shows if mkviolet meets it.

## 15. Open questions

1. Do an idle engine's GPU queues slow the other engines on its device (3.3)? A measurement answers this.
2. What makes `close()` safe for mk, so that engines can load and unload at runtime (3.3)?
3. How long does the conversion to canonical safetensors take on mk and mkviolet (10.3, D10)?
