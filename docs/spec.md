# nagaya spec v0 (draft)

This spec says what the nagaya is and what it does, one rule at a time. It is written in Simplified Technical
English (`CLAUDE.md`).

- `docs/ontology.md` defines each term once.
- `docs/engine-abi.md` defines the calls and the decisions D1-D9. The spec refers to them by number.
- `docs/prior-art.md` and `docs/reviews.md` record where the rules came from.

When this spec and another doc do not agree, correct one of them. Neither one wins without a decision.

Status: draft, 2026-10-17. The sources are the design conversation with Amy and inventories of mk
(`~/src/megakernel-qwen38-flashnext-strixhalo`) and mkviolet (`~/src/mkviolet`). There is no code yet. Each open
question has the mark **Open**, and section 15 lists all of them.

## 1. Scope

The nagaya is one process on each machine. It owns the machine's GPUs, and engines run inside it. It decides three
things: where engines and state live, what runs next on each device, and what the store keeps. It does not decide
what a client asks a model.

| Layer | Owns | Does not own |
|---|---|---|
| kaijutsu (the agent) | coordination: casts, seats, judges, bumps, which model answers what | devices, memory, scheduling |
| nagaya (the host) | placement, the ledger, scheduling, the store, IO, machine telemetry, lifecycle, the client API | what a request means; kernels |
| engine | a model's code and weights, its kernels, its private layouts and caches | anything that another engine uses |

Amy, 2026-10-17: "Coordination belongs in kaijutsu, yeah. It's the agent."

Goals for v0:

- Several engines run on one machine at the same time: inference endpoints with different models, image batch, and
  training as filler.
- Each engine gives the same bits for the same identity and inputs.
- Nothing is lost or corrupted without a report.

Not in v0:

- One model split across machines. The links are gigabit today (README, "Between machines").
- Engines that load and unload while the process runs (3.3).
- Cooperative yields inside a pass (D8).
- Tensor transfer between engines (13.3).

## 2. Machines and placement

**2.1 Inventory.** A static file names each machine, its devices, its tiers and the measured edges between tiers.

| Machine | Devices | Memory |
|---|---|---|
| gfx1151host | Strix Halo, one gfx1151 GPU | 128 GB unified |
| b70host | two Intel Arc Pro B70, a BlueField-2 | 64 GB GDDR6, 128 GB DDR5, 4 TB NVMe |
| sm121host | DGX Spark, one Blackwell GPU | 128 GB unified |

Telemetry takes hardware attributes from the inventory, not from constants in code. (mkviolet puts B70 values in
`telemetry.rs` as constants.)

**2.2 Tier edges.** An edge records more than a bandwidth (`docs/prior-art.md` 5):

- the rate for each read size and queue depth;
- whether direct IO works into the destination;
- the alignment that the edge needs.

The nagaya measures edges at install and can measure them again. Each measurement is data with a date.

**2.3 Placement profile.** Each machine has one profile. It names the engines that load, their devices, their
config, and each engine's weight for spare memory (D9). The nagaya reads the profile at start. If the profile does
not fit the ledger (section 4), the nagaya refuses it before it reads a weight. The refusal gives the shortfall in
bytes for each tier.

In v0, each machine has one profile. A later controller can change profiles between day and night (README,
"Later").

## 3. Process

**3.1 One process per machine.** The nagaya owns every GPU on the machine (Amy: "we'd do 1 process per host that
owns the gpus"). The nagaya is the only GPU client. A tool that needs a device sends its work through the nagaya, or
stops the nagaya first. (ds4-recipe: a shader fault with a second GPU client made the SMU stop responding.)

**3.2 One thread per device.** Each device has one thread. Every engine call for that device comes from that
thread, and the call is synchronous. A pass blocks the thread. mkviolet's rule "one `ze_context_handle_t` per card,
never share across cards" is an example. Only the device thread writes its device's part of the ledger. Other
threads send requests to it.

**3.3 Engines live as long as the process (v0).** The nagaya opens each engine at start, from the profile. It
closes the engines at exit. mk cannot close inside a process today: "the HIP runtime's own teardown segfaults" (mk
`service/__main__.py`). Thus the nagaya calls `close()` only at exit, and it can exit without the call. Engines can
load and unload at runtime (README door 1) only when each engine's `close()` is safe.

**Open:** Do the GPU queues of an idle engine slow the other engines on its device? mk has a note that its idle
queues slow another process, but we did not find the source. A measurement answers the question.

## 4. Ledger

The ledger is the machine's account of memory (ontology "Ledger", D9, ds4-recipe `05-memory-ledger.md`).

**4.1 Budgets.** Each tier has one budget. A budget is a byte target in the config. It is not a measurement of free
memory.

- gfx1151host and sm121host: one shared pool for device and host memory.
- b70host: one budget for each card's device memory, and one for host memory.
- Every machine: one budget for the store on NVMe.

**4.2 Terms.** Each engine declares its terms in `limits().memory`:

- mandatory terms `{name, bytes, lifetime}`. The lifetime is per engine, per context, or loaned (one holder at a
  time);
- elastic terms `{name, unit_bytes, min_units, max_units}`;
- a key by kind of quantum, where the bytes change by phase. A prefill's scratch is not a decode's scratch.

The nagaya's own memory (staging, store buffers, IO) is in terms too. The ledger charges every allocation. Bytes
without a charge are a defect. (ds4-recipe: vision weights without a charge "show up as untracked bytes".)

**4.3 Admission.** The ledger admits each call that allocates before the call runs. These calls are `open`,
`ctx_new`, `ctx_fork`, `part_write`, `slot_load`, and each resize. The check includes transient peaks. A refusal
charges nothing and changes nothing. The job that waits keeps its place.

**4.4 Spare memory.** The ledger first charges all mandatory terms. It then divides what remains among the elastic
terms by the profile's weights (D9). Each term gets at least its `min_units`. Counts come from this arithmetic, never
from free memory. Some engines have bits that depend on their cache size. Such an engine must get the same size at
each start.

**4.5 Resize.** An elastic term changes size only between quanta, through `resize(term, units)`. It does not shrink
while a lease or a pin refers to what would go.

**4.6 Reclaim.** When admission needs bytes, the ledger tries these steps in this order:

1. Park idle contexts that have no pin to a lower tier, the oldest first (D7).
2. Shrink the elastic terms of idle engines to their floors.
3. Refuse the request.

Reclaim never seals. It never discards state that has no other copy.

**4.7 Host safety floor.** mk keeps 12 GiB of MemAvailable free. In the nagaya, such a floor is a safety check
for what the ledger cannot see: other processes, tmpfs and swap. It does not set any size. When free memory goes
below the floor, the ledger refuses new admissions, resizes nothing, and reports the event.

On machines with unified memory, RSS and the OOM scorer do not show GTT. Thus the nagaya sets its own OOM score.

## 5. Scheduling

**5.1 Units.** A job is the quanta of one request, in order, with a class (ontology). A quantum is one pass or one
stage. It is not longer than the engine's `max_quantum_ms` (D8), and it runs to its end. The nagaya chooses what
runs next only between quanta. If an engine's natural pass is longer than its bound, the engine splits the pass.
(mk splits a prefill into 256-token chunks in its Python service, and that split moves into mk's engine wrapper.)

**5.2 Kinds.** Each quantum has a kind: `read`, `prefill`, `decode`, `embed`, `copy`, `step` (an image step), or
`train` (a micro-batch). Metrics and the cost model use kinds. A kind does not set urgency (mk `mk.gpu.kind`).

**5.3 Classes.** The classes are `interactive`, `batch` and `filler`. Between quanta, a ready quantum of a higher
class always goes first. A council decision is interactive. A job's class starts to apply at its first quantum, not
when the job arrives. mkviolet's `lazy_high` takes the card at the council's first GPU step. That change cut
decisions beside a prefill from 383-702 ms to 300-390 ms.

**5.4 Order in a class.** The scheduler orders the jobs in a class by three keys:

1. integer priority, 0 by default;
2. owed seconds;
3. time of arrival.

Owed seconds is the device time that a job is owed under equal sharing, clamped to ±30 s. A job that leaves takes
only its own balance (ds4-recipe `11-multi-session-serving.md` section 6). Filler jobs do not collect a debt.

**5.5 Filler.** A filler job declares a minimum runtime: the time between its checkpoints. It must be resumable:
its state parks when it gives way (Ai2's "unallocated" time). It runs only when no interactive or batch quantum is
ready on its device.

**5.6 No preemption.** Nothing interrupts a quantum that runs. Priority orders the work that waits. Priority never
moves a continuation off the context that holds it (a pin).

**5.7 Segments.** Quanta from several jobs on one engine can share a pass as segments. This is permitted only when
the engine declares `batch_exact` (`docs/prior-art.md` 1). Each class has a join window. The scheduler waits that
long for jobs that are ready, and never for jobs that are not ready.

**5.8 Deadlines.** A job can have a deadline (council v1 `timeout_ms`). The scheduler checks it between quanta. A
job after its deadline ends with a timeout, and its segments do not run again.

**5.9 Cancellation.** A client cancels one job. The scheduler drops the job's next segments, and the quantum that
runs goes to its end. Cancellation is not failure: a cancelled job keeps its source snapshots. (ds4-recipe: the
loss of one such snapshot cost a cold prefill of 35 minutes.) Every terminal state of a job releases its loans and
leases. The terminal states are done, cancelled, failed and expired.

**5.10 Admission at the door.** The client API admits a limited number of requests in flight for each engine (mk:
16). Above that limit it answers 429 with `Retry-After`.

**5.11 Cost model (D4, proposed).** The nagaya fits a robust affine cost from production quanta. The fit has a key
for each engine, kind and segment count, and it uses context depth. Quanta that ran beside other work are not in the
fit. Before a fit exists, the nagaya records "no model yet" and schedules by class and arrival. An engine can have
its own private planner, which stays in the engine. (mkviolet's council planner uses 39 ms + 0.21 ms per token,
`council_server.rs`.)

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
- the tags of each approximation.

mk's `context_id` and mkviolet's `snap:<sha256>` are recipes in this sense. mk's id is a SHA-256 of the tokens and
boundaries with `|gdn-bf16-v1|`.

**6.3 Sealed bytes are canonical.** Padding and unused rows are zero. Times, hit counts, owners and reasons go in
the store's index, never in the sealed bytes. If they go in the sealed bytes, two copies of one state get different
content ids, and dedup fails.

**6.4 Entries.** Each entry in the index records:

- the content id, the recipe, the parent, and the coverage;
- the tier, the size, and the owner;
- the times of creation and last use;
- the reason for the store: anchor, interval, prefill done, evict, reclaim, or shutdown;
- the provenance of an approximate entry;
- the strength of its last check.

**6.5 Chains.** Each engine declares a maximum chain depth and a size fraction. (ds4-recipe: depth 1, and a delta
only when it is at most a quarter of a full snapshot.) The store never evicts a base while a delta needs it.

**6.6 Lookup.** A lookup is by recipe. The store keeps only the entries that fit the slot (coverage) and the
request's tolerance (ontology). It picks the deepest exact match. It gives an approximate match only to a request
whose tolerance names that method.

**6.7 The tripwire.** Some engines declare `always` or `pinned` (D3). For those engines, two content ids under one
recipe with the same coverage are a defect. The store refuses that recipe and reports it. For other engines, the
audit job (8.4) does the check.

**6.8 Restore.** The nagaya checks the content id, the identity and the fit before it calls the engine.
`slot_load` and `part_write` return one of three results:

- ok;
- refused, and nothing changed;
- failed after a change. The state is poisoned (D6).

When a restore fails, the store quarantines the entry with the reason. A failure never deletes content.

**6.9 Eviction.** The store chooses victims in this order:

1. Pinned entries are exempt.
2. Thin the chains: interiors go before tips and roots.
3. Remove whole entries by value per byte: recency, hits, and the cost to build them again.

The store chooses all victims before it deletes one. It also chooses them before it writes the new entry. Each
owner has a quota, so one long conversation cannot remove the others (a ds4-recipe incident). If a pin alone is
larger than a budget, the pin loses its protection, so eviction cannot deadlock.

**6.10 Pins.** A pin is a lease with an owner and an expiry (mkviolet v2: one day at most). A request pin holds a
snapshot from its lookup to the end of its load. Pins stay through a restart (mkviolet `pins.json`).

**6.11 Restart.** At start, the store loads its index again and checks entries when they are used. It wakes pinned
entries into slots when the profile asks for them. (mk wakes its pinned parked contexts before `READY=1`.)

**6.12 Steering and lab inputs.** Some state changes bits but is not an input, for example mk's attention cell mask
or a steering vector. Such state goes into the recipe. Thus a steered or masked result never has a plain recipe.

## 7. IO

**7.1 One IO service per machine.** Engines and the store send batches `(file, offset, len, dest)` with a class to
the IO service. The service owns these jobs:

- the rate on each edge and the queue depth;
- hedged reads across copies on different drives;
- the loss of a drive, and its return.

A batch holds a lease on its destination until its final wait ends. This is also true after an error.

**7.2 Direct and staged bytes.** The service counts direct bytes and staged bytes separately. Thus a change from
direct IO to staged IO is visible.

**7.3 Replicas.** A store on several NVMe drives names each copy by content id. "Same size is not same bytes". A
copy that has no verification gives no reads.

**7.4 Watchdog.** A drive that hangs can hold a quantum until the kernel times out (about 30 s in ds4-recipe).
Thus each device thread has a watchdog. It reports a quantum that is past `max_quantum_ms` by a margin.

## 8. Correctness at runtime

**8.1 Canary at start.** Each engine records a canary: fixed inputs and the exact outputs that they gave. mk's
canary has pilot reads and an 8-step greedy decode (mk `service/canary.py`). It keeps letter logits as f32 bit
patterns, the top 5, and a SHA-256 of all logits. The nagaya runs the canary again before the engine is ready, with
no tolerance. A mismatch refuses the engine, and the process exits with the code for "do not restart" (10.4). The
canary's hash is part of the engine identity.

**8.2 Pass failures.** A pass returns ok, poison (D6), or timeout. After a timeout, the nagaya does these steps:

1. Fail the jobs in that pass and free their contexts.
2. Run the canary again on the device thread, before any other work.
3. If the canary does not match, or if this is the fourth timeout in 600 s, mark the engine fatal. Refuse every
   later job for it.

(mk `service/gpu.py` does this.) Outputs that are not finite are poison.

**8.3 Conformance.** Each engine has a conformance suite in the nagaya's tests. It checks these properties:

- park and restore keep every bit;
- seal runs only at boundaries;
- the quantum conditions hold (ontology "Quantum");
- an admission refusal changes nothing.

Equality is by integer bit pattern. NaN or Inf fails, also when the two values are equal. Counts must also match.
Each gate keeps a broken variant that it must fail (ds4-recipe `01-correctness-and-measurement.md`).

**8.4 Audit.** For some engines the tripwire is not reliable, because of their determinism class. For them, a
filler job builds a stored recipe again from time to time and compares the content ids (D3). It runs in a fresh
process under a pinned plan table.

**8.5 Crash over corruption.** An unexpected failure in the nagaya stops the process (exit 1). The process does not
continue in a bad state.

## 9. Observability

The nagaya follows kaijutsu's conventions (`~/src/kaijutsu/docs/telemetry.md`). mkviolet already follows them.

**9.1 Activation.** The nagaya always logs to stderr. It also exports in these cases:

- OTLP over gRPC to the host collector, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set;
- OTLP JSON-lines files, when `NAGAYA_OTEL_FILE_DIR` or `KAIJUTSU_OTEL_FILE_DIR` is set.

`OTEL_SDK_DISABLED=true` stops all export. The `service.name` is `nagaya`. The resource attributes carry
`host.name` and the hardware from the inventory.

**9.2 Traces.** When a request has a W3C `traceparent` and `tracestate`, its span is a child of the caller's span.
An invalid header is an attribute, not an error. The spans nest in this order:

1. the HTTP server span, `<METHOD> <route template>`;
2. `nagaya.job`, with engine, class and request kind;
3. `nagaya.quantum` for each quantum, with engine, device, kind, class, segments, and wait and run times in ms.

Decode quanta make metrics only: "one span per generated token would bury a generation's trace" (mk). The trace
context goes to the device thread with the job. Today, neither engine carries it onto GPU work. Engines add their
own spans under the quantum through the telemetry capability.

**9.3 Sampling.** The nagaya uses kaijutsu's rule. It always keeps errors and spans whose parent is sampled. It
keeps 100% of the spans in some dot-qualified prefixes and 10% of the others. `nagaya.job.`, `nagaya.quantum.` and
`council.` are 100% prefixes. Decode quanta make no spans, so the volume stays low.

**9.4 Metrics.** Metric attributes have a bounded set of values. Context, trace and job ids are never attributes.

| Metric | Unit | Attributes |
|---|---|---|
| `nagaya.quantum.wait.duration`, `nagaya.quantum.run.duration` | ms | engine, device, kind, class |
| `nagaya.quantum.tokens` | {token} | engine, kind |
| `nagaya.ledger.bytes` | By | tier, engine, term |
| `nagaya.ledger.refusals` | {refusal} | tier, engine, gate (budget, floor, resize) |
| `nagaya.store.bytes`, `nagaya.store.entries` | By, {entry} | tier, engine |
| `nagaya.store.lookups` | {lookup} | engine, outcome (exact, approx, miss, refused) |
| `nagaya.store.evictions` | {entry} | tier, reason |
| `nagaya.io.bytes` | By | edge, path (direct, staged), class |
| `nagaya.arena.handoff` | {handoff} | engine, status (hot, cold) |
| `nagaya.engine.state` | 1 | engine, state (loading, ready, fatal) |
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
| `nagaya.device.throttle` | counter | device, reason |
| `nagaya.device.evicted` | ms | device (amdgpu KFD only) |

mk's `mk.gpu.*` names and mkviolet's `mkviolet.*` names map onto these names. Council decision metrics (count,
spread, confidence) belong to the council layer (11.1).

**9.5 Each instrument has a call site under test.** mkviolet defines `record_gpu_pass` but calls it only from its
smoke test. It calls `record_council_decision` only in tests. Its docs name instruments for routing entropy and
J-Lens drift that do not exist. Thus a test shows that each instrument records on the path that it describes.

**9.6 Evidence on failure.** Before and after a quantum that fails, the nagaya reads the machine's counters. It logs
the change with the failure. mk reads these counters: KFD evicted ms, MemAvailable, GTT, and from `/proc/vmstat`
compaction stalls, migrations, swap-outs and direct scans.

**9.7 Status stream.** Each machine has one SSE status stream (mk `/mk/v1/status`, mkviolet `PassBus`):

- a snapshot when a client connects;
- one `quantum` event for each quantum, with engine, device, kind, ms, and queue length by class;
- a keepalive every 15 s.

The stream disconnects a slow subscriber. It does not skip events for it.

## 10. Lifecycle

**10.1 systemd.** The unit is `Type=notify`. The process sends `READY=1` when every engine in the profile has
opened, passed its canary, and woken its pinned state. `STATUS=` gives the state of each engine.

**10.2 Arenas through a restart.** systemd's file-descriptor store keeps the arenas (`FDSTORE=1`). Each fd has the
name `<engine>/<device>/<part>`. The nagaya derives `FileDescriptorStoreMax` from the profile. (mkviolet keeps 61
fds under a fixed limit of 64.) At start, the nagaya does these steps:

1. Check each stored arena against the engine's build, and record the strength of the check.
2. If an arena does not match, drop that engine's stored fds and load it cold.
3. Duplicate each fd before use. (In mkviolet on 2026-10-16, an import closed the stored fd and caused a crash
   loop.)

`systemctl stop` clears the store. A restart keeps it.

**10.3 Stop.** On SIGTERM, the nagaya does these steps:

1. Accept no new jobs.
2. Let each running quantum go to its end.
3. Park pinned and wanted contexts to NVMe in a time budget (ds4-recipe: 10 s).
4. Exit.

A second signal makes the process exit at once. Cancellation of running work follows 5.9.

**10.4 Exit codes.**

| Code | Meaning | systemd |
|---|---|---|
| 0 | a clean stop | no restart |
| 1 | a crash | restart |
| 3 | a refusal to start: a canary mismatch, an identity mismatch, or a profile that does not fit | `RestartPreventExitStatus=3` |

The restart policy follows mk's unit: `Restart=on-failure`, `RestartSec=30`, and at most 3 starts in 600 s.

## 11. Client API

**11.1 Council.** The nagaya serves council v1 and v2 for all its engines (`~/src/kaijutsu/docs/council-api.md`,
types in `gogi-proto`). It routes each request by its `model` field. The API has contexts, specs, decisions,
snapshots, pins, reads, and generations with riding decisions (mkviolet `council_v2.rs`). Pooling and the decision
numbers come from `gogi_proto::math`, which all engines share. The council `identity.engine` covers two things: the
engine identity, and the host code that turns logits into decisions. (mk: `libmkengine:<sha>+numbers:<sha>`.)

**11.2 Generation.** The host runs one sampler for all engines. It applies top_k, top_p, min_p, then temperature.
The draw is seeded, and the response reports the seed. Each engine has its own defaults.

All engines share the SSE framing. Speculative drafts change how the sampler uses the seed (mkviolet's
prompt-lookup drafts). Thus a request says if drafts are on, and the reported seed includes that setting.

**11.3 Text front-end.** Each model family has its own tokenizer, template, escapes for special tokens in client
text, and tool-call parser.

**Open:** Is the text front-end an ABI capability (`text`) that each engine exports? Or is it a host library that
the engine identity selects?

**11.4 `/mk/v1`.** kaijutsu's mk provider uses `model`, `render` and `generate` (mkviolet `mk_api.rs`).

**Open:** Does the nagaya keep `/mk/v1` as an API for each engine? Or does kaijutsu move to council v2 generations,
so that `/mk/v1` can go?

**11.5 Errors.** All routes use one error shape:

| Code | Meaning |
|---|---|
| 400 | a bad request; a field that the schema does not know is a 400 |
| 404 | not found |
| 409 | a conflict |
| 413 | too large |
| 429 | busy, with `Retry-After` |
| 503 | unavailable: the engine is fatal, or the deadline passed |
| 507 | not enough memory: the ledger refused |

An unexpected failure is a 500, and the process stops (8.5).

**11.6 Binding.** The listen address is in the config. (mkviolet has a tailnet address in its code.)

## 12. Identity

**12.1 Fields.** An engine identity has these fields:

- the model;
- the weights, as two ids (12.2);
- the hash of the tokenizer file that the engine loads;
- the template;
- the numbers: the engine binary, and the host code that turns its outputs into answers;
- the lineage of the plan table;
- the bit settings, each with its class (`docs/prior-art.md` 1);
- the canary hash.

**12.2 Two weight ids.** The checkpoint digest is logical. It hashes the canonical config and each tensor's payload,
so a new packing does not change it. The arena id is the content id of the packed bytes that the engine runs.

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
and runs on a CPU device. Two models cannot share a lens. Models are compared in the lens's concept coordinates, not
in raw residuals.

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
- mk's "park" writes through to disk and seals at each hold. In nagaya terms, a hold is a declared boundary, and a
  write to disk is a park to NVMe (D7).
- mk's MemAvailable floor becomes the safety floor in 4.7. It does not set any size.
- mkviolet's three caches merge into the store (6.1). Its fixed budgets become ledger terms.
- mkviolet's telemetry setup leaks a runtime (`Box::leak`). Its HTTP server is hand-written, and its bind address is
  in its code.
- mkviolet has instruments without call sites (9.5) and wrong model labels (12.3).
- mkviolet's README gives `GEN_SEQ` 16384 and its `main.rs` gives 65536. Its `docs/07` is out of date.
- mkviolet's determinism class is not known. The first port runs one recipe in two processes and compares the
  content ids (engine-abi, "How mkviolet maps").

## 15. Open questions

1. Does the nagaya keep `/mk/v1`, or does it go in favor of council v2 generations (11.4)?
2. Is the text front-end an ABI capability or a host library (11.3)?
3. Do an idle engine's GPU queues slow the other engines on its device (3.3)? A measurement answers this.
4. What makes `close()` safe for mk, so that engines can load and unload at runtime (3.3)?
5. D4-D7 are not yet reviewed with Amy (engine-abi, "Decisions").
6. Do kinds of work get fair shares over a window, as Ai2's teams do? Or do strict classes do the job (5.3)?
