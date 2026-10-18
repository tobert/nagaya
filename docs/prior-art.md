# Prior art

What other systems teach the nagaya, and what we take from each. Items are proposals until Amy decides them; each
says where it would land (ontology term, ABI call, scheduler or store rule, decision).

## ds4-recipe (`~/src/ds4-recipe`)

A recipe for a fast DeepSeek V4.1 Flash runtime on one Strix Halo (128 GB unified memory, three NVMe drives): 278 GB
of experts streamed from NVMe, 4,667 of 15,360 routed experts resident, 98.75-99.51% of prompt tokens served from a
disk KV cache, about 14.5 tok/s decode with DSpark. It is one engine on one machine, so it has no cross-engine
answers, but nearly every host-level question we have, it has met once in production.

Read 2026-10-17 by four readers (chapters 01, 03, 04, 05, 06, 10, 11, 12, 13; the kernel chapters 02, 07, 08, 09,
14, 15 skipped as engine-internal). Load-bearing claims were checked against the text.

### 1. Bits: what changes them, and who says so

- **Exactness tags** (README "Conventions", 01): `[exact]`, `[reference-defining]` (a free choice of evaluation order
  made once; "once chosen it *is* the reference"), `[approx — approved]`, `[policy]`. Proposed: each bit setting in
  `identity()` becomes `{name, value, class}`; reference-defining settings are identity, approved approximations
  are recipe inputs and mark the snapshot non-exact, policy settings are neither unless declared output-relevant.
- **Same binary, different bits per process.** Timed GEMM plan selection is "reference-defining per process": 32
  processes of one binary at λ = 0 "picked 28 distinct plan sets, and their dumps fell into 13 groups" (01). This
  breaks D3's refusal rule (two content ids under one recipe with the same coverage) on every restart. Proposed:
  `limits()` declares `determinism: always | pinned | per-process`; the refusal applies to `always` and `pinned`; a
  `per-process` engine puts its plan-set hash in the identity. Our engines should be `pinned`.
- **History changes bits.** With cache-conditional routing, "Every eviction, residency and sizing decision is
  `[policy]`, but it is **output-relevant**" (06), and "execution order changes expert-cache history". Residency
  state is read by every pass but lives in no state object and is never parked; a restore starts cold, so "restore
  is not resume". Proposed: quantum condition 7, "engine history that feeds arithmetic is identical"; an engine
  declares `bit_dependence: residency` in `limits()`, and the nagaya then makes no interleave-identity claim for it
  and refuses no recipe over it. Policy that never changes bits holds only for engines that do not declare it.
- **Pass shape changes bits.** Changing GEMM N changes the library's algorithm (01); prefill call boundaries matter;
  ds4 says "Never choose the job size independently of the engine cohort" (11). Our Segment entry claims each
  segment gets the bits of a pass of it alone. Proposed: `limits()` declares `batch_exact` and whether quantum split
  points change bits; if they do, split points join the boundaries in the recipe.
- **Logical digest, not layout digest** (03 §12, Pitfall 16): re-encoding the weights bit-identically must not change
  identity ("Keyed on the layout digest, every checkpoint would have been refused"). Proposed: `weights_hash` is a
  logical digest (canonical config plus shard payload hashes); the arena's layout hash is recorded separately. This
  amends our "the weights hash is the content id of the weights state object".
- **Equality in tests** (01 "What exact means"): compare integer bit patterns, NaN or Inf fails even when equal,
  counters must match too, report the first differing row; keep a deliberately broken variant that the gate must
  fail. For the nagaya's engine conformance and park/seal tests. Non-finite outputs from `pass` poison (D6).

### 2. Memory: one ledger per machine

- **One byte target, mandatory first, elastic as the remainder** (05 "1. One byte target"). Proposed: a Ledger noun
  under Holding, one budget per tier (one shared pool on gfx1151host and sm121host, two on b70host). Each engine declares
  terms in `limits().memory`: mandatory `{name, bytes, lifetime}` (per engine, per context, or loaned to one holder
  at a time) and elastic `{name, unit_bytes, min_units, max_units}`, keyed by phase where bytes differ by phase.
- **Counts from arithmetic, never from free memory** (05 "2.", 06 technique 12): "the slot count changes routes", so
  a restart must size identically. Free memory only enforces a safety floor.
- **Reserve before allocate**, transient peaks included (05 invariant 4); a refused admission changes nothing
  (invariant 7); the worst phase is checked at launch before reading any weight ("Fail at launch, not at hour 9").
  Proposed: every allocating call is admitted by the nagaya first; `open` reports minimum bytes per tier first.
- **Reclaim parks**, in a fixed order (05 "7.", 11 "11."): idle unpinned contexts, oldest first, parked to NVMe;
  then idle engines' elastic terms; then refuse. Agrees with D7.
- **Resize only between quanta** (05: "refuse while leases, pins, cohort references ... exist"). Proposed: a
  `resize(term, units)` capability, called only between quanta.
- **Charge everything.** Uncharged vision weights and a capture pool "show up as untracked bytes"; co-tenant
  processes (tmpfs, swap) caused ds4's OOM. The nagaya sets its own OOM score.
- Open: when two engines both have elastic terms, who gets the remainder? ds4 has one elastic consumer. Proposed: a
  weight or floor per engine in the placement profile, not a heuristic.

### 3. Scheduling

- **No preemption of a running request** ("No active request is preempted"). Agrees with quanta. Priority orders
  waiting work only and never moves a continuation off its required context.
- **Head-of-line blocking is the cost.** A 201-row turn waited 283.95 s behind a 112,257-row prefill of 32k-row jobs
  (11). ds4's fix is cooperative yields inside a pass and nested guests in the yield. Proposed for v0: no yields; a
  `max_quantum_ms` in `limits()`, and an engine whose natural pass exceeds it splits into shorter quanta or (later)
  stages. Nested guests are deferred; their "refused, nothing changed, the waiter keeps its place" protocol is not.
- **Owed-share fairness within a class** (11 "6."): order by priority, then owed seconds (clamped ±30 s), then
  arrival; "Leaving members take only their own balance". Time-based, not token-based. Fillers accrue no debt.
- **Gathering segments**: wait a bounded join window for ready jobs, never for unready ones ("Nothing waits on a
  member that is not about to post"). A per-class scheduler parameter.
- **Cancel is per job, between quanta; cancel is not failure.** A cancelled prefill that deleted its checkpoint made
  the retry prefill 442,546 tokens cold (about 35 minutes). Only poison discards state. Draining a lease gets a time
  budget (ds4: 10 s to persist resident sessions).
- **Loans are leases the nagaya enforces.** A stranded prefill-scratch loan made every other prefill fail behind
  it. Every terminal state of a job (done, cancelled, failed, expired) releases its loans.
- **Cost model (D4)**: fit robust affine costs per engine and segment count (Theil-Sen; a median fit stalled
  admission), keyed by context depth; exclude passes that ran beside other work; fit from production passes with
  telemetry on (profiled and observed costs differed 2x). Cold start: record "no model yet" and schedule by class
  and arrival until a fit exists; never guess silently.
- **Phase transitions are a cost term.** ds4's prefill-to-decode transition (draft reload, regrow) was 1.594 s, the
  largest fixed cost for short turns. A co-resident loadable piece (a draft model, an image model) declares its
  bytes and reload time so the scheduler can price keep versus unload.

### 4. The store (snapshots)

- **Keep metadata out of sealed bytes.** ds4's header carries hit count and last-used time, so two checkpoints of
  one state differ in 16 bytes ("mask those 16 bytes"). For us that breaks content-id dedup: times, counts and
  reasons live in the store index. Seal zeroes padding and unused rows ("never uninitialized device memory").
- **Recipe inputs**: ds4 keys by rendered text because token-id keys miss when "BPE merges across the boundary"
  (12). Proposed: the recipe hashes the logical input (rendered text or bytes) plus the tokenizer split and
  template; exact token ids travel in the payload. A tokenizer change is a new recipe space (ds4 had no check and
  emptied its store, a 1,661 s cold prefill).
- **Boundaries by anchoring** (12 technique 5): stores happen only where the live state already is ("the session is
  never rolled back"), with a reason (cold anchor, interval, prefill done, evict, reclaim, shutdown). Anchors are
  seals; evict and reclaim stores are parks (D7). A boundary never falls inside an atomic input span (an image).
- **Chains**: ds4 allows depth 1 only ("never delta-of-delta"), and writes a delta only if it is at most a quarter
  of a full checkpoint; a delta stores fixed and ring parts whole and cells as a range. Proposed: a maximum depth
  and a size fraction per engine; a base is never evicted while a delta needs it, deltas go first. A delta is legal
  only while no cell below its start was rewritten, which the engine must report.
- **Eviction**: a forest of prefix chains, thin interiors, keep tips and roots, choose all victims before deleting,
  evict before writing. Their incident: "Tier-first eviction wipes other conversations". Add per-owner quotas (ds4
  has none) and an owner on every entry. A pin that alone exceeds the budget loses protection rather than
  deadlock.
- **Restore**: validate everything before mutation, one reason per failed check. Proposed: a three-valued status
  for `slot_load` and `part_write`: ok, refused with nothing changed, failed after mutation (poison, D6). A failed
  restore quarantines the store entry; content is never deleted by a failure.
- **Approximate reuse** (crop and bounded replay, suffix cache reuse: KV moved to a new position, `[approx —
  approved]`, opt-in per request). Proposed: an approximate snapshot has its own recipe tagged with the method and
  its parameters and naming its source content ids; lookups take an exactness predicate, so a request that did not
  opt in never sees one; an approximate piece is never the parent of an exact one; the D3 refusal never compares
  across tags. Exact restores win every tie, as in ds4.

### 5. IO and experts (one machine, model bigger than device memory)

- **A shared IO service is the nagaya's job.** ds4's readers share drive health but "there is no cross-reader
  reservation". Proposed: an `io` capability (submit `(file, offset, len, dest)` batches with a class, wait for a
  prefix or the whole). The nagaya owns rates, hedging across replicated drives, loss and re-admission. A batch
  holds a lease on its destination until its final wait. A hung drive can hold a quantum about 30 s, so the nagaya
  needs a watchdog.
- **Tier edges carry more than a bandwidth**: rate by read size and queue depth, whether direct IO works into that
  destination (GTT refused `O_DIRECT`, 6.5% of prefill), alignment. Count direct and staged bytes separately so no
  fallback is silent.
- **Replicated NVMe**: a holder set of byte-identical copies, ids from content hashes, "Same size is not same
  bytes"; a copy with nothing verified "serves nothing".
- **Experts are weights, not a new part kind.** Proposed: a weights part may be **records**: N equal-size read-only
  records read by key, an authoritative copy in a lower tier and a resident subset above. A cached record is a
  copy, so evicting it is not a park and writes nothing back.
- **Ownership split at what routing can see.** The engine owns its device slots, their eviction and its routing;
  the nagaya owns the budget, the backing store, IO and host-memory staging. The nagaya's moves below the slots are
  bit-neutral, so it may make them freely; a host-memory expert tier (ds4 has none) is ours to add.
- **06 doubts its own lever**: real traffic is capacity-bound, and "The only large lever is not needing the
  expert": the large miss cut came from approximate routing. So exact routing is the baseline and cache-conditional
  routing is an opt-in, declared approximation.

### Engine-internal (stays out of the nagaya)

Kernels, fold orders and GEMM plan mechanics (only the determinism class crosses), FP4/MXFP4 and record formats,
expert eviction scores and prefetch, DSpark's scheduler, cohort sizes, RoPE relocation math, the replay band, vision
policy, telemetry side channels.

## Ai2's allocations scheduler (https://allenai.org/blog/impactful-scheduling)

Ai2 schedules thousands of H100/B200/B300 GPUs for about 150 researchers, with demand 2-3x supply. Whole GPUs, jobs of
hours; ours are shares of a GPU in quanta of milliseconds. The rules still carry:

- **Minimum runtime**, protected from preemption (at most 8 hours), declared per workload: our quantum's "runs to its
  end", at job scale. For training as filler, the time between checkpoints.
- **Resumable** is declared; preempted resumable work is requeued. For us: a filler job must be resumable, meaning
  its state parks.
- **"Unallocated" time**: minimum runtime zero, free, always preemptible. That is our filler class.
- **Hierarchical fair share over a 7-day window**; priority orders work only within a team. For us the "teams" could
  be job kinds (interactive endpoints, image batch, training), each with a share; ds4's owed-share rule works inside
  a class.
- **Hosts drain** as their workloads reach their minimum runtimes: our lease drain.
- Their stated cost: interactive sessions did worse under the cap, and they plan a separate place for them. For us,
  interactive is a class above everything, not a share.
