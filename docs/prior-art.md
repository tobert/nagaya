# Prior art

This doc records what other systems teach the nagaya, the exokernel of `docs/spec.md`, and what we take from
each. An item is a proposal until Amy decides it. Each item says where it lands: an ontology term, an ABI call, a
scheduler or store rule, or a decision.
When `docs/spec.md` now states the rule, the item gives the spec section. This doc is history: where it and the
spec differ, the spec is the rule (2026-10-20, after kaibo found stale items here).

## ds4-recipe (`~/src/ds4-recipe`)

ds4-recipe is a recipe for a fast DeepSeek V4.1 Flash runtime. It runs on one Strix Halo with 128 GB of unified
memory and three NVMe drives. Its numbers:

- 278 GB of experts streamed from NVMe;
- 4,667 of 15,360 routed experts resident;
- 98.75-99.51% of prompt tokens served from a disk KV cache;
- about 14.5 tok/s decode with DSpark.

It is one engine on one machine, so it has no answers across engines. But it met nearly every host-level question
that we have, once, in production.

Four readers read it on 2026-10-17: chapters 01, 03, 04, 05, 06, 10, 11, 12 and 13. They skipped the kernel
chapters 02, 07, 08, 09, 14 and 15 as engine-internal. We checked the load-bearing claims against the text.

### 1. Bits: what changes them, and who says so

- **Exactness tags** (README "Conventions", 01). The tags are `[exact]`, `[reference-defining]`, `[approx —
  approved]` and `[policy]`. A reference-defining choice is a free choice of evaluation order, made once: "once
  chosen it *is* the reference". Proposed: each bit setting in `identity()` becomes `{name, value, class}`.
  - Reference-defining settings are part of the identity.
  - Approved approximations are recipe inputs, and they mark the snapshot as not exact.
  - Policy settings are neither, unless the engine declares them output-relevant.
- **Same binary, different bits per process.** Timed GEMM plan selection is "reference-defining per process". 32
  processes of one binary at λ = 0 "picked 28 distinct plan sets, and their dumps fell into 13 groups" (01). Runs
  with identical plan tables matched. This breaks D3's refusal rule at each restart.
  - Decided: the D3 amendment (`determinism: always | pinned | per-process`) and an append-only plan table
    (ontology). The table covers the ragged shapes that kept ds4 from pinning in production (07).
  - Pinned plans cost ds4 nothing measurable. A cold 32k prefill took 85.87-86.07 s pinned against 86.20-86.99 s
    timed (07).
- **History changes bits.** With cache-conditional routing, "Every eviction, residency and sizing decision is
  `[policy]`, but it is **output-relevant**" (06). Also, "execution order changes expert-cache history". Every pass
  reads the residency state, but no state object holds it, and it never parks. A restore starts cold, so "restore
  is not resume".
  - Decided: quantum condition 7, "engine history that feeds arithmetic is identical" (D3 amendment).
  - An engine declares `bit_dependence: residency` in `limits()`. The nagaya then claims no identical bits under
    interleaving for it, and refuses no recipe for it.
  - Policy that never changes bits is true only for engines that do not declare it.
- **Pass shape changes bits.** A change of GEMM N changes the library's algorithm (01). Prefill call boundaries
  also matter. ds4 says "Never choose the job size independently of the engine cohort" (11). Our Segment entry
  claims that each segment gets the bits of a pass of it alone.
  - Proposed: `limits()` declares `segments_exact`, and whether quantum split points change bits (spec 5.7).
  - If they do, the split points join the boundaries in the recipe.
- **Logical digest, not layout digest** (03 §12, Pitfall 16). A bit-identical new encoding of the weights must not
  change the identity: "Keyed on the layout digest, every checkpoint would have been refused".
  - Decided (spec 12.2): `checkpoint_digest` is a logical digest: the canonical config plus the shard payload
    hashes. The arena's layout hash is a separate record.
  - This amends our line "the weights hash is the content id of the weights state object".
- **Equality in tests** (01 "What exact means"). This is for the nagaya's engine conformance and park/seal tests
  (spec 8.3).
  - Compare integer bit patterns. NaN or Inf fails, also when the bit patterns are equal.
  - Counters must also match. Report the first row that differs.
  - Keep a broken variant that the gate must fail.
  - Outputs from `pass` that are not finite are poison (D6).

### 2. Memory: one ledger per machine

- **One byte target, mandatory first, elastic as the remainder** (05 "1. One byte target"). Proposed: a Ledger noun
  under Holding, with one budget per tier (spec 4.1). gfx1151host and sm121host have one shared pool, and b70host has
  two. Each engine declares terms in `limits().memory` (spec 4.2):
  - mandatory `{name, bytes, lifetime}`, per engine, per context, or loaned to one holder at a time;
  - elastic `{name, unit_bytes, min_units, max_units}`. `min_units` is the term's minimum;
  - a key by phase, where the bytes differ by phase.
- **Counts from arithmetic, never from free memory** (05 "2.", 06 technique 12). "The slot count changes routes", so
  a restart must give the same sizes. Free memory only enforces the safety floor (spec 4.4, 4.7).
- **Reserve before allocate** (05 invariant 4), transient peaks included. A refused admission changes nothing
  (invariant 7). The worst phase is checked at launch, before any weight is read: "Fail at launch, not at hour 9".
  Proposed: the nagaya admits every call that allocates before it runs, and `open` reports the minimum bytes per
  tier first (spec 4.3, 2.4).
- **Reclaim parks** (05 "7.", 11 "11."), in a fixed order (spec 4.6). Agrees with D7.
  1. Park idle contexts that have no pin or lease to a lower tier, the oldest first. (ds4 parks them to NVMe.)
  2. Shrink the elastic terms of idle engines.
  3. Refuse.
- **Resize only between quanta** (05: "refuse while leases, pins, cohort references ... exist"). Proposed: a
  `resize(term, units)` capability, called only between quanta (spec 4.5).
- **Charge everything.** Vision weights and a capture pool without a charge "show up as untracked bytes". Other
  processes on the machine (tmpfs, swap) caused ds4's OOM. The nagaya sets its own OOM score (spec 4.2, 4.7).
- **Two elastic consumers.** When two engines both have elastic terms, which engine gets the remainder? ds4 has one
  elastic consumer. Decided (D9): a share for each engine in each machine's placement profile (spec 4.4).

### 3. Scheduling

- **No preemption of a running request** ("No active request is preempted"). This agrees with quanta. Priority
  orders only the work that waits. It never moves a continuation off its required context (spec 5.6).
- **Head-of-line blocking is the cost.** A 201-row turn waited 283.95 s behind a 112,257-row prefill of 32k-row
  jobs (11). ds4's fix is cooperative yields inside a pass, and nested guests in the yield.
  - Decided (D8): no yields. Each engine declares `max_quantum_ms` in `limits()`. An engine with a longer natural
    pass splits it into shorter quanta, or later into stages (spec 5.1).
  - Nested guests wait for later. Their protocol stays: "refused, nothing changed, the waiter keeps its place".
- **Owed-share fairness in `interactive`, FIFO flow in `batch` and `filler`** (11 "6."). Decided (spec 5.4): in
  `interactive`, the order is priority, then owed seconds (clamped to ±30 s), then arrival. "Leaving members take
  only their own balance", in time, not in tokens. In `batch` and `filler`, the order is priority and arrival, so one
  job runs in flow until higher-priority work is ready.
- **Gathering segments.** Wait a bounded join window for jobs that are ready, never for jobs that are not ready:
  "Nothing waits on a member that is not about to post". The window is a scheduler parameter for each class (spec
  5.7).
- **Cancel is per job, between quanta. Cancel is not failure.** A cancelled prefill deleted its snapshot, so the
  retry did a cold prefill of 442,546 tokens (about 35 minutes). Only poison discards state. A lease drain gets a
  time budget (ds4: 10 s to persist resident sessions) (spec 5.9, 10.3).
- **Loans are leases that the nagaya enforces.** A stranded loan of prefill scratch made every other prefill fail
  behind it. Every terminal state of a job releases its loans: done, cancelled, failed, expired (spec 5.9).
- **Cost model (D4)** (spec 5.11, decided).
  - Fit robust affine costs for each engine and segment count, with Theil-Sen. ds4's first profile used a median
    per cell, and one noisy step in it stalled admission. Theil-Sen over every round corrected it (11).
  - Key the fit by context depth.
  - Exclude passes that ran beside other work.
  - Fit from production passes with telemetry on. In ds4, an extra row cost 0.060-0.073 s in production and
    0.026-0.037 s in the profile (11, "Production per-row cost").
  - Use the fit for store eviction value-per-byte and deadline checks between quanta. At cold start, record "no model
    yet" until a fit exists. Never guess silently.
- **Phase transitions are a cost term.** ds4's change from prefill to decode (draft reload, regrow) took 1.594 s.
  It was the largest fixed cost for short turns. A loadable piece that shares the engine's memory (a draft model, an
  image model) declares its bytes and reload time. Then the scheduler can compare the cost to keep it and to unload
  it.

### 4. The store (snapshots)

- **Keep metadata out of sealed bytes.** ds4's header carries a hit count and a last-used time. Thus two
  snapshots of one state differ in 16 bytes ("mask those 16 bytes"). For us that breaks dedup by content id. So
  times, counts and reasons live in the store index. Seal sets padding and unused rows to zero ("never
  uninitialized device memory") (spec 6.3).
- **Recipe inputs.** ds4 keys by rendered text, because token-id keys miss when "BPE merges across the boundary"
  (12).
  - Proposed: the recipe hashes the logical input (rendered text or bytes), the tokenizer split and the template.
    The exact token ids travel in the payload (spec 6.2).
  - A tokenizer change is a new recipe space. ds4 had no check and emptied its store, which cost a cold prefill of
    1,661 s.
- **Boundaries by anchoring** (12 technique 5). A store happens only where the live state already is: "the session
  is never rolled back". Each store has a reason: cold anchor, interval, prefill done, evict, reclaim, shutdown
  (spec 6.4). Anchors are seals. Evict and reclaim stores are parks (D7). A boundary never falls inside an atomic
  input span, such as an image.
- **Chains.** ds4 permits depth 1 only ("never delta-of-delta"). It writes a delta only if the delta is at most a
  quarter of a full snapshot. A delta stores fixed and ring parts whole, and cells as a range.
  - Proposed: each engine declares a maximum depth and a size fraction (spec 6.5).
  - The store never evicts a base while a delta needs it. Deltas go first.
  - A delta is legal only while no cell below its start was written again. The engine must report this.
- **Eviction** (spec 6.9).
  - Keep a forest of prefix chains. Thin the interiors, and keep the tips and roots.
  - Choose all victims before a deletion. Evict before the write.
  - Their incident: "Tier-first eviction wipes other conversations". Add a quota for each owner (ds4 has none), and
    an owner on every entry.
  - A pin that alone is larger than the budget loses its protection, so eviction cannot deadlock.
- **Restore.** Validate everything before a change. Give one reason for each failed check.
  - Decided: `part_write` and `cells_write` return one of three results (spec 6.8). These are ok, refused with
    nothing changed, and failed after a change (poison, D6).
  - A failed restore quarantines the store entry only when verification fails. A failure never deletes content.
- **Approximate reuse.** ds4 has crop and bounded replay, and suffix cache reuse, which moves KV to a new position.
  Both are `[approx — approved]` and opt-in for each request. Proposed (spec 6.6):
  - An approximate snapshot has its own recipe. The recipe has a tag for the method and its parameters, and names
    the source content ids.
  - Lookups take the request's tolerance. Thus a request that did not opt in never sees an approximate snapshot.
  - An approximate piece is never the parent of an exact one.
  - The D3 refusal never compares across tags.
  - Exact restores win every tie, as in ds4.

### 5. IO and experts (one machine, a model larger than device memory)

- **A shared IO service is the nagaya's job.** ds4's readers share drive health, but "there is no cross-reader
  reservation" (spec 7.1).
  - Proposed: an `io` capability. A caller sends `(file, offset, len, dest)` batches with a class. It waits for a
    prefix or for the whole batch.
  - The nagaya owns rates, hedged reads across replicated drives, and the loss and return of drives.
  - A batch holds a lease on its destination until its final wait.
  - A drive that hangs can hold a quantum for about 30 s, so the nagaya needs a watchdog (spec 7.4).
- **Tier edges carry more than a bandwidth** (spec 2.2). An edge records the rate by read size and queue depth. It
  records whether direct IO works into the destination: GTT refused `O_DIRECT`, at a cost of 6.5% of prefill. It
  records the alignment. Count direct and staged bytes separately, so no fallback is silent (spec 7.2).
- **Replicated NVMe** (spec 7.3). A holder set has byte-identical copies, with ids from content hashes. "Same size is
  not same bytes". A copy with nothing verified "serves nothing".
- **Experts are weights, not a new part kind.** Proposed: a weights part can be **records**. Records are N read-only
  records of equal size, read by key. The authoritative copy is in a lower tier, and a resident subset is above. A
  cached record is a copy, so its eviction is not a park and writes nothing back.
- **Ownership splits at what routing can see.**
  - The engine owns its device slots, their eviction, and its routing.
  - The nagaya owns the budget, the backing store, IO, and staging in host memory.
  - The nagaya's moves below the slots do not change bits, so it can make them freely.
  - A tier of experts in host memory is ours to add. ds4 has none.
- **06 doubts its own lever.** Real traffic is bound by capacity, and "The only large lever is not needing the
  expert". The large cut in misses came from approximate routing. So exact routing is the baseline, and
  cache-conditional routing is an opt-in, declared approximation.

### Engine-internal (stays out of the nagaya)

These stay in the engine:

- kernels, fold orders and GEMM plan mechanics (only the determinism class crosses the ABI);
- FP4/MXFP4 and record formats;
- expert eviction scores and prefetch;
- DSpark's scheduler and cohort sizes;
- RoPE relocation math and the replay band;
- vision policy and telemetry side channels.

## Ai2's allocations scheduler (https://allenai.org/blog/impactful-scheduling)

Ai2 schedules thousands of H100, B200 and B300 GPUs for about 150 researchers, and demand is 2-3x supply. Ai2
allocates whole GPUs, in jobs of hours. We allocate parts of a GPU's time, in quanta of milliseconds. The rules
still apply:

- **Minimum runtime.** Each workload declares a minimum runtime (8 hours at most) that is safe from preemption. It is
  our quantum's "runs to its end", at the scale of a job. For training as filler, the nearest thing is its checkpoint
  interval (spec 5.5). Our filler gives its device to other work between quanta, so the interval protects nothing
  from preemption.
- **Resumable.** A workload declares it, and preempted resumable work goes back in the queue. For us, a filler job
  must be resumable: its state parks (spec 5.5).
- **"Unallocated" time.** The minimum runtime is zero, the time is free, and it is always preemptible. That is our
  filler class.
- **Hierarchical fair share over a 7-day window.** Priority orders work only in a team. Decided for v0 (spec 5.3,
  5.4): strict classes (`interactive` > `batch` > `filler`) do the job. `interactive` interleaves by owed seconds,
  while `batch` and `filler` run FIFO within each priority so one job runs in flow until higher-priority work arrives.
- **Hosts drain** as their workloads reach their minimum runtimes. That is our lease drain.
- **Their stated cost.** Interactive sessions did worse under the cap. Ai2 plans a separate place for interactive
  sessions. For
  us, interactive is a class above everything, not a share (spec 5.3).
