# Reviews

This doc records each review of the nagaya's docs, the spec of an exokernel for megakernels, and what we did with
each finding. The reviews read the drafts against mk and mkviolet, the two learning megakernels that
`docs/ontology.md` defines.

## 2026-10-17: kaibo consult on the first drafts

Cast `deepseek` (explorer and synth `deepseek-flash`), 539k tokens in, 83k out. kaibo could read only gmk. Thus we
staged the drafts there for the review, with copies of mk's `capi.hip` and `engine.h` and of mkviolet's
`gemma4.rs`, `held.rs`, `protocol.rs` and `hot_handoff.rs`. We deleted the copies after the review.

Verdict: the inventory of both engines was largely right. That covers names, part numbers, fork sharing, slots,
the 0-14 / 15-29 split, the priority locks, arena attach and `FDSTORE`. The generalizations were not right.
"The first engine that breaks the ABI is the second one, not the third."

| # | Finding | What we did |
|---|---|---|
| 1 | mk reads cells whole (no range), so an mk snapshot is about 316 MB at n = 3584, not "fixed + new rows" | ontology: storage per engine; ABI: `cells_read(from, n)` |
| 2 | Parking by sealing changes mk's bits; `fused` and `MK_FAST_QMM` change bits too | ontology: seal vs park, six conditions for identical bits; ABI D7; bit settings in identity |
| 3 | mkviolet's id omits slot geometry; the same chain joins for one slot size and fails for another (checked: `held.rs:312-325`, `gemma4.rs:1692-1693`) | D3: `geometry_key` in the id; later corrected: geometry changes coverage, not bits, and ids became content ids |
| 4 | The call list was mk's `Context`; mkviolet has slots, no fork, no seal | ABI split into core and declared capabilities |
| 5 | `stage_pass` lacks positions, meta, logit rows, snapshot, slot | D1: mkviolet starts with (a) |
| 6 | Sliding layers are rings, not append-only | fourth part kind: ring |
| 7 | mkviolet has 60 slabs and a spot check (checked: `gemma4.rs:284-287`) | arena = named fd set with check strength |
| 8 | `open` needs config and devices; mk has no device argument | `open(config)`; the wrapper sets the HIP device |
| 9 | Taps are single-segment, unfused only on mk | taps declared with conditions |
| 10 | mk needs token ids for PLE even with rows | D2: tokens, rows, or both |
| 11 | "One thread per card" was not mkviolet's rule | stated as a new nagaya rule |
| 12 | mkviolet stores no token count (`kv_high`, `pos0`) | slot entry corrected |
| 13 | mkviolet's `memfd` arena is host state, not weights | arena entry corrected |
| 14 | A branch block is not a segment: one slot, overwritten rows | segment entry: atomic, not a snapshot |
| 15 | Missing nouns: pass, stage, boundary, residual; "build" overloaded; pin vs lease | added; build = binary, history = chain; pin is a lease on a snapshot |
| 16 | Streams need period, deadline, late policy | stream entry; ABI "not covered" |

Unverified by the review (files outside the copies): the 256-token target, `YieldHook`, mkviolet `docs/03`
bandwidths, `council_v2.rs`, `stage.h`, `test_segments.py`, `docs/painter.md`.

## 2026-10-17: consistency pass (Claude, after D1 and D3)

Amy asked for a read of every file for consistency and use of the ontology.

- mk's "stage" (`stage.h`, "A stage is one op with its arguments") is our op. Noted in Op; ABI telemetry is
  `op_times()`.
- Weights became a state object, but State object listed two kinds. Now three: weights, context, snapshot. (The
  identity's weights hash was the weights' content id. spec 12.2 later replaced it with two weight ids.)
- Leftovers from geometry in the id:
  - Boundary said that boundaries are in the id. Now they are in the recipe.
  - Quantum condition 4 said "same slot geometry". Now it says "a slot the coverage fits".
  - "Layout" and "geometry" named one thing. Now the term is geometry.
- Content id: the parent's id is hashed only for a piece of a chain; mk's whole snapshots have no parent.
- D1 said per-card use makes mkviolet two engines; Gemma 4 does not fit on one card, so per-card means smaller
  engines. D2 deferred to the first port. Decision status listed: D4-D7 not yet reviewed with Amy.
- Machines: the third lab machine is the DGX Spark; gfx1151host and the Spark are 128 GB unified LPDDR5X.
- References into other repos named their repo (mkviolet `docs/03`, `council_v2.rs`; mk `docs/painter.md`).
- README said the ABI "names as a contract" the residual boundary; it does not, D1 defers it.

## 2026-10-17: kaibo consult on the spec (job-1)

Cast `deepseek` (explorer and synth `deepseek-flash`), 238k tokens in, 85k out. kaibo read all seven files whole:
the spec, the ontology, the ABI, the README, `CLAUDE.md`, and the prior art. We asked for contradictions, gaps,
rules that cannot work, ontology misuse, and STE problems that the length checker cannot see.

kaibo listed about 95 findings. Claude checked the claims about the text. The triage:

| Verdict | About | Examples |
|---|---|---|
| Real, and changes the design | 12 | below |
| Real and small | 50 | D10 not in the "D1-D9" pointer; two weight ids against `weights_hash`; Store, Geometry, Loan, Owner and Chain not defined; span prefix `nagaya.job.` against span name `nagaya.job`; "weight" for D9 and for weights; "batch" with four meanings |
| Duplicates | 15 | the thread and ledger findings, three times each |
| Not right | 15 | below |

The 12 that change the design, and what we did:

| # | Finding | What we did |
|---|---|---|
| 1 | The ledger must admit `open`, but `limits()` exists only after `open` | a static `plan(config) -> limits` call |
| 2 | No thread writes the host and NVMe parts of the ledger | one ledger owner; device threads send requests |
| 3 | A D1(a) pass on two cards breaks one thread per device | one thread per device group (Amy) |
| 4 | The audit's fresh process is a second GPU client | the audit runs after each start, in-process (Amy) |
| 5 | v0 scope differs between the spec, README and ABI | image batch and training are in v0 (Amy); the ABI gains both sections |
| 6 | D10 shapes by slot capacity, so equal coverage gives different content ids | cells and ring parts are shaped by coverage |
| 7 | D10 stores ring parts whole in a delta | a delta holds the rows after its parent's end |
| 8 | A refused restore would quarantine a good entry | quarantine only when verification fails |
| 9 | A lookup returns one entry, but a delta needs its chain | lookup returns the covering chain |
| 10 | An engine whose bits depend on cache size cannot have an elastic cache | that term is mandatory |
| 11 | The watchdog runs on the thread that it watches | the watchdog has its own thread |
| 12 | systemd reads `FileDescriptorStoreMax` before the nagaya starts | the unit file sets it, from the profile; fds drop only after a cold load fits |

Not right:

- Reclaim does not park through `part_write`. A park out uses `part_read`, which changes nothing. We kept one part:
  the device copy is freed only after the lower-tier copy is durable.
- `set_n` and `cells_write` do not allocate: mk sizes a context at `ctx_new(max)`. `attach` and `ctx_copy` do
  allocate, and we added them to admission.
- The count of six conditions in this file's first review is a dated record. It was right at that time.
- ds4's failed fit used medians per cell, and Theil-Sen was the correction. The fault was our wording.
- Boundaries need no ABI declaration. Clients mark them (`snap: true`); the engine only seals.

Amy's answers:

- v0 is inference, image batch and training.
- The audit runs across restarts.
- D9's number is a "share".
- An engine that spans devices gets one thread for its device group.

The checker gap that kaibo found: `tools/ste_check.py` applies 25 words to every sentence. CLAUDE.md says 20 for an
instruction. The checker cannot yet tell an instruction from a description.

## 2026-10-18: kaibo consult on the agent brief (`AGENTS.md`)

Cast `deepseek` (explorer and synth `deepseek-flash`), 253k tokens in, 46k out. We asked kaibo to read the docs as
the deepseek-flash agent that writes the first code. It could not reach mk, so Claude checked the mk claims in
`capi.hip`. They hold, and `mk_engine_attach` takes a `/proc/<pid>/fd/<n>` path.

kaibo listed 24 findings. We corrected these in `AGENTS.md`:

- the fake engine's parts, slots, seal and determinism class; a broken fake engine for the gates;
- a `nagaya-conformance` crate, and milestones for store entries and conformance;
- the scheduler as a state machine, not a pure order function;
- mk through `libloading` and `MK_ENGINE_LIB`, its shards in `QWEN38_SHARDS`, and no mk build by the agent;
- the ledger as an owned type with no lock;
- a commit prefix for changes to more than one crate, and `.gitignore` for weights;
- a rule for gaps: the agent writes a **Proposed** choice into the doc and commits it alone.

Amy decided the nine design gaps in an interview on the same day:

| Gap | Decision | Where |
|---|---|---|
| trait shape | a core trait, one trait per capability, accessors that return `Option`; config is a TOML table | ABI "Rust shape" |
| memory terms | each term has a tier (`device(i)` or `host`) and kinds | spec 4.2 |
| byte rules | tensors in name order, data from a 4 KiB boundary, no padding between tensors | ABI D10 |
| recipe | BLAKE3 derive-key, tagged fields, chained at boundaries, tokens not text | spec 6.2 |
| join windows | 0 ms in v0, in the bootstrap config | spec 5.7 |
| shares | per engine, then by each term's minimum bytes; round down; report the rest | spec 4.4 |
| coverage | in the content id; fit is not | ABI D10, ontology |
| `close()` | a capability; mk does not declare it | ABI "Capabilities" |
| mk writes | the wrapper checks preconditions first, then reads poison | ABI "How mk maps" |

Claude also renamed the `admission` capability to `pass_check`, so "admission" names only the ledger's check. Check
strength is one enum: `full`, `spot` or `none`.
