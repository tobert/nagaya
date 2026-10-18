# Reviews

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
