# History

This doc keeps the finished milestones of the nagaya, as `AGENTS.md` wrote them, with what each one built and its
merges. `AGENTS.md` keeps one line for each of them and the full text of the milestones that run now. The nagaya is
the exokernel of `docs/spec.md`; these milestones built its reference implementation, in this order.

The design came on 2026-10-17 from a conversation with Amy. The code began with mk's engine on 2026-10-18
(`docs/ontology.md` defines mk, mkviolet, gogi and kaijutsu). Senpai and Amy cut gfx1151host over to the nagaya at 00:24 on
2026-10-21.

## Milestones 1 to 14

1. **Workspace.** Make the workspace, the seven crates, and a `rustfmt.toml` with `max_width = 120`. Run each check
   once, with `--features hip` too, so that each one works before there is real code. In the workspace
   `Cargo.toml`, set `[profile.dev] debug = "line-tables-only"` and `[profile.dev.package."*"] debug = false`.
   Backtraces keep their lines, and `target/` stays small (Amy, 2026-10-18: "we almost never use debug info").
2. **Engine trait.** Write the ABI core and capabilities as a trait, plus a fake engine. The fake engine is
   `always`. It has fixed, cells, ring and derived parts, slots of two sizes, a seal that changes bits, and the
   `parts`, `coverage` and `seal` capabilities. A second fake engine is broken on purpose, for the gates (8.3).
3. **Store keys.** Write and read canonical safetensors (D10), the BLAKE3 content id, and the recipe hash (6.2).
   Test that equal coverage gives equal bytes from slots of different sizes. Test that a delta names its parent.
   Keep a fixed file with its content id in the tests, so a change to the byte rules fails a test.
4. **Ledger.** Admit a profile from `plan()` results (2.4), divide spare memory by shares (4.4), and refuse with the
   shortfall in bytes for each tier. Test that a refusal changes nothing (4.3).
5. **Scheduler order.** Write the scheduler as a state machine. A pure step function takes the scheduler state and
   one event. It gives the new state and the next quantum. Test each class rule (5.3-5.9).
6. **Store entries.** Write entries, chains, lookup, the tripwire, restore results, eviction and pins (6.4-6.11).
7. **Conformance.** Write the properties of 8.3 that need no GPU, and run them against the fake engine. Milestone
   11.5 added the gate `segments_exact_bits`.
8. **mk engine.** Bind mk's C ABI and implement the trait on top. Run the conformance suite on gfx1151.
9. **Decisions.** Amy decided the eight open Proposed notes on 2026-10-18. The signoff lists them under "Decided
   2026-10-18 evening". Write each one as a rule in its doc, and remove its Proposed mark. Three need code:
   - A pass that the mk binding refuses before it calls mk returns a `PassError`. It does not return `Ok`.
   - The binding refuses a library whose SHA-256 is not the pin in the profile. Tests can leave the pin unset.
   - mk's pass scratch is a mandatory term. Measure the pool once on gfx1151, and declare that size. Refuse the
     engine when a later measurement is larger.
10. **Process.** In `nagaya`, read the bootstrap config (2.1, 11.6) and a profile for one machine (2.4). Admit the
    profile with the ledger, and open each engine on its own thread (3.2). Send `READY=1` (10.1), stop on SIGTERM
    (10.3), and exit with the codes of 10.4. Test with the fake engine. Then join the cluster with `chitchat` (2.3)
    before the HTTP server, and test it with two processes on loopback (Amy, 2026-10-19: "step e plus gossip now").
11. **Text front-end.** Write the family library for Qwen3.8 (11.3): the tokenizer, the chat template and the
    detokenizer, from the GGUF metadata. Test against fixed outputs of mk's Python front-end (`harness/tokenizer.py`,
    `service/chatfmt.py`). Make those outputs with mk's Python, with no service and no GPU. Keep them in the tests.
11.5. **Spec compliance.** Amy, 2026-10-20: "rework our specs and do a M11.5 pass of bringin the code into
    compliance with the spec after updating it with what we learned." The spec changed first, on 2026-10-20:
    - An audit compares each crate with its spec sections. It lists each gap with `file:line` in the code and in
      the doc.
    - Each gap is a step with a test that fails first. A gap that needs a decision goes to Amy.
    - A rule for a part that no milestone built yet waits for the milestone that builds that part. The audit lists
      it as deferred.
12. **Council v2.** Serve council v2 for the engines of the profile, with one sampler (11.1-11.3b, 11.5). gogi is
    the contract: gogi `docs/council-api.md` and the types in `gogi-proto`. Coordinator routing waits for a
    second machine. Serve the core capabilities only (11.1). Do the steps in this order:
    1. **gogi-toy.** In gogi, move the toy tokenizer, template and model out of `gogi-fake` into a new crate
       `gogi-toy`. Add a reply parser for calls and thinking. `gogi-fake` uses the new crate. gogi's `AGENTS.md`
       applies there: do not run `cargo fmt` in gogi. Amy decided this on 2026-10-20.
    2. **Sampler.** In `nagaya-council`, write the sampler of 11.2. Test it against fixed outputs of mk's
       `service/sampling.py`, made from fixed logits, settings and seeds, as in milestone 11. The draw keeps no state,
       so it does not match mk's PCG64 bits. Match the filter order and the probabilities.
    3. **Family interface.** In `nagaya-council`, write the interface of a family library. It renders a context with its
       boundaries and a council read (11.3a), parses a reply, and gives the template id. Write
       `nagaya-text-toy` on `gogi-toy`. Give the fake engine logits over the toy vocabulary.
    4. **Qwen3.8 council.** In `nagaya-text-qwen38`, port mkviolet's council rendering (`src/template.rs` and its
       council code) to Qwen3.8 tokens, and the reply parser of mk's `service/chatfmt.py`. Test the parser against
       fixed outputs of mk's Python. Test the template id: a comment edit keeps it, and a token change changes it.
    5. **Job path.** In `nagaya`, take a request to an engine through `nagaya-sched`. Build a context: look up the
       longest held prefix by recipe (6.6), restore it, feed the rest, and put a snapshot at each boundary.
    6. **HTTP.** Serve the council routes with `axum` on the control runtime: identity, contexts, snapshots, pins,
       specs, decisions, reads and generations with SSE. Use gogi's error body and request ids (11.5). Specs persist
       in the store directory. Run `gogi-conformance` against the process with the fake engine, in a test.
    7. **Follow-ups.** Build the scheduler's part of interval copies (6.13), and declare mk's grain when it works.
       Restore through `Store::restore` in the conformance gates (8.3), and refuse an interval copy as a parent
       (6.13). The ledger counts per-context terms when it admits `ctx_new` (4.2). This step does not wait for the
       others.

    Done on 2026-10-20 night. Step 1 is in gogi (`gogi-toy`). Step 7 declares mk's grain of 256 tokens. The merges:

    - step 2: `cf81337`;
    - step 3: `3dcaea2`;
    - step 4: `960484e` and `40e21f0`;
    - step 5: `766d5bd`;
    - step 6: `9c99735`;
    - step 7: `0b45c95` and `6121891`.
13. **Observability.** Log to stderr. Export traces and metrics over OTLP when the endpoint is set (9.1-9.5). Each
    instrument has a call site under test.

    Done on 2026-10-20 night, merge `4bdd501`. The crate is `nagaya-telemetry`. Not built: the JSON-lines file
    export, the OTLP export of logs, and the instruments that have no call site yet (spec 9.5).
14. **Deploy on gfx1151host.** Write a systemd user unit in `deploy/` (10.1). The bootstrap config binds
    `198.51.100.20:8090`, mk's address now (Amy, 2026-10-18). Run `gogi-conformance` against the process on gfx1151
    with the mk engine. Then drive one kaijutsu context against it: kaijutsu speaks council v2 through gogi. Write
    the cutover steps in `docs/deploy.md`, and keep arenas through a restart (10.2) for later. Do not start, stop
    or disable a service: senpai does the cutover with Amy.

    Also build these, which Amy put here on 2026-10-20:

    - the inventory probe that reads files under a root path, with fixed trees for three machines (2.1);
    - CPU isolation with `sched_setaffinity`, and its test of the CPU masks (3.2a);
    - the heartbeat of a group while its engine opens (7.4).

    Done on 2026-10-20 night. Senpai and Amy did the cutover at 00:24 on 2026-10-21. `nagaya.service` serves
    council v2 on `198.51.100.20:8090` with mk on gfx1151, and `megakernel.service` is stopped and disabled.
    `gogi-conformance` gave 215 ok, 0 FAIL and 5 skip. Arenas through a restart (10.2) were not built then; milestone
    15s built them on b70host. The merges:

    - the unit, the config and `docs/deploy.md`: `ec87a36`;
    - the file probe: `010ae47`;
    - the CPU sets: `7927da9`;
    - the mk factory and the probe at start: `b89b014`;
    - the heartbeat during an open: `cee227d`;
    - the stop path: `9a79c3a`;
    - the park of held bytes: `5c5dd88`.
