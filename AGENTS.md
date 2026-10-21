# nagaya

One process per machine that hosts our engines. Start with `README.md`, then `docs/spec.md`.

## Writing style: Simplified Technical English

Amy, 2026-10-17: "make sure we're still doing the STE style in the docs I want it to bleed into the code so it's
crisp and easy to read; we will be working on this project for a long time."

We write docs, comments, commit messages and names in the style of ASD-STE100.

- One word, one meaning. Use the term from `docs/ontology.md` for each thing, and only that term. Do not use a
  synonym for variety.
- Short sentences: 20 words or fewer for an instruction, 25 or fewer for a description. One idea per sentence.
- Active voice. Present tense for what is true; past tense for what happened.
- Simple verbs. "Use", not "make use of". "Remove", not "get rid of". No phrasal verbs where one verb works.
- Articles stay in ("the", "a"). Do not drop them to make text shorter.
- No more than three nouns in a row. "The store's eviction rule", not "store entry eviction policy rule".
- A paragraph has one topic and at most six sentences. Use a list for steps or for parallel items.
- Instructions are commands: "Admit the call before it allocates."
- Numbers carry their unit and their source.
- Quotes stay as they were said. They are exempt from these rules.

In code:

- Names use ontology terms: `Quantum`, `Ledger`, `Recipe`, `ContentId`, not `WorkItem`, `MemMgr`, `CacheKey`.
- A comment says what a thing is or why it is so, in one or two STE sentences. It does not repeat the code.
- An error message says what failed and what the reader can do, in STE.

## Code

The nagaya is a Cargo workspace in Rust. The first machine is gfx1151host: one Strix Halo GPU (gfx1151, HIP), 128 GB of
unified memory. The first engine is mk (`~/src/megakernel-qwen38-flashnext-strixhalo`) through its C ABI (Amy,
2026-10-18). mkviolet's port comes later, on b70host.

### Rules

- Write the test first. Watch it fail for the right reason. Then write the code that makes it pass.
- Falsify each test that you write. Break the thing that it checks, see it fail, and put the thing back. Write the
  mutation in the commit message. A test that does not fail when you break the code is a defect: report it.
- If a test shows a real defect, stop and report it. Do not change the test to make it pass.
- Crash before you corrupt data (spec 8.5). Do not add a silent fallback. An unexpected state is an error or a panic.
- Report the real counts: tests run, tests passed, and each thing that you chose not to do.
- Cite `file:line` for each claim about code. Read the line again before you report it.
- The spec is the contract. When the code and a doc do not agree, stop and report. Do not choose one.
- When two docs do not agree, or a doc leaves a choice open, do not guess in code. Write the simplest choice into the
  doc with the mark **Proposed**, your reason, and your model name. Commit it alone, then continue. Amy decides later.
  When she accepts it, senpai writes it as a rule with the mark "Decided <date>" (spec, header).
- Do not edit other repositories. mk's HIP code stays in mk. If the work needs a change in mk, stop and report.

### Layout

The crates live in directories under `crates/`. Use these crates:

| Crate | Holds | Spec |
|---|---|---|
| `nagaya-abi` | the `Engine` trait, its types, and a fake engine behind the feature `fake` | `docs/engine-abi.md` |
| `nagaya-store` | canonical safetensors, content ids, recipes, entries, lookup, eviction | 6, D3, D10 |
| `nagaya-ledger` | budgets, terms, admission, shares, reclaim order | 4, D9 |
| `nagaya-sched` | jobs, quanta, classes, the order of ready quanta | 5 |
| `nagaya-mk-qwen38-gfx1151` | the mk engine: an FFI binding to `libmkengine.so` and the `Engine` trait on top | ABI "How mk maps" |
| `nagaya-mk-gemma4-bmg` | the mkviolet engine: Level Zero, the OpenCL C kernels, and the Gemma 4 host code; a card needs `xe` | ABI "How mkviolet maps" |
| `nagaya-council` | council v2 logic with no I/O: the family interface, the sampler, request checks | 11 |
| `nagaya-jinja` | the Jinja interpreter of the chat templates, with the results of Python's jinja2 | 11.3 |
| `nagaya-text-qwen38` | the Qwen3.8 family library: tokenizer, templates, detokenizer, reply parser | 11.3, 11.3a |
| `nagaya-text-gemma4` | the Gemma 4 family library: tokenizer, template, escapes, council rendering, reply parser | 11.3, 11.3a |
| `nagaya-text-toy` | the toy family library on gogi's `gogi-toy`, for tests with the fake engine | 11.3b |
| `nagaya-conformance` | the conformance suite (8.3), generic over the `Engine` trait | 8.3 |
| `nagaya-telemetry` | the instruments, the stderr log, the OTLP export and the sampler | 9 |
| `nagaya` | the process: bootstrap config, threads, HTTP, gossip, systemd | 2, 3, 9-11 |

Crate names follow `nagaya-<kind>-<model>-<target>`. The kind `mk` is a megakernel engine. The model is the family
and version without dots. The target is the compiler's device target, for example `gfx1151`. The Intel target is
`bmg`, so the Gemma 4 engine is `nagaya-mk-gemma4-bmg` (Amy, 2026-10-21, by interview). The device count is the
profile's business.

Keep pure logic out of `nagaya-mk-qwen38-gfx1151` and `nagaya`. The ledger, the store, the scheduler and the
council logic run in tests with no GPU. The ledger is a plain owned type with `&mut self` methods. Do not wrap it
in a lock: one thread owns it (3.2).

### Milestones

Do them in this order. Each milestone ends with all checks green and one or more commits. Milestones 1 to 14 are
done. Milestone 15 is next on gfx1151host, and milestone 15s is next on b70host.

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
    the contract: `~/src/gogi/docs/council-api.md` and the types in `gogi-proto`. Coordinator routing waits for a
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
    `gogi-conformance` gave 215 ok, 0 FAIL and 5 skip. Arenas through a restart (10.2) are not built. The merges:

    - the unit, the config and `docs/deploy.md`: `ec87a36`;
    - the file probe: `010ae47`;
    - the CPU sets: `7927da9`;
    - the mk factory and the probe at start: `b89b014`;
    - the heartbeat during an open: `cee227d`;
    - the stop path: `9a79c3a`;
    - the park of held bytes: `5c5dd88`.
15. **Import mk.** Copy mk's engine into `nagaya-mk-qwen38-gfx1151`, after the cutover. We own the copy, and the old
    repository stays an archive (Amy, 2026-10-19: "copy code in, we own all of it 100%"). The rules:
    - The HIP device kernels stay HIP. The host code is Rust.
    - Copy only what the engine needs: the sources of `libmkengine` and what they include. Leave `lab/`, `probes/`,
      `reference/`, the service and the harness behind. Amy: "don't copy extra stuff".
    - `build.rs` calls `hipcc` for `gfx1151`. No CMake (Amy: "I'd rather not drag cmake in"). Keep mk's float
      flags, `-ffp-contract=off -fno-fast-math`, because the conformance gates compare bits.
    - Link the engine statically. The `libloading` binding and the library pin go, and the build identity takes the
      pin's place in the engine identity (12.1).
    - First copy the code with no change in behavior, and run the hip suite and the conformance gates on gfx1151.
      Then port host code to Rust one part at a time, each with the gates green. Start with the GGUF reader.
    - Before step one, interview Amy on the copy list and the order of the ports.
15s. **mkviolet on b70host.** Replace `mkviolet.service` with the nagaya on b70host. Amy, 2026-10-21: "pull
    the engine in now, you can stop mkviolet, our goal is to replace it." The branch is `b70host`. The milestone
    number is **Proposed** (claude-opus-5-5). Do the steps in this order:
    1. **Spec.** Write the b70host rules into the docs: engine-abi D1 and D11, spec 2.1-2.4, 3.2b, 6.14, 10.2, 11.3
       and 12.1. Amy decides each Proposed note.
    2. **Host code.** Build the engine registry (11.3) and the devices with PCI addresses in `Factory` (engine-abi
       "The factory gets its devices"). Build the CPU sets for each card (3.2b) and the feed rule (6.14). Build the
       stage scheduling in `nagaya-sched` and its threads, tested with a fake engine of two stages (engine-abi D1).
    3. **Gemma 4 family.** Write `nagaya-text-gemma4` (11.3) and its oracle fixture (8.3).
    4. **Copy the engine.** Copy mkviolet's engine into `nagaya-mk-gemma4-bmg`, with no change in behavior. Copy
       branch blocks, prompt-lookup drafts and the long slots (65,536 positions, and the 262,144 work in mkviolet
       `docs/07-long-context.md`). Do not copy the VNNI packer, the memfd stub, or other parts that nothing calls.
       The host code stays Rust. The kernels stay OpenCL C, and `build.rs` compiles them with `ocloc` for
       `bmg-g31` into `OUT_DIR`.
    5. **The trait.** Implement the `Engine` trait with `stages` and `stage_pass` on the copy (engine-abi D1).
    6. **Gates.** Run the conformance gates and the Gemma 4 oracle on the two B70 cards.
    7. **Arenas.** Keep the arenas through a restart (10.2), with the head in a slab.
    8. **Deploy.** Write the unit and the profile for b70host. The bootstrap config binds `198.51.100.10:8091`,
       mkviolet's address. Do not start, stop or enable a service: senpai does the cutover with Amy.
16. Stop and ask Amy for the next milestone.

### mk on gfx1151host

- Do not build mk. Amy builds it. The library is `build/src/engine/libmkengine.so` in mk's repo. If it is missing or
  older than mk's last commit, stop and report.
- Load the library at runtime with `libloading`, from the path in `MK_ENGINE_LIB`. Do not link it at build time, so
  that a machine without mk builds and tests the workspace. Do not copy the library into this repo.
- mk's weights are three GGUF shards. mk's `harness/paths.py` (`SHARDS`) gives their paths. GPU tests read them
  from `MK_SHARDS`, three paths with `:` between them. Do not put the paths in code.
- `mk_engine_attach` takes a path, not an fd. For an fd from systemd's store, give `/proc/self/fd/<n>`.
- `mk_ctx_state_write` returns nothing. The wrapper maps mk's result onto the three write results (ABI
  "Capabilities") through `mk_ctx_poisoned`.
- The nagaya must be the only GPU client (spec 3.1). Since 2026-10-21, `nagaya.service` serves council v2 on gfx1151host.
  Before you run GPU tests, check that both units are inactive:
  `systemctl --user is-active megakernel.service nagaya.service`.
  - They are user units. The command without `--user` says "inactive" while they run.
  - If either one runs, stop and report. Do not stop a service yourself.
  - In a container with no systemd, ask your reviewer to check it.
- mk aborts the process on some bad arguments (`MK_DIE` in `src/engine/capi.hip`). Call `mk_segments_problem`
  before `mk_pass_segments`.
- mk's HIP device is the device that is current on the calling thread. Make every mk call from one thread.
- Do not call `mk_engine_close` or free the engine in tests. HIP teardown in a process segfaults (spec 3.3).
- GPU tests are behind the cargo feature `hip`, so `cargo test` with no features needs no GPU.

### mkviolet on b70host

- mkviolet is `~/src/mkviolet`. It is a reference: read it, and do not edit it.
- The nagaya must be the only compute client of the B70 cards (spec 3.1). Amy stopped and disabled `mkviolet.service`
  on 2026-10-21 ("cards and cpus are all yours"). Before you run GPU tests, check that it is inactive:
  `systemctl --user is-active mkviolet.service nagaya.service`. If either one runs, stop and report.
- The iGPU (gfx1036) drives the display. Do not run work on it.
- GPU tests read the Gemma 4 checkpoint directory from `GEMMA4_MODEL_DIR` (**Proposed**, claude-opus-5-5). The
  directory holds `config.json`, the safetensors shards and `tokenizer.json`. Do not put the path in code.
- GPU tests are behind the cargo feature `xe` (**Proposed**, claude-opus-5-5), so `cargo test` with no features needs
  no card. On b70host, also run `cargo test -p nagaya-mk-gemma4-bmg --tests --features xe -- --test-threads=1`.
- The reference Python for the Gemma 4 oracle is `~/.venvs/nagaya-ref`. Do not run it with a GPU job.

### Checks

Run these before each commit. Each one must pass.

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features hip -- -D warnings
cargo clippy --workspace --all-targets --features xe -- -D warnings
cargo test --workspace --tests
cargo test --workspace --tests --no-default-features
python3 tools/ste_check.py README.md AGENTS.md docs/*.md
awk 'length > 120 && !/^\|/ {print FILENAME":"FNR; bad = 1} END {exit bad}' README.md AGENTS.md docs/*.md
```

The check with `--features xe` needs `ocloc`, because `build.rs` compiles the kernels. Run it on b70host, and skip it
on a machine with no `ocloc`. Run `cargo fmt --all` often, not only before a commit. On gfx1151host, after a change to
`nagaya-mk-qwen38-gfx1151`, also run this:

```
cargo test -p nagaya-mk-qwen38-gfx1151 --tests --features hip -- --test-threads=1
```

`cargo test --lib` does not compile integration tests. Name `--tests`.

#### Tests that need a model file

Amy, 2026-10-21: "adjust the tests so they don't break our tests when we don't have a model downloaded locally. we
won't replicate all models to all machines so this state is normal. we should figure out a way to ensure the tests
get ran sometimes, perhaps just a bump in our AGENTS.md"

- A host test that needs a model file skips when the file is absent. It prints one line to stderr.
- A path that exists but cannot be read fails the test. Only absence skips.
- The helper is `crates/nagaya-text-qwen38/tests/common/model.rs`. Call `absent()` first in each such test.
- On a machine that holds a model, run the checks with `NAGAYA_REQUIRE_MODELS=1` and the model variables set.
  Then an absent file fails the test.
- Do this at least before each merge that touches the crates of that model.
- gfx1151host holds the Qwen3.8 GGUF at the default path of `NAGAYA_QWEN38_GGUF`.
- b70host holds Gemma 4 in `~/src/mkviolet/checkpoints/gemma-4-26B-A4B`. The `xe` tests of
  `nagaya-mk-gemma4-bmg` and the tests of `nagaya-text-gemma4` read it from `GEMMA4_MODEL_DIR`. The variable has no
  default, so the tests of `nagaya-text-gemma4` skip when it is unset.
- Those tests also read HF reference files from `target/gemma4-golden` or `GEMMA4_GOLDEN_DIR`. Write them with
  `tools/gemma4_golden.sh`, with no GPU job. An absent reference file skips the test, as an absent model does.
- The report of a check names how many tests skipped for a missing model.

On gfx1151host, run the oracle bench with `MK_ENGINE_LIB` and `MK_SHARDS` set and both units inactive:
`cargo bench -p nagaya-mk-qwen38-gfx1151 --features hip --bench oracle`. It exits with 1 when a gate breaks.
Set `MK_FAST_QMM=0` for the gate: the exact path must match the reference. Leave it unset to measure the fast path.
Make the fixture again with `PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <mk venv python> -I tools/oracle_fixture.py
OUT.json`. It needs about 30 GiB of RAM, and it must not run with a GPU job.

`tools/ste_check.py` checks Markdown only, and it allows 25 words in each sentence. Comments and commit messages
have no check. Keep instructions at 20 words or fewer yourself.

### Git

- Commit small steps on the current branch. One commit holds one idea with its tests.
- Do not push. Do not amend, rebase, reset or force. Amy reads the history as it is.
- Do not commit `signoff.md`, `target/`, weights or snapshots. Tests write files only under a temporary directory.
  Small test fixtures go under `tests/fixtures/` in their crate.
- A commit message is STE. Its first line is `<crate>: <what changed>`, or `nagaya: <what changed>` when the commit
  changes more than one crate. The body says why and names the mutations.
- End each commit message with a `Co-Authored-By:` line that names the model that wrote the change.

## Working notes

- `signoff.md` is the handoff between sessions and seats in a run. Read it first, and keep it current. It is never
  committed.
- Commit messages are in STE and end with the attribution lines.

頑張って！
