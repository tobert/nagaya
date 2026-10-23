# nagaya

One process per machine that hosts our engines. The nagaya is an exokernel for megakernels, its reference
implementation, and the council v2 API as served. Start with `README.md`, then `docs/spec.md`. `docs/ontology.md`
defines the terms and the four other projects that these docs name: mk, mkviolet, gogi and kaijutsu.

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
unified memory. The first engine is the Qwen3.8 megakernel from mk, through its C ABI (Amy, 2026-10-18). The second
is the Gemma 4 megakernel from mkviolet, on b70host's two B70 cards.

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
done, and `docs/history.md` keeps their full text, and the merges where the run recorded them. Milestone 15
runs now on gfx1151host. Milestone 15s is done except the cutover on b70host.

1. **Workspace.** The Cargo workspace, its crates, `rustfmt.toml` and the dev profile. Done.
2. **Engine trait.** The ABI core and capabilities as traits, with a fake engine and a broken fake for the gates. Done.
3. **Store keys.** Canonical safetensors (D10), the BLAKE3 content id and the recipe hash (6.2). Done.
4. **Ledger.** Admission of a profile from `plan()`, shares, and refusals with the shortfall in bytes (4.3, 4.4).
   Done.
5. **Scheduler order.** The scheduler as a state machine with a pure step function (5.3-5.9). Done.
6. **Store entries.** Entries, chains, lookup, the tripwire, restore results, eviction and pins (6.4-6.11). Done.
7. **Conformance.** The properties of 8.3 that need no GPU, run against the fake engine. Done.
8. **mk engine.** The binding of mk's C ABI, with the conformance suite on gfx1151. Done.
9. **Decisions.** The eight Proposed notes that Amy decided on 2026-10-18, as rules and code. Done.
10. **Process.** The bootstrap config, the profile, the threads, systemd, the exit codes and gossip (2, 3, 10). Done.
11. **Text front-end.** The Qwen3.8 family library: tokenizer, template and detokenizer, from the GGUF (11.3). Done.
11.5. **Spec compliance.** The audit of each crate against the spec, and a step for each gap. Done on 2026-10-20.
12. **Council v2.** The sampler, the family interface, the job path and the HTTP routes (11.1-11.3b, 11.5). Done on
    2026-10-20 night, in seven steps; `gogi-toy` is in gogi.
13. **Observability.** The log, and the OTLP export of spans and metrics (9.1-9.5), in `nagaya-telemetry`. Done on
    2026-10-20 night.
14. **Deploy on gfx1151host.** The unit, the config, `docs/deploy.md`, the file probe, the CPU sets and the heartbeat of an
    open. Done on 2026-10-20 night; the cutover ran at 00:24 on 2026-10-21.
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

    Amy decided the plan on 2026-10-22, by interview. The copy is from mk `555a2a3` into
    `crates/nagaya-mk-qwen38-gfx1151/hip/`. It keeps mk's folder layout, so the includes do not change.

    - **Copy:** all of `src/engine`, all of `src/rt`, the GGUF reader and dequantizer (`gguf.cpp`, `dequant.cpp`
      and their headers), and every file of `src/ops`. Leave out `mkgguf_c` (the C ABI for Python), the oracles'
      Python, `lab/`, `probes/`, `reference/`, the service and the harness.
    - **Op tests:** port mk's Python op tests (`tests/ops`) to Rust tests behind the feature `hip`. mk's Python
      writes their fixed inputs and outputs once, with no GPU. Keep the fixtures small. No Python runs at test time.
      `build.rs` builds the op libraries for these tests only.
    - **Steps, in this order.** Each step ends with the hip suite and the oracle gate green.
      1. The copy, built by `build.rs` and linked statically, with no change in behavior.
      2. The op tests in Rust.
      3. The GGUF reader and dequantizer in Rust. Check the bits against the C++ copy.
      4. The C ABI (`capi.hip`) in Rust, over a thin extern layer.
      5. The stage schedule (`stage_sched.h`) in Rust.
      6. The host code of `engine.hip`. `rt.cpp` stays C++ as the HIP runtime layer.
    - **Identity:** accept one refill. The canary (8.1) is not built, so the build facts stay in the digest (12.1).
      The new build gets a new digest, and each held context fills again once.
    - **Tonight:** do every step in order. Stop at the first step that does not go green, and report. Deploy the
      last green build.
    - mk and mkviolet become archives when milestones 15 and 15s are done (Amy, 2026-10-22: "they were our
      learning projects for this and their megakernels live on").
15s. **mkviolet on b70host.** Replace `mkviolet.service` with the nagaya on b70host. Amy, 2026-10-21: "pull
    the engine in now, you can stop mkviolet, our goal is to replace it." The branch is `b70host`. The milestone
    number is Decided 2026-10-22. Do the steps in this order:
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
- GPU tests read the Gemma 4 checkpoint directory from `GEMMA4_MODEL_DIR` (Decided 2026-10-22). The
  directory holds `config.json`, the safetensors shards and `tokenizer.json`. Do not put the path in code.
- GPU tests are behind the cargo feature `xe` (Decided 2026-10-22), so `cargo test` with no features needs
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
