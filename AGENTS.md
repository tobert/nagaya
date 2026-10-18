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
- Do not edit other repositories. mk's HIP code stays in mk. If the work needs a change in mk, stop and report.

### Layout

Use these crates:

| Crate | Holds | Spec |
|---|---|---|
| `nagaya-abi` | the `Engine` trait, its types, and a fake engine behind the feature `fake` | `docs/engine-abi.md` |
| `nagaya-store` | canonical safetensors, content ids, recipes, entries, lookup, eviction | 6, D3, D10 |
| `nagaya-ledger` | budgets, terms, admission, shares, reclaim order | 4, D9 |
| `nagaya-sched` | jobs, quanta, classes, the order of ready quanta | 5 |
| `nagaya-mk` | the mk engine: an FFI binding to `libmkengine.so` and the `Engine` trait on top | ABI "How mk maps" |
| `nagaya-conformance` | the conformance suite (8.3), generic over the `Engine` trait | 8.3 |
| `nagaya` | the process: bootstrap config, threads, HTTP, gossip, systemd | 2, 3, 9-11 |

Keep pure logic out of `nagaya-mk` and `nagaya`. The ledger, the store and the scheduler run in tests with no GPU.
The ledger is a plain owned type with `&mut self` methods. Do not wrap it in a lock: one thread owns it (3.2).

### Milestones

Do them in this order. Each milestone ends with all checks green and one or more commits.

1. **Workspace.** Make the workspace, the seven crates, and a `rustfmt.toml` with `max_width = 120`. Run each check
   once, with `--features hip` too, so that each one works before there is real code.
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
7. **Conformance.** Write the properties of 8.3 that need no GPU, and run them against the fake engine.
8. **mk engine.** Bind mk's C ABI and implement the trait on top. Run the conformance suite on gfx1151.
9. Stop and ask Amy for the next milestone.

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
- The nagaya must be the only GPU client (spec 3.1). Before you run GPU tests, check that `megakernel.service` is
  inactive. If it runs, stop and report. Do not stop the service yourself.
- mk aborts the process on some bad arguments (`MK_DIE` in `src/engine/capi.hip`). Call `mk_segments_problem`
  before `mk_pass_segments`.
- mk's HIP device is the device that is current on the calling thread. Make every mk call from one thread.
- Do not call `mk_engine_close` or free the engine in tests. HIP teardown in a process segfaults (spec 3.3).
- GPU tests are behind the cargo feature `hip`, so `cargo test` with no features needs no GPU.

### Checks

Run these before each commit. Each one must pass.

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --tests
python3 tools/ste_check.py README.md AGENTS.md docs/*.md
awk 'length > 120 && !/^\|/ {print FILENAME":"FNR}' README.md AGENTS.md docs/*.md
```

Run `cargo fmt --all` often, not only before a commit. On gfx1151host, after a change to `nagaya-mk`, also run this:

```
cargo test -p nagaya-mk --tests --features hip -- --test-threads=1
```

`cargo test --lib` does not compile integration tests. Name `--tests`.

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

- `signoff.md` is the handoff between sessions; it is never committed. The lasting copy is
  `~/exomemory/nagaya/signoff.md`.
- Commit messages are in STE and end with the attribution lines.
