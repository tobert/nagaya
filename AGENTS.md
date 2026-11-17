# nagaya

One process per machine that hosts our engines. The nagaya is an exokernel for megakernels, its reference
implementation, and the council v2 API as served. Start with `README.md`, then `docs/spec.md`. `docs/ontology.md`
defines the terms. `docs/history.md` keeps milestones 1 to 30 as they were written.

@tobert (Amy) owns the project, and she decides. An agent is a model session that works here.

## Direction

Amy, 2026-11-11: "I want to start scrubbing nagaya to consider for posting on github."

The next reader of this repository is a stranger. Milestone 31 prepares the repository for that reader. Until it
is done, each change must move the repository toward that reader, not away:

- Do not write an old name in new text or new code ("Names").
- Do not add a host name, a network address or a home path to code, tests or tools. Read them from config.
- Do not add a quote, a private note or a session record to a doc. `signoff.md` holds those, and it stays local.
- Name the owner less, and name her machines less. Where a doc must name her, write @tobert, which GitHub makes
  a link. A rule stands without the name of the person who decided it.
- Write each doc for a reader who has only the repository.

## For agents working here

- Read `signoff.md` first. It is the handoff between sessions and machines, and it is never committed.
- Keep `signoff.md` short and current. Move each durable fact into a doc before the fact goes stale.
- Keep status explicit. If you change direction, say what changed and why.
- If a tool or a sandbox stops the work, report the exact cause and the safe next option.
- Report the real counts: tests run, tests passed, and each thing that you chose not to do.
- Use a worktree for a lane of work: `~/src/wt/nagaya-<topic>`. Give each worktree its own target directory.
- Tell another session of the project before you change a file that it uses.
- Use a little 日本語（にほんご） when it fits; add ふりがな for kanji.

頑張（がんば）って！

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
- An error message says what failed, why, and what the reader can do, in STE.

An error message is the text that a client reads most, and it reads it when it cannot continue. Give it the most
care.

## Names

Amy, 2026-11-11: "We will want to start filtering out some of our old codenames that are now confusing, like 'mk',
'mkviolet', and 'gmk'; those were practice projects, some code came forward, but they are now part of the nagaya."

The practice projects are part of the nagaya now. A reader does not need their names. In new text and new code,
write the name of the thing in the nagaya:

| Old name | What it was | Write this |
|---|---|---|
| mk | the practice megakernel of Qwen3.8 on gfx1151 | the Qwen3.8 engine, or `nagaya-mk-qwen38-gfx1151` |
| mkviolet | the practice megakernel of Gemma 4 on two B70 cards | the Gemma 4 engine, or `nagaya-mk-gemma4-bmg` |
| gmk | the practice project that fitted the first J-Lens | the first J-Lens fit, of Gemma 4 12B |
| `megakernel.service`, `mkviolet.service`, `libmkengine`, `/mk/v1` | the old services and their parts | nothing: they are gone |
| diffusepal | the old Qwen-Image tool | the diffusers reference |

- "Megakernel" stays. It is the ontology's term for an engine that runs a model's pass as one kernel.
- `mk` stays in crate names as the kind "megakernel engine" ("Layout"). Decided 2026-11-11. This is the one
  meaning of `mk` in the nagaya.
- `mk` also stays in the internal names of a megakernel, such as `MK_LFM3` and `mk_persistent_interpreter`. The
  interface of the exokernel does not show them. Decided 2026-11-17.
- Each identifier that carries the name of an old project gets a new name, in the C++ and HIP code too. Decided
  2026-11-11: the repository is private and all the code is here, so make the names as good as they can be.
  The Qwen3.8 engine has its new names since 2026-11-12: `QWEN38_SHARDS`, `HIP_CHECK`, `namespace qwen38`, and the
  engine name `qwen38` on gfx1151host.
- The names of the old Gemma 4 project are still in the code. Until milestone 31 step 2 changes such an identifier,
  use it as it is, and do not copy the pattern.
- `docs/history.md` can keep the old names. It is the record of how the code came here.

## Code

The nagaya is a Cargo workspace in Rust. It has engines for two targets:

- **gfx1151**, one AMD Strix Halo GPU with 128 GB of unified memory, through HIP: Qwen3.8 and Qwen-Image 2.1.
- **bmg**, Intel Arc Pro B70 cards, through Level Zero and OpenCL C kernels: Gemma 4, Qwen3-TTS, SDXL,
  EmbeddingGemma 2 and LFM3.

### Rules

- Write the test first. Watch it fail for the right reason. Then write the code that makes it pass.
- Falsify each test that you write. Break the thing that it checks, see it fail, and put the thing back. Write the
  mutation in the commit message. A test that does not fail when you break the code is a defect: report it.
- Run the mutation before the commit that names it. A message must not name a mutation that did not run.
- If a test shows a real defect, stop and report it. Do not change the test to make it pass.
- Crash before you corrupt data (spec 8.5). Do not add a silent fallback. An unexpected state is an error or a panic.
- Report the real counts: tests run, tests passed, and each thing that you chose not to do.
- Cite `file:line` for each claim about code. Read the line again before you report it.
- The spec is the contract. When the code and a doc do not agree, stop and report. Do not choose one.
- When two docs do not agree, or a doc leaves a choice open, do not guess in code. Write the simplest choice into the
  doc with the mark **Proposed**, your reason, and your model name. Commit it alone, then continue. Amy decides later.
  When she accepts it, remove the mark and the model name. Git keeps the history of the rule (spec, header).
- Engines that we build give the same bits for the same identity and inputs. A change that moves bits changes the
  engine identity, and the commit message says so.
- Do not edit other repositories. The exception is gogi (Amy, 2026-10-30). An agent may edit it on a branch, merge
  it to gogi `main`, push it and move the pin.
- The nagaya does not check a model license. The user accepts it at the download (Amy, 2026-11-06).

### Layout

The crates live in directories under `crates/`. Use these crates:

| Crate | Holds | Spec |
|---|---|---|
| `nagaya-abi` | the `Engine` trait, its types, and a fake engine behind the feature `fake` | `docs/engine-abi.md` |
| `nagaya-store` | canonical safetensors, content ids, recipes, entries, lookup, eviction | 6, D3, D10 |
| `nagaya-ledger` | budgets, terms, admission, shares, reclaim order | 4, D9 |
| `nagaya-sched` | jobs, quanta, classes, the order of ready quanta | 5 |
| `nagaya-mk-qwen38-gfx1151` | the Qwen3.8 engine: its HIP code in `hip/`, the FFI binding, and the `Engine` trait on top | ABI "How the Qwen3.8 engine maps" |
| `nagaya-mk-hip` | the shared HIP ops of gfx1151: hipBLASLt with the plan table, the kernels of Qwen-Image | 12.1 |
| `nagaya-mk-qwenimage21-gfx1151` | the Qwen-Image 2.1 engine on gfx1151: the encoder, the transformer, the VAE | 11.9c |
| `nagaya-mk-bmg` | the megakernel interpreter of the B70 cards: Level Zero, the task protocol, the kernel source, one binary per engine | ABI "How the Gemma 4 engine maps" |
| `nagaya-mk-gemma4-bmg` | the Gemma 4 engine: Level Zero, the OpenCL C kernels, and the host code; a card needs `xe` | ABI "How the Gemma 4 engine maps" |
| `nagaya-council` | council v2 logic with no I/O: the family interface, the sampler, request checks | 11 |
| `nagaya-jinja` | the Jinja interpreter of the chat templates, with the results of Python's jinja2 | 11.3 |
| `nagaya-text-qwen38` | the Qwen3.8 family library: tokenizer, templates, detokenizer, reply parser | 11.3, 11.3a |
| `nagaya-text-gemma4` | the Gemma 4 family library: tokenizer, template, escapes, council rendering, reply parser | 11.3, 11.3a |
| `nagaya-mk-qwen3tts-bmg` | the Qwen3-TTS VoiceDesign engine on one B70 card: weights, the passes as rings, the speech calls | ABI "Speech" |
| `nagaya-text-qwen3tts` | the Qwen3-TTS family library: tokenizer, the prompt of a speech request, the codec ids | 11.3, 11.8 |
| `nagaya-mk-sdxl-bmg` | the SDXL engine on one B70 card: weights and their planes, encoders, UNet, VAE, the sampler | ABI "Image" |
| `nagaya-text-sdxl` | the SDXL family library: CLIP tokenizer, the A1111 prompt, its chunks and weights | 11.9a |
| `nagaya-mk-embeddinggemma2-bmg` | the EmbeddingGemma 2 engine on one B70 card: weights, the passes, the `embed` call | ABI "Embedding" |
| `nagaya-text-embeddinggemma2` | the EmbeddingGemma 2 family library: tokenizer, task prompts, the template id | 11.10 |
| `nagaya-text-qwenimage21` | the Qwen-Image 2.1 family library: tokenizer, the template, the cut of the system ids | 11.9c |
| `nagaya-mk-lfm3-bmg` | the LFM3 engine of both models on one B70 card: the backbone, vision, audio, speech, control vectors, batched passes | ABI "LFM3" |
| `nagaya-text-lfm3` | the LFM3 family library of both models: tokenizer, templates, escapes, XML tool calls, media positions | 11.3, 11.11 |
| `nagaya-text-toy` | the toy family library on gogi's `gogi-toy`, for tests with the fake engine | 11.3b |
| `nagaya-conformance` | the conformance suite (8.3), generic over the `Engine` trait | 8.3 |
| `nagaya-telemetry` | the instruments, the stderr log, the OTLP export and the sampler | 9 |
| `nagaya` | the process: bootstrap config, threads, HTTP, gossip, systemd | 2, 3, 9-11 |

Crate names follow `nagaya-<kind>-<model>-<target>`. The kind `mk` is a megakernel engine. The model is the family
and version without dots. The target is the compiler's device target, for example `gfx1151`. The Intel target is
`bmg`, so the Gemma 4 engine is `nagaya-mk-gemma4-bmg` (Amy, 2026-10-21, by interview). The device count is the
profile's business.

Keep pure logic out of the engine crates and out of `nagaya`. The ledger, the store, the scheduler and the council
logic run in tests with no GPU. The ledger is a plain owned type with `&mut self` methods. Do not wrap it in a
lock: one thread owns it (3.2).

`docs/new-engine.md` is the checklist for a new engine, family, card or machine.

### The lab

The nagaya runs on two machines of Amy's lab today. `docs/deploy.md` tells how to build and deploy on each. The
profiles are in `deploy/`.

- **gfx1151host** has the gfx1151. Qwen-Image is resident, and Qwen3.8 is on demand under the engine name `qwen38`.
- **b70host** has two B70 cards. Its profile is a copy of `deploy/b70-pair/profile.toml`, in the store directory. Its
  iGPU (gfx1036) drives the display: do not run work on it.

The rules for a GPU test, on each machine:

- The nagaya must be the only compute client of its GPUs (spec 3.1). A GPU test is a client too.
- Check the unit first: `systemctl --user is-active nagaya.service`. It is a user unit. The command without `--user`
  says "inactive" while it runs.
- An agent may stop and start `nagaya.service` for a gate or a deploy (Amy, 2026-11-01 for gfx1151host, 2026-11-06 for
  b70host). Write each stop and start in `signoff.md`, with the time and the build. End each session with the
  unit active.
- If another GPU client runs, stop and report. Do not stop a process that is not ours.
- systemd allows 3 starts of the unit in 10 minutes. Put several gates in one window.
- Build tests into another target directory (`CARGO_TARGET_DIR`). The unit runs `target/release/nagaya`, and a test
  build replaces that file.
- Do not mutate a pointer or a size that a kernel reads. Such a mutant faulted a card on 2026-11-07.
- In a container with no systemd, ask your reviewer to check the unit.

An agent may also stop and start the web app on b70host (`python3 tools/nagaya_web.py --host <address> --seed
http://<address>:8091`, port 8096). `signoff.md` gives the address. Write each restart in `signoff.md`.

#### The Qwen3.8 engine on gfx1151

- The engine's HIP code is in `crates/nagaya-mk-qwen38-gfx1151/hip/`. We own it (milestone 15). The kernels stay
  HIP, and the host code is Rust.
- Under the feature `hip`, `build.rs` compiles the code with ROCm's `clang++` and links it statically. A build
  without `hip` needs no ROCm, so a machine without ROCm builds and tests the workspace.
- Keep the compile flags, `-ffp-contract=off -fno-fast-math`: the gates compare bits. `tests/build_flags.rs`
  compares the flags with a fixture.
- The weights are three GGUF shards. GPU tests read them from `QWEN38_SHARDS`, three paths with `:` between them.
  Do not put the paths in code. `signoff.md` gives them.
- The engine has no attach. The export of the arena (spec 10.2) builds it in Rust, from `arena.rs`. For an fd from
  systemd's store, give `/proc/self/fd/<n>`.
- The context calls (`context.rs`) return nothing. The worker maps the result onto the three write results (ABI
  "Capabilities") through the poison flag of the context.
- The C++ runtime layer aborts the process on a failed HIP call (`HIP_CHECK`), and the Rust host stops it on a
  defect. The extern layer `extern.hip` checks nothing, so the worker checks each rule first.
  `host::segments_problem` runs before each pass.
- The HIP device is the device that is current on the calling thread. Make every engine call from one thread.
- Do not free the engine in tests. HIP teardown in a process segfaults (spec 3.3).
- The drop of the last engine handle frees the weights (engine-abi "The close of the Qwen3.8 engine"). A test
  binary that opens many handles keeps a `hold()`, as `tests/hip.rs` does. `tests/hip_close.rs` tests the close.
- GPU tests are behind the cargo feature `hip`, so `cargo test` with no features needs no GPU.

#### The engines on the B70 cards

- The shared interpreter is `nagaya-mk-bmg`. Each engine adds its ops and builds its own task rings. The kernels
  are OpenCL C, and `build.rs` compiles them with `ocloc` for `bmg-g31` into `OUT_DIR`.
- GPU tests are behind the cargo feature `xe` (Decided 2026-10-22), so `cargo test` with no features needs no card.
- Each engine reads its checkpoint folder from a variable with no default ("Tests that need a model file").
- Run the card tests of a crate with one thread, for example:
  `cargo test -p nagaya-mk-gemma4-bmg --tests --features xe -- --test-threads=1`.
- A card test opens only the cards that `NAGAYA_TEST_CARDS` names: PCI addresses with `,` between them. The variable
  has no default, and a test of Gemma 4 needs two cards. Name only a card that no engine of the active profile
  names. Such a test can run with the unit active (Decided 2026-11-15).
- The reference Python of each oracle runs on the CPU. Do not run it with a GPU job.
- A video part needs the feature `video` and ffmpeg's libraries. The deploy of b70host builds with
  `--features xe,video`.

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

The check with `--features xe` needs `ocloc`, because `build.rs` compiles the kernels. Run it on b70host. On a
machine with no `ocloc`, such as gfx1151host, run it with the stub, which writes empty kernels. It checks the xe code;
nothing that it builds runs on a card:

```
PATH="$PWD/tools/ocloc-stub:$PATH" cargo clippy --workspace --all-targets --features xe -- -D warnings
```

Run `cargo fmt --all` often, not only before a commit. On gfx1151host, after a change to `nagaya-mk-qwen38-gfx1151`, also
run this:

```
cargo test -p nagaya-mk-qwen38-gfx1151 --tests --features hip -- --test-threads=1
```

On gfx1151host, after a change to `nagaya-mk-hip`, also run this, with `nagaya.service` stopped:

```
cargo test -p nagaya-mk-hip --features hip --tests -- --test-threads=1
```

On gfx1151host, run this before each commit that changes `crates/nagaya`, a profile of `deploy/`, or the Qwen3.8 crate.
The test file compiles only with the feature `hip`, so no check of the first list runs it. It opens no engine:

```
cargo test -p nagaya --features hip --test qwen38_wire
```

`crates/nagaya/tests/qwen38_stop.rs` opens the engine, and it needs `QWEN38_SHARDS`. Run it only with `nagaya.service`
stopped. Do not run `cargo test -p nagaya --features hip --tests` with the unit active, because it runs that file too.

After a change to `nagaya-mk-qwenimage21-gfx1151`, also run its `hip` tests with the model variables set:

```
cargo test -p nagaya-mk-qwenimage21-gfx1151 --features hip --tests -- --test-threads=1
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
- b70host holds Gemma 4 in `~/models/gemma-4-26B-A4B`. The `xe` tests of
  `nagaya-mk-gemma4-bmg` and the tests of `nagaya-text-gemma4` read it from `GEMMA4_MODEL_DIR`. The variable has no
  default, so the tests of `nagaya-text-gemma4` skip when it is unset.
- Those tests also read HF reference files from `target/gemma4-golden` or `GEMMA4_GOLDEN_DIR`. Write them with
  `tools/gemma4_golden.sh`, with no GPU job. An absent reference file skips the test, as an absent model does.
- b70host holds Qwen3-TTS VoiceDesign in the Hugging Face cache. The tests of `nagaya-text-qwen3tts` and
  `nagaya-mk-qwen3tts-bmg` read it from `QWEN3TTS_MODEL_DIR`, which has no default. `signoff.md` gives the path.
- The oracle of `nagaya-mk-qwen3tts-bmg` reads `target/qwen3tts-golden` or `QWEN3TTS_GOLDEN_DIR`. Write the files
  with `tools/qwen3tts_golden.sh`, with no GPU job.
- b70host holds SDXL base 1.0 in the Hugging Face cache. The tests of `nagaya-text-sdxl` and the `xe` tests of
  `nagaya-mk-sdxl-bmg` read it from `SDXL_MODEL_DIR`, which has no default. `signoff.md` gives the path.
  `tools/sdxl_golden.sh` writes the oracle into `target/sdxl-golden` or `SDXL_GOLDEN_DIR`, with no GPU job.
- The single-file tests of `nagaya-mk-sdxl-bmg` also read `SDXL_SINGLE_FILE`: `sd_xl_base_1.0.safetensors` of the same
  repository. They need `vae_1_0/` of the folder too, because the single file holds that VAE.
- b70host holds EmbeddingGemma 2 in the Hugging Face cache. The tests of `nagaya-text-embeddinggemma2` and the `xe`
  tests of `nagaya-mk-embeddinggemma2-bmg` read it from `EMBEDDINGGEMMA2_MODEL_DIR`, which has no default.
  `signoff.md` gives the path. `tools/embeddinggemma2_golden.sh` writes the oracle into
  `target/embeddinggemma2-golden` or `EMBEDDINGGEMMA2_GOLDEN_DIR`, with no GPU job. The family tests read it too,
  and an absent oracle file skips the test, as an absent model does.
- gfx1151host holds Qwen-Image 2.1 and FUSAL under `/tank/ml/models/qwen-image-2.1/`. The tests of `nagaya-text-qwenimage21`
  read the base folder from `QWENIMAGE21_MODEL_DIR`. The `hip` tests of `nagaya-mk-qwenimage21-gfx1151` read FUSAL's
  folder from `QWENIMAGE21_FUSAL_DIR`, and the base folder too. Neither variable has a default.
  `tools/qwenimage21_golden.sh` writes the oracle into `target/qwenimage21-golden` or `QWENIMAGE21_GOLDEN_DIR`, with
  `nagaya.service` stopped. An absent oracle file skips a test, as an absent model does.
- The report of a check names how many tests skipped for a missing model.

On gfx1151host, run the oracle bench with `QWEN38_SHARDS` set and the unit inactive:
`cargo bench -p nagaya-mk-qwen38-gfx1151 --features hip --bench oracle`. It exits with 1 when a gate breaks.
Set `QWEN38_FAST_QMM=0` for the gate: the exact path must match the reference. Leave it unset to measure the fast path.

The bench prints the SHA-256 of the logits of each prompt. A port step that keeps the bits keeps each hash.

Make the fixture again with the Python of the reference venv (`signoff.md` gives it):

```
PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <python> -I tools/oracle_fixture.py OUT.json
```

It needs about 30 GiB of RAM, and it must not run with a GPU job.

`tools/ste_check.py` checks Markdown only, and it allows 25 words in each sentence. Comments and commit messages
have no check. Keep instructions at 20 words or fewer yourself.

Gate each commit on the exit status of each check. Do not pipe a check into `tail` or `tee` before a commit: the
pipe hides the status. This mistake put three red commits into the history.

### Git

- Commit small steps on the current branch. One commit holds one idea with its tests.
- Commit each time the checks pass after a change (Amy, 2026-11-08).
- Stage each file by name. Do not use `git add -A` or `git add .`. Read `git status --short` before each commit.
- Do not push. Do not amend, rebase, reset or force. Amy reads the history as it is.
- Do not commit `signoff.md`, `target/`, weights or snapshots. Tests write files only under a temporary directory.
  Small test fixtures go under `tests/fixtures/` in their crate.
- A commit message is STE. Its first line is `<crate>: <what changed>`, or `nagaya: <what changed>` when the commit
  changes more than one crate. The body says why and names the mutations.
- The body gives the decision behind the change, from the conversation with Amy, more than a list of the edits.
- End each commit message with a `Co-Authored-By:` line that names the model that wrote the change.
- When a milestone is almost done, run review rounds with kaibo (`docs/reviews.md`), and fix what they find. Use a
  cast of another model family than the one that wrote the change.

## Milestones

Milestones 1 to 30 are in `docs/history.md`, with their rules, their steps and their commits. A comment that says
"`AGENTS.md` milestone N" points there now.

Amy, 2026-11-11: "go ahead and rework milestones in whatever order makes sense now." This is the order (Claude
Fable 5.2, 2026-11-11):

1. Milestone 31 steps 1 and 2 for the Qwen3.8 engine: done on 2026-11-12, and deployed on gfx1151host.
2. Milestone 34, Qwen3.8 catches up, on gfx1151host. It holds milestone 30 steps 7 to 11. Its steps 1 to 6 are done. It
   can run beside the next item.
3. Milestone 31 steps 1 and 2 for the other engines, on b70host. Then steps 3 to 7: the docs, the machines,
   the build, the license and the review.
4. Milestone 31 step 8, the publication, when Amy says so.

Milestone 32, context records, starts in gogi and runs beside these. It changes the contract, so it must be done
before the publication.

These steps of the older milestones are open. Do not start one of them unless Amy asks:

- Milestone 21, image speed on the B70 cards: not started.
- Milestone 24 step 3, the refusal probe: it waits for Amy's photos.
- Milestone 25 step 8 and milestone 26 step 4: trials with Amy.
- Milestone 28 step 5: its first trial ran. More trials are Amy's choice.
- Milestone 29 step 5, the merge, and step 6, the latent swarm: they wait.

### Milestone 30: Qwen3.8 batches and latent steps

Several contexts of Qwen3.8 share one decode pass on gfx1151host, and a context can think in latent steps. Amy started
it on 2026-11-10. `docs/history.md` has its rules and its full steps. Steps 1 to 6 are done and deployed on gfx1151host
from `f7d9ea9`. The engine joins its decode quanta, and its weight arena is locked in memory.

These steps are open. Each step ends with the checks green:

7. **Docs.** The tap hook of the engine in engine-abi: "the residual after layer `l`", with the head's row as the
   last layer. Also its latent rows: the placeholder, the scale and the record. Amy decides each Proposed note.
8. **Oracle fixture.** The residual after each layer, and latent steps with the logits after each, from the
   reference on the CPU (8.3).
9. **The tap hook.** A general hook, not one for the head's row alone. Later milestones read it for control
   vectors, `project` and `trace`. Card gates: a pass with a tap gives the bits of the same pass with none, and the
   rows match the oracle.
10. **The latent span.** Card gate: the latent row of the placeholder's embedding gives the bits of the
    placeholder token.
11. **Route and trial.** `ensemble` with `latent` on Qwen3.8. Deploy, then run a trial with Amy.

Two things from steps 5 and 6 are open too:

- A bound of the barrier wait that counts polls, not time. Its diagnostic faulted the GPU in 5 of 5 stress runs on
  2026-11-10. Decided 2026-11-12: do not hunt that fault unless it happens again. The patch is on gfx1151host, outside
  the repository.
- 8 contexts give no more tokens each second than 4 (46.6 against 47.3), and 16 give 65.0. Decided 2026-11-12: this
  is not a defect. Each row reads 10 of 512 experts in each layer, so more rows read more weights.

### Milestone 31: the scrub for a public repository

Make the repository fit for a public post on GitHub. Amy started it on 2026-11-11, and the quotes are in
"Direction" and "Names". She decided these rules on 2026-11-11:

- **Names:** `mk` stays as the kind in crate names. Every other use of an old name goes, in all code.
- **History:** the public repository starts from a new root commit, later. This repository stays the private
  archive, so its history needs no scrub.
- **License:** MIT, but not now. Step 6 waits for her word.
- **People and machines:** the docs name her and her machines less. A doc that must name her writes @tobert.
- **New names of the Qwen3.8 identifiers:** a variable of the engine takes `QWEN38_`, as in `QWEN38_SHARDS`. A
  variable of the whole process takes `NAGAYA_`, as in `NAGAYA_REQUIRE_MODELS`. Decided 2026-11-12: only the engine
  reads `FAST_QMM` and `TILES`, so they are `QWEN38_FAST_QMM` and `QWEN38_TILES`. A macro of the HIP runtime takes
  `HIP_`, as in `HIP_CHECK`. The engine of the gfx1151host profile has the name `qwen38`. The served model name is the
  file stem of the shards, and it did not change.
- **No old name beside a new one.** Decided 2026-11-12: remove old code and old names. Add no guard and no second
  name. A deploy changes the binary, the unit file and the profile in one window.
- **The merge came first.** The Qwen work of gfx1151host is on `main` since 2026-11-11, so a rename can start.

The rules of the work:

- **No change in behavior.** A scrub commit changes names, text, config and tests. It does not change what an
  engine calculates. The gates give the same results before and after.
- **Know the cost of a rename.** A name in an engine identity, a recipe or a kernel source changes a digest. Then
  each held context fills again once. Find these names first, and say the cost in the commit.
- **One class in one commit.** A commit changes one name or one kind of detail, across the files that hold it.
- **The repository must stand alone.** A fresh clone builds, and its checks pass, on a machine with no GPU and no
  model file. No path points out of the repository.
- **Our lab is an example.** The profiles of gfx1151host and b70host stay as examples of a deploy. No test and no
  tool depends on their names, their addresses or their paths (Amy, 2026-11-10).
- **Private data stays out.** That is Amy's prompts, her photos, captions, session notes and keys. Only counts go
  into git.
- **The publication is Amy's act.** Do not push, do not create a remote repository, and do not rewrite history.

Do the steps in this order. Each step ends with the checks green.

1. **Inventory.** Count each class of detail, and list its files. The classes are old names, people and roles,
   host names, addresses and home paths. They are also license headers, and fixtures that come from model files.
   Write a table of each old identifier and its new name. Amy reads the table and answers the open questions.
   Done on 2026-11-17 in `docs/scrub.md`, which goes at step 8.
2. **Code names.** Remove the old names from comments, test names, identifiers, variables, profiles and the C++
   and HIP code. Do the Qwen3.8 engine first, on gfx1151host, with the `hip` tests and the oracle bench. Do the names
   that change a digest last, in one deploy window on each machine. Tell the kaijutsu session before the served
   model changes its name.
3. **Docs.** Write `README.md` for a stranger: what the nagaya is, the hardware, the build, the checks with no
   GPU, and the status. Remove the old names from the spec, the ABI and the ontology. Move the story of each
   decision to `docs/history.md`.
4. **Machines.** Tests take what they need from config: a card, a model folder, an address. Tools take their
   hosts from flags. `deploy/` becomes a set of examples.
5. **Build.** gogi comes from its GitHub URL, not from a local path. Remove home paths from the fixtures and from
   `Cargo.lock`. Add a workflow that runs the checks that need no GPU.
6. **License.** Add the MIT license file when Amy says so. Keep the notice of each copied source, such as the ggml
   notice of the GGUF code. Write the source of each fixture that comes from a model file.
7. **Review.** Run kaibo review rounds on the whole tree as a stranger reads it. Then clone the tree on a machine
   with no GPU, and run the checks there.
8. **Publication.** Make the new root commit from the scrubbed tree. Amy makes the post.

Each stop and start of `nagaya.service` goes into `signoff.md`.

### Milestone 32: context records

A change of the gogi contract. Amy, 2026-11-11: "of course I want the contract change, gogi is not used by anyone
else yet, let's get it aligned." She chose the shape by interview with the session of the web app.

- The server holds each context as a record, and a content id names it.
- A reply returns the id of "this context and my reply".
- The next request sends `parent` with that id, and only its new messages.
- A miss is a 404 that names `context.parent`. Then the client sends the whole context.

gogi `docs/council-api.md`, "Context records", has each rule. The rules below the shape are senpai's, and each is
Proposed. `gogi-proto` has them as code in `record`: `context_id`, `resolve_request` and `with_reply`.

Its cause is a finding of the swarm trials of 2026-11-10. The driver marked no boundary, so each run fed 60% to
100% of each prompt again. The recipe member `snapshots` marks them now (`53ee84c`).

The goal is not speed. Amy, 2026-11-11: "it was never about speed. more... mechanical sympathy kinda things, and
cutting out round trips it shouldn't need". Thus the measure of a step is what a client sends again, what the
nagaya runs again, and the count of round trips.

Do the steps in this order. Each step ends with the checks green.

1. **gogi.** The contract, `gogi-proto`, `gogi-fake` and the suite checks. Done on 2026-11-11: `8012f82` and
   `d9ce5d8`, merged as `4c063e0`. Several replies are `aa4c089` and `6a8eef4`, merged as `16cea61`. gogi main is
   pushed (Amy, 2026-11-11: "you may push gogi at will").
2. **Spec.** Where the nagaya holds a record, its budget, and its life through a restart (spec 6 and 10). The
   forward of a request that names a `parent` (spec 11.1a). The lab routes that read contexts (`docs/lab.md`). Amy
   decides each Proposed note.
3. **Host code.** Move the pin. Make each context that names a `parent` whole before a council route decodes its
   body. Hold a record of each context that a request reads. Serve `GET /council/v2/contexts/{id}`, the pin of a
   record and the limit `context_records`. Gate: the gogi suite with the toy engine.
4. **The head after a reply.** A generation of LFM3 gives `head: null` now, so the next turn feeds the reply again.
   Amy decided on 2026-11-11: hold the decoded state as a head with its own recipe (spec 6.14, "The decoded
   head"). Do this step after steps 5 and 6. Card gates: a second turn by `parent` feeds only its new message. The
   same request from the same head gives the same bits. The gate measures the difference from a build of the text.
5. **A shared prefix.** Two requests in flight whose contexts begin with the same tokens feed those tokens once
   (gogi "Contexts", "Requests in flight share a prefix"). Thus a client builds no context only for its snapshot.
6. **Several replies.** A generation with `forks` or `contexts` (gogi "Several replies"). Move the pin to
   `16cea61`. The host splits the request with `gogi_proto::replies::split`, runs each reply as a job, and joins
   them. The decode quanta of the replies join in one pass, as milestone 27 built.
7. **Clients.** The chat of the web app sends `parent` and its new message: done, `aff157f`. The swarm driver uses
   `contexts`, `forks` and `parent`, and it loses its build call. Each client sends the whole context after a 404.
8. **Deploy.** Deploy both machines, and run the gogi suite through each. senpai may deploy, update the binaries in
   `~/bin` and restart the nagaya as the steps come together (Amy, 2026-11-11).

Each stop and start of `nagaya.service` goes into `signoff.md`.

Steps 1 to 7 are done on b70host on 2026-11-11, and `nagaya.service` runs the build of `9d71d2c`. The merges are
`1941df0` (records), `a11ccae` (a shared prefix), `831c362` and `72db614` (several replies), and `9d71d2c` (the
decoded head). Live on `lfm3`:

- A second chat turn by `parent` sent 221 bytes. With the decoded head, a turn after a reply of 142 tokens fed 13
  of its 175 prompt tokens.
- 8 generations at once with a new head of 217 tokens fed 343 tokens, where they fed 1,736 before.
- `forks: 8` gave 8 replies in 1.25 s. A swarm round makes 3 requests, where it made 13.
- The card gate `xe_lfm3_head` compares a turn from a decoded head with a build of the text. The next-token log
  probabilities differ by 7.2e-6 at most.

Step 8 is done on gfx1151host too. The build of `9d71d2c` ran there from 2026-11-11, and the gogi suite on Qwen3.8 gave
490 ok, 0 FAIL and 12 skip. Clippy with `hip` runs in the checks of each commit since 2026-11-12.

Open:

- A reply whose decode row shared a pass holds no head, so forks and chats that decode at once get a null head.
- A reply that `max_tokens` cut, and that the next turn closes, has no head for that turn.
- A build that marks an earlier message of a held context can be a 500 (`Service::starts`). It is older than this
  milestone.
- No kaibo review round ran on this milestone.

### Milestone 33: swarms and the stream of thought

Amy chose this list and its order on 2026-11-11: "I like yours let's go with it." Milestone 32 gave the nagaya
context records and several replies, and these steps use them. Do the steps in this order. Each step ends with the
checks green.

1. **Swarms in the web app.** The Chat tab sends one message as N forks with a temperature ladder, and the person
   picks one to go on from. A Swarm tab runs think, select, distill and act, with the person as the judge
   (`docs/swarm-ideas.md`, "A swarm in the chat of the web app"). A session saves and loads as a recipe of
   `tools/swarm`. The page drives the round with `contexts`, `forks` and `parent`.
2. **The stream of thought.** With thinking off, a second turn of `lfm3` writes a first draft, then a second
   `</think>`, then its answer. The reference model does the same (2026-11-11: 6 to 23 of 24 replies). Amy: "think
   of it (hah!) more as being able to dial up the stream of consciousness nature of non-thinking calls to the model;
   I think they'll be handy for creative stuff, and then we'll use thinking where we need it to be precise."
   - Make a control vector from contexts that write a draft and contexts that do not. The trigger is an old reply
     with an empty think block, and an old reply with a reasoning stops it.
   - Measure how often a reply has the draft at each strength, with `add` and with `ablate`. Write the counts.
   - No reply loses text: a stray tag stays in the content, and the client shows it (Amy, 2026-11-11).
   - **Latent steps as another way to think.** Amy: "I'm also curious if we give it some latent steps as an
     alternate thinking option, what changes? I wanna try it." Run the same prompts with thinking off and N latent
     steps before the reply (`docs/lab.md`, "Latent steps in a request"). Compare the count of drafts, the replies
     and the tokens with thinking on and with thinking off. The web app gets a field for the latent steps of a
     chat turn.
   - **Soft inputs.** Amy, 2026-11-12, asked for a middle ground between text and latent steps. The next input
     mixes rows of the embedding table (`docs/lab.md`, "Soft inputs in a request"). Trial `moi` and `gumbel` on
     `lfm3` first, then Qwen3.8 on gfx1151host, then Gemma 4. A stop for latent steps at an end token follows.
3. **One identity on two engines.** `lfm3` and `lfm3-1` have one identity, so they share their snapshots. The gogi
   suite fails "builds the same context into a snapshot of its own". Amy decides: the contract lets equal engines
   share, or the second engine gets an identity of its own.
4. **A repetition penalty.** An optional member of `sample`. It is off unless a request sets it (Amy, 2026-11-11:
   "if we add the repetition penalty, I think I want it optional"). gogi writes it first.
5. **The sampler that a model recommends.** Identity gives the start values of each served model, from its model
   card. Then the chat page reads them, and it holds no values of LFM3.
6. **The response as an object.** A response holds its counts, its stats and its seed under an id. Then a client
   reads them later, as it reads a reply by its record.

### Milestone 34: Qwen3.8 catches up

The Qwen3.8 engine on gfx1151host gets what the LFM3 engine has for the lab: a tap, control vectors and latent steps.
It started on 2026-11-12. The order is by Claude Fable 5.2. These rules were decided on 2026-11-12:

- **The residual.** The state between two layers of Qwen3.8 is 4 streams of 2,560 values. `residual` is the 4
  streams, 10,240 values, for a tap and for a control vector. The head's row is a tap of its own, `fold`.
- **The canary.** The build hash is out of the identity digest (spec 12.1). A canary of the Qwen3.8 engine takes
  its place (spec 8.1). Its record is a table in the crate. A build with other bits does not open until a commit
  records its hash. A set with no record refuses the start. The canary runs at each load.
- **Control vectors.** The op runs on the host first. Then a device op takes its place (step 7).
- **The GPU fault of 2026-11-10** is closed unless it happens again. A small change of the device code faulted
  the GPU in 5 of 5 stress runs, and no cause was found. The machine works hard, and the cause can be many
  things. It does not stop a change of the device code. The stress run stays a gate of each new device op.
- **No old code beside new code.** A deploy is a flag day ("Milestone 31").
- **The J-Lens waits** for hardware. A fit of Qwen3.8 needs more memory than gfx1151host has.

Steps 1 to 6 are done and deployed on gfx1151host (`1471a00`, 2026-11-12). The results:

1. **Repairs.** The load route refuses a load that the engine's memory check refuses (spec 3.3b). A reply to an
   unmarked message holds a head. A second chat turn fed 146 of its 180 prompt tokens before, and 20 of 171 after.
2. **The canary.** 5 tokens, 300 tokens in two passes, and 8 greedy steps as joined passes of 2 segments. It adds
   1.2 s to a load of about 30 s on the fast path, and 4.2 s on the exact path. The fast path gave the same bits in
   5 of 5 processes. Two checkouts now give one identity.
3. **Docs.** `docs/engine-abi.md`, "The residual of the Qwen3.8 engine".
4. **Oracle fixture.** `tests/fixtures/oracle/qwen38_residual.json`, and the full rows in `target/qwen38-golden`
   (`tools/oracle_fixture.py --residual`).
5. **The taps `residual` and `fold`.** A pass with a tap or a control list runs as launches cut at layer ends. The
   host reads or changes rows between two launches. No device code changed. On the exact path the rows match
   the reference to 1.8e-5 of the size of a row.
6. **Control vectors.** A table of 8 entries, and `add`, `ablate` and `clamp` on the host. A decode step under a
   list on 48 layers takes 80 ms, where a plain step takes 63 ms. A prefill of 256 tokens takes 1,865 ms, where a
   plain one takes 499 ms. The cause: each row goes to the host and back at each layer.

These steps are open. Do them in this order. Each step ends with the checks green.

7. **A device op for the three control ops.** Decided 2026-11-12: it is the right form. The host op is its oracle, and
   the gate is equal bits. So the device op keeps the order of the sums: 8 lanes by index modulo 8, joined in pairs,
   with no fused multiply-add. First show that the HIP compiler keeps that order. Two forms, the simpler first: a
   small kernel of its own that runs at the launch cuts, then an op inside the megakernel. Measure a prefill of 256
   tokens again.
8. **The lab routes.** A route chooses a model by what its engine declares, not by its kind. `project` and `trace`
   read the GGUF head and the output fold. The tap `lens` needs 48 logit rows, and the scratch holds 16.
9. **Latent steps.** Milestone 30 steps 10 and 11. A latent step reads `fold`. The placeholder of a latent position
   is the padding token.
10. **Trial and review.** A steering trial and a latent trial with the owner, and a kaibo review round with a cast
    of another model family. No review round ran on this milestone yet.
11. **The J-Lens.** Last, when the hardware is there.

Small things that are open, in no order:

- `crates/nagaya/tests/qwen38_stop.rs` fails with the feature `hip` since 2026-11-01. It reads the gfx1151host profile,
  where Qwen3.8 is now on demand, with a registry of one kind. Give the test a profile of its own.
- Not built for the canary: the fatal state of spec 8.2, the `build` of a store entry, and the canary of the
  other engines.
- A reply that ends in white space, or that the tokenizer would split another way, holds no head.
- A tapped pass allocates on the host: about 2 MB for each logit row.
- `qmmh.h` keeps the buffer-load variants (`P >= 0`) and the field `wbulk`, which only a removed knob chose.
- The memory check before a swap is built for the Qwen3.8 engine only.
- The profile of each machine must not name `pin_identity`. b70host must copy its new profile into its store
  directory at its next deploy, or its unit does not start.

Each stop and start of `nagaya.service` goes into `signoff.md`.

## Working notes

- `signoff.md` is the handoff between sessions and machines. Read it first, and keep it current. It is never
  committed.
- The old signoff, until 2026-11-11, is `signoff-archive-2026-11-11.md` on b70host. It is not committed either.
- Commit messages are in STE and end with the attribution line.
