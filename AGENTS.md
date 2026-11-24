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
| `nagaya-mk-inkling2-bmg` | the Inkling 2 engine of both models on one B70 card: the weight formats, the host form, the ring of a pass, cycles | ABI "Inkling 2" |
| `nagaya-mk-cuda` | the megakernel interpreter of CUDA GPUs: Driver API, the task protocol, the CUDA C++ kernel source, one cubin per engine | ABI "The GB10 Inkling 2 binding" |
| `nagaya-mk-inkling2-sm121` | the Inkling 2 engine on one GB10 GPU: native NVFP4 and MXFP8 activation quantization, host oracle, long-context spec | ABI "The GB10 Inkling 2 binding" |
| `nagaya-text-inkling2` | the Inkling 2 family library of both models: tokenizer, template, effort, escapes, council rendering, reply parser | 11.3, 11.18 |
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

#### The engines on the GB10

- The shared interpreter is `nagaya-mk-cuda`. Its kernels are CUDA C++ with inline PTX, compiled ahead of time by
  `nvcc` to `sm_121a` `.cubin` files into `OUT_DIR` under the feature `cuda`.
- The crate links the CUDA Driver API (`libcuda`) only when `cuda` is enabled. With no feature, `cargo test` needs no
  GPU and no `nvcc`.
- Run GPU tests with one thread (`--test-threads=1`) and with `NAGAYA_TEST_CARDS` set to the free GPU PCI address.
- Check `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` before every GPU test window.

### Checks

Run these before each commit. Each one must pass.

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features hip -- -D warnings
cargo clippy --workspace --all-targets --features xe -- -D warnings
cargo clippy --workspace --all-targets --features cuda -- -D warnings
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

The check with `--features cuda` needs `nvcc`. On a machine with no `nvcc`, run it with `tools/nvcc-stub`:

```
PATH="$PWD/tools/nvcc-stub:$PATH" cargo clippy --workspace --all-targets --features cuda -- -D warnings
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
- gfx1151host holds the Qwen3.8 GGUF. `NAGAYA_QWEN38_GGUF` names its first shard, and it has no default. `signoff.md`
  gives the path.
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
- b70host holds Inkling 2 Mini and the variable-loop research model in the Hugging Face cache. The tests of
  `nagaya-text-inkling2`, `nagaya-mk-inkling2-bmg` and `nagaya` read Mini from `INKLING2_MODEL_DIR` and the research
  model from `INKLING2_VARLOOP_MODEL_DIR`. Neither variable has a default. The B70 unit names the folder of Mini.
  `tools/inkling2_golden.sh` writes the oracle into `target/inkling2-golden` or `INKLING2_GOLDEN_DIR`, with no GPU job.
  An absent oracle file skips a test, as an absent model does.
- The tests `deploy` and `qwen38_wire` of `nagaya` plan the Strix Halo example. They read both Qwen-Image folders.
- The report of a check names how many tests skipped for a missing model.

On gfx1151host, run the oracle bench with `QWEN38_SHARDS` set and the unit inactive:
`cargo bench -p nagaya-mk-qwen38-gfx1151 --features hip --bench oracle`. It exits with 1 when a gate breaks.
Set `QWEN38_FAST_QMM=0` for the gate: the exact path must match the reference. Leave it unset to measure the fast path.

The bench prints the SHA-256 of the logits of each prompt. A port step that keeps the bits keeps each hash.

The reference of Qwen3.8 is the package `tools/qwen38_reference`. Each fixture tool of the Qwen3.8 crates imports it.
Make a venv from its `requirements.txt`, and set `QWEN38_SHARDS`. Then make the fixture again:

```
PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <venv>/bin/python -I tools/oracle_fixture.py OUT.json
```

It needs about 30 GiB of RAM, and it must not run with a GPU job. Test the package from `tools/` with
`<venv>/bin/python -m unittest qwen38_reference.test_reference`.

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

The open milestones, their order and their rules are in `docs/milestones.md`. Read the milestone of your lane
before you start. The finished milestones are in `docs/history.md`.

## Working notes

- `signoff.md` is the handoff between sessions and machines. Read it first, and keep it current. It is never
  committed.
- The old signoff, until 2026-11-11, is `signoff-archive-2026-11-11.md` on b70host. It is not committed either.
- Commit messages are in STE and end with the attribution line.
