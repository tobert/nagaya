# Add an engine, a family, a card or a machine

This doc is the checklist for the next engine, family, card or machine. Each item has its reason. The items come
from the first two ports: Qwen3.8 on gfx1151host (milestones 8 to 14) and Gemma 4 on b70host (milestone 15s). Four
engines came after them, in milestones 17 to 22: Qwen3-TTS, SDXL, EmbeddingGemma 2 and Qwen-Image 2.1. The
rules are in `docs/spec.md`, `docs/engine-abi.md` and `AGENTS.md`. This doc points at them and does not repeat them.

This doc names mkviolet, gogi and kaijutsu. `docs/ontology.md`, "The other projects", defines each one.

Read `AGENTS.md` first. Its rules apply to every step here: a test first, a falsified test, no silent fallback, and
no edit in another repository.

## 1. Before step one

- Interview Amy on the copy list and the order of the ports (`AGENTS.md`, milestone 15). Write her answers into the
  milestone as rules with the mark "Decided <date>".
- Write the spec first. The b70host port began with a spec pass (`f421ae7`, the merge of sg-spec). A rule
  that the docs leave open gets a **Proposed** note with the simplest choice. Amy decides later. Do not choose in
  code.
- Use a reference implementation only to make fixtures. The code comes from the spec and from the crates in this
  tree. Amy, 2026-10-22: "we will archive mk and mkviolet when this is all done; they were our learning projects
  for this and their megakernels live on".
- Work in lanes on a branch for the machine, with one worktree for each lane under `~/src/wt/`. The lanes of 15s
  were `sg-spec`, `sg-probe`, `sg-devices`, `sg-registry`, `sg-gemma-text`, `sg-engine`, `sg-stages`, `sg-cpus`,
  `sg-trait`, `sg-ignore`, `sg-deploy` and `sg-arenas`. Each merge names what it proved.
- Commit the evidence. A merge body names the measurement: "READY in 1233-1251 ms hot, 7130-7138 ms cold"
  (`d2cc40a`). A commit body names each mutation that failed a test (`AGENTS.md`, "Git").

## 2. Names and crates

- A crate is `nagaya-<kind>-<model>-<target>`. The kind `mk` is a megakernel engine. The model is the family and
  version with no dot. The target is the compiler's device target: `gfx1151`, `bmg` (`AGENTS.md`, "Layout").
- The engine kind in a profile is the crate name: `mk-qwen38-gfx1151`, `mk-gemma4-bmg`
  (`crates/nagaya/src/qwen38.rs:12`, `crates/nagaya-mk-gemma4-bmg/src/spec.rs:24`). The engine name in the profile is
  short: `qwen38`, `gemma`.
- A family library is `nagaya-text-<family>`. Its name in the identity is the family: `qwen38`, `gemma4`, `toy`
  (`crates/nagaya-mk-gemma4-bmg/src/spec.rs:30`).
- One engine crate for each vendor and device. Amy, 2026-10-20: "we generate a megakernel per vendor/device and they
  get their own whole crate" (spec 8.3).

A port adds these things:

| Thing | Where |
|---|---|
| the engine crate | `crates/nagaya-<kind>-<model>-<target>/` |
| the family library, when the family is new | `crates/nagaya-text-<family>/` |
| the factory module and the family opener | `crates/nagaya/src/<engine>.rs`, as `qwen38.rs` and `gemma4.rs`, named in `crates/nagaya/src/lib.rs` |
| the kind in the engine registry | `crates/nagaya/src/admission.rs:191` (`Registry::compiled_in`) |
| the family in the family registry | `crates/nagaya/src/serving.rs:53` (`FamilyRegistry::compiled_in`) |
| the cargo feature that registers both | `crates/nagaya/Cargo.toml`, as `hip` and `xe` |
| the fixtures and their tools | `crates/<crate>/tests/fixtures/`, `tools/` |
| the probe tree of the machine | `crates/nagaya/tests/fixtures/probe/<machine>/` |
| the unit, the config and the profile | `deploy/<machine>/` |
| the install, the cutover and the rollback | `docs/deploy.md` |
| the rows and the sections of `AGENTS.md` | section 8 below |

**Pure logic stays out of the engine crate.** The ledger, the store, the scheduler and the council logic run in tests
with no GPU (`AGENTS.md`, "Layout"). The engine crate holds the device code and the trait on it. Its `spec.rs`
holds the plan, the limits and the checks that need no card (`crates/nagaya-mk-gemma4-bmg/src/spec.rs:3`). Reason:
the host checks run on every machine, and a defect in a plan shows before a card is busy.

## 3. The family library

The family library turns text into tokens and tokens into text. The engine never sees text (spec 11.3, D2). One
library serves every engine of its family, on any vendor's card.

**The interface.** The library implements `Family` in `crates/nagaya-council/src/family.rs:297`: `render`,
`decision_read`, `raw_read`, `reply_parser`, `tokenize`, `tokenize_special`, `detokenize`, `pieces`, `template_id`
and `vocab_size`. The vocabulary size is here, not in `limits()` (engine-abi "Logit outputs").

- **Tokenizer.** Read the model's own tokenizer file, or the tokenizer keys of its GGUF. mkviolet read the
  `tokenizer.json` of another model, and its hash named the wrong file (spec 12.3). Hash the whole file, and write
  `sha256:` and 64 hex digits (spec 12.3). The engine computes the same hash, and the registry compares the two at
  start. A difference exits 3 (spec 11.3, "The engine registry").
- **Template.** Take the template text from the model: the GGUF key `tokenizer.chat_template`, or the file in
  `assets/`. Do not write a copy by hand. Run it with `nagaya-jinja`, the one interpreter for every family (spec
  11.3). A feature that the template needs and the interpreter lacks is a step with a test that fails first. Gemma
  4 needed `dict.get` (`crates/nagaya-jinja/tests/dict_get.rs`).
- **Escapes.** Client text that spells a marker is content. Refuse all 66 Unicode noncharacters, and name the
  character and its offset in the error (spec 11.3, "The Gemma 4 choices").
- **Reply parser.** Write it from the rules of spec 11.3 ("The finish", "The reply parser"), with the parsers in
  this tree as the pattern: `crates/nagaya-text-qwen38/src/reply.rs` and
  `crates/nagaya-text-gemma4/src/council/reply.rs`. Name its stop tokens, its phases and the finish map. Test it
  against a fixture of the reference implementation (`tools/reply_fixture.py`).
- **Council rendering.** Follow spec 11.3a with the family's tokens. The code is in this tree:
  `crates/nagaya-text-qwen38/src/council/` and `crates/nagaya-text-gemma4/src/council/`. Keep one form across
  families. A failed tool result renders as `Error: ` and the content, unless the model's template has its own form
  (spec 11.3, "One form in every family").
- **Template id.** `nagaya-council:<hash>+<family>-chat:<hash>` (`crates/nagaya-council/src/template_id.rs`). The
  first hash is over the tokens of a fixed set of test renders that the library keeps. Test that a comment edit keeps
  the id, and that a token change changes it (`AGENTS.md`, milestone 12 step 4). Reason: a client's thresholds are
  fitted to one id.
- **The registry.** The identity's `family` selects the library. The table `[engines.family]` of the profile names
  its files: `gguf` for `qwen38`, `model_dir` for `gemma4`. With no key, the library reads the environment, as the
  engine does (spec 11.3, "Registry choices"). Reason: a new kind needs no change in `serving.rs`.

**Fixtures.** A fixture holds fixed outputs of the reference implementation. A tool in `tools/` makes it, with the
reference's own Python, with no service and no GPU job (`tools/oracle_fixture.py`, `tools/gemma4_text_fixture.py`).

- Run the tool with `PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 <venv python> -I`. The tool reads the reference
  repository and never writes into it.
- The fixture records its source: the revision of the reference and the files it read
  (`crates/nagaya-text-qwen38/tests/fixtures.rs`).
- A crate's fixtures stay under 1 MB in total. This is a convention of the two family crates, not a rule of the
  spec. Each crate has a test for it (`crates/nagaya-text-qwen38/tests/fixtures.rs:24`,
  `crates/nagaya-text-gemma4/tests/model_skip.rs:40`).
- A fixture lives in the family crate when each engine of the family reads it. The oracle fixture is one
  (`crates/nagaya-text-gemma4/tests/fixtures/oracle.json`).

**The oracle.** Each family has one reference that runs on the CPU (spec 8.3, "One oracle for each family"). Its
fixture holds five prompts with the top-64 logits and the logsumexp of the last position, in the format of
`tools/oracle_fixture.py`. Follow these rules:

- Measure the gap first, with a spike. The reference does not give the bits of a GPU, so the oracle has a tolerance.
  The Qwen3.8 gap is 1e-3 on the exact path. The Gemma 4 gap is 1e-2 (spec 8.3).
- Keep the prompts under the length where the gap is explained. Past about 640 tokens, the Qwen3.8 engine and its
  reference differ by 0.29, and the cause is not known (spec 15 item 8). The fixtures stop at 596 and 586 tokens.
- Test that the ids of the fixture are the ids of the library (`crates/nagaya-text-gemma4/tests/oracle.rs`). Then a
  tokenizer change shows in the fixture test, not in the engine gate.
- The engine's bench compares the engine with the fixture and exits 1 when a gate breaks
  (`crates/nagaya-mk-qwen38-gfx1151/benches/oracle.rs`).

**Model files.** A host test that needs a model file skips when the file is absent, and prints one line to stderr.
Only absence skips. A path that exists but does not read fails the test (`AGENTS.md`, "Tests that need a model
file"). The helpers are `crates/nagaya-text-qwen38/tests/common/model.rs`,
`crates/nagaya-text-gemma4/tests/common/model.rs` and `crates/nagaya-mk-gemma4-bmg/tests/common/mod.rs`. Each test
calls `absent()` first.

The variable is one for each family: `NAGAYA_QWEN38_GGUF` has a default path, and `GEMMA4_MODEL_DIR` has none.
Before a merge that touches the crates of a model, run the checks with `NAGAYA_REQUIRE_MODELS=1` on the machine
that holds the model.

## 4. The engine crate

**The shape.** The engine is a Rust type that implements `Engine` (engine-abi "Rust shape"). The core calls are
`identity`, `limits`, `pass` and `last_error`. Each capability is its own trait with one accessor, so an engine has
no stub method. An engine in another language exports a C ABI, and a thin wrapper implements the trait on it (the
Qwen3.8 engine). An engine in Rust implements the trait directly (Gemma 4).

**The factory.** The factory lives in the `nagaya` crate (`crates/nagaya/src/admission.rs`). It has `plan`, `open`,
`arena_parts` and `attach`, and each one gets the group's devices with their PCI addresses (engine-abi "The factory
gets its devices").

- `plan` reads no weights. It reads the config and the tensor headers (`crates/nagaya-mk-gemma4-bmg/src/spec.rs:259`).
  What the code cannot compute goes in the config as a measured value: the Qwen3.8 engine's `weights_bytes` (engine-abi
  "The factory plans from `weights_bytes`"). `open` compares the real size with the plan, and a difference exits 3 with
  the real size in the message.
- The config refuses a key that it does not know (`crates/nagaya/src/gemma4.rs:46-54`).
- The engine finds each card by its PCI address, and it refuses a count that is not its own. It never sees a machine
  device id. `limits().devices` holds group indexes 0 to n - 1 (spec 2.4, "One name for each device"). A Level Zero
  engine maps the address with `zeDevicePciGetPropertiesExt` (`crates/nagaya-mk-gemma4-bmg/src/level_zero.rs:821`).
  Reason: the runtime's device order can differ from the PCI order.
- `limits()` after `open` equals the result of `plan` (engine-abi "Core").

**How the two engines map.** Read engine-abi "How the Qwen3.8 engine maps" and "How mkviolet maps" before you map the
next one. The two engines took different capabilities for the same rule:

| Rule | Qwen3.8 engine | Gemma 4 |
|---|---|---|
| a live state | a context in a slot; `contexts` not bound | a slot; no contexts |
| segments in a pass | 16, `segments_exact: true` | 1, `segments_exact: false` |
| bits and the split of a pass | the same for any split; a checkpoint grain of 256 | depend on the token count; a feed rule, `block` 512 (D11) |
| seal | `qwen38_ctx_round_gdn_bf16` changes bits | none |
| parts | 4 fixed, 2 cells, 1 derived; snapshots whole | `k.<ll>`, `v.<ll>` for 30 layers; cells on global layers, rings on sliding layers |
| coverage | rows 0 to n | per part; a ring holds the last 4,096 positions |
| stages | one | two, one card each; the residual through host memory |
| arena | one fd, full check | 31 DMA-BUF slabs, spot check |
| `max_quantum_ms` | 10,000, its own watchdog | 2,000, measured on the longest stage quantum |
| checkpoint digest | `spot-sha256:` of the shard headers | `blake3:` over the canonical config and each tensor |

- **Determinism.** The engine declares `always` or `pinned` (D3). When the bits depend on the token count of a
  pass, declare a feed rule, `segments_exact: false`, and no grain (D11 rule 4). When the split does not change the
  bits, declare `segments_exact: true` and a measured grain (spec 6.13). Measure each claim on the card.
  The Qwen3.8 engine's grain has two GPU tests. mkviolet's feed rule has a measurement of three splits (D11).
- **Terms.** Declare every byte in `limits().memory`: the weights, the slots, the scratch, the pinned host buffers
  and the residual (spec 4.2). A pass never allocates. Measure a term that the code cannot sum: the Qwen3.8 engine's
  context pool (`examples/context_pool.rs`) and pass scratch (`examples/pass_scratch.rs`). `open` refuses a pool that
  takes more than it declares. Leave room in the budget: card 1 of b70host had about 2.3 GB left beside `gemma` on
  2026-10-21 (`docs/deploy.md`).
- **Identity.** Fill the fields of spec 12.1: `model`, `checkpoint_digest`, `tokenizer_hash`, `template`, `family`,
  `numbers`, the plan table lineage, the bit settings with their class, and `canary_hash`. Today `numbers` holds the
  kernel's SHA-256, the compiler versions and the driver version. The plan table names the PCI ids of the card
  (engine-abi "The first Gemma 4 binding"). The digest does not cover `family`. Decided by senpai 2026-10-22, accepted
  by Amy 2026-11-02: the canary decides for a new build, and the digest will not cover a `build` member. Until the
  canary runs at each start, the build facts stay in the digest (spec 12.1).
- **The family crate.** The engine crate depends on its family crate. The identity's `template` is the family's
  template text, and the engine hashes the tokenizer file with the family's function
  (`crates/nagaya-mk-gemma4-bmg/Cargo.toml:17-18`, `crates/nagaya-mk-gemma4-bmg/src/engine.rs:93`). Reason: one
  hash function, so the registry's check at start compares like with like.
- **Stages.** An engine that spans devices declares `stages` and gives one `StageHandle` for each stage with
  `take_handles` (D1). Each part names its stage, and a handle reads and writes only its parts. After `take_handles`,
  `Engine::pass` refuses. The residual buffers are a mandatory host term, two for each boundary. An engine with
  stages declares no seal and no grain (engine-abi "The stage path in v0"). The process checks the limits with
  `Limits::check` at open (`crates/nagaya-abi/src/lib.rs:420`).
- **Arenas through a restart.** Declare `arena` when the weights can outlive the process (spec 10.2). `arena_export`
  gives one fd for each part, and the process names it `<engine>/device<id>/<part>`. `arena_parts` gives the names
  before the open, so admission counts them against `FileDescriptorStoreMax`. `attach` checks first and opens
  nothing on a mismatch. Give every part a slab: mkviolet stored no head, so each restart uploaded 1.5 GB again.
- **Coverage.** Declare `coverage` when the row parts of a slot hold different rows, as rings do (engine-abi
  "Per-part coverage"). `cells_write` takes `from`, and a write that cannot place its rows refuses and changes
  nothing (engine-abi "The first row of a cells write").
- **Seal.** Declare `seal` only when the boundary rule changes bits. A park never seals (D7).
- **Threads.** Every call to a device comes from one thread: the group thread, or the stage thread of the device
  (spec 3.2). Call the runtime's init on that thread. A thread that a runtime starts keeps the mask of the thread
  that starts it (spec 3.2b).
- **Refuse, do not abort.** A pass that the wrapper refuses returns `PassError`, not `Ok` (engine-abi "A refused
  pass returns an error"). Check every precondition before a C call that aborts the process: the Qwen3.8 engine's worker
  runs `host::segments_problem` before `qwen38_launch_fused` (`AGENTS.md`, "The Qwen3.8 engine on gfx1151"). A bad write
  returns `Refused` and changes nothing, and a failure after a change poisons the state (D6).
- **Close.** An engine whose runtime teardown segfaults declares no `close`, and its `Drop` frees nothing
  (engine-abi "Capabilities"). The binary ends through one exit path with `_exit` (spec 10.4, "One exit path").
- **The wake call.** A card that idles between the quanta of a job loses speed at its next command. The call and
  its shape are decided, and it is not built (engine-abi "The wake call", "The shape of the wake call").

**Copy first, then port.** Copy the reference engine with no change in behavior, and prove the bits before the
trait. The Gemma 4 trait test keeps the hashes of the copy's logits in the test
(`crates/nagaya-mk-gemma4-bmg/tests/xe.rs`, `the_trait_gives_the_bits_of_the_copy`). The golden tests of the
reference move into the crate, one test for each file. Device memory is free only when the process exits, so two
engines do not fit (`crates/nagaya-mk-gemma4-bmg/tests/common/mod.rs`). Then port host code one part at a time,
each with the gates green (`AGENTS.md`, milestone 15).

## 5. The build

- **One feature for each engine.** The engine crate has a feature for its device: `hip`, `xe`. The `nagaya` crate
  has a feature of the same name that registers the kind and the family (`crates/nagaya/Cargo.toml`). With no
  feature, `cargo test --workspace --tests` needs no GPU and no compiler of the vendor (`AGENTS.md`, "Checks").
  Reason: a machine without the card builds and tests the workspace.
- **What builds with no feature.** The config parse, the plan, the limits and the family opener build with no
  feature, so the host tests cover them (`d503317`).
- **`build.rs` compiles the kernels.** It runs the vendor's compiler only when `CARGO_FEATURE_<X>` is set, and it
  links the runtime library only then (`crates/nagaya-mk-gemma4-bmg/build.rs`). It writes the compiler versions into
  environment variables for the identity: `GEMMA4_OCLOC_VERSION`, `GEMMA4_IGC_REVISION`. It names the source in
  `rerun-if-changed`. A compiler that is missing stops the build with a message that names the feature.
- **Keep the reference's float flags.** The gates compare bits, so the kernels keep the flags of the reference. For
  the Qwen3.8 engine, they are `-ffp-contract=off -fno-fast-math` (`AGENTS.md`, milestone 15). For Gemma 4, they are
  mkviolet's `ocloc` options (`crates/nagaya-mk-gemma4-bmg/build.rs`).
- **No CMake.** `build.rs` calls the compiler itself (`AGENTS.md`, milestone 15).
- **A static link.** Both engines link at build time. The Gemma 4 engine links Level Zero. The Qwen3.8 engine links its
  own HIP code statically, with its own flags, and a test checks the flags (`crates/nagaya-mk-qwen38-gfx1151/build.rs`,
  `tests/build_flags.rs`). Build objects to stdout from the crate directory, so the archive digest does not take in
  the build directory.
- **`serde_json` features.** Cargo joins the features of one build, so a workspace build has `preserve_order` and
  `float_roundtrip` (spec 11.3a). A build of one crate alone has only what its own `Cargo.toml` names. Every crate
  that parses or writes tool schemas or config JSON names both features in its own `Cargo.toml`.
- **The clippy check with the feature** runs only on a machine that has the compiler: `ocloc` for `xe`
  (`AGENTS.md`, "Checks").
- **The workspace members.** Add the crate to `members` in the workspace `Cargo.toml`. Every crate under `crates/`
  is a member of `Cargo.toml`.

## 6. Tests

**The conformance gates** (spec 8.3) are generic over `Engine` in `crates/nagaya-conformance/src/lib.rs`:
`park_and_restore`, `seal_only_at_boundaries`, the four quantum gates, `admission_refusal_changes_nothing`,
`segments_exact_bits`, `stage_path_bits` and `feed_rule_bits`. `run_all` makes a fresh engine for each gate. A
gate that an engine's limits do not reach gives "not applicable".

- Each new gate adds its own broken fake, which the gate must fail (engine-abi "Fake engines"). `FakeFeed` has a
  feed rule in one stage or two, and its broken variants reverse the residual or read their history.
- A gate that restores state restores it through `Store::restore`, the path that production uses (spec 8.3).
- The suite compares by integer bit pattern. NaN fails, also against NaN (spec 8.3).
- Run the suite on the card in the engine's GPU test (`crates/nagaya-mk-qwen38-gfx1151/tests/hip.rs`,
  `crates/nagaya-mk-gemma4-bmg/tests/xe.rs`).

**The host layers.** Each port adds tests that need no card:

- the plan, the limits and the checks, as unit tests in the engine crate's `spec.rs`;
- the factory: the config, the plan with no card, and the family opener (`crates/nagaya/tests/gemma4_factory.rs`);
- the text routes with the real tokenizer and template of the family (`crates/nagaya/tests/extensions_qwen38.rs`);
- the deploy files, read with the real loaders (`crates/nagaya/tests/deploy.rs`);
- the probe tree of the machine, with the check that all trees stay under 200 KB
  (`crates/nagaya/tests/probe.rs:229`).

**GPU tests.** They sit behind the feature, and they run with `--test-threads=1`, because the tests share one engine
and the weights fill the card (`crates/nagaya-mk-gemma4-bmg/tests/xe.rs:4`). Before a run, check that the old
service and the nagaya are inactive, and never stop a service yourself (`AGENTS.md`, "The Qwen3.8 engine on gfx1151",
"The engines on the B70 cards"). Do not close the engine in a test when the runtime's teardown segfaults (spec 3.3).

**Process tests.** A process test starts the binary with the profile of the machine on a free loopback port. It
serves a decision and a generation, and it runs gogi's conformance suite (`crates/nagaya/tests/xe_process.rs`). A
restart test
plays systemd: it keeps the fds of `FDSTORE=1` and starts the next process with them at fd 3 and up
(`crates/nagaya/tests/xe_restart.rs`).

**The oracle bench.** Run it on the exact path for the gate, and on the fast path for the numbers (`AGENTS.md`,
"Checks"). It opens the engine once and exits 1 when a tolerance breaks.

**The report.** Name the tests run, the tests passed, and the tests that skipped for a missing model. Name each
thing that you chose not to do (`AGENTS.md`, "Rules"). On the card, give the numbers: `gogi-conformance` gave 215
ok, 0 FAIL and 8 skip on b70host (`docs/deploy.md`).

## 7. The machine

**The inventory probe** reads files under a root path and opens no GPU runtime (spec 2.1, "The probe reads files").
For a new machine:

- Capture its tree into `crates/nagaya/tests/fixtures/probe/<machine>/`: only the files that the probe reads, with
  symlinks resolved. Keep the trees well under 200 KB in total; a test checks it
  (`crates/nagaya/tests/fixtures/probe/README.md`).
- A tree made by hand marks each guessed value, and the test for that machine shows each guess when a capture
  replaces it. The sm121host tree is such a tree.
- Know what the probe cannot see for the vendor (spec 2.1, "What the file probe cannot see"). NVIDIA gives no
  memory size without NVML. Xe has no size file, so the probe reads a BAR. AMD needs two files to tell unified from
  dedicated memory. The probe gives a true size or stops.
- The probe has code for each vendor: `Vendor` (`crates/nagaya/src/probe.rs:13`), `amd_devices` (`:310`),
  `xe_devices` (`:391`) and `nvidia_devices` (`:456`). A new vendor is a step in that file, with a test on its tree.
  A new NVIDIA model goes in `UNIFIED_NVIDIA_MODELS` (`:516`) when its memory is unified. The probe refuses another
  model and names the list (spec 2.1, "What the file probe cannot see").
- Read a PCIe link from the upstream port of the card's switch. The endpoint of a B70 reports 2.5 GT/s x1 behind
  its switch (spec 2.2).

**Devices.** The device ids follow the PCI address for every vendor (spec 2.1). A GPU that drives the display goes
in `ignore_devices` of the bootstrap config. It keeps its id and stays in the inventory. It gets no budget, no
thread and no work. An address that the probe does not find exits 3 (spec 2.1, "Ignored devices"). Set no runtime
variable that hides a card, such as `ZE_AFFINITY_MASK` (`deploy/b70host/nagaya.service:29`).

**CPU sets.** The housekeeping set is the two lowest online CPUs, and the pool set is the rest (spec 3.2a). A machine
with two cards or more gives each stage thread a set of its own, from `local_cpulist` or one L3 domain (spec 3.2b).
A machine with one card keeps the pool set. Measure the masks of the threads that the runtime starts
(spec 15 items 14 and 17).

**The profile** (spec 2.4) has three tables. Each `[[engines]]` entry has `name`, `kind`, `devices` (machine device
ids), `config`, `share` and `pin_identity`. Each `[[budgets]]` entry has the tiers of one pool and its bytes. The
`[store]` table has the store's budgets.

A machine with unified memory pools `device(0)` and `host` in one budget. A machine with discrete cards has one
budget for each card and one for host memory (`deploy/b70host/profile.toml`). The host safety floor of spec 4.7
is a runtime check of `MemAvailable`, 12 GiB in the Qwen3.8 engine. It sets no size, and the ledger does not check it
yet (`docs/spec.md:578`).

The two profiles keep a margin of their own, and each says why in a comment. gfx1151host's pool is 112 GiB of 125.1 GiB
(`deploy/gfx1151host/profile.toml:22-24`). b70host's host budget is 96 GiB of 123.5 GiB
(`deploy/b70host/profile.toml:34-38`). The test `crates/nagaya/tests/deploy.rs` reads the real files with the
real loaders.

**The unit.** One unit file for each machine, in `deploy/<machine>/` next to its config and profile (spec 10.1).
Copy gfx1151host's or b70host's unit and keep its comments. Set `FileDescriptorStoreMax` and
`FileDescriptorStorePreserve=restart` when the engine declares an arena. Set the model variable of the family in
the unit. Keep the OTLP line a comment until a collector listens (`deploy/b70host/nagaya.service`).

**The only GPU client** (spec 3.1). Before a GPU test and before the cutover, check that the old service and
`nagaya.service` are inactive, with `--user`. List the holders of the render nodes with `fuser`. The display
server holds the iGPU, and that is correct (`docs/deploy.md`, "Cut over from mkviolet").

**The cutover.** Write the install, the cutover and the rollback into `docs/deploy.md` before the first start. The
steps of the two cutovers were these:

1. Check the GPU clients.
2. Enable the unit, and read the log.
3. Ask for the identity.
4. Run `gogi-conformance`.
5. Read the kernel log for card errors.
6. Check that the clients' fitted thresholds match the served `template` id. The id changes only when the rendered
   tokens change (spec 11.3a). kaijutsu's gate holds such thresholds (`docs/deploy.md`, "The kaijutsu gate").

Senpai does the cutover with Amy. Do not start, stop or enable a service yourself (`AGENTS.md`, milestones 14 and
15s).

## 8. The `AGENTS.md` updates

- Add the crate rows to the table under "Layout", with the spec sections of each.
- Add the clippy check with the new feature under "Checks", and say where it runs. Add the GPU test command with
  `--test-threads=1`.
- Add a machine section, as "The Qwen3.8 engine on gfx1151" and "The engines on the B70 cards". It names the reference
  repository as read-only, the two services to check, and the display device to leave alone. It names the model
  variable, the feature and the reference venv.
- Add the model to "Tests that need a model file": which machine holds it, the variable, and its default or the lack
  of one.
- Add the milestone with its steps in order, and the rule that the cutover is senpai's with Amy.

## 9. Open

These items are places where the docs and the code do not agree, or where a choice is still Proposed. This doc does
not choose. Amy decides.

- Spec 8.3, "The Gemma 4 oracle" (`docs/spec.md:1608-1609`), says that a bench reports the Gemma 4 oracle numbers.
  No test or bench in `crates/nagaya-mk-gemma4-bmg` reads `oracle.json`. Only
  `crates/nagaya-text-gemma4/tests/oracle.rs` reads it.
- Spec 6.14: drafts are a setting of the served model (Decided 2026-10-28). A test on the B70 cards must still show
  that drafts change no bit of Gemma 4. `f1ab9ad` decided the notes on the build in the digest, the wake call and the
  pause rule. The test stays open.
- Three copies of the model-file helper exist (section 3). `AGENTS.md:618` names only the Qwen3.8 copy. Where the
  helper of the third model lives is not decided.
