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
- Do not edit other repositories. mk's HIP code is now in `crates/nagaya-mk-qwen38-gfx1151/hip/`, and we own it
  (milestone 15). mkviolet stays a reference: read it, and do not edit it.

### Layout

The crates live in directories under `crates/`. Use these crates:

| Crate | Holds | Spec |
|---|---|---|
| `nagaya-abi` | the `Engine` trait, its types, and a fake engine behind the feature `fake` | `docs/engine-abi.md` |
| `nagaya-store` | canonical safetensors, content ids, recipes, entries, lookup, eviction | 6, D3, D10 |
| `nagaya-ledger` | budgets, terms, admission, shares, reclaim order | 4, D9 |
| `nagaya-sched` | jobs, quanta, classes, the order of ready quanta | 5 |
| `nagaya-mk-qwen38-gfx1151` | the mk engine: mk's HIP code in `hip/`, its FFI binding, and the `Engine` trait on top | ABI "How mk maps" |
| `nagaya-mk-hip` | the shared HIP ops of gfx1151: hipBLASLt with the plan table, the kernels of Qwen-Image | 12.1 |
| `nagaya-mk-qwenimage21-gfx1151` | the Qwen-Image 2.1 engine on gfx1151: the encoder, the transformer, the VAE | 11.9c |
| `nagaya-mk-bmg` | the megakernel interpreter of the B70 cards: Level Zero, the task protocol, the kernel source, one binary per engine | ABI "How mkviolet maps" |
| `nagaya-mk-gemma4-bmg` | the mkviolet engine: Level Zero, the OpenCL C kernels, and the Gemma 4 host code; a card needs `xe` | ABI "How mkviolet maps" |
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
| `nagaya-text-lfm3` | the LFM3 family library of both models: tokenizer, templates, escapes, XML tool calls, media positions | 11.3, 11.11 |
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
is done on gfx1151host, and milestones 15s, 16, 17, 18, 19 and 20 are done on b70host. Milestone 21 waits, and
milestone 22 is done on gfx1151host.

Milestone 23, the cluster, is done on both machines. Milestones 24, LFM3 Omni, 25, control vectors, and 26, the lab
tools, are deployed on b70host, and each has open steps. Senpai chose this order on 2026-11-08 (Amy: "you may decide
order"):

1. Milestone 26 step 5, the J-Lens. Its fit runs on the CPU while the other steps run. The host code is done, and
   the lens deploys after the fit.
2. Milestone 25 step 9, Omni's control vector. Done on 2026-11-08: `5b26dcd`, deployed with `dd5b9b5`.
3. Milestone 24 step 9: media inputs and speech outputs of `lfm3-omni` through council v2. Done on 2026-11-08:
   lanes J and K of 2026-11-06 merged (`c16934d`, `c9c0bea`), and media in decisions (`dd5b9b5`). The gogi suite
   gave 399 ok, 0 FAIL and 18 skip for `lfm3-omni`, and 380 ok, 0 FAIL and 18 skip for `lfm3`.
4. Milestone 24 step 10, video, and step 11, the Omni tab of the web app. Done on 2026-11-08: video `3cfeff1` to
   `c52287a`, deployed with `--features xe,video`; the card gate of both videos gives the oracle's first token. The
   tab is `a9c476e`. Step 12, the read back of an assistant audio part, is done too: `c312d3d` to `b6fe817`, deployed
   at 16:15 on 2026-11-08.
5. Milestone 25 step 8 and milestone 26 step 4, with Amy. Milestone 24 step 3 waits for her photos.
6. Milestone 27, batched passes on LFM3. Amy put it next on 2026-11-09. Done on 2026-11-09.

Amy decided on 2026-11-08: commit a small step each time the checks pass after a change. When a milestone is almost
done, run review rounds with kaibo (`docs/reviews.md`), and fix what they find.

senpai may stop and start the web app on b70host at will (`python3 tools/nagaya_web.py --host 198.51.100.10`, port
8096). Write each restart in `signoff.md`.

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

    Done on 2026-10-23 at 02:28. The load and the PLE rows (`arena.rs`, `ple.rs`), the contexts (`context.rs`), and
    the pass (`pass.rs`, `host.rs`) are Rust. The C++ that is left is the kernels, `rt.cpp`, and the launches in
    `engine.hip` and `extern.hip`. `build.rs` calls ROCm's `clang++` directly, as mk's CMake did, because `hipcc` is
    a wrapper.

    At each step the oracle bench gave the same five logit hashes, and the arena digest stayed the same.
    `nagaya.service` runs the build of `92b071f`, and `gogi-conformance` gave 274 ok, 0 FAIL and 7 skip. The merges:

    - step 1, the copy and the static link: `dd89a9b`;
    - step 2, the op tests: `2f65fae`;
    - step 3, the GGUF reader: `4aadc34`;
    - step 4, the C ABI: `2c710aa`;
    - step 5, the stage schedule: `11493fe`;
    - step 6, the host code: `92b071f`.
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

    Senpai and Amy did the cutover at 11:01 on 2026-10-23, from `5d7c5eb`. `nagaya.service` serves council v2
    on `198.51.100.10:8091` with Gemma 4 on the two B70 cards, ready after 7 s. `mkviolet.service` stays disabled.
    `gogi-conformance` at gogi `a07d85e` gave 266 ok, 0 FAIL and 11 skip. The kernel log has no hang or reset.
    b70host has no kaijutsu gate config, so the gate check has no target here.
16. **Engine swap in place.** The nagaya loads and unloads an engine while it runs, with no restart. Amy decided
    this on 2026-10-24, by interview, so that voice and image engines can use the B70 cards when Gemma 4 is idle.
    - **On demand:** the nagaya swaps by itself. A request for an engine that is not loaded starts the swap, and the
      request waits for the load (Amy: "Nagaya swaps itself").
    - **Order:** the swap is milestone 16, and voice is milestone 17.
    - **Cards:** senpai may stop and start `nagaya.service` on b70host for GPU tests in milestones 16 and 17. Write
      each stop and start in `signoff.md`.

    Do the steps in this order:
    1. **Spec.** Write the swap rules into spec 1, 2.4, 3.3, 4, 10 and 11, and into engine-abi `close`. Amy decides
       each Proposed note.
    2. **Close.** Measure `close` on the Gemma 4 engine on the B70 cards. Open it, run a pass and close it many
       times, and check the free device memory each time. Declare `close` when the memory comes back.
    3. **Ledger.** Release the terms of one engine, and admit one engine into a running ledger.
    4. **Group.** Unload and load an engine on its group thread, with the fake engines first.
    5. **On demand.** A job for an unloaded engine starts the swap, and an idle engine swaps back. Test it with the
       fake engines.
    6. **Routes.** Show the states `loading` and `unloaded` on the HTTP routes and on the status feed.
    7. **Gates and deploy.** Run the gates on the cards, then deploy on b70host.

    Done on 2026-10-24 at 20:09, on b70host. A drop of the Gemma 4 engine frees its device memory, and it declares
    `close`. On the cards, a swap to an on-demand engine took 244 ms and freed 56.9 GB. The swap back and a generation
    took 8.0 s, with the same reply. `nagaya.service` runs the build of `ff197d2`, and `gogi-conformance` gave 266 ok,
    0 FAIL and 11 skip.

    No Proposed note of this milestone is open (2026-11-02). The commits:

    - step 2, close: `a82e59f`;
    - step 3, the ledger check: `589bfbc`;
    - step 4, residency, admission and the swap on a group: `043b02a`, `c542f9e`, `5be9a0c`, `36c6162`;
    - step 5, the swap back: `7421fa5`;
    - step 6, the states: `1efdad5`;
    - step 7, the ledger of an unloaded engine and the swap on the cards: `ff197d2`.
17. **Voice.** Serve Qwen3-TTS from an engine on the B70 cards. Amy chose the model on 2026-10-24 after she
    listened to CPU samples. She decided these rules on 2026-10-24, by interview:
    - **Weights:** `Qwen3-TTS-12Hz-1.7B-VoiceDesign` only. A text prompt designs each voice.
    - **Requests:** one request is a line or a paragraph. The agent orders the requests and paces them. The nagaya
      has no stream for speech. A speech job can wait behind other work: "that's by design, it should be getting
      enough back to stay ahead".
    - **Replies:** the reply sends the audio in chunks while the frames decode.
    - **Route:** `POST /v1/audio/speech`, in the shape of OpenAI's speech API.
    - **Kernels:** the opcode interpreter of the Gemma 4 engine becomes a module that both engines use. The speech
      engine adds its ops and builds its own task rings.
    - **Codec:** the codec decoder runs on the card from the start.
    - **Placement:** the speech engine swaps in on demand (milestone 16). The placement profile chooses its card.
    - **Cards in one group:** a pass of one engine can run on one card while a stage of another engine runs on the
      other card. Write this rule now (Amy: "Yes, design it now").
    - **Voice drift:** each request designs its voice again, so the voice drifts a little between requests. Amy:
      "a bit of drift, hmm, inevitable for now, let's let it be and fix up later".

    The model name in the crate names is `qwen3tts`. Do the steps in this order:
    1. **Spec.** Write the speech rules into the docs. They are the route, the outputs of the engine, the sampler of
       the audio codes, and the passes on separate cards (engine-abi D1). Amy decides each Proposed note.
    2. **Oracle fixture.** Write the fixed inputs and outputs from the Python reference once, on the CPU (8.3).
    3. **Family.** Write `nagaya-text-qwen3tts`: the tokenizer and the template of a speech request.
    4. **Shared interpreter.** Move the interpreter of the Gemma 4 engine into a shared module. The Gemma 4 gates must
       give the same bits.
    5. **Ops.** Add the ops of the talker, the code predictor, the sampler and the codec decoder, each with a test.
    6. **Engine.** Write `nagaya-mk-qwen3tts-bmg` and its `Engine` trait. Run the conformance gates and the oracle.
    7. **Route.** Serve `POST /v1/audio/speech`, with the audio in chunks.
    8. **Cards in one group.** Build the passes on separate cards in one group.
    9. **Deploy.** Add the speech engine to the profile of b70host as an engine on demand.

    Done on 2026-10-24 at 22:18, on b70host. The forced oracle on the card gave relative errors of 7.3e-7 for the
    talker logits and 9.6e-7 for the audio. The live route swapped gemma out in 691 ms and sent its first byte after
    1.2 s. The swap back took 7.2 s. `nagaya.service` runs the build of `3fd481d`, and `gogi-conformance` gave 266
    ok, 0 FAIL and 11 skip. The speech engine plans 5.6 GB on card 0, and gemma leaves 4.4 GB, so they swap.

    No Proposed note of this milestone is open (2026-11-02). The commits:

    - steps 1 to 3, the spec, the oracle and the family: `cc7ffa7`, `529cf4f`, `5d78334`, `94760ba`, `5600fda`;
    - steps 4 and 5, the shared interpreter and the ops: `44a5545`, `b4c53a8`, `9610748`, `dadbde0`;
    - step 6, the engine: `804c3f3`, `91bd122`;
    - step 7, the route: `1d3e11d`, `91728f2`, `7d46c93`;
    - step 8, the cards in one group: `30143a5`, `0b15ba3`, `83d6f6c`;
    - step 9, the deploy: `3450734`, `12d6a9e`, and the stall fix of milestone 16, `3fd481d`.
18. **Media.** Council v2 carries media as objects with `blake3:` ids. The B70 cards do image generation "mostly
    for a while" (Amy, 2026-10-25). Council moves to gfx1151host's Strix Halo later, and Gemma 4 goes on demand on
    b70host. Amy decided the plan on 2026-10-25 and 2026-10-27, by interview.
    - **Image model:** start with SDXL base 1.0. Amy, 2026-10-27: "could we start with a general sdxl and pony etc
      would be easier later?" Pony Diffusion V6 and its merges, such as AutismMix, are SDXL fine-tunes. They come
      later as other weights on the same engine.
    - **Weights:** the loader reads the diffusers folders and the single file with Stability's tensor names. Thus a
      Civitai checkpoint loads with no new code.
    - **No refiner.** Pony and its merges do not use it.
    - **Few steps:** Lightning, DMD2 or Turbo weights come after the engine, as other weights and settings. Read the
      license of each first.
    - Qwen-Image 2.1 with FUSAL (`~/src/diffusepal`) stays a later engine for batch work.
    - **Route:** council v2 only. An image is an output object, with a pending id while it decodes.
    - **Prompt weights:** the family parses A1111 weights, such as `(word:1.2)` and `[word]`. Thus a prompt from
      Civitai works as it is.
    - **Previews:** a request can ask for a preview at each step. First a linear map from latent to RGB, and TAESD
      later as a second choice.
    - **VAE:** the VAE that the checkpoint ships, in f32. Amy, 2026-11-02: "seems my kinda rule is we follow the best
      ergonomics we can find for the model+gpu" changed the first rule, "the stock VAE, not the fp16 fix".
    - **Samplers:** Euler, Euler a, DPM++ 2M Karras and DPM++ 2M SDE Karras.
    - **Batch:** a request asks for at most 4 images, and each is its own output object.
    - **Cards:** one image engine on each card, and each engine is its own served model. Amy, 2026-10-27: "if higher
      quality models fit on these cards, we'll run them in parallel, one model per card". Gemma 4 spans both cards,
      so they are one group. A device thread for each card runs them at once (spec 3.2, Decided 2026-10-28).
    - **Request:** Amy decided the rest on 2026-10-28, by interview (spec 11.9). The rules name the prompt, the PNG,
      the seed, the previews, the sizes, the quanta, the defaults and `clip_skip`. img2img, inpainting, LoRAs and a
      hires pass are designed, not built.
    - **Loads:** a request never starts a load (spec 3.3, Decided 2026-10-27). A client loads an engine with a
      profile `PUT` or a load route. `POST /v1/audio/speech` goes.

    Do the steps in this order:
    1. **Objects.** Objects and keeps in `nagaya-store`, and their routes (spec 6.15). Done.
    2. **Speech in council.** The speech engine as a council served model, with held state (spec 11.8a). Done.
    3. **Explicit loads.** A request for a served model that is not ready gets a 503 `model_unavailable`. Build
       the load routes (spec 3.3b) and the profile `PUT` in place (spec 3.3). Remove `POST /v1/audio/speech`.
    4. **Spec.** Write the image rules into the docs: the route, the outputs, the `step` quanta, and the image engine
       in engine-abi. Amy decides each Proposed note.
    5. **Oracle fixture.** Write the fixed inputs and outputs of SDXL base from diffusers once, on the CPU (8.3).
       Keep the image small.
    6. **Family.** Write `nagaya-text-sdxl`: the CLIP tokenizer and the prompt of an image request.
    7. **Ops.** Add the 2D convolution, GroupNorm, attention, and the up and down samplers to the shared interpreter.
       Give each op a test.
    8. **Engine.** Write `nagaya-mk-sdxl-bmg`: the text encoders, the UNet steps and the VAE decode. Run the
       conformance gates and the oracle.
    9. **Route.** Serve images through council v2, with each image as an object.
    10. **Deploy.** Add an image engine on each card to the profile of b70host, each on demand.

    The commits so far:

    - before step 1, the council contract with `blake3:` ids and media: `3a5c106`, `5b4fece`, `8173c95`, `e9eb276`,
      `abef7af`;
    - step 1, objects: `69a4f0d`, `6469b63`, `5eaed78`;
    - step 2, speech in council: `e57eaeb` to `7c414b0`, deployed at 17:53 on 2026-10-25;
    - Amy's decisions of 2026-10-27: `6d61ea2`, `483b81c`, and the gogi pin `af34fa7`.
    - step 3, explicit loads, the load routes, the profile routes and the swap states in gossip: `70a3f1a` to
      `5a80b25`, deployed at 10:54 on 2026-10-28;
    - step 4, the image rules: `362f3fc`, `00f2682` and Amy's decisions `db5f5f6`;
    - step 5, the oracle of SDXL base from diffusers: `15b94b6`. `tools/sdxl_golden.sh` writes it.
    - step 6, the family `nagaya-text-sdxl`: `a9c1ce7`, and Amy's decision on the ids `061efed`;
    - step 7, the image ops on the shared interpreter: `8dc7277`. The card tests are in
      `crates/nagaya-mk-bmg/tests/xe_sdxl_ops.rs`.
    - step 8, the engine: `38ae84c`, `fab01eb`, `a7d0b09`, `cae1560`, and the matrix units `c13d5a6`. A step of a
      1024 by 1024 image takes 1.77 s. `tests/xe_oracle.rs` compares it with diffusers on the card.
    - step 9, the route: `e4c5c87`, `4557225`, `eb3c12e`, `158383f`, and the admission fix `16a0107`;
    - step 10, the deploy: `04299e6`, deployed at 14:58 on 2026-10-28. `sdxl-0` and `sdxl-1` are on demand.

    - after step 10, Amy's decisions `10b4bcd`, and gogi `48d2d1a` with its pin `ed07837`;
    - the hashes from the open, `e21f3d6`, and the single file, `baddb59` (Proposed) and `a8a27f3`;
    - the device threads of spec 3.2, `48c3370`;
    - the fixes of a review by a fresh Claude session and kaibo: `47e31bb`, `a96b8ce` and `2985888`.

    Done on 2026-10-28, on b70host. `nagaya.service` runs the build of the deploy commit of `signoff.md`. Live, an
    image of 1024 by 1024 with 30 steps takes 57 s on each card, and the two cards run at once. The gogi suite gave
    343 ok, 0 FAIL and 15 skip for gemma, and 88 ok, 0 FAIL and 2 skip for `sdxl-0`.

    Amy accepted the single-file checkpoint of `baddb59` on 2026-10-28.

    After the deploy, on 2026-10-29: Lightning and LoRAs. `lightning-0` serves SDXL-Lightning 4-step on card 0.
    The commits are `348a3ea` to `afe40b3`. On 2026-10-30, two limits of M18 were fixed on the host:

    - A single-card swap runs on the thread of its card, so the other card does not wait (`9a0294b`). An engine on
      both cards still swaps on the group thread, and Amy accepted that.
    - A full queue of a slow image stream ends the stream, and the job seals its images (`3aad340`). A slow speech
      reader never held the group thread (spec 11.8a).
19. **Embeddings.** Serve EmbeddingGemma 2 for the semantic search of kaijutsu. Amy decided these rules on 2026-10-30,
    by interview:
    - **Route:** council v2 gets a native route, `POST /council/v2/embeddings`. gogi writes the contract first.
    - **Task:** the request names its task, for example `query` or `document`. The family adds the task prompt of
      the model. The client never writes the prompt.
    - **Dimensions:** a request can ask for fewer dimensions (768, 512, 256 or 128). The engine cuts the vector and
      normalizes it again.
    - **Placement:** the engine runs on a B70 card on the shared interpreter, resident beside Gemma 4.
    - **Repositories:** senpai may edit gogi on a branch, merge it to gogi main and move the pin. kaijutsu uses the
      route later, in its own session.
    - **Inputs:** text only. The model also takes audio and images, and those come later.
    - **Layout:** no auto-swap. Engines load and unload only through the API, and a pin keeps an engine loaded
      (spec 3.3a, "Explicit layout"). Gemma 4 does not come back by itself after an image engine.

    Do the steps in this order:
    1. **gogi.** The route, the members of a served model, `gogi-fake` and the suite checks. Merge, then move the pin.
    2. **Spec.** The route, the family, the embedding capability of the engine ABI and the batch quanta. Amy decides
       each Proposed note.
    3. **Oracle fixture.** Write the fixed inputs and outputs of sentence-transformers once, on the CPU (8.3).
    4. **Family.** Write `nagaya-text-embeddinggemma2`: the tokenizer and the task prompts.
    5. **Engine.** Write `nagaya-mk-embeddinggemma2-bmg` on the shared interpreter. Run the oracle on the card.
    6. **Route and deploy.** Serve the route, and add the engine to the profile of b70host as resident.

    Done on 2026-10-30, on b70host. On the card, the engine matches sentence-transformers to 9.5e-7. An embedding
    of a short input takes 11 ms, and one of 2048 tokens takes 601 ms. `nagaya.service` runs the build of `b4fb5a1`,
    with `embeddinggemma-2` pinned on card 1 beside Gemma 4. The gogi suite gave 70 ok, 0 FAIL and 0 skip for
    `embeddinggemma-2`, and 367 ok, 0 FAIL and 15 skip for gemma.

    Amy accepted the Proposed notes of spec 3.3a, 3.3b, 11.9a and 11.10 on 2026-11-02. The commits:

    - step 1, gogi `9f60194`, `a3a0006` and `6db7e98`, and the pin `432cf8b`;
    - steps 2 to 4, the spec, the oracle and the family: `f3d78cb`, `ce6d967`, `62ef500`;
    - step 5, the engine: `6bec50b`, `df0ba2a`, `1a8250e`;
    - step 6, the route and the deploy: `93f9d8d`;
    - the explicit layout: `38662d4`, `8ec6afb`, and the device threads of the group `8a29dc4`, `61120b7`, `920dabe`;
    - the fixes of four reviews: `9f9bb44` to `b4fb5a1`.
20. **img2img and inpainting.** Amy chose this milestone on 2026-10-30, and she decided these rules by interview:
    - **Inputs:** spec 11.9, "The design of later image work". The image and the mask are object parts.
    - **Inpainting:** the latent blend works with any checkpoint. Also load a 9-channel inpainting UNet, such as
      diffusers' `stable-diffusion-xl-1.0-inpainting-0.1`, as other weights.
    - **Size:** with no size in the request, the input image sets it. An image that breaks the size rules is a 400
      that names the part. A request size that does not match the image is a 400 too.
    - **Cards:** senpai may stop and start `nagaya.service` on b70host for gates, and deploy at will. Write each
      stop and start in `signoff.md`.

    Do the steps in this order:
    1. **Spec.** Write the request, the mask rules, the VAE encode quantum and the 9-channel UNet into the docs.
       Amy decides each Proposed note.
    2. **Oracle fixture.** Write the fixed inputs and outputs of img2img and both kinds of inpainting from diffusers
       once, on the CPU (8.3). Keep the images small.
    3. **VAE encode.** Add the VAE encoder to the engine, with an op test for each new op.
    4. **img2img.** The start latent, the strength and the shorter schedule. Run the oracle on the card.
    5. **Inpainting with a blend.** The mask, the blend at each step, and the paste of the pixels after the decode.
    6. **The 9-channel UNet.** Its loader and its extra inputs. Run its oracle on the card.
    7. **Route and deploy.** Serve both through council v2, and add the inpainting weights to the profile.

    The commits so far:

    - step 1, the spec: `06f2317`, `60dcd35`, and Amy's decisions `1017422` and `97bb138`;
    - step 2, the oracle: `a77e9a4`. `tools/sdxl_inpaint_golden.sh` writes it into `target/sdxl-golden`;
    - the host rules of 11.9b (the request, the PNG, the masks, the paste and the chunk): `bdc072f` to `52dde8e`;
    - step 3, the VAE encoder: `06704a9` to `d6006d1`, `0da11fa`, and the test fix `e1928b1`;
    - step 4, img2img: `2e07187`, `d75f1ab`, deployed at 12:29 on 2026-11-02 from `cb35641`.
    - step 5, inpainting with a blend: `59dbaa3`, `73ace04`, `b8e7dc4`, deployed at 13:05 on 2026-11-02.
    - step 6, the 9-channel UNet: `d0b43d5` (Proposed) to `d1d3916`;
    - step 7, `inpaint-0` in the profile: `096d2fa`.

    Done on 2026-11-02 at 13:47, on b70host. `nagaya.service` runs the build of `096d2fa`, with `inpaint-0` on
    demand on card 1. On the cards, img2img and both kinds of inpainting match diffusers for each sampler. Live, a
    load of `inpaint-0` took 10 s, and an inpainting job of 1024 by 1024 with 30 steps took 58.8 s. A request without
    a mask got the 400 that names `messages[0].content`. The gogi suite gave 290 ok, 0 FAIL and 16 skip, in the card
    gate `xe_process`.

    Amy accepted the note of engine-abi "The declaration of a 9-channel UNet" on 2026-11-02.
21. **Image speed.** Make the SDXL engine faster on the B70 cards. Amy decided on 2026-10-30, by interview: a faster
    kernel can change the output bits if the oracle still passes. The tolerance goes into the spec as a Proposed
    note. The same identity, job and seed must still give the same bits on each run.
22. **Qwen-Image on gfx1151host.** Serve Qwen-Image 2.1 with the FUSAL LoRA from the nagaya on gfx1151host, and retire
    `~/src/diffusepal`. Amy, 2026-10-31: "I'm thinking about adding qwen-image to the nagaya. we'd use nagaya instead
    of diffusepal and retire that project." She decided these rules on 2026-10-31, by interview:
    - **Machine:** gfx1151host. Amy: "the goal of nagaya is to make our lab flexible, eventually we'll have models across
      all the hosts to choose from."
    - **Variant:** text-to-image with FUSAL only. Amy: "we'll add more variants over time so keep that in mind but
      just build one." The profile names the base, the text encoder and the LoRA. Thus a variant is other weights.
    - **Text encoder:** the Heretic Qwen3-VL-8B of FUSAL, in bf16 (Amy: "keep bf16"). FUSAL trained against its
      output. Use the tokenizer of the base model's `processor/`, not Heretic's (FUSAL README). Load only the text
      tower: 14.10 GiB of the 16.33 GiB file.
    - **Conditioning cache:** the store keeps the encoder output of each prompt (Amy: "yes"). Its recipe names the
      encoder's content id, the tokenizer and the tokens. A hit skips the encoder.
    - **Kernels:** hipBLASLt does the large bf16 matrix multiplies, and our HIP kernels do the rest. Amy: "hipBLASLt
      for GEMMs but we let it ride with changes over time, ok with some change in behavior if upstream changes". The
      hipBLASLt version is a build fact of the identity (12.1). A new version gives a new identity.
    - **Ops:** the new crate `nagaya-mk-hip` holds the shared HIP ops of gfx1151, as `nagaya-mk-bmg` does for the B70
      cards. The engine is `nagaya-mk-qwenimage21-gfx1151`, and the family is `nagaya-text-qwenimage21`. mk's Qwen3.8
      code stays in its own crate.
    - **Memory:** Qwen3.8 and Qwen-Image are both loaded while gfx1151host only serves. Amy: "what if we make it possible
      to load both models when gfx1151host is just doing inference, and we can unload them and restart nagaya as we do
      when we're doing builds of nagaya here." Large builds move to b70host, which has host memory for them.
      Before a large build on gfx1151host, unload Qwen-Image through the load routes (3.3b), or stop the nagaya.
    - **Pool:** the pool grows to hold both engines. Senpai's estimate of 2026-10-31: Qwen3.8 holds 72.7 GiB,
      Qwen-Image about 31.6 GiB, and the store's host tier 16 GiB. That is 120.3 GiB of gfx1151host's 125.1 GiB. The
      deploy step measures the real sizes. Write the pool, the store tier and the floor (4.7) as a Proposed note.
    - **Service:** senpai may stop and start `nagaya.service` on gfx1151host for fixtures and gates (Amy: "Yes, at will").
      Write each stop and start in `signoff.md`. End each session with `nagaya.service` running.
    - **Weights:** tests read the base model's directory from `QWENIMAGE21_MODEL_DIR`, and FUSAL's directory from
      `QWENIMAGE21_FUSAL_DIR`. Do not put the paths in code. Both are under `/tank/ml/models/qwen-image-2.1/`.
    - **Reference:** `~/.venvs/nagaya-qwenimage21-ref` has diffusers 0.41.0.dev0 with `QwenImage21Pipeline`. It is a
      copy of diffusepal's venv, without diffusepal. Run it with `-I`, on the CPU only, with `nagaya.service`
      stopped. Its weights need 31 GiB of RAM.
    - **Decisions of 2026-11-01**, by interview. Senpai decided the other notes of step 1, and Amy
      accepted them on 2026-11-02.
      - A step never splits into smaller quanta.
      - The client writes the trigger phrase, and the PNG stays RGBA.
      - True CFG and the negative prompt are a design for later.
      - Variants with the same encoder share their conditionings.
      - The LoRA folds into the weights at the load.
      - gfx1151host gets an object budget of 200 GiB at step 8, as b70host has.
    - **Decisions of 2026-11-01, after the deploy**, by interview.
      - A host victim of the store moves to the NVMe tier (spec 6.9).
      - Qwen-Image is resident on gfx1151host, and mk is on demand. The API swaps them (`docs/deploy.md`, "The layout of
        gfx1151host").

    Do the steps in this order. Each step ends with the checks green.
    1. **Spec.** Write the Qwen-Image rules into the docs: the served model, the family, the job and the sampler.
       Also write the sizes, the conditioning cache and hipBLASLt in the identity. Amy decides each Proposed note.
    2. **Oracle fixture.** Write the fixed inputs and outputs of diffusers once, on the CPU (8.3). Keep the image
       small. Keep the conditioning, the output of one step and the decoded pixels, so each stage has a gate.
    3. **Family.** Write `nagaya-text-qwenimage21`: the tokenizer, the prompt template and the cut of the system
       tokens.
    4. **Ops.** Write `nagaya-mk-hip`: its `build.rs`, the hipBLASLt binding and the ops of the three models. Give
       each op a test.
    5. **Encoder.** The text tower of Qwen3-VL, and the conditioning cache in the store. Run the conditioning gate.
    6. **Transformer and sampler.** The steps, the cached keys and values of the condition, and FlowMatch Euler with
       its shift. Run the step gate.
    7. **VAE and engine.** The VAE decode, and the `Engine` trait with the image calls. Run the conformance gates and
       the whole oracle.
    8. **Route and deploy.** Serve the model through council v2, and add it to the profile of gfx1151host. Run the gogi
       suite. Then Amy retires diffusepal.

    Done on 2026-11-01 at 16:50, on gfx1151host. `nagaya.service` runs the build of `23f7b86`. Qwen-Image is resident and
    mk is on demand; the load route swaps them. A step of 1024 by 1024 takes 4.2 s with flash attention, and an image
    of 40 steps takes 176 s. A swap to mk took 28.5 s, and the swap back 8.0 s. The image after the round trip had the
    same object id as before.

    The gogi suite gave 83 ok, 0 FAIL and 4 skip for `qwen-image-fusal`. For mk after a swap, it gave 343 ok, 0 FAIL
    and 12 skip.

    The approaches to the speed of a step, each tested on gfx1151host:

    - flash attention: 5.8 s to 4.2 s at 1024 by 1024, with the same gaps of the gates. It is the default.
    - attention of several heads in one GEMM: no gain, so the change went.
    - the fastest deterministic GEMM pick: 4.2 s to 4.06 s, but the bits could change when a table fills again. It
      stays out (spec 12.1).

    Amy accepted the Proposed notes of spec 11.1 ("A text engine on demand") and of `docs/deploy.md` ("Two engines
    on gfx1151host") on 2026-11-02. She also accepted each note that senpai decided in this milestone. The
    nagaya needs nothing in `~/src/diffusepal` now: the weights and the reference venv have copies (`b1694e6` and
    after). The commits:

    - step 1, the spec: `74034db` to `ca7c6e1`, the merge `9a559eb`, and Amy's decisions `65246fe`;
    - step 2, the oracle: `ae47b39`, and the bf16 baselines `7021537`;
    - step 3, the family: `3399eae`;
    - step 4, `nagaya-mk-hip`: `3ee48b9`, `ad70019`, `88a243e`, `d8c71b1`, `2fb024b`;
    - steps 5 to 7, the engine: `c9c1b4a`, `b675422`, `cb10e8b`, `15be802`, `5b2baf3`, `2f08cba`, `6ba048b`;
    - step 8, the route and the deploy: `ef055ea`, `5649914`, `9e691cf`;
    - the speed: flash attention `d5e65c6` and `fde517e`;
    - the fixes of the deploy: `0e26a1e`, `711f3ac`, `1598924`, `28e6eb5`;
    - the layout and the swap: `13227bf`, mk's close `6e8e013`, and the served model on demand `ce3a52e`, `23f7b86`.
23. **Cluster.** gfx1151host and b70host join one cluster, and each machine serves the models of both. Amy, 2026-11-02:
    "Let's get the work rolling to get the cluster up though." She decided these rules on 2026-11-02, by interview:
    - **Work where the files are:** "ideally we do the work with what's available on the cluster - so if the
      tokenizer is on this machine great we tokenize here. otherwise we forward to whereever it is." The routes
      `tokenize`, `detokenize` and `render` run on the machine that takes them when it holds the family. A job route
      forwards the whole request to the machine that holds the engine. A job split across machines comes later.
    - **Objects:** a `GET` of an object that this machine lacks asks each peer and gives the first hit. A forwarded
      request takes its objects to the target first.
    - **Names:** a served model name is the id of a model in the whole cluster (Amy: "no 2 models can have the same
      text name we use as an identifier"). A profile that adds a name that a peer serves is refused.
    - **The cluster call:** one layer makes every call to peers. It routes when gossip names the holder, and else it
      broadcasts to all and lets each peer answer 4xx or 5xx. Bloom filters or routes come later (spec 11.1a).
    - **Control:** `GET /nagaya/v1/cluster` gives the gossip view. A load or an unload of an engine on any node goes
      to the node that holds it. A profile `PUT` for another host publishes `profile:desired:<host>` (spec 2.4).
    - **Drafts:** prompt-lookup drafts wait for the next pass on gfx1151host (spec 6.14).

    Do the steps in this order:
    1. **Spec.** Write the forward, the objects, the names and the control routes into spec 2.3, 2.4, 3.3b, 6.15 and
       11. Amy decides each Proposed note.
    2. **Join.** Seed each machine with the other, and publish what spec 2.3 names. Build the cluster call.
    3. **Identity.** `identity.models` lists every served model in the cluster, with its status from gossip.
    4. **Forward.** A request for a model of a peer goes to that peer, and its stream comes back as it is.
    5. **Objects and names.** The `GET` on a miss, the objects of a forwarded request, and the name conflict.
    6. **Control.** The cluster route, the load and unload of a peer's engine, and the profile of another host.
    7. **Deploy.** Deploy both machines. Run the gogi suite through each machine for the models of both. The web app
       finds its hosts from the cluster route.

    Done on 2026-11-04. Steps 1 to 6 are `d5efc8c` to `41fb290`, and step 7 is `0e8a69c`. Both nodes list the same 11
    served models, and the gogi suite passed through each machine. One check of `qwen-image-fusal` through gfx1151host
    aborted gfx1151host's nagaya (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, `signoff.md` 2026-11-04). It did not happen
    again. Amy, 2026-11-08: ignore it for now.

    Amy accepted the four Proposed notes of this milestone on 2026-11-08.
24. **LFM3 Omni.** Serve LiquidAI's `LFM3-Omni-2B` and `LFM3-1.7B` from one engine kind on a B70 card. Omni reads
    text, images and audio, and it writes text and speech. Amy decided these rules on 2026-11-06, by interview:
    - **Order:** after the cluster (milestone 23). Thus every machine of the cluster reaches the new models.
    - **Scope:** text, image and audio in; text and speech out. Video comes after the deploy, as its own step. It
      uses the vision tower on frames.
    - **Placement:** both models are resident on card 0 of b70host. Card 1 holds the image engines, the embedding
      engine and speech. Gemma 4 goes on demand, and it loads only through the load routes (spec 3.3b).
    - **Text model:** `LFM3-1.7B` is other weights on the same engine kind, as its own served model. Its backbone has
      the same tensor names and shapes as Omni's language model, but other bits.
    - **Licenses:** the nagaya does not check a model license. The user accepts it at the download (Amy, 2026-11-06:
      "anyone using it has to accept the model license on their own when they download it for use imo. so we just
      build, let users make their own decisions").
    - **Speech:** Qwen3-TTS VoiceDesign stays the speech engine. Omni's speech is a second choice, with its own
      voices.

    The facts from the checkpoints, read on 2026-11-06 from the safetensors headers and `config.json`:

    - Omni is 2,166,928,384 bf16 parameters in `model.safetensors` (4.33 GB). LiquidCodec is
      `audio_tokenizer/model.safetensors` (0.15 GB). `LFM3-1.7B` is 1,728,801,536 parameters (3.46 GB).
    - The backbone has 24 layers of width 2,048. There are 13 gated short convolutions, 5 sliding attention layers
      with a window of 256, and 6 full attention layers. The norms are sandwich norms, and full attention rotates
      half of each head.
    - The vision tower has 14 layers of width 1,024, with patches of 16 and axial rotary positions. The audio encoder
      has 17 layers of width 768, on 128 mel bins at 16 kHz.
    - The depth decoder has 3 layers. It fills 12 codebooks of 2,017 codes for each frame, 4 at a time
      (`maskgit_schedule`). The reply puts 6 text tokens, then 12 audio tokens, in turn.
    - LiquidCodec decodes 1,920 samples at 24 kHz from each frame. Speech input does not use the codec encoder.
    - The two `tokenizer.json` files hold the same tokens, but Omni's lacks the last newline, so the hashes differ.
      The chat templates differ: only Omni's has audio parts. Both embedding tables have 128,000 rows.

    The model name in the crate names is `lfm3`. The engine is `nagaya-mk-lfm3-bmg`, and the family is
    `nagaya-text-lfm3`. Do the steps in this order:
    1. **Spec.** Write the LFM3 rules into the docs. They are the served models, the family, the state parts and
       the image and audio inputs. They also include the interleaved reply and the sampler of the audio codes. Amy
       decides each Proposed note.
    2. **Oracle fixture.** Write the fixed inputs and outputs from transformers once, on the CPU (8.3). Keep a gate for
       each stage: the vision rows, the audio rows, the backbone logits, the depth codes and the decoded audio.
       Pin the revision of each model. Read Liquid's remote code before it runs.
    3. **Refusal probe.** Run a sample of Amy's photos through the reference on the CPU. Count the refusals, the
       vague captions and the full captions. Write only the counts. The photos and the captions stay on b70host.
       Amy runs it later (2026-11-06: "no photos on this machine yet, I'll test that later. you can generate some
       test images with kaibo as needed"). Thus the other steps do not wait for it.
    4. **Family.** Write `nagaya-text-lfm3`: both tokenizers, both templates, the XML tool calls, and the reply
       parser. Also write the image patches and the mel features on the CPU.
    5. **Ops.** Add the gated short convolution, the sliding attention and the partial rotation to the shared
       interpreter. Give each op a test.
    6. **Backbone.** Write `nagaya-mk-lfm3-bmg` with the text path, and serve `LFM3-1.7B` first. Run the conformance
       gates and the logit gate on the card.
    7. **Vision and audio in.** The vision tower, the audio encoder and their projectors, with their gates.
    8. **Speech out.** The depth decoder, the sampler of the audio codes, and the LiquidCodec decoder on the card.
       Run the gates of the codes and the audio.
    9. **Route and deploy.** Serve both models through council v2 with media inputs and audio outputs. Put them on
       card 0 in the profile of b70host, and put Gemma 4 on demand. Run the gogi suite through each machine.
    10. **Video.** Frames from a video through the vision tower, with a gate. Amy decided these rules on 2026-11-08,
        by interview:
        - **Input:** a video object, for example `video/mp4` or `video/webm`. The family decodes it on the CPU.
        - **Decoder:** ffmpeg's libraries (libavformat, libavcodec, libswscale and libswresample), behind a cargo
          feature. Their versions go into the recipe item of a video part, because a new decoder can change the rows
          (Amy, 2026-11-08). With no feature, a video part is a 400 that names the part.
        - **Frames:** the rules of the checkpoint's video processor. It takes 2 frames each second, from 4 to 384
          frames, within its pixel budget (spec 11.11, "The video of a request").
        - **Sound:** the sound track of a video is an audio part beside its frames.
        - **Resample:** the resampler of the decoder also serves `audio/wav` at any rate and channel count. A WAV of
          16 kHz mono in 16 bits skips it, so its bits do not change.
        - **Live input:** PipeWire and capture from a microphone or a camera are a later design, not this step.
    11. **Omni tab.** A tab of the web app for `lfm3-omni`: it attaches images, audio and video, and it shows the text
        reply and plays its speech. Today the page puts Omni in its speech tab, because Omni writes audio. Amy,
        2026-11-08: "maybe we need to add an omni tab to the web app".
    12. **Audio read back.** An `assistant` audio part feeds the kept codes of its output: one position for each frame,
        with the frame's summed codebook embedding. Amy, 2026-11-08: "Read it back". Then the Omni tab sends a spoken
        reply back as its audio. Gate: the teacher-forced rows of the oracle's speech run.

    Each stop and start of `nagaya.service` goes into `signoff.md`. The tests read Omni's folder from
    `LFM3_OMNI_MODEL_DIR` and the text model's folder from `LFM3_MODEL_DIR`. Neither has a default.
    `tools/lfm3_golden.sh` writes the oracle into `target/lfm3-golden` or `LFM3_GOLDEN_DIR`, with no GPU job.
    Control vectors (`docs/cv.md`) are not in this milestone.
25. **Control vectors.** A council v2 context can change the residual of an LFM3 model while it runs. Amy, 2026-11-07:
    "I'm getting some refusals on LFM3 for storytelling". Someone ships Omni weights with the refusal removed. Amy:
    "I'd like to try this runtime technique and figure out how to work it into the protocol as an option". She
    decided these rules on 2026-11-07, by interview. They replace the Proposed notes of `docs/cv.md` where the two
    do not agree.
    - **Protocol:** a context names its control vectors, beside `thinking`. Each item names a vector object
      (`blake3:` id), an operation and a strength. A served model declares a capability for it. The list goes into
      the root recipe (spec 6.12), so contexts with the same list share their prefixes.
    - **Operations:** `add`, `ablate` and `clamp` (`docs/cv.md`, "What it does"), in one op.
    - **Extraction:** the nagaya makes a vector on the card. A residual tap (spec 13.1) reads the residual after
      each layer, and a job takes the difference of the means of two prompt sets.
    - **Route:** the route that makes a vector is council v2. gogi writes its contract first, with suite checks.
    - **Prompt sets:** public sets of refused requests and answered requests (Arditi et al., 2024), and the
      storytelling prompts that LFM3 refused for Amy. Her prompts stay on b70host, and only counts
      go into git. The public sets come first (Amy: "so we can work without seeing that data").
    - **Models:** `LFM3-1.7B` first. Omni gets its own vector after the gates of the text model are green.
    - **Strength:** Amy sets each strength by hand through the context. A search for strengths, as Heretic does,
      comes later.
    - **Repositories:** senpai may edit gogi on a branch, merge it to gogi main, push it and move the pin.

    Do the steps in this order:
    1. **gogi.** The control list of a context, the capability, the route that makes a vector, `gogi-fake` and the
       suite checks. Merge, push, then move the pin.
    2. **Spec.** The context field, the route, the recipe item (6.2, 6.12), the residual tap (13.1) and the engine
       hook "the residual after layer `l`" in engine-abi. Write Amy's decisions into `docs/cv.md`. Amy decides each
       Proposed note.
    3. **Oracle fixture.** Write the fixed inputs and outputs from transformers once, on the CPU (8.3). Keep the
       mean residuals of two small prompt sets, and the logits with each operation at a fixed strength.
    4. **Op.** Add the op of the three operations to the shared interpreter, with a test for each operation.
    5. **Tap and job.** The residual tap on the LFM3 engine, and the job that takes the means on the card. Run the
       gate of the means.
    6. **Engine.** The control list on the pass of the LFM3 engine, and in the root recipe. Run the gates: a strength
       of 0 gives the bits of the plain model, and each operation matches the oracle.
    7. **Route.** Serve the route and the context field through council v2. Run the gogi suite.
    8. **Trial.** Make a refusal vector of `LFM3-1.7B` from the prompt sets. Amy tries strengths for storytelling.
       Write only the counts of refusals into `signoff.md`.
    9. **Omni.** Make Omni's vector, with image prompts if its refusals need them. Run its gates, then deploy.
       The vector acts on Omni's text passes only (Amy, 2026-11-07: "we would only need to do this for omni's
       text"). Its speech frames take no list.

    Each stop and start of `nagaya.service` goes into `signoff.md`.
26. **Lab tools.** Look inside a model while it runs, with `project` and `trace` (`docs/lab.md`). Amy, 2026-11-07,
    after loops in some steered replies: "how might we analyze that? could we build something into the nagaya for
    looking at weights to see what's up?" Then "ok yeah, start on /nagaya/v1/lab/". The rules:
    - **Routes:** nagaya routes under `/nagaya/v1/lab/` first. `trace` moves to gogi when its shape settles and a
      client needs it. Routes that read tensors stay in the nagaya.
    - **Model:** `lfm3` first, then Omni's text passes.
    - **The J-Lens:** a reference tool outside the nagaya fits it, as gmk did on gfx1151host. The nagaya applies it.

    Do the steps in this order:
    1. **`project`.** A vector through the unembedding, with the weight of the final norm. Its oracle is the product
       of the reference on the CPU. Then project the refusal vector, and report the rank of `<|im_end|>`.
    2. **`trace`.** A generation that writes a trace object (`docs/lab.md`, "`trace`"). Its card gate: a trace gives
       the bits of the same generation with no trace.
    3. **Web app.** A Lab tab that projects a saved vector, and draws a trace of a plain and a steered reply.
    4. **Deploy.** Deploy on b70host, and trace Amy's loops with her.
    5. **J-Lens.** Fit the lens of LFM3-1.7B with the reference, and read through it in `project` and `trace`. The fit
       runs on the CPU of b70host at `nice` 19, with no GPU job (Amy, 2026-11-08). A client keeps a trace object
       with a keep (`docs/lab.md`, "Open questions").

    Each stop and start of `nagaya.service` goes into `signoff.md`.
27. **Batched passes on LFM3.** Several contexts share one decode pass of the LFM3 engine, so the weights are read
    once for each step of all of them. Amy, 2026-11-09, after the swarm ideas (`docs/swarm-ideas.md`): "what's the
    lift to enable batching for LFM3? I'd like to do that". She decided these rules on 2026-11-09, by interview:
    - **Bits:** a context's bits never depend on the other segments of its pass. One context decodes on the path of
      the matrix multiply for 2 to 8 rows. Thus one row and eight rows sum in one order. The engine identity
      changes, and each held context fills again once.
    - **Order:** this milestone comes now. The J-Lens deploys when its fit ends, and the steps with Amy stay open.
    - **Models:** both `lfm3` and `lfm3-omni`. Omni's speech frames stay one context in a pass for now.

    Do the steps in this order. Each step ends with the checks green.
    1. **Probe.** A card test of the matrix multiply and of SwiGLU: each row gives the same bits at 1 to 8 rows.
       Also time the decode of one context on both paths. Stop and report if a row differs.
    2. **One path.** The LFM3 decode of one row takes the path for 2 to 8 rows. Run the oracle gates on the card.
    3. **The batched pass.** `model::pass` builds its tasks for a list of slots and positions. Each row has its own
       rotation, attention, convolution and K and V rows. Declare 8 segments; `segments_exact` stays false (D11
       rule 4). The card gate: B segments give the bits of B passes alone.
    4. **Control lists.** A pass joins only segments with equal control lists, at first.
    5. **The process.** The scheduler joins the decode quanta of an engine with a feed rule (D11 rule 6). The group
       driver puts every segment of a quantum into one pass. Then it gives each job its rows. Now
       `crates/nagaya/src/jobs.rs:249` asserts one segment.
    6. **Deploy and measure.** Decode tokens each second at 1, 4 and 8 contexts. Run the gogi suite for both models.

    Each stop and start of `nagaya.service` goes into `signoff.md`.

    Done on 2026-11-09, on b70host. `nagaya.service` runs the build of `abcaccd`, with eight slots for each LFM3
    model. Live, greedy replies of 200 tokens ran at 1, 4 and 8 contexts. `lfm3` gave 109, 249 and 317 tokens each
    second in total, and `lfm3-omni` gave 109, 247 and 313. A reply among 8 was the reply alone, on both models. The
    gogi suite gave 438 ok, 0 FAIL and 16 skip for `lfm3`, and 457 ok, 0 FAIL and 16 skip for `lfm3-omni`.

    The correction of `19413de` is Proposed: decode segments join by D11 rule 6, and `segments_exact` stays false.
    The commits:

    - steps 1 and 2, the tiled path: `d2380de`, `912113e`;
    - steps 3 and 4, the batched pass: `911fc3b`;
    - step 5, the schedulers and the drivers: `e057205`, `3f46d47`, `edd5bee`, `3ae6345`, `27c5cee`, and the fix of a
      red test `7558623`;
    - step 6, eight slots: `200461d`;
    - the fixes of a kaibo review: `abcaccd`.
28. Stop and ask Amy for the next milestone.

### mk on gfx1151host

- mk's engine is in `crates/nagaya-mk-qwen38-gfx1151/hip/`, a copy of mk commit `555a2a3` (milestone 15, Amy
  2026-10-22). mk's repository is an archive: do not edit it, and do not build it.
- Under the feature `hip`, `build.rs` compiles the copy with ROCm's `clang++` and links it statically. A build
  without `hip` needs no ROCm, so a machine without ROCm builds and tests the workspace.
- Keep mk's compile flags. `tests/build_flags.rs` compares them with mk's `compile_commands.json`.
- mk's weights are three GGUF shards. mk's `harness/paths.py` (`SHARDS`) gives their paths. GPU tests read them
  from `MK_SHARDS`, three paths with `:` between them. Do not put the paths in code.
- mk's engine has no attach. Milestone 15 step 6 removed the C++ attach, which had no call. The export of the arena
  (spec 10.2) builds it in Rust, from `arena.rs`. For an fd from systemd's store, give `/proc/self/fd/<n>`.
- The context calls (`context.rs`) return nothing. The worker maps the result onto the three write results (ABI
  "Capabilities") through the poison flag of the context.
- The nagaya must be the only GPU client (spec 3.1). Since 2026-10-21, `nagaya.service` serves council v2 on gfx1151host.
  Before you run GPU tests, check that both units are inactive:
  `systemctl --user is-active megakernel.service nagaya.service`.
  - They are user units. The command without `--user` says "inactive" while they run.
  - If either one runs, stop and report. Do not stop a service yourself.
  - In a container with no systemd, ask your reviewer to check it.
- The C++ runtime layer aborts the process on a failed HIP call (`MK_CHECK`), and the Rust host stops it on a
  defect. The extern layer `extern.hip` checks nothing, so the worker checks each rule first. `host::segments_problem`
  runs before each pass.
- mk's HIP device is the device that is current on the calling thread. Make every mk call from one thread.
- Do not call `mk_engine_close` or free the engine in tests. HIP teardown in a process segfaults (spec 3.3).
- The drop of the last mk handle frees the weights (engine-abi "The close of mk"). A test binary that opens many
  handles keeps a `hold()`, as `tests/hip.rs` does. `tests/hip_close.rs` tests the close.
- GPU tests are behind the cargo feature `hip`, so `cargo test` with no features needs no GPU.

### mkviolet on b70host

- mkviolet is `~/src/mkviolet`. It is a reference: read it, and do not edit it.
- The nagaya must be the only compute client of the B70 cards (spec 3.1). Amy stopped and disabled `mkviolet.service`
  on 2026-10-21 ("cards and cpus are all yours"). Before you run GPU tests, check that it is inactive:
  `systemctl --user is-active mkviolet.service nagaya.service`. If either one runs, stop and report.
  In milestones 16 and 17, senpai may stop `nagaya.service` for a GPU test, and start it again after.
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
- b70host holds Gemma 4 in `~/src/mkviolet/checkpoints/gemma-4-26B-A4B`. The `xe` tests of
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

On gfx1151host, run the oracle bench with `MK_SHARDS` set and both units inactive:
`cargo bench -p nagaya-mk-qwen38-gfx1151 --features hip --bench oracle`. It exits with 1 when a gate breaks.
Set `MK_FAST_QMM=0` for the gate: the exact path must match the reference. Leave it unset to measure the fast path.

The bench prints the SHA-256 of the logits of each prompt. A port step that keeps the bits keeps each hash.

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
