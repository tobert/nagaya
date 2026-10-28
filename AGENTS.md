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
| `nagaya-mk-bmg` | the megakernel interpreter of the B70 cards: Level Zero, the task protocol, the kernel source, one binary per engine | ABI "How mkviolet maps" |
| `nagaya-mk-gemma4-bmg` | the mkviolet engine: Level Zero, the OpenCL C kernels, and the Gemma 4 host code; a card needs `xe` | ABI "How mkviolet maps" |
| `nagaya-council` | council v2 logic with no I/O: the family interface, the sampler, request checks | 11 |
| `nagaya-jinja` | the Jinja interpreter of the chat templates, with the results of Python's jinja2 | 11.3 |
| `nagaya-text-qwen38` | the Qwen3.8 family library: tokenizer, templates, detokenizer, reply parser | 11.3, 11.3a |
| `nagaya-text-gemma4` | the Gemma 4 family library: tokenizer, template, escapes, council rendering, reply parser | 11.3, 11.3a |
| `nagaya-mk-qwen3tts-bmg` | the Qwen3-TTS VoiceDesign engine on one B70 card: weights, the passes as rings, the speech calls | ABI "Speech" |
| `nagaya-text-qwen3tts` | the Qwen3-TTS family library: tokenizer, the prompt of a speech request, the codec ids | 11.3, 11.8 |
| `nagaya-text-sdxl` | the SDXL family library: CLIP tokenizer, the A1111 prompt, its chunks and weights | 11.9a |
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
is done on gfx1151host, and milestones 15s, 16 and 17 are done on b70host. Milestone 18 runs now.

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

    Amy decides the Proposed notes of spec 3.3a. The commits:

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

    Amy decides the Proposed notes of spec 11.8 and of engine-abi "Speech" and "How a group loads both". The commits:

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
    - **VAE:** the stock VAE of the checkpoint, in fp32. Not the fp16 fix.
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
19. Stop and ask Amy for the next milestone.

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
- b70host holds Qwen3-TTS VoiceDesign in the Hugging Face cache. The tests of `nagaya-text-qwen3tts` and
  `nagaya-mk-qwen3tts-bmg` read it from `QWEN3TTS_MODEL_DIR`, which has no default. `signoff.md` gives the path.
- The oracle of `nagaya-mk-qwen3tts-bmg` reads `target/qwen3tts-golden` or `QWEN3TTS_GOLDEN_DIR`. Write the files
  with `tools/qwen3tts_golden.sh`, with no GPU job.
- b70host holds SDXL base 1.0 in the Hugging Face cache. The tests of `nagaya-text-sdxl` read it from
  `SDXL_MODEL_DIR`, which has no default. `signoff.md` gives the path. `tools/sdxl_golden.sh` writes the oracle into
  `target/sdxl-golden` or `SDXL_GOLDEN_DIR`, with no GPU job.
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
