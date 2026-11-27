# History

This doc keeps the finished milestones of the nagaya, as `AGENTS.md` wrote them, with what each one built and its
merges. `docs/milestones.md` keeps their open steps and the full text of the milestones that run now. The nagaya is
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
       `nagaya-family-toy` on `gogi-toy`. Give the fake engine logits over the toy vocabulary.
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

## The order of 2026-11-08

This is the order that `AGENTS.md` held from 2026-11-08 to 2026-11-11, as it was written.

Do them in this order. Each milestone ends with all checks green and one or more commits. Milestones 1 to 14 are
done, and `docs/history.md` keeps their full text, and the merges where the run recorded them. Milestone 15
is done on gfx1151host, and milestones 15s, 16, 17, 18, 19 and 20 are done on b70host. Milestone 21 waits, and
milestone 22 is done on gfx1151host.

Milestone 23, the cluster, is done on both machines. Milestones 24, LFM3 Omni, 25, control vectors, and 26, the lab
tools, are deployed on b70host, and each has open steps. Senpai chose this order on 2026-11-08 (Amy: "you may decide
order"):

1. Milestone 26 step 5, the J-Lens. Done on 2026-11-09: the fit ended at 04:31, and the lens of `lfm3` is in the
   profile (`01991db`). The first live trace found a defect, fixed in `a7af717`. A live trace matched an
   independent NumPy readout in 144 of 144 lists.
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
7. Milestone 28, tuned control vectors. Amy put it next on 2026-11-09. Steps 1 to 4 are done and deployed:
   the docs `26ad93f`, gogi `62847b0` and the pin `8e90c3f`, and `compose` `42e77d5`. The first trial of step 5 ran.
8. Milestone 29, swarms. Amy put it next on 2026-11-09.

## Milestones 15 to 29

This is the text of `AGENTS.md` for these milestones, as it was on 2026-11-11. Milestone 21 did not start.

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
    6. **Family.** Write `nagaya-family-sdxl`: the CLIP tokenizer and the prompt of an image request.
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
    - step 6, the family `nagaya-family-sdxl`: `a9c1ce7`, and Amy's decision on the ids `061efed`;
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
28. **Tuned control vectors.** An agent composes a control vector set and tunes it with the J-Lens. Amy, 2026-11-09:
    "maybe we expose what an agent will need to compose a control vector set that is tuned using the j-lens. and
    maybe we add something to it to indicate strength by layer?" She decided these rules on 2026-11-09, by
    interview, and she is "cool with this being a lil haphazard just to try it":
    - **Strength by layer:** in the control item of council v2, not in the vector. gogi writes it first.
    - **Steer toward words:** a direction toward a list of words, through the J-Lens run backwards.
    - **Mix in safe directions:** a part of a composed vector can add a direction, or take one out, at chosen layers.

    Do the steps in this order:
    1. **Docs.** The member `layers` of a control item (spec 11.12), and the route `compose` (`docs/lab.md`). Amy
       decides each Proposed note.
    2. **gogi.** The member `layers` in the contract, `gogi-proto`, `gogi-fake` and the suite checks. Merge, push,
       then move the pin.
    3. **The engine.** A control item acts on its listed layers only, each at its own strength. Card gate: `layers`
       with every layer at 1 gives the bits of no `layers`.
    4. **`compose`.** The route under `/nagaya/v1/lab/`, with the parts `vector`, `words` and `remove`. Its oracle is
       the same sums in NumPy.
    5. **Try it.** Compose a few vectors of `lfm3` with an agent, read them through the J-Lens, and count refusals on
       the public prompt sets. Write only counts into `signoff.md`.

    Each stop and start of `nagaya.service` goes into `signoff.md`.
29. **Swarms.** Agents of one model work in parallel, and one context reads their work (`docs/swarm-ideas.md`).
    Amy, 2026-11-09, on latent steps: "still interesting, might be good for our decisions too". She chose this order
    on 2026-11-09: "Yeah, that order, and let's get some running tonight. I'd like to at least get some
    measurements."

    Do the steps in this order:
    1. **Text swarm.** Agents write short ideas, a novelty filter keeps some, and a summarizer reads them as text.
       Measure it against direct mutations at the same wall time, on a task with a CPU evaluator. This step needs no
       new nagaya code. Write only counts and times into `signoff.md`.
    2. **Docs.** Latent steps, the read of a latent step, and the merge of held contexts, in engine-abi, spec 6.12
       and `docs/lab.md`. Amy decides each Proposed note.
    3. **Oracle fixture.** Latent steps and a merge in transformers on the CPU, with the logits after each (8.3).
    4. **Latent steps.** The engine feeds the last hidden state as the next input row, in batched passes. A latent
       step can also give its logit row, so a client reads the text along the way. A held latent context continues
       in tokens like any other.
    5. **The merge.** A new held context from blocks of held contexts, on the host, with the keys rotated to their
       new offsets. It is a lab route first.
    6. **Latent swarm.** The task of step 1 with latent agents and a merged summarizer. Compare it with the text
       swarm. Stop if it gives no gain.

    Step 1 ran on 2026-11-09 (`docs/swarm-ideas.md`, "The first trials"). One summary of 8 ideas never beat best fit,
    and a vote that picked one idea gave the best result. Amy, 2026-11-10: "I feel like we shouldn't try to get good
    code out of the thinking swarm. we need to distill it into something focused". She put two steps before step 4:

    - **1a. Plan, then code.** Swarm 1 thinks, one step distills, and forks of one context code the plan. Also run the
      summary with the parent last. This needs no new nagaya code.
    - **1b. Joined forks.** The lab route `ensemble` (`docs/lab.md`), with the rules `logprob_sum`, `prob_mean` and
      `contrast`. Card gate: one context gives the bits of the same generation. Compare the rules on the code step and
      on the distill step.

    A tune of the distill step, and a coder from `LFM3-1.7B-Base`, wait for another day (Amy, 2026-11-10).

    Steps 1a, 1b, 3 and 4 are done on 2026-11-10 (`docs/swarm-ideas.md`, "The second trials"). The route `ensemble`
    (`e2b70db`) and latent steps (`a36b117`) passed their card gates and are deployed, and `lfm3-1` serves card 1
    (`3a281d0`). The recipe driver is `tools/swarm`. Plan, then code (`s2s-combine`) won each seed. Joined forks and
    latent steps gave no reliable gain on bin packing. Step 5, the merge, and step 6 wait.

    Each stop and start of `nagaya.service` goes into `signoff.md`.

## Milestone 30

This is the text of `AGENTS.md` on gfx1151host for this milestone, as it was on 2026-11-10. Steps 7 to 11 are open.

30. **Qwen3.8 batches and latent steps.** Several contexts of Qwen3.8 share one decode pass on gfx1151host, and a context
    can think in latent steps. Amy, 2026-11-10: "I see latent steps as the more potentially useful addon for qwen, so
    we can start with join but let's plan on doing latent steps, which I want to experiment with on regular inference
    too." She decided these rules on 2026-11-10, by interview:
    - **Order:** the join first, then latent steps.
    - **Bits:** mk declares `segments_exact`, so a context's bits never depend on the other segments of its pass.
    - **Where:** latent steps go on the lab route `ensemble` first, as on LFM3. One context there is a plain
      generation. A council v2 member comes later, when the shape settles, and gogi writes it first.
    - **The row:** a latent step feeds back the row that the head reads, after the 4 hyper-connection streams fold.
      The host scales it to the mean norm of the embedding rows. It goes in at the embedding row, so the pass
      copies it into the 4 streams.
    - **PLE:** a latent position holds one placeholder token. The n-gram hash of layer 1 reads it as any token.
    - **Machine:** gfx1151host is senpai's for the milestone, and Qwen-Image can unload (Amy, 2026-11-10: "the machine is
      yours today, qwen-image is idle and can be unloaded").

    Do the steps in this order. Each step ends with the checks green.
    1. **Deploy.** Deploy `main` on gfx1151host: the checks, the mk `hip` tests and the oracle bench, then a start.
    2. **Gate.** A card gate of decode rows at deep positions and beside a long prefill, alone and joined. Run it with
       `MK_FAST_QMM` unset and with `MK_FAST_QMM=0`. Stop and report if a row differs.
    3. **Docs.** The join of an engine with `segments_exact` and no feed rule (engine-abi "Batched passes", D11).
       Amy decides each Proposed note.
    4. **The process.** The scheduler and the drivers join the decode quanta of mk. Test it first on a fake engine
       with `segments_exact` and no feed rule. `ensemble` joins mk's forks too.
    5. **mk's spin bound.** On 2026-11-10, mk stopped twice on a spin-timeout while a cargo build ran beside it. A
       barrier wait counts 1 s of wall clock (`hip/src/rt/rt.h:97`), so a paused queue looks like a hang. Amy chose
       to fix it before the deploy, with Qwen-Image resident until the fix passes. What the card window found:
       - The weight arena is a memfd that the GPU reads through a host registration, so swap could move its pages.
         Amy, 2026-11-10: "oh, mlock or pin for sure". The open locks it with `mlock2(MLOCK_ONFAULT)`: the load
         stays at about 21 s, and a card test checks that each page is locked.
       - A diagnostic that names the polls of a spin-timeout faulted the GPU in 5 of 5 stress runs. The code without
         it passed in 4 of 4. Its defect is not found, so it waits in `~/exomemory/nagaya/patches/`.
       - A bound that counts polls waits for that diagnostic. A real hang must still stop in about 1 s, so the gfx
         ring keeps its margin (`rt.h:93-95`).
    6. **Deploy and measure.** Decode tokens each second at 1, 4, 8 and 16 contexts. A reply among 16 must be the
       reply alone. Run the gogi suite for Qwen3.8.
    7. **Docs.** The tap hook of mk in engine-abi: "the residual after layer `l`", with the head's row as the last
       layer. Also the latent rows of mk: the placeholder, the scale and the record. Amy decides each Proposed note.
    8. **Oracle fixture.** The residual after each layer, and latent steps with the logits after each, from the
       reference on the CPU (8.3).
    9. **The tap hook.** A general hook on mk, not one for the head's row alone. Later milestones read it for
       control vectors, `project` and `trace` (Amy, 2026-11-10: "let's get the tap hook earlier in the plan").
       Card gates: a pass with a tap gives the bits of the same pass with none, and the rows match the oracle.
    10. **The latent span.** The latent span on mk. Card gate: the latent row of the placeholder's embedding gives
       the bits of the placeholder token.
    11. **Route and trial.** `ensemble` with `latent` on Qwen3.8. Deploy, then run a trial with Amy.

    Each stop and start of `nagaya.service` goes into `signoff.md`.

    Steps 1 to 6 are done on gfx1151host, deployed at 16:07 on 2026-11-10 from `f7d9ea9`. Live, greedy replies of 200
    tokens ran at 1, 4, 8 and 16 contexts: 17.8, 47.3, 46.6 and 65.0 tokens each second in total. Before the join, 4
    contexts gave 18.4. Each reply among 16 was the reply alone, and the gogi suite gave 411 ok, 0 FAIL and 12 skip.

    Why 8 contexts give no more than 4 is open, because no record shows the segments of each pass. The commits:

    - step 1, the test fix `723aa6b` and the milestone `911ae76`;
    - step 2, the gate `1d02ac5`;
    - steps 3 and 4, the note `5a2f50f` and the join `2f40184`;
    - step 5, `38e1dfa`, the lock `4e31aae` and its record `f7d9ea9`.

## Milestone 31, the names of the Qwen3.8 engine

Steps 1 and 2 of milestone 31 ran for the Qwen3.8 engine on 2026-11-12, on gfx1151host. The merge is `d1d02af`. The old
practice project had the name mk, and this section is the record of its names. A reader of the code does not need
them.

The rules, decided on 2026-11-12:

- A name goes when it points at the old project. `mk` stays where it means the kind "megakernel engine".
- No old name works beside a new one. The code has no guard for an old variable.
- Old code goes. Two build knobs that no build set went with their branches.

| Old name | New name |
|---|---|
| `MK_SHARDS`, `MK_FAST_QMM`, `MK_TILES`, `MK_ELIDE`, `MK_FUSE*`, `MK_TAIL` | the same names with `QWEN38_` |
| `MK_DEBUG_LAYERS`, `MK_MEGA_GRID`, `MK_MUTATE_*`, `MK_WATCHDOG_DEADLINE_S` | the same names with `QWEN38_` |
| `MK_CHECK`, `MK_DIE`, `MK_CHECK_BYTES`, `MK_TRAP` | `HIP_CHECK`, `HIP_DIE`, `HIP_CHECK_BYTES`, `HIP_TRAP` |
| `MK_FIELD`, `MK_SIZE`, `MK_CONST`, `MK_UNION` | `STAGE_FIELD`, `STAGE_SIZE`, `STAGE_CONST`, `STAGE_UNION` |
| `kMkBlock`, `MK_BLOCK` | `kBlockThreads`, `BLOCK_THREADS` |
| `MK_EXPERT_LOOKAHEAD` | `QWEN38_EXPERT_LOOKAHEAD` |
| `MK_ENGINE_BUILD_SHA256`, `_COMPILER`, `_HIP` | `QWEN38_BUILD_SHA256`, `_COMPILER`, `_HIP` |
| `namespace mk`, `mkgguf` | `namespace qwen38`, `gguf` |
| `mk_engine_open`, `mk_relayout`, `mk_launch_fused`, `mk_launch_each` | the same names with `qwen38_` |
| `mk_ctx_round_gdn_bf16`, `mk_max_coop_grid`, `mk_status_name` | the same names with `qwen38_` |
| `mk_rt_*`, `mk_trap_sink_install`, `mk_trap_format` | `hip_rt_*`, `hip_trap_sink_install`, `hip_trap_format` |
| `mk_tid` | `thread_id` |
| the op wrappers `mk_qmm`, `mk_qmmh`, `mk_moe`, `mk_attn`, `mk_gdn`, `mk_tile_*`, `mk_hc_*` | `qwen38_*` |
| `libmkengine.a`, `libmk_<op>.so` | `libqwen38engine.a`, `libqwen38_<op>.so` |
| `MkEngine`, `MkError`, `MkContext`, `MkPaths`, `MkLaunch`, `MkSegment`, `MkFactory` | `Qwen38Engine` and so on |
| `crates/nagaya/src/mk.rs`, `tests/mk_wire.rs`, `tests/mk_stop.rs` | `qwen38.rs`, `qwen38_wire.rs`, `qwen38_stop.rs` |
| the engine name `mk` of the gfx1151host profile | `qwen38` |
| the memfd `mk-weights`, the thread `mk-engine` | `qwen38-weights`, `qwen38-engine` |
| the fixture keys `mk_rev`, `mk_files`, `mk_src`, and the toolset `mk` | `reference_rev`, `source_files`, `reference_src`, `reference` |
| `MK_REPO`, `MK_REF_THREADS`, `MK_OPS_LIBDIR` | `QWEN38_REFERENCE_REPO`, `QWEN38_REFERENCE_THREADS`, `QWEN38_OPS_LIBDIR` |

These went with no new name:

- the build knobs `MK_WEIGHT_POLICY` and `MK_GDN_CORE_EXPERIMENT`;
- the list of seven refused knobs: `MK_ATTACH_WAIT_S`, `MK_MEGA_EVENTS`, `MK_MEGA_PER_STAGE`, `MK_SCRATCH_WARMUP`,
  `MK_STAGE_TIMING`, `MK_STAGE_TIMING_PRINT` and `MK_UNFUSED_GRID`;
- `src/table.rs`, which no module named.

The gates. The `hip` suite of the engine gave 274 passed, 0 failed and 10 ignored on the card. The oracle bench on
the exact path gave the five logit hashes of 2026-10-31. Thus the rename changed no bit. The engine identity
changed, because the build hash and the names of two bit settings are in it. Each held context filled again once.

The two sections below came out of `docs/deploy.md` on 2026-11-12. They tell how the nagaya took the place of the
old service on gfx1151host, on 2026-10-21. The old service is gone, so nobody can do these steps now.

### Cut over from mk

Do these steps in this order. Stop at the first step that fails, and do the rollback below.

1. Check that nothing else is a GPU client (spec 3.1). The nagaya is the only client.

   ```
   systemctl --user is-active megakernel.service
   fuser -v /dev/kfd /dev/dri/renderD* 2>&1
   ```

   The first command says `active` while mk runs. The second command lists the processes that hold the GPU. Stop any
   job of ours that it lists. Do not stop a process that you do not know.

2. Stop and disable mk. The stop frees mk's weights and its GPU queues.

   ```
   systemctl --user stop megakernel.service
   systemctl --user disable megakernel.service
   ```

3. Enable and start the nagaya.

   ```
   systemctl --user enable --now nagaya.service
   ```

   The command returns when the process sends `READY=1`, or when `TimeoutStartSec` (300 s) ends. While mk opens, the
   process sends `EXTEND_TIMEOUT_USEC=` every 5 s, so a slow load does not time the start out. The process ends a
   hung open itself after 300 s and exits 1.

4. Check the state of the unit and its log.

   ```
   systemctl --user status nagaya.service
   journalctl --user -u nagaya.service -b --no-pager | tail -40
   journalctl --user -u nagaya.service -f
   ```

   `status` shows `Active: active (running)` and a `Status:` line with the state of each engine. The log has the
   line "ready with 1 engines". Use `journalctl -f` to follow the log. Stop it with Ctrl-C.

5. Ask for the identity of the served models.

   ```
   curl -s http://198.51.100.20:8090/council/v2/identity
   ```

   The reply is JSON. It lists the served model with its identity, limits, capabilities and status.

6. Run gogi's conformance suite against the process. It needs no restart.

   ```
   cd ~/src/gogi
   nice -n 10 cargo run --release -p gogi-conformance -- http://198.51.100.20:8090
   ```

   The last line counts the checks. On 2026-10-21 it was `215 ok, 0 FAIL, 5 skip`. Each `FAIL` line names a check. Add
   `--model <name>` to check one served model. The flag `--allow-restart <cmd>` also checks a restart: since
   `47f2191`, a pinned snapshot stays through a clean restart (spec 6.11). b70host ran the check on 2026-10-23, and
   on gfx1151host each restart loads the weights again ("Arenas through a restart").

7. Look for GPU errors in the kernel log.

   ```
   journalctl -k -b --no-pager | grep -i -E 'amdgpu|kfd' | tail -20
   ```

   Any `amdgpu` error after the start is a failed gate. Do the rollback.

### Roll back to mk

Use this when a gate fails, or when Amy asks for mk. mk is an archive since milestone 15 (`docs/ontology.md`). The
rollback needs its unit file and its built library, which the archive keeps.

```
systemctl --user disable --now nagaya.service
cmp ~/src/megakernel-qwen38-flashnext-strixhalo/deploy/megakernel.service ~/.config/systemd/user/megakernel.service
systemctl --user enable --now megakernel.service
systemctl --user is-active megakernel.service
curl -s http://198.51.100.20:8090/mk/v1/health
```

`systemctl --user disable` removes a unit that is a link into a repository. The cutover removed mk's link that way
(seen on 2026-10-21). Thus mk's unit in `~/.config/systemd/user/` is now a copy of the file in mk's repository, and
`is-enabled` says `disabled`. The second line checks that the copy is the same as mk's file. If `cmp` reports a
difference, copy mk's file again and run `systemctl --user daemon-reload`.

The third line starts mk, which loads its weights. The fourth line says `active`. The `curl` answers when
the load ends. mk serves its own route, `/mk/v1`, on port 8090. Its answer proves that mk is back.

To go back to the nagaya later, link its unit again first, as in "Install".

Nothing in the store directory needs a change for the rollback. The nagaya leaves its store and its profile there.
