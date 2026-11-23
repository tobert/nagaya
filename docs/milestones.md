# Milestones

This doc holds the open milestones of the nagaya: their order, their rules and their steps. The finished milestones
are in `docs/history.md`, with their rules, their steps and their commits. A comment that says "`AGENTS.md`
milestone N" points to `docs/history.md` for milestones 1 to 30, and to this doc for the later ones.

## The order

Amy, 2026-11-11: "go ahead and rework milestones in whatever order makes sense now." This is the order (Claude
Fable 5.2, 2026-11-11):

1. Milestone 31 steps 1 and 2 for the Qwen3.8 engine: done on 2026-11-12, and deployed on gfx1151host.
2. Milestone 34, Qwen3.8 catches up, on gfx1151host. It holds milestone 30 steps 7 to 11. Its steps 1 to 6 are done. It
   can run beside the next item.
3. Milestone 31 steps 1 and 2 for the other engines, on b70host. Then steps 3 to 7: the docs, the machines,
   the build, the license and the review.
4. Milestone 31 step 8, the publication, when Amy says so.
5. Milestone 35, Inkling 2, on b70host: planned on 2026-11-21. Amy decides how much of it to do.
6. Milestone 36, the speed of Inkling 2: a list of findings, not started. An agent of kaijutsu may take an item.
7. Milestone 37, Inkling 2 on GB10 (`sm_121`): started on 2026-11-22 on a DGX Spark.

Milestone 32, context records, starts in gogi and runs beside these. It changes the contract, so it must be done
before the publication.

These steps of the older milestones are open. Do not start one of them unless Amy asks:

- Milestone 21, image speed on the B70 cards: not started.
- Milestone 24 step 3, the refusal probe: it waits for Amy's photos.
- Milestone 25 step 8 and milestone 26 step 4: trials with Amy.
- Milestone 28 step 5: its first trial ran. More trials are Amy's choice.
- Milestone 29 step 5, the merge, and step 6, the latent swarm: they wait.

## Milestone 30: Qwen3.8 batches and latent steps

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

## Milestone 31: the scrub for a public repository

Make the repository fit for a public post on GitHub. Amy started it on 2026-11-11, and the quotes are in
`AGENTS.md`, "Direction" and "Names". She decided these rules on 2026-11-11:

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

## Milestone 32: context records

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

## Milestone 33: swarms and the stream of thought

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

## Milestone 34: Qwen3.8 catches up

The Qwen3.8 engine on gfx1151host gets what the LFM3 engine has for the lab: a tap, control vectors and latent steps.
It started on 2026-11-12. The order is by Claude Fable 5.2. These rules were decided on 2026-11-12:

- **The residual.** The state between two layers of Qwen3.8 is 4 streams of 2,560 values. `residual` is the 4
  streams, 10,240 values, for a tap and for a control vector. The head's row is a tap of its own, `fold`.
- **The canary.** The build hash is out of the identity digest (spec 12.1). A canary of the Qwen3.8 engine takes
  its place (spec 8.1). Its record is a table in the crate. A build with other bits does not open until a commit
  records its hash. A set with no record refuses the start. The canary runs at each load.
- **Control vectors.** The op ran on the host first. A device op at the launch cuts took its place (step 7).
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
   with no fused multiply-add. Two forms, the simpler first: a small kernel of its own that runs at the launch cuts,
   then an op inside the megakernel.
   - The first form is done (2026-11-17, `hip/src/engine/control.hip`). Its ISA keeps the order, and the card gate
     `control_card.rs` gives equal bits. A prefill of 256 tokens under `ablate` on 48 layers takes 468 ms, where a
     plain one takes 472 ms. A decode step under that list takes 79 ms, where a plain one takes 64 ms.
   - The second form is open. The cost of a steered decode step is now the 48 launch cuts. Not deployed yet.
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

## Milestone 35: Inkling 2

Thinking Machines released Inkling 2 Mini and a research model on 2026-11-20. Amy started this milestone on
2026-11-21: "Let's plan it all, we'll decide how much to do later. We will be exploring this model thoroughly so I
want to design for its features." The engine runs on one B70 card of b70host. Its place in the profile waits for
the deploy. The plan is by Claude Opus 5.5, 2026-11-21.

The models are `thinkingmachines/Inkling-2-Mini` and `thinkingmachines/TML-d24-varloop-research` on Hugging Face,
under Apache 2.0. The reference is the transformers model `inkling2_loop`, added in commit `046bdb8cc2`. The facts
below come from its `config.json` and `modeling_inkling2_loop.py`.

- **Layers.** 24 layers of width 2,304, with 24 query heads and 8 key-value heads of 128. Layers 5, 11, 17 and 23
  attend to all positions. The other 20 layers attend to a window of 1,024 positions.
- **Cycles.** Layers 6 to 17 run 3 times for each position. Each cycle after the first starts again from the
  residual after layer 5. It reads a message from the cycle before it, through learned gates of rank 64.
- **The state of a cycle.** Each cycle has its own keys, values and convolution states. Mini is post-trained at 3
  cycles. The research model is trained at 1 to 5, and the reference reads the count from its config at load.
- **Positions.** There is no rotary embedding. Each layer adds a relative bias that the hidden row selects, over
  the distances 0 to 1,023. The full layers also scale the query by the log of the position.
- **Attention details.** Each head has a learned sink logit for each token. Queries and keys are clipped by their
  L2 norm. Values get an RMS norm with no weight.
- **Residuals.** The residuals are attention residuals over blocks of 6 layers. Each layer input is a softmax over
  the stacked outputs of the blocks before it, in `f32`.
- **Convolutions.** Each layer has four short convolutions of kernel 4. In the full layers, the taps of the key and
  value convolutions change with each token. A context holds the convolution states, as an LFM3 context does.
- **Experts.** 128 experts, 8 for each token, with sigmoid routing and a bias. The SwiGLU has a softcap of 4 on the
  gate and 25 on the up row. The logits have a softcap of 30.
- **Weights.** One file of about 20 GB. The experts of layers 0 to 19 are NVFP4, and those of layers 20 to 23 are
  MXFP8. The rest is `bf16` and `f32`. The B70 has no FP4 unit, so the kernels expand the values.
- **Other parts.** A ViT of 23 layers for images of up to 4,096 px on each side. An MTP head of 3 layers that drafts
  blocks of 8 tokens. A tiktoken vocabulary of 201,024 entries. A template with `reasoning_effort` from 0.0 to 0.99.

The cost on a card: the weights stay in the formats of the checkpoint, about 19.5 GB. Expanded to `bf16`, the
experts alone take 49 GB. The keys and values are `f32`, so a slot of 32,768 positions takes about 2.9 GiB
(`docs/engine-abi.md`, "Inkling 2"). Thus a card of 32 GiB holds the weights and about 3 such slots.

The names are `nagaya-mk-inkling2-bmg` and `nagaya-text-inkling2`. The family is `inkling2`. Each model is an engine
of its own with an identity of its own.

The ontology defines "Pass" as one run of an engine. Thus this plan writes "cycle" for one walk through layers 6
to 17. The reference uses the same word (`middle_cycle_cross_pass_queries`).

The questions of the plan, and their answers:

1. **Cycles and layer loops.** A layer loop of LFM3 runs a span again on its own output. A cycle of Inkling 2 starts
   again from layer 6 and reads a message. Amy, 2026-11-21: "perhaps cycles should just be called cycles if they're
   different things?" Thus a cycle has a capability and a context member of its own, `cycles`. The engine declares
   no `layer_loops`. A context with no `cycles` runs 3 cycles (`docs/engine-abi.md`, "Inkling 2").
2. **Activations of the experts.** The quant config asks for NVFP4 activations before each expert. The reference
   expands each expert to `bf16` at load, and its activations stay `bf16` (`Inkling2LoopDequantizeExperts` in
   transformers). It rounds each expanded weight to `bf16`. In the checkpoint, row `2i` of `w13` is gate row `i`,
   and row `2i + 1` is up row `i`. Proposed: the engine expands the weights as the reference does.
3. **Drafts of the MTP head.** They wait. Amy, 2026-11-21: "MTP can wait, I'm not a big fan, would rather make the
   model fast". The head learned the plain model at 3 cycles. Under a control vector or another count of cycles, its
   drafts miss more often.
4. **The effort.** The template writes `Thinking effort level: <n>` from a number of 0.0 to 0.99, and a gogi context
   has only `thinking: true | false`. Proposed: `false` gives 0 and `true` gives the default of the template, 0.9.
   A number for the levels between them is a later change of gogi. Amy agreed on 2026-11-21.

Do the steps in this order. Each step ends with the checks green.

1. **Spikes and the oracle.** Make a CPU venv with transformers at `046bdb8cc2`. Measure the time and the memory of
   one prompt on the CPU.
   `tools/inkling2_golden.sh` writes the rows after each run and the block outputs, messages and gates of each cycle.
   - Done on 2026-11-21. The reference keeps the experts in `bf16` and expands each one to `f32` in its call, so
     every sum is `f32`. A model holds 74 GiB of RAM, and a prompt of 637 tokens takes 19 s.
   - A run in `bf16` differs from the `f32` reference by 0.18 to 0.22 on average in each logit. Its argmax differs
     at 2% to 4% of the positions. Thus the oracle is `f32`, and the engine's sums are `f32`.
   - The self checks: 48 runs in order at 3 cycles, and the cache of the reference gives the logits of no cache to
     3.6e-5. The two models share every file but `config.json`, so one family library serves both.
2. **Spec and ABI.** `docs/engine-abi.md`, "Inkling 2": the parts of a state, the cycles, the taps, the weight
   formats. The ontology gets "Cycle". Each rule that the docs leave open is **Proposed**. gogi writes the context
   member `cycles` and its suite checks.
3. **The family library.** The tiktoken tokenizer and the template through `nagaya-jinja`. The reply parser with the
   thinking block and the tool calls, the escapes and the council rendering. Fixtures of the reference.
   **Proposed (Gemini 4 Hydrogen):** `vocab_size()` returns `unpadded_vocab_size` (200,058), not `vocab_size` (201,024).
   Reason: `lm_head` and the logit rows have width 200,058, and the tokenizer holds 200,058 entries.
4. **Weights.** The safetensors reader for NVFP4, MXFP8 and `f32`. A host op expands each format, and it is the
   oracle of the device op. The card gate is equal bits.
   - The host ops are done (`24ab9dc`): the bits of the reference on 8 real slices and on edge cases. A scan of the
     checkpoint found no NaN code. The factor of an expert follows the loader of the reference.
5. **The text engine, 3 cycles.** The ops: the relative bias, the sinks, the clips, the log scale, the convolutions,
   the attention residuals, the gates, the routing and the experts. Prefill, decode and snapshots. Card gates
   against the oracle, and two plans with equal bits.
   - The host forward is done (`00940c5`): within 3.3e-5 of the `f32` oracle on both models, 1 to 5 cycles.
   - The device ops are done, and each card gate compares an op with its host form. The gaps: the experts
     (`ee07cbc`) 7.6e-7, the layer parts (`348c403`) 5e-7 at most, the attention (`556f81b`) 1.3e-5 at
     most. Each op gives the
     same bits in any block, and a decode step gives the bits of its position in a block.
   - The host spec is done (`b0655d5`): a slot of 32,768 positions takes 2.94 GiB, and the weights 18.9 GB.
   - The ABI has `segment.cycles` (`ce9f45a`). The engine refuses a count until step 7.
   - The engine is done on the branch `inkling2-engine`: the upload, the ring of a pass and `impl Engine`. The card
     gates compare it with the `f32` oracle:
     - the 48 run rows of "plain": within 9.1e-7 (relative L2);
     - the top-64 logits and the logsumexp of five prompts of both models: within 2.0e-5;
     - 8 decode rows: within 4.1e-5.
   - Past the end of a ring, at 2,200 positions, the logits are within 1.5e-5 of the host form. A prefill of 512
     tokens takes 799 ms, and a decode step about 70 ms.
6. **Council and deploy.** The factory, the registries, an example profile, and the gogi suite on the card.
   - Done on 2026-11-21 (`crates/nagaya/src/inkling2.rs`, spec 11.18). The B70 example has `inkling2-mini` on
     demand on card 0.
   - The gogi suite through a process on card 0 (`xe_inkling2.rs`, a reserve of 48 runs): Mini gave 446 ok, 0 FAIL
     and 22 skip. The research model gave 443 ok, 0 FAIL and 23 skip. Each gave 27 of 28 checks of cycles. The
     other check needs `layer_loops`.
   - The first run found a defect of the family: a generation was a 500, because the template has no generation
     prompt. The family now gives `<|message_model|>` (spec 11.18).
   - With `thinking: false`, both models still write a thinking block, so the suite holds no head after such a reply.
   - The place on b70host: card 0 is for Inkling 2. Card 1 keeps its pinned engines and `lfm3-omni`. Amy,
     2026-11-21: "card 0 can load inkling 2, if it lives up to the hype it will likely be there a while".
   - Deployed on 2026-11-21 from `c936337`. A text engine on demand with a family table is refused (`deploy.rs`), and
     the family of LFM3 needs its table. Thus `lfm3` stays resident, and `inkling2-mini` is on demand on card 0. A
     load names `lfm3` in `unload` and takes about 15 s, and the engine stays until another load. Its family reads
     `INKLING2_MODEL_DIR` from the unit. Live: a chat and the counts 1 and 5 gave replies.
   - The gogi suite through the live process on `inkling2-mini`, build `b803170`: 500 ok, 0 FAIL, 19 skip.
7. **Cycles in a call.** The context member `cycles` chooses 1 to 5 cycles. Card gate: the research model against
   the reference at each count. Mini takes the same counts, out of the range of its training.
   - The engine is done (`xe_cycles.rs`, 2026-11-21). With a reserve of 48 runs, it runs 1 to 5 cycles. A pass of 4
     or 5 cycles runs as 2 launches.
   - The card gate compares "plain" and "chat" at 1 to 5 cycles with the `f32` oracle. The top-64 logits are within
     2.5e-5 on Mini and 2.4e-5 on the research model. The logsumexp is within 7.7e-6.
   - The count 3 by name gives the bits of no count. The same split at 5 cycles gives the same bits twice.
8. **The lab.** The taps: the row after each run, the block outputs, the messages, the gates and the routes. Control
   vectors, latent steps, and `ensemble` over counts of cycles.
   - Done on 2026-11-21. The card gate `tests/xe_lab.rs` passes on card 0. Each tap gives the rows of the host
     form, and steering stays within 3.0e-5 of the host form. A latent row of the placeholder gives its bits.
   - A tap or a control list adds 0.2 to 1.0 ms to a decode step of 68.5 ms.
   - The norms remove the scale of a row: twice a latent row moves a logit by 8e-6 in the host form.
9. **Images.** The ViT, the placeholders of an image and its sizes. Card gate against the reference.
   - The oracle stage and the host form are done (`617dfb6`): the logits of a prompt with two images within 2.2e-5
     of the oracle.
   - The rows and the records of the family, and the device ops of the tower (`tests/xe_vision.rs`), are on main.
   - The engine runs the tower in the pass (engine-abi "The choices of the pass"). It declares `tokens+rows` and
     the term `vision`, and it holds the projector rows of the last image for the later blocks of its span. The
     process gives the family its reader of objects. A council test with the fake engine checks the placeholders,
     the records and the rows of a chat with an image part.
   - The card gate `tests/xe_images.rs` passed on 2026-11-21. It feeds the oracle prompt with two images, 235
     positions, in one block, in a cut inside image 1, and in blocks of 64. Each split gives the top-64 logits
     within 2.1e-5 of the oracle and the logsumexp within 3.8e-6. The three splits give the same bits, and the same
     split twice too. The tower runs once for each image, also for image 1 in 4 blocks. One block takes 523 ms.
   - The largest image is 4,096 by 4,096 px, 10,609 tokens. Its tower takes about 56,780 ms in the pass of its first
     block (`tests/xe_long.rs`, 2026-11-21). That is 28 times `max_quantum_ms` of 2,000 ms. A later block of
     the image takes 843 ms. A second feed of the same image reads the held rows, and its tower does not run.
   - Each side of an image is 1 to 4,096 px, so an image has at most 10,609 tokens. Amy, 2026-11-21: "yep, match
     the model card. use the 4096 rule". A side below 40 px pads, as the reference does. Amy chose it the same day,
     when the gogi suite sent an image of 8 by 8 px. The council never resizes an image.
10. **Drafts.** The MTP head drafts tokens, and a pass checks them. This step waits (question 3).
11. **Long contexts.** The model reads up to 1,048,576 positions. Measure the ledger cost and the speed at 256K.
    - The cost of a slot is measured on 2026-11-21 (engine-abi "The cost of a slot"). One card holds a slot of
      164,352 positions at 3 cycles, and 106,496 at up to 5. A position costs 68,656 bytes at 3 cycles.
    - The open and the ledger do not refuse a slot that does not fit the card. A slot of 262,144 positions opens,
      and the driver puts a part of it in system memory. Then a block takes 4.0 to 5.0 s, not 0.78 s. Thus 256K
      does not run on one card at the speed of the card. Since step 12, the open refuses such a slot.
    - The card gate `tests/xe_long.rs` feeds 130,560 seeded ids into a slot of 131,072, in blocks of 512, and
      decodes 8 steps after them. The host form runs from position 0 up to 1,536. At 1,527 a route flips at a near
      tie (engine-abi "A route near a tie"). Thus later, the host form runs each checked block from the state that
      the card read. There are 8 such blocks, from 512 to 130,048.
    - The gate passed on 2026-11-21. Its 57 rows are within 2.0e-5 of the host form, and each argmax agrees. The
      rows from position 0, the rows of the 8 blocks and the 8 decode steps each stay within 2.0e-5. The gate takes
      about 20 minutes, most of it in the host form.
    - The gate found a defect of the host form: it summed the attention in `f32`, and that sum drifts with the
      count of keys. Its logits moved by 4.8e-2 at a decode step after 130,560 positions. The host form sums in
      `f64` now (`acbccb2`).
    - The time of a block of 512 tokens at each position:

      | Position | 0 | 8,192 | 33,280 | 66,048 | 130,048 |
      |---|---|---|---|---|---|
      | Block | 790 ms | 848 ms | 1,027 ms | 1,279 ms | 1,761 ms |
      | Tokens each second | 648 | 604 | 499 | 400 | 291 |

      The feed of 130,568 positions took 339 s. The slowest pass took 1,768 ms, within `max_quantum_ms` of 2,000 ms.
    - The time of a decode step, the median of 8 steps:

      | Position | 1,024 | 8,704 | 32,768 | 65,536 | 130,560 |
      |---|---|---|---|---|---|
      | Step | 69.1 ms | 71.7 ms | 73.9 ms | 82.6 ms | 97.9 ms |
12. **Trial and review.** A trial with Amy, and a kaibo review round with a cast of another model family.
    - The trial ran on 2026-11-22 with an image, on `2a3f433`. Amy: "It seems to be doing well on a test image
      right now. 12 tok/s". That is about 83 ms for each token, near the decode step of step 5.
    - Round 1 ran on 2026-11-21 with DeepSeek, Gemini and ChatGPT (`gpt-plan`). The cast `gpt` had no credits.
    - Not taken: Gemini said the window reads 2,048 keys and the block outputs overflow. The window is 1,024
      (`inkling2_attn_helpers.cl:24`), `BLOCK` is 6 (`host/weights.rs:22`), and the card gates of the window and
      of the 24 messages pass.
    - The fixes of round 1, each with a test and its mutations (branch `inkling2-review`, 2026-11-21):
      - The identity names `extra_runs` (`0c01111`).
      - The plan checks the name, the dtype and the shape of each tensor that the load reads (`497a9fc`).
      - A snapshot holds the part `cycles`, and the runs that its count does not reach read as zeros (`b744330`,
        `c0a6258`). The identity names the layout of the parts, `state` 2.
      - The probe of the oracle checks before it clears, and `last_error` clears on a call that does not fail
        (`442e272`).
      - The stale text of the ABI is gone (`df44845`).
      - The family must have the template of the identity (`f1b854a`), and the template id covers the placeholders
        of an image (`50ec3ed`).
      - The open refuses a slot that the card does not hold (`e8d66eb`, `553842c`). The budget of a card in the
        example profile stays 32 GiB. A budget of the usable memory waits for the owner.
    - Not fixed: the tower of an image of 10,609 tokens takes about 56,780 ms, and `max_quantum_ms` is 2,000 ms. The
      watchdog of the process stops a group with no beat for 30 s. The ABI holds a **Proposed** rule: the tower in
      quanta of its own (`dc37002`, engine-abi "The time of a large image").
    - Round 2 ran on 2026-11-22 with ChatGPT (`gpt-plan`), on the changes `46a1551..cffa0a5`. It gave 6
      findings, and each was true. The fixes, each with a test and its mutations (branch `inkling2-review2`):
      - The open fills the two buffers of the card and launches the halt before a second check (`337ffbf`).
        Before, the first pass took 983,040 bytes of the card after the check, with deferred memory.
      - The active rings of a restore start at one position (`0d9e7ef`).
      - The plan checks each tensor of the tower (`29c637e`).
      - The family compares its template before it renders, and a failed test render is an error (`3083900`).
      - The probe checks its slot before the cast to `u32` (`beabc39`). The ABI says which calls set
        `last_error`.
    - Not changed in round 2: the memory check reads the whole process on the card. The ABI states its limits
      (`07739d2`). A refused open in a swap still stops the process. An `open_problem` of the factory of
      Inkling 2 waits, on the route of milestone 34 step 1.

**Speed.** Each finding about the speed of the engine goes into milestone 36, not into this milestone.

**Ports, later.** Amy, 2026-11-21: "not sure we'll do either port yet, pretty excited to see how it does on intel
here first". These notes are for a port to a machine of 128 GB of unified memory, gfx1151host or a DGX Spark:

- Keep the experts packed in memory, and expand them in registers or local memory before each matrix op. A
  decode step reads each weight once, so it is bound by bandwidth. In `bf16` it reads about 6 GB of experts for
  each token, where packed it reads about 1.7 GB.
- The expansion changes no bit: it gives the `bf16` of the reference. Thus a copy in `bf16` is a choice of
  bandwidth and capacity only. A prefill block uses each expert for many tokens, so a tile that is expanded once
  into local memory can serve many token tiles.
- The Qwen3.8 engine on gfx1151host expands its IQ4_XS weights in its matmuls (`qmmh.h`). A port to gfx1151host follows it,
  with the expansion helpers of `kernels/inkling2_helpers.cl`.
- A DGX Spark has NVFP4 units. The checkpoint asks for NVFP4 activations too (`hf_quant_config.json`, per-token,
  `fouroversix_mae`), and the oracle of this milestone keeps them `bf16`. A port that uses the NVFP4 units needs
  an oracle and an identity of its own.

**Lanes.** Amy, 2026-11-21: "our plan should use subagents to drive a lot of the work efficiently in passes,
identifying where things can go in parallell. we may also farm some out to other agents via text files". The work
goes in waves. Each lane has its own worktree and target directory (`~/src/wt/nagaya-inkling2-<lane>`).

- **Wave 1, after step 2.** Three lanes share no file: the oracle (step 1), the family library (step 3) and
  the weight formats (step 4). One more lane writes a host op of each layer part, in Rust, against
  the rows of the oracle. These ops are the oracles of the device ops of step 5.
- **Wave 2, step 5.** Each device op is a lane with its host op as its gate. One session owns the task rings and
  the card, because card tests run one at a time on a released card.
- **Wave 3, after step 6.** Cycles in a call (step 7), the lab (step 8) and images (step 9) share only the
  engine's `spec.rs`. Merge them in that order.
- **One owner of the card.** One session at a time opens card 0. It releases the card, runs its card gates, and
  reclaims it (spec "The release of a device"). The lead session gives the card to one lane at a time.
- **Briefs for another agent.** A lane that needs no card and no model run can go to an agent of another family.
  Write its brief as `BRIEF.md` in the worktree, and do not commit it. The brief names the files, the fixture, the
  test that must pass, and the rules of `AGENTS.md` that apply.
- **The return of a brief.** The agent commits on the branch of its worktree. The lead session reviews each commit
  and runs the mutations again before the merge.

These lanes fit a brief: the tokenizer, the reply parser, the host expansion of NVFP4 and MXFP8, and the host
ops of the layer parts. The card work, the task rings and the spec stay with the lead session.

Each stop and start of `nagaya.service` goes into `signoff.md`.

## Milestone 36: the speed of Inkling 2

Amy, 2026-11-21: "as we notice things that can help future perf let's add them to a next milestone. not for today,
might have a kaijutsu agent go after some". This milestone is a list of findings about the speed of the Inkling 2
engine on one B70 card. Each item stands alone, and an agent may take any one. Add each new finding here.

Amy set the order on 2026-11-21: "mostly we'll want some batching for when we want to try swarm or parallel
contexts I think; we can focus on other perf gains first". On 2026-11-22 she let the agent choose the order: "if
pulling swarm forward puts us on the right path I'm all for making it next". Thus the batch (items 10 to 12) comes
next after item 1, and then the other items. Reasons:

- With 8 rows, each run reads about 51 experts, so each expert gets one or two rows. That is the path of item 1.
- Item 1 and `FLAG_TILED` both move bits. Back to back, the engine identity moves once, and the gap is measured once.
- A batch shares the fixed costs of a step: the head, the dense weights and the task boundaries.

**The ceiling of one row.** A decode step must read each weight that it uses once. At 3 cycles it reads these:

- about 1.7 GB of experts and 0.92 GB of the head;
- 0.5 to 1 GB of the other dense weights, an estimate (item 13).

At 608 GB/s, with about 2 ms of task boundaries, a step takes at least 7 to 8 ms: 120 to 150 tokens each second.
Persistent kernels seldom hold more than 60% to 70% of the bandwidth, so 60 to 90 tokens each second is a fair goal.
A batch can pass this ceiling in the tokens of all its rows.

**Where the engine is.** Milestone 35 step 5 measured it on 2026-11-21 (`effd4d2`):

- A prefill of 512 tokens takes 799 ms, about 640 tokens each second.
- A decode step takes 65 to 70 ms. A pass of 3 cycles is 1,635 tasks in one launch.
- A decode step reads about 1.7 GB of packed experts: 8 experts in each of 48 runs. At 608 GB/s, that sets a floor
  near 3 ms.

**The rules of each item.**

- Measure before and after, on the same prompt and slot, and write both numbers in the commit.
- Profile the change with `tests/xe_profile.rs` of the Inkling 2 crate (item 0). Write the row of the op before and
  after in the commit.
- Keep the gates of milestone 35 green: `xe_oracle`, `xe_feed` and the card gate of each op. A change of the order
  of a sum changes bits, so it changes the engine identity. Say so in the commit, and measure the gap again.
- A host form in `src/host/` stays the oracle of each op. A new op gets a card gate against it.
- One owner of card 0 at a time (milestone 35, "Lanes").

**0. The profile of a decode step: done on 2026-11-22.** The profile binary of Inkling 2 is the engine binary with
`MK_PROFILE`. Each work-group writes three clocks of each task: at its start, after its wait, and at its end.
`Inkling2Engine::open_profiled` loads it, and `take_profile` gives the clocks. `src/profile.rs` sums them for each
op kind. The test `xe_profile` prints the tables below.

- The clock is `clock_read_device`. It counts at 2,800 MHz, and the Xe cores agree to about 600 counts.
- The profile binary gives the bits of the engine binary. Its tasks cover 99.9% of the host time of each launch.
- "Work" runs from the first work-group past the wait to the last work-group at the end. "Boundary" is the time
  before it. "Spread" runs from the first work-group at the end to the last one. A spread near the work says that
  one work-group did the op.
- The engine binary takes 70.8 ms for a decode step at position 1,024 at 3 cycles. The profile binary takes 72.1 ms
  in its tasks. The host part of a pass is about 0.3 ms.
- The noise is small. Over 16 decode steps, the sum of the tasks is 72.02 to 72.14 ms. Over 5 blocks, it is 803.32
  to 803.43 ms.

A decode step at position 1,024 at 3 cycles, the mean of 16 steps (prompt "longer" of the oracle, repeated):

| op | tasks | work ms | boundary ms | mean work us | spread ms | share |
|---|---:|---:|---:|---:|---:|---:|
| EXPERT_UP | 48 | 24.35 | 0.02 | 507.3 | 0.26 | 33.8% |
| MESSAGE_GATE | 96 | 10.11 | 0.05 | 105.4 | 10.01 | 14.1% |
| MATMUL | 385 | 9.07 | 0.23 | 23.6 | 0.42 | 12.9% |
| EXPERT_DOWN | 48 | 8.82 | 0.04 | 183.7 | 0.14 | 12.3% |
| INKLING2_ATTN | 48 | 7.14 | 0.02 | 148.8 | 3.04 | 9.9% |
| ATTN_RESIDUAL | 120 | 6.24 | 0.08 | 52.0 | 6.11 | 8.8% |
| RMSNORM | 154 | 1.24 | 0.08 | 8.0 | 0.98 | 1.8% |
| MOE_PLAN | 48 | 1.08 | 0.02 | 22.5 | 1.00 | 1.5% |
| RESIDUAL_CONV | 192 | 0.99 | 0.10 | 5.2 | 0.25 | 1.5% |
| HEAD_CLIP | 192 | 0.63 | 0.14 | 3.3 | 0.45 | 1.1% |
| SIGMOID_ROUTER | 48 | 0.35 | 0.02 | 7.4 | 0.30 | 0.5% |
| INKLING2_ATTN_SUM | 48 | 0.32 | 0.04 | 6.8 | 0.13 | 0.5% |
| MOE_GATHER | 48 | 0.28 | 0.02 | 5.8 | 0.22 | 0.4% |
| EXPERT_SUM | 48 | 0.17 | 0.03 | 3.6 | 0.09 | 0.3% |
| OUTPUT_GATE | 48 | 0.14 | 0.02 | 3.0 | 0.02 | 0.2% |
| EMBED | 49 | 0.12 | 0.02 | 2.4 | 0.07 | 0.2% |
| COPY | 15 | 0.04 | 0.01 | 2.7 | 0.01 | 0.1% |
| sum | 1,635 | 71.11 | 0.96 | | | 100% |

The variants of a decode step that the table joins:

- `EXPERT_UP` of NVFP4 takes 504 us and of MXFP8 541 us. `EXPERT_DOWN` takes 183 us and 197 us.
- `INKLING2_ATTN` takes 98 us on a sliding layer (40 tasks) and 401 us on a full layer (8 tasks). On a full layer,
  3.0 ms of 3.2 ms is spread. One chunk of 2,048 keys holds all 1,024 positions, so the op has 24 items, one for
  each head. Thus 24 teams of 4 subgroups work, of 512 teams.
- The head (`MATMUL`, 200,064 rows) takes 1.63 ms, about 570 GB/s. The matmuls of 3,072 rows take 35 us each.

One cycle costs about 20.5 ms. A decode step takes 31.1 ms at 1 cycle (736 tasks) and 72.1 ms at 3. At 5 cycles it
takes 113.1 ms (2,533 tasks in 2 launches), and the shares are those at 3, within 4 points.

A prefill block of 512 tokens from position 0 at 3 cycles, the mean of 5 blocks:

| op | tasks | work ms | boundary ms | mean work us | spread ms | share |
|---|---:|---:|---:|---:|---:|---:|
| EXPERT_UP | 48 | 309.61 | 0.05 | 6450.1 | 1.74 | 38.5% |
| MESSAGE_GATE | 96 | 142.94 | 0.08 | 1489.0 | 8.54 | 17.8% |
| EXPERT_DOWN | 48 | 140.09 | 0.09 | 2918.5 | 0.19 | 17.4% |
| MATMUL | 385 | 84.55 | 0.28 | 219.6 | 57.52 | 10.6% |
| ATTN_RESIDUAL | 120 | 67.81 | 0.08 | 565.0 | 3.37 | 8.5% |
| INKLING2_ATTN | 48 | 20.94 | 0.03 | 436.3 | 14.61 | 2.6% |
| RMSNORM | 154 | 9.63 | 0.11 | 62.5 | 0.29 | 1.2% |
| RESIDUAL_CONV | 192 | 6.96 | 0.15 | 36.3 | 0.36 | 0.9% |
| EXPERT_SUM | 48 | 4.73 | 0.13 | 98.5 | 0.19 | 0.6% |
| MOE_PLAN | 48 | 3.58 | 0.02 | 74.6 | 3.50 | 0.4% |
| MOE_GATHER | 48 | 3.38 | 0.03 | 70.5 | 0.39 | 0.4% |
| INKLING2_ATTN_SUM | 48 | 2.65 | 0.04 | 55.1 | 0.09 | 0.3% |
| HEAD_CLIP | 192 | 2.50 | 0.13 | 13.0 | 0.38 | 0.3% |
| OUTPUT_GATE | 48 | 1.44 | 0.04 | 30.1 | 0.04 | 0.2% |
| SIGMOID_ROUTER | 48 | 0.68 | 0.03 | 14.2 | 0.13 | 0.1% |
| COPY | 15 | 0.31 | 0.01 | 20.9 | 0.02 | 0.0% |
| EMBED | 49 | 0.23 | 0.03 | 4.7 | 0.07 | 0.0% |
| sum | 1,635 | 802.03 | 1.33 | | | 100% |

**What the profile shows.** The task boundaries take 0.96 ms of a decode step, not 1.6 ms. The ops take the rest.
The experts take 33.2 ms, 11 times their floor of about 3 ms. Two small ops run in one work-group of 32 and take
16.4 ms. Thus the items below are in the order of their measured gain.

**The findings, first the gains of one row.**

1. **A decode path in the expert ops: 33.2 ms of a decode step.** `OP_EXPERT_UP` and `OP_EXPERT_DOWN` always take
   the XMX path. With one token, 15 of the 16 rows of each dpas are idle. A path of SIMD dot products for one token
   can read the packed planes as fast with less work. The review of the binding by ChatGPT asked to measure both.
   - An up task reads about 24 MB in 507 us, about 47 GB/s. A down task reads about 12 MB in 184 us.
   - At half of 608 GB/s, the two ops take about 6 ms. Thus the gain is about 27 ms of a step.
2. **Wider ops of the residual and the gates: done on 2026-11-22.** `OP_ATTN_RESIDUAL` and `OP_MESSAGE_GATE` took
   one work-group for each position, and one load for each step of a sum. They now spread over all 32 work-groups,
   and each sum keeps the order of the host `dot`. Thus the bits and the engine identity did not change.
   - A team of subgroups takes a position of the residual. The gate is two tasks: `OP_MESSAGE_GATE_DOWN` writes `g`
     into the router rows, and `OP_MESSAGE_GATE` reads it.
   - The sums read their rows with subgroup block reads: one read gives each lane 8 steps in the order of the host.
     A position alone first copies its rows into local memory.
   - A decode step at position 1,024 at 3 cycles: the residual takes 1.02 ms, not 6.35 ms. The gate takes 1.55 ms,
     not 9.83 ms. The tasks of a step take 59.82 ms, not 71.85 ms.
   - A prefill block of 512: the residual takes 11.74 ms, not 67.64 ms. The gate takes 32.06 ms, not 142.27 ms.
   - `xe_feed` measured 17.3 tokens each second for a decode step, from 14.4. A prefill of 512 took 657 ms, from
     801 ms.
   - `tests/xe_bits.rs` compares the SHA-256 of the logits of 21 cases with a fixture. A change of bits fails it.
   - Left: a decode task of the gate still takes 6 to 10 us, and the megakernel spills registers (6.8 KB a work
     item). The gate as matmuls over a block can gain most of its 32 ms.
3. **The chunk size of the attention: 7.1 ms of a decode step.** It is 256 keys on a sliding layer and 2,048 on a
   full layer (Proposed in the binding). At position 1,024, a full layer has 24 items for 512 teams: 401 us. Measure
   other sizes for a decode step and for a prefill. A new size changes bits.
4. **Reuse of an expanded tile in a prefill: 450 ms of a block.** A block of 512 tokens has about 32 picks for each
   expert. Expand a tile of an expert once into local memory, and run many tiles of tokens against it. The review
   asked for it. The expert ops take 56% of a block.
5. **The balance of the matmuls of a block: 57.5 ms of spread.** The matmuls of a block take 84.6 ms, and 57.5 ms
   of it is spread. The matmul of 3,072 rows takes 345 us, and its work-groups end about 170 us apart on average.
   Give the work-groups tiles of equal work.
6. **The small ops in one work-group: about 2.3 ms of a decode step.** `OP_RMSNORM` of one row (1.24 ms) and
   `OP_MOE_PLAN` (1.08 ms) run in one work-group. Join each one to the op after it, or split it.
7. **Fewer tasks in a run: 0.96 ms of a decode step.** A task boundary takes about 0.6 microseconds. Thus 1,635
   tasks take about 1 ms. Join a small op to the task before it when it reads only its output. Examples: the clips
   with their matmul, or the gate with the output matmul.
8. **The tower of an image** (milestone 35 step 9). An image of 4,096 by 4,096 px has 42,436 patches, and its
   attention reads every pair. Its tower takes about 56,780 ms in one pass. Since 2026-11-22 it runs as a long pass
   with a declared bound (engine-abi D8), and its card gate passed through the process. A flash form of that
   attention, or a tower in several quanta, is now for its speed and for the wait of other work.
9. **Long contexts.** A slot of one card holds 164,352 positions at 3 cycles, and 106,496 at up to 5 (milestone 35
   step 11). The keys and values in `bf16` would hold about twice as many, but they change bits: a new identity and a
   new gap. Two cards could hold the keys of the full layers on the second card.
   - The attention of a block grows with the count of keys. A block of 512 at 130,048 takes 971 ms more than at
     position 0. The 8 full runs then do about 3.3 TFLOP of `f32` sums, so the op runs at about 3.4 TFLOP/s.
   - Extended in a line, a block at 164,352 takes about 2,020 ms, past `max_quantum_ms`. A slot of that length needs
     a faster attention or a smaller feed block.
   - A decode step at 130,560 takes 28.8 ms more than at 1,024. It reads the keys and values of 8 full runs: about
     8.6 GB. At 608 GB/s that sets a floor near 14 ms, so the op reads at about half of the bandwidth.

**Then the batch of decode rows** (Amy's order: after the items above).

10. **`FLAG_TILED` in the Inkling 2 binary.** `OP_MATMUL` takes another path for one row than for 2 to 8 rows. Thus a
   decode step differs from its position in a block by up to 9.1e-6. The LFM3 binary has the flag. With it, a row
   gets the same bits for 1 to 8 rows, and a decode step gets the bits of the block.
11. **A ring of several slots.** A decode pass of rows from several contexts, each with its own slot and position,
   as LFM3's `model::batch`. Raise `MAX_SEGMENTS` from 1. The rows of one pass have one count of cycles, and the
   council joins only equal counts.
12. **What a batch gains.** With 8 rows, about 51 of 128 experts run in each run. Thus a row reads about 80% of the
    expert bytes of a row alone. Most of the gain comes from the fixed costs that the rows share. They are the
    tasks, the weights of the attention and the idle rows of each dpas. Measure tokens each second at 1, 2, 4 and 8
    rows.

**Also.**

13. **Bytes in the profile.** Give each row of the profile the bytes that its tasks read, and their rate against
    608 GB/s. Then each op shows its distance from its floor, and the ceiling above stops being an estimate. Amy,
    2026-11-22, of the profile binary: "a super cool feature". The README names it as a feature to grow.
14. **The register spills.** The megakernel spills about 6.8 KB of scratch for each work item at 128 GRF (5.8 KB
    before item 2). A work-group of 1,024 threads cannot use the large GRF. Thus the time of each op depends on the
    compiler. Find the ops that spill most, and measure a binary with fewer threads in each work-group.

## Milestone 37: Inkling 2 on GB10 (`sm_121`)

The first NVIDIA engine of the nagaya runs Inkling 2 Mini on one GB10 (`sm_121`) of a DGX Spark. The machine has
121 GB of memory, unified with the GPU. The target context is 400K to 512K positions. This milestone started on
2026-11-22.

This engine is the exemplar of a CUDA engine in the nagaya. A stranger reads it to learn how we build one. Thus it
must be clean, and its speed must come from its structure. A later optimization is a tweak, not a new design.
The B70 engine shows the cost of the other way: milestone 36 lists changes that are new designs of its ops.

### The decisions

The decisions of 2026-11-22:

- **Crates and build.** `nagaya-mk-cuda` holds the shared CUDA persistent interpreter and the Driver API wrapper.
  `nagaya-mk-inkling2-sm121` holds the Inkling 2 engine. `build.rs` calls `nvcc` under the feature `cuda`.
- **Expert numerics.** The experts quantize their activations to NVFP4 and MXFP8, as the checkpoint requires
  (`hf_quant_config.json`). This engine has a quantized host oracle and an engine identity of its own.
- **The source of the activation rules.** transformers at `046bdb8cc2` expands the experts to `bf16` and quantizes
  no activation. The Inkling 2 path of vLLM (Apache-2.0, `vllm/models/inkling2/nvidia/ops/`) holds the rules that
  ship: `nvfp4.py`, `mxfp8.py`, `moe.py` and `moe_sm120.py`. Our oracle follows them.
- **The trunk is `f32`.** vLLM runs the trunk in `bf16`. Our oracle keeps every row and every sum outside the
  experts in `f32`, and it quantizes the `f32` rows. A `bf16` trunk is a later choice, if a measure asks for it.
- **The rule of the port.** Follow the model card, unless a change gives a clear gain of bandwidth.
- **Kernels.** Clean-room CUDA C++ with inline PTX. No external C++ dependency. The source of vLLM informs a
  numeric rule; we do not copy its code.
- **Long context.** Target 400K to 512K positions (`model_max_length` 1,048,576). Here 400K means 409,600
  positions. At 3 cycles in `f32`, a slot of 409,600 positions takes 27.854 GB, and 524,288 take 35.370 GB. Each
  figure includes the 40 fixed sliding rings. The attention scratch is extra, a provisional B70 bound until the GB10
  attention exists (step 9, wave C). The weights take 19.5 GB.
- **The machine.** sm121host serves only the nagaya. The vLLM of k8s goes for good. The control plane of the cluster
  scales it to 0 replicas.

### The byte budget of a decode step

A decode step at batch 1 is bound by memory bandwidth. These counts come from the checkpoint header, at 3 cycles,
with each weight in its checkpoint format:

| part | read for each token |
|---|---:|
| dense weights of the 48 runs (`bf16`): attention, gates, controllers | 2.79 GB |
| experts, 8 of 128 in each run (NVFP4 in layers 0 to 19, MXFP8 in 20 to 23) | 1.84 GB |
| the head (`bf16`, 200,064 rows of 2,304 values, 922 MB) | 0.92 GB |
| sum | 5.55 GB |

- NVIDIA gives the GB10 273 GB/s of memory bandwidth. The probe measured a read peak of about 252 GB/s, 92 to 94% of
  that figure (`crates/nagaya-mk-cuda/tests/cuda_probe.rs`, step 7 substep 1). At the measured peak, the floor of a
  decode step is about 22 ms (5.55 GB at 252 GB/s). That is about 45 tokens each second at batch 1. The figure of
  273 GB/s would give about 20 ms and about 49 tokens each second.
- The checkpoint pads the head to 201,024 rows. The engine needs 200,064: the 200,058 rows of the vocabulary,
  rounded up to a multiple of 16. The 960 rows of the padding are not read in a decode step.
- The dense weights take half of the bytes. Thus a matmul of one row is as important as an expert op.
- A weight in memory stays in its checkpoint format. An expansion to `f32` in memory doubles the dense bytes.
- The keys and values of a full run take 8 KB of each position in `f32`. Full runs are layers 5 and 23 once, and
  layers 11 and 17 at each cycle: 8 runs. At 400K positions, a decode step reads about 26 GB of them, about 96 ms.
  Thus at a long context the cache sets the speed, not the weights. Its format is a decision for Amy (step 11).

**More rows on each read.** The rows of one pass share each read of the dense weights and the head. A row adds only
its own experts until all 128 run. These estimates are the reason for design rule 3:

- A pass of 16 rows picks about 81 experts in each run. It reads about 22 GB: about 195 tokens each second in all.
- A pass of 64 rows reads about 33 GB: about 530 tokens each second in all.
- A block of a prefill and the drafts of the MTP head (milestone 35 step 10) gain the same way.

### The design rules

Each op follows these rules from its first commit. A review checks each rule.

1. **Measure bandwidth.** Each op knows the bytes that it reads. Its card gate prints its GB/s and its share of the
   measured peak. A step of the pass prints a table of each op, in the form of milestone 36 item 0.
2. **Keep the checkpoint formats.** A weight stays `bf16`, NVFP4 or MXFP8 in memory. The load can repack it into the
   fragment order of the MMA, so that each warp reads whole 16-byte vectors. A repack changes no value. Each value
   expands in registers.
3. **One path for 1 to 16 rows.** A matmul or expert op runs tiles of 16 rows on the tensor cores at each count of
   rows. A tile pads its unused rows. At one row the MMA work is free, because the op waits on memory.
4. **The bits of a row do not depend on the count of rows.** The order of each sum is a function of the shapes only.
   Thus a decode step gives the bits of its position in a block, and a batch gives the bits of each row alone. The
   B70 engine has no such rule (milestone 36 item 10).
5. **Each op uses each SM.** No op runs in one block of threads. A row of 2,304 values splits over the SMs by output
   columns or by key chunks. The split comes from constants of the shapes, not from the count of SMs, so the bits
   stay the same on another GB10.
6. **Fetch the next weights early.** A weight does not depend on the task before it. While a block waits at the
   scoreboard, it starts the reads of its next weights into L2 or shared memory. This hides the task boundary.
7. **Fuse at the design, not later.** A norm runs in the prologue of the matmul that reads it. Clips, gates and the
   residual add run in the epilogue. The quantizer of an expert input runs in the epilogue of the op that writes the
   row. The router, the plan and the gather are one op. Count the tasks of a step, and keep the count low.
8. **No race in a sum.** No sum uses an atomic add. Each reduction has a fixed tree. Each split constant is part of
   the engine identity.
9. **The `f32` trunk on tensor cores.** Each `f32` activation splits into three `bf16` terms, and the MMA multiplies
   each term by the `bf16` weight. The three terms hold the 24 bits of an `f32` mantissa. Thus each product is
   exact, and the MMA sums in `f32`. **Proposed** (Claude Opus 5.5, 2026-11-22): the work triples, but a decode step
   waits on memory. Step 9 measures the cost in a prefill. The other form is CUDA-core sums.
10. **One owner of the GPU.** One session at a time runs on the GPU. It runs every card gate and every mutation.

### The steps

Do the steps in this order. Each step ends with the checks green.

1. **GB10 probe fix.** Capture the real `dgx-spark` sysfs and procfs tree, and map `Model: Unknown` with PCI id
   `10de:2e12` to `NVIDIA GB10`. Done (`809d8514`).
2. **Spec and ABI.** The GB10 binding in `docs/engine-abi.md`, and `AGENTS.md`. Done (`21b7dd9e`). Step 6 wrote
   its rules of the activations again.
3. **The shared CUDA interpreter (`nagaya-mk-cuda`).** The 64-byte task protocol, the safetensors reader, the Driver
   API wrapper, `megakernel.cu` compiled to an `sm_121a` cubin with `--fmad=false`, `%globaltimer` profiles and
   `tools/nvcc-stub`. Done (`594b1bc4`).
4. **The host spec (`nagaya-mk-inkling2-sm121`).** The slot plan of 400K to 512K positions and the registration in
   `nagaya` under `cuda`. Done (`bca25a29`, `c81e171e`). Step 6 wrote its `quant.rs` again.
5. **GPU smoke gate.** `cuModuleLoadData` loads both cubins on the GB10. `OP_COPY` copies 16,384 values bit for bit
   over 48 blocks, and `MK_PROFILE` writes monotonic times. Done (`1fb96d0e`).
6. **Corrections.** Make the work of steps 2 to 5 fit the rules of this milestone:
   - **A fixture from the kernel that ships.** Done (`19816e11`, merged in `2e455c86`). The fixture tool runs the
     Triton kernels of vLLM at `186882d9e8`, unmodified, on the GB10, from a venv with torch and Triton. The image
     of vLLM that the node holds is an older build with no Inkling 2 path, so it is not the source. The rows hold
     ties, zero rows, zero blocks, saturation and a tiny `amax`. Each row goes in as `bf16` values, so that the
     `f32` rows of the fixture are exact.
   - **`quant.rs`** follows the rules of vLLM. Done (`4a2af81e`, merged in `2e455c86`). The global scale is
     `amax / 1536`. The order of each operation is vLLM's. The error is a pairwise sum. The UE8M0 scale rounds up
     by the bits.
     - One mutant survives. No row of the fixture decides the order of the reconstruction `(v × s) × outer`
       against `v × (s × outer)`. Add a row that decides it.
   - **The binding.** Write the rules of the activations again in `docs/engine-abi.md`, from the source of vLLM.
     They stay **Proposed**. Done (`53e26fa3`, merged in `ed82db6d`).
   - **A review** of steps 3 to 5 by a model of another family than the writer's (`docs/reviews.md`), against the
     design rules. Then run the full list of checks of `AGENTS.md` on sm121host.
     The review is done. Its findings are defects to fix in this step. A lane fixes them on a branch:
     - The smoke gate does not test the scoreboard.
     - The layout test does not compare the Rust layout with the CUDA layout.
     - A grid that is not all resident hangs.
     - The SM count falls back to 1 with no error.
     - Error messages have no name and no action for the reader.
     - Driver lifetimes: the context is not always current, and `Ring` has no `Drop`.
     - `op_copy` uses a 32-bit index.
     - `Model::parse` checks too few fields of `config.json`.
     - The slot tests are loose.
     - `identity_numbers` lacks `ring_rows` and `cycles`.
7. **The facts of the GPU and the shape of the interpreter.** Measure the facts with ops of the interpreter. They
   are the peak read bandwidth and the size of L2. They are also the cost of a task boundary and the time of each
   kind of MMA tile. Write them in the binding. Then write the budget of each op of a decode step from the table
   above. The budget is the target of each card gate.

   The interpreter of step 3 fought the design rules. Each task was a barrier of the grid. It had no prefetch
   hook, a 64-byte descriptor, no shared memory and 256 threads. Substeps 2 to 6 fixed each of them.

   The design is in `docs/engine-abi.md`, "The CUDA interpreter". It is **Proposed** (Claude Opus 5.5,
   2026-11-22). It answers three questions:
   - A task waits on two events through counters.
   - A split of one op over all SMs is a cut into work items from the shapes (rule 5).
   - Its split constants enter the engine identity (rule 8).

   Migrate in six substeps. Each ends with the checks green and its GPU gate:
   1. **Probes** on the current interpreter. Done: `2aef3a64`, merged in `213cf10d`. The facts are in the binding.

      Read 1 GB in items of 32 KB, with direct loads and with bulk copies into shared memory, each with and
      without L2 prefetch. Re-read 1 to 64 MB to find the knee of L2. Time the tiles of `bf16` `m16n8k16`,
      `mxf4nvf4` and `mxf8f6f4`. Gate: the numbers go in the binding, and each probe tile equals its host value.
   2. **The 128-byte descriptor.** A Rust layout test and a `static_assert` compare the layouts. Gate: `OP_COPY`
      gives the same bits. Done: `70e88a5b`, merged in `f4fc865d`. `OP_COPY`, the scoreboard self-test and each
      probe gave the same bits.
   3. **Counters and the host builder** of the graph of dependencies. Host tests: a reduction is right, the
      removal of a hazard fails, and a forward wait panics. Gate: a ring of copy chains and branches gives exact
      results, and the profile shows overlap. Done: `789a5754`, merged in `f4fc865d`. The builder also finds
      write-after-write hazards and takes explicit `after` links. 15 of 16 blocks of a chain left their wait about
      1.2 ms before the slow fill ended.
   4. **Work items and static assignment**, with the grid size as a launch parameter. Gate: the bits are equal at
      G of 48, 24, 7 and 1. Done: `31dfb91f`, merged in `f4fc865d`. Five buffers were bit-identical at each G.
   5. **Merge groups**, tested with `OP_REDUCE_PROBE`. Gate: the bits are equal across G, and equal to the fixed
      tree of the host. Done: `b823f850`, merged in `d5e5d493`. Groups of 11, 16 and 66 items matched the host
      sum at each G. The host still zeroes the merge counters.
   6. **The producer warp, dynamic shared memory and prefetch.** Gates: the read probe holds at least 85% of the
      peak of substep 1. A run of 1,000 dependent no-op tasks gives the cost of a boundary, old chain against new.
      The build shows 0 bytes of spill. Done: `8c0cc25d` and `8816080b`, merged in `d5e5d493`. The read rate is
      244 to 246 GB/s, about 97% of the peak. A boundary costs about 750 ns, from 978 ns, at 114 registers.

   Wave A builds on substep 6.
8. **The host oracle.** The B70 crate has a host form of each op in `f32` (`nagaya-mk-inkling2-bmg/src/host/`).
   Wave B settles the shared host crate for now: no move is needed. The GB10 crate takes `nagaya-mk-inkling2-bmg`
   as a dev-dependency with no feature, and its tests call the public `host::ops` functions. With no `xe`, the build
   of `nagaya-mk-bmg` needs no `ocloc` and no Level Zero. A later move into a crate with no GPU code stays possible.
   Then:
   - Add the quantized expert op, with `quant.rs`, as a host form.
   - Make a reference venv with transformers at `046bdb8cc2`. Add a mode to `tools/inkling2_reference.py` that
     quantizes the inputs of each expert by the rules of `quant.rs`. Its Python must match the fixture of step 6.
   - Write the golden files of both modes. Measure the logit gap between the quantized oracle and the `f32` oracle.
9. **The device ops.** Each op has a card gate against its host form, and the gate prints its GB/s. Do them in
   waves:
   - **Wave A, the matmul core.** Done. Two merges on `main`:
     - `d7390c03`, the dense `bf16` matmul of 1 to 16 rows with the `f32` split of rule 9. The head of 201,024 rows
       at one row reads 235 to 238 GB/s, 93 to 95% of the measured peak. A host model of the MMA rounding gives
       the bits of the card, so a change of the sum order fails a gate.
     - `9a1019d9`, the quantizers and the expert ops. The quantizers give the bytes of the shipped vLLM kernels for
       all 154 rows of the fixture. The NVFP4 expert op equals the host dot on real weights. The MXFP8 op is
       within 7.4e-8 of the sum of the absolute terms. For 8 experts at one row, the ops read at 79 to 105% of the
       measured peak.

     The binding describes both: "The ops of wave A" in `docs/engine-abi.md`.
   - **Wave B, the layer.** The plan is in "The task graph of a layer run" of the binding. It is **Proposed**
     (Claude Opus 5.5, 2026-11-22). The oracle of each lane is the host form of the B70 crate, through the
     dev-dependency of step 8. Do the work in this order:
     1. **The foundation commit**, by the lead. It reserves every opcode in `protocol.rs`, adds empty cases to
        `megakernel.cu`, and puts the task table behind `d`. It adds `weight` to `ExpertPair` and makes the
        merger reset its counter the rule of every group.
     2. **Lane D, the dense ops.** Opcodes 40 to 49. The row tiles and the norm prologue, then `PROJ`, `OUT_MIX`
        and the head. Oracle: `ops::{rms_norm, scaled_l2_clip, log_scale, short_conv, dynamic_conv, l2_clip,
        rms_norm_unweighted, sigmoid, softcap, matvec_bf16}`. Gate: the error is under 4e-7 of the sum of the
        absolute terms. The bits are equal at 1, 7, 16 and 512 rows and at G of 48, 24, 7 and 1. The speed is 90%
        of 252 GB/s or more.
     3. **Lane R, the row ops.** Opcodes 70 to 79: `EMBED`, `MIX`, `GATE_DOWN`, `GATE_UP`, `MIX_OUT` and the row
        finishers. Oracle: `ops::{attention_residual, message_gate, rms_norm}`. Gate: a tolerance, the same bits
        as lane D, and the time against the budget.
     4. **Lane M, the MoE ops.** Opcodes 55 to 69: `ROUTE` and the epilogues of `UP` and `DOWN`. Oracle:
        `ops::{route, swiglu, short_conv}` with `quant.rs`. The lane must not call `Model::experts`, because it
        applies the route weight at another place. Gate: the same picks, unless a gap is under 2e-7, and the bytes
        of the quantized rows equal.
     5. **Lane P, the pass.** No opcode. The buffers, the task tables, the spans and the run ring. It needs a public
        entry to a run of the B70 host form, because `Model::layer`, `kv` and `experts` are private. Gate: each run
        against the host form, and a row of a decode step equals its position in a block.
     6. **Lane A, the attention**, with opcodes 80 to 89, is wave C. Oracle: `ops::{attend, relative_logits,
        relative_bias}`. Gate: a tolerance, and the same bits at 1, 7, 16 and 512 rows.

     Lanes D, R, M and A run in parallel after the foundation. Lane P starts with the run of layer 0, then a
     sliding run, a full run and a run with messages.

     Status of wave B: the ops of lanes D, R and M are done, and lane P is open.
     - The foundation merged as `ef9fb430`. The ops merged with the dense, row and MoE lanes in `6b3066ee`. That
       merge also holds the register structure: one kernel, 168 registers, 0 spill ("The register structure" in the
       binding).
     - `PROJ` writes the head-major cache of the attention (`32589716`, merged in `eb0bf566`).
     - `OP_OUT_MIX` (45) is open. It is not merged. The kernel traps on it.
     - "The ops of wave B" in `docs/engine-abi.md` gives each op with its gates and its speed.
   - **Wave C, the attention.** Lane A. The sliding layers, and the full layers over key chunks of a fixed size on
     all SMs, with a merge in a fixed order. The cache follows "The layout of the cache" of step 11. Build `f32` and the
     first packed format together, so that the gate of step 11 can compare them.

     Status of wave C: done for the attention op (`946ddebc`, merged in `4f01b40f`). It builds `f32` and `e4m3` with
     a unit scale.
     - An `f32` decode step at 65,536 positions reads 204 GB/s, 81% of 252 GB/s.
     - An `e4m3` decode step reads at 49%. One step of 16 keys for each sync limits it.
     - The `e4m3` form gives the bits of the `f32` form on the same packed values.
10. **The pass.** The task ring of a decode step and of a prefill block, at 1 to 5 cycles. Gates: the oracle of
    step 8, and a row of a decode step gives the bits of its position in a block. Print the table of the ops, and
    compare each op with its budget.
    - **Flips of the quantizer.** A sum that changes in its last bit can flip a code of the quantizer. On a tiny
      model, such a flip moved a logit by 0.3 to 8 (`dc574559`). The engine sums in another order than the CPU
      reference. Thus a pass of the engine cannot give the logits of the quantized oracle within a tight bound.
    - Thus each op has a card gate that reads the quantized rows of the oracle, not rows that it quantized itself.
      The reference records them (its tape of quantized rows).
      - The `layers` stage of the quantized mode writes the tape, `inkling2-mini.quant.tape.safetensors`
        (`ebc6f51e`). For each run it holds the attention output, `r`, `h_e`, the route, the MoE output and the run
        output. It holds the expert input and the down input as bytes in the form of `QuantRowsLayout`.
      - The route picks are in ascending expert id. The key `route_tie` marks a tie at the eighth place. The route
        weight acts before the down quantizer.
      - The tape is a realization of its own. Another Python and CPU build moved run 0 by 2e-5 (measured on
        2026-11-22). The flips grew that to 1.9 in the last logits. Thus a gate reads only the tape, and never
        mixes it with the rows of the layers file.
    - The gate of a whole pass is a gate of coherence, as in step 11. It compares the top token, the greedy text
      and the logit gap over the oracle prompts. Write the thresholds into the binding with the first measure.
    - Rule 4 keeps the engine consistent with itself: a decode step gives the bits of its position in a block. Thus
      a flip cannot come from the cache path of the engine.
11. **The engine and the long context.** The `Engine` trait, cycles 1 to 5, snapshots and the conformance suite.
    Slots of 400K to 512K positions, with gates of latency and accuracy.
    - **The format of the cache.** Decided 2026-11-22: a format of fewer bytes is allowed, and it may move bits, if
      the model stays coherent. Each format has an engine identity of its own.
    - The candidates come from the GB10 path of vLLM (`vllm/models/inkling2/nvidia/attention.py`). It keeps `bf16`
      by default, and it allows `float16` and `fp8_e4m3` with a unit scale on `sm_120` and `sm_121`. A unit scale
      can work because the model clips the norm of each query and key, and it normalizes each value.
    - **The layout of the cache.** **Proposed** (Claude Opus 5.5, 2026-11-22). Store the cache packed, and compute
      the attention in `bf16`. Each value expands to `bf16` in registers before the MMA.
      - Each position holds one row of 128 values for each run and KV head. The rows of a head are contiguous, so
        the attention reads them as a stream.
      - A scale covers channels of one row only, never more than one position. Thus the row of a new token is
        complete when its projection writes it. The epilogue of the k and v projection quantizes and writes it.
      - The format is a parameter of the attention op. Only the expansion changes between formats.
      - Reason: a block-scaled MMA needs both of its operands in the format. Then the queries and the softmax
        weights must be quantized, and 8 or 4 bits in the softmax weights hurt the attention most.
      - Reason: a scale over 32 positions has no value until all 32 keys exist. Then the newest block needs a buffer
        and a second write.
      - Reason: a decode step of the attention waits on memory. The MMA can do about 60 times the work that the
        bandwidth feeds. One `cvt` expands two `e4m3` or `e2m1` values to `f16`, and each expanded value is exact
        in `bf16`.
    - **The formats.** The table gives the approximate reads of a decode step at 409,600 positions and 252 GB/s:

      | format | bytes per value | read per step | time | risk |
      |---|---:|---:|---:|---|
      | `f32` (the oracle) | 4 | 26 GB | 104 ms | none |
      | `float16` | 2 | 13 GB | 52 ms | low |
      | `e4m3`, unit scale (vLLM, `sm_120`) | 1 | 6.5 GB | 26 ms | low: the model clips q and k |
      | `e4m3`, `ue8m0` per 32 channels | 1.03 | 6.7 GB | 27 ms | lower: an outlier stays in its block |
      | `e2m1`, `e4m3` per 16 channels | 0.56 | 3.7 GB | 15 ms | high: 4-bit keys move the softmax |
    - **The gate of coherence.** Compare each format with the `f32` cache on the oracle prompts. Compare the logit
      gap, the top token and the greedy text. Then compare them at long positions with a
      retrieval of a fact from early in a context of 400K. A format passes when its greedy text and its retrievals
      match those of `f32`. The thresholds of the gap go into the binding with the first measure.
    - Start with `e4m3` with a unit scale, because it is the format that ships. If it fails the gate, try the scale
      per 32 channels, then `float16`. The `e2m1` format is an experiment for the gate. It comes after the others.
12. **Rows.** A ring of several slots, as LFM3's `model::batch`. The ops take 16 rows from wave A, so this step is
    work of the ring and the scheduler. Measure tokens each second at 1, 2, 4, 8 and 16 rows. The MTP head waits
    for Amy's word (milestone 35 step 10).
13. **Vision and the lab.** The vision tower, the taps and the control vectors.
14. **Deploy and review.** Add `deploy/dgx-spark/` and a `nagaya.service` on sm121host. Run `gogi-conformance`. Run
    kaibo review rounds with a cast of another model family.

### The tuning list

The owner put the tuning after the plan, in later passes. The plan comes first: the pass of step 10, and the engine
of step 11. Each item below is a gap that the card gates of wave B and wave C measured. Each stays open until a
tuning pass takes it.

- **`PROJ` at 188 us, against the budget of 165 us.**
- **`MIX` at 4 us, and the gates at 12 to 13 us.** Latency chains limit them.
- **`ROUTE` at 8 us.** Three serial merges limit it.
- **The tail of the finisher of `DOWN`.**
- **The `e4m3` attention at 49% of the peak.** One step of 16 keys for each sync limits it.
- **The attention prefill at depth.** 512 rows at 8,192 positions take 9.78 ms.
- **The tail of 8% in the head.**
- **The floor of 80% for the NVFP4 `UP` op.** The op runs at 82 to 88%, so the gate can flake.
- **The finisher of `OP_OUT_MIX` at many rows.** One block runs the rows one after another: 151 us at 16 rows and
  4.3 ms at 512 rows (`7a0d74f8`).
- **The block form of the matmul.** `OP_PROJ` takes 3.03 ms at 512 rows, against the design's 0.7 ms (`a59e4adf`).
  - The copies and the MMAs do not overlap well. The causes: one sync for each stage, and the split and the A loads
    in the chain of each warp.
  - A repack by group puts the 8 tiles of a stage together, so that one bulk copy loads a stage.
  - The order of the head's items is not tuned. The head takes 31.7 ms at 512 rows.
- **The speed floor of the attention at 65,536 positions.** It reads 200.4 to 201.3 GB/s against a floor of
  201.6 GB/s, on `main` too. The gate can flake.
- **Fixture gaps.** No row decides the order `(v * s) * outer`. No MXFP8 raw scale lies just above a power of two.
- **Mutants that survived.**
  - The order of the lag 1 and lag 2 terms in the dynamic convolution of `PROJ`.
  - The order of the slices in the merge of `GATE_DOWN`. No host model gives that rounding.

### The lanes

- The lead session plans, reviews each diff, and owns the GPU (rule 10). Subagents write the code: Sonnet where a
  name or a design needs judgment, Haiku for a mechanical class with an exact list.
- Each lane has its own worktree and target directory: `~/src/wt/nagaya-gb10-<lane>`.
- Each brief names the files that the lane must not touch, the checks, and "do not pipe a check". It tells the lane
  to stop at a choice with two meanings.
- The waves of step 9 can run in parallel lanes. One lane writes the ring of the pass and merges the ops.
- The B70 engine lives on b70host. Read its ops and milestone 36 before you write an op here. Its code for the
  Xe cards is not a model for this engine.
