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
- **Other parts.** A ViT of 23 layers for images of 40 to 4,096 px. An MTP head of 3 layers that drafts blocks of
  8 tokens. A tiktoken vocabulary of 201,024 entries. A template with `reasoning_effort` from 0.0 to 0.99.

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
   - Open: the ring of a pass, the upload, `impl Engine` and the gates of the whole engine.
6. **Council and deploy.** The factory, the registries, an example profile, and the gogi suite on the card.
7. **Cycles in a call.** The context member `cycles` chooses 1 to 5 cycles. Card gate: the research model against
   the reference at each count. Mini takes the same counts, out of the range of its training.
8. **The lab.** The taps: the row after each run, the block outputs, the messages, the gates and the routes. Control
   vectors, latent steps, and `ensemble` over counts of cycles.
9. **Images.** The ViT, the placeholders of an image and its sizes. Card gate against the reference.
10. **Drafts.** The MTP head drafts tokens, and a pass checks them. This step waits (question 3).
11. **Long contexts.** The model reads up to 1,048,576 positions. Measure the ledger cost and the speed at 256K.
12. **Trial and review.** A trial with Amy, and a kaibo review round with a cast of another model family.

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
