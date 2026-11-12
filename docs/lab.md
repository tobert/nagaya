# Lab tools

This doc studies tools that look inside a model while it runs, and inside its weights. It asks what the nagaya can
give, and where each tool lives. It is a design note, not a milestone. Amy accepted each Proposed note here on
2026-11-08 (spec, header).

Amy, 2026-11-07, after the first refusal vector worked: "got a couple places where it got stuck in a loop. without
digging in quite yet, how might we analyze that? could we build something into the nagaya for looking at weights to
see what's up?" Then: "do some doc updates first then let's see project and trace".

## What came before

**Weights only.** `~/src/research/weights-only-replication` replays an MIT CSAIL paper (Takehi, Teoh, Zhang and
Khattab, "Sparks of Weight-Space Programming", 2026). There, GPT-6 Astra named the game of four small PPO policies
from their weights. In the replay, Claude Opus 5.5 named all four from the numbers alone, with no tools.

- The networks are small: 17,253 weights for a game, and 1,218 for a physics system.
- Opus used three rules. A row that copies the bias belongs to an input that is always on. A row near its initial
  values belongs to an input that is never on. Two equal rows belong to two inputs that are on together.
- On the physics systems, Opus found the kind of system 4 times in 5, and the constants once in 5. The limit was the
  arithmetic of the chain rule by hand. `physics/jac.py` computes the exact Jacobian in a few lines.

The lesson for the nagaya: a model reads structure well and computes badly by hand. Thus a lab tool computes the
quantities and gives back small results. LFM3-1.7B has 1.7 billion weights, so nobody reads them raw.

**The J-Lens of gmk.** gmk on gfx1151host fitted a Jacobian lens for Gemma 4 12B on 2026-10-11 to 14
(`tools/jlens_fit.py`). It uses Anthropic's `jacobian-lens` code (Gurnee et al., "Verbalizable Representations Form a
Global Workspace in Language Models", 2026), cloned on gfx1151host to `~/src/research/jacobian-lens`.

- The lens of layer `l` is `J_l`, the mean of `d h_final / d h_l` over generic text. `lens_l(h)` is the unembedding
  of `J_l h`, with the final norm.
- The corpus was WikiText-103 paragraphs of more than 700 characters. The cost was one forward pass and
  `ceil(3840 / 8)` backward passes for each prompt.
- The lens of 48 layers at a width of 3,840 is 1.39 GB (`/tank/ml/gmk/jlens/gemma4-12b-it.pt`).
- `jlens_listen`, `jlens_wander` and `jlens_vary` read Gemma's residual through it.

**mkviolet.** mkviolet `docs/04-jlens-and-multi-model-stack.md` designs a tap and a steer op, and a lens on the CPU.
Its `AGENTS.md` names `src/jlens.rs`, but the file does not exist. The port was not built.

## The terms

- The **logit lens** of a residual row is its unembedding, as if the later layers did nothing.
- The **J-Lens** of a residual row at layer `l` is the unembedding of `J_l h`. It corrects the logit lens for the
  later layers. Spec 13.2 calls it a lens.
- A **trace** is a record of one generation: for each step and each layer, numbers that the taps and the lenses give.

**Proposed** (Claude Opus 5.5, 2026-11-07): add "lens" and "trace" to the ontology with these meanings. Reason: the
routes below use both words, and spec 13.2 uses "lens" now with no definition.

## `project`: a vector through a matrix

`project` multiplies one vector by one matrix of the model, and gives the largest and the smallest entries. Its input
needs no context and no pass. It reads the weights only.

```json
POST /nagaya/v1/lab/project
{"model": "lfm3", "vector": "blake3:…", "layers": [12, 23], "through": "unembed", "top_k": 20}
-> {"model": "lfm3", "through": "unembed",
    "layers": [{"layer": 12, "up": [{"id": 62590, "token": " seawater", "score": 1.84}, …], "down": […]}, …]}
```

- `vector` is a control vector object (spec 11.12), or a list of `width` numbers for one layer.
- `through` names the matrix. `unembed` is the logit lens. `jlens` is the J-Lens of the layer, when the served model
  has a lens object. An MLP or an attention matrix can come later.
- With `unembed`, the vector first gets the weight of the final norm, element by element, and no division by its
  root mean square. Reason: a vector is a direction, so it has no scale to normalize.

The first use is the loops. Project the refusal vector through `unembed`, and look at the rank of `<|im_end|>`.

### The rules of `project`

**Proposed** (Claude Opus 5.5, 2026-11-07, milestone 26 step 1). The route follows these rules. Reason: lab.md
gives the shape and leaves these choices open. Each rule is the simplest choice that answers the loop question.

```json
POST /nagaya/v1/lab/project
{"model": "lfm3", "vector": "blake3:…", "through": "unembed"}
-> {"model": "lfm3", "through": "unembed",
    "layers": [{"layer": 23,
                "up": [{"id": 44523, "token": " seafood", "score": 1.245}, …],
                "down": […],
                "end": [{"id": 124900, "token": "<|im_end|>", "score": -0.285, "rank": 122593}]}, …]}
```

- **Members.** `model`, `vector` and `through` are required. `layers` and `top_k` are optional. Another member is a
  400 that names it.
- **Models.** The route serves a model whose engine kind is `mk-lfm3-bmg`: `lfm3` and `lfm3-omni`. Another served
  model is a 400 that names `model`. A model of a peer is a 400 too: the route does not forward.
- **No engine.** The route reads the checkpoint of the model folder of the profile, not the engine. Thus it runs
  while the engine is `unloaded` or `loading`, and it never gives a 503.
- **`through`.** Only `unembed` now. Another value is a 400 that names `through`. `jlens` comes in step 5.
- **An object.** A `blake3:` id names a control vector object. The route checks it as a context checks its control
  list. It checks a held object, the media type, the bytes and the fit with the served model. An object that the
  store does not hold is a 404 that names `vector`. Each other fault is a 400 that names `vector`.
- **Its layers.** With no `layers`, the response has each layer of the vector that is not all zeros, in layer order.
  `layers` names layers of the vector, in the order of the response. A layer that the vector does not hold, a repeat
  or an empty list is a 400 that names `layers`.
- **A list.** `vector` can also be a list of `width` numbers, each finite as a 32-bit float. Another length is a 400
  that names `vector`. Then `layers` is absent or names one layer below the layer count. The response gives that layer,
  or `null` with no `layers`. The layer does not change the scores of `unembed`.
- **`top_k`.** The default is 20, and the limit is 256. A value of 0 or past 256 is a 400 that names `top_k`.
- **Entries.** `up` has the largest scores first, and `down` has the smallest first. A tie goes to the smaller id.
  `token` is the text of the id alone, from the family library. An id past the vocabulary of the family gets `null`.
- **The end tokens.** `end` gives each id of `eos_token_id` in `generation_config.json` of the model folder, with its
  score and its rank. The rank counts from 0 at the top, in the order of `up`.
- **The bits.** Each product of a BF16 weight and an `f32` value is exact in `f64`. The sums run in `f64`, in a fixed
  order. Thus the same request gives the same bytes on each run.

**The cost of `project`.** Proposed (Claude Opus 5.5, 2026-11-07). The route maps `model.safetensors` for each request,
and it holds no copy of the table. The page cache holds the file, and the kernel can take those pages back. Reason:
a copy of the table of LFM3-1.7B is 500 MiB of host memory outside the ledger (spec 4). The work runs on a call
thread and on at most 8 threads of the CPUs of that thread. The scores are 500 KiB for each layer, for the time of
the request.

**The gate of `project`.** The oracle stage `project` of `tools/lfm3_reference.py` puts layers 0, 12 and 23 of the
control vector of the stage `control` through the modules of transformers. Its self check shows that the final norm of
LFM3 multiplies by its weight, not by 1 plus its weight.

The gap of each score is under the error bound of an `f32` sum
of 2,048 products. That bound is 2,048 times half of the `f32` epsilon, times the sum of the sizes of the products. The
worst gap was 0.001 of that bound (2026-11-07). The ids of the top 64 and the bottom 64 match. The test allows a swap
only within the bounds of the two scores.

**The facts of the first run** (2026-11-07, `crates/nagaya-mk-lfm3-bmg/tests/lab.rs`, a build with `--release` on
b70host):

- At layer 23 of the sea vector, " seawater" (id 62590) has the rank 5, with the score 1.116. The top is " seafood".
- At that layer, `<|im_end|>` (id 124900) has the rank 122,593 of 128,000, with the score -0.285.
- The product of 1 layer took 9.7 ms, and the product of 24 layers took 137 ms, with the file in the page cache.

## `trace`: a generation with its numbers

`trace` is a generation (gogi "Generations") that also writes a trace object. The request is a generation request with
one more member.

```json
POST /nagaya/v1/lab/trace
{"model": "lfm3", "context": {"thinking": false, "control": [...], "messages": [...]},
 "max_tokens": 400, "sample": {"temperature": 0},
 "trace": {"layers": "all", "vectors": ["blake3:…"], "top_k": 8, "lens": "logit", "rows": false}}
-> the generation response, and "trace": "blake3:…"
```

The trace object is canonical safetensors (D10). For `S` steps, `L` layers, `V` vectors and `K` top entries, it holds:

| Tensor | Shape | What it is |
|---|---|---|
| `tokens` | `[S]` | the id of each step |
| `entropy` | `[S]` | the entropy of the sampled distribution, in nats |
| `top_ids`, `top_logprobs` | `[S, K]` | the top entries of each step |
| `end_logprob` | `[S, E]` | the log probability of each of the `E` end tokens of the model |
| `norms` | `[S, L]` | the norm of the residual after each layer |
| `along` | `[S, L, V]` | `h · v̂` for each vector, after the control items of that layer |
| `lens_ids`, `lens_logprobs` | `[S, L, K]` | the top entries of the lens at each layer |
| `rows` | `[S, L, W]` | the residual rows themselves, only with `"rows": true` |

The rows are large: 192 KiB for each step of LFM3-1.7B. Thus they are off unless the request asks for them. The
other tensors are a few KiB for each step. The prompt pass gives the row of its last position as step 0.

For a loop, the trace shows which number collapses first. The entropy can fall, the end token can lose its mass, or
`along` can grow at one layer. Compare a plain and a steered trace of the same context.

### The rules of `trace`

**Proposed** (Claude Opus 5.5, 2026-11-07, milestone 26 step 2). The route follows these rules. Reason: the shape above
leaves these choices open. Each rule is the simplest choice that lets a client compare a plain and a steered reply.

- **Members.** The request is a generation request (gogi "Generations") and the member `trace`, which is required. The
  members of `trace` are `layers`, `vectors`, `top_k`, `lens` and `rows`, and each one is optional. Another member of
  `trace` is a 400 that names it, such as `trace.layer`.
- **Models.** The route serves a model of the engine kind `mk-lfm3-bmg` whose engine declares the tap `residual`:
  `lfm3`, and the text passes of `lfm3-omni`. Another served model is a 400 that names `model`. A model of a peer is a
  400 too: the route does not forward.
- **The engine.** A trace is a job of the engine, so a model that is not `ready` is a 503 `model_unavailable`.
- **No stream.** `stream: true` is a 400 that names `stream`. The object is complete only at the end of the reply.
- **The response.** It is the generation response, with each member and value as the same generation with no trace
  gives. One member follows: `trace`, the id of the object. No keep holds the object, so the client keeps it.
- **Steps.** Step `i` is the logit row that gave the token `i` of the reply. Thus step 0 is the last prompt position,
  and `S` is `usage.completion`. The tokens that the job feeds after the reply to hold a head are not steps.
- **The distribution.** `entropy`, `top_logprobs` and `end_logprob` come from the softmax of the logit row over the
  whole vocabulary, in `f64`. That is before `temperature`, `top_k`, `top_p` and `min_p`. It is the distribution of
  `p` and `h` in the token events. Reason: a loop is a fact of the model, and the request names the sampler.
- **`layers`.** `"all"` or a list of layers below the layer count, in the order of the object. A repeat or an empty
  list is a 400 that names `trace.layers`. With no `layers`, the trace has every layer.
- **`top_k`.** The default is 8, and the limit is 64. A value of 0 or past 64 is a 400 that names `trace.top_k`. An
  entry with a larger value comes first, and a tie goes to the smaller id.
- **`vectors`.** Each id names a control vector object. The route checks it as `project` checks it, and a fault is a
  404 or a 400 that names `trace.vectors[i]`. A repeat is a 400. `along` is absent when the list is empty.
- **`along`.** `v̂` is the vector of the layer over its norm, summed in `f64`. `h · v̂` sums in `f64` and gives `f32`. A
  layer that the vector does not hold, or that is all zeros, has no direction, so its `along` is NaN.
- **`norms`.** The Euclidean norm of the row, summed in `f64`.
- **The end tokens.** They are the ids of `eos_token_id` in `generation_config.json`, as in `project`. The tensor
  `end_ids` `[E]` gives them in that order.
- **`lens`.** `"logit"` gives the logit lens of each layer. With no `lens`, the object has no lens tensors. Another
  value is a 400 that names `trace.lens`. `"jlens"` comes in step 5.
- **`rows`.** `true` puts the residual rows into the object. The default is `false`.

**The object.** The tensors are in bytewise order of their names, and the header follows D10 rules 3 and 5. The ids
are `U32`, and the numbers are `F32`. The tensor `layers` `[L]` gives the layer of each place on the axis `L`.
`__metadata__` names the `model`, the `weight_hash`, the `lens`, and the `vectors` in order, with a comma between two
ids. The same request with the same seed gives the same bytes. The object has the media type
`application/x-safetensors`, as a control vector has.

**The logit lens of a row.** This answers open question 3. The lens of a residual row is the final RMSNorm of the
model, with its division and its epsilon, then the head. A row has a scale, so the lens divides by it, as the model
does before the head. The log softmax sums over the whole vocabulary in `f64`.

**Where the lens runs.** The card computes it with the tap `lens` (spec 13.1). After the head of a pass, one matmul
of `L` rows takes the final norm of each tapped row through the head. The host copies the `L` logit rows back. It
takes the log softmax and the top entries on at most 8 threads of the CPUs of the job's thread.

- The card always computes every layer. `layers` chooses what the object holds, not the work on the card.
- The CPU product of `project` takes 9.7 ms for each layer. Thus a CPU lens of 24 layers and 400 steps takes 93 s.
  The matmul on the card reads the table once for all `L` rows.
- The lens of the last layer reads the same row as the head. But its matmul of `L` rows uses another reduction than
  the head's matmul of one row (engine-abi D11). Thus its values can differ from the logits by `f32` rounding.

**The budget.** The job keeps the numbers in host memory, outside the ledger, until the reply ends. The route takes
the size of the object for `max_tokens` steps before the job. When it is past the largest object of the store, or
past 1 GiB, the route gives a 413. It names `trace.rows` when `rows` is `true`, and `max_tokens` when not. Reason:
the rows of `LFM3-1.7B` are 192 KiB for each step.

**The bits.** The taps do not change the generation. The tap `residual` copies each row, and the lens runs after the
head, into its own buffers. Thus a trace gives the tokens and the logit bits of the same generation with no trace.

**The facts of the first trace** (2026-11-07, `LFM3-1.7B` on card 0 of b70host, builds with `--release`):

- `crates/nagaya-mk-lfm3-bmg/tests/xe_trace.rs`: a pass of one token took 9.05 ms with no tap, 9.11 ms with the tap
  `residual`, and 12.69 ms with both taps. These are means of 200 greedy steps.
- `crates/nagaya/tests/xe_lab_trace.rs`: a generation of 200 tokens took 1,875 ms. Its trace took 2,017 ms with no
  lens, and 3,504 ms with the lens of 24 layers. With the lens of 4 layers, it took 3,032 ms.
- The taps changed no logit bit in 32 greedy steps, plain and with an ablate of 1.
- The lens of layer 23 differed from the logits by at most 2.7e-5, and it gave the same top token at each step.
- After an ablate of 1, `along` was at most 7.5e-8 of the norm of the row. Plain `along` at step 0 differed from the
  oracle's rows by at most 4.7e-7 of the row.

## The J-Lens in the nagaya

A lens is a state object of one model (spec 13.2). The nagaya applies it, and a tool outside the process fits it.

- **The fit.** The fit needs backward passes, and the engines run forward only. One backward pass gives a row of
  `J_l` for every layer at once. Thus a fit of LFM3-1.7B copies gmk's `jlens_fit` into a reference tool, on the CPU
  or on gfx1151host's GPU, with no job of the nagaya.
- **The object.** The lens is canonical safetensors with one tensor `layers.<l>` of `[W, W]` for each layer. Its
  metadata names the weight hash and the corpus. For LFM3-1.7B in BF16, that is 24 × 2,048² × 2 bytes, about 201 MB.
- **The use.** `project` with `through: "jlens"` and `trace` with `lens: "jlens"` multiply on the CPU, as spec 13.2
  says. A row of `W_U J_l` is a concept vector of one token at layer `l`. With `add`, it pushes that concept
  (`docs/cv.md`, "With the J-Lens and the other tools").

**The lens of a served model.** Decided 2026-11-08 (Amy, by interview: "Profile names it"). The profile entry of the
served model names its lens, and a request names only `jlens`. Thus a served model has at most one lens.

**The rules of the J-Lens.** **Proposed** (Claude Opus 5.5, 2026-11-08, milestone 26 step 5). Reason: the decision
above and the reference leave these choices open, and each is the simplest one that a test can check.

- **The profile.** The entry gets the table `[engines.lab]` with one key, `jlens`, the path of the lens file. Another
  key, or the table on an engine kind that the lab does not read, stops the start with exit 3.
- **A change.** A profile change to `[engines.lab]` reloads the engine, as a change to `[engines.config]` does
  (3.3). Reason: the lab then never reads a lens that the profile no longer names.
- **The file.** `tools/lfm3_jlens_fit.py` writes it in canonical safetensors (D10). It holds an F16 tensor
  `layers.<l>` of `[W, W]` for each layer before the last. The reference stores F16, because the entries are near 1.
  The lens of the last layer is the identity, so the file does not hold it. For LFM3-1.7B, the file is 193 MB.
- **The metadata.** `model`, `weight_hash`, and `nagaya.lens` with the value `jlens`, as a control vector names
  its weights (spec 11.12). The route refuses a lens of other weights with a 400 that names `through` or
  `trace.lens`. It checks `weight_hash` and `nagaya.lens`; `model` is for a person who reads the file. Reason: an
  alias of a served model has another name and the same weights.
- **The values.** A value that is not finite stops the read, because it would give scores of NaN with no error.
- **The read.** The lab reads the file at its first use and keeps it in host memory, outside the ledger, as it keeps
  the head. A file that is missing or does not read is a 500 that names the path. The start does not read it, so a
  bad lens does not stop the served model.
- **`project`.** With `through: "jlens"`, the route multiplies the vector of layer `l` by `J_l`. Then it follows the
  rule of `unembed`: the weight of the final norm, and no division.
- **`trace`.** With `lens: "jlens"`, the job reads the tap `residual`, not the tap `lens`. For each traced layer, the
  CPU computes `J_l h`, the final norm with its division, and the head. The trace keeps the shape of the logit lens,
  and its metadata names `jlens` and the content id of the lens. The head of LFM3 has 128,000 rows for 125,033 ids.
  Thus the trace keeps the first 125,033 values of each row, as the engine does with its logits. The first live trace
  found it on 2026-11-09.
- **The cost.** The readout runs in the step of the trace, on the group thread. Thus the other jobs of the group
  wait for it at each step. Each step reads the head once for all traced layers. The deploy measures the time.
- A served model with no lens refuses `jlens` with a 400 that names `through` or `trace.lens`.

A fit with forward passes only is possible: an `add` of a small step at layer `l` and a tap give one column of `J_l`.
But each column needs its own pass, for each layer and each prompt. For LFM3, that is 24 times the passes of the
backward fit. Thus it is not the plan.

## `compose`: a control vector from parts

**Proposed** (Claude Opus 5.5, 2026-11-09, milestone 28). Amy wants an agent to compose a control vector set and tune
it with the J-Lens. Reason for the shape: each part is one idea that a projection can show, and the result is a plain
control vector object. Thus every client can use it with no new member.

```json
POST /nagaya/v1/lab/compose
{"model": "lfm3",
 "parts": [{"vector": "blake3:…", "weight": 1.0, "layers": [15, 16, 17, 18, 19, 20, 21, 22]},
           {"words": [" story", " imagine"], "through": "jlens", "weight": 4.0, "layers": [14, 15, 16]},
           {"vector": "blake3:…", "remove": true, "layers": [10, 11]}]}
-> {"model": "lfm3", "vector": "blake3:…", "layers": [14, 15, …], "norms": {"14": 4.0, …},
    "words": [{"text": " story", "ids": [3290]}, …]}
```

- **A `vector` part** adds `weight` times the vector at each of its `layers` that the vector holds. Without
  `layers`, it adds at each layer of the vector.
- **A `words` part** adds `weight` times a unit direction at each of its `layers`, which it names. The direction
  at layer `l` is the backward lens of the sum of the head rows of the ids of the words. For `jlens` it is `J_lᵀ
  (g ⊙ u)`; for `unembed` it is `g ⊙ u`. Here `u` is the sum of the rows, and `g` is the weight of the final norm. The
  family tokenizes each text as it is, so a leading space is part of the word.
- **A part with `remove`** takes the part of the sum along its unit direction out, at each of its `layers`. A
  `remove` part runs after every part that adds, in list order. It has no `weight`.
- **The result** holds each layer where a part added something. The route writes it as a control vector object of the
  model, with its weights, and it gives the norm of each layer. A layer of norm 0 is left out.
- **Errors.** A part with both `vector` and `words`, or with neither, is a 400 that names `parts[i]`. A vector of other
  weights is a 400 that names `parts[i].vector`. `through: "jlens"` on a model with no lens is a 400. A result with no
  layer is a 400 that names `parts`.

**The loop of an agent.** These are the steps of one round:

1. Make a vector from two prompt sets: `POST /council/v2/control-vectors`.
2. Read it through the J-Lens at each layer: `POST /nagaya/v1/lab/project` with `through: "jlens"`. Note the layers
   where it reads as the idea you want, and the words that you do not want.
3. Compose: keep those layers, take out a direction of the words that you do not want, or add a direction toward words.
4. Try it: a generation with a control item, with `layers` for the strength of each layer (spec 11.12).
5. Trace one reply with `lens: "jlens"`, and count the replies that you want over a prompt set. Go to step 2.

### Latent steps in a request

**Proposed** (Claude Opus 5.5, 2026-11-10, milestone 29 step 4). Amy, 2026-11-10: "I feel like it might be most
effective on the summarizer. but probably everywhere we can over time, and somewhat tunable per api call ideally."
Thus a request of `ensemble` takes the member `latent`. One context with `logprob_sum` is a generation, so any
generation of the lab can think in latent steps first.

```json
"latent": {"steps": 16, "read": true}
-> "latent": {"steps": 16, "read": [{"id": 3290, "text": " story"}, …]}
```

- **Where.** Each context takes its steps after its last token, before the first token of the reply. A context with
  a prefill takes them after the prefill.
- **A step.** The pass of the last input gives the residual after the last layer, through the residual tap (spec
  13.1). The host applies the final norm, and it scales the row to the mean norm of the embedding rows. The next pass
  feeds that row at the next position, with a latent span (engine-abi "Latent rows of LFM3"). The logit row of the last
  step gives the first token of the reply.
- **`steps`.** 0 to 256. With 0, or with no `latent`, the request gives the bits of the same request with no member.
- **`read`.** With `true`, the response gives the top token of the logit row of each step of context 0. The token
  does not go into the context.
- **Joins.** A pass with a tap has one segment (engine-abi "Batched passes"). Thus the steps of each context run alone,
  in the build turn of that context. The decode of the reply joins as before.
- **Errors.** `steps` past 256, or not a whole number, is a 400 that names `latent.steps`. A model with no tap
  `residual` is a 400 that names `latent`.

**Proposed** by Claude Opus 5.5, 2026-11-10, for milestone 29 step 4: the choices that the rules above leave open.

- **No head.** A reply after latent steps holds no head, and its slot makes no interval copy after the first step.
  Reason: a recipe names tokens, and a latent row is not a token.
- **The response.** `latent` echoes `steps`. It has `read` only when the request reads, and `read` is empty for 0
  steps. The `text` of a read token is its piece.
- **The seed.** The sampler takes the position in the slot, so the first token of the reply is at the prompt length
  plus the step count.
- **The norm.** The host takes the mean of the squares in `f64`, then each value in `f32` takes the weight. The scale
  is the mean of the row norms of the BF16 table, in `f64`. On the CPU it is 0.90660375 for `LFM3-1.7B`; the
  oracle gives 0.9066035.
- **Members.** `latent` has only `steps` and `read`. Another member is a 400 that names `latent.<member>`, and
  `read` that is not a boolean names `latent.read`.

### Soft inputs in a request

**Proposed** (Claude Opus 5.5, 2026-11-12, milestone 33 step 2). Amy, 2026-11-12: "I'm curious if there is a middle
ground where we get some of the compute cost and precision (not round tripping through English, or does that help?)
benefits of latent steps, while making them more productive."

Three papers measured this. Each is training-free, and each feeds a mix of embedding rows in place of one token:

- **Mixture of Inputs.** Yufan Zhuang et al., *Text Generation Beyond Discrete Token Sampling*, arXiv 2505.14827,
  NeurIPS 2025. The sampler picks a token as before, and the next input mixes it with the distribution. The weight of
  the mix grows with the entropy. It gave 1.8 points more accuracy on average, over 16 pairs of model and task.
- **Soft Thinking.** Zhen Zhang et al., *Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous
  Concept Space*, arXiv 2505.15778, 2025. The input of each step of the reasoning is the mix of the top 10 to 15
  rows. With it, replies were 12% to 22% shorter, and accuracy rose by 0.5 to 2.5 points. Its own ablation fed the last
  hidden state back, as our latent steps do. That scored 0.0 on AIME 2024 and LiveCodeBench, and each reply ran to
  the limit.
- **Stochastic Soft Thinking.** Junhong Wu et al., *LLMs are Single-threaded Reasoners: Demystifying the Working
  Mechanism of Soft Thinking*, arXiv 2508.03440, ICLR 2026. A plain mix follows its top token, and it scored below
  sampled text. Gumbel noise in the mix, with `tau` 0.5, scored above sampled text on each of the three models.

All three used reasoning models of 27B parameters and more, on math, code and questions. Nobody measured a small model
or creative text. LFM3 is both, so these trials are new.

```json
"soft": {"map": "moi", "beta": 1.0, "top": 16, "span": "all"}
"soft": {"map": "gumbel", "tau": 0.5, "top": 16, "span": "think"}
```

- **Where.** `soft` is a member of `ensemble`, as `latent` is. It changes the input row after each sampled token. The
  token goes into the reply and the record as before.
- **The distribution.** `p` is the draw probabilities of the survivors of the sampler (spec 11.2). The mix uses the
  `top` most probable survivors, scaled to a sum of 1. `top` is 1 to 64, and 16 when it is absent.
- **`moi`.** The sampler picks the token `t` as before. Let `H` be the entropy of `p` over all survivors, divided by
  the log of the vocabulary size. The weight of each row is `(1 - a) onehot(t) + a p`, with `a = H / (beta + 1)`.
  `beta` is finite and 0 or more, and 1 when it is absent. The paper calls this the posterior mean of a Dirichlet
  prior.
- **`gumbel`.** For each survivor `i` of the top, `g_i = -ln(-ln u_i)`, with `u_i` from the seed, the position and
  `i`. The weights are `softmax((ln p_i + g_i) / tau)`. The token is the survivor of the largest weight. It is a draw
  from `p`, by the Gumbel-max trick. `tau` is finite and above 0, and 0.5 when it is absent.
- **The input row.** It is the sum of the weights times the rows of the embedding table, in `f64`, then `f32`. The
  pass feeds it at the next position, as a latent row (engine-abi "Latent rows of LFM3").
- **One row is a token.** When the weight of one row is 1, the pass feeds that token, not a latent row. Thus `moi`
  at temperature 0 gives the bits of the same request with no `soft`. The host test checks this.
- **`span`.** With `all`, each token of the reply takes a soft input. With `think`, only a token of the reasoning
  does, and the token that ends the reasoning feeds as a token. `all` is the default.
- **No head.** A reply with soft inputs holds no head, as a reply after latent steps does. The next turn by `parent`
  feeds the tokens of the reply as text.
- **One context.** `soft` takes one context with the weight 1, and the ensemble runs it as a generation. Another count
  is a 400 that names `soft`.
- **Joins.** A soft row is a latent row, and a latent row shares no pass (engine-abi "Batched passes"). Thus the first
  build feeds each soft row in a pass alone. A join of soft rows is later work, if the trials show a gain.
- **The response.** `soft` echoes the request with each default filled in. It also gives `mixed`, the count of soft
  rows, and `mean_share`. The share of a step is 1 minus the weight of its token. `mean_share` is the mean over the
  steps of the span, and a token that feeds as a token has a share of 0.
- **Errors.** An unknown `map` is a 400 that names `soft.map`. A member of the other map, or an unknown member, is a
  400 that names `soft.<member>`. A value out of range is a 400 that names its member.

**Proposed**, later: a stop for steps that do not end by themselves. Amy, 2026-11-12: "I think detecting </think> or
other end tokens would be helpful so the model isn't trapped in a loop it has no use for."

- **`latent.stop`.** With `true`, the latent steps end at the first step whose top token ends the reasoning or the
  turn. The reply starts after that step. The response gives `latent.ran`, the steps that ran.
- **`soft.cold`.** This is "Cold Stop" of Soft Thinking: `{"entropy": 0.05, "run": 128}`. After `run` tokens in a row
  with an entropy below `entropy`, the reply takes the token that ends the reasoning. It needs `span: "think"`.

The trials of these rules are in `docs/cv.md`, "The stream of thought". The paper files are not in git.

## `latent` and `merge`: the parts of a swarm

**Proposed** (Claude Opus 5.5, 2026-11-09, milestone 29 step 2). Amy chose a text swarm first, then latent steps and
the merge as a pair (`docs/swarm-ideas.md`). Reason for the shape: each route makes a held snapshot, so a generation,
a decision or another route continues from it with no new member.

```json
POST /nagaya/v1/lab/latent
{"model": "lfm3", "from": "blake3:…", "steps": 40, "read": true}
-> {"model": "lfm3", "snapshot": "blake3:…", "steps": 40, "mean": "blake3:…",
    "read": [{"id": 3290, "text": " story"}, …]}

POST /nagaya/v1/lab/merge
{"model": "lfm3", "parts": [{"from": "blake3:…", "start": 0, "end": 410},
                            {"from": "blake3:…", "start": 380, "end": 420}, …]}
-> {"model": "lfm3", "snapshot": "blake3:…", "positions": 1610}
```

- **A latent step** feeds the last hidden state as the input row of the next position. The hidden state is the input
  of the head: the residual after the last layer, through the final norm. The engine scales it to the mean norm of the
  embedding rows. LFM3's embeddings are tied, so this is the map of LatentMAS with a ridge weight near 0.
- **`from`** is a held snapshot of the model. `steps` is 1 to 256. The result is a new held snapshot, and its recipe
  names `from`, the step count and the map (spec 6.12).
- **`mean`** is an object of `width` f32 values: the mean hidden state of the steps. A novelty filter reads it.
- **`read: true`** also gives the top token of the logit row of each step. The token does not go into the context.
  It is a view of the agent, as a trace is.
- **A merge part** names a held snapshot and a range of its positions. The merge puts the ranges in list order, with
  no gap. The key of each row at each full attention layer turns from its old position to its new one. A value does
  not change. A sliding layer keeps the last 256 positions, and the convolution state comes from the last part.
- **Bits.** A merge is not the bits of a pass over the same tokens, because two rotations round apart from one. Thus
  its recipe names each part and the order, not the tokens.
- **Errors.** A snapshot of another model or other weights is a 400 that names `from` or `parts[i].from`. A range
  outside its snapshot is a 400 that names `parts[i]`.

## `ensemble`: joined forks

**Proposed** (Claude Opus 5.5, 2026-11-10, milestone 29). Amy chose plan, then code (`docs/swarm-ideas.md`, "Plan, then
code"). Joined forks are one way to code a plan, or to distill ideas into one plan. Reason for a lab route: the rules of
the combination are a trial, and a council v2 contract can follow when one rule wins.

```json
POST /nagaya/v1/lab/ensemble
{"model": "lfm3", "contexts": [{"thinking": false, "messages": [...]}, …],
 "combine": "logprob_sum", "weights": [1.0, 1.0], "strength": 0.5,
 "max_tokens": 96, "sample": {"temperature": 1.0, "seed": 7}}
-> {"model": "lfm3", "content": "…", "finish": "stop", "usage": {"prompt": 812, "fed": 140, "completion": 41}}
```

- **The job.** One job holds each context in its own slot. Each step is one batched pass, with one row of each
  context. The job combines the logit rows into one row, and the sampler of spec 11.2 picks one token from it. Each
  context takes that token at the next step.
- **`contexts`.** 1 to 8 council v2 contexts of the model, with the same control list. A prefill with `continue: true`
  is allowed. The store shares their prefixes, as it does for generations.
- **`combine`.** Let `l_k` be the log softmax of row `k`, in `f64`, over the whole vocabulary. Then:
  - `logprob_sum`: the row is `sum_k w_k l_k`. A token needs the support of every context.
  - `prob_mean`: the row is `log(sum_k w_k exp(l_k))`, with the weights scaled to a sum of 1. One context can carry a
    token.
  - `contrast`: exactly 2 contexts, and the row is `(1 + s) l_0 - s l_1`, with `s` the `strength`. Context 0 has the
    idea, and context 1 does not. This is context-aware decoding (Shi et al., 2023). With no `strength`, `s` is 0.5.
- **`weights`.** One finite value above 0 for each context. With no `weights`, each weight is 1. A vote of a decision
  can give them.
- **The sampler.** The combined row goes to `f32`, and the sampler of spec 11.2 reads it with `sample`. The seed of a
  step is the position of the reply in context 0, as for a generation.
- **The reply.** The parser of context 0 reads the tokens, and its end ends the reply. `max_tokens` ends it too, and so
  does a full context. `content` is the reply after the prefill of context 0. There is no stream, and no held head.
  `finish` and `usage` are those of context 0.
- **Bits.** One context with `logprob_sum` and the weight 1 gives its row to the sampler unchanged. Thus it gives the
  bits of the same generation. The card gate checks this.
- **Errors.** A model with no batched passes is a 400 that names `model`. 0 or more than 8 contexts is a 400 that names
  `contexts`. A context with another control list is a 400 that names `contexts[i].control`. `contrast` with other than
  2 contexts is a 400 that names `combine`. A weight that is not finite or not above 0 is a 400 that names
  `weights[i]`. A `strength` that is not finite is a 400 that names `strength`.

## Context records in the lab routes

**Proposed** (Claude Fable 5.2, 2026-11-11, milestone 32 step 2). `trace` and `ensemble` read council v2 contexts, so
they take a context that names a `parent` (gogi "Context records", spec 11.13). Reason: a client of a swarm names the
record of its shared head in each route that reads it. @tobert decides.

- **Whole before the decode.** Each route makes each context whole before it decodes the body, as a generation does.
  A parent that the node does not hold is gogi's 404. It names `context.parent` for `trace`, and
  `contexts[i].parent` for `ensemble`.
- **An open parent.** A context that is its parent alone continues the open message of the parent, as in a
  generation.
- **`trace` holds what a generation holds.** Its response is the generation response, so `context` names the record
  with the reply.
- **`ensemble` holds the record of each context when it succeeds.** It names no record. Reason: the reply comes from
  the combined row of all the contexts, so it is the reply of no single context. A client that wants a record builds
  the context with the reply after it.
- `project` and `compose` read no context, so they do not change.

## Where the routes live

Decided 2026-11-07 (Amy: "ok yeah, start on /nagaya/v1/lab/"): put `project` and `trace` on nagaya routes under
`/nagaya/v1/lab/` first. Move `trace` into gogi when its shape settles and a client needs it. Keep the routes that read
tensors in the nagaya.

- `project` and `trace` use ideas that each language model has: layers, a residual width, an unembedding and tokens.
  gogi already has layers and widths in its control vectors, so they can go there later.
- A tool that is still changing costs less in the nagaya. A gogi route needs a contract, the fake and suite checks
  before its first use.
- A route that names tensors exposes the engine's own layout. That is not part of a council contract.
- The requests use gogi's types now: a context, a control list, object ids and a generation request. Thus a move to
  gogi changes the path, and little else.

## Later: the weights

These read the weights only, and give small results, as the replay asks. They wait until `project` and `trace` show
what is needed.

- `GET /nagaya/v1/lab/tensors`: the name, the shape, the type and the norms of each tensor.
- `POST /nagaya/v1/lab/rows`: the rows nearest to a row, the rows near the spread of their initial values, and the
  pairs of rows that are almost equal.
- `POST /nagaya/v1/lab/jacobian`: the Jacobian of the output at one point, as `jac.py` gives for a small network.

## Tests

- `project` with `unembed` of a vector of one layer equals the product of the reference, to the tolerance of the
  logit gate. Break it: leave out the weight of the final norm.
- A trace of a plain context gives the tokens and the logits of the same generation with no trace, bit for bit. Break
  it: run the taps before the control items.
- `along` with an `ablate` of 1 is 0 at each layer that the vector holds. Break it: read the row before the items.
- A request with `"rows": true` past the budget of the store gets a 413 before the job. Break it: count the rows as 0.

## Open questions

1. Answered 2026-11-08: `trace` writes the object at the end, and the stream carries the tokens only. The note of
   milestone 26 step 2 says so, and Amy accepted it.
2. Answered 2026-11-08 (Amy, by interview): the J-Lens fit of LFM3-1.7B runs on the CPU of b70host, at `nice` 19
   and with no GPU job. The cards stay free for the served models. Measure the time on a small corpus first.
3. Answered 2026-11-08: yes ("The logit lens of a row"), accepted by Amy.
4. Answered 2026-11-08 (Amy, by interview): the client keeps a trace object with a keep. The route sets no keep, and
   the store evicts an object with no keep by its usual rule (6.15). The web app keeps each trace under its owner.
