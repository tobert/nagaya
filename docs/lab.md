# Lab tools

This doc studies tools that look inside a model while it runs, and inside its weights. It asks what the nagaya can
give, and where each tool lives. It is a design note, not a milestone. Each choice here is a proposal until Amy
decides it.

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
| `end_logprob` | `[S]` | the log probability of each end token of the family |
| `norms` | `[S, L]` | the norm of the residual after each layer |
| `along` | `[S, L, V]` | `h · v̂` for each vector, after the control items of that layer |
| `lens_ids`, `lens_logprobs` | `[S, L, K]` | the top entries of the lens at each layer |
| `rows` | `[S, L, W]` | the residual rows themselves, only with `"rows": true` |

The rows are large: 192 KiB for each step of LFM3-1.7B. Thus they are off unless the request asks for them. The
other tensors are a few KiB for each step. The prompt pass gives the row of its last position as step 0.

For a loop, the trace shows which number collapses first. The entropy can fall, the end token can lose its mass, or
`along` can grow at one layer. Compare a plain and a steered trace of the same context.

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

A fit with forward passes only is possible: an `add` of a small step at layer `l` and a tap give one column of `J_l`.
But each column needs its own pass, for each layer and each prompt. For LFM3, that is 24 times the passes of the
backward fit. Thus it is not the plan.

## Where the routes live

**Proposed** (Claude Opus 5.5, 2026-11-07): put `project` and `trace` on nagaya routes under `/nagaya/v1/lab/`
first. Move `trace` into gogi when its shape settles and a client needs it. Keep the routes that read tensors in the
nagaya.

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

1. Does `trace` stream its numbers with the tokens, or only write the object at the end?
2. Does the J-Lens fit run on gfx1151host's GPU with the reference, or on the CPU of b70host?
3. Does the logit lens of a residual row use the final norm with its division? gmk's lens does, with the final norm.
4. Who keeps a trace object: the client with a keep, or the nagaya for a fixed time?
