# Control vectors

This doc studies one tool: a control vector that changes the residual of a model while it runs. It asks if the
nagaya can carry the tool, and what the tool costs. It is a design note, not a milestone. No code exists for it.
Each choice here is a proposal until Amy decides it.

Amy, 2026-11-06: "I'm mostly interested in the technique's viability. could we write a docs/cv.md about adding
control vector support, which I'll come back to later. I want to think about it alongside the delayed j-lens feature
and other model debugging/tweaking capabilities we want to have options for."

## Decisions of 2026-11-07

Amy decided these rules by interview, for milestone 25 (`AGENTS.md`). Amy: "I'd like to try this runtime technique
and figure out how to work it into the protocol as an option". Spec 11.12 and engine-abi "Control vectors" hold the
rules.

- A council v2 context names its control vectors, beside `thinking`. This replaces the note on choice 1 below.
- One op gives `add`, `ablate` and `clamp`.
- The nagaya makes a vector on the card, with a residual tap and a job. The route is council v2, in gogi.
- The first prompt sets are public. Amy, 2026-11-07: "we'll use some public data at first so we can work without
  seeing that data".
- `LFM3-1.7B` comes first, and Omni follows. Amy sets each strength by hand.

The open questions below get these answers, each Proposed by Claude Opus 5.5 on 2026-11-07:

1. The op runs on the residual, before the norm of the next layer.
2. The op runs on the prompt pass and on each decode step.
3. The op does not touch the output of the embedding. It runs only after a layer.
4. A turn with another list is another context, with another root recipe.
5. The search for strengths waits. Amy sets them by hand.

## The term

A **control vector** is a set of vectors, one for each layer that it touches, with an operation and a strength for
each. It is a state object (ontology "State object"): it has a content id and belongs to one model. Spec 6.12 and 13.1
call the same thing a "steering vector". llama.cpp calls it a control vector, and so does this doc.

**Proposed** (Claude Opus 5.5, 2026-11-06): use the term "control vector" in the ontology, spec 6.12 and spec 13.1.
Reason: it names the file and its strengths together, and llama.cpp uses that name. Amy decides.

## What it does

The residual is the stream of rows that each layer reads and adds to. A control vector changes the residual after
layer `l`, before layer `l + 1` reads it. Let `h` be one row of the residual, `v` the vector of the layer, `v̂` its
unit vector, and `α` the strength. mkviolet `docs/06-council-jlens-extensions.md` names three operations:

| Operation | Change to the row | Use |
|---|---|---|
| `add` | `h + α·v` | push the model toward a concept |
| `ablate` | `h − α·(h·v̂)·v̂` | remove a concept; with `α` below 1, only weaken it |
| `clamp` | `h + (α − h·v̂)·v̂` | set the size of a concept to `α` |

The known use is the refusal direction of Arditi et al. (2024, "Refusal in language models is mediated by a single
direction"). The difference of the mean residuals of two prompt sets gives one direction. One set gets refusals and
the other set gets answers. An `ablate` of that direction at each layer removes most refusals.

## Weights or runtime

The same change can go into the weights. With a fixed `α`, an `ablate` at each layer gives the same result as a
projection of each matrix that writes to the residual. Abliteration and Heretic ship such weights. Heretic also
searches for the strength at each layer that removes the most refusals with the smallest change of the outputs.

The nagaya can already serve such weights with no new code, as other weights of the same engine. A control vector
at runtime gives three things that other weights do not:

- one copy of the weights for each strength, with no new load;
- `α = 0` as the plain model, for a test in the same process;
- operations that no weight projection gives, such as `add` and `clamp`.

## Why the nagaya can carry it

These parts exist now:

- **The hook point.** A stage boundary is a residual at a layer (engine-abi D1). The Gemma 4 engine sends the
  residual after layer 14 through host memory (engine-abi "How mkviolet maps", `stages`).
- **The recipe slot.** The root recipe has an item for lab state, tag 4 (spec 6.2). The code hashes it in
  `recipe_root` (`crates/nagaya-store/src/lib.rs:457-475`). No caller gives a lab state yet.
- **The rule.** Spec 6.12 says that a steered result never has a plain recipe. Spec 13.1 says that a steering
  vector goes into the recipe.
- **The shared interpreter.** The engines on the B70 cards run ops from `nagaya-mk-bmg`. One new op there serves
  each engine on those cards.

## What it costs

- **Compute.** An `ablate` is one dot product and one multiply-add for each row at each layer. For LFM3-1.7B the row
  is 2,048 values (the shape of `embed_tokens` in its safetensors header). That is about 8,192 floating-point
  operations for each token at each layer. The matrices of the same layer need millions.
- **Memory.** A vector for each of 24 layers at 2,048 F32 values is 196,608 bytes. The ledger charges it as an engine
  term with lifetime `engine`.
- **Cache reuse.** A control vector changes every row after its first layer, so it changes the keys and values. A
  snapshot with one control vector and strength never serves a request with another. This is the largest cost.
- **Bits.** The op must give the same bits each run (engine-abi D3). The dot product sums in a fixed order in F32.

## Where it is chosen

There are two places to choose a control vector. They have different costs.

1. **In the served model.** The profile names the weights, the control vector and the strengths. Thus
   `lfm3-omni` and `lfm3-omni-open` are two served models on the same weights. Each has its own root recipe, and
   each keeps its own prefix cache. The engine identity does not change, so the weights load once.
2. **In the request.** The request names a control vector by content id, and the strengths. Each request with new
   strengths starts a new root recipe, with no prefix reuse.

**Proposed** (Claude Opus 5.5, 2026-11-06): build choice 1 first. Leave choice 2 as a design. Reason: choice 1 needs
no change to the council v2 contract. Its cache rule is the rule of a served model today. Amy decides.

A request route needs a contract in gogi first (spec 11.1). mkviolet `docs/06-council-jlens-extensions.md` drafted
`steer` for council v1, with a rule that a steered decision says `"steered": true` in its identity. That draft is a
start for the gogi text.

## How a control vector is made

A control vector comes from a job, not from code. The job reads residuals through a tap (spec 13.1), on the same
engine that serves the model. These are the steps for a refusal direction:

1. Run two sets of prompts, and keep the residual of the last prompt token at each layer.
2. At each layer, subtract the mean of one set from the mean of the other.
3. Pick the layers and the strengths, then measure refusals and the change of the outputs.
4. Write the vectors as canonical safetensors (D10), with one tensor for each layer. The store gives the content id.

Step 3 is the search that Heretic does. The first version can copy its rule. llama.cpp writes control vectors as
GGUF files. An import of that format can come later, but a vector only fits the model that made it.

A model with images, such as LFM3-Omni-2B, may refuse on an image and not on its text. A direction from text prompts
may not remove that refusal. Thus the prompt sets for such a model hold images too.

## With the J-Lens and the other tools

The J-Lens waits for a later milestone (spec 13.2). The J-Lens and control vectors use the same point in the pass:
the residual after a layer. The J-Lens reads it, and a control vector writes it. mkviolet's design puts both into
one op (`OP_JLENS_TAP` and `OP_JLENS_STEER`, mkviolet `docs/04-jlens-and-multi-model-stack.md`).

| Tool | Reads or writes | Changes bits | Where it lives |
|---|---|---|---|
| residual tap | reads the residual at layer `l` | no | engine, `limits().taps` (spec 13.1) |
| route tap, gate tap | reads the expert picks or gates | no | engine (spec 13.1) |
| J-Lens | reads a tap, then maps it through the Jacobian | no | a lens on a CPU device (spec 13.2) |
| control vector | writes the residual at layer `l` | yes | engine op, and a state object in the store |
| steer in the J-Lens basis | writes `α·Jᵀ·W_Uᵀ·e` for a token `e` | yes | the same op, with a vector from the lens |
| logit bias | writes the logits | yes | the sampler (spec 11) |

The last two rows show how the tools meet. A J-Lens finds a concept at a layer. Its Jacobian then gives the
vector that pushes that concept, which a control vector applies. Thus a lens and a control vector share the format of
a vector for each layer.

**Proposed** (Claude Opus 5.5, 2026-11-06): design one engine hook, "the residual after layer `l`", for both taps and
control vectors. An engine declares the layers where it supports the hook. Reason: one op, one test and one
declaration serve both tools. Amy decides.

## Tests

These tests show that the tool works. Each one can fail.

- With `α = 0`, an `add` or an `ablate` gives the bits of no control vector. Break it: add a fixed offset in the op.
  A `clamp` with `α = 0` removes the concept, so it is not in this test.
- An `ablate` with `α = 1` gives a residual row with a dot product of zero against `v̂`, to F32 rounding. Break it:
  skip the normalization of `v`.
- A served model with a control vector never takes a snapshot of the plain served model. Break it: leave the lab
  state out of `recipe_root`.
- On the reference in Python, the same control vector gives the same logits as the engine, to the gate of the oracle.

## Open questions

1. Does the op run before or after the norm of the next layer? Arditi et al. change the residual, so the first
   answer is "before the norm".
2. Does a control vector act on the prompt pass, on decode, or on both? llama.cpp acts on both.
3. Does `ablate` touch the embedding output and the output of the last layer too? Weight abliteration touches the
   embedding matrix.
4. Can a request turn a control vector off for one turn, and how does the prefix cache treat that turn?
5. Where does the refusal search run: in the nagaya as a job, or in a tool outside it that uses the routes?

## Verdict

The tool fits the nagaya. The hook point, the recipe slot and the rule exist. The op is small, and its compute cost
is less than one percent of a layer. The real cost is the cache, and a control vector for each served model keeps
that cost the same as today. Build it after the first LFM3 engine, together with the residual tap that the J-Lens
needs.
