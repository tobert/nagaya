# Control vectors

This doc studies one tool: a control vector that changes the residual of a model while it runs. It asks if the
nagaya can carry the tool, and what the tool costs. It is a design note, not a milestone. No code exists for it.
Amy accepted each Proposed note here on 2026-11-08 (spec, header).

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
  residual after layer 14 through host memory (engine-abi "How the Gemma 4 engine maps", `stages`).
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

The J-Lens waits for a later milestone (spec 13.2), and `docs/lab.md` designs it. The J-Lens and control vectors use
the same point in the pass:
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

## The stream of thought: the trials of 2026-11-11

Milestone 33 step 2. With thinking off, a second turn of `lfm3` can write a first draft, then a second `</think>`,
then its answer. The reference model in transformers does the same, so the cause is the model. Amy, 2026-11-11:
"think of it (hah!) more as being able to dial up the stream of consciousness nature of non-thinking calls to the
model". Each count below is of 24 replies at temperature 1.0 on b70host, unless the row says another value.

**The trigger.** An old reply with an empty think block in the history. A first turn gave a draft in 0 of 72
replies. A second turn gave 15 (a color), 6 (a story) and 23 (arithmetic). An old reply with a short reasoning gave
1, and thinking on gave 0.

**Temperature.** For arithmetic the draft is the most probable reply: 24, 24, 24, 23 and 20 at 0.1, 0.3, 0.6, 1.0
and 1.3. For a story it comes with temperature: 0, 0, 2, 6 and 3.

**The vector.** Toward: 12 second turns with an old reply that has no reasoning. Away: the same 12 with a short
reasoning in the old reply. Its id is `blake3:face3f09c6e79b1be15658d93e8ac53db16928edf9a2a9adeda8472bcbbf0378`.
The table gives the drafts with `add` at each strength, and the right answers of the arithmetic.

| Strength | Story, turn 2 | Arithmetic, turn 2 | Right, turn 2 | Story, turn 1 | Arithmetic, turn 1 |
|---|---|---|---|---|---|
| -0.2 | 0 | 0 | 19 | 0 | 2 |
| -0.1 | 0 | 7 | 21 | 0 | 0 |
| -0.05 | 1 | 19 | 22 | 0 | 1 |
| 0 | 6 | 23 | 24 | 1 | 1 |
| +0.05 | 8 | 24 | 24 | 0 | 1 |
| +0.1 | 24 | 24 | 22 | 2 | 1 |
| +0.2 | 19 | 14 | 8 | 9 | 0 |

- The vector is a dial on a second turn, from -0.2 to +0.1. On a first turn it adds few drafts.
- At +0.2 the replies start to break. At 0.4 and above, with either sign, each reply is noise. A strength of 1, 2
  or 4 gave noise in each of 576 replies.
- `ablate` at strength 1 is weaker: 23 to 16 for the arithmetic and 6 to 1 for the story on turn 2. It raised the
  drafts of the arithmetic on turn 1 from 1 to 10. A small strength below 0 with `add` is the better way down.

**Latent steps** (`docs/lab.md`, "Latent steps in a request"), with thinking off:

- One step is enough to change how a reply starts. With one step, 22 to 24 story replies started in the middle of
  a sentence. With no step, 0 did. The first steps hold the first words of the reply, and the reply then starts
  after them.
- With 16 to 64 steps it is the same. With 128 and 256 steps it goes away (7 of 24). The arithmetic stays right at
  each count, 23 or 24 of 24, and the replies get longer.
- 20 arithmetic problems with 6 replies each gave 120, 120, 119 and 118 right of 120 at 0, 1, 2 and 4 steps. Thus
  the steps do not make this arithmetic better or worse. The task is too easy to show a difference.

**Latent steps with thinking on.** The steps come after `<think>`, and the model then writes its thinking.

- The steps do not take the place of the written thinking. For `17 times 23` its median stayed near 95 tokens at
  0, 1 and 2 steps. It grew to 165 to 177 tokens at 16 steps and more.
- Thinking needs room. With 160 tokens, 18 of 24 replies had the right answer. With 600 tokens, 24 of 24 had it. The
  shortest thinking was 50 to 75 tokens for `17 times 23`, and 140 to 170 tokens for a puzzle of two equations.
- For a story, 1 or 2 steps cut the median thinking from 439 tokens to 209 to 271. Then 23 of 24 replies reached
  their answer, where 17 did with no step.

Not tried: latent steps, then a `</think>` that the request writes. The route puts the steps last, so it cannot
feed a token after them. The raw replies of these trials are in `target/trials-2026-11-11/` on b70host, which
git does not hold.

**Soft inputs** (`docs/lab.md`, "Soft inputs in a request"), 2026-11-12, the build of `9367f8c`. Each row is 24
replies of `lfm3` at temperature 1.0, with the seeds 0 to 23. Thus the rows of one prompt are paired by seed. The
share is the mean weight of the rows other than the token (`mean_share`).

With thinking on and `span: "think"`, for 600 tokens at most:

| Row | 17 times 23: thinking | Share | Rabbits: thinking | Share |
|---|---|---|---|---|
| no `soft` | 86 | 0 | 212.5 | 0 |
| `moi`, `beta` 1 | 86 | 0.0044 | 213 | 0.0018 |
| `moi`, `beta` 0.25 | 91 | 0.0071 | 214.5 | 0.0030 |
| `moi`, `beta` 4 | 88.5 | 0.0018 | 212.5 | 0.0008 |
| `gumbel`, `tau` 0.5 | 92.5 | 0.0473 | 227 | 0.0225 |

- Each of the 10 rows had 24 of 24 right answers. "Rabbits" asks for the rabbits of 35 heads and 94 legs.
- The median length of the thinking did not change. As with latent steps, these tasks are too easy to show a gain.
- `moi` mixes little here: thinking about easy math has a low entropy, and `moi` divides the entropy by the log of
  the vocabulary size.

With thinking off and `span: "all"`, for 200 tokens at most:

| Row | Owl story: share | Distinct pairs | Ended | Sea poem: share | Distinct pairs | Ended |
|---|---|---|---|---|---|---|
| no `soft` | 0 | 0.770 | 23 | 0 | 0.840 | 24 |
| `moi`, `beta` 1 | 0.053 | 0.786 | 23 | 0.071 | 0.828 | 24 |
| `moi`, `beta` 0.25 | 0.083 | 0.760 | 24 | 0.114 | 0.842 | 24 |
| `gumbel`, `tau` 0.5 | 0.193 | 0.760 | 21 | 0.198 | 0.783 | 24 |
| `gumbel`, `tau` 0.9 | 0.349 | 0.776 | 24 | 0.372 | 0.834 | 24 |

- "Distinct pairs" is the count of distinct word pairs over all pairs of the 24 replies. "Ended" counts the replies
  that ended before the limit.
- The counts do not show the main result. At a share of 0.2 and more, words break inside the reply: for example
  "scrainkets", "whereumbling" and "peruck". Two replies of each `gumbel` row show it, at `tau` 0.5 and more at 0.9.
  Two replies of `moi` with `beta` 0.25, a share near 0.1, read as plain text.
- The cause: the reply holds the token, but the next pass reads the mix. Then the model continues the mix, not
  the token, and the text of the reply leaves what the model read. Soft Thinking and Stochastic Soft Thinking mix
  only the reasoning, and they decode the answer as tokens. These trials show why.
- A count of words that are not one token of the tokenizer did not find the breaks. Plain replies gave 14% to 15%
  of such words, and `gumbel` with `tau` 0.9 gave 11% to 13%. A broken piece is often a token too. A count of the
  right kind needs the log probability of each token of a reply with no soft input. No route gives that now.

Thus the next trials are these. Use a harder task for the reasoning. Give a creative reply its soft inputs in its
reasoning, then decode the reply as tokens. The raw counts are in `target/trials-2026-11-12/`, which git does not
hold.

**The second trials of 2026-11-12**, on the build of `30e4d86`. The score of a reply is the mean log probability of
its content with no soft input (`docs/lab.md`, "The score of a text"). The force puts the content after the same
prompt, with thinking off. Each row of these trials ran on `lfm3` or `lfm3-1`, which give the same score for a text.

The score finds the broken words. The rows of thinking off above, run again with the same seeds:

| Row | Owl story: score | Median of the minimum | Sea poem: score | Median of the minimum |
|---|---|---|---|---|
| no `soft` | -1.70 | -11.1 | -2.13 | -10.9 |
| `moi`, `beta` 1 | -1.71 | -11.7 | -2.11 | -10.9 |
| `moi`, `beta` 0.25 | -1.70 | -11.9 | -2.19 | -10.1 |
| `gumbel`, `tau` 0.5 | -2.35 | -19.2 | -2.26 | -16.6 |
| `gumbel`, `tau` 0.9 | -3.37 | -21.7 | -3.57 | -19.0 |

Harder reasoning: 12 tasks with 6 seeds each, thinking on, 1500 tokens at most. The tasks are four products of two
numbers, two-step word problems, a count of primes, and a trick question.

| Row | Right of 72 | Median thinking | Share |
|---|---|---|---|
| no `soft` | 65 | 249.5 | 0 |
| `moi`, `beta` 0.25, `span: "think"` | 65 | 265 | 0.0056 |
| `gumbel`, `tau` 0.5, `span: "think"` | 51 | 262 | 0.041 |
| 16 latent steps with `stop` | 46 | 337 | none |

- `gumbel` breaks digit work. Three of the four products were right in 1 of 6, where no `soft` gave 4 or 5. A mix of
  two digits is neither digit.
- Latent steps lost on two of the products (0 of 6 each), the train, the primes and the trick question.
- The stop never ended latent steps. In each request with latent steps, all 16 ran. The top tokens of the steps are
  words such as " need" and ":". The stop works in the host tests, but `lfm3` gives no end token in its steps.

Creative replies with soft inputs in their reasoning, thinking on, and the reply decoded as tokens. A run of 600
tokens cut most owl stories inside their reasoning, so the owl story ran again with 1500.

| Run | Row | Median thinking | Ended | Score |
|---|---|---|---|---|
| Sea poem, seeds 0 to 23, 600 tokens | no `soft` | 258 | 17 of 24 | -1.95 |
| | `moi`, `beta` 0.25 | 145.5 | 21 of 24 | -1.83 |
| | `moi`, `beta` 1 | 205 | 19 of 24 | -1.91 |
| Sea poem, seeds 24 to 71, 600 tokens | no `soft` | 241.5 | 34 of 48 | -1.85 |
| | `moi`, `beta` 0.25 | 209.5 | 42 of 48 | -1.93 |
| | `moi`, `beta` 1 | 224.5 | 38 of 48 | -1.86 |
| Owl story, seeds 0 to 23, 1500 tokens | no `soft` | 858.5 | 20 of 24 | -1.93 |
| | `moi`, `beta` 0.25 | 687.5 | 23 of 24 | -1.90 |
| | `moi`, `beta` 1 | 1025 | 20 of 24 | -1.93 |

- In each of the three runs, `moi` with `beta` 0.25 gave shorter reasoning and more replies that ended. The reply
  scored the same, within 0.1. The gain is 13% to 44% of the median reasoning, and it is smaller in the larger run.
  This agrees with Soft Thinking, which gave 12% to 22% shorter replies.
- `moi` with `beta` 1 is not a gain: it made the reasoning of the owl story longer.
- With thinking on, `gumbel` in the reasoning did not break the reply. With `tau` 0.9, the sea poem reasoned for a
  median of 419.5 tokens, and 16 of 24 replies ended.
- Each score is of the replies that ended, and the median thinking counts a cut reply at the limit.

Thus `moi` with a low `beta` in the reasoning is the first setting that gives a measured gain. It is small, and two
prompts hold it. Trial it on more prompts before the web app gets a field for it.

**Ten more prompts**, the same day and build: no `soft` against `moi` with `beta` 0.25 and `span: "think"`, thinking
on, 1500 tokens at most, the seeds 0 to 23. Five prompts ran on `lfm3`, and five on `lfm3-1`.

| Prompt | Median thinking | Ended of 24 | Score | Share |
|---|---|---|---|---|
| Dialogue: a robot and a cat | 1034 -> 745.5 | 18 -> 23 | -2.36 -> -2.27 | 0.072 |
| Letter of a grandmother | 922 -> 681.5 | 11 -> 17 | -1.99 -> -1.87 | 0.069 |
| Advertisement of a time machine | 939.5 -> 773 | 14 -> 17 | -1.92 -> -1.83 | 0.077 |
| A night market on clouds | 530.5 -> 476.5 | 23 -> 24 | -1.90 -> -1.87 | 0.058 |
| A lighthouse keeper | 1500 -> 1436 | 6 -> 9 | -1.98 -> -2.05 | 0.039 |
| A myth of the moon | 1500 -> 1500 | 0 -> 4 | -2.27 -> -2.26 | 0.090 |
| A folk song of a river | 1500 -> 1500 | 1 -> 1 | -1.97 -> -2.01 | 0.089 |
| Three haiku | 1500 -> 1500 | 0 -> 0 | none | 0.028 |
| A recipe of a dragon | 1500 -> 1500 | 0 -> 0 | -1.80 -> -1.57 | 0.079 |
| A sonnet of a city | 1500 -> 1500 | 0 -> 0 | none | 0.048 |

- Each arrow goes from no `soft` to `moi`. A score is the mean of the replies that ended. A reply cut at the limit
  that has content gets a score too.
- In no prompt did fewer replies end with `moi`. Over the ten prompts, 73 of 240 ended with no `soft`, and 95 with
  `moi`.
- Four prompts had a median reasoning below the limit: the dialogue, the letter, the advertisement and the market.
  In each, `moi` cut the median reasoning by 10% to 28%, and the score rose by 0.03 to 0.12.
- Five prompts reasoned past 1500 tokens in most replies, with either setting. Haiku and the sonnet never ended.
  The first 400 characters of their reasoning count syllables and rhymes. Nobody read the rest, so a loop is a guess.
  A loop is the trap that Amy named on 2026-11-12. A stop of the kind of Cold Stop is the tool for it.

Thus `moi` with `beta` 0.25 in the reasoning gains over twelve creative prompts. The reasoning is shorter, more
replies end, and the score stays. The gain is not large, but no prompt went against it.

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
