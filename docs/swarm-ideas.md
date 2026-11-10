# Swarm ideas

This doc holds ideas, not rules. Milestone 29 builds some of them (`AGENTS.md`). Amy asked for it on 2026-11-09,
after she read SwarmEvolve: "I'm more interested in how we can do the parallel processing and then the merge." A copy
of the paper is in `~/papers/2611.08483-swarmevolve.pdf`.

## The paper

*SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization*, by Qile Jiang and George Em
Karniadakis, arXiv 2611.08483, 2026-11-05. The code is `github.com/Qile-J/SwarmEvolve`.

It is an evolutionary search for programs, as AlphaEvolve is. Each round runs 10 swarms of 45 agents on one model:

- Each agent reads the problem, a focus prompt and one parent program. The agents of a parent share its KV cache.
- Each agent takes 40 latent steps. In a latent step, the last hidden state becomes the next input embedding, after a
  map and a rescale (LatentMAS). No text decodes.
- A novelty filter keeps the 30 agents of each swarm whose mean hidden states are the farthest apart.
- A summarizer reads one KV cache: the problem, the 30 agents, the parents and its instruction. It rotates the keys
  of each agent block to the offset of that block. Then it writes one program, and a CPU evaluator scores it.

The numbers, from the paper:

- The 450 agents of a round take about 4 s. A round takes about 1 minute.
- It reaches the 60-minute scores of OpenEvolve, ShinkaEvolve and CodeEvolve in a median of 3 to 8 minutes.
- With text agents of 40 tokens, the mean score is 0.796, against 0.804 with latent steps (Supplementary Table 1).
  Thus latent steps give speed more than quality.
- The paper names hybrid models as a limit, because their state is a summary of the past.

## LFM3 and the paper's limit

LFM3 is a hybrid of a friendly kind (`config.json` of `LFM3-1.7B`):

- The 13 convolution layers have a kernel of 7. Their state is the last 6 inputs, not a summary of the past.
- The 6 full attention layers see every block of a merged context.
- The 5 sliding attention layers see the last 256 positions only.

Thus the signal between agents flows through 6 of the 24 layers. Whether that is enough is a question for a trial.

The embeddings of LFM3 are tied. Then the map of a latent step is close to the identity for a small ridge weight. A
latent step is about this: take the last hidden state, and scale it to the mean norm of the embedding rows.

## What the nagaya has now

- **Decisions** run in parallel and merge at the output. A decision reads one case after each of up to 8 contexts,
  each from its own held snapshot (gogi "Decisions"). The pool merges the answer distributions: `linear`, or
  `loglinear`, where one context can veto. `leave_one_out` shows how much each context moves the result.
- **Branch blocks** feed a shared tail and several branches in one pass on Gemma 4 (spec 6.14, "Branch blocks").
- **The store** keeps each prefix once, by its recipe. Agents with one problem and one parent share that prefix.
- **The residual tap** reads the residual after a layer (spec 13.1). The mean hidden state of an agent comes from it.
- **Media rows** put rows of numbers at positions of a context (spec 11.11). A latent step is a row of this kind.
- **Segments.** The ABI lets several segments share a pass (`Engine::pass`, engine-abi `max_segments`). The scheduler
  joins them only when the engine declares `segments_exact` (spec 5.7).

A decision is the nearest thing to a swarm today. Its contexts are the agents, and its held snapshots are their KV
caches. The difference is the place of the merge. A decision merges probabilities after each read. SwarmEvolve merges
the caches, so one read attends to every agent.

## Batched passes on the LFM3 engine

A survey of the engine on 2026-11-09 found these facts:

- The engine refuses more than one segment: `MAX_SEGMENTS` is 1 (`nagaya-mk-lfm3-bmg/src/spec.rs:50`).
- The process sends one segment to each pass (`nagaya/src/council/build.rs:1131`).
- The weight ops already take `n` rows, and the shared buffers hold 512 rows. A prompt pass is already a matrix
  multiply of many rows. Thus a decode step of B contexts reads the weights once for B tokens.
- The state of a context is its slot: the convolution state, and the K and V planes. The rest is shared.
- The per-row work is small, and its ops exist. Each row needs a rotation at its position, an attention task on its
  slot, and a convolution task on its state.
- The head gives at most 16 logit rows (`spec.rs:53`), so B is at most 16. The task count at B = 16 is about 1,300,
  under the limit of 2,048.

The hard part is the bits, not the kernels. The matrix multiply takes another path for each range of row counts
(`nagaya-mk-bmg/kernels/megakernel.cl:785`, `:803`, `:827`). Thus the bits of a context would depend on its
neighbors in the pass. That breaks `segments_exact`, and the rule that a job and a seed give the same bits.

Two ways out:

1. A decode path whose sum order for a row does not change with the row count, also for one row. This is a new
   identity and one refill, and maybe a slower decode of one context.
2. Accept bits that depend on the batch for some served models, and declare it.

The steps, smallest first:

1. Decide the bits question.
2. Write `model::pass` over a list of slot and position pairs, behind `MAX_SEGMENTS` above 1. The card gate: B
   segments give the bits of B single passes.
3. Give each row its own control list, or join only segments with equal lists.
4. Declare `segments_exact`, and let `build.rs` pack the ready segments into one pass.
5. If the task count is too high: a rotation that reads a buffer of positions, and attention over many rows.

Batched passes help every client, not only a swarm. kaijutsu's seats and a text swarm would gain the most.

## Latent steps

A latent step could be a call: "take N latent steps on this context". The engine loops on the card. It takes the last
hidden state, scales it, and feeds it as the input row of the next position.

- The steps sample nothing, so they are deterministic. A recipe item can name the count and the map. Then the store
  keeps the result and shares it like any prefix.
- The client reads the mean hidden state of the steps through a tap. The novelty filter runs in the client.
- With batched passes, the agents of a swarm take their steps together, as the paper does.

What a latent step gives, from the paper and from the LFM3 engine (senpai, 2026-11-09):

- **A message that carries more.** A hidden state holds the whole leaning of the model at a step, not one sampled
  token. The paper measured a small gain: 0.804 against 0.796 for text agents of 40 tokens.
- **A little speed.** A latent step is still a pass through all 24 layers. It skips the head and the sampler. The head
  is the table of 128,000 rows of width 2,048, about 15% of the weights that a step reads. The paper's larger gain in
  speed comes from its Python loop.
- **Fixed bits.** A latent step samples nothing, so the store keeps and shares it like a prefix. Two agents with the
  same prompt give the same states, so the diversity comes from their prompts only.
- **A need for the merge.** No text comes out, so only a context that attends to the agents' K and V rows reads them.
  A text swarm needs no merge: its summarizer reads the texts as a prompt.

The read along the way. LFM3 has no separate encoder: its head is its embedding table. Thus a latent step can also
give its logit row, and a client can read the top token of each step as text. The text does not go back into the
context. It is a view of the agent, as the trace of the J-Lens is.

A held latent context is an ordinary context. It can continue in tokens and put its state into words. A decision can
read its case after latent steps, too.

## The merge

A merge makes one context from blocks of several held contexts. The summarizer then continues it. On LFM3, the
survey found that the host can build it with calls that exist:

- Read the K and V rows of each block with `cells_read` (`nagaya-mk-lfm3-bmg/src/engine.rs:944-1015`).
- Rotate each key by the offset of its block, in the rotated pairs only, with the theta of its layer. The values do
  not change.
- Write the rows into the new slot with `cells_write`, then set its length with `set_n`.
- The sliding layers keep the last 256 positions of the result. The convolution state comes from the block before
  the summarizer's instruction, and that instruction replaces it after 6 tokens.

Rules for a design:

- A merged context is not the bits of a contiguous prompt pass, because two rotations round differently from one. It
  is a recipe of its own: the content id of each block, its offset, and the order.
- Do not leave a gap of empty rows. Attention reads each row before the position, with no mask.
- A merge reads blocks of one engine identity only. The KV caches of two models do not mix.

## A merged read

A decision with a merge could read the case once, after the concatenation of its contexts. That is the decision of
today with an early merge. The fields of a decision stay. A new pool method, such as `merged`, names the change.
`leave_one_out` keeps its meaning: read the merge again without one block. gogi writes such a contract first.

## A text swarm

Amy is thinking about a text swarm in kaijutsu. The nagaya gives it these parts now:

- The store shares the prefix of a problem and a parent across the agents.
- `continuation` and generations write each agent's short text. The paper's 40 tokens of text came close to its
  latent steps.
- A decision pools the agents' answers to a fixed question.

Milestone 27 added batched passes, so 8 agents of a round decode in one pass on `lfm3`.

## The first trials

Milestone 29 step 1 ran on 2026-11-09 and 2026-11-10, on b70host. The task is online bin packing, after
FunSearch: 5 sets of 5,000 items, with sizes from a Weibull shape. A program is one numpy expression that scores
each bin. The score is the percent of excess bins over the lower bound. Best fit scores 4.08 on the train seed and
4.14 on the test seed. A sandbox with no network runs each program on the CPU.

Each run had 20 minutes on `lfm3`. The modes:

- **direct:** 8 calls at once. Each call mutates a parent into a new program.
- **swarm:** 16 agents write ideas. A novelty filter keeps 8, and one summary writes the code from all 8.
- **hint:** direct, with one focus line at random.
- **pick:** 8 agents write ideas, one idea is picked, and one call codes it. The pick is by consensus (the idea
  nearest the others in embedding space) or by vote (the model names a number).

| Mode | Seed | Programs | Valid | Train | Test |
|---|---|---|---|---|---|
| direct | 1 | 2,488 | 82% | 1.58 | 1.26 |
| direct | 2 | 2,521 | 92% | 3.92 | 3.92 |
| swarm | 1 | 193 | 43% | 4.08 | 4.14 |
| swarm | 2 | 191 | 67% | 4.08 | 4.14 |
| hint | 1 | 2,682 | 61% | 3.92 | 3.94 |
| hint | 2 | 2,422 | 95% | 2.81 | 2.83 |
| pick, consensus | 1 | 341 | 60% | 3.95 | 3.92 |
| pick, consensus | 2 | 335 | 76% | 4.08 | 4.14 |
| pick, vote | 1 | 320 | 88% | 1.20 | 0.97 |
| pick, vote | 2 | 325 | 77% | 4.08 | 4.14 |

What the logs show:

- The summary of 8 ideas never beat best fit. 30% of its programs were a constant, such as `return 0.0`. Direct
  gave 1% constants, and pick by vote gave 4%.
- In the summary prompt, the parent program ends 434 tokens before the first code token. LFM3's sliding layers see
  256 positions, and its convolutions are local. Thus only the 6 full attention layers see the program that changes.
  In pick mode, the gap is 128 tokens. This is a hypothesis, and a trial with the parent last tests it.
- 91% of the valid programs of direct scored the same as their parent. Only 0.2% to 0.6% of valid programs improved.
- The best result is pick by vote on seed 1. Each mode won on at most one seed, so two seeds decide nothing.

## Plan, then code

Amy, 2026-11-10: "I feel like we shouldn't try to get good code out of the thinking swarm. we need to distill it into
something focused". LFM3 writes good ideas at a high temperature, but it loses precision in a long context. Thus a
round has three parts:

1. **Swarm 1 thinks.** Agents write short ideas at a high temperature.
2. **One step distills.** A vote picks one idea, or one call combines the ideas into one plan of two sentences.
3. **Swarm 2 codes.** Forks of one context code the plan. The store shares their prefix, so a fork costs little.

The forks of swarm 2 can end in two ways:

- **Independent forks** give K programs, and the evaluator or a vote picks after. This needs no nagaya code.
- **Joined forks** give one reply. Each step combines the K logit rows into one, the sampler picks one token, and
  all K contexts take it. This needs a lab route (`docs/lab.md`, "`ensemble`").

Joined forks can also do the distill step. Each fork holds the problem, the parent and one idea, and together they
write the plan. No one context reads every idea, so the sliding window does not cut the parent.

Two ideas wait for another day. A small tune of the distill step could learn from the logged plans and the scores of
their code. `LiquidAI/LFM3-1.7B-Base` could become a narrow coder.

## Open questions

- Is a context's bits dependent on its batch acceptable for any served model?
- Do the 6 full attention layers of LFM3 carry enough between agents? A trial on the card can answer this.
- Does the summary fail because the parent is outside the sliding window? The trial with the parent last answers it.
- Which rule of joined forks helps: the sum of the log probabilities, the mean of the probabilities, or a contrast?
- Is the merge a council v2 feature, or a lab route first (`/nagaya/v1/lab/`)?
- Does a swarm round run as one job, or does the client drive it with many requests?
