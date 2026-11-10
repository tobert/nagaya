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

Batched passes are what it lacks. Today the agents of a round run one after the other on `lfm3`.

## Open questions

- Is a context's bits dependent on its batch acceptable for any served model?
- Do the 6 full attention layers of LFM3 carry enough between agents? A trial on the card can answer this.
- Is the merge a council v2 feature, or a lab route first (`/nagaya/v1/lab/`)?
- Does a swarm round run as one job, or does the client drive it with many requests?
