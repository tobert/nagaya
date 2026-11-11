# The swarm driver

`tools/swarm` runs a swarm recipe against the nagaya. A recipe is a JSON object that describes one round. The
driver runs rounds for a fixed wall time, and it logs each program and each stage. `docs/swarm-ideas.md`, "An agent's
interface to a swarm", defines the recipe. The task is the online bin packing of the first trials.

## A round

1. **Parent.** The driver picks a parent program: a tournament of two among the top 5 of the archive.
2. **think.** `think.agents` agents each write one short idea, with one focus line each.
3. **select.** `vote` asks the model for the number of the best idea. `novelty` keeps the `keep` ideas whose
   embeddings are farthest apart. `all` keeps every idea.
4. **distill.** `pick` uses the kept idea as the plan. `combine` asks one call to combine the kept ideas into a plan,
   with the parent program last in the prompt. `joined` asks `ensemble` with `prob_mean`, with one context for each
   kept idea.
5. **act.** `plain` makes `forks` independent generations of the plan. `contrast` makes `forks` calls of `ensemble`
   with `contrast`: the context with the plan against the same context with no plan. `joined` makes one call of
   `ensemble` with `forks` copies of the plan context, and it gives one program.
6. **Evaluate.** A sandbox runs each program on the CPU, with no network. The archive keeps each program that scores.

## The models

`models` lists one or more served models. The driver spreads the independent calls over them in turn: the think
agents, the vote, the combine call and each fork of `plain` or `contrast`. A call of `ensemble` goes to one model.
The embeddings of `novelty` go to `select.model`, which is `embeddinggemma-2` by default.

## The call options

The stages `think`, `distill` and `act` take three optional members: `system`, `sample` and `thinking`. `select`
with `by` `vote` takes `system` and `thinking`. `prime` takes `system` and `sample`, and it always thinks. A
recipe without these members sends the same bytes as the driver before them. `tests/fixtures/requests-a36b117.json`
holds the hashes of those bodies.

- **`system`.** A line of text. The context of each call of the stage starts with
  `{"role": "system", "content": ...}`. A top-level `system` acts on each stage that sets none.
- **`sample`.** An object with any of `temperature`, `top_p`, `top_k` and `min_p`, the members of the council v2
  sampler. The driver sets the seed. `sample.temperature` and the member `temperature` of the stage are one setting,
  so a recipe that sets both is refused. The vote stays greedy, so `select` takes no `sample`.
- **`sample.ladder`.** A list of one or more temperatures, each with the rules of `temperature`. Member `i` of the
  stage uses step `i` modulo the length of the list. The members are the think agents and the forks of `plain` or
  `contrast`. The other members of `sample` act on each member. A ladder and a temperature cannot both be set. A
  stage that makes one call refuses a ladder: `prime`, `distill` and `act` with `joined`.
- **`thinking`.** True or false, sent as `context.thinking`. The default is false.
- **No `repetition_penalty`.** The sampler of the nagaya (spec 11.2) has none, so a recipe with it is refused.

The model card of LFM3-1.7B gives two sets of settings. For general work and code, it gives `temperature` 0.1,
`top_k` 50 and a repetition penalty of 1.1. For math and science, it gives `temperature` 1.0, `top_p` 0.95 and
`top_k` 20.

### Thinking and the prefill

Each call of think, the vote, distill and act ends with a prefill: an open assistant message with `content`, such
as `Idea:`. The council closes the thinking at a `content` member (gogi "Continuations"). The nagaya code shows this:

- `ReplyStart::of` (`crates/nagaya-council/src/family.rs`) starts the reply in the reasoning only when the open
  message has no `content` member.
- `run_open` (`crates/nagaya-text-lfm3/src/council/mod.rs`) renders `<think>`, the open reasoning, `</think>`, then
  the prefill.

Thus a prefilled call with `thinking` true writes no reasoning. Its `content` is the prefill and the new text, as
with thinking off, and the driver reads it the same way. To think first, use `prime`. `lfm3-omni` does not think: a
context with `thinking` true gets a 400 that names `thinking`.

## Snapshots

`snapshots` is a top-level member, true or false. The default is false, and then a recipe sends the bytes of the
driver before this member. Each recipe in `recipes/` sets it to true.

With `snapshots` true, the driver lets the nagaya hold the part that the calls of a stage share:

- The head of a round, which is the problem and the parent, is its own user message with `snap` true. The rest of
  each prompt is a second user message. Thus the model reads two user messages in place of one.
- After a prime, the request of the prime and its text are shared too, and the text of the prime ends with a
  boundary. Then an agent feeds only its focus.
- Each fork of `plain` or `contrast` has the same prompt, so the end of the prompt is a boundary. Then a fork feeds
  only its prefill.
- The driver makes no call only for a snapshot. The calls of a stage run at once. The nagaya feeds a prefix once
  for the requests in flight that share it (spec 11.14). Until 2026-11-11 the driver built the shared messages first,
  with one more call for each model, because each call fed the shared part.

`distill` with `combine` shares nothing: its prompt has the parent last, after the ideas.

On 2026-11-11, three rounds of `s2s-combine.json` on `lfm3` gave these counts. Without snapshots, think fed 4,624
of 6,366 prompt tokens, and act fed 4,532 of 4,532. With snapshots and a build call, think fed 1,504 of 6,929, and
act fed 767 of 5,514. The runs of 2026-11-10 had no snapshots: they fed 60% to 100% of each prompt again.

## The prime

`prime` is an optional stage. It runs once in each round, before think. It makes one generation with `thinking`
true and no prefill. Its user message is the head of the round (the problem and the parent) and a request to find
where the parent wastes bins.

- `prime.max_tokens` is required, because a thought can be long. `prime.temperature` is 0.6 by default. `prime`
  also takes `system` and `sample`.
- The text of the prime is its reasoning, then its content, with a blank line between them. A prime that
  `max_tokens` cut inside the thinking has text in its reasoning only.
- Each think agent gets these messages in order. The prefill `Idea:` comes last, as before.
  1. the system prompt of think, if one is set;
  2. the user message of the prime;
  3. the text of the prime, as an assistant message;
  4. the focus and the request for one idea, as a user message.
- Every agent shares the messages before its focus. With `snapshots` true, the nagaya holds that part, and an
  agent feeds only its focus. Without it, each agent feeds the whole prompt.
- The round line gives the text in `prime`, and the stage record `prime` gives its calls and tokens.

The text of the prime goes into `content`, not into `reasoning`. The chat template of LFM3 removes the reasoning of
each turn before the last user message, unless `preserve_thinking` is set. The focus is a later user message, so a
`reasoning` member would not reach the agents. `</think>` is not part of the text, because the parser removes the
markers.

## Latent steps

`think.latent`, `distill.latent` and `act.latent` give the latent steps of the calls of that stage. With a value above
0, a generation goes to `POST /nagaya/v1/lab/ensemble` with one context and `"latent": {"steps": N}`. The member
`latent` of `ensemble` is Proposed (`docs/lab.md`, "Latent steps in a request"). The driver sends it only when the
value is above 0, so a nagaya without latent steps serves the other recipes.

## Run a recipe

Run the driver in `tools/`:

```
cd tools
python3 -m swarm swarm/recipes/pick-vote.json --out /tmp/pick-vote-1.jsonl --seed 1
```

- `--url` gives the base URL of the nagaya. With no `--url`, the driver reads `NAGAYA_URL`. With neither, it uses
  `http://198.51.100.10:8091`.
- `--seed` gives the seed of the parents, the foci and the seeds of the calls.
- `loop.minutes` sets the wall time. `loop.in_flight` sets the count of rounds that run at once. `loop.rounds`, if
  present, stops the run after that count of rounds.
- At the end, the driver prints the summary as one line of JSON.

The recipes in `recipes/` are the best modes of the first trials:

| Recipe | think | select | distill | act |
|---|---|---|---|---|
| `pick-vote.json` | 8 agents | vote | pick | 1 plain fork |
| `s2s-vote.json` | 8 agents | vote | pick | 4 plain forks |
| `s2s-combine.json` | 8 agents | all | combine | 4 plain forks |
| `s2s-combine-latent.json` | 8 agents | all | combine, 16 latent steps | 4 plain forks |

These recipes add the settings of the model card to `s2s-combine.json`, with a short system prompt for each stage:

| Recipe | prime | think | distill | act |
|---|---|---|---|---|
| `s2s-combine-card.json` | none | 1.0, `top_p` 0.95, `top_k` 20 | 0.6, `top_p` 0.95 | 0.1, `top_k` 50 |
| `s2s-combine-card-prime.json` | 384 tokens, 0.6, `top_p` 0.95 | as above | as above | as above |

The first number of each cell is the temperature.

`s2s-combine-warm-ladder.json` is `s2s-combine-warm.json` with a ladder in think and in act. The 8 think agents use
0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2 and 1.2, with `top_p` 0.95 and `top_k` 20. The 4 act forks use 0.3, 0.6, 0.9 and
1.2, with `top_p` 0.95. On 2026-11-10, code forks at 0.1 gave 146 distinct programs of 1,576. Forks at 1.0 found the
best scores, with more programs that failed.

## The log

The log is a JSONL file with three kinds of lines.

A **program line** is one evaluated program:

- `excess` is the score on the train seed (1), or `fail` gives the reason of the failure.
- `valid` is true when the program scored. `constant` is true when `priority` returns one number, such as
  `return 0.0`. `improved` is true when the score is below the score of the parent.
- `parent` is the score of the parent. `best` is the best score of the archive after this program.
- `kind` names the methods of select, distill and act, for example `vote-pick-plain`.
- `round`, `fork`, `model`, `plan`, `ideas` and `src` give the origin of the program.
- `sample` gives the temperature of the act call that made the program, and the other members of `act.sample`.

A **round line** has the member `round_stats`:

- `programs`, `valid`, `constant` and `improved` are counts. Each `_share` member is the count over `programs`.
- `vote` and `bad_vote` give the result of a vote. A bad vote named no idea of the list, so the driver picked one at
  random. `kept` gives the indices that `novelty` kept.
- `stages` has one record for each stage: `prime` if the recipe has it, then `think`, `select`, `distill`, `act`
  and `evaluate`. A record gives `calls`, the tokens `prompt`, `fed` and `completion`, and `seconds` of wall time.
  The tokens are the sums of the usage of each reply. `models` gives the calls of each model.
- `prime` gives the text of the prime, if the recipe has a prime.
- `think_temperatures` gives the temperature of each think agent, in the order of the agents.

The **summary line** has the member `summary`:

- `best_train` is the best score on the train seed (1). `best_test` is the score of that program on the test seed
  (2). `best_src` is its source.
- `rounds`, the counts and the shares are the sums of all rounds. `stages` gives the sums of each stage.
- `recipe` is the recipe with each default filled.
- `by_temperature` has one row for each act temperature, in rising order. A row gives `programs`, `valid` and
  `improved`, and `best_excess`, the best score of its valid programs. Thus a run shows which temperatures pay off.

## Tests

The tests use a stub transport, so they make no call to a nagaya. Run them in `tools/`:

```
python3 -m unittest discover -s swarm/tests -t .
```

The sandbox tests run `bwrap` on the CPU. They skip when `bwrap` or `prlimit` is absent.
