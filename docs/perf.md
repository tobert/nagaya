# Starting places for speed and scale

This doc holds measurements and ideas, not rules. It is the start of the next round of work on the speed of a pass
and on the count of contexts in a pass. It started on 2026-11-14 (Claude Fable 5.2). Each idea names a test that
shows if it is true. Do the test before the change.

All numbers are of `lfm3` (LFM3-1.7B) on one B70 card, through the route of generations with `forks`, unless a line
says other. A step is one decode token for each member. The step time comes from two runs: one of 160 or 224
tokens, and one of 32 tokens. It is the difference for each token, so it holds no prompt and no start cost.

## The step time by member count

| Members | Temperature 0.8, no filter | Temperature 0.8, `top_k` 64 | Greedy |
|---|---|---|---|
| 1 | 10.9 ms | 9.3 ms | 9.1 ms |
| 2 | 14.8 ms | 11.7 ms | 11.3 ms |
| 3 | 18.8 ms | 14.0 ms | 13.5 ms |
| 4 | 22.9 ms | 16.5 ms | 15.7 ms |
| 5 | 27.0 ms | 25.8 ms | 24.8 ms |
| 6 | 37.9 ms | 28.2 ms | 27.0 ms |
| 7 | 41.9 ms | 23.5 ms | 22.2 ms |
| 8 | 45.9 ms | 25.9 ms | 24.3 ms |

One formula fits each cell to 0.3 ms: `T = p * a + n * b`.

- `a` is the fixed cost of a pass: 6.9 ms.
- `b` is the cost of each member: 4.0 ms with no filter, 2.4 ms with `top_k` 64, and 2.2 ms greedy.
- `n` is the member count, and `p` is the count of passes in a step: 1 or 2.

Thus three things make the step time, and each is a place to start.

## 1. A step of one swarm runs as two passes

The engine declares 8 segments in a pass (`crates/nagaya-mk-lfm3-bmg/src/spec.rs`, `MAX_SEGMENTS`). But `p` is 2 in
these cells: 6, 7 and 8 members with no filter, and 5 and 6 members with `top_k` 64 and greedy. With 7 and 8
members, the last two columns have one pass. The second pass costs 6.9 ms in each step.

The count does not rise with the member count, so it is not a limit. An idea, not checked in the code:

- The scheduler joins the decode quanta that are ready when it chooses (`crates/nagaya-sched`, `choose`).
- The members start at other times, because each takes its copy of the head in turn.
- When a pass of some members ends, the other members are ready, and they run at once. Then the first members
  are ready, and they run alone again. The two sets never meet.

The test: count the segments of each decode quantum of one request. The histogram `nagaya.quantum.run.duration`
counts the quanta, so its count for a request of 8 members and 100 tokens is 100 or 200.

If the idea is true, these are the changes to weigh:

- A decode quantum waits a short time for the other members of its request before it runs alone.
- A request with `forks` starts each member at the same step.
- A clock (`docs/beats.md`) starts each member on one grid, so they join with no other rule.

The same question is open for the Qwen3.8 engine. There, 8 contexts gave 46.6 tokens each second and 4 gave 47.3
(`AGENTS.md`, milestone 30). The cause was given as the experts that each row reads. Run this test there too.

## 2. The sampler sorts the vocabulary for each token

After a pass, one device thread ends each member in turn (`crates/nagaya/src/jobs.rs`, `run_joined_bodies`). The card
does no work in that time. For each member, the sampler does this on a row of 125,033 logits
(`crates/nagaya-council/src/sampler.rs`, `candidates` and `pick_with_except`):

- It copies the row to `f64`, and it builds a sort key for each logit.
- With no `top_k`, it sorts every key. With `top_k`, it selects the top keys first, and it sorts only them.
- It allocates about 6 vectors of the size of the row.
- It computes the exponential of each logit two times: for the draw, and for the probability and the entropy.

Measured alone on a CPU, with no card, as the mean of 200 rows:

| Sampler | Time for each row |
|---|---|
| Greedy | 0.44 ms |
| Temperature 0.8, `top_k` 64 | 0.63 ms |
| Temperature 0.8, `top_p` 0.95 | 2.22 ms |
| Temperature 0.8, no filter | 2.82 ms |

The differences agree with the table of step times. Thus with no filter, the sampler is 2.8 ms of the 4.0 ms of
each member, and the card waits for it.

What a client can do today: send `top_k`. With 8 members, a step takes 25.9 ms where it took 45.9 ms. `top_k` cuts
the tail of the distribution, so it is another request, not the same one faster.

Changes that keep the bits, in the order of their size:

- Run the members of a pass on the pool threads. The process has 30 of them. Each row is apart from the others.
- Sort the keys with a radix sort. The keys are `u64` and their order is total, so the order stays.
- Keep the vectors of a job from one token to the next.
- Compute the exponentials one time, if the two sums can keep their order.

The sum of the draw adds the terms in sorted order. Thus a change that sorts less changes the bits of the sum, and
it is a change of spec 11.2. Do not make it as a speed change.

## 3. The head runs one time for each member

Greedy still costs 2.2 ms for each member, and the sampler is 0.44 ms of it. The rest is not measured in parts.
These are the parts, with the largest first by estimate:

- **The head.** A batch runs the weight ops one time for all rows, but the head runs one time for each row
  (`crates/nagaya-mk-lfm3-bmg/src/model.rs`, `batch`: "Each row is its own head matmul of one row"). The head is
  128,000 rows of 2,048 values in BF16: 524 MB, read again for each member.
- **The read of the logits.** Each row is 512 kB, from the card to the host. Check if each row is its own copy
  command with its own wait (`crates/nagaya-mk-bmg/src/level_zero.rs`).
- **The tasks of a row.** Each row has its own rotation, attention, convolution and copies in each layer, and each
  task has a wait at the scoreboard.
- **The host work of a token.** The parser, the token event, the stats and the record.

An idea for the head: one matmul of `n` rows. A weight of more than 65,536 rows takes one path for 1 to 8 rows
(`megakernel.cl`, `OP_MATMUL`, "At most 8 tokens"). That path sums each row in one order (`docs/engine-abi.md`,
"Batched passes"). Thus the bits can stay. The card gate: `n` rows in one head matmul give the bits of `n` head
matmuls of one row. If the gate fails, stop: the head stays as it is.

The test of each part: a count of time inside the pass call and outside it, for each step. Today the process has no
such count. The first tool to build is that split. It has three parts: the time of the launch to its end, the time
of the reads, and the time of the host after them.

## 4. The fixed cost of a pass

`a` is 6.9 ms. Nobody split it. It holds the weight ops of the backbone, over about 2.9 GB of weights. It also
holds the build of the task ring, the write of the ring, the launch and the wait.

- The step of one member is the floor of each chat. Thus this cost matters most for one person at a time.
- The test: the same split as above, and the task count of a ring.

## What the numbers say about the card

With 4 members and no filter, a step is 22.9 ms. The host sampler takes about 11 ms of it. Thus the card is idle for
about half of each step at the full speed of today. The limit is one host thread, not the card.

A first guess of the gains for 8 members, each on top of the last. Each number is from the formula, not measured,
but the second, which is measured:

| Change | Step time | Tokens each second |
|---|---|---|
| Today, no filter | 45.9 ms | 174 |
| One pass in a step | 38.9 ms | 206 |
| A sampler at the cost of `top_k` 64 | 26.1 ms (25.9 ms measured with `top_k`) | 306 |
| The members end on pool threads | about 21 ms | about 380 |
| One head matmul, if it is most of the rest | about 16 ms | about 500 |

## The long slot

The profile had a slot of 262,144 positions on `lfm3-1`. Its card test ran for the first time on 2026-11-14
(`crates/nagaya-mk-lfm3-bmg/tests/xe_feed.rs`). The bits of a long slot are the bits of a short slot. But the longest
block of 512 tokens took 2,965 ms, and the engine declares a quantum of 2,000 ms at most. The last decode step took
37 ms. The slot is out of the profile until the bound holds (Decided 2026-11-14).

The owner expects that contexts of 32,768 positions are enough for now, and that 65,536 comes next. Places to start:

- Measure the longest block at 65,536 positions first. It can fit the bound with no change.
- A block of fewer tokens for a long slot. The scores of a block are its tokens times the keys.
- A bound that the engine declares for each slot size.
- The scores of a block of 512 over 262,144 keys take 8.6 GB. A smaller block takes less.

## Other open things from the same day

- The test file `crates/nagaya/tests/council_media.rs` fails 9 of its 10 tests when `LFM3_OMNI_MODEL_DIR` is set.
  The checks skip it on a machine with no model. 7 tests build a context with no `thinking`, which gogi requires.
  1 test expects 3 media inputs, and a build with the feature `video` has 5. 1 test gets "post the spec first".
- Gemma 4 fails 3 checks of the gogi suite at pin `8830357`: a reply to an unmarked message holds no head. The
  Qwen3.8 engine got this repair in milestone 34 step 1.
- One generation of `lfm3-omni` with an audio part stopped the process with a Level Zero error `0x70000003`. The
  process had run for 43 hours. The same request passed 4 times after the restart. The cause is not known.

## The tools of these measurements

They are not in the repository yet. Each is short, and a later session can write it again:

- A client that times a request of `forks` at two values of `max_tokens`, for each member count and sampler.
- A bench of `nagaya_council::sampler::pick` on rows of 125,033 logits.
- A reader of `act_freq` of a card each 5 ms (`docs/beats.md`, "The speed of a swarm").
