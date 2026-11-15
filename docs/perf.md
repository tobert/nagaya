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

That table is of the build before the row threads (spec 3.2, "The row threads"). Since `c75285a`, the rows of a
shared pass draw their tokens at the same time. The same measurement on that build, on 2026-11-14:

| Members | Temperature 0.8, no filter | Temperature 0.8, `top_k` 64 | Greedy |
|---|---|---|---|
| 1 | 10.9 ms | 9.3 ms | 9.1 ms |
| 2 | 13.0 ms | 11.5 ms | 11.3 ms |
| 3 | 15.0 ms | 13.3 ms | 13.0 ms |
| 4 | 17.0 ms | 15.1 ms | 14.9 ms |
| 5 | 19.0 ms | 17.0 ms | 16.6 ms |
| 6 | 20.9 ms | 18.7 ms | 18.3 ms |
| 7 | 30.1 ms | 20.7 ms | 20.0 ms |
| 8 | 24.9 ms | 22.4 ms | 22.0 ms |

- With 8 members and no filter, a step takes 24.9 ms where it took 45.9 ms: 321 tokens each second, where it
  gave 174.
- With 4 members and no filter, a step takes 17.0 ms where it took 22.9 ms: 235 tokens each second, where it gave 175.
- Each member now costs 1.8 ms to 2.0 ms with each sampler. The draw is not a part of that cost now. The slowest
  draw of a step is a part of the fixed cost: about 2.1 ms with no filter.
- One cell has two passes in a step: 7 members with no filter. Before, 7 cells had them. Section 1 is still open.
- The bits did not change. 39 seeded replies of 1, 4 and 8 members gave the same text on the two builds. 22 greedy
  replies gave one text, alone and in shared passes.

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

- Run the members of a pass on other threads. Each row is apart from the others. Done in `c75285a`: the row
  threads. One member alone still draws on its device thread.
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

Before the row threads, a step of 4 members with no filter was 22.9 ms, and the host sampler took about 11 ms of
it. Thus the card was idle for about half of each step. The limit was one host thread, not the card. With the row
threads the same step is 17.0 ms. What the card does in the 2 ms of each member is not measured yet (section 3).

A first guess of the gains for 8 members, each on top of the last. Each number is from the formula, not measured,
but the second, which is measured:

| Change | Step time | Tokens each second |
|---|---|---|
| Today, no filter | 45.9 ms | 174 |
| One pass in a step | 38.9 ms | 206 |
| A sampler at the cost of `top_k` 64 | 26.1 ms (25.9 ms measured with `top_k`) | 306 |
| The members end on pool threads | about 21 ms | about 380 |
| One head matmul, if it is most of the rest | about 16 ms | about 500 |

The row threads took the place of the second and third rows, with no change of the sampler: 24.9 ms measured, 321
tokens each second. The cost of each member that stays, about 2 ms, is the subject of section 3.

## The long slot

The profile had a slot of 262,144 positions on `lfm3-1`. Its card test ran for the first time on 2026-11-14
(`crates/nagaya-mk-lfm3-bmg/tests/xe_feed.rs`). The bits of a long slot are the bits of a short slot. But the longest
block of 512 tokens took 2,965 ms, and the engine declares a quantum of 2,000 ms at most. The last decode step took
37 ms. The slot is out of the profile until the bound holds (Decided 2026-11-14).

What failed is the time bound only. The cause is the size of the work. In each of the 6 full layers, a block
writes its scores over each key of the slot (`tasks::score_floats`). The softmax and the sum of the values read
them again.

A first curve came through the live service on 2026-11-14. It is the time of a prefill of `lfm3-1` at 8 prompt
lengths, as the difference for each block of 512. It holds the host time of a block too.

| Positions before the block | Time of a block of 512 |
|---|---|
| about 6,000 | 121 ms |
| about 10,000 | 139 ms |
| about 14,500 | 182 ms |
| about 19,000 | 182 ms |
| about 23,000 | 200 ms |
| about 28,000 | 242 ms |
| 262,144 (card test) | 2,965 ms |

- To 32,768 positions, a block costs about 100 ms and 5.5 ms for each 1,000 keys.
- That line gives about 1,540 ms at 262,144 positions, and the card gave 2,965 ms. Thus the cost grows faster than
  the keys in some part of the range. No measurement is there.
- By both slopes, a block at 65,536 positions takes 460 ms to 680 ms. At 131,072 it takes 820 ms to 1,440 ms.
  Thus both sizes can fit the bound of 2,000 ms with no change. A card test must show it.
- A prefill of 29,961 tokens took 10.1 s. A decode step at 262,144 positions took 37 ms, where a short one takes 9 ms.
- The memory is the second limit. The plan of card 1 has 31.9 GB of its 32 GiB in use. A slot of 65,536 positions
  takes 1.74 GB of `kv`, and the scores of its block take 2.1 GB of `work`. A longer slot means fewer slots.

**The cause of the bend, and its repair (2026-11-14).** An ignored test of `xe_lfm3_ops`
measures each op on the card, on a cache of random values. These are its times for one layer and a block of 512:

| Keys | Scores | Softmax | Sum of the values | The three, before | The three, after |
|---|---|---|---|---|---|
| 32,768 | 13.7 ms | 10.7 ms | 6.4 ms | 31.3 ms | 25.2 ms |
| 65,536 | 74.2 ms | 22.0 ms | 13.2 ms | 107.3 ms | 55.6 ms |
| 131,072 | 160.0 ms | 44.6 ms | 25.8 ms | 235.3 ms | 103.7 ms |
| 262,144 | 351.4 ms | 88.0 ms | 52.2 ms | 493.9 ms | 208.0 ms |

- `OP_ATTN_SCORES` took 71% of the time at 262,144 keys, and it grew by 5.4 from 32,768 keys to 65,536.
- Its work items had the keys as the fastest index. Thus it read the whole key plane of a head one time for
  each head and each tile of 16 tokens. That is 512 times in a layer. Past about 32,768 keys the plane does not
  fit the cache of the card, and each of those reads came from memory.
- Now the keys are the slowest index. The items that run at one time read the same key rows. At 262,144 keys the
  op takes 66.4 ms where it took 351.4 ms.
- An item computes what it computed before, so no bit changes. The kernel binaries change, so the identity of each
  engine on the cards changes, and each held context fills again one time.
- On the engine, the longest block of a slot of 262,144 positions takes 1,330 ms where it took 2,965 ms. The long
  slot fits the bound of 2,000 ms, and it gives the bits of a short slot (`xe_feed`).
- A cache of zeros gives times that are too short by a factor of 3. Measure with values.
- What stays is the softmax and the sum of the values: 140 ms of 208 ms. They are passes over the scores.

**The attention with no score buffer (2026-11-15).** The backbone of LFM3 now runs the attention of a block as
`OP_ATTN_STREAM` and `OP_ATTN_STREAM_SUM` (`kernels/lfm3_ops.cl`). The three ops stay for the other engines, and for
the audio encoder and the codec of LFM3, whose keys have a small bound.

- A team of 4 subgroups takes a chunk of keys, a head and a tile of 16 tokens. It walks the chunk in rounds of 128
  keys. In a round each subgroup scores 32 keys, and the team shares the probabilities of the round. Then each
  subgroup adds them against 32 of the 128 values of V.
- The shared probabilities of a round are 8 KB for each team, 4.5 MB for the card. They stay in the cache of the
  card, so no pass over a score buffer reads memory.
- A team keeps a running max and sum of each token, and it writes them with its output for each chunk. The second
  op merges the chunks in their order. A chunk has at most 8,192 keys.
- The size of a chunk reads the block and its keys, and no slot size. Thus each slot size gives the same bits.

These are the times of one layer and a block of 512, from the same ignored test:

| Keys | The three ops | No score buffer |
|---|---|---|
| 32,768 | 24.8 ms | 19.7 ms |
| 65,536 | 56.1 ms | 38.7 ms |
| 131,072 | 101.6 ms | 76.1 ms |
| 262,144 | 205.0 ms | 147.0 ms |

- On the engine, the longest block of a slot of 262,144 positions takes 1,072 ms where it took 1,330 ms (`xe_feed`).
  At 32,768 positions it takes 206 ms where it took 230 ms.
- The term `work` of 8 slots of 32,768 positions is 135 MB, where the scores alone took 1.07 GB. With a slot of
  262,144 positions it is 345 MB, where the scores took 8.6 GB.
- The sums have another order, so the bits changed, and the identity of each LFM3 engine changed. The logits of the
  five oracle prompts follow the reference as before. The largest gap of the top 64 logits is 2.5e-5.
- A first form failed, and its cause is a rule for the next op. Each subgroup kept the scores of its chunk in a
  buffer of its own, 128 KB. The 2,048 buffers did not fit the cache, and each subgroup read its buffer 4 times.
  The op took 651 ms at 262,144 keys. The arithmetic alone takes about 80 ms of it.
- Thus an op must keep the state that its subgroups read again inside the cache of the card. The team form does
  it: 4 subgroups read one small buffer at the same time, where one subgroup read a large buffer 4 times.

What is open for long slots:

- The profile holds no long slot. The time bound holds and the scores are gone, so the limit is the term `kv`.
  That is 6.5 GB for a slot of 262,144 positions, and 1.74 GB for one of 65,536.
- The team form is not tuned. Its arithmetic takes about 80 ms of the 147 ms at 262,144 keys. The rest is the
  exponentials, the barriers of each round, and the reads of the keys.
- A decode step at 262,144 positions takes 39 ms, where a short one takes 9 ms to 13 ms. One-token attention keeps
  its own form.

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
