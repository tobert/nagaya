# Research: the next passes of Inkling 2 on the GB10

This doc lists the ways to make the GB10 engine of Inkling 2 Mini faster or more able. It is research. Each item is
a proposal until @tobert decides it. A later session takes one item, measures it first, and then builds it.

Decided 2026-11-27: the order of the last section stands, and the drafts of the MTP head (D1) no longer wait. The
engine finds its own design for each item. The GB10 ends at its memory bandwidth, so the items that cut bytes for
each token count most.

The engine is `nagaya-mk-inkling2-sm121`, on the interpreter `nagaya-mk-cuda`. Milestone 37 of `docs/milestones.md`
built it. Its "tuning list" holds the gaps that the card gates found. This doc adds a measure of a prefill block,
the ideas of another engine, and an order.

Each item gives four things:

- the fact, with its source;
- the idea;
- the gain, as an estimate from the fact;
- the first measure, which a session makes before it writes code.

An item also says whether it moves bits. An item that moves bits changes the engine identity.

## The measure of 2026-11-27

The test `cuda_pass_profile` of the engine crate printed these times. The build was a release build of `234eb511`,
with the profile cubin, an `f32` cache and 3 cycles. The context was short: at most 1,024 positions.

| call | time | tokens each second |
|---|---:|---:|
| a decode step of 1 row | 27.3 ms | 36.6 |
| a block of 16 rows | 68.4 ms | 234 |
| a block of 512 rows, at position 512 | 733 ms | 699 |

The profile cubin adds about 0.8 ms to a decode step. The binding gives 26.5 to 26.7 ms with no profile
(`docs/engine-abi.md`, "The pass as built").

The times of each op, in ms for the whole call. A call has 48 runs:

| op | 1 row | 16 rows | 512 rows | share at 512 rows |
|---|---:|---:|---:|---:|
| `OUT_MIX` | 3.35 | 6.98 | 191.9 | 26% |
| `UP`, NVFP4 (44 runs) | 5.18 | 25.04 | 157.9 | 22% |
| `DOWN`, NVFP4 (44 runs) | 3.18 | 11.97 | 128.2 | 17% |
| `PROJ` | 7.69 | 9.69 | 81.4 | 11% |
| `ATTN` | 0.78 | 1.05 | 59.7 | 8% |
| `UP`, MXFP8 (4 runs) | 0.78 | 3.61 | 26.2 | 4% |
| `DOWN`, MXFP8 (4 runs) | 0.44 | 1.71 | 16.7 | 2% |
| the four gate ops | 0.85 | 1.90 | 42.3 | 6% |
| `ROUTE` | 0.71 | 1.16 | 12.3 | 2% |
| `MIX` | 0.22 | 0.44 | 8.9 | 1% |
| `QUANT` | 0.31 | 0.32 | 3.3 | 0.5% |
| `HEAD` (one row of logits) | 4.00 | 4.14 | 4.0 | 0.5% |
| sum | 27.5 | 68.0 | 733.0 | |

What the table says:

- A decode step at 1 row is near its floor. The weights of a step are 5.49 GB, which take 21.8 ms at 252 GB/s. The
  step is at 80% of that rate. No change of an op can give more than 25%.
- A block of 512 rows is far from its floor. The next section gives the floor: about 190 ms.
- The experts take 45% of a block, and `OUT_MIX` takes 26%. These two are the first work of a prefill.

## The prefill

### What another engine does

`erisinky` is an engine of Inkling 2 Mini in Rust and Metal, for Apple silicon
(<https://github.com/Eriskii/erisinky>, read at commit `0e8c39d`). Its license is GPL-3.0-or-later. We read it to
learn, and we copy no code from it.

Its README gives 2,200 tokens each second for a prompt of 2,394 tokens on an M5 Pro. It gives 63 tokens each second
for a decode step. Its weights are Q4_K, about 3.2 GB for each token, and its activations are `fp16`. Thus its decode
number does not compare with ours. Its prefill number does: our block of 512 rows gives 699 tokens each second on a
GPU with more tensor throughput.

The parts of its prefill that matter here:

- **One batch for the prompt.** A block has up to 4,096 tokens. Our largest block has 512 rows (`MAX_ROWS`,
  `crates/nagaya-mk-cuda/src/attn.rs:31`).
- **The experts by tile.** A counting sort on the GPU puts the pairs of a token and an expert in the order of the
  expert. Then each expert runs its tokens as tiles of a GEMM, and the weight tile stays on the chip.
- **Epilogues in registers.** The activation, the route weight and the gates run on the accumulators of the GEMM.
- **A tail on the last layer.** Only the last 8 rows of a block run the attention and the experts of the last run.
  The keys and the values of that run still cover each row.
- **Memory by demand.** The cache of the full runs grows in steps of 64 positions. The activation arena of a prefill
  exists only while a prefill runs.

### The floor of a block

A block of 512 rows picks each of the 128 experts in each run. Thus it reads all the weights once: about 33 GB with
the head, which take 131 ms at 252 GB/s. The bytes do not grow with the rows.

The MMA work grows with the rows. One row needs about 290 MFLOP in each run. The dense matmuls take 161 of them,
with the three terms of design rule 9, and the experts take 127. That is 13.9 GFLOP for each row. The tensor cores
give 126 TFLOP/s in `bf16`, so the MMA bound is about 9,000 rows each second with no attention.

| rows of a block | bytes | MMA | floor | tokens each second |
|---:|---:|---:|---:|---:|
| 512 | 131 ms | 57 ms | about 190 ms | 2,700 |
| 2,048 | 131 ms | 228 ms | about 360 ms | 5,700 |

These floors are estimates from counts, with no attention and no epilogue. A fair target is half of each floor:
1,300 tokens each second at 512 rows, and 2,800 at 2,048 rows.

### P1. The finisher of `OUT_MIX` runs the rows one after another

- Fact: `OUT_MIX` takes 4.0 ms for each run at 512 rows, and 192 ms of a block. One block of threads mixes every
  row, in order (`crates/nagaya-mk-cuda/kernels/ops_out_mix.cuh:22`). The matmul part needs about 0.13 ms.
- Idea: give the rows to all SMs, in chunks. A chunk needs only the 3 rows before it for the state of the
  convolution. The finisher of `DOWN` runs chunks of 64 rows in this way.
- Gain: about 185 ms of 733 ms.
- Bits: no change. Each row keeps its sums.
- First measure: the time of the finisher alone at 512 rows, from the profile clocks of its block.
- Done (2026-11-27): finisher items of 2 rows each, after the 576 matmul items. `OUT_MIX` at 512 rows takes 1.53 ms
  (was 4.01), and a block of 512 rows takes 615 ms (was 735): 832 tokens each second. 1 row is unchanged; no bits
  moved. The tile phase (about 1.4 ms at 512 rows) is the next part: the matmul, the merge of the 4 slices and the
  convolution.

### P2. The experts of a block read each weight tile many times

- Fact: `UP` and `DOWN` take 329 ms of a block. A pair holds at most 16 rows of one expert
  (`crates/nagaya-mk-cuda/src/protocol.rs:207`). At 512 rows an expert has about 32 rows, thus two pairs or more.
  Each pair reads the weights of its expert again. A pair of 9 to 16 rows reads them two times.
- Fact: the bytes of a run take 2.0 ms at the peak rate, for NVFP4. `UP` and `DOWN` take 6.5 ms together.
- Idea: a block form of the expert ops, as the dense matmul has. One item holds one weight tile and loops over each
  row tile of its expert. The weight tile comes from memory once and stays in shared memory.
- Gain: about 200 ms of 733 ms, if the ops reach 90% of the peak rate.
- Bits: no change, if each chain keeps the order of the decode form. The block form of the dense matmul kept it.
- First measure: the count of pairs and the bytes read in a block of 512 rows, against the bytes of the weights.

### P3. `PROJ` in the block form

- Fact: `PROJ` takes 1.7 ms for each run at 512 rows in the pass. The design gave 0.7 ms. The tuning list of
  milestone 37 names the causes: one sync for each stage, and the split in the chain of each warp.
- Idea: the repack by group of the tuning list, so that one bulk copy loads a stage.
- Gain: about 45 ms of 733 ms.
- Bits: no change.

### P4. The small ops at many rows

- Fact: the gate ops, `ROUTE` and `MIX` take 64 ms of a block. `GATE_UP_e` alone takes 0.82 ms for each run.
- Idea: the same as P1. Find the part of each op that one block runs for all rows, and give it to all SMs.
- Gain: about 40 ms of 733 ms.
- Bits: no change.

### P5. A larger block

- Fact: the bytes of a block do not grow with its rows, so a larger block pays them for more rows.
- Idea: raise `MAX_ROWS` to 2,048. The scratch of the attention and the work term grow with the rows. The work term
  is 238 MB at 512 rows.
- Gain: none before P1 and P2. After them, the table of the floor gives about 2 times.
- Bits: no change (design rule 4).
- Limit: a block must end inside `max_quantum_ms`, or it must declare a bound. At depth, a larger block is a longer
  pass. The engine picks the rows of a block from the position.
- First measure: the op table at 1,024 and 2,048 rows after P1 and P2.

### P6. The tail of the last run

- Fact: nothing after the block reads the output of the last run, except its last rows. The convolution states need
  the last 3 rows, and the logits need the last row.
- Idea: this is for a block that gives one row of logits. Run `ATTN`, `OUT_MIX` and the experts of the last run
  for the last 16 rows only. `PROJ` still writes the keys and the values of each row.
- Gain: about 2% of a block, from one run of 48.
- Bits: no change for the rows that run.

### P7. The attention of a block at depth

- Fact: with an `e4m3` cache, a block of 512 rows takes 764 ms at 1,024 positions. At 409,600 positions it takes
  3,815 ms (`docs/engine-abi.md`, "The latency by position and format"). The 8 full runs take the 3,050 ms between
  them: about 380 ms each.
- Fact: the MMA work of one full run of that block is about 7.7 TFLOP, which is about 61 ms at 126 TFLOP/s.
- Fact: 60 to 70% of the time is the latency of one step of 16 keys (milestone 37, the tuning list).
- Idea: stage the keys and the values of a chunk in shared memory with bulk copies. Then a step can hold 64 keys or
  more, and the read of the cache does not wait in the chain of the MMA.
- Idea: let one warp compute the whole score of its rows, with 8 chains. The parts then add in registers in the
  order of today, and the first sync goes.
- Gain: 2 to 3 times at depth. A prefill of 409,000 positions takes 1,853 s today.
- Bits: no change, if the order of the channels and of the chunks stays.
- First measure: the probe of the MMA from shared memory exists (`tests/cuda_probe_prefill.rs`). Add a probe of one
  step of 64 keys with staged operands, and read its registers from `ptxas`.

### P8. The prefill that does not run

- Fact: a restore of a snapshot of 409,000 positions takes 10 s with an `e4m3` cache. The prefill takes 1,853 s
  (`docs/engine-abi.md`, "The retrieval of a fact from early in 400K positions").
- Fact: the restore reads 6.83 GB in those 10 s, which is 0.68 GB/s. An NVMe drive reads faster.
- Idea: measure where the 10 s go: the read of the file, the BLAKE3 check, and the part writes. Then read the file
  into pinned memory in pieces, and hash and write the pieces in parallel.
- Idea: the store holds the state of a context by its recipe (spec 6). A long system prompt or a document that many
  requests share is one entry. Check that the GB10 engine serves such a prefix from the store with no prefill.
- Gain: a stored prefix costs seconds, not minutes.
- Bits: no change. A restore gives the bits of the prefill.

### P9. A prefill in chunks that run apart

- Fact: the trunk uses no RoPE, so a key does not hold its position. A key of a chunk that ran from position 0 can
  go to another position with no change.
- Fact: 40 of the 48 runs read only the 1,024 positions before a row. Only the 8 full runs and the convolution
  states read further back.
- Idea: prefill each chunk of a document alone, from an empty slot, on any card or node and at any time. Then splice
  the chunks: put the keys and values of the full runs of each chunk into one slot, at the chunk's positions. Two
  forms:
  - (a) no repair;
  - (b) a repair: run the first 1,024 positions of each chunk again with the true context before them, so that its
    sliding rings and convolution states are right.
- Gain: a prefill that is parallel across nodes and runs as text arrives. A stored chunk loads with no prefill (P8).
  A document that many contexts share is prefilled once.
- Bits: they move against a straight prefill. The chunks are part of the recipe (spec 6), so the same chunks give
  the same bits: a spliced context is a new recipe with its own identity, not a fallback.
- First measure: a PG-19 text of 32K positions, straight and in chunks of 4K and 8K, forms (a) and (b). Compare the
  top token and the KL of the next 256 tokens against the straight prefill, and the needles of L4.

## The decode step at short context

The step is at 27.3 ms against a floor of 21.8 ms. The items below are in the order of their gain.

### D1. Drafts, checked as rows

- Fact: a block of 16 rows costs 2.5 times a step of 1 row. Thus 8 more rows cost less than 1 more step.
- Fact: design rule 4 gives each row the bits of its decode step. Thus a checked draft is exact: an accepted token
  has the bits of the plain decode.
- Idea: draft `K` tokens, run the last token and the drafts as one block, and keep the drafts that the model
  confirms. Two sources of drafts exist:
  - a lookup in the context: the tokens that followed the last match of the newest tokens. It needs no model.
  - the MTP head of the checkpoint, which drafts blocks of 8 tokens. "The draft head" below gives its facts.
- Gain: with `a` accepted drafts for each block, the speed grows by `(a + 1) / cost(K + 1)`. At `a` of 3 and a cost
  of 1.8, that is 2.2 times. This is the only item that can pass the floor of the bytes.
- Open: a refused draft wrote its keys, its values and its convolution states. The slot must go back. The sliding
  ring has spare rows for a block, so the keys of the window survive. Check this before the design.
- Open: with a sampler, a draft is accepted when it equals the sampled token. The rate falls with the temperature.
- First measure: on the host, with texts that the engine wrote. Count the accepted tokens of the lookup for `K` of
  2, 4 and 8. Then measure `cost(K + 1)` on the GPU for 2, 4, 8 and 16 rows (milestone 37 step 12 measures this).

#### The draft head

The facts come from `mtp_config` of the checkpoint, from the header of `mtp.safetensors`, and from the path of vLLM
(`vllm/models/inkling2/nvidia/dspark.py` and `configs.py`, Apache-2.0). vLLM names the head DSpark.

- The file `mtp.safetensors` holds 54 tensors in `bf16`, 636 MB in all. A draft reads them once: about 2.5 ms at
  the peak rate.
- The head is a dense transformer of 3 layers, with a width of 2,304 and an MLP of 9,216. Layers 0 and 2 have a
  sliding window of 1,024, and layer 1 reads each position. It uses RoPE, and the trunk does not.
- A block is one anchor token and 7 mask tokens. Each mask predicts the token at its own position. Thus a draft has
  at most 7 tokens, and the head runs once for a block, not once for each token.
- The head reads features of the trunk at the layer visits 5, 23 and 41, and the output of the final norm (48). A
  matrix of 4 by 4 (`fuse_weight`) mixes them. The head has its own key and value projections for this context
  (`k_proj_ctx`, `v_proj_ctx`), so a slot holds a draft cache too.
- A Markov head follows: an RNN state of rank 256 and a table of 201,024 rows of 256 values (103 MB). Each token of
  the draft depends on the one before it.
- The visits are fixed, so the head works only at 3 cycles. A context with another count of cycles gets no drafts.
- The reference of transformers that our oracle uses holds no code for the head. The path of vLLM is the only
  source of its rules.

The cost of a block is an estimate. The head and its Markov table take about 3 ms. The base logits of the 7 masks
share one read of the head matrix, about 4 ms. The check of 8 rows costs about 1.8 steps. That is about 56 ms in
all. With 3 accepted drafts, 4 tokens take 56 ms: about 71 tokens each second.

The steps of D1, in order:

1. The lookup drafts on the host, as the first measure above. This gives the acceptance of drafts that cost nothing.
2. An oracle of the head. Run the path of vLLM on the CPU for the golden prompts, and record the features, the
   draft ids and the logits. Measure the count of accepted drafts for each block on the texts of step 1.
3. Go or stop: build the head only if step 2 shows a clear gain over the lookup.
4. The rule of the slot after a refused draft (the open question above), in the binding, with the mark Proposed.
5. The head as ops of the megakernel. Its matmuls are the dense matmul that exists. Its attention with RoPE is new.
6. The draft cache as a part of the slot, with its bytes in the plan and its place in a snapshot.

### D2. Fewer bytes in the dense weights

- Fact: the dense weights are 2.79 GB of the 5.49 GB of a step, in `bf16`. The head is 0.92 GB.
- Idea: quantize the dense weights to 8 bits with a block scale, as MXFP8. A step then reads about 4.1 GB.
- Gain: about 5.5 ms of a step, thus about 46 tokens each second. A format of 4 bits gives about 52.
- Bits: they move. This is the "custom quant" that the plan of milestone 37 puts last. It needs the gate of
  coherence, and a measure of quality that is wider than 15 prompts.
- First measure: on the host, the gap of the logits with the dense weights in 8 bits, by the reference.

### D3. The tails of `UP` and `DOWN`, and the small ops

- Fact: `DOWN` of NVFP4 reads at 66% of the peak rate and `UP` at 80%. `GATE_UP`, the two `_e` gates and `ROUTE`
  read at 16 to 48%. `OUT_MIX` reads at 81%.
- Gain: about 3 ms of a step if each reaches 90%. The tuning list of milestone 37 holds these items.
- Bits: no change.

### D4. The sampler on the device

- Fact: the host reads 800 KB of logits for each step, which takes most of 0.5 ms.
- Idea: a greedy step takes the top token on the device and feeds it to the next step. Two steps can then be in
  flight, with no wait on the host. A sampled step reads only the top `k` logits.
- Gain: 0.5 ms of a step, about 2%.
- Bits: no change for a greedy step.

### D5. The head in two stages

- Fact: the head takes 4.0 ms, 15% of a step.
- Idea: keep a copy of the head in 8 bits (0.46 GB). Score each row with it, and then score the best rows again with
  the `bf16` head. A bound on the error of the first stage tells which rows the second stage must read.
- Gain: about 1.9 ms of a step, when the sampler needs only the top rows.
- Bits: no change for the rows of the second stage. The sampler must not read the others.
- First measure: the count of rows inside the error bound, on the logits of the golden prompts.

## The decode step at depth

At depth the cache sets the speed. A step at 409,600 positions takes 88 ms with an `e4m3` cache, against 30 ms at
1,024 positions.

### L1. The read of the `e4m3` cache

- Fact: the 8 full runs read 6.7 GB of keys and values in 58 ms. That is 116 GB/s, 46% of the peak rate.
- Idea: the staged step of P7. A decode step has only 3 rows for each KV head, so the MMA work is small. The read
  must run as a stream, as the weight reads do.
- Gain: at 90% of the peak rate, the step takes about 60 ms: 16.7 tokens each second, from 11.4.
- Bits: no change.

### L2. Fewer keys for each step

- Fact: `erisinky` holds experiments that score the far keys of the full runs and keep a share of them. Its README
  gives no result for them.
- Fact: our retrieval at 409,000 positions mixes up needles in each format, and in `f32` too.
- Idea: measure how peaked the attention of the full runs is. If a few chunks hold nearly all the weight, a first
  stage can pick the chunks, and the second stage reads only those.
- Gain: not known. It can be large, because the step is bound by the bytes of the cache.
- Bits: they move, and retrieval can break. This item is an experiment for the gate of retrieval.
- First measure: for each head of the full runs, the share of the chunks that hold 99% and 99.9% of the softmax
  weight. Restore a snapshot of about 409,000 positions (no prefill), and run some decode steps on a variant cubin
  that writes the weight of each chunk of keys.

### L3. A format for the keys and another for the values

- Fact: the keys set the softmax, and the values only add. The formats table of milestone 37 says that 4-bit keys
  move the softmax.
- Idea: keep the keys in `e4m3` and try the values in 4 bits, or keep `bf16` keys with `e4m3` values.
- Gain: up to 25% of the cache bytes, or a better retrieval at the same bytes.
- Bits: they move. It needs the gates of coherence and retrieval.

### L4. More needles for the gate of retrieval

- Fact: 14 needles do not rank the formats (milestone 37, step 11).
- Idea: one context can hold many more needles for the cost of one prefill. Ask each question from the same
  snapshot.
- Gain: a rate with an error bar, so that L2 and L3 have a gate.

### L5. A cache in tiers

- Fact: a decode step at 409,600 positions reads 6.7 GB of keys and values (L1). The bytes grow with the position.
- Idea: if L2 shows a peaked attention, keep a small summary of the keys of each chunk on the card (a few bits), and
  keep the chunks themselves in a compact format, part of them on NVMe. A step scores the summaries, then reads only
  the chunks it picks, in a finer format.
- Gain: a cache that is not bound by the RAM of the card, and fewer bytes for each step. Not known before L2.
- Bits: they move. It needs the gates of coherence and retrieval (L4).
- First measure: the measure of L2. Then the read rate of NVMe for chunks of 64 to 1,024 positions.

## Memory

### M1. Slots that grow

- Fact: the engine declares the cache of every slot as one mandatory term (`src/spec.rs:462` of the engine crate).
  The profile of the DGX Spark example has 4 slots of 131,072 positions: 9.1 GB, also for a chat of 100 tokens.
- Fact: a position costs 16 KB in the 8 full runs with an `e4m3` cache. The 40 sliding rings have a fixed size.
- Fact: the spec has the mechanism. An elastic term has a unit, a minimum and a maximum, and it changes size only
  between quanta (spec 4.2, 4.4, 4.5). The bits of this engine do not depend on the size of a slot.
- Idea: make the full planes of a slot an elastic term, with a unit of some thousand positions. The attention
  reads the cache in chunks of keys, so a plane can be a table of chunks.
- Gain: more slots in the same budget. The rows of milestone 37 step 12 need one slot for each row, and 16 slots of
  131,072 positions take 36 GB today.
- Bits: no change.
- This is not particular to the model. Each engine with a cache that grows by position can declare it this way.
- First measure: whether the driver commits the pages of an allocation at once on this GPU. Read `MemAvailable`
  before and after an open. If the pages come at the first touch, only the ledger needs the change.

### M2. The scratch of a prefill

- Fact: the spec lets a term change its bytes by the kind of quantum (spec 4.2). The work term is small today.
- Idea: when P5 raises the block, declare the scratch of a prefill as its own term, so that a decode does not hold
  it.

### M3. Touch each weight page at the load

- Fact: `erisinky` reads each page of the weights from the GPU at load. It reports a slow first read of an expert
  that no token picked before.
- First measure: the time of the first decode steps after an open against later steps, by expert.

## Intelligence

These items change what the model can do, not how fast a step runs.

### I1. The cycles as a dial

- Fact: the engine runs 1 to 5 cycles, and a context picks the count. A step at 5 cycles takes 44 ms at short
  context, against 30 ms at 3 (the latency table of the binding). A step at 1 cycle runs 24 runs, not 48.
- Idea: measure the quality at each count of cycles on one set of tasks. Then map the effort of a request to a
  count of cycles.
- Open: the model card and the varloop research model say what the cycles are trained for. Read them first.
- First measure: the loss of the next token on PG-19 text at 1 to 5 cycles, on the GPU. It needs no new code.

### I2. Many rows for one answer

- Fact: 16 rows give 234 tokens each second in all, 6.4 times the rate of 1 row.
- Idea: run several samples of one question as rows, and let a vote or a judge pick. `docs/swarm-ideas.md` holds
  such ideas for another engine. They need M1, because each sample is a slot, and a fork of a slot copies its cache.

### I3. The parts that are not built

- The vision tower, the taps and the control vectors are step 13 of milestone 37.
- The MTP head is D1. It no longer waits (2026-11-27).

## The order

1. P1, then P2: the two largest parts of a block. They move no bits, and the profile test shows each gain.
2. D1: steps 1 and 2 of "The draft head", then the go or stop of step 3. They need no kernel work, so they run
   beside item 1.
3. P7 and L1 together: one staged step serves the prefill and the decode at depth.
4. M1, with the rows of step 12.
5. P3 to P6 and D3 to D5: small gains, in any order.
6. I1 and L4: measures that need GPU time and no kernel work. They can run between the lanes.
7. D2, L2 and L3 last: they move bits, and they need the wider gates of L4.
8. Later: P9 and L5. They change how a context is made and kept, not how a step runs. The first measure of L2 tells
   whether L5 can pay. P8 comes first, because P9 loads its chunks by restore.

## The tools

- `cuda_pass_profile` prints the op table for 1, 16 and 512 rows. Its header gives the command. Run it in a release
  build, with `nagaya.service` stopped.
- `cuda_latency` gives the times by position and format. `cuda_retrieval` gives the retrieval and the snapshots.
- `cuda_probe_prefill` holds the probes of the block form.
