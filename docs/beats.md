# Beat ideas

This doc holds ideas, not rules. No milestone builds them yet. It started on 2026-11-14 (Claude Fable 5.2), from one
question. Can a musical clock drive a part of a pass, or change an input of it by a small amount?

## The terms

- **Beat.** One quarter note of an external clock.
- **Pulse.** One tick of a MIDI clock. A MIDI clock sends 24 pulses for each beat. At 120 beats each minute, a pulse
  is 20.8 ms.
- **Phase.** The place in the beat, from 0 to 1.
- **Nudge.** A small change of one input of a pass, as a function of the phase.

## Where a clock is today

- The megakernel of the B70 cards has no clock. Its work-groups walk the task ring, and they wait on the scoreboard
  between two tasks (`crates/nagaya-mk-bmg/kernels/megakernel.cl`).
- The Qwen3.8 engine reads a device counter of 100 MHz. It bounds each wait at a barrier, and it stamps the events
  (`hip/src/rt/device.h`, `hip/src/engine/mega.hip`).
- The host owns the sequence. A quantum runs to its end, and the nagaya changes what runs only between quanta
  (`docs/ontology.md`). The scheduler takes the time as an argument (`crates/nagaya-sched`).
- A speech frame of LFM3 Omni is 1,920 samples (`codec.rs`: a stack of 6, a hop of 320). Thus speech is the one output
  with a fixed grid of time.

Thus the place for a clock is the quantum boundary on the host. Do not hold a kernel in a wait until the beat. A
queue that waits cannot be preempted on gfx1151, and a long wait once reset the GPU.

## Two layers

Keep these two layers apart.

- **Timing** changes when a quantum starts. It changes no bits.
- **Modulation** changes bits. Thus the phase is a declared input. A request or a response records the phase of each
  position. Then a replay gives the same bits with no clock.

Do not gate each quantum on a pulse. Read the phase at the start of each quantum. A gate costs up to one pulse for
each token, and a read costs nothing.

## The entry points of a pass

A pass is a function of fixed weights and five inputs. A nudge enters through one of them.

| Entry point | What it is | Where a nudge lasts |
|---|---|---|
| The input row | the embedding row of the last token, a soft row, or a latent row | in each layer of the position |
| The position | the angle of the rotation of the queries and the keys | see "A position offset" |
| The KV cache | the keys and the values of the earlier positions | until the context ends |
| The control list | `add`, `ablate` and `clamp` on the residual between two layers | in each layer above |
| The draw | the uniform value that picks a token from the logits | in the token |

A seed is not an entry point for a small nudge. Each draw is a hash of the seed, the position and the purpose
(`crates/nagaya-council/src/sampler.rs`). Thus a seed that differs by one bit gives a draw with no relation to the
first. Nudge the uniform value after the hash: `pick_with` takes it as an argument.

## A position offset

The rotation of the megakernel is a bank of oscillators, and the position is its clock. Pair `i` of a head turns
`theta^(-2i/K)` radians for each position. The fastest pair turns 1 radian, and the slowest turns about `1/theta`.

The kernel turns by a float, and only its source is an integer (`OP_RMSNORM` with `FLAG_ROPE`). The row of the KV
cache comes from the integer position, apart from the angle. In a decode step, the query and the key of a row are two
tasks (`crates/nagaya-mk-lfm3-bmg/src/model.rs`). Thus three nudges are possible:

| Offset on | What it means |
|---|---|
| the query | the token looks back from a place where it is not |
| the key | the context remembers the token at a place where it was not |
| both | the token is at a position between two whole positions |

- An offset on the query changes no key of its own layer. But it changes the residual, so the layers above write
  other keys and values. No offset leaves the KV cache clean.
- The same offset on each position changes almost nothing, because attention reads the difference of two angles. This
  is the first gate of a trial, with "an offset of 0 gives equal bits".
- The position is `f32`. Near position 1,000, its step is about 6e-5. A smaller offset needs a second rotation with
  its own sine and cosine.
- A small offset turns only the fast pairs. An offset of 0.01 turns the fastest pair by 0.6 degrees.
- An offset is outside what the model saw in training. That is the point of the trial, not a defect of it.
- SwarmEvolve rotates the keys of a block to another offset (`docs/swarm-ideas.md`). It is prior art for a key offset.
- The kernel source changes. Thus the engine identity changes, and each held context fills again once.
- LFM3 is part convolution. Only its attention layers read the position.

## The latent step as the place to listen

In a latent step, the host holds the next input row before it feeds it (`docs/lab.md`, "Latent steps in a request").
Two facts make it the best place for a nudge:

- A change of the row is host code. It needs no change of a kernel.
- A token step puts the logits into one token, so a small nudge stays only if it changes the token. A latent step
  feeds the whole row, so a small nudge stays in it.

The audio encoder of LFM3 Omni makes rows for the input of the backbone, one for each 8 mel frames. Thus the next
input of a latent step can be the latent row and a small share of a row of live audio.

## Sound out

The sampler of the audio codes runs on the host (`crates/nagaya-mk-lfm3-bmg/src/codes.rs`). Most of the output side
is ours to change:

- **The sampler.** The temperature is 0.4, and the penalty is 1.2 over 64 frames. They are the defaults of the
  reference, as constants. A request member for each is a small change.
- **The feed.** The feed of a frame and the decode of a frame are two calls. Thus the model can read one frame while
  the codec decodes another.
- **The code.** A code is 4 integers with the levels 8, 7, 6 and 6, so a codebook has 2,016 codes. Two codes that
  differ by 1 in one level are near in sound. A nudge of one level stays inside what the codec knows.
- **The latents.** A frame is 48 values. A trial can write values between two levels there.
- **The synthesis.** The overlap-add runs on the host, with an FFT size of 1,280 and a hop of 320. Another hop or
  another window changes the length and the timbre.

The frame number is a counter of the fixed grid. Each draw of a frame is a hash of the seed and the frame number.

## Other nudges

- **The strength of a control vector.** It is a free `f32`, with a strength for each layer. The phase can drive it as
  a slow oscillator.
- **The mix of a soft row.** `beta` of `moi` and `tau` of `gumbel`: whole tokens on the beat, mixed rows between two
  beats.
- **The count of latent steps.** A rest: the context takes latent steps for one bar before it replies.
- **An image step.** The sigma, the guidance scale and the noise of each step. One step for each beat shows the image
  as it forms.
- **The order of ready quanta.** Each fork of a swarm is a voice, and a joined pass is a chord.

## A swarm with a beat

A swarm already has a pulse. The decode quanta of its members join in one pass, so each member takes one step at the
same time (`docs/swarm-ideas.md`, "Batched passes on the LFM3 engine"). An external clock gives that pulse a phase.

`ensemble` is the strongest coupling: each member takes the same token (`docs/lab.md`, "`ensemble`: joined forks"). A
dance is a weak coupling. Each member keeps its own token, and it reads a small share of the other members. These are
the places for that share, from the smallest change of code to the largest:

| Place | The coupling | What exists |
|---|---|---|
| The logits | a member's row moves a share `e` toward the mean row of the swarm | the rows of `ensemble`, on the host |
| The input row | the next input is the member's token row and a share of the other token rows | soft rows |
| The latent row | the next latent row moves a share toward the mean row of the swarm | `latent` and its `mean` |
| The residual | the residual of member A at layer `l` is an `add` on member B at layer `l` | taps and control lists |
| The KV cache | a member reads the last bar of another member | `merge` |

- **The share follows the phase.** A large share on the beat and a small share between two beats.
- **A share below 0** moves a member away from the swarm. `contrast` has this form for 2 contexts.
- **A canon.** Member `k` has the position offset `k * d`. Each member reads the same context, a little later.
- **Two voices at one time.** A merge can turn the keys of another member to the positions of the member's own last
  bar. Then two rows are at each position. No model saw this in training.
- **Leader and follower.** With a coupling, the order of the members in a step has a meaning: who reads whom first.
- **A pop.** At some steps, one member moves its latent row away from the mean row by one standard deviation of the
  swarm. It is the opposite of the coupling, and it keeps the members apart.
- **A little noise over time.** A small random share in each coupling, from a hash of the seed, the member and the
  step. Thus the dance does not repeat, and a replay gives the same bits.

Limits of the code today: a soft row shares no pass, and a pass with a tap has one segment. Thus only the coupling
at the logits keeps the joined pass with no change of an engine.

The engines run forward only, so "teach" has three meanings here:

- **The coupling alone.** Weak coupling makes oscillators lock with no teacher. Nobody knows if contexts do this.
- **Selection.** The select step of a swarm keeps the members with the best score, and the score can be a measure
  of the dance.
- **A control vector.** Fit a direction from contexts that follow a partner and contexts that do not.

The measure of a dance needs a number for each member at each step. The entropy of the step is the simplest, and the
norm of a middle layer is another. The members dance when these numbers rise and fall together at the period of the
beat. The test of a lesson: remove the coupling, and count the steps for which the rhythm stays.

The first trial: 8 members on `lfm3`, a coupling at the logits, and a share that follows the phase. With a share of
0, each member gives the bits of its own generation.

## The speed of a swarm

Measured on 2026-11-14 on `lfm3`, on one B70 card, through the route of generations with `forks`. Each member wrote
224 tokens. The rate is the difference of a run of 224 tokens and a run of 32 tokens.

| Members | Steps each second | One step | Tokens each second |
|---|---|---|---|
| 1 | 92.1 | 10.9 ms | 92 |
| 2 | 67.3 | 14.9 ms | 135 |
| 4 | 43.7 | 22.9 ms | 175 |
| 8 | 22.0 | 45.5 ms | 176 |

8 members give no more tokens each second than 4. Nobody found the cause yet.

The fastest beat follows from the step. The table gives it for one step on each part of a beat, and 60% of it:

| Members | One step on each | Fastest beat | 60% of it |
|---|---|---|---|
| 8 | 16th note (4 for each beat) | 330 beats each minute | 198 |
| 8 | 32nd note (8 for each beat) | 165 | 99 |
| 8 | pulse (24 for each beat) | 55 | 33 |
| 4 | 32nd note | 328 | 197 |
| 4 | pulse | 109 | 66 |

At 60%, a card works for 45.5 ms and rests for 30 ms in each step of 8 members.

**A rest and the power state.** The file `act_freq` of a card reads 2,800 in a pass, and 0 from 20 ms after it. The
table gives the time of a request of 8 members and 6 steps after a rest, as the median of 7 runs:

| Rest | The request | More than no rest |
|---|---|---|
| 0 ms | 362 ms | |
| 50 ms to 500 ms | 366 ms to 369 ms | 4 ms to 7 ms |
| 1,000 ms | 372 ms | 10 ms |
| 3,000 ms | 378 ms | 16 ms |

Thus a rest costs the next step a few milliseconds, and a longer rest costs more. A beat that needs a step within
1 ms of its time must start the step early by this cost, or keep the card awake.

**To see the beat.** A rest of 30 ms in each step does not show in a monitor that samples each second. Two ways show
it. Sample `act_freq` each 5 ms, because it reads 0 in each rest of 20 ms or more. Or put the steps in groups. Then
some parts of each beat have a step, and the other parts are a rest of 200 ms or more.

**What it needs.** A gate of the host at the quantum boundary, which is not built. A client can pace only whole
requests, and the 6 steps above took 362 ms where 6 steps of a long run take 273 ms.

## A plan at 120 beats each minute

The owner chose 4 members and 120 beats each minute on 2026-11-14. The goal is not speed. It is to learn if a
context can take a small bias from other players, as a performer does. A weak effect is enough at first.

**The grid is the audio row.** LFM3 Omni has one grid of time in both directions:

- Audio in is 16 kHz with a hop of 160 samples, and one position is 8 mel frames. Thus one audio row is 80 ms
  (`crates/nagaya-text-lfm3/src/media/audio.rs`).
- Audio out is 1,920 samples at 24 kHz for each frame. Thus one frame is 80 ms too (`docs/engine-abi.md`, "The speech
  calls of LFM3").

A step of 4 members is 22.9 ms. Two steps for each audio row give a step each 40 ms, and a card works 57% of the time.
At 120 beats each minute, a bar of 4 beats is 2,000 ms: 25 audio rows and 50 steps. A beat is 6.25 rows, so only
the bar is on the grid. At 125 beats each minute, a beat is 6 rows and 12 steps.

**The quantum boundary.** A device thread takes one quantum from its channel, runs it, and sends the result to the
group thread (`crates/nagaya/src/process/staged.rs`). The group thread then asks the scheduler for the next quantum.
Between the result and the next quantum, no kernel runs, and the host holds each row. The gate goes there. The
simplest form is a rule of the scheduler: a job with a clock is not ready before the time of its next step. The
scheduler takes the time as an argument, so a test with no GPU checks the rule.

**The host is the clock, and the card holds the score.** The megakernel of the B70 cards has no clock, so it cannot
find the time of a step. A kernel that waits for a signal of the host keeps the card at work in each rest. Thus the
host starts each step. But the data of a bar can be in device memory before the bar starts. That data is the audio
rows, and the share and the offset of each step. Then a step sends no row to the card.

**The buffer.** A lab route takes audio and a place on the grid. The audio encoder makes its rows, and a ring holds
them by their row number. A step at row `r` reads a small share of a row of the ring.

- A background track comes before its time. Thus a step at row `r` can read row `r + 1`. This is the anticipation
  of a performer who knows the piece.
- A live player comes after its time, by one row and the time of the encoder at least. A step then reads the past.
- The frames that the swarm speaks are on the same grid. Thus they can go into the ring for the other players.

**The control of a trial.** Feed the same rows in a random order. They have the same sound and no timing. An effect
that stays with the random order is not an effect of the beat.

Open for this plan:

- The rows of the audio encoder are inputs of `lfm3-omni`. Nobody checked that `lfm3`, the text model, reads them.
- Nobody checked that `lfm3-omni` joins the decode quanta of 4 members, or its step time.
- The time of the audio encoder for one row is not measured.
- Other jobs can run in a rest. Then a step starts late by up to one quantum of the other job.

## The readout

A trace with the J-Lens shows a nudge before it changes a token (`docs/lab.md`, "`trace`"). One trace of `lfm3` on
2026-11-14 had 10 steps and 24 layers. Its top entries fall into three bands:

- Layers 2 to 8: the word that the position read, and words near it.
- Layers 10 to 16: the layout that the reply can take, such as a heading or a list.
- Layers 18 to 23: the token of the reply, and tokens near it.

The trial of a nudge runs the same trace two times, plain and with the nudge. It prints only the rows that differ.

## The first trial

A position offset on the query, in decode steps only, on `lfm3`. Change the offset from 0 to a few positions, and
read the J-Lens trace at each value. Then make the offset a function of the phase.

## Open questions

- The `trace` route serves the text passes of `lfm3-omni`. Does it take a pass with an audio part?
- `lfm3-omni` has no J-Lens. The lens of `lfm3` is of other weights.
- Only the text model declares `control`. Thus no control list runs on a pass of Omni.
- Does one request take audio and give speech in turns, or does it take one clip and then speak?
- The rotation of the Qwen3.8 engine is in its HIP code. Nobody read it for this doc.
- An ablation of the layout direction of the middle band is a separate idea, for later.
