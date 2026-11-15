# 長屋 nagaya

A *nagaya* (長屋（ながや）) is a row house: one building, several households, each with its own door. This nagaya is
the one process on each machine that owns the GPUs. Engines live in it, and each engine has its own door.

The nagaya is three things:

- **An exokernel for megakernels.** The spec (`docs/spec.md`) and the engine ABI (`docs/engine-abi.md`) say how one
  process divides a machine among its engines. The parts are device memory, host memory, NVMe, and time on each
  device. The nagaya decides where state lives, what runs next on each device, and what the store keeps. It never
  decides what a model computes. Each engine keeps its own abstractions: its key/value layout, its recurrent state,
  its kernels.
- **A reference implementation.** The Cargo workspace in `crates/` is that process in Rust, with six megakernel
  engines. Qwen3.8 and Qwen-Image 2.1 run on one AMD gfx1151 (`nagaya-mk-qwen38-gfx1151`,
  `nagaya-mk-qwenimage21-gfx1151`). Gemma 4, Qwen3-TTS, SDXL and EmbeddingGemma 2 run on two Intel Arc Pro B70 cards
  (the `nagaya-mk-*-bmg` crates). `AGENTS.md` gives the layout, the rules and the milestones.
- **The council v2 API.** Clients speak council v2 to the nagaya. The contract is gogi, and the nagaya serves it for
  every engine of the cluster (`docs/spec.md` 11). Its own routes are under `/nagaya/v1`.

An engine is a model's code and weights. The first two engines came from our learning projects, the first Qwen3.8 engine
and mkviolet. `docs/ontology.md` defines the first Qwen3.8 engine, mkviolet, gogi and kaijutsu once, for every doc.
Later engines are lfm and gmk's instruments.

v0 covers three kinds of work: inference, image batch, and training as filler. Amy chose all three for v0 on
2026-10-17.

## Status

The design came on 2026-10-17 from a conversation with Amy. It began with renting a GPU for a weekend and ended with
a common host for our megakernels. The code began with the first Qwen3.8 engine on 2026-10-18.

- Since 00:24 on 2026-10-21, `nagaya.service` serves council v2 on `198.51.100.20:8090` on gfx1151host's gfx1151.
  Since 2026-10-23, the engine of Qwen3.8 is the nagaya's own copy. Since
  2026-11-01, `qwen-image-fusal` is resident, and the Qwen3.8 engine is on demand (`docs/deploy.md`, "The layout of
  gfx1151host").
- Since 11:01 on 2026-10-23, `nagaya.service` serves council v2 on `198.51.100.10:8091` on b70host's two B70 cards.
  Since 2026-11-15 `autismmix-1`, `embeddinggemma-2` and `lfm3-omni` are resident on card 1, and `lfm3` on card 0.
  `gemma` is on demand (`deploy/b70host/profile.toml`).
- Milestones 1 to 20 and 22 are done (`docs/history.md`, `AGENTS.md`). Milestone 21 waits, and milestone 23 is next.
- `docs/deploy.md` tells how to run the nagaya on each machine.

## Why a new repo

Amy, 2026-10-17: "tempted to expand mkviolet but it was made to be just for those cards for us to learn, I am
tempted to come up with a new name and port all the code into a fresh repo and clean as we go."

- mkviolet is the seed. It is Rust, and one process owns both B70 cards. It served council v2 and kept its weights
  alive across restarts through systemd's file-descriptor store. Its engine is now `nagaya-mk-gemma4-bmg`
  (milestone 15s): copied in, and cleaned as it moved.
- The first Qwen3.8 engine joined first, through its C ABI. Milestone 15 copied its engine in, HIP kernels and all. Its
  host code is Rust now, so the nagaya owns both engines (Amy, 2026-10-19: "copy code in, we own all of it 100%").
- One nagaya process per machine (Amy: "we'd do 1 process per host that owns the gpus").
- Engines we build give the same bits for the same identity and inputs: `always` or `pinned` (`docs/engine-abi.md`
  D3). Engines we borrow declare what they promise instead.

Milestones 15 and 15s are done, so the first Qwen3.8 engine and mkviolet are archives (Amy, 2026-10-22: "they were our
learning projects for this and their megakernels live on"). The docs name them only as the source of a copy or of a
measurement.

## Documents

- `docs/spec.md`: the spec. Read it after this file.
- `docs/ontology.md`: the nouns, defined once, and the four other projects that the docs name.
- `docs/engine-abi.md`: the engine ABI v0, how the Qwen3.8 engine and mkviolet map onto it, the image, embedding and
  speech calls, and its decisions D1-D11.
- `docs/deploy.md`: how to build the nagaya, install its unit on gfx1151host and on b70host, cut over, and roll back.
- `docs/history.md`: the finished milestones, with what each one built, and the merges where the run recorded them.
- `docs/new-engine.md`: the checklist for the next engine, family, card or machine, from the first two ports.
- `docs/reviews.md`: reviews of these documents and what we did with each finding.
- `docs/prior-art.md`: what ds4-recipe and Ai2's scheduler teach us, as proposals.
- `docs/deepinfer.md`: what DeepSeek's DeepInfer teaches us, as proposals.
- `docs/cv.md`: control vectors that change the residual while a model runs. Milestone 25 builds them.
- `docs/lab.md`: tools that look inside a model: a vector through a matrix, a trace of a generation, the J-Lens.

## Later (doors to keep open, nothing to build in the first pass)

Amy, 2026-10-17: "could we down the road do some dynamic model swapping and say, reconfigure the fleet overnight to
focus on some task that fits on all 3 machines in the lab in a particular way, but be in multi-model during the day
when we're coding and interacting more? ... would need more orchestration and cluster membership than I want to
build in the first pass."

Machines (Amy, 2026-10-17): the B70 cards are "the fastest memory we have at home"; "We have the dgx spark for 128GB
and strix halo for 128GB bigger models where we maybe don't care about speed as much as overall capability for
certain tasks." Today's Gemma 4 needs both B70 cards; one card would take a smaller Gemma, "or probably a qwen and
lfm".

Between machines (Amy, 2026-10-17): "gigabit rn, could be 10g with a little work ... so not dma, would need to be
mostly control traffic or final product movement. so the moe idea probably isn't right. probably whole model on the
gpu and we look more at how to combine different models to achieve more, like the kaijutsu bump".

A model stays on one machine. What crosses the network is jobs, their answers, and now and then a snapshot by
content id.

A split of one MoE's experts across machines would move about 123 MB per 256-token pass. That is 48 round trips
of 10 KB per token per layer. A pipeline split would move about 5 MB once per pass. At gigabit, neither
split is practical. Neither is worth it while gfx1151host and sm121host each hold 128 GB.

The first pass keeps three doors open:

1. Engines load and unload while the nagaya runs. Weight arenas are file descriptors that we can export (the
   arena of the Qwen3.8 engine, mkviolet's DMA-BUF slabs). Thus with an arena staged in host memory or on NVMe, a swap
   is a copy between tiers. Milestone 16 built the swap in place, with a cold load (`docs/spec.md` 3.3a). An arena
   staged for a swap is not built.
2. Placement is live cluster data. The nagaya processes gossip their probed hardware, active profiles, and loaded
   engine identities through `chitchat` (`docs/spec.md` 2.3). Any machine can receive a request, tokenize it locally,
   and coordinate or proxy it (`docs/spec.md` 11.1).
3. State objects (weights, contexts, snapshots) are canonical safetensors with BLAKE3 content ids. Later, plain HTTP
   (`/nagaya/v1/cas/{content_id}`) can serve them between machines that run the same engine identity.

A model spread across machines over Ethernet means pipeline stages per machine. The Gemma 4 engine splits its
layers across its two cards, and it hands the residual between them through host memory. Thus a stage boundary is a
residual at a layer. `docs/engine-abi.md` D1 exposes stages, and the scheduler runs each stage on its card. A split
across machines would use the same boundary.

lfm (Amy, 2026-10-17): "we'd add lfm over time and likely deprecate lfm2d". lfm2d is candle-based on a fork of
candle; it may stay as a second council API implementation. A new lfm engine (perhaps LFM 2.5 on one Intel card)
comes when we're ready for another model or need one to validate against.

Other later items:

- a model router (lfm or council decisions);
- rented burst machines that join as leases;
- streams (home video, audio for the cochlea, beat tracking).
