# 長屋 nagaya

A *nagaya* (長屋（ながや）) is a row house: one building, several households, each with its own door. This nagaya is
the one process on each machine that owns the GPUs. Engines live in it, and each engine has its own door.

An engine is a model's code and weights. The first engines are mk's Qwen engine (`libmkengine`) and mkviolet's
Gemma 4 engine. Later engines are lfm and gmk's instruments. The nagaya divides the machine among its engines:
device memory, host memory, NVMe, and time on each device. Each engine keeps its own abstractions: its key/value
layout, its recurrent state, its kernels.

v0 covers three kinds of work: inference, image batch, and training as filler. Amy chose all three for v0 on
2026-10-17.

Clients speak council v2 to the nagaya. The contract is gogi (`~/src/gogi`). The nagaya speaks
the engine ABI (`docs/engine-abi.md`) to its engines.

Status: design only. Written 2026-10-17 from a conversation with Amy that began with renting a GPU for a weekend and
ended with a common host for our megakernels. No code yet.

## Why a new repo

Amy, 2026-10-17: "tempted to expand mkviolet but it was made to be just for those cards for us to learn, I am
tempted to come up with a new name and port all the code into a fresh repo and clean as we go."

- mkviolet (`~/src/mkviolet`) is the seed. It is Rust, and one process owns both B70 cards. It serves council v1
  and v2. It keeps weights alive across restarts through systemd's file-descriptor store. Its code moves here
  engine by engine, and we clean it as it moves.
- mk (`~/src/megakernel-qwen38-flashnext-strixhalo`) joins through its existing C ABI (`src/engine/capi.hip`). The
  HIP code stays in mk.
- One nagaya process per machine (Amy: "we'd do 1 process per host that owns the gpus").
- Engines we build give the same bits for the same identity and inputs: `always` or `pinned` (`docs/engine-abi.md`
  D3). Engines we borrow declare what they promise instead.

## Documents

- `docs/spec.md`: the spec. Read it after this file.
- `docs/ontology.md`: the nouns, defined once, with where each one already exists in mk and mkviolet.
- `docs/engine-abi.md`: the draft engine ABI v0, how mk and mkviolet map onto it, and its decisions.
- `docs/reviews.md`: reviews of these documents and what we did with each finding.
- `docs/prior-art.md`: what ds4-recipe and Ai2's scheduler teach us, as proposals.

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

1. Engines load and unload while the nagaya runs. Weight arenas are file descriptors that we can export (mk's
   arena, mkviolet's DMA-BUF slabs). Thus with an arena staged in host memory or on NVMe, a swap is a copy between
   tiers.
2. Placement is live cluster data. The nagaya processes gossip their probed hardware, active profiles, and loaded
   engine identities through `chitchat` (`docs/spec.md` 2.3). Any machine can receive a request, tokenize it locally,
   and coordinate or proxy it (`docs/spec.md` 11.1).
3. State objects (weights, contexts, snapshots) are canonical safetensors with BLAKE3 content ids, served over plain
   HTTP (`/nagaya/v1/cas/{content_id}`) between machines that run the same engine identity.

A model spread across machines over Ethernet means pipeline stages per machine. mkviolet already splits Gemma 4 by
layers across its two cards. It hands the residual between them through host memory (`gemma4.rs`
`pipeline_stage`). Thus a stage boundary is a residual at a layer. `docs/engine-abi.md` D1 defers exposing stages
until a split across machines needs it.

lfm (Amy, 2026-10-17): "we'd add lfm over time and likely deprecate lfm2d". lfm2d is candle-based on a fork of
candle; it may stay as a second council API implementation. A new lfm engine (perhaps LFM 2.5 on one Intel card)
comes when we're ready for another model or need one to validate against.

Other later items:

- a model router (lfm or council decisions);
- rented burst machines (`~/src/rented-gpu`) that join as leases;
- streams (home video, audio for the cochlea, beat tracking).
