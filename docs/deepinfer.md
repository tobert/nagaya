# DeepInfer

DeepInfer is DeepSeek's inference framework for DeepSeek V4.1 Flash (Apache-2.0). This doc records what it teaches
the nagaya, the exokernel of `docs/spec.md`, and where each lesson lands. It follows the rules of
`docs/prior-art.md`: an item is a proposal until Amy decides it.

The clone is `~/src/research/DeepInfer` at `774ce24`, with the submodules DeepGEMM, FlashMLA, DeepEP and DeepSelect.
Three Claude readers read the code on 2026-10-20: the kernels, the scheduler and state, and the cluster code. They
ran nothing. Senpai (Claude Opus 5.5) checked the load-bearing citations against the code.

Amy, 2026-10-20: "we don't have a deepseek on the roadmap yet, but maybe we can learn from their code for our
megakernel and clustering".

## What DeepInfer is

DeepInfer serves one model family on NVIDIA SM100 and Huawei Ascend 950. A deployment is wide: 64-128 GPUs for
decode, with experts spread over all of them. It started from vLLM. The release is the engine for one machine and
the parallel code inside it. Its doc, `doc/inference_deployment_qa.md`, describes the whole cluster.

## What is not in the code

Read the doc as a description, not as code. These parts are prose only:

- the load balancers for prefill, decode and the vision encoder;
- the KV cache store, its hash-chain prefix match, and its eviction;
- the prefill time budget and its fitted cost `cost(q, c)` (doc line 115). No file in the repository holds it;
- the policy that decides when to copy state for a store (`should_copy_state` returns false,
  `deepinfer/engine/llm_engine.py:811`);
- the drain at shutdown and the change of expert count at restart.

The hooks for the store are empty methods (`llm_engine.py:801-830`). The doc says why (line 21): the store and the
load balancers are "tightly coupled to our internal infrastructure".

## What we already have

Four of their designs match rules that the spec already states. This is evidence for those rules, not new work.

- **Prefix keys chain at boundaries.** Their store matches a prefix by a hash chain over metadata, at 128-token
  blocks (doc line 157). Our recipes chain at boundaries in the same way (spec 6.2).
- **The engine picks its chunk size from a time bound.** Their prefill solves a cost model for the chunk that fits a
  time budget. Our engines size each unit of work to fit `max_quantum_ms` (spec 5.1, D8).
- **A profile curve keyed by a hash.** They measure decode time against batch size at start. They keep the curve in a
  file named by the SHA-256 of the model and the parallel layout (`deepinfer/worker/decode_perf_cache.py:40-52`).
  This is the plan table's idea (D3 amendment): a measured choice, saved under a key, and used again.
- **Spin waits time out.** Every wait in DeepGEMM's grid barrier stops after 60 s with a message
  (`vendor/DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh:11-24`). The Qwen3.8 engine has spin timeouts too
  (engine-abi, "test limits of the spin-timeout opt-in").

## Proposed changes to the spec

Each item gives the lesson, its source, and the place in our docs where it lands. Claude Opus 5.5 proposed them on
2026-10-20, and Amy decided them the same day by interview. Each item starts with its decision.

### 1. A sampler draw that is a function of the seed and the position (spec 11.2)

**Decided:** adopted. Milestone 12 matches the first Qwen3.8 engine's filter order and probabilities, not its bits.

DeepInfer keeps no random state for a sequence. Each draw seeds the generator from the sequence seed and a counter
(`deepinfer/model_executor/layers/tilelang_ops/cuda/random_sampling.py:113`). The counter is three times the
absolute position, plus a case: sample, resample, or accept test (`random_sampling.py:247`). The draw then uses an
integer CDF over fixed-point probabilities, so the thread layout does not change the result.

The sampler of the first Qwen3.8 engine is different. It draws from one PCG64 stream for each request, one uniform for
each step (`tools/qwen38_reference/sampling.py:21`, `:104`, `:151`). The draw at step k thus depends on how many draws
came before it.

Proposed rule: the draw for a token is a function of the seed, the absolute position, and the purpose of the draw.
The sampler holds no state between tokens. A fork gets a child seed: a hash of the parent seed and a branch number.

Reason: park, restore and fork then keep the sampler's bits with nothing to save. A generation that resumes after a
restart gives the same tokens. Speculative drafts (11.2) get their own purpose, so a draft never moves a later draw.

**For Amy:** this conflicts with milestone 12. Milestone 12 asks the sampler to match the PCG64 draws of the first
Qwen3.8 engine bit for bit. We can match its draws and keep a step counter in each job, or take the new rule and match
only its filter order.

### 2. A reduction order that depends only on the shape (D3, spec 5.7)

**Decided:** adopted as the meaning of `segments_exact`. Conformance varies the segment count (spec 8.3).

DeepInfer's batch-invariant mode has one rule. The order of every sum depends on the problem shape, never on the
batch size or the count of compute units. Their switches show it:

- The gate GEMM never splits its K dimension in this mode
  (`vendor/DeepGEMM/csrc/jit_kernels/heuristics/mega_gate.hpp:110`, `:126`).
- The hyper-connection kernel uses a fixed split count of 16 (`vendor/DeepGEMM/csrc/apis/mega_mhc.hpp:15-17`).
- The MoE combine sums its expert outputs in a fixed slot order, with no float atomics.

Proposed: write this rule into D3 as the meaning of `segments_exact`. An engine that declares `segments_exact`
promises that no reduction order depends on the segment count or the row count of a pass.

Reason: `segments_exact` now says what is true, but not what an engine must do to make it true. DeepInfer's rule is
a test that a kernel author can check.

### 3. Interval checkpoints at a fixed grain (spec 6.13)

**Decided:** an interval copy that does not seal (spec 6.13, ontology "Interval copy"). The latest copy for each
context stays in host memory, and eviction takes those copies first, with a cap.

DeepInfer stores state only at positions that are multiples of 128 (`KVSTORE_GRAIN`,
`deepinfer/worker/cache_engine.py:32`). During decode, it copies a sequence's state on the device each time the
sequence crosses a multiple of 128 (doc line 165). At the end of a request, the store takes the last copy.

Thus a restore replays fewer than 128 tokens. The cost of a restore has a fixed bound, and the cost model can check
that bound against a deadline (5.8).

Their replay of the window is approximate, and that part does not transfer. A GDN state depends on the whole
prefix, so a short replay gives a different state, not a near one. Our replay must be exact: from the checkpoint to
the position, with the same boundaries.

Proposed: an engine can declare a checkpoint grain in `limits()`. The scheduler then makes an interval copy at each
multiple of the grain. (The decision above replaces this text.)

The Qwen3.8 engine's seal rounds the GDN state to bf16 (ontology "Seal"), so a seal changes the bits that follow. There
were two choices for an interval checkpoint:

- a seal at a declared boundary, so the grain positions join the recipe;
- a copy that does not seal, which has no content id until it goes into the store (engine-abi, D7).

The first choice changes every generation's bits at each grain. The second choice keeps the bits but needs a new
store rule. This connects to the "shadow rounding points" from goal 06 of the first Qwen3.8 engine.

### 4. A cost model with a measured prior (spec 5.11)

**Decided:** adopted (spec 5.11).

Spec 5.11 recorded "no model yet" until a fit exists. DeepInfer measures a curve at start instead: five timed runs
for each batch size, a trimmed mean (`deepinfer/worker/model_runner.py:586-619`), saved under a hash
(`decode_perf_cache.py:40-52`). It then corrects the curve
online with the median of the recent errors, and keeps the slope at zero or more
(`csrc/engine/mtp_varlen.hpp:44-45`, `:226`).

Proposed: an engine's first start measures a short curve for each kind and saves it in the plan table. The fit starts
from that curve. A quantum that ran beside other work stays out of the fit, as now.

Their FLOPs estimator gives closed forms for the attention work of a chunk, with a window or a cap
(`csrc/engine/flops_estimator.hpp:237-278`). These are good inputs for our fit, because only our attention layers
grow with depth.

### 5. Device discovery that tests can fake (spec 2.1)

**Decided:** adopted (spec 2.1).

DeepInfer finds devices from sysfs, procfs and `/dev`. It reads NVML through `ctypes`, so it never opens a GPU
runtime to count devices (`deep_cluster/hardware/nvidia.py:18-129`). Each probe takes a root path, so a test runs it
against a fake sysfs tree.

Proposed: the inventory probe in 2.1 takes a root path, and its tests use fixed trees for gfx1151host, b70host and
sm121host. DeepInfer assumes one vendor for each machine and records no links between devices. Our inventory keeps the
measured edges of the tier graph (ontology "Tier").

### 6. One directory of kernels for each vendor

**Decided:** not this layout. Each engine crate already serves one vendor and device. Adopted instead: one
reference for each model family is the oracle for every engine of that family (spec 8.3).

DeepInfer keeps a neutral interface and a Python reference for each kernel. The kernels for each vendor live in
their own directory (`deepinfer/model_executor/layers/tilelang_ops/{cuda,asc}/`). A missing kernel raises
`KernelUnavailableError`, and the tests make that a skip (`tilelang_ops/ops.py:15-32`). The reference is the test
oracle for every vendor.

Proposed: when an engine has kernels for two vendors, use this layout. Do not take their other method: a patch of
`torch.cuda` and of imports so that CUDA code runs on Ascend (`deepinfer/platform.py:120-136`). It works for one
extra vendor, but we have three.

### 7. Routing goes to the chain (spec 11.1)

**Decided:** deferred. Amy expects fleets with sticky sessions for each model and machine (spec 11.1).

DeepInfer's store is shared and fast, so any engine can load any prefix. Their prefill balancer thus ignores where a
prefix lives (doc line 111). Our links are gigabit. A few hundred MB of state takes seconds to move.

Proposed: when coordinator routing comes (11.1), it routes a request to the machine whose store holds the deepest
chain for its recipe. Load breaks ties. Revisit this rule when the links are faster.

## Lessons for the kernels of the Qwen3.8 engine

These lessons are inside the engine, so the spec does not hold them. They apply to the HIP code of the Qwen3.8 engine,
which comes into this repository at milestone 15. The Qwen3.8 engine already has a grid barrier whose generation needs
no reset (`src/rt/rt.h:184`).

1. **Counters for each tile, not a barrier for each stage.** DeepGEMM's down projection waits on a 64-bit mask, with
   one bit for each up-projection tile (`deep_gemm/include/deep_gemm/layout/mega_moe.cuh:82-89`). A producer flips its
   bit with a release XOR. The expected value flips each round, so the mask needs no reset. A consumer thus starts
   when its own inputs are done, not when the whole stage is done. First measure how long the Qwen3.8 engine waits at
   stage barriers in decode.
2. **A global queue of tasks.** Each workgroup takes the next task number from an atomic counter. One wave turns the
   number into an expert and a block, with a prefix sum over the expert counts. A closed form proves that the order
   cannot deadlock (`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:15-45`).
3. **The shared expert fills the gap during routing.** Its work does not need the routing result, so it runs while
   the top-k runs (`scheduler/mega_moe.cuh:442-452`).
4. **The producer writes the consumer's layout.** The up-projection's epilogue does the clamp, SwiGLU, the routing
   weight and the quantization. It writes in the layout that the down projection reads. Gate and up rows interleave
   in groups of 8, so SwiGLU stays in one thread.
5. **A timeout and a message on every wait.** The Qwen3.8 engine has spin timeouts. Each one should print what it waited
   for, as DeepGEMM's `wait_until` does.

These do not port to gfx1151: TMEM, UMMA, TMA, clusters of two blocks, and warps that trade registers. AMD fixes
the register count for each kernel. Thus waves that change role for each task suit RDNA3.5 better than fixed roles.

## What we do not take

These parts pay off only with many GPUs and fast links:

- experts spread over 64-128 GPUs, and the load balancer that moves them;
- prefill and decode on separate machines, which moves KV over a 400 Gb/s link;
- the 3FS store with erasure codes, and GPUDirect RDMA into device memory;
- the vote across ranks before each prefill step;
- variable-length speculative checks. Their own table shows -1.5% at 16 sequences for each GPU (doc line 251). The
  method also breaks batch invariance (README).

Their asynchronous loop plans step N+1 while the GPU runs step N. Our pass blocks its group thread (spec 3.2), so it
does not fit v0. Look at it again if the time between quanta becomes large. Their guard fails on any hidden host sync
inside a step (`deepinfer/utils.py:119-132`). An equivalent would be a test for our group thread.
