# AGENTS.md

Contributor and coding-agent guide for **nagaya** — an exokernel for megakernels, its Rust reference implementation, and the council v2 server. Start with [`README.md`](README.md) for the user-facing overview, [`docs/spec.md`](docs/spec.md) for the normative contract, and [`docs/ontology.md`](docs/ontology.md) for canonical terminology.

@tobert owns the project and makes final decisions.

## Public repository hygiene

Every change must be safe for a public GitHub reader who only has this repository:

- **No old codenames** in new code or docs: do not use `mk` (as a project name), `mkviolet`, `gmk`, `diffusepal`, `megakernel.service`, `mkviolet.service`, `libmkengine`, or `/mk/v1`.
  - "Megakernel" stays. `mk` stays in crate names (`nagaya-mk-*`) and internal kernel symbols (`MK_LFM3`, `mk_persistent_interpreter`).
  - `docs/history.md` retains historical names as an archive.
- **No hostnames, real IPs, or home paths** in code, tests, tools, docs, or commit messages. Read them from config or environment variables.
  - For example/test IPv4 addresses, use RFC 5737 documentation blocks: `192.0.2.0/24`, `198.51.100.0/24`, or `203.0.113.0/24`.
  - For example/test hostnames, use hardware placeholders: `gfx1151host`, `b70host`, `sm121host`.
- **No private notes or session logs in committed docs.** Keep those in `signoff.md` (local only, never committed).

## Style & communication

- **Clarity and high information density** in docs, comments, error messages, and commit messages. Avoid filler.
- **Use canonical ontology terms** from [`docs/ontology.md`](docs/ontology.md) in both prose and code: `Quantum`, `Ledger`, `Recipe`, `ContentId` (not `WorkItem`, `MemMgr`, `CacheKey`).
- **Comments** explain *what* a thing is or *why* it exists; never restate the code.
- **Error messages** state what failed, why, and what the caller can do. Clients read them when they cannot continue—give them the most care.
- Use a little 日本語（にほんご） when it fits, with ふりがな for kanji. 頑張（がんば）って！

## Agent workflow & engineering rules

- **Read `signoff.md` first** (when present locally). It is the handoff between sessions and machines and is never committed. Keep it concise and move durable facts into `docs/`.
- **Use git worktrees** for isolated work lanes (`~/src/wt/nagaya-<topic>`) with a separate `CARGO_TARGET_DIR`. Coordinate with other active sessions before editing shared files.
- **Be explicit about status and blockers.** If a tool or sandbox blocks you, report the exact cause and the safe next option. Report real counts: tests run, passed, skipped, and anything intentionally not run.
- **Verify citations.** Cite `file:line` for code claims and re-read the line before reporting.
- **Test-first & mutation testing:**
  1. Write the test first and watch it fail for the right reason.
  2. Write the code that makes it pass.
  3. Falsify the test: mutate the code under test, verify the test fails, and restore the code. Record the tested mutation in the commit message (never cite a mutation you did not run).
  - If a test exposes a real defect, stop and report it—do not weaken the test.
  - **Never mutate a pointer or size read by a GPU kernel** (can fault/wedge hardware).
- **Fail fast, never corrupt data (`docs/spec.md` 8.5).** No silent fallbacks; an unexpected state is an error or panic.
- **The spec is the contract.** If code and spec disagree, stop and report. If a doc leaves a choice open, commit the simplest choice into the doc marked **Proposed** with your rationale, then proceed.
- **Bit-exact determinism.** Engines produce identical bits for identical engine identity and inputs. Any change that moves bits must update engine identity and say so in the commit message.
- **External repos:** Do not edit other repositories except `gogi` (edit on a branch, merge to `main`, push, and update the pinned `rev` in `Cargo.toml`).
- **Model licenses:** nagaya does not check model licenses; the user accepts them at download.

## Workspace layout

nagaya is a 30-crate Cargo workspace targeting three GPU architectures:

- **`gfx1151`** — AMD Strix Halo (128 GB unified memory, HIP): Qwen3.8, Qwen-Image 2.1.
- **`bmg`** — Intel Arc Pro B70 (Level Zero + OpenCL C persistent interpreter): Gemma 4, LFM3 / LFM3-Omni, Inkling 2, SDXL, Qwen3-TTS, EmbeddingGemma 2.
- **`sm121`** — NVIDIA DGX Spark / GB10 (128 GB unified memory, CUDA Driver API + CUDA C++/PTX persistent interpreter): Inkling 2.

| Crate | Purpose | Spec / ABI |
|---|---|---|
| `nagaya-abi` | `Engine` trait, ABI types, `fake` engine feature | `docs/engine-abi.md` |
| `nagaya-store` | Canonical safetensors, BLAKE3 content IDs, recipes, index, eviction | Spec 6, D3, D10 |
| `nagaya-ledger` | Tier budgets, mandatory/elastic terms, admission, shares, reclaim | Spec 4, D9 |
| `nagaya-sched` | Jobs, bounded quanta, priority classes, ready-queue ordering | Spec 5 |
| `nagaya-council` | Pure council v2 logic (no I/O): family interface, sampler, validation | Spec 11 |
| `nagaya-jinja` | Jinja2-compatible chat-template interpreter | Spec 11.3 |
| `nagaya-conformance` | Generic `Engine` conformance test suite | Spec 8.3 |
| `nagaya-telemetry` | Metrics, stderr logging, OTLP gRPC export, sampler | Spec 9 |
| `nagaya` | Daemon: inventory probe, device threads, HTTP, gossip, systemd | Spec 2, 3, 9–11 |
| `nagaya-mk-hip` | Shared `gfx1151` HIP ops: hipBLASLt + plan table, Qwen-Image kernels | Spec 12.1 |
| `nagaya-mk-qwen38-gfx1151` | Qwen3.8 megakernel (`hip/`), FFI binding, `Engine` impl | ABI "Qwen3.8" |
| `nagaya-mk-qwenimage21-gfx1151` | Qwen-Image 2.1 on `gfx1151`: encoder, transformer, VAE | Spec 11.9c |
| `nagaya-mk-bmg` | Shared B70 persistent interpreter: Level Zero, task protocol, OpenCL C | ABI "Gemma 4" |
| `nagaya-mk-gemma4-bmg` | Gemma 4 2-card pipeline engine on B70 (`xe`) | ABI "Gemma 4" |
| `nagaya-mk-qwen3tts-bmg` | Qwen3-TTS VoiceDesign engine on B70 | ABI "Speech" |
| `nagaya-mk-sdxl-bmg` | SDXL engine on B70: encoders, UNet, VAE, sampler, LoRA/inpaint | ABI "Image" |
| `nagaya-mk-embeddinggemma2-bmg` | EmbeddingGemma 2 engine on B70 | ABI "Embedding" |
| `nagaya-mk-lfm3-bmg` | LFM3 & LFM3-Omni engine on B70: backbone, vision, audio, speech, control, loops | ABI "LFM3" |
| `nagaya-mk-inkling2-bmg` | Inkling 2 engine on B70: weight formats, host oracle, cycles, vision | ABI "Inkling 2" |
| `nagaya-mk-cuda` | Shared CUDA persistent interpreter: Driver API, task protocol, CUDA/PTX | ABI "GB10 Inkling 2" |
| `nagaya-mk-inkling2-sm121` | Inkling 2 on GB10: NVFP4/MXFP8 quantization, FP8 KV cache, host oracle | ABI "GB10 Inkling 2" |
| `nagaya-text-qwen38` | Qwen3.8 family: tokenizer, templates, detokenizer, reply parser | Spec 11.3, 11.3a |
| `nagaya-text-gemma4` | Gemma 4 family: tokenizer, template, escapes, council rendering, parser | Spec 11.3, 11.3a |
| `nagaya-text-qwen3tts` | Qwen3-TTS family: tokenizer, speech prompt, codec IDs | Spec 11.3, 11.8 |
| `nagaya-text-sdxl` | SDXL family: CLIP tokenizer, A1111 prompt chunks and weights | Spec 11.9a |
| `nagaya-text-embeddinggemma2` | EmbeddingGemma 2 family: tokenizer, task prompts, template ID | Spec 11.10 |
| `nagaya-text-qwenimage21` | Qwen-Image 2.1 family: tokenizer, template, system ID cut | Spec 11.9c |
| `nagaya-text-lfm3` | LFM3 family: tokenizer, templates, escapes, XML tool calls, media positions | Spec 11.3, 11.11 |
| `nagaya-text-inkling2` | Inkling 2 family: tokenizer, template, effort, escapes, reply parser | Spec 11.3, 11.18 |
| `nagaya-text-toy` | Toy family over `gogi-toy` for tests with the `fake` engine | Spec 11.3b |

Naming convention: `nagaya-<kind>-<model>-<target>` (e.g., `nagaya-mk-gemma4-bmg`). Device count belongs in the placement profile, not the crate name.

Keep pure logic out of engine crates and `crates/nagaya` so it tests without a GPU. `Ledger` is a plain owned struct with `&mut self` methods—do not wrap it in a lock (single-thread owned, spec 3.2).

See [`docs/new-engine.md`](docs/new-engine.md) when adding an engine, family, card, or machine.

## GPU targets & lab rules

### General GPU test rules

- **Single compute client per GPU (`spec 3.1`).** Check `systemctl --user is-active nagaya.service` first (it is a **user** unit; omitting `--user` falsely reports `inactive`).
- If an unrelated GPU client is running, stop and report—never kill a process that is not ours.
- Agents may stop/start `nagaya.service` for test windows or deploys (and restart `tools/nagaya_web.py` on `b70host`). Log every stop, start, and deploy in `signoff.md` with timestamp and commit hash, and leave `nagaya.service` active when finished.
- systemd allows at most 3 unit starts per 10 minutes (`systemctl --user reset-failed nagaya.service` resets the counter). Batch gates into one window.
- **Use a separate `CARGO_TARGET_DIR` for test builds.** `nagaya.service` runs `target/release/nagaya`.

### `gfx1151` (HIP)

- Enabled by `--features hip`; compiles `crates/nagaya-mk-qwen38-gfx1151/hip/` via ROCm `clang++` and links statically. Builds without `hip` need no ROCm.
- Preserve `-ffp-contract=off -fno-fast-math` (checked by `tests/build_flags.rs`).
- `QWEN38_SHARDS` holds the three GGUF shard paths separated by `:`.
- Make all engine calls from one thread. Do not free the engine in tests (HIP teardown segfaults, spec 3.3); keep a `hold()` handle (`tests/hip.rs`), except in `tests/hip_close.rs`.
- `extern.hip` performs no validation—Rust workers must validate first (`host::segments_problem`).
- Run `cargo test -p nagaya --features hip --test qwen38_wire` when changing `crates/nagaya`, `deploy/`, or the Qwen3.8 crate (opens no engine). Only run `qwen38_stop` or `nagaya-mk-hip` GPU tests with `nagaya.service` stopped.

### `bmg` (Level Zero / OpenCL C)

- Enabled by `--features xe`; `build.rs` compiles OpenCL C kernels via `ocloc` for `bmg-g31` into `OUT_DIR`.
- Run card tests single-threaded: `cargo test -p <crate> --tests --features xe -- --test-threads=1`.
- Set `NAGAYA_TEST_CARDS` to comma-separated PCI addresses of **free** cards (never an active engine's card or the display iGPU `gfx1036`). With a card released via `POST /nagaya/v1/devices/0/release`, tests can run while `nagaya.service` stays active; reclaim afterward.
- Run CPU Python reference oracles only when no GPU job is active.
- Video support requires `--features video` (FFmpeg + `libclang`). Production builds on `b70host` use `--features xe,video`.

### `sm121` (CUDA / PTX)

- Enabled by `--features cuda`; `build.rs` compiles CUDA C++ / inline PTX via `nvcc` to `sm_121a` `.cubin` files into `OUT_DIR` and links `libcuda`.
- Check `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` before every GPU test window.
- Run GPU tests with `--test-threads=1` and `NAGAYA_TEST_CARDS` set to the free GPU PCI address.

## Pre-commit checks

Run these before every commit (gate directly on exit codes—never pipe through `tail` or `tee` before committing):

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features hip -- -D warnings
cargo clippy --workspace --all-targets --features xe -- -D warnings
cargo clippy --workspace --all-targets --features cuda -- -D warnings
cargo test --workspace --tests
cargo test --workspace --tests --no-default-features
```

On machines without `ocloc` or `nvcc`, use the compiler stubs to type-check and lint GPU code (stubs emit empty kernels—do not run stub-built binaries on hardware):

```sh
PATH="$PWD/tools/ocloc-stub:$PATH" cargo clippy --workspace --all-targets --features xe -- -D warnings
PATH="$PWD/tools/nvcc-stub:$PATH" cargo clippy --workspace --all-targets --features cuda -- -D warnings
```

Python tool tests (from `tools/`):

```sh
python3 -m unittest test_nagaya_web test_nagaya_web_page test_nagaya_image_batch test_oracle_fixture
python3 -m unittest discover -s swarm/tests -t .
```

Always pass `--tests` to `cargo test` (`--lib` skips integration tests). When touching a GPU engine crate on its target host, also run its feature-gated GPU tests (`--features hip|xe|cuda -- --test-threads=1`).

## Model-dependent tests & reference oracles

Host tests that require local model weights skip with a one-line stderr notice when the path variable is unset (an existing but unreadable path still fails). Set `NAGAYA_REQUIRE_MODELS=1` on a host that holds the weights to turn skips into failures—do this at least before merging changes to that model's crates, and report how many tests skipped.

| Model | Model env var(s) | Golden oracle dir / script |
|---|---|---|
| Qwen3.8 | `NAGAYA_QWEN38_GGUF` (first shard), `QWEN38_SHARDS` (`:`-joined) | `tools/oracle_fixture.py` (uses `tools/qwen38_reference`, ~30 GiB RAM); `cargo bench -p nagaya-mk-qwen38-gfx1151 --features hip --bench oracle` (`QWEN38_FAST_QMM=0` for exact gate) |
| Qwen-Image 2.1 | `QWENIMAGE21_MODEL_DIR`, `QWENIMAGE21_FUSAL_DIR` | `QWENIMAGE21_GOLDEN_DIR` / `tools/qwenimage21_golden.sh` (stop service first) |
| Gemma 4 | `GEMMA4_MODEL_DIR` | `GEMMA4_GOLDEN_DIR` / `tools/gemma4_golden.sh` |
| Qwen3-TTS | `QWEN3TTS_MODEL_DIR` | `QWEN3TTS_GOLDEN_DIR` / `tools/qwen3tts_golden.sh` |
| SDXL | `SDXL_MODEL_DIR`, `SDXL_SINGLE_FILE` (needs `vae_1_0/`) | `SDXL_GOLDEN_DIR` / `tools/sdxl_golden.sh` |
| EmbeddingGemma 2 | `EMBEDDINGGEMMA2_MODEL_DIR` | `EMBEDDINGGEMMA2_GOLDEN_DIR` / `tools/embeddinggemma2_golden.sh` |
| LFM3 / LFM3-Omni | `LFM3_MODEL_DIR`, `LFM3_OMNI_MODEL_DIR` | `LFM3_GOLDEN_DIR` / `tools/lfm3_golden.sh` |
| Inkling 2 | `INKLING2_MODEL_DIR`, `INKLING2_VARLOOP_MODEL_DIR` | `INKLING2_GOLDEN_DIR` / `tools/inkling2_golden.sh` |

## Git & reviews

- Commit small, single-idea steps on the current branch whenever checks pass.
- Stage files explicitly by name (`git add <file>`). Never use `git add -A` or `git add .`. Check `git status --short` before committing.
- Never push, amend, rebase, reset, or force-push unless explicitly asked.
- Never commit `signoff.md`, `target/`, model weights, or snapshots. Small fixtures go in `<crate>/tests/fixtures/`.
- Commit message format:
  - Subject: `<crate>: <what changed>` (or `nagaya: <what changed>` for multi-crate changes).
  - Body: the *why* / design decision and any mutations run.
  - Trailers: `Co-Authored-By: <Model> <email>` for authoring models; `Reviewed-by: <Model> <email>` for reviewing models (e.g., kaibo peers).
- Near milestone completion, run review rounds via kaibo using a different model family than the author (`docs/reviews.md`).
- Open milestones live in [`docs/milestones.md`](docs/milestones.md); completed milestones 1–30 are archived in [`docs/history.md`](docs/history.md).
