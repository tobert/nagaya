# 長屋 nagaya

A *nagaya* (長屋（ながや）) is a row house: one building, several households, each with its own door. This nagaya is
the one process on each machine that owns the GPUs. Engines live in it, and each engine has its own door.

The nagaya is three things:

- **An exokernel for megakernels.** A megakernel engine runs the pass of a model as one GPU kernel. The nagaya divides
  a machine among its engines: device memory, host memory, NVMe, and time on each device. It decides where state
  lives, what runs next on each device, and what the store keeps. It never decides what a model calculates. Each
  engine keeps its own key and value layout, its own recurrent state and its own kernels.
- **A reference implementation.** The Cargo workspace in `crates/` is that process in Rust, with its engines.
- **The council v2 API, as served.** Clients speak council v2 to the nagaya. The contract is
  [gogi](https://github.com/tobert/gogi). The nagaya serves it for each engine of a cluster of machines. Its own
  routes are under `/nagaya/v1`.

## The engines

Each engine gives the same bits for the same identity and the same inputs. A change that moves a bit changes the
identity of the engine.

| Target | Hardware | Engines |
|---|---|---|
| `gfx1151` | one AMD Strix Halo GPU with 128 GB of unified memory, through HIP | Qwen3.8, Qwen-Image 2.1 |
| `bmg` | Intel Arc Pro B70 cards, through Level Zero and OpenCL C kernels | Gemma 4 (two cards), Qwen3-TTS, SDXL, EmbeddingGemma 2, LFM3 |

A crate name tells the kind, the model and the target: `nagaya-mk-qwen38-gfx1151` is the megakernel engine of
Qwen3.8 on `gfx1151`. A family library, such as `nagaya-text-lfm3`, holds the tokenizer, the chat template and the
reply parser of a model. It needs no GPU.

## Status

The nagaya is a research project. It runs on two machines of one home lab: a Strix Halo machine and a machine with two
B70 cards. The API of `/nagaya/v1` can change. The council v2 routes follow gogi.

The nagaya trusts its network. Its HTTP routes have no authentication. Bind it to an address that only your clients
can reach.

## Build

You need a Rust toolchain. A build with no feature needs no GPU, no driver and no model file:

```
cargo build --release -p nagaya
```

That binary has no engine. The features add the engines:

| Feature | Adds | Needs |
|---|---|---|
| `hip` | the `gfx1151` engines | ROCm, with its `clang++` |
| `xe` | the `bmg` engines | the Level Zero loader, and `ocloc` to compile the kernels |
| `video` | the video inputs of LFM3 | the libraries of ffmpeg, and libclang |

`docs/deploy.md` tells how to install the process as a systemd user unit. `deploy/` holds the config, the profile
and the unit of each machine of the lab, as examples.

## Checks

These checks need no GPU and no model file:

```
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --tests
cargo test --workspace --tests --no-default-features
python3 tools/ste_check.py README.md docs/*.md
```

A test that needs a model file reads its path from a variable, and it skips when the file is absent. A GPU test
needs its feature (`hip` or `xe`) and the hardware. `AGENTS.md` lists each variable and each GPU check.

## Documents

- `docs/spec.md`: the spec. It says what the nagaya is and what it does, one rule at a time. Read it after this file.
- `docs/engine-abi.md`: the engine ABI: the calls of an engine, how each engine maps onto them, and the decisions
  D1 to D11.
- `docs/ontology.md`: each term, defined once.
- `docs/new-engine.md`: the checklist for a new engine, a new family, a new card or a new machine.
- `docs/deploy.md`: the build and the install on each machine.
- `docs/lab.md`: the lab routes, which look inside a model: taps, control vectors, traces and lenses.

The docs and the code comments use Simplified Technical English (ASD-STE100). `tools/ste_check.py` checks the docs.
