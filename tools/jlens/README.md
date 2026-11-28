# J-Lens Fitting Toolkit (`tools/jlens`)

GPU-capable, prompt-shardable [Jacobian Lens (J-Lens)](https://github.com/anthropics/jacobian-lens) fitting, merging, validation, and Hugging Face upload utilities for **Inkling-2-Mini** and **Qwen3.8-Flash-Next** in [nagaya](../../README.md).

Designed for one-off or repeatable runs on managed Kubernetes GPU clusters (H100 / B200 / B300 nodes) or standalone multi-GPU VMs. Everything here is self-contained and public-safe: supply your registry, `HF_TOKEN`, and `weight_hash` values at runtime.

> **Current nagaya integration status:** Today, only `lfm3` loads a J-Lens in `nagaya` (`layers.0..layers.<L-2>`, square `[W, W]`, [`docs/lab.md`](../../docs/lab.md)). The lenses produced here are fitted in BF16 on the upstream Hugging Face checkpoints (with `nagaya.lens.fit_dtype` recorded in metadata) and use rectangular `[2560, 10240]` (all 48 layers for Qwen3.8) and per-run `runs.0..runs.47` (all 48 looped runs for Inkling-2-Mini). Updating `nagaya`'s lab loader to consume these shapes is tracked in the follow-up section below.

---

## Why these two models need custom hooks

A J-Lens estimates the corpus-averaged Jacobian $J_k = \mathbb{E}\big[\partial y / \partial h_k\big]$ mapping an intermediate residual activation $h_k$ to the pre-final-norm representation $y$ read by the unembedding head ([`docs/lab.md`](../../docs/lab.md)). Reverse-mode autodiff computes one row of $J_k$ for **every** hooked layer/run simultaneously, so cost scales with $\dim(y)$ rather than depth or $\dim(h_k)$.

| Model | HF Repo | Residual $h_k$ | Target $y$ (pre-norm) | Output Shape (F16) | Approx. BF16 Footprint |
| --- | --- | --- | --- | --- | --- |
| **Inkling-2-Mini** | `thinkingmachines/Inkling-2-Mini` (`ed40e7da...`) | 24 physical layers executed as **48 runs** at 3 cycles (`[2304]`) | Folded attention-residual (`[2304]`) | `48 × [2304, 2304]` (~510 MB, `runs` view) or `24 × [2304, 2304]` (~255 MB, `layers` view) | ~37–40 GiB (**1 GPU/shard**) |
| **Qwen3.8-Flash-Next** | `Qwen/Qwen3.8-Flash-Next` | 48 layers, **4-stream mHC residual** (`[10240]`) | Folded single stream (`[2560]`) | `48 × [2560, 10240]` (~2.52 GB, `layers` view) | ~250 GiB MoE on GPU + ~102 GiB PLE n-gram table pinned to **CPU RAM** |

Because a J-Lens is simply the **mean** over $N$ prompts, shards run independently with zero DDP/NCCL communication, accumulate in `float64` on CPU, checkpoint every 5 prompts, and merge into a canonical D10 `.jlens.safetensors` file ([`docs/engine-abi.md`](../../docs/engine-abi.md)).

---

## Hardware Sizing Guide

| Node Type | Inkling-2-Mini (`INKLING2_NUM_SHARDS`) | Qwen3.8-Flash-Next (`QWEN38_TOPOLOGY`) | Expected Wall Time ($N=100$) |
| --- | --- | --- | --- |
| **8× H100 (80 GB)** | `8` shards × 1 GPU | `8xh100`: `2` shards × 4 GPUs (320 GB/shard) | Inkling-2: ~15–30 min; Qwen3.8: ~1–3 h |
| **8× B200 (180 GB)** | `8` shards × 1 GPU | `8xb200`: `4` shards × 2 GPUs (360 GB/shard) | Inkling-2: ~10–15 min; Qwen3.8: ~30–90 min |
| **8× B300 (288 GB)** | `8` shards × 1 GPU | `8xb300`: `4` shards × 2 GPUs (or override `8` × 1 GPU) | Fastest |
| **4× H100 (80 GB)** | `4` shards × 1 GPU | `4xh100`: `1` shard × 4 GPUs | Inkling-2: ~30–60 min; Qwen3.8: ~2–6 h |

**Host RAM note for Qwen3.8:** Each Qwen3.8 shard pins the ~102 GiB PLE n-gram table to CPU RAM plus ~10 GiB for the `float64` Jacobian accumulator. Plan for **256–384 GiB host RAM per Qwen3.8 shard** (8-GPU H100/B200/B300 nodes typically have 1–2 TiB RAM).

---

## Quickstart (Kubernetes)

### Prerequisites
- `git`, `docker` (or `podman`), `kubectl`, `envsubst` (`gettext`)
- A Hugging Face token (`HF_TOKEN`) with read access (and write access if uploading results)
- A Kubernetes cluster with GPU nodes and a shared `ReadWriteMany` (RWX) StorageClass (`kubectl get sc`)
- The `weight_hash` of your served models from `GET /council/v2/identity` on your nagaya nodes (or use a placeholder for an exploratory run and re-stamp later—note nagaya refuses a lens whose `weight_hash` does not match the served model)

### Step 0: Configure `env.sh`

```bash
cp tools/jlens/scripts/env.sh.example tools/jlens/scripts/env.sh
# Edit tools/jlens/scripts/env.sh:
#   - IMAGE_REGISTRY (your container registry)
#   - STORAGE_CLASS (from `kubectl get sc`)
#   - INKLING2_WEIGHT_HASH / QWEN38_WEIGHT_HASH (from `GET /council/v2/identity`)
#   - QWEN38_TOPOLOGY (8xh100 | 8xb200 | 8xb300 | 4xh100)
#   - Optional: HF_UPLOAD_ORG if you want Step 5 to upload validated lenses to HF
```

### Step 1: Build and push container images

Two targets share one base layer so each model gets its pinned `transformers` version (`5.17.0` for Qwen3.8; `046bdb8cc2` for Inkling-2-Mini):

```bash
./tools/jlens/scripts/01-build-push.sh all
# Or dry-run first:
./tools/jlens/scripts/01-build-push.sh all --dry-run
```

### Step 2: Set up namespace, HF Secret, and PVC

```bash
export HF_TOKEN="hf_..."
./tools/jlens/scripts/02-k8s-setup.sh
```

### Step 3: Run Inkling-2-Mini first (fast end-to-end smoke test)

```bash
./tools/jlens/scripts/03-run-inkling2.sh
kubectl -n nagaya-jlens get pods -l app=jlens-inkling2 -w
```

Each shard checkpoints every 5 prompts to `/data/runs/inkling2/shard-XXX-of-YYY.pt`. If a pod is preempted, Kubernetes restarts it and `fit.py` automatically resumes from its last checkpoint.

### Step 4: Run Qwen3.8-Flash-Next

```bash
./tools/jlens/scripts/04-run-qwen38.sh
kubectl -n nagaya-jlens get pods -l app=jlens-qwen38 -w
```

### Step 5: Merge, validate, and optionally upload to Hugging Face

For **Inkling-2-Mini**, you can emit both views from the same shard checkpoints without re-running GPUs:

```bash
# All 48 execution runs (runs.0..runs.47, ~510 MB) — recommended for inspecting cycle-by-cycle evolution
./tools/jlens/scripts/05-merge-validate.sh inkling2 runs

# Or 24 physical layers' final runs (layers.0..layers.23, ~255 MB)
./tools/jlens/scripts/05-merge-validate.sh inkling2 layers
```

For **Qwen3.8-Flash-Next**:

```bash
./tools/jlens/scripts/05-merge-validate.sh qwen38
```

If `HF_UPLOAD_ORG` is set in `env.sh`, Step 5 uploads:
1. `<model>.jlens.safetensors` (canonical D10 safetensors)
2. `<model>.jlens.validation.json` (SHA-256, Frobenius norm bounds, optional probe readouts)
3. `README.md` (auto-generated model card)

---

## Standalone GPU VM Option (No Kubernetes)

If you grab a single multi-GPU VM instead of a Kubernetes cluster, use the exact same images:

```bash
export HF_TOKEN="hf_..."
export DATA_DIR="/mnt/nvme/jlens"
./tools/jlens/scripts/run-local-docker.sh inkling2
./tools/jlens/scripts/run-local-docker.sh qwen38
```

---

## Local Verification (CPU, no weights required)

The unit tests verify exact reverse-mode Jacobians against central finite differences (`atol=1e-7`) on synthetic 4-stream mHC and 3-cycle looped models, canonical D10 safetensors alignment/validation, shard-and-merge equivalence, and duplicate-prompt rejection:

```bash
python3 -m unittest discover -s tools/jlens -p "test_*.py" -v
```

---

## Troubleshooting

- **OOM during backward pass:** Lower `DIM_BATCH` in `env.sh` from `16` to `8` or `4`. For Qwen3.8, you can also pass `--max-memory-gib` to leave more activation headroom per GPU.
- **Batched VJP unsupported by a custom op or cross-device `device_map`:** `fit.py` defaults to `--vjp-mode auto`, which catches batched-VJP `RuntimeError` and falls back to the sequential VJP loop automatically. Pass `--vjp-mode loop` explicitly to skip the batched attempt.
- **No `ReadWriteMany` StorageClass:** On a single 8-GPU node, set `PVC_ACCESS_MODE="ReadWriteOnce"` in `env.sh` (or use `run-local-docker.sh` directly on the node's local NVMe).
- **Preemption / partial run:** Re-applying the job resumes every incomplete shard from its last `.pt` checkpoint. Even a partial set of prompts (e.g., 20–50 prompts) can be merged right away to start probing while more shards run.

---

## Future / Longer-Term Refinements

- **Higher-precision / broader-domain fits:** Because shards store additive `(sum_J, n_prompts)` in `float64`, you can later run additional prompt slices (e.g., $N = 100 \to 400$, or code/dialogue corpora) and merge them with existing shards without redoing earlier work.
- **Model-backed probe job:** Add a dedicated 1-GPU (Inkling-2) / multi-GPU (Qwen3.8) probe job manifest to run `validate.py --probe` automatically and attach top-k layer readouts to the Hugging Face model card.
- **Nagaya lab integration:** Wire rectangular `[2560, 10240]` J-Lens support into Qwen3.8 lab routes (Milestone 34 steps 8 & 11) and per-run `[2304, 2304]` J-Lens support into Inkling-2 lab routes so `project`, `trace`, and `compose` read these files directly.
- **Nagaya on Kubernetes:** Reuse the namespace/PVC/GPU-node setup from this toolkit as the foundation for deploying `nagaya` itself on Kubernetes later.
