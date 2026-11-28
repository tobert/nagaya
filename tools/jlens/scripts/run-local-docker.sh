#!/usr/bin/env bash
# Standalone GPU VM runner (no Kubernetes needed).
# Runs all shards in parallel on one multi-GPU node using the same container images, then merges and validates.
#
# Usage:
#   DATA_DIR=/mnt/nvme/jlens HF_TOKEN=hf_... ./tools/jlens/scripts/run-local-docker.sh inkling2
#   DATA_DIR=/mnt/nvme/jlens HF_TOKEN=hf_... ./tools/jlens/scripts/run-local-docker.sh qwen38
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.sh"

if [ ! -f "${ENV_FILE}" ]; then
    echo "error: ${ENV_FILE} not found. Copy env.sh.example to env.sh first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${ENV_FILE}"

MODEL_KIND="${1:-}"
DATA_DIR="${DATA_DIR:-/mnt/nvme/jlens}"
CONTAINER_CLI="${CONTAINER_CLI:-docker}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: export HF_TOKEN=hf_... before running." >&2
    exit 1
fi
mkdir -p "${DATA_DIR}/hf-cache" "${DATA_DIR}/runs/${MODEL_KIND}" "${DATA_DIR}/out"

gpu_flags() {
    local devices="$1"
    if [ "${CONTAINER_CLI}" = "podman" ]; then
        # Podman uses CDI device specs (nvidia.com/gpu=<id>) per GPU.
        local IFS=,
        for dev in ${devices}; do
            printf -- '--device nvidia.com/gpu=%s ' "${dev}"
        done
    else
        printf -- '--gpus "device=%s"' "${devices}"
    fi
}

pids=()
case "${MODEL_KIND}" in
    inkling2)
        for s in $(seq 0 $((INKLING2_NUM_SHARDS - 1))); do
            # shellcheck disable=SC2046
            "${CONTAINER_CLI}" run --rm $(gpu_flags "${s}") \
                -e HF_TOKEN="${HF_TOKEN}" -e HF_HOME=/data/hf-cache \
                -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                -v "${DATA_DIR}:/data" \
                "${IMAGE_INKLING2}" \
                --model-kind=inkling2 \
                --revision=ed40e7da625f594fc9c4990f5f66be24873c22d7 \
                --weight-hash="${INKLING2_WEIGHT_HASH}" \
                --n="${PROMPTS_N}" --shard-index="${s}" --num-shards="${INKLING2_NUM_SHARDS}" \
                --dim-batch="${DIM_BATCH}" --max-seq-len="${MAX_SEQ_LEN}" \
                --out-dir=/data/runs/inkling2 \
                > "${DATA_DIR}/runs/inkling2/shard-${s}.log" 2>&1 &
            pids+=($!)
        done
        ;;
    qwen38)
        if [ -z "${SHARD_GPU_GROUPS:-}" ]; then
            case "${QWEN38_TOPOLOGY:-8xh100}" in
                8xh100) SHARD_GPU_GROUPS="0,1,2,3 4,5,6,7" ;;
                8xb200|8xb300) SHARD_GPU_GROUPS="0,1 2,3 4,5 6,7" ;;
                4xh100) SHARD_GPU_GROUPS="0,1,2,3" ;;
                *) echo "error: unknown QWEN38_TOPOLOGY=${QWEN38_TOPOLOGY}" >&2; exit 2 ;;
            esac
        fi
        IFS=' ' read -r -a GROUPS <<< "${SHARD_GPU_GROUPS}"
        NUM_SHARDS="${#GROUPS[@]}"
        for s in "${!GROUPS[@]}"; do
            # shellcheck disable=SC2046
            "${CONTAINER_CLI}" run --rm $(gpu_flags "${GROUPS[$s]}") \
                -e HF_TOKEN="${HF_TOKEN}" -e HF_HOME=/data/hf-cache \
                -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                -v "${DATA_DIR}:/data" \
                "${IMAGE_QWEN38}" \
                --model-kind=qwen38 \
                --weight-hash="${QWEN38_WEIGHT_HASH}" \
                --n="${PROMPTS_N}" --shard-index="${s}" --num-shards="${NUM_SHARDS}" \
                --dim-batch="${DIM_BATCH}" --max-seq-len="${MAX_SEQ_LEN}" \
                --out-dir=/data/runs/qwen38 \
                > "${DATA_DIR}/runs/qwen38/shard-${s}.log" 2>&1 &
            pids+=($!)
        done
        ;;
    *)
        echo "usage: $0 <inkling2|qwen38>" >&2
        exit 2
        ;;
esac

echo "Started ${#pids[@]} shards for ${MODEL_KIND}. Waiting..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done
if [ "${fail}" -ne 0 ]; then
    echo "error: one or more shards failed; inspect ${DATA_DIR}/runs/${MODEL_KIND}/shard-*.log" >&2
    exit 1
fi

echo "All shards complete. Merging and validating..."
if [ "${MODEL_KIND}" = "inkling2" ]; then
    "${CONTAINER_CLI}" run --rm -v "${DATA_DIR}:/data" --entrypoint /bin/bash "${IMAGE_INKLING2}" -c "
        python3 -m tools.jlens.merge --shards-dir /data/runs/inkling2 --view runs --out /data/out/inkling2-mini.runs.jlens.safetensors &&
        python3 -m tools.jlens.merge --shards-dir /data/runs/inkling2 --view layers --out /data/out/inkling2-mini.layers.jlens.safetensors &&
        python3 -m tools.jlens.validate --lens /data/out/inkling2-mini.runs.jlens.safetensors --weight-hash '${INKLING2_WEIGHT_HASH}' &&
        python3 -m tools.jlens.validate --lens /data/out/inkling2-mini.layers.jlens.safetensors --weight-hash '${INKLING2_WEIGHT_HASH}'
    "
else
    "${CONTAINER_CLI}" run --rm -v "${DATA_DIR}:/data" --entrypoint /bin/bash "${IMAGE_QWEN38}" -c "
        python3 -m tools.jlens.merge --shards-dir /data/runs/qwen38 --view layers --out /data/out/qwen38-flash-next.jlens.safetensors &&
        python3 -m tools.jlens.validate --lens /data/out/qwen38-flash-next.jlens.safetensors --weight-hash '${QWEN38_WEIGHT_HASH}'
    "
fi
