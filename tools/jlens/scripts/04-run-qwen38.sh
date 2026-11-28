#!/usr/bin/env bash
# Step 4: Launch the Qwen3.8-Flash-Next J-Lens IndexedJob.
# Automatically maps QWEN38_TOPOLOGY (8xh100 | 8xb200 | 8xb300 | 4xh100) to shard/GPU counts.
#
# Usage:
#   ./tools/jlens/scripts/04-run-qwen38.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(cd "${SCRIPT_DIR}/../k8s" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.sh"

if [ ! -f "${ENV_FILE}" ]; then
    echo "error: ${ENV_FILE} not found. Copy env.sh.example to env.sh first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${ENV_FILE}"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

case "${QWEN38_TOPOLOGY:-8xh100}" in
    8xh100)
        export QWEN38_NUM_SHARDS="${QWEN38_NUM_SHARDS:-2}"
        export QWEN38_GPUS_PER_SHARD="${QWEN38_GPUS_PER_SHARD:-4}"
        export QWEN38_MEM_REQUEST="${QWEN38_MEM_REQUEST:-160Gi}"
        export QWEN38_MEM_LIMIT="${QWEN38_MEM_LIMIT:-256Gi}"
        ;;
    8xb200)
        export QWEN38_NUM_SHARDS="${QWEN38_NUM_SHARDS:-4}"
        export QWEN38_GPUS_PER_SHARD="${QWEN38_GPUS_PER_SHARD:-2}"
        export QWEN38_MEM_REQUEST="${QWEN38_MEM_REQUEST:-160Gi}"
        export QWEN38_MEM_LIMIT="${QWEN38_MEM_LIMIT:-256Gi}"
        ;;
    8xb300)
        export QWEN38_NUM_SHARDS="${QWEN38_NUM_SHARDS:-4}"
        export QWEN38_GPUS_PER_SHARD="${QWEN38_GPUS_PER_SHARD:-2}"
        export QWEN38_MEM_REQUEST="${QWEN38_MEM_REQUEST:-160Gi}"
        export QWEN38_MEM_LIMIT="${QWEN38_MEM_LIMIT:-256Gi}"
        ;;
    4xh100)
        export QWEN38_NUM_SHARDS="${QWEN38_NUM_SHARDS:-1}"
        export QWEN38_GPUS_PER_SHARD="${QWEN38_GPUS_PER_SHARD:-4}"
        export QWEN38_MEM_REQUEST="${QWEN38_MEM_REQUEST:-160Gi}"
        export QWEN38_MEM_LIMIT="${QWEN38_MEM_LIMIT:-256Gi}"
        ;;
    *)
        echo "error: unknown QWEN38_TOPOLOGY=${QWEN38_TOPOLOGY}; use 8xh100, 8xb200, 8xb300, or 4xh100" >&2
        exit 2
        ;;
esac

: "${K8S_NAMESPACE:?}" "${IMAGE_QWEN38:?}" "${QWEN38_WEIGHT_HASH:?}"
: "${PROMPTS_N:?}" "${DIM_BATCH:?}" "${MAX_SEQ_LEN:?}"

if [[ "${QWEN38_WEIGHT_HASH}" == *0000000000000000* ]]; then
    echo "warning: QWEN38_WEIGHT_HASH looks like the placeholder; update env.sh from GET /council/v2/identity" >&2
fi

VARS='${K8S_NAMESPACE} ${QWEN38_NUM_SHARDS} ${QWEN38_GPUS_PER_SHARD} ${QWEN38_MEM_REQUEST} ${QWEN38_MEM_LIMIT} ${IMAGE_QWEN38} ${QWEN38_WEIGHT_HASH} ${PROMPTS_N} ${DIM_BATCH} ${MAX_SEQ_LEN}'
RENDERED="$(envsubst "${VARS}" < "${K8S_DIR}/qwen38-job.yaml")"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${RENDERED}"
    exit 0
fi

kubectl -n "${K8S_NAMESPACE}" delete job jlens-qwen38 --ignore-not-found
echo "${RENDERED}" | kubectl apply -f -

echo ""
echo "Launched jlens-qwen38 (topology=${QWEN38_TOPOLOGY}: ${QWEN38_NUM_SHARDS} shards x ${QWEN38_GPUS_PER_SHARD} GPUs, N=${PROMPTS_N})."
echo "Monitor with:"
echo "  kubectl -n ${K8S_NAMESPACE} get pods -l app=jlens-qwen38 -w"
echo "  kubectl -n ${K8S_NAMESPACE} logs -l app=jlens-qwen38 --tail=40 -f"
