#!/usr/bin/env bash
# Step 3: Launch the Inkling-2-Mini J-Lens IndexedJob (1 GPU per shard).
#
# Usage:
#   ./tools/jlens/scripts/03-run-inkling2.sh [--dry-run]
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

: "${K8S_NAMESPACE:?}" "${INKLING2_NUM_SHARDS:?}" "${IMAGE_INKLING2:?}" "${INKLING2_WEIGHT_HASH:?}"
: "${PROMPTS_N:?}" "${DIM_BATCH:?}" "${MAX_SEQ_LEN:?}"

if [[ "${INKLING2_WEIGHT_HASH}" == *0000000000000000* ]]; then
    echo "warning: INKLING2_WEIGHT_HASH looks like the placeholder; update env.sh from GET /council/v2/identity" >&2
fi

VARS='${K8S_NAMESPACE} ${INKLING2_NUM_SHARDS} ${IMAGE_INKLING2} ${INKLING2_WEIGHT_HASH} ${PROMPTS_N} ${DIM_BATCH} ${MAX_SEQ_LEN}'
RENDERED="$(envsubst "${VARS}" < "${K8S_DIR}/inkling2-job.yaml")"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${RENDERED}"
    exit 0
fi

kubectl -n "${K8S_NAMESPACE}" delete job jlens-inkling2 --ignore-not-found
echo "${RENDERED}" | kubectl apply -f -

echo ""
echo "Launched jlens-inkling2 (${INKLING2_NUM_SHARDS} shards x 1 GPU, N=${PROMPTS_N})."
echo "Monitor with:"
echo "  kubectl -n ${K8S_NAMESPACE} get pods -l app=jlens-inkling2 -w"
echo "  kubectl -n ${K8S_NAMESPACE} logs -l app=jlens-inkling2 --tail=40 -f"
