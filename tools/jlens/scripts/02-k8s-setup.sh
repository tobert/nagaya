#!/usr/bin/env bash
# Step 2: Create the Kubernetes namespace, hf-token Secret, and shared PVC.
#
# Usage:
#   HF_TOKEN=hf_... ./tools/jlens/scripts/02-k8s-setup.sh [--dry-run]
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

if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: HF_TOKEN is not set. Export HF_TOKEN=hf_... before running this script." >&2
    exit 1
fi

CTX="$(kubectl config current-context 2>/dev/null || true)"
echo "kubectl context: ${CTX:-<none>}"
echo "namespace:       ${K8S_NAMESPACE}"
echo "storage class:   ${STORAGE_CLASS} (${PVC_ACCESS_MODE}, ${PVC_SIZE})"

PVC_YAML="$(envsubst '${K8S_NAMESPACE} ${PVC_ACCESS_MODE} ${STORAGE_CLASS} ${PVC_SIZE}' < "${K8S_DIR}/pvc.yaml")"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--- dry-run: would create namespace ${K8S_NAMESPACE}, secret hf-token, and apply:"
    echo "${PVC_YAML}"
    exit 0
fi

kubectl create namespace "${K8S_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "${K8S_NAMESPACE}" create secret generic hf-token \
    --from-literal=token="${HF_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -
echo "${PVC_YAML}" | kubectl apply -f -

echo "Waiting for PVC jlens-data to bind (if WaitForFirstConsumer, it binds when the first Job starts)..."
kubectl -n "${K8S_NAMESPACE}" get pvc jlens-data
