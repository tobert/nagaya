#!/usr/bin/env bash
# Step 5: Merge shard checkpoints into canonical D10 .jlens.safetensors, validate, and optionally upload to HF.
#
# Usage:
#   ./tools/jlens/scripts/05-merge-validate.sh inkling2 [runs|layers] [--dry-run]
#   ./tools/jlens/scripts/05-merge-validate.sh qwen38 [layers] [--dry-run]
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
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -*)
            echo "error: unknown flag ${arg}" >&2
            exit 2
            ;;
        *) POSITIONAL+=("${arg}") ;;
    esac
done

MODEL_KIND="${POSITIONAL[0]:-}"
VIEW="${POSITIONAL[1]:-auto}"

case "${MODEL_KIND}" in
    inkling2)
        export MERGE_MODEL_KIND="inkling2"
        export MERGE_IMAGE="${IMAGE_INKLING2}"
        export MERGE_WEIGHT_HASH="${INKLING2_WEIGHT_HASH}"
        if [ "${VIEW}" = "auto" ]; then
            export MERGE_VIEW="runs"
        elif [ "${VIEW}" = "runs" ] || [ "${VIEW}" = "layers" ]; then
            export MERGE_VIEW="${VIEW}"
        else
            echo "error: invalid view '${VIEW}' for inkling2 (expected runs or layers)" >&2
            exit 2
        fi
        export MERGE_OUT_NAME="inkling2-mini.${MERGE_VIEW}.jlens.safetensors"
        export MERGE_MEM_REQUEST="32Gi"
        export MERGE_MEM_LIMIT="64Gi"
        if [ -n "${HF_UPLOAD_ORG:-}" ]; then
            export UPLOAD_REPO="${HF_UPLOAD_ORG}/inkling2-mini-jlens"
        else
            export UPLOAD_REPO=""
        fi
        ;;
    qwen38)
        if [ "${VIEW}" != "auto" ] && [ "${VIEW}" != "layers" ]; then
            echo "error: qwen38 only supports view 'layers', got '${VIEW}'" >&2
            exit 2
        fi
        export MERGE_MODEL_KIND="qwen38"
        export MERGE_IMAGE="${IMAGE_QWEN38}"
        export MERGE_WEIGHT_HASH="${QWEN38_WEIGHT_HASH}"
        export MERGE_VIEW="layers"
        export MERGE_OUT_NAME="qwen38-flash-next.jlens.safetensors"
        export MERGE_MEM_REQUEST="48Gi"
        export MERGE_MEM_LIMIT="96Gi"
        if [ -n "${HF_UPLOAD_ORG:-}" ]; then
            export UPLOAD_REPO="${HF_UPLOAD_ORG}/qwen38-flash-next-jlens"
        else
            export UPLOAD_REPO=""
        fi
        ;;
    *)
        echo "usage: $0 <inkling2|qwen38> [runs|layers] [--dry-run]" >&2
        exit 2
        ;;
esac

export UPLOAD_PRIVATE="${UPLOAD_PRIVATE:-1}"
: "${K8S_NAMESPACE:?}" "${MERGE_IMAGE:?}" "${MERGE_WEIGHT_HASH:?}" "${PROMPTS_N:?}"

VARS='${K8S_NAMESPACE} ${MERGE_MODEL_KIND} ${MERGE_IMAGE} ${MERGE_VIEW} ${MERGE_OUT_NAME} ${MERGE_WEIGHT_HASH} ${PROMPTS_N} ${UPLOAD_REPO} ${UPLOAD_PRIVATE} ${MERGE_MEM_REQUEST} ${MERGE_MEM_LIMIT}'
RENDERED="$(envsubst "${VARS}" < "${K8S_DIR}/merge-job.yaml")"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "${RENDERED}"
    exit 0
fi

kubectl -n "${K8S_NAMESPACE}" delete job "jlens-merge-${MERGE_MODEL_KIND}" --ignore-not-found
echo "${RENDERED}" | kubectl apply -f -

echo ""
echo "Launched jlens-merge-${MERGE_MODEL_KIND} -> /data/out/${MERGE_OUT_NAME}"
if [ -n "${UPLOAD_REPO}" ]; then
    echo "Will upload validated lens to https://huggingface.co/${UPLOAD_REPO} (private=${UPLOAD_PRIVATE})"
fi
echo "Follow logs with:"
echo "  kubectl -n ${K8S_NAMESPACE} logs -l app=jlens-merge-${MERGE_MODEL_KIND} -f"
