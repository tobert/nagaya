#!/usr/bin/env bash
# Step 1: Build and push the Inkling-2-Mini and Qwen3.8 J-Lens container images.
#
# Usage:
#   ./tools/jlens/scripts/01-build-push.sh [all|inkling2|qwen38] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/env.sh"

if [ ! -f "${ENV_FILE}" ]; then
    echo "error: ${ENV_FILE} not found. Copy env.sh.example to env.sh and set IMAGE_REGISTRY first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${ENV_FILE}"

TARGET="${1:-all}"
DRY_RUN=0
for arg in "$@"; do
    if [ "${arg}" = "--dry-run" ]; then
        DRY_RUN=1
    fi
done

CONTAINER_CLI="${CONTAINER_CLI:-docker}"
run_cmd() {
    echo "+ $*"
    if [ "${DRY_RUN}" -eq 0 ]; then
        "$@"
    fi
}

build_one() {
    local target="$1"
    local image="$2"
    run_cmd "${CONTAINER_CLI}" build \
        --target "${target}" \
        -f "${REPO_ROOT}/tools/jlens/Dockerfile" \
        -t "${image}" \
        "${REPO_ROOT}"
    run_cmd "${CONTAINER_CLI}" push "${image}"
}

case "${TARGET}" in
    all)
        build_one inkling2 "${IMAGE_INKLING2}"
        build_one qwen38 "${IMAGE_QWEN38}"
        ;;
    inkling2)
        build_one inkling2 "${IMAGE_INKLING2}"
        ;;
    qwen38)
        build_one qwen38 "${IMAGE_QWEN38}"
        ;;
    *)
        echo "usage: $0 [all|inkling2|qwen38] [--dry-run]" >&2
        exit 2
        ;;
esac
