#!/usr/bin/env bash
# Writes the HF reference files that the golden tests of nagaya-mk-gemma4-bmg read.
#
# The token ids are fixed in the crate's tests/fixtures. The first Gemma 4 engine made them from its prompts.
# Do not run it with a GPU job.
#
#   GEMMA4_MODEL_DIR=<checkpoint> REF_PYTHON=<reference venv>/bin/python tools/gemma4_golden.sh
#
# The files go to $GEMMA4_GOLDEN_DIR, or to target/gemma4-golden. Do not commit them.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:?set REF_PYTHON to a python with torch and transformers}"
MODEL="${GEMMA4_MODEL_DIR:?set GEMMA4_MODEL_DIR to the Gemma 4 checkpoint directory}"
OUT="${GEMMA4_GOLDEN_DIR:-target/gemma4-golden}"
FIX=crates/nagaya-mk-gemma4-bmg/tests/fixtures
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
nice -n 10 "$PY" -I tools/gemma4_reference.py --model "$MODEL" --ids "$(cat "$FIX/golden_chat_ids.txt")" \
    --dump-layers 0,5 --expert-sweep "$(seq -s, 0 29)" --out "$OUT/chat.safetensors"
nice -n 10 "$PY" -I tools/gemma4_reference.py --model "$MODEL" --ids "$(cat "$FIX/golden_long_ids.txt")" \
    --dump-routes --out "$OUT/long.safetensors"
