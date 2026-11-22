#!/usr/bin/env bash
# Writes the reference files of the Inkling 2 oracle (spec 8.3) for Inkling 2 Mini and the variable-loop model.
#
# The reference is transformers at commit 046bdb8cc2, on the CPU, with every sum in f32 and the weights as the
# checkpoint holds them. One model holds about 74 GiB of RAM, and the tool loads one at a time. Do not run it with a
# GPU job or with another big Python job.
#
#   INKLING2_MODEL_DIR=<snapshot of thinkingmachines/Inkling-2-Mini> \
#   INKLING2_VARLOOP_MODEL_DIR=<snapshot of thinkingmachines/TML-d24-varloop-research> \
#       tools/inkling2_golden.sh [--stages text,layers,cycles,decode,vision] [--quant]
#
# The option --quant runs the quantized mode: each expert quantizes its input rows as the checkpoint requires (the rules
# of tools/inkling2_quant.py, and the layer split of hf_quant_config.json). It writes the same files with the infix
# ".quant" (inkling2-mini.quant.oracle.json, inkling2-mini.quant.layers.safetensors, quant.facts.json, and so on), so
# the f32 files stay. Run the script twice, with and without the option, for both sets. Each run takes the time and the
# RAM of the f32 run.
#
# The pinned revisions are Inkling-2-Mini ed40e7da625f594fc9c4990f5f66be24873c22d7 and TML-d24-varloop-research
# 657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2. The script refuses another revision.
#
# REF_PYTHON names the Python of the reference venv, with no default. The venv holds Python 3.13, torch 2.15.0+cpu,
# transformers 5.24.0.dev0 from git commit 046bdb8cc2, and safetensors. The stage "vision" also needs torchvision
# 0.29.2+cpu and pillow 13.0.0, because the image processor reads them. The files go to $INKLING2_GOLDEN_DIR, or to
# target/inkling2-golden. Do not commit them.
#
# A cycle is one walk through layers 6 to 17. A run is one execution of one layer. At 3 cycles a position has 48 runs:
# layers 0 to 5, then layers 6 to 17 three times, then layers 18 to 23. P is a count of prompt positions.
#
# <model>.oracle.json, for <model> in inkling2-mini, inkling2-varloop: the gate of the text model, in the format of
#   tools/oracle_fixture.py. Five prompts (plain, code, chat, long, longer); for each, the ids, the top-64 logits and
#   the logsumexp of the last position. No prompt has a start token, because the template writes none. "chat" is the
#   model's own template with the effort "none", rendered by transformers.
# inkling2-mini.layers.safetensors: the prompt "plain" at 3 cycles, for the bring-up of the engine.
#   ids [P], run00 .. run47 [P, 2304] (the row after each run, in the order of the runs), message00 .. message23
#   [P, 4, 2304] (the message of each layer of cycles 2 and 3), gate00 .. gate95.<gate> [P, 2304] (the output of each
#   message gate, in call order), final_norm [P, 2304], logits_last [200058].
# <model>.cycles.json: the prompts "plain" and "chat" at 1 to 5 cycles, in the format of the oracle, with "cycles".
# <model>.decode.safetensors: 8 greedy tokens after "chat" with the cache of the reference.
#   ids [P] i64, tokens [8] i64, logits [8, 200058] (the logits of the position before each token, with no cache)
# facts.json: the measured facts: the prompt sizes, the gap of the cache, the greedy text.
#
# The stage "vision" runs Mini only, and it writes new files only. Its facts go to vision.facts.json. T is a count of
# image tokens, and each one is a square of 40 px, or 2 x 2 patches of 20 px. N = 4 T is a count of patches.
# inkling2-vision-image0.png, inkling2-vision-image1.png: two synthetic RGB images of 213 x 118 and 637 x 479 px, from
#   fixed seeds. No side is a multiple of 40, so the processor pads each one.
# inkling2-mini.vision.safetensors, for each image i in 0 and 1:
#   image<i>.rgb [H, W, 3] i64 (the bytes of the PNG), image<i>.pixel_values [T, 40, 40, 3] (the output of the
#   processor), image<i>.grid [3] i64 (1, rows, columns of image tokens), image<i>.patch_embed [N, 1024] (the patch
#   embedding alone), image<i>.embeddings [N + 1, 1024] (the class row, then the patches with their position
#   embedding), image<i>.pre_layernorm [N + 1, 1024], image<i>.tower [N + 1, 1024] (the output of the last layer),
#   image<i>.projector [T, 2304] (the rows that replace the placeholders). Image 0 also has image0.layer00 ..
#   image0.layer22 [N + 1, 1024], the rows after each layer of the tower. The patches go in the order of the merge
#   blocks. prompt.ids [P] i64: a chat with both images and a question, with the placeholders expanded.
#   prompt.logits_last [200058]: the logits of its last position at 3 cycles.
# inkling2-mini.vision.json: that prompt in the format of the oracle, with the top-64 logits and the logsumexp.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:?set REF_PYTHON to the Python of the reference venv}"
MINI="${INKLING2_MODEL_DIR:?set INKLING2_MODEL_DIR to the Inkling-2-Mini snapshot directory}"
VARLOOP="${INKLING2_VARLOOP_MODEL_DIR:?set INKLING2_VARLOOP_MODEL_DIR to the snapshot directory of the research model}"
OUT="${INKLING2_GOLDEN_DIR:-target/inkling2-golden}"
check_revision() {
    if [ "$(basename "$(realpath "$1")")" != "$2" ]; then
        echo "inkling2_golden: $1 is not the pinned revision $2" >&2
        exit 1
    fi
}
check_revision "$MINI" ed40e7da625f594fc9c4990f5f66be24873c22d7
check_revision "$VARLOOP" 657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nice -n 19 ionice -c3 "$PY" -I tools/inkling2_reference.py --mini "$MINI" --varloop "$VARLOOP" --out "$OUT" "$@"
