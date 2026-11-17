#!/usr/bin/env bash
# Writes the reference files of the embedding oracle (spec 8.3) for EmbeddingGemma 2, text inputs only.
#
# The reference is sentence-transformers on the CPU, in float32 from the bf16 weights, with eager attention. Do not
# run it with a GPU test or with another big Python job: it holds about 8 GB of RAM.
#
#   EMBEDDINGGEMMA2_MODEL_DIR=<snapshot of google/embeddinggemma-2> tools/embeddinggemma2_golden.sh
#
# REF_PYTHON names the Python of the reference venv, with no default. The venv holds torch 2.8.0+cpu,
# torchvision 0.23.0+cpu, transformers 5.21.0, sentence-transformers 6.1.0 and pillow 11.3.0. sentence-transformers
# loads the image and video processors of the model, so it needs torchvision and pillow even for text. The files go
# to $EMBEDDINGGEMMA2_GOLDEN_DIR, or to target/embeddinggemma2-golden. Do not commit them.
#
# The cases are ascii, code, cjk, emoji, space, empty, long (past the sliding window of 512 positions) and
# near_limit (2040 tokens with the prompt of "document"). The tasks are query, document, Clustering,
# Classification, SentenceSimilarity, CodeRetrieval, QuestionAnswering and FactChecking: names in the prompts of
# config_sentence_transformers.json. Floats are f32.
#
# golden.safetensors:
#   <case>.<task>.ids        [T]           i64 ids that the model saw: <bos>, the prompt and the text, <eos>
#   <case>.<task>.v768       [768]         the vector: mean of all T token embeddings, then normalized
#   <case>.<task>.v{512,256,128}           the first d values of v768, normalized again
#   <case>.<task>.bf16       [768]         the vector of a bf16 run, for the tasks query and document only
#   parts.embed              [T, 512]      the token embeddings times sqrt(512) in bf16 (22.625), case ascii, query
#   parts.ple                [T, 24, 512]  the per-layer inputs, from the embeddings alone
#   parts.layer{0..23}       [T, 512]      the hidden state after each layer, with its scalar
#   parts.final_norm         [T, 512]      after the final RMSNorm
#   parts.token_embeddings   [T, 768]      after embedding_projection
#   parts.pooled             [768]         the mean before the normalize
#   parts.vector             [768]         equal to ascii.query.v768
#
# cases.json and the metadata of golden.safetensors hold the versions, the prompts, the ids of the short cases, the
# difference of a padded batch from one input at a time, and the difference of bf16 from f32.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:?set REF_PYTHON to the Python of the reference venv}"
MODEL="${EMBEDDINGGEMMA2_MODEL_DIR:?set EMBEDDINGGEMMA2_MODEL_DIR to the EmbeddingGemma 2 snapshot directory}"
OUT="${EMBEDDINGGEMMA2_GOLDEN_DIR:-target/embeddinggemma2-golden}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nice -n 19 ionice -c3 "$PY" -I tools/embeddinggemma2_reference.py --model "$MODEL" --out "$OUT"
