#!/usr/bin/env bash
# Writes the reference files of the Beat This! oracle (spec 8.3), and the safetensors form of its checkpoints.
#
# The reference is the package beat_this (CPJKU, MIT for the code and the weights) on the CPU in float32, one thread.
# It uses no GPU, so it may run beside the service. It holds about 2 GB of RAM.
#
#   BEATTHIS_CKPT_DIR=<folder of beat_this-final0.ckpt and beat_this-small0.ckpt> tools/beatthis_golden.sh
#
# REF_PYTHON names the Python of the reference venv, with no default. The venv holds torch 2.10.0+cpu, torchaudio
# 2.10.0+cpu, numpy 2.3.5, einops 0.8.1, rotary-embedding-torch 0.8.9, safetensors, soxr, and beat_this from
# https://github.com/CPJKU/beat_this/archive/b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c.zip. The files go to
# $BEATTHIS_GOLDEN_DIR, or to target/beatthis-golden, and the weights to $BEATTHIS_MODEL_DIR, or to
# target/beatthis-golden/weights. Do not commit them.
#
# The audio is synthetic (click tracks over seeded noise), at 22,050 Hz mono. The cases are click120_44 (30 s, one
# chunk), click97_34 (10 s, a live window), ramp100_130 (70 s, three chunks), silence (5 s) and short (0.5 s).
# The models are final0 and small0. Floats are f32 unless named.
#
# golden.safetensors:
#   mel.window                [1024]        the periodic Hann window
#   mel.filterbank            [513, 128]    the Slaney mel filters of torchaudio, 30 Hz to 11 kHz
#   mel.window64, mel.filterbank64  f64    the same, made by torchaudio with f64 as the default dtype
#   <case>.wave               [N]           the input samples
#   <case>.spect              [T, 128]      log1p(1000 * mel), T = N / 441 + 1 frames
#   <case>.spect64            [T, 128] f64  the same from the f64 window and filters: the gate of an f64 engine
#   <model>.<case>.beat       [T]           beat logits after the sum head, chunks joined keep_first
#   <model>.<case>.downbeat   [T]           downbeat logits
#   <model>.<case>.beat64, .downbeat64  f64 the logits of the f64 run (the sum head of beat_this rounds the beat to f32)
#   <model>.<case>.beats_s    [B]  f64      beat times in seconds, minimal postprocess
#   <model>.<case>.downbeats_s [D] f64      downbeat times in seconds, each moved onto a beat
#   join.<T>.beat, .downbeat  [T]           upstream's split and keep_first join of fake logits for T frames: the
#                                           value at a frame is 10000 * chunk + the global frame that it read
#   parts.stft_mag            [T, 513]      |STFT| of click120_44, reflect pad 512, divided by sqrt(1024)
#   parts.stft_mag64          [T, 513] f64  the same in f64
#   parts.stem                [32, 32, T']  final0, click120_44, chunk 0, after the stem (channels, freqs, frames)
#   parts.block{0,1,2}        [C, F, T']    after each frontend block: 64x16, 128x8, 256x4
#   parts.linear              [T', 512]     after the projection to the transformer
#   parts.layer{0..5}.attn    [T', 512]     the attention output of each layer, before its residual add
#   parts.layer{0..5}.ff      [T', 512]     the feed-forward output of each layer, before its residual add
#   parts.final_norm          [T', 512]     after the last RMSNorm
#   parts64.<name>            f64           each parts.* row from the f64 run, for a gate as tight as torch's own f32 error
#
# <weights>/<model>.safetensors holds the tensors of BeatThis.state_dict in f32, without num_batches_tracked. Its
# metadata holds the SHA-256 of the source checkpoint and its hyper_parameters. cases.json holds the versions, the
# true beat times of each case, the chunk starts, and the gap of f32 from a run in f64 (the sum head of beat_this
# rounds to f32, so that run is nearly f64, not all f64).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:?set REF_PYTHON to the Python of the reference venv}"
CKPT="${BEATTHIS_CKPT_DIR:?set BEATTHIS_CKPT_DIR to the folder of the Beat This! checkpoints}"
OUT="${BEATTHIS_GOLDEN_DIR:-target/beatthis-golden}"
WEIGHTS="${BEATTHIS_MODEL_DIR:-$OUT/weights}"
mkdir -p "$OUT" "$WEIGHTS"
export PYTHONDONTWRITEBYTECODE=1
nice -n 19 ionice -c3 "$PY" -I tools/beatthis_reference.py --ckpt-dir "$CKPT" --out "$OUT" --weights-out "$WEIGHTS"
