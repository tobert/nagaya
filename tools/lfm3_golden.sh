#!/usr/bin/env bash
# Writes the reference files of the LFM3 oracle (spec 8.3, 11.11) for LFM3-Omni-2B and LFM3-1.7B.
#
# The reference is transformers with the checkpoint's own code (trust_remote_code), on the CPU in float32, with
# eager attention. Do not run it with a GPU job or with another big Python job: it holds about 20 GB of RAM.
#
#   LFM3_OMNI_MODEL_DIR=<snapshot of LiquidAI/LFM3-Omni-2B> LFM3_MODEL_DIR=<snapshot of LiquidAI/LFM3-1.7B> \
#       tools/lfm3_golden.sh [--stages text,vision,audio,speech]
#
# The pinned revisions are LFM3-Omni-2B 21d75123f86c1f7ffb4bef1ea202710c4e7ebe67 and LFM3-1.7B
# 0dc4268b561499c648c7ba7c9466bf3d8d01517a (spec 11.11). The script refuses another revision.
#
# REF_PYTHON overrides the python (default ~/.venvs/nagaya-lfm3-ref/bin/python). The venv holds Python 3.13,
# torch 2.14.1+cpu, torchvision 0.29.2+cpu, torchcodec 0.17.0+cpu, transformers 5.20.2, tokenizers 0.23.3,
# accelerate 1.15.0, numpy 2.5.4, pillow 13.0.0, librosa 1.0.0, soundfile 0.14.0 and safetensors 0.8.0. The files go
# to $LFM3_GOLDEN_DIR, or to target/lfm3-golden. Do not commit them. transformers copies the remote code into
# $OUT/.modules (HF_MODULES_CACHE), not into ~/.cache.
#
# Every float tensor is f32. Ids, codes and grids are i64; masks and kinds are u8. Each safetensors file carries
# its inputs and the versions in its metadata. P is a count of prompt positions, F a count of speech frames.
#
# lfm3.oracle.json, lfm3-omni.oracle.json: the backbone gate, in the format of tools/oracle_fixture.py. Five
#   prompts (plain, code, chat, long, longer); for each, the ids, the top-64 logits and the logsumexp of the last
#   position. Each id list starts with <|startoftext|>. "chat" is the model's own template, rendered by
#   transformers with its defaults. The prose of long and longer is the prose of the Qwen3.8 fixture.
# lfm3.layers.safetensors, lfm3-omni.layers.safetensors: the prompt "plain", for the bring-up of the backbone.
#   ids [P], embed [P, 2048], layer{0..23} [P, 2048] (the residual after each layer), final_norm [P, 2048],
#   logits_last [128000].
# lfm3-omni.media.json: the top-64 logits of the last position of three media prompts (image_small, image_mid,
#   audio_chirp), with their ids after the placeholder expansion. Same format as the oracle.
# vision.safetensors, image_small.png, image_mid.png: two synthetic RGB images. For <n> in small, mid:
#   <n>.png_rgb [H, W, 3] u8         the decoded PNG
#   <n>.resized_rgb [h, w, 3] u8     torchvision bicubic with antialias, on the u8 image, to the size of the grid
#   <n>.pixel_values [N, 1536]       the processor's patches, in merge-block order; N = grid_h * grid_w
#   <n>.grid_thw [1, 3]              (1, grid_h, grid_w); the resized size is 16 * grid_h by 16 * grid_w
#   <n>.patches [N, 1024]            after the patchifier and the embedding norm
#   <n>.tower [N, 1024]              the output of the vision tower
#   <n>.rows [N / 4, 2048]           the projector rows that replace the <|image|> placeholders
#   <n>.ids [P]                      the prompt ids with the expanded placeholders
# audio_in.safetensors, chirp.wav: a synthetic 1.5 s mono WAV at 16 kHz in 16 bits.
#   pcm [n] i16, samples [n]         the samples, and the samples divided by 32768
#   mel [128, T]                     the log-mel features of the processor; T mel frames
#   mel_filters [201, 128]           the slaney mel filter bank of the processor (n_fft 400)
#   downsample [ceil(T / 8), 768]    after the three stride-2 convolutions
#   encoder [ceil(T / 8), 768]       the output of the audio encoder
#   rows [ceil(T / 8), 2048]         the projector rows that replace the <|audio|> placeholders
#   ids [P]                          the prompt ids with the expanded placeholders
# speech.safetensors: a TTS run and an interleaved run of Omni, seed 1234, the audio sampler of
#   generation_config.json, greedy text. A run keeps its end frame (all codes 2016) when it ended.
#   tts.prompt_ids [P]               the prompt, ending with <|audio_start|>
#   tts.sequence [P + F + 1]         the prompt, one <|audio_out|> for each frame, then <|im_end|>
#   tts.codes [F, 12]                the codes of each frame, the end frame last when the run ended
#   tts.hidden [F + 1, 2048]         the backbone output (after the final norm) of a teacher-forced pass, at the
#                                    position before each frame; row F follows the last frame
#   tts.end_logits [128000]          the text logits after the last frame
#   tts.audio [S]                    the codec decode of the frames without the end frame, 24 kHz
#   depth.hidden [4, 2048], depth.codes [4, 12]   the first 4 frames of the TTS run
#   depth.order [4, 12]              a fixed reveal order for each frame
#   depth.mask{0..3} [4, 12] u8      1 is masked; 0, 4, 8 and 11 codebooks revealed in the order
#   depth.logits{0..3} [4, 12, 2017] the depth decoder logits of every codebook, with the codes forced
#   depth.in0, depth.trunk0 [4, 12, 1024]  the input and the output of the depth trunk for mask0
#   depth.audio_out_embeds [4, 2048] the summed codebook embedding of each frame (get_audio_out_embeds)
#   codec.codes [20, 12]             the first 20 speech frames of the TTS run
#   codec.audio_full [38400]         codec.decode of the 20 frames at once
#   codec.audio_frames [38400]       the streaming decode, one frame at a time
#   codec.audio_chunks4 [38400]      the streaming decode, 4 frames at a time
#   mixed.prompt_ids [P]             the prompt, ending with <|mixed_start|>
#   mixed.sequence [P + G]           the prompt and the G generated slots, <|im_end|> last
#   mixed.kinds [G] u8               0 for a text slot, 1 for an audio slot
#   mixed.codes [F, 12]              the codes of the audio slots in order
#   mixed.hidden [G + 1, 2048]       the teacher-forced backbone output at the position before each slot
#   mixed.text_top_ids [G, 8]        the 8 best text ids at each slot
#   mixed.audio [S]                  the codec decode of the frames without the end frame
# lfm3.control.safetensors (--stages control, not in the default): the control vector gate of LFM3-1.7B (11.12).
#   Each prompt is one user message with the generation prompt and thinking off. L is 24 layers, W is 2048.
#   positive{0..3}.ids [P], positive{0..3}.rows [L, W]   the residual after each layer at the last position
#   negative{0..3}.ids [P], negative{0..3}.rows [L, W]
#   vector [L, W]                    mean(positive rows) - mean(negative rows); the sums in f64 in prompt order
#   test.ids [P]                     the test prompt
#   plain.logits [128000], plain.rows [L, W]   the test prompt with no control item
#   <op>.logits [128000], <op>.rows [L, W]     the test prompt with one item: add 4, ablate 1, clamp 8
#   <op>.decode_ids [8]              the greedy decode of 8 tokens with the item, for plain and each op
#   <op>.decode_top_ids [8, 64] i64, <op>.decode_top [8, 64]   the top-64 logits of each decode step
# lfm3-omni.control.safetensors (--stages control-omni, not in the default): the same gate for LFM3-Omni-2B, with the
#   same prompts and strengths, through Omni's language model and its tokenizer. Its vector acts on the text passes.
# lfm3.project.safetensors (--stages project, not in the default): the project gate of LFM3-1.7B (docs/lab.md). It
#   reads lfm3.control.safetensors, so run the control stage first. For each layer l in 0, 12 and 23, u is the weight
#   of the final norm times vector[l], with no division by its root mean square, and the scores are the head of u.
#   vector [L, W]                    the vector of lfm3.control.safetensors
#   layers [3] i64                   the layers of the gate
#   layer<l>.scores [128000]         the score of each row of the head
#   layer<l>.up_ids [64] i64, layer<l>.down_ids [64] i64   the ids of the 64 largest and the 64 smallest scores
# facts.json: the measured facts for spec 11.11: sizes, counts, the codec checks and the run patterns.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:-$HOME/.venvs/nagaya-lfm3-ref/bin/python}"
OMNI="${LFM3_OMNI_MODEL_DIR:?set LFM3_OMNI_MODEL_DIR to the LFM3-Omni-2B snapshot directory}"
TEXT="${LFM3_MODEL_DIR:?set LFM3_MODEL_DIR to the LFM3-1.7B snapshot directory}"
OUT="${LFM3_GOLDEN_DIR:-target/lfm3-golden}"
check_revision() {
    if [ "$(basename "$(realpath "$1")")" != "$2" ]; then
        echo "lfm3_golden: $1 is not the pinned revision $2" >&2
        exit 1
    fi
}
check_revision "$OMNI" 21d75123f86c1f7ffb4bef1ea202710c4e7ebe67
check_revision "$TEXT" 0dc4268b561499c648c7ba7c9466bf3d8d01517a
mkdir -p "$OUT/.modules"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_MODULES_CACHE="$OUT/.modules"
nice -n 19 ionice -c3 "$PY" -I tools/lfm3_reference.py --omni "$OMNI" --text "$TEXT" --out "$OUT" "$@"
