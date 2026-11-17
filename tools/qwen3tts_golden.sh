#!/usr/bin/env bash
# Writes the reference files of the speech oracle (spec 8.3) for Qwen3-TTS 12 Hz 1.7B VoiceDesign.
#
# The reference is the qwen_tts package, on the CPU in float32. Do not run it with a GPU job or with another
# big Python job: it holds about 10 GB of RAM.
#
#   QWEN3TTS_MODEL_DIR=<snapshot of Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign> tools/qwen3tts_golden.sh
#
# REF_PYTHON names the Python of the reference venv, with no default. The files go to $QWEN3TTS_GOLDEN_DIR,
# or to target/qwen3tts-golden. Do not commit them. The request: instruct "A calm female voice, clear and soft,
# unhurried.", text "The nagaya is awake.", language english, torch.manual_seed(1234), max_new_tokens 24.
# Each file carries the request, the processors and the versions in its safetensors metadata.
#
# F is the count of frames. P is the count of prompt positions. Every float tensor is f32. Ids and codes are i64.
# Channel-first tensors are [C, T]; "-1" in an id tensor is "absent".
#
# forced.safetensors: the speech oracle. The engine runs with `codes` in place of its samples.
#   instruct_ids        [Ni]            text ids of "<|im_start|>user\n{instruct}<|im_end|>\n"
#   input_ids           [Nt]            text ids of "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
#   prompt_text_ids     [P]             the text id of each prompt position, or -1
#   prompt_codec_ids    [P]             the codec id of each prompt position, or -1
#   talker_input_embeds [P, 2048]       the prompt embeddings: text_projection(text_embedding(text id))
#                                       + codec_embedding(codec id), with an absent id adding nothing
#   tts_pad_embed       [2048]          text_projection(text_embedding(tts_pad)); each frame input adds it
#   cb0_sampled         [F + 1]         each sampled codebook 0 id; the last one is 2150 (end of speech) when
#                                       the run ended, else a sample that got no frame
#   codes               [F, 16]         the codes of each frame, codebook 0 first
#   talker_logits       [F + 1, 3072]   raw codebook 0 logits: row 0 at the end of the prompt, row f + 1 after
#                                       frame f (the input of a frame is the sum of its 16 codec embeddings
#                                       plus tts_pad_embed)
#   talker_hidden       [F + 1, 2048]   the talker hidden state after the final norm, at the same steps; row f
#                                       is position 0 of the code predictor of frame f
#   cp_logits           [F, 15, 2048]   raw code predictor logits of codebooks 1..15 of each frame
#   audio_full          [F * 1920]      the codec decoder over all frames at once, 24 kHz, in [-1, 1]
#   audio_chunked       [F * 1920]      chunked_decode(chunk_size=4, left_context_size=25) of the same codes
#
# ops.safetensors: references for kernel bring-up, from the real weights.
#   talker.L0.prefill.{in,attn,attn_out,mlp_out,out}  [5, 2048]  talker layer 0 over prompt positions 0..4;
#                                       attn is before o_proj, attn_out after it, out is the residual after MLP
#   talker.L0.prefill.q                 [5, 16, 128]    q after q_norm and RoPE (theta 1e6, rotate_half pairs)
#   talker.L0.prefill.{k,v}             [5, 8, 128]     k after k_norm and RoPE; v
#   talker.L0.decode.*                  the same with 1 row: position 5, with the KV cache of positions 0..4
#   cp.codes            [16]            the codes of frame 0
#   cp.in_embeds        [16, 2048]      code predictor inputs: talker_hidden[0], the talker codec embedding of
#                                       code 0, and the code predictor embeddings of codes 1..14
#   cp.in_proj          [16, 1024]      after small_to_mtp_projection
#   cp.L{0..4}.out      [16, 1024]      the output of each of the 5 layers at each position
#   cp.norm             [16, 1024]      after the final norm
#   cp.logits           [15, 2048]      lm_head[g] at position g + 1 gives the logits of codebook g + 1
#   dec.codes           [4, 16]         the codes of frames 0..3
#   dec.quantizer       [512, 4]        the sum of the 16 codebook vectors, after output_proj
#   dec.pre_conv        [1024, 4]
#   dec.pre_transformer [1024, 4]       after the 8 layers, the norm and output_proj
#   dec.upsample0       [1024, 8]       after the transposed conv and the ConvNeXt block
#   dec.upsample1       [1024, 16]
#   dec.decoder_in      [1536, 16]      after the first causal conv
#   dec.block0..3       [768, 128], [384, 640], [192, 2560], [96, 7680]   after each decoder block
#   dec.final_snake     [96, 7680]
#   dec.final_conv      [1, 7680]       before the clamp
#   dec.audio           [7680]          after the clamp to [-1, 1]
#
# sampler.safetensors: 3 rows of real codebook 0 logits from the sampling run, through each processor.
#   sampler.step        [3]             the step of each row: 0, (F + 1) / 2 rounded down, F
#   sampler.history     [3, F]          the ids before the step (the repetition penalty history), -1 after
#   sampler.sampled     [3]             the id that the reference sampled at the step
#   sampler.logits      [3, 3072]       raw logits
#   sampler.after_penalty, after_min_new, after_suppress, after_temperature   [3, 3072]
#   sampler.scores      [3, 3072]       after top-k; equal to the scores of transformers
#   sampler.probs       [3, 3072]       softmax(scores)
#   cp_sampler.logits, cp_sampler.scores, cp_sampler.probs   [15, 2048]   the code predictor of frame 0
# The processors, in order (transformers 4.57): RepetitionPenalty(1.05), MinLength(2) and MinNewTokensLength(2)
# (both mask 2150 at steps 0 and 1), SuppressTokens(2048..3071 except 2150), Temperature(0.9), TopK(50). The
# code predictor: Temperature(0.9), TopK(50). The script checks this order. after_min_new is after both.
#
# chunked.safetensors: a second run with a longer text and the same seed, max_new_tokens 96. L is its frame count.
#   long_codes          [L, 16]         the codes, without the end-of-speech step
#   long_audio_full     [L * 1920]      the codec decoder over all frames at once
#   long_audio_chunked  [L * 1920]      chunked_decode(chunk_size=4, left_context_size=25); equal to the full
#                                       decode up to rounding until frame 28, the first chunk with a cut context
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${REF_PYTHON:?set REF_PYTHON to the Python of the reference venv}"
MODEL="${QWEN3TTS_MODEL_DIR:?set QWEN3TTS_MODEL_DIR to the Qwen3-TTS VoiceDesign checkpoint directory}"
OUT="${QWEN3TTS_GOLDEN_DIR:-target/qwen3tts-golden}"
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
nice -n 10 "$PY" -I tools/qwen3tts_reference.py --model "$MODEL" --out "$OUT"
