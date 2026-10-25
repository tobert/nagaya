#!/usr/bin/env python3
"""f32 CPU reference for Qwen3-TTS 12 Hz 1.7B VoiceDesign: the speech oracle (spec 8.3) and op references.

Run it through tools/qwen3tts_golden.sh. The direct command is:

  HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 ~/.venvs/tts-proto/bin/python -I \
      tools/qwen3tts_reference.py --model "$QWEN3TTS_MODEL_DIR" --out target/qwen3tts-golden

The script uses the `qwen_tts` package of ~/.venvs/tts-proto (torch CPU, transformers 4.57). It loads the
checkpoint in float32 with eager attention. It runs on the CPU only. It writes four safetensors files into
--out, and the tensor names, shapes and meanings are in tools/qwen3tts_golden.sh.

  forced.safetensors   the speech oracle: the prompt, the sampled codes, the logits, the hidden states, the audio
  ops.safetensors      kernel bring-up: talker layer 0, the code predictor of frame 0, the codec decoder stages
  sampler.safetensors  the processed probabilities of 3 rows of codebook 0 logits
  chunked.safetensors  a longer run, to show what the left context of the chunked decode changes

The script stops with an AssertionError when a self check fails. The checks:

  - the prompt pairs (text id, codec id) give the prompt embeddings of the reference;
  - the teacher-forced pass gives the hidden states of the sampling run within 1e-4, and the logits of codebook 0
    and of the code predictor within 1e-3;
  - the talker RoPE (3 sections, interleaved) is plain RoPE, because the 3 position rows are equal;
  - the manual parts of talker layer 0 give the output of the layer in the model;
  - one forward of the code predictor over 16 positions gives the logits of the stepwise run;
  - the stages of the codec decoder give the output of the decoder;
  - the full decode gives the audio of generate_voice_design;
  - the chunked decode gives the full decode until the first chunk with a cut left context (frame 28);
  - the explicit sampler pipeline gives the processed scores of transformers at each step, for the talker and for
    the code predictor;
  - each sampled code has a probability above zero in the pipeline.
"""

import argparse
import os
import sys
import time

import torch
from safetensors.torch import save_file
from transformers.cache_utils import DynamicCache

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import (
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)

FRAME_SAMPLES = 1920  # 24 kHz / 12.5 Hz


def check(ok, what):
    """Stop with the name of the self check that failed."""
    if not ok:
        raise AssertionError(f"self check failed: {what}")


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def close(a, b, tol, what):
    d = max_abs(a, b)
    print(f"check {what}: max abs {d:.3g} (tolerance {tol:g})", file=sys.stderr)
    check(a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
    check(d <= tol, f"{what}: max abs {d:.3g} > {tol:g}")


class Capture:
    """Wraps the generate calls of the talker and the code predictor and keeps what they return."""

    def __init__(self, talker):
        self.talker_kwargs = None
        self.talker_result = None
        self.cp_results = []
        self.processors = {}
        self.processor_objects = {}
        tgen = talker.generate
        cgen = talker.code_predictor.generate

        def talker_generate(**kw):
            self.talker_kwargs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kw.items()}
            kw.update(output_logits=True, output_scores=True)
            self.talker_result = tgen(**kw)
            return self.talker_result

        def cp_generate(**kw):
            kw.update(output_logits=True, output_scores=True)
            r = cgen(**kw)
            self.cp_results.append(r)
            return r

        talker.generate = talker_generate
        talker.code_predictor.generate = cp_generate
        for name, mod in (("talker", talker), ("code_predictor", talker.code_predictor)):
            orig = mod._get_logits_processor

            def wrapped(*a, _orig=orig, _name=name, **kw):
                procs = _orig(*a, **kw)
                self.processors[_name] = [type(p).__name__ for p in procs]
                self.processor_objects[_name] = list(procs)
                return procs

            mod._get_logits_processor = wrapped


# The sampler of codebook 0, in the order of transformers 4.57 GenerationMixin._get_logits_processor:
#   1. RepetitionPenaltyLogitsProcessor(1.05): for each id in the history (the codebook 0 ids sampled before
#      this step, not the prompt), a score s becomes s / 1.05 when s > 0, else s * 1.05. An id that occurs
#      twice is penalized once.
#   2. MinLengthLogitsProcessor(2, eos 2150) and 3. MinNewTokensLengthLogitsProcessor(skip 0, 2, eos 2150):
#      both set the score of 2150 to -inf at steps 0 and 1. generate() passes min_new_tokens=2, and
#      transformers also makes min_length = 2 from it, because the input ids of an embeds prompt are empty.
#   4. SuppressTokensLogitsProcessor: the ids 2048..3071 except 2150 become -inf.
#   5. TemperatureLogitsWarper(0.9): scores / 0.9.
#   6. TopKLogitsWarper(50): scores below the 50th largest become -inf (ties at the 50th stay).
# top_p is 1.0, so transformers adds no TopPLogitsWarper. Then probs = softmax(scores) and torch.multinomial.
# The code predictor has only steps 4 and 5 (temperature 0.9, top-k 50, no penalty, no suppress).
def sampler_stages(logits, history, step, *, vocab, eos, penalty, min_new, temperature, top_k):
    s = logits.clone().float()
    if len(history):
        ids = torch.tensor(sorted(set(history)), dtype=torch.long)
        g = s[ids]
        s[ids] = torch.where(g < 0, g * penalty, g / penalty)
    after_penalty = s.clone()
    if step < min_new:
        s[eos] = -float("inf")
    after_min_new = s.clone()
    suppress = torch.tensor([i for i in range(vocab - 1024, vocab) if i != eos], dtype=torch.long)
    s[suppress] = -float("inf")
    after_suppress = s.clone()
    s = s / temperature
    after_temperature = s.clone()
    kth = torch.topk(s, top_k).values[-1]
    s = s.masked_fill(s < kth, -float("inf"))
    return {
        "after_penalty": after_penalty,
        "after_min_new": after_min_new,
        "after_suppress": after_suppress,
        "after_temperature": after_temperature,
        "scores": s,
        "probs": torch.softmax(s, dim=-1),
    }


def cp_sampler(logits, *, temperature, top_k):
    s = logits.float() / temperature
    kth = torch.topk(s, top_k).values[-1]
    return s.masked_fill(s < kth, -float("inf"))


def prompt_pairs(model, input_id, instruct_id, language):
    """The (text id, codec id) pair of each prompt position of generate() in non-streaming mode; -1 is absent."""
    cfg = model.config
    tc = cfg.talker_config
    text, codec = [], []
    for t in instruct_id[0].tolist():
        text.append(t)
        codec.append(-1)
    for t in input_id[0, :3].tolist():  # <|im_start|>assistant\n
        text.append(t)
        codec.append(-1)
    tags = [tc.codec_think_id, tc.codec_think_bos_id, tc.codec_language_id[language], tc.codec_think_eos_id]
    for c in tags:
        text.append(cfg.tts_pad_token_id)
        codec.append(c)
    text.append(cfg.tts_bos_token_id)
    codec.append(tc.codec_pad_id)
    for t in input_id[0, 3:-5].tolist():  # the text, without <|im_end|>\n<|im_start|>assistant\n
        text.append(t)
        codec.append(tc.codec_pad_id)
    text.append(cfg.tts_eos_token_id)
    codec.append(tc.codec_pad_id)
    text.append(cfg.tts_pad_token_id)
    codec.append(tc.codec_bos_id)
    return torch.tensor(text, dtype=torch.long), torch.tensor(codec, dtype=torch.long)


@torch.no_grad()
def embed_pairs(talker, text, codec):
    out = []
    for t, c in zip(text.tolist(), codec.tolist()):
        e = torch.zeros(1, 1, talker.config.hidden_size)
        if t >= 0:
            e = e + talker.text_projection(talker.get_text_embeddings()(torch.tensor([[t]])))
        if c >= 0:
            e = e + talker.get_input_embeddings()(torch.tensor([[c]]))
        out.append(e)
    return torch.cat(out, dim=1)


@torch.no_grad()
def code_predictor_forced(cp, talker_hidden, cb0_embed, codes):
    """Logits of codebooks 1..15 for one frame, fed with the given codes. talker_hidden: [1, 1, 2048]."""
    cache = DynamicCache()
    x = cp.small_to_mtp_projection(torch.cat((talker_hidden, cb0_embed), dim=1))
    h = cp.model(inputs_embeds=x, past_key_values=cache, use_cache=True).last_hidden_state
    logits = [cp.lm_head[0](h)[:, -1]]
    for g in range(1, 15):
        e = cp.small_to_mtp_projection(cp.model.codec_embedding[g - 1](codes[None, g : g + 1]))
        h = cp.model(inputs_embeds=e, past_key_values=cache, use_cache=True).last_hidden_state
        logits.append(cp.lm_head[g](h)[:, -1])
    return torch.cat(logits, dim=0)  # [15, 2048]


@torch.no_grad()
def frame_embed(talker, codes, tts_pad):
    """The talker input of one frame: the sum of its 16 codec embeddings, plus the projected tts_pad."""
    cp = talker.code_predictor
    parts = [talker.get_input_embeddings()(codes[None, :1])]
    parts += [cp.get_input_embeddings()[i](codes[None, i + 1 : i + 2]) for i in range(15)]
    return torch.cat(parts, dim=1).sum(1, keepdim=True) + tts_pad


@torch.no_grad()
def forced_run(talker, prompt, codes, tts_pad):
    """Teacher-forced talker and code predictor over the sampled codes."""
    cache = DynamicCache()
    h = talker.model(inputs_embeds=prompt, past_key_values=cache, use_cache=True).last_hidden_state[:, -1:]
    hiddens, logits, cp_logits = [h[0, 0]], [talker.codec_head(h)[0, 0]], []
    for f in range(codes.shape[0]):
        c = codes[f]
        cb0 = talker.get_input_embeddings()(c[None, :1])
        cp_logits.append(code_predictor_forced(talker.code_predictor, h, cb0, c))
        x = frame_embed(talker, c, tts_pad)
        h = talker.model(inputs_embeds=x, past_key_values=cache, use_cache=True).last_hidden_state
        hiddens.append(h[0, 0])
        logits.append(talker.codec_head(h)[0, 0])
    return torch.stack(hiddens), torch.stack(logits), torch.stack(cp_logits)


@torch.no_grad()
def talker_layer_parts(talker, layer_idx, x, pos, past=None):
    """Talker layer `layer_idx` by hand. x: [1, S, 2048]; pos: [S]; past: (k, v) after RoPE, or None."""
    layer = talker.model.layers[layer_idx]
    a = layer.self_attn
    s = x.shape[1]
    cos, sin = talker.model.rotary_emb(x, pos.view(1, 1, -1).expand(3, 1, -1))
    h = layer.input_layernorm(x)
    shape = (1, s, -1, a.head_dim)
    q = a.q_norm(a.q_proj(h).view(shape)).transpose(1, 2)
    k = a.k_norm(a.k_proj(h).view(shape)).transpose(1, 2)
    v = a.v_proj(h).view(shape).transpose(1, 2)
    sections, interleaved = a.rope_scaling["mrope_section"], a.rope_scaling["interleaved"]
    q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, sections, interleaved)
    # The 3 position rows are equal, so the sectioned RoPE must be plain NeoX RoPE (rotate_half).
    c1, s1 = cos[0].unsqueeze(1), sin[0].unsqueeze(1)
    q0 = a.q_norm(a.q_proj(h).view(shape)).transpose(1, 2)
    q_plain = q0 * c1 + rotate_half(q0) * s1
    close(q, q_plain, 0.0, f"talker L{layer_idx} sectioned RoPE equals plain RoPE")
    kk, vv = (k, v) if past is None else (torch.cat((past[0], k), 2), torch.cat((past[1], v), 2))
    t = kk.shape[2]
    qpos = torch.arange(t - s, t)[:, None]
    kpos = torch.arange(t)[None, :]
    mask = torch.where(kpos <= qpos, 0.0, float("-inf"))[None, None]
    attn, _ = eager_attention_forward(a, q, kk, vv, mask, scaling=a.scaling)
    attn = attn.reshape(1, s, -1)
    o = a.o_proj(attn)
    r1 = x + o
    mlp = layer.mlp(layer.post_attention_layernorm(r1))
    out = r1 + mlp
    sq = lambda z: z[0]
    return {
        "in": sq(x),
        "q": q[0].transpose(0, 1).contiguous(),  # [S, 16, 128] after q_norm and RoPE
        "k": k[0].transpose(0, 1).contiguous(),  # [S, 8, 128] after k_norm and RoPE
        "v": v[0].transpose(0, 1).contiguous(),  # [S, 8, 128]
        "attn": sq(attn),  # [S, 2048] before o_proj
        "attn_out": sq(o),  # [S, 2048] after o_proj
        "mlp_out": sq(mlp),
        "out": sq(out),
    }, (k, v)


@torch.no_grad()
def layer_outputs(model, layers, fn):
    """Runs fn() and returns the output of each module in `layers` (forward hooks)."""
    outs = [None] * len(layers)
    hooks = []
    for i, m in enumerate(layers):
        def hook(_m, _inp, out, i=i):
            outs[i] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        hooks.append(m.register_forward_hook(hook))
    try:
        res = fn()
    finally:
        for h in hooks:
            h.remove()
    return outs, res


@torch.no_grad()
def decoder_stages(dec, codes):
    """The codec decoder by stage. codes: [1, 16, T]. Each stage is channels first, [C, T']."""
    st = {}
    h = dec.quantizer.decode(codes)
    st["quantizer"] = h[0]
    h = dec.pre_conv(h)
    st["pre_conv"] = h[0]
    h = dec.pre_transformer(inputs_embeds=h.transpose(1, 2)).last_hidden_state.permute(0, 2, 1)
    st["pre_transformer"] = h[0]
    for i, blocks in enumerate(dec.upsample):
        for b in blocks:
            h = b(h)
        st[f"upsample{i}"] = h[0]
    names = ["decoder_in", "block0", "block1", "block2", "block3", "final_snake", "final_conv"]
    check(len(dec.decoder) == len(names), f"the decoder has {len(dec.decoder)} parts, not {len(names)}")
    for name, b in zip(names, dec.decoder):
        h = b(h)
        st[name] = h[0]
    st["audio"] = h.clamp(min=-1, max=1)[0, 0]
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the checkpoint directory (with speech_tokenizer/)")
    ap.add_argument("--out", required=True, help="the output directory")
    ap.add_argument("--instruct", default="A calm female voice, clear and soft, unhurried.")
    ap.add_argument("--text", default="The nagaya is awake.")
    ap.add_argument("--language", default="english")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-frames", type=int, default=24, help="max_new_tokens of the talker")
    ap.add_argument(
        "--long-text",
        default="The nagaya is awake. It hosts our engines, one process for each machine, and it keeps every card"
        " busy through the night.",
        help="the text of the run that checks the chunked decode",
    )
    ap.add_argument("--long-frames", type=int, default=96, help="max_new_tokens of the long run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    t0 = time.time()

    tts = Qwen3TTSModel.from_pretrained(args.model, dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    model = tts.model
    talker = model.talker
    tc = model.config.talker_config
    dec = model.speech_tokenizer.model.decoder
    for name, mod in (("talker", talker), ("decoder", dec)):
        dt = {p.dtype for p in mod.parameters()}
        check(dt == {torch.float32}, f"{name} parameters are float32, not {dt}")
    check(talker.config._attn_implementation == "eager", "talker attention is eager")
    print(f"loaded in {time.time() - t0:.1f} s", file=sys.stderr)

    # 1. The sampling run, through the public call.
    cap = Capture(talker)
    t1 = time.time()
    torch.manual_seed(args.seed)
    wavs, sr = tts.generate_voice_design(
        text=args.text, instruct=args.instruct, language=args.language, max_new_tokens=args.max_frames
    )
    gen_seconds = time.time() - t1
    check(sr == 24000, f"sample rate 24000, not {sr}")
    res = cap.talker_result
    seq = res.sequences[0]  # every sampled codebook 0 id, the last one included
    steps = seq.shape[0]
    codes = torch.stack([hid[-1][0] for hid in res.hidden_states if hid[-1] is not None])  # [F, 16]
    frames = codes.shape[0]
    check(frames == steps - 1, f"frames {frames} = steps {steps} - 1")
    check(torch.equal(codes[:, 0], seq[:frames]), "codebook 0 of each frame is the talker sample")
    ended = bool(seq[-1].item() == tc.codec_eos_token_id)
    check(not (codes[:, 0] == tc.codec_eos_token_id).any(), "no frame holds the end-of-speech id")
    sampled_hidden = torch.stack([hid[0][-1][0, -1] for hid in res.hidden_states])  # [F + 1, 2048]
    sampled_logits = torch.stack([l[0] for l in res.logits])  # [F + 1, 3072]
    sampled_scores = torch.stack([s[0] for s in res.scores])
    check(len(cap.cp_results) == frames, f"{len(cap.cp_results)} code predictor runs for {frames} frames")
    cp_sampled_logits = torch.stack([torch.stack([l[0] for l in r.logits]) for r in cap.cp_results])
    cp_sampled_scores = torch.stack([torch.stack([s[0] for s in r.scores]) for r in cap.cp_results])
    for f, r in enumerate(cap.cp_results):
        check(torch.equal(r.sequences[0], codes[f, 1:]), f"code predictor codes of frame {f}")
    print(f"sampled {frames} frames in {gen_seconds:.1f} s, end of speech: {ended}", file=sys.stderr)
    print(f"processors: {cap.processors}", file=sys.stderr)

    # 2. The prompt as pairs, and its embeddings.
    input_id = tts._tokenize_texts([tts._build_assistant_text(args.text)])[0]
    instruct_id = tts._tokenize_texts([tts._build_instruct_text(args.instruct)])[0]
    prompt = cap.talker_kwargs["inputs_embeds"]  # [1, P, 2048]
    check(cap.talker_kwargs["attention_mask"].all(), "no padding in the prompt")
    text_ids, codec_ids = prompt_pairs(model, input_id, instruct_id, args.language)
    close(embed_pairs(talker, text_ids, codec_ids), prompt, 1e-5, "prompt pairs give the prompt embeddings")
    tts_pad = cap.talker_kwargs["tts_pad_embed"]  # [1, 1, 2048]
    trailing = cap.talker_kwargs["trailing_text_hidden"]
    check(trailing.shape[1] == 1, "non-streaming mode: the trailing text is one tts_pad")
    close(trailing, tts_pad, 0.0, "the trailing text is tts_pad")

    # 3. The forced run.
    f_hidden, f_logits, f_cp_logits = forced_run(talker, prompt, codes, tts_pad)
    close(f_hidden, sampled_hidden, 1e-4, "forced hidden states = sampling run")
    close(f_logits, sampled_logits, 1e-3, "forced codebook 0 logits = sampling run")
    close(f_cp_logits, cp_sampled_logits, 1e-3, "forced code predictor logits = sampling run")

    # 4. The sampler pipeline against the processed scores of transformers, at each step.
    knobs = dict(vocab=tc.vocab_size, eos=tc.codec_eos_token_id, penalty=1.05, min_new=2, temperature=0.9, top_k=50)
    check(
        cap.processors["talker"]
        == [
            "RepetitionPenaltyLogitsProcessor",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
        ],
        f"talker processors: {cap.processors['talker']}",
    )
    check(
        cap.processors["code_predictor"] == ["TemperatureLogitsWarper", "TopKLogitsWarper"],
        f"code predictor processors: {cap.processors['code_predictor']}",
    )
    po = cap.processor_objects["talker"]
    check(po[0].penalty == 1.05, f"repetition penalty {po[0].penalty}")
    check(po[1].min_length == 2, f"min_length {po[1].min_length}")
    check(po[2].prompt_length_to_skip == 0 and po[2].min_new_tokens == 2, "min_new_tokens 2 over no prompt ids")
    check(po[4].temperature == 0.9 and po[5].top_k == 50, "temperature 0.9 and top-k 50")
    hist = seq.tolist()
    for t in range(steps):
        st = sampler_stages(sampled_logits[t], hist[:t], t, **knobs)
        check(torch.equal(st["scores"], sampled_scores[t]), f"talker sampler step {t} = transformers scores")
        check(st["probs"][hist[t]] > 0, f"sampled id {hist[t]} of step {t} has a probability")
    for f in range(frames):
        for g in range(15):
            s = cp_sampler(cp_sampled_logits[f, g], temperature=0.9, top_k=50)
            check(torch.equal(s, cp_sampled_scores[f, g]), f"code predictor sampler frame {f} codebook {g + 1}")
            check(torch.isfinite(s[codes[f, g + 1]]), f"code {g + 1} of frame {f} is in the top-k")

    # 5. The audio.
    codes_bt = codes.T.unsqueeze(0)  # [1, 16, F]
    audio_full = dec(codes_bt)[0, 0]
    audio_chunked = dec.chunked_decode(codes_bt, chunk_size=4, left_context_size=25)[0, 0]
    check(audio_full.shape[0] == frames * FRAME_SAMPLES, f"{audio_full.shape[0]} samples for {frames} frames")
    check(audio_chunked.shape == audio_full.shape, "the chunked decode has the length of the full decode")
    gen_audio = torch.as_tensor(wavs[0]).float()
    close(gen_audio, audio_full, 1e-5, "full decode = audio of generate_voice_design")
    diff = audio_chunked - audio_full
    chunk_max = diff.abs().max().item()
    chunk_rms = diff.pow(2).mean().sqrt().item()
    sig_rms = audio_full.pow(2).mean().sqrt().item()
    print(f"chunked vs full: max abs {chunk_max:.4g}, rms {chunk_rms:.4g}, signal rms {sig_rms:.4g}", file=sys.stderr)

    meta = {
        "instruct": args.instruct,
        "text": args.text,
        "language": args.language,
        "seed": str(args.seed),
        "max_new_tokens": str(args.max_frames),
        "frames": str(frames),
        "ended": str(ended),
        "talker_processors": ",".join(cap.processors["talker"]),
        "code_predictor_processors": ",".join(cap.processors["code_predictor"]),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "model": os.path.realpath(args.model),
    }
    forced = {
        "instruct_ids": instruct_id[0].clone(),
        "input_ids": input_id[0].clone(),
        "prompt_text_ids": text_ids,
        "prompt_codec_ids": codec_ids,
        "talker_input_embeds": prompt[0].contiguous(),
        "tts_pad_embed": tts_pad[0, 0].contiguous(),
        "cb0_sampled": seq.clone(),
        "codes": codes.contiguous(),
        "talker_logits": f_logits.contiguous(),
        "talker_hidden": f_hidden.contiguous(),
        "cp_logits": f_cp_logits.contiguous(),
        "audio_full": audio_full.contiguous(),
        "audio_chunked": audio_chunked.contiguous(),
    }

    # 6. Op references. Talker layer 0 over the first 5 prompt positions, then 1 decode position.
    ops = {}
    pre, kv = talker_layer_parts(talker, 0, prompt[:, :5], torch.arange(5))
    one, _ = talker_layer_parts(talker, 0, prompt[:, 5:6], torch.arange(5, 6), past=kv)
    outs, _ = layer_outputs(
        talker, [talker.model.layers[0]], lambda: talker.model(inputs_embeds=prompt[:, :6], use_cache=False)
    )
    close(pre["out"], outs[0][0, :5], 1e-4, "talker L0 by hand = model, prefill")
    close(one["out"], outs[0][0, 5:6], 1e-4, "talker L0 by hand = model, decode")
    for k, v in pre.items():
        ops[f"talker.L0.prefill.{k}"] = v.contiguous()
    for k, v in one.items():
        ops[f"talker.L0.decode.{k}"] = v.contiguous()

    # The code predictor of frame 0, in one forward over its 16 positions.
    cp = talker.code_predictor
    c0 = codes[0]
    h0 = f_hidden[0].view(1, 1, -1)
    cp_in = torch.cat(
        [h0, talker.get_input_embeddings()(c0[None, :1])]
        + [cp.model.codec_embedding[g](c0[None, g + 1 : g + 2]) for g in range(14)],
        dim=1,
    )  # [1, 16, 2048]
    cp_proj = cp.small_to_mtp_projection(cp_in)
    louts, out = layer_outputs(cp, list(cp.model.layers), lambda: cp.model(inputs_embeds=cp_proj, use_cache=False))
    norm = out.last_hidden_state
    cp_logits = torch.stack([cp.lm_head[g](norm[0, g + 1]) for g in range(15)])
    close(cp_logits, f_cp_logits[0], 1e-4, "code predictor in one forward = stepwise")
    ops["cp.codes"] = c0.clone()
    ops["cp.in_embeds"] = cp_in[0].contiguous()
    ops["cp.in_proj"] = cp_proj[0].contiguous()
    for i, lo in enumerate(louts):
        ops[f"cp.L{i}.out"] = lo[0].contiguous()
    ops["cp.norm"] = norm[0].contiguous()
    ops["cp.logits"] = cp_logits.contiguous()

    # The codec decoder by stage, for the first 4 frames.
    c4 = codes[:4].T.unsqueeze(0)
    stages = decoder_stages(dec, c4)
    close(stages["audio"], dec(c4)[0, 0], 0.0, "decoder stages = decoder")
    ops["dec.codes"] = codes[:4].contiguous()
    for k, v in stages.items():
        ops[f"dec.{k}"] = v.contiguous()

    # 7. The sampler, for 3 rows: step 0 (the end-of-speech id is masked), a middle step, the last step.
    rows = [0, steps // 2, steps - 1]
    hlen = max(rows)
    samp = {
        "sampler.step": torch.tensor(rows, dtype=torch.int64),
        "sampler.history": torch.full((3, max(hlen, 1)), -1, dtype=torch.int64),
        "sampler.logits": torch.stack([sampled_logits[t] for t in rows]),
        "sampler.sampled": torch.tensor([hist[t] for t in rows], dtype=torch.int64),
    }
    stage_rows = {}
    for i, t in enumerate(rows):
        if t:
            samp["sampler.history"][i, :t] = seq[:t]
        for k, v in sampler_stages(sampled_logits[t], hist[:t], t, **knobs).items():
            stage_rows.setdefault(k, []).append(v)
    for k, v in stage_rows.items():
        samp[f"sampler.{k}"] = torch.stack(v)
    samp["cp_sampler.logits"] = cp_sampled_logits[0].contiguous()
    samp["cp_sampler.scores"] = cp_sampled_scores[0].contiguous()
    samp["cp_sampler.probs"] = torch.softmax(cp_sampled_scores[0], dim=-1)

    # 8. The chunked decode over more frames than the left context. The decoder is causal, so a chunk with all
    # the frames before it as context gives the full decode up to rounding. The first chunk with a cut context
    # starts at frame 28 (chunks of 4, context 25), and only there can the audio differ.
    torch.manual_seed(args.seed)
    long_wavs, _ = tts.generate_voice_design(
        text=args.long_text, instruct=args.instruct, language=args.language, max_new_tokens=args.long_frames
    )
    lres = cap.talker_result
    long_codes = torch.stack([hid[-1][0] for hid in lres.hidden_states if hid[-1] is not None])
    lseq = lres.sequences[0]
    if lseq[-1].item() == tc.codec_eos_token_id:
        long_codes = long_codes[: lseq.shape[0] - 1]
    lf = long_codes.shape[0]
    check(lf > 40, f"the long run has {lf} frames; it needs more than 40 to cut the left context")
    lbt = long_codes.T.unsqueeze(0)
    long_full = dec(lbt)[0, 0]
    long_chunked = dec.chunked_decode(lbt, chunk_size=4, left_context_size=25)[0, 0]
    close(torch.as_tensor(long_wavs[0]).float(), long_full, 1e-5, "long full decode = generate_voice_design")
    cut = 28 * FRAME_SAMPLES
    close(long_chunked[:cut], long_full[:cut], 1e-4, "long chunked = full before the first cut context")
    ld = long_chunked[cut:] - long_full[cut:]
    long_max = ld.abs().max().item()
    long_rms = ld.pow(2).mean().sqrt().item()
    long_sig = long_full[cut:].pow(2).mean().sqrt().item()
    print(
        f"long run, {lf} frames, after frame 28: chunked vs full max abs {long_max:.4g}, rms {long_rms:.4g},"
        f" signal rms {long_sig:.4g}",
        file=sys.stderr,
    )
    chunked = {
        "long_codes": long_codes.contiguous(),
        "long_audio_full": long_full.contiguous(),
        "long_audio_chunked": long_chunked.contiguous(),
    }
    meta["long_text"] = args.long_text
    meta["long_frames"] = str(lf)

    for name, tensors in (("forced", forced), ("ops", ops), ("sampler", samp), ("chunked", chunked)):
        path = os.path.join(args.out, f"{name}.safetensors")
        save_file({k: v.contiguous() for k, v in tensors.items()}, path, metadata=meta)
        print(f"wrote {path}: {os.path.getsize(path) / 1e6:.1f} MB", file=sys.stderr)
        for k, v in tensors.items():
            print(f"  {k} {list(v.shape)} {str(v.dtype).removeprefix('torch.')}", file=sys.stderr)
    print(f"done in {time.time() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
