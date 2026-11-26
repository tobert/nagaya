"""Writes the reference files of the Beat This! oracle (spec 8.3) and the safetensors form of its two checkpoints.

Run it through tools/beatthis_golden.sh, which names the files and their contents. The reference is the upstream
package `beat_this` (CPJKU) on the CPU in float32, one thread. A second run in float64 gives the size of the f32
error, for the tolerance of an engine. The audio is synthetic and made here from a fixed seed, so the files hold no
recording.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys

import numpy
import torch
from safetensors.numpy import save_file

from beat_this.inference import aggregate_prediction, load_checkpoint, load_model, split_piece, split_predict_aggregate
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect
from beat_this.utils import replace_state_dict_key

SR = 22050
FPS = 50
CHUNK = 1500
BORDER = 6
MODELS = ["final0", "small0"]
# The model whose inner rows go into parts.*: one chunk of the case below.
PARTS_MODEL = "final0"
PARTS_CASE = "click120_44"


def click_track(seconds, bpm_at, meter, seed):
    """A click track: a short decaying tone on each beat, higher and louder on the downbeat, over quiet noise.

    `bpm_at(t)` gives the tempo at time t in seconds, so a ramp is a function of t. The beat times are the sum of
    the beat periods, so a ramp has exact beat times too.
    """
    rng = numpy.random.default_rng(seed)
    n = int(round(seconds * SR))
    out = (rng.standard_normal(n) * 0.003).astype(numpy.float64)
    beats, downbeats = [], []
    t, i = 0.25, 0
    tone = numpy.arange(int(0.06 * SR)) / SR
    while t < seconds - 0.1:
        down = i % meter == 0
        freq, amp = (1760.0, 0.6) if down else (880.0, 0.35)
        click = amp * numpy.sin(2 * math.pi * freq * tone) * numpy.exp(-tone * 60.0)
        s = int(round(t * SR))
        e = min(n, s + len(click))
        out[s:e] += click[: e - s]
        beats.append(t)
        if down:
            downbeats.append(t)
        t += 60.0 / bpm_at(t)
        i += 1
    return out.astype(numpy.float32), beats, downbeats


def cases():
    """The cases: (name, wave f32 at 22,050 Hz mono, beat times, downbeat times, description)."""
    out = []
    w, b, d = click_track(30.0, lambda t: 120.0, 4, 1)
    out.append(("click120_44", w, b, d, "120 BPM in 4/4, 30 s: one chunk of 1,501 frames"))
    w, b, d = click_track(10.0, lambda t: 97.0, 3, 2)
    out.append(("click97_34", w, b, d, "97 BPM in 3/4, 10 s: the size of a live window"))
    w, b, d = click_track(70.0, lambda t: 100.0 + 30.0 * t / 70.0, 4, 3)
    out.append(("ramp100_130", w, b, d, "100 to 130 BPM in 4/4, 70 s: three chunks, the last moved left"))
    out.append(("silence", numpy.zeros(5 * SR, dtype=numpy.float32), [], [], "5 s of zeros"))
    w, b, d = click_track(0.5, lambda t: 120.0, 4, 4)
    out.append(("short", w, b, d, "0.5 s: shorter than two borders"))
    return out


def predict(model, spect):
    """Upstream `split_predict_aggregate` (keep_first), with the chunk starts and sizes kept for cases.json."""
    chunks, starts = split_piece(spect, CHUNK, border_size=BORDER, avoid_short_end=True)
    p = split_predict_aggregate(spect, CHUNK, BORDER, "keep_first", model)
    return p["beat"], p["downbeat"], [int(s) for s in starts], [int(c.shape[0]) for c in chunks]


# Lengths in frames around the edges of the chunk rule: one frame, a live window, one chunk less and more than its
# borders, two and three chunks, and the length where the last chunk moves left by one frame.
JOIN_FRAMES = [1, 26, 263, 501, 1487, 1488, 1489, 1493, 1494, 1495, 1500, 1501, 2976, 2977, 2989, 3501]


def join_oracle(golden, info):
    """Runs upstream's split and keep_first join on fake predictions, with no model.

    The fake beat logit at chunk position i of chunk c is 10000 * c + (start + i), its global frame. Thus the joined
    row names, at each frame, the chunk that wins and the frame that it read. The downbeat logit is its negative.
    """
    info["join"] = {}
    for t in JOIN_FRAMES:
        spect = torch.zeros(t, 128)
        chunks, starts = split_piece(spect, CHUNK, border_size=BORDER, avoid_short_end=True)
        preds = []
        for c, (chunk, start) in enumerate(zip(chunks, starts)):
            tag = torch.arange(chunk.shape[0], dtype=torch.float32) + float(start) + 10000.0 * c
            preds.append({"beat": tag, "downbeat": -tag})
        beat, down = aggregate_prediction(preds, starts, t, CHUNK, BORDER, "keep_first", "cpu")
        golden[f"join.{t}.beat"] = beat.numpy()
        golden[f"join.{t}.downbeat"] = down.numpy()
        info["join"][str(t)] = {"starts": [int(x) for x in starts], "sizes": [int(c.shape[0]) for c in chunks]}


def parts_hooks(model, store):
    names = {
        "stem": model.frontend.stem,
        "block0": model.frontend.blocks[0],
        "block1": model.frontend.blocks[1],
        "block2": model.frontend.blocks[2],
        "linear": model.frontend.linear,
        "final_norm": model.transformer_blocks.norm,
    }
    for i, layer in enumerate(model.transformer_blocks.layers):
        names[f"layer{i}.attn"] = layer[0]
        names[f"layer{i}.ff"] = layer[1]
    handles = []
    for name, module in names.items():

        def hook(_m, _args, out, name=name):
            # A piece of more than one chunk runs the model once for each; keep chunk 0, as the docs say.
            if name not in store:
                store[name] = out.detach()[0].clone()

        handles.append(module.register_forward_hook(hook))
    return handles


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(ckpt_path, out_path):
    """Writes the model weights of one checkpoint as safetensors, f32, with the names of `BeatThis.state_dict`."""
    ckpt = load_checkpoint(ckpt_path, "cpu")
    state = replace_state_dict_key(ckpt["state_dict"], "model.", "")
    state = replace_state_dict_key(state, "_orig_mod.", "")
    tensors = {}
    for k, v in state.items():
        if k.endswith("num_batches_tracked"):
            continue  # a training counter; inference does not read it
        if v.dtype != torch.float32:
            sys.exit(f"{ckpt_path}: tensor {k} is {v.dtype}, not float32; the converter keeps f32 only")
        tensors[k] = v.contiguous().numpy()
    hparams = {k: v for k, v in ckpt["hyper_parameters"].items() if isinstance(v, (int, float, str, bool, dict))}
    meta = {"source_sha256": sha256(ckpt_path), "hyper_parameters": json.dumps(hparams, sort_keys=True)}
    save_file(tensors, out_path, metadata=meta)
    return meta, len(tensors), sum(t.size for t in tensors.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="the folder of beat_this-final0.ckpt and beat_this-small0.ckpt")
    ap.add_argument("--out", required=True, help="the folder of golden.safetensors and cases.json")
    ap.add_argument("--weights-out", required=True, help="the folder of final0.safetensors and small0.safetensors")
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.weights_out, exist_ok=True)

    golden, info = {}, {"models": {}, "cases": {}}
    spect_f32 = LogMelSpect(device="cpu")
    # A cast of the f32 module keeps its f32 window and filters. Built with f64 as the default dtype, torchaudio makes
    # the window and the filters in f64 too, so this is the exact form that an engine in f64 can match.
    torch.set_default_dtype(torch.float64)
    spect_f64 = LogMelSpect(device="cpu")
    torch.set_default_dtype(torch.float32)
    golden["mel.window"] = spect_f32.spect_class.spectrogram.window.numpy()
    golden["mel.filterbank"] = spect_f32.spect_class.mel_scale.fb.numpy()  # [513, 128]
    golden["mel.window64"] = spect_f64.spect_class.spectrogram.window.numpy()
    golden["mel.filterbank64"] = spect_f64.spect_class.mel_scale.fb.numpy()

    join_oracle(golden, info)
    post = Postprocessor(type="minimal", fps=FPS)
    for name in MODELS:
        ckpt = os.path.join(a.ckpt_dir, f"beat_this-{name}.ckpt")
        meta, count, params = convert(ckpt, os.path.join(a.weights_out, f"{name}.safetensors"))
        info["models"][name] = {**meta, "tensors": count, "values": params}

    for case, wave, beats, downbeats, desc in cases():
        golden[f"{case}.wave"] = wave
        with torch.inference_mode():
            x = torch.from_numpy(wave)
            spect = spect_f32(x)
            spect64 = spect_f64(x.double())
        golden[f"{case}.spect"] = spect.numpy()
        golden[f"{case}.spect64"] = spect64.numpy()
        c = {
            "description": desc,
            "samples": len(wave),
            "frames": int(spect.shape[0]),
            "true_beats_s": beats,
            "true_downbeats_s": downbeats,
            "spect_f32_vs_f64_max_abs": float((spect.double() - spect64).abs().max()),
            "models": {},
        }
        if case == PARTS_CASE:
            stft = torch.stft(
                torch.nn.functional.pad(x[None], (512, 512), mode="reflect")[0],
                1024,
                441,
                window=spect_f32.spect_class.spectrogram.window,
                center=False,
                normalized=True,
                return_complex=True,
            )
            golden["parts.stft_mag"] = stft.abs().T.contiguous().numpy()  # [T, 513]
            x64 = x.double()
            stft64 = torch.stft(
                torch.nn.functional.pad(x64[None], (512, 512), mode="reflect")[0],
                1024,
                441,
                window=spect_f64.spect_class.spectrogram.window,
                center=False,
                normalized=True,
                return_complex=True,
            )
            golden["parts.stft_mag64"] = stft64.abs().T.contiguous().numpy()
        for name in MODELS:
            ckpt = os.path.join(a.ckpt_dir, f"beat_this-{name}.ckpt")
            model = load_model(ckpt, "cpu")
            store, handles = {}, []
            if name == PARTS_MODEL and case == PARTS_CASE:
                handles = parts_hooks(model, store)
            with torch.inference_mode():
                beat, down, starts, sizes = predict(model, spect)
            for h in handles:
                h.remove()
            for k, v in store.items():
                golden[f"parts.{k}"] = v.numpy()
            model64 = load_model(ckpt, "cpu").double()
            store64, handles64 = {}, []
            if name == PARTS_MODEL and case == PARTS_CASE:
                handles64 = parts_hooks(model64, store64)
            with torch.inference_mode():
                beat64, down64, _, _ = predict(model64, spect64)
            for h in handles64:
                h.remove()
            for k, v in store64.items():
                golden[f"parts64.{k}"] = v.numpy()
            golden[f"{name}.{case}.beat64"] = beat64.double().numpy()
            golden[f"{name}.{case}.downbeat64"] = down64.double().numpy()
            b_s, d_s = post(beat, down)
            golden[f"{name}.{case}.beat"] = beat.numpy()
            golden[f"{name}.{case}.downbeat"] = down.numpy()
            golden[f"{name}.{case}.beats_s"] = numpy.asarray(b_s, dtype=numpy.float64)
            golden[f"{name}.{case}.downbeats_s"] = numpy.asarray(d_s, dtype=numpy.float64)
            c["models"][name] = {
                "chunk_starts": starts,
                "chunk_frames": sizes,
                "beats": len(b_s),
                "downbeats": len(d_s),
                "f32_vs_f64_max_abs_logit": float(
                    max((beat.double() - beat64).abs().max(), (down.double() - down64).abs().max())
                ),
            }
        info["cases"][case] = c

    versions = {p: importlib.metadata.version(p) for p in ["torch", "torchaudio", "beat_this", "numpy", "einops",
                                                            "rotary-embedding-torch", "safetensors"]}
    info["versions"] = versions
    info["settings"] = {"sample_rate": SR, "fps": FPS, "chunk": CHUNK, "border": BORDER, "threads": 1,
                        "postprocess": "minimal", "parts": f"{PARTS_MODEL}.{PARTS_CASE}, chunk 0"}
    save_file(golden, os.path.join(a.out, "golden.safetensors"), metadata={"versions": json.dumps(versions)})
    with open(os.path.join(a.out, "cases.json"), "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: {m: v["models"][m] for m in v["models"]} for k, v in info["cases"].items()}, indent=1))


if __name__ == "__main__":
    main()
