#!/usr/bin/env python3
"""Render a batch of images from one job file, through council v2 (spec 11.9a).

The job is one TOML file. It names a served image model, the prompts, the seeds, and the variants. A variant names
its own model or changes the render settings. Renders run seed-major, so a batch stopped early still holds complete
A/B sets.

Examples:

    tools/nagaya_image_batch.py fox.toml --dry-run
    tools/nagaya_image_batch.py fox.toml
    tools/nagaya_image_batch.py fox.toml --out /tmp/foxes --unload-after

The images go to `out/<name>` beside the job, or to `--out`. The folder keeps `job.toml` as it was at the start, and
`manifest.jsonl` with one line for each image. Each PNG holds its settings in the A1111 `parameters` chunk. A rerun
skips each image that exists, and a changed prompt, setting or model renders a new file.

It needs only the Python standard library, and it imports `nagaya_image.py` from this folder.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import sys
import time
import tomllib
import urllib.error

import nagaya_image

NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
MAX_PIXELS = 1_572_864
JOB_KEYS = {"name", "model", "negative", "seeds", "prompts", "render", "variant"}
RENDER_KEYS = {"steps", "width", "height", "cfg", "sampler", "clip_skip", "previews"}
VARIANT_KEYS = RENDER_KEYS | {"name", "model"}


class JobError(Exception):
    """A mistake in the job file. The message names the file and the key."""


@dataclasses.dataclass(frozen=True)
class Settings:
    steps: int = 30
    width: int = 1024
    height: int = 1024
    cfg: float = 7.0
    sampler: str = "dpm++ 2m karras"
    clip_skip: int | None = None
    previews: bool = False


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    model: str
    settings: Settings


@dataclasses.dataclass(frozen=True)
class Job:
    name: str
    model: str
    negative: str
    seeds: tuple
    prompts: dict
    variants: tuple


@dataclasses.dataclass(frozen=True)
class Render:
    seed: int
    prompt_name: str
    prompt: str
    negative: str
    variant: Variant

    @property
    def model(self):
        return self.variant.model

    @property
    def settings(self):
        return self.variant.settings

    @property
    def hash(self):
        """A short hash of what shapes the image: the model, the prompt, the negative prompt, the seed, the settings."""
        shaping = {
            "model": self.model,
            "prompt": self.prompt,
            "negative": self.negative,
            "seed": self.seed,
            "settings": dataclasses.asdict(self.settings),
        }
        return hashlib.sha256(json.dumps(shaping, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def stem(self):
        return f"{self.prompt_name}--{self.variant.name}--s{self.seed}--{self.hash}"

    def body(self):
        """The generation request of spec 11.9a. The seed is `sample.seed`, and the image count is 1."""
        options = {
            "width": self.settings.width,
            "height": self.settings.height,
            "steps": self.settings.steps,
            "cfg": self.settings.cfg,
            "sampler": self.settings.sampler,
            "count": 1,
            "negative": self.negative,
        }
        if self.settings.clip_skip is not None:
            options["clip_skip"] = self.settings.clip_skip
        body = {
            "model": self.model,
            "max_tokens": 1,
            "context": {"thinking": False, "messages": [{"role": "user", "content": self.prompt}]},
            "output": {"mime": "image/png", "options": options},
            "sample": {"seed": self.seed},
        }
        if self.settings.previews:
            body["stream"] = True
            body["previews"] = True
        return body


def check_keys(path, table, allowed, where):
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise JobError(f"{path}: {where} has no key {unknown[0]!r}. The keys are {', '.join(sorted(allowed))}.")


def check_name(path, value, where):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise JobError(f"{path}: {where} must be a name of lowercase letters, digits, - and _, not {value!r}.")
    return value


def check_int(path, where, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise JobError(f"{path}: {where} must be a whole number from {low} to {high}, not {value!r}.")
    return value


def read_settings(path, where, table, base):
    """The settings of a table, over the settings of `base`. Each key is checked against its range (spec 11.9a)."""
    check_keys(path, table, RENDER_KEYS, where)
    values = dataclasses.asdict(base) | table
    width = check_int(path, f"{where}.width", values["width"], 512, 2048)
    height = check_int(path, f"{where}.height", values["height"], 512, 2048)
    for label, size in (("width", width), ("height", height)):
        if size % 64:
            raise JobError(f"{path}: {where}.{label} must be a multiple of 64, not {size}.")
    if width * height > MAX_PIXELS:
        raise JobError(f"{path}: {where} is {width} by {height}, which is over the pixels limit of {MAX_PIXELS}.")
    cfg = values["cfg"]
    if type(cfg) not in (int, float) or not 1.0 <= cfg <= 30.0:
        raise JobError(f"{path}: {where}.cfg must be a number from 1.0 to 30.0, not {cfg!r}.")
    if values["sampler"] not in nagaya_image.SAMPLERS:
        raise JobError(f"{path}: {where}.sampler must be one of {', '.join(nagaya_image.SAMPLERS)}.")
    clip_skip = values["clip_skip"]
    if clip_skip is not None and clip_skip not in (2, 3):
        raise JobError(f"{path}: {where}.clip_skip must be 2 or 3 (A1111's count), not {clip_skip!r}.")
    if type(values["previews"]) is not bool:
        raise JobError(f"{path}: {where}.previews must be true or false, not {values['previews']!r}.")
    return Settings(
        steps=check_int(path, f"{where}.steps", values["steps"], 1, 150),
        width=width,
        height=height,
        cfg=float(cfg),
        sampler=values["sampler"],
        clip_skip=clip_skip,
        previews=values["previews"],
    )


def load_job(path):
    """Read and check a job file. Every mistake raises JobError, with the file and the key."""
    path = pathlib.Path(path)
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise JobError(f"{path}: {error}") from error
    check_keys(path, data, JOB_KEYS, "the job")

    name = check_name(path, data.get("name"), "name")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise JobError(f"{path}: model is required, and names a served image model such as pony-0.")
    negative = data.get("negative", "")
    if not isinstance(negative, str):
        raise JobError(f"{path}: negative must be text.")

    seeds = data.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise JobError(f"{path}: seeds must be a list of whole numbers, with at least one.")
    for seed in seeds:
        check_int(path, "seeds", seed, 0, 2**63 - 1)
    if len(set(seeds)) != len(seeds):
        raise JobError(f"{path}: seeds has a repeated seed.")

    prompts = data.get("prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise JobError(f"{path}: prompts is required: a table of names and prompt text.")
    for key, text in prompts.items():
        check_name(path, key, f"prompts.{key}")
        if not isinstance(text, str) or not text.strip():
            raise JobError(f"{path}: prompts.{key} must be prompt text, not empty.")

    base = read_settings(path, "render", data.get("render", {}), Settings())
    variants = []
    for index, table in enumerate(data.get("variant", [{}])):
        where = f"variant[{index}]"
        check_keys(path, table, VARIANT_KEYS, where)
        variant_name = check_name(path, table.get("name", "base"), f"{where}.name")
        if any(v.name == variant_name for v in variants):
            raise JobError(f"{path}: the variant name {variant_name!r} is used twice.")
        variant_model = table.get("model", model)
        if not isinstance(variant_model, str) or not variant_model:
            raise JobError(f"{path}: {where}.model must be a served image model name.")
        settings = read_settings(path, where, {k: v for k, v in table.items() if k not in ("name", "model")}, base)
        variants.append(Variant(variant_name, variant_model, settings))

    return Job(name, model, negative, tuple(seeds), dict(prompts), tuple(variants))


def renders(job):
    """Every render of the job, seed-major: seed, then prompt, then variant."""
    for seed in job.seeds:
        for prompt_name, prompt in job.prompts.items():
            for variant in job.variants:
                yield Render(seed, prompt_name, prompt, job.negative, variant)


def run(job, job_path, out, base, no_load, unload_after, load_wait):
    """Render every image that is not yet in `out`. Each PNG is written to a .tmp file and then renamed."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(job_path, out / "job.toml")
    previews = out / "previews"
    manifest = out / "manifest.jsonl"
    models_used = []
    for render in renders(job):
        target = out / f"{render.stem}.png"
        if target.exists():
            print(f"skip {target}", file=sys.stderr)
            continue
        if not no_load:
            nagaya_image.load(base, render.model, load_wait)
        if render.model not in models_used:
            models_used.append(render.model)
        started = time.monotonic()
        if render.settings.previews:
            previews.mkdir(exist_ok=True)
            done = nagaya_image.generate_stream(base, render.body(), previews, render.stem)
        else:
            code, done = nagaya_image.call(f"{base}/council/v2/generations", "POST", render.body(), timeout=3600)
            if code != 200:
                sys.exit(f"the generation failed with {code}: {json.dumps(done)}")
        if len(done["outputs"]) != 1:
            sys.exit(f"the generation of {render.stem} gave {len(done['outputs'])} images, not 1")
        output_id = done["outputs"][0]["id"]
        partial = out / f"{render.stem}.png.tmp"
        partial.write_bytes(nagaya_image.fetch(base, output_id))
        partial.replace(target)
        record = {
            "file": target.name,
            "written": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "seconds": round(time.monotonic() - started, 1),
            "id": output_id,
            "model": render.model,
            "variant": render.variant.name,
            "prompt": render.prompt_name,
            "prompt_text": render.prompt,
            "negative": render.negative,
            "seed": render.seed,
            "settings": dataclasses.asdict(render.settings),
        }
        with manifest.open("a") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        print(target)
    if unload_after:
        for model in models_used:
            code, reply = nagaya_image.call(f"{base}/nagaya/v1/engines/{model}/unload", "POST")
            if code not in (200, 202):
                sys.exit(f"the unload of `{model}` failed with {code}: {reply}")


def describe(job, out):
    """Print each render and whether its file exists. No request is sent."""
    for render in renders(job):
        target = out / f"{render.stem}.png"
        state = "done" if target.exists() else "todo"
        print(f"{state}  {render.model}  seed {render.seed}  {render.variant.name}  {render.prompt_name}  {target.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", help="the job file, a TOML file")
    parser.add_argument("--out", help="the folder of the images (default: out/<name> beside the job)")
    parser.add_argument("--url", default=nagaya_image.DEFAULT_URL, help=f"the base URL of the nagaya (default {nagaya_image.DEFAULT_URL})")
    parser.add_argument("--dry-run", action="store_true", help="print the renders and send nothing")
    parser.add_argument("--no-load", action="store_true", help="do not load the engines; a model not ready is a 503")
    parser.add_argument("--unload-after", action="store_true", help="unload each engine after the batch")
    parser.add_argument("--load-wait", type=int, default=300, help="seconds to wait for a load (default 300)")
    args = parser.parse_args()

    job_path = pathlib.Path(args.job)
    try:
        job = load_job(job_path)
    except JobError as error:
        sys.exit(f"{error}")
    out = pathlib.Path(args.out) if args.out else job_path.parent / "out" / job.name
    if args.dry_run:
        describe(job, out)
        return
    try:
        run(job, job_path, out, args.url.rstrip("/"), args.no_load, args.unload_after, args.load_wait)
    except urllib.error.URLError as error:
        sys.exit(f"the nagaya at {args.url} did not answer: {error.reason}")


if __name__ == "__main__":
    main()
