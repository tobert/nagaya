#!/usr/bin/env python3
"""Make images with a served image model of a nagaya, through council v2 (spec 11.9a).

The script loads the engine of the model when it is not ready, sends one generation, and writes each PNG. The PNG
holds the A1111 `parameters` chunk, so A1111 and Civitai tools read its settings. With `--previews`, the script
streams the generation and writes the preview of each step too.

Examples:

    tools/nagaya_image.py --model sdxl-0 "a red fox in a snowy forest, (watercolor:1.2)"
    tools/nagaya_image.py --model sdxl-1 --count 4 --seed 7 --negative "blurry" --out foxes "a fox"
    tools/nagaya_image.py --model sdxl-0 --unload-after --previews "a lighthouse at dusk"
    tools/nagaya_image.py --model sdxl-0 --unload gemma "a fox"

It needs only the Python standard library.
"""

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://198.51.100.10:8091"
SAMPLERS = ["euler", "euler a", "dpm++ 2m karras", "dpm++ 2m sde karras"]


def call(url, method="GET", body=None, timeout=900):
    """Send one request. Give the status and the decoded JSON body."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def status_of(base, model):
    """The status of a served model in identity, or None when the server does not serve it."""
    _, identity = call(f"{base}/council/v2/identity")
    for served in identity["models"]:
        if served["model"] == model:
            return served["status"]
    return None


def load(base, model, wait_s, unload=()):
    """Load the engine of the model after the engines of `unload` unload, and wait until the model is ready. The served
    image model has the name of its engine (spec 11.9a, "The served image model"). A load that does not fit is a 409
    that names the engines in its way (spec 3.3a, "Explicit layout")."""
    status = status_of(base, model)
    if status is None:
        sys.exit(f"the server serves no model `{model}`; GET {base}/council/v2/identity lists the served models")
    if status == "ready":
        return
    code, reply = call(f"{base}/nagaya/v1/engines/{model}/load", "POST", {"unload": list(unload)})
    if code == 409:
        sys.exit(f"the load of `{model}` does not fit: {reply['error']['message']}; give each engine in the way with "
                 f"--unload")
    if code not in (200, 202):
        sys.exit(f"the load of `{model}` failed with {code}: {reply}")
    if reply.get("unload"):
        print(f"loading {model}; it unloads {', '.join(reply['unload'])}", file=sys.stderr)
    deadline = time.monotonic() + wait_s
    while status_of(base, model) != "ready":
        if time.monotonic() > deadline:
            sys.exit(f"`{model}` did not become ready in {wait_s} s")
        time.sleep(0.5)


def request_body(args):
    """The generation of spec 11.9a: the prompt as the last `user` message, the options in `output.options`."""
    options = {
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler": args.sampler,
        "count": args.count,
    }
    if args.negative:
        options["negative"] = args.negative
    if args.clip_skip is not None:
        options["clip_skip"] = args.clip_skip
    body = {
        "model": args.model,
        "max_tokens": 1,
        "context": {"thinking": False, "messages": [{"role": "user", "content": args.prompt}]},
        "output": {"mime": "image/png", "options": options},
    }
    if args.seed is not None:
        body["sample"] = {"seed": args.seed}
    if args.previews:
        body["stream"] = True
        body["previews"] = True
    return body


def fetch(base, object_id):
    with urllib.request.urlopen(f"{base}/council/v2/objects/{object_id}", timeout=120) as response:
        return response.read()


def events(response):
    """The server-sent events of a stream: each name with its decoded data."""
    name, data = None, []
    for raw in response:
        line = raw.decode().rstrip("\n").rstrip("\r")
        if not line:
            if name is not None:
                yield name, json.loads("".join(data))
            name, data = None, []
        elif line.startswith("event:"):
            name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].strip())


def generate_stream(base, body, out, stem):
    """Stream a generation: print each `progress` event, write each preview as it comes, and give the `done` reply."""
    request = urllib.request.Request(
        f"{base}/council/v2/generations",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=3600)
    except urllib.error.HTTPError as error:
        sys.exit(f"the generation failed with {error.code}: {error.read().decode()}")
    with response:
        for name, data in events(response):
            if name == "progress":
                print(f"image {data['index']}: step {data['step']} of {data['of']}", file=sys.stderr)
            elif name == "preview":
                path = out / f"{stem}-{data['index']}-step{data['step']:03d}.png"
                path.write_bytes(fetch(base, data["id"]))
                print(f"preview {path}", file=sys.stderr)
            elif name == "done":
                return data
            elif name == "error":
                sys.exit(f"the generation failed: {data}")
    sys.exit("the stream ended with no `done` event")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="the prompt, with A1111 weights such as (word:1.2) and [word]")
    parser.add_argument("--model", required=True, help="the served image model, such as sdxl-0")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"the base URL of the nagaya (default {DEFAULT_URL})")
    parser.add_argument("--negative", default="", help="the negative prompt")
    parser.add_argument("--width", type=int, default=1024, help="a multiple of 64 from 512 to 2048 (default 1024)")
    parser.add_argument("--height", type=int, default=1024, help="a multiple of 64 from 512 to 2048 (default 1024)")
    parser.add_argument("--steps", type=int, default=30, help="1 to 150 (default 30)")
    parser.add_argument("--cfg", type=float, default=7.0, help="1.0 to 30.0 (default 7)")
    parser.add_argument("--sampler", choices=SAMPLERS, default="dpm++ 2m karras")
    parser.add_argument("--count", type=int, default=1, help="1 to 4 images; image i uses the seed plus i")
    parser.add_argument("--seed", type=int, help="the seed; with none, the server chooses one and the PNG records it")
    parser.add_argument("--clip-skip", type=int, choices=[2, 3],
                        help="in A1111's count: 2 is the penultimate layer; the default comes from the served model")
    parser.add_argument("--previews", action="store_true", help="stream, and write the preview of each step")
    parser.add_argument("--out", default=".", help="the directory of the PNGs (default: the current directory)")
    parser.add_argument("--no-load", action="store_true", help="do not load the engine; a model not ready is a 503")
    parser.add_argument("--unload", action="append", default=[], metavar="ENGINE",
                        help="an engine to unload before the load, such as gemma; give it once for each engine")
    parser.add_argument("--unload-after", action="store_true", help="unload the engine after the images")
    parser.add_argument("--load-wait", type=int, default=300, help="seconds to wait for a load (default 300)")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_load:
        load(base, args.model, args.load_wait, args.unload)
    started = time.monotonic()
    stem = f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.previews:
        done = generate_stream(base, request_body(args), out, stem)
    else:
        code, done = call(f"{base}/council/v2/generations", "POST", request_body(args), timeout=3600)
        if code != 200:
            sys.exit(f"the generation failed with {code}: {json.dumps(done)}")
    seconds = time.monotonic() - started
    for output in done["outputs"]:
        path = out / f"{stem}-{output['index']}.png"
        path.write_bytes(fetch(base, output["id"]))
        print(path)
    print(f"{len(done['outputs'])} images in {seconds:.1f} s, seed {done.get('seed')}", file=sys.stderr)
    if args.unload_after:
        code, reply = call(f"{base}/nagaya/v1/engines/{args.model}/unload", "POST")
        if code not in (200, 202):
            sys.exit(f"the unload of `{args.model}` failed with {code}: {reply}")


if __name__ == "__main__":
    main()
