"""Tests for nagaya_image_batch: the job file, the render order, the request body, and a run against a fake server."""
import http.server
import json
import pathlib
import tempfile
import threading
import unittest

import nagaya_image_batch as batch

JOB = """
name = "fox-study"
model = "pony-0"
negative = "blurry"
seeds = [1, 2]

[prompts]
snow = "score_9, a red fox in snow"
dusk = "score_9, a fox at dusk"

[render]
steps = 20
clip_skip = 2

[[variant]]
name = "hot"
model = "autismmix-1"
cfg = 5.0
"""


def write_job(text, directory):
    path = pathlib.Path(directory) / "job.toml"
    path.write_text(text)
    return path


class JobFile(unittest.TestCase):
    def test_an_unknown_key_names_the_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_job(JOB.replace("steps = 20", "steps = 20\ncfg_scale = 6"), d)
            with self.assertRaises(batch.JobError) as caught:
                batch.load_job(path)
        self.assertIn("cfg_scale", str(caught.exception))

    def test_a_width_off_the_grid_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_job(JOB.replace("[render]", "[render]\nwidth = 1000"), d)
            with self.assertRaises(batch.JobError) as caught:
                batch.load_job(path)
        self.assertIn("width", str(caught.exception))

    def test_a_clip_skip_of_1_is_refused(self):
        # clip_skip counts as A1111 does, and the server takes 2 or 3 (docs/spec.md 11.9a).
        with tempfile.TemporaryDirectory() as d:
            path = write_job(JOB.replace("clip_skip = 2", "clip_skip = 1"), d)
            with self.assertRaises(batch.JobError) as caught:
                batch.load_job(path)
        self.assertIn("2 or 3", str(caught.exception))

    def test_a_render_over_the_pixel_limit_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_job(JOB.replace("[render]", "[render]\nwidth = 2048\nheight = 2048"), d)
            with self.assertRaises(batch.JobError) as caught:
                batch.load_job(path)
        self.assertIn("pixels", str(caught.exception))

    def test_a_variant_takes_its_model_and_keeps_the_render_settings(self):
        with tempfile.TemporaryDirectory() as d:
            job = batch.load_job(write_job(JOB, d))
        hot = job.variants[0]
        self.assertEqual((hot.name, hot.model), ("hot", "autismmix-1"))
        self.assertEqual((hot.settings.steps, hot.settings.cfg, hot.settings.clip_skip), (20, 5.0, 2))

    def test_no_variant_gives_one_base_variant_on_the_job_model(self):
        with tempfile.TemporaryDirectory() as d:
            job = batch.load_job(write_job(JOB.split("[[variant]]")[0], d))
        self.assertEqual([(v.name, v.model) for v in job.variants], [("base", "pony-0")])

    def test_a_name_with_a_space_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_job(JOB.replace('name = "fox-study"', 'name = "fox study"'), d)
            with self.assertRaises(batch.JobError):
                batch.load_job(path)


class RenderOrder(unittest.TestCase):
    def test_renders_run_seed_major_then_prompt_then_variant(self):
        with tempfile.TemporaryDirectory() as d:
            job = batch.load_job(write_job(JOB.replace("[[variant]]", "[[variant]]\nname = \"plain\"\n[[variant]]"), d))
        order = [(r.seed, r.prompt_name, r.variant.name) for r in batch.renders(job)]
        self.assertEqual(
            order,
            [
                (1, "snow", "plain"), (1, "snow", "hot"),
                (1, "dusk", "plain"), (1, "dusk", "hot"),
                (2, "snow", "plain"), (2, "snow", "hot"),
                (2, "dusk", "plain"), (2, "dusk", "hot"),
            ],
        )

    def test_an_edit_of_the_prompt_renames_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            before = next(batch.renders(batch.load_job(write_job(JOB, d)))).stem
            after = next(batch.renders(batch.load_job(write_job(JOB.replace("red fox", "grey fox"), d)))).stem
        self.assertNotEqual(before, after)


class RequestBody(unittest.TestCase):
    def first_render(self, text=JOB):
        with tempfile.TemporaryDirectory() as d:
            return next(batch.renders(batch.load_job(write_job(text, d))))

    def test_the_body_names_the_model_seed_prompt_and_options(self):
        body = self.first_render(JOB.split("[[variant]]")[0]).body()
        self.assertEqual(body["model"], "pony-0")
        self.assertEqual(body["sample"], {"seed": 1})
        self.assertEqual(body["context"]["messages"], [{"role": "user", "content": "score_9, a red fox in snow"}])
        options = body["output"]["options"]
        self.assertEqual(options["steps"], 20)
        self.assertEqual(options["negative"], "blurry")
        self.assertEqual(options["clip_skip"], 2)
        self.assertEqual(options["count"], 1)
        self.assertNotIn("stream", body)

    def test_clip_skip_is_left_out_when_the_job_does_not_name_it(self):
        body = self.first_render(JOB.replace("clip_skip = 2", "")).body()
        self.assertNotIn("clip_skip", body["output"]["options"])

    def test_previews_make_the_request_a_stream(self):
        body = self.first_render(JOB.replace("clip_skip = 2", "previews = true")).body()
        self.assertTrue(body["stream"])
        self.assertTrue(body["output"]["options"]["previews"])


class FakeServer(http.server.BaseHTTPRequestHandler):
    """Serves one ready model, one generation and its PNG. It counts the generations it gets."""

    generations = 0
    png = b"\x89PNG fake"

    def log_message(self, *args):
        pass

    def send_json(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/council/v2/identity":
            self.send_json(200, {"models": [{"model": "pony-0", "status": "ready"}, {"model": "autismmix-1", "status": "ready"}]})
        elif self.path.startswith("/council/v2/objects/"):
            self.send_response(200)
            self.send_header("content-length", str(len(self.png)))
            self.end_headers()
            self.wfile.write(self.png)
        else:
            self.send_json(404, {"error": self.path})

    def do_POST(self):
        length = int(self.headers["content-length"])
        body = json.loads(self.rfile.read(length))
        FakeServer.generations += 1
        self.send_json(200, {"outputs": [{"index": 0, "id": f"blake3:{body['sample']['seed']:064x}"}], "seed": body["sample"]["seed"]})


class RunAgainstFakeServer(unittest.TestCase):
    def setUp(self):
        FakeServer.generations = 0
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_a_run_writes_each_png_and_a_second_run_skips_them_all(self):
        with tempfile.TemporaryDirectory() as d:
            job_path = write_job(JOB, d)
            out = pathlib.Path(d) / "out"
            job = batch.load_job(job_path)
            batch.run(job, job_path, out, self.base, no_load=True, unload_after=False, load_wait=5)
            files = sorted(p.name for p in out.glob("*.png"))
            manifest = (out / "manifest.jsonl").read_text().splitlines()
            self.assertEqual(len(files), 4)
            self.assertEqual(len(manifest), 4)
            self.assertEqual(FakeServer.generations, 4)
            self.assertTrue((out / "job.toml").exists())
            self.assertFalse(list(out.glob("*.tmp")))

            batch.run(job, job_path, out, self.base, no_load=True, unload_after=False, load_wait=5)
            self.assertEqual(FakeServer.generations, 4)
            self.assertEqual(sorted(p.name for p in out.glob("*.png")), files)


if __name__ == "__main__":
    unittest.main()
