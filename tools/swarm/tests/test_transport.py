"""Tests of the HTTP transport against a fake server on a local port. No test calls a live nagaya."""
import http.server
import json
import os
import threading
import unittest
from unittest import mock

from swarm import transport


class Fake(http.server.BaseHTTPRequestHandler):
    """A 503 for the first POST of /busy, a 400 for /bad, and else the body back."""

    busy = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        if self.path == "/busy" and Fake.busy == 0:
            Fake.busy += 1
            self.send_response(503)
            self.end_headers()
            return
        code = 400 if self.path == "/bad" else 200
        out = json.dumps({"path": self.path, "body": json.loads(body)}).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class TransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_a_post_gives_the_json_reply(self):
        t = transport.HttpTransport(self.url)
        self.assertEqual(t.post("/x", {"a": 1}), {"path": "/x", "body": {"a": 1}})

    def test_a_503_tries_again(self):
        Fake.busy = 0
        t = transport.HttpTransport(self.url)
        self.assertEqual(t.post("/busy", {})["path"], "/busy")
        self.assertEqual(Fake.busy, 1)

    def test_a_400_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            transport.HttpTransport(self.url).post("/bad", {})
        self.assertIn("400", str(cm.exception))

    def test_the_base_url(self):
        with mock.patch.dict(os.environ, {"NAGAYA_URL": "http://a:1/"}):
            self.assertEqual(transport.base_url("http://b:2"), "http://b:2")
            self.assertEqual(transport.base_url(), "http://a:1")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(transport.base_url(), "http://127.0.0.1:8090")


if __name__ == "__main__":
    unittest.main()
