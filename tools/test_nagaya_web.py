"""Tests of nagaya_web.py against a fake nagaya on a local port. Run: python3 -m unittest test_nagaya_web (in tools/)."""

import http.client
import http.server
import json
import threading
import unittest

import nagaya_web


class FakeNagaya(http.server.BaseHTTPRequestHandler):
    """A JSON route, a stream whose second event waits for the test, and a redirect of a pending id."""

    protocol_version = "HTTP/1.0"
    second = threading.Event()
    seen = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/council/v2/identity":
            self.reply(200, b'{"models": []}', "application/json")
        elif self.path.startswith("/council/v2/objects/pending:"):
            self.send_response(308)
            self.send_header("location", "/council/v2/objects/blake3:ab")
            self.end_headers()
        else:
            self.reply(404, b'{"error": {"message": "no route"}}', "application/json")

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        FakeNagaya.seen.append((self.path, self.headers.get("content-type"), json.loads(body)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'event: progress\ndata: {"index": 0, "step": 1, "of": 2}\n\n')
        self.wfile.flush()
        # The second event waits until the test has read the first. A pass that buffers the stream holds back the
        # first event, so the wait runs out, and the stream ends with `late` in place of `done`.
        if FakeNagaya.second.wait(3):
            self.wfile.write(b'event: done\ndata: {"outputs": []}\n\n')
        else:
            self.wfile.write(b'event: late\ndata: {}\n\n')

    def reply(self, status, body, kind):
        self.send_response(status)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class Proxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeNagaya))
        url = f"http://127.0.0.1:{cls.upstream.server_address[1]}"
        cls.app = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), nagaya_web.handler(url)))

    @classmethod
    def tearDownClass(cls):
        for server in (cls.app, cls.upstream):
            server.shutdown()
            server.server_close()

    def connect(self):
        return http.client.HTTPConnection("127.0.0.1", self.app.server_address[1], timeout=10)

    def test_the_root_is_the_page(self):
        c = self.connect()
        c.request("GET", "/")
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertTrue(r.getheader("content-type").startswith("text/html"))
        self.assertIn(b"<title>nagaya</title>", r.read())

    def test_a_council_route_passes_with_its_status_and_body(self):
        c = self.connect()
        c.request("GET", "/council/v2/identity")
        r = c.getresponse()
        self.assertEqual((r.status, r.getheader("content-type"), json.loads(r.read())), (200, "application/json", {"models": []}))
        c = self.connect()
        c.request("GET", "/council/v2/nothing")
        self.assertEqual(c.getresponse().status, 404)

    def test_a_redirect_passes_with_its_location(self):
        c = self.connect()
        c.request("GET", "/council/v2/objects/pending:" + "0" * 64)
        r = c.getresponse()
        self.assertEqual((r.status, r.getheader("location")), (308, "/council/v2/objects/blake3:ab"))

    def test_a_stream_reaches_the_page_event_by_event(self):
        FakeNagaya.second.clear()
        c = self.connect()
        c.request("POST", "/council/v2/generations", body=b'{"stream": true}', headers={"content-type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.getheader("content-type"), "text/event-stream")
        first = r.readline() + r.readline() + r.readline()
        self.assertEqual(first, b'event: progress\ndata: {"index": 0, "step": 1, "of": 2}\n\n')
        FakeNagaya.second.set()
        self.assertEqual(r.read(), b'event: done\ndata: {"outputs": []}\n\n')
        self.assertEqual(FakeNagaya.seen[-1], ("/council/v2/generations", "application/json", {"stream": True}))

    def test_a_path_outside_the_nagaya_routes_is_a_404_and_does_not_pass(self):
        before = len(FakeNagaya.seen)
        c = self.connect()
        c.request("POST", "/elsewhere", body=b"{}")
        self.assertEqual(c.getresponse().status, 404)
        self.assertEqual(len(FakeNagaya.seen), before)

    def test_a_nagaya_that_does_not_answer_is_a_502(self):
        dead = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), nagaya_web.handler("http://127.0.0.1:9")))
        try:
            c = http.client.HTTPConnection("127.0.0.1", dead.server_address[1], timeout=10)
            c.request("GET", "/council/v2/identity")
            self.assertEqual(c.getresponse().status, 502)
        finally:
            dead.shutdown()
            dead.server_close()


if __name__ == "__main__":
    unittest.main()
