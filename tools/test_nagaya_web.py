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
    identity = b'{"models": []}'

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/council/v2/identity":
            self.reply(200, self.identity, "application/json")
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


class OtherNagaya(FakeNagaya):
    """A second nagaya, which serves another model."""

    identity = b'{"models": [{"model": "qwen-image-fusal"}]}'


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


class Nagayas(unittest.TestCase):
    """One app in front of two nagayas: the first under the plain paths, each one under /at/<name>/."""

    @classmethod
    def setUpClass(cls):
        cls.first = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeNagaya))
        cls.other = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), OtherNagaya))
        cls.nagayas = [("b70host", f"http://127.0.0.1:{cls.first.server_address[1]}"),
                       ("gfx1151host", f"http://127.0.0.1:{cls.other.server_address[1]}")]
        cls.app = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), nagaya_web.handler(cls.nagayas)))

    @classmethod
    def tearDownClass(cls):
        for server in (cls.app, cls.first, cls.other):
            server.shutdown()
            server.server_close()

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.app.server_address[1], timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        return r, r.read()

    def test_the_app_lists_its_nagayas_in_order(self):
        r, body = self.get("/app/nagayas")
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(body), {"nagayas": [{"name": n, "url": u} for n, u in self.nagayas]})

    def test_a_path_under_at_a_name_goes_to_that_nagaya_and_a_plain_path_to_the_first(self):
        self.assertEqual(json.loads(self.get("/at/gfx1151host/council/v2/identity")[1]), {"models": [{"model": "qwen-image-fusal"}]})
        self.assertEqual(json.loads(self.get("/at/b70host/council/v2/identity")[1]), {"models": []})
        self.assertEqual(json.loads(self.get("/council/v2/identity")[1]), {"models": []})

    def test_a_redirect_under_at_a_name_keeps_the_name(self):
        r, _ = self.get("/at/gfx1151host/council/v2/objects/pending:" + "0" * 64)
        self.assertEqual((r.status, r.getheader("location")), (308, "/at/gfx1151host/council/v2/objects/blake3:ab"))

    def test_an_unknown_name_is_a_404_and_does_not_pass(self):
        before = len(FakeNagaya.seen)
        r, body = self.get("/at/elsewhere/council/v2/identity")
        self.assertEqual(r.status, 404)
        self.assertIn(b"b70host, gfx1151host", body)
        r, _ = self.get("/at/gfx1151host/elsewhere")
        self.assertEqual(r.status, 404)
        self.assertEqual(len(FakeNagaya.seen), before)

    def test_the_names_parse_from_the_command_line(self):
        self.assertEqual(nagaya_web.nagaya("gfx1151host=http://h:8090/"), ("gfx1151host", "http://h:8090"))
        for bad in ("gfx1151host", "=http://h:1", "a/b=http://h:1"):
            with self.assertRaises(Exception):
                nagaya_web.nagaya(bad)

    def test_the_machine_of_the_proxy_comes_first_in_the_default_list(self):
        names = lambda host: [n for n, _ in nagaya_web.local_first(nagaya_web.DEFAULT_NAGAYAS, host)]
        self.assertEqual(names("gfx1151host"), ["gfx1151host", "b70host"])
        self.assertEqual(names("b70host"), ["b70host", "gfx1151host"])
        self.assertEqual(names("sm121host"), ["b70host", "gfx1151host"], "another machine keeps the order of the list")



class ClusterSeed(http.server.BaseHTTPRequestHandler):
    """A nagaya that answers only the cluster route, with the view in `view`."""

    protocol_version = "HTTP/1.0"
    view = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps(self.view).encode()
        status = 200 if self.path == "/nagaya/v1/cluster" else 404
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Discovery(unittest.TestCase):
    """The app finds its nagayas from the cluster route of a seed (AGENTS.md milestone 23 step 7)."""

    @classmethod
    def setUpClass(cls):
        cls.seed = serve(http.server.ThreadingHTTPServer(("127.0.0.1", 0), ClusterSeed))
        cls.seed_url = f"http://127.0.0.1:{cls.seed.server_address[1]}"
        # A closed port: bind one, take its number, and close it.
        closed = http.server.HTTPServer(("127.0.0.1", 0), ClusterSeed)
        cls.dead_url = f"http://127.0.0.1:{closed.server_address[1]}"
        closed.server_close()
        ClusterSeed.view = {"node": "b70host", "dead": ["sm121host"], "nodes": [
            {"name": "b70host", "self": True, "http": "198.51.100.10:8091"},
            {"name": "yuki", "self": False, "http": None},
            {"name": "gfx1151host", "self": False, "http": "198.51.100.20:8090"}]}

    @classmethod
    def tearDownClass(cls):
        cls.seed.shutdown()
        cls.seed.server_close()

    def test_the_nodes_come_from_the_first_seed_that_answers(self):
        found = nagaya_web.discover([self.dead_url, self.seed_url], "sm121host")
        self.assertEqual(found, [("b70host", "http://198.51.100.10:8091"), ("gfx1151host", "http://198.51.100.20:8090")])

    def test_the_machine_of_the_proxy_comes_first(self):
        found = nagaya_web.discover([self.seed_url], "gfx1151host")
        self.assertEqual([n for n, _ in found], ["gfx1151host", "b70host"])

    def test_no_seed_that_answers_is_an_error_that_names_the_seeds(self):
        with self.assertRaises(SystemExit) as caught:
            nagaya_web.discover([self.dead_url], "gfx1151host")
        self.assertIn(self.dead_url, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
