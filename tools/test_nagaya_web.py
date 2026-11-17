"""Tests of nagaya_web.py against a fake nagaya on a local port. Run: python3 -m unittest test_nagaya_web (in tools/)."""

import http.client
import http.server
import json
import re
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

    def get(self, path):
        c = self.connect()
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.getheader("content-type"), r.read()

    def test_the_page_names_only_files_that_the_app_serves(self):
        page = self.get("/")[2].decode()
        names = re.findall(r'(?:src|href)="(/app/web/[^"]+)"', page)
        # The stylesheet, the core, the start script and one script or more for the tabs.
        self.assertGreaterEqual(len(names), 4)
        for name in names:
            status, kind, body = self.get(name)
            self.assertEqual(status, 200, name)
            want = "text/css" if name.endswith(".css") else "text/javascript"
            self.assertEqual(kind, f"{want}; charset=utf-8", name)
            self.assertEqual(body, nagaya_web.WEB.joinpath(name.rpartition("/")[2]).read_bytes())

    def test_a_file_that_is_not_of_the_page_is_a_404(self):
        self.assertTrue(nagaya_web.WEB.joinpath("README.md").is_file())
        for path in ("/app/web/nothing.js", "/app/web/README.md", "/app/web/", "/app/web/../nagaya_web.py",
                     "/app/web/..%2Fnagaya_web.py", "/app/web/%2e%2e/nagaya_web.py", "/app/web/sub/core.js", "/app/web/../web/core.js"):
            self.assertEqual(self.get(path)[0], 404, path)

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
        cls.nagayas = [("north", f"http://127.0.0.1:{cls.first.server_address[1]}"),
                       ("south", f"http://127.0.0.1:{cls.other.server_address[1]}")]
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
        self.assertEqual(json.loads(self.get("/at/south/council/v2/identity")[1]), {"models": [{"model": "qwen-image-fusal"}]})
        self.assertEqual(json.loads(self.get("/at/north/council/v2/identity")[1]), {"models": []})
        self.assertEqual(json.loads(self.get("/council/v2/identity")[1]), {"models": []})

    def test_a_redirect_under_at_a_name_keeps_the_name(self):
        r, _ = self.get("/at/south/council/v2/objects/pending:" + "0" * 64)
        self.assertEqual((r.status, r.getheader("location")), (308, "/at/south/council/v2/objects/blake3:ab"))

    def test_an_unknown_name_is_a_404_and_does_not_pass(self):
        before = len(FakeNagaya.seen)
        r, body = self.get("/at/elsewhere/council/v2/identity")
        self.assertEqual(r.status, 404)
        self.assertIn(b"north, south", body)
        r, _ = self.get("/at/south/elsewhere")
        self.assertEqual(r.status, 404)
        self.assertEqual(len(FakeNagaya.seen), before)

    def test_the_names_parse_from_the_command_line(self):
        self.assertEqual(nagaya_web.nagaya("south=http://h:8090/"), ("south", "http://h:8090"))
        for bad in ("south", "=http://h:1", "a/b=http://h:1"):
            with self.assertRaises(Exception):
                nagaya_web.nagaya(bad)

    def test_the_machine_of_the_proxy_comes_first_in_the_list(self):
        names = lambda host: [n for n, _ in nagaya_web.local_first(self.nagayas, host)]
        self.assertEqual(names("south"), ["south", "north"])
        self.assertEqual(names("north"), ["north", "south"])
        self.assertEqual(names("east"), ["north", "south"], "another machine keeps the order of the list")

    def test_the_seeds_come_from_the_flags_else_the_variable_else_this_machine(self):
        env = {"NAGAYA_URLS": "http://192.0.2.10:8091/, http://192.0.2.20:8090"}
        self.assertEqual(nagaya_web.seeds(["http://192.0.2.30:1/"], env), ["http://192.0.2.30:1"])
        self.assertEqual(nagaya_web.seeds(None, env), ["http://192.0.2.10:8091", "http://192.0.2.20:8090"])
        self.assertEqual(nagaya_web.seeds(None, {"NAGAYA_URLS": " , "}), ["http://127.0.0.1:8090"])
        self.assertEqual(nagaya_web.seeds(None, {}), ["http://127.0.0.1:8090"])



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
        ClusterSeed.view = {"node": "north", "dead": ["east"], "nodes": [
            {"name": "north", "self": True, "http": "192.0.2.10:8091"},
            {"name": "yuki", "self": False, "http": None},
            {"name": "south", "self": False, "http": "192.0.2.20:8090"}]}

    @classmethod
    def tearDownClass(cls):
        cls.seed.shutdown()
        cls.seed.server_close()

    def test_the_nodes_come_from_the_first_seed_that_answers(self):
        found = nagaya_web.discover([self.dead_url, self.seed_url], "east")
        self.assertEqual(found, [("north", "http://192.0.2.10:8091"), ("south", "http://192.0.2.20:8090")])

    def test_the_machine_of_the_proxy_comes_first(self):
        found = nagaya_web.discover([self.seed_url], "south")
        self.assertEqual([n for n, _ in found], ["south", "north"])

    def test_no_seed_that_answers_is_an_error_that_names_the_seeds(self):
        with self.assertRaises(SystemExit) as caught:
            nagaya_web.discover([self.dead_url], "south")
        self.assertIn(self.dead_url, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
