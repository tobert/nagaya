#!/usr/bin/env python3
"""A web app in front of one or more nagayas: engines, images, chat, speech and embeddings, through council v2.

The server sends the page `web/index.html` and its files under `/app/web/`. It passes each request under `/council/`
and `/nagaya/` to the first nagaya as it is. A request under `/at/<name>/council/` or `/at/<name>/nagaya/` goes to
the nagaya with that name. It streams each response, so the events of a generation reach the page as the nagaya
sends them. The page and the nagayas then have one origin, and the browser needs no CORS. `GET /app/nagayas` lists
the names, and the page shows them.

Examples:

    tools/nagaya_web.py                      # http://127.0.0.1:8096, in front of each node of the cluster
    tools/nagaya_web.py --port 9000 --nagaya local=http://127.0.0.1:8091
    tools/nagaya_web.py --url http://127.0.0.1:8091    # one nagaya, with the name `nagaya`

The app has no login. Bind it to another address than 127.0.0.1 only on a network that you trust.
It needs only the Python standard library.
"""

import argparse
import http.client
import http.server
import json
import pathlib
import re
import socket
import sys
import urllib.parse

# The seeds of the lab. The app asks the first seed that answers for the cluster route, and the nodes there become the
# nagayas of the page. The first nagaya also serves the plain paths, so `local_first` puts the machine of the proxy
# first.
DEFAULT_NAGAYAS = [("b70host", "http://198.51.100.10:8091"), ("gfx1151host", "http://198.51.100.20:8090")]


def local_first(nagayas, host):
    """The list with the nagaya named `host` first, and the others in their order."""
    return [n for n in nagayas if n[0] == host] + [n for n in nagayas if n[0] != host]


def discover(seeds, host):
    """The (name, URL) of each live node in the cluster view of the first seed that answers, with `host` first."""
    failures = []
    for seed in seeds:
        target = urllib.parse.urlsplit(seed)
        connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=5)
        try:
            connection.request("GET", "/nagaya/v1/cluster")
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                failures.append(f"{seed} gave {response.status}")
                continue
            view = json.loads(body)
        except (OSError, ValueError) as error:
            failures.append(f"{seed} gave {error}")
            continue
        finally:
            connection.close()
        nagayas = []
        for node in view["nodes"]:
            # A node without the key `server` has no HTTP address (spec 11.4a), so the app cannot pass to it.
            if node["http"] is None:
                print(f"node {node['name']!r} publishes no HTTP address, so the app leaves it out", file=sys.stderr)
                continue
            nagayas.append((node["name"], "http://" + node["http"]))
        return local_first(nagayas, host)
    sys.exit("no seed gave the cluster route: " + "; ".join(failures) + ". Start a nagaya, or give --nagaya.")
# The folder of the page: `index.html`, its stylesheet and its scripts (web/README.md).
WEB = pathlib.Path(__file__).with_name("web")
# The kinds of file that the app serves from the folder. A file of another kind, such as the README, is a 404.
KINDS = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
PREFIXES = ("/council/", "/nagaya/")
# The headers that the server passes on, in each direction. The others belong to one connection.
REQUEST_HEADERS = ("content-type", "accept", "range")
RESPONSE_HEADERS = ("content-type", "location", "content-range", "accept-ranges", "request-id", "cache-control")
NAME = re.compile(r"[A-Za-z0-9_.-]+")


def nagaya(text):
    """The (name, URL) of a command-line value `name=http://host:port`."""
    name, _, url = text.partition("=")
    if not NAME.fullmatch(name) or not url:
        raise argparse.ArgumentTypeError(f"give a nagaya as name=http://host:port, with a name of letters, digits, "
                                         f"`.`, `_` and `-`, not {text!r}")
    return name, url.rstrip("/")


def handler(nagayas):
    """The request handler class for `nagayas`: a list of (name, URL), or one URL such as http://127.0.0.1:8091."""
    if isinstance(nagayas, str):
        nagayas = [("nagaya", nagayas)]
    targets = {}
    for name, url in nagayas:
        target = urllib.parse.urlsplit(url)
        if target.scheme != "http" or not target.hostname:
            raise ValueError(f"the URL of nagaya {name!r} must be http://host:port, not {url!r}")
        targets[name] = (url, target)
    listing = json.dumps({"nagayas": [{"name": name, "url": url} for name, url in nagayas]}).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        # HTTP/1.0: the connection closes after each response, so a stream needs no length and no chunks.
        protocol_version = "HTTP/1.0"

        def log_message(self, format, *args):
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.file(WEB / "index.html", "text/html; charset=utf-8")
            elif self.path.startswith("/app/web/"):
                self.web(self.path[len("/app/web/"):])
            elif self.path == "/app/nagayas":
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(listing)))
                self.end_headers()
                self.wfile.write(listing)
            else:
                self.forward()

        def do_HEAD(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def do_PUT(self):
            self.forward()

        def do_DELETE(self):
            self.forward()

        def web(self, name):
            """Send one file of the page. The name is one plain file name, so no path leaves the folder."""
            kind = KINDS.get(pathlib.PurePosixPath(name).suffix)
            if not NAME.fullmatch(name) or kind is None or not (WEB / name).is_file():
                self.send_error(404, f"the page has no file {name!r}")
                return
            self.file(WEB / name, kind)

        def file(self, path, kind):
            # The app reads the file at each request, and the browser keeps no copy. Thus a change shows at a reload.
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def forward(self):
            name, path, at = nagayas[0][0], self.path, ""
            if self.path.startswith("/at/"):
                name, _, rest = self.path[4:].partition("/")
                path, at = "/" + rest, f"/at/{name}"
                if name not in targets:
                    self.send_error(404, f"this app knows no nagaya {name!r}; it knows {', '.join(targets)}")
                    return
            if not path.startswith(PREFIXES):
                self.send_error(404, f"this app passes on only {' and '.join(PREFIXES)}, and the same under /at/<name>/")
                return
            upstream, target = targets[name]
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {name: self.headers[name] for name in REQUEST_HEADERS if self.headers.get(name)}
            # A generation can run for minutes, as a load can. The nagaya ends a request that runs too long.
            connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=3600)
            try:
                connection.request(self.command, path, body=body, headers=headers)
                response = connection.getresponse()
            except OSError as error:
                self.send_error(502, f"the nagaya at {upstream} did not answer: {error}")
                return
            try:
                self.send_response(response.status, response.reason)
                for header in RESPONSE_HEADERS:
                    value = response.getheader(header)
                    # A redirect to a path of the nagaya stays under the name of that nagaya.
                    if header == "location" and value and value.startswith("/"):
                        value = at + value
                    if value is not None:
                        self.send_header(header, value)
                if response.getheader("content-length") is not None:
                    self.send_header("content-length", response.getheader("content-length"))
                self.end_headers()
                if self.command == "HEAD":
                    return
                # read1 gives what has come so far, so each event goes to the page when the nagaya sends it.
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The page went away. Closing the connection to the nagaya cancels the generation.
                pass
            finally:
                connection.close()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nagaya", type=nagaya, action="append", metavar="NAME=URL",
                        help="a nagaya to pass requests to; give it once for each nagaya. The first one serves the "
                             "plain paths. Default: the nodes of the cluster route of the first seed that answers, "
                             "from " + ", ".join(u for _, u in DEFAULT_NAGAYAS))
    parser.add_argument("--url", help="the base URL of one nagaya, with the name `nagaya`, in place of --nagaya")
    parser.add_argument("--host", default="127.0.0.1", help="the address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8096, help="the port to bind (default 8096)")
    args = parser.parse_args()
    if args.url and args.nagaya:
        parser.error("give --url or --nagaya, not both")
    if args.url:
        nagayas = [("nagaya", args.url.rstrip("/"))]
    else:
        nagayas = args.nagaya or discover([u for _, u in local_first(DEFAULT_NAGAYAS, socket.gethostname())],
                                          socket.gethostname())
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler(nagayas))
    names = ", ".join(f"{n} at {u}" for n, u in nagayas)
    print(f"nagaya web app on http://{args.host}:{args.port}, in front of {names}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
