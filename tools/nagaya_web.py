#!/usr/bin/env python3
"""A web app in front of a nagaya: engines, images, chat and speech, through council v2.

The server sends the page `nagaya_web.html`, and it passes each request under `/council/` and `/nagaya/` to the nagaya
as it is. It streams each response, so the events of a generation reach the page as the nagaya sends them. The page
and the nagaya then have one origin, and the browser needs no CORS.

Examples:

    tools/nagaya_web.py                      # http://127.0.0.1:8096, nagaya at the default URL
    tools/nagaya_web.py --port 9000 --url http://127.0.0.1:8091

The app has no login. Bind it to another address than 127.0.0.1 only on a network that you trust.
It needs only the Python standard library.
"""

import argparse
import http.client
import http.server
import pathlib
import sys
import urllib.parse

DEFAULT_URL = "http://198.51.100.10:8091"
PAGE = pathlib.Path(__file__).with_name("nagaya_web.html")
PREFIXES = ("/council/", "/nagaya/")
# The headers that the server passes on, in each direction. The others belong to one connection.
REQUEST_HEADERS = ("content-type", "accept", "range")
RESPONSE_HEADERS = ("content-type", "location", "content-range", "accept-ranges", "request-id", "cache-control")


def handler(upstream):
    """The request handler class for the nagaya at `upstream`, a URL such as http://127.0.0.1:8091."""
    target = urllib.parse.urlsplit(upstream)
    if target.scheme != "http" or not target.hostname:
        raise ValueError(f"the nagaya URL must be http://host:port, not {upstream!r}")

    class Handler(http.server.BaseHTTPRequestHandler):
        # HTTP/1.0: the connection closes after each response, so a stream needs no length and no chunks.
        protocol_version = "HTTP/1.0"

        def log_message(self, format, *args):
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.page()
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

        def page(self):
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def forward(self):
            if not self.path.startswith(PREFIXES):
                self.send_error(404, f"this app passes on only {' and '.join(PREFIXES)}")
                return
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {name: self.headers[name] for name in REQUEST_HEADERS if self.headers.get(name)}
            # A generation can run for minutes, as a load can. The nagaya ends a request that runs too long.
            connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=3600)
            try:
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
            except OSError as error:
                self.send_error(502, f"the nagaya at {upstream} did not answer: {error}")
                return
            try:
                self.send_response(response.status, response.reason)
                for name in RESPONSE_HEADERS:
                    value = response.getheader(name)
                    if value is not None:
                        self.send_header(name, value)
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
    parser.add_argument("--url", default=DEFAULT_URL, help=f"the base URL of the nagaya (default {DEFAULT_URL})")
    parser.add_argument("--host", default="127.0.0.1", help="the address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8096, help="the port to bind (default 8096)")
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler(args.url.rstrip("/")))
    print(f"nagaya web app on http://{args.host}:{args.port}, in front of {args.url}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
