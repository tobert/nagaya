"""The transport of the driver: one POST of a JSON body to the nagaya, and its JSON reply. A test gives a stub with
the same method `post` in its place."""
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://198.51.100.10:8091"


def base_url(arg=None):
    """The base URL: the argument, else NAGAYA_URL, else b70host's nagaya."""
    return (arg or os.environ.get("NAGAYA_URL") or DEFAULT_URL).rstrip("/")


class HttpTransport:
    """HTTP to one nagaya. A 429 or a 503 waits and tries again, up to `tries` times; another error raises."""

    def __init__(self, url, timeout=900, tries=20):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.tries = tries

    def post(self, path, body):
        data = json.dumps(body).encode()
        for attempt in range(self.tries):
            req = urllib.request.Request(self.url + path, data, {"content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    e.close()
                    time.sleep(0.5 + attempt * 0.5)
                    continue
                raise RuntimeError(f"POST {path} failed with {e.code}: {e.read()[:300]!r}") from e
        raise RuntimeError(f"POST {path} stayed busy after {self.tries} tries. Wait, or lower loop.in_flight.")
