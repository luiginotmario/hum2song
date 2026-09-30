"""Serve the hum screen and forward search to the live API.

    python web/serve.py
    python web/serve.py --api http://127.0.0.1:8000 --port 8080

The page and the API share this origin, so the browser can post a hum without
CORS on the already-running search process.
"""

from __future__ import annotations

import argparse
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

WEB_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_API = "http://127.0.0.1:8000"
UPSTREAM_TIMEOUT_S = 180
MAX_PROXY_BYTES = 12 * 1024 * 1024
PROXY_PATHS = ("/search", "/health", "/decide")
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.css": "app.css",
    "/app.js": "app.js",
    "/capture-worklet.js": "capture-worklet.js",
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def api_url(raw: str) -> str:
    text = raw.strip().rstrip("/")
    if not text.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("API URL must start with http:// or https://")
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hum screen in front of the live search API")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api", default=None, help="live search base URL (or set H2S_API)")
    args = parser.parse_args(argv)
    raw = args.api or os.environ.get("H2S_API", DEFAULT_API)
    try:
        args.api = api_url(raw)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return args


def is_proxy_path(path: str) -> bool:
    bare = urlsplit(path).path
    return bare in PROXY_PATHS or any(bare.startswith(f"{name}/") for name in PROXY_PATHS)


def static_file(url_path: str) -> Path | None:
    name = STATIC_FILES.get(urlsplit(url_path).path)
    if name is None:
        return None
    path = (WEB_DIR / name).resolve()
    try:
        path.relative_to(WEB_DIR)
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix, "application/octet-stream")


def read_body(handler: BaseHTTPRequestHandler) -> bytes | None:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length < 0 or length > MAX_PROXY_BYTES:
        return None
    if length == 0:
        return b""
    return handler.rfile.read(length)


def forward_request(
    api: str, handler: BaseHTTPRequestHandler, body: bytes
) -> tuple[int, str, bytes]:
    headers = {}
    content_type = handler.headers.get("Content-Type")
    if content_type:
        headers["Content-Type"] = content_type
    accept = handler.headers.get("Accept")
    if accept:
        headers["Accept"] = accept
    request = Request(
        f"{api}{handler.path}",
        data=body if handler.command == "POST" else None,
        headers=headers,
        method=handler.command,
    )
    try:
        with urlopen(request, timeout=UPSTREAM_TIMEOUT_S) as response:
            payload = response.read()
            kind = response.headers.get("Content-Type", "application/json")
            return response.status, kind, payload
    except HTTPError as error:
        payload = error.read()
        fallback = "text/plain; charset=utf-8"
        kind = error.headers.get("Content-Type", fallback) if error.headers else fallback
        return error.code, kind, payload


class HumServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], api: str) -> None:
        self.api_base = api_url(api)
        self.daemon_threads = True
        super().__init__(server_address, HumHandler)


class HumHandler(BaseHTTPRequestHandler):
    server: HumServer

    def do_GET(self) -> None:
        if is_proxy_path(self.path):
            self.proxy()
            return
        if urlsplit(self.path).path == "/favicon.ico":
            self.send_bytes(204, b"", "image/x-icon")
            return
        self.serve_static()

    def do_POST(self) -> None:
        if is_proxy_path(self.path):
            self.proxy()
            return
        self.send_error(404)

    def proxy(self) -> None:
        body = read_body(self)
        if body is None:
            message = b"audio larger than 12 MB"
            self.send_bytes(413, message, "text/plain; charset=utf-8")
            return
        try:
            status, kind, payload = forward_request(self.server.api_base, self, body)
        except URLError:
            message = b'{"message":"search API unreachable"}'
            self.send_bytes(502, message, "application/json")
            return
        self.send_bytes(status, payload, kind)

    def serve_static(self) -> None:
        path = static_file(self.path)
        if path is None:
            self.send_error(404)
            return
        self.send_bytes(200, path.read_bytes(), content_type(path), cache="no-cache")

    def send_bytes(self, status: int, body: bytes, kind: str, cache: str = "no-store") -> None:
        self.send_response(status)
        if kind:
            self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")


def print_banner(host: str, port: int, api: str) -> None:
    shown = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    print(f"Open http://{shown}:{port}")
    print(f"Forwarding search to {api}")


def listen(host: str, port: int, api: str) -> HumServer:
    server = HumServer((host, port), api)
    thread = threading.Thread(target=server.serve_forever, name="hum-screen", daemon=True)
    thread.start()
    return server


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    server = HumServer((args.host, args.port), args.api)
    print_banner(args.host, server.server_address[1], server.api_base)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
