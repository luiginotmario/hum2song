"""Serve the hum screen and forward search to the live API.

    python web/serve.py
    python web/serve.py --api http://127.0.0.1:8000 --port 8080

The page and the API share this origin, so the browser can post a hum without
CORS on the already-running search process.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urlsplit
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
META_TIMEOUT_S = 8
META_UA = "hum2song/0.1 (github.com/luiginotmario/hum2song)"
ITUNES_SEARCH = "https://itunes.apple.com/search"
DEEZER_SEARCH = "https://api.deezer.com/search"
MB_RECORDING = "https://musicbrainz.org/ws/2/recording"
COVER_ART = "https://coverartarchive.org/release"
META_CACHE: dict[str, dict] = {}
META_LOCK = threading.Lock()


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


def https_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("https://"):
        return text
    return None


def words(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {piece for piece in cleaned.split() if piece}


def overlap(wanted: set[str], found: set[str]) -> float:
    if not wanted or not found:
        return 0.0
    return len(wanted & found) / len(wanted)


def match_score(title: str, artist: str, got_title: str, got_artist: str) -> float:
    return overlap(words(title), words(got_title)) * 2 + overlap(words(artist), words(got_artist))


def fetch_json(url: str, fetch, headers: dict | None = None) -> dict | None:
    request_headers = {"Accept": "application/json", "User-Agent": META_UA}
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    try:
        with fetch(request, timeout=META_TIMEOUT_S) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    try:
        data = json.loads(payload.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def larger_itunes_art(url: str) -> str:
    return url.replace("100x100bb", "300x300bb")


def itunes_hit(title: str, artist: str, fetch) -> dict | None:
    term = " ".join(part for part in (title, artist) if part)
    url = f"{ITUNES_SEARCH}?{urlencode({'term': term, 'entity': 'song', 'limit': 5})}"
    data = fetch_json(url, fetch)
    if not data:
        return None
    best = None
    best_score = 0.0
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        score = match_score(title, artist, str(item.get("trackName") or ""), str(item.get("artistName") or ""))
        if score > best_score:
            best = item
            best_score = score
    if best is None or best_score < 1:
        return None
    art = https_url(best.get("artworkUrl100"))
    return {
        "cover": larger_itunes_art(art) if art else None,
        "preview": https_url(best.get("previewUrl")),
    }


def deezer_hit(title: str, artist: str, fetch) -> dict | None:
    query = f'track:"{title}"'
    if artist:
        query = f'{query} artist:"{artist}"'
    url = f"{DEEZER_SEARCH}?q={quote(query)}&limit=5"
    data = fetch_json(url, fetch)
    if not data:
        return None
    best = None
    best_score = 0.0
    for item in data.get("data") or []:
        if not isinstance(item, dict):
            continue
        got_artist = item.get("artist") if isinstance(item.get("artist"), dict) else {}
        score = match_score(title, artist, str(item.get("title") or ""), str(got_artist.get("name") or ""))
        if score > best_score:
            best = item
            best_score = score
    if best is None or best_score < 1:
        return None
    album = best.get("album") if isinstance(best.get("album"), dict) else {}
    cover = https_url(album.get("cover_xl") or album.get("cover_big") or album.get("cover_medium"))
    return {"cover": cover, "preview": https_url(best.get("preview"))}


def musicbrainz_cover(title: str, artist: str, fetch) -> dict | None:
    clauses = [f'recording:"{title}"']
    if artist:
        clauses.append(f'artist:"{artist}"')
    query = " AND ".join(clauses)
    url = f"{MB_RECORDING}?query={quote(query)}&fmt=json&limit=3"
    data = fetch_json(url, fetch)
    if not data:
        return None
    for recording in data.get("recordings") or []:
        if not isinstance(recording, dict):
            continue
        for release in recording.get("releases") or []:
            if not isinstance(release, dict):
                continue
            release_id = str(release.get("id") or "")
            if not release_id:
                continue
            cover = https_url(f"{COVER_ART}/{release_id}/front-250")
            if cover:
                return {"cover": cover, "preview": None}
    return None


def lookup_meta(title: str, artist: str, fetch=urlopen) -> dict:
    title = " ".join(title.split())
    artist = " ".join(artist.split())
    key = f"{title.casefold()}|{artist.casefold()}"
    with META_LOCK:
        cached = META_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    found = itunes_hit(title, artist, fetch) or deezer_hit(title, artist, fetch) or musicbrainz_cover(title, artist, fetch)
    result = {
        "title": title,
        "artist": artist,
        "cover": (found or {}).get("cover"),
        "preview": (found or {}).get("preview"),
    }
    with META_LOCK:
        META_CACHE[key] = result
    return dict(result)


def meta_from_query(path: str) -> tuple[int, dict]:
    params = parse_qs(urlsplit(path).query)
    title = (params.get("title") or [""])[0].strip()
    artist = (params.get("artist") or [""])[0].strip()
    if not title or len(title) > 200 or len(artist) > 200:
        return 400, {"message": "title is required"}
    return 200, lookup_meta(title, artist)


class HumServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], api: str) -> None:
        self.api_base = api_url(api)
        self.daemon_threads = True
        super().__init__(server_address, HumHandler)


class HumHandler(BaseHTTPRequestHandler):
    server: HumServer

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/meta":
            self.serve_meta()
            return
        if is_proxy_path(self.path):
            self.proxy()
            return
        if urlsplit(self.path).path == "/favicon.ico":
            self.send_bytes(204, b"", "image/x-icon")
            return
        self.serve_static()

    def serve_meta(self) -> None:
        status, payload = meta_from_query(self.path)
        body = json.dumps(payload).encode()
        cache = "private, max-age=86400" if status == 200 else "no-store"
        self.send_bytes(status, body, "application/json", cache=cache)

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
