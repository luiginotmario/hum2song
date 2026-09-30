"""Hum screen: static page plus a same-origin proxy to the live search API."""

import importlib.util
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def load_screen():
    spec = importlib.util.spec_from_file_location("hum_screen", ROOT / "web" / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav_bytes() -> bytes:
    frames = b"\x00\x00" * 800
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(frames),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        8000,
        16000,
        2,
        16,
        b"data",
        len(frames),
    )
    return header + frames


def multipart(wav: bytes) -> tuple[bytes, str]:
    boundary = "----humtest"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; filename="hum.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body += wav + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class RecordingApi(BaseHTTPRequestHandler):
    last = None
    payload = b'{"ok":true}'

    def do_GET(self) -> None:
        self._reply()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        RecordingApi.last = {
            "path": self.path,
            "type": self.headers.get("Content-Type"),
            "body": self.rfile.read(length),
        }
        self._reply()

    def _reply(self) -> None:
        body = RecordingApi.payload
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


def serve_api() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingApi)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def fetch(url: str, data: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes]:
    method = "POST" if data is not None else "GET"
    request = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=5) as response:
            return response.getcode(), response.read()
    except HTTPError as error:
        return error.code, error.read()


def test_page_is_served_and_search_is_proxied():
    screen = load_screen()
    api = serve_api()
    host, port = api.server_address
    RecordingApi.payload = b'{"decision":{"action":"show","message":"Best match: Clear Hit."}}'
    ui = screen.listen("127.0.0.1", 0, f"http://{host}:{port}")
    ui_port = ui.server_address[1]
    try:
        status, page = fetch(f"http://127.0.0.1:{ui_port}/")
        assert status == 200
        text = page.decode()
        assert 'id="listen"' in text
        assert "Tap to Listen" in text
        assert 'aria-label="Tap to listen"' in text
        style, css = fetch(f"http://127.0.0.1:{ui_port}/app.css")
        script, _ = fetch(f"http://127.0.0.1:{ui_port}/app.js")
        worklet, _ = fetch(f"http://127.0.0.1:{ui_port}/capture-worklet.js")
        assert style == script == worklet == 200
        assert b"#60a088" not in css
        assert b"backdrop-filter" not in css
        assert b"#ffffff" in css
        assert b"background: #000" in css

        wav = wav_bytes()
        body, kind = multipart(wav)
        status, payload = fetch(
            f"http://127.0.0.1:{ui_port}/search?decide=true&mode=windows",
            data=body,
            headers={"Content-Type": kind},
        )
        assert status == 200
        assert json.loads(payload)["decision"]["action"] == "show"
        assert RecordingApi.last["path"] == "/search?decide=true&mode=windows"
        assert wav in RecordingApi.last["body"]
        assert b"hum.wav" in RecordingApi.last["body"]
    finally:
        ui.shutdown()
        ui.server_close()
        api.shutdown()
        api.server_close()


def test_missing_api_is_a_clear_error():
    screen = load_screen()
    ui = screen.listen("127.0.0.1", 0, "http://127.0.0.1:1")
    try:
        status, payload = fetch(
            f"http://127.0.0.1:{ui.server_address[1]}/health",
        )
        assert status == 502
        assert json.loads(payload)["message"] == "search API unreachable"
    finally:
        ui.shutdown()
        ui.server_close()


class Body:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_meta_is_answered_locally():
    screen = load_screen()
    screen.lookup_meta = lambda title, artist: {
        "title": title,
        "artist": artist,
        "cover": "https://example.com/cover.jpg",
        "preview": "https://example.com/preview.m4a",
    }
    api = serve_api()
    RecordingApi.last = None
    ui = screen.listen("127.0.0.1", 0, f"http://{api.server_address[0]}:{api.server_address[1]}")
    try:
        status, body = fetch(
            f"http://127.0.0.1:{ui.server_address[1]}/meta?title=Purple%20Rain&artist=Prince"
        )
        assert status == 200
        meta = json.loads(body)
        assert meta["cover"] == "https://example.com/cover.jpg"
        assert meta["preview"].endswith(".m4a")
        assert RecordingApi.last is None
        status, _ = fetch(f"http://127.0.0.1:{ui.server_address[1]}/meta")
        assert status == 400
    finally:
        ui.shutdown()
        ui.server_close()
        api.shutdown()
        api.server_close()


def test_lookup_meta_reads_itunes_then_deezer():
    screen = load_screen()
    itunes = json.dumps(
        {
            "results": [
                {
                    "trackName": "Purple Rain",
                    "artistName": "Prince",
                    "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/x/100x100bb.jpg",
                    "previewUrl": "https://audio-ssl.itunes.apple.com/preview.m4a",
                }
            ]
        }
    ).encode()
    deezer = json.dumps(
        {
            "data": [
                {
                    "title": "When Doves Cry",
                    "artist": {"name": "Prince"},
                    "album": {"cover_xl": "https://e-cdns-images.dzcdn.net/images/cover/abc.jpg"},
                    "preview": "https://cdns-preview.dzcdn.net/stream/preview.mp3",
                }
            ]
        }
    ).encode()

    def fetch(request, timeout=8):
        url = request.full_url
        if "itunes.apple.com" in url and "When" in url:
            return Body(b'{"results":[]}')
        if "itunes.apple.com" in url:
            return Body(itunes)
        if "deezer.com" in url:
            return Body(deezer)
        raise AssertionError(url)

    hit = screen.lookup_meta("Purple Rain", "Prince", fetch=fetch)
    assert "300x300bb" in hit["cover"]
    assert hit["preview"].endswith(".m4a")
    fallback = screen.lookup_meta("When Doves Cry", "Prince", fetch=fetch)
    assert fallback["cover"].startswith("https://e-cdns-images.dzcdn.net/")
    assert fallback["preview"].endswith(".mp3")
    youtube = "https://www.youtube.com/results?search_query="
    page = (ROOT / "web" / "app.js").read_text()
    assert youtube in page
    assert "youtube.com/watch" not in page


def test_unknown_paths_are_not_served():
    screen = load_screen()
    ui = screen.listen("127.0.0.1", 0, "http://127.0.0.1:1")
    try:
        status, _ = fetch(f"http://127.0.0.1:{ui.server_address[1]}/serve.py")
        assert status == 404
        status, _ = fetch(f"http://127.0.0.1:{ui.server_address[1]}/../README.md")
        assert status == 404
    finally:
        ui.shutdown()
        ui.server_close()
