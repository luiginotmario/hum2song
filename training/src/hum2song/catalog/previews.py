"""30-second previews from the iTunes Search API and the Deezer public API (D-018).

Both are public, login-free endpoints that return a preview URL per track. Only metadata
and preview URLs are collected here; audio is fetched transiently by extract_previews.py
and deleted once melody tracks are cached (D-009: store features, not audio).
Rate limits: iTunes Search about 20 calls/minute; Deezer 50 calls per 5 seconds.
"""

import re
import threading
import time
import unicodedata
from difflib import SequenceMatcher

import httpx

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"
APPLE_CHART_URL = (
    "https://rss.applemarketingtools.com/api/v2/{country}/music/most-played/100/songs.json"
)
DEEZER_API = "https://api.deezer.com"
YOUTUBE_OEMBED_URL = "https://www.youtube.com/oembed"
ITUNES_INTERVAL_S = 3.1
DEEZER_INTERVAL_S = 0.15
LOOKUP_BATCH = 150
TITLE_WEIGHT = 0.7
ARTIST_THRESHOLD = 0.8
TITLE_THRESHOLD = 0.9
RETRIES = 3
TIMEOUT_S = 20.0
BRACKETS = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
FEATURING = re.compile(r"\b(feat|ft|featuring)\b\.?.*$")
VIDEO_WORDS = re.compile(
    r"\b(official|video|audio|lyrics?|lyric|hd|hq|4k|remaster(ed)?|music|mv|clip|visualizer)\b"
)
NON_WORD = re.compile(r"[^a-z0-9 ]+")


class RateLimiter:
    """Blocks so that calls are at least `interval_s` apart."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            delay = self.last + self.interval_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


def plain(text: str) -> str:
    """Lowercase ASCII words: accents, brackets, 'feat.' tails and punctuation removed."""
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    lowered = BRACKETS.sub(" ", ascii_text.lower()).replace("&", " and ")
    words = NON_WORD.sub(" ", FEATURING.sub(" ", lowered))
    return " ".join(words.split())


def clean_video_title(title: str) -> str:
    """YouTube title -> search text: drops bracketed parts and words like 'official video'."""
    return " ".join(VIDEO_WORDS.sub(" ", plain(title)).split())


def title_score(wanted: str, found: str) -> float:
    wanted, found = plain(wanted), plain(found)
    if not wanted or not found:
        return 0.0
    if wanted == found:
        return 1.0
    contained = wanted in found or found in wanted
    return max(0.9 if contained else 0.0, SequenceMatcher(None, wanted, found).ratio())


def artist_score(wanted: str, found: str) -> float:
    """Share of the wanted artist's words found in the result's artist."""
    words = set(plain(wanted).split())
    return len(words & set(plain(found).split())) / max(len(words), 1)


def match_score(target: dict, row: dict) -> float:
    """Title similarity, blended 70/30 with artist overlap when the target has an artist."""
    title = title_score(target["title"], row["title"])
    if not target.get("artist"):
        return title
    return TITLE_WEIGHT * title + (1.0 - TITLE_WEIGHT) * artist_score(
        target["artist"], row["artist"]
    )


def best_match(rows: list[dict], target: dict) -> dict | None:
    """Best-scoring row with a preview (first wins ties, i.e. the service's ranking)."""
    threshold = ARTIST_THRESHOLD if target.get("artist") else TITLE_THRESHOLD
    scored = [
        (match_score(target, row), -i, row) for i, row in enumerate(rows) if row["preview_url"]
    ]
    if not scored:
        return None
    score, _order, row = max(scored, key=lambda item: item[:2])
    return {**row, "match_score": round(score, 3)} if score >= threshold else None


def split_video_title(title: str) -> dict:
    """'Artist - Song (Official Video)' -> {'artist', 'title'}; no dash -> title only."""
    parts = re.split(r"\s+[-\u2013\u2014|]\s+", title or "", maxsplit=1)
    if len(parts) == 2:
        return {"artist": clean_video_title(parts[0]), "title": clean_video_title(parts[1])}
    return {"artist": "", "title": clean_video_title(title or "")}


def song_key(artist: str, title: str) -> str:
    return f"{plain(artist)}|{plain(title)}"


def itunes_row(result: dict) -> dict:
    return {
        "song_id": f"itunes:{result['trackId']}",
        "source": "itunes_preview",
        "title": result.get("trackName", ""),
        "artist": result.get("artistName", ""),
        "genre": result.get("primaryGenreName", ""),
        "isrc": None,
        "preview_url": result.get("previewUrl"),
        "full_duration_s": (result.get("trackTimeMillis") or 0) / 1000.0,
    }


def deezer_row(track: dict) -> dict:
    return {
        "song_id": f"deezer:{track['id']}",
        "source": "deezer_preview",
        "title": track.get("title", ""),
        "artist": (track.get("artist") or {}).get("name", ""),
        "genre": "",
        "isrc": track.get("isrc"),
        "preview_url": track.get("preview") or None,
        "full_duration_s": float(track.get("duration") or 0),
    }


class PreviewClient:
    """Rate-limited JSON calls to the three public endpoints."""

    def __init__(self) -> None:
        self.http = httpx.Client(timeout=TIMEOUT_S, follow_redirects=True)
        self.itunes = RateLimiter(ITUNES_INTERVAL_S)
        self.deezer = RateLimiter(DEEZER_INTERVAL_S)
        self.light = RateLimiter(DEEZER_INTERVAL_S)

    def get_json(self, url: str, limiter: RateLimiter, params: dict | None = None):
        for attempt in range(RETRIES):
            limiter.wait()
            try:
                response = self.http.get(url, params=params)
            except httpx.HTTPError:
                time.sleep(2.0 * (attempt + 1))
                continue
            if response.status_code == 200:
                return response.json()
            if response.status_code in (401, 403, 404):
                return None
            time.sleep(5.0 * (attempt + 1))
        return None

    def itunes_search(self, term: str, limit: int = 10) -> list[dict]:
        params = {"term": term, "entity": "song", "media": "music", "limit": limit}
        data = self.get_json(ITUNES_SEARCH_URL, self.itunes, params) or {}
        return [itunes_row(r) for r in data.get("results", []) if r.get("kind") == "song"]

    def itunes_lookup(self, track_ids: list[str]) -> list[dict]:
        rows = []
        for first in range(0, len(track_ids), LOOKUP_BATCH):
            ids = ",".join(track_ids[first : first + LOOKUP_BATCH])
            data = self.get_json(ITUNES_LOOKUP_URL, self.itunes, {"id": ids}) or {}
            rows += [itunes_row(r) for r in data.get("results", []) if r.get("kind") == "song"]
        return rows

    def apple_chart_ids(self, country: str) -> list[str]:
        data = self.get_json(APPLE_CHART_URL.format(country=country), self.light) or {}
        return [item["id"] for item in data.get("feed", {}).get("results", [])]

    def deezer_search(self, query: str, limit: int = 10) -> list[dict]:
        data = self.get_json(f"{DEEZER_API}/search", self.deezer, {"q": query, "limit": limit})
        return [deezer_row(t) for t in (data or {}).get("data", [])]

    def deezer_chart_playlists(self, limit: int = 100) -> list[int]:
        data = self.get_json(f"{DEEZER_API}/chart/0/playlists", self.deezer, {"limit": limit})
        return [p["id"] for p in (data or {}).get("data", [])]

    def deezer_playlist_tracks(self, playlist_id: int, limit: int = 300) -> list[dict]:
        url = f"{DEEZER_API}/playlist/{playlist_id}/tracks"
        data = self.get_json(url, self.deezer, {"limit": limit})
        return [deezer_row(t) for t in (data or {}).get("data", [])]

    def deezer_track(self, track_id: str) -> dict | None:
        """Fresh track row; Deezer preview URLs are signed and expire within minutes."""
        data = self.get_json(f"{DEEZER_API}/track/{track_id}", self.deezer)
        return deezer_row(data) if data and "id" in data else None

    def download(self, url: str, path) -> None:
        with self.http.stream("GET", url) as response:
            response.raise_for_status()
            with open(path, "wb") as handle:
                for block in response.iter_bytes():
                    handle.write(block)

    def youtube_title(self, video_id: str) -> str | None:
        url = f"https://www.youtube.com/watch?v={video_id}"
        data = self.get_json(YOUTUBE_OEMBED_URL, self.light, {"url": url, "format": "json"})
        return None if data is None else f"{data.get('title', '')}"
