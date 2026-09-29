"""Query audio -> contour -> top-k songs from the library (D-017, D-027).

Queries use the same path as every contour eval: RMVPE on 16 kHz audio, the cleaned
contour with unvoiced ends trimmed, then the D-012 contour encoder.

Two modes:
  windows  (default, D-027) the D-025 pipeline: chunk-index and window-index first stages,
           then re-ranking of the candidates (catalog/window_search.py). The query is cut
           into 5 s windows (up to WINDOW_QUERY_MAX_S of it); its whole-query embedding uses
           the first 20 s as before.
  chunks   the D-017 search: the best 10 s chunk per song from the chunk index.
The windows mode falls back to chunks when the library has no window index yet.
Whistles would need the spectral-peak tracker (D-007); not wired in yet.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hum2song.audio import load_audio
from hum2song.catalog.db import has_windows, song_hits
from hum2song.catalog.matching import query_windows
from hum2song.catalog.window_search import HOP_S, MIN_VOICED, WIN_S, SongCache, window_search
from hum2song.contour.evaluate import embed_contours, query_contour
from hum2song.contour.features import FRAME_S, rmvpe_contour, trim_unvoiced, voiced_fraction
from hum2song.contour.melody import TRACK_RATE
from hum2song.contour.trackers import load_rmvpe, rmvpe_track
from hum2song.contour.train import load_checkpoint

MIN_QUERY_VOICED_S = 1.0
WINDOW_QUERY_MAX_S = 60.0
MODES = ("windows", "chunks")


class QueryEncoder:
    """RMVPE plus the contour encoder, loaded once."""

    def __init__(self, ckpt: Path, rmvpe_weights: Path, device: torch.device) -> None:
        self.device = device
        self.model, _config, _step = load_checkpoint(ckpt, device)
        self.rmvpe = load_rmvpe(rmvpe_weights, device)
        self.model_ver = f"{ckpt.parent.name}/{ckpt.name}"

    def contour(self, audio_16k: np.ndarray) -> np.ndarray:
        """The trimmed query contour (no length cap)."""
        return trim_unvoiced(rmvpe_contour(rmvpe_track(self.rmvpe, audio_16k)))

    def embed(self, contour: np.ndarray) -> np.ndarray:
        return embed_contours(self.model, [query_contour(contour)], self.device)[0]

    def embed_windows(self, contour: np.ndarray) -> np.ndarray:
        capped = contour[: int(round(WINDOW_QUERY_MAX_S / FRAME_S))]
        return embed_contours(
            self.model, query_windows(capped, WIN_S, HOP_S, MIN_VOICED), self.device
        )


def voiced_seconds(contour: np.ndarray) -> float:
    return voiced_fraction(contour) * len(contour) * FRAME_S if len(contour) else 0.0


def chunk_results(encoder, connection, contour: np.ndarray, top_k: int) -> list[dict]:
    return [asdict(hit) for hit in song_hits(connection, encoder.embed(contour), top_k)]


def window_results(encoder, connection, cache, contour: np.ndarray, top_k: int) -> list[dict]:
    capped = contour[: int(round(WINDOW_QUERY_MAX_S / FRAME_S))]
    ranked = window_search(
        connection, cache, capped, encoder.embed(contour), encoder.embed_windows(contour), top_k
    )
    titles = song_titles(connection, [song.song_id for song in ranked])
    return [
        {
            "song_id": song.song_id,
            "title": titles.get(song.song_id, ("", ""))[0],
            "artist": titles.get(song.song_id, ("", ""))[1],
            "score": round(song.score, 4),
            "best_start_s": song.best_start_s,
        }
        for song in ranked
    ]


def song_titles(connection, song_ids: list[str]) -> dict[str, tuple]:
    rows = connection.execute(
        "SELECT song_id, title, artist FROM songs WHERE song_id = ANY(%s)", (song_ids,)
    ).fetchall()
    return {song: (title or "", artist or "") for song, title, artist in rows}


def search_audio(
    encoder, connection, path: Path, top_k: int, mode: str = "windows", cache=None
) -> dict:
    """Top-k songs for one audio file; too little voicing returns no results."""
    audio = load_audio(path, TRACK_RATE, trim=False)
    if len(audio) < MIN_QUERY_VOICED_S * TRACK_RATE:
        return {"voiced_s": 0.0, "results": [], "message": "query shorter than 1 s"}
    contour = encoder.contour(audio)
    voiced_s = voiced_seconds(contour)
    if voiced_s < MIN_QUERY_VOICED_S:
        return {"voiced_s": round(voiced_s, 2), "results": [], "message": "no melody heard"}
    used = mode if mode == "chunks" or has_windows(connection) else "chunks"
    results = (
        chunk_results(encoder, connection, contour, top_k)
        if used == "chunks"
        else window_results(encoder, connection, cache or SongCache(), contour, top_k)
    )
    return {"voiced_s": round(voiced_s, 2), "mode": used, "results": results}
