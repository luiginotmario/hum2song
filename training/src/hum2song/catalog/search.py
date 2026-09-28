"""Query audio -> contour embedding -> top-k songs from the library (D-017).

Queries use the same path as every contour eval: RMVPE on 16 kHz audio, the cleaned
contour with unvoiced ends trimmed and capped at 20 s, then the D-012 contour encoder.
Whistles would need the spectral-peak tracker (D-007); not wired in yet.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hum2song.audio import load_audio
from hum2song.catalog.db import song_hits
from hum2song.contour.evaluate import embed_contours, query_contour
from hum2song.contour.features import rmvpe_contour, voiced_fraction
from hum2song.contour.melody import TRACK_RATE
from hum2song.contour.trackers import load_rmvpe, rmvpe_track
from hum2song.contour.train import load_checkpoint

MIN_QUERY_VOICED_S = 1.0


class QueryEncoder:
    """RMVPE plus the contour encoder, loaded once."""

    def __init__(self, ckpt: Path, rmvpe_weights: Path, device: torch.device) -> None:
        self.device = device
        self.model, _config, _step = load_checkpoint(ckpt, device)
        self.rmvpe = load_rmvpe(rmvpe_weights, device)
        self.model_ver = f"{ckpt.parent.name}/{ckpt.name}"

    def contour(self, audio_16k: np.ndarray) -> np.ndarray:
        return query_contour(rmvpe_contour(rmvpe_track(self.rmvpe, audio_16k)))

    def embed(self, audio_16k: np.ndarray) -> tuple[np.ndarray, float]:
        """Embedding and seconds of voiced query audio."""
        contour = self.contour(audio_16k)
        voiced_s = voiced_fraction(contour) * len(contour) * 0.02
        return embed_contours(self.model, [contour], self.device)[0], voiced_s


def search_audio(encoder: QueryEncoder, connection, path: Path, top_k: int) -> dict:
    """Top-k songs for one audio file; too little voicing returns no results."""
    audio = load_audio(path, TRACK_RATE, trim=False)
    if len(audio) < MIN_QUERY_VOICED_S * TRACK_RATE:
        return {"voiced_s": 0.0, "results": [], "message": "query shorter than 1 s"}
    embedding, voiced_s = encoder.embed(audio)
    if voiced_s < MIN_QUERY_VOICED_S:
        return {"voiced_s": round(voiced_s, 2), "results": [], "message": "no melody heard"}
    hits = song_hits(connection, embedding, top_k)
    return {"voiced_s": round(voiced_s, 2), "results": [asdict(hit) for hit in hits]}
