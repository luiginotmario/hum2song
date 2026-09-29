"""The D-025 window pipeline for live search (D-027).

  1. first stage  the top FIRST_K songs by the chunk HNSW index (the D-017 score) and the
                  top FIRST_K by window votes from the window HNSW index (db.window_votes)
  2. features     for every candidate, from its stored vectors: base (best chunk), window
                  vote, time-consistent path score (seq) and minus the key-normalized DTW
                  error on the span the path picked (dtw), exactly as scripts/eval_rerank.py
  3. re-rank      per-query z-scored weighted sum with the weights chosen on CHAD val (D-025)

Song vectors are cached in memory after their first use (SongCache), so repeated candidates
cost one database round trip only once.
"""

from dataclasses import dataclass

import numpy as np

from hum2song.catalog.db import (
    SongVectors,
    best_per_song,
    nearest_chunks,
    song_vectors,
    window_votes,
)
from hum2song.catalog.matching import (
    NO_MATCH,
    dtw_mae,
    frames,
    fuse,
    key_normalized,
    sequence_path,
)

WIN_S = 5.0
HOP_S = 1.0
MIN_VOICED = 0.25
FIRST_K = 200
CHUNK_POOL = 1000
WINDOW_POOL = 400
WEIGHTS = (1.0, 1.0, 0.5, 1.0)  # base, window, seq, dtw: D-025, chosen on CHAD val only


@dataclass
class RankedSong:
    song_id: str
    score: float
    best_start_s: float
    base: float
    window: float


class SongCache:
    """song_id -> SongVectors, loaded from the database on first use."""

    def __init__(self) -> None:
        self.songs: dict[str, SongVectors] = {}

    def get(self, connection, song_ids: list[str]) -> dict[str, SongVectors]:
        missing = [song for song in song_ids if song not in self.songs]
        if missing:
            self.songs.update(song_vectors(connection, missing))
        return {song: self.songs[song] for song in song_ids if song in self.songs}


def candidates(connection, query_emb: np.ndarray, window_embs: np.ndarray) -> list[str]:
    """Union of the chunk-index and window-index first stages, FIRST_K songs each."""
    by_chunk = [
        row[0] for row in best_per_song(nearest_chunks(connection, query_emb, CHUNK_POOL), FIRST_K)
    ]
    by_window = [song for song, _ in window_votes(connection, window_embs, WINDOW_POOL)][:FIRST_K]
    return list(dict.fromkeys(by_chunk + by_window))


def grid_matrix(similarity: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Query windows x the song's window grid (index = start / HOP_S), NO_MATCH for gaps."""
    keys = np.round(starts / HOP_S).astype(np.int64)
    matrix = np.full((similarity.shape[0], int(keys.max()) + 1), NO_MATCH, dtype=np.float32)
    matrix[:, keys] = similarity
    return matrix


def song_features(song: SongVectors, contour, query_emb, window_embs) -> tuple:
    """(base, window, seq, dtw, best start in s) of one candidate."""
    base = float((song.chunks @ query_emb).max()) if len(song.chunks) else NO_MATCH
    if not len(song.windows):
        return base, NO_MATCH, -np.inf, -np.inf, 0.0
    similarity = window_embs @ song.windows.T
    seq, first, last = sequence_path(grid_matrix(similarity, song.window_starts))
    span = song.contour[frames(first * HOP_S) : frames(last * HOP_S + WIN_S)]
    dtw = -dtw_mae(key_normalized(span), key_normalized(contour)) if len(span) else -np.inf
    return base, float(similarity.max(axis=1).mean()), seq, dtw, first * HOP_S


def rank_songs(songs: dict[str, SongVectors], contour, query_emb, window_embs) -> list:
    """All candidates, best first, by the fused D-025 score."""
    ids = list(songs)
    if not ids:
        return []
    features = np.array(
        [song_features(songs[s], contour, query_emb, window_embs) for s in ids], dtype=np.float64
    )
    fused = fuse([features[:, column] for column in range(4)], list(WEIGHTS))
    order = np.argsort(-fused, kind="stable")
    return [
        RankedSong(ids[i], float(fused[i]), float(features[i, 4]), features[i, 0], features[i, 1])
        for i in order
    ]


def window_search(connection, cache: SongCache, contour, query_emb, window_embs, top_k: int):
    songs = cache.get(connection, candidates(connection, query_emb, window_embs))
    return rank_songs(songs, contour, query_emb, window_embs)[:top_k]
