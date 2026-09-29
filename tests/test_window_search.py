import numpy as np
import soundfile
from fastapi.testclient import TestClient

from hum2song.catalog import search
from hum2song.catalog.db import SongVectors, contour_bytes, contour_from_bytes, vote
from hum2song.catalog.window_search import SongCache, grid_matrix, rank_songs
from hum2song.server import api


def unit(vectors: np.ndarray) -> np.ndarray:
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def melody(seconds: float, offset: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds / 0.02)) * 0.02
    return (60 + offset + 5 * np.sin(1.3 * t) + 2 * np.sin(3.1 * t)).astype(np.float32)


def library(rng) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    query_windows = unit(rng.normal(size=(4, 16)))
    query_emb = unit(rng.normal(size=(1, 16)))[0]
    song = melody(60.0, 3.0)
    query = song[500:1000].copy()  # 10 s starting at 10 s
    target = SongVectors(
        chunks=unit(np.vstack([query_emb + 0.3 * rng.normal(size=16), rng.normal(size=(3, 16))])),
        window_starts=np.arange(0.0, 20.0),
        windows=unit(rng.normal(size=(20, 16))),
        contour=song,
    )
    target.windows[10:14] = unit(query_windows + 0.05 * rng.normal(size=(4, 16)))
    decoys = {
        f"d{i}": SongVectors(
            chunks=unit(rng.normal(size=(4, 16))),
            window_starts=np.arange(0.0, 20.0),
            windows=unit(rng.normal(size=(20, 16))),
            contour=melody(60.0, -2.0 * i)[::-1].copy(),
        )
        for i in range(6)
    }
    return {"target": target, **decoys}, query, query_emb, query_windows


def test_grid_matrix_places_windows_on_their_grid():
    matrix = grid_matrix(np.ones((2, 2), dtype=np.float32), np.array([1.0, 3.0]))
    assert matrix.shape == (2, 4)
    assert (matrix[:, [1, 3]] == 1).all() and (matrix[:, [0, 2]] < 0).all()


def test_rank_songs_puts_the_time_consistent_song_first():
    songs, query, query_emb, query_windows = library(np.random.default_rng(0))
    ranked = rank_songs(songs, query, query_emb, query_windows)
    assert ranked[0].song_id == "target"
    assert ranked[0].best_start_s == 10.0
    assert len(ranked) == len(songs)


def test_rank_songs_handles_songs_without_windows():
    songs, query, query_emb, query_windows = library(np.random.default_rng(1))
    songs["bare"] = SongVectors(chunks=unit(np.random.default_rng(2).normal(size=(2, 16))))
    ranked = rank_songs(songs, query, query_emb, query_windows)
    assert ranked[-1].song_id == "bare"


def test_song_cache_loads_each_song_once(monkeypatch):
    calls = []

    def fake_vectors(_connection, ids):
        calls.append(list(ids))
        return {song: SongVectors(chunks=np.zeros((1, 2), dtype=np.float32)) for song in ids}

    monkeypatch.setattr("hum2song.catalog.window_search.song_vectors", fake_vectors)
    cache = SongCache()
    cache.get(None, ["a", "b"])
    assert set(cache.get(None, ["b", "c"])) == {"b", "c"}
    assert calls == [["a", "b"], ["c"]]


def test_contour_bytes_round_trip_keeps_nan():
    contour = np.array([60.0, np.nan, 61.5], dtype=np.float32)
    back = contour_from_bytes(contour_bytes(contour))
    assert np.isnan(back[1]) and np.allclose(back[[0, 2]], [60.0, 61.5])


def test_vote_is_best_first_input_for_candidates():
    votes = vote([[("a", 0.9)], [("a", 0.7), ("b", 0.6)]])
    assert votes["a"] > votes["b"]


class FakeConnection:
    def __init__(self, windows: bool) -> None:
        self.windows = windows

    def execute(self, *_args):
        return self

    def fetchone(self):
        return (self.windows,)


class FakeEncoder:
    def contour(self, _audio):
        return np.full(200, 60.0, dtype=np.float32)


def test_search_audio_falls_back_to_chunks_without_window_index(tmp_path, monkeypatch):
    path = tmp_path / "hum.wav"
    soundfile.write(path, np.zeros(32000, dtype=np.float32), 16000)
    monkeypatch.setattr(search, "chunk_results", lambda *a: [{"song_id": "chunk"}])
    monkeypatch.setattr(search, "window_results", lambda *a: [{"song_id": "window"}])
    no_index = search.search_audio(FakeEncoder(), FakeConnection(False), path, 5)
    with_index = search.search_audio(FakeEncoder(), FakeConnection(True), path, 5)
    forced = search.search_audio(FakeEncoder(), FakeConnection(True), path, 5, mode="chunks")
    assert (no_index["mode"], no_index["results"][0]["song_id"]) == ("chunks", "chunk")
    assert (with_index["mode"], with_index["results"][0]["song_id"]) == ("windows", "window")
    assert forced["mode"] == "chunks"


def test_search_rejects_unknown_mode():
    response = TestClient(api.app).post(
        "/search", params={"mode": "nope"}, files={"audio": ("q.wav", b"1")}
    )
    assert response.status_code == 422
