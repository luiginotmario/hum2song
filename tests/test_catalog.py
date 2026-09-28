import numpy as np
import pandas as pd
import pytest
import soundfile
from fastapi.testclient import TestClient

from hum2song.catalog.db import best_per_song, vector_text
from hum2song.catalog.fma import member_name, select_tracks, song_id
from hum2song.catalog.library import (
    CHUNK_S,
    HOP_S,
    chunk_contour,
    decode_audio,
    segment_weight,
    segment_windows,
    to_track_rate,
)
from hum2song.catalog.probe import hum_crop, rank_of, render_hum, summarize_ranks
from hum2song.catalog.search import search_audio
from hum2song.contour.features import FRAME_S
from hum2song.server import api

FRAMES_PER_S = int(round(1.0 / FRAME_S))


def tracks() -> pd.DataFrame:
    genres = ["Rock", "Pop", "Electronic", "Folk", "Hip-Hop"] * 40
    durations = [30.0, 120.0, 200.0, 500.0, 240.0] * 40
    return pd.DataFrame(
        {"genre": genres, "duration_s": durations, "title": "t", "artist": "a", "license": "l"},
        index=np.arange(1000, 1200),
    )


def voiced_contour(seconds: float) -> np.ndarray:
    return np.full(int(seconds * FRAMES_PER_S), 60.0, dtype=np.float32)


def test_select_tracks_is_deterministic_and_filters():
    first = select_tracks(tracks(), 30, seed=1)
    second = select_tracks(tracks(), 30, seed=1)
    assert list(first.index) == list(second.index)
    assert len(first) == 30
    assert set(first["genre"]) <= {"Rock", "Pop", "Folk", "Hip-Hop"}
    assert first["duration_s"].between(60.0, 420.0).all()
    assert list(first.index) == sorted(first.index)


def test_select_tracks_seed_changes_selection():
    assert list(select_tracks(tracks(), 30, seed=1).index) != list(
        select_tracks(tracks(), 30, seed=2).index
    )


def test_member_name_and_song_id():
    assert member_name(155066) == "fma_full/155/155066.mp3"
    assert member_name(2) == "fma_full/000/000002.mp3"
    assert song_id(2) == "fma:000002"


def test_chunk_contour_windows_and_hop():
    chunks = chunk_contour(voiced_contour(30.0))
    assert [chunk.start_s for chunk in chunks] == [0.0, 5.0, 10.0, 15.0, 20.0]
    assert all(len(chunk.contour) == int(CHUNK_S * FRAMES_PER_S) for chunk in chunks)
    assert chunks[1].start_s - chunks[0].start_s == HOP_S


def test_chunk_contour_skips_unvoiced_windows():
    contour = voiced_contour(30.0)
    contour[: 15 * FRAMES_PER_S] = np.nan
    starts = [chunk.start_s for chunk in chunk_contour(contour)]
    assert starts == [10.0, 15.0, 20.0]
    assert all(chunk.voiced >= 0.25 for chunk in chunk_contour(contour))


def test_chunk_contour_trims_unvoiced_edges():
    contour = voiced_contour(10.0)
    contour[:100] = np.nan
    (chunk,) = chunk_contour(contour)
    assert len(chunk.contour) == 400
    assert chunk.voiced == 0.8


def test_chunk_contour_short_song_gives_one_chunk():
    chunks = chunk_contour(voiced_contour(4.0))
    assert len(chunks) == 1
    assert len(chunks[0].contour) == 4 * FRAMES_PER_S


def test_chunk_contour_empty_song():
    assert chunk_contour(np.full(1000, np.nan, dtype=np.float32)) == []


def test_best_per_song_keeps_best_chunk_and_orders():
    rows = [("a", 0.0, 0.5), ("b", 5.0, 0.7), ("a", 10.0, 0.9), ("c", 0.0, 0.1)]
    assert best_per_song(rows, 2) == [("a", 10.0, 0.9), ("b", 5.0, 0.7)]


def test_vector_text():
    assert vector_text(np.array([0.5, -1.0], dtype=np.float32)) == "[0.500000,-1.000000]"


def test_search_rejects_large_upload(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    response = TestClient(api.app).post("/search", files={"audio": ("q.wav", b"12345")})
    assert response.status_code == 413


def test_render_hum_has_the_contour_pitch():
    rate = 16000
    contour = np.full(50, 69.0, dtype=np.float32)
    audio = render_hum(contour, rate)
    assert len(audio) == 50 * int(FRAME_S * rate)
    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * rate / len(audio)
    assert abs(peak_hz - 440.0) < 2.0


def test_render_hum_is_silent_when_unvoiced():
    audio = render_hum(np.full(50, np.nan, dtype=np.float32), 16000)
    assert np.abs(audio).max() < 0.1


def test_rank_of_and_summarize_ranks():
    results = [{"song_id": "a"}, {"song_id": "b"}]
    assert rank_of("b", results) == 2
    assert rank_of("z", results) is None
    summary = summarize_ranks([1, 2, None, 11])
    assert summary["top1"] == 0.25
    assert summary["top5"] == 0.5
    assert summary["mrr"] == pytest.approx((1.0 + 0.5 + 1.0 / 11) / 4)


def test_decode_audio_reads_mono_at_rate(tmp_path):
    path = tmp_path / "tone.wav"
    stereo = np.stack([np.sin(np.arange(22050) / 10.0)] * 2, axis=1).astype(np.float32)
    soundfile.write(path, stereo, 22050)
    audio = decode_audio(path, 44100)
    assert audio.dtype == np.float32
    assert abs(len(audio) - 44100) < 200


def test_to_track_rate_length():
    assert len(to_track_rate(np.zeros(44100, dtype=np.float32), 44100)) == 16000


def test_hum_crop_length_and_voicing():
    contour = voiced_contour(60.0)
    contour[: 30 * FRAMES_PER_S] = np.nan
    crop = hum_crop(contour, np.random.default_rng(0))
    assert 8 * FRAMES_PER_S - 1 <= len(crop) <= 12 * FRAMES_PER_S
    assert not np.isnan(crop[0])


def test_search_audio_rejects_short_query(tmp_path):
    path = tmp_path / "short.wav"
    soundfile.write(path, np.zeros(1600, dtype=np.float32), 16000)
    assert search_audio(None, None, path, 5)["results"] == []


def test_segment_windows_cover_the_song_and_center_the_tail():
    windows = segment_windows(250, 100, 0.25)
    assert [offset for offset, _kept, _left in windows] == [0, 75, 150, 225]
    assert windows[0] == (0, 100, 0)
    assert windows[-1] == (225, 25, 37)


def test_segment_weight_is_a_unit_triangle():
    weight = segment_weight(10, "cpu")
    assert len(weight) == 10
    assert float(weight.max()) == 1.0
    assert float(weight[0]) == pytest.approx(0.2)
