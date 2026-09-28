import numpy as np

from hum2song.catalog.matching import (
    NO_MATCH,
    dtw_mae,
    fuse,
    key_normalized,
    query_windows,
    sequence_path,
    windows,
)
from hum2song.catalog.real_pool import charts_pool


def melody(seconds: float, offset: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds / 0.02)) * 0.02
    return (60 + offset + 5 * np.sin(t)).astype(np.float32)


def test_sequence_path_follows_diagonal_and_reports_span():
    similarity = np.full((3, 6), 0.0, dtype=np.float32)
    similarity[0, 2], similarity[1, 3], similarity[2, 4] = 1.0, 1.0, 1.0
    score, first, last = sequence_path(similarity)
    assert score == 1.0 and (first, last) == (2, 4)


def test_sequence_path_respects_max_step():
    similarity = np.full((2, 6), NO_MATCH, dtype=np.float32)
    similarity[0, 0], similarity[1, 5] = 1.0, 1.0
    score, _first, _last = sequence_path(similarity, max_step=2)
    assert score < 1.0


def test_dtw_mae_is_key_and_tempo_invariant():
    reference = key_normalized(melody(10.0))
    transposed_slower = key_normalized(
        np.interp(np.linspace(0, 499, 700), np.arange(500), melody(10.0, 7))
    )
    other = key_normalized(melody(10.0)[::-1].copy())
    assert dtw_mae(reference, transposed_slower) < 0.2
    assert dtw_mae(reference, other) > dtw_mae(reference, transposed_slower)
    assert dtw_mae(reference, reference[:3]) == float("inf")


def test_folded_dtw_forgives_octave_errors():
    reference = key_normalized(melody(6.0))
    octave = reference.copy()
    octave[20:40] += 12.0
    assert dtw_mae(reference, octave, fold_octaves=True) < dtw_mae(reference, octave)


def test_key_normalized_drops_unvoiced_and_centres():
    contour = np.array([np.nan] * 5 + [62.0] * 10, dtype=np.float32)
    assert np.allclose(key_normalized(contour), 0.0) and len(key_normalized(contour)) == 2


def test_windows_grid_and_short_query():
    grid = windows(melody(12.0), 5.0, 1.0, 0.25)
    assert sorted(grid) == list(range(8))
    assert len(query_windows(melody(3.0), 5.0, 1.0, 0.25)) == 1


def test_fuse_zscores_each_signal():
    fused = fuse([np.array([1.0, 2.0, 3.0]), np.array([30.0, 20.0, 10.0])], [1.0, 1.0])
    assert np.allclose(fused, 0.0)
    assert np.argmax(fuse([np.array([1.0, 2.0, -np.inf])], [1.0])) == 1


def test_charts_pool_keeps_split_targets_previews_and_fma():
    target = {"role": "target", "source": "youtube_full", "target_id": "chad:a"}
    assert charts_pool(target, "chad", {"chad:a"})
    assert not charts_pool(target, "chad", {"chad:b"})
    assert not charts_pool(target, "mlend", None)
    assert charts_pool({"role": "distractor", "source": "deezer_preview"}, "chad", None)
    assert not charts_pool({"role": "distractor", "source": "youtube_full"}, "chad", None)
    assert charts_pool({"role": "fma"}, "chad", None)
