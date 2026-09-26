"""Top-k and MRR, including the CHAD number the report has to print."""

import numpy as np
import pytest

from hum2song.eval.metrics import CHAD_TOP10, ranks_for_queries, retrieval_scores


def test_known_ranks() -> None:
    scores = retrieval_scores([0, 9, 10])
    assert scores.count == 3
    assert scores.top1 == pytest.approx(1.0 / 3.0)
    assert scores.top10 == pytest.approx(2.0 / 3.0)
    expected_mrr = (1.0 + 1.0 / 10.0 + 1.0 / 11.0) / 3.0
    assert scores.mrr == pytest.approx(expected_mrr)


def test_empty_ranks_have_no_rate() -> None:
    scores = retrieval_scores([])
    assert scores.count == 0
    assert scores.top10 is None


def test_best_clip_groups_by_song() -> None:
    query = np.array([[1.0, 0.0]], dtype=np.float64)
    refs = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.1]], dtype=np.float64)
    ranks = ranks_for_queries(query, ["song-a"], refs, ["song-b", "song-a", "song-a"])
    assert ranks == [0]


def test_chad_midi_baseline_is_the_published_top10() -> None:
    assert CHAD_TOP10["chad_top10_jang_midi"] == 0.921
