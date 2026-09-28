"""Sequence-level matching on top of chunk embeddings (literature review E0/E1, D-022, D-023).

windows          fixed short windows on a regular grid (index = start / hop), for the
                 query and for reference songs
sequence_path    best monotone path through a query-window x reference-window similarity
                 matrix: each next query window (one hop later) moves the reference by
                 0..MAX_STEP hops, i.e. a local tempo ratio of about 0.5..2
key_normalized   voiced frames only, minus their median (key invariance for DTW)
dtw_mae          slope-constrained DTW (steps (1,1), (1,2), (2,1)) after resampling the
                 query to the reference length; mean absolute semitone error on the path
fuse             per-query z-scored sum of several candidate scores
"""

import numpy as np
from numba import njit

from hum2song.catalog.library import chunk_contour
from hum2song.contour.features import FRAME_S

MAX_STEP = 2
NO_MATCH = -2.0
DTW_DECIMATE = 5
MIN_DTW_FRAMES = 10


def windows(contour: np.ndarray, win_s: float, hop_s: float, min_voiced: float) -> dict:
    """{grid index: window contour} for windows with enough voiced frames."""
    return {
        int(round(chunk.start_s / hop_s)): chunk.contour
        for chunk in chunk_contour(contour, win_s, hop_s, min_voiced)
    }


def query_windows(contour: np.ndarray, win_s: float, hop_s: float, min_voiced: float) -> list:
    """Query sub-windows in time order; a query shorter than one window is used whole."""
    found = windows(contour, win_s, hop_s, min_voiced)
    return [found[k] for k in sorted(found)] or [contour]


def sequence_path(similarity: np.ndarray, max_step: int = MAX_STEP) -> tuple[float, int, int]:
    """Best monotone path (rows: query windows one hop apart; cols: reference windows on the
    grid, NO_MATCH where missing): (mean similarity, first column, last column). Each step
    moves 0..max_step columns, a local tempo ratio of about 0..2."""
    best = similarity[0].copy()
    start = np.arange(similarity.shape[1])
    for row in similarity[1:]:
        shifted = np.stack([shift_right(best, step) for step in range(max_step + 1)])
        starts = np.stack([shift_right(start, step, fill=0) for step in range(max_step + 1)])
        choice = shifted.argmax(axis=0)
        columns = np.arange(len(row))
        best = shifted[choice, columns] + row
        start = starts[choice, columns]
    end = int(best.argmax())
    return float(best[end] / len(similarity)), int(start[end]), end


def sequence_score(similarity: np.ndarray, max_step: int = MAX_STEP) -> float:
    return sequence_path(similarity, max_step)[0]


def shift_right(values: np.ndarray, step: int, fill=-np.inf) -> np.ndarray:
    if step == 0:
        return values
    return np.concatenate([np.full(step, fill, dtype=values.dtype), values[:-step]])


def key_normalized(contour: np.ndarray, decimate: int = DTW_DECIMATE) -> np.ndarray:
    voiced = contour[~np.isnan(contour)]
    usable = len(voiced) // decimate * decimate
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    reduced = voiced[:usable].reshape(-1, decimate).mean(axis=1)
    return (reduced - np.median(reduced)).astype(np.float64)


def resample(values: np.ndarray, length: int) -> np.ndarray:
    return np.interp(np.linspace(0, len(values) - 1, length), np.arange(len(values)), values)


def dtw_mae(ref: np.ndarray, query: np.ndarray, fold_octaves: bool = False) -> float:
    """Key-normalized contours -> path-mean |semitone difference| (inf when too short)."""
    if min(len(ref), len(query)) < MIN_DTW_FRAMES:
        return float("inf")
    diff = ref[:, None] - resample(query, len(ref))[None, :]
    cost = np.abs(diff)
    if fold_octaves:
        cost = np.minimum(cost, np.abs(np.abs(diff) - 12.0))
    return float(accumulated_mean(np.ascontiguousarray(cost)))


@njit(cache=True)
def accumulated_mean(cost: np.ndarray) -> float:
    """Slope-constrained DTW over a square cost matrix: weighted path cost / path weight."""
    size = cost.shape[0]
    total = np.full((size + 2, size + 2), np.inf)
    weight = np.zeros((size + 2, size + 2))
    total[2, 2], weight[2, 2] = cost[0, 0], 1.0
    for i in range(2, size + 2):
        for j in range(2, size + 2):
            if i == 2 and j == 2:
                continue
            here = cost[i - 2, j - 2]
            best, best_weight = np.inf, 0.0
            for di, dj, step in ((1, 1, 1.0), (1, 2, 1.5), (2, 1, 1.5)):
                candidate = total[i - di, j - dj] + step * here
                if candidate < best:
                    best, best_weight = candidate, weight[i - di, j - dj] + step
            total[i, j], weight[i, j] = best, best_weight
    end = total[size + 1, size + 1]
    return end / weight[size + 1, size + 1] if np.isfinite(end) else np.inf


def zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.where(finite, 0.0, -1.0e3)
    mean, std = values[finite].mean(), values[finite].std() + 1.0e-9
    return np.where(finite, (values - mean) / std, -1.0e3)


def fuse(scores: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted sum of per-query z-scored candidate scores (higher is better)."""
    return sum(w * zscore(s) for s, w in zip(scores, weights, strict=True) if w)


def frames(seconds: float) -> int:
    return int(round(seconds / FRAME_S))
