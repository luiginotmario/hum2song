"""Candidate generation and fused re-ranking for the real-hum evaluations (D-023, D-025).

Each query item holds, for a pool of candidate songs, several scores (higher is better):
  base    best 10 s chunk (the API score)
  window  order-free window vote: mean over query windows of the song's best 5 s window
  seq     time-consistent window path score (matching.sequence_path)
  dtw     minus key-normalized DTW error on the span the path picked
plus each candidate's rank in the whole library under `base` and `window`. A first stage
(`source`, `k`) picks the candidates; a z-scored weighted sum re-ranks them. A target that
the first stage misses keeps its first-stage rank.
"""

import itertools

import numpy as np

from hum2song.catalog.matching import fuse
from hum2song.catalog.probe import summarize_ranks

FEATURES = ("base", "window", "seq", "dtw")
SOURCES = ("base", "window", "union")
K_GRID = (50, 100, 200)
WEIGHT_STEPS = (0.0, 0.5, 1.0, 2.0)


def rank_all(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every entry (ties broken by position)."""
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return ranks


def target_rank(scores: np.ndarray, is_target: np.ndarray, keep: np.ndarray) -> int | None:
    """Rank of the best target version among the kept songs (None when absent)."""
    if not (is_target & keep).any():
        return None
    best = scores[is_target & keep].max()
    return int((scores[~is_target & keep] > best).sum()) + 1


def members(item: dict, source: str, k: int) -> np.ndarray:
    """Candidate mask of the first stage `source` keeping the top `k` songs."""
    in_base, in_window = item["base_rank"] <= k, item["window_rank"] <= k
    return {"base": in_base, "window": in_window, "union": in_base | in_window}[source]


def fused_rank(item: dict | None, weights: tuple, source: str, k: int) -> int | None:
    if item is None:
        return None
    mask = members(item, source, k)
    hit = item["hit"][mask]
    if not hit.any():
        return item["fallback"][source]
    fused = fuse([item[name][mask] for name in FEATURES], list(weights))
    return int((fused[~hit] > fused[hit].max()).sum()) + 1


def summarize(items: list, weights: tuple, source: str, k: int) -> dict:
    return summarize_ranks([fused_rank(item, weights, source, k) for item in items])


def recall_at(items: list, source: str, k: int) -> float:
    """Share of queries whose target is among the first stage's candidates."""
    found = [item is not None and item["hit"][members(item, source, k)].any() for item in items]
    return float(np.mean(found)) if found else 0.0


def weight_grid(allow_window: bool = True) -> list[tuple]:
    window_steps = WEIGHT_STEPS if allow_window else (0.0,)
    grid = itertools.product((0.0, 1.0), window_steps, WEIGHT_STEPS, WEIGHT_STEPS)
    return [w for w in grid if any(w)]


def choose(items: list, configs: list[tuple]) -> tuple:
    """Best (top-1, then top-10) (weights, source, k) on the selection items."""
    scored = [(summarize(items, *config), config) for config in configs]
    return max(scored, key=lambda sc: (sc[0]["top1"], sc[0]["top10"]))[1]
