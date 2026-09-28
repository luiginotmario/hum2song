"""Real queries against a song library held in memory, with exact cosine search (D-018).

Scores match the API: a song's score is its best chunk's cosine similarity. A query's rank
is 1 + the number of non-target songs scoring above its best target version; queries whose
target has no version in the library get rank None and count as misses.
"""

import numpy as np
import torch

from hum2song.catalog.library import song_chunks

TRACK_METHODS = ("vocals", "mix", "vocals_or_mix")
QUERY_BATCH = 256
NEGATIVE_QUANTILE = 0.99


def pick_track(tracks: dict, method: str) -> tuple[str, list]:
    """(method used, chunks); vocals_or_mix falls back to the mix when vocals give no chunk."""
    if method != "vocals_or_mix":
        return method, song_chunks(tracks[method])
    vocal_chunks = song_chunks(tracks["vocals"])
    if vocal_chunks:
        return "vocals", vocal_chunks
    return "mix", song_chunks(tracks["mix"])


def song_scores(queries: torch.Tensor, chunks: torch.Tensor, owner: torch.Tensor, songs: int):
    """(Q, songs) best-chunk cosine similarity; embeddings are already L2-normalized."""
    rows = []
    for first in range(0, len(queries), QUERY_BATCH):
        similarity = queries[first : first + QUERY_BATCH] @ chunks.T
        best = torch.full((len(similarity), songs), -2.0, device=similarity.device)
        index = owner.expand(len(similarity), -1)
        rows.append(best.scatter_reduce(1, index, similarity, reduce="amax").cpu())
    return torch.cat(rows).numpy()


def target_ranks(scores: np.ndarray, song_targets: list, query_targets: list) -> list:
    """1-based rank of each query's best target version, None when none is in the library."""
    song_targets = np.array([t or "" for t in song_targets], dtype=object)
    ranks = []
    for row, target in zip(scores, query_targets, strict=True):
        is_target = song_targets == target
        if not is_target.any():
            ranks.append(None)
            continue
        best = row[is_target].max()
        ranks.append(int((row[~is_target] > best).sum()) + 1)
    return ranks


def positive_negative(scores: np.ndarray, song_targets: list, query_targets: list):
    """Per query: best score of its target versions, and of every other target song."""
    song_targets = np.array([t or "" for t in song_targets], dtype=object)
    positives, negatives = [], []
    for row, target in zip(scores, query_targets, strict=True):
        is_target = song_targets == target
        other = (song_targets != "") & ~is_target
        positives.append(row[is_target].max() if is_target.any() else np.nan)
        negatives.extend(row[other].tolist())
    return np.array(positives), np.array(negatives)


def covered_targets(positives: np.ndarray, negatives: np.ndarray, query_targets: list) -> dict:
    """Targets whose median query similarity beats the 99th percentile of wrong-song scores.

    A model-based estimate of 'the hummed part is inside the stored audio'.
    """
    threshold = float(np.quantile(negatives, NEGATIVE_QUANTILE)) if len(negatives) else np.inf
    by_target: dict[str, list[float]] = {}
    for score, target in zip(positives, query_targets, strict=True):
        if not np.isnan(score):
            by_target.setdefault(target, []).append(float(score))
    covered = {t: bool(np.median(s) > threshold) for t, s in by_target.items()}
    return {"threshold": threshold, "covered": covered}
