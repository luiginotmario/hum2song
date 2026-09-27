"""Retrieval with references extracted from song audio vs annotated melody (D-014).

MIR-1K has 1,000 clips from 110 songs. Two targets: `clip` (the query's own clip among
all 1,000, chance top-1 = 0.001) and `song` (any clip of the query's song counts, a song
scores its best clip; the query-by-humming question, since songs repeat melodies across
clips). References are whole clips, unvoiced ends trimmed. Reference sets differ only in
where the melody comes from: the manual annotation (the MIDI-like ceiling) or one
extraction method run on the mixture. Two query types, both cropped to a seeded
QUERY_FRACTION of the voiced span:
  sung       RMVPE on the isolated singing channel, transposed and time-stretched. Same
             performance as the reference, so the absolute level is optimistic; the drop
             between reference sets is the measurement.
  synthetic  the annotation, humanized and augmented like training queries: a different
             "performance" of the same melody that never touches the audio.
"""

import numpy as np

from hum2song.contour.augment import ContourAugment, augment_contour, humanize
from hum2song.contour.features import FRAME_S, rmvpe_contour, trim_unvoiced
from hum2song.eval.metrics import ranks_for_queries, retrieval_scores

QUERY_FRACTION = (0.5, 0.8)
MIN_QUERY_FRAMES = 25
REPORTED = ("top1", "top10", "mrr")


def crop(contour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random QUERY_FRACTION-long slice of the voiced span (whole span when short)."""
    span = trim_unvoiced(contour)
    length = max(int(len(span) * rng.uniform(*QUERY_FRACTION)), MIN_QUERY_FRAMES)
    start = int(rng.integers(0, max(len(span) - length, 0) + 1))
    return trim_unvoiced(span[start : start + length])


def synthetic_hum(melody: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return augment_contour(humanize(melody, rng, FRAME_S), rng, ContourAugment(), FRAME_S)


def sung_queries(tracks: dict, names: list[str], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [crop(rmvpe_contour(tracks[f"query_sung/{name}"]), rng) for name in names]


def synthetic_queries(melodies: list[np.ndarray], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [crop(synthetic_hum(melody, rng), rng) for melody in melodies]


def extracted_references(tracks: dict, method: str, names: list[str]) -> list[np.ndarray]:
    return [trim_unvoiced(rmvpe_contour(tracks[f"{method}/{name}"])) for name in names]


def retrieval(query_emb: np.ndarray, ref_emb: np.ndarray, ids: list[str]) -> dict[str, float]:
    """Row i of both embedding matrices carries ids[i]; a song id scores its best clip."""
    values = retrieval_scores(ranks_for_queries(query_emb, ids, ref_emb, ids)).as_dict()
    return {metric: float(values[metric]) for metric in REPORTED}
