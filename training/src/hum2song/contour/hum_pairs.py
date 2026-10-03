"""D-039: hummed or sung queries against a song's vocal melody, for CLEWS training.

Not a cover-song pair. The query is a contour-level stand-in for a person humming or
singing the song's extracted vocal melody (Google's pitch-to-hum idea, in the F0 space
this encoder actually sees). The references are windows of that same vocal melody, which
is what the live index stores. Other songs in the batch are the negatives (CLEWS).

A hum query is `humanize` (glides and vibrato) then the usual contour augment. A sung
query skips `humanize` and only applies that augment (pitch drift, tempo, contour noise).
Whistles are not built here. Global key is already removed by median centering, so
"off-pitch" is drift, interval error, and jitter, not a transposition.
"""

import numpy as np
from torch.utils.data import Dataset

from hum2song.contour.augment import ContourAugment, augment_contour, humanize
from hum2song.contour.data import (
    EpochCounter,
    features_tensor,
    seconds_to_frames,
    stretch_nearest,
    voiced_start,
)
from hum2song.contour.features import FRAME_S, trim_unvoiced

HUM_STYLE_PROB = 0.75
REF_STRETCH = (0.85, 1.15)
QUERY_TRIES = 8
MIN_QUERY_S = 1.0


def style_name(rng: np.random.Generator, hum_prob: float) -> str:
    """`hum` with probability `hum_prob`, otherwise `sung`."""
    return "hum" if float(rng.random()) < hum_prob else "sung"


def hum_query(contour: np.ndarray, rng: np.random.Generator, augment: ContourAugment) -> np.ndarray:
    """Synthetic hum: glides and vibrato, then off-pitch, off-tempo, and contour noise."""
    return augment_contour(humanize(contour, rng, FRAME_S), rng, augment, FRAME_S)


def sung_query(
    contour: np.ndarray, rng: np.random.Generator, augment: ContourAugment
) -> np.ndarray:
    """Sung-style query: the vocal contour with the same augment, no hum resynthesis."""
    return augment_contour(contour, rng, augment, FRAME_S)


def styled_query(
    contour: np.ndarray, rng: np.random.Generator, augment: ContourAugment, style: str
) -> np.ndarray:
    queries = {"hum": hum_query, "sung": sung_query}
    return queries[style](contour, rng, augment)


def voiced_crop(
    contour: np.ndarray, rng: np.random.Generator, query_s: tuple[float, float]
) -> tuple[np.ndarray, int, int]:
    """(crop, start, length). The last try keeps the raw slice if trimming would empty it."""
    start, length, raw = 0, 1, contour[:1]
    for _attempt in range(QUERY_TRIES):
        length = seconds_to_frames(float(rng.uniform(*query_s)))
        start = voiced_start(contour, length, rng)
        raw = contour[start : start + length]
        trimmed = trim_unvoiced(raw)
        if len(trimmed) >= seconds_to_frames(MIN_QUERY_S):
            return trimmed, start, length
    return raw, start, length


def vocal_window(
    ref: np.ndarray,
    start: int,
    length: int,
    rng: np.random.Generator,
    offset_s: float,
    extra_s: tuple[float, float],
) -> np.ndarray:
    """A vocal-melody window near `start`. Tempo stays close to the recording."""
    offset = int(round(float(rng.uniform(-offset_s, offset_s)) / FRAME_S))
    begin = int(np.clip(start + offset, 0, max(len(ref) - 1, 0)))
    extra = int(round(float(rng.uniform(*extra_s)) / FRAME_S))
    end = begin + max(length + extra, seconds_to_frames(MIN_QUERY_S))
    window = stretch_nearest(ref[begin:end], float(rng.uniform(*REF_STRETCH)))
    trimmed = trim_unvoiced(window)
    if len(trimmed) >= seconds_to_frames(MIN_QUERY_S):
        return trimmed
    return window


class HumSongDataset(Dataset):
    """One augmented hum or sung query and several vocal-melody windows per song."""

    def __init__(
        self,
        routes: dict[str, dict],
        augment: ContourAugment,
        seed: int,
        refs: int = 4,
        offset_s: float = 3.0,
        query_s: tuple[float, float] = (3.0, 12.0),
        extra_s: tuple[float, float] = (-1.0, 3.0),
        hum_prob: float = HUM_STYLE_PROB,
    ) -> None:
        self.ids = sorted(routes)
        self.routes = routes
        self.augment = augment
        self.seed = seed
        self.refs = refs
        self.offset_s = offset_s
        self.query_s = query_s
        self.extra_s = extra_s
        self.hum_prob = hum_prob
        self.epoch = EpochCounter()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 5])
        ref = self.routes[self.ids[index]]["ref"]
        crop, start, length = voiced_crop(ref, rng, self.query_s)
        query = styled_query(crop, rng, self.augment, style_name(rng, self.hum_prob))
        refs = [
            vocal_window(ref, start, length, rng, self.offset_s, self.extra_s)
            for _ in range(self.refs)
        ]
        return {
            "query": features_tensor(query),
            "refs": [features_tensor(window) for window in refs],
            "song_id": self.ids[index],
        }
