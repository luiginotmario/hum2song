"""Library sanity probe: hum-like audio rendered from indexed songs' own melodies (D-017).

A rendered hum is a proxy, not a person humming. It checks that the whole search path
(audio -> RMVPE -> contour -> encoder -> pgvector -> best chunk per song) finds a song
from a humanized, transposed, cropped copy of its extracted melody.
"""

import numpy as np

from hum2song.contour.features import FRAME_S, trim_unvoiced, voiced_fraction
from hum2song.contour.melody_data import semitones_to_hz

HARMONIC_WEIGHTS = (1.0, 0.5, 0.25, 0.12)
NOISE_LEVEL = 0.01
AMPLITUDE = 0.3
FADE_S = 0.01
QUERY_S = (8.0, 12.0)
MIN_CROP_VOICED = 0.5
CROP_TRIES = 20


def hum_crop(contour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A QUERY_S-long window starting on a voiced frame, preferring mostly voiced windows."""
    length = int(rng.uniform(*QUERY_S) / FRAME_S)
    voiced_starts = np.flatnonzero(~np.isnan(contour[: max(len(contour) - length, 1)]))
    pool = voiced_starts if len(voiced_starts) else np.arange(max(len(contour) - length, 1))
    windows = [contour[start : start + length] for start in rng.choice(pool, CROP_TRIES)]
    voiced = [window for window in windows if voiced_fraction(window) >= MIN_CROP_VOICED]
    return trim_unvoiced((voiced or windows)[0])


def voicing_envelope(voiced: np.ndarray, samples_per_frame: int, rate: int) -> np.ndarray:
    """Per-sample 0/1 voicing with short linear fades so note edges do not click."""
    gate = np.repeat(voiced.astype(np.float64), samples_per_frame)
    width = max(int(FADE_S * rate), 1)
    return np.convolve(gate, np.ones(width) / width, mode="same")


def render_hum(contour: np.ndarray, rate: int, seed: int = 0) -> np.ndarray:
    """Contour in MIDI semitones at FRAME_S (NaN = silence) -> harmonic tone plus noise."""
    samples_per_frame = int(round(FRAME_S * rate))
    voiced = ~np.isnan(contour)
    hz = np.repeat(semitones_to_hz(np.nan_to_num(contour, nan=0.0)), samples_per_frame)
    phase = 2.0 * np.pi * np.cumsum(hz) / rate
    tone = sum(weight * np.sin((k + 1) * phase) for k, weight in enumerate(HARMONIC_WEIGHTS))
    envelope = voicing_envelope(voiced, samples_per_frame, rate)
    noise = np.random.default_rng(seed).normal(0.0, NOISE_LEVEL, len(tone))
    audio = AMPLITUDE * tone * envelope / sum(HARMONIC_WEIGHTS) + noise
    return audio.astype(np.float32)


def rank_of(song_id: str, results: list[dict]) -> int | None:
    """1-based rank of song_id in search results, None when it is not returned."""
    ids = [result["song_id"] for result in results]
    return ids.index(song_id) + 1 if song_id in ids else None


def summarize_ranks(ranks: list[int | None], cutoffs: tuple[int, ...] = (1, 5, 10)) -> dict:
    found = np.array([rank if rank is not None else np.inf for rank in ranks], dtype=float)
    summary = {f"top{k}": float(np.mean(found <= k)) for k in cutoffs}
    summary["mrr"] = float(np.mean(np.where(np.isfinite(found), 1.0 / found, 0.0)))
    summary["queries"] = len(ranks)
    return summary
