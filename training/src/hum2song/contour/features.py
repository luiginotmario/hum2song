"""Pitch contours: RMVPE F0 or MIDI notes -> key-normalized frames.

A contour is a float32 array of MIDI semitones on a fixed FRAME_S grid, NaN where
nothing is voiced. Queries come from RMVPE (10 ms hop, decimated to FRAME_S); references
come straight from the melody MIDI. Model input subtracts the median voiced pitch of the
crop, so any transposition of the whole clip gives the same input.
"""

import numpy as np
from scipy.signal import medfilt

from hum2song.midi_render import MidiNote

RMVPE_HOP_S = 0.01
FRAME_S = 0.02
DECIMATE = int(round(FRAME_S / RMVPE_HOP_S))
CONFIDENCE_THRESHOLD = 0.3
F0_MIN_HZ = 50.0
F0_MAX_HZ = 2000.0
MIN_RUN_FRAMES = 3
MEDIAN_KERNEL = 3
OCTAVE_FOLD_WINDOW = 25
OCTAVE_FOLD_SEMITONES = 9.0
SEMITONE_SCALE = 12.0
FEATURE_DIM = 2
RMVPE_CENTS_BINS = 360
RMVPE_CENTS_OFFSET = 1997.3794084376191
RMVPE_CENTS_STEP = 20.0
RMVPE_LOCAL_BINS = 4


def hz_to_semitones(f0_hz: np.ndarray) -> np.ndarray:
    """MIDI note numbers (A4 = 69)."""
    return 69.0 + 12.0 * np.log2(np.maximum(f0_hz, 1.0e-6) / 440.0)


def salience_to_f0(salience: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RMVPE (frames, 360) salience -> F0 in Hz (weighted cents around the peak) and peak value."""
    cents = RMVPE_CENTS_OFFSET + RMVPE_CENTS_STEP * np.arange(RMVPE_CENTS_BINS)
    bins = np.argmax(salience, axis=1)[:, None] + np.arange(-RMVPE_LOCAL_BINS, RMVPE_LOCAL_BINS + 1)
    inside = (bins >= 0) & (bins < RMVPE_CENTS_BINS)
    bins = np.clip(bins, 0, RMVPE_CENTS_BINS - 1)
    weights = np.take_along_axis(salience, bins, axis=1) * inside
    local_cents = (weights * cents[bins]).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0e-9)
    f0 = 10.0 * 2.0 ** (local_cents / 1200.0)
    return f0.astype(np.float32), salience.max(axis=1).astype(np.float32)


def rmvpe_contour(track: np.ndarray) -> np.ndarray:
    """(2, frames) RMVPE F0 and confidence at 10 ms -> cleaned contour at FRAME_S."""
    f0, confidence = track[0].astype(np.float64), track[1].astype(np.float64)
    voiced = (confidence >= CONFIDENCE_THRESHOLD) & (f0 > F0_MIN_HZ) & (f0 < F0_MAX_HZ)
    semitones = np.where(voiced, hz_to_semitones(f0), np.nan)
    return clean_contour(decimate(semitones, DECIMATE))


def decimate(semitones: np.ndarray, factor: int) -> np.ndarray:
    """Mean of each block of `factor` frames over voiced frames; NaN when none is voiced."""
    count = len(semitones) // factor
    blocks = semitones[: count * factor].reshape(count, factor)
    voiced = ~np.isnan(blocks)
    sums = np.where(voiced, blocks, 0.0).sum(axis=1)
    counts = voiced.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def clean_contour(semitones: np.ndarray) -> np.ndarray:
    """Drop voiced runs shorter than MIN_RUN_FRAMES, fold octave jumps, median-filter."""
    keep = drop_short_runs(~np.isnan(semitones), MIN_RUN_FRAMES)
    cleaned = np.where(keep, semitones, np.nan)
    index = np.flatnonzero(keep)
    if len(index) >= OCTAVE_FOLD_WINDOW:
        cleaned[index] = fold_octave_jumps(cleaned[index])
    if len(index) >= MEDIAN_KERNEL:
        cleaned[index] = medfilt(cleaned[index], MEDIAN_KERNEL)
    return cleaned.astype(np.float32)


def drop_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only True runs of at least min_run frames."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[0::2], edges[1::2]
    keep = np.zeros(len(mask), dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        if end - start >= min_run:
            keep[start:end] = True
    return keep


def fold_octave_jumps(voiced: np.ndarray) -> np.ndarray:
    """Move frames far from a running median by whole octaves toward it."""
    running = medfilt(voiced, OCTAVE_FOLD_WINDOW)
    distance = running - voiced
    octaves = np.round(distance / 12.0)
    return voiced + 12.0 * octaves * (np.abs(distance) > OCTAVE_FOLD_SEMITONES)


def midi_contour(notes: list[MidiNote], end_s: float | None = None) -> np.ndarray:
    """Highest sounding note per FRAME_S frame, NaN in rests. Time 0 is the file start."""
    if not notes:
        return np.full(1, np.nan, dtype=np.float32)
    last = max(note.end_s for note in notes) if end_s is None else end_s
    count = max(int(np.ceil(last / FRAME_S)), 1)
    contour = np.full(count, -np.inf)
    centers = (np.arange(count) + 0.5) * FRAME_S
    for note in notes:
        inside = (centers >= note.start_s) & (centers < note.end_s)
        contour[inside] = np.maximum(contour[inside], note.pitch)
    return np.where(np.isinf(contour), np.nan, contour).astype(np.float32)


def trim_unvoiced(contour: np.ndarray) -> np.ndarray:
    """Drop leading and trailing unvoiced frames."""
    index = np.flatnonzero(~np.isnan(contour))
    if len(index) == 0:
        return contour[:1]
    return contour[index[0] : index[-1] + 1]


def voiced_fraction(contour: np.ndarray) -> float:
    return float(np.mean(~np.isnan(contour))) if len(contour) else 0.0


def contour_features(contour: np.ndarray) -> np.ndarray:
    """(frames, 2): pitch minus the voiced median in octaves (0 when unvoiced), voicing flag."""
    voiced = ~np.isnan(contour)
    features = np.zeros((len(contour), FEATURE_DIM), dtype=np.float32)
    if not voiced.any():
        return features
    median = float(np.median(contour[voiced]))
    features[voiced, 0] = (contour[voiced] - median) / SEMITONE_SCALE
    features[:, 1] = voiced
    return features
