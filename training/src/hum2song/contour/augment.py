"""Contour augmentations that imitate how people hum a melody they remember.

Operates on semitone contours (NaN = unvoiced) at features.FRAME_S. Key needs no
augmentation because model input is median-normalized; these cover what remains:
tempo (global and local), interval size, pitch drift, and voicing gaps.
"""

from dataclasses import dataclass

import numpy as np

WARP_KNOTS = 4
DRIFT_KNOTS = 3


@dataclass(frozen=True)
class ContourAugment:
    """Ranges for each augmentation. A zero probability or unit range disables it."""

    stretch_min: float = 0.6
    stretch_max: float = 1.7
    warp_depth: float = 0.2
    interval_scale: float = 0.15
    drift_semitones: float = 0.7
    jitter_semitones: float = 0.15
    dropout_prob: float = 0.5
    dropout_max_s: float = 0.4
    octave_error_prob: float = 0.1


def augment_contour(
    contour: np.ndarray, rng: np.random.Generator, spec: ContourAugment, frame_s: float
) -> np.ndarray:
    """Apply every augmentation once, in a fixed order."""
    stretched = time_warp(contour, draw_stretch(rng, spec), draw_warp(rng, spec))
    scaled = scale_intervals(stretched, float(rng.uniform(-1, 1)) * spec.interval_scale + 1.0)
    drifted = scaled + slow_curve(len(scaled), rng, spec.drift_semitones)
    jittered = drifted + rng.normal(0.0, spec.jitter_semitones, size=len(drifted))
    folded = octave_error(jittered, rng, spec.octave_error_prob, frame_s)
    return voicing_gaps(folded, rng, spec.dropout_prob, spec.dropout_max_s, frame_s).astype(
        np.float32
    )


def draw_stretch(rng: np.random.Generator, spec: ContourAugment) -> float:
    """Duration factor, log-uniform in [stretch_min, stretch_max]."""
    return float(np.exp(rng.uniform(np.log(spec.stretch_min), np.log(spec.stretch_max))))


def draw_warp(rng: np.random.Generator, spec: ContourAugment) -> np.ndarray:
    """Relative local speeds at WARP_KNOTS points, mean 1."""
    speeds = 1.0 + rng.uniform(-spec.warp_depth, spec.warp_depth, size=WARP_KNOTS)
    return speeds / speeds.mean()


def time_warp(contour: np.ndarray, stretch: float, speeds: np.ndarray) -> np.ndarray:
    """Resample with nearest-frame lookup (keeps NaN gaps) along a smooth monotone time map."""
    length = max(int(round(len(contour) * stretch)), 1)
    knots = np.linspace(0.0, 1.0, len(speeds))
    local_speed = np.interp(np.linspace(0.0, 1.0, length), knots, speeds)
    source = np.cumsum(1.0 / local_speed)
    source = (source - source[0]) / max(source[-1] - source[0], 1.0e-9) * (len(contour) - 1)
    return contour[np.clip(np.round(source).astype(int), 0, len(contour) - 1)]


def scale_intervals(contour: np.ndarray, factor: float) -> np.ndarray:
    """Stretch every interval around the median by `factor` (people compress big leaps)."""
    voiced = ~np.isnan(contour)
    if not voiced.any():
        return contour
    median = float(np.median(contour[voiced]))
    return median + (contour - median) * factor


def slow_curve(length: int, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """Smooth random offset through DRIFT_KNOTS points, for singers going off key."""
    knots = rng.uniform(-amplitude, amplitude, size=DRIFT_KNOTS)
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, DRIFT_KNOTS), knots)


def octave_error(
    contour: np.ndarray, rng: np.random.Generator, probability: float, frame_s: float
) -> np.ndarray:
    """Occasionally move one short stretch by an octave, like an F0 tracker error."""
    if float(rng.random()) >= probability or len(contour) < 2:
        return contour
    width = max(int(rng.uniform(0.1, 0.5) / frame_s), 1)
    start = int(rng.integers(0, max(len(contour) - width, 1)))
    shifted = contour.copy()
    shifted[start : start + width] += 12.0 * float(rng.choice([-1.0, 1.0]))
    return shifted


def voicing_gaps(
    contour: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    max_s: float,
    frame_s: float,
) -> np.ndarray:
    """Blank up to three short stretches (breaths, unvoiced consonants, tracker dropouts)."""
    if float(rng.random()) >= probability:
        return contour
    gapped = contour.copy()
    for _index in range(int(rng.integers(1, 4))):
        width = max(int(rng.uniform(0.05, max_s) / frame_s), 1)
        start = int(rng.integers(0, max(len(gapped) - width, 1)))
        gapped[start : start + width] = np.nan
    return gapped
