"""Octave-error correction of song-side melody contours (D-024).

E0 (D-022) found that most large hum-vs-song pitch errors disappear when octaves are folded:
the song's extracted melody jumps by 12 semitones for a note or a phrase. Correction moves
a voiced frame by whole octaves towards a running median of the voiced frames around it
only when it lies more than `limit` semitones from that median, so a short octave jump is
pulled back while ordinary melodic intervals (within about a sixth of the median) are kept.
A first version folded every frame into ±6 semitones; that destroyed real leaps (D-024).

octave_correct   contour (semitones, NaN unvoiced) -> contour with octave jumps removed
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from hum2song.contour.features import FRAME_S

DEFAULT_WINDOW_S = 3.0
DEFAULT_LIMIT = 9.0
PASSES = 2


def running_median(values: np.ndarray, size: int) -> np.ndarray:
    """Centred median over `size` neighbours (edges padded by reflection)."""
    size = max(1, min(size, len(values)) // 2 * 2 + 1)
    half = size // 2
    padded = np.pad(values, half, mode="reflect") if half else values
    return np.median(sliding_window_view(padded, size), axis=1)


def fold_towards(values: np.ndarray, reference: np.ndarray, limit: float) -> np.ndarray:
    """Shift values further than `limit` from the reference by whole octaves towards it."""
    offset = values - reference
    shift = np.where(np.abs(offset) > limit, 12.0 * np.round(offset / 12.0), 0.0)
    return values - shift


def octave_correct(
    contour: np.ndarray, window_s: float = DEFAULT_WINDOW_S, limit: float = DEFAULT_LIMIT
) -> np.ndarray:
    """Remove octave jumps; the window counts voiced frames only, so gaps do not dilute it."""
    voiced = ~np.isnan(contour)
    if voiced.sum() < 3:
        return contour
    values = contour[voiced].astype(np.float64)
    size = int(round(window_s / FRAME_S))
    for _ in range(PASSES):
        values = fold_towards(values, running_median(values, size), limit)
    out = contour.astype(np.float64).copy()
    out[voiced] = values
    return out.astype(contour.dtype)
