"""Human-error edits applied to a melody contour before it is rendered.

The continuous part reuses contour.augment (tempo warp, interval scaling, drift,
jitter, short octave slips, voicing gaps) and humanize (glides, vibrato). Two
extra edits match the query types people actually produce: one note a step or
two off, and one note left out. Key is not shifted here. Median centering
removes a global transposition (SIGNALS S5).
"""

import numpy as np

from hum2song.contour.augment import ContourAugment, augment_contour, humanize
from hum2song.contour.features import FRAME_S

WRONG_INTERVAL_PROB = 0.5
WRONG_INTERVAL_ST = (1.0, 2.0)
DROP_NOTE_PROB = 0.4
MIN_RUN = 3


def note_runs(contour: np.ndarray) -> list[tuple[int, int]]:
    """Nearly constant voiced stretches. A jump above half a semitone starts a new note."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(contour):
        if np.isnan(value):
            start = _close_run(runs, start, index)
            continue
        if start is None:
            start = index
            continue
        if abs(float(value) - float(contour[index - 1])) > 0.5:
            start = _close_run(runs, start, index)
            start = index
    _close_run(runs, start, len(contour))
    return [run for run in runs if run[1] - run[0] >= MIN_RUN]


def _close_run(runs: list[tuple[int, int]], start: int | None, end: int) -> None:
    if start is not None:
        runs.append((start, end))
    return None


def shift_one_interval(
    contour: np.ndarray,
    rng: np.random.Generator,
    probability: float = WRONG_INTERVAL_PROB,
) -> np.ndarray:
    """Move one note by ±1 or ±2 semitones. Needs two notes, so a single tone is unchanged."""
    runs = note_runs(contour)
    if float(rng.random()) >= probability or len(runs) < 2:
        return contour
    start, end = runs[int(rng.integers(0, len(runs)))]
    step = float(rng.choice([-2.0, -1.0, 1.0, 2.0]))
    shifted = contour.copy()
    shifted[start:end] = shifted[start:end] + step
    return shifted


def drop_one_note(
    contour: np.ndarray,
    rng: np.random.Generator,
    probability: float = DROP_NOTE_PROB,
) -> np.ndarray:
    """Blank one note. A contour with a single note is left alone."""
    runs = note_runs(contour)
    if float(rng.random()) >= probability or len(runs) < 2:
        return contour
    start, end = runs[int(rng.integers(0, len(runs)))]
    dropped = contour.copy()
    dropped[start:end] = np.nan
    return dropped


def apply_human_errors(
    contour: np.ndarray,
    rng: np.random.Generator,
    spec: ContourAugment,
    interval_prob: float = WRONG_INTERVAL_PROB,
    drop_prob: float = DROP_NOTE_PROB,
) -> np.ndarray:
    """Wrong interval, dropped note, then the shared contour augment and glides."""
    noted = drop_one_note(shift_one_interval(contour, rng, interval_prob), rng, drop_prob)
    warped = augment_contour(noted, rng, spec, FRAME_S)
    return humanize(warped, rng, FRAME_S)
