"""Render one whistle or wrong-lyrics query from a semitone contour."""

from dataclasses import dataclass

import numpy as np

from hum2song.contour.augment import ContourAugment
from hum2song.contour.features import FRAME_S
from hum2song.contour.whistle import SAMPLE_RATE
from hum2song.query_synth.errors import apply_human_errors
from hum2song.query_synth.signal import f0_hz_from_contour, frame_grid, semitone_to_hz
from hum2song.query_synth.singing import singing_waveform, syllable_plan
from hum2song.query_synth.track import track_singing_cepstrum, track_whistle
from hum2song.query_synth.whistle_audio import draw_h2_db, whistle_waveform

STYLE_WHISTLE = "whistle"
STYLE_WRONG_LYRICS = "wrong_lyrics"
STYLES = (STYLE_WHISTLE, STYLE_WRONG_LYRICS)
WHISTLE_OCTAVES = (2, 3)
WHISTLE_HZ = (500.0, 4000.0)
ORIGIN_SYNTHETIC = "synthetic"
PRESENTATION = "synthetic; not a recording of a person"


@dataclass(frozen=True)
class RenderedQuery:
    """Audio plus the contour the tracker read. origin is always synthetic."""

    audio: np.ndarray
    sample_rate: int
    control: np.ndarray
    tracked: np.ndarray
    style: str
    syllables: tuple[str, ...]
    h2_db: float | None
    origin: str = ORIGIN_SYNTHETIC


def shift_to_whistle(contour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add two or three octaves, then step by octaves until the median sits in the whistle band.

    A global octave is removed again by median centering (SIGNALS S5). It is still
    rendered, because the tracker and the timbre live in that register (SIGNALS S4).
    """
    voiced = ~np.isnan(contour)
    if not bool(voiced.any()):
        return contour.astype(np.float32, copy=True)
    shifted = contour + 12.0 * int(rng.integers(WHISTLE_OCTAVES[0], WHISTLE_OCTAVES[1] + 1))
    for _step in range(6):
        median_hz = float(semitone_to_hz(np.median(shifted[voiced])))
        if WHISTLE_HZ[0] <= median_hz <= WHISTLE_HZ[1]:
            return shifted.astype(np.float32)
        shifted = shifted + (-12.0 if median_hz > WHISTLE_HZ[1] else 12.0)
    return shifted.astype(np.float32)


def render_query(
    contour: np.ndarray,
    rng: np.random.Generator,
    style: str,
    spec: ContourAugment,
    room: bool = True,
    interval_prob: float = 0.5,
    drop_prob: float = 0.4,
) -> RenderedQuery:
    """Human-error the contour, render audio, re-track it. style is whistle or wrong_lyrics."""
    if style not in STYLES:
        raise ValueError(f"unknown query style: {style}")
    erred = apply_human_errors(contour, rng, spec, interval_prob, drop_prob)
    control = shift_to_whistle(erred, rng) if style == STYLE_WHISTLE else erred
    return _render_control(control, rng, style, room)


def render_control(
    contour: np.ndarray,
    rng: np.random.Generator,
    style: str,
    room: bool = True,
) -> RenderedQuery:
    """Render a contour as-is (no extra human-error pass). Used by the timbre checks."""
    if style not in STYLES:
        raise ValueError(f"unknown query style: {style}")
    return _render_control(np.asarray(contour, dtype=np.float32), rng, style, room)


def _render_control(
    control: np.ndarray, rng: np.random.Generator, style: str, room: bool
) -> RenderedQuery:
    renders = {STYLE_WHISTLE: _whistle, STYLE_WRONG_LYRICS: _singing}
    audio, syllables, h2_db = renders[style](control, rng, room)
    tracked = track_whistle(audio) if style == STYLE_WHISTLE else track_singing_cepstrum(audio)
    return RenderedQuery(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        control=control.astype(np.float32),
        tracked=tracked.astype(np.float32),
        style=style,
        syllables=syllables,
        h2_db=h2_db,
    )


def _whistle(
    control: np.ndarray, rng: np.random.Generator, room: bool
) -> tuple[np.ndarray, tuple[str, ...], float]:
    h2_db = draw_h2_db(rng)
    f0 = f0_hz_from_contour(control, SAMPLE_RATE, FRAME_S)
    return whistle_waveform(f0, rng, h2_db=h2_db, room=room), (), h2_db


def _singing(
    control: np.ndarray, rng: np.random.Generator, room: bool
) -> tuple[np.ndarray, tuple[str, ...], None]:
    _n_samples, samples_per_frame = frame_grid(max(len(control), 1), SAMPLE_RATE, FRAME_S)
    names, formants, noise, scale = syllable_plan(max(len(control), 1), rng, FRAME_S)
    f0 = f0_hz_from_contour(control, SAMPLE_RATE, FRAME_S)
    audio = singing_waveform(f0, formants, noise, scale, samples_per_frame, rng, room=room)
    return audio, names, None
