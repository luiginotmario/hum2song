"""Whistle audio from a per-sample F0: one partial, a weak second, breath, then room.

The second-harmonic draw is centered on the MLEnd median of −45 dB (SIGNALS S2).
Register (people whistle higher) is applied to the contour before this function.
"""

import numpy as np

from hum2song.contour.whistle import SAMPLE_RATE
from hum2song.query_synth.signal import finish, smooth

H2_DB = (-50.0, -40.0)
BREATH_DB = (30.0, 40.0)
ATTACK_S = 0.015
GAP_BREATH_DB = 50.0


def draw_h2_db(rng: np.random.Generator) -> float:
    """Second harmonic relative to the fundamental, dB. Uniform around the S2 median."""
    return float(rng.uniform(*H2_DB))


def whistle_waveform(
    f0_hz: np.ndarray,
    rng: np.random.Generator,
    h2_db: float | None = None,
    room: bool = True,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Near-sinusoid at f0_hz (0 = unvoiced), plus breath and an optional room."""
    level = h2_db if h2_db is not None else draw_h2_db(rng)
    phase = np.cumsum(2.0 * np.pi * f0_hz / float(sample_rate))
    partial = 10.0 ** (level / 20.0)
    tone = np.sin(phase) + partial * np.sin(2.0 * phase)
    voiced = (f0_hz > 1.0).astype(np.float64)
    envelope = smooth(voiced, int(ATTACK_S * sample_rate))
    tone = tone * envelope
    audio = tone + _breath(tone, envelope, rng)
    return finish(audio, rng, sample_rate, room)


def _breath(tone: np.ndarray, envelope: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """High-passed noise under the tone, and a quieter noise in the gaps."""
    white = rng.normal(0.0, 1.0, size=len(tone))
    colored = np.diff(white, append=white[-1] if len(white) else 0.0)
    colored /= np.std(colored) + 1.0e-8
    tone_rms = float(np.sqrt(np.mean(tone**2))) + 1.0e-8
    under = tone_rms * 10.0 ** (-float(rng.uniform(*BREATH_DB)) / 20.0)
    gap = tone_rms * 10.0 ** (-GAP_BREATH_DB / 20.0)
    return colored * (under * envelope + gap * (1.0 - envelope))
