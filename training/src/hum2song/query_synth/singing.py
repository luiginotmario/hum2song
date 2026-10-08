"""Nonsense-syllable singing locked to a given F0.

A harmonic source through a three-resonance filter is the source-filter voice
(SIGNALS S2). Syllables are drawn from a fixed inventory. No lyric string is
an input, so the words cannot be the song's words.

The resonance centers are fixed constants for this synthesizer. They are not
a new measurement of a singer.
"""

import numpy as np

from hum2song.contour.whistle import SAMPLE_RATE
from hum2song.query_synth.signal import finish

N_HARMONICS = 16
SYLLABLE_S = (0.18, 0.28)
CONSONANT_S = 0.04
FORMANT_BW = (90.0, 110.0, 170.0)
FORMANT_GAIN = (1.0, 0.55, 0.25)
VOWEL_HZ = {
    "a": (730.0, 1090.0, 2440.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
}
NASAL_HZ = (280.0, 1200.0, 2200.0)
LIQUID_HZ = (400.0, 1400.0, 2400.0)
SYLLABLES = (
    ("la", "l", "a"),
    ("na", "n", "a"),
    ("ma", "m", "a"),
    ("du", "d", "u"),
    ("lu", "l", "u"),
    ("ni", "n", "i"),
    ("me", "m", "e"),
    ("lo", "l", "o"),
)
SYLLABLE_NAMES = tuple(name for name, _cons, _vowel in SYLLABLES)
VOWEL_NOISE = 0.04
STOP_NOISE = 1.0


def syllable_plan(
    n_frames: int, rng: np.random.Generator, frame_s: float
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """(names, formant Hz (frames, 3), noise gain, harmonic scale)."""
    formants = np.zeros((n_frames, 3), dtype=np.float64)
    noise = np.zeros(n_frames, dtype=np.float64)
    scale = np.ones(n_frames, dtype=np.float64)
    names: list[str] = []
    frame = 0
    while frame < n_frames:
        name, consonant, vowel = SYLLABLES[int(rng.integers(0, len(SYLLABLES)))]
        length = max(int(rng.uniform(*SYLLABLE_S) / frame_s), 4)
        cons = max(int(CONSONANT_S / frame_s), 1)
        end = min(frame + length, n_frames)
        cons_end = min(frame + cons, end)
        names.append(name)
        _paint(formants, noise, scale, frame, cons_end, end, consonant, vowel)
        frame = end
    return tuple(names), formants, noise, scale


def harmonic_amplitudes(f0_frames: np.ndarray, formant_hz: np.ndarray) -> np.ndarray:
    """(frames, harmonics). Each voiced frame is peak-normalized. Unvoiced frames are 0."""
    k = np.arange(1, N_HARMONICS + 1, dtype=np.float64)
    freq = np.maximum(f0_frames[:, None], 0.0) * k[None, :]
    gain = np.zeros_like(freq)
    for center, bandwidth, weight in zip(formant_hz.T, FORMANT_BW, FORMANT_GAIN, strict=True):
        half = bandwidth / 2.0
        gain += weight / (1.0 + ((freq - center[:, None]) / half) ** 2)
    tilted = gain / k
    tilted[f0_frames < 1.0] = 0.0
    peak = np.maximum(tilted.max(axis=1, keepdims=True), 1.0e-8)
    return tilted / peak


def harmonic_bank(f0_hz: np.ndarray, amplitudes: np.ndarray, sample_rate: int) -> np.ndarray:
    """Sum of sinusoids. amplitudes is per sample, shape (samples, harmonics)."""
    phase = np.cumsum(2.0 * np.pi * f0_hz / float(sample_rate))
    k = np.arange(1, amplitudes.shape[1] + 1, dtype=np.float64)
    return (np.sin(phase[:, None] * k[None, :]) * amplitudes).sum(axis=1)


def singing_waveform(
    f0_hz: np.ndarray,
    formants: np.ndarray,
    noise_gain: np.ndarray,
    harm_scale: np.ndarray,
    samples_per_frame: int,
    rng: np.random.Generator,
    room: bool = True,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Harmonic source plus consonant noise, optionally with room and mic noise."""
    f0_frames = _frame_rate(f0_hz, samples_per_frame)
    amps = harmonic_amplitudes(f0_frames, formants) * harm_scale[:, None]
    per_sample = np.repeat(amps, samples_per_frame, axis=0)[: len(f0_hz)]
    tone = harmonic_bank(f0_hz, per_sample, sample_rate)
    noise = rng.normal(0.0, 1.0, size=len(f0_hz))
    burst = np.repeat(noise_gain, samples_per_frame)[: len(f0_hz)]
    audio = tone + noise * burst * 0.2
    return finish(audio, rng, sample_rate, room)


def _paint(
    formants: np.ndarray,
    noise: np.ndarray,
    scale: np.ndarray,
    frame: int,
    cons_end: int,
    end: int,
    consonant: str,
    vowel: str,
) -> None:
    formants[cons_end:end] = VOWEL_HZ[vowel]
    noise[cons_end:end] = VOWEL_NOISE
    if consonant == "d":
        scale[frame:cons_end] = 0.0
        noise[frame:cons_end] = STOP_NOISE
        return
    formants[frame:cons_end] = NASAL_HZ if consonant in ("n", "m") else LIQUID_HZ


def _frame_rate(f0_hz: np.ndarray, samples_per_frame: int) -> np.ndarray:
    count = int(np.ceil(len(f0_hz) / float(samples_per_frame)))
    index = np.minimum(np.arange(count) * samples_per_frame, len(f0_hz) - 1)
    return f0_hz[index]
