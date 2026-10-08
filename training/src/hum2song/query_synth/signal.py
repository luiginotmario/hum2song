"""Sample-rate helpers shared by the whistle and singing renders.

Physics of the two sources is docs/paper/SIGNALS.md S2–S5. This file does not restate it.
"""

import numpy as np

from hum2song.audio import peak_normalize

ROOM_DELAY_S = (0.012, 0.035)
ROOM_GAIN = (0.04, 0.18)
ROOM_SNR_DB = (18.0, 36.0)
HUNT_HZ = 40.0


def semitone_to_hz(midi: np.ndarray) -> np.ndarray:
    """MIDI note numbers to Hz. NaN stays out of this call."""
    return 440.0 * np.power(2.0, (np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def frame_grid(n_frames: int, sample_rate: int, frame_s: float) -> tuple[int, int]:
    """(n_samples, samples_per_frame) with one constant frame length."""
    samples_per_frame = max(int(round(frame_s * sample_rate)), 1)
    return n_frames * samples_per_frame, samples_per_frame


def f0_hz_from_contour(contour: np.ndarray, sample_rate: int, frame_s: float) -> np.ndarray:
    """Per-sample Hz. Unvoiced frames are 0. Gaps are not interpolated across."""
    n_samples, samples_per_frame = frame_grid(max(len(contour), 1), sample_rate, frame_s)
    if len(contour) == 0:
        return np.zeros(n_samples, dtype=np.float64)
    index = np.minimum(np.arange(n_samples) // samples_per_frame, len(contour) - 1)
    midi = contour[index]
    hz = np.zeros(n_samples, dtype=np.float64)
    voiced = ~np.isnan(midi)
    hz[voiced] = semitone_to_hz(midi[voiced])
    return hz


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    """Moving average. Width 1 returns a float copy."""
    width = max(int(width), 1)
    if width <= 1:
        return values.astype(np.float64, copy=False)
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def add_room(
    audio: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int,
    snr_db: tuple[float, float] = ROOM_SNR_DB,
) -> np.ndarray:
    """One short reflection plus noise, then peak-normalize. Silence stays silence."""
    delay = int(rng.uniform(*ROOM_DELAY_S) * sample_rate)
    gain = float(rng.uniform(*ROOM_GAIN))
    wet = np.array(audio, dtype=np.float64, copy=True)
    if delay > 0 and delay < len(wet):
        wet[delay:] += gain * audio[: len(audio) - delay]
    noise = rng.normal(0.0, 1.0, size=len(wet))
    colored = np.diff(noise, append=noise[-1] if len(noise) else 0.0)
    colored /= np.std(colored) + 1.0e-8
    signal = float(np.sqrt(np.mean(wet**2))) + 1.0e-8
    level = signal / (10.0 ** (float(rng.uniform(*snr_db)) / 20.0))
    mixed = wet + colored * level
    mixed -= float(np.mean(mixed))
    return peak_normalize(mixed.astype(np.float32))


def finish(audio: np.ndarray, rng: np.random.Generator, sample_rate: int, room: bool) -> np.ndarray:
    """DC-block, then either the room/mic path or a plain peak normalize."""
    centered = np.asarray(audio, dtype=np.float64)
    if len(centered) == 0:
        return np.zeros(1, dtype=np.float32)
    centered = centered - float(np.mean(centered))
    if room:
        return add_room(centered, rng, sample_rate)
    return peak_normalize(centered.astype(np.float32))


def component_ratio_db(audio: np.ndarray, sample_rate: int, f0_hz: float, harmonic: int) -> float:
    """Level of harmonic*f0 relative to f0, in dB, on a 1 s slice from the middle."""
    start = max(len(audio) // 4, 0)
    frame = np.asarray(audio[start : start + sample_rate], dtype=np.float64)
    if len(frame) < 256:
        frame = np.asarray(audio, dtype=np.float64)
    window = frame * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), 1.0 / float(sample_rate))
    fundamental = _peak_mag(spectrum, freqs, f0_hz)
    overtone = _peak_mag(spectrum, freqs, harmonic * f0_hz)
    return float(20.0 * np.log10(overtone / fundamental))


def _peak_mag(spectrum: np.ndarray, freqs: np.ndarray, hz: float) -> float:
    band = np.flatnonzero(np.abs(freqs - hz) <= HUNT_HZ)
    if len(band) == 0:
        return 1.0e-12
    return float(max(spectrum[band].max(), 1.0e-12))


def median_abs_semitones(left: np.ndarray, right: np.ndarray) -> float | None:
    """Median |left-right| on frames both call voiced. None when fewer than 5 agree."""
    n = min(len(left), len(right))
    if n == 0:
        return None
    both = ~np.isnan(left[:n]) & ~np.isnan(right[:n])
    if int(both.sum()) < 5:
        return None
    return float(np.median(np.abs(left[:n][both] - right[:n][both])))
