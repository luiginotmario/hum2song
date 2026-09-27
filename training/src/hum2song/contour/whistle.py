"""Whistle F0: STFT spectral peak with a tonal-ratio voicing gate (docs/DECISIONS.md D-007).

Same rule as the paper pipeline (docs/paper/scripts/contour_pipeline.py): the F0 is the
argmax of the magnitude spectrum in 200-6000 Hz with parabolic interpolation, and a frame
is voiced when the energy within +-50 cents of the peak is at least 6 dB above the rest of
the band. Output uses the extract_f0 track layout, (2, frames) at a 10 ms hop, with the
voicing flag (0 or 1) in the confidence row.
"""

import numpy as np

SAMPLE_RATE = 32000
HOP_S = 0.01
N_FFT = 8192
WINDOW = 2048
BAND_HZ = (200.0, 6000.0)
TONAL_DB = 6.0
CENTS_HALF_WIDTH = 0.5
WHISTLE_F0_MAX_HZ = 6000.0


def frame_spectra(samples: np.ndarray) -> np.ndarray:
    """(frames, bins) magnitudes: 64 ms Hann window zero-padded to N_FFT, centered frames."""
    hop = int(round(SAMPLE_RATE * HOP_S))
    padded = np.pad(samples.astype(np.float64), N_FFT // 2, mode="reflect")
    count = 1 + (len(padded) - N_FFT) // hop
    window = np.zeros(N_FFT)
    offset = (N_FFT - WINDOW) // 2
    window[offset : offset + WINDOW] = np.hanning(WINDOW + 1)[:-1]
    starts = np.arange(count) * hop
    frames = np.stack([padded[start : start + N_FFT] for start in starts]) * window
    return np.abs(np.fft.rfft(frames, axis=1))


def parabolic_offset(level_db: np.ndarray, peak: np.ndarray) -> np.ndarray:
    """Sub-bin peak offset from a parabola through the three dB values around each peak."""
    rows = np.arange(len(peak))
    left, center, right = level_db[rows, peak - 1], level_db[rows, peak], level_db[rows, peak + 1]
    denominator = left - 2.0 * center + right
    safe = np.where(np.abs(denominator) > 1.0e-9, denominator, 1.0)
    offset = np.where(np.abs(denominator) > 1.0e-9, 0.5 * (left - right) / safe, 0.0)
    return np.clip(offset, -0.5, 0.5)


def spectral_peak_track(samples: np.ndarray) -> np.ndarray:
    """Mono audio at SAMPLE_RATE -> (2, frames): peak frequency in Hz and 0/1 voicing."""
    spectra = frame_spectra(samples)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    low, high = np.searchsorted(freqs, BAND_HZ[0]), np.searchsorted(freqs, BAND_HZ[1])
    band = spectra[:, low:high] + 1.0e-10
    peak = np.clip(np.argmax(band, axis=1), 1, band.shape[1] - 2)
    f0 = freqs[low] + (peak + parabolic_offset(20.0 * np.log10(band), peak)) * (freqs[1] - freqs[0])
    power = band**2
    near = np.abs(12.0 * np.log2(freqs[low:high][None, :] / f0[:, None])) <= CENTS_HALF_WIDTH
    tonal = (power * near).sum(axis=1)
    rest = np.maximum(power.sum(axis=1) - tonal, 1.0e-20)
    voiced = 10.0 * np.log10(tonal / rest) >= TONAL_DB
    return np.stack([f0, voiced.astype(np.float64)]).astype(np.float32)
