"""Re-track rendered queries so the encoder sees a tracker, not the control curve.

Whistles use the spectral peak (D-007). Singing uses a cepstrum by default, or
RMVPE when the generate script is given a loaded model. RMVPE is what real sung
queries use (extract_f0.py).
"""

import numpy as np

from hum2song.contour.features import F0_MAX_HZ, rmvpe_contour
from hum2song.contour.whistle import SAMPLE_RATE, WHISTLE_F0_MAX_HZ, spectral_peak_track

CEPSTRUM_S = 0.04
FMIN_HZ = 70.0
FMAX_HZ = 900.0
CEPSTRUM_MARGIN = 0.08


def track_whistle(audio: np.ndarray) -> np.ndarray:
    """32 kHz whistle audio -> semitone contour at FRAME_S."""
    return rmvpe_contour(spectral_peak_track(audio), f0_max_hz=WHISTLE_F0_MAX_HZ)


def track_singing_cepstrum(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Harmonic audio -> semitone contour. Numpy only, no checkpoint."""
    return rmvpe_contour(_cepstrum(audio, sample_rate), f0_max_hz=F0_MAX_HZ)


def track_singing_rmvpe(audio: np.ndarray, model, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """32 kHz singing -> RMVPE contour. `model` is contour.trackers.load_rmvpe."""
    from hum2song.audio import resample_audio
    from hum2song.contour.trackers import RMVPE_SAMPLE_RATE, rmvpe_track

    audio_16 = resample_audio(np.asarray(audio, dtype=np.float32), sample_rate, RMVPE_SAMPLE_RATE)
    return rmvpe_contour(rmvpe_track(model, audio_16), f0_max_hz=F0_MAX_HZ)


def _cepstrum(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """(2, frames) at 10 ms: cepstral F0 in Hz and a 0/1 peak gate."""
    hop = int(round(sample_rate * 0.01))
    win = int(round(sample_rate * CEPSTRUM_S))
    if len(audio) < win:
        audio = np.pad(audio.astype(np.float64), (0, win - len(audio)))
    padded = np.pad(audio.astype(np.float64), (win // 2, win // 2))
    count = 1 + max(len(padded) - win, 0) // hop
    if count < 1:
        return np.zeros((2, 1), dtype=np.float32)
    window = np.hanning(win)
    frames = np.stack([padded[i * hop : i * hop + win] * window for i in range(count)])
    cepstrum = np.fft.irfft(np.log(np.abs(np.fft.rfft(frames, axis=1)) + 1.0e-8), axis=1)
    qmin = max(int(sample_rate / FMAX_HZ), 1)
    qmax = min(int(sample_rate / FMIN_HZ), cepstrum.shape[1] - 2)
    band = cepstrum[:, qmin:qmax]
    local = np.argmax(band, axis=1)
    peak = local + qmin
    refined = np.array(
        [_parabola(cepstrum[i], int(peak[i])) for i in range(len(peak))], dtype=np.float64
    )
    f0 = sample_rate / np.maximum(refined, 1.0)
    margin = band.max(axis=1) - np.median(band, axis=1)
    energy = frames.std(axis=1)
    voiced = (margin > CEPSTRUM_MARGIN) & (energy > 1.0e-4)
    return np.stack([f0, voiced.astype(np.float64)]).astype(np.float32)


def _parabola(values: np.ndarray, index: int) -> float:
    left, center, right = values[index - 1], values[index], values[index + 1]
    denom = left - 2.0 * center + right
    if abs(float(denom)) < 1.0e-12:
        return float(index)
    offset = float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
    return float(index) + offset


def best_median_abs(control: np.ndarray, tracked: np.ndarray, max_lag: int = 20) -> float | None:
    """Median semitone error over a small lag, ignoring the tracker's frame offset."""
    from hum2song.query_synth.signal import median_abs_semitones

    best: float | None = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            err = median_abs_semitones(control, tracked[lag:])
        else:
            err = median_abs_semitones(control[-lag:], tracked)
        if err is not None and (best is None or err < best):
            best = err
    return best
