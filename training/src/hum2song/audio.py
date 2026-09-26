"""Load, resample, trim, and length-fit mono audio."""

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 24000
PEAK_TARGET = 0.95
TRIM_THRESHOLD = 0.01
TRIM_FRAME_MS = 20.0


def resample_to_length(samples: np.ndarray, new_length: int) -> np.ndarray:
    """Linearly resample a 1-D waveform to new_length."""
    if new_length <= 0:
        raise ValueError("new_length must be positive")
    if len(samples) == new_length:
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return np.zeros(new_length, dtype=np.float32)
    if len(samples) == 1:
        return np.full(new_length, float(samples[0]), dtype=np.float32)
    positions = np.linspace(0.0, len(samples) - 1, num=new_length)
    resampled = np.interp(positions, np.arange(len(samples)), samples)
    return resampled.astype(np.float32)


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio from orig_sr to target_sr."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sr == target_sr:
        return samples.astype(np.float32, copy=False)
    duration_s = len(samples) / float(orig_sr)
    new_length = max(int(round(duration_s * target_sr)), 1)
    return resample_to_length(samples, new_length)


def peak_normalize(samples: np.ndarray, peak: float = PEAK_TARGET) -> np.ndarray:
    """Scale audio so its peak amplitude is `peak`. Silence stays silence."""
    current = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if current <= 0.0:
        return samples.astype(np.float32, copy=False)
    scaled = samples * (peak / current)
    return scaled.astype(np.float32)


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float = TRIM_THRESHOLD,
    frame_ms: float = TRIM_FRAME_MS,
) -> np.ndarray:
    """Drop leading and trailing frames below `threshold`. Keep all-quiet audio intact."""
    if len(samples) == 0:
        return samples.astype(np.float32, copy=False)
    frame = max(int(sample_rate * frame_ms / 1000.0), 1)
    frame_count = int(np.ceil(len(samples) / frame))
    keep = np.zeros(frame_count, dtype=bool)
    for index in range(frame_count):
        start = index * frame
        chunk = samples[start : start + frame]
        keep[index] = float(np.max(np.abs(chunk))) >= threshold
    if not bool(np.any(keep)):
        return samples.astype(np.float32, copy=False)
    first = int(np.argmax(keep))
    last = int(len(keep) - 1 - np.argmax(keep[::-1]))
    return samples[first * frame : min((last + 1) * frame, len(samples))].astype(np.float32)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels down to one mono waveform."""
    if samples.ndim == 1:
        return samples.astype(np.float32)
    return samples.mean(axis=1).astype(np.float32)


def load_audio(
    path: str | Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    trim: bool = True,
) -> np.ndarray:
    """Load a file as mono float32 at `sample_rate`, peak-normalized."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    mono = to_mono(np.asarray(data))
    resampled = resample_audio(mono, int(file_sr), sample_rate)
    if trim:
        resampled = trim_silence(resampled, sample_rate)
    return peak_normalize(resampled)


def audio_duration_s(path: str | Path) -> float:
    """Return duration in seconds without reading every sample."""
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def fit_length(
    samples: np.ndarray,
    target_length: int,
    rng: np.random.Generator,
    random_start: bool,
) -> np.ndarray:
    """Crop or loop `samples` so the result has target_length samples."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(samples) == target_length:
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return np.zeros(target_length, dtype=np.float32)
    if len(samples) > target_length:
        start = 0
        if random_start:
            start = int(rng.integers(0, len(samples) - target_length + 1))
        return samples[start : start + target_length].astype(np.float32, copy=False)
    repeats = int(np.ceil(target_length / len(samples)))
    tiled = np.tile(samples, repeats)
    return tiled[:target_length].astype(np.float32)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float audio as 16-bit PCM wav."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), samples.astype(np.float32), sample_rate, subtype="PCM_16")
