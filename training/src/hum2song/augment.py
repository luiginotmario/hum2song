"""Lightweight waveform augmentations. Codec round-trip runs only when ffmpeg exists."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from hum2song.audio import load_audio, resample_to_length, write_wav

AUG_PROBABILITY = 0.5


def augment_wave(
    samples: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    config,
) -> np.ndarray:
    """Apply the enabled augmentations. Each one fires with probability 0.5."""
    if not config.augment:
        return samples.astype(np.float32, copy=False)
    current = samples.astype(np.float32, copy=False)
    current = _maybe(rng, config.aug_pitch, _pitch, current, rng, config)
    current = _maybe(rng, config.aug_time, _time_stretch, current, rng, config)
    current = _maybe(rng, config.aug_noise, _noise, current, rng, config)
    current = _maybe(rng, config.aug_gain, _gain, current, rng, config)
    current = _maybe(rng, config.aug_eq, _eq, current, rng)
    current = _maybe(rng, config.aug_rir, _rir, current, sample_rate, rng)
    current = _maybe(rng, config.aug_codec, _codec, current, sample_rate)
    return current.astype(np.float32)


def add_noise(samples: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add gaussian noise at the requested SNR."""
    power = float(np.mean(samples.astype(np.float64) ** 2)) + 1.0e-8
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=samples.shape)
    return (samples + noise).astype(np.float32)


def _maybe(rng: np.random.Generator, enabled: bool, fn, *args):
    if not enabled or float(rng.random()) >= AUG_PROBABILITY:
        return args[0]
    return fn(*args)


def _pitch(samples: np.ndarray, rng: np.random.Generator, config) -> np.ndarray:
    semitones = float(rng.uniform(-config.pitch_semitones, config.pitch_semitones))
    factor = 2.0 ** (semitones / 12.0)
    new_length = max(int(round(len(samples) / factor)), 1)
    return resample_to_length(samples, new_length)


def _time_stretch(samples: np.ndarray, rng: np.random.Generator, config) -> np.ndarray:
    factor = float(rng.uniform(config.time_stretch_min, config.time_stretch_max))
    new_length = max(int(round(len(samples) * factor)), 1)
    return resample_to_length(samples, new_length)


def _noise(samples: np.ndarray, rng: np.random.Generator, config) -> np.ndarray:
    snr_db = float(rng.uniform(config.snr_min_db, config.snr_max_db))
    return add_noise(samples, snr_db, rng)


def _gain(samples: np.ndarray, rng: np.random.Generator, config) -> np.ndarray:
    gain_db = float(rng.uniform(-config.gain_db, config.gain_db))
    return (samples * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def _eq(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(samples)
    tilt = float(rng.uniform(-1.0, 1.0))
    freqs = np.linspace(0.0, 1.0, num=spectrum.shape[0])
    gain = 10.0 ** ((tilt * 6.0) * freqs / 20.0)
    return np.fft.irfft(spectrum * gain, n=len(samples)).astype(np.float32)


def _rir(samples: np.ndarray, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    rt60 = float(rng.uniform(0.1, 0.4))
    length = max(int(sample_rate * rt60), 1)
    decay = np.exp(-np.linspace(0.0, 6.0, length))
    impulse = rng.normal(0.0, 1.0, size=length) * decay
    impulse[0] = 1.0
    size = len(samples) + length
    wet = np.fft.irfft(np.fft.rfft(samples, n=size) * np.fft.rfft(impulse, n=size), n=size)
    return wet[: len(samples)].astype(np.float32)


def _codec(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return samples
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / "in.wav"
        encoded = folder / "clip.opus"
        decoded = folder / "out.wav"
        write_wav(source, samples, sample_rate)
        encoded_ok = _run(
            [ffmpeg, "-y", "-i", str(source), "-c:a", "libopus", "-b:a", "32k", str(encoded)]
        )
        if not encoded_ok:
            return samples
        decoded_ok = _run([ffmpeg, "-y", "-i", str(encoded), str(decoded)])
        if not decoded_ok:
            return samples
        return load_audio(decoded, sample_rate, trim=False)


def _run(command: list[str]) -> bool:
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except OSError:
        return False
    return result.returncode == 0
