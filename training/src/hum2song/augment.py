"""Waveform augmentations for the query tower.

Pitch and tempo are independent: transposition keeps the duration, and time-stretch
keeps the pitch. Both run through one resample plus one phase-vocoder pass.
Codec round-trip runs only when ffmpeg exists.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from hum2song.audio import load_audio, peak_normalize, resample_to_length, write_wav

AUG_PROBABILITY = 0.5
OCTAVE_SEMITONES = 12.0
PV_FFT = 1024
PV_HOP = 256
PV_OVERLAP = PV_FFT // PV_HOP
PV_WINDOW = np.hanning(PV_FFT + 1)[:-1]
PV_WINDOW_GAIN = float(np.sum(PV_WINDOW**2) / PV_HOP)
PV_BIN_ADVANCE = 2.0 * np.pi * PV_HOP * np.arange(PV_FFT // 2 + 1) / PV_FFT
UNIT_TOLERANCE = 1.0e-3
FFMPEG_THREADS = "1"


def augment_wave(
    samples: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    config,
) -> np.ndarray:
    """Apply the enabled augmentations. Pitch and tempo fire with their configured
    probabilities; the others fire with probability 0.5."""
    if not config.augment:
        return samples.astype(np.float32, copy=False)
    current = samples.astype(np.float32, copy=False)
    current = _pitch_and_tempo(current, rng, config)
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


def transpose_and_stretch(samples: np.ndarray, semitones: float, stretch: float) -> np.ndarray:
    """Shift pitch by `semitones` and scale duration by `stretch`, independently.

    Resampling moves the pitch (and the duration by the same factor). One phase-vocoder
    pass then sets the final duration to len(samples) * stretch without touching pitch.
    """
    pitch_factor = 2.0 ** (semitones / OCTAVE_SEMITONES)
    shifted_length = max(int(round(len(samples) / pitch_factor)), 1)
    shifted = resample_to_length(samples, shifted_length)
    target_length = max(int(round(len(samples) * stretch)), 1)
    return peak_normalize(phase_vocoder_stretch(shifted, target_length))


def phase_vocoder_stretch(samples: np.ndarray, target_length: int) -> np.ndarray:
    """Change duration to target_length samples while keeping pitch (STFT phase vocoder)."""
    if len(samples) == target_length:
        return samples.astype(np.float32, copy=False)
    if len(samples) < PV_FFT:
        return resample_to_length(samples, target_length)
    spectrum = _stft(samples)
    speed = len(samples) / float(target_length)
    positions = np.arange(0.0, spectrum.shape[0] - 1, speed)
    stretched = _interpolate_frames(spectrum, positions)
    return _istft(stretched, target_length)


def ffmpeg_command(ffmpeg: str, source: Path, dest: Path, *codec_args: str) -> list[str]:
    """One single-threaded ffmpeg call, so dataloader workers do not oversubscribe cores."""
    return [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-threads",
        FFMPEG_THREADS,
        "-y",
        "-i",
        str(source),
        "-threads",
        FFMPEG_THREADS,
        *codec_args,
        str(dest),
    ]


def draw_semitones(rng: np.random.Generator, config) -> float:
    """Uniform in +-pitch_semitones, or a whole-octave jump with octave_jump_prob."""
    if float(rng.random()) < config.octave_jump_prob:
        return float(rng.choice([-OCTAVE_SEMITONES, OCTAVE_SEMITONES]))
    return float(rng.uniform(-config.pitch_semitones, config.pitch_semitones))


def draw_stretch(rng: np.random.Generator, config) -> float:
    """Log-uniform duration factor in [time_stretch_min, time_stretch_max]."""
    low = np.log(config.time_stretch_min)
    high = np.log(config.time_stretch_max)
    return float(np.exp(rng.uniform(low, high)))


def _pitch_and_tempo(samples: np.ndarray, rng: np.random.Generator, config) -> np.ndarray:
    semitones = _maybe_draw(rng, config.aug_pitch_prob, draw_semitones, 0.0, config)
    stretch = _maybe_draw(rng, config.aug_time_prob, draw_stretch, 1.0, config)
    if abs(semitones) < UNIT_TOLERANCE and abs(stretch - 1.0) < UNIT_TOLERANCE:
        return samples
    return transpose_and_stretch(samples, semitones, stretch)


def _maybe_draw(rng: np.random.Generator, probability: float, draw, neutral: float, config):
    if float(rng.random()) >= probability:
        return neutral
    return draw(rng, config)


def _maybe(rng: np.random.Generator, enabled: bool, fn, *args):
    if not enabled or float(rng.random()) >= AUG_PROBABILITY:
        return args[0]
    return fn(*args)


def _stft(samples: np.ndarray) -> np.ndarray:
    padded = np.pad(samples.astype(np.float64), PV_FFT // 2)
    frame_count = 1 + (len(padded) - PV_FFT) // PV_HOP
    frames = np.lib.stride_tricks.sliding_window_view(padded, PV_FFT)[::PV_HOP][:frame_count]
    return np.fft.rfft(frames * PV_WINDOW, axis=1)


def _interpolate_frames(spectrum: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Magnitudes interpolate between frames; phases advance by each bin's measured rate."""
    base = positions.astype(np.int64)
    fraction = (positions - base)[:, None]
    left = spectrum[base]
    right = spectrum[base + 1]
    magnitude = (1.0 - fraction) * np.abs(left) + fraction * np.abs(right)
    deviation = np.angle(right) - np.angle(left) - PV_BIN_ADVANCE
    deviation -= 2.0 * np.pi * np.round(deviation / (2.0 * np.pi))
    advance = PV_BIN_ADVANCE + deviation
    phase = np.angle(spectrum[0]) + np.concatenate(
        [np.zeros((1, spectrum.shape[1])), np.cumsum(advance[:-1], axis=0)]
    )
    return magnitude * np.exp(1j * phase)


def _istft(spectrum: np.ndarray, target_length: int) -> np.ndarray:
    frames = np.fft.irfft(spectrum, n=PV_FFT, axis=1) * PV_WINDOW
    frame_count = frames.shape[0]
    output = np.zeros((frame_count + PV_OVERLAP - 1) * PV_HOP)
    for part in range(PV_OVERLAP):
        block = frames[:, part * PV_HOP : (part + 1) * PV_HOP].reshape(-1)
        output[part * PV_HOP : part * PV_HOP + len(block)] += block
    trimmed = output[PV_FFT // 2 : PV_FFT // 2 + target_length] / PV_WINDOW_GAIN
    missing = target_length - len(trimmed)
    return np.pad(trimmed, (0, max(missing, 0))).astype(np.float32)


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
        encode = ffmpeg_command(ffmpeg, source, encoded, "-c:a", "libopus", "-b:a", "32k")
        if not _run(encode):
            return samples
        if not _run(ffmpeg_command(ffmpeg, encoded, decoded)):
            return samples
        return load_audio(decoded, sample_rate, trim=False)


def _run(command: list[str]) -> bool:
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except OSError:
        return False
    return result.returncode == 0
