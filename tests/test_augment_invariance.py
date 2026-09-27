"""Transposition keeps duration, time-stretch keeps pitch, ffmpeg stays single-threaded."""

from pathlib import Path

import numpy as np

from hum2song.augment import (
    draw_semitones,
    draw_stretch,
    ffmpeg_command,
    ffmpeg_env,
    frame_positions,
    phase_vocoder_stretch,
    transpose_and_stretch,
)
from hum2song.config import TrainConfig

SAMPLE_RATE = 24000


def _tone(frequency: float, seconds: float = 2.0) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (0.5 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _peak_frequency(wave: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
    return float(np.argmax(spectrum)) * SAMPLE_RATE / len(wave)


def test_transpose_moves_pitch_and_keeps_duration() -> None:
    for semitones in (7.0, -12.0, 12.0):
        shifted = transpose_and_stretch(_tone(220.0), semitones, 1.0)
        expected = 220.0 * 2.0 ** (semitones / 12.0)
        assert len(shifted) == 2 * SAMPLE_RATE
        assert abs(_peak_frequency(shifted) - expected) < 2.0


def test_stretch_changes_duration_and_keeps_pitch() -> None:
    for factor in (0.7, 1.4):
        stretched = transpose_and_stretch(_tone(220.0), 0.0, factor)
        assert len(stretched) == int(round(2 * SAMPLE_RATE * factor))
        assert abs(_peak_frequency(stretched) - 220.0) < 2.0


def test_stretched_tone_keeps_its_level() -> None:
    stretched = phase_vocoder_stretch(_tone(220.0), int(2.8 * SAMPLE_RATE))
    middle = stretched[SAMPLE_RATE // 2 : -SAMPLE_RATE // 2]
    assert float(np.sqrt(np.mean(middle**2))) > 0.25


def test_draws_cover_octaves_and_stay_in_range() -> None:
    config = TrainConfig(pitch_semitones=12.0, octave_jump_prob=0.5)
    config.time_stretch_min = 0.7
    config.time_stretch_max = 1.4
    rng = np.random.default_rng(0)
    semitones = [draw_semitones(rng, config) for _ in range(400)]
    stretches = [draw_stretch(rng, config) for _ in range(400)]
    assert max(abs(value) for value in semitones) <= 12.0
    assert {-12.0, 12.0} <= set(semitones)
    assert 0.7 <= min(stretches) and max(stretches) <= 1.4


def test_ffmpeg_runs_single_threaded() -> None:
    command = ffmpeg_command("ffmpeg", Path("in.wav"), Path("out.opus"), "-c:a", "libopus")
    assert command.count("-threads") == 2
    assert command[command.index("-threads") + 1] == "1"


def test_ffmpeg_env_caps_openmp_and_blas_threads() -> None:
    env = ffmpeg_env()
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert "PATH" in env


def test_frame_positions_never_reach_the_last_frame() -> None:
    overshooting = np.arange(0.0, 1133, 290048 / 225792)
    assert overshooting[-1] >= 1133
    assert float(frame_positions(1134, 290048 / 225792).max()) < 1133
    for frame_count in range(3, 1400, 37):
        for speed in np.linspace(0.3, 3.0, 97):
            positions = frame_positions(frame_count, float(speed))
            assert positions.size > 0
            assert float(positions.max()) < frame_count - 1


def test_stretch_handles_many_lengths() -> None:
    wave = np.random.default_rng(0).normal(size=5000).astype(np.float32)
    for target in range(2000, 16000, 997):
        assert len(phase_vocoder_stretch(wave, target)) == target
