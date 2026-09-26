"""Resample, trim, and length fitting."""

import numpy as np
import pytest

from hum2song.audio import fit_length, peak_normalize, resample_audio, trim_silence


def test_resample_changes_length() -> None:
    source = np.ones(8000, dtype=np.float32)
    resampled = resample_audio(source, 8000, 24000)
    assert len(resampled) == 24000


def test_trim_drops_edges_and_keeps_silence_when_everything_is_quiet() -> None:
    tone = np.concatenate([np.zeros(2000), np.ones(2000), np.zeros(2000)]).astype(np.float32)
    trimmed = trim_silence(tone, 24000, threshold=0.5, frame_ms=10)
    assert len(trimmed) < len(tone)
    assert float(np.max(trimmed)) == 1.0
    leading_zeros = int(np.argmax(trimmed != 0))
    assert leading_zeros < 2000
    quiet = np.zeros(1000, dtype=np.float32)
    assert len(trim_silence(quiet, 24000)) == len(quiet)


def test_peak_normalize_and_fit_length() -> None:
    samples = np.array([0.0, 0.5, -0.25], dtype=np.float32)
    peaked = peak_normalize(samples)
    assert float(np.max(np.abs(peaked))) == pytest.approx(0.95)
    rng = np.random.default_rng(0)
    fitted = fit_length(samples, 8, rng, random_start=False)
    assert len(fitted) == 8
    cropped = fit_length(np.arange(20, dtype=np.float32), 5, rng, random_start=False)
    assert np.array_equal(cropped, np.arange(5, dtype=np.float32))
