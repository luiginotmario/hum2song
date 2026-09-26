"""Augmentations can be turned off, and noise changes the waveform."""

import numpy as np

from hum2song.augment import add_noise, augment_wave
from hum2song.config import TrainConfig


def test_disabled_augment_is_identity() -> None:
    samples = np.linspace(-0.2, 0.2, 400, dtype=np.float32)
    config = TrainConfig(augment=False)
    output = augment_wave(samples, 24000, np.random.default_rng(0), config)
    assert np.array_equal(output, samples)


def test_noise_changes_the_signal() -> None:
    samples = np.ones(800, dtype=np.float32) * 0.2
    noisy = add_noise(samples, snr_db=10.0, rng=np.random.default_rng(1))
    assert not np.allclose(noisy, samples)
