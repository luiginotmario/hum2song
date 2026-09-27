import numpy as np

from hum2song.contour.augment import (
    ContourAugment,
    augment_contour,
    scale_intervals,
    time_warp,
    voicing_gaps,
)
from hum2song.contour.features import FRAME_S


def melody(length: int = 200) -> np.ndarray:
    contour = 60.0 + (np.arange(length) // 20 % 5).astype(np.float32)
    contour[50:60] = np.nan
    return contour


def test_time_warp_scales_length_and_keeps_values():
    contour = melody()
    warped = time_warp(contour, 1.5, np.array([0.8, 1.2, 1.0, 1.0]))
    assert len(warped) == 300
    assert set(np.unique(warped[~np.isnan(warped)])) <= set(np.unique(contour[~np.isnan(contour)]))
    assert np.isnan(warped).any()


def test_scale_intervals_keeps_median():
    contour = np.array([60.0, 62.0, 64.0])
    assert np.allclose(scale_intervals(contour, 2.0), [58.0, 62.0, 66.0])


def test_voicing_gaps_only_removes_frames():
    rng = np.random.default_rng(0)
    gapped = voicing_gaps(melody(), rng, 1.0, 0.4, FRAME_S)
    assert np.isnan(gapped).sum() > np.isnan(melody()).sum()


def test_augment_is_deterministic_per_seed():
    spec = ContourAugment()
    first = augment_contour(melody(), np.random.default_rng(3), spec, FRAME_S)
    second = augment_contour(melody(), np.random.default_rng(3), spec, FRAME_S)
    assert np.array_equal(first, second, equal_nan=True)
    assert first.dtype == np.float32
