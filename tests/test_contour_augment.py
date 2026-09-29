import numpy as np

from hum2song.contour.augment import (
    ContourAugment,
    WhistleAugment,
    augment_contour,
    humanize,
    octave_fold,
    scale_intervals,
    time_warp,
    voicing_gaps,
    whistle_like,
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


def test_humanize_keeps_gaps_and_stays_near_the_notes():
    contour = melody()
    human = humanize(contour, np.random.default_rng(1), FRAME_S)
    assert np.array_equal(np.isnan(human), np.isnan(contour))
    assert np.nanmax(np.abs(human - contour)) < 2.0


def test_octave_fold_keeps_pitches_near_the_median():
    contour = np.array([60.0, 72.0, 48.0, np.nan], dtype=np.float32)
    folded = octave_fold(contour, range_st=6.0)
    voiced = folded[~np.isnan(folded)]
    assert np.nanmax(np.abs(voiced - 60.0)) <= 6.0
    assert np.isnan(folded[3])


def test_whistle_like_always_compresses_when_probability_is_one():
    rng = np.random.default_rng(0)
    contour = np.array([55.0, 60.0, 65.0, 70.0], dtype=np.float32)
    spec = WhistleAugment(probability=1.0, fold_prob=1.0, gap_prob=0.0)
    out = whistle_like(contour, rng, spec, FRAME_S)
    span = float(np.nanmax(out) - np.nanmin(out))
    assert span < float(np.nanmax(contour) - np.nanmin(contour))


def test_humanize_handles_contours_shorter_than_the_glide():
    short = np.array([60.0, 62.0, 64.0], dtype=np.float32)
    out = humanize(short, np.random.default_rng(0), FRAME_S, glide_s=0.08)
    assert len(out) == 3 and out.dtype == np.float32
