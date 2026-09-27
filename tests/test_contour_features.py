import numpy as np

from hum2song.contour.features import (
    FRAME_S,
    RMVPE_CENTS_OFFSET,
    RMVPE_CENTS_STEP,
    contour_features,
    decimate,
    drop_short_runs,
    hz_to_semitones,
    midi_contour,
    rmvpe_contour,
    salience_to_f0,
    trim_unvoiced,
)
from hum2song.midi_render import MidiNote


def test_hz_to_semitones_a4_and_octave():
    values = hz_to_semitones(np.array([440.0, 880.0]))
    assert np.allclose(values, [69.0, 81.0])


def test_salience_peak_bin_gives_its_frequency():
    salience = np.zeros((2, 360), dtype=np.float32)
    salience[0, 100] = 0.9
    salience[1, 0] = 0.2
    f0, confidence = salience_to_f0(salience)
    expected = 10.0 * 2.0 ** ((RMVPE_CENTS_OFFSET + RMVPE_CENTS_STEP * 100) / 1200.0)
    assert np.isclose(f0[0], expected, rtol=1e-5)
    assert np.allclose(confidence, [0.9, 0.2])


def test_decimate_averages_voiced_frames_only():
    values = np.array([60.0, np.nan, np.nan, np.nan, 62.0, 64.0])
    assert np.allclose(decimate(values, 2), [60.0, np.nan, 63.0], equal_nan=True)


def test_drop_short_runs():
    mask = np.array([True, False, True, True, True, False, True, True])
    assert drop_short_runs(mask, 3).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_rmvpe_contour_drops_low_confidence_and_decimates():
    frames = 200
    f0 = np.full(frames, 440.0)
    confidence = np.full(frames, 0.9)
    confidence[:40] = 0.05
    contour = rmvpe_contour(np.stack([f0, confidence]))
    assert len(contour) == frames // 2
    assert np.all(np.isnan(contour[:20]))
    assert np.allclose(contour[25:], 69.0)


def test_midi_contour_places_notes_and_rests():
    notes = [MidiNote(0.0, 0.1, 60, 100), MidiNote(0.2, 0.3, 67, 100)]
    contour = midi_contour(notes)
    per_note = int(round(0.1 / FRAME_S))
    assert np.allclose(contour[:per_note], 60)
    assert np.all(np.isnan(contour[per_note : 2 * per_note]))
    assert np.allclose(contour[2 * per_note :], 67)


def test_features_ignore_transposition():
    contour = np.array([60.0, 62.0, np.nan, 64.0, 67.0], dtype=np.float32)
    assert np.allclose(contour_features(contour), contour_features(contour + 7.0))
    assert contour_features(contour)[:, 1].tolist() == [1, 1, 0, 1, 1]


def test_trim_unvoiced():
    contour = np.array([np.nan, 60.0, np.nan, 61.0, np.nan])
    assert len(trim_unvoiced(contour)) == 3
