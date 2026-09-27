"""Song-melody references (D-014): annotation loading, frame scoring, retrieval helpers."""

import numpy as np
import pytest
import soundfile as sf

from hum2song.contour.features import FRAME_S
from hum2song.contour.melody import frame_scores, track_to_hz
from hum2song.contour.melody_data import (
    LabelledClip,
    mir1k_clip,
    ref_txt_clip,
    reference_contour,
    semitones_to_hz,
    vocal_subset,
)
from hum2song.contour.song_eval import QUERY_FRACTION, crop, retrieval, synthetic_hum

pytest.importorskip("mir_eval")


def write_wav(path, seconds=1.0, rate=16000, channels=1):
    sf.write(str(path), np.zeros((int(seconds * rate), channels), dtype=np.float32), rate)


def test_semitones_to_hz_keeps_zero_unvoiced():
    np.testing.assert_allclose(semitones_to_hz(np.array([0.0, 69.0, 81.0])), [0.0, 440.0, 880.0])


def test_mir1k_clip_reads_pitch_on_20ms_grid(tmp_path):
    (tmp_path / "Wavfile").mkdir()
    (tmp_path / "PitchLabel").mkdir()
    write_wav(tmp_path / "Wavfile" / "abjones_1_01.wav", channels=2)
    np.savetxt(tmp_path / "PitchLabel" / "abjones_1_01.pv", [0.0, 60.0, 62.0])
    clip = mir1k_clip(tmp_path / "Wavfile" / "abjones_1_01.wav", tmp_path / "PitchLabel")
    assert clip.song == "abjones_1"
    np.testing.assert_allclose(clip.ref_times, [0.02, 0.04, 0.06])
    assert clip.ref_hz[0] == 0.0 and clip.ref_hz[1] == pytest.approx(261.63, abs=0.01)


def test_ref_txt_clip_finds_suffixed_wav(tmp_path):
    write_wav(tmp_path / "train13MIDI.wav", rate=44100)
    np.savetxt(tmp_path / "train13REF.txt", [[0.0, 0.0], [0.01, 220.0]])
    clip = ref_txt_clip(tmp_path / "train13REF.txt")
    assert clip.name == "train13" and clip.wav.name == "train13MIDI.wav"
    np.testing.assert_allclose(clip.ref_hz, [0.0, 220.0])


def test_vocal_subset_only_for_adc2004():
    names = ("pop1", "jazz1", "daisy2", "opera_male3")
    clips = [LabelledClip(name, None, name, np.zeros(1), np.zeros(1)) for name in names]
    assert [c.name for c in vocal_subset("adc2004", clips)] == ["pop1", "daisy2", "opera_male3"]
    assert vocal_subset("mirex05", clips) == []


def test_reference_contour_samples_frame_centers():
    times = np.arange(0.0, 1.0, 0.01)
    hz = np.where(times < 0.5, 440.0, 0.0)
    contour = reference_contour(times, hz)
    assert len(contour) == int(np.ceil(times[-1] / FRAME_S))
    assert np.allclose(contour[:24], 69.0)
    assert np.isnan(contour[26:]).all()


def test_track_to_hz_marks_unvoiced_guess_negative():
    track = np.array([[220.0, 330.0, 0.0], [0.9, 0.1, 0.0]])
    np.testing.assert_allclose(track_to_hz(track), [220.0, -330.0, 0.0])


def test_frame_scores_perfect_track():
    times = np.arange(100) * 0.01
    hz = np.where(np.arange(100) < 60, 300.0, 0.0)
    track = np.stack([np.where(hz > 0, hz, 300.0), (hz > 0).astype(float)])
    scores = frame_scores(times, hz, track)
    assert scores["oa"] == pytest.approx(1.0) and scores["vfa"] == pytest.approx(0.0)


def test_crop_length_within_fraction():
    rng = np.random.default_rng(0)
    contour = np.concatenate([[np.nan] * 10, np.linspace(60, 70, 400), [np.nan] * 10])
    for _ in range(20):
        piece = crop(contour, rng)
        assert QUERY_FRACTION[0] * 400 - 1 <= len(piece) <= QUERY_FRACTION[1] * 400 + 1


def test_synthetic_hum_is_voiced_and_finite():
    melody = np.repeat([60.0, 62.0, 64.0, np.nan, 65.0], 30)
    hum = synthetic_hum(melody, np.random.default_rng(1))
    assert np.isfinite(hum[~np.isnan(hum)]).all() and (~np.isnan(hum)).sum() > 30


def test_retrieval_identity_embeddings_rank_first():
    embeddings = np.eye(5, dtype=np.float32)
    scores = retrieval(embeddings, embeddings, [f"clip{i}" for i in range(5)])
    assert scores["top1"] == 1.0 and scores["mrr"] == 1.0
