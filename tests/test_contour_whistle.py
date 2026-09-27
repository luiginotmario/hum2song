import numpy as np

from hum2song.contour.example_eval import ExampleSet, example_metrics, song_scores
from hum2song.contour.features import hz_to_semitones, rmvpe_contour
from hum2song.contour.whistle import SAMPLE_RATE, WHISTLE_F0_MAX_HZ, spectral_peak_track


def tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return 0.3 * np.sin(2 * np.pi * frequency * time)


def test_spectral_peak_finds_a_high_whistle_tone():
    track = spectral_peak_track(tone(2500.0))
    middle = track[:, 20:-20]
    assert np.allclose(middle[0], 2500.0, rtol=0.003)
    assert middle[1].min() == 1.0


def test_noise_is_not_voiced():
    rng = np.random.default_rng(0)
    track = spectral_peak_track(0.3 * rng.normal(size=SAMPLE_RATE))
    assert track[1].mean() < 0.05


def test_whistle_contour_keeps_frequencies_above_2khz():
    contour = rmvpe_contour(spectral_peak_track(tone(3000.0)), WHISTLE_F0_MAX_HZ)
    voiced = contour[~np.isnan(contour)]
    assert len(voiced) > 30
    assert np.allclose(voiced[2:-2], hz_to_semitones(np.array([3000.0]))[0], atol=0.05)


def example(songs, people, vectors) -> ExampleSet:
    embeddings = np.array(vectors, dtype=np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return ExampleSet(embeddings=embeddings, songs=songs, people=people)


def test_own_performer_is_left_out_of_the_reference():
    refs = example(["a", "a", "b"], ["p1", "p2", "p3"], [[1, 0], [0, 1], [0.6, 0.8]])
    query = example(["a"], ["p1"], [[1, 0]])
    scores, songs = song_scores(query, refs, "max")
    assert songs == ["a", "b"]
    assert np.isclose(scores[0, 0], 0.0) and np.isclose(scores[0, 1], 0.6)
    assert example_metrics(query, refs, "max")["top1"] == 0.0


def test_centroid_ranks_the_right_song():
    refs = example(
        ["a", "a", "b", "b"], ["p2", "p3", "p2", "p3"], [[1, 0.1], [1, -0.1], [0, 1], [0.1, 1]]
    )
    query = example(["a", "b"], ["p1", "p1"], [[1, 0], [0, 1]])
    assert example_metrics(query, refs, "centroid")["top1"] == 1.0
