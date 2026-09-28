import numpy as np

from hum2song.contour.augment import ContourAugment
from hum2song.contour.chad import (
    SPLIT_FRACTIONS,
    ChadPairDataset,
    ChadSong,
    fragment_window,
    read_split,
    split_groups,
    write_split,
)
from hum2song.contour.features import FRAME_S

FRAMES_PER_S = int(round(1.0 / FRAME_S))


def test_split_groups_is_deterministic_and_complete():
    groups = [f"g{i:03d}" for i in range(100)]
    first = split_groups(groups, seed=1)
    assert first == split_groups(list(reversed(groups)), seed=1)
    assert set(first) == set(groups)
    counts = {name: sum(v == name for v in first.values()) for name, _ in SPLIT_FRACTIONS}
    assert counts == {"train": 40, "val": 10, "test": 50}
    assert first != split_groups(groups, seed=2)


def test_split_round_trip(tmp_path):
    path = tmp_path / "split.json"
    write_split(path, {"a": "train", "b": "test"})
    assert read_split(path) == {"a": "train", "b": "test"}


def test_fragment_window_covers_the_interval():
    contour = np.arange(60 * FRAMES_PER_S, dtype=np.float32)
    window = fragment_window(contour, (10.0, 20.0), np.random.default_rng(0), 0.0, (0.0, 0.0))
    assert window[0] == 10 * FRAMES_PER_S
    assert len(window) == 10 * FRAMES_PER_S
    jittered = fragment_window(contour, (10.0, 20.0), np.random.default_rng(0), 1.0, (-1.0, 3.0))
    assert 9 * FRAMES_PER_S <= jittered[0] <= 10 * FRAMES_PER_S


def test_fragment_window_before_start_is_clamped():
    contour = np.arange(10 * FRAMES_PER_S, dtype=np.float32)
    window = fragment_window(contour, (0.2, 5.0), np.random.default_rng(0), 1.0, (0.0, 0.0))
    assert window[0] == 0


def test_chad_pair_dataset_item():
    song = ChadSong("g1", "vid", (5.0, 15.0))
    contour = 60.0 + np.zeros(40 * FRAMES_PER_S, dtype=np.float32)
    hums = [
        ("g1", 62.0 + np.zeros(10 * FRAMES_PER_S, dtype=np.float32)),
        ("g1", np.full(3, np.nan)),
    ]
    dataset = ChadPairDataset(
        hums, {"g1": song}, {"g1": contour}, ContourAugment(), seed=0, repeat=2
    )
    assert len(dataset) == 2
    item = dataset[1]
    assert item["song_id"] == "chad:g1"
    assert item["query"].shape[-1] == 2 and item["song"].shape[-1] == 2
