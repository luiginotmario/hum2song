"""MLEnd performer split, pair dataset and whistle-like augmentation (D-015)."""

import numpy as np

from hum2song.contour.augment import ContourAugment
from hum2song.contour.mlend import (
    SPLIT_MANIFEST,
    MLEndClip,
    MLEndPairDataset,
    clips_in,
    crop_fraction,
    read_split,
    split_performers,
    write_split,
)

PEOPLE = [f"P{i:03d}" for i in range(100)]


def test_split_is_deterministic_and_covers_everyone():
    first = split_performers(PEOPLE * 2)
    assert first == split_performers(list(reversed(PEOPLE)))
    assert set(first) == set(PEOPLE)
    counts = {name: sum(v == name for v in first.values()) for name in ("train", "val", "test")}
    assert counts == {"train": 60, "val": 20, "test": 20}


def test_split_changes_with_seed():
    assert split_performers(PEOPLE, seed=1) != split_performers(PEOPLE, seed=2)


def test_write_and_read_split_round_trip(tmp_path):
    assignment = split_performers(PEOPLE)
    clips = [MLEndClip(f"q{i}.wav", "Song", person, "hum") for i, person in enumerate(PEOPLE)]
    path = tmp_path / "split.json"
    write_split(path, assignment, clips)
    assert read_split(path) == assignment


def test_committed_manifest_is_disjoint():
    split = read_split(SPLIT_MANIFEST)
    assert set(split.values()) == {"train", "val", "test"}
    assert len(split) > 200


def test_clips_in_filters_split_and_type():
    split = {"a": "train", "b": "test"}
    clips = [
        MLEndClip("1", "S", "a", "hum"),
        MLEndClip("2", "S", "b", "hum"),
        MLEndClip("3", "S", "b", "whistle"),
    ]
    assert [c.path for c in clips_in(clips, split, "test", "hum")] == ["2"]
    assert [c.path for c in clips_in(clips, split, "all", "hum")] == ["1", "2"]


def test_crop_fraction_stays_within_voiced_span():
    contour = np.concatenate([[np.nan] * 5, np.linspace(60, 70, 100), [np.nan] * 5])
    piece = crop_fraction(contour, np.random.default_rng(0))
    assert 60 <= len(piece) <= 100 and not np.isnan(piece[0])


def test_pair_dataset_pairs_other_people_of_same_song():
    whistles = [MLEndClip("w1", "Frozen", "a", "whistle")]
    hums = [
        MLEndClip("h1", "Frozen", "a", "hum"),
        MLEndClip("h2", "Frozen", "b", "hum"),
        MLEndClip("h3", "Potter", "c", "hum"),
    ]
    contours = {name: np.linspace(60, 67, 300) for name in ("w1", "h1", "h2", "h3")}
    dataset = MLEndPairDataset(whistles, hums, contours, ContourAugment(), seed=0, repeat=3)
    assert len(dataset) == 3
    rng = np.random.default_rng(0)
    assert {dataset.partner(whistles[0], rng).path for _ in range(20)} == {"h2"}
    item = dataset[1]
    assert item["song_id"] == "mlend:Frozen" and item["query"].shape[1] == 2


def test_whistle_like_compresses_range_and_is_off_at_zero():
    from hum2song.contour.augment import WhistleAugment, whistle_like

    contour = np.linspace(55.0, 65.0, 200)
    rng = np.random.default_rng(0)
    assert whistle_like(contour, rng, WhistleAugment(), 0.02) is contour
    spec = WhistleAugment(probability=1.0, gap_prob=0.0)
    out = whistle_like(contour, rng, spec, 0.02)
    span = np.nanmax(out) - np.nanmin(out)
    assert 0.55 * 10.0 - 1e-3 <= span <= 0.85 * 10.0 + 1e-3


def test_selection_score_is_mean_of_listed_metrics():
    from hum2song.contour.train import selection_score

    metrics = {"a": 0.5, "b": 1.0}
    assert selection_score(metrics, "a,b") == 0.75
    assert selection_score(metrics, "a") == 0.5
    assert selection_score(metrics, "a,c") == float("-inf")


def test_parse_sets_reads_yaml_values():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "training" / "scripts" / "train_contour.py"
    spec = importlib.util.spec_from_file_location("train_contour_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_sets(["whistle_aug_prob=0.3", "mlend_pairs=false"]) == {
        "whistle_aug_prob": 0.3,
        "mlend_pairs": False,
    }
