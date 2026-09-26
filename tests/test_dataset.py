"""Crops and augmentation change with the epoch and the worker, not the row index."""

from pathlib import Path

import numpy as np

import hum2song.data.dataset as dataset_mod
from hum2song.config import TrainConfig
from hum2song.data.dataset import PairDataset
from hum2song.manifest import PairRecord
from tests.support import write_tone


def _dataset(root: Path) -> PairDataset:
    query = root / "query.wav"
    song = root / "song.wav"
    write_tone(query, 220.0, seconds=2.0)
    write_tone(song, 330.0, seconds=2.0)
    record = PairRecord(
        pair_id="humtrans:clip",
        query_path="query.wav",
        qtype="hum",
        qsource="real",
        song_id="humtrans:0001",
        song_start_s=0.0,
        song_dur_s=2.0,
        split="train",
        group="humtrans",
        song_path="song.wav",
        title=None,
    )
    config = TrainConfig(
        sample_rate=24000,
        crop_seconds=0.25,
        query_min_seconds=0.25,
        query_max_seconds=0.25,
        song_jitter_s=0.0,
        augment=False,
        seed=0,
        num_workers=0,
    )
    return PairDataset([record], root, config, training=True)


def test_a_new_epoch_changes_the_crop(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    dataset.set_epoch(0)
    first = dataset[0]["query"].numpy().copy()
    dataset.set_epoch(1)
    second = dataset[0]["query"].numpy().copy()
    dataset.set_epoch(0)
    repeated = dataset[0]["query"].numpy().copy()
    assert not np.array_equal(first, second)
    assert np.array_equal(first, repeated)


def test_worker_id_changes_the_crop(tmp_path: Path, monkeypatch) -> None:
    dataset = _dataset(tmp_path)

    class _Info:
        id = 3

    monkeypatch.setattr(dataset_mod, "get_worker_info", lambda: _Info())
    dataset.set_epoch(0)
    from_worker = dataset[0]["query"].numpy().copy()
    monkeypatch.setattr(dataset_mod, "get_worker_info", lambda: None)
    dataset.set_epoch(0)
    from_main = dataset[0]["query"].numpy().copy()
    assert not np.array_equal(from_worker, from_main)
