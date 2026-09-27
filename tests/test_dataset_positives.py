"""Other takes of a HumTrans segment can be positives; the reference crop can start late."""

from pathlib import Path

import numpy as np
import torch

from hum2song.config import TrainConfig
from hum2song.data.dataset import PairDataset, _prepare_song, segment_key, worker_init
from hum2song.manifest import PairRecord
from tests.support import write_tone


def _record(take: str, song_path: str, group: str = "humtrans") -> PairRecord:
    return PairRecord(
        pair_id=f"{group}:{take}",
        query_path=f"queries/{take}.wav",
        qtype="hum",
        qsource="real",
        song_id=f"{group}:0001",
        song_start_s=0.0,
        song_dur_s=1.0,
        split="train",
        group=group,
        song_path=song_path,
        title=None,
    )


def _config(**overrides) -> TrainConfig:
    base = {
        "crop_seconds": 0.25,
        "query_min_seconds": 0.25,
        "query_max_seconds": 0.25,
        "song_jitter_s": 0.0,
        "augment": False,
        "num_workers": 0,
    }
    return TrainConfig(**{**base, **overrides})


def test_segment_key_groups_singers_and_takes() -> None:
    first = _record("F01_0001_0002_1_D", "catalog/a.wav")
    second = _record("M03_0001_0002_2", "catalog/b.wav")
    other_segment = _record("F01_0001_0003_1", "catalog/c.wav")
    assert segment_key(first) == segment_key(second) == "humtrans:0001_0002"
    assert segment_key(other_segment) != segment_key(first)
    assert segment_key(_record("year2003:person1:00013", "x.wav", group="mirqbsh")) is None


def test_cross_take_positive_uses_another_takes_reference(tmp_path: Path) -> None:
    records = [
        _record("F01_0001_0002_1", "catalog/low.wav"),
        _record("M01_0001_0002_1", "catalog/high.wav"),
    ]
    for record in records:
        write_tone(tmp_path / record.query_path, 220.0)
    write_tone(tmp_path / "catalog/low.wav", 110.0)
    write_tone(tmp_path / "catalog/high.wav", 440.0)
    dataset = PairDataset(records, tmp_path, _config(cross_take_prob=1.0), training=True)
    rng = np.random.default_rng(0)
    assert dataset._song_path(records[0], rng) == "catalog/high.wav"
    assert dataset._song_path(records[1], rng) == "catalog/low.wav"
    item = dataset[0]
    assert item["song"].shape == item["query"].shape


def test_cross_take_is_off_at_eval(tmp_path: Path) -> None:
    records = [
        _record("F01_0001_0002_1", "catalog/low.wav"),
        _record("M01_0001_0002_1", "catalog/high.wav"),
    ]
    dataset = PairDataset(records, tmp_path, _config(cross_take_prob=1.0), training=False)
    assert dataset._song_path(records[0], np.random.default_rng(0)) == "catalog/low.wav"


def test_song_offset_starts_the_reference_crop_later() -> None:
    wave = np.arange(24000 * 4, dtype=np.float32)
    config = _config(song_offset_max_s=2.0)
    starts = [
        float(_prepare_song(wave, 0.0, config, np.random.default_rng(seed), True)[0]) / 24000
        for seed in range(20)
    ]
    assert min(starts) >= 0.0
    assert max(starts) <= 2.0
    assert max(starts) > 0.5


def test_worker_init_uses_one_torch_thread() -> None:
    before = torch.get_num_threads()
    try:
        worker_init(0)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
