"""Pair dataset: query crop plus a song-side positive."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hum2song.audio import fit_length, load_audio
from hum2song.augment import augment_wave
from hum2song.manifest import PairRecord
from hum2song.model import QTYPE_TO_INDEX


class PairDataset(Dataset):
    """Load hum, whistle, and sung pairs. The positive is the melody, not the words."""

    def __init__(
        self,
        records: list[PairRecord],
        data_root: Path,
        config,
        training: bool,
    ) -> None:
        self.records = list(records)
        self.data_root = Path(data_root)
        self.config = config
        self.training = training
        self.partners = _partner_index(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rng = np.random.default_rng(self.config.seed + index)
        query = load_audio(self.data_root / record.query_path, self.config.sample_rate)
        if self.training:
            query = augment_wave(query, self.config.sample_rate, rng, self.config)
        song = load_audio(
            self.data_root / _positive_path(record, self.partners, rng),
            self.config.sample_rate,
        )
        return {
            "query": torch.from_numpy(_prepare_query(query, self.config, rng, self.training)),
            "song": torch.from_numpy(
                _prepare_song(song, record.song_start_s, self.config, rng, self.training)
            ),
            "qtype": QTYPE_TO_INDEX[record.qtype],
            "song_id": record.song_id,
        }


def collate_pairs(rows: list[dict]) -> dict:
    """Stack variable rows into one batch. song_id stays a list of strings."""
    return {
        "query": torch.stack([row["query"] for row in rows]),
        "song": torch.stack([row["song"] for row in rows]),
        "qtype": torch.tensor([row["qtype"] for row in rows], dtype=torch.long),
        "song_id": [row["song_id"] for row in rows],
    }


def _partner_index(records: list[PairRecord]) -> dict[str, list[str]]:
    partners: dict[str, list[str]] = defaultdict(list)
    for record in records:
        partners[record.song_id].append(record.query_path)
    return partners


def _positive_path(
    record: PairRecord,
    partners: dict[str, list[str]],
    rng: np.random.Generator,
) -> str:
    if record.song_path:
        return record.song_path
    choices = [path for path in partners.get(record.song_id, []) if path != record.query_path]
    if not choices:
        return record.query_path
    return choices[int(rng.integers(0, len(choices)))]


def _prepare_query(wave: np.ndarray, config, rng: np.random.Generator, training: bool) -> np.ndarray:
    target = _target_length(config)
    if not training:
        return fit_length(wave, target, rng, random_start=False)
    low = max(int(round(config.query_min_seconds * config.sample_rate)), 1)
    high = max(int(round(config.query_max_seconds * config.sample_rate)), 1)
    high = min(high, max(len(wave), 1))
    low = min(low, high)
    if high <= low:
        cropped = fit_length(wave, low, rng, random_start=True)
        return fit_length(cropped, target, rng, random_start=False)
    length = int(rng.integers(low, high + 1))
    cropped = fit_length(wave, length, rng, random_start=True)
    return fit_length(cropped, target, rng, random_start=False)


def _prepare_song(
    wave: np.ndarray,
    start_s: float | None,
    config,
    rng: np.random.Generator,
    training: bool,
) -> np.ndarray:
    target = _target_length(config)
    if start_s is None:
        return fit_length(wave, target, rng, random_start=training)
    start = float(start_s)
    if training and config.song_jitter_s > 0:
        start = max(0.0, start + float(rng.uniform(-config.song_jitter_s, config.song_jitter_s)))
    start_index = int(start * config.sample_rate)
    start_index = min(max(start_index, 0), max(len(wave) - 1, 0))
    return fit_length(wave[start_index:], target, rng, random_start=False)


def _target_length(config) -> int:
    return max(int(round(config.crop_seconds * config.sample_rate)), 1)
