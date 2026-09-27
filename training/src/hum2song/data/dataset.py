"""Pair dataset: query crop plus a song-side positive.

HumTrans references are locked to their hums in key and time, so training breaks that
lock: the query is transposed and time-stretched independently (augment.py), the
reference crop may start later than the melody, and the positive may be another
singer's reference for the same segment, which is often a whole octave away.
"""

import multiprocessing
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from hum2song.audio import fit_length, load_audio
from hum2song.augment import augment_wave
from hum2song.manifest import PairRecord
from hum2song.model import QTYPE_TO_INDEX

HUMTRANS_GROUP = "humtrans"
HUMTRANS_ID_PARTS = 3


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
        self.segment_refs = _segment_index(self.records)
        self._epoch = multiprocessing.Value("i", 0)
        self._rng: np.random.Generator | None = None
        self._rng_key: tuple[int, int] | None = None

    def set_epoch(self, epoch: int) -> None:
        """Reseed crops and augmentation. Workers read this shared counter."""
        self._epoch.value = int(epoch)

    def _generator(self) -> np.random.Generator:
        """One stream per epoch and worker. It advances across items in that epoch."""
        worker_id = _worker_id()
        epoch = int(self._epoch.value)
        key = (epoch, worker_id)
        if self._rng is None or self._rng_key != key:
            self._rng = np.random.default_rng([self.config.seed, epoch, worker_id])
            self._rng_key = key
        return self._rng

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        rng = self._generator()
        query = load_audio(self.data_root / record.query_path, self.config.sample_rate)
        if self.training:
            query = _precrop_query(query, self.config, rng)
            query = augment_wave(query, self.config.sample_rate, rng, self.config)
        song = load_audio(
            self.data_root / self._song_path(record, rng),
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


    def _song_path(self, record: PairRecord, rng: np.random.Generator) -> str:
        other = self._maybe_other_take(record, rng)
        return other or _positive_path(record, self.partners, rng)

    def _maybe_other_take(self, record: PairRecord, rng: np.random.Generator) -> str | None:
        if not self.training or float(rng.random()) >= self.config.cross_take_prob:
            return None
        return _other_take_path(record, self.segment_refs, rng)


def segment_key(record: PairRecord) -> str | None:
    """HumTrans takes of one segment share song and segment ids: F01_0001_0002_1 -> 0001_0002."""
    parts = record.pair_id.split(":", 1)[-1].split("_")
    if record.group != HUMTRANS_GROUP or len(parts) < HUMTRANS_ID_PARTS or not record.song_path:
        return None
    return f"{record.group}:{parts[1]}_{parts[2]}"


def worker_init(_worker_index: int) -> None:
    """One torch thread per dataloader worker, so workers do not fight over cores."""
    torch.set_num_threads(1)


def _worker_id() -> int:
    info = get_worker_info()
    if info is None:
        return 0
    return int(info.id)


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


def _segment_index(records: list[PairRecord]) -> dict[str, list[str]]:
    references: dict[str, list[str]] = defaultdict(list)
    for record in records:
        key = segment_key(record)
        if key is not None:
            references[key].append(str(record.song_path))
    return references


def _other_take_path(
    record: PairRecord,
    segment_refs: dict[str, list[str]],
    rng: np.random.Generator,
) -> str | None:
    """A different singer's or take's reference for the same segment, if one exists."""
    key = segment_key(record)
    choices = [path for path in segment_refs.get(key or "", []) if path != record.song_path]
    if not choices:
        return None
    return choices[int(rng.integers(0, len(choices)))]


def _precrop_query(wave: np.ndarray, config, rng: np.random.Generator) -> np.ndarray:
    """Cut the longest window a training crop can need before the costly augmentations."""
    window_s = config.query_max_seconds / min(config.time_stretch_min, 1.0)
    window = max(int(np.ceil(window_s * config.sample_rate)), 1)
    if len(wave) <= window:
        return wave
    start = int(rng.integers(0, len(wave) - window + 1))
    return wave[start : start + window]


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
    if training:
        start = max(0.0, start + _song_shift_s(config, rng))
    start_index = int(start * config.sample_rate)
    start_index = min(max(start_index, 0), max(len(wave) - 1, 0))
    return fit_length(wave[start_index:], target, rng, random_start=False)


def _song_shift_s(config, rng: np.random.Generator) -> float:
    """Jitter around the aligned start plus a one-sided offset of up to song_offset_max_s."""
    jitter = float(rng.uniform(-config.song_jitter_s, config.song_jitter_s))
    offset = float(rng.uniform(0.0, config.song_offset_max_s))
    return jitter + offset


def _target_length(config) -> int:
    return max(int(round(config.crop_seconds * config.sample_rate)), 1)
