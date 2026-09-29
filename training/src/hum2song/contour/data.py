"""Contour banks and the training pair dataset.

Query contours come from the cached RMVPE tracks (scripts/extract_f0.py); reference
contours come from the melody MIDI named by each record's song_path. A training pair is
a random window of a HumTrans hum and a loosely aligned window of its MIDI; the hum
window is then augmented (tempo, intervals, drift, gaps). Key needs no augmentation
because features are median-normalized.
"""

import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hum2song.contour.augment import (
    ContourAugment,
    WhistleAugment,
    augment_contour,
    humanize,
    whistle_like,
)
from hum2song.contour.features import (
    FRAME_S,
    contour_features,
    contour_salience_features,
    midi_contour,
    rmvpe_contour,
    trim_unvoiced,
    voiced_fraction,
)
from hum2song.manifest import PairRecord
from hum2song.midi_render import parse_midi

MIN_VOICED_FRACTION = 0.3
WINDOW_TRIES = 8


@dataclass(frozen=True)
class PairWindows:
    """Crop lengths and offsets for training pairs, in seconds."""

    query_min_s: float = 3.0
    query_max_s: float = 12.0
    ref_start_jitter_s: float = 1.0
    ref_extra_min_s: float = -1.0
    ref_extra_max_s: float = 3.0
    ref_stretch_min: float = 0.8
    ref_stretch_max: float = 1.25


def load_query_contours(npz_path: Path, paths: set[str] | None = None) -> dict[str, np.ndarray]:
    """RMVPE tracks from one extract_f0 file -> cleaned contours, keyed by query path."""
    with np.load(npz_path) as tracks:
        keys = [key for key in tracks.files if paths is None or key in paths]
        return {key: rmvpe_contour(tracks[key]) for key in keys}


def midi_index(raw_root: Path) -> dict[str, Path]:
    """MIDI file stem -> path, for every .mid under raw_root."""
    return {path.stem: path for path in raw_root.rglob("*.mid")}


def load_reference_contours(song_paths: set[str], midis: dict[str, Path]) -> dict[str, np.ndarray]:
    """Reference contour per song_path, from the MIDI with the same stem."""
    contours: dict[str, np.ndarray] = {}
    for song_path in sorted(song_paths):
        midi = midis.get(Path(song_path).stem)
        if midi is None:
            raise FileNotFoundError(f"no MIDI for reference {song_path}")
        contours[song_path] = midi_contour(parse_midi(midi.read_bytes()))
    return contours


def seconds_to_frames(seconds: float) -> int:
    return max(int(round(seconds / FRAME_S)), 1)


INPUT_KIND = "contour"


def set_input_kind(kind: str) -> None:
    """Switch hard-F0 features (D-012) and soft-salience features (E3) for this process."""
    global INPUT_KIND
    if kind not in ("contour", "salience"):
        raise ValueError(f"unknown input_kind: {kind}")
    INPUT_KIND = kind


def features_tensor(contour: np.ndarray) -> torch.Tensor:
    """Hard-F0 features (D-012) or soft-salience features from the same contour (E3)."""
    if INPUT_KIND == "salience":
        return torch.from_numpy(contour_salience_features(contour))
    return torch.from_numpy(contour_features(contour))


def collate_contours(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad feature rows to one length. Returns features and a validity mask."""
    length = max(row.shape[0] for row in rows)
    length += length % 2
    features = torch.zeros(len(rows), length, rows[0].shape[1])
    valid = torch.zeros(len(rows), length, dtype=torch.bool)
    for index, row in enumerate(rows):
        features[index, : row.shape[0]] = row
        valid[index, : row.shape[0]] = True
    return features, valid


def collate_pairs(items: list[dict]) -> dict:
    query, query_valid = collate_contours([item["query"] for item in items])
    song, song_valid = collate_contours([item["song"] for item in items])
    return {
        "query": query,
        "query_valid": query_valid,
        "song": song,
        "song_valid": song_valid,
        "song_id": [item["song_id"] for item in items],
    }


def stretch_nearest(contour: np.ndarray, factor: float) -> np.ndarray:
    length = max(int(round(len(contour) * factor)), 1)
    source = np.clip(np.round(np.linspace(0, len(contour) - 1, length)).astype(int), 0, None)
    return contour[source]


def voiced_start(contour: np.ndarray, length: int, rng: np.random.Generator) -> int:
    """A start whose window is at least MIN_VOICED_FRACTION voiced, if one is found."""
    last = max(len(contour) - length, 0)
    start = int(rng.integers(0, last + 1))
    for _attempt in range(WINDOW_TRIES):
        if voiced_fraction(contour[start : start + length]) >= MIN_VOICED_FRACTION:
            return start
        start = int(rng.integers(0, last + 1))
    return start


def crop_pair(
    query: np.ndarray, reference: np.ndarray, rng: np.random.Generator, spec: PairWindows
) -> tuple[np.ndarray, np.ndarray]:
    """Query window with enough voicing, and the reference window around the same time."""
    length = seconds_to_frames(rng.uniform(spec.query_min_s, spec.query_max_s))
    start = voiced_start(query, length, rng)
    query_crop = query[start : start + length]
    jitter = rng.uniform(-spec.ref_start_jitter_s, spec.ref_start_jitter_s)
    ref_start = int(np.clip(start + int(round(jitter / FRAME_S)), 0, max(len(reference) - 1, 0)))
    extra = int(round(rng.uniform(spec.ref_extra_min_s, spec.ref_extra_max_s) / FRAME_S))
    ref_length = max(len(query_crop) + extra, seconds_to_frames(1.0))
    ref_crop = stretch_nearest(
        reference[ref_start : ref_start + ref_length],
        float(rng.uniform(spec.ref_stretch_min, spec.ref_stretch_max)),
    )
    return trim_unvoiced(query_crop), trim_unvoiced(ref_crop)


class EpochCounter:
    """Epoch number in shared memory, so persistent dataloader workers see each new epoch."""

    def __init__(self) -> None:
        self._value = multiprocessing.Value("i", 0)

    def get(self) -> int:
        return int(self._value.value)

    def set(self, epoch: int) -> None:
        self._value.value = int(epoch)


def set_epoch(dataset: Dataset, epoch: int) -> None:
    """Reseed crops for one dataset or every part of a ConcatDataset."""
    for part in getattr(dataset, "datasets", [dataset]):
        part.epoch.set(epoch)


class ContourPairDataset(Dataset):
    """Training pairs of (augmented hum window, loosely aligned MIDI window)."""

    def __init__(
        self,
        records: list[PairRecord],
        queries: dict[str, np.ndarray],
        references: dict[str, np.ndarray],
        augment: ContourAugment,
        windows: PairWindows,
        seed: int,
        whistle: WhistleAugment | None = None,
    ) -> None:
        self.records = [r for r in records if r.query_path in queries and r.song_path in references]
        self.whistle = whistle or WhistleAugment()
        self.queries = queries
        self.references = references
        self.augment = augment
        self.windows = windows
        self.seed = seed
        self.epoch = EpochCounter()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index])
        record = self.records[index]
        query, reference = crop_pair(
            self.queries[record.query_path], self.references[record.song_path], rng, self.windows
        )
        query = augment_contour(query, rng, self.augment, FRAME_S)
        query = whistle_like(query, rng, self.whistle, FRAME_S)
        return {
            "query": features_tensor(query),
            "song": features_tensor(reference),
            "song_id": record.song_id,
        }


class SyntheticPairDataset(Dataset):
    """Pairs made from melody MIDI alone: the query is the same window, humanized and augmented.

    Adds melodies HumTrans lacks (e.g. Essen folk songs not used as distractors).
    """

    def __init__(
        self,
        melodies: dict[str, np.ndarray],
        augment: ContourAugment,
        windows: PairWindows,
        seed: int,
    ) -> None:
        self.ids = sorted(melodies)
        self.melodies = melodies
        self.augment = augment
        self.windows = windows
        self.seed = seed
        self.epoch = EpochCounter()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 1])
        melody = self.melodies[self.ids[index]]
        query, reference = crop_pair(melody, melody, rng, self.windows)
        query = augment_contour(humanize(query, rng, FRAME_S), rng, self.augment, FRAME_S)
        return {
            "query": features_tensor(query),
            "song": features_tensor(reference),
            "song_id": self.ids[index],
        }


class WhistleSynthDataset(Dataset):
    """E4: whistle-like queries from many melodies, paired with clean refs.

    Always applies compress / octave-fold / gaps so the model sees whistle-shaped
    queries without memorizing the tiny MLEnd song set (D-016 failure mode).
    """

    def __init__(
        self,
        melodies: dict[str, np.ndarray],
        whistle: WhistleAugment,
        augment: ContourAugment,
        windows: PairWindows,
        seed: int,
    ) -> None:
        self.ids = sorted(melodies)
        self.melodies = melodies
        self.whistle = whistle
        self.augment = augment
        self.windows = windows
        self.seed = seed
        self.epoch = EpochCounter()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 4])
        melody = self.melodies[self.ids[index]]
        query, reference = crop_pair(melody, melody, rng, self.windows)
        always = WhistleAugment(
            probability=1.0,
            compress_min=self.whistle.compress_min,
            compress_max=self.whistle.compress_max,
            gap_prob=self.whistle.gap_prob,
            gap_max_s=self.whistle.gap_max_s,
            fold_prob=self.whistle.fold_prob,
            fold_range_st=self.whistle.fold_range_st,
        )
        query = whistle_like(query, rng, always, FRAME_S)
        query = augment_contour(humanize(query, rng, FRAME_S), rng, self.augment, FRAME_S)
        return {
            "query": features_tensor(query),
            "song": features_tensor(reference),
            "song_id": self.ids[index],
        }
