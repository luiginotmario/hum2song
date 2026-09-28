"""CHAD real hum <-> real recording pairs for training and validation (D-019).

CHAD gives, per song group, hums and the interval of the original YouTube recording they
hum. The full recordings (downloaded for D-019, features only) are cached as melody tracks
in library/youtube_v1/tracks/youtube_<video id>.npz. Songs are split once with a fixed
seed (training/splits/chad_songs.json): test songs are never trained on or used to pick a
checkpoint; val songs pick checkpoints.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hum2song.catalog.library import song_chunks
from hum2song.catalog.real_eval import song_scores, target_ranks
from hum2song.catalog.targets import chad_originals, chad_queries
from hum2song.contour.augment import ContourAugment, augment_contour
from hum2song.contour.data import EpochCounter, features_tensor
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.features import FRAME_S, rmvpe_contour, trim_unvoiced

SPLIT_MANIFEST = Path(__file__).resolve().parents[3] / "splits" / "chad_songs.json"
SPLIT_SEED = 20260928
SPLIT_FRACTIONS = (("train", 0.4), ("val", 0.1), ("test", 0.5))
TRACK_DIR = "library/youtube_v1/tracks"
QUERY_CACHE = "library/query_contours.npz"
MIN_REFERENCE_FRAMES = 25


@dataclass(frozen=True)
class ChadSong:
    group: str
    youtube_id: str
    fragment_s: tuple[float, float]


def track_path(root: Path, youtube_id: str) -> Path:
    return root / TRACK_DIR / f"youtube_{youtube_id}.npz"


def available_songs(root: Path) -> list[ChadSong]:
    """CHAD originals whose full recording was extracted."""
    songs = [
        ChadSong(o["target_id"].split(":", 1)[1], o["youtube_id"], tuple(o["fragment_s"]))
        for o in chad_originals(root)
    ]
    return [s for s in songs if track_path(root, s.youtube_id).exists()]


def split_groups(groups: list[str], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Deterministic song split: sorted groups, one permutation, fixed fractions."""
    order = [sorted(groups)[i] for i in np.random.default_rng(seed).permutation(len(groups))]
    assignment, start = {}, 0
    for index, (name, fraction) in enumerate(SPLIT_FRACTIONS):
        end = (
            len(order)
            if index == len(SPLIT_FRACTIONS) - 1
            else start + round(fraction * len(order))
        )
        assignment.update({group: name for group in order[start:end]})
        start = end
    return assignment


def write_split(path: Path, assignment: dict[str, str]) -> None:
    counts = {name: sum(v == name for v in assignment.values()) for name, _ in SPLIT_FRACTIONS}
    payload = {
        "seed": SPLIT_SEED,
        "unit": "song (CHAD group)",
        "counts": counts,
        "groups": dict(sorted(assignment.items())),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def read_split(path: Path = SPLIT_MANIFEST) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["groups"]


def hum_contours(root: Path, groups: set[str]) -> list[tuple[str, np.ndarray]]:
    """(group, cached query contour) for every CHAD hum of `groups`."""
    with np.load(root / QUERY_CACHE, allow_pickle=True) as cache:
        return [
            (q["target_id"].split(":", 1)[1], cache[q["path"]])
            for q in chad_queries(root)
            if q["target_id"].split(":", 1)[1] in groups and q["path"] in cache
        ]


def song_contour(root: Path, song: ChadSong) -> np.ndarray:
    with np.load(track_path(root, song.youtube_id)) as tracks:
        return rmvpe_contour(tracks["vocals"])


def fragment_window(contour: np.ndarray, fragment_s, rng, jitter_s: float, extra_s) -> np.ndarray:
    """The hummed interval of the full-song contour, start jittered earlier and end extended."""
    start_s = fragment_s[0] - rng.uniform(0.0, jitter_s)
    end_s = fragment_s[1] + rng.uniform(*extra_s)
    start, end = max(int(start_s / FRAME_S), 0), max(int(end_s / FRAME_S), 0)
    return trim_unvoiced(contour[start:end]) if end > start else contour[:1]


class ChadPairDataset(Dataset):
    """(augmented real hum, matching window of the real recording's vocal melody)."""

    def __init__(
        self,
        hums,
        songs: dict,
        contours: dict,
        augment: ContourAugment,
        seed: int,
        repeat: int = 1,
        jitter_s: float = 1.0,
        extra_s=(-1.0, 3.0),
    ) -> None:
        self.hums = [(g, c) for g, c in hums if len(trim_unvoiced(c)) >= MIN_REFERENCE_FRAMES]
        self.songs = songs
        self.contours = contours
        self.augment = augment
        self.seed = seed
        self.repeat = repeat
        self.jitter_s = jitter_s
        self.extra_s = extra_s
        self.epoch = EpochCounter()

    def __len__(self) -> int:
        return len(self.hums) * self.repeat

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 3])
        group, hum = self.hums[index % len(self.hums)]
        song = self.songs[group]
        query = augment_contour(trim_unvoiced(hum), rng, self.augment, FRAME_S)
        window = fragment_window(
            self.contours[group], song.fragment_s, rng, self.jitter_s, self.extra_s
        )
        return {
            "query": features_tensor(query),
            "song": features_tensor(window),
            "song_id": f"chad:{group}",
        }


def training_pairs(
    root: Path, augment: ContourAugment, seed: int, repeat: int, config
) -> ChadPairDataset:
    split = read_split()
    songs = {s.group: s for s in available_songs(root) if split.get(s.group) == "train"}
    contours = {group: song_contour(root, song) for group, song in songs.items()}
    hums = hum_contours(root, set(songs))
    extra = (config.ref_extra_min_s, config.ref_extra_max_s)
    return ChadPairDataset(
        hums, songs, contours, augment, seed, repeat, config.ref_start_jitter_s, extra
    )


class ChadValidation:
    """Val-song hums ranked against every train and val full recording (best 10 s chunk)."""

    def __init__(self, root: Path) -> None:
        split = read_split()
        songs = [s for s in available_songs(root) if split.get(s.group) in ("train", "val")]
        val_groups = {s.group for s in songs if split[s.group] == "val"}
        self.hums = hum_contours(root, val_groups)
        self.chunks, self.owner, self.groups = [], [], []
        for song in songs:
            chunks = song_chunks(np.load(track_path(root, song.youtube_id))["vocals"])
            self.chunks += [chunk.contour for chunk in chunks]
            self.owner += [len(self.groups)] * len(chunks)
            self.groups.append(song.group if chunks else None)

    def metrics(self, model, device) -> dict[str, float]:
        if not self.hums:
            return {}
        queries = torch.from_numpy(embed_contours(model, [c for _, c in self.hums], device)).to(
            device
        )
        chunks = torch.from_numpy(embed_contours(model, self.chunks, device)).to(device)
        owner = torch.tensor(self.owner, device=device)
        scores = song_scores(queries, chunks, owner, len(self.groups))
        ranks = target_ranks(scores, self.groups, [g for g, _ in self.hums])
        found = np.array([r if r is not None else np.inf for r in ranks], dtype=float)
        return {
            "val/chad_val_top1": float(np.mean(found <= 1)),
            "val/chad_val_top10": float(np.mean(found <= 10)),
        }
