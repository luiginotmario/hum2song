"""MLEnd hums and whistles: performer split, contours, training pairs, query-by-example sets.

MLEnd (8 songs, about 200 people) has no melody references, so it is used query-by-example
(D-013). D-015 adds whistle training, which needs people who are never evaluated on: the
performers are split once into train / val / test with a fixed seed, and the assignment is
saved in training/splits/mlend_performers.json. Code reads that file and never re-draws it.
D-016 also holds out 2 of the 8 songs (training/splits/mlend_heldout_songs.json, drawn
with a fixed seed): with song_holdout on, their clips leave whistle-pair training and
MLEnd validation, and any HumTrans songs listed there leave HumTrans training.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from hum2song.contour.augment import ContourAugment, augment_contour
from hum2song.contour.data import EpochCounter, features_tensor
from hum2song.contour.evaluate import embed_contours, query_contour
from hum2song.contour.example_eval import ExampleSet, example_metrics
from hum2song.contour.features import F0_MAX_HZ, FRAME_S, rmvpe_contour, trim_unvoiced
from hum2song.contour.whistle import WHISTLE_F0_MAX_HZ
from hum2song.manifest import read_pairs

ATTRIBUTES = "raw/mlend_hums_whistles/MLEndHWD_audio_attributes_benchmark.csv"
SPLIT_MANIFEST = Path(__file__).resolve().parents[3] / "splits" / "mlend_performers.json"
SPLIT_SEED = 20260927
SPLIT_FRACTIONS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
SONG_MANIFEST = SPLIT_MANIFEST.with_name("mlend_heldout_songs.json")
SONG_SEED = 20260928
HELDOUT_SONG_COUNT = 2
TRACKERS = {"hum": "rmvpe", "whistle": "peak"}
F0_LIMITS = {"rmvpe": F0_MAX_HZ, "peak": WHISTLE_F0_MAX_HZ, "rmvpe_half": WHISTLE_F0_MAX_HZ}
QUERY_FRACTION = (0.6, 1.0)
PAIR_TRIES = 8


@dataclass(frozen=True, order=True)
class MLEndClip:
    path: str
    song: str
    person: str
    qtype: str


def performers(root: Path) -> dict[str, str]:
    """MLEnd file name (0000.wav) -> interpreter id."""
    with open(root / ATTRIBUTES, newline="", encoding="utf-8") as handle:
        return {row["filename"]: row["Interpreter"] for row in csv.DictReader(handle)}


def mlend_clips(root: Path) -> list[MLEndClip]:
    """Every MLEnd clip in pairs_real.jsonl, sorted by path."""
    people = performers(root)
    records = [r for r in read_pairs(root / "pairs_real.jsonl") if r.group == "mlend"]
    return sorted(
        MLEndClip(
            r.query_path, r.song_id, people[Path(r.query_path).name.removeprefix("mlend_")], r.qtype
        )
        for r in records
    )


def split_performers(people: list[str], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Shuffle the sorted unique ids with `seed` and cut by SPLIT_FRACTIONS."""
    unique = sorted(set(people))
    order = np.random.default_rng(seed).permutation(len(unique))
    cuts = np.cumsum([fraction for _name, fraction in SPLIT_FRACTIONS])
    bounds = np.round(cuts * len(unique)).astype(int)
    assignment = {}
    for position, index in enumerate(order):
        split = next(
            name
            for (name, _f), bound in zip(SPLIT_FRACTIONS, bounds, strict=True)
            if position < bound
        )
        assignment[unique[index]] = split
    return assignment


def write_split(path: Path, assignment: dict[str, str], clips: list[MLEndClip]) -> None:
    counts = {
        name: {
            "performers": sum(split == name for split in assignment.values()),
            "hums": sum(assignment[c.person] == name for c in clips if c.qtype == "hum"),
            "whistles": sum(assignment[c.person] == name for c in clips if c.qtype == "whistle"),
        }
        for name, _fraction in SPLIT_FRACTIONS
    }
    payload = {
        "seed": SPLIT_SEED,
        "fractions": dict(SPLIT_FRACTIONS),
        "counts": counts,
        "performers": dict(sorted(assignment.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_split(path: Path = SPLIT_MANIFEST) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["performers"]


def choose_heldout_songs(songs: list[str], seed: int = SONG_SEED) -> list[str]:
    """HELDOUT_SONG_COUNT songs drawn from the sorted unique ids with `seed`."""
    unique = sorted(set(songs))
    picked = np.random.default_rng(seed).choice(len(unique), HELDOUT_SONG_COUNT, replace=False)
    return sorted(unique[int(index)] for index in picked)


def write_song_holdout(path: Path, songs: list[str], humtrans_exclude: list[str]) -> None:
    payload = {"seed": SONG_SEED, "songs": songs, "humtrans_exclude": sorted(humtrans_exclude)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_song_holdout(path: Path = SONG_MANIFEST) -> dict:
    """{"songs": held-out MLEnd song ids, "humtrans_exclude": HumTrans song ids to drop}."""
    return json.loads(path.read_text(encoding="utf-8"))


def without_songs(clips: list[MLEndClip], songs: set[str]) -> list[MLEndClip]:
    return [c for c in clips if c.song not in songs]


def clips_in(
    clips: list[MLEndClip], split: dict[str, str], name: str, qtype: str
) -> list[MLEndClip]:
    """Clips of one query type whose performer is in split `name` ("all" keeps everyone)."""
    return [c for c in clips if c.qtype == qtype and (name == "all" or split[c.person] == name)]


def load_contours(root: Path, method: str, clips: list[MLEndClip]) -> dict[str, np.ndarray]:
    """Cleaned contours from f0/<method>_mlend.npz, keyed by clip path."""
    with np.load(root / "f0" / f"{method}_mlend.npz") as tracks:
        return {c.path: rmvpe_contour(tracks[c.path], F0_LIMITS[method]) for c in clips}


def example_set(model, clips: list[MLEndClip], contours: dict, device) -> ExampleSet:
    queries = [query_contour(contours[c.path]) for c in clips]
    return ExampleSet(
        embeddings=embed_contours(model, queries, device),
        songs=[c.song for c in clips],
        people=[c.person for c in clips],
    )


class MLEndValidation:
    """Whistle -> hum and hum -> hum top-1 (centroid) among one split's performers."""

    def __init__(
        self,
        root: Path,
        split_name: str,
        split: dict[str, str] | None = None,
        exclude_songs: set[str] = frozenset(),
    ) -> None:
        clips = without_songs(mlend_clips(root), exclude_songs)
        split = split if split is not None else read_split()
        self.name = split_name
        self.hums = clips_in(clips, split, split_name, "hum")
        self.whistles = clips_in(clips, split, split_name, "whistle")
        self.contours = {
            **load_contours(root, TRACKERS["hum"], self.hums),
            **load_contours(root, TRACKERS["whistle"], self.whistles),
        }

    def metrics(self, model, device) -> dict[str, float]:
        hums = example_set(model, self.hums, self.contours, device)
        whistles = example_set(model, self.whistles, self.contours, device)
        prefix = f"val/mlend_{self.name}"
        return {
            f"{prefix}_whistle_top1": example_metrics(whistles, hums, "centroid")["top1"],
            f"{prefix}_hum_top1": example_metrics(hums, hums, "centroid")["top1"],
        }


def crop_fraction(contour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random QUERY_FRACTION-long slice of the voiced span."""
    span = trim_unvoiced(contour)
    length = max(int(len(span) * rng.uniform(*QUERY_FRACTION)), 1)
    start = int(rng.integers(0, len(span) - length + 1))
    return trim_unvoiced(span[start : start + length])


class MLEndPairDataset(Dataset):
    """Training pairs (augmented whistle crop, whole hum of the same song by someone else).

    `repeat` lists every whistle that many times per epoch; each copy draws its own crop,
    augmentation and hum partner. song_id is "mlend:<song>", so the loss masks other
    clips of the same song instead of using them as negatives.
    """

    def __init__(
        self,
        whistles: list[MLEndClip],
        hums: list[MLEndClip],
        contours: dict[str, np.ndarray],
        augment: ContourAugment,
        seed: int,
        repeat: int = 1,
    ) -> None:
        self.whistles = whistles
        self.hums_by_song = {
            song: [h for h in hums if h.song == song] for song in {h.song for h in hums}
        }
        self.contours = contours
        self.augment = augment
        self.seed = seed
        self.repeat = repeat
        self.epoch = EpochCounter()

    def __len__(self) -> int:
        return len(self.whistles) * self.repeat

    def partner(self, whistle: MLEndClip, rng: np.random.Generator) -> MLEndClip:
        """A hum of the same song, by a different performer when one can be drawn."""
        candidates = self.hums_by_song[whistle.song]
        choice = candidates[int(rng.integers(len(candidates)))]
        for _attempt in range(PAIR_TRIES):
            if choice.person != whistle.person:
                return choice
            choice = candidates[int(rng.integers(len(candidates)))]
        return choice

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 2])
        whistle = self.whistles[index % len(self.whistles)]
        query = crop_fraction(self.contours[whistle.path], rng)
        query = augment_contour(query, rng, self.augment, FRAME_S)
        hum = query_contour(self.contours[self.partner(whistle, rng).path])
        return {
            "query": features_tensor(query),
            "song": features_tensor(hum),
            "song_id": f"mlend:{whistle.song}",
        }


def training_pairs(
    root: Path,
    augment: ContourAugment,
    seed: int,
    repeat: int,
    exclude_songs: set[str] = frozenset(),
) -> MLEndPairDataset:
    """Train-split whistles (spectral-peak F0, D-007) paired with train-split hums."""
    clips = without_songs(mlend_clips(root), exclude_songs)
    split = read_split()
    whistles = clips_in(clips, split, "train", "whistle")
    hums = clips_in(clips, split, "train", "hum")
    contours = {
        **load_contours(root, TRACKERS["hum"], hums),
        **load_contours(root, TRACKERS["whistle"], whistles),
    }
    return MLEndPairDataset(whistles, hums, contours, augment, seed, repeat)
