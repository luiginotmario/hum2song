"""E2a (D-026): self-supervised pairs from real song melody tracks, for CLEWS-style training.

Songs: the YouTube chart library (distractors, never targets) plus one half of the FMA
library by track-id parity, so the other FMA half stays untouched as evaluation distractors.
Songs whose normalized title matches any evaluation target are dropped.

One item = one song:
  query  a 3-12 s window of the song's indexed melody track (vocals; with probability
         `mix_prob` the same window from the full-mix track when the song has one, so the
         model sees two extraction routes), humanized and augmented like a hum
  refs   `refs` reference windows of the vocal track starting within ±`offset_s` of the
         query window, each a little longer or shorter and slightly stretched
The CLEWS loss (losses.clews_loss) takes the best reference window as the positive and the
closest window of every other song as its negative, so the model is never told which
reference window is "the" match.
"""

import re
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.real_eval import pick_track
from hum2song.catalog.real_pool import FMA_LIBRARY, fma_rows, library_rows, load_tracks
from hum2song.contour.augment import ContourAugment, augment_contour, humanize
from hum2song.contour.data import (
    EpochCounter,
    collate_contours,
    features_tensor,
    seconds_to_frames,
    stretch_nearest,
    voiced_start,
)
from hum2song.contour.features import FRAME_S, rmvpe_contour, trim_unvoiced

TARGET_LIBRARIES = ("youtube_v1", "previews_v1")
METHOD = "vocals_or_mix"
MIN_SONG_S = 20.0
MIN_QUERY_S = 1.0
QUERY_TRIES = 8


def title_key(title: str | None) -> str:
    """Lower-case letters and digits of the title before any bracket or dash."""
    head = re.split(r"[\(\[\-–]", title or "")[0]
    return re.sub(r"[^a-z0-9]", "", head.lower())


def target_titles(root: Path) -> set[str]:
    rows = library_rows(root, list(TARGET_LIBRARIES))
    return {title_key(r.get("title")) for r in rows if r["role"] == "target"} - {""}


def keep_fma(song_id: str, parity: str) -> bool:
    number = int(song_id.split(":")[1])
    return {"all": True, "even": number % 2 == 0, "odd": number % 2 == 1}.get(parity, False)


def training_songs(root: Path, libraries: list[str], fma_parity: str) -> list[dict]:
    """Rows of the training songs (see module docstring), with their track files."""
    rows = library_rows(root, libraries)
    fma_titles = {
        s["song_id"]: s.get("title")
        for s in read_jsonl(root / "library" / FMA_LIBRARY / "songs.jsonl")
    }
    fma = [
        {**r, "title": fma_titles.get(r["song_id"])}
        for r in fma_rows(root)
        if keep_fma(r["song_id"], fma_parity)
    ]
    blocked = target_titles(root)
    return [
        r for r in rows + fma if r["role"] != "target" and title_key(r.get("title")) not in blocked
    ]


def song_routes(row: dict) -> dict | None:
    """{"ref": indexed-track contour, "alt": full-mix contour or None} of one song."""
    tracks = load_tracks(row["_track"])
    if tracks is None:
        return None
    kind, chunks = pick_track(tracks, METHOD)
    if not chunks:
        return None
    ref = rmvpe_contour(tracks[kind])
    alt = (
        rmvpe_contour(tracks["mix"])
        if kind == "vocals" and row["_track"].endswith(".npz")
        else None
    )
    return {"ref": ref, "alt": alt} if len(ref) * FRAME_S >= MIN_SONG_S else None


def load_song_routes(rows: list[dict]) -> dict[str, dict]:
    routes = {row["song_id"]: song_routes(row) for row in rows}
    return {song: route for song, route in routes.items() if route is not None}


class SongWindowDataset(Dataset):
    """One augmented query window and several nearby reference windows per song."""

    def __init__(
        self,
        routes: dict[str, dict],
        augment: ContourAugment,
        seed: int,
        refs: int = 4,
        offset_s: float = 3.0,
        mix_prob: float = 0.5,
        query_s: tuple[float, float] = (3.0, 12.0),
        extra_s: tuple[float, float] = (-1.0, 3.0),
        stretch: tuple[float, float] = (0.8, 1.25),
    ) -> None:
        self.ids = sorted(routes)
        self.routes = routes
        self.augment = augment
        self.seed = seed
        self.refs = refs
        self.offset_s = offset_s
        self.mix_prob = mix_prob
        self.query_s = query_s
        self.extra_s = extra_s
        self.stretch = stretch
        self.epoch = EpochCounter()

    def __len__(self) -> int:
        return len(self.ids)

    def query_source(self, route: dict, rng: np.random.Generator) -> np.ndarray:
        use_alt = route["alt"] is not None and rng.random() < self.mix_prob
        return route["alt"] if use_alt else route["ref"]

    def reference(self, ref: np.ndarray, start: int, length: int, rng) -> np.ndarray:
        offset = int(round(rng.uniform(-self.offset_s, self.offset_s) / FRAME_S))
        begin = int(np.clip(start + offset, 0, max(len(ref) - 1, 0)))
        extra = int(round(rng.uniform(*self.extra_s) / FRAME_S))
        window = ref[begin : begin + max(length + extra, seconds_to_frames(1.0))]
        window = stretch_nearest(window, float(rng.uniform(*self.stretch)))
        trimmed = trim_unvoiced(window)
        return trimmed if len(trimmed) >= seconds_to_frames(MIN_QUERY_S) else window

    def query_window(self, route: dict, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
        """(query crop, start, length) with at least MIN_QUERY_S after trimming; the last try
        keeps the untrimmed crop so a nearly unvoiced window cannot break the augmentation."""
        for _attempt in range(QUERY_TRIES):
            length = seconds_to_frames(rng.uniform(*self.query_s))
            start = voiced_start(route["ref"], length, rng)
            crop = self.query_source(route, rng)[start : start + length]
            if len(trim_unvoiced(crop)) >= seconds_to_frames(MIN_QUERY_S):
                return trim_unvoiced(crop), start, length
        return crop, start, length

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 2])
        route = self.routes[self.ids[index]]
        query, start, length = self.query_window(route, rng)
        query = augment_contour(humanize(query, rng, FRAME_S), rng, self.augment, FRAME_S)
        refs = [self.reference(route["ref"], start, length, rng) for _ in range(self.refs)]
        return {
            "query": features_tensor(query),
            "refs": [features_tensor(r) for r in refs],
            "song_id": self.ids[index],
        }


def collate_song_windows(items: list[dict]) -> dict:
    query, query_valid = collate_contours([item["query"] for item in items])
    refs, refs_valid = collate_contours([ref for item in items for ref in item["refs"]])
    return {
        "query": query,
        "query_valid": query_valid,
        "refs": refs,
        "refs_valid": refs_valid,
        "refs_per_song": len(items[0]["refs"]),
        "song_id": [item["song_id"] for item in items],
    }
