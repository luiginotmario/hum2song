"""The real-hum evaluation pool shared by eval_previews.py, the E0 breakdown and E1 rerank.

Rows come from library/<name>/previews.jsonl (preview and full-song targets and
distractors) plus the FMA library; each row carries its cached melody-track file.
"""

from pathlib import Path

import numpy as np

from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.real_eval import pick_track
from hum2song.contour import chad
from hum2song.contour.features import rmvpe_contour

FMA_LIBRARY = "fma_full_3k"
FMA_LIBRARIES = ("fma_full_3k", "fma_full_extra_3k", "fma_electronic_1k")
QUERY_CACHE = "library/query_contours.npz"
PREVIEW_SOURCES = ("itunes_preview", "deezer_preview")
FULL_SOURCE = "youtube_full"


def library_rows(root: Path, names: list[str]) -> list[dict]:
    """previews.jsonl rows of each library, tagged with their track file."""
    rows = []
    for name in names:
        tracks = root / "library" / name / "tracks"
        for row in read_jsonl(root / "library" / name / "previews.jsonl"):
            rows.append({**row, "_track": str(tracks / f"{row['song_id'].replace(':', '_')}.npz")})
    return rows


def fma_rows(root: Path) -> list[dict]:
    """All open-licence FMA library folders (D-017 base + D-033 extras)."""
    rows = []
    seen: set[str] = set()
    for name in FMA_LIBRARIES:
        songs_path = root / "library" / name / "songs.jsonl"
        if not songs_path.exists():
            continue
        tracks = root / "library" / name / "tracks"
        for song in read_jsonl(songs_path):
            if song["song_id"] in seen:
                continue
            seen.add(song["song_id"])
            rows.append(
                {
                    "song_id": song["song_id"],
                    "role": "fma",
                    "target_id": None,
                    "source": "fma_full",
                    "genre": song.get("genre"),
                    "_track": str(tracks / f"{song['song_id'].replace(':', '_')}.npy"),
                }
            )
    return rows


def load_tracks(path: str) -> dict | None:
    """npz with vocals and mix; FMA .npy caches hold the vocal track only (used for both)."""
    if not Path(path).exists():
        return None
    if path.endswith(".npy"):
        vocals = np.load(path)
        return {"vocals": vocals, "mix": vocals}
    with np.load(path) as tracks:
        return dict(tracks)


def song_contour(row: dict, method: str = "vocals_or_mix") -> np.ndarray | None:
    """The full-song contour of the track the library would index (None if nothing usable)."""
    tracks = load_tracks(row["_track"])
    if tracks is None:
        return None
    kind, chunks = pick_track(tracks, method)
    return rmvpe_contour(tracks[kind]) if chunks else None


def charts_pool(row: dict, dataset: str, split_groups: set[str] | None) -> bool:
    """D-019 headline pool: full-song targets of `dataset`, chart previews and FMA songs.
    For CHAD, only targets of the given song split (e.g. test) are kept."""
    if row["role"] == "fma":
        return True
    if row["role"] == "distractor":
        return row["source"] in PREVIEW_SOURCES
    in_split = split_groups is None or row["target_id"] in split_groups
    return row["source"] == FULL_SOURCE and row["target_id"].startswith(dataset + ":") and in_split


def chad_split_targets(split_name: str) -> set[str]:
    return {f"chad:{g}" for g, s in chad.read_split().items() if s == split_name}


def query_contour_cache(root: Path) -> dict:
    with np.load(root / QUERY_CACHE, allow_pickle=True) as cache:
        return dict(cache)
