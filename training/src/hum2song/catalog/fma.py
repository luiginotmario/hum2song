"""FMA full-track subset: metadata, a seeded selection, and ranged downloads (D-017).

fma_full.zip is 879 GiB, but its server accepts HTTP ranges, so single tracks are read
straight out of the remote zip (remotezip) without fetching the archive.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FMA_FULL_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
METADATA_CSV = "raw/fma/fma_metadata/tracks.csv"
VOCAL_GENRES = (
    "Pop",
    "Rock",
    "Folk",
    "Hip-Hop",
    "Soul-RnB",
    "Country",
    "Blues",
    "International",
)
DURATION_S = (60.0, 420.0)
SELECTION_SEED = 20260928


def load_tracks(csv_path: Path) -> pd.DataFrame:
    """tracks.csv (two header rows) -> one row per track with the fields we store."""
    raw = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    return pd.DataFrame(
        {
            "title": raw[("track", "title")],
            "artist": raw[("artist", "name")],
            "genre": raw[("track", "genre_top")],
            "license": raw[("track", "license")],
            "duration_s": raw[("track", "duration")].astype(float),
        },
        index=raw.index,
    )


def select_tracks(tracks: pd.DataFrame, count: int, seed: int = SELECTION_SEED) -> pd.DataFrame:
    """`count` tracks drawn with `seed` among vocal-leaning genres and 1-7 minute durations."""
    eligible = tracks[
        tracks["genre"].isin(VOCAL_GENRES) & tracks["duration_s"].between(*DURATION_S)
    ].sort_index()
    order = np.random.default_rng(seed).permutation(len(eligible))[:count]
    return eligible.iloc[np.sort(order)]


def member_name(track_id: int) -> str:
    return f"fma_full/{track_id // 1000:03d}/{track_id:06d}.mp3"


def song_id(track_id: int) -> str:
    return f"fma:{track_id:06d}"


def song_rows(selected: pd.DataFrame, audio_dir: Path) -> list[dict]:
    """songs.jsonl rows: metadata plus the local audio path."""
    return [
        {
            "song_id": song_id(int(track_id)),
            "audio": str(audio_dir / f"{int(track_id):06d}.mp3"),
            "source": "fma_full",
            "source_tier": "P",
            "coverage": "full",
            "title": str(row.title),
            "artist": str(row.artist),
            "genre": str(row.genre),
            "license": str(row.license),
            "duration_s": float(row.duration_s),
        }
        for track_id, row in selected.iterrows()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def download_tracks(track_ids: list[int], audio_dir: Path, workers: int, url: str = FMA_FULL_URL):
    """Fetch each missing track from the remote zip; one RemoteZip per thread."""
    from remotezip import RemoteZip

    local = threading.local()
    audio_dir.mkdir(parents=True, exist_ok=True)

    def fetch(track_id: int) -> str:
        target = audio_dir / f"{track_id:06d}.mp3"
        if target.exists():
            return "cached"
        if not hasattr(local, "archive"):
            local.archive = RemoteZip(url)
        partial = target.with_suffix(".part")
        partial.write_bytes(local.archive.read(member_name(track_id)))
        partial.rename(target)
        return "downloaded"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, track_ids))
