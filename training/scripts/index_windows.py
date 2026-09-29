"""Index 5 s windows of every indexed song into pgvector for the first stage (D-025).

    python scripts/index_windows.py --db postgresql://...

Windows come from the same melody track the chunk index used (vocals, or the mix for
instrumental songs), cut exactly as in scripts/eval_rerank.py. Songs that already have
windows are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.db import (
    connect,
    ensure_schema,
    insert_windows,
    library_counts,
    searchable_songs,
    windowed_songs,
)
from hum2song.catalog.library import chunk_contour
from hum2song.catalog.real_eval import pick_track
from hum2song.catalog.real_pool import FMA_LIBRARY, load_tracks
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.features import rmvpe_contour
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
LIBRARIES = ("youtube_v1", "youtube_charts_v1", "previews_v1", FMA_LIBRARY)
WIN_S = 5.0
HOP_S = 1.0
MIN_VOICED = 0.25
METHOD = "vocals_or_mix"


def track_files(root: Path) -> dict[str, Path]:
    """song_id -> cached track file, first library that has it."""
    found: dict[str, Path] = {}
    for name in LIBRARIES:
        for path in sorted((root / "library" / name / "tracks").glob("*.np[yz]")):
            found.setdefault(path.stem.replace("_", ":", 1), path)
    return found


def song_windows(path: Path) -> list:
    tracks = load_tracks(str(path))
    kind, chunks = pick_track(tracks, METHOD)
    if not chunks:
        return []
    return chunk_contour(rmvpe_contour(tracks[kind]), WIN_S, HOP_S, MIN_VOICED)


def index_song(connection, song_id: str, path: Path, model, device) -> int:
    found = song_windows(path)
    if not found:
        return 0
    embeddings = embed_contours(model, [w.contour for w in found], device)
    insert_windows(connection, song_id, [w.start_s for w in found], embeddings)
    return len(found)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index 5 s windows into pgvector")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--db", required=True, help="postgresql:// URL")
    parser.add_argument("--ckpt", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _step = load_checkpoint(args.ckpt or args.data_root / DEFAULT_CKPT, device)
    connection = connect(args.db)
    ensure_schema(connection)
    files = track_files(args.data_root)
    done = windowed_songs(connection)
    todo = [s for s in searchable_songs(connection) if s not in done and s in files]
    counts = [index_song(connection, s, files[s], model, device) for s in todo]
    windows = connection.execute("SELECT count(*) FROM windows").fetchone()[0]
    LOGGER.info(
        "windowed %s songs (%s windows); table: %s windows; %s",
        len(todo),
        int(np.sum(counts)) if counts else 0,
        windows,
        json.dumps(library_counts(connection)),
    )


if __name__ == "__main__":
    main()
