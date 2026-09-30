"""Index 5 s windows and the melody contour of every indexed song (D-025, D-027).

    python scripts/index_windows.py --db postgresql://...

Windows come from the same melody track the chunk index used (vocals, or the mix for
instrumental songs), cut exactly as in scripts/eval_rerank.py; the contour of that track is
stored for the re-ranking stage of live search. Songs that already have both are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.db import (
    connect,
    contoured_songs,
    ensure_schema,
    insert_contour,
    insert_windows,
    library_counts,
    searchable_songs,
    windowed_songs,
)
from hum2song.catalog.library import chunk_contour
from hum2song.catalog.real_eval import pick_track
from hum2song.catalog.real_pool import FMA_LIBRARIES, load_tracks
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.features import rmvpe_contour
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
LIBRARIES = ("youtube_v1", "youtube_charts_v1", "previews_v1", *FMA_LIBRARIES)
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


def indexed_contour(path: Path) -> np.ndarray | None:
    """Contour of the track the chunk index used, None when it has no voiced chunk."""
    tracks = load_tracks(str(path))
    kind, chunks = pick_track(tracks, METHOD)
    return rmvpe_contour(tracks[kind]) if chunks else None


def index_song(connection, song_id: str, path: Path, model, device, windowed: bool) -> int:
    contour = indexed_contour(path)
    if contour is None:
        return 0
    insert_contour(connection, song_id, contour)
    if windowed:
        return 0
    found = chunk_contour(contour, WIN_S, HOP_S, MIN_VOICED)
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
    parser.add_argument("--force", action="store_true", help="re-embed songs already in the DB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _step = load_checkpoint(args.ckpt or args.data_root / DEFAULT_CKPT, device)
    connection = connect(args.db)
    ensure_schema(connection)
    files = track_files(args.data_root)
    windowed = set() if args.force else windowed_songs(connection)
    contoured = set() if args.force else contoured_songs(connection)
    todo = [
        s
        for s in searchable_songs(connection)
        if s in files and (s not in windowed or s not in contoured)
    ]
    counts = [index_song(connection, s, files[s], model, device, s in windowed) for s in todo]
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
