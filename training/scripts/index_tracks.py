"""Index cached melody tracks (tracks/<song_id>.npz) into pgvector (D-019, D-020).

    python scripts/index_tracks.py --names youtube_v1 youtube_charts_v1 --db postgresql://...

The vocal track is used when it gives voiced chunks, otherwise the full-mix track
(instrumental and house songs, D-020). Songs already in the database are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.db import connect, ensure_schema, indexed_songs, insert_song
from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.library import embed_chunks
from hum2song.catalog.real_eval import pick_track
from hum2song.catalog.youtube import db_song
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
METHOD = "vocals_or_mix"
TRACK_HOP_S = 0.01


def track_file(root: Path, name: str, song_id: str) -> Path:
    return root / "library" / name / "tracks" / f"{song_id.replace(':', '_')}.npz"


def library_songs(root: Path, names: list[str]) -> list[tuple[str, dict]]:
    """(library, row) for every row whose track exists, first occurrence of a song only."""
    seen, songs = set(), []
    for name in names:
        for row in read_jsonl(root / "library" / name / "previews.jsonl"):
            if row["song_id"] in seen or not track_file(root, name, row["song_id"]).exists():
                continue
            seen.add(row["song_id"])
            songs.append((name, row))
    return songs


def index_one(connection, root: Path, name: str, row: dict, encoder, model_ver, device) -> dict:
    with np.load(track_file(root, name, row["song_id"])) as data:
        tracks = dict(data)
    kind, chunks = pick_track(tracks, METHOD)
    embeddings = embed_chunks(encoder, chunks, device)
    song = db_song(row, round(tracks["mix"].shape[1] * TRACK_HOP_S, 2))
    starts, voiced = [c.start_s for c in chunks], [c.voiced for c in chunks]
    insert_song(connection, song, model_ver, starts, voiced, embeddings)
    return {
        "song_id": row["song_id"],
        "track": kind if chunks else "no_chunk",
        "chunks": len(chunks),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index cached melody tracks into pgvector")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--names", nargs="+", default=["youtube_v1", "youtube_charts_v1"])
    parser.add_argument("--db", required=True, help="postgresql:// URL")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="re-embed songs already in the DB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or args.data_root / DEFAULT_CKPT
    encoder, _config, _step = load_checkpoint(ckpt, device)
    model_ver = f"{ckpt.parent.name}/{ckpt.name}"
    connection = connect(args.db)
    ensure_schema(connection)
    done = set() if args.force else indexed_songs(connection)
    songs = [
        (n, r) for n, r in library_songs(args.data_root, args.names) if r["song_id"] not in done
    ]
    results = [
        index_one(connection, args.data_root, n, r, encoder, model_ver, device) for n, r in songs
    ]
    counts = {}
    for result in results:
        counts[result["track"]] = counts.get(result["track"], 0) + 1
    LOGGER.info("indexed %s songs: %s", len(results), json.dumps(counts))


if __name__ == "__main__":
    main()
