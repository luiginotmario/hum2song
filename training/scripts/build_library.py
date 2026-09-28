"""Index songs: htdemucs vocals -> RMVPE -> 10 s / 5 s contour chunks -> pgvector (D-017).

Reads songs.jsonl from fetch_fma.py. Melody tracks are cached as .npy next to it, so a
re-chunk or a new checkpoint does not repeat separation. Songs already in the database
are skipped, and --shard/--shards split the list so several processes share one GPU.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hum2song.catalog.db import connect, ensure_schema, indexed_songs, insert_song
from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.library import decode_audio, embed_chunks, melody_track, song_chunks
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.melody import load_separator
from hum2song.contour.trackers import load_rmvpe
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
DEFAULT_RMVPE = "fig_contours/rmvpe.pt"
LOG_EVERY = 25
MAIN_THREADS = 4


class SongAudio(Dataset):
    """Decodes songs to mono 44.1 kHz in worker processes; cached tracks skip decoding."""

    def __init__(self, songs: list[dict], track_dir: Path) -> None:
        self.songs = songs
        self.track_dir = track_dir

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, index: int) -> dict:
        song = self.songs[index]
        if track_path(self.track_dir, song).exists():
            return {"index": index, "audio": None}
        try:
            return {"index": index, "audio": decode_audio(song["audio"])}
        except Exception as error:  # noqa: BLE001 - a broken file must not stop the run
            return {"index": index, "audio": None, "error": repr(error)}


def single_thread_worker(_worker_id: int) -> None:
    torch.set_num_threads(1)


def track_path(track_dir: Path, song: dict) -> Path:
    return track_dir / f"{song['song_id'].replace(':', '_')}.npy"


def song_track(item: dict, song: dict, models: dict, track_dir: Path, device):
    """Cached melody track, or a new one from the decoded audio (None if it failed)."""
    cached = track_path(track_dir, song)
    if cached.exists():
        return np.load(cached)
    if item["audio"] is None:
        LOGGER.warning("skip %s: %s", song["song_id"], item.get("error", "no audio"))
        return None
    track = melody_track(models["separator"], models["rmvpe"], item["audio"], device)
    np.save(cached, track)
    return track


def index_song(connection, song: dict, track: np.ndarray, models: dict, model_ver, device):
    chunks = song_chunks(track)
    embeddings = embed_chunks(models["encoder"], chunks, device)
    starts = [chunk.start_s for chunk in chunks]
    voiced = [chunk.voiced for chunk in chunks]
    fields = {k: v for k, v in song.items() if k != "audio"}
    insert_song(connection, fields, model_ver, starts, voiced, embeddings)
    return len(chunks)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the song library in pgvector")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--name", default="fma_full_3k")
    parser.add_argument("--db", required=True, help="postgresql:// URL")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    torch.set_num_threads(MAIN_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library_dir = args.data_root / "library" / args.name
    track_dir = library_dir / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.ckpt or args.data_root / DEFAULT_CKPT
    encoder, _config, _step = load_checkpoint(ckpt, device)
    models = {
        "separator": load_separator(device),
        "rmvpe": load_rmvpe(args.data_root / DEFAULT_RMVPE, device),
        "encoder": encoder,
    }
    model_ver = f"{ckpt.parent.name}/{ckpt.name}"
    connection = connect(args.db)
    ensure_schema(connection)
    done = indexed_songs(connection)
    songs = read_jsonl(library_dir / "songs.jsonl")[args.shard :: args.shards]
    songs = [s for s in songs if s["song_id"] not in done]
    LOGGER.info("shard %s/%s: %s songs to index", args.shard, args.shards, len(songs))
    loader = DataLoader(
        SongAudio(songs, track_dir),
        batch_size=None,
        num_workers=args.workers,
        prefetch_factor=2 if args.workers else None,
        worker_init_fn=single_thread_worker,
    )
    started, chunk_total = time.time(), 0
    for count, item in enumerate(loader, start=1):
        song = songs[int(item["index"])]
        audio = item["audio"]
        item["audio"] = None if audio is None else np.asarray(audio, dtype=np.float32)
        track = song_track(item, song, models, track_dir, device)
        if track is not None:
            chunk_total += index_song(connection, song, track, models, model_ver, device)
        if count % LOG_EVERY == 0 or count == len(songs):
            rate = count / max(time.time() - started, 1.0e-9)
            LOGGER.info(
                "%s / %s songs, %s chunks, %.2f songs/s", count, len(songs), chunk_total, rate
            )


if __name__ == "__main__":
    main()
