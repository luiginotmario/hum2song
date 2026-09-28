"""Select and download an FMA full-track subset; write songs.jsonl (docs/DECISIONS.md D-017)."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from hum2song.catalog.fma import (
    METADATA_CSV,
    SELECTION_SEED,
    download_tracks,
    load_tracks,
    select_tracks,
    song_rows,
    write_jsonl,
)
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMA full-track subset for the song library")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--name", default="fma_full_3k")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    selected = select_tracks(load_tracks(args.data_root / METADATA_CSV), args.count, args.seed)
    audio_dir = args.data_root / "raw" / "fma" / "full_audio"
    library_dir = args.data_root / "library" / args.name
    library_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(library_dir / "songs.jsonl", song_rows(selected, audio_dir))
    LOGGER.info("selected %s tracks: %s", len(selected), dict(Counter(selected["genre"])))
    outcome = download_tracks([int(i) for i in selected.index], audio_dir, args.workers)
    LOGGER.info("download: %s", dict(Counter(outcome)))


if __name__ == "__main__":
    main()
