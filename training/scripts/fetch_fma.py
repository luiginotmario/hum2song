"""Select and download an FMA full-track subset; write songs.jsonl (D-017, D-033)."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from hum2song.catalog.fma import (
    METADATA_CSV,
    SELECTION_SEED,
    VOCAL_GENRES,
    download_tracks,
    load_tracks,
    read_jsonl,
    select_tracks,
    song_rows,
    write_jsonl,
)
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_genres(text: str) -> tuple[str, ...]:
    if not text.strip():
        return VOCAL_GENRES
    return tuple(part.strip() for part in text.split(",") if part.strip())


def exclude_from_libraries(root: Path, names: list[str]) -> set[int]:
    """Track ids already listed in other FMA songs.jsonl libraries."""
    ids: set[int] = set()
    for name in names:
        path = root / "library" / name / "songs.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            song = row["song_id"]
            if song.startswith("fma:"):
                ids.add(int(song.split(":", 1)[1]))
    return ids


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMA full-track subset for the song library")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--name", default="fma_full_3k")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--genres",
        default="",
        help="comma list (default: D-017 vocal genres)",
    )
    parser.add_argument(
        "--exclude-libraries",
        nargs="*",
        default=[],
        help="skip track ids already in these library/*/songs.jsonl files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    genres = parse_genres(args.genres)
    exclude = exclude_from_libraries(args.data_root, args.exclude_libraries)
    selected = select_tracks(
        load_tracks(args.data_root / METADATA_CSV),
        args.count,
        args.seed,
        genres=genres,
        exclude_ids=exclude,
    )
    audio_dir = args.data_root / "raw" / "fma" / "full_audio"
    library_dir = args.data_root / "library" / args.name
    library_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(library_dir / "songs.jsonl", song_rows(selected, audio_dir))
    LOGGER.info(
        "selected %s tracks (excluded %s): %s",
        len(selected),
        len(exclude),
        dict(Counter(selected["genre"])),
    )
    outcome = download_tracks([int(i) for i in selected.index], audio_dir, args.workers)
    LOGGER.info("download: %s", dict(Counter(outcome)))


if __name__ == "__main__":
    main()
