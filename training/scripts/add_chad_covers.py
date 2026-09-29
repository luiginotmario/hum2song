"""Append one downloaded E2b batch (CHAD originals + covers) to library/chad_covers_v1 (D-028).

    python scripts/add_chad_covers.py --batch-dir /tmp/e2b_01

Segment downloads (<tag>.s<start>.<ext>) are first placed on one silent timeline per
recording (<tag>.wav), and id-named downloads are renamed to their tags (--id-list).
Rows point at the local audio files; extract_previews.py --name chad_covers_v1 then decodes
each file once, deletes it and caches the melody tracks. Recordings whose video title
contains an evaluation target's title are deleted here and never extracted.
"""

import argparse
import json
import sys
from pathlib import Path

from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.library import decode_audio
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.cover_pairs import (
    COVER_LIBRARY,
    cover_batch_rows,
    eval_targets,
    rename_id_files,
    stitch_sections,
)
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)


def known_songs(path: Path) -> set[str]:
    return {r["song_id"] for r in read_jsonl(path)} if path.exists() else set()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add an E2b batch to the cover library")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--id-list", type=Path, default=None, help="list mapping ids to tags")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    library = args.data_root / "library" / COVER_LIBRARY
    library.mkdir(parents=True, exist_ok=True)
    previews = library / "previews.jsonl"
    renamed = rename_id_files(args.batch_dir, args.id_list) if args.id_list else 0
    stitched = stitch_sections(args.batch_dir / "audio", decode_audio)
    LOGGER.info("renamed %s id-named files, stitched %s segment sets", renamed, stitched)
    batch = cover_batch_rows(args.batch_dir, eval_targets(args.data_root), known_songs(previews))
    with open(previews, "a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in batch["rows"])
    LOGGER.info("added %s recordings, dropped %s", len(batch["rows"]), batch["dropped"])


if __name__ == "__main__":
    main()
