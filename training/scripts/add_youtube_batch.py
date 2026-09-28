"""Append one downloaded YouTube batch to a library's previews.jsonl (D-020).

    python scripts/add_youtube_batch.py --batch-dir /tmp/ytb --tag-meta /tmp/ytb/tag_meta.json

Rows point at the local audio files; extract_previews.py --name <library> then decodes
each file once, deletes it and caches the melody tracks.
"""

import argparse
import json
import sys
from pathlib import Path

from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.youtube import batch_rows
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
TARGET_LIBRARY = "previews_v1"
YOUTUBE_LIBRARIES = ("youtube_v1", "youtube_charts_v1")


def known_videos(root: Path) -> set[str]:
    known = set()
    for name in YOUTUBE_LIBRARIES:
        path = root / "library" / name / "previews.jsonl"
        known |= {r["youtube_id"] for r in read_jsonl(path)} if path.exists() else set()
    return known


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add a YouTube batch to a library")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--name", default="youtube_charts_v1")
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--tag-meta", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    library = args.data_root / "library" / args.name
    library.mkdir(parents=True, exist_ok=True)
    targets = read_jsonl(args.data_root / "library" / TARGET_LIBRARY / "targets.jsonl")
    tag_meta = json.loads(args.tag_meta.read_text(encoding="utf-8"))
    batch = batch_rows(args.batch_dir, tag_meta, targets, known_videos(args.data_root))
    with open(library / "previews.jsonl", "a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in batch["rows"])
    LOGGER.info("added %s songs, dropped %s", len(batch["rows"]), batch["dropped"])


if __name__ == "__main__":
    main()
