"""Write the fixed MLEnd performer split (docs/DECISIONS.md D-015).

Run once; the manifest is committed and every later script reads it. Refuses to overwrite
an existing manifest unless --force is given, so the split cannot drift silently.
"""

import argparse
import sys
from pathlib import Path

from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.mlend import SPLIT_MANIFEST, mlend_clips, split_performers, write_split
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed train/val/test split of MLEnd performers")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; the split is fixed (use --force to rewrite)")
    clips = mlend_clips(args.data_root)
    assignment = split_performers([clip.person for clip in clips])
    write_split(args.out, assignment, clips)
    LOGGER.info("wrote %s performers to %s", len(assignment), args.out)


if __name__ == "__main__":
    main()
