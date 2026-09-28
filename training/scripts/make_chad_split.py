"""Fix the CHAD song split (D-019) before any full recording is downloaded or trained on.

    python scripts/make_chad_split.py

All CHAD groups that have hums on disk and an available original are split by song:
40% train, 10% val (checkpoint logging), 50% test.
"""

import sys
from pathlib import Path

from hum2song.catalog.targets import chad_originals
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.chad import SPLIT_MANIFEST, split_groups, write_split


def main(argv: list[str] | None = None) -> None:
    root = Path((argv or sys.argv[1:] or [DEFAULT_DATA_ROOT])[0])
    groups = [o["target_id"].split(":", 1)[1] for o in chad_originals(root)]
    write_split(SPLIT_MANIFEST, split_groups(groups))
    print(SPLIT_MANIFEST.read_text(encoding="utf-8")[:200])


if __name__ == "__main__":
    main()
