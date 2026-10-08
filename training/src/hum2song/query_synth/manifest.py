"""Manifest rows for synthetic queries. Every row is labeled synthetic."""

import json
import zlib
from pathlib import Path

from hum2song.query_synth.render import ORIGIN_SYNTHETIC, PRESENTATION

SPLIT_MOD = 10
TEST_BUCKET = 0
VAL_BUCKET = 1
SPLIT_NAMES = {TEST_BUCKET: "test", VAL_BUCKET: "val"}


def split_name(song_id: str) -> str:
    """Stable 10/10/80 test/val/train split from the song id. Not a random draw per run."""
    bucket = zlib.crc32(song_id.encode()) % SPLIT_MOD
    return SPLIT_NAMES.get(bucket, "train")


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Generate it with training/scripts/generate_query_synth.py."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def train_rows(path: Path) -> list[dict]:
    """Train-split rows. Refuses a row that is not marked synthetic."""
    chosen = []
    for row in read_manifest(path):
        require_synthetic(row)
        if row["split"] == "train":
            chosen.append(row)
    if not chosen:
        raise FileNotFoundError(f"{path} has no synthetic train rows")
    return chosen


def require_synthetic(row: dict) -> None:
    if row.get("origin") != ORIGIN_SYNTHETIC:
        raise ValueError(f"{row.get('query_id')} is not marked {ORIGIN_SYNTHETIC}")
    if row.get("presentation") != PRESENTATION:
        raise ValueError(f"{row.get('query_id')} is missing the synthetic presentation label")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["query_id"])
    lines = [json.dumps(row, sort_keys=True) for row in ordered]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
