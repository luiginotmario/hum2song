"""Hum query sets whose songs are commercial recordings, and the targets to look up (D-018).

mlend   MLEnd hums (8 movie songs); the reference fragment offsets come from the dataset's
        own brief (youtu.be links with ?t=).
mtgqbh  MTG-QBH sung queries (Salamon et al. 2013), song = class label.
chad    CHAD hummings subset (Amatov et al. 2023); each group's original is a YouTube
        video, `interval` is where the hummed fragment sits in it.
"""

import ast
import csv
import json
from pathlib import Path

MLEND_TARGETS = {
    "Potter": ("Hedwig's Theme", "John Williams", 0),
    "StarWars": ("The Imperial March", "John Williams", 9),
    "Panther": ("The Pink Panther Theme", "Henry Mancini", 10),
    "Rain": ("Singin' in the Rain", "Gene Kelly", 65),
    "Hakuna": ("Hakuna Matata", "Nathan Lane", 79),
    "Mamma": ("Mamma Mia", "ABBA", 50),
    "Showman": ("This Is Me", "Keala Settle", 115),
    "Frozen": ("Let It Go", "Idina Menzel", 126),
}
MTGQBH_QUERIES = "raw/mtgqbh/metadata/Queries.csv"
MTGQBH_CANONICALS = "raw/mtgqbh/metadata/Collection_Canonicals.csv"
CHAD_METADATA = "raw/chad/repo/metadata/dataset.csv"
CHAD_DIR = "raw/chad"
PAIRS = "pairs_real.jsonl"


def read_csv_rows(path: Path) -> list[dict]:
    """MTG-QBH CSVs use bare CR line endings and padded fields."""
    lines = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
    rows = csv.DictReader([line for line in lines if line.strip()])
    return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in rows]


def mlend_targets() -> list[dict]:
    return [
        {
            "target_id": f"mlend:{label}",
            "dataset": "mlend",
            "title": title,
            "artist": artist,
            "fragment_start_s": start,
        }
        for label, (title, artist, start) in MLEND_TARGETS.items()
    ]


def mtgqbh_targets(root: Path) -> list[dict]:
    """One target per queried class label, with the canonical version's artist."""
    canonical = {
        row["Class label"]: row
        for row in read_csv_rows(root / MTGQBH_CANONICALS)
        if row.get("Canonical") == "YES"
    }
    labels = sorted({row["Class label"] for row in read_csv_rows(root / MTGQBH_QUERIES)})
    return [
        {
            "target_id": f"mtgqbh:{label}",
            "dataset": "mtgqbh",
            "title": canonical.get(label, {}).get("Title", ""),
            "artist": canonical.get(label, {}).get("Artist", ""),
        }
        for label in labels
    ]


def parse_interval(text: str) -> tuple[float, float]:
    start, end = ast.literal_eval(text)
    return float(start), float(end)


def chad_originals(root: Path) -> list[dict]:
    """Groups with hummings on disk and an available original: youtube id and fragment."""
    on_disk = {p.name for p in (root / CHAD_DIR).iterdir() if p.is_dir() and len(p.name) == 16}
    with open(root / CHAD_METADATA, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["audio_type"] == "original"]
    originals = {}
    for row in rows:
        if row["group_id"] in on_disk and row["is_available"] == "True":
            originals.setdefault(row["group_id"], row)
    return [
        {
            "target_id": f"chad:{group}",
            "dataset": "chad",
            "youtube_id": row["youtube_id"],
            "fragment_s": parse_interval(row["interval"]),
        }
        for group, row in sorted(originals.items())
    ]


def pair_queries(root: Path, group: str, qtype: str) -> list[dict]:
    with open(root / PAIRS, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    return [
        {"path": str(root / r["query_path"]), "target_id": r["song_id"], "dataset": group}
        for r in records
        if r["group"] == group and r["qtype"] == qtype
    ]


def chad_queries(root: Path) -> list[dict]:
    return [
        {"path": str(wav), "target_id": f"chad:{wav.parts[-3]}", "dataset": "chad"}
        for wav in sorted((root / CHAD_DIR).glob("*/*/*.wav"))
    ]


def query_sets(root: Path) -> dict[str, list[dict]]:
    return {
        "chad_hum": chad_queries(root),
        "mtgqbh_sing": pair_queries(root, "mtgqbh", "sing"),
        "mlend_hum": pair_queries(root, "mlend", "hum"),
        "mlend_whistle": pair_queries(root, "mlend", "whistle"),
    }
