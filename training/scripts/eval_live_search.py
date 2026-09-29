"""Real hums through the live search code path (D-027): windows mode vs chunks mode.

    python scripts/eval_live_search.py --db postgresql://... --out results/d027_live.json

Uses the cached query contours (no RMVPE) and the served library, calling exactly what
POST /search calls after the contour is extracted. Reports top-1/5/10 per mode and the
mean search time. Queries: CHAD val and test songs and MTG-QBH, whose targets are indexed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.db import connect, library_counts
from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.probe import summarize_ranks
from hum2song.catalog.real_pool import chad_split_targets, query_contour_cache
from hum2song.catalog.search import QueryEncoder, chunk_results, window_results
from hum2song.catalog.targets import query_sets
from hum2song.catalog.window_search import SongCache
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.features import trim_unvoiced
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
DEFAULT_RMVPE = "fig_contours/rmvpe.pt"
TARGET_LIBRARY = "youtube_v1"
TOP_K = 10
QUERY_SETS = {
    "chad_val": ("chad_hum", "val"),
    "chad_test": ("chad_hum", "test"),
    "mtgqbh_sing": ("mtgqbh_sing", None),
}


def rank_in(results: list[dict], wanted: set[str]) -> int | None:
    return next((i + 1 for i, r in enumerate(results) if r["song_id"] in wanted), None)


def timed(function, *args) -> tuple[list, float]:
    started = time.perf_counter()
    results = function(*args)
    return results, time.perf_counter() - started


def run_set(encoder, connection, cache, contours: list, wanted: list[set]) -> dict:
    ranks = {"windows": [], "chunks": []}
    seconds = {"windows": [], "chunks": []}
    for contour, songs in zip(contours, wanted, strict=True):
        found, spent = timed(window_results, encoder, connection, cache, contour, TOP_K)
        ranks["windows"].append(rank_in(found, songs))
        seconds["windows"].append(spent)
        found, spent = timed(chunk_results, encoder, connection, contour, TOP_K)
        ranks["chunks"].append(rank_in(found, songs))
        seconds["chunks"].append(spent)
    return {
        mode: {**summarize_ranks(ranks[mode]), "mean_s": float(np.mean(seconds[mode]))}
        for mode in ranks
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live search path on real hums")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--db", required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=300, help="per set, evenly spaced")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or args.data_root / DEFAULT_CKPT
    encoder = QueryEncoder(ckpt, args.data_root / DEFAULT_RMVPE, device)
    connection = connect(args.db)
    indexed = {row[0] for row in connection.execute("SELECT song_id FROM songs").fetchall()}
    versions: dict[str, set] = {}
    for row in read_jsonl(args.data_root / "library" / TARGET_LIBRARY / "previews.jsonl"):
        if row["song_id"] in indexed:
            versions.setdefault(row["target_id"], set()).add(row["song_id"])
    cache_contours = query_contour_cache(args.data_root)
    song_cache = SongCache()
    report = {"library": library_counts(connection), "model": encoder.model_ver, "sets": {}}
    for name, (source, split) in QUERY_SETS.items():
        groups = chad_split_targets(split) if split else None
        queries = [q for q in query_sets(args.data_root)[source] if q["path"] in cache_contours]
        queries = [q for q in queries if q["target_id"] in versions]
        queries = [q for q in queries if groups is None or q["target_id"] in groups]
        queries = queries[:: max(1, len(queries) // args.max_queries)][: args.max_queries]
        contours = [trim_unvoiced(cache_contours[q["path"]]) for q in queries]
        wanted = [versions[q["target_id"]] for q in queries]
        report["sets"][name] = run_set(encoder, connection, song_cache, contours, wanted)
        LOGGER.info("%s: %s", name, json.dumps(report["sets"][name]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
