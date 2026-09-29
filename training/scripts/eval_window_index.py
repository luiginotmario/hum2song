"""Fidelity of the pgvector window index (D-025): approximate (HNSW) vs exact window votes.

    python scripts/eval_window_index.py --db postgresql://... --out results/d025_ann.json

On the served library (songs table), for real hum queries whose target is indexed:
  chunk   the API's current first stage (best 10 s chunk via the chunk HNSW index)
  ann     window votes from the window HNSW index (db.window_votes)
  exact   the same votes computed exhaustively over every stored window
Reports target recall@K of each first stage and the overlap of ann and exact top-K.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.db import best_per_song, connect, library_counts, nearest_chunks, window_votes
from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.matching import query_windows
from hum2song.catalog.real_eval import song_scores
from hum2song.catalog.real_pool import chad_split_targets, query_contour_cache
from hum2song.catalog.targets import query_sets
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
TARGET_LIBRARY = "youtube_v1"
WIN_S, HOP_S, MIN_VOICED = 5.0, 1.0, 0.25
DEPTHS = (10, 50, 100, 200)
QUERY_SETS = {
    "chad_val": ("chad_hum", "val"),
    "chad_test": ("chad_hum", "test"),
    "mtgqbh_sing": ("mtgqbh_sing", None),
}


def stored_windows(connection) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(song ids, owner index per window, window embeddings) of the whole window table."""
    rows = connection.execute("SELECT song_id, embedding FROM windows ORDER BY song_id").fetchall()
    songs = sorted({song for song, _ in rows})
    index = {song: i for i, song in enumerate(songs)}
    owners = np.array([index[song] for song, _ in rows])
    return songs, owners, np.stack([np.asarray(e, dtype=np.float32) for _, e in rows])


def exact_votes(query_win: np.ndarray, stored: tuple, device) -> list[str]:
    songs, owners, embeddings = stored
    best = song_scores(
        torch.from_numpy(query_win).to(device),
        torch.from_numpy(embeddings).to(device),
        torch.from_numpy(owners).long().to(device),
        len(songs),
    )
    return [songs[i] for i in np.argsort(-best.mean(axis=0))[: max(DEPTHS)]]


def rank_in(order: list[str], wanted: set[str]) -> int | None:
    return next((i + 1 for i, song in enumerate(order) if song in wanted), None)


def query_orders(connection, model, device, contour, stored) -> dict[str, list[str]]:
    embedding = embed_contours(model, [contour], device)[0]
    chunk = [row[0] for row in best_per_song(nearest_chunks(connection, embedding), max(DEPTHS))]
    wins = embed_contours(model, query_windows(contour, WIN_S, HOP_S, MIN_VOICED), device)
    ann = [song for song, _ in window_votes(connection, wins)][: max(DEPTHS)]
    return {"chunk": chunk, "ann": ann, "exact": exact_votes(wins, stored, device)}


def summarize(orders: list[dict], wanted: list[set]) -> dict:
    out = {}
    for stage in ("chunk", "ann", "exact"):
        ranks = [rank_in(o[stage], w) for o, w in zip(orders, wanted, strict=True)]
        out[stage] = {
            f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in DEPTHS
        }
    out["ann_exact_overlap"] = {
        f"@{k}": float(np.mean([len(set(o["ann"][:k]) & set(o["exact"][:k])) / k for o in orders]))
        for k in DEPTHS
    }
    out["queries"] = len(orders)
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Window-index fidelity check")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=600, help="per set, evenly spaced")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _step = load_checkpoint(args.data_root / DEFAULT_CKPT, device)
    connection = connect(args.db)
    stored = stored_windows(connection)
    indexed = set(stored[0])
    versions: dict[str, set] = {}
    for row in read_jsonl(args.data_root / "library" / TARGET_LIBRARY / "previews.jsonl"):
        if row["song_id"] in indexed:
            versions.setdefault(row["target_id"], set()).add(row["song_id"])
    cache = query_contour_cache(args.data_root)
    report = {"library": library_counts(connection), "window_songs": len(indexed), "sets": {}}
    for name, (source, split) in QUERY_SETS.items():
        groups = chad_split_targets(split) if split else None
        queries = [q for q in query_sets(args.data_root)[source] if q["path"] in cache]
        queries = [q for q in queries if q["target_id"] in versions]
        queries = [q for q in queries if groups is None or q["target_id"] in groups]
        step = max(1, len(queries) // args.max_queries)
        queries = queries[::step][: args.max_queries]
        orders = [
            query_orders(connection, model, device, cache[q["path"]], stored) for q in queries
        ]
        report["sets"][name] = summarize(orders, [versions[q["target_id"]] for q in queries])
        LOGGER.info("%s: %s", name, json.dumps(report["sets"][name]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
