"""E1 (D-023): multi-window queries, time-consistent matching and key-invariant DTW reranking.

    python scripts/eval_rerank.py --out results/e1.json

No retraining. In the D-019 headline pool (full-song targets, chart previews, FMA):
  base   the API score: best 10 s chunk (5 s hop) per song
  seq    the query is cut into WIN_S windows every HOP_S; each song into the same windows;
         score = mean similarity along the best monotone path (local tempo about 0..2)
  dtw    minus the key-normalized, slope-constrained DTW error between the whole query and
         the reference span that path picked
The top TOP_K songs by `base` are reranked by a per-query z-scored sum of the three. Fusion
weights are chosen on CHAD **val** songs only, then applied unchanged to CHAD test songs,
MTG-QBH and MLEnd.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.library import chunk_contour
from hum2song.catalog.matching import (
    NO_MATCH,
    dtw_mae,
    frames,
    fuse,
    key_normalized,
    query_windows,
    sequence_path,
    windows,
)
from hum2song.catalog.probe import summarize_ranks
from hum2song.catalog.real_eval import song_scores
from hum2song.catalog.real_pool import (
    chad_split_targets,
    charts_pool,
    fma_rows,
    library_rows,
    query_contour_cache,
    song_contour,
)
from hum2song.catalog.targets import query_sets
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
LIBRARIES = ["previews_v1", "youtube_v1"]
WIN_S = 5.0
HOP_S = 1.0
MIN_VOICED = 0.25
TOP_K = 50
WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0)
SELECT_SET = "chad_val"
QUERY_SETS = {  # name: (query list in targets.query_sets, dataset prefix, CHAD split)
    "chad_val": ("chad_hum", "chad", "val"),
    "chad_test": ("chad_hum", "chad", "test"),
    "mtgqbh_sing": ("mtgqbh_sing", "mtgqbh", None),
    "mlend_hum": ("mlend_hum", "mlend", None),
}


def library(root: Path, model, device) -> dict:
    """Every pool song's contour, 10 s chunk embeddings and short-window embeddings."""
    rows, contours = [], []
    for row in library_rows(root, LIBRARIES) + fma_rows(root):
        contour = song_contour(row)
        if contour is not None and chunk_contour(contour):
            rows.append(row)
            contours.append(contour)
    chunk_owner, chunk_list, grids, window_list = [], [], [], []
    for index, contour in enumerate(contours):
        chunks = chunk_contour(contour)
        chunk_owner += [index] * len(chunks)
        chunk_list += [c.contour for c in chunks]
        grid = windows(contour, WIN_S, HOP_S, MIN_VOICED)
        rows_of = np.full(max(grid, default=-1) + 1, -1, dtype=np.int64)
        for position, key in enumerate(sorted(grid)):
            rows_of[key] = len(window_list) + position
        window_list += [grid[k] for k in sorted(grid)]
        grids.append(rows_of)
    LOGGER.info("%s songs, %s chunks, %s windows", len(rows), len(chunk_list), len(window_list))
    return {
        "rows": rows,
        "contours": contours,
        "chunk_owner": np.array(chunk_owner),
        "chunk_emb": embed_contours(model, chunk_list, device),
        "grids": grids,
        "window_emb": embed_contours(model, window_list, device),
    }


def base_scores(lib: dict, query_emb: np.ndarray, device) -> np.ndarray:
    return song_scores(
        torch.from_numpy(query_emb).to(device),
        torch.from_numpy(lib["chunk_emb"]).to(device),
        torch.from_numpy(lib["chunk_owner"]).long().to(device),
        len(lib["rows"]),
    )


def similarity_matrix(lib: dict, song: int, query_win_emb: np.ndarray) -> np.ndarray | None:
    rows_of = lib["grids"][song]
    if len(rows_of) == 0:
        return None
    present = rows_of >= 0
    matrix = np.full((len(query_win_emb), len(rows_of)), NO_MATCH, dtype=np.float32)
    matrix[:, present] = query_win_emb @ lib["window_emb"][rows_of[present]].T
    return matrix


def candidate_scores(lib: dict, song: int, query: np.ndarray, query_win_emb) -> tuple:
    """(sequence score, minus DTW error) of one candidate song for one query."""
    matrix = similarity_matrix(lib, song, query_win_emb)
    if matrix is None:
        return -np.inf, -np.inf
    score, first, last = sequence_path(matrix)
    span = lib["contours"][song][frames(first * HOP_S) : frames(last * HOP_S + WIN_S)]
    return score, -dtw_mae(key_normalized(span), key_normalized(query))


def rerank_inputs(lib, keep, contours, targets, query_emb, window_embs, device) -> list[dict]:
    """Per query: top-K candidates with base / seq / dtw scores and the baseline rank."""
    scores = base_scores(lib, query_emb, device)
    scores[:, ~keep] = -np.inf
    song_targets = np.array([r["target_id"] or "" for r in lib["rows"]], dtype=object)
    out = []
    for i, (contour, target) in enumerate(zip(contours, targets, strict=True)):
        order = np.argsort(-scores[i])[:TOP_K]
        pairs = [candidate_scores(lib, int(s), contour, window_embs[i]) for s in order]
        is_target = song_targets == target
        best = scores[i][is_target & keep].max() if (is_target & keep).any() else None
        out.append(
            {
                "base": scores[i][order],
                "seq": np.array([p[0] for p in pairs]),
                "dtw": np.array([p[1] for p in pairs]),
                "hit": song_targets[order] == target,
                "base_rank": None
                if best is None
                else int((scores[i][~is_target] > best).sum()) + 1,
            }
        )
    return out


def fused_rank(item: dict, weights: tuple) -> int | None:
    """Target rank after reranking the top-K; outside the top-K the baseline rank stays."""
    if item["base_rank"] is None:
        return None
    if not item["hit"].any():
        return item["base_rank"]
    fused = fuse([item["base"], item["seq"], item["dtw"]], list(weights))
    best = fused[item["hit"]].max()
    return int((fused[~item["hit"]] > best).sum()) + 1


def summarize(items: list[dict], weights: tuple) -> dict:
    return summarize_ranks([fused_rank(item, weights) for item in items])


def choose_weights(items: list[dict], grid: list[tuple]) -> tuple:
    """Best (top-1, then top-10) weights on the selection set."""
    scored = [(summarize(items, w), w) for w in grid]
    return max(scored, key=lambda sw: (sw[0]["top1"], sw[0]["top10"]))[1]


def weight_grid() -> dict[str, list[tuple]]:
    grid = [w for w in itertools.product((1.0,), WEIGHT_GRID, WEIGHT_GRID) if w[1] or w[2]]
    return {
        "base+seq": [w for w in grid if not w[2]],
        "base+dtw": [w for w in grid if not w[1]],
        "base+seq+dtw": [w for w in grid if w[1] and w[2]],
    }


def query_set(root: Path, lib: dict, name: str, cache: dict, model, device) -> tuple:
    """(keep mask over the library, query contours, targets) for one query set."""
    source, dataset, split = QUERY_SETS[name]
    groups = chad_split_targets(split) if split else None
    keep = np.array([charts_pool(r, dataset, groups) for r in lib["rows"]])
    targets_kept = {r["target_id"] for r, k in zip(lib["rows"], keep, strict=True) if k}
    queries = [q for q in query_sets(root)[source] if q["path"] in cache]
    queries = [q for q in queries if groups is None or q["target_id"] in groups]
    LOGGER.info(
        "%s: %s queries, %s songs, %s targets", name, len(queries), keep.sum(), len(targets_kept)
    )
    return keep, [cache[q["path"]] for q in queries], [q["target_id"] for q in queries]


def evaluate_set(root, lib, name, cache, model, device) -> list[dict]:
    keep, contours, targets = query_set(root, lib, name, cache, model, device)
    query_emb = embed_contours(model, contours, device)
    subwindows = [query_windows(c, WIN_S, HOP_S, MIN_VOICED) for c in contours]
    flat = embed_contours(model, [w for ws in subwindows for w in ws], device)
    bounds = np.cumsum([0] + [len(ws) for ws in subwindows])
    window_embs = [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    return rerank_inputs(lib, keep, contours, targets, query_emb, window_embs, device)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1: multi-window + DTW reranking")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _step = load_checkpoint(args.ckpt or args.data_root / DEFAULT_CKPT, device)
    lib = library(args.data_root, model, device)
    cache = query_contour_cache(args.data_root)
    items = {
        name: evaluate_set(args.data_root, lib, name, cache, model, device) for name in QUERY_SETS
    }
    chosen = {k: choose_weights(items[SELECT_SET], grid) for k, grid in weight_grid().items()}
    variants = {
        "base": (1.0, 0.0, 0.0),
        "seq_only": (0.0, 1.0, 0.0),
        "dtw_only": (0.0, 0.0, 1.0),
        **chosen,
    }
    report = {
        "top_k": TOP_K,
        "window_s": WIN_S,
        "hop_s": HOP_S,
        "weights_chosen_on": SELECT_SET,
        "weights": {k: list(v) for k, v in variants.items()},
        "results": {
            name: {k: summarize(set_items, w) for k, w in variants.items()}
            for name, set_items in items.items()
        },
    }
    for name, result in report["results"].items():
        LOGGER.info("%s: %s", name, {k: round(v["top1"], 3) for k, v in result.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
