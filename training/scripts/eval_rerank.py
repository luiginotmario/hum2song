"""E1 / D-023..D-025: multi-window queries, time-consistent matching, DTW re-ranking and a
window-level first stage.

    python scripts/eval_rerank.py --out results/e1.json [--octave-window 3] [--first-stage]

No retraining. In the D-019 headline pool (full-song targets, chart previews, FMA), every
song is cut into 10 s chunks (the API index) and WIN_S windows every HOP_S. Scores are
described in catalog/rerank.py. Without --first-stage the candidates are the top TOP_K songs
by `base` (D-023). With it, the first stage (base, window vote or their union) and its depth
are chosen too (D-025). Everything is chosen on CHAD **val** songs only, then applied
unchanged to CHAD test songs, MTG-QBH and MLEnd. --octave-window octave-corrects the
song-side contours (D-024).
"""

import argparse
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
    key_normalized,
    query_windows,
    sequence_path,
    windows,
)
from hum2song.catalog.octave import octave_correct
from hum2song.catalog.real_eval import song_scores
from hum2song.catalog.real_pool import (
    chad_split_targets,
    charts_pool,
    fma_rows,
    library_rows,
    query_contour_cache,
    song_contour,
)
from hum2song.catalog.rerank import (
    K_GRID,
    SOURCES,
    choose,
    rank_all,
    recall_at,
    summarize,
    target_rank,
    weight_grid,
)
from hum2song.catalog.targets import query_sets
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.song_pairs import keep_fma
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
LIBRARIES = ["previews_v1", "youtube_v1"]
WIN_S = 5.0
HOP_S = 1.0
MIN_VOICED = 0.25
TOP_K = 50
SELECT_SET = "chad_val"
QUERY_SETS = {  # name: (query list in targets.query_sets, dataset prefix, CHAD split)
    "chad_val": ("chad_hum", "chad", "val"),
    "chad_test": ("chad_hum", "chad", "test"),
    "mtgqbh_sing": ("mtgqbh_sing", "mtgqbh", None),
    "mlend_hum": ("mlend_hum", "mlend", None),
}


def contour_fix(window_s: float, limit: float):
    """Octave correction with the given window and limit (D-024), or the identity when 0."""
    if not window_s:
        return lambda contour: contour
    return lambda contour: octave_correct(contour, window_s, limit)


def library(root: Path, model, device, fix) -> dict:
    """Every pool song's contour, 10 s chunk embeddings and short-window embeddings."""
    rows, contours = [], []
    for row in library_rows(root, LIBRARIES) + fma_rows(root):
        contour = song_contour(row)
        contour = None if contour is None else fix(contour)
        if contour is not None and chunk_contour(contour):
            rows.append(row)
            contours.append(contour)
    chunk_owner, chunk_list, grids, window_owner, window_list = [], [], [], [], []
    for index, contour in enumerate(contours):
        chunks = chunk_contour(contour)
        chunk_owner += [index] * len(chunks)
        chunk_list += [c.contour for c in chunks]
        grid = windows(contour, WIN_S, HOP_S, MIN_VOICED)
        rows_of = np.full(max(grid, default=-1) + 1, -1, dtype=np.int64)
        for position, key in enumerate(sorted(grid)):
            rows_of[key] = len(window_list) + position
        window_list += [grid[k] for k in sorted(grid)]
        window_owner += [index] * len(grid)
        grids.append(rows_of)
    LOGGER.info("%s songs, %s chunks, %s windows", len(rows), len(chunk_list), len(window_list))
    return {
        "rows": rows,
        "contours": contours,
        "chunk_owner": np.array(chunk_owner),
        "chunk_emb": embed_contours(model, chunk_list, device),
        "grids": grids,
        "window_owner": np.array(window_owner),
        "window_emb": embed_contours(model, window_list, device),
    }


def best_per_song(lib: dict, kind: str, queries: np.ndarray, device) -> np.ndarray:
    """(queries, songs) best cosine similarity over each song's chunks or windows."""
    return song_scores(
        torch.from_numpy(queries).to(device),
        torch.from_numpy(lib[f"{kind}_emb"]).to(device),
        torch.from_numpy(lib[f"{kind}_owner"]).long().to(device),
        len(lib["rows"]),
    )


def window_votes(lib: dict, window_embs: list[np.ndarray], device) -> np.ndarray:
    """(Q, songs): mean over each query's windows of the song's best window similarity."""
    return np.stack([best_per_song(lib, "window", w, device).mean(axis=0) for w in window_embs])


def similarity_matrix(lib: dict, song: int, query_win_emb: np.ndarray) -> np.ndarray | None:
    rows_of = lib["grids"][song]
    if len(rows_of) == 0:
        return None
    present = rows_of >= 0
    matrix = np.full((len(query_win_emb), len(rows_of)), NO_MATCH, dtype=np.float32)
    matrix[:, present] = query_win_emb @ lib["window_emb"][rows_of[present]].T
    return matrix


def candidate_scores(lib: dict, song: int, query: np.ndarray, query_win_emb, fold) -> tuple:
    """(sequence score, minus DTW error) of one candidate song for one query."""
    matrix = similarity_matrix(lib, song, query_win_emb)
    if matrix is None:
        return -np.inf, -np.inf
    score, first, last = sequence_path(matrix)
    span = lib["contours"][song][frames(first * HOP_S) : frames(last * HOP_S + WIN_S)]
    return score, -dtw_mae(key_normalized(span), key_normalized(query), fold)


def query_item(lib, keep, is_target, base, window, contour, win_emb, pool_k, fold) -> dict | None:
    """Candidate pool (top pool_k by base or by window vote) with all scores of one query."""
    fallback = {
        "base": target_rank(base, is_target, keep),
        "window": target_rank(window, is_target, keep),
    }
    if fallback["base"] is None:
        return None
    fallback["union"] = min(fallback["base"], fallback["window"])
    base, window = np.where(keep, base, -np.inf), np.where(keep, window, -np.inf)
    base_rank, window_rank = rank_all(base), rank_all(window)
    pool = np.flatnonzero((base_rank <= pool_k) | (window_rank <= pool_k))
    pairs = [candidate_scores(lib, int(s), contour, win_emb, fold) for s in pool]
    return {
        "base": base[pool],
        "window": window[pool],
        "seq": np.array([p[0] for p in pairs]),
        "dtw": np.array([p[1] for p in pairs]),
        "hit": is_target[pool],
        "base_rank": base_rank[pool],
        "window_rank": window_rank[pool],
        "fallback": fallback,
    }


def pooled(row: dict, dataset: str, groups, drop_fma: str) -> bool:
    """Headline pool membership, minus FMA songs of the parity used in E2a training."""
    trained = bool(drop_fma) and row["role"] == "fma" and keep_fma(row["song_id"], drop_fma)
    return charts_pool(row, dataset, groups) and not trained


def query_set(root: Path, lib: dict, name: str, cache: dict, fix, drop_fma: str) -> tuple:
    """(keep mask over the library, query contours, targets) for one query set."""
    source, dataset, split = QUERY_SETS[name]
    groups = chad_split_targets(split) if split else None
    keep = np.array([pooled(r, dataset, groups, drop_fma) for r in lib["rows"]])
    targets_kept = {r["target_id"] for r, k in zip(lib["rows"], keep, strict=True) if k}
    queries = [q for q in query_sets(root)[source] if q["path"] in cache]
    queries = [q for q in queries if groups is None or q["target_id"] in groups]
    LOGGER.info(
        "%s: %s queries, %s songs, %s targets", name, len(queries), keep.sum(), len(targets_kept)
    )
    return keep, [fix(cache[q["path"]]) for q in queries], [q["target_id"] for q in queries]


def evaluate_set(root, lib, name, cache, model, device, args) -> list:
    query_fix = contour_fix(args.octave_window if args.fix_queries else 0.0, args.octave_limit)
    keep, contours, targets = query_set(root, lib, name, cache, query_fix, args.drop_fma)
    base = best_per_song(lib, "chunk", embed_contours(model, contours, device), device)
    subwindows = [query_windows(c, WIN_S, HOP_S, MIN_VOICED) for c in contours]
    flat = embed_contours(model, [w for ws in subwindows for w in ws], device)
    bounds = np.cumsum([0] + [len(ws) for ws in subwindows])
    win_embs = [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    window = window_votes(lib, win_embs, device)
    song_targets = np.array([r["target_id"] or "" for r in lib["rows"]], dtype=object)
    pool_k = max(K_GRID) if args.first_stage else TOP_K
    return [
        query_item(
            lib, keep, song_targets == t, base[i], window[i], c, win_embs[i], pool_k, args.dtw_fold
        )
        for i, (c, t) in enumerate(zip(contours, targets, strict=True))
    ]


def configurations(first_stage: bool) -> dict[str, list[tuple]]:
    """Named families of (weights, source, k) to choose from on CHAD val."""
    e1 = [(w, "base", TOP_K) for w in weight_grid(allow_window=False) if w[0]]
    families = {"base+seq+dtw (D-023)": e1}
    if first_stage:
        full = weight_grid()
        families["first stage + rerank"] = [
            (w, s, k) for w in full for s in SOURCES for k in K_GRID
        ]
    return families


def report_for(items: dict, args) -> dict:
    fixed = {
        "base": ((1.0, 0.0, 0.0, 0.0), "base", TOP_K),
        "window_vote": ((0.0, 1.0, 0.0, 0.0), "window", TOP_K),
    }
    chosen = {k: choose(items[SELECT_SET], v) for k, v in configurations(args.first_stage).items()}
    variants = {**fixed, **chosen}
    depths = K_GRID if args.first_stage else (TOP_K,)
    return {
        "window_s": WIN_S,
        "hop_s": HOP_S,
        "octave_window_s": args.octave_window,
        "octave_limit_st": args.octave_limit,
        "fix_queries": args.fix_queries,
        "dtw_fold": args.dtw_fold,
        "dropped_fma_parity": args.drop_fma,
        "chosen_on": SELECT_SET,
        "variants": {
            k: {"weights": list(w), "source": s, "k": d} for k, (w, s, d) in variants.items()
        },
        "results": {
            name: {k: summarize(set_items, *v) for k, v in variants.items()}
            for name, set_items in items.items()
        },
        "candidate_recall": {
            name: {f"{s}@{k}": recall_at(set_items, s, k) for s in SOURCES for k in depths}
            for name, set_items in items.items()
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-window re-ranking and first stage")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", default=list(QUERY_SETS), choices=list(QUERY_SETS))
    parser.add_argument("--octave-window", type=float, default=0.0, help="song-side, s; 0 off")
    parser.add_argument("--octave-limit", type=float, default=9.0, help="semitones")
    parser.add_argument("--fix-queries", action="store_true", help="octave-correct queries too")
    parser.add_argument("--dtw-fold", action="store_true", help="octave-folded DTW cost")
    parser.add_argument(
        "--drop-fma", default="", choices=["", "even", "odd"], help="FMA parity to drop"
    )
    parser.add_argument("--first-stage", action="store_true", help="also choose a first stage")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _step = load_checkpoint(args.ckpt or args.data_root / DEFAULT_CKPT, device)
    fix = contour_fix(args.octave_window, args.octave_limit)
    lib = library(args.data_root, model, device, fix)
    cache = query_contour_cache(args.data_root)
    names = [SELECT_SET] + [n for n in args.sets if n != SELECT_SET]
    items = {n: evaluate_set(args.data_root, lib, n, cache, model, device, args) for n in names}
    report = report_for(items, args)
    for name, result in report["results"].items():
        LOGGER.info("%s: %s", name, {k: round(v["top1"], 3) for k, v in result.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
