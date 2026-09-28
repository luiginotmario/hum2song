"""Real hums vs real commercial recordings: 30 s previews or full songs (D-018, D-019).

    python scripts/eval_previews.py --libraries previews_v1 youtube_v1 \
        --out docs/paper/results/youtube/real_hum_eval_full.json

Queries: CHAD hums, MTG-QBH sung queries, MLEnd hums and whistles, through the API query
path (RMVPE -> cleaned contour -> contour encoder). Library settings:
  targets_only   the query set's target recordings only
  charts         + chart distractor previews (D-018)
  charts_fma     + the searchable FMA songs (D-017), re-embedded from cached vocal tracks
Target versions: iTunes preview, Deezer preview, both previews, or the full YouTube song.
--chad-split test keeps only CHAD test-split songs (D-019) as queries and targets, so a
model trained on CHAD train songs is compared on the same footing.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hum2song.audio import load_audio
from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.probe import summarize_ranks
from hum2song.catalog.real_eval import (
    TRACK_METHODS,
    covered_targets,
    pick_track,
    positive_negative,
    song_scores,
    target_ranks,
)
from hum2song.catalog.search import QueryEncoder
from hum2song.catalog.targets import query_sets
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour import chad
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.melody import TRACK_RATE
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
DEFAULT_RMVPE = "fig_contours/rmvpe.pt"
FMA_LIBRARY = "fma_full_3k"
QUERY_CACHE = "library/query_contours.npz"
SOURCES = {
    "itunes": ("itunes_preview",),
    "deezer": ("deezer_preview",),
    "both": ("itunes_preview", "deezer_preview"),
    "youtube": ("youtube_full",),
}
SETTINGS = ("targets_only", "charts", "charts_fma")
DETAIL_SETTING = "charts_fma"
FRAGMENT_BINS = ((0, 30), (30, 60), (60, 90), (90, 10_000))


def query_contours(encoder: QueryEncoder, queries: list[dict], cache: Path) -> dict:
    """Contour per query path, cached; unreadable files are left out."""
    done = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    missing = [q for q in queries if q["path"] not in done and Path(q["path"]).exists()]
    for query in missing:
        done[query["path"]] = encoder.contour(load_audio(query["path"], TRACK_RATE, trim=False))
    if missing:
        np.savez(cache, **done)
    return done


def library_rows(root: Path, names: list[str]) -> list[dict]:
    """previews.jsonl rows of each library, tagged with their track file."""
    rows = []
    for name in names:
        tracks = root / "library" / name / "tracks"
        for row in read_jsonl(root / "library" / name / "previews.jsonl"):
            rows.append({**row, "_track": str(tracks / f"{row['song_id'].replace(':', '_')}.npz")})
    return rows


def fma_rows(root: Path) -> list[dict]:
    tracks = root / "library" / FMA_LIBRARY / "tracks"
    return [
        {
            "song_id": s["song_id"],
            "role": "fma",
            "target_id": None,
            "source": "fma_full",
            "_track": str(tracks / f"{s['song_id'].replace(':', '_')}.npy"),
        }
        for s in read_jsonl(root / "library" / FMA_LIBRARY / "songs.jsonl")
    ]


def load_tracks(path: str) -> dict | None:
    """npz with vocals and mix; FMA .npy caches hold the vocal track only (used for both)."""
    if not Path(path).exists():
        return None
    if path.endswith(".npy"):
        vocals = np.load(path)
        return {"vocals": vocals, "mix": vocals}
    with np.load(path) as tracks:
        return dict(tracks)


def embed_library(rows: list[dict], method: str, model, device) -> tuple:
    """(songs with a chunk, chunk owner index, chunk embeddings, track use counts)."""
    songs, owners, contours, used = [], [], [], Counter()
    for row in rows:
        tracks = load_tracks(row["_track"])
        kind, chunks = pick_track(tracks, method) if tracks else ("missing", [])
        used[f"{row['role']}:{kind if chunks else 'no_chunk'}"] += 1
        if chunks:
            owners += [len(songs)] * len(chunks)
            contours += [chunk.contour for chunk in chunks]
            songs.append(row)
    return songs, np.array(owners), embed_contours(model, contours, device), dict(used)


def keep_song(song: dict, sources: tuple, setting: str, dataset: str) -> bool:
    if song["role"] == "target":
        return song["source"] in sources and song["target_id"].startswith(dataset + ":")
    if song["role"] == "distractor":
        return setting != "targets_only"
    return setting == "charts_fma"


def select(library: tuple, keep: np.ndarray) -> tuple:
    """Subset of songs (keep[i] True) with their chunks, owners re-indexed."""
    songs, owners, embeddings = library[:3]
    chunk_mask = keep[owners]
    kept = [s for s, k in zip(songs, keep, strict=True) if k]
    return kept, (np.cumsum(keep) - 1)[owners[chunk_mask]], embeddings[chunk_mask]


def rank_queries(library: tuple, query_emb: np.ndarray, query_targets: list, device) -> dict:
    songs, owners, embeddings = library
    scores = song_scores(
        torch.from_numpy(query_emb).to(device),
        torch.from_numpy(embeddings).to(device),
        torch.from_numpy(owners).long().to(device),
        len(songs),
    )
    targets = [song["target_id"] for song in songs]
    return {
        "songs": len(songs),
        "scores": scores,
        "targets": targets,
        "ranks": target_ranks(scores, targets, query_targets),
    }


def per_target_top1(ranks: list, query_targets: list) -> dict:
    by_target: dict[str, list[bool]] = {}
    for rank, target in zip(ranks, query_targets, strict=True):
        by_target.setdefault(target, []).append(rank == 1)
    return {t: float(np.mean(hits)) for t, hits in sorted(by_target.items())}


def fragment_breakdown(ranks: list, query_targets: list, fragments: dict) -> dict:
    """CHAD: accuracy by where the hummed fragment starts in the original recording."""
    out = {}
    for low, high in FRAGMENT_BINS:
        picked = [
            r
            for r, t in zip(ranks, query_targets, strict=True)
            if low <= fragments.get(t, (-1.0, 0.0))[0] < high
        ]
        out[f"{low}-{high}s"] = summarize_ranks(picked) if picked else None
    return out


def details(result: dict, query_targets: list, fragments: dict) -> dict:
    positives, negatives = positive_negative(result["scores"], result["targets"], query_targets)
    coverage = covered_targets(positives, negatives, query_targets)
    return {
        "hook_coverage_estimate": {
            "threshold": coverage["threshold"],
            "targets": len(coverage["covered"]),
            "covered_fraction": float(np.mean(list(coverage["covered"].values()) or [0.0])),
        },
        "per_target_top1": per_target_top1(result["ranks"], query_targets),
        "by_fragment_start": fragment_breakdown(result["ranks"], query_targets, fragments),
    }


def entry_for(result: dict, query_targets: list) -> dict:
    available = {t for t in result["targets"] if t}
    present = [r for r, t in zip(result["ranks"], query_targets, strict=True) if t in available]
    return {
        "library_songs": result["songs"],
        "queries": len(query_targets),
        "queries_with_target": len(present),
        "all_queries": summarize_ranks(result["ranks"]),
        "target_present": summarize_ranks(present),
    }


def chad_filter(rows: list[dict], sets: dict, split_name: str | None) -> tuple:
    """Keep only CHAD songs of one split, as queries and as targets."""
    if not split_name:
        return rows, sets
    split = chad.read_split()
    in_split = {f"chad:{g}" for g, s in split.items() if s == split_name}
    rows = [
        r
        for r in rows
        if not (r["target_id"] or "").startswith("chad:") or r["target_id"] in in_split
    ]
    sets = {**sets, "chad_hum": [q for q in sets["chad_hum"] if q["target_id"] in in_split]}
    return rows, sets


def evaluate_method(library: tuple, sets: dict, contours: dict, model, fragments, device) -> dict:
    results = {}
    for set_name, queries in sets.items():
        queries = [q for q in queries if q["path"] in contours]
        query_emb = embed_contours(model, [contours[q["path"]] for q in queries], device)
        query_targets = [q["target_id"] for q in queries]
        dataset = set_name.split("_")[0]
        for source, names in SOURCES.items():
            for setting in SETTINGS:
                keep = np.array([keep_song(s, names, setting, dataset) for s in library[0]])
                result = rank_queries(select(library, keep), query_emb, query_targets, device)
                entry = entry_for(result, query_targets)
                if setting == DETAIL_SETTING:
                    entry.update(details(result, query_targets, fragments))
                results[f"{set_name}/{source}/{setting}"] = entry
                LOGGER.info("%s/%s/%s: %s", set_name, source, setting, entry["all_queries"])
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real hums vs commercial recordings")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--libraries", nargs="+", default=["previews_v1"])
    parser.add_argument("--targets", default="previews_v1", help="library holding targets.jsonl")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=list(TRACK_METHODS))
    parser.add_argument("--chad-split", default=None, help="e.g. test (D-019 song split)")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or args.data_root / DEFAULT_CKPT
    encoder = QueryEncoder(ckpt, args.data_root / DEFAULT_RMVPE, device)
    sets = query_sets(args.data_root)
    contours = query_contours(
        encoder, [q for s in sets.values() for q in s], args.data_root / QUERY_CACHE
    )
    rows, sets = chad_filter(library_rows(args.data_root, args.libraries), sets, args.chad_split)
    rows += fma_rows(args.data_root)
    targets = read_jsonl(args.data_root / "library" / args.targets / "targets.jsonl")
    fragments = {t["target_id"]: t["fragment_s"] for t in targets if t.get("fragment_s")}
    report = {
        "ckpt": str(ckpt),
        "chad_split": args.chad_split,
        "libraries": args.libraries,
        "track_use": {},
        "results": {},
    }
    for method in args.methods:
        songs, owners, embeddings, used = embed_library(rows, method, encoder.model, device)
        report["track_use"][method] = used
        results = evaluate_method(
            (songs, owners, embeddings), sets, contours, encoder.model, fragments, device
        )
        report["results"].update({f"{method}/{k}": v for k, v in results.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
