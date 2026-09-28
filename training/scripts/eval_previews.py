"""Real hums vs real commercial recordings (30 s previews), plus extraction accounting (D-018).

    python scripts/eval_previews.py --name previews_v1 --out docs/paper/results/previews

Queries: CHAD hums, MTG-QBH sung queries, MLEnd hums and whistles, all through the API
query path (RMVPE -> cleaned contour -> D-012 encoder). Library settings:
  targets_only   the query set's target previews only
  charts         + chart distractor previews
  charts_fma     + the 2,783 searchable FMA songs already in pgvector (vocal stem chunks)
Target versions: iTunes preview, Deezer preview, or both (either counts as correct).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hum2song.audio import load_audio
from hum2song.catalog.db import connect
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
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.melody import TRACK_RATE
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
DEFAULT_RMVPE = "fig_contours/rmvpe.pt"
SOURCES = {
    "itunes": ("itunes_preview",),
    "deezer": ("deezer_preview",),
    "both": ("itunes_preview", "deezer_preview"),
}
SETTINGS = ("targets_only", "charts", "charts_fma")
HEADLINE = ("vocals_or_mix", "itunes", "charts_fma")
FRAGMENT_BINS = ((0, 30), (30, 60), (60, 90), (90, 10_000))


def query_contours(encoder: QueryEncoder, queries: list[dict], cache: Path) -> dict:
    """Contour per query path, cached; unreadable files are left out."""
    done = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    for query in queries:
        if query["path"] in done or not Path(query["path"]).exists():
            continue
        done[query["path"]] = encoder.contour(load_audio(query["path"], TRACK_RATE, trim=False))
    np.savez(cache, **done)
    return done


def preview_library(rows: list[dict], track_dir: Path, method: str, model, device):
    """Chunk embeddings of every extracted preview under one track method."""
    songs, owners, contours, used = [], [], [], Counter()
    for row in rows:
        path = track_dir / f"{row['song_id'].replace(':', '_')}.npz"
        if not path.exists():
            continue
        with np.load(path) as tracks:
            kind, chunks = pick_track(dict(tracks), method)
        if not chunks:
            used["no_chunk"] += 1
            continue
        used[kind] += 1
        owners += [len(songs)] * len(chunks)
        contours += [chunk.contour for chunk in chunks]
        songs.append(row)
    embeddings = embed_contours(model, contours, device)
    return songs, np.array(owners), embeddings, dict(used)


def fma_library(url: str):
    connection = connect(url)
    rows = connection.execute(
        "SELECT song_id, embedding FROM chunks WHERE song_id LIKE 'fma:%' ORDER BY song_id"
    ).fetchall()
    ids = sorted({song for song, _ in rows})
    position = {song: i for i, song in enumerate(ids)}
    owners = np.array([position[song] for song, _ in rows])
    embeddings = np.stack([np.asarray(vector.to_numpy(), dtype=np.float32) for _, vector in rows])
    return (
        [{"song_id": s, "role": "fma", "target_id": None, "source": "fma_full"} for s in ids],
        owners,
        embeddings,
    )


def select(songs, owners, embeddings, keep) -> tuple:
    """Subset of songs (keep[i] True) with their chunks, owners re-indexed."""
    new_index = np.cumsum(keep) - 1
    chunk_mask = keep[owners]
    return (
        [s for s, k in zip(songs, keep, strict=True) if k],
        new_index[owners[chunk_mask]],
        embeddings[chunk_mask],
    )


def setting_mask(songs: list[dict], sources: tuple, setting: str, dataset: str) -> np.ndarray:
    def keep(song: dict) -> bool:
        if song["role"] == "target":
            return song["source"] in sources and song["target_id"].startswith(dataset + ":")
        if song["role"] == "distractor":
            return setting != "targets_only"
        return setting == "charts_fma"

    return np.array([keep(song) for song in songs])


def run_setting(songs, owners, embeddings, query_emb, query_targets, device) -> dict:
    chunks = torch.from_numpy(embeddings).to(device)
    owner = torch.from_numpy(owners).long().to(device)
    scores = song_scores(torch.from_numpy(query_emb).to(device), chunks, owner, len(songs))
    targets = [song["target_id"] for song in songs]
    ranks = target_ranks(scores, targets, query_targets)
    return {"songs": len(songs), "ranks": ranks, "scores": scores, "targets": targets}


def with_target(ranks: list, query_targets: list, available: set) -> list:
    return [r for r, t in zip(ranks, query_targets, strict=True) if t in available]


def fragment_breakdown(ranks: list, query_targets: list, targets: dict) -> dict:
    """CHAD: accuracy by where the hummed fragment starts in the original video."""
    out = {}
    for low, high in FRAGMENT_BINS:
        picked = [
            r
            for r, t in zip(ranks, query_targets, strict=True)
            if low <= targets.get(t, {}).get("fragment_s", (-1, 0))[0] < high
        ]
        out[f"{low}-{high}s"] = summarize_ranks(picked) if picked else None
    return out


def extraction_report(targets: list[dict], rows: list[dict], status: list[dict]) -> dict:
    by_dataset = {}
    for dataset in sorted({t["dataset"] for t in targets}):
        group = [t for t in targets if t["dataset"] == dataset]
        by_dataset[dataset] = {
            "targets": len(group),
            "no_title": sum(not t["title"] for t in group),
            "itunes_match": sum(bool(t["itunes"]) for t in group),
            "deezer_match": sum(bool(t["deezer"]) for t in group),
            "either_match": sum(bool(t["itunes"] or t["deezer"]) for t in group),
        }
    outcome = Counter((line["role"], line["status"]) for line in status)
    return {
        "targets": by_dataset,
        "previews": len(rows),
        "download_extract": {f"{r}:{s}": n for (r, s), n in sorted(outcome.items())},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real hums vs commercial previews")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--name", default="previews_v1")
    parser.add_argument("--db", default="postgresql://h2s:h2s@127.0.0.1:5432/h2s")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    library_dir = args.data_root / "library" / args.name
    encoder = QueryEncoder(args.data_root / DEFAULT_CKPT, args.data_root / DEFAULT_RMVPE, device)
    rows = read_jsonl(library_dir / "previews.jsonl")
    targets = {t["target_id"]: t for t in read_jsonl(library_dir / "targets.jsonl")}
    status = read_jsonl(library_dir / "status.jsonl")
    sets = query_sets(args.data_root)
    contours = query_contours(
        encoder,
        [q for s in sets.values() for q in s],
        args.data_root / "library" / "query_contours.npz",
    )
    fma = fma_library(args.db)
    report = {"extraction": extraction_report(list(targets.values()), rows, status), "results": {}}
    for method in TRACK_METHODS:
        songs, owners, embeddings, used = preview_library(
            rows, library_dir / "tracks", method, encoder.model, device
        )
        report.setdefault("track_use", {})[method] = used
        all_songs = songs + fma[0]
        all_owners = np.concatenate([owners, fma[1] + len(songs)])
        all_emb = np.concatenate([embeddings, fma[2]])
        for set_name, queries in sets.items():
            queries = [q for q in queries if q["path"] in contours]
            query_emb = embed_contours(
                encoder.model, [contours[q["path"]] for q in queries], device
            )
            query_targets = [q["target_id"] for q in queries]
            dataset = set_name.split("_")[0]
            for source, source_names in SOURCES.items():
                for setting in SETTINGS:
                    mask = setting_mask(all_songs, source_names, setting, dataset)
                    subset = select(all_songs, all_owners, all_emb, mask)
                    result = run_setting(*subset, query_emb, query_targets, device)
                    available = {t for t in result["targets"] if t}
                    entry = {
                        "library_songs": result["songs"],
                        "queries": len(queries),
                        "queries_with_target": len(
                            with_target(result["ranks"], query_targets, available)
                        ),
                        "all_queries": summarize_ranks(result["ranks"]),
                        "target_present": summarize_ranks(
                            with_target(result["ranks"], query_targets, available)
                        ),
                    }
                    if (method, source, setting) == HEADLINE:
                        pos, neg = positive_negative(
                            result["scores"], result["targets"], query_targets
                        )
                        coverage = covered_targets(pos, neg, query_targets)
                        entry["hook_coverage_estimate"] = {
                            "threshold": coverage["threshold"],
                            "targets": len(coverage["covered"]),
                            "covered_fraction": float(
                                np.mean(list(coverage["covered"].values()) or [0])
                            ),
                        }
                        entry["per_target_top1"] = (
                            {
                                t: float(
                                    np.mean(
                                        [
                                            r == 1
                                            for r, q in zip(
                                                result["ranks"], query_targets, strict=True
                                            )
                                            if q == t
                                        ]
                                    )
                                )
                                for t in sorted(set(query_targets) & available)
                            }
                            if dataset in ("mlend", "mtgqbh")
                            else None
                        )
                        if dataset == "chad":
                            entry["by_fragment_start"] = fragment_breakdown(
                                result["ranks"], query_targets, targets
                            )
                    report["results"][f"{set_name}/{method}/{source}/{setting}"] = entry
                    LOGGER.info(
                        "%s/%s/%s/%s: %s", set_name, method, source, setting, entry["all_queries"]
                    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "real_hum_eval.json").write_text(json.dumps(report, indent=1))
    headline = {
        k: v["all_queries"] for k, v in report["results"].items() if k.endswith("/".join(HEADLINE))
    }
    print(
        json.dumps(
            {
                "extraction": report["extraction"],
                "track_use": report["track_use"],
                "headline": headline,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
