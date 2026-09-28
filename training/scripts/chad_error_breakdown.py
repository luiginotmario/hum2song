"""E0 (D-022): where do CHAD test hums fail against full recordings?

    python scripts/chad_error_breakdown.py --out results/e0.json

For every CHAD test hum, in the D-019 headline pool (full-song targets, chart previews, FMA):
  grid     rank with the normal 10 s / 5 s chunks (the API)
  oracle   rank when the target also gets one chunk at exactly the annotated hummed
           interval: grid -> oracle is the chunking (window placement) cost
  pitch    key-normalized DTW error (semitones, D-008-style slope constraint) between the
           hum contour and the reference contour in that interval, plain and octave-folded,
           plus the voiced fraction of the reference interval
Low reference voicing or large pitch error point at the song-melody extraction; failures
with small pitch error point at the embedding model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.catalog.fma import read_jsonl
from hum2song.catalog.matching import dtw_mae, frames, key_normalized
from hum2song.catalog.probe import summarize_ranks
from hum2song.catalog.real_eval import pick_track, song_scores
from hum2song.catalog.real_pool import (
    chad_split_targets,
    charts_pool,
    fma_rows,
    library_rows,
    load_tracks,
    query_contour_cache,
)
from hum2song.catalog.targets import chad_queries
from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.features import FRAME_S, rmvpe_contour, trim_unvoiced, voiced_fraction
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_CKPT = "ckpt/contour_v2_s0/last.pt"
LIBRARIES = ["previews_v1", "youtube_v1"]
TARGETS = "previews_v1"
METHOD = "vocals_or_mix"
MIN_REF_VOICED = 0.3
MAE_BINS = (0.0, 1.0, 1.5, 2.0, 3.0, 1.0e9)
DURATION_BINS = (0.0, 5.0, 8.0, 12.0, 1.0e9)
VOICED_BINS = (0.0, 0.3, 0.5, 0.7, 1.01)


def pool(root: Path, split_name: str) -> list[dict]:
    split = chad_split_targets(split_name)
    rows = library_rows(root, LIBRARIES) + fma_rows(root)
    return [r for r in rows if charts_pool(r, "chad", split)]


def embed_pool(rows: list[dict], model, device) -> tuple:
    """Songs with chunks, chunk owners, chunk embeddings, and each target's contour."""
    songs, owners, chunk_contours, contours = [], [], [], {}
    for row in rows:
        tracks = load_tracks(row["_track"])
        kind, chunks = pick_track(tracks, METHOD) if tracks else ("missing", [])
        if not chunks:
            continue
        if row["role"] == "target":
            contours[row["target_id"]] = rmvpe_contour(tracks[kind])
            contours["mix:" + row["target_id"]] = rmvpe_contour(tracks["mix"])
        owners += [len(songs)] * len(chunks)
        chunk_contours += [chunk.contour for chunk in chunks]
        songs.append(row)
    return songs, np.array(owners), embed_contours(model, chunk_contours, device), contours


def interval_window(contour: np.ndarray, fragment_s) -> np.ndarray:
    return contour[frames(fragment_s[0]) : frames(fragment_s[1])]


def rank_with(scores_row: np.ndarray, target_index: int, target_score: float) -> int:
    others = np.delete(scores_row, target_index)
    return int((others > target_score).sum()) + 1


def pitch_agreement(hum: np.ndarray, reference: np.ndarray) -> dict:
    ref, query = key_normalized(reference), key_normalized(hum)
    return {
        "mae": dtw_mae(ref, query),
        "mae_folded": dtw_mae(ref, query, fold_octaves=True),
        "ref_voiced": voiced_fraction(reference) if len(reference) else 0.0,
    }


def mix_agreement(hum: np.ndarray, mix_reference: np.ndarray) -> dict:
    """The same interval on the full-mix track: does the melody live outside the vocals?"""
    agreement = pitch_agreement(hum, mix_reference)
    return {"mae_mix": agreement["mae"], "ref_voiced_mix": agreement["ref_voiced"]}


def query_record(query, hum, scores_row, index, oracle_sim, reference, mix_reference) -> dict:
    grid_score = scores_row[index]
    return {
        "path": query["path"],
        "target_id": query["target_id"],
        "grid_rank": rank_with(scores_row, index, grid_score),
        "oracle_rank": rank_with(scores_row, index, max(grid_score, oracle_sim)),
        "oracle_only_rank": rank_with(scores_row, index, oracle_sim),
        "hum_s": round(len(trim_unvoiced(hum)) * FRAME_S, 2),
        "hum_voiced": voiced_fraction(hum),
        **pitch_agreement(hum, reference),
        **mix_agreement(hum, mix_reference),
    }


def bucket(records: list[dict], key: str, edges) -> dict:
    out = {}
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        picked = [r for r in records if low <= r[key] < high]
        out[f"{low:g}-{high:g}"] = {
            "queries": len(picked),
            "grid": summarize_ranks([r["grid_rank"] for r in picked]) if picked else None,
            "oracle": summarize_ranks([r["oracle_rank"] for r in picked]) if picked else None,
        }
    return out


def decomposition(records: list[dict]) -> dict:
    """Share of queries in each failure class (grid top-1 missed)."""
    missed = [r for r in records if r["grid_rank"] != 1]
    no_ref = [r for r in missed if r["ref_voiced"] < MIN_REF_VOICED]
    rest = [r for r in missed if r["ref_voiced"] >= MIN_REF_VOICED]
    fixed_by_oracle = [r for r in rest if r["oracle_rank"] == 1]
    still = [r for r in rest if r["oracle_rank"] != 1]
    median_mae = float(np.median([r["mae"] for r in records if np.isfinite(r["mae"])]))
    high_mae = [r for r in still if r["mae"] > median_mae]
    n = len(records)
    return {
        "queries": n,
        "grid_top1": 1 - len(missed) / n,
        "missed": len(missed) / n,
        "missed_reference_unvoiced": len(no_ref) / n,
        "missed_fixed_by_oracle_window": len(fixed_by_oracle) / n,
        "missed_with_oracle_high_pitch_error": len(high_mae) / n,
        "missed_with_oracle_low_pitch_error": (len(still) - len(high_mae)) / n,
        "median_mae": median_mae,
    }


def unvoiced_on_mix(records: list[dict]) -> dict:
    """Queries whose vocal-stem interval is (nearly) unvoiced: what the mix track shows."""
    picked = [r for r in records if r["ref_voiced"] < MIN_REF_VOICED]
    voiced_mix = [r for r in picked if r["ref_voiced_mix"] >= MIN_REF_VOICED]
    close = [r for r in voiced_mix if r["mae_mix"] <= MAE_BINS[2]]
    return {
        "queries": len(picked),
        "mix_voiced": len(voiced_mix),
        "mix_voiced_and_pitch_close": len(close),
        "median_mae_mix": float(np.median([r["mae_mix"] for r in voiced_mix]))
        if voiced_mix
        else None,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E0: CHAD test error breakdown")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _config, _step = load_checkpoint(args.ckpt or args.data_root / DEFAULT_CKPT, device)
    songs, owners, chunk_emb, contours = embed_pool(pool(args.data_root, args.split), model, device)
    fragments = {
        t["target_id"]: t["fragment_s"]
        for t in read_jsonl(args.data_root / "library" / TARGETS / "targets.jsonl")
        if t.get("fragment_s")
    }
    cache = query_contour_cache(args.data_root)
    index = {s["target_id"]: i for i, s in enumerate(songs) if s["role"] == "target"}
    queries = [
        q for q in chad_queries(args.data_root) if q["target_id"] in index and q["path"] in cache
    ]
    queries = [q for q in queries if q["target_id"] in fragments]
    hums = [cache[q["path"]] for q in queries]
    query_emb = embed_contours(model, hums, device)
    references = [
        interval_window(contours[q["target_id"]], fragments[q["target_id"]]) for q in queries
    ]
    mix_refs = [
        interval_window(contours["mix:" + q["target_id"]], fragments[q["target_id"]])
        for q in queries
    ]
    oracle_emb = embed_contours(
        model, [trim_unvoiced(r) if len(r) else r[:1] for r in references], device
    )
    scores = song_scores(
        torch.from_numpy(query_emb).to(device),
        torch.from_numpy(chunk_emb).to(device),
        torch.from_numpy(owners).long().to(device),
        len(songs),
    )
    oracle_sims = (query_emb * oracle_emb).sum(axis=1)
    records = [
        query_record(
            q, hum, scores[i], index[q["target_id"]], oracle_sims[i], references[i], mix_refs[i]
        )
        for i, (q, hum) in enumerate(zip(queries, hums, strict=True))
    ]
    report = {
        "split": args.split,
        "library_songs": len(songs),
        "grid": summarize_ranks([r["grid_rank"] for r in records]),
        "oracle_window": summarize_ranks([r["oracle_rank"] for r in records]),
        "oracle_window_only": summarize_ranks([r["oracle_only_rank"] for r in records]),
        "decomposition": decomposition(records),
        "by_pitch_mae": bucket(records, "mae", MAE_BINS),
        "by_pitch_mae_folded": bucket(records, "mae_folded", MAE_BINS),
        "by_reference_voiced": bucket(records, "ref_voiced", VOICED_BINS),
        "by_hum_seconds": bucket(records, "hum_s", DURATION_BINS),
        "by_hum_voiced": bucket(records, "hum_voiced", VOICED_BINS),
        "unvoiced_reference_on_mix": unvoiced_on_mix(records),
    }
    LOGGER.info("E0: %s", json.dumps(report["decomposition"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    args.out.with_suffix(".queries.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )


if __name__ == "__main__":
    main()
