"""Melody extraction from song audio: frame accuracy and retrieval drop (docs/DECISIONS.md D-014).

Needs extract_song_melody.py output. Frame measures (mir_eval, clip-averaged as in MIREX)
for every dataset and method; with --ckpt also the MIR-1K 1,000-way retrieval of
contour/song_eval.py, one row per reference source (annotation or extraction method).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.config import DEFAULT_DATA_ROOT
from hum2song.contour.evaluate import embed_contours
from hum2song.contour.features import trim_unvoiced
from hum2song.contour.melody import METHODS, frame_scores
from hum2song.contour.melody_data import (
    DATASETS,
    dataset_clips,
    reference_contour,
    vocal_subset,
)
from hum2song.contour.song_eval import (
    extracted_references,
    retrieval,
    sung_queries,
    synthetic_queries,
)
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
QUERY_SEED = 1234
CLEAN = "clean"


def load_tracks(root: Path, dataset: str) -> dict[str, np.ndarray] | None:
    path = root / "f0" / f"songmel_{dataset}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def mean_scores(rows: list[dict]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def frame_table(clips, tracks: dict) -> dict[str, dict]:
    methods = [m for m in (*METHODS, CLEAN) if f"{m}/{clips[0].name}" in tracks]
    return {
        method: mean_scores(
            [frame_scores(c.ref_times, c.ref_hz, tracks[f"{method}/{c.name}"]) for c in clips]
        )
        for method in methods
    }


def retrieval_table(model, clips, tracks: dict, device) -> dict[str, dict]:
    names = [clip.name for clip in clips]
    songs = [clip.song for clip in clips]
    melodies = [reference_contour(clip.ref_times, clip.ref_hz) for clip in clips]
    queries = {
        "sung": sung_queries(tracks, names, QUERY_SEED),
        "synthetic": synthetic_queries(melodies, QUERY_SEED),
    }
    references = {"annotation": [trim_unvoiced(melody) for melody in melodies]}
    for method in (*METHODS, CLEAN):
        if f"{method}/{names[0]}" in tracks:
            references[method] = extracted_references(tracks, method, names)
    query_emb = {kind: embed_contours(model, items, device) for kind, items in queries.items()}
    table = {}
    for source, contours in references.items():
        ref_emb = embed_contours(model, contours, device)
        for kind, embedded in query_emb.items():
            table[f"{kind}->{source}"] = {
                "clip": retrieval(embedded, ref_emb, names),
                "song": retrieval(embedded, ref_emb, songs),
            }
    return table


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Song melody extraction eval")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--ckpt", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {"frame": {}, "retrieval": {}}
    for dataset in DATASETS:
        tracks = load_tracks(args.data_root, dataset)
        if tracks is None:
            LOGGER.info("no tracks for %s, skipped", dataset)
            continue
        clips = dataset_clips(args.data_root, dataset)
        report["frame"][dataset] = frame_table(clips, tracks)
        vocal = vocal_subset(dataset, clips)
        if vocal:
            report["frame"][f"{dataset}_vocal"] = frame_table(vocal, tracks)
        if dataset != "mir1k":
            continue
        for ckpt in args.ckpt:
            model, _config, _step = load_checkpoint(ckpt, device)
            report["retrieval"][str(ckpt)] = retrieval_table(model, clips, tracks, device)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
