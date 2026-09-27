"""Whistle and hum eval on MLEnd with other people's clips as references (D-013, D-015).

Needs F0 tracks from extract_f0.py: rmvpe for all MLEnd clips, and peak / rmvpe_half for
the whistles. Every query is ranked over the 8 songs; the query's performer never
contributes to a reference (example_eval.py). --split restricts queries and references
to one part of the fixed performer split (training/splits/mlend_performers.json); D-015
reports --split test, D-013 used everyone (--split all).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.contour.example_eval import example_metrics
from hum2song.contour.mlend import clips_in, example_set, load_contours, mlend_clips, read_split
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging

MIN_VOICED_FRAMES = 50
QUERY_SETS = (
    ("hum", "rmvpe", "hum"),
    ("whistle_peak", "peak", "whistle"),
    ("whistle_rmvpe_half", "rmvpe_half", "whistle"),
    ("whistle_rmvpe", "rmvpe", "whistle"),
)
REFERENCE_SETS = ("hum", "whistle_peak")
MODES = ("centroid", "max")


def build_sets(model, root: Path, split_name: str, device) -> tuple[dict, dict]:
    clips, split = mlend_clips(root), read_split()
    sets, counts = {}, {}
    for name, method, qtype in QUERY_SETS:
        chosen = clips_in(clips, split, split_name, qtype)
        contours = load_contours(root, method, chosen)
        voiced = [int(np.sum(~np.isnan(contours[c.path]))) for c in chosen]
        counts[name] = {
            "clips": len(chosen),
            "performers": len({c.person for c in chosen}),
            "under_1s_voiced": sum(v < MIN_VOICED_FRAMES for v in voiced),
        }
        sets[name] = example_set(model, chosen, contours, device)
    return sets, counts


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLEnd hum/whistle query-by-example eval")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_checkpoint(args.ckpt, device)
    sets, counts = build_sets(model, config.resolved_data_root(), args.split, device)
    results = {
        f"{query}->{ref}/{mode}": example_metrics(sets[query], sets[ref], mode)
        for ref in REFERENCE_SETS
        for query, _method, _qtype in QUERY_SETS
        for mode in MODES
    }
    report = {
        "ckpt": str(args.ckpt),
        "step": step,
        "split": args.split,
        "counts": counts,
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
