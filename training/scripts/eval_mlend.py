"""Whistle and hum eval on MLEnd with other people's clips as references (D-013).

Needs F0 tracks from extract_f0.py: rmvpe for all MLEnd clips, and peak / rmvpe_half for
the whistles. Every query is ranked over the 8 songs; the query's performer never
contributes to a reference (example_eval.py).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from hum2song.contour.evaluate import embed_contours, query_contour
from hum2song.contour.example_eval import ExampleSet, example_metrics
from hum2song.contour.features import rmvpe_contour
from hum2song.contour.train import load_checkpoint
from hum2song.contour.whistle import WHISTLE_F0_MAX_HZ
from hum2song.logutil import configure_logging
from hum2song.manifest import read_pairs

ATTRIBUTES = "raw/mlend_hums_whistles/MLEndHWD_audio_attributes_benchmark.csv"
MIN_VOICED_FRAMES = 50
QUERY_SETS = (
    ("hum", "rmvpe", "hum"),
    ("whistle_peak", "peak", "whistle"),
    ("whistle_rmvpe_half", "rmvpe_half", "whistle"),
    ("whistle_rmvpe", "rmvpe", "whistle"),
)
REFERENCE_SETS = ("hum", "whistle_peak")
MODES = ("centroid", "max")


def performers(root: Path) -> dict[str, str]:
    """MLEnd file name (0000.wav) -> interpreter id."""
    with open(root / ATTRIBUTES, newline="", encoding="utf-8") as handle:
        return {row["filename"]: row["Interpreter"] for row in csv.DictReader(handle)}


def clip_contours(root: Path, method: str, paths: list[str]) -> list[np.ndarray]:
    limit = WHISTLE_F0_MAX_HZ if method != "rmvpe" else 2000.0
    with np.load(root / "f0" / f"{method}_mlend.npz") as tracks:
        return [query_contour(rmvpe_contour(tracks[path], limit)) for path in paths]


def build_sets(model, root: Path, device) -> tuple[dict[str, ExampleSet], dict[str, dict]]:
    records = [r for r in read_pairs(root / "pairs_real.jsonl") if r.group == "mlend"]
    people = performers(root)
    sets: dict[str, ExampleSet] = {}
    counts: dict[str, dict] = {}
    for name, method, qtype in QUERY_SETS:
        chosen = sorted((r for r in records if r.qtype == qtype), key=lambda r: r.query_path)
        contours = clip_contours(root, method, [r.query_path for r in chosen])
        voiced = [int(np.sum(~np.isnan(c))) for c in contours]
        counts[name] = {
            "clips": len(chosen),
            "under_1s_voiced": sum(v < MIN_VOICED_FRAMES for v in voiced),
        }
        sets[name] = ExampleSet(
            embeddings=embed_contours(model, contours, device),
            songs=[r.song_id for r in chosen],
            people=[people[Path(r.query_path).name.removeprefix("mlend_")] for r in chosen],
        )
    return sets, counts


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLEnd hum/whistle query-by-example eval")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_checkpoint(args.ckpt, device)
    sets, counts = build_sets(model, config.resolved_data_root(), device)
    results = {
        f"{query}->{ref}/{mode}": example_metrics(sets[query], sets[ref], mode)
        for ref in REFERENCE_SETS
        for query, _method, _qtype in QUERY_SETS
        for mode in MODES
    }
    report = {"ckpt": str(args.ckpt), "step": step, "counts": counts, "results": results}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
