"""Evaluate a contour-branch checkpoint on MIR-QBSH (48 targets) and HumTrans val or test."""

import argparse
import json
import sys
from pathlib import Path

import torch

from hum2song.contour.config import ContourConfig
from hum2song.contour.evaluate import evaluate
from hum2song.contour.train import build_model, load_contours, validation_sets
from hum2song.logutil import configure_logging
from hum2song.manifest import read_pairs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contour-branch retrieval eval")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ContourConfig(**payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), config, int(payload["step"])


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, step = load_checkpoint(args.ckpt, device)
    root = config.resolved_data_root()
    records = read_pairs(root / "pairs_real.jsonl")
    contours = load_contours(root, records, config.shift_semitones)
    metrics = evaluate(model, validation_sets(records, contours, config, args.split), device)
    report = {"ckpt": str(args.ckpt), "step": step, "split": args.split, "metrics": metrics}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
