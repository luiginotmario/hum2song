"""Train the key-invariant contour encoder (docs/DECISIONS.md D-011)."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from hum2song.contour.config import load_contour_config
from hum2song.contour.train import run_training
from hum2song.logutil import configure_logging
from hum2song.wandb_tracker import build_tracker


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contour-branch contrastive training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the result JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    overrides = {
        "run_name": args.run_name,
        "steps": args.steps,
        "batch_size": args.bs,
        "lr": args.lr,
        "val_every": args.val_every,
        "seed": args.seed,
        "num_workers": args.workers,
    }
    config = load_contour_config(args.config, overrides)
    tracker = build_tracker(config.run_name, asdict(config))
    result = run_training(config, tracker)
    tracker.finish()
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
