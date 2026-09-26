"""Score a checkpoint and write top-1 / top-10 / MRR next to the CHAD baselines."""

import argparse
import sys
from pathlib import Path

import torch

from hum2song.build_model import build_model
from hum2song.config import load_eval_config, train_config_from_mapping
from hum2song.eval.run_eval import evaluate_pairs, log_eval_plan, write_report
from hum2song.logutil import configure_logging, get_logger
from hum2song.manifest import read_pairs

LOGGER = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Evaluate test rows in the manifest. --dry-run prints the plan and the baselines."""
    configure_logging()
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = load_eval_config(args.config, _overrides(args))
    if config.dry_run:
        log_eval_plan(config)
        return
    if not config.ckpt:
        raise SystemExit("pass --ckpt")
    payload = torch.load(config.ckpt, map_location="cpu", weights_only=False)
    train_config = train_config_from_mapping(payload["config"])
    model = build_model(train_config)
    model.load_state_dict(payload["model"])
    pairs = read_pairs(config.resolved_manifest())
    report = evaluate_pairs(model, pairs, config)
    out = config.resolved_out()
    write_report(out, report)
    LOGGER.info("wrote %s", out)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval eval")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--sets", default=None, help="Comma-separated groups, or all")
    parser.add_argument("--distractors", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict:
    return {
        "ckpt": args.ckpt,
        "sets": args.sets,
        "distractors": args.distractors,
        "out": args.out,
        "manifest": args.manifest,
        "dry_run": True if args.dry_run else None,
    }


if __name__ == "__main__":
    main()
