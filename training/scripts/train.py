"""Stage A contrastive fine-tune of MERT-v1-95M."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from hum2song.build_model import build_model
from hum2song.config import load_train_config
from hum2song.logutil import configure_logging
from hum2song.train_loop import run_training
from hum2song.wandb_tracker import build_tracker


def main(argv: list[str] | None = None) -> None:
    """Train from a YAML config. --dry-run runs a single step when data is present."""
    configure_logging()
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = load_train_config(args.config, _overrides(args))
    model = build_model(config)
    tracker = build_tracker(config.run_name, asdict(config))
    run_training(config, model, tracker)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A contrastive training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", default=None, choices=["A", "B", "C"])
    parser.add_argument("--init", default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--init-weights",
        default=None,
        help="Checkpoint whose model weights start a new run (fresh optimizer, step 0)",
    )
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lr-bb", type=float, default=None)
    parser.add_argument("--unfreeze-top", type=int, default=None)
    parser.add_argument("--conf-loss-w", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict:
    return {
        "stage": args.stage,
        "resume": args.init,
        "init_weights": args.init_weights,
        "batch_size": args.bs,
        "steps": args.steps,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_bb,
        "unfreeze_top_layers": args.unfreeze_top,
        "conf_loss_weight": args.conf_loss_w,
        "limit": args.limit,
        "dry_run": True if args.dry_run else None,
    }


if __name__ == "__main__":
    main()
