"""Score whistle, wrong-lyrics, and the hum baseline. Does not reindex or replace E2b."""

import argparse
import json
import sys
from pathlib import Path

import torch

from hum2song.contour.config import load_contour_config
from hum2song.contour.train import load_checkpoint
from hum2song.logutil import configure_logging, get_logger
from hum2song.query_synth.report import build_report

LOGGER = get_logger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval synthetic queries and the real baselines")
    parser.add_argument("--config", type=Path, default=Path("configs/train_contour_qsynth.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref-windows", type=int, default=24)
    parser.add_argument("--synthetic-limit", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = load_contour_config(args.config)
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_config, step = load_checkpoint(args.ckpt, device)
    report = build_report(
        config.resolved_data_root(),
        model,
        device,
        ckpt_config,
        config.query_synth_manifest,
        args.ref_windows,
        args.synthetic_limit,
    )
    report["checkpoint"] = str(args.ckpt)
    report["checkpoint_step"] = step
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote %s (replaces_e2b=%s)", args.out, report["replaces_e2b"])


if __name__ == "__main__":
    main()
