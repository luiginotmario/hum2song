"""Render two synthetic queries on CPU and write smoke.json. No dataset and no GPU."""

import argparse
import json
import sys
from pathlib import Path

from hum2song.query_synth.smoke import run_smoke


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU smoke test of the query synth")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_smoke(args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
