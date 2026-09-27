"""Convert Essen Folksong Collection **kern files to MIDI, for MIREX-style distractors (D-011).

MIREX QBSH ranks MIR-QBSH queries against the 48 targets plus 2,000 Essen melodies.
Clone https://github.com/ccarh/essen-folksong-collection, then run this with music21
installed (it is not a project dependency). Scores carry no tempo, so music21's default
of 120 bpm applies, as in the MIREX MIDI set; the contour eval's tempo handling covers it.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def convert(job: tuple[str, str]) -> bool:
    source, dest = job
    import music21

    try:
        music21.converter.parse(source).write("midi", fp=dest)
    except Exception:  # noqa: BLE001 - a few Essen files do not parse; they are skipped
        return False
    return True


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Essen **kern -> MIDI")
    parser.add_argument("--essen", type=Path, required=True, help="essen-folksong-collection root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subdir", default="europa/deutschl")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted((args.essen / args.subdir).rglob("*.krn"))
    jobs = [(str(path), str(args.out / f"{path.stem}.mid")) for path in sources]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        converted = sum(pool.map(convert, jobs, chunksize=32))
    print(f"converted {converted} of {len(jobs)} files into {args.out}")


if __name__ == "__main__":
    main()
