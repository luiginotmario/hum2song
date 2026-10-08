"""Generate whistle and wrong-lyrics queries from vocal F0 (or MIDI).

Writes $DATA/query_synth/. Every row is origin=synthetic. Does not touch checkpoints.
"""

import argparse
import sys
from pathlib import Path

from hum2song.contour.config import load_contour_config
from hum2song.logutil import configure_logging, get_logger
from hum2song.query_synth.generate import (
    build_rows,
    library_melodies,
    midi_melodies,
    payloads_for,
    write_set,
)

LOGGER = get_logger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic whistle and wrong-lyrics queries")
    parser.add_argument("--config", type=Path, default=Path("configs/train_contour_qsynth.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--per-song", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="First N songs. 0 means all.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--midi-dir", default="", help="Optional MIDI folder under the data root.")
    parser.add_argument(
        "--sing-tracker",
        default="cepstrum",
        choices=("cepstrum", "rmvpe"),
        help="F0 tracker for wrong-lyrics audio. Whistles always use the spectral peak.",
    )
    return parser.parse_args(argv)


def melodies(root: Path, config, midi_dir: str, limit: int) -> tuple[dict, str]:
    songs = library_melodies(root, config)
    midis = midi_melodies(root, midi_dir)
    merged = {**songs, **midis}
    source = "vocal_f0+midi" if songs and midis else ("midi" if midis and not songs else "vocal_f0")
    if limit > 0:
        keep = set(sorted(merged)[:limit])
        merged = {key: merged[key] for key in keep}
    return merged, source


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = load_contour_config(args.config)
    if args.data_root is not None:
        config.data_root = str(args.data_root)
    root = config.resolved_data_root()
    chosen, source = melodies(root, config, args.midi_dir, args.limit)
    if not chosen:
        raise SystemExit(f"no vocal contours or MIDI under {root}")
    LOGGER.info("rendering %s songs (%s) into %s/query_synth", len(chosen), source, root)
    rows = build_rows(
        payloads_for(root, chosen, config, args.per_song, source, args.sing_tracker),
        args.workers,
    )
    path = write_set(root, rows)
    LOGGER.info("wrote %s queries to %s", len(rows), path)


if __name__ == "__main__":
    main()
