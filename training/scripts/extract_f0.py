"""Extract F0 for every query clip of the chosen groups (docs/DECISIONS.md D-011, D-013).

Writes one float32 array of shape (2, frames) per clip, rows = F0 in Hz and voicing
confidence, 10 ms hop, into <data_root>/f0/<method>_<group>[_shift<k>].npz, keyed by the
manifest query path. --method picks the tracker (contour/trackers.py): rmvpe (default,
hums and singing), peak (spectral peak, whistles, D-007) or rmvpe_half. --qtypes limits
the clips, e.g. to whistles. Audio is not trimmed, so frame 0 is the start of the
file and HumTrans hums stay aligned with their MIDI. --shift transposes the audio first
(the same resample + phase vocoder used by validation), for key-robustness checks.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hum2song.audio import load_audio
from hum2song.augment import transpose_and_stretch
from hum2song.config import DEFAULT_DATA_ROOT, default_num_workers
from hum2song.contour.trackers import SAMPLE_RATES, load_rmvpe, track_audio
from hum2song.logutil import configure_logging, get_logger
from hum2song.manifest import read_pairs

LOGGER = get_logger(__name__)
DEFAULT_WEIGHTS = "fig_contours/rmvpe.pt"


class ClipDataset(Dataset):
    """Loads mono audio at one sample rate, optionally transposed, for one list of paths."""

    def __init__(self, paths: list[str], data_root: Path, semitones: float, rate: int) -> None:
        self.paths = paths
        self.data_root = data_root
        self.semitones = semitones
        self.rate = rate

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray]:
        path = self.paths[index]
        audio = load_audio(self.data_root / path, self.rate, trim=False)
        if self.semitones:
            audio = transpose_and_stretch(audio, self.semitones, 1.0)
        return path, audio


def query_paths(data_root: Path, group: str, splits: set[str], qtypes: set[str]) -> list[str]:
    records = read_pairs(data_root / "pairs_real.jsonl")
    chosen = {
        r.query_path
        for r in records
        if r.group == group and r.split in splits and (not qtypes or r.qtype in qtypes)
    }
    return sorted(chosen)


def output_path(data_root: Path, method: str, group: str, semitones: float) -> Path:
    suffix = f"_shift{semitones:+g}" if semitones else ""
    return data_root / "f0" / f"{method}_{group}{suffix}.npz"


def split_names(text: str) -> list[str]:
    return [name.strip() for name in text.split(",") if name.strip()]


def extract_group(model, data_root: Path, group: str, splits: set[str], args) -> None:
    paths = query_paths(data_root, group, splits, split_names(args.qtypes))
    dataset = ClipDataset(paths, data_root, args.shift, SAMPLE_RATES[args.method])
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        prefetch_factor=8 if args.workers else None,
    )
    tracks: dict[str, np.ndarray] = {}
    for index, (path, audio) in enumerate(loader):
        tracks[path] = track_audio(args.method, model, np.asarray(audio, dtype=np.float32))
        if index % 1000 == 0:
            LOGGER.info("%s: %s / %s", group, index, len(paths))
    destination = output_path(data_root, args.method, group, args.shift)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **tracks)
    LOGGER.info("wrote %s tracks to %s", len(tracks), destination)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F0 tracks for query clips")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--groups", default="humtrans,mirqbsh")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--method", default="rmvpe", choices=sorted(SAMPLE_RATES))
    parser.add_argument("--qtypes", default="", help="Comma list, e.g. whistle (default all)")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=default_num_workers())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model = None
    if args.method != "peak":
        weights = args.weights or args.data_root / DEFAULT_WEIGHTS
        model = load_rmvpe(weights, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    splits = set(split_names(args.splits))
    for group in split_names(args.groups):
        extract_group(model, args.data_root, group, splits, args)


if __name__ == "__main__":
    main()
