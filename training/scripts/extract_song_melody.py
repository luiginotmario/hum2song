"""Melody tracks of recorded songs for every extraction method (docs/DECISIONS.md D-014).

Writes <data_root>/f0/songmel_<dataset>.npz with one (2, frames) 10 ms track per
"<method>/<clip>" key (contour/melody.py METHODS). MIR-1K also gets two query-side tracks
from its isolated singing channel, both RMVPE:
  clean       the vocal channel as is (the ceiling for any separation)
  query_sung  the vocal channel transposed by 2-5 semitones up or down and time-stretched
              by 0.85-1.15 (seeded per clip), used as a sung query
CPU work (audio loading, resampling, Melodia) runs in DataLoader workers; separation and
the neural trackers run on the GPU in the main process.
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hum2song.augment import transpose_and_stretch
from hum2song.config import DEFAULT_DATA_ROOT, default_num_workers
from hum2song.contour.melody import (
    METHODS,
    TRACK_RATE,
    VOCAL_METHODS,
    load_separator,
    melodia_track,
    separate_vocals,
    vocal_track,
)
from hum2song.contour.melody_data import DATASETS, dataset_clips, mir1k_vocal, mixture
from hum2song.contour.trackers import load_rmvpe, rmvpe_track
from hum2song.logutil import configure_logging, get_logger

LOGGER = get_logger(__name__)
DEFAULT_WEIGHTS = "fig_contours/rmvpe.pt"
SHIFT_RANGE = (2.0, 5.0)
STRETCH_RANGE = (0.85, 1.15)
SEED = 20260927


def sung_query(vocal: np.ndarray, index: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + index)
    shift = float(rng.uniform(*SHIFT_RANGE)) * float(rng.choice([-1.0, 1.0]))
    return transpose_and_stretch(vocal, shift, float(rng.uniform(*STRETCH_RANGE)))


class SongDataset(Dataset):
    """Loads one clip's mixture (and MIR-1K vocal channel) and runs Melodia on the CPU."""

    def __init__(self, clips, dataset: str, methods: list[str]) -> None:
        self.clips = clips
        self.dataset = dataset
        self.methods = methods

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict:
        clip = self.clips[index]
        item = {"name": clip.name, "mix": mixture(clip.wav, TRACK_RATE)}
        if "melodia_mix" in self.methods:
            item["melodia_mix"] = melodia_track(item["mix"], TRACK_RATE)
        if self.dataset == "mir1k":
            vocal = mir1k_vocal(clip.wav, TRACK_RATE)
            item["vocal"], item["sung"] = vocal, sung_query(vocal, index)
        return item


def as_numpy(item: dict) -> dict:
    """DataLoader turns arrays into shared-memory tensors; the trackers take numpy.

    Copying releases the shared-memory handle, so kept tracks do not pile up open files.
    """
    return {k: v.numpy().copy() if isinstance(v, torch.Tensor) else v for k, v in item.items()}


def gpu_tracks(item: dict, models: dict, methods: list[str], device, clock: dict) -> dict:
    """Every GPU method for one clip; `clock` accumulates seconds per stage."""
    tracks = {}
    if "rmvpe_mix" in methods:
        start = time.perf_counter()
        tracks["rmvpe_mix"] = rmvpe_track(models["rmvpe"], item["mix"])
        clock["rmvpe_mix"] += time.perf_counter() - start
    vocal_methods = [method for method in methods if method in VOCAL_METHODS]
    if vocal_methods:
        start = time.perf_counter()
        vocals = separate_vocals(models["separator"], item["mix"], TRACK_RATE, device)
        clock["separation"] += time.perf_counter() - start
    for method in vocal_methods:
        start = time.perf_counter()
        tracks[method] = vocal_track(method, models, vocals, device)
        clock[method] += time.perf_counter() - start
    if "vocal" in item:
        tracks["clean"] = rmvpe_track(models["rmvpe"], item["vocal"])
        tracks["query_sung"] = rmvpe_track(models["rmvpe"], item["sung"])
    return tracks


def load_models(methods: list[str], weights: Path, device) -> dict:
    from torchfcpe import spawn_bundled_infer_model

    models = {"rmvpe": load_rmvpe(weights, device)}
    if any(method in VOCAL_METHODS for method in methods):
        models["separator"] = load_separator(device)
    if "fcpe_vocals" in methods:
        models["fcpe"] = spawn_bundled_infer_model(device=str(device))
    return models


def extract_dataset(models: dict, root: Path, dataset: str, args, device) -> None:
    clips = dataset_clips(root, dataset)
    loader = DataLoader(
        SongDataset(clips, dataset, args.methods),
        batch_size=None,
        num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
    )
    tracks: dict[str, np.ndarray] = {}
    clock: dict[str, float] = defaultdict(float)
    for index, item in enumerate(map(as_numpy, loader)):
        if "melodia_mix" in item:
            tracks[f"melodia_mix/{item['name']}"] = item["melodia_mix"]
        for method, track in gpu_tracks(item, models, args.methods, device, clock).items():
            tracks[f"{method}/{item['name']}"] = track
        if index % 100 == 0:
            LOGGER.info("%s: %s / %s", dataset, index, len(clips))
    destination = root / "f0" / f"songmel_{dataset}.npz"
    np.savez(destination, **tracks)
    LOGGER.info("wrote %s tracks to %s; GPU seconds %s", len(tracks), destination, dict(clock))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Melody tracks for songs with annotations")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=default_num_workers())
    args = parser.parse_args(argv)
    args.methods = [name for name in args.methods.split(",") if name]
    return args


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.methods, args.weights or args.data_root / DEFAULT_WEIGHTS, device)
    for dataset in (name for name in args.datasets.split(",") if name):
        extract_dataset(models, args.data_root, dataset, args, device)


if __name__ == "__main__":
    main()
