"""Check that no MIR-QBSH target melody is also an Essen training melody (D-005 hygiene).

For each of the 48 MIR-QBSH MIDIs, the first 12 s of its contour is compared with the
start of every Essen melody used for synthetic training (key removed by median offset,
7 tempo scales, mean absolute semitone error on frames voiced in both). Prints the
closest Essen melody per target, sorted; a near-zero error would mean an overlap.
"""
import sys
from pathlib import Path

import numpy as np

from hum2song.contour.evaluate import distractor_paths
from hum2song.contour.features import FRAME_S, midi_contour, trim_unvoiced
from hum2song.midi_render import parse_midi

ROOT = Path("/lambda/nfs/hum2song-data")
SCALES = np.linspace(0.7, 1.4, 7)
WINDOW = int(12.0 / FRAME_S)


def contour(path):
    return trim_unvoiced(midi_contour(parse_midi(path.read_bytes())))


def error(query, ref):
    length = len(query)
    idx = np.clip(np.round(np.linspace(0, len(ref) - 1, length)).astype(int), 0, None)
    diff = query - ref[idx]
    both = ~np.isnan(diff)
    if both.sum() < 0.5 * length:
        return np.inf
    d = diff[both]
    return float(np.mean(np.abs(d - np.median(d))))


def best(query, ref):
    return min(error(query, ref[: max(int(WINDOW * k), 2)]) for k in SCALES)


def main():
    held = set(distractor_paths(ROOT / "raw/essen_midi/deutschl", 2000))
    essen = [p for d in ("deutschl", "china") for p in sorted((ROOT / "raw/essen_midi" / d).rglob("*.mid"))
             if p not in held]
    refs = [(p.stem, contour(p)) for p in essen]
    rows = []
    for mir in sorted((ROOT / "raw/mirqbsh").rglob("*.mid")):
        query = contour(mir)[:WINDOW]
        scores = [(best(query, ref), name) for name, ref in refs]
        rows.append((min(scores), mir.stem))
    for (score, name), target in sorted(rows):
        print(f"{target} closest={name} mae={score:.3f}")


if __name__ == "__main__":
    sys.exit(main())
