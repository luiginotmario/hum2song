"""RMVPE pitch-only baseline (pitch_baseline_rmvpe.py method) with 2,000 Essen distractors.

Distractors are the same evenly spaced 2,000 files eval_contour.py --distractors picks.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pitch_baseline_rmvpe import ROOT, SCALES, distance, midi_frames, query_semitones  # noqa: E402

from hum2song.midi_render import parse_midi  # noqa: E402

NOTES = {}


def load_notes():
    recs = [json.loads(l) for l in open(ROOT / "pairs_real.jsonl")]
    recs = [r for r in recs if r["group"] == "mirqbsh" and r["song_path"]]
    midis = {p.stem: p for p in (ROOT / "raw/mirqbsh").rglob("*.mid")}
    targets = sorted({Path(r["song_path"]).stem for r in recs})
    essen = sorted((ROOT / "raw/essen_midi/deutschl").rglob("*.mid"))
    essen = [essen[int(i)] for i in np.linspace(0, len(essen) - 1, 2000).round()]
    notes = {s: parse_midi(midis[s].read_bytes()) for s in targets}
    notes.update({f"distractor:{p.stem}": parse_midi(p.read_bytes()) for p in essen})
    return recs, notes


def rank(job):
    q, secs, target = job
    if len(q) < 20:
        return len(NOTES) - 1
    names = list(NOTES)
    scores = []
    for name in names:
        cands = [midi_frames(NOTES[name], secs * k) for k in SCALES]
        valid = [distance(q, c) for c in cands if len(c) > 1]
        scores.append(min(valid) if valid else np.inf)
    return int(np.where(np.argsort(scores, kind="stable") == names.index(target))[0][0])


def main():
    global NOTES
    recs, NOTES = load_notes()
    tracks = np.load(ROOT / "f0/rmvpe_mirqbsh.npz")
    jobs = [(*query_semitones(tracks[r["query_path"]]), Path(r["song_path"]).stem) for r in recs]
    with Pool(12) as pool:
        ranks = np.array(pool.map(rank, jobs, chunksize=8))
    print(f"songs={len(NOTES)} queries={len(ranks)} top1={np.mean(ranks < 1):.4f} "
          f"top10={np.mean(ranks < 10):.4f} mrr={np.mean(1.0 / (ranks + 1)):.4f}")


if __name__ == "__main__":
    sys.exit(main())
