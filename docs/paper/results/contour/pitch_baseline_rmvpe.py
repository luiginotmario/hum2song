"""Pitch-only MIR-QBSH baseline with RMVPE queries instead of the hand-labelled .pv files.

Same method as pitch_baseline.py (voiced frames, median key removal, 13 tempo scales,
linear scaling vs the start of each MIDI), 32 ms frames. Queries are all MIR-QBSH test
records with a reference (hum + sing).
"""
import json
import sys
from pathlib import Path

import numpy as np

from hum2song.contour.features import clean_contour, hz_to_semitones
from hum2song.midi_render import parse_midi

ROOT = Path("/lambda/nfs/hum2song-data")
FRAME_S = 0.032
SCALES = np.linspace(0.6, 1.8, 13)


def query_semitones(track):
    f0, conf = track[0], track[1]
    voiced = (conf >= 0.3) & (f0 > 50) & (f0 < 2000)
    st = clean_contour(np.where(voiced, hz_to_semitones(f0), np.nan))  # 10 ms
    times = np.arange(0, len(st) * 0.01, FRAME_S)
    idx = np.minimum((times / 0.01).astype(int), len(st) - 1)
    grid = st[idx]
    return grid[~np.isnan(grid)], len(grid) * FRAME_S


def midi_frames(notes, seconds):
    times = np.arange(0.0, seconds, FRAME_S)
    pitch = np.zeros_like(times)
    for n in notes:
        pitch[(times >= n.start_s) & (times < n.end_s)] = n.pitch
    return pitch[pitch > 0]


def resample(values, length):
    return np.interp(np.linspace(0, len(values) - 1, length), np.arange(len(values)), values)


def distance(q, r):
    d = q - resample(r, len(q))
    return float(np.mean(np.abs(d - np.median(d))))


def main():
    recs = [json.loads(l) for l in open(ROOT / "pairs_real.jsonl")]
    recs = [r for r in recs if r["group"] == "mirqbsh" and r["song_path"]]
    songs = sorted({Path(r["song_path"]).stem for r in recs})
    midis = {p.stem: p for p in (ROOT / "raw/mirqbsh").rglob("*.mid")}
    notes = {s: sorted(parse_midi(midis[s].read_bytes()), key=lambda n: n.start_s) for s in songs}
    tracks = np.load(ROOT / "f0/rmvpe_mirqbsh.npz")
    ranks = []
    for r in recs:
        q, secs = query_semitones(tracks[r["query_path"]])
        if len(q) < 20:
            ranks.append(len(songs) - 1)
            continue
        scores = []
        for s in songs:
            cands = [midi_frames(notes[s], secs * k) for k in SCALES]
            scores.append(min(distance(q, c) for c in cands if len(c) > 1))
        target = songs.index(Path(r["song_path"]).stem)
        ranks.append(int(np.where(np.argsort(scores, kind="stable") == target)[0][0]))
    ranks = np.array(ranks)
    print(f"queries={len(ranks)} top1={np.mean(ranks < 1):.4f} top10={np.mean(ranks < 10):.4f} "
          f"mrr={np.mean(1.0 / (ranks + 1)):.4f}")


if __name__ == "__main__":
    sys.exit(main())
