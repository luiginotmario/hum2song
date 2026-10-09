"""Print the D-040 tables used in docs/paper/paper.md from the JSON files in this folder.

Run from anywhere: python docs/paper/results/d040/summarize.py
Every number in the paper's D-040 tables is printed here; nothing is computed elsewhere.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = (("E2b (contour_e2b_s0/best.pt, step 800)", "e2b", "_e2b"),
          ("D-040 best.pt (selected step 0 = E2b weights)", None, ""),
          ("D-040 last.pt (step 3000)", "last", "_last"))


def load(name):
    return json.loads((HERE / name).read_text())


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    print("Synthetic held-out queries (origin=synthetic), top-1 / top-10 / MRR")
    for label, style_tag, tag in MODELS:
        ev = load(f"eval{tag}.json")["sets"]["synthetic_holdout"]
        print(f"  {label}: all {ev['count']} q vs {ev['n_songs']} songs: "
              f"{fmt(ev['top1'])} / {fmt(ev['top10'])} / {fmt(ev['mrr'])}")
        if style_tag:
            st = load(f"eval_style_{style_tag}.json")
            for style in ("whistle", "wrong_lyrics"):
                s = st[style]
                print(f"      {style:12s} {s['count']} q: {fmt(s['top1'])} / {fmt(s['top10'])} / {fmt(s['mrr'])}")
    print("\nHum baseline (real hums, MIDI references), top-1 / top-10 / MRR")
    for label, _s, tag in MODELS:
        sets = load(f"eval{tag}.json")["sets"]["hum_baseline"]["sets"]
        for k in ("humtrans_test", "humtrans_test_shift+7", "mir48_anywhere"):
            s = sets[k]
            print(f"  {label:48s} {k:22s} n={s['count']}: {fmt(s['top1'])} / {fmt(s['top10'])} / {fmt(s['mrr'])}")
    print("\nMLEnd test people, query-by-example (8-way, centroid), top-1 / top-3 / MRR")
    keys = ("whistle_peak->hum/centroid", "whistle_rmvpe_half->hum/centroid",
            "whistle_peak->whistle_peak/centroid", "hum->hum/centroid")
    for label, _s, tag in MODELS:
        ml = load(f"eval_mlend{tag}.json")
        for sec, name in (("results", "all 8 songs"), ("results_heldout_songs", "Hakuna+Potter"),
                          ("results_seen_songs", "other 6 songs")):
            for k in keys:
                r = ml[sec][k]
                print(f"  {label[:22]:22s} {name:14s} {k:38s} {fmt(r['top1'])} / {fmt(r['top3'])} / {fmt(r['mrr'])}")
    print("\nBlocked real sets (no target audio):")
    for k in ("mlend_whistle", "mtgqbh_sing"):
        print(f"  {k}: {load('eval_last.json')['sets'][k]['status']}")


if __name__ == "__main__":
    main()
