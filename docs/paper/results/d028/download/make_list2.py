"""E2b v2: top ~2000 CHAD cover groups by aligned-fragment evidence; segments only (+-4 s)."""

import ast
import collections
import csv
import json

MARGIN = 4
TOP = 2000
plan = json.load(open("e2b_plan.json"))["plan"]
done = set(open("done_snap.txt").read().split())
have = {n.split(".")[0] for n in open("have_snap.txt").read().split()}
frag = collections.defaultdict(dict)
corr = collections.defaultdict(float)
for x in csv.DictReader(open("dataset.csv")):
    if x["audio_type"] == "humming":
        continue
    frag[(x["group_id"], x["youtube_id"])][x["fragment_id"]] = ast.literal_eval(x["interval"])
    if x["audio_type"] == "cover" and x["correlation"]:
        corr[(x["group_id"], x["youtube_id"], x["fragment_id"])] = float(x["correlation"])


def shared(g, p):
    o, c = frag[(g, p["original"])], frag[(g, p["covers"][0])]
    return sorted(set(o) & set(c))


def score(g):
    p = plan[g]
    return sum(corr[(g, p["covers"][0], f)] for f in shared(g, p))


def sections(g, vid, fids):
    spans = sorted(
        (max(frag[(g, vid)][f][0] - MARGIN, 0), frag[(g, vid)][f][1] + MARGIN) for f in fids
    )
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 20:
            merged[-1][1] = max(merged[-1][1], b)
            continue
        merged.append([a, b])
    return ";".join(f"{a}-{b}" for a, b in merged)


full = [g for g in plan if f"o_{g}" in have and f"c1_{g}" in have]
rest = sorted((g for g in plan if g not in full and shared(g, plan[g])), key=lambda g: -score(g))
chosen = rest[: TOP - len(full)]
lines = []
for g in chosen:
    p, fids = plan[g], shared(g, plan[g])
    for tag, vid in ((f"o_{g}", p["original"]), (f"c1_{g}", p["covers"][0])):
        if tag in have:
            continue
        lines.append(f"{tag}\thttps://www.youtube.com/watch?v={vid}\t{sections(g, vid, fids)}")
open("e2b_v2.tsv", "w").write("\n".join(lines) + "\n")
secs = [
    sum(float(s.split("-")[1]) - float(s.split("-")[0]) for s in line.split("\t")[2].split(";"))
    for line in lines
]
print(
    "full groups already",
    len(full),
    "chosen",
    len(chosen),
    "downloads",
    len(lines),
    "mean seg s",
    sum(secs) / len(secs),
)
print(
    "min score chosen",
    score(chosen[-1]),
    "median frags",
    sorted(len(shared(g, plan[g])) for g in chosen)[len(chosen) // 2],
)
