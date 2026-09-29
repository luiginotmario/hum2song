"""E2b download list: per CHAD cover group, the original and its best covers (random order)."""

import collections
import csv
import json
import random

SEED = 20260929
r = list(csv.DictReader(open("dataset.csv")))
split = json.load(open("../../../../../training/splits/chad_songs.json"))["groups"]
hum_groups = {x["group_id"] for x in r if x["audio_type"] == "humming"}
blocked_groups = {g for g in hum_groups if split.get(g) != "train"}
blocked_ids = {x["youtube_id"] for x in r if x["group_id"] in blocked_groups}


def avail(x):
    return x["is_available"] == "True" and x["youtube_id"] not in blocked_ids


by = collections.defaultdict(list)
for x in r:
    by[x["group_id"]].append(x)


def cover_ranking(rows):
    score = collections.defaultdict(float)
    for x in rows:
        if x["audio_type"] == "cover" and avail(x) and x["correlation"]:
            score[x["youtube_id"]] += float(x["correlation"])
    return sorted(score, key=lambda v: -score[v])


def original(rows):
    ids = collections.Counter(
        x["youtube_id"] for x in rows if x["audio_type"] == "original" and avail(x)
    )
    return ids.most_common(1)[0][0] if ids else None


plan = {}
for g, rows in by.items():
    if g in blocked_groups:
        continue
    o, covers = original(rows), cover_ranking(rows)
    if o and covers:
        plan[g] = {"original": o, "covers": [c for c in covers if c != o]}
plan = {g: p for g, p in plan.items() if p["covers"]}
groups = sorted(plan)
random.Random(SEED).shuffle(groups)
seen = set()
tier = {1: [], 2: []}
for g in groups:
    p = plan[g]
    for tag, vid, t in [(f"o_{g}", p["original"], 1), (f"c1_{g}", p["covers"][0], 1)] + (
        [(f"c2_{g}", p["covers"][1], 2)] if len(p["covers"]) > 1 else []
    ):
        if vid in seen:
            continue
        seen.add(vid)
        tier[t].append(f"{tag}\thttps://www.youtube.com/watch?v={vid}")
open("e2b_t1.tsv", "w").write("\n".join(tier[1]) + "\n")
open("e2b_t2.tsv", "w").write("\n".join(tier[2]) + "\n")
json.dump(
    {"groups": groups, "plan": plan, "blocked_groups": sorted(blocked_groups)},
    open("e2b_plan.json", "w"),
)
print(
    "blocked hum groups",
    len(blocked_groups),
    "groups",
    len(groups),
    "tier1",
    len(tier[1]),
    "tier2",
    len(tier[2]),
)
