"""Markdown table from results.json (D-044). Usage: python summarize.py results.json"""
import json, sys

r = json.load(open(sys.argv[1]))["results"]
print("| Model | train performers | whistle->hums top-1 [95% CI clip; performer] | top-3 | whistle->whistles top-1 | hum->hums top-1 |")
print("|---|---|---|---|---|---|")
for name, v in r.items():
    w = v["whistle->hums"]
    f = lambda m: f"{m['value']:.3f} [{m['ci95_clip'][0]:.3f}-{m['ci95_clip'][1]:.3f}; {m['ci95_performer'][0]:.3f}-{m['ci95_performer'][1]:.3f}]"
    tp = v.get("train", {}).get("train_performers", 0)
    print(f"| {name} | {tp} | {f(w['top1'])} | {f(w['top3'])} | {f(v['whistle->whistles']['top1'])} | {f(v['hum->hums']['top1'])} |")
