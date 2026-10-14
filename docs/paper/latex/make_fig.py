"""Bar chart of top-1 from the committed result JSONs (docs/paper/results/d042). No numbers typed by hand."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "..", "results", "d042")
models = [("e2b", "E2b (baseline)"), ("d040_last", "D-040"), ("d041_best", "D-041"), ("d042_best", "D-042")]
def j(n): return json.load(open(os.path.join(R, n)))
groups = ["Synthetic whistle\n[synthetic]", "Synthetic wrong lyrics\n[synthetic]",
          "MLEnd real whistle,\nall 8 songs [real]", "MLEnd real whistle,\n2 held-out songs [real]"]
vals = {}
for k, _ in models:
    s = j(f"eval_style_{k}.json"); m = j(f"eval_mlend_{k}.json")
    key = "whistle_peak->hum/centroid"
    vals[k] = [s["whistle"]["top1"], s["wrong_lyrics"]["top1"],
               m["results"][key]["top1"], m["results_heldout_songs"][key]["top1"]]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
w = 0.2
for i, (k, lab) in enumerate(models):
    xs = [g + (i - 1.5) * w for g in range(len(groups))]
    b = ax.bar(xs, vals[k], w, label=lab)
    for x, v in zip(xs, vals[k]):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=7)
ax.set_ylabel("Top-1"); ax.set_ylim(0, 0.8)
ax.axvline(1.5, color="gray", lw=0.6, ls="--")
ax.text(0.5, 0.74, "11,050-song gallery", ha="center", fontsize=7, color="gray")
ax.text(2.5, 0.74, "8-way closed set (chance 0.125)", ha="center", fontsize=7, color="gray")
ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "figures", "fig_results.pdf"))
print(json.dumps(vals, indent=1))
