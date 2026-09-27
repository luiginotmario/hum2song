"""Summarize D-016: unseen-song whistles and the pairs-only vs combined config.

python summarize.py --d015 DIR --d016 DIR --train DIR [--out summary.json]
  d015   eval JSONs of D-015 (mir_whistle_s*_best, mir_pairsonly_s0_best)
  d016   eval JSONs of D-016 (mlend_test_*, mir_holdout_*, mir_pairsonly_s[12]_best)
  train  training result JSONs (contour_whistle*_s*.json) with the val selection scores
Values are mean ± sd over seeds of top-1 (MIR also top-10).
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD_MIR = HERE.parent / "eval_contour_v2_s{seed}_last_test.json"
SEEDS = (0, 1, 2)
MLEND_KEYS = {
    "whistle->hums": "whistle_peak->hum/centroid",
    "hum->hums": "hum->hum/centroid",
    "whistle->whistles": "whistle_peak->whistle_peak/centroid",
}
MIR_KEYS = (
    "mir48+2000_anywhere_top1",
    "mir48+2000_anywhere_top10",
    "mir48+2000_start_multi_top10",
    "humtrans_test_top1",
    "humtrans_test_shift+7_top1",
)
VAL_KEYS = ("val/select", "val/mlend_val_whistle_top1", "val/mlend_val_hum_top1")
HOLDOUT_MODELS = {
    "old (D-012, no MLEnd)": "mlend_test_v2_s{seed}.json",
    "whistle, all 8 songs (D-015)": "mlend_test_whistle_s{seed}_best.json",
    "whistle, 2 songs held out": "mlend_test_holdout_s{seed}_best.json",
}
CONFIGS = {
    "combined (D-015)": ("whistle_s{seed}_best", "contour_whistle_s{seed}.json"),
    "pairs only": ("pairsonly_s{seed}_best", "contour_whistle_pairsonly_s{seed}.json"),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def stat(values: list[float]) -> str:
    return f"{np.mean(values):.3f}±{np.std(values):.3f}"


def holdout_table(d016: Path) -> dict:
    table = {}
    for name, pattern in HOLDOUT_MODELS.items():
        reports = [read(d016 / pattern.format(seed=s)) for s in SEEDS]
        table[name] = {
            f"{part}:{label}": stat([r[f"results_{part}_songs"][key]["top1"] for r in reports])
            for part in ("heldout", "seen")
            for label, key in MLEND_KEYS.items()
        }
    return table


def mir_report(d015: Path, d016: Path, stem: str) -> dict:
    return (
        read(d016 / f"mir_{stem}.json")
        if (d016 / f"mir_{stem}.json").exists()
        else read(d015 / f"mir_{stem}.json")
    )


def config_table(d015: Path, d016: Path, train: Path) -> dict:
    table = {}
    for name, (stem, train_name) in CONFIGS.items():
        mirs = [mir_report(d015, d016, stem.format(seed=s)) for s in SEEDS]
        mlends = [read(d016 / f"mlend_test_{stem.format(seed=s)}.json") for s in SEEDS]
        trains = [read(train / train_name.format(seed=s)) for s in SEEDS]
        row = {key: stat([m["metrics"][f"val/{key}"] for m in mirs]) for key in MIR_KEYS}
        row |= {
            f"mlend_test:{label}": stat([r["results"][key]["top1"] for r in mlends])
            for label, key in MLEND_KEYS.items()
        }
        row |= {key: stat([t["selected"][key] for t in trains]) for key in VAL_KEYS}
        row["selected_steps"] = [t["selected_step"] for t in trains]
        table[name] = row
    old = [read(Path(str(OLD_MIR).format(seed=s))) for s in SEEDS]
    table["old (D-012)"] = {
        key: stat([m["metrics"][f"val/{key}"] for m in old]) for key in MIR_KEYS
    }
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D-016 summary")
    for name in ("--d015", "--d016", "--train"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = {
        "unseen_songs_test_people": holdout_table(args.d016),
        "pairs_only_vs_combined": config_table(args.d015, args.d016, args.train),
    }
    text = json.dumps(summary, indent=2)
    args.out.write_text(text + "\n")
    print(text)
