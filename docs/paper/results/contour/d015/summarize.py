"""Summarize D-015: MLEnd test people per query type, MIR-QBSH and HumTrans (mean ± sd)."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD_MIR = HERE.parent / "eval_contour_v2_s{seed}_last_test.json"
MLEND_KEYS = (
    "hum->hum/centroid",
    "whistle_peak->hum/centroid",
    "whistle_peak->whistle_peak/centroid",
    "hum->whistle_peak/centroid",
    "whistle_rmvpe_half->hum/centroid",
)
MIR_KEYS = (
    "mir48_start_multi_top10",
    "mir48_anywhere_top1",
    "mir48_anywhere_top10",
    "mir48+2000_start_multi_top1",
    "mir48+2000_start_multi_top10",
    "mir48+2000_anywhere_top1",
    "mir48+2000_anywhere_top10",
    "humtrans_test_top1",
    "humtrans_test_shift+7_top1",
)
MODELS = {
    "old (D-012, last.pt)": ("mlend_test_v2_s{seed}.json", str(OLD_MIR), (0, 1, 2)),
    "whistle, val-selected": (
        "mlend_test_whistle_s{seed}_best.json",
        "mir_whistle_s{seed}_best.json",
        (0, 1, 2),
    ),
    "whistle, last.pt": (
        "mlend_test_whistle_s{seed}_last.json",
        "mir_whistle_s{seed}_last.json",
        (0, 1, 2),
    ),
    "aug only (s0, val-selected)": (
        "mlend_test_augonly_s{seed}_best.json",
        "mir_augonly_s{seed}_best.json",
        (0,),
    ),
    "pairs only (s0, val-selected)": (
        "mlend_test_pairsonly_s{seed}_best.json",
        "mir_pairsonly_s{seed}_best.json",
        (0,),
    ),
}


def load(pattern: str, seed: int) -> dict:
    path = Path(pattern.format(seed=seed))
    return json.loads((path if path.is_absolute() else HERE / path).read_text())


def mir_value(report: dict, key: str) -> float:
    return report["metrics"][f"val/{key}"]


def stat(values: list[float]) -> str:
    return f"{np.mean(values):.3f}" + (f"±{np.std(values):.3f}" if len(values) > 1 else "")


def summarize() -> dict:
    summary = {}
    for name, (mlend, mir, seeds) in MODELS.items():
        mlend_reports = [load(mlend, s) for s in seeds]
        mir_reports = [load(mir, s) for s in seeds]
        row = {key: stat([r["results"][key]["top1"] for r in mlend_reports]) for key in MLEND_KEYS}
        row |= {key: stat([mir_value(r, key) for r in mir_reports]) for key in MIR_KEYS}
        row["steps"] = [r["step"] for r in mir_reports]
        summary[name] = row
    return summary


if __name__ == "__main__":
    result = summarize()
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, row in result.items():
        print(name)
        for key, value in row.items():
            print(f"  {key:40s} {value}")
