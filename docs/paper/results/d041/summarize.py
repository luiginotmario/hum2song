"""Print the D-041 comparison tables (markdown) from the JSON files in this folder.

Run from anywhere: python docs/paper/results/d041/summarize.py
Every D-041 number in docs/paper/paper.md and README.md is printed here; nothing is
computed elsewhere. All four checkpoints were scored by one chain (run_chain.sh) on
identical sets.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = (
    ("D-041 best.pt (step 9500, selected on synthetic val)", "d041_best"),
    ("D-041 last.pt (step 11500, early stop)", "d041_last"),
    ("D-040 last.pt (step 3000)", "d040_last"),
    ("E2b (live, step 800)", "e2b"),
)
SYNTH_STYLES = (("whistle", "Synthetic whistle"), ("wrong_lyrics", "Synthetic wrong-lyrics"))
HUM_SETS = (
    ("humtrans_test", "HumTrans test"),
    ("humtrans_test_shift+7", "HumTrans test +7 st"),
    ("mir48_anywhere", "MIR-QBSH 48 anywhere"),
)
MLEND_KEY = "whistle_peak->hum/centroid"
MLEND_SECTIONS = (
    ("results_heldout_songs", "MLEnd held-out songs (Hakuna+Potter) [real, clean]"),
    ("results_seen_songs", "MLEnd seen songs (other 6) [real, NOT clean for D-041]"),
    ("results", "MLEnd all 8 [real, mixed]"),
)
VAL_FIELDS = ("val/select", "val/qsynth_whistle_top1", "val/qsynth_wrong_lyrics_top1",
              "val/humtrans_val_shift+7_top1")


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def triple(scores: dict, second: str) -> str:
    return f"{fmt(scores['top1'])} / {fmt(scores[second])} / {fmt(scores['mrr'])}"


def markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def header(first: str) -> list[str]:
    names = [label for label, _tag in MODELS]
    return [markdown_row([first] + names), markdown_row(["---"] * (len(names) + 1))]


def synthetic_rows() -> list[str]:
    rows = header("Synthetic hold-out [synthetic], top-1 / top-10 / MRR")
    for style, label in SYNTH_STYLES:
        cells = [triple(load(f"eval_style_{tag}.json")[style], "top10") for _l, tag in MODELS]
        rows.append(markdown_row([f"{label} (4,263 q)"] + cells))
    both = [triple(load(f"eval_{tag}.json")["sets"]["synthetic_holdout"], "top10") for _l, tag in MODELS]
    rows.append(markdown_row(["Both styles (8,526 q)"] + both))
    return rows


def hum_rows() -> list[str]:
    rows = header("Real hums vs MIDI [real], top-1 / top-10 / MRR")
    for key, label in HUM_SETS:
        sets = [load(f"eval_{tag}.json")["sets"]["hum_baseline"]["sets"][key] for _l, tag in MODELS]
        rows.append(markdown_row([f"{label} ({sets[0]['count']})"] + [triple(s, "top10") for s in sets]))
    return rows


def mlend_rows() -> list[str]:
    rows = header(f"MLEnd test people, {MLEND_KEY} [real], top-1 / top-3 / MRR")
    for section, label in MLEND_SECTIONS:
        cells = [triple(load(f"eval_mlend_{tag}.json")[section][MLEND_KEY], "top3") for _l, tag in MODELS]
        rows.append(markdown_row([label] + cells))
    return rows


def checkpoint_steps() -> list[str]:
    return [f"- {label}: step {load(f'eval_{tag}.json').get('checkpoint_step')}" for label, tag in MODELS]


def validation_curve() -> list[str]:
    pattern = re.compile(r"step (\d+) (val/\S+=.*)")
    rows = [markdown_row(["step"] + list(VAL_FIELDS)), markdown_row(["---"] * (len(VAL_FIELDS) + 1))]
    for match in pattern.finditer((HERE / "train.log").read_text()):
        values = dict(item.split("=") for item in match.group(2).split())
        rows.append(markdown_row([match.group(1)] + [values[name] for name in VAL_FIELDS]))
    return rows


def main() -> None:
    sections = (synthetic_rows(), hum_rows(), mlend_rows(), checkpoint_steps(), validation_curve())
    print("\n\n".join("\n".join(lines) for lines in sections))


if __name__ == "__main__":
    main()
