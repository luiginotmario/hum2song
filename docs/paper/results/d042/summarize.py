"""Print the D-042 comparison tables (markdown) from the JSON files in this folder.

Run from anywhere: python docs/paper/results/d042/summarize.py
Every D-042 number in docs/paper/paper.md and README.md is printed here; nothing is
computed elsewhere. All five checkpoints were scored by one chain (run_chain.sh) on
identical sets.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = (
    ("D-042 best.pt", "d042_best"),
    ("D-042 last.pt", "d042_last"),
    ("D-041 best.pt", "d041_best"),
    ("D-040 last.pt", "d040_last"),
    ("E2b (live)", "e2b"),
)
SYNTH_STYLES = (("whistle", "Synthetic whistle"), ("wrong_lyrics", "Synthetic wrong-lyrics"))
HUM_SETS = (
    ("humtrans_test", "HumTrans test"),
    ("humtrans_test_shift+7", "HumTrans test +7 st"),
    ("mir48_anywhere", "MIR-QBSH 48 anywhere"),
)
MLEND_KEY = "whistle_peak->hum/centroid"
MLEND_SECTIONS = (
    ("results_heldout_songs", "MLEnd held-out songs (Hakuna+Potter) [real, clean for all]"),
    ("results_seen_songs", "MLEnd other 6 songs [real, clean except D-041]"),
    ("results", "MLEnd all 8 songs [real, clean except D-041]"),
)
VAL_FIELDS = ("val/select", "val/qsynth_whistle_top1", "val/qsynth_wrong_lyrics_top1",
              "val/humtrans_val_shift+7_top1", "val/guard_ok")


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def label(name: str, tag: str) -> str:
    return f"{name} (step {load(f'eval_{tag}.json').get('checkpoint_step')})"


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def triple(scores: dict, second: str) -> str:
    return f"{fmt(scores['top1'])} / {fmt(scores[second])} / {fmt(scores['mrr'])}"


def markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def header(first: str) -> list[str]:
    names = [label(name, tag) for name, tag in MODELS]
    return [markdown_row([first] + names), markdown_row(["---"] * (len(names) + 1))]


def synthetic_rows() -> list[str]:
    rows = header("Synthetic hold-out [synthetic], top-1 / top-10 / MRR")
    for style, name in SYNTH_STYLES:
        cells = [triple(load(f"eval_style_{tag}.json")[style], "top10") for _n, tag in MODELS]
        count = load(f"eval_style_{MODELS[0][1]}.json")[style]["count"]
        rows.append(markdown_row([f"{name} ({count:,} q)"] + cells))
    both = [triple(load(f"eval_{tag}.json")["sets"]["synthetic_holdout"], "top10") for _n, tag in MODELS]
    rows.append(markdown_row(["Both styles"] + both))
    return rows


def hum_rows() -> list[str]:
    rows = header("Real hums vs MIDI [real], top-1 / top-10 / MRR")
    for key, name in HUM_SETS:
        sets = [load(f"eval_{tag}.json")["sets"]["hum_baseline"]["sets"][key] for _n, tag in MODELS]
        rows.append(markdown_row([f"{name} ({sets[0]['count']})"] + [triple(s, "top10") for s in sets]))
    return rows


def mlend_rows() -> list[str]:
    rows = header(f"MLEnd test people, {MLEND_KEY} [real], top-1 / top-3 / MRR")
    for section, name in MLEND_SECTIONS:
        cells = [triple(load(f"eval_mlend_{tag}.json")[section][MLEND_KEY], "top3") for _n, tag in MODELS]
        rows.append(markdown_row([name] + cells))
    return rows


# whistle_peak test clips: 90 on the two held-out songs, 249 on the other six, 339 total
# (eval_mlend counts.whistle_peak.clips = 339; held-out count from the D-041 README).
MLEND_N = {"results_heldout_songs": 90, "results_seen_songs": 249, "results": 339}


def mlend_counts() -> list[str]:
    """Correct whistles (top-1 x query count) for each MLEnd section."""
    rows = [markdown_row(["MLEnd correct top-1 / whistles [real]"] + [n for n, _t in MODELS]),
            markdown_row(["---"] * (len(MODELS) + 1))]
    for section, name in MLEND_SECTIONS:
        n = MLEND_N[section]
        cells = [f"{load(f'eval_mlend_{tag}.json')[section][MLEND_KEY]['top1'] * n:.0f} / {n}"
                 for _n, tag in MODELS]
        rows.append(markdown_row([name] + cells))
    return rows


def validation_curve() -> list[str]:
    pattern = re.compile(r"step (\d+) (val/\S+=.*)")
    rows = [markdown_row(["step"] + list(VAL_FIELDS)), markdown_row(["---"] * (len(VAL_FIELDS) + 1))]
    for match in pattern.finditer((HERE / "train.log").read_text()):
        values = dict(item.split("=") for item in match.group(2).split())
        rows.append(markdown_row([match.group(1)] + [values.get(name, "n/a") for name in VAL_FIELDS]))
    return rows


def main() -> None:
    sections = (synthetic_rows(), hum_rows(), mlend_rows(), mlend_counts(), validation_curve())
    print("\n\n".join("\n".join(lines) for lines in sections))


if __name__ == "__main__":
    main()
