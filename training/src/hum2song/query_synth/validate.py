"""Checkpoint selection on the synthetic whistle and wrong-lyrics VAL split (D-041).

D-040 picked checkpoints on HumTrans hum val, which sits near 0.995 from step 0, so
best.pt never moved off the init. This validator ranks val-split synthetic queries
over vocal-melody windows of the val songs only (same windows-per-song as the test
report). The gallery is the ~1.1k val songs, so these numbers are for selection and
are not comparable to the 11k-song test numbers. Every row is synthetic.
"""

from pathlib import Path

import numpy as np

from hum2song.contour.evaluate import query_contour
from hum2song.query_synth.manifest import read_manifest, require_synthetic
from hum2song.query_synth.report import _index, score_lists

STYLES = ("whistle", "wrong_lyrics")


class QuerySynthValidation:
    def __init__(
        self,
        root: Path,
        manifest: str,
        ref_windows: int = 24,
        max_per_style: int = 3000,
        seed: int = 0,
    ) -> None:
        val = [row for row in read_manifest(root / manifest) if row["split"] == "val"]
        songs: dict[str, np.ndarray] = {}
        for row in val:
            require_synthetic(row)
            if row["song_id"] not in songs:
                songs[row["song_id"]] = np.load(root / row["ref_contour"])
        self.refs, self.ref_ids = _index(songs, ref_windows)
        rng = np.random.default_rng(seed)
        self.queries: dict[str, tuple[list[np.ndarray], list[str]]] = {}
        for style in STYLES:
            chosen = [row for row in val if row["style"] == style]
            if 0 < max_per_style < len(chosen):
                keep = np.sort(rng.choice(len(chosen), max_per_style, replace=False))
                chosen = [chosen[int(i)] for i in keep]
            self.queries[style] = (
                [query_contour(np.load(root / row["contour"])) for row in chosen],
                [row["song_id"] for row in chosen],
            )

    def metrics(self, model, device) -> dict[str, float]:
        out: dict[str, float] = {}
        for style, (queries, ids) in self.queries.items():
            scores = score_lists(model, queries, ids, self.refs, self.ref_ids, device)
            for key in ("top1", "top10", "mrr"):
                value = scores[key]
                out[f"val/qsynth_{style}_{key}"] = float("nan") if value is None else float(value)
        return out
