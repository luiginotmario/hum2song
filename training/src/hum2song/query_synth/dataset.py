"""CLEWS pairs: a synthetic whistle or wrong-lyrics contour against vocal-melody windows."""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from hum2song.contour.augment import ContourAugment, augment_contour
from hum2song.contour.data import EpochCounter, features_tensor
from hum2song.contour.features import FRAME_S
from hum2song.contour.hum_pairs import vocal_window
from hum2song.query_synth.manifest import require_synthetic

MIN_REF_FRAMES = 1
MIN_AUG_VOICED = 20


class QuerySynthDataset(Dataset):
    """One re-tracked synthetic query and several vocal windows of the same song.

    The query contour is what the tracker read off the rendered audio. With
    `augment` (D-041) it gets a fresh light tempo/voicing augmentation each epoch. Reference windows are the song vocal melody, near the cut
    the query was made from.
    """

    def __init__(
        self,
        root: Path,
        rows: list[dict],
        seed: int,
        refs: int = 4,
        offset_s: float = 3.0,
        extra_s: tuple[float, float] = (-1.0, 3.0),
        augment: ContourAugment | None = None,
    ) -> None:
        self.root = root
        self.rows = [row for row in rows if _ref_exists(root, row)]
        for row in self.rows:
            require_synthetic(row)
        self.seed = seed
        self.refs = refs
        self.offset_s = offset_s
        self.extra_s = extra_s
        self.augment = augment
        self.epoch = EpochCounter()
        self._songs: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def song(self, song_id: str, ref_contour: str) -> np.ndarray:
        cached = self._songs.get(song_id)
        if cached is None:
            cached = np.load(self.root / ref_contour)
            self._songs[song_id] = cached
        return cached

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch.get(), index, 7])
        row = self.rows[index]
        query = np.load(self.root / row["contour"])
        if self.augment is not None:
            augmented = augment_contour(query, rng, self.augment, FRAME_S)
            if int(np.sum(~np.isnan(augmented))) >= MIN_AUG_VOICED:
                query = augmented
        ref = self.song(row["song_id"], row["ref_contour"])
        windows = [
            vocal_window(
                ref,
                int(row["start_frame"]),
                max(int(row["length_frames"]), MIN_REF_FRAMES),
                rng,
                self.offset_s,
                self.extra_s,
            )
            for _slot in range(self.refs)
        ]
        return {
            "query": features_tensor(query),
            "refs": [features_tensor(window) for window in windows],
            "song_id": row["song_id"],
        }


def _ref_exists(root: Path, row: dict) -> bool:
    return (root / row["contour"]).exists() and (root / row["ref_contour"]).exists()
